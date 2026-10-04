//! Asks a remote System One API instead of a local model: TypeSafe's Jev,
//! Cloudflare Workers AI (Clef, Clef-flash, or other hosted decision
//! models), or any `/v1/systemone` server such as `rverdict serve`,
//! laya-serve or von.

use std::time::Duration;

use reqwest::StatusCode;
use rverdict_core::{Request, Response};
use serde_json::Value;

const TYPESAFE_ENDPOINT: &str = "https://api.typesafe.ai/v1/systemone";
const TYPESAFE_MODEL: &str = "jev-latest";
/// Cloudflare's own decision model; `@cf/cloudflare/clef-flash` is the fast one.
pub const CLOUDFLARE_CLEF: &str = "@cf/cloudflare/clef";

/// Statuses that mean "try again later": rate limits and overload.
const RETRYABLE: [u16; 4] = [429, 502, 503, 529];

#[derive(Debug, thiserror::Error)]
pub enum RemoteError {
    #[error("request to {endpoint} failed")]
    Http {
        endpoint: String,
        #[source]
        source: reqwest::Error,
    },
    /// The API rejected the request as malformed (HTTP 422).
    #[error("the API rejected the request: {0}")]
    Invalid(String),
    #[error("{endpoint} answered {status}: {body}")]
    Status {
        endpoint: String,
        status: u16,
        body: String,
    },
    #[error("unexpected response from {endpoint}: {detail}")]
    Response { endpoint: String, detail: String },
}

/// How the API wraps its answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Envelope {
    /// The System One response is the body.
    Plain,
    /// Cloudflare's REST envelope: `{"result": …, "success": …, "errors": […]}`.
    Cloudflare,
}

#[derive(Debug, Clone)]
pub struct RemoteClient {
    http: reqwest::Client,
    endpoint: String,
    api_key: Option<String>,
    model: Option<String>,
    envelope: Envelope,
    retries: u32,
}

impl RemoteClient {
    /// Any System One server. `base` may be the server root
    /// (`http://127.0.0.1:8090`) or the full `/v1/systemone` URL.
    pub fn system_one(base: &str) -> Self {
        let base = base.trim_end_matches('/');
        let endpoint = if base.ends_with("/v1/systemone") {
            base.to_owned()
        } else {
            format!("{base}/v1/systemone")
        };
        Self {
            http: reqwest::Client::new(),
            endpoint,
            api_key: None,
            model: None,
            envelope: Envelope::Plain,
            retries: 3,
        }
    }

    /// TypeSafe's hosted Jev.
    pub fn typesafe(api_key: impl Into<String>) -> Self {
        Self::system_one(TYPESAFE_ENDPOINT)
            .with_api_key(api_key)
            .with_model(TYPESAFE_MODEL)
    }

    /// A Workers AI decision model, by its Workers AI path (e.g.
    /// [`CLOUDFLARE_CLEF`]). The request's `model` is the path's last segment
    /// (`clef`, `clef-flash`) unless set with [`RemoteClient::with_model`].
    pub fn cloudflare(account_id: &str, api_token: impl Into<String>, model_path: &str) -> Self {
        let model = model_path
            .rsplit('/')
            .next()
            .unwrap_or(model_path)
            .to_owned();
        Self {
            endpoint: format!(
                "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_path}"
            ),
            envelope: Envelope::Cloudflare,
            ..Self::system_one("")
        }
        .with_api_key(api_token)
        .with_model(model)
    }

    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// The `model` sent with every request, overriding the caller's.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Attempts after the first for rate-limited or overloaded responses.
    #[must_use]
    pub fn with_retries(mut self, retries: u32) -> Self {
        self.retries = retries;
        self
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Whether requests stay on this machine. Anything else sends the
    /// state to a third party, which privacy rules may forbid.
    pub fn is_local(&self) -> bool {
        let Ok(url) = reqwest::Url::parse(&self.endpoint) else {
            return false;
        };
        let host = url.host_str().unwrap_or_default().trim_matches(['[', ']']);
        host == "localhost"
            || host
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
    }

    pub async fn decide(&self, request: &Request) -> Result<Response, RemoteError> {
        let mut request = request.clone();
        if let Some(model) = &self.model {
            request.model = Some(model.clone());
        }
        let mut attempt = 0;
        loop {
            let mut call = self.http.post(&self.endpoint).json(&request);
            if let Some(key) = &self.api_key {
                call = call.bearer_auth(key);
            }
            let reply = call
                .send()
                .await
                .map_err(|source| self.http_error(source))?;
            let status = reply.status();
            if RETRYABLE.contains(&status.as_u16()) && attempt < self.retries {
                tokio::time::sleep(backoff(attempt, retry_after(reply.headers()))).await;
                attempt += 1;
                continue;
            }
            let body = reply
                .text()
                .await
                .map_err(|source| self.http_error(source))?;
            return match status {
                StatusCode::OK => self.parse(&body),
                StatusCode::UNPROCESSABLE_ENTITY => Err(RemoteError::Invalid(error_message(&body))),
                _ => Err(RemoteError::Status {
                    endpoint: self.endpoint.clone(),
                    status: status.as_u16(),
                    body: error_message(&body),
                }),
            };
        }
    }

    fn http_error(&self, source: reqwest::Error) -> RemoteError {
        RemoteError::Http {
            endpoint: self.endpoint.clone(),
            source,
        }
    }

    fn parse(&self, body: &str) -> Result<Response, RemoteError> {
        let invalid = |detail: String| RemoteError::Response {
            endpoint: self.endpoint.clone(),
            detail,
        };
        let value: Value = serde_json::from_str(body).map_err(|e| invalid(e.to_string()))?;
        let value = match self.envelope {
            Envelope::Plain => value,
            Envelope::Cloudflare => unwrap_cloudflare(value).map_err(invalid)?,
        };
        serde_json::from_value(value).map_err(|e| invalid(e.to_string()))
    }
}

/// Cloudflare's REST API wraps results as `{"result", "success", "errors"}`;
/// a body that already looks like a System One response is used as-is.
fn unwrap_cloudflare(mut value: Value) -> Result<Value, String> {
    if value.get("answers").is_some() {
        return Ok(value);
    }
    if value.get("success") == Some(&Value::Bool(false)) {
        return Err(format!("Cloudflare reported failure: {}", value["errors"]));
    }
    match value.get_mut("result").map(Value::take) {
        Some(result) if result.is_object() => Ok(result),
        _ => Err("no `result` in Cloudflare's response".into()),
    }
}

/// The most useful text in an error body: `error.message`, `errors[0].message`, or the body.
fn error_message(body: &str) -> String {
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return body.chars().take(500).collect();
    };
    value["error"]["message"]
        .as_str()
        .or_else(|| value["errors"][0]["message"].as_str())
        .or_else(|| value["detail"].as_str())
        .map_or_else(|| value.to_string(), str::to_owned)
}

fn retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let seconds: u64 = headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()?;
    Some(Duration::from_secs(seconds.min(60)))
}

/// `Retry-After` when given, else 0.5 s doubling per attempt, capped at 8 s.
fn backoff(attempt: u32, retry_after: Option<Duration>) -> Duration {
    retry_after.unwrap_or_else(|| Duration::from_millis(500 * 2u64.pow(attempt.min(4))))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn cloudflare_envelopes_are_unwrapped_and_failures_reported() {
        let response = json!({"model": "clef", "answers": {}, "usage": {"input_tokens": 1, "output_tokens": 0}});
        let wrapped =
            json!({"result": response.clone(), "success": true, "errors": [], "messages": []});
        assert_eq!(unwrap_cloudflare(wrapped).unwrap(), response);
        assert_eq!(unwrap_cloudflare(response.clone()).unwrap(), response);
        let failed = json!({"result": null, "success": false, "errors": [{"code": 7, "message": "no route"}]});
        assert!(unwrap_cloudflare(failed).unwrap_err().contains("no route"));
    }

    #[test]
    fn only_loopback_endpoints_count_as_local() {
        assert!(RemoteClient::system_one("http://127.0.0.1:8090").is_local());
        assert!(RemoteClient::system_one("http://localhost:8090/v1/systemone").is_local());
        assert!(RemoteClient::system_one("http://[::1]:8090").is_local());
        assert!(!RemoteClient::typesafe("key").is_local());
        assert!(!RemoteClient::cloudflare("acct", "token", CLOUDFLARE_CLEF).is_local());
    }
}
