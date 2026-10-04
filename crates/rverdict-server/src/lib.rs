//! Serves any [`Decider`] on the System One wire format, so existing Jev,
//! Clef and Von clients can use a local rverdict by changing their base URL.
//!
//! - `POST /v1/systemone`: answer a request.
//! - `GET /v1/models`: the model being served.
//! - `GET /health`: liveness.

use std::sync::Arc;

use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use rverdict_core::{DecideError, Decider, Request};
use serde_json::json;
use tokio::sync::Semaphore;

/// Largest accepted request body, enough for long documents.
const MAX_BODY: usize = 16 << 20;

#[derive(Debug, Clone)]
pub struct Options {
    /// When set, requests must carry `Authorization: Bearer <key>`.
    pub api_key: Option<String>,
    /// Requests decided at once. Each runs a forward pass, so on a GPU more
    /// than one mostly adds memory pressure.
    pub concurrency: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            api_key: None,
            concurrency: 1,
        }
    }
}

struct AppState {
    decider: Arc<dyn Decider>,
    api_key: Option<String>,
    permits: Semaphore,
}

pub fn router(decider: Arc<dyn Decider>, options: Options) -> Router {
    let state = Arc::new(AppState {
        decider,
        api_key: options.api_key,
        permits: Semaphore::new(options.concurrency.max(1)),
    });
    Router::new()
        .route("/v1/systemone", post(systemone))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .with_state(state)
}

/// Serves `router` until the process is stopped.
pub async fn serve(listener: tokio::net::TcpListener, router: Router) -> std::io::Result<()> {
    axum::serve(listener, router).await
}

struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            kind,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({"error": {"type": self.kind, "message": self.message}});
        (self.status, axum::Json(body)).into_response()
    }
}

impl AppState {
    fn authorize(&self, headers: &HeaderMap) -> Result<(), ApiError> {
        let Some(key) = &self.api_key else {
            return Ok(());
        };
        let given = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        match given {
            Some(given) if constant_time_eq(given.as_bytes(), key.as_bytes()) => Ok(()),
            _ => Err(ApiError::new(
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "missing or invalid API key",
            )),
        }
    }
}

/// Compares without an early exit, so response timing does not reveal how
/// much of a guessed key was right.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn systemone(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    state.authorize(&headers)?;
    let request: Request = serde_json::from_slice(&body).map_err(|e| {
        ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request",
            e.to_string(),
        )
    })?;
    let _permit = state.permits.acquire().await.map_err(|_| {
        ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable",
            "server is shutting down",
        )
    })?;
    let decider = Arc::clone(&state.decider);
    let outcome = tokio::task::spawn_blocking(move || decider.decide(&request))
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))?;
    match outcome {
        Ok(response) => Ok(axum::Json(response).into_response()),
        Err(DecideError::Invalid(e)) => Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request",
            e.to_string(),
        )),
        Err(DecideError::Failed(e)) => Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            e,
        )),
    }
}

async fn models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    state.authorize(&headers)?;
    let body =
        json!({"object": "list", "data": [{"id": state.decider.model(), "object": "model"}]});
    Ok(axum::Json(body).into_response())
}

async fn health(State(state): State<Arc<AppState>>) -> Response {
    axum::Json(json!({"status": "ok", "model": state.decider.model()})).into_response()
}
