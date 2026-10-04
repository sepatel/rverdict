//! The remote client against a real `rverdict-server` on a loopback port,
//! with a stub decider standing in for a model.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rverdict_core::{Answer, DecideError, Decider, OrderedMap, Request, Response, Usage};
use rverdict_remote::{RemoteClient, RemoteError};
use rverdict_server::{Options, router};
use serde_json::json;

/// Answers every noul with 0.9, after validating the request as the engine does.
struct Stub;

impl Decider for Stub {
    fn decide(&self, request: &Request) -> Result<Response, DecideError> {
        let answers: OrderedMap<Answer> = request
            .parse_questions()?
            .into_iter()
            .map(|(id, _)| {
                (
                    id,
                    Answer::Noul {
                        noul: 0.9,
                        noul_raw: None,
                    },
                )
            })
            .collect();
        Ok(Response {
            model: "stub".into(),
            usage: Usage {
                input_tokens: 3,
                output_tokens: answers.len(),
            },
            answers,
            truncation: None,
        })
    }

    fn model(&self) -> &'static str {
        "stub"
    }
}

async fn start(app: axum::Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(rverdict_server::serve(listener, app));
    format!("http://{address}")
}

fn request(question: &serde_json::Value) -> Request {
    serde_json::from_value(
        json!({"state": "Refund my duplicate charge", "questions": {"q": question}}),
    )
    .unwrap()
}

#[tokio::test]
async fn answers_round_trip_and_malformed_questions_are_422() {
    let base = start(router(Arc::new(Stub), Options::default())).await;
    let client = RemoteClient::system_one(&base);

    let response = client
        .decide(&request(
            &json!({"type": "noul", "instructions": "Refund?"}),
        ))
        .await
        .unwrap();
    assert_eq!(
        response.answers.get("q"),
        Some(&Answer::Noul {
            noul: 0.9,
            noul_raw: None
        })
    );

    let malformed =
        request(&json!({"type": "score", "instructions": "How bad?", "criteria": ["only one"]}));
    assert!(matches!(
        client.decide(&malformed).await,
        Err(RemoteError::Invalid(_))
    ));
}

#[tokio::test]
async fn a_keyed_server_rejects_clients_without_the_key() {
    let options = Options {
        api_key: Some("secret".into()),
        ..Options::default()
    };
    let base = start(router(Arc::new(Stub), options)).await;
    let noul = request(&json!({"type": "noul", "instructions": "Refund?"}));

    let anonymous = RemoteClient::system_one(&base).decide(&noul).await;
    assert!(matches!(
        anonymous,
        Err(RemoteError::Status { status: 401, .. })
    ));
    assert!(
        RemoteClient::system_one(&base)
            .with_api_key("secret")
            .decide(&noul)
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn rate_limited_requests_are_retried() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let flaky = axum::Router::new().route(
        "/v1/systemone",
        axum::routing::post(move || {
            let counter = Arc::clone(&counter);
            async move {
                if counter.fetch_add(1, Ordering::SeqCst) == 0 {
                    return (axum::http::StatusCode::TOO_MANY_REQUESTS, [("retry-after", "0")], String::new());
                }
                let body = json!({"model": "m", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}});
                (axum::http::StatusCode::OK, [("retry-after", "0")], body.to_string())
            }
        }),
    );
    let base = start(flaky).await;
    let noul = request(&json!({"type": "noul", "instructions": "Refund?"}));
    assert!(RemoteClient::system_one(&base).decide(&noul).await.is_ok());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}
