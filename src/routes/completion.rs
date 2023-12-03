use async_stream::stream;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::Json;
use futures::stream::Stream;
use oxpilot::types::{Choice, Completion, CompletionRequest, Usage};
use serde_json::{json, to_string};
use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

pub async fn completion(
    Json(body): Json<CompletionRequest>,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    Sse::new(stream! {
      yield Ok(
        SseEvent::default().data(
          to_string(
            &json!(
              Completion {
                id: "cmpl-".to_string(),
                object: "text_completion".to_string(),
                created: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                model: body.model.unwrap_or("unknown".to_string()),
                choices: vec![Choice {
                    text: " world!".to_string(),
                    index: 0,
                    logprobs: None,
                    finish_reason: Some("stop".to_string()),
                }],
                usage: Usage {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0
                },
                system_fingerprint: "".to_string(),
              }
            )).unwrap()
          )
      );
    })
    .keep_alive(KeepAlive::default())
}
