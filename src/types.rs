use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// Acknowledgements:
// https://github.com/AmineDiro/cria/blob/main/src/routes/completions.rs
// https://github.com/64bit/async-openai/blob/main/async-openai/src/types/types.rs

/// Represents a OpenAI completion response from the API. Note: both the streamed and non-streamed
/// response objects share the same shape (unlike the chat endpoint).
/// https://platform.openai.com/docs/api-reference/completions
#[derive(Debug, Serialize, Deserialize)]
pub struct Completion {
    /// A unique identifier for the completion.
    pub id: String,
    /// The list of completion choices the model generated for the input prompt.
    pub choices: Vec<Choice>,
    /// The Unix timestamp (in seconds) of when the completion was created.
    pub created: u64,
    /// The model used for completion.
    pub model: String,
    /// This fingerprint represents the backend configuration that the model runs with.
    ///
    /// Can be used in conjunction with the `seed` request parameter to understand when backend changes have been
    /// made that might impact determinism.
    pub system_fingerprint: String,
    /// The object type, which is always "text_completion"
    pub object: String,
    /// Usage statistics for the completion request.
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Choice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<Logprobs>,
    // The reason the model stopped generating tokens. This will be stop if the model hit a
    // natural stop point or a provided stop sequence, length if the maximum number of tokens
    // specified in the request was reached, or content_filter if content was omitted due to a
    // flag from our content filters.
    pub finish_reason: Option<String>,
}

/// Usage statistics for the completion request.
#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,
    /// Number of tokens in the generated completion.
    pub completion_tokens: usize,
    /// Total number of tokens used in the request (prompt + completion).
    pub total_tokens: usize,
}

/// Not well-documented in OpenAI doc https://platform.openai.com/docs/api-reference/completions/object
#[derive(Debug, Serialize, Deserialize)]
pub struct Logprobs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<Option<usize>>,
    pub top_logprobs: Vec<serde_json::Value>,
    pub text_offset: Vec<usize>,
}

#[derive(Deserialize, Debug)]
pub enum LogitBias {
    TokenIds,
    Tokens,
}

/// The request body for the completion endpoint.
/// Only makes `prompt` and `model` required, the rest are optional.
/// (so we can observe what passes from the copilot clients)
#[derive(Deserialize, Debug)]
pub struct CompletionRequest {
    pub prompt: Option<String>,
    pub model: Option<String>,
    pub suffix: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub mirostat_mode: Option<usize>,
    pub mirostat_tau: Option<f32>,
    pub mirostat_eta: Option<f32>,
    pub echo: Option<bool>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
    pub logprobs: Option<usize>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub logit_bias: Option<HashMap<String, f32>>,
    pub top_k: Option<usize>,
    pub repeat_penalty: Option<f32>,
    pub last_n_tokens: Option<usize>,
    pub logit_bias_type: Option<LogitBias>,
    pub n: Option<usize>,
    pub best_of: Option<usize>,
    pub seed: Option<u64>,
    pub user: Option<String>,
}
