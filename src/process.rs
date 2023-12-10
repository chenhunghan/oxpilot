use crate::llm::LLM;
use crate::token::token_to_text;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::MAX_SEQ_LEN;

/// A function that takes a prompt and returns the generated text to a responder.
/// ```no_run
/// use tokio::sync::mpsc;
/// use oxpilot::process::process;
///  
/// let mut llm = llm_builder.build().await.expect("Failed to build LLM");
/// let (sender, mut receiver) = mpsc::channel(32);
/// let manager = tokio::spawn(async move {
///   while let Some(cmd) = receiver.recv().await {
///     let seed = 299792458;
///     let temperature: f64 = 1.0;
///     let top_p = 1.1;
///     let n = 10;
///     let repeat_last_n = 64;
///     let repeat_penalty = 1.1;
///     match cmd {
///         cmd::Command::Prompt { prompt, responder } => {
///             process(
///               prompt,
///               &mut llm,
///               responder,
///               n,
///               seed,
///               temperature,
///               top_p,
///               repeat_last_n,
///               repeat_penalty,
///          ).await;
///       }
///     }
///   }
/// });
///
/// ```
///
pub async fn process(
    prompt: String,
    llm: &mut LLM,
    responder: tokio::sync::mpsc::Sender<String>,
    to_sample: usize,
    seed: u64,
    temperature: f64,
    top_p: Option<f64>,
    repeat_last_n: usize,
    repeat_penalty: f32,
) {
    let tokens = match llm.tokenizer.encode(prompt, true) {
        Ok(tokens) => tokens,
        Err(_) => {
            todo!();
        }
    };
    let pre_prompt_tokens = vec![];
    let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();
    let prompt_tokens = if prompt_tokens.len() + to_sample > MAX_SEQ_LEN - 10 {
        let to_remove = prompt_tokens.len() + to_sample + 10 - MAX_SEQ_LEN;
        prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
    } else {
        prompt_tokens
    };
    let mut all_tokens: Vec<u32> = vec![];
    let mut logits_processor = LogitsProcessor::new(seed, Some(temperature), top_p);
    let mut next_token = {
        let input = Tensor::new(prompt_tokens.as_slice(), &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let logits = llm.model_weights.forward(&input, 0).unwrap();
        let logits = logits.squeeze(0).unwrap();
        logits_processor.sample(&logits).unwrap()
    };
    all_tokens.push(next_token);
    for index in 0..to_sample {
        let input = Tensor::new(&[next_token], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let logits = llm
            .model_weights
            .forward(&input, prompt_tokens.len() + index)
            .unwrap();
        let logits = logits.squeeze(0).unwrap();
        let start_at = all_tokens.len().saturating_sub(repeat_last_n);
        let _ = candle_transformers::utils::apply_repeat_penalty(
            &logits,
            repeat_penalty,
            &all_tokens[start_at..],
        );
        next_token = logits_processor.sample(&logits).unwrap();
        all_tokens.push(next_token);
        let text = token_to_text(next_token, &llm.tokenizer);
        responder.send((text).to_string()).await.unwrap();
    }
}
