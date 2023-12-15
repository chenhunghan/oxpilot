use crate::llm::LLM;
use crate::token::token_to_text;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;

/// A function that takes a prompt and returns the generated text to a responder.
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
    eos_token: String,
    max_sampled: usize,
) {
    let tokens = llm
        .tokenizer
        .encode(prompt, true)
        .expect("Failed to encode prompt as tokens.");
    let prompt_tokens = tokens.get_ids().to_vec();
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
    responder
        .send((token_to_text(next_token, &llm.tokenizer)).to_string())
        .await
        .unwrap();

    let eos_token_id = *llm.tokenizer.get_vocab(true).get(&eos_token).unwrap();

    let mut sampled = 0;
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
        sampled += 1;
        if next_token == 32000 {
            break;
        };
        if sampled >= max_sampled {
            break;
        }
        if next_token == eos_token_id {
            break;
        }
        all_tokens.push(next_token);
        let text = token_to_text(next_token, &llm.tokenizer);
        responder.send((text).to_string()).await.unwrap();
    }
}
