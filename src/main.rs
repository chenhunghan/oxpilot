use oxpilot::llm::LLMBuilder;

#[tokio::main]
async fn main() {
    let llm_builder = LLMBuilder::new()
        .tokenizer_repo_id("hf-internal-testing/llama-tokenizer")
        .model_repo_id("TheBloke/CodeLlama-7B-GGU")
        .model_file_name("codellama-7b.Q2_K.gguf");
    let _ = llm_builder.build().await;
}
