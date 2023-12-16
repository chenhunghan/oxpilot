/// returns a string of the prompt to be sent to the Mistral Instruct Models
/// e.g. `<s>[INST] {instruction} [/INST]`
/// see https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
pub fn instruct(instruction: impl Into<String>) -> String {
    let instruction = instruction.into();
    format!("<s>[INST] {} [/INST] ", instruction)
}
