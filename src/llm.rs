use anyhow::{anyhow, Context, Result};

/// In this file we are using the "Builder" pattern to create `LLM` struc instances. "Builder" pattern is a common design
/// pattern that used in Rust. It is a creational pattern that lets you construct complex objects step by step.
///
/// ```
/// use oxpilot::llm::LLMBuilder;
/// #[tokio::main]
/// async fn main() {
///    let llm_builder = LLMBuilder::new()
///         .tokenizer_repo_id("hf-internal-testing/llama-tokenizer")
///         .model_repo_id("TheBloke/CodeLlama-7B-GGU")
///         .model_file_name("codellama-7b.Q2_K.gguf");
///    let llm = llm_builder.build().await;
/// }
/// ```
/// See [The Ultimate Builder Pattern Tutorial](https://www.youtube.com/watch?v=Z_3WOSiYYFY)
/// See https://www.lurklurk.org/effective-rust/builders.html

pub struct LLM {
    pub tokenizer_repo_id: String,
    pub tokenizer_repo_revision: String,
    pub tokenizer_file_name: String,
    pub model_repo_id: String,
    pub model_repo_revision: String,
    pub model_file_name: String,
    pub model_weights: candle_transformers::models::quantized_llama::ModelWeights,
    pub tokenizer: tokenizers::Tokenizer,
}

/// `Default` is a trait for giving a type a useful default value.
/// https://doc.rust-lang.org/std/default/trait.Default.html
/// `Option<String>` would be `None` by default with `#[derive(Default)]`
///
/// `PartialEq` is a trait for equality comparisons that is used in tests.
/// https://doc.rust-lang.org/std/cmp/trait.PartialEq.html
/// e.g. `assert!(LLMBuilder::default() == LLMBuilder::new())`
#[derive(Default, PartialEq)]
pub struct LLMBuilder<State> {
    tokenizer_repo_id: Option<String>,
    tokenizer_repo_revision: Option<String>,
    tokenizer_file_name: Option<String>,
    model_repo_id: Option<String>,
    model_repo_revision: Option<String>,
    model_file_name: Option<String>,
    state: State,
}

/// Type state for LLMBuilder
/// Init state when `::new()` is called.
/// 
/// `pub struct StrucName` is a unit struct. A unit struct is a struct that has no fields.
/// They are most commonly used as marker types.
/// 
/// Common struct types in Rust:
///   - Unit struct: `struct Unit;`
///   - Classic struct: `struct Classic { a: i32, b: f32 }` Each field in the struct has a name and a data type. 
///     After a classic struct is defined, the fields in the struct can be accessed by using the syntax <struct>.<field>.
///   - Tuple struct: `struct Tuple(i32, f32);` are similar to classic structs, but the fields don't have names To access 
///     the fields in a tuple struct: <tuple>.<index>. The index values in the tuple struct start at zero.
#[derive(PartialEq)]
pub struct InitState;

/// Intermedia state, with token repo id, ready to accept model repo id
#[derive(PartialEq)]
pub struct WithTokenizerRepoId;

/// Intermedia state, with model repo id, ready to accept model file name
#[derive(PartialEq)]
pub struct WithModelRepoId;

/// With both token repo id and model repo id
#[derive(PartialEq)]
pub struct ReadyState;

impl LLMBuilder<InitState> {
    pub fn new() -> Self {
        LLMBuilder {
            tokenizer_repo_id: None,
            tokenizer_repo_revision: None,
            tokenizer_file_name: None,
            model_repo_id: None,
            model_repo_revision: None,
            model_file_name: None,
            state: InitState,
        }
    }
}

impl<State> LLMBuilder<State> {
    // Here we can add methods that are common to all states.
}

// The initial state
impl LLMBuilder<InitState> {
    /// Should function parameter be `String` or `&str`....or both?
    /// Short answer: `impl Into<String>` is preferred as it allows both `&str` and `String`.
    ///
    /// <https://rhai.rs/book/rust/strings.html> suggests that The parameter type `String` involves always converting an
    /// `ImmutableString` into a String which mandates cloning it.
    /// Using `ImmutableString` or `&str` is much more efficient.
    ///
    /// <https://hermanradtke.com/2015/05/03/string-vs-str-in-rust-functions.html/#introducing-struct> suggests that
    /// `&str` is preferred over `String` because it's more flexible, however, we need lifetime specifiers in `LLM` when using `&str`.
    ///
    /// If it's a public API, using `impl Into<String>` or `AsRef` make it nicer for users of your API, so they can pass both `&str` and `String`.
    ///
    /// If you need to own the string, use `impl Into<String>`.
    /// ```
    /// struct LLMBuilder {}
    /// impl LLMBuilder {
    ///    pub fn tokenizer_repo_id(mut self, tokenizer_repo_id: impl Into<String>) -> Self { self }
    /// }
    /// ```
    /// If it's ok to borrow the string, use `AsRef`, but you need to specify the lifetime.
    /// https://doc.rust-lang.org/std/convert/trait.AsRef.html#examples
    /// ```
    /// struct LLMBuilder {}
    /// impl LLMBuilder {
    ///    pub fn tokenizer_repo_id<T: AsRef<str>>(mut self, tokenizer_repo_id: T) -> Self { self }
    /// }
    /// ```
    /// Which is more idiomatic?
    /// https://users.rust-lang.org/t/idiomatic-string-parmeter-types-str-vs-asref-str-vs-into-string/7934
    pub fn tokenizer_repo_id(
        self,
        tokenizer_repo_id: impl Into<String>,
    ) -> LLMBuilder<WithTokenizerRepoId> {
        LLMBuilder {
            tokenizer_repo_id: Some(tokenizer_repo_id.into()),
            tokenizer_repo_revision: self.tokenizer_repo_revision,
            tokenizer_file_name: self.tokenizer_file_name,
            model_repo_id: self.model_repo_id,
            model_repo_revision: self.model_repo_revision,
            model_file_name: self.model_file_name,
            state: WithTokenizerRepoId,
        }
    }
}

impl LLMBuilder<WithTokenizerRepoId> {
    pub fn model_repo_id(self, model_repo_id: impl Into<String>) -> LLMBuilder<WithModelRepoId> {
        LLMBuilder {
            tokenizer_repo_id: self.tokenizer_repo_id,
            tokenizer_repo_revision: self.tokenizer_repo_revision,
            tokenizer_file_name: self.tokenizer_file_name,
            model_repo_id: Some(model_repo_id.into()),
            model_repo_revision: self.model_repo_revision,
            model_file_name: self.model_file_name,
            state: WithModelRepoId,
        }
    }
}

impl LLMBuilder<WithModelRepoId> {
    pub fn model_file_name(self, model_file_name: impl Into<String>) -> LLMBuilder<ReadyState> {
        LLMBuilder {
            tokenizer_repo_id: self.tokenizer_repo_id,
            tokenizer_repo_revision: self.tokenizer_repo_revision,
            tokenizer_file_name: self.tokenizer_file_name,
            model_repo_id: self.model_repo_id,
            model_repo_revision: self.model_repo_revision,
            model_file_name: Some(model_file_name.into()),
            state: ReadyState,
        }
    }
}

impl LLMBuilder<ReadyState> {
    pub fn tokenizer_repo_revision(mut self, tokenizer_repo_revision: impl Into<String>) -> Self {
        // `into()` is a trait that converts the value of one type into the value of another type.
        // Since `tokenizer_repo_revision: impl Into<String>` we will convert a string slice into a String.
        // and if we pass a String, no allocation will happenend.
        self.tokenizer_repo_revision = Some(tokenizer_repo_revision.into());
        self
    }
    pub fn tokenizer_file_name(mut self, tokenizer_file_name: impl Into<String>) -> Self {
        self.tokenizer_file_name = Some(tokenizer_file_name.into());
        self
    }

    pub fn model_repo_revision(mut self, model_repo_revision: impl Into<String>) -> Self {
        self.model_repo_revision = Some(model_repo_revision.into());
        self
    }

    pub async fn build(self) -> Result<LLM> {
        let tokenizer_repo_id = self
            .tokenizer_repo_id
            .context("tokenizer_repo_id is None, forgot to .tokenizer_repo_id()?")?;
        let tokenizer_repo_revision = self.tokenizer_repo_revision.unwrap_or("main".to_string());
        let tokenizer_file_name = self
            .tokenizer_file_name
            .unwrap_or("tokenizer.json".to_string());

        let hf_hub_api = match hf_hub::api::tokio::Api::new() {
            Ok(hf_hub) => hf_hub,
            Err(error) => {
                return Result::Err(anyhow!(
                    "hf_hub_api initialization failed because of {:?}",
                    error
                ));
            }
        };

        let tokenizer_repo = hf_hub_api.repo(hf_hub::Repo::with_revision(
            tokenizer_repo_id.clone(),
            hf_hub::RepoType::Model,
            tokenizer_repo_revision.clone(),
        ));
        let tokenizer_file_path = tokenizer_repo
            .get(&tokenizer_file_name)
            .await
            .context("Failed to fetch tokenizer file")?;
        let tokenizer = match tokenizers::Tokenizer::from_file(tokenizer_file_path) {
            Ok(tokenizer) => tokenizer,
            Err(error) => {
                return Result::Err(anyhow!("Failed to init Tokenizer from file {:?}", error))
            }
        };

        let model_repo_id = self
            .model_repo_id
            .context("model_repo_id is None, forgot to .model_repo_id()?")?;
        let model_repo_revision = self.model_repo_revision.unwrap_or("main".to_string());
        let model_file_name = self
            .model_file_name
            .context("model_file_name is None, forgot to .model_file_name()?")?;

        let model_repo = hf_hub_api.repo(hf_hub::Repo::with_revision(
            model_repo_id.clone(),
            hf_hub::RepoType::Model,
            model_repo_revision.clone(),
        ));
        let model_file_path = model_repo
            .get(&model_file_name)
            .await
            .context("Failed to fetch model file")?;
        let mut model_file =
            std::fs::File::open(model_file_path).context("Failed to open model file")?;
        let model_content = candle_core::quantized::gguf_file::Content::read(&mut model_file)
            .context("gguf file read failed")?;
        let model_weights = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
            model_content,
            &mut model_file,
        )
        .context("Failed creating model weights from gguf")?;

        Ok(LLM {
            tokenizer_repo_id,
            tokenizer_repo_revision,
            tokenizer_file_name,
            tokenizer,
            model_repo_id,
            model_repo_revision,
            model_file_name,
            model_weights,
        })
    }
}
#[cfg(test)]
mod llm_builder_tests {
    use super::*;

    #[tokio::test]
    async fn can_accept_string() {
        let _ = LLMBuilder::new()
            // mandatory parameters
            .tokenizer_repo_id(String::from("repo"))
            .model_repo_id(String::from("repo"))
            .model_file_name(String::from("model.file"))
            // optional parameters
            .tokenizer_repo_revision(String::from("main"))
            .tokenizer_file_name(String::from("tokenizer.json"))
            .model_repo_revision(String::from("main"));
    }

    #[tokio::test]
    async fn can_accept_string_slice() {
        let _ = LLMBuilder::new()
            // mandatory parameters
            .tokenizer_repo_id("string_slice")
            .model_repo_id("repo")
            .model_file_name("model.file")
            // optional parameters
            .tokenizer_repo_revision("main")
            .tokenizer_file_name("tokenizer.json")
            .model_repo_revision("main");
    }

    #[tokio::test]
    async fn can_access_values_before_build() {
        let builder_init_state = LLMBuilder::new();
        assert!(builder_init_state.state == InitState);
        let with_token_repo_id_state = builder_init_state.tokenizer_repo_id("token_repo_id");
        let with_model_repo_id_state = with_token_repo_id_state.model_repo_id("model_repo_id");
        let ready_state = with_model_repo_id_state.model_file_name("model.file");
        assert!(ready_state.tokenizer_repo_id.unwrap() == "token_repo_id");
        assert!(ready_state.model_repo_id.unwrap() == "model_repo_id");
        assert!(ready_state.model_file_name.unwrap() == "model.file");
        // all none when not set
        assert!(ready_state.tokenizer_file_name.is_none());
        assert!(ready_state.tokenizer_repo_revision.is_none());
        assert!(ready_state.model_repo_revision.is_none());
    }
}
