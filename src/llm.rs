use anyhow::{anyhow, Context, Result};

/// In this file we are using the "Builder" pattern to create `LLM` struc instances. "Builder" pattern is a common design
/// pattern that used in Rust. It is a creational pattern that lets you construct complex objects step by step.
///
/// ```
/// use oxpilot::llm::LLMBuilder;
/// #[tokio::main]
/// async fn main() {
///    let llm_builder = LLMBuilder::default()
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

pub struct LLMBuilder {
    hf_hub_api: Option<hf_hub::api::tokio::Api>,
    tokenizer_repo_id: Option<String>,
    tokenizer_repo_revision: String,
    tokenizer_file_name: String,
    model_repo_id: Option<String>,
    model_repo_revision: String,
    model_file_name: Option<String>,
}

/// `Default` is a trait for giving a type a useful default value.
/// https://doc.rust-lang.org/std/default/trait.Default.html
impl Default for LLMBuilder {
    /// Rust doesn't have built-in support for static constructors, but you can implement the Default trait for
    /// a type and use the default function to create a new instance. See <https://doc.rust-lang.org/nomicon/constructors.html>
    ///
    /// If the struct needs to be initialized with some parameters, the convention is to use an `new` to create an object
    /// <https://rust-unofficial.github.io/patterns/idioms/ctor.html>
    ///
    /// Example:
    /// ```
    /// struct LLMBuilder { hf_hub_api: Option<hf_hub::api::tokio::Api> }
    /// impl LLMBuilder {
    ///    pub fn new(hf_hub_api: hf_hub::api::tokio::Api) -> Self {
    ///       Self { hf_hub_api: Some(hf_hub_api) }
    ///   }
    /// }
    /// ```
    /// In our case, we don't need to initialize the LLMBuilder struct with any parameters, so we can use the Default trait.
    ///
    /// Example:
    /// ```
    /// struct LLMBuilder {}
    /// impl Default for LLMBuilder {
    ///   fn default() -> Self { Self {} }
    /// }
    /// let llm = LLMBuilder::default();
    /// ````
    fn default() -> Self {
        let hf_hub_api = match hf_hub::api::tokio::Api::new() {
            Ok(hf_hub) => Some(hf_hub),
            Err(_error) => None,
        };
        Self {
            hf_hub_api,
            tokenizer_repo_id: None,
            tokenizer_repo_revision: "main".to_string(),
            tokenizer_file_name: "tokenizer.json".to_string(),
            model_repo_id: None,
            model_repo_revision: "main".to_string(),
            model_file_name: None,
        }
    }
}

impl LLMBuilder {
    /// Same as `LLMBuilder::default()`, for those who prefer `LLMBuilder::new()`.
    pub fn new() -> Self {
        Self::default()
    }
    /// Should function parameter be `String` or `&str`....or both?
    /// Short answer: `impl Into<String>` is preferred as it allows both `&str` and `String`.
    ///
    /// ```
    /// use oxpilot::llm::LLMBuilder;
    /// let repo_id_string_slice = "string_slice";
    /// let _ = LLMBuilder::default().tokenizer_repo_id(repo_id_string_slice);
    /// let repo_id_string = String::from("String");
    /// let _ = LLMBuilder::default().tokenizer_repo_id(repo_id_string);
    /// ```
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
    pub fn tokenizer_repo_id(mut self, tokenizer_repo_id: impl Into<String>) -> Self {
        self.tokenizer_repo_id = Some(tokenizer_repo_id.into());
        self
    }

    pub fn tokenizer_repo_revision(mut self, tokenizer_repo_revision: impl Into<String>) -> Self {
        // `into()` is a trait that converts the value of one type into the value of another type.
        // Since `tokenizer_repo_revision: impl Into<String>` we will convert a string slice into a String.
        // and if we pass a String, no allocation will happenend.
        self.tokenizer_repo_revision = tokenizer_repo_revision.into();
        self
    }
    pub fn tokenizer_file_name(mut self, tokenizer_file_name: impl Into<String>) -> Self {
        self.tokenizer_file_name = tokenizer_file_name.into();
        self
    }

    pub fn model_repo_id(mut self, model_repo_id: impl Into<String>) -> Self {
        self.model_repo_id = Some(model_repo_id.into());
        self
    }

    pub fn model_repo_revision(mut self, model_repo_revision: impl Into<String>) -> Self {
        self.model_repo_revision = model_repo_revision.into();
        self
    }

    pub fn model_file_name(mut self, model_file_name: impl Into<String>) -> Self {
        self.model_file_name = Some(model_file_name.into());
        self
    }

    pub async fn build(self) -> Result<LLM> {
        let tokenizer_repo_id = self
            .tokenizer_repo_id
            .context("tokenizer_repo_id is None, forgot to .tokenizer_repo_id()?")?;
        let tokenizer_repo_revision = self.tokenizer_repo_revision;
        let tokenizer_file_name = self.tokenizer_file_name;
        let model_repo_id = self
            .model_repo_id
            .context("model_repo_id is None, forgot to .model_repo_id()?")?;
        let model_repo_revision = self.model_repo_revision;
        let model_file_name = self
            .model_file_name
            .context("model_file_name is None, forgot to .model_file_name()?")?;

        // The `hf_hub_api` is created in the `default()` is unlikely to fail, but if it does,
        // we will retry and return an error this time.
        let hf_hub_api = match self.hf_hub_api {
            Some(hf_hub_api) => hf_hub_api,
            None => {
                // Let's try again ton `new()` the hf_hub_api
                match hf_hub::api::tokio::Api::new() {
                    Ok(hf_hub) => hf_hub,
                    Err(error) => {
                        // Let's return an error this time.
                        // `anyhow!` is a macro that creates an error type that implements the Error trait.
                        // see https://github.com/dtolnay/anyhow/blob/master/src/macros.rs
                        return Result::Err(anyhow!(
                            "hf_hub_api initialization failed because of {:?}",
                            error
                        ));
                    }
                }
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

#[tokio::test]
async fn builder_test() {
    let repo_id_string_slice = "string_slice";
    let _ = LLMBuilder::default().tokenizer_repo_id(repo_id_string_slice);
    let repo_id_string = String::from("String");
    let _ = LLMBuilder::default().tokenizer_repo_id(repo_id_string);
}
