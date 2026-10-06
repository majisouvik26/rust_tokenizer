use bpe::{BpeModel, EncodeOptions, RuntimeConfig, Tokenizer};
use wasm_bindgen::prelude::*;
fn error(e: bpe::TokenizerError) -> JsValue {
    JsValue::from_str(&e.to_string())
}
#[wasm_bindgen]
pub struct WasmTokenizer {
    inner: Tokenizer,
}
#[wasm_bindgen]
impl WasmTokenizer {
    #[wasm_bindgen(constructor)]
    pub fn new(model_json: &str) -> Result<WasmTokenizer, JsValue> {
        let model = BpeModel::from_json(model_json.as_bytes()).map_err(error)?;
        Ok(Self {
            inner: Tokenizer::new(model).map_err(error)?,
        })
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, JsValue> {
        self.inner.encode(text).map_err(error)
    }
    pub fn encode_with(&self, text: &str, backend: &str, heap_threshold: Option<u32>) -> Result<Vec<u32>, JsValue> {
        let options = EncodeOptions { backend: backend.parse().map_err(error)?, ..Default::default() };
        let config = RuntimeConfig { heap_threshold: heap_threshold.map(|value| value as usize), ..Default::default() };
        self.inner.session(config).map_err(error)?.encode(text, &options).map_err(error)
    }
    pub fn trace(&self, text: &str, backend: &str) -> Result<String, JsValue> {
        let options = EncodeOptions { backend: backend.parse().map_err(error)?, ..Default::default() };
        let trace = self.inner.trace(text, &options, &RuntimeConfig::default()).map_err(error)?;
        serde_json::to_string(&trace).map_err(|error| JsValue::from_str(&error.to_string()))
    }
    pub fn decode(&self, ids: &[u32]) -> Result<String, JsValue> {
        self.inner.decode_utf8(ids).map_err(error)
    }
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, JsValue> {
        self.inner.decode_bytes(ids).map_err(error)
    }
}
