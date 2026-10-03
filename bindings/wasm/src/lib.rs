use bpe::{BpeModel, Tokenizer};
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
    pub fn decode(&self, ids: &[u32]) -> Result<String, JsValue> {
        self.inner.decode_utf8(ids).map_err(error)
    }
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, JsValue> {
        self.inner.decode_bytes(ids).map_err(error)
    }
}
