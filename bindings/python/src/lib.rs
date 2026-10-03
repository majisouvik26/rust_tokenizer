use bpe::{EncodeOptions, SpecialMode, Tokenizer};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
fn error(e: bpe::TokenizerError) -> PyErr {
    PyValueError::new_err(e.to_string())
}
fn options(allowed: Option<Vec<String>>, reject_special: bool) -> PyResult<EncodeOptions> {
    if reject_special && allowed.is_some() {
        return Err(PyValueError::new_err(
            "allowed_special and reject_special are mutually exclusive",
        ));
    }
    Ok(EncodeOptions {
        special: if reject_special {
            SpecialMode::Reject
        } else if let Some(names) = allowed {
            SpecialMode::Allow(names.into_iter().collect())
        } else {
            SpecialMode::Ordinary
        },
        ..Default::default()
    })
}
#[pyclass(name = "Tokenizer")]
struct PyTokenizer {
    inner: Tokenizer,
}
#[pymethods]
impl PyTokenizer {
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Tokenizer::load(model_path).map_err(error)?,
        })
    }
    #[pyo3(signature = (text, allowed_special=None, reject_special=false))]
    fn encode(
        &self,
        text: &str,
        allowed_special: Option<Vec<String>>,
        reject_special: bool,
    ) -> PyResult<Vec<u32>> {
        self.inner
            .encode_with(text, &options(allowed_special, reject_special)?)
            .map_err(error)
    }
    #[pyo3(signature = (texts, allowed_special=None, reject_special=false))]
    fn encode_batch(
        &self,
        texts: Vec<String>,
        allowed_special: Option<Vec<String>>,
        reject_special: bool,
    ) -> PyResult<Vec<Vec<u32>>> {
        self.inner
            .encode_batch(&texts, &options(allowed_special, reject_special)?)
            .map_err(error)
    }
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.inner.decode_utf8(&ids).map_err(error)
    }
    fn decode_bytes<'py>(&self, py: Python<'py>, ids: Vec<u32>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(
            py,
            &self.inner.decode_bytes(&ids).map_err(error)?,
        ))
    }
    fn model_sha256(&self) -> PyResult<String> {
        self.inner.model().sha256().map_err(error)
    }
}
#[pymodule]
fn rust_tokenizer(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTokenizer>()?;
    Ok(())
}
