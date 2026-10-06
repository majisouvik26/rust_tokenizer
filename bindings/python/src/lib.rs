use bpe::{Backend, EncodeOptions, RuntimeConfig, SpecialMode, Tokenizer};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
fn error(e: bpe::TokenizerError) -> PyErr { PyValueError::new_err(e.to_string()) }
fn options(allowed: Option<Vec<String>>, reject_special: bool, backend: &str) -> PyResult<EncodeOptions> {
    if reject_special && allowed.is_some() {
        return Err(PyValueError::new_err("allowed_special and reject_special are mutually exclusive"));
    }
    Ok(EncodeOptions {
        backend: backend.parse::<Backend>().map_err(error)?,
        special: if reject_special { SpecialMode::Reject }
            else if let Some(names) = allowed { SpecialMode::Allow(names.into_iter().collect()) }
            else { SpecialMode::Ordinary },
    })
}
fn config(heap_threshold: Option<usize>, cache_capacity: usize, cache_bytes: usize) -> RuntimeConfig {
    RuntimeConfig { heap_threshold, cache_capacity, cache_bytes, ..Default::default() }
}
#[pyclass(name = "Tokenizer")]
struct PyTokenizer { inner: Tokenizer }
#[pymethods]
impl PyTokenizer {
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        Ok(Self { inner: Tokenizer::load(model_path).map_err(error)? })
    }
    #[allow(clippy::too_many_arguments)] // Explicit Python keyword options.
    #[pyo3(signature = (text, allowed_special=None, reject_special=false, *, backend="reference", heap_threshold=None, cache_capacity=0, cache_bytes=8388608))]
    fn encode(&self, py: Python<'_>, text: String, allowed_special: Option<Vec<String>>, reject_special: bool, backend: &str, heap_threshold: Option<usize>, cache_capacity: usize, cache_bytes: usize) -> PyResult<Vec<u32>> {
        let options = options(allowed_special, reject_special, backend)?;
        let config = config(heap_threshold, cache_capacity, cache_bytes);
        py.allow_threads(|| self.inner.session(config)?.encode(&text, &options)).map_err(error)
    }
    #[allow(clippy::too_many_arguments)] // Explicit Python keyword options.
    #[pyo3(signature = (texts, allowed_special=None, reject_special=false, *, backend="reference", threads=1, heap_threshold=None, cache_capacity=0, cache_bytes=8388608))]
    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>, allowed_special: Option<Vec<String>>, reject_special: bool, backend: &str, threads: usize, heap_threshold: Option<usize>, cache_capacity: usize, cache_bytes: usize) -> PyResult<Vec<Vec<u32>>> {
        let options = options(allowed_special, reject_special, backend)?;
        let config = config(heap_threshold, cache_capacity, cache_bytes);
        // Both pool setup and encoding release the GIL. This convenience API
        // creates workers/cache per call; caches do not survive Python requests.
        py.allow_threads(|| self.inner.batch_encoder(threads, config)?.encode(&texts, &options)).map_err(error)
    }
    #[pyo3(signature = (text, *, backend="reference", heap_threshold=None))]
    fn trace(&self, py: Python<'_>, text: String, backend: &str, heap_threshold: Option<usize>) -> PyResult<String> {
        let options = options(None, false, backend)?;
        let trace = py.allow_threads(|| self.inner.trace(&text, &options, &config(heap_threshold, 0, 8388608))).map_err(error)?;
        serde_json::to_string(&trace).map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> { self.inner.decode_utf8(&ids).map_err(error) }
    fn decode_bytes<'py>(&self, py: Python<'py>, ids: Vec<u32>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.inner.decode_bytes(&ids).map_err(error)?))
    }
    fn model_sha256(&self) -> PyResult<String> { self.inner.model().sha256().map_err(error) }
}
#[pymodule]
fn rust_tokenizer(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTokenizer>()?;
    Ok(())
}
