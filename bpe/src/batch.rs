use crate::{
    CacheStats, EncodeOptions, Result, RuntimeConfig, TokenId, Tokenizer, TokenizerError,
    WorkerState,
};
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
use rayon::prelude::*;
use std::sync::Mutex;

/// Persistent native pool with one bounded state per deterministic input lane.
/// Pool creation is explicit, so steady-state timing can exclude it.
pub struct BatchEncoder<'a> {
    tokenizer: &'a Tokenizer,
    config: RuntimeConfig,
    workers: Vec<Mutex<WorkerState>>,
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    pool: Option<rayon::ThreadPool>,
}
impl<'a> BatchEncoder<'a> {
    pub(crate) fn new(
        tokenizer: &'a Tokenizer,
        threads: usize,
        config: RuntimeConfig,
    ) -> Result<Self> {
        config.validate()?;
        if threads == 0 {
            return Err(TokenizerError::InvalidConfig(
                "threads must be positive".into(),
            ));
        }
        #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
        if threads != 1 {
            return Err(TokenizerError::InvalidConfig(
                "multiple workers require the native parallel feature".into(),
            ));
        }
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        let pool = if threads > 1 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|error| TokenizerError::InvalidConfig(error.to_string()))?,
            )
        } else {
            None
        };
        Ok(Self {
            tokenizer,
            config,
            workers: (0..threads)
                .map(|_| Mutex::new(WorkerState::default()))
                .collect(),
            #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
            pool,
        })
    }
    pub fn threads(&self) -> usize {
        self.workers.len()
    }
    pub fn encode<S: AsRef<str> + Sync>(
        &mut self,
        texts: &[S],
        options: &EncodeOptions,
    ) -> Result<Vec<Vec<TokenId>>> {
        if texts.is_empty() {
            // Validate special allow lists even for an empty request.
            self.tokenizer.encode_with("", options)?;
            return Ok(Vec::new());
        }
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        if let Some(pool) = &self.pool {
            let width = texts.len().div_ceil(self.workers.len());
            let results: Vec<Result<Vec<Vec<TokenId>>>> = pool.install(|| {
                texts
                    .par_chunks(width)
                    .enumerate()
                    .map(|(lane, chunk)| {
                        let mut state = self.workers[lane]
                            .lock()
                            .map_err(|_| TokenizerError::WorkerState)?;
                        chunk
                            .iter()
                            .map(|text| {
                                self.tokenizer.encode_state(
                                    text.as_ref(),
                                    options,
                                    &self.config,
                                    &mut state,
                                )
                            })
                            .collect()
                    })
                    .collect()
            });
            // Indexed parallel collection preserves input/error order.
            return Ok(results
                .into_iter()
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect());
        }
        let mut state = self.workers[0]
            .lock()
            .map_err(|_| TokenizerError::WorkerState)?;
        texts
            .iter()
            .map(|text| {
                self.tokenizer
                    .encode_state(text.as_ref(), options, &self.config, &mut state)
            })
            .collect()
    }
    pub fn clear_cache(&mut self) -> Result<()> {
        for worker in &self.workers {
            worker
                .lock()
                .map_err(|_| TokenizerError::WorkerState)?
                .clear_cache();
        }
        Ok(())
    }
    pub fn cache_stats(&self) -> Result<Vec<CacheStats>> {
        self.workers
            .iter()
            .map(|worker| {
                Ok(worker
                    .lock()
                    .map_err(|_| TokenizerError::WorkerState)?
                    .stats())
            })
            .collect()
    }
}
