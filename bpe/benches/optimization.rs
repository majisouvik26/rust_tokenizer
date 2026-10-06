//! Explicit process worker; the Python driver owns manifests and run ordering.
use bpe::{Backend, EncodeOptions, RuntimeConfig, Tokenizer};
use serde::Deserialize;
use std::{alloc::{GlobalAlloc, Layout, System}, env, fs, hint::black_box, sync::atomic::{AtomicBool, AtomicU64, Ordering}, time::Instant};

struct CountingAllocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
fn count(bytes: usize) {
    if COUNTING.load(Ordering::Relaxed) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    }
}
// Benchmark-only instrumentation delegates every allocation to the system.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 { count(layout.size()); System.alloc(layout) }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 { count(layout.size()); System.alloc_zeroed(layout) }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) { System.dealloc(ptr, layout) }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 { count(size); System.realloc(ptr, layout, size) }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    model: String,
    input: String,
    backend: Backend,
    runtime: RuntimeConfig,
    threads: usize,
    batch_size: usize,
    iterations: usize,
    cache_state: String,
    profile: bool,
}
#[derive(Deserialize)]
struct Record { text: String, ids: Vec<u32> }
fn validate(records: &[Record], outputs: &[Vec<Vec<u32>>]) {
    let actual: Vec<_> = outputs.iter().flatten().collect();
    assert_eq!(records.len(), actual.len());
    for (index, (record, ids)) in records.iter().zip(actual).enumerate() {
        assert_eq!(&record.ids, ids, "full ID mismatch at record {index}");
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request: Request = serde_json::from_slice(&fs::read(env::var("BPE_BENCH_REQUEST")?)?)?;
    if request.iterations == 0 || request.batch_size == 0 { return Err("positive iterations and batch size required".into()); }
    if !["empty_each_sweep", "warm"].contains(&request.cache_state.as_str()) { return Err("invalid cache_state".into()); }
    let records: Vec<Record> = fs::read_to_string(&request.input)?.lines().map(serde_json::from_str).collect::<Result<_, _>>()?;
    if records.is_empty() { return Err("nonempty workload required".into()); }
    let start = Instant::now();
    let tokenizer = Tokenizer::load(&request.model)?;
    let load_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut encoder = tokenizer.batch_encoder(request.threads, request.runtime.clone())?;
    let pool_setup_seconds = start.elapsed().as_secs_f64();
    let texts: Vec<_> = records.iter().map(|row| row.text.as_str()).collect();
    let options = EncodeOptions { backend: request.backend, ..Default::default() };
    for row in &records {
        assert_eq!(tokenizer.encode(&row.text)?, row.ids, "reference fixture mismatch");
        assert_eq!(tokenizer.decode_bytes(&row.ids)?, row.text.as_bytes(), "byte mismatch");
    }
    let warmup: Vec<_> = texts.chunks(request.batch_size).map(|batch| encoder.encode(batch, &options)).collect::<bpe::Result<_>>()?;
    validate(&records, &warmup);
    drop(warmup);
    let mut encode_seconds = Vec::new();
    let mut batch_seconds = Vec::new();
    for _ in 0..request.iterations {
        if request.cache_state == "empty_each_sweep" { encoder.clear_cache()?; }
        let mut outputs = Vec::with_capacity(texts.len().div_ceil(request.batch_size));
        let mut latencies = Vec::with_capacity(outputs.capacity());
        let start = Instant::now();
        for batch in texts.chunks(request.batch_size) {
            let call = Instant::now();
            let ids = encoder.encode(black_box(batch), &options)?;
            latencies.push(call.elapsed().as_secs_f64());
            outputs.push(ids);
        }
        black_box(&outputs);
        encode_seconds.push(start.elapsed().as_secs_f64());
        batch_seconds.push(latencies);
        validate(&records, &outputs);
    }
    let cache_stats = encoder.cache_stats()?;
    // A separate pass counts allocation requests; it is not used as a timing.
    if request.cache_state == "empty_each_sweep" { encoder.clear_cache()?; }
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let allocations: Vec<_> = texts.chunks(request.batch_size).map(|batch| encoder.encode(batch, &options)).collect::<bpe::Result<_>>()?;
    COUNTING.store(false, Ordering::Relaxed);
    let allocation_calls = ALLOCATIONS.load(Ordering::Relaxed);
    let allocated_bytes = ALLOCATED_BYTES.load(Ordering::Relaxed);
    validate(&records, &allocations);
    drop(allocations);
    let profiles = if request.profile {
        Some(records.iter().map(|row| bpe::profile::measure(&tokenizer, &row.text, request.backend, &request.runtime)).collect::<bpe::Result<Vec<_>>>()?)
    } else { None };
    let peak_rss_bytes = fs::read_to_string("/proc/self/status").ok().and_then(|status| {
        status.lines().find(|line| line.starts_with("VmHWM:")).and_then(|line| line.split_whitespace().nth(1)).and_then(|value| value.parse::<u64>().ok()).map(|kb| kb * 1024)
    });
    println!("{}", serde_json::json!({
        "schema_version":2, "backend":request.backend, "runtime":request.runtime,
        "threads":request.threads, "batch_size":request.batch_size, "cache_state":request.cache_state,
        "input_bytes":records.iter().map(|row|row.text.len()).sum::<usize>(), "documents":records.len(),
        "tokens":records.iter().map(|row|row.ids.len()).sum::<usize>(),
        "model_sha256":tokenizer.model().sha256()?, "encode_seconds":encode_seconds,
        "batch_seconds":batch_seconds, "cache_stats":cache_stats,
        "allocation_calls":allocation_calls, "allocated_bytes_requested":allocated_bytes,
        "peak_rss_bytes":peak_rss_bytes, "load_seconds":load_seconds, "pool_setup_seconds":pool_setup_seconds,
        "profiles":profiles, "mismatches":0
    }));
    Ok(())
}
