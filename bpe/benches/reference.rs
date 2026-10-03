use bpe::Tokenizer;
use serde::Deserialize;
use std::{env, fs, hint::black_box, time::Instant};

#[derive(Deserialize)]
struct Record {
    text: String,
    ids: Vec<u32>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = env::var("BPE_BENCH_MODEL")?;
    let input = env::var("BPE_BENCH_INPUT")?;
    let iterations: usize = env::var("BPE_BENCH_ITERATIONS")
        .unwrap_or_else(|_| "5".into())
        .parse()?;
    let records: Vec<Record> = fs::read_to_string(input)?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    if records.is_empty() || iterations == 0 {
        return Err("nonempty input and positive iterations required".into());
    }
    let tokenizer = Tokenizer::load(model)?;
    for record in &records {
        assert_eq!(
            tokenizer.encode(&record.text)?,
            record.ids,
            "preflight ID mismatch"
        );
        assert_eq!(tokenizer.decode_bytes(&record.ids)?, record.text.as_bytes());
    }
    let input_bytes: usize = records.iter().map(|r| r.text.len()).sum();
    let mut encode_seconds = Vec::new();
    let mut decode_seconds = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        let outputs: Vec<_> = records
            .iter()
            .map(|r| tokenizer.encode(black_box(&r.text)))
            .collect::<Result<_, _>>()?;
        black_box(&outputs);
        encode_seconds.push(start.elapsed().as_secs_f64());
        // Validate consumed outputs outside the measured region.
        for (record, ids) in records.iter().zip(&outputs) {
            assert_eq!(ids, &record.ids);
        }
        let start = Instant::now();
        let decoded: Vec<_> = outputs
            .iter()
            .map(|ids| tokenizer.decode_bytes(black_box(ids)))
            .collect::<Result<_, _>>()?;
        black_box(&decoded);
        decode_seconds.push(start.elapsed().as_secs_f64());
        for (record, bytes) in records.iter().zip(&decoded) {
            assert_eq!(bytes, record.text.as_bytes());
        }
    }
    println!(
        "{}",
        serde_json::json!({"schema_version":1,"engine":"rust_reference_native","threads":1,"cache_capacity":0,"input_bytes":input_bytes,"documents":records.len(),"tokens":records.iter().map(|r|r.ids.len()).sum::<usize>(),"model_sha256":tokenizer.model().sha256()?,"encode_seconds":encode_seconds,"decode_seconds":decode_seconds})
    );
    Ok(())
}
