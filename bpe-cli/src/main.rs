use bpe::{
    Backend, BpeTrainer, EncodeOptions, Pretokenizer, RuntimeConfig, SpecialMode, Tokenizer,
    TrainConfig, TrainerBackend,
};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::Deserialize;
use std::{
    collections::BTreeSet,
    fs,
    io::{self, Read, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
#[command(version, about = "Lossless byte BPE tokenizer")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Clone, Copy, ValueEnum)]
enum Pre {
    Raw,
    Gpt2,
}
#[derive(Clone, Copy, ValueEnum)]
enum Engine {
    Reference,
    Auto,
    Heap,
}
#[derive(Clone, Copy, ValueEnum)]
enum Policy {
    Ordinary,
    Allow,
    Reject,
}
#[derive(Clone, Copy, ValueEnum)]
enum Trainer {
    Reference,
    Incremental,
}
#[derive(Args)]
struct RuntimeArgs {
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Auto uses the heap for pre-tokens at least this many UTF-8 bytes long.
    #[arg(long)]
    heap_threshold: Option<usize>,
    /// Per-worker FIFO cache entries; zero disables caching.
    #[arg(long, default_value_t = 0)]
    cache_capacity: usize,
    #[arg(long, default_value_t = 8388608)]
    cache_bytes: usize,
    /// Ablation: allocate fresh merge buffers for every pre-token.
    #[arg(long)]
    no_reuse_buffers: bool,
}
impl RuntimeArgs {
    fn config(&self) -> RuntimeConfig {
        RuntimeConfig {
            heap_threshold: self.heap_threshold,
            cache_capacity: self.cache_capacity,
            cache_bytes: self.cache_bytes,
            reuse_buffers: !self.no_reuse_buffers,
            ..Default::default()
        }
    }
}
impl From<Engine> for Backend {
    fn from(engine: Engine) -> Self {
        match engine {
            Engine::Reference => Self::Reference,
            Engine::Auto => Self::Auto,
            Engine::Heap => Self::Heap,
        }
    }
}
#[derive(Subcommand)]
enum Command {
    Train {
        /// JSONL records, each containing exactly {"text": "..."}; '-' reads stdin.
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 8192)]
        vocab_size: usize,
        #[arg(long, default_value_t = 2)]
        min_frequency: u64,
        #[arg(long, value_enum, default_value_t = Pre::Gpt2)]
        pretokenizer: Pre,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value_t = Trainer::Reference)]
        trainer: Trainer,
    },
    Encode {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, conflicts_with = "input")]
        text: Option<String>,
        /// UTF-8 text file or JSONL with --jsonl; omitted/'-' reads stdin.
        #[arg(long)]
        input: Option<PathBuf>,
        #[arg(long, conflicts_with = "text")]
        jsonl: bool,
        #[arg(long, value_enum, default_value_t = Engine::Reference)]
        backend: Engine,
        #[arg(long, value_enum, default_value_t = Policy::Ordinary)]
        special: Policy,
        /// Repeat for each allowed special. Requires --special allow.
        #[arg(long = "allow-special", requires = "special")]
        allow_special: Vec<String>,
        #[command(flatten)]
        runtime: RuntimeArgs,
    },
    /// Output the actual merge events with absolute UTF-8 byte offsets.
    Trace {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: String,
        #[arg(long, value_enum, default_value_t = Engine::Reference)]
        backend: Engine,
        #[arg(long)]
        heap_threshold: Option<usize>,
    },
    /// Diagnostic phase timings on ordinary text; not a throughput benchmark.
    Profile {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long, value_enum, default_value_t = Engine::Reference)]
        backend: Engine,
    },
    Decode {
        #[arg(long)]
        model: PathBuf,
        /// JSON array of integer IDs; omitted/'-' reads stdin.
        #[arg(long)]
        ids_file: Option<PathBuf>,
        #[arg(long, conflicts_with = "lossy")]
        bytes: bool,
        #[arg(long)]
        lossy: bool,
    },
    Inspect {
        #[arg(long)]
        model: PathBuf,
    },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    text: String,
}
fn read(path: Option<&PathBuf>) -> Result<String, Box<dyn std::error::Error>> {
    if let Some(path) = path.filter(|path| path.as_os_str() != "-") {
        return Ok(fs::read_to_string(path)?);
    }
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    Ok(input)
}
fn records(input: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    input
        .lines()
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str::<Record>(line)
                .map(|r| r.text)
                .map_err(|error| format!("JSONL line {}: {error}", index + 1).into())
        })
        .collect()
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Train {
            input,
            vocab_size,
            min_frequency,
            pretokenizer,
            out,
            trainer,
        } => {
            let texts = records(&read(Some(&input))?)?;
            let pretokenizer = match pretokenizer {
                Pre::Raw => Pretokenizer::Raw,
                Pre::Gpt2 => Pretokenizer::gpt2(),
            };
            let start = Instant::now();
            let (model, report) = BpeTrainer::new(TrainConfig {
                vocab_size,
                min_frequency,
                pretokenizer,
            })
            .train_with_report(
                &texts,
                match trainer {
                    Trainer::Reference => TrainerBackend::Reference,
                    Trainer::Incremental => TrainerBackend::Incremental,
                },
            )?;
            let training_seconds = start.elapsed().as_secs_f64();
            model.save(&out)?;
            let peak_rss_bytes = fs::read_to_string("/proc/self/status")
                .ok()
                .and_then(|status| {
                    status
                        .lines()
                        .find(|line| line.starts_with("VmHWM:"))
                        .and_then(|line| line.split_whitespace().nth(1))
                        .and_then(|value| value.parse::<u64>().ok())
                        .map(|kb| kb * 1024)
                });
            println!(
                "{}",
                serde_json::json!({"vocab_size":model.vocab_size(), "merges":model.merges().len(), "model_sha256":model.sha256()?, "out":out, "training_seconds":training_seconds, "training_report":report, "peak_rss_bytes":peak_rss_bytes})
            );
        }
        Command::Encode {
            model,
            text,
            input,
            jsonl,
            backend,
            special,
            allow_special,
            runtime,
        } => {
            if !matches!(special, Policy::Allow) && !allow_special.is_empty() {
                return Err("--allow-special requires --special allow".into());
            }
            let tokenizer = Tokenizer::load(model)?;
            let options = EncodeOptions {
                backend: backend.into(),
                special: match special {
                    Policy::Ordinary => SpecialMode::Ordinary,
                    Policy::Reject => SpecialMode::Reject,
                    Policy::Allow => {
                        SpecialMode::Allow(allow_special.into_iter().collect::<BTreeSet<_>>())
                    }
                },
            };
            let input = match text {
                Some(text) => text,
                None => read(input.as_ref())?,
            };
            if jsonl {
                let mut encoder = tokenizer.batch_encoder(runtime.threads, runtime.config())?;
                for ids in encoder.encode(&records(&input)?, &options)? {
                    println!("{}", serde_json::to_string(&ids)?);
                }
            } else {
                if runtime.threads != 1 {
                    return Err("--threads requires --jsonl; one document remains serial".into());
                }
                let mut session = tokenizer.session(runtime.config())?;
                println!(
                    "{}",
                    serde_json::to_string(&session.encode(&input, &options)?)?
                );
            }
        }
        Command::Trace {
            model,
            text,
            backend,
            heap_threshold,
        } => {
            let tokenizer = Tokenizer::load(model)?;
            let options = EncodeOptions {
                backend: backend.into(),
                ..Default::default()
            };
            let config = RuntimeConfig {
                heap_threshold,
                ..Default::default()
            };
            println!(
                "{}",
                serde_json::to_string(&tokenizer.trace(&text, &options, &config)?)?
            );
        }
        Command::Profile {
            model,
            input,
            backend,
        } => {
            let tokenizer = Tokenizer::load(model)?;
            let text = read(Some(&input))?;
            println!(
                "{}",
                serde_json::to_string(&bpe::profile::measure(
                    &tokenizer,
                    &text,
                    backend.into(),
                    &RuntimeConfig::default()
                )?)?
            );
        }
        Command::Decode {
            model,
            ids_file,
            bytes,
            lossy,
        } => {
            let tokenizer = Tokenizer::load(model)?;
            let ids: Vec<u32> = serde_json::from_str(&read(ids_file.as_ref())?)?;
            if bytes {
                println!("{}", serde_json::to_string(&tokenizer.decode_bytes(&ids)?)?);
            } else {
                let decoded = if lossy {
                    tokenizer.decode_lossy(&ids)?
                } else {
                    tokenizer.decode_utf8(&ids)?
                };
                io::stdout().write_all(decoded.as_bytes())?; // No invented trailing newline.
            }
        }
        Command::Inspect { model } => {
            let tokenizer = Tokenizer::load(model)?;
            let m = tokenizer.model();
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"format_version":1, "profile":m.data().profile, "pretokenizer":m.pretokenizer(), "vocab_size":m.vocab_size(), "merges":m.merges().len(), "special_tokens":m.special_tokens(), "model_sha256":m.sha256()?})
                )?
            );
        }
    }
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
