mod domain;
mod application;
mod infrastructure;

use std::path::Path;

use crate::domain::tensor::{softmax, entropy_from_logits};
use crate::domain::ngram::TokenNgram;
use crate::domain::bias_head::BiasHead;
use crate::domain::mixer::AdaptiveMixer;
use crate::domain::coder::{Cdf, RangeEncoder, RangeDecoder};
use crate::domain::cm::ContextMixer;
use crate::domain::bridge::byte_probs_to_bit_preds;
use crate::domain::backbone::{ByteBackbone, RwkvBackbone, BackboneOrchestrator};
use crate::domain::match_model::MatchModel;
use crate::domain::lstm_expert::LstmExpert;
use crate::domain::preprocess::{self, Transform};
use crate::application::telemetry::{ProgressTracker, JsonLogger, LogSnapshot, HybridLogger, HybridLogSnapshot};
use crate::infrastructure::rwkv7::model::{Rwkv7Config, Rwkv7Model, Rwkv7State};
use crate::infrastructure::rwkv7::tokenizer::WorldTokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "compress" => cmd_compress(&args[2..]),
        "decompress" => cmd_decompress(&args[2..]),
        "cm-eval" => cmd_cm_eval(&args[2..]),
        "hybrid-eval" => cmd_hybrid_eval(&args[2..]),
        "baseline" => cmd_baseline(&args[2..]),
        "rwkv-test" => cmd_rwkv_test(&args[2..]),
        "info" => todo!("Checkpoint info"),
        _ => print_usage(),
    }
}

fn print_usage() {
    eprintln!("azathoth-lm — hybrid CM + neural byte-level predictor");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  compress    --input PATH --output PATH [--weights DIR] [--bytes N] [--lr F] [--ngram-scale F] [--mix-eta F]");
    eprintln!("  decompress  --input PATH --output PATH [--weights DIR]");
    eprintln!("  cm-eval     --input PATH [--bytes N] [--e8e9]");
    eprintln!("  hybrid-eval --input PATH [--weights DIR] [--backbone2 DIR] [--bytes N] [--skip THRESHOLD] [--no-hierarchical] [--no-match] [--no-emb-surgery] [--lstm-hidden N] [--lstm-lr F] [--lstm-layers N] [--expert] [--expert-lr F] [--neural-blend] [--blend-lr F] [--order-chain] [--log FILE.jsonl] [--emb-surgery METHOD] [--e8e9] [--preprocess auto|identity|delta:N|byteplane:N] [--save-state PATH]");
    eprintln!("  baseline    --input PATH [--weights DIR] [--bytes N] [--ensemble] [--lr F] [--tau F] [--ngram-scale F] [--log FILE.jsonl]");
    eprintln!("  rwkv-test   --weights DIR [--prompt TEXT]");
    eprintln!("  info        --ckpt PATH");
}

/// Compressed format:
/// [8 bytes: magic "AZTH\x01\x00\x00\x00"]
/// [4 bytes: total_input_bytes as u32 LE]
/// [4 bytes: token_count as u32 LE]
/// [N bytes: range-coded token stream]
const MAGIC: &[u8; 8] = b"AZTH\x01\x00\x00\x00";

/// Build CDF from logits for current prediction step.
/// Uses f64 softmax to preserve tail precision for 65K vocab.
fn cdf_from_logits(logits: &[f32]) -> Cdf {
    let n = logits.len();
    // f64 log-sum-exp for numerical stability with large vocab
    let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, |a, b| a.max(b as f64));
    let mut probs = Vec::with_capacity(n);
    let mut sum = 0.0f64;
    for &l in logits {
        let e = ((l as f64) - max_l).exp();
        probs.push(e);
        sum += e;
    }
    let inv_sum = 1.0 / sum;
    let probs_f32: Vec<f32> = probs.iter().map(|&p| (p * inv_sum) as f32).collect();
    Cdf::from_probs(&probs_f32)
}

fn cmd_compress(args: &[String]) {
    let mut input_path = "data/enwik8".to_string();
    let mut output_path = String::new();
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();
    let mut max_bytes: usize = 0;
    let mut bias_lr: f32 = 0.30;
    let mut ngram_scale: f32 = 0.5;
    let mut mix_eta: f32 = 0.01;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { i += 1; input_path = args[i].clone(); }
            "--output" => { i += 1; output_path = args[i].clone(); }
            "--weights" => { i += 1; weights_dir = args[i].clone(); }
            "--bytes" => { i += 1; max_bytes = args[i].parse().unwrap(); }
            "--lr" => { i += 1; bias_lr = args[i].parse().unwrap(); }
            "--ngram-scale" => { i += 1; ngram_scale = args[i].parse().unwrap(); }
            "--mix-eta" => { i += 1; mix_eta = args[i].parse().unwrap(); }
            _ => {}
        }
        i += 1;
    }

    if output_path.is_empty() {
        eprintln!("error: --output PATH required");
        std::process::exit(1);
    }

    // Load input
    let raw_bytes = std::fs::read(&input_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", input_path, e));
    let total_bytes = if max_bytes > 0 { max_bytes.min(raw_bytes.len()) } else { raw_bytes.len() };
    let input_slice = &raw_bytes[..total_bytes];
    eprintln!("[compress] input: {} ({} bytes)", input_path, total_bytes);

    // Load tokenizer and model
    let model_path = Path::new(&weights_dir).join("model.safetensors");
    let vocab_path = Path::new(&weights_dir).join("rwkv_vocab_v20230424.txt");
    let tokenizer = WorldTokenizer::load(&vocab_path);
    let config = Rwkv7Config::from_weights_dir(&weights_dir);
    let model = Rwkv7Model::load(&model_path, config);
    let v = model.config.vocab_size;

    // Tokenize
    let tokens = tokenizer.encode(input_slice);
    eprintln!("[compress] tokenized: {} tokens ({:.2} bytes/token)",
              tokens.len(), total_bytes as f64 / tokens.len() as f64);

    // Initialize ensemble + coder
    let mut ngram = TokenNgram::new(4, v, ngram_scale);
    let mut bias = BiasHead::new(v, bias_lr);
    let mut mixer = AdaptiveMixer::new(mix_eta);
    let mut enc = RangeEncoder::new();

    let mut state = Rwkv7State::new(&model.config);
    let mut scratch = model.create_scratch();
    let mut logits = vec![0.0f32; v];
    let mut ce_bits = 0.0f64; // cross-entropy bits for comparison

    eprintln!("[compress] encoding ...");
    let t_start = std::time::Instant::now();

    for t in 0..tokens.len() {
        let tok = tokens[t] as usize;

        if t == 0 {
            // First token: no context yet, encode with uniform CDF
            let cdf = Cdf::uniform(v);
            enc.encode_symbol(tok, &cdf);
            let prob = 1.0 / v as f64;
            ce_bits += -prob.log2();
        } else {
            // Build ensemble logits
            let ng_bias = ngram.compute_bias(None);
            let b_vec = bias.bias_vector();
            let ensemble_logits = mixer.combine(&logits, &ng_bias, b_vec);

            // Build CDF and encode
            let cdf = cdf_from_logits(&ensemble_logits);
            enc.encode_symbol(tok, &cdf);

            // Track cross-entropy for comparison
            let tensor = crate::domain::tensor::Tensor::from_data(
                ensemble_logits, vec![v],
            );
            let probs = softmax(&tensor);
            let prob = probs.data[tok] as f64;
            ce_bits += -prob.max(1e-30).log2();

            // Update online components AFTER encoding
            mixer.update(&probs.data, &ngram.compute_bias(None), bias.bias_vector(), tok);
            bias.update(&probs.data, tok);
        }

        ngram.observe(tok as u32);
        model.forward_into(tok, &mut state, &mut scratch, &mut logits);

        // Progress
        if t > 0 && (t % 1000 == 0 || t == tokens.len() - 1) {
            let elapsed = t_start.elapsed().as_secs_f64();
            let pct = 100.0 * t as f64 / tokens.len() as f64;
            let enc_size = enc.size();
            eprint!("\r[compress] {:.1}% | {} tokens | ~{} bytes | {:.1} tok/s   ",
                    pct, t, enc_size, t as f64 / elapsed);
        }
    }

    let compressed = enc.finish();
    let elapsed = t_start.elapsed().as_secs_f64();
    eprintln!();

    // Write output: header + compressed stream
    let mut output = Vec::with_capacity(16 + compressed.len());
    output.extend_from_slice(MAGIC);
    output.extend_from_slice(&(total_bytes as u32).to_le_bytes());
    output.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    output.extend_from_slice(&compressed);

    std::fs::write(&output_path, &output)
        .unwrap_or_else(|e| panic!("cannot write {}: {}", output_path, e));

    let compressed_bpb = output.len() as f64 * 8.0 / total_bytes as f64;
    let ce_bpb = ce_bits / total_bytes as f64;
    let ratio = output.len() as f64 / total_bytes as f64;

    eprintln!();
    eprintln!("[compress] results:");
    eprintln!("  input:          {} bytes", total_bytes);
    eprintln!("  compressed:     {} bytes (ratio: {:.4})", output.len(), ratio);
    eprintln!("  compressed BPB: {:.4}", compressed_bpb);
    eprintln!("  cross-ent BPB:  {:.4}", ce_bpb);
    eprintln!("  coder overhead: {:.4} BPB", compressed_bpb - ce_bpb);
    eprintln!("  time:           {:.1}s ({:.1} tok/s)", elapsed, tokens.len() as f64 / elapsed);
    eprintln!("  output:         {}", output_path);
}

fn cmd_decompress(args: &[String]) {
    let mut input_path = String::new();
    let mut output_path = String::new();
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { i += 1; input_path = args[i].clone(); }
            "--output" => { i += 1; output_path = args[i].clone(); }
            "--weights" => { i += 1; weights_dir = args[i].clone(); }
            _ => {}
        }
        i += 1;
    }

    if input_path.is_empty() || output_path.is_empty() {
        eprintln!("error: --input PATH and --output PATH required");
        std::process::exit(1);
    }

    // Read compressed file
    let data = std::fs::read(&input_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", input_path, e));

    if data.len() < 16 || &data[..8] != MAGIC {
        eprintln!("error: invalid compressed file (bad magic)");
        std::process::exit(1);
    }

    let total_bytes = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let token_count = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let stream = &data[16..];

    eprintln!("[decompress] compressed: {} bytes, original: {} bytes, {} tokens",
              data.len(), total_bytes, token_count);

    // Load tokenizer and model
    let model_path = Path::new(&weights_dir).join("model.safetensors");
    let vocab_path = Path::new(&weights_dir).join("rwkv_vocab_v20230424.txt");
    let tokenizer = WorldTokenizer::load(&vocab_path);
    let config = Rwkv7Config::from_weights_dir(&weights_dir);
    let model = Rwkv7Model::load(&model_path, config);
    let v = model.config.vocab_size;

    // Initialize ensemble + decoder — must match compressor exactly
    let mut ngram = TokenNgram::new(4, v, 0.5);
    let mut bias = BiasHead::new(v, 0.30);
    let mut mixer = AdaptiveMixer::new(0.01);
    let mut dec = RangeDecoder::new(stream);

    let mut state = Rwkv7State::new(&model.config);
    let mut scratch = model.create_scratch();
    let mut logits = vec![0.0f32; v];
    let mut decoded_tokens: Vec<u32> = Vec::with_capacity(token_count);

    eprintln!("[decompress] decoding ...");
    let t_start = std::time::Instant::now();

    for t in 0..token_count {
        let tok = if t == 0 {
            // First token: uniform CDF (no context), must match compressor
            let cdf = Cdf::uniform(v);
            dec.decode_symbol(&cdf) as u32
        } else {
            let ng_bias = ngram.compute_bias(None);
            let b_vec = bias.bias_vector();
            let ensemble_logits = mixer.combine(&logits, &ng_bias, b_vec);

            let cdf = cdf_from_logits(&ensemble_logits);
            let sym = dec.decode_symbol(&cdf);

            // Update online components — must mirror compressor exactly
            let tensor = crate::domain::tensor::Tensor::from_data(
                ensemble_logits, vec![v],
            );
            let probs = softmax(&tensor);
            mixer.update(&probs.data, &ngram.compute_bias(None), bias.bias_vector(), sym);
            bias.update(&probs.data, sym);

            sym as u32
        };

        decoded_tokens.push(tok);
        ngram.observe(tok);
        model.forward_into(tok as usize, &mut state, &mut scratch, &mut logits);

        if t > 0 && (t % 1000 == 0 || t == token_count - 1) {
            let elapsed = t_start.elapsed().as_secs_f64();
            let pct = 100.0 * t as f64 / token_count as f64;
            eprint!("\r[decompress] {:.1}% | {} tokens | {:.1} tok/s   ",
                    pct, t, t as f64 / elapsed);
        }
    }
    let elapsed = t_start.elapsed().as_secs_f64();
    eprintln!();

    // Decode tokens to bytes
    let mut output_bytes = Vec::with_capacity(total_bytes);
    for &tok in &decoded_tokens {
        let bytes = tokenizer.decode_token(tok);
        output_bytes.extend_from_slice(bytes);
    }

    if output_bytes.len() != total_bytes {
        eprintln!("WARNING: decoded {} bytes, expected {}", output_bytes.len(), total_bytes);
    }

    std::fs::write(&output_path, &output_bytes)
        .unwrap_or_else(|e| panic!("cannot write {}: {}", output_path, e));

    eprintln!();
    eprintln!("[decompress] output: {} ({} bytes)", output_path, output_bytes.len());
    eprintln!("[decompress] time: {:.1}s ({:.1} tok/s)", elapsed, token_count as f64 / elapsed);
    eprintln!("[decompress] done.");
}

fn cmd_cm_eval(args: &[String]) {
    let mut input_path = "data/enwik8".to_string();
    let mut max_bytes: usize = 0;
    let mut use_e8e9 = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { i += 1; input_path = args[i].clone(); }
            "--bytes" => { i += 1; max_bytes = args[i].parse().unwrap(); }
            "--e8e9" => { use_e8e9 = true; }
            _ => {}
        }
        i += 1;
    }

    let raw_bytes = std::fs::read(&input_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", input_path, e));
    let (eval_bytes, e8e9_applied) = if use_e8e9 {
        let (full_transformed, stats) = preprocess::e8e9_encode_with_stats(&raw_bytes);
        eprintln!("[cm-eval] E8/E9 transform (full file {} bytes): {} E8 + {} E9, {} transformed, {} skipped",
                  raw_bytes.len(), stats.e8_count, stats.e9_count, stats.transformed, stats.skipped);
        (full_transformed, true)
    } else {
        (raw_bytes, false)
    };
    let total_bytes = if max_bytes > 0 { max_bytes.min(eval_bytes.len()) } else { eval_bytes.len() };
    let input_slice = &eval_bytes[..total_bytes];
    let _ = e8e9_applied;

    let mut cm = ContextMixer::new();
    let mem_mb = cm.memory_bytes() as f64 / (1024.0 * 1024.0);
    eprintln!("[cm-eval] input: {} ({} bytes)", input_path, total_bytes);
    eprintln!("[cm-eval] models: {} (9 order + 2 sparse + 1 indirect), hash tables: {:.1} MB", cm.n_models(), mem_mb);
    eprintln!("[cm-eval] evaluating ...");
    eprintln!();

    let t_start = std::time::Instant::now();
    let mut total_bits = 0.0f64;
    let report_interval = (total_bytes / 4).max(1000);

    for (pos, &byte) in input_slice.iter().enumerate() {
        total_bits += cm.process_byte(byte);

        if (pos + 1) % report_interval == 0 || pos + 1 == total_bytes {
            let elapsed = t_start.elapsed().as_secs_f64();
            let bpb = total_bits / (pos + 1) as f64;
            let bps = (pos + 1) as f64 / elapsed;
            eprint!("\r[cm-eval] {:.1}% | {}/{} bytes | {:.4} BPB | {:.0} B/s   ",
                    100.0 * (pos + 1) as f64 / total_bytes as f64,
                    pos + 1, total_bytes, bpb, bps);
        }
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    let final_bpb = total_bits / total_bytes as f64;
    eprintln!();
    eprintln!();
    eprintln!("[cm-eval] results:");
    eprintln!("  input:       {} bytes", total_bytes);
    eprintln!("  total bits:  {:.1}", total_bits);
    eprintln!("  BPB:         {:.4}", final_bpb);
    eprintln!("  time:        {:.1}s ({:.0} B/s)", elapsed, total_bytes as f64 / elapsed);
    eprintln!("  hash memory: {:.1} MB", mem_mb);
}

fn cmd_hybrid_eval(args: &[String]) {
    let mut input_path = "data/enwik8".to_string();
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();
    let mut max_bytes: usize = 0;
    let mut skip_threshold: f32 = 0.0;
    let mut use_lstm = false;
    let mut use_hierarchical = true;
    let mut use_match = true;
    let mut use_expert = false;
    let mut expert_lr: f32 = 0.01;
    let mut neural_blend = false;
    let mut blend_lr: f32 = 0.005;
    let mut lstm_hidden: usize = 128;
    let mut lstm_lr: f32 = 0.002;
    let mut lstm_layers: usize = 1;
    let mut log_path: Option<String> = None;
    let mut emb_surgery: Option<String> = Some("center0.3".to_string());
    let mut use_e8e9 = false;
    let mut save_state_path: Option<String> = None;
    let mut preprocess_arg: Option<String> = None;
    let mut order_chain = false;
    let mut backbone2_dir: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { i += 1; input_path = args[i].clone(); }
            "--weights" => { i += 1; weights_dir = args[i].clone(); }
            "--backbone2" => { i += 1; backbone2_dir = Some(args[i].clone()); }
            "--bytes" => { i += 1; max_bytes = args[i].parse().unwrap(); }
            "--skip" => { i += 1; skip_threshold = args[i].parse().unwrap(); }
            "--lstm" => { use_lstm = true; }
            "--hierarchical" => { use_hierarchical = true; }
            "--no-hierarchical" => { use_hierarchical = false; }
            "--match" => { use_match = true; }
            "--no-match" => { use_match = false; }
            "--no-emb-surgery" => { emb_surgery = None; }
            "--expert" => { use_expert = true; }
            "--expert-lr" => { i += 1; expert_lr = args[i].parse().unwrap(); use_expert = true; }
            "--neural-blend" => { neural_blend = true; }
            "--blend-lr" => { i += 1; blend_lr = args[i].parse().unwrap(); neural_blend = true; }
            "--lstm-hidden" => { i += 1; lstm_hidden = args[i].parse().unwrap(); use_lstm = true; }
            "--lstm-lr" => { i += 1; lstm_lr = args[i].parse().unwrap(); use_lstm = true; }
            "--lstm-layers" => { i += 1; lstm_layers = args[i].parse().unwrap(); }
            "--log" => { i += 1; log_path = Some(args[i].clone()); }
            "--emb-surgery" => { i += 1; emb_surgery = Some(args[i].clone()); }
            "--e8e9" => { use_e8e9 = true; }
            "--preprocess" => { i += 1; preprocess_arg = Some(args[i].clone()); }
            "--save-state" => { i += 1; save_state_path = Some(args[i].clone()); }
            "--order-chain" => { order_chain = true; }
            _ => {}
        }
        i += 1;
    }

    // Load input — apply transforms: E8/E9 first, then adaptive preprocess
    let raw_bytes = std::fs::read(&input_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", input_path, e));
    let (after_e8e9, e8e9_applied) = if use_e8e9 {
        let (full_transformed, stats) = preprocess::e8e9_encode_with_stats(&raw_bytes);
        eprintln!("[hybrid] E8/E9 transform (full file {} bytes): {} E8 + {} E9, {} transformed, {} skipped",
                  raw_bytes.len(), stats.e8_count, stats.e9_count, stats.transformed, stats.skipped);
        (full_transformed, true)
    } else {
        (raw_bytes, false)
    };

    // Adaptive preprocessing (after E8/E9, before eval)
    let (eval_bytes, preprocess_transform) = if let Some(ref arg) = preprocess_arg {
        match preprocess::parse_transform_arg(arg) {
            None => {
                // auto-detect
                let (transformed, stats) = preprocess::adaptive_encode(&after_e8e9);
                eprintln!("[hybrid] preprocess auto: {} (raw H={:.2}, best H={:.2}, sample={}B)",
                          stats.transform, stats.raw_entropy, stats.best_entropy, stats.sample_size);
                (transformed, stats.transform)
            }
            Some(t) => {
                let transformed = preprocess::apply_transform(&after_e8e9, t);
                eprintln!("[hybrid] preprocess manual: {}", t);
                (transformed, t)
            }
        }
    } else {
        (after_e8e9, Transform::Identity)
    };

    let total_bytes = if max_bytes > 0 { max_bytes.min(eval_bytes.len()) } else { eval_bytes.len() };
    let input_slice = &eval_bytes[..total_bytes];

    let mut preprocess_tags = String::new();
    if e8e9_applied { preprocess_tags.push_str(" [E8/E9]"); }
    if preprocess_transform != Transform::Identity {
        preprocess_tags.push_str(&format!(" [{}]", preprocess_transform));
    }
    eprintln!("[hybrid] input: {} ({} bytes){}", input_path, total_bytes, preprocess_tags);

    // Load RWKV backbone via orchestrator (model + tokenizer + bridge)
    let rwkv = RwkvBackbone::load(
        &weights_dir,
        emb_surgery.as_deref(),
    );
    let mut orchestrator = BackboneOrchestrator::new(rwkv);
    orchestrator.rwkv_mut().prepare(input_slice);
    eprintln!("[hybrid] tokenized: {} tokens ({:.2} bytes/token)",
              orchestrator.rwkv().token_count(), orchestrator.rwkv().bytes_per_token(total_bytes));

    // Load optional second backbone (C.0.4: e.g., G1k 1.5B)
    if let Some(ref bb2_dir) = backbone2_dir {
        eprintln!("[hybrid] loading second backbone from {} ...", bb2_dir);
        let mut bb2 = RwkvBackbone::load(bb2_dir, emb_surgery.as_deref());
        bb2.prepare(input_slice);
        eprintln!("[hybrid] backbone2: {} tokens ({:.2} bytes/token)",
                  bb2.token_count(), bb2.bytes_per_token(total_bytes));
        orchestrator.add_backbone(Box::new(bb2));
    }

    if orchestrator.backbone_count() > 1 {
        eprintln!("[hybrid] backbones: {}", orchestrator.names().join(", "));
    }

    // Initialize components
    let mut cm = if use_hierarchical {
        eprintln!("[hybrid] hierarchical mixer: hidden={}, lr={}, layers={}", lstm_hidden, lstm_lr, lstm_layers);
        ContextMixer::new_with_hierarchical(lstm_hidden, lstm_lr, lstm_layers)
    } else if use_lstm {
        eprintln!("[hybrid] LSTM mixer: hidden={}, lr={}", lstm_hidden, lstm_lr);
        ContextMixer::new_with_lstm(lstm_hidden, lstm_lr)
    } else {
        ContextMixer::new()
    };
    if order_chain {
        cm.set_chain_orders(true);
    }
    let cm_mem_mb = cm.memory_bytes() as f64 / (1024.0 * 1024.0);
    let mixer_params = cm.mixer_param_count();
    let mixer_str = if use_hierarchical {
        format!("hierarchical(H={},lr={},L={}, {}params)", lstm_hidden, lstm_lr, lstm_layers, mixer_params)
    } else if use_lstm {
        format!("LSTM(H={},lr={}, {}params)", lstm_hidden, lstm_lr, mixer_params)
    } else {
        "logistic".to_string()
    };
    let mut match_model = if use_match { Some(MatchModel::new()) } else { None };
    let match_mem_mb = match_model.as_ref().map(|m| m.memory_bytes() as f64 / (1024.0 * 1024.0)).unwrap_or(0.0);
    let mut expert = if use_expert { Some(LstmExpert::new(expert_lr)) } else { None };
    let expert_str = if let Some(ref e) = expert {
        format!(" | expert: {}params, lr={}", e.param_count(), expert_lr)
    } else { String::new() };
    let mut blend_expert = if neural_blend { Some(LstmExpert::new(expert_lr)) } else { None };
    let mut blend_logit: f32 = 4.6; // sigmoid(4.6) ≈ 0.99 → almost pure RWKV initially
    let blend_str = if let Some(ref be) = blend_expert {
        format!(" | blend: {}params, blend_lr={}", be.param_count(), blend_lr)
    } else { String::new() };
    let chain_str = if order_chain { " | order-chain" } else { "" };
    eprintln!("[hybrid] CM: {} models, {:.1} MB | mixer: {} | trie: {} nodes{}{}{}{}",
              cm.n_models(),
              cm_mem_mb, mixer_str, orchestrator.rwkv().trie_node_count(),
              if use_match { format!(" | match: {:.1} MB", match_mem_mb) } else { String::new() },
              expert_str, blend_str, chain_str);

    let mut logger = log_path.as_ref().map(|p| HybridLogger::new(p, total_bytes));

    let skip_active = skip_threshold > 0.0;
    let mode_str = if skip_active {
        format!("hybrid + skip (threshold={:.2})", skip_threshold)
    } else {
        "hybrid (CM + RWKV bridge)".to_string()
    };
    eprintln!("[hybrid] mode: {}", mode_str);

    let mut total_bits = 0.0f64;
    let mut byte_count = 0usize;
    let mut skipped_tokens = 0usize;
    let mut skipped_bytes = 0usize;
    let mut cum_bit_costs = [0.0f64; 8];
    let mut cum_bit_count = 0u64;

    eprintln!("[hybrid] evaluating ...");
    eprintln!();

    let t_start = std::time::Instant::now();
    let report_interval = (total_bytes / 4).max(1000);
    let n_tokens = orchestrator.rwkv().token_count();

    for t in 0..n_tokens {
        let tok_bytes = orchestrator.rwkv().current_token_bytes().to_vec();
        let have_rwkv = orchestrator.rwkv().has_prediction();

        if have_rwkv {
            // Check confidence skip: top-1 probability
            let top1_prob = if skip_active {
                orchestrator.rwkv().token_probs().iter().copied().fold(0.0f32, f32::max)
            } else {
                0.0
            };

            if skip_active && top1_prob >= skip_threshold {
                // SKIP: use RWKV token-level cross-entropy for this token
                let tok = orchestrator.rwkv().current_token_id();
                let prob_correct = orchestrator.rwkv().token_probs()[tok] as f64;
                let token_bits = -prob_correct.max(1e-30).log2();
                let n_bytes = tok_bytes.len();
                let bits_per_byte = token_bits / n_bytes as f64;

                for &byte in tok_bytes.iter() {
                    total_bits += bits_per_byte;
                    cm.observe_byte(byte);
                    if let Some(ref mut mm) = match_model {
                        mm.observe(byte);
                    }
                    if let Some(ref mut be) = blend_expert {
                        be.observe_byte(byte);
                    }
                    orchestrator.observe_byte(byte);
                    byte_count += 1;

                    if byte_count % report_interval == 0 || byte_count == total_bytes {
                        let elapsed = t_start.elapsed().as_secs_f64();
                        let bpb = total_bits / byte_count as f64;
                        let bps = byte_count as f64 / elapsed;
                        eprint!("\r[hybrid] {:.1}% | {}/{} bytes | {:.4} BPB | {:.0} B/s | skip {:.0}%   ",
                                100.0 * byte_count as f64 / total_bytes as f64,
                                byte_count, total_bytes, bpb, bps,
                                100.0 * skipped_tokens as f64 / t.max(1) as f64);
                    }
                }

                skipped_tokens += 1;
                skipped_bytes += n_bytes;
            } else {
                // FULL HYBRID: backbone byte probs + CM
                for &byte in tok_bytes.iter() {
                    let rwkv_byte_probs = orchestrator.byte_probs();

                    // Neural blend (Phase 3, R55): blend expert with RWKV at byte level
                    let blended_byte_probs;
                    let neural_bit_preds = if let Some(ref be) = blend_expert {
                        let expert_bp = be.predict_byte();
                        let alpha = 1.0 / (1.0 + (-blend_logit).exp()); // sigmoid
                        let mut bp = [0.0f32; 256];
                        for k in 0..256 {
                            bp[k] = alpha * rwkv_byte_probs[k] + (1.0 - alpha) * expert_bp[k];
                        }
                        blended_byte_probs = bp;
                        byte_probs_to_bit_preds(&blended_byte_probs, byte)
                    } else {
                        byte_probs_to_bit_preds(&rwkv_byte_probs, byte)
                    };
                    let mut bit_costs = [0.0f64; 8];

                    // Build externals list dynamically
                    let mut externals: Vec<[f32; 8]> = Vec::new();
                    externals.push(neural_bit_preds);

                    let match_len = if let Some(ref mm) = match_model {
                        let (match_byte_probs, ml) = mm.predict();
                        externals.push(byte_probs_to_bit_preds(&match_byte_probs, byte));
                        ml
                    } else { 0 };

                    if let Some(ref expert_ref) = expert {
                        let expert_byte_probs = expert_ref.predict_byte();
                        externals.push(byte_probs_to_bit_preds(expert_byte_probs, byte));
                    }

                    let ext_refs: Vec<&[f32; 8]> = externals.iter().collect();
                    let need_detailed = logger.is_some() || save_state_path.is_some();
                    let bits = if ext_refs.len() > 1 {
                        if need_detailed {
                            cm.process_byte_with_externals_detailed(byte, &ext_refs, &mut bit_costs)
                        } else {
                            cm.process_byte_with_externals(byte, &ext_refs)
                        }
                    } else {
                        cm.process_byte_with_external(byte, &neural_bit_preds)
                    };

                    if need_detailed {
                        for i in 0..8 {
                            cum_bit_costs[i] += bit_costs[i];
                        }
                        cum_bit_count += 1;
                    }

                    total_bits += bits;
                    if let Some(ref mut log) = logger {
                        log.record_byte(bits, &HybridLogSnapshot { bit_costs, match_len });
                    }
                    if let Some(ref mut mm) = match_model {
                        mm.observe(byte);
                    }
                    if let Some(ref mut exp) = expert {
                        exp.observe_byte(byte);
                    }
                    // Update neural blend: train expert, update alpha
                    if let Some(ref mut be) = blend_expert {
                        let rwkv_loss = -(rwkv_byte_probs[byte as usize].max(1e-10)).ln();
                        let be_loss = -(be.predict_byte()[byte as usize].max(1e-10)).ln();
                        blend_logit += blend_lr * (be_loss - rwkv_loss);
                        blend_logit = blend_logit.clamp(-2.0, 8.0);
                        be.observe_byte(byte);
                    }
                    orchestrator.observe_byte(byte);
                    byte_count += 1;

                    if byte_count % report_interval == 0 || byte_count == total_bytes {
                        let elapsed = t_start.elapsed().as_secs_f64();
                        let bpb = total_bits / byte_count as f64;
                        let bps = byte_count as f64 / elapsed;
                        eprint!("\r[hybrid] {:.1}% | {}/{} bytes | {:.4} BPB | {:.0} B/s | skip {:.0}%   ",
                                100.0 * byte_count as f64 / total_bytes as f64,
                                byte_count, total_bytes, bpb, bps,
                                100.0 * skipped_tokens as f64 / t.max(1) as f64);
                    }
                }
            }
        } else {
            // First token: CM only (no RWKV context yet)
            for &byte in tok_bytes.iter() {
                let bit_costs = [0.0f64; 8];
                let mut externals: Vec<[f32; 8]> = Vec::new();

                let match_len = if let Some(ref mm) = match_model {
                    let (match_byte_probs, ml) = mm.predict();
                    externals.push(byte_probs_to_bit_preds(&match_byte_probs, byte));
                    ml
                } else { 0 };

                if let Some(ref expert_ref) = expert {
                    let expert_byte_probs = expert_ref.predict_byte();
                    externals.push(byte_probs_to_bit_preds(expert_byte_probs, byte));
                }

                let ext_refs: Vec<&[f32; 8]> = externals.iter().collect();
                let bits = if !ext_refs.is_empty() {
                    cm.process_byte_with_externals(byte, &ext_refs)
                } else {
                    cm.process_byte(byte)
                };
                total_bits += bits;
                if let Some(ref mut log) = logger {
                    log.record_byte(bits, &HybridLogSnapshot { bit_costs, match_len });
                }
                if let Some(ref mut mm) = match_model {
                    mm.observe(byte);
                }
                if let Some(ref mut exp) = expert {
                    exp.observe_byte(byte);
                }
                if let Some(ref mut be) = blend_expert {
                    be.observe_byte(byte);
                }
                orchestrator.observe_byte(byte);
                byte_count += 1;
            }
        }
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    let final_bpb = total_bits / byte_count as f64;

    eprintln!();
    eprintln!();
    eprintln!("[hybrid] results:");
    eprintln!("  input:       {} bytes ({} tokens)", byte_count, n_tokens);
    eprintln!("  BPB:         {:.4} ({})", final_bpb, mode_str);
    eprintln!("  time:        {:.1}s ({:.0} B/s)", elapsed, byte_count as f64 / elapsed);
    eprintln!("  CM memory:   {:.1} MB", cm_mem_mb);
    if skip_active {
        let predictable = n_tokens - 1; // exclude first token
        let skip_pct = 100.0 * skipped_tokens as f64 / predictable.max(1) as f64;
        eprintln!("  skipped:     {}/{} tokens ({:.1}%), {} bytes",
                  skipped_tokens, predictable, skip_pct, skipped_bytes);
    }
    if let Some(ref mut log) = logger {
        log.finalize();
        eprintln!("[hybrid] log written: {}", log_path.as_ref().unwrap());
    }

    // Save state if requested
    if let Some(ref state_path) = save_state_path {
        use crate::domain::state_io::{StateMetadata, sha256_first_n};
        use crate::application::telemetry::now_barcelona;

        let avg_bit_costs = if cum_bit_count > 0 {
            let n = cum_bit_count as f64;
            [cum_bit_costs[0]/n, cum_bit_costs[1]/n, cum_bit_costs[2]/n, cum_bit_costs[3]/n,
             cum_bit_costs[4]/n, cum_bit_costs[5]/n, cum_bit_costs[6]/n, cum_bit_costs[7]/n]
        } else {
            [0.0; 8]
        };

        let bpt = orchestrator.rwkv().bytes_per_token(total_bytes);

        let meta = StateMetadata {
            input_path: input_path.clone(),
            input_sha256: sha256_first_n(&eval_bytes, 100_000),
            input_size_total: eval_bytes.len() as u64,
            bytes_evaluated: byte_count as u64,
            bpb_final: final_bpb,
            per_bit_costs: avg_bit_costs,
            throughput_bps: byte_count as f64 / elapsed,
            elapsed_secs: elapsed,
            token_count: n_tokens as u64,
            bytes_per_token: bpt,
            model_name: orchestrator.rwkv().name().to_string(),
            weights_path: weights_dir.clone(),
            mixer_type: mixer_str.clone(),
            mixer_params: mixer_params as u64,
            emb_surgery: emb_surgery.clone().unwrap_or_else(|| "none".to_string()),
            e8e9_enabled: use_e8e9,
            cm_model_count: cm.n_models() as u32,
            cm_memory_mb: cm_mem_mb,
            match_memory_mb: match_mem_mb,
            lstm_adam_t: cm.lstm_adam_t(),
            match_total_bytes: match_model.as_ref().map(|m| m.data_len() as u64).unwrap_or(0),
            domain_cluster: StateMetadata::classify_domain(final_bpb, bpt),
            timestamp: now_barcelona(),
        };

        crate::domain::state_io::save_state(
            state_path,
            &cm,
            match_model.as_ref(),
            Some(&meta),
        ).unwrap_or_else(|e| eprintln!("[state] error saving: {}", e));
    }

    eprintln!();
    eprintln!("[hybrid] reference points:");
    eprintln!("  CM standalone (100KB):   2.41 BPB");
    eprintln!("  RWKV ensemble (100KB):   1.30 BPB");
    eprintln!("  hybrid no-skip (100KB):  1.2924 BPB");
    eprintln!("  target:                  < 1.0 BPB");
}

fn cmd_baseline(args: &[String]) {
    let mut input_path = "data/enwik8".to_string();
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();
    let mut max_bytes: usize = 0; // 0 = entire file
    let mut use_ensemble = false;
    let mut skip_threshold: f32 = 0.0;
    let mut bias_lr: f32 = 0.001;
    let mut ngram_scale: f32 = 1.0; // multiplier on N-gram weights
    let mut lr_tau: f32 = 0.0; // surprise modulation smoothness (0 = static)
    let mut adaptive_ngram = false; // StateSMix-style entropy-adaptive N-gram
    let mut use_mix = false; // adaptive component weighting
    let mut mix_eta: f32 = 0.01; // mixer learning rate
    let mut log_path: Option<String> = None; // Level 2 structured log

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { i += 1; input_path = args[i].clone(); }
            "--weights" => { i += 1; weights_dir = args[i].clone(); }
            "--bytes" => { i += 1; max_bytes = args[i].parse().unwrap(); }
            "--ensemble" => { use_ensemble = true; }
            "--skip" => { i += 1; skip_threshold = args[i].parse().unwrap(); }
            "--lr" => { i += 1; bias_lr = args[i].parse().unwrap(); }
            "--ngram-scale" => { i += 1; ngram_scale = args[i].parse().unwrap(); }
            "--tau" => { i += 1; lr_tau = args[i].parse().unwrap(); }
            "--adaptive" => { adaptive_ngram = true; }
            "--mix" => { use_mix = true; use_ensemble = true; }
            "--mix-eta" => { i += 1; mix_eta = args[i].parse().unwrap(); use_mix = true; use_ensemble = true; }
            "--log" => { i += 1; log_path = Some(args[i].clone()); }
            _ => {}
        }
        i += 1;
    }

    // Load input file
    let raw_bytes = std::fs::read(&input_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", input_path, e));
    let total_bytes = if max_bytes > 0 { max_bytes.min(raw_bytes.len()) } else { raw_bytes.len() };
    let input_slice = &raw_bytes[..total_bytes];
    eprintln!("[baseline] input: {} ({} bytes)", input_path, total_bytes);
    let mode_str = if !use_ensemble {
        "RWKV only".to_string()
    } else if skip_threshold > 0.0 {
        format!("ensemble + skip (threshold={:.2})", skip_threshold)
    } else if use_mix {
        format!("ensemble mix (lr={}, scale={}, eta={})", bias_lr, ngram_scale, mix_eta)
    } else if adaptive_ngram {
        format!("ensemble adaptive (lr={}, scale={})", bias_lr, ngram_scale)
    } else if lr_tau > 0.0 {
        format!("ensemble (lr={}, tau={}, scale={})", bias_lr, lr_tau, ngram_scale)
    } else {
        format!("ensemble (lr={}, scale={})", bias_lr, ngram_scale)
    };
    eprintln!("[baseline] mode: {}", mode_str);

    // Load tokenizer and model
    let model_path = Path::new(&weights_dir).join("model.safetensors");
    let vocab_path = Path::new(&weights_dir).join("rwkv_vocab_v20230424.txt");
    let tokenizer = WorldTokenizer::load(&vocab_path);
    let config = Rwkv7Config::from_weights_dir(&weights_dir);
    let model = Rwkv7Model::load(&model_path, config);

    let v = model.config.vocab_size;

    // Tokenize
    let tokens = tokenizer.encode(input_slice);
    eprintln!("[baseline] tokenized: {} tokens ({:.2} bytes/token)",
              tokens.len(), total_bytes as f64 / tokens.len() as f64);

    // Map each token to its byte length for BPB tracking
    let mut token_byte_lengths = Vec::with_capacity(tokens.len());
    let mut byte_offset = 0;
    for &tok in &tokens {
        let tok_bytes = tokenizer.decode_token(tok);
        let len = tok_bytes.len();
        token_byte_lengths.push(len);
        byte_offset += len;
    }
    assert_eq!(byte_offset, total_bytes,
               "tokenizer roundtrip mismatch: {} encoded bytes vs {} input bytes",
               byte_offset, total_bytes);

    // Initialize ensemble components
    let mut ngram = TokenNgram::new(4, v, ngram_scale);
    let mut bias = BiasHead::new(v, bias_lr).with_decay(lr_tau);
    let mut mixer = AdaptiveMixer::new(mix_eta);

    // Run forward pass and measure cross-entropy
    let mut state = Rwkv7State::new(&model.config);
    let mut scratch = model.create_scratch();
    let mut tracker = ProgressTracker::new(total_bytes);
    let mut logger = log_path.as_ref().map(|p| {
        eprintln!("[baseline] logging to: {}", p);
        JsonLogger::new(p, total_bytes)
    });

    eprintln!("[baseline] starting evaluation ...");
    eprintln!();

    let mut logits = vec![0.0f32; v];
    let mut skipped = 0usize;
    let skip_active = skip_threshold > 0.0 && use_ensemble;

    for t in 0..tokens.len() {
        let tok = tokens[t] as usize;

        if t > 0 {
            // Check confidence skip: can we use N-gram alone?
            let (conf, _best_tok, _best_order) = if skip_active {
                ngram.confidence()
            } else {
                (0.0, 0, 0)
            };
            let do_skip = skip_active && conf >= skip_threshold;

            let ensemble_logits = if do_skip {
                // Use N-gram standalone prediction (skip RWKV head)
                skipped += 1;
                let mut ng_logits = vec![0.0f32; v];
                ngram.predict_standalone(&mut ng_logits);
                // Still apply bias head
                bias.apply(&mut ng_logits);
                ng_logits
            } else if use_mix {
                // Adaptive mixer: compute components separately, combine with learned weights
                let h = if adaptive_ngram {
                    Some(entropy_from_logits(&logits))
                } else {
                    None
                };
                let ng_bias = ngram.compute_bias(h);
                let b_vec = bias.bias_vector();
                mixer.combine(&logits, &ng_bias, b_vec)
            } else {
                // Normal ensemble: RWKV + N-gram + bias
                let mut el = logits.clone();
                if use_ensemble {
                    let h = if adaptive_ngram {
                        Some(entropy_from_logits(&logits))
                    } else {
                        None
                    };
                    ngram.predict(&mut el, h);
                    bias.apply(&mut el);
                }
                el
            };

            // Compute probability of this token given context
            let ensemble_tensor = crate::domain::tensor::Tensor::from_data(
                ensemble_logits, vec![v],
            );
            let probs = softmax(&ensemble_tensor);
            let prob = probs.data[tok] as f64;

            // Update online components AFTER measuring (no lookahead)
            if use_mix {
                // Update mixer weights based on component contributions
                let h = if adaptive_ngram {
                    Some(entropy_from_logits(&logits))
                } else {
                    None
                };
                let ng_bias = ngram.compute_bias(h);
                let b_vec = bias.bias_vector();
                mixer.update(&probs.data, &ng_bias, b_vec, tok);
                bias.update(&probs.data, tok);
            } else if use_ensemble {
                bias.update(&probs.data, tok);
            }

            // Update telemetry extra info + build log snapshot
            let (eff_lr, ema_s) = bias.telemetry();
            let snap = LogSnapshot {
                w_ngram: if use_mix { mixer.w_ngram } else { 0.0 },
                w_bias: if use_mix { mixer.w_bias } else { 0.0 },
                eff_lr,
                ema_surprise: ema_s,
            };
            if use_mix {
                tracker.set_extra(format!("w_ng={:.3} w_b={:.3}", snap.w_ngram, snap.w_bias));
            } else if use_ensemble && lr_tau > 0.0 {
                tracker.set_extra(format!("lr={:.4} surp={:.2}", snap.eff_lr, snap.ema_surprise));
            } else if use_ensemble && adaptive_ngram {
                let h = entropy_from_logits(&logits);
                let beta = 0.6f32;
                let s = ((1.0 - beta) + beta * (h / 5.5)).clamp(0.2, 2.5);
                tracker.set_extra(format!("H={:.2} s={:.3}", h, s * ngram_scale));
            }

            // Level 2: structured log
            if let Some(ref mut log) = logger {
                log.record_token(token_byte_lengths[t], prob, &snap);
            }

            // Distribute this token's bits across its bytes
            let n_bytes = token_byte_lengths[t];
            let bits_per_byte = -prob.max(1e-30).log2() / n_bytes as f64;
            let byte_prob = (-(bits_per_byte * std::f64::consts::LN_2)).exp();
            for _ in 0..n_bytes {
                tracker.record_byte(byte_prob);
            }
        }

        // Update N-gram history (always, even at t=0)
        if use_ensemble {
            ngram.observe(tok as u32);
        }

        // Always run RWKV forward to maintain state
        model.forward_into(tok, &mut state, &mut scratch, &mut logits);
    }

    if skip_active {
        let pct = 100.0 * skipped as f64 / (tokens.len() - 1) as f64;
        eprintln!("[baseline] confidence skip: {}/{} tokens skipped ({:.1}%)",
                  skipped, tokens.len() - 1, pct);
    }

    // Finalize Level 2 log
    if let Some(ref mut log) = logger {
        let (eff_lr, ema_s) = bias.telemetry();
        let snap = LogSnapshot {
            w_ngram: if use_mix { mixer.w_ngram } else { 0.0 },
            w_bias: if use_mix { mixer.w_bias } else { 0.0 },
            eff_lr,
            ema_surprise: ema_s,
        };
        log.finalize(&snap);
        eprintln!("[baseline] log written: {}", log_path.as_ref().unwrap());
    }

    tracker.final_report();

    eprintln!();
    eprintln!("[baseline] {} BPB on {}: {:.4}",
              mode_str, input_path, tracker.bpb());
    eprintln!("[baseline] done.");
}

fn cmd_rwkv_test(args: &[String]) {
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();
    let mut prompt = "The meaning of life is".to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => { i += 1; weights_dir = args[i].clone(); }
            "--prompt" => { i += 1; prompt = args[i].clone(); }
            _ => {}
        }
        i += 1;
    }

    let model_path = Path::new(&weights_dir).join("model.safetensors");
    let vocab_path = Path::new(&weights_dir).join("rwkv_vocab_v20230424.txt");

    eprintln!("[rwkv-test] loading tokenizer ...");
    let tokenizer = WorldTokenizer::load(&vocab_path);
    eprintln!("[rwkv-test] tokenizer loaded: {} entries", tokenizer.vocab_size());

    let config = Rwkv7Config::from_weights_dir(&weights_dir);
    let model = Rwkv7Model::load(&model_path, config);

    let tokens = tokenizer.encode(prompt.as_bytes());
    eprintln!("[rwkv-test] prompt: {:?} ({} tokens)", prompt, tokens.len());

    let mut state = Rwkv7State::new(&model.config);
    let mut scratch = model.create_scratch();

    // Prefill: process all prompt tokens
    let mut logits = vec![0.0f32; model.config.vocab_size];
    let t_infer = std::time::Instant::now();
    for &tok in &tokens {
        model.forward_into(tok as usize, &mut state, &mut scratch, &mut logits);
    }
    let prefill_ms = t_infer.elapsed().as_millis();

    // Show top-10 predictions after prompt
    let logits_tensor = crate::domain::tensor::Tensor::from_data(logits.clone(), vec![model.config.vocab_size]);
    let probs = softmax(&logits_tensor);
    let mut indexed: Vec<(usize, f32)> = probs.data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!();
    eprintln!("[rwkv-test] top-10 predictions after prompt:");
    for (idx, prob) in indexed.iter().take(10) {
        let token_bytes = tokenizer.decode_token(*idx as u32);
        let token_str = String::from_utf8_lossy(token_bytes);
        eprintln!("  {:>6.2}%  {:?}", prob * 100.0, token_str);
    }

    // Generate 20 tokens greedily
    let gen_count = 20;
    eprintln!();
    eprint!("[rwkv-test] greedy generation: {}", prompt);
    let t_gen = std::time::Instant::now();
    for _ in 0..gen_count {
        let logits_tensor = crate::domain::tensor::Tensor::from_data(logits.clone(), vec![model.config.vocab_size]);
        let probs = softmax(&logits_tensor);
        let mut best_idx = 0;
        let mut best_prob = 0.0f32;
        for (i, &p) in probs.data.iter().enumerate() {
            if p > best_prob {
                best_prob = p;
                best_idx = i;
            }
        }
        let token_bytes = tokenizer.decode_token(best_idx as u32);
        let token_str = String::from_utf8_lossy(token_bytes);
        eprint!("{}", token_str);
        model.forward_into(best_idx, &mut state, &mut scratch, &mut logits);
    }
    let gen_ms = t_gen.elapsed().as_millis();
    let total_tokens = tokens.len() + gen_count;
    let total_ms = t_infer.elapsed().as_millis();
    eprintln!();
    eprintln!();
    eprintln!("[rwkv-test] performance:");
    eprintln!("  prefill: {} tokens in {} ms ({:.1} ms/tok)",
              tokens.len(), prefill_ms, prefill_ms as f64 / tokens.len() as f64);
    eprintln!("  generate: {} tokens in {} ms ({:.1} ms/tok)",
              gen_count, gen_ms, gen_ms as f64 / gen_count as f64);
    eprintln!("  total: {} tokens in {} ms ({:.1} tok/s)",
              total_tokens, total_ms, total_tokens as f64 / (total_ms as f64 / 1000.0));
    eprintln!();

    // Estimate enwik8 time
    let ms_per_tok = total_ms as f64 / total_tokens as f64;
    // enwik8 ≈ 100M bytes, avg ~2.5 bytes/token with World tokenizer
    let est_tokens_enwik8 = 100_000_000.0 / 2.5;
    let est_hours = (est_tokens_enwik8 * ms_per_tok) / 3_600_000.0;
    eprintln!("[rwkv-test] enwik8 estimate: ~{:.1}h ({:.0} ms/tok, ~{:.0}M tokens)",
              est_hours, ms_per_tok, est_tokens_enwik8 / 1_000_000.0);
    eprintln!("[rwkv-test] done.");
}

#[cfg(test)]
mod tests {
    #[test]
    fn skeleton_compiles() {
        assert!(true);
    }
}
