mod domain;
mod application;
mod infrastructure;

use std::path::Path;

use crate::domain::tensor::softmax;
use crate::domain::ngram::TokenNgram;
use crate::domain::bias_head::BiasHead;
use crate::application::telemetry::ProgressTracker;
use crate::infrastructure::rwkv7::model::{Rwkv7Config, Rwkv7Model, Rwkv7State};
use crate::infrastructure::rwkv7::tokenizer::WorldTokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "compress" => todo!("Phase 1: CM + RWKV hybrid predictor"),
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
    eprintln!("  compress    --input PATH [--ckpt PATH]");
    eprintln!("  baseline    --input PATH [--weights DIR] [--bytes N]");
    eprintln!("  rwkv-test   --weights DIR [--prompt TEXT]");
    eprintln!("  info        --ckpt PATH");
}

fn cmd_baseline(args: &[String]) {
    let mut input_path = "data/enwik8".to_string();
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();
    let mut max_bytes: usize = 0; // 0 = entire file
    let mut use_ensemble = false;
    let mut skip_threshold: f32 = 0.0;
    let mut bias_lr: f32 = 0.001;
    let mut ngram_scale: f32 = 1.0; // multiplier on N-gram weights

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
    } else {
        "ensemble (RWKV + N-gram + bias)".to_string()
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
    let mut bias = BiasHead::new(v, bias_lr);

    // Run forward pass and measure cross-entropy
    let mut state = Rwkv7State::new(&model.config);
    let mut tracker = ProgressTracker::new(total_bytes);

    eprintln!("[baseline] starting evaluation ...");
    eprintln!();

    let mut logits = crate::domain::tensor::Tensor::zeros(&[v]);
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
            } else {
                // Normal ensemble: RWKV + N-gram + bias
                let mut el = logits.data.clone();
                if use_ensemble {
                    ngram.predict(&mut el);
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
            if use_ensemble {
                bias.update(&probs.data, tok);
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
        logits = model.forward(tok, &mut state);
    }

    if skip_active {
        let pct = 100.0 * skipped as f64 / (tokens.len() - 1) as f64;
        eprintln!("[baseline] confidence skip: {}/{} tokens skipped ({:.1}%)",
                  skipped, tokens.len() - 1, pct);
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

    // Prefill: process all prompt tokens
    let mut logits = crate::domain::tensor::Tensor::zeros(&[model.config.vocab_size]);
    let t_infer = std::time::Instant::now();
    for &tok in &tokens {
        logits = model.forward(tok as usize, &mut state);
    }
    let prefill_ms = t_infer.elapsed().as_millis();

    // Show top-10 predictions after prompt
    let probs = softmax(&logits);
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
        let probs = softmax(&logits);
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
        logits = model.forward(best_idx, &mut state);
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
