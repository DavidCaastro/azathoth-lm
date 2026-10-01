mod domain;
mod application;
mod infrastructure;

use std::path::Path;

use crate::domain::tensor::softmax;
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
    eprintln!("  rwkv-test   --weights DIR [--prompt TEXT]");
    eprintln!("  info        --ckpt PATH");
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

    let config = Rwkv7Config::default_0_1b();
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
