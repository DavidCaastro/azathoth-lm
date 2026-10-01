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
    for &tok in &tokens {
        logits = model.forward(tok as usize, &mut state);
    }

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
    eprintln!();
    eprint!("[rwkv-test] greedy generation: {}", prompt);
    for _ in 0..20 {
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
    eprintln!();
    eprintln!();
    eprintln!("[rwkv-test] done.");
}

#[cfg(test)]
mod tests {
    #[test]
    fn skeleton_compiles() {
        assert!(true);
    }
}
