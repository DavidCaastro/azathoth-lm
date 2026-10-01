fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "compress" => todo!("Phase 0: CM + RWKV hybrid predictor"),
        "info" => todo!("Checkpoint info"),
        _ => print_usage(),
    }
}

fn print_usage() {
    eprintln!("azathoth-lm — hybrid CM + neural byte-level predictor");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  compress  --input PATH [--ckpt PATH]");
    eprintln!("  info      --ckpt PATH");
}

#[cfg(test)]
mod tests {
    #[test]
    fn skeleton_compiles() {
        assert!(true);
    }
}
