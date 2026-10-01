//! RWKV World tokenizer — 65,536 entries.
//!
//! Format of rwkv_vocab_v20230424.txt:
//!   <id> <repr> <len>
//! where repr is a Python literal (string or bytes).

use std::collections::HashMap;
use std::path::Path;

pub struct WorldTokenizer {
    idx_to_token: Vec<Vec<u8>>,
    token_to_idx: HashMap<Vec<u8>, u32>,
}

impl WorldTokenizer {
    pub fn load(path: &Path) -> Self {
        let content = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("failed to read tokenizer: {}", e));

        let mut idx_to_token: Vec<Vec<u8>> = Vec::with_capacity(65536);
        let mut token_to_idx = HashMap::with_capacity(65536);

        // Pre-fill with empty entries
        for _ in 0..65536 {
            idx_to_token.push(Vec::new());
        }

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() { continue; }

            // Parse: <id> <repr> <len>
            let first_space = line.find(' ').expect("bad tokenizer line");
            let id: usize = line[..first_space].parse().expect("bad token id");
            let rest = &line[first_space + 1..];
            let last_space = rest.rfind(' ').expect("bad tokenizer line");
            let repr = &rest[..last_space];
            // let _len: usize = rest[last_space + 1..].parse().expect("bad token len");

            let bytes = parse_python_literal(repr);

            if id < 65536 {
                token_to_idx.insert(bytes.clone(), id as u32);
                idx_to_token[id] = bytes;
            }
        }

        Self { idx_to_token, token_to_idx }
    }

    pub fn encode(&self, text: &[u8]) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut i = 0;
        while i < text.len() {
            // Greedy: try longest match first
            let max_len = (text.len() - i).min(64);
            let mut best_len = 1;
            let mut best_id = self.token_to_idx
                .get(&text[i..i + 1].to_vec())
                .copied()
                .unwrap_or(0);

            for len in (2..=max_len).rev() {
                let candidate = &text[i..i + len];
                if let Some(&id) = self.token_to_idx.get(candidate) {
                    best_id = id;
                    best_len = len;
                    break;
                }
            }

            tokens.push(best_id);
            i += best_len;
        }
        tokens
    }

    pub fn decode_token(&self, id: u32) -> &[u8] {
        &self.idx_to_token[id as usize]
    }

    pub fn vocab_size(&self) -> usize {
        self.idx_to_token.len()
    }
}

fn parse_python_literal(s: &str) -> Vec<u8> {
    let s = s.trim();
    if s.starts_with("b'") || s.starts_with("b\"") {
        // Bytes literal
        let quote = s.as_bytes()[1] as char;
        let inner = &s[2..s.len() - 1];
        unescape_bytes(inner, quote)
    } else if s.starts_with('\'') || s.starts_with('"') {
        // String literal — encode as UTF-8
        let quote = s.as_bytes()[0] as char;
        let inner = &s[1..s.len() - 1];
        let unescaped = unescape_string(inner, quote);
        unescaped.into_bytes()
    } else {
        s.as_bytes().to_vec()
    }
}

fn unescape_bytes(s: &str, _quote: char) -> Vec<u8> {
    let mut result = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            match bytes[i + 1] {
                b'\\' => { result.push(b'\\'); i += 2; }
                b'\'' => { result.push(b'\''); i += 2; }
                b'"' => { result.push(b'"'); i += 2; }
                b'n' => { result.push(b'\n'); i += 2; }
                b'r' => { result.push(b'\r'); i += 2; }
                b't' => { result.push(b'\t'); i += 2; }
                b'0' => { result.push(0); i += 2; }
                b'x' if i + 3 < bytes.len() => {
                    let hex = std::str::from_utf8(&bytes[i + 2..i + 4]).unwrap_or("00");
                    let val = u8::from_str_radix(hex, 16).unwrap_or(0);
                    result.push(val);
                    i += 4;
                }
                _ => { result.push(bytes[i + 1]); i += 2; }
            }
        } else {
            result.push(bytes[i]);
            i += 1;
        }
    }
    result
}

fn unescape_string(s: &str, _quote: char) -> String {
    let mut result = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('\\') => result.push('\\'),
                Some('\'') => result.push('\''),
                Some('"') => result.push('"'),
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('t') => result.push('\t'),
                Some('0') => result.push('\0'),
                Some('x') => {
                    let mut hex = String::new();
                    for _ in 0..2 {
                        if let Some(&c) = chars.peek() {
                            if c.is_ascii_hexdigit() {
                                hex.push(c);
                                chars.next();
                            }
                        }
                    }
                    let val = u32::from_str_radix(&hex, 16).unwrap_or(0);
                    if let Some(ch) = char::from_u32(val) { result.push(ch); }
                }
                Some('u') => {
                    let mut hex = String::new();
                    for _ in 0..4 {
                        if let Some(&c) = chars.peek() {
                            if c.is_ascii_hexdigit() {
                                hex.push(c);
                                chars.next();
                            }
                        }
                    }
                    let val = u32::from_str_radix(&hex, 16).unwrap_or(0);
                    if let Some(ch) = char::from_u32(val) { result.push(ch); }
                }
                Some(other) => { result.push('\\'); result.push(other); }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }
    result
}
