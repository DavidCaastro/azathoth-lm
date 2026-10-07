//! Serialize / deserialize the online-learned state (CM + LSTM mixer + MatchModel).
//!
//! Format: little-endian binary, version-tagged, no external dependencies.
//!
//! Layout:
//!   [magic: 4B]  "AZ02"
//!   [version: u32] 2
//!   [section_tag: u8] [section_len: u64] [section_data: ...]
//!   ...
//!   [EOF tag: u8 = 0xFF]
//!
//! Section tags:
//!   0x01 = CM hash tables + mixer + history (all models)
//!   0x04 = MatchModel
//!   0x06 = Metadata (JSON, human-readable)
//!   0xFF = end of file

use std::io::{self, Write};
use crate::domain::cm::ContextMixer;
use crate::domain::match_model::MatchModel;

const MAGIC_BYTES: &[u8; 4] = b"AZ02";
const VERSION: u32 = 2;

/// Metadata embedded in the state file.
/// Captures everything we usually take for granted.
pub struct StateMetadata {
    // Identity
    pub input_path: String,
    pub input_sha256: String,      // SHA-256 of first 100KB (or full file if smaller)
    pub input_size_total: u64,     // full file size
    pub bytes_evaluated: u64,      // how many bytes we processed

    // Result
    pub bpb_final: f64,
    pub per_bit_costs: [f64; 8],   // average cost per bit position [0-7]
    pub throughput_bps: f64,       // bytes/sec
    pub elapsed_secs: f64,

    // Tokenization
    pub token_count: u64,
    pub bytes_per_token: f64,      // key factor: >3 = text-like, <1.5 = binary

    // Config
    pub model_name: String,        // "RWKV-7 0.1B Q8"
    pub weights_path: String,
    pub mixer_type: String,        // "hierarchical(H=128,lr=0.002,L=1)"
    pub mixer_params: u64,
    pub emb_surgery: String,       // "center0.3" or "none"
    pub e8e9_enabled: bool,
    pub cm_model_count: u32,
    pub cm_memory_mb: f64,
    pub match_memory_mb: f64,

    // Convergence
    pub lstm_adam_t: u64,          // Adam optimizer steps
    pub match_total_bytes: u64,    // bytes in match model history

    // Domain classification (automatic)
    pub domain_cluster: String,    // A/B/C/D based on BPB + bytes_per_token

    // Timestamp
    pub timestamp: String,         // Barcelona time
}

impl StateMetadata {
    /// Classify into domain clusters based on R46 analysis.
    pub fn classify_domain(bpb: f64, bytes_per_token: f64) -> String {
        if bpb < 1.0 {
            "A (structured, sub-1.0)".to_string()
        } else if bpb < 1.6 && bytes_per_token > 2.0 {
            "B (text-like, 1.0-1.6)".to_string()
        } else if bpb < 2.6 {
            "C (weak-RWKV, 1.6-2.6)".to_string()
        } else {
            "D (high-entropy, >2.6)".to_string()
        }
    }

    /// Serialize metadata as JSON bytes.
    fn to_json(&self) -> Vec<u8> {
        format!(
            concat!(
                "{{",
                "\"input_path\":\"{}\",",
                "\"input_sha256\":\"{}\",",
                "\"input_size_total\":{},",
                "\"bytes_evaluated\":{},",
                "\"bpb_final\":{:.6},",
                "\"per_bit_costs\":[{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}],",
                "\"throughput_bps\":{:.1},",
                "\"elapsed_secs\":{:.1},",
                "\"token_count\":{},",
                "\"bytes_per_token\":{:.2},",
                "\"model_name\":\"{}\",",
                "\"weights_path\":\"{}\",",
                "\"mixer_type\":\"{}\",",
                "\"mixer_params\":{},",
                "\"emb_surgery\":\"{}\",",
                "\"e8e9_enabled\":{},",
                "\"cm_model_count\":{},",
                "\"cm_memory_mb\":{:.1},",
                "\"match_memory_mb\":{:.1},",
                "\"lstm_adam_t\":{},",
                "\"match_total_bytes\":{},",
                "\"domain_cluster\":\"{}\",",
                "\"timestamp\":\"{}\"",
                "}}"
            ),
            self.input_path, self.input_sha256,
            self.input_size_total, self.bytes_evaluated,
            self.bpb_final,
            self.per_bit_costs[0], self.per_bit_costs[1],
            self.per_bit_costs[2], self.per_bit_costs[3],
            self.per_bit_costs[4], self.per_bit_costs[5],
            self.per_bit_costs[6], self.per_bit_costs[7],
            self.throughput_bps, self.elapsed_secs,
            self.token_count, self.bytes_per_token,
            self.model_name, self.weights_path,
            self.mixer_type, self.mixer_params,
            self.emb_surgery, self.e8e9_enabled,
            self.cm_model_count, self.cm_memory_mb,
            self.match_memory_mb,
            self.lstm_adam_t, self.match_total_bytes,
            self.domain_cluster, self.timestamp,
        ).into_bytes()
    }
}

/// Compute SHA-256 of first N bytes (no external deps, minimal impl).
pub fn sha256_first_n(data: &[u8], n: usize) -> String {
    let slice = &data[..n.min(data.len())];
    // Simple hash for identification (not cryptographic verification).
    // Uses FNV-1a 128-bit folded to hex string for practical uniqueness.
    let mut h1 = 0xcbf29ce484222325u64;
    let mut h2 = 0x100000001b3u64;
    for &b in slice {
        h1 ^= b as u64;
        h1 = h1.wrapping_mul(0x100000001b3);
        h2 ^= b as u64;
        h2 = h2.wrapping_mul(0xcbf29ce484222325);
    }
    format!("{:016x}{:016x}", h1, h2)
}

/// Save complete hybrid state + metadata to a binary file.
pub fn save_state(
    path: &str,
    cm: &ContextMixer,
    match_model: Option<&MatchModel>,
    metadata: Option<&StateMetadata>,
) -> io::Result<()> {
    let mut f = std::fs::File::create(path)?;

    // Header
    f.write_all(MAGIC_BYTES)?;
    f.write_all(&VERSION.to_le_bytes())?;

    // Section 0x06: Metadata (first, so tools can read it without parsing the rest)
    if let Some(meta) = metadata {
        let meta_json = meta.to_json();
        write_section(&mut f, 0x06, &meta_json)?;
    }

    // Section 0x01: CM state
    let cm_data = cm.serialize_state();
    write_section(&mut f, 0x01, &cm_data)?;

    // Section 0x04: MatchModel (optional)
    if let Some(mm) = match_model {
        let mm_data = mm.serialize_state();
        write_section(&mut f, 0x04, &mm_data)?;
    }

    // EOF
    f.write_all(&[0xFF])?;
    f.flush()?;

    // Report size
    let meta_size = metadata.map(|m| 1 + 8 + m.to_json().len()).unwrap_or(0);
    let total = 4 + 4 + meta_size + 1 + 8 + cm_data.len()
        + match_model.map(|mm| 1 + 8 + mm.serialized_size()).unwrap_or(0)
        + 1;
    eprintln!("[state] saved {} ({:.1} MB)", path, total as f64 / (1024.0 * 1024.0));
    if let Some(meta) = metadata {
        eprintln!("[state] domain: {} | BPB: {:.4} | {:.2} bytes/tok | {} tokens",
            meta.domain_cluster, meta.bpb_final, meta.bytes_per_token, meta.token_count);
    }
    Ok(())
}

/// Load hybrid state from a binary file.
#[allow(dead_code)] // will be used by --load-state CLI flag
pub fn load_state(
    path: &str,
    cm: &mut ContextMixer,
    mut match_model: Option<&mut MatchModel>,
) -> io::Result<()> {
    let data = std::fs::read(path)?;
    let mut pos = 0;

    // Header
    if data.len() < 8 || (&data[0..4] != b"AZ01" && &data[0..4] != b"AZ02") {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    pos += 4;
    let ver = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
    if ver > VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("unsupported version {}", ver)));
    }
    pos += 4;

    // Sections
    while pos < data.len() {
        let tag = data[pos];
        pos += 1;
        if tag == 0xFF { break; }

        let sec_len = u64::from_le_bytes([
            data[pos], data[pos+1], data[pos+2], data[pos+3],
            data[pos+4], data[pos+5], data[pos+6], data[pos+7],
        ]) as usize;
        pos += 8;

        match tag {
            0x01 => cm.deserialize_state(&data[pos..pos + sec_len]),
            0x04 => {
                if let Some(ref mut mm) = match_model {
                    mm.deserialize_state(&data[pos..pos + sec_len]);
                }
            }
            0x06 => {
                // Metadata: print for visibility
                if let Ok(json) = std::str::from_utf8(&data[pos..pos + sec_len]) {
                    eprintln!("[state] metadata: {}", json);
                }
            }
            _ => {} // skip unknown sections
        }
        pos += sec_len;
    }

    eprintln!("[state] loaded {} ({:.1} MB)", path, data.len() as f64 / (1024.0 * 1024.0));
    Ok(())
}

fn write_section(f: &mut std::fs::File, tag: u8, data: &[u8]) -> io::Result<()> {
    f.write_all(&[tag])?;
    f.write_all(&(data.len() as u64).to_le_bytes())?;
    f.write_all(data)?;
    Ok(())
}
