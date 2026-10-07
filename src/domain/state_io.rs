//! Serialize / deserialize the online-learned state (CM + LSTM mixer + MatchModel).
//!
//! Format: little-endian binary, version-tagged, no external dependencies.
//!
//! Layout:
//!   [magic: u32]  "AZ01"
//!   [version: u32] 1
//!   [section_tag: u8] [section_len: u64] [section_data: ...]
//!   ...
//!   [EOF tag: u8 = 0xFF]
//!
//! Section tags:
//!   0x01 = CM hash tables (all models)
//!   0x02 = CM mixer weights (logistic or hierarchical sub-mixers)
//!   0x03 = LSTM mixer (top-level)
//!   0x04 = MatchModel
//!   0x05 = CM history + word state
//!   0xFF = end of file

use std::io::{self, Write};
use crate::domain::cm::ContextMixer;
use crate::domain::match_model::MatchModel;

const MAGIC_BYTES: &[u8; 4] = b"AZ01";
const VERSION: u32 = 1;

/// Save complete hybrid state to a binary file.
pub fn save_state(
    path: &str,
    cm: &ContextMixer,
    match_model: Option<&MatchModel>,
) -> io::Result<()> {
    let mut f = std::fs::File::create(path)?;

    // Header
    f.write_all(MAGIC_BYTES)?;
    f.write_all(&VERSION.to_le_bytes())?;

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

    let total = 4 + 4 + 1 + 8 + cm_data.len()
        + match_model.map(|mm| 1 + 8 + mm.serialized_size()).unwrap_or(0)
        + 1;
    eprintln!("[state] saved {} ({:.1} MB)", path, total as f64 / (1024.0 * 1024.0));
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
    if data.len() < 8 || &data[0..4] != MAGIC_BYTES {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    pos += 4;
    let ver = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
    if ver != VERSION {
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

        let sec_data = &data[pos..pos + sec_len];
        match tag {
            0x01 => cm.deserialize_state(sec_data),
            0x04 => {
                if let Some(ref mut mm) = match_model {
                    mm.deserialize_state(sec_data);
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
