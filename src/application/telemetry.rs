//! Telemetry — Level 1 (stderr) + Level 2 (structured .jsonl log).
//!
//! Level 1: real-time progress on stderr.
//!   Format: [timestamp Barcelona] progress% | BPB | B/s | ETA
//!   Frequency: every ~1M bytes (or 4 reports for corpus <4MB).
//!
//! Level 2: structured JSON-lines log for post-analysis.
//!   One record per N bytes with cumulative + windowed BPB, throughput,
//!   mixer weights, bias head state. Queryable with jq/Python/Excel.
//!   Overhead: < 0.01%.

use std::io::Write;
use std::time::Instant;

pub struct ProgressTracker {
    total_bytes: usize,
    processed_bytes: usize,
    total_log_loss: f64,
    start_time: Instant,
    last_report_bytes: usize,
    report_interval: usize,
    extra: String,
}

impl ProgressTracker {
    pub fn new(total_bytes: usize) -> Self {
        let report_interval = if total_bytes < 4_000_000 {
            total_bytes / 4
        } else {
            1_000_000
        }.max(1);

        Self {
            total_bytes,
            processed_bytes: 0,
            total_log_loss: 0.0,
            start_time: Instant::now(),
            last_report_bytes: 0,
            report_interval,
            extra: String::new(),
        }
    }

    /// Record one byte prediction. `prob` is the predicted probability of the actual byte.
    pub fn record_byte(&mut self, prob: f64) {
        self.processed_bytes += 1;
        self.total_log_loss += -prob.max(1e-30).ln();

        if self.processed_bytes - self.last_report_bytes >= self.report_interval
            || self.processed_bytes == self.total_bytes
        {
            self.report();
            self.last_report_bytes = self.processed_bytes;
        }
    }

    /// Set extra telemetry info to display in progress reports.
    pub fn set_extra(&mut self, extra: String) {
        self.extra = extra;
    }

    pub fn bpb(&self) -> f64 {
        if self.processed_bytes == 0 { return 0.0; }
        self.total_log_loss / (self.processed_bytes as f64 * std::f64::consts::LN_2)
    }

    fn report(&self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let pct = 100.0 * self.processed_bytes as f64 / self.total_bytes as f64;
        let bpb = self.bpb();
        let bps = self.processed_bytes as f64 / elapsed.max(0.001);
        let remaining = (self.total_bytes - self.processed_bytes) as f64 / bps.max(1.0);

        let eta_h = (remaining / 3600.0) as u32;
        let eta_m = ((remaining % 3600.0) / 60.0) as u32;
        let eta_s = (remaining % 60.0) as u32;

        let timestamp = now_barcelona();

        if self.extra.is_empty() {
            eprintln!(
                "[{}] {:5.1}% | BPB {:.4} | {:.0} B/s | ETA {}h{:02}m{:02}s",
                timestamp, pct, bpb, bps, eta_h, eta_m, eta_s,
            );
        } else {
            eprintln!(
                "[{}] {:5.1}% | BPB {:.4} | {:.0} B/s | ETA {}h{:02}m{:02}s | {}",
                timestamp, pct, bpb, bps, eta_h, eta_m, eta_s, self.extra,
            );
        }
    }

    pub fn final_report(&self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let bpb = self.bpb();
        let bps = self.processed_bytes as f64 / elapsed.max(0.001);
        let timestamp = now_barcelona();

        eprintln!();
        eprintln!("[{}] DONE | {} bytes | BPB {:.6} | {:.0} B/s | {:.1}s",
                  timestamp, self.processed_bytes, bpb, bps, elapsed);
    }
}

/// Current timestamp in Barcelona timezone (CET=UTC+1, CEST=UTC+2).
pub fn now_barcelona() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Approximate CEST (last Sunday of March to last Sunday of October)
    // For correctness: just use UTC+2 during summer, UTC+1 during winter.
    // Simple heuristic: months 4-10 are CEST (UTC+2), rest CET (UTC+1).
    let utc_secs = now as i64;
    let days = utc_secs / 86400;
    // Jan 1, 1970 was Thursday (day 4). Compute month approximately.
    let year_approx = 1970 + (days as f64 / 365.25) as i64;
    let day_of_year = days - ((year_approx - 1970) as f64 * 365.25) as i64;
    let month_approx = (day_of_year as f64 / 30.44) as i64 + 1;
    let offset = if (4..=10).contains(&month_approx) { 2 } else { 1 };

    let local_secs = utc_secs + offset * 3600;
    let local_day_secs = ((local_secs % 86400) + 86400) % 86400;
    let h = local_day_secs / 3600;
    let m = (local_day_secs % 3600) / 60;
    let s = local_day_secs % 60;

    format!("{:02}:{:02}:{:02} UTC+{}", h, m, s, offset)
}

// ---- Level 2: Structured JSON-lines log ----

/// Snapshot of component state for structured logging (baseline command).
#[derive(Clone, Copy, Default)]
pub struct LogSnapshot {
    pub w_ngram: f32,
    pub w_bias: f32,
    pub eff_lr: f32,
    pub ema_surprise: f32,
}

/// Snapshot for hybrid-eval structured logging.
#[derive(Clone, Default)]
pub struct HybridLogSnapshot {
    /// Per-bit cost for current byte (bits 0-7, MSB first)
    pub bit_costs: [f64; 8],
    /// Match model longest match length (0 = no match)
    pub match_len: usize,
}

/// Structured .jsonl logger for hybrid-eval post-analysis.
/// Records per-byte BPB, per-bit costs, throughput, match quality.
pub struct HybridLogger {
    file: std::io::BufWriter<std::fs::File>,
    interval_bytes: usize,
    start_time: Instant,
    // Cumulative
    cum_bytes: usize,
    cum_bits: f64,
    // Window (between flushes)
    win_bytes: usize,
    win_bits: f64,
    // Per-bit cost accumulators (window)
    win_bit_costs: [f64; 8],
    win_bit_counts: usize,
    // Match stats (window)
    win_match_hits: usize,
    win_match_len_sum: usize,
}

impl HybridLogger {
    pub fn new(path: &str, total_bytes: usize) -> Self {
        let file = std::fs::File::create(path)
            .unwrap_or_else(|e| panic!("cannot create log file {}: {}", path, e));
        let interval = (total_bytes / 100).clamp(100, 100_000);
        eprintln!("[hybrid] logging to: {}", path);
        Self {
            file: std::io::BufWriter::new(file),
            interval_bytes: interval,
            start_time: Instant::now(),
            cum_bytes: 0,
            cum_bits: 0.0,
            win_bytes: 0,
            win_bits: 0.0,
            win_bit_costs: [0.0; 8],
            win_bit_counts: 0,
            win_match_hits: 0,
            win_match_len_sum: 0,
        }
    }

    /// Record one byte's total cost and per-bit breakdown.
    pub fn record_byte(&mut self, byte_bits: f64, snap: &HybridLogSnapshot) {
        self.cum_bytes += 1;
        self.cum_bits += byte_bits;
        self.win_bytes += 1;
        self.win_bits += byte_bits;

        for i in 0..8 {
            self.win_bit_costs[i] += snap.bit_costs[i];
        }
        self.win_bit_counts += 1;

        if snap.match_len > 0 {
            self.win_match_hits += 1;
            self.win_match_len_sum += snap.match_len;
        }

        if self.win_bytes >= self.interval_bytes {
            self.flush();
        }
    }

    pub fn finalize(&mut self) {
        if self.win_bytes > 0 {
            self.flush();
        }
    }

    fn flush(&mut self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let bpb = self.cum_bits / self.cum_bytes.max(1) as f64;
        let bpb_w = self.win_bits / self.win_bytes.max(1) as f64;
        let bps = self.cum_bytes as f64 / elapsed.max(0.001);
        let ts = now_barcelona();

        // Average per-bit costs for this window
        let n = self.win_bit_counts.max(1) as f64;
        let bc: Vec<String> = self.win_bit_costs.iter()
            .map(|c| format!("{:.4}", c / n))
            .collect();

        let avg_match = if self.win_match_hits > 0 {
            self.win_match_len_sum as f64 / self.win_match_hits as f64
        } else {
            0.0
        };

        let _ = writeln!(self.file,
            "{{\"ts\":\"{}\",\"sec\":{:.1},\"bytes\":{},\"bpb\":{:.6},\"bpb_w\":{:.6},\"bps\":{:.0},\"bit_costs\":[{}],\"match_hits\":{},\"match_avg_len\":{:.1}}}",
            ts, elapsed, self.cum_bytes, bpb, bpb_w, bps,
            bc.join(","), self.win_match_hits, avg_match,
        );

        self.win_bytes = 0;
        self.win_bits = 0.0;
        self.win_bit_costs = [0.0; 8];
        self.win_bit_counts = 0;
        self.win_match_hits = 0;
        self.win_match_len_sum = 0;
    }
}

/// Structured .jsonl logger for post-run analysis.
pub struct JsonLogger {
    file: std::io::BufWriter<std::fs::File>,
    interval_bytes: usize,
    start_time: Instant,
    // Cumulative
    cum_bytes: usize,
    cum_log_loss: f64,
    cum_tokens: usize,
    // Window (between flushes)
    win_bytes: usize,
    win_log_loss: f64,
    win_tokens: usize,
}

impl JsonLogger {
    /// Create a new logger writing to `path`.
    /// `total_bytes` is used to auto-size the flush interval (~100-1000 records).
    pub fn new(path: &str, total_bytes: usize) -> Self {
        let file = std::fs::File::create(path)
            .unwrap_or_else(|e| panic!("cannot create log file {}: {}", path, e));
        let interval = (total_bytes / 100).clamp(100, 100_000);
        Self {
            file: std::io::BufWriter::new(file),
            interval_bytes: interval,
            start_time: Instant::now(),
            cum_bytes: 0,
            cum_log_loss: 0.0,
            cum_tokens: 0,
            win_bytes: 0,
            win_log_loss: 0.0,
            win_tokens: 0,
        }
    }

    /// Record one token's prediction. `n_bytes` = byte length of token, `prob` = predicted probability.
    pub fn record_token(&mut self, n_bytes: usize, prob: f64, snap: &LogSnapshot) {
        let log_loss = -prob.max(1e-30).ln(); // total nats for this token
        self.cum_bytes += n_bytes;
        self.cum_log_loss += log_loss;
        self.cum_tokens += 1;
        self.win_bytes += n_bytes;
        self.win_log_loss += log_loss;
        self.win_tokens += 1;

        if self.win_bytes >= self.interval_bytes {
            self.flush(snap);
        }
    }

    /// Flush remaining window data at end of run.
    pub fn finalize(&mut self, snap: &LogSnapshot) {
        if self.win_bytes > 0 {
            self.flush(snap);
        }
    }

    fn flush(&mut self, snap: &LogSnapshot) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let bpb = self.cum_log_loss / (self.cum_bytes as f64 * std::f64::consts::LN_2);
        let bpb_w = self.win_log_loss / (self.win_bytes.max(1) as f64 * std::f64::consts::LN_2);
        let bps = self.cum_bytes as f64 / elapsed.max(0.001);
        let ts = now_barcelona();

        let _ = writeln!(self.file,
            "{{\"ts\":\"{}\",\"sec\":{:.1},\"bytes\":{},\"tokens\":{},\"bpb\":{:.6},\"bpb_w\":{:.6},\"bps\":{:.0},\"w_ng\":{:.4},\"w_b\":{:.4},\"lr\":{:.4},\"surp\":{:.4}}}",
            ts, elapsed, self.cum_bytes, self.cum_tokens, bpb, bpb_w, bps,
            snap.w_ngram, snap.w_bias, snap.eff_lr, snap.ema_surprise,
        );

        self.win_bytes = 0;
        self.win_log_loss = 0.0;
        self.win_tokens = 0;
    }
}
