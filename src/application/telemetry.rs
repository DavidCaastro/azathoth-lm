//! Telemetry — Level 1: real-time progress on stderr.
//!
//! Format: [timestamp Barcelona] progress% | BPB | B/s | ETA | MB_RAM
//! Frequency: every ~1M bytes (or 4 reports for corpus <4MB).

use std::time::Instant;

pub struct ProgressTracker {
    total_bytes: usize,
    processed_bytes: usize,
    total_log_loss: f64,
    start_time: Instant,
    last_report_bytes: usize,
    report_interval: usize,
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

        eprintln!(
            "[{}] {:5.1}% | BPB {:.4} | {:.0} B/s | ETA {}h{:02}m{:02}s",
            timestamp, pct, bpb, bps, eta_h, eta_m, eta_s,
        );
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
fn now_barcelona() -> String {
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
