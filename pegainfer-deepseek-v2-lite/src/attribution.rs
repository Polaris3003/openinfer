use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use anyhow::Result;
use serde::Serialize;

#[derive(Clone, Debug, Default)]
pub struct DecodeAttributionProfile {
    enabled: bool,
    total_generation_us: u64,
    prefill_next_token_us: Option<u64>,
    per_token_decode_us: Vec<u64>,
    samples: Vec<SectionSample>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SectionSample {
    pub phase: &'static str,
    pub section: &'static str,
    pub call_site: String,
    pub layer: Option<usize>,
    pub token_index: Option<usize>,
    pub elapsed_us: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SectionRollup {
    pub section: String,
    pub calls: usize,
    pub total_us: u64,
    pub mean_us: f64,
    pub min_us: u64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub max_us: u64,
    pub pct: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CallSiteRollup {
    pub call_site: String,
    pub section: String,
    pub calls: usize,
    pub total_us: u64,
    pub mean_us: f64,
    pub min_us: u64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub max_us: u64,
    pub pct: f64,
}

impl DecodeAttributionProfile {
    pub(crate) fn disabled() -> Self {
        Self::default()
    }

    pub(crate) fn enabled() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }

    pub(crate) fn set_total_generation(&mut self, elapsed: Duration) {
        if self.enabled {
            self.total_generation_us = micros(elapsed);
        }
    }

    pub(crate) fn set_prefill_next_token(&mut self, elapsed: Duration) {
        if self.enabled {
            self.prefill_next_token_us = Some(micros(elapsed));
        }
    }

    pub(crate) fn push_decode_token(&mut self, elapsed: Duration) {
        if self.enabled {
            self.per_token_decode_us.push(micros(elapsed));
        }
    }

    pub(crate) fn record_result<T, C, S>(
        &mut self,
        phase: &'static str,
        section: &'static str,
        call_site: C,
        layer: Option<usize>,
        token_index: Option<usize>,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T>
    where
        C: FnOnce() -> S,
        S: Into<String>,
    {
        if !self.enabled {
            return f();
        }
        let start = Instant::now();
        let result = f();
        self.samples.push(SectionSample {
            phase,
            section,
            call_site: call_site().into(),
            layer,
            token_index,
            elapsed_us: micros(start.elapsed()),
        });
        result
    }

    pub fn total_generation_us(&self) -> u64 {
        self.total_generation_us
    }

    pub fn prefill_next_token_us(&self) -> Option<u64> {
        self.prefill_next_token_us
    }

    pub fn per_token_decode_us(&self) -> &[u64] {
        &self.per_token_decode_us
    }

    pub fn by_section(&self) -> Vec<SectionRollup> {
        let total = self.samples.iter().map(|sample| sample.elapsed_us).sum();
        let mut groups: BTreeMap<&str, Vec<u64>> = BTreeMap::new();
        for sample in &self.samples {
            groups
                .entry(sample.section)
                .or_default()
                .push(sample.elapsed_us);
        }
        let mut rows: Vec<_> = groups
            .into_iter()
            .map(|(section, samples)| {
                let stats = sample_stats(samples);
                SectionRollup {
                    section: section.to_string(),
                    calls: stats.calls,
                    total_us: stats.total_us,
                    mean_us: stats.mean_us,
                    min_us: stats.min_us,
                    p50_us: stats.p50_us,
                    p95_us: stats.p95_us,
                    p99_us: stats.p99_us,
                    max_us: stats.max_us,
                    pct: pct(stats.total_us, total),
                }
            })
            .collect();
        rows.sort_by(|left, right| {
            right
                .total_us
                .cmp(&left.total_us)
                .then(left.section.cmp(&right.section))
        });
        rows
    }

    pub fn by_call_site(&self) -> Vec<CallSiteRollup> {
        let total = self.samples.iter().map(|sample| sample.elapsed_us).sum();
        let mut groups: BTreeMap<(&str, &str), Vec<u64>> = BTreeMap::new();
        for sample in &self.samples {
            groups
                .entry((sample.call_site.as_str(), sample.section))
                .or_default()
                .push(sample.elapsed_us);
        }
        let mut rows: Vec<_> = groups
            .into_iter()
            .map(|((call_site, section), samples)| {
                let stats = sample_stats(samples);
                CallSiteRollup {
                    call_site: call_site.to_string(),
                    section: section.to_string(),
                    calls: stats.calls,
                    total_us: stats.total_us,
                    mean_us: stats.mean_us,
                    min_us: stats.min_us,
                    p50_us: stats.p50_us,
                    p95_us: stats.p95_us,
                    p99_us: stats.p99_us,
                    max_us: stats.max_us,
                    pct: pct(stats.total_us, total),
                }
            })
            .collect();
        rows.sort_by(|left, right| {
            right
                .total_us
                .cmp(&left.total_us)
                .then(left.call_site.cmp(&right.call_site))
                .then(left.section.cmp(&right.section))
        });
        rows
    }
}

#[derive(Clone, Copy)]
struct SampleStats {
    calls: usize,
    total_us: u64,
    mean_us: f64,
    min_us: u64,
    p50_us: u64,
    p95_us: u64,
    p99_us: u64,
    max_us: u64,
}

fn sample_stats(mut samples: Vec<u64>) -> SampleStats {
    debug_assert!(!samples.is_empty());
    samples.sort_unstable();
    let total_us = samples.iter().sum();
    SampleStats {
        calls: samples.len(),
        total_us,
        mean_us: total_us as f64 / samples.len() as f64,
        min_us: samples[0],
        p50_us: percentile(&samples, 0.50),
        p95_us: percentile(&samples, 0.95),
        p99_us: percentile(&samples, 0.99),
        max_us: samples[samples.len() - 1],
    }
}

fn percentile(sorted: &[u64], quantile: f64) -> u64 {
    let idx = ((sorted.len() as f64 - 1.0) * quantile).ceil() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn pct(value: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        value as f64 / total as f64 * 100.0
    }
}

fn micros(duration: Duration) -> u64 {
    duration.as_micros().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollups_sort_by_total_time() {
        let mut profile = DecodeAttributionProfile::enabled();
        profile.samples.push(SectionSample {
            phase: "decode",
            section: "short",
            call_site: "layer.0.short".to_string(),
            layer: Some(0),
            token_index: Some(0),
            elapsed_us: 10,
        });
        profile.samples.push(SectionSample {
            phase: "decode",
            section: "long",
            call_site: "layer.0.long".to_string(),
            layer: Some(0),
            token_index: Some(0),
            elapsed_us: 30,
        });
        profile.samples.push(SectionSample {
            phase: "decode",
            section: "long",
            call_site: "layer.1.long".to_string(),
            layer: Some(1),
            token_index: Some(0),
            elapsed_us: 20,
        });

        let rows = profile.by_section();

        assert_eq!(rows[0].section, "long");
        assert_eq!(rows[0].calls, 2);
        assert_eq!(rows[0].total_us, 50);
        assert_eq!(rows[1].section, "short");
    }
}
