#![feature(portable_simd)]

use bytemuck;

use core::simd::num::SimdFloat;
use core::simd::Simd;

#[derive(Debug, Clone)]
pub struct TelemetryColumnar<'a> {
    pub timestamp: &'a [f32],
    pub q_x: &'a [f32],
    pub q_y: &'a [f32],
    pub q_z: &'a [f32],
    pub cpu_utility: &'a [f32],
    pub entropy_flux: &'a [f32],
    pub entanglement_map: &'a [f32],
}

impl TelemetryColumnar<'_> {
    pub fn len(&self) -> usize {
        self.timestamp.len()
    }
}

pub fn parse_telemetry_binary(bytes: &[u8]) -> TelemetryColumnar {
    let data = bytemuck::cast_slice::<u8, f32>(bytes);
    let len = data.len() / 7;
    TelemetryColumnar {
        timestamp: &data[0..len],
        q_x: &data[len..2 * len],
        q_y: &data[2 * len..3 * len],
        q_z: &data[3 * len..4 * len],
        cpu_utility: &data[4 * len..5 * len],
        entropy_flux: &data[5 * len..6 * len],
        entanglement_map: &data[6 * len..7 * len],
    }
}

fn sum_simd_4(xs: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut i = 0usize;
    const LANES: usize = 4;

    while i + LANES <= xs.len() {
        let v = Simd::<f32, LANES>::from_slice(&xs[i..i + LANES]);
        acc += v.reduce_sum();
        i += LANES;
    }
    while i < xs.len() {
        acc += xs[i];
        i += 1;
    }
    acc
}

pub fn pearson_r_simd4(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }

    let mean_x = sum_simd_4(&x[..n]) / n as f32;
    let mean_y = sum_simd_4(&y[..n]) / n as f32;

    let mut num = 0.0f32;
    let mut denom_x = 0.0f32;
    let mut denom_y = 0.0f32;

    let mut i = 0usize;
    const LANES: usize = 4;

    while i + LANES <= n {
        let vx = Simd::<f32, LANES>::from_slice(&x[i..i + LANES]);
        let vy = Simd::<f32, LANES>::from_slice(&y[i..i + LANES]);

        let dx = vx - Simd::splat(mean_x);
        let dy = vy - Simd::splat(mean_y);

        num += (dx * dy).reduce_sum();
        denom_x += (dx * dx).reduce_sum();
        denom_y += (dy * dy).reduce_sum();
        i += LANES;
    }

    while i < n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        denom_x += dx * dx;
        denom_y += dy * dy;
        i += 1;
    }

    let denom = (denom_x * denom_y).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        num / denom
    }
}

pub fn variance_simd_4(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }

    let mean = sum_simd_4(xs) / xs.len() as f32;
    let mut acc = 0.0f32;
    let mut i = 0usize;
    const LANES: usize = 4;

    while i + LANES <= xs.len() {
        let v = Simd::<f32, LANES>::from_slice(&xs[i..i + LANES]);
        let d = v - Simd::splat(mean);
        acc += (d * d).reduce_sum();
        i += LANES;
    }

    while i < xs.len() {
        let d = xs[i] - mean;
        acc += d * d;
        i += 1;
    }

    acc / xs.len() as f32
}

pub fn cpu_correlations(telemetry: &TelemetryColumnar) -> [f32; 3] {
    // Correlate cpu_utility with the other scalar columns.
    // - timestamp
    // - entropy_flux
    // - entanglement_map
    let cpu = telemetry.cpu_utility;
    let r_t = pearson_r_simd4(cpu, telemetry.timestamp);
    let r_f = pearson_r_simd4(cpu, telemetry.entropy_flux);
    let r_e = pearson_r_simd4(cpu, telemetry.entanglement_map);
    [r_t, r_f, r_e]
}

pub fn select_top2_variance(telemetry: &TelemetryColumnar) -> (&'static str, &'static str) {
    // Only scalar properties participate for axis selection.
    let candidates: [(&'static str, f32); 4] = [
        ("timestamp", variance_simd_4(telemetry.timestamp)),
        ("cpu_utility", variance_simd_4(telemetry.cpu_utility)),
        ("entropy_flux", variance_simd_4(telemetry.entropy_flux)),
        ("entanglement_map", variance_simd_4(telemetry.entanglement_map)),
    ];

    let mut sorted = candidates;
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    (sorted[0].0, sorted[1].0)
}
