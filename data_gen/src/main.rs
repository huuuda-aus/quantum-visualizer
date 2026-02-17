use serde::Serialize;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

const DEFAULT_N: usize = 10_000;

#[derive(Serialize)]
struct TelemetryColumnar {
    timestamp: Vec<f32>,
    q_coords: Vec<[f32; 3]>,
    cpu_utility: Vec<f32>,
    entropy_flux: Vec<f32>,
    entanglement_map: Vec<f32>,
}

fn clamp01(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

fn parse_n_from_args() -> usize {
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--n" {
            if let Some(v) = args.next() {
                if let Ok(n) = v.parse::<usize>() {
                    return n.max(1);
                }
            }
        }
    }
    DEFAULT_N
}

fn main() -> anyhow::Result<()> {
    let n = parse_n_from_args();

    // Internal generator state (kept simple + deterministic enough to be reproducible-ish).
    let mut drift = 0.0f32;
    let mut flux_drift = 0.0f32;
    let mut regime: f32 = 1.0;
    let mut burst_energy = 0.0f32;

    let mut timestamp = Vec::with_capacity(n);
    let mut q_coords = Vec::with_capacity(n);
    let mut cpu_utility = Vec::with_capacity(n);
    let mut entropy_flux = Vec::with_capacity(n);
    let mut entanglement_map = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f32 * 0.016;
        timestamp.push(t);

        // Occasionally flip the coupling regime so the correlation scan has something
        // interesting to detect (positive/negative coupling windows).
        if i % 700 == 0 {
            if quad_rand::gen_range(0.0, 1.0) > 0.55 {
                regime *= -1.0;
            }
        }

        // Burst envelopes (rare spikes) to break up the smoothness.
        if quad_rand::gen_range(0.0, 1.0) > 0.993 {
            burst_energy += quad_rand::gen_range(0.4, 1.2);
        }
        burst_energy *= 0.965;

        // Bloch-ish coords: precession + wobble + occasional burst-driven radial kick.
        let wobble = 0.35 * (t * 0.17).sin() + 0.22 * (t * 0.041).cos();
        let theta = t * (0.28 + 0.08 * wobble) + quad_rand::gen_range(-0.06, 0.06);
        let phi = t * (0.19 + 0.05 * (t * 0.071).sin()) + quad_rand::gen_range(-0.06, 0.06);
        let r = 1.0 + quad_rand::gen_range(-0.03, 0.03) + 0.06 * burst_energy;
        let x = r * phi.cos() * theta.sin();
        let y = r * phi.sin() * theta.sin();
        let z = r * theta.cos();
        q_coords.push([x, y, z]);

        // CPU utility: multi-frequency load + slow random-walk drift + burst coupling.
        drift = (drift + quad_rand::gen_range(-0.015, 0.015)).clamp(-0.25, 0.25);
        let base = 0.52
            + 0.18 * (t * 0.65).sin()
            + 0.11 * (t * 1.37).sin()
            + 0.08 * (t * 0.13).cos()
            + 0.06 * (t * 3.9 + 0.3 * drift).sin();

        let util = clamp01(base + drift + 0.12 * burst_energy + quad_rand::gen_range(-0.06, 0.06));
        cpu_utility.push(util);

        // Entropy flux: correlated/anti-correlated windows + its own drift + noise.
        flux_drift = (flux_drift + quad_rand::gen_range(-0.02, 0.02)).clamp(-0.35, 0.35);
        let flux = (util - 0.5) * 0.9 * regime
            + 0.25 * (t * 0.9).cos()
            + 0.18 * (t * 2.7).sin()
            + 0.35 * quad_rand::gen_range(-1.0, 1.0)
            + flux_drift
            + 0.25 * burst_energy;
        entropy_flux.push(flux);

        // Entanglement map: richer mixture (nonlinear-ish) with rare excursions.
        let exc = if quad_rand::gen_range(0.0, 1.0) > 0.997 {
            quad_rand::gen_range(-1.4, 1.4)
        } else {
            0.0
        };
        let ent = 0.35 * (t * 1.1).cos()
            + 0.22 * (t * 0.33).sin()
            + 0.55 * (util - 0.5)
            + 0.28 * (flux * 0.6).tanh()
            + 0.15 * quad_rand::gen_range(-1.0, 1.0)
            + 0.35 * burst_energy
            + exc;
        entanglement_map.push(ent);
    }

    let telemetry = TelemetryColumnar {
        timestamp,
        q_coords,
        cpu_utility,
        entropy_flux,
        entanglement_map,
    };

    // Minified JSON output.
    let file = File::create("telemetry.json")?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &telemetry)?;
    writer.write_all(b"\n")?;
    writer.flush()?;

    Ok(())
}
