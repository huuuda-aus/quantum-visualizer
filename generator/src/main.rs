use std::fs::File;
use std::io::{BufWriter, Write};

use bytemuck;

const N: usize = 1_000_000;

fn clamp01(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

fn main() -> anyhow::Result<()> {
    // quad-rand is fast and works well cross-platform; seed is implicit.
    // We keep everything SoA for cache-friendly SIMD scanning.
    let mut timestamp = Vec::with_capacity(N);
    let mut q_x = Vec::with_capacity(N);
    let mut q_y = Vec::with_capacity(N);
    let mut q_z = Vec::with_capacity(N);
    let mut cpu_utility = Vec::with_capacity(N);
    let mut entropy_flux = Vec::with_capacity(N);
    let mut entanglement_map = Vec::with_capacity(N);

    // Generate pseudo-physics-like patterns (sin/cos + noise) with stable ranges.
    for i in 0..N {
        let t = i as f32 * 0.016; // ~60Hz steps
        timestamp.push(t);

        // Bloch sphere coords: mostly on/near unit sphere with slight radial noise.
        let theta = t * 0.35 + quad_rand::gen_range(-0.05, 0.05);
        let phi = t * 0.22 + quad_rand::gen_range(-0.05, 0.05);
        let r = 1.0 + quad_rand::gen_range(-0.02, 0.02);
        let x = r * phi.cos() * theta.sin();
        let y = r * phi.sin() * theta.sin();
        let z = r * theta.cos();
        q_x.push(x);
        q_y.push(y);
        q_z.push(z);

        // cpu utility: bounded 0..1 with periodic load + small noise
        let base = 0.55 + 0.35 * (t * 0.9).sin();
        let util = clamp01(base + quad_rand::gen_range(-0.05, 0.05));
        cpu_utility.push(util);

        // entropy flux: noise + weak coupling to util to create discoverable correlation
        let flux = quad_rand::gen_range(-1.0, 1.0) * 0.35 + (util - 0.5) * 0.6;
        entropy_flux.push(flux);

        // entanglement map: mixed signal with util + oscillation
        let ent = (t * 1.7).cos() * 0.4 + (util - 0.5) * 0.8 + quad_rand::gen_range(-0.15, 0.15);
        entanglement_map.push(ent);
    }

    // Write binary columnar data: [timestamps][q_x][q_y][q_z][cpu_load][entropy][entanglement]
    let file = File::create("telemetry.bin")?;
    let mut writer = BufWriter::new(file);
    writer.write_all(bytemuck::cast_slice(&timestamp))?;
    writer.write_all(bytemuck::cast_slice(&q_x))?;
    writer.write_all(bytemuck::cast_slice(&q_y))?;
    writer.write_all(bytemuck::cast_slice(&q_z))?;
    writer.write_all(bytemuck::cast_slice(&cpu_utility))?;
    writer.write_all(bytemuck::cast_slice(&entropy_flux))?;
    writer.write_all(bytemuck::cast_slice(&entanglement_map))?;
    writer.flush()?;

    Ok(())
}
