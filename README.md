# Quantum Visualizer (Quantum-CPU Telemetry Dashboard)

A small Rust demo that visualizes a synthetic **quantum telemetry** dataset as a 3D point cloud on/around a Bloch-sphere-like space.
It also runs a background **SIMD Pearson correlation** scan (WASM `simd128` / `core::simd`) between telemetry channels.

This repo is meant to be:

- **Fast to run** (WASM build output is a static `dist/` folder)
- **Easy to tweak** (single Macroquad render loop + egui controls)
- **Good-looking by default** (trail, breadcrumbs-as-artifacts, grid/axes)

## Features

- 3D point cloud (Macroquad 3D + GLSL point quads)
- Playback + time scrub
- Active trail (sphere trail or ribbon trail)
- Breadcrumb history as connected line artifacts
- Grid + XYZ axes
- Background correlation scan: Pearson `r(cpu_utility, entropy_flux)`

## Project layout

- `data_gen/`
  - CLI generator that produces `telemetry.json` (columnar **Struct-of-Arrays**)
- `dashboard/`
  - Macroquad + egui-macroquad app
  - `src/main.rs` is the dashboard entrypoint
  - `assets/` contains GLSL shaders used by the point cloud
- `telemetry.json`
  - generated data file (repo root)
- `build.sh`
  - builds the dashboard to WASM (`dist/` output)
- `serve.sh`
  - serves `dist/` locally (`python3 -m http.server`)

## Prerequisites

- Rust **nightly** (this project uses `core::simd`)
- Python 3 (only for local serving)

If you plan to run the dashboard **natively** on Linux, see the ALSA note in Troubleshooting.

## Generate telemetry

This produces `telemetry.json` in the repo root:

```bash
cargo run -p data_gen --release
```

## Run (web / WASM)

Build the static web output:

```bash
./build.sh
```

Serve it:

```bash
./serve.sh 8000
```

Open:

- `http://127.0.0.1:8000/`

### What `build.sh` does

- Adds `wasm32-unknown-unknown` target if needed
- Builds `quantum_dashboard_app` for WASM with `simd128`
- Copies output to `dist/`:
  - `quantum_dashboard_app.wasm`
  - Macroquad loader `mq_js_bundle.js`
  - `index.html`

## Run (native / desktop)

```bash
cargo run -p quantum_dashboard --bin quantum_dashboard_app
```

## UI / controls

Right panel:

- **Play / Pause**: start/stop playback
- **Step**: advance one sample (pauses playback)
- **Hz**: playback speed
- **Time Scrub**: jump in time
- **Wireframe Sphere**: show a reference sphere
- **Sphere Rotation**: toggles camera orbit
- **Waveform Plot**: CPU waveform plot
- **Active Trail**: show the active trail (sphere or ribbon)
- **Ribbon Trail**: use a strip/ribbon for the trail
- **Trail Width**: ribbon width (when ribbon enabled)
- **Breadcrumbs**: breadcrumb artifact line history
- **Grid / Axes**: scene reference guides

Left panel:

- Optional discovery log (phase-lock events)

Overlay:

- Quick sanity stats (record count, meshes, breadcrumb count, FPS)

## Troubleshooting

### Native build fails with `unable to find library -lasound`

Macroquad’s audio backend links against ALSA on Linux. Install the ALSA development package for your distro.

- Debian/Ubuntu: `libasound2-dev`
- Fedora: `alsa-lib-devel`
- Arch: `alsa-lib`

If you don’t need native, use the WASM path (`./build.sh` + `./serve.sh`).

### `can't find crate for core` when building WASM

You’re missing the target:

```bash
rustup target add wasm32-unknown-unknown
```

### `wasm-bindgen: command not found`

This repo uses Macroquad’s JS loader bundle (`mq_js_bundle.js`). You should *not* need `wasm-bindgen-cli` for the current pipeline.

## Notes

- The point cloud uses custom GLSL to expand points into camera-facing quads.
- Telemetry data is columnar (SoA) to keep the SIMD correlation scan cache-friendly.
