use egui_macroquad::egui::{Color32, RichText};
use egui_plot::{Line, Plot, PlotPoints};
use macroquad::miniquad::CullFace;
use macroquad::prelude::*;

use quantum_dashboard::{pearson_r_simd4, TelemetryColumnar};

const TELEMETRY_PATH_CANDIDATES: [&str; 2] = ["telemetry.bin", "../telemetry.bin"];

struct AppState {
    telemetry: TelemetryColumnar<'static>,
    points_meshes: Vec<Mesh>,
    points_material_solid: Material,
    ribbon_material: Material,
    points_max: usize,
    log: Vec<String>,
    phase_lock: bool,
    time_idx: usize,
    last_scan_t: f32,
    frame_idx: u32,
    playing: bool,
    playback_hz: f32,
    last_time_idx: usize,
    breadcrumbs: Vec<[f32; 3]>,
    breadcrumb_frames: Vec<u32>,
    show_sphere: bool,
    last_r: f32,
    last_phase_lock: bool,
    rotate_scene: bool,
    show_plot: bool,
    show_breadcrumbs: bool,
    show_trail: bool,
    trail_ribbon: bool,
    trail_width: f32,
    show_grid: bool,
    show_axes: bool,
    show_log: bool,
    max_breadcrumbs: usize,
    viz_mode: String,
    macro_scale: bool,
    gpu_memory_pressure: f32,
}

fn theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.visuals = egui::Visuals::dark();
    style.visuals.window_fill = Color32::from_rgb(5, 5, 5);
    style.visuals.panel_fill = Color32::from_rgb(5, 5, 5);
    style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(10, 10, 10);
    style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(14, 14, 14);
    style.visuals.widgets.active.bg_fill = Color32::from_rgb(18, 18, 18);
    style.spacing.item_spacing = egui::vec2(6.0, 4.0);
    style.spacing.window_margin = egui::Margin::same(8);
    ctx.set_style(style);
}

fn make_points_meshes(telemetry: &TelemetryColumnar, max_points: usize) -> Vec<Mesh> {
    // Macroquad batches geometry with a per-drawcall limit. Rendering 10k points as quads
    // can exceed that limit, so we chunk into multiple meshes.
    //
    // Each point = 4 vertices + 6 indices.
    // Keep this conservative to stay under Macroquad/miniquad internal geometry limits.
    const POINTS_PER_MESH: usize = 200;

    let n = telemetry.len();
    if n == 0 {
        return Vec::new();
    }

    let max_points = max_points.max(1);
    let stride = ((n + max_points - 1) / max_points).max(1);
    let sampled: Vec<usize> = (0..n).step_by(stride).collect();

    let mut meshes = Vec::new();

    let mut start = 0usize;
    while start < sampled.len() {
        let end = (start + POINTS_PER_MESH).min(sampled.len());
        let count = end - start;

        let mut vertices = Vec::with_capacity(count * 4);
        let mut indices = Vec::with_capacity(count * 6);

        for (local_i, global_i) in sampled[start..end].iter().copied().enumerate() {
            let p = vec3(telemetry.q_x[global_i], telemetry.q_y[global_i], telemetry.q_z[global_i]);
            let base = (local_i * 4) as u16;
            let c = p;

            // Color by cpu_utility: cyan -> magenta.
            let u = telemetry.cpu_utility[global_i].clamp(0.0, 1.0);
            let col = Color::new(1.0 * u, 0.95 * (1.0 - u), 1.0, 1.0);

            let corners = [vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)];

            for uv in corners {
                vertices.push(Vertex {
                    position: c,
                    uv,
                    color: col.into(),
                    normal: vec4(uv.x, uv.y, 1.0, 0.0),
                });
            }

            indices.extend_from_slice(&[
                base,
                base + 1,
                base + 2,
                base,
                base + 2,
                base + 3,
            ]);
        }

        meshes.push(Mesh {
            vertices,
            indices,
            texture: None,
        });

        start = end;
    }

    meshes
}

fn make_points_material_solid() -> Material {
    let vertex = include_str!("../../assets/vertex_solid.glsl");
    let fragment = include_str!("../../assets/fragment_solid.glsl");

    let pipeline_params = PipelineParams {
        // Solid/opaque point cloud: no blending (faster), proper depth.
        depth_write: false,
        depth_test: Comparison::LessOrEqual,
        cull_face: CullFace::Nothing,
        color_blend: None,
        alpha_blend: None,
        ..Default::default()
    };

    load_material(
        ShaderSource::Glsl { vertex, fragment },
        MaterialParams {
            pipeline_params,
            uniforms: vec![UniformDesc::new("time", UniformType::Float1)],
            ..Default::default()
        },
    )
    .unwrap()
}

fn make_ribbon_material() -> Material {
    let vertex = r#"precision highp float;

attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color0;

varying lowp vec4 vColor;

uniform mat4 Model;
uniform mat4 View;
uniform mat4 Projection;

void main() {
    gl_Position = Projection * View * Model * vec4(position, 1.0);
    vColor = color0;
}
"#;

    let fragment = r#"precision mediump float;

varying lowp vec4 vColor;

void main() {
    gl_FragColor = vColor;
}
"#;

    let pipeline_params = PipelineParams {
        depth_write: false,
        depth_test: Comparison::Always,
        cull_face: CullFace::Nothing,
        color_blend: None,
        alpha_blend: None,
        ..Default::default()
    };

    load_material(
        ShaderSource::Glsl { vertex, fragment },
        MaterialParams {
            pipeline_params,
            uniforms: vec![],
            ..Default::default()
        },
    )
    .unwrap()
}

fn make_trail_ribbon_mesh(
    telemetry: &TelemetryColumnar,
    start: usize,
    end: usize,
    camera: &Camera3D,
    width: f32,
) -> Option<Mesh> {
    let n = telemetry.q_x.len();
    if n == 0 {
        return None;
    }
    if end <= start {
        return None;
    }

    let start = start.min(n - 1);
    let end = end.min(n);
    if end <= start + 1 {
        return None;
    }

    let forward = (camera.target - camera.position).normalize();
    let span = (end - start).max(2);

    let mut vertices = Vec::with_capacity((end - start) * 2);
    let mut indices = Vec::with_capacity((end - start - 1) * 6);

    for j in start..end {
        let idx = j;
        let q = vec3(telemetry.q_x[idx], telemetry.q_y[idx], telemetry.q_z[idx]);
        let p = q;

        let prev_idx = idx.saturating_sub(1).max(start);
        let next_idx = (idx + 1).min(end - 1);
        let q0 = vec3(telemetry.q_x[prev_idx], telemetry.q_y[prev_idx], telemetry.q_z[prev_idx]);
        let q1 = vec3(telemetry.q_x[next_idx], telemetry.q_y[next_idx], telemetry.q_z[next_idx]);
        let prev = q0;
        let next = q1;

        let delta = next - prev;
        let tangent = if delta.length() > 1e-6 {
            delta / delta.length()
        } else {
            // Degenerate segment (repeated points). Pick a stable direction orthogonal-ish
            // to the camera forward vector so we still generate a visible strip.
            let mut t = camera.up.cross(forward);
            if t.length() < 1e-6 {
                t = vec3(1.0, 0.0, 0.0);
            }
            t / t.length()
        };

        let mut side = forward.cross(tangent);
        if side.length() < 1e-6 {
            side = forward.cross(camera.up);
        }
        if side.length() < 1e-6 {
            side = vec3(1.0, 0.0, 0.0);
        }
        let side = side / side.length();

        let t01 = ((j - start) as f32) / (span as f32);
        let intensity = 0.12 + 0.88 * t01;
        // Orange-ish gradient: dim at tail -> bright at head.
        let col = Color::new(1.0 * intensity, 0.45 * intensity, 0.0, 1.0);
        let rgba: [u8; 4] = col.into();

        let left = p - side * (width * 0.5);
        let right = p + side * (width * 0.5);

        vertices.push(Vertex {
            position: left,
            uv: vec2(0.0, t01),
            color: rgba,
            normal: vec4(0.0, 0.0, 1.0, 0.0),
        });
        vertices.push(Vertex {
            position: right,
            uv: vec2(1.0, t01),
            color: rgba,
            normal: vec4(0.0, 0.0, 1.0, 0.0),
        });
    }

    let mut i = 0u16;
    while (i as usize) + 2 < vertices.len() {
        indices.extend_from_slice(&[i, i + 1, i + 2, i + 1, i + 3, i + 2]);
        i += 2;
    }

    Some(Mesh {
        vertices,
        indices,
        texture: None,
    })
}

fn draw_grid_and_axes(show_grid: bool, show_axes: bool) {
    if show_grid {
        let desired_extent: f32 = 1.15;
        let step: f32 = 0.25;
        let n = (desired_extent / step).ceil() as i32;
        let extent = (n as f32) * step;

        for i in -n..=n {
            let x = (i as f32) * step;
            let is_major = i % 4 == 0;
            let a = if is_major { 60 } else { 28 };
            let col = Color::from_rgba(255, 255, 255, a);
            draw_line_3d(vec3(x, 0.0, -extent), vec3(x, 0.0, extent), col);
            draw_line_3d(vec3(-extent, 0.0, x), vec3(extent, 0.0, x), col);
        }
    }

    if show_axes {
        let l = 1.35;
        draw_line_3d(vec3(0.0, 0.0, 0.0), vec3(l, 0.0, 0.0), Color::from_rgba(255, 60, 60, 200));
        draw_line_3d(vec3(0.0, 0.0, 0.0), vec3(0.0, l, 0.0), Color::from_rgba(60, 255, 60, 200));
        draw_line_3d(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, l), Color::from_rgba(60, 120, 255, 200));
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn load_telemetry_native() -> anyhow::Result<TelemetryColumnar<'static>> {
    for path in TELEMETRY_PATH_CANDIDATES {
        if let Ok(bytes) = std::fs::read(path) {
            let bytes_slice = Box::leak(bytes.into_boxed_slice());
            let telemetry = parse_telemetry_binary(bytes_slice);
            return Ok(telemetry);
        }
    }

    Err(anyhow::anyhow!(
        "telemetry.bin not found (tried: {:?})",
        TELEMETRY_PATH_CANDIDATES
    ))
}

#[cfg(not(target_arch = "wasm32"))]
#[macroquad::main("Quantum Dashboard")]
async fn main() {
    let telemetry = load_telemetry_native().expect("failed to load telemetry.bin");
    let telemetry_len = telemetry.len();

    let points_max = 100_000usize;
    let points_meshes = make_points_meshes(&telemetry, points_max);
    let points_material_solid = make_points_material_solid();
    let ribbon_material = make_ribbon_material();

    let mut state = AppState {
        telemetry,
        points_meshes,
        points_material_solid,
        ribbon_material,
        points_max,
        log: Vec::new(),
        phase_lock: false,
        time_idx: 0,
        last_scan_t: 0.0,
        frame_idx: 0,
        playing: true,
        playback_hz: 960.0,
        last_time_idx: 0,
        breadcrumbs: Vec::with_capacity(telemetry_len),
        breadcrumb_frames: Vec::with_capacity(telemetry_len),
        show_sphere: false,
        last_r: 0.0,
        last_phase_lock: false,
        rotate_scene: true,
        show_plot: true,
        show_breadcrumbs: true,
        show_trail: true,
        trail_ribbon: true,
        trail_width: 0.06,
        show_grid: true,
        show_axes: true,
        show_log: false,
        max_breadcrumbs: 10_000,
        viz_mode: "Detailed".to_string(),
        macro_scale: false,
        gpu_memory_pressure: 0.0,
    };

    // Seed the breadcrumb history with the initial position so it's visible immediately.
    if !state.telemetry.q_x.is_empty() {
        state.breadcrumbs.push([state.telemetry.q_x[0], state.telemetry.q_y[0], state.telemetry.q_z[0]]);
        state.breadcrumb_frames.push(state.frame_idx);
    }

    loop {
        state.frame_idx = state.frame_idx.wrapping_add(1);
        let t = get_time() as f32;
        let dt = get_frame_time();
        clear_background(Color::from_rgba(5, 5, 5, 255));

        if state.playing {
            let n = state.telemetry.len();
            if n > 0 {
                let max_dt = 1.0 / 30.0; // Assume min 30 FPS to prevent large jumps
                let clamped_dt = dt.min(max_dt);
                state.time_idx = ((state.time_idx as f32 + state.playback_hz * clamped_dt).round() as usize) % n;
            }
        }

        // Breadcrumb history: keep past head positions visible.
        if !state.telemetry.q_x.is_empty() && state.time_idx != state.last_time_idx {
            let idx = state.time_idx.min(state.telemetry.q_x.len() - 1);

            if state.breadcrumbs.len() >= state.max_breadcrumbs {
                let excess = state.breadcrumbs.len() - state.max_breadcrumbs + 1;
                state.breadcrumbs.drain(0..excess);
                state.breadcrumb_frames.drain(0..excess);
            }

            state.breadcrumbs.push([state.telemetry.q_x[idx], state.telemetry.q_y[idx], state.telemetry.q_z[idx]]);
            state.breadcrumb_frames.push(state.frame_idx);
            state.last_time_idx = state.time_idx;
        }

        // Orbit camera so the 3D structure is obvious (toggleable).
        let orbit_t = if state.rotate_scene { t } else { 0.0 };
        let orbit = orbit_t * 0.35;
        let camera = Camera3D {
            position: vec3(2.6 * orbit.cos(), 1.5 + 0.25 * (t * 0.6).sin(), 2.6 * orbit.sin()),
            up: vec3(0.0, 1.0, 0.0),
            target: vec3(0.0, 0.0, 0.0),
            ..Default::default()
        };

        set_camera(&camera);

        draw_grid_and_axes(state.show_grid, state.show_axes);

        if state.show_sphere {
            draw_sphere_wires(
                vec3(0.0, 0.0, 0.0),
                1.0,
                None,
                Color::from_rgba(0, 242, 255, 80),
            );
        }

        // Solid base pass so points are always visible.
        state.points_material_solid.set_uniform("time", t);
        gl_use_material(&state.points_material_solid);
        for m in &state.points_meshes {
            draw_mesh(m);
        }

        // Reset material so immediate-mode primitives (spheres/lines) use the default pipeline.
        gl_use_default_material();

        // Selected sample marker + short active trail (draw after point cloud so it stays visible).
        if state.show_trail && !state.telemetry.q_x.is_empty() {
            let idx = state.time_idx.min(state.telemetry.q_x.len() - 1);
            let p = vec3(state.telemetry.q_x[idx], state.telemetry.q_y[idx], state.telemetry.q_z[idx]);

            let trail_len = 256usize;
            let start = idx.saturating_sub(trail_len);
            let end = (idx + 1).min(state.telemetry.q_x.len());

            let r = 0.03;
            if state.trail_ribbon {
                if let Some(mesh) = make_trail_ribbon_mesh(&state.telemetry, start, end, &camera, state.trail_width) {
                    gl_use_default_material();
                    draw_mesh(&mesh);
                }
            } else if end > start {
                let span = (end - start).max(1);
                for j in start..end {
                    let q_vec = vec3(state.telemetry.q_x[j], state.telemetry.q_y[j], state.telemetry.q_z[j]);
                    let t01 = ((j - start) as f32) / (span as f32);
                    let intensity = 0.15 + 0.85 * t01;
                    draw_sphere(
                        vec3(q_vec.x, q_vec.y, q_vec.z),
                        r * 0.9,
                        None,
                        Color::new(1.0 * intensity, 0.45 * intensity, 0.0, 0.75),
                    );
                }
            }

            draw_sphere(
                vec3(p.x, p.y, p.z),
                r * 1.6,
                None,
                Color::from_rgba(255, 140, 20, 255),
            );
        }

        // Breadcrumbs: persistent tiny spheres.
        if state.show_breadcrumbs {
            let bn = state.breadcrumbs.len();
            if bn >= 2 {
                // Fade out over a fixed frame window, but never to black.
                const FADE_FRAMES: u32 = 300;
                let current = state.frame_idx;
                let mut prev = state.breadcrumbs[0];
                let mut prev_frame = state.breadcrumb_frames.get(0).copied().unwrap_or(current);

                for b in 1..bn {
                    let p0 = prev;
                    let p1 = state.breadcrumbs[b];
                    let f0 = prev_frame;
                    let f1 = state.breadcrumb_frames.get(b).copied().unwrap_or(current);
                    prev = p1;
                    prev_frame = f1;

                    // Age is based on the newer endpoint so segments stay visible right after being added.
                    let seg_frame = f0.max(f1);
                    let age = current.saturating_sub(seg_frame);
                    let k = (1.0 - (age as f32 / FADE_FRAMES as f32)).clamp(0.0, 1.0);

                    // Dark grey at tail -> near-white at head.
                    let dark = 25.0;
                    let bright = 230.0;
                    let g = (dark + (bright - dark) * k).clamp(0.0, 255.0) as u8;
                    let col = Color::from_rgba(g, g, g, 255);

                    draw_line_3d(vec3(p0[0], p0[1], p0[2]), vec3(p1[0], p1[1], p1[2]), col);
                }
            }
        }

        gl_use_default_material();

        set_default_camera();

        // Minimal sanity check overlay.
        draw_text(
            &format!(
                "records: {}  points_drawn: {}  point_meshes: {}  bc: {}  bc_show: {}  t: {}  play: {}  fps: {:.0}",
                state.telemetry.len(),
                state.points_max.min(state.telemetry.len()),
                state.points_meshes.len(),
                state.breadcrumbs.len(),
                if state.show_breadcrumbs { 1 } else { 0 },
                state.time_idx,
                if state.playing { 1 } else { 0 },
                get_fps()
            ),
            12.0,
            22.0,
            18.0,
            Color::from_rgba(255, 255, 255, 220),
        );

        if t - state.last_scan_t > 0.25 {
            state.last_scan_t = t;
            let r = pearson_r_simd4(state.telemetry.cpu_utility, state.telemetry.entropy_flux);
            state.phase_lock = r.abs() > 0.8;

            // Log only meaningful events.
            let crossed = state.phase_lock != state.last_phase_lock;
            let changed = (r - state.last_r).abs() > 0.05;
            if crossed || changed {
                let line = if crossed {
                    format!("Phase Lock {} | r={:.3}", if state.phase_lock { "ON" } else { "OFF" }, r)
                } else {
                    format!("r(cpu,entropy_flux)={:.3}", r)
                };
                state.log.push(line);
                if state.log.len() > 5 {
                    let drain = state.log.len() - 5;
                    state.log.drain(0..drain);
                }
            }

            state.last_r = r;
            state.last_phase_lock = state.phase_lock;
        }

        egui_macroquad::ui(|ctx| {
            theme(ctx);

            if state.show_log {
                egui::SidePanel::left("log_panel")
                    .resizable(true)
                    .default_width(360.0)
                    .show(ctx, |ui| {
                        ui.heading(RichText::new("DISCOVERY LOG").color(Color32::from_rgb(0, 242, 255)));
                        ui.separator();
                        egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                            for line in &state.log {
                                ui.label(RichText::new(line).monospace());
                            }
                        });
                    });
            }

            egui::SidePanel::right("charts_panel")
                .resizable(true)
                .default_width(420.0)
                .show(ctx, |ui| {
                    ui.heading(RichText::new("CPU WAVEFORMS").color(Color32::from_rgb(255, 0, 85)));
                    ui.label(
                        RichText::new(format!("FPS: {}", get_fps()))
                            .color(Color32::from_rgb(160, 160, 160))
                            .monospace(),
                    );
                    ui.separator();

                    ui.horizontal(|ui| {
                        let label = if state.playing { "Pause" } else { "Play" };
                        if ui.button(label).clicked() {
                            state.playing = !state.playing;
                        }

                        if ui.button("Step").clicked() {
                            state.playing = false;
                            if !state.telemetry.timestamp.is_empty() {
                                state.time_idx = (state.time_idx + 1).min(state.telemetry.len() - 1);
                            }
                        }

                        ui.add(
                            egui::Slider::new(&mut state.time_idx, 0..=(state.telemetry.len() - 1))
                                .text("Time Scrub")
                                .clamping(egui::SliderClamping::Always),
                        );
                    });

                    ui.separator();

                    if state.show_plot {
                        let window = 256usize;
                        let start = state.time_idx.saturating_sub(window / 2);
                        let end = (start + window).min(state.telemetry.len());
                        let xs = &state.telemetry.timestamp[start..end];
                        let y_cpu = &state.telemetry.cpu_utility[start..end];
                        let mut pts = Vec::with_capacity(xs.len() / 8 + 1);
                        for i in (0..xs.len()).step_by(8) {
                            pts.push([xs[i] as f64, y_cpu[i] as f64]);
                        }
                        let pts_cpu: PlotPoints = pts.into();
                        Plot::new("cpu_plot").height(240.0).show(ui, |plot_ui| {
                            plot_ui.line(
                                Line::new(pts_cpu)
                                    .name("cpu_utility")
                                    .color(Color32::from_rgb(0, 242, 255)),
                            );
                        });
                    }
                });
            });
            egui_macroquad::draw();

        // Breadcrumbs: persistent tiny spheres.
        if state.show_breadcrumbs {
            let bn = state.breadcrumbs.len();
            if bn >= 2 {
                // Fade out over a fixed frame window, but never to black.
                const FADE_FRAMES: u32 = 300;
                let current = state.frame_idx;
                let mut prev = state.breadcrumbs[0];
                let mut prev_frame = state.breadcrumb_frames.get(0).copied().unwrap_or(current);

                for b in 1..bn {
                    let p0 = prev;
                    let p1 = state.breadcrumbs[b];
                    let f0 = prev_frame;
                    let f1 = state.breadcrumb_frames.get(b).copied().unwrap_or(current);
                    prev = p1;
                    prev_frame = f1;

                    // Age is based on the newer endpoint so segments stay visible right after being added.
                    let seg_frame = f0.max(f1);
                    let age = current.saturating_sub(seg_frame);
                    let k = (1.0 - (age as f32 / FADE_FRAMES as f32)).clamp(0.0, 1.0);

                    // Dark grey at tail -> near-white at head.
                    let dark = 25.0;
                    let bright = 230.0;
                    let g = (dark + (bright - dark) * k).clamp(0.0, 255.0) as u8;
                    let col = Color::from_rgba(g, g, g, 255);

                    draw_line_3d(vec3(p0[0], p0[1], p0[2]), vec3(p1[0], p1[1], p1[2]), col);
                }
            }
        }

        gl_use_default_material();

        set_default_camera();

        // Minimal sanity check overlay.
        draw_text(
            &format!(
                "records: {}  points_drawn: {}  point_meshes: {}  bc: {}  bc_show: {}  t: {}  play: {}  fps: {:.0}",
                state.telemetry.len(),
                state.points_max.min(state.telemetry.len()),
                state.points_meshes.len(),
                state.breadcrumbs.len(),
                if state.show_breadcrumbs { 1 } else { 0 },
                state.time_idx,
                if state.playing { 1 } else { 0 },
                get_fps()
            ),
            12.0,
            22.0,
            18.0,
            Color::from_rgba(255, 255, 255, 220),
        );

        if t - state.last_scan_t > 0.25 {
            state.last_scan_t = t;
            let r = pearson_r_simd4(state.telemetry.cpu_utility, state.telemetry.entropy_flux);
            state.phase_lock = r.abs() > 0.8;

            // Log only meaningful events.
            let crossed = state.phase_lock != state.last_phase_lock;
            let changed = (r - state.last_r).abs() > 0.05;
            if crossed || changed {
                let line = if crossed {
                    format!("Phase Lock {} | r={:.3}", if state.phase_lock { "ON" } else { "OFF" }, r)
                } else {
                    format!("r(cpu,entropy_flux)={:.3}", r)
                };
                state.log.push(line);
                if state.log.len() > 5 {
                    let drain = state.log.len() - 5;
                    state.log.drain(0..drain);
                }
            }

            state.last_r = r;
            state.last_phase_lock = state.phase_lock;
        }

        egui_macroquad::ui(|ctx| {
            theme(ctx);

            if state.show_log {
                egui::SidePanel::left("log_panel")
                    .resizable(true)
                    .default_width(360.0)
                    .show(ctx, |ui| {
                        ui.heading(RichText::new("DISCOVERY LOG").color(Color32::from_rgb(0, 242, 255)));
                        ui.separator();
                        egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                            for line in &state.log {
                                ui.label(RichText::new(line).monospace());
                            }
                        });
                    });
            }

            egui::SidePanel::right("charts_panel")
                .resizable(true)
                .default_width(420.0)
                .show(ctx, |ui| {
                    ui.heading(RichText::new("CPU WAVEFORMS").color(Color32::from_rgb(255, 0, 85)));
                    ui.label(
                        RichText::new(format!("FPS: {}", get_fps()))
                            .color(Color32::from_rgb(160, 160, 160))
                            .monospace(),
                    );
                    ui.separator();

                    ui.horizontal(|ui| {
                        let label = if state.playing { "Pause" } else { "Play" };
                        if ui.button(label).clicked() {
                            state.playing = !state.playing;
                        }

                        if ui.button("Step").clicked() {
                            state.playing = false;
                            if !state.telemetry.timestamp.is_empty() {
                                state.time_idx = (state.time_idx + 1).min(state.telemetry.len() - 1);
                            }
                        }

                        ui.add(
                            egui::Slider::new(&mut state.playback_hz, 0.1..=10.0)
                                .text("Playback Hz")
                                .clamping(egui::SliderClamping::Always),
                        );
                    });

                    ui.separator();
                });
        });
// Reset material so immediate-mode primitives (spheres/lines) use the default pipeline.
gl_use_default_material();

// Selected sample marker + short active trail (draw after point cloud so it stays visible).
if state.show_trail && !state.telemetry.q_x.is_empty() {
let idx = state.time_idx.min(state.telemetry.q_x.len() - 1);
let p = vec3(state.telemetry.q_x[idx], state.telemetry.q_y[idx], state.telemetry.q_z[idx]);

let trail_len = 256usize;
let start = idx.saturating_sub(trail_len);
let end = (idx + 1).min(state.telemetry.q_x.len());

let r = 0.03;
if state.trail_ribbon {
if let Some(mesh) = make_trail_ribbon_mesh(&state.telemetry, start, end, &camera, state.trail_width) {
gl_use_default_material();
draw_mesh(&mesh);
}
} else if end > start {
let span = (end - start).max(1);
for j in start..end {
let q_vec = vec3(state.telemetry.q_x[j], state.telemetry.q_y[j], state.telemetry.q_z[j]);
let t01 = ((j - start) as f32) / (span as f32);
let intensity = 0.15 + 0.85 * t01;
draw_sphere(
vec3(q_vec.x, q_vec.y, q_vec.z),
r * 0.9,
None,
Color::new(1.0 * intensity, 0.45 * intensity, 0.0, 0.75),
);
}
}

draw_sphere(
vec3(p.x, p.y, p.z),
r * 1.6,
None,
Color::from_rgba(255, 140, 20, 255),
);
}

// Breadcrumbs: persistent tiny spheres.
if state.show_breadcrumbs {
let bn = state.breadcrumbs.len();
if bn >= 2 {
// Fade out over a fixed frame window, but never to black.
const FADE_FRAMES: u32 = 300;
let current = state.frame_idx;
let mut prev = state.breadcrumbs[0];
let mut prev_frame = state.breadcrumb_frames.get(0).copied().unwrap_or(current);

for b in 1..bn {
let p0 = prev;
let p1 = state.breadcrumbs[b];
let f0 = prev_frame;
let f1 = state.breadcrumb_frames.get(b).copied().unwrap_or(current);
prev = p1;
prev_frame = f1;

// Age is based on the newer endpoint so segments stay visible right after being added.
let seg_frame = f0.max(f1);
let age = current.saturating_sub(seg_frame);
let k = (1.0 - (age as f32 / FADE_FRAMES as f32)).clamp(0.0, 1.0);

// Dark grey at tail -> near-white at head.
let dark = 25.0;
let bright = 230.0;
let g = (dark + (bright - dark) * k).clamp(0.0, 255.0) as u8;
let col = Color::from_rgba(g, g, g, 255);

draw_line_3d(vec3(p0[0], p0[1], p0[2]), vec3(p1[0], p1[1], p1[2]), col);
}
}
}

gl_use_default_material();

set_default_camera();

// Minimal sanity check overlay.
draw_text(
&format!(
"records: {}  points_drawn: {}  point_meshes: {}  bc: {}  bc_show: {}  t: {}  play: {}  fps: {:.0}",
state.telemetry.len(),
state.points_max.min(state.telemetry.len()),
state.points_meshes.len(),
state.breadcrumbs.len(),
if state.show_breadcrumbs { 1 } else { 0 },
state.time_idx,
if state.playing { 1 } else { 0 },
get_fps()
),
12.0,
22.0,
18.0,
Color::from_rgba(255, 255, 255, 220),
);

if t - state.last_scan_t > 0.25 {
state.last_scan_t = t;
let r = pearson_r_simd4(state.telemetry.cpu_utility, state.telemetry.entropy_flux);
state.phase_lock = r.abs() > 0.8;

// Log only meaningful events.
let crossed = state.phase_lock != state.last_phase_lock;
let changed = (r - state.last_r).abs() > 0.05;
if crossed || changed {
let line = if crossed {
format!("Phase Lock {} | r={:.3}", if state.phase_lock { "ON" } else { "OFF" }, r)
} else {
format!("r(cpu,entropy_flux)={:.3}", r)
};
state.log.push(line);
if state.log.len() > 5 {
let drain = state.log.len() - 5;
state.log.drain(0..drain);
}
}

state.last_r = r;
state.last_phase_lock = state.phase_lock;
}

#[cfg(target_arch = "wasm32")]
#[macroquad::main("Quantum Dashboard")]
async fn main() {
    let telemetry = load_telemetry_wasm().await.expect("failed to load telemetry.bin");
    let telemetry_len = telemetry.len();

    let points_max = 100_000usize;
    let points_meshes = make_points_meshes(&telemetry, points_max);
    let points_material_solid = make_points_material_solid();
    let ribbon_material = make_ribbon_material();

    let mut state = AppState {
        telemetry,
        points_meshes,
        points_material_solid,
        ribbon_material,
        points_max,
        log: Vec::new(),
        phase_lock: false,
        time_idx: 0,
        last_scan_t: 0.0,
        frame_idx: 0,
        playing: true,
        playback_hz: 960.0,
        last_time_idx: 0,
        breadcrumbs: Vec::with_capacity(telemetry_len),
        breadcrumb_frames: Vec::with_capacity(telemetry_len),
        show_sphere: false,
        last_r: 0.0,
        last_phase_lock: false,
        rotate_scene: true,
        show_plot: true,
        show_breadcrumbs: true,
        show_trail: true,
        trail_ribbon: true,
        trail_width: 0.06,
        show_grid: true,
        show_axes: true,
        show_log: false,
        max_breadcrumbs: 10_000,
        viz_mode: "Detailed".to_string(),
        macro_scale: false,
        gpu_memory_pressure: 0.0,
    };

    if !state.telemetry.q_x.is_empty() {
        state.breadcrumbs.push([state.telemetry.q_x[0], state.telemetry.q_y[0], state.telemetry.q_z[0]]);
        state.breadcrumb_frames.push(state.frame_idx);
    }

    loop {
        state.frame_idx = state.frame_idx.wrapping_add(1);
        let t = get_time() as f32;
        let dt = get_frame_time();
        clear_background(Color::from_rgba(5, 5, 5, 255));

        if state.playing {
            let n = state.telemetry.len();
            if n > 0 {
                let max_dt = 1.0 / 30.0;
                let clamped_dt = dt.min(max_dt);
                state.time_idx = ((state.time_idx as f32 + state.playback_hz * clamped_dt).round() as usize) % n;
            }
        }

        if !state.telemetry.q_x.is_empty() && state.time_idx != state.last_time_idx {
            let idx = state.time_idx.min(state.telemetry.q_x.len() - 1);
            if state.breadcrumbs.len() >= state.max_breadcrumbs {
                let excess = state.breadcrumbs.len() - state.max_breadcrumbs + 1;
                state.breadcrumbs.drain(0..excess);
                state.breadcrumb_frames.drain(0..excess);
            }
            state.breadcrumbs.push([state.telemetry.q_x[idx], state.telemetry.q_y[idx], state.telemetry.q_z[idx]]);
            state.breadcrumb_frames.push(state.frame_idx);
            state.last_time_idx = state.time_idx;
        }

        let orbit_t = if state.rotate_scene { t } else { 0.0 };
        let orbit = orbit_t * 0.35;
        let camera = Camera3D {
            position: vec3(2.6 * orbit.cos(), 1.5 + 0.25 * (t * 0.6).sin(), 2.6 * orbit.sin()),
            up: vec3(0.0, 1.0, 0.0),
            target: vec3(0.0, 0.0, 0.0),
            ..Default::default()
        };
        set_camera(&camera);

        draw_grid_and_axes(state.show_grid, state.show_axes);

        if state.show_sphere {
            draw_sphere_wires(
                vec3(0.0, 0.0, 0.0),
                1.0,
                None,
                Color::from_rgba(0, 242, 255, 80),
            );
        }

        state.points_material_solid.set_uniform("time", t);
        gl_use_material(&state.points_material_solid);
        for m in &state.points_meshes {
            draw_mesh(m);
        }
        gl_use_default_material();

        if state.viz_mode == "Filament Web" {
            for i in 0..50 {
                let idx1 = i * 20000;
                let idx2 = (i + 1) * 20000;
                if idx2 < state.telemetry.len() {
                    let p1 = vec3(state.telemetry.q_x[idx1], state.telemetry.q_y[idx1], state.telemetry.q_z[idx1]);
                    let p2 = vec3(state.telemetry.q_x[idx2], state.telemetry.q_y[idx2], state.telemetry.q_z[idx2]);
                    draw_line_3d(p1, p2, Color::from_rgba(0, 255, 255, 150));
                }
            }
        } else if state.viz_mode == "Probability Manifold" {
            for x in -10..10 {
                for z in -10..10 {
                    let idx = (((x + 10) as usize + (z + 10) as usize * 20) % state.telemetry.len());
                    let y = state.telemetry.cpu_utility[idx] * 0.5;
                    let p1 = vec3(x as f32 * 0.1, y, z as f32 * 0.1);
                    let p2 = vec3((x + 1) as f32 * 0.1, y, z as f32 * 0.1);
                    draw_line_3d(p1, p2, Color::from_rgba(255, 0, 255, 100));
                    let p3 = vec3(x as f32 * 0.1, y, (z + 1) as f32 * 0.1);
                    draw_line_3d(p1, p3, Color::from_rgba(255, 0, 255, 100));
                }
            }
        } else if state.viz_mode == "Interference Voxel" {
            for i in 0..1000 {
                let p = vec3(state.telemetry.q_x[i], state.telemetry.q_y[i], state.telemetry.q_z[i]);
                let col = Color::new(state.telemetry.cpu_utility[i], state.telemetry.entropy_flux[i].abs(), state.telemetry.entanglement_map[i].abs(), 0.5);
                draw_cube(p, vec3(0.005, 0.005, 0.005), None, col);
            }
        }

        if state.show_trail && !state.telemetry.q_x.is_empty() {
            let idx = state.time_idx.min(state.telemetry.q_x.len() - 1);
            let p = vec3(state.telemetry.q_x[idx], state.telemetry.q_y[idx], state.telemetry.q_z[idx]);
            let trail_len = 256usize;
            let start = idx.saturating_sub(trail_len);
            let end = (idx + 1).min(state.telemetry.q_x.len());
            let r = 0.03;
            if state.trail_ribbon {
                if let Some(mesh) = make_trail_ribbon_mesh(&state.telemetry, start, end, &camera, state.trail_width) {
                    gl_use_default_material();
                    draw_mesh(&mesh);
                }
            } else if end > start {
                let span = (end - start).max(1);
                for j in start..end {
                    let q_vec = vec3(state.telemetry.q_x[j], state.telemetry.q_y[j], state.telemetry.q_z[j]);
                    let t01 = ((j - start) as f32) / (span as f32);
                    let intensity = 0.15 + 0.85 * t01;
                    draw_sphere(
                        vec3(q_vec.x, q_vec.y, q_vec.z),
                        r * 0.9,
                        None,
                        Color::new(1.0 * intensity, 0.45 * intensity, 0.0, 0.75),
                    );
                }
            }
            draw_sphere(
                vec3(p.x, p.y, p.z),
                r * 1.6,
                None,
                Color::from_rgba(255, 140, 20, 255),
            );
        }

        if state.show_breadcrumbs {
            let bn = state.breadcrumbs.len();
            if bn >= 2 {
                const FADE_FRAMES: u32 = 300;
                let current = state.frame_idx;
                let mut prev = state.breadcrumbs[0];
                let mut prev_frame = state.breadcrumb_frames.get(0).copied().unwrap_or(current);
                for b in 1..bn {
                    let p0 = prev;
                    let p1 = state.breadcrumbs[b];
                    let f0 = prev_frame;
                    let f1 = state.breadcrumb_frames.get(b).copied().unwrap_or(current);
                    prev = p1;
                    prev_frame = f1;
                    let seg_frame = f0.max(f1);
                    let age = current.saturating_sub(seg_frame);
                    let k = (1.0 - (age as f32 / FADE_FRAMES as f32)).clamp(0.0, 1.0);
                    let dark = 25.0;
                    let bright = 230.0;
                    let g = (dark + (bright - dark) * k).clamp(0.0, 255.0) as u8;
                    let col = Color::from_rgba(g, g, g, 255);
                    draw_line_3d(vec3(p0[0], p0[1], p0[2]), vec3(p1[0], p1[1], p1[2]), col);
                }
            }
        }

        gl_use_default_material();
        set_default_camera();

        draw_text(
            &format!(
                "records: {}  points_drawn: {}  point_meshes: {}  bc: {}  bc_show: {}  t: {}  play: {}  fps: {:.0}",
                state.telemetry.len(),
                state.points_max.min(state.telemetry.len()),
                state.points_meshes.len(),
                state.breadcrumbs.len(),
                if state.show_breadcrumbs { 1 } else { 0 },
                state.time_idx,
                if state.playing { 1 } else { 0 },
                get_fps()
            ),
            12.0,
            22.0,
            18.0,
            Color::from_rgba(255, 255, 255, 220),
        );

        if t - state.last_scan_t > 0.25 {
            state.last_scan_t = t;
            let r = pearson_r_simd4(state.telemetry.cpu_utility, state.telemetry.entropy_flux);
            state.phase_lock = r.abs() > 0.8;
            let crossed = state.phase_lock != state.last_phase_lock;
            let changed = (r - state.last_r).abs() > 0.05;
            if crossed || changed {
                let line = if crossed {
                    format!("Phase Lock {} | r={:.3}", if state.phase_lock { "ON" } else { "OFF" }, r)
                } else {
                    format!("r(cpu,entropy_flux)={:.3}", r)
                };
                state.log.push(line);
                if state.log.len() > 5 {
                    let drain = state.log.len() - 5;
                    state.log.drain(0..drain);
                }
            }
            state.last_r = r;
            state.last_phase_lock = state.phase_lock;
        }

        egui_macroquad::ui(|ctx| {
            theme(ctx);

            if state.show_log {
                egui::SidePanel::left("log_panel")
                    .resizable(true)
                    .default_width(360.0)
                    .show(ctx, |ui| {
                        ui.heading(RichText::new("DISCOVERY LOG").color(Color32::from_rgb(0, 242, 255)));
                        ui.separator();
                        egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                            for line in &state.log {
                                ui.label(RichText::new(line).monospace());
                            }
                        });
                    });
            }

            egui::SidePanel::right("charts_panel")
                .resizable(true)
                .default_width(420.0)
                .show(ctx, |ui| {
                    ui.heading(RichText::new("CPU WAVEFORMS").color(Color32::from_rgb(255, 0, 85)));
                    ui.label(
                        RichText::new(format!("FPS: {}", get_fps()))
                            .color(Color32::from_rgb(160, 160, 160))
                            .monospace(),
                    );
                    ui.separator();

                    ui.horizontal(|ui| {
                        let label = if state.playing { "Pause" } else { "Play" };
                        if ui.button(label).clicked() {
                            state.playing = !state.playing;
                        }

                        if ui.button("Step").clicked() {
                            state.playing = false;
                            if !state.telemetry.timestamp.is_empty() {
                                state.time_idx = (state.time_idx + 1).min(state.telemetry.len() - 1);
                            }
                        }

                        ui.add(
                            egui::Slider::new(&mut state.playback_hz, 0.1..=10.0)
                                .text("Playback Hz")
                                .clamping(egui::SliderClamping::Always),
                        );
                    });

                    ui.separator();
                    ui.checkbox(&mut state.show_sphere, "Wireframe Sphere");
                    ui.checkbox(&mut state.rotate_scene, "Sphere Rotation");
                    ui.checkbox(&mut state.show_plot, "Waveform Plot");
                    ui.separator();
                    ui.checkbox(&mut state.show_log, "Show Log Panel");
                    ui.separator();
                    ui.checkbox(&mut state.macro_scale, "Macro-Scale (limit to 10k points)");
                    ui.label(format!("GPU Memory Pressure: {:.1}%", state.gpu_memory_pressure));

                    ui.add(
                        egui::Slider::new(&mut state.time_idx, 0..=(state.telemetry.len() - 1))
                            .text("Time Scrub")
                            .clamping(egui::SliderClamping::Always),
                    );

                    if state.phase_lock {
                        ui.separator();
                        ui.colored_label(Color32::from_rgb(255, 0, 85), RichText::new("PHASE LOCK ∣r∣>0.8").strong());
                    }

                    ui.separator();

                    if state.show_plot {
                        let window = 256usize;
                        let start = state.time_idx.saturating_sub(window / 2);
                        let end = (start + window).min(state.telemetry.len());
                        let xs = &state.telemetry.timestamp[start..end];
                        let y_cpu = &state.telemetry.cpu_utility[start..end];
                        let mut pts = Vec::with_capacity(xs.len() / 8 + 1);
                        for i in (0..xs.len()).step_by(8) {
                            pts.push([xs[i] as f64, y_cpu[i] as f64]);
                        }
                        let pts_cpu: PlotPoints = pts.into();
                        Plot::new("cpu_plot").height(240.0).show(ui, |plot_ui| {
                            plot_ui.line(
                                Line::new(pts_cpu)
                                    .name("cpu_utility")
                                    .color(Color32::from_rgb(0, 242, 255)),
                            );
                        });
                    }
                });
        });

        egui_macroquad::draw();

        next_frame().await;
    }
}
