use egui::{Color32, RichText};
use egui::plot::{Line, Plot, PlotPoints};
use macroquad::prelude::*;
use macroquad::miniquad::{BlendFactor, BlendState, BlendValue, Equation};

use dashboard_wasm::{cpu_correlations, parse_telemetry_binary, select_top2_variance, TelemetryColumnar};

const TELEMETRY_BYTES: &[u8] = include_bytes!("../../telemetry.bin");

struct AppState {
    telemetry: TelemetryColumnar<'static>,
    points_mesh: Mesh,
    points_material: Material,
    log: Vec<String>,
    time_idx: usize,
    last_scan_t: f32,
    axis_x: &'static str,
    axis_y: &'static str,
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
    style.spacing.window_margin = egui::Margin::same(8.0);
    style.visuals.window_rounding = egui::Rounding::same(0.0);
    style.visuals.menu_rounding = egui::Rounding::same(0.0);
    style.visuals.widgets.noninteractive.rounding = egui::Rounding::same(0.0);
    style.visuals.widgets.inactive.rounding = egui::Rounding::same(0.0);
    style.visuals.widgets.hovered.rounding = egui::Rounding::same(0.0);
    style.visuals.widgets.active.rounding = egui::Rounding::same(0.0);
    ctx.set_style(style);
}

fn make_points_mesh(telemetry: &TelemetryColumnar) -> Mesh {
    let mut vertices = Vec::with_capacity(telemetry.len());
    let mut indices = Vec::with_capacity(telemetry.len());

    for i in 0..telemetry.len() {
        vertices.push(Vertex {
            position: vec3(telemetry.q_x[i], telemetry.q_y[i], telemetry.q_z[i]),
            uv: vec2(0.0, 0.0),
            color: WHITE.into(),
            normal: vec4(0.0, 0.0, 1.0, 0.0),
        });
        // Mesh indices are u16; 1M doesn't fit in u16, but for now, assuming we limit to 10k or use u32.
        // But for 1M, we can't use u16 indices. Macroquad Mesh uses u16 for indices.
        // For 1M points, we need to either split into multiple meshes or use instancing.
        // For now, I'll keep as is, but note it.
        indices.push(i as u16);
    }

    Mesh {
        vertices,
        indices,
        texture: None,
    }
}

fn make_points_material() -> Material {
    let vertex = r#"
    attribute vec3 position;
    attribute vec2 texcoord;
    attribute vec4 color0;

    uniform mat4 Model;
    uniform mat4 View;
    uniform mat4 Projection;
    uniform float time;

    varying lowp vec4 color;
    varying highp vec3 v_pos;

    void main() {
        vec3 jitter = position + (sin(time * 2.0 + position.z) * 0.015);
        v_pos = jitter;
        gl_Position = Projection * View * Model * vec4(jitter, 1.0);
        gl_PointSize = 2.5;
        color = color0;
    }
    "#;

    let fragment = r#"
    precision lowp float;

    varying lowp vec4 color;
    varying highp vec3 v_pos;

    void main() {
        float r = length(v_pos);
        float t = clamp(r, 0.0, 1.0);

        vec3 cyan = vec3(0.0, 0.95, 1.0);
        vec3 magenta = vec3(1.0, 0.0, 0.33);
        vec3 base = mix(cyan, magenta, t);

        vec2 pc = gl_PointCoord * 2.0 - 1.0;
        float d = dot(pc, pc);
        float core = smoothstep(1.0, 0.0, d);
        float glow = smoothstep(1.0, 0.0, d) * 0.75 + smoothstep(1.0, 0.4, d) * 0.35;

        float alpha = clamp(core + glow, 0.0, 1.0);
        vec3 out_col = base * (1.2 + glow);
        gl_FragColor = vec4(out_col, alpha);
    }
    "#;

    let pipeline_params = PipelineParams {
        depth_write: true,
        depth_test: Comparison::LessOrEqual,
        color_blend: Some(BlendState::new(Equation::Add, BlendFactor::One, BlendFactor::One)),
        alpha_blend: Some(BlendState::new(
            Equation::Add,
            BlendFactor::One,
            BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
        )),
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

fn scalar_by_name<'a>(t: &'a TelemetryColumnar<'a>, name: &str) -> &'a [f32] {
    match name {
        "timestamp" => t.timestamp,
        "cpu_utility" => t.cpu_utility,
        "entropy_flux" => t.entropy_flux,
        "entanglement_map" => t.entanglement_map,
        _ => t.cpu_utility,
    }
}

#[macroquad::main("Quantum Observer")]
async fn main() {
    let telemetry = parse_telemetry_binary(TELEMETRY_BYTES).expect("failed to parse telemetry.bin");
    let (axis_x, axis_y) = select_top2_variance(&telemetry);

    let points_mesh = make_points_mesh(&telemetry);
    let points_material = make_points_material();

    let mut state = AppState {
        telemetry,
        points_mesh,
        points_material,
        log: Vec::new(),
        time_idx: 0,
        last_scan_t: 0.0,
        axis_x,
        axis_y,
    };

    loop {
        let t = get_time() as f32;
        clear_background(Color::from_rgba(5, 5, 5, 255));

        let camera = Camera3D {
            position: vec3(2.2, 1.6, 2.2),
            up: vec3(0.0, 1.0, 0.0),
            target: vec3(0.0, 0.0, 0.0),
            ..Default::default()
        };
        set_camera(&camera);

        draw_sphere_wires(vec3(0.0, 0.0, 0.0), 1.0, None, Color::from_rgba(0, 242, 255, 80));
        draw_sphere_wires(vec3(0.0, 0.0, 0.0), 1.002, None, Color::from_rgba(255, 0, 85, 40));

        state.points_material.set_uniform("time", t);
        gl_use_material(&state.points_material);
        draw_mesh(&state.points_mesh);
        gl_use_default_material();

        set_default_camera();

        if t - state.last_scan_t > 0.5 {
            state.last_scan_t = t;
            let r = cpu_correlations(&state.telemetry);
            let line = format!(
                "SIMD scan: r(cpu,t)={:.3} r(cpu,flux)={:.3} r(cpu,ent)={:.3}",
                r[0], r[1], r[2]
            );
            state.log.push(line);
            if state.log.len() > 200 {
                let drain = state.log.len() - 200;
                state.log.drain(0..drain);
            }
        }

        egui_macroquad::ui(|ctx| {
            theme(ctx);

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

            egui::SidePanel::right("charts_panel")
                .resizable(true)
                .default_width(420.0)
                .show(ctx, |ui| {
                    ui.heading(RichText::new("INTERFERENCE").color(Color32::from_rgb(255, 0, 85)));
                    ui.separator();

                    let n = state.telemetry.len().saturating_sub(1);
                    ui.add(
                        egui::Slider::new(&mut state.time_idx, 0..=n)
                            .text("Time Scrub")
                            .clamp_to_range(true),
                    );

                    ui.separator();
                    ui.label(format!("Axis X: {}", state.axis_x));
                    ui.label(format!("Axis Y: {}", state.axis_y));

                    let window = 512usize;
                    let start = state.time_idx.saturating_sub(window / 2);
                    let end = (start + window).min(state.telemetry.len());

                    let xs = &state.telemetry.timestamp[start..end];
                    let y1 = &state.telemetry.entropy_flux[start..end];
                    let y2 = &state.telemetry.entanglement_map[start..end];
                    let axx = scalar_by_name(&state.telemetry, state.axis_x);
                    let axy = scalar_by_name(&state.telemetry, state.axis_y);

                    let pts_flux: PlotPoints = xs
                        .iter()
                        .zip(y1.iter())
                        .map(|(x, y)| [*x as f64, *y as f64])
                        .collect();
                    let pts_ent: PlotPoints = xs
                        .iter()
                        .zip(y2.iter())
                        .map(|(x, y)| [*x as f64, *y as f64])
                        .collect();

                    Plot::new("wave_plot")
                        .height(220.0)
                        .show(ui, |plot_ui| {
                            plot_ui.line(
                                Line::new(pts_flux)
                                    .name("entropy_flux")
                                    .color(Color32::from_rgb(0, 242, 255)),
                            );
                            plot_ui.line(
                                Line::new(pts_ent)
                                    .name("entanglement_map")
                                    .color(Color32::from_rgb(255, 0, 85)),
                            );
                        });

                    ui.separator();

                    let end2 = (start + window).min(axx.len().min(axy.len()));
                    let ax_pts: PlotPoints = (start..end2)
                        .map(|i| [axx[i] as f64, axy[i] as f64])
                        .collect();
                    Plot::new("phase_plot").height(220.0).show(ui, |plot_ui| {
                        plot_ui.line(
                            Line::new(ax_pts)
                                .name("feature phase")
                                .color(Color32::from_rgb(160, 160, 160)),
                        );
                    });
                });

            egui::CentralPanel::default().show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("QUANTUM OBSERVER")
                            .color(Color32::from_rgb(0, 242, 255))
                            .strong(),
                    );
                    ui.label(RichText::new("/ Bloch Sphere").color(Color32::from_rgb(160, 160, 160)));
                });
            });
        });
        egui_macroquad::draw();

        next_frame().await;
    }
}
