precision highp float;

attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color0;
attribute vec4 normal;

varying lowp vec4 vColor;
varying highp vec2 vUv;
varying mediump float vDist;

uniform mat4 Model;
uniform mat4 View;
uniform mat4 Projection;
uniform float time;

void main() {
    vec4 view_pos = View * Model * vec4(position, 1.0);
    vDist = length(view_pos.xyz);

    // Some Macroquad pipelines/materials can fail to bind `texcoord` as expected.
    // We also encode UVs into `normal.xy` from Rust; use it as a fallback.
    vec2 uv = texcoord;
    if (abs(uv.x) + abs(uv.y) < 0.00001) {
        uv = normal.xy;
    }

    // Quads are expanded on the CPU; do not expand again here.

    gl_Position = Projection * view_pos;
    vUv = uv;
    vColor = color0;
}
