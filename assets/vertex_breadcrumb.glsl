precision highp float;

attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color0;

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

    vec2 corner = texcoord - vec2(0.5, 0.5);

    float size = 0.45;
    view_pos.xy += corner * size;
    gl_Position = Projection * view_pos;

    vUv = texcoord;
    vColor = color0;
}
