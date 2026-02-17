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
    float waveX = sin(time * 2.0 + position.z * 0.5) * 0.02;
    float waveY = cos(time * 2.5 + position.x * 0.5) * 0.02;
    float waveZ = sin(time * 1.5 + position.y * 0.5) * 0.02;

    vec3 jittered_pos = position + vec3(waveX, waveY, waveZ);

    vec4 view_pos = View * Model * vec4(jittered_pos, 1.0);
    vDist = length(view_pos.xyz);
    vec2 corner = texcoord - vec2(0.5, 0.5);

    float pulse = 1.0 + sin(time * 3.0 + position.x) * 0.2;
    float size = 0.14 * pulse;

    view_pos.xy += corner * size;
    gl_Position = Projection * view_pos;

    vUv = texcoord;
    vColor = color0;
}
