precision mediump float;

varying lowp vec4 vColor;
varying highp vec2 vUv;
varying mediump float vDist;

uniform float glow_strength;

void main() {
    vec3 col = vec3(1.0) * glow_strength;
    gl_FragColor = vec4(col, 1.0);
}
