precision mediump float;

varying lowp vec4 vColor;
varying highp vec2 vUv;

void main() {
    gl_FragColor = vec4(vColor.rgb, 1.0);
}
