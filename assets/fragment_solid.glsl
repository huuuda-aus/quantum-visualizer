precision mediump float;

varying lowp vec4 vColor;
varying highp vec2 vUv;
varying mediump float vDist;

void main() {
    gl_FragColor = vec4(vColor.rgb, 1.0);
}
