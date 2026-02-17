precision mediump float;

varying lowp vec4 vColor;
varying highp vec2 vUv;
varying mediump float vDist;

void main() {
    // Solid point sprites (no blending): shade by distance.
    // Near = bright, far = dim.
    float atten = 1.0 / (1.0 + 0.35 * vDist * vDist);
    vec3 col = vColor.rgb * (0.6 + 1.8 * atten);
    gl_FragColor = vec4(col, 1.0);
}
