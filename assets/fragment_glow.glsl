precision mediump float;

varying lowp vec4 vColor;
varying highp vec2 vUv;
varying mediump float vDist;

uniform float glow_strength;

void main() {
    // Radial glow around sprite center.
    float dist = length(vUv - vec2(0.5));
    if (dist > 0.5) {
        discard;
    }

    float alpha = 1.0 - smoothstep(0.0, 0.5, dist);

    // Slight distance attenuation so far points don't over-saturate.
    float atten = 1.0 / (1.0 + 0.25 * vDist * vDist);
    vec3 glowColor = vColor.rgb * (2.0 + 2.0 * atten) * glow_strength;

    // Additive blending ignores alpha; bake the falloff into RGB.
    gl_FragColor = vec4(glowColor * alpha, 1.0);
}
