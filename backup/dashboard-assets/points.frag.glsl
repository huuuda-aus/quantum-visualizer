precision lowp float;

varying lowp vec4 color;
varying highp vec3 v_pos;
varying highp vec2 uv;

void main() {
    vec3 base = color.rgb;

    vec2 pc = uv * 2.0 - 1.0;
    float d = dot(pc, pc);

    float core = smoothstep(1.0, 0.0, d);
    float glow = smoothstep(1.0, 0.0, d) * 1.15 + smoothstep(1.0, 0.35, d) * 0.65;

    float alpha = clamp((core + glow) * 1.25, 0.0, 1.0);
    vec3 out_col = base * (2.2 + glow * 1.25);

    gl_FragColor = vec4(out_col, alpha);
}
