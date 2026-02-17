precision highp float;

attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color0;
attribute vec4 normal;

varying lowp vec4 vColor;

uniform mat4 Model;
uniform mat4 View;
uniform mat4 Projection;

void main() {
    vec4 view_pos = View * Model * vec4(position, 1.0);

    if (view_pos.z > -0.05) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        vColor = color0;
        return;
    }

    // Expand a camera-facing quad in view space.
    vec2 tc = texcoord;
    if (!(tc.x == tc.x)) tc.x = 0.5;
    if (!(tc.y == tc.y)) tc.y = 0.5;
    tc = clamp(tc, vec2(0.0, 0.0), vec2(1.0, 1.0));
    vec2 corner = tc - vec2(0.5, 0.5);
    float size = 0.01;
    view_pos.xy += corner * size;

    gl_Position = Projection * view_pos;
    vColor = color0;
}
