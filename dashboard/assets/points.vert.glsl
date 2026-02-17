attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color0;

uniform mat4 Model;
uniform mat4 View;
uniform mat4 Projection;
uniform float time;

varying lowp vec4 color;
varying highp vec3 v_pos;
varying highp vec2 uv;

void main() {
    float j = 0.035;
    vec3 wobble = vec3(
        sin(time * 2.4 + position.y * 6.0),
        cos(time * 2.1 + position.z * 6.0),
        sin(time * 2.7 + position.x * 6.0)
    ) * j;
    vec3 center = position + wobble;
    v_pos = center;
    uv = texcoord;

    // Expand into a screen-aligned quad in view space (billboard).
    vec4 view_pos = View * Model * vec4(center, 1.0);
    vec2 corner = (texcoord - vec2(0.5, 0.5));
    float size = 0.16;
    view_pos.xy += corner * size;

    gl_Position = Projection * view_pos;
    color = color0;
}
