precision highp float;

attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color0;

varying lowp vec4 vColor;
varying highp vec2 vUv;

uniform mat4 Model;
uniform mat4 View;
uniform mat4 Projection;

void main() {
    vUv = texcoord;
    vColor = color0;
    gl_Position = Projection * View * Model * vec4(position, 1.0);
}
