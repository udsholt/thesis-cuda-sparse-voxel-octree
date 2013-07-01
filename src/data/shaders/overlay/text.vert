#version 330
uniform mat4 projectionMatrix;
uniform vec2 offset;

in  vec3 vertexPosition;
in  vec2 vertexTexcoord;

out vec2 st;

void main() {
    st = vertexTexcoord;
    gl_Position = projectionMatrix * vec4(vertexPosition.xy + offset, vertexPosition.z, 1.0);
}

