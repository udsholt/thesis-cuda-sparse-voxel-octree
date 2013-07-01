#version 330
uniform mat4 projectionMatrix;

in  vec3 vertexPosition;

void main() {
    gl_Position = projectionMatrix * vec4(vertexPosition.xyz, 1.0);
}

