#version 330
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 viewMatrixInverse;

in vec3 vertexPosition;
in vec3 vertexColor;

out vec3 interpVexterColor;

void main() {
    interpVexterColor = vertexColor;
    gl_Position = projectionMatrix * viewMatrix * vec4(vertexPosition, 1.0);
}

