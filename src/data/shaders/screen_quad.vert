#version 330
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;

in vec3 vertexPosition;
in vec2 vertexTexcoord;

out vec3 fragTexcoord;

void main() {
    fragTexcoord = vec3(vertexTexcoord, 0.0);
    gl_Position  = projectionMatrix * vec4(vertexPosition, 1.0);
}

