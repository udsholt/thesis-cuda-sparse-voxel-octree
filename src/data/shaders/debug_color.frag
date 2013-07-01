#version 330

uniform vec4 debugColor;

in vec3 interpVexterColor;

out vec4 fragColor;

void main() {
    fragColor = vec4(interpVexterColor, 1.0) * debugColor;
}
