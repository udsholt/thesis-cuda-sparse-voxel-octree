#version 330

uniform sampler2D textTexture;
uniform vec4 tint;

in  vec2 st;
out vec4 color;

void main() {
    color = texture(textTexture, st) * tint;
}
