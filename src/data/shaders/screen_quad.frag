#version 330

uniform ivec2 dimensions;
uniform samplerBuffer buffer;

in  vec3 fragTexcoord;
out vec4 color;

void main() {

    int i = int(fragTexcoord.x * float(dimensions.x));
    int j = int(fragTexcoord.y * float(dimensions.y));
    
    color = texelFetch(buffer, i + dimensions.x * j);
    
    //color = vec4(fragTexcoord, 1.0);
}
