#version 450

layout(location = 0) out vec4 outColor;
layout(location = 0) in vec2 fragTexCoord;

layout(binding = 0) uniform sampler2D texSampler;

void main() {
    float value = texture(texSampler, fragTexCoord).r;
    outColor = vec4(value, value, value, 1.0);
}