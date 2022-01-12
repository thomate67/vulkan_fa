#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 fragColor;

layout(binding = 1) uniform sampler2D texSampler;

void main()
{
	fragColor = texture(texSampler, fragTexCoord);
}