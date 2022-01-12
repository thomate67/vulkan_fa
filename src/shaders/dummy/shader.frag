#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 fragColor;

void main()
{
	fragColor = vec4(0.698f, 0.1333f, 0.1333f, 1.0f);
}