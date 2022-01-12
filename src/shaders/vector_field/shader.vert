#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject
{
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

layout (location = 2) in vec3 boid_position;
layout (location = 3) in vec3 boid_velocity;
layout (location = 4) in vec4 boid_color;

layout (location = 0) out vec4 color;

mat4 make_lookat(vec3 forward, vec3 up)
{
    vec3 side = cross(forward, up);
    vec3 u_frame = cross(side, forward);

    return mat4(vec4(side, 0.0),
                vec4(u_frame, 0.0),
                vec4(forward, 0.0),
                vec4(0.0, 0.0, 0.0, 1.0));
}


void main()
{
	mat4 lookat = make_lookat(normalize(boid_velocity), vec3(0.0, 1.0, 0.0));
    vec4 obj_coord = lookat * vec4(position.xyz, 1.0);
    gl_Position = ubo.proj * ubo.view * ubo.model * (obj_coord + vec4(boid_position, 0.0));
	gl_PointSize = 5.0f;
	color = boid_color;
}

///////////////////////////////////////////////// TEST TEST TEST //////////////////////////////////////////////////////////////////////

/*layout(binding = 0) uniform UniformBufferObject
{
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout (location = 2) in vec3 boid_position;
layout (location = 3) in vec3 boid_velocity;

layout(location = 0) out vec3 fragColor;

void main()
{
	vec4 vertPos = vec4(inPosition, 0.0, 1.0);
	gl_Position = ubo.proj * ubo.view * ubo.model * (vertPos + vec4(boid_position, 0.0));
	fragColor = inColor;
}*/
