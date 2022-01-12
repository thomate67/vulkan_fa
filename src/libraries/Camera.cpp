#include "Camera.h"
#include <iostream>
#include "glm/gtx/quaternion.hpp"

Camera::Camera(int width, int height, glm::vec3 position, glm::vec3 target,
	glm::vec3 up, float fov, float near, float far)
	: position(position)
	, rotation(glm::quatLookAt(glm::normalize(target - position), up))
	, m_speed(SPEED)
	, m_sensitivity(SENSITIVTY)
{
	m_startPosition = position;
	m_startRotation = rotation;
	m_lastTime = glfwGetTime();
	m_width = width;
	m_height = height;

	m_projection = glm::perspective(glm::radians(fov), m_width / (float)m_height, near, far);
}

void Camera::update(GLFWwindow* window)
{
	// update time
	double currentFrameTime = glfwGetTime();
	double deltaTime = currentFrameTime - m_lastTime;
	m_lastTime = currentFrameTime;

	// check keyboard input
	bool speedMod = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		processKeyboard(Direction::FORWARD, deltaTime, speedMod);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		processKeyboard(Direction::BACKWARD, deltaTime, speedMod);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		processKeyboard(Direction::LEFT, deltaTime, speedMod);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		processKeyboard(Direction::RIGHT, deltaTime, speedMod);
	if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
		processKeyboard(Direction::UP, deltaTime, speedMod);
	if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
		processKeyboard(Direction::DOWN, deltaTime, speedMod);
	if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS)
		updateSpeed(m_speed + 0.01f);
	if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)
		updateSpeed(m_speed - 0.01f);
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		reset();
	if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS && !m_cPressed)
	{
		m_cPressed = true;
		m_cameraActive = !m_cameraActive;
		glfwGetCursorPos(window, &m_lastX, &m_lastY);
	}
	else if (glfwGetKey(window, GLFW_KEY_C) == GLFW_RELEASE)
		m_cPressed = false;
	// check mouse input
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS && !m_mousePressed)
	{
		m_mousePressed = true;
		m_cameraActive = true;
		glfwGetCursorPos(window, &m_lastX, &m_lastY);
	}
	else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE && m_mousePressed)
	{
		m_mousePressed = false;
		m_cameraActive = false;
	}

	if (m_cameraActive)
	{
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		double xoffset = xpos - m_lastX;
		double yoffset = m_lastY - ypos;
		m_lastX = xpos;
		m_lastY = ypos;

		processMouseMovement(xoffset, yoffset);
	}
	else
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

void Camera::follow(GLFWwindow* window, glm::vec3 boidPos)
{
	// update time
	double currentFrameTime = glfwGetTime();
	double deltaTime = currentFrameTime - m_lastTime;
	m_lastTime = currentFrameTime;

	// check keyboard input
	bool speedMod = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		processKeyboard(Direction::FORWARD, deltaTime, speedMod);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		processKeyboard(Direction::BACKWARD, deltaTime, speedMod);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		processKeyboard(Direction::LEFT, deltaTime, speedMod);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		processKeyboard(Direction::RIGHT, deltaTime, speedMod);
	if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
		processKeyboard(Direction::UP, deltaTime, speedMod);
	if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
		processKeyboard(Direction::DOWN, deltaTime, speedMod);
	if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS)
	{
		position = boidPos;
	}
	if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS)
		updateSpeed(m_speed + 0.01f);
	if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)
		updateSpeed(m_speed - 0.01f);
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		reset();
	if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS && !m_cPressed)
	{
		m_cPressed = true;
		m_cameraActive = !m_cameraActive;
		glfwGetCursorPos(window, &m_lastX, &m_lastY);
	}
	else if (glfwGetKey(window, GLFW_KEY_C) == GLFW_RELEASE)
		m_cPressed = false;
	// check mouse input
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS && !m_mousePressed)
	{
		m_mousePressed = true;
		m_cameraActive = true;
		glfwGetCursorPos(window, &m_lastX, &m_lastY);
	}
	else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE && m_mousePressed)
	{
		m_mousePressed = false;
		m_cameraActive = false;
	}

	if (m_cameraActive)
	{
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		double xoffset = xpos - m_lastX;
		double yoffset = m_lastY - ypos;
		m_lastX = xpos;
		m_lastY = ypos;

		processMouseMovement(xoffset, yoffset);
	}
	else
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

void Camera::reset()
{
	position = m_startPosition;
	rotation = m_startRotation;
}

glm::mat4 Camera::view() const
{
	return glm::lookAt(position, position + forward(), up());
}

glm::mat4  Camera::projection() const
{
	return m_projection;
}



void Camera::processKeyboard(Direction direction, double deltaTime, bool speedModifier)
{
	float velocity = m_speed * static_cast<float>(deltaTime);
	if (speedModifier)
		velocity *= m_speed;
	switch (direction)
	{
	case Direction::FORWARD:
		position += forward() * velocity;
		break;
	case Direction::BACKWARD:
		position += backward() * velocity;
		break;
	case Direction::RIGHT:
		position += right() * velocity;
		break;
	case Direction::LEFT:
		position += left() * velocity;
		break;
	case Direction::UP:
		position += up() * velocity;
		break;
	case Direction::DOWN:
		position += down() * velocity;
		break;
	default:
		break;
	}
}

void Camera::processMouseMovement(double xoffset, double yoffset)
{
	xoffset *= this->m_sensitivity;
	yoffset *= this->m_sensitivity;

	rotation = glm::angleAxis(glm::radians(static_cast<float>(xoffset)), glm::vec3(0, -1, 0)) *
		rotation;
	rotation = rotation *
		glm::angleAxis(glm::radians(static_cast<float>(yoffset)), glm::vec3(1, 0, 0));
}

inline void Camera::updateSpeed(float speed)
{
	if (speed > 0)
		this->m_speed = speed;
}
glm::vec3 Camera::forward() const { return rotation * glm::vec3(0, 0, -1); }
glm::vec3 Camera::backward() const { return rotation * glm::vec3(0, 0, 1); }
glm::vec3 Camera::up() const { return rotation * glm::vec3(0, 1, 0); }
glm::vec3 Camera::down() const { return rotation * glm::vec3(0, -1, 0); }
glm::vec3 Camera::left() const { return rotation * glm::vec3(-1, 0, 0); }
glm::vec3 Camera::right() const { return rotation * glm::vec3(1, 0, 0); }