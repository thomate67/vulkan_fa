#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <GLFW/glfw3.h>
#include <vector>

#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

enum class Direction
{
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT,
	UP,
	DOWN
};

// default camera values
constexpr float SPEED = 10.0f;
constexpr float SENSITIVTY = 0.05f;

/**
 * @brief first person camera
 * @details camera movements:
 *		WASD:		move left/right/forward/backward
 *		MOUSE:		hold right mouse button to turn camera
 *		C:			toggle turning camera as alternative to right mouse button
 *		Left Shift:	speed up
 *		R/F:		move up/down
 *		T/G:		increase/decrease camera movement speed
 *		SPACE:		reset camera
 */
class Camera
{
public:
	/**
	 * @brief creates camera object and sets starting parameters
	 * @param position set starting position
	 * @param up set starting up vector
	 * @param yaw set starting rotation
	 * @param pitch set starting rotation
	 */
	Camera(int width, int height, glm::vec3 position = glm::vec3(0.0f), glm::vec3 target = { 0, 0, 0 },
		glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float fov = 60.0f, float near = 0.1f, float far = 3000.0f);

	/**
	 * @brief handles input and updates camera values
	 * @param window GLFWwindow to get inputs from
	 */
	void update(GLFWwindow* window);

	/**
	 * @brief gets Position of first Boid so the camera follows it
	 * @param window GLFWwindow to get inputs from
	 * @param boidPos Position of first Boid
	 */
	void follow(GLFWwindow* window, glm::vec3 boidPos);

	/**
	 * @brief resets camera to starting position and orientation
	 */
	void reset();

	/** @return The view matrix. */
	glm::mat4 view() const;

	/** @return The projection matrix. */
	glm::mat4 projection() const;

	// camera attributes
	glm::vec3 position; //!< The camera position.
	glm::quat rotation; //!< The camera rotation quaternion.

	/** @return The local forward vector ({0, 0, -1} rotated). */
	glm::vec3 forward() const;
	/** @return The local backward vector ({0, 0, 1} rotated). */
	glm::vec3 backward() const;
	/** @return The local up vector ({0, 1, 0} rotated). */
	glm::vec3 up() const;
	/** @return The local down vector ({0, -1, 0} rotated). */
	glm::vec3 down() const;
	/** @return The local left vector ({-1, 0, 0} rotated). */
	glm::vec3 left() const;
	/** @return The local right vector ({1, 0, 0} rotated). */
	glm::vec3 right() const;

private:
	void processKeyboard(Direction direction, double deltaTime, bool speedModifier);
	void processMouseMovement(double xoffset, double yoffset);
	void updateSpeed(float speed);

	// starting paramters for reset
	glm::vec3 m_startPosition;
	glm::quat m_startRotation;
	glm::mat4 m_projection;

	int m_width;
	int m_height;

	// camera options
	float m_speed;
	float m_sensitivity;

	double m_lastTime;
	double m_lastX = 0.0;
	double m_lastY = 0.0;
	bool   m_cameraActive = false;
	bool   m_mousePressed = false;
	bool   m_cPressed = false;
	bool   m_bPressed = false;
};