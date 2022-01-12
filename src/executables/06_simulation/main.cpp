#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include "Camera.h"
//#include "Grid.h"

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <functional>
#include <cstdlib>
#include <optional>
#include <set>
#include <fstream>
#include <array>
#include <chrono>
#include <cmath>
#include <random>

const int WIDTH = 800;
const int HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;

uint32_t readSet = 0;

Camera camera(WIDTH, HEIGHT, glm::vec3(10.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
int neighbor_amount = 15;
int old_t = 0;



struct Uniform_grid_params {

	Uniform_grid_params(const float grid_size_x, const float grid_size_y, const float grid_size_z,
		const float cell_size_x, const float cell_size_y, const float cell_size_z,
		const int capacity, const glm::vec4 pos = glm::vec4(0.0f))
		:grid_pos(pos), cell_capacity(capacity)
	{
		const float cell_count_x = glm::ceil(grid_size_x / cell_size_x);
		const float cell_count_y = glm::ceil(grid_size_y / cell_size_y);
		const float cell_count_z = glm::ceil(grid_size_z / cell_size_z);
		cell_count = glm::ivec4(cell_count_x, cell_count_y, cell_count_z, 0);
		total_cell_count = static_cast<int>(cell_count_x * cell_count_y * cell_count_z);
		cell_size = glm::vec4(cell_size_x, cell_size_y, cell_size_z, 0);
		const auto half_size = glm::vec4(cell_size_x * cell_count_x, cell_size_y * cell_count_y,
			cell_size_z * cell_count_z, 0) / 2.0f;
		grid_min = grid_pos - half_size;
		grid_capacity = total_cell_count * cell_capacity;
	}

	glm::vec4 grid_min;
	glm::ivec4 cell_count;
	glm::vec4 cell_size;
	glm::vec4 grid_pos;
	int total_cell_count;
	int cell_capacity;
	int grid_capacity;
};

//Uniform_grid_params grid_params(5.0f, 5.0f, 5.0f, 1.0f, 1.0f, 1.0f, 15);
//Uniform_grid_params grid_params(4.0f, 4.0f, 4.0f, 1.0f, 1.0f, 1.0f, 15);
//Uniform_grid_params grid_params(4.0f, 4.0f, 4.0f, 0.1f, 0.1f, 0.1f, 15);
//Uniform_grid_params grid_params(300.0f, 300.0f, 300.0f, 5.0f, 5.0f, 5.0f, 1);
//Uniform_grid_params grid_params(50.0f, 50.0f, 50.0f, 1.0f, 1.0f, 1.0f, 25);
Uniform_grid_params grid_params(100.0f, 100.0f, 100.0f, 1.0f, 1.0f, 1.0f, 50);

//int FLOCK_SIZE = grid_params.total_cell_count;
int plasma_count = 2000;
int atoms_count = 50000;
int total_particle_count = plasma_count + atoms_count;

enum
{
	WORKGROUP_SIZE = 128,
	NUM_WORKGROUPS = 1,
	//FLOCK_SIZE = (NUM_WORKGROUPS * WORKGROUP_SIZE)
	//FLOCK_SIZE = 15
};

const std::vector<const char*> validationLayers =
{
	"VK_LAYER_LUNARG_standard_validation"
};

const std::vector<const char*> deviceExtensions =
{
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
{
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr)
	{
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else
	{
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr)
	{
		func(instance, debugMessenger, pAllocator);
	}
}

struct QueueFamilyIndices
{
	std::optional<uint32_t> computeFamily;
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete()
	{
		return computeFamily.has_value() && graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

struct Plasma
{
	glm::vec4 pos;
	glm::vec4 vel;
	glm::vec4 color;
	float lifetime;
	float max_lifetime;
	int hit_counter;
	float pad1;
};

struct Atom
{
	glm::vec4 pos;
	glm::vec4 vel;
	glm::vec4 color;
	int type;				//0 -> oxygen(red); 1 -> oxygen(green); 2 -> nitrogen
	int is_excited;			//0 -> no collision with plasma; 1 -> collision with plasma
	int is_dead;			//technically the atome doesn't die, but the name is_no_longer_capable_to_be_excited is unpractically to write all the time
	float lifetime;			//will be initialised with max_lifetime and then reduced in the compute shader
	float max_lifetime;		//corresponds to transition state of the atome type		
	int pad1;
	int pad2;
	int pad3;
};

struct Vector_Rods
{
	glm::vec4 position;
	glm::vec4 velocity;
};

struct Grid
{
	glm::vec3 pos;
	float pad3;
	glm::vec3 vel;
	float pad4;
};

struct Vertex_Plasma
{
	glm::vec3 position;
	glm::vec3 normal;
};

struct Vertex_Atoms
{
	glm::vec3 position;
	glm::vec3 normal;
};

struct Vertex_Vector_Rods
{
	glm::vec3 position;
	glm::vec3 normal;
};

struct UniformBufferObject
{
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

struct Uniform_Variables
{
	float delta_time;
	float theta;
};

struct UBO
{
	glm::vec3 goal;
} uboCompute;

const std::vector<Vertex_Plasma> plasma_vertices =
{
	{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}}
	//{{1.5f, 0.0f}, {0.0f, 0.0f, 0.0f}}
};

const std::vector<Vertex_Plasma> atoms_vertices =
{
	{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}}
};

const std::vector<Vertex_Vector_Rods> rods_vertices =
{
	{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
	{{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}}
};

struct VectorField
{
	glm::vec4 pos;
	glm::vec4 vector;
};

struct Skybox
{
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;

	static VkVertexInputBindingDescription getBindingDescription()
	{
		VkVertexInputBindingDescription bindingDescription = {};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Skybox);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions()
	{
		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Skybox, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Skybox, color);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Skybox, texCoord);

		return attributeDescriptions;
	}
};

const std::vector<Skybox> sky_vertices =
{
	//front face
	{{-50.0f, -50.0f, -50.0f}, {1.0f, 0.0f, 0.0f}, {1.0f / 4.0f, 1.0f / 3.0f}},		//vertex 0
	{{50.0f, -50.0f, -50.0f}, {1.0f, 0.0f, 0.0f}, {2.0f / 4.0f, 1.0f / 3.0f}},		//vertex 1
	{{50.0f, 50.0f, -50.0f}, {1.0f, 0.0f, 0.0f}, {2.0f / 4.0f, 2.0f / 3.0f}},		//vertex 2
	{{-50.0f, 50.0f, -50.0f}, {1.0f, 0.0f, 0.0f}, {1.0f / 4.0f, 2.0f / 3.0f}},		//vertex 3

	//right face																	
	{{50.0f, -50.0f, 50.0f}, {1.0f, 0.0f, 0.0f}, {3.0f / 4.0f, 1.0f / 3.0f}},		//vertex 4
	{{50.0f, 50.0f, 50.0f}, {1.0f, 0.0f, 0.0f}, {3.0f / 4.0f, 2.0f / 3.0f}},		//vertex 5

	//back face																		
	{{-50.0f, -50.0f, 50.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f / 3.0f}},				//vertex 6
	{{-50.0f, 50.0f, 50.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 2.0f / 3.0f}},				//vertex 7

	//left face																		
	{{-50.0f, -50.0f, 50.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f / 3.0f}},				//vertex 8
	{{-50.0f, 50.0f, 50.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 2.0f / 3.0f}},				//vertex 9

	//bottom face																	
	{{-50.0f, -50.0f, 50.0f}, {1.0f, 0.0f, 0.0f}, {1.0f / 4.0f, 0.0f}},				//vertex 10
	{{50.0f, -50.0f, 50.0f}, {1.0f, 0.0f, 0.0f}, {2.0f / 4.0f, 0.0f}},				//vertex 11

	//top face																		
	{{50.0f, 50.0f, 50.0f}, {1.0f, 0.0f, 0.0f}, {2.0f / 4.0f, 1.0f}},				//vertex 12
	{{-50.0f, 50.0f,50.0f}, {1.0f, 0.0f, 0.0f}, {1.0f / 4.0f, 1.0f}}				//vertex 13
};

const std::vector<uint16_t> sky_indices =
{
	//front face
	0,1,2,
	2,3,0,

	//right face
	1,4,5,
	5,2,1,

	//back face
	4,6,7,
	7,5,4,

	//left face
	8,0,3,
	3,9,8,

	//bottom face
	10,11,1,
	1,0,10,

	//top face
	3,2,12,
	12,13,3
};

class BoidApplication
{
public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:

	GLFWwindow* window;

	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR surface;

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device;

	VkQueue graphicsQueue;
	VkQueue presentQueue;

	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;
	std::vector<VkFramebuffer> swapChainFramebuffers;

	VkRenderPass renderPass;
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;
	VkPipelineLayout skyboxPipelineLayout;
	VkPipelineLayout atomsPipelineLayout;
	VkPipelineLayout plasmaPipelineLayout;
	VkPipelineLayout vectorRodsPipelineLayout;

	VkPipeline graphicsPipeline;
	VkPipeline skyboxPipeline;
	VkPipeline atomsPipeline;
	VkPipeline plasmaPipeline;
	VkPipeline vectorRodsPipeline;

	VkCommandPool commandPool;

	VkBuffer plasmaVertexBuffer;
	VkDeviceMemory plasmaVertexBufferMemory;

	VkBuffer atomsVertexBuffer;
	VkDeviceMemory atomsVertexBufferMemory;

	VkBuffer vectorRodsVertexBuffer;
	VkDeviceMemory vectorRodsVertexBufferMemory;

	VkBuffer skyboxVertexBuffer;
	VkDeviceMemory skyboxVertexBufferMemory;

	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;

	struct StorageBuffers
	{
		VkBuffer input;
		VkBuffer output;
		VkBuffer plasma;
		VkBuffer atoms;
		VkBuffer grid_params;
		VkBuffer grid_data;
		VkBuffer grid_cell_counter;
		VkBuffer grid_neighborhood_data;
		VkBuffer grid_neighborhood_counter;
		VkBuffer vector_field;
		VkBuffer vector_rods;
	} storageBuffers;

	VkDeviceMemory storageInputMemory;
	VkDeviceMemory storageOutputMemory;
	VkDeviceMemory plasmaBufferMemory;
	VkDeviceMemory atomsBufferMemory;
	VkDeviceMemory gridParamsBufferMemory;
	VkDeviceMemory gridDataBufferMemory;
	VkDeviceMemory gridCounterBufferMemory;
	VkDeviceMemory gridNeighborhoodDataBufferMemory;
	VkDeviceMemory gridNeighborhoodCounterBufferMemory;
	VkDeviceMemory vectorFieldBufferMemory;
	VkDeviceMemory vectorRodsBufferMemory;

	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;

	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;

	VkDescriptorPool descriptorPoolSkybox;
	std::vector<VkDescriptorSet> descriptorSetsSkybox;

	std::vector<VkCommandBuffer> commandBuffers;

	VkImage textureImage;
	VkDeviceMemory textureImageMemory;
	VkImageView textureImageView;
	VkSampler textureSampler;

	std::vector<VkSemaphore> imageAvailableSemaphore;
	std::vector<VkSemaphore> renderFinishedSemaphore;
	std::vector<VkFence> inFlightFences;
	size_t currentFrame = 0;

	//compute Stuff
	VkQueue computeQueue;

	VkDescriptorSetLayout descriptorSetLayoutCompute;
	VkPipelineLayout pipelineLayoutCompute;
	VkPipelineLayout pipelineLayoutGridReset;
	VkPipelineLayout pipelineLayoutGridSetup;
	VkPipelineLayout pipelineLayoutGridFindNeighbor;
	VkPipelineLayout pipelineLayoutVectorField;
	VkPipelineLayout pipelineLayoutPlasmaSimulation;
	VkPipeline computePipeline;
	VkPipeline gridResetPipeline;
	VkPipeline gridSetupPipeline;
	VkPipeline gridFindNeighborPipeline;
	VkPipeline vectorFieldPipeline;
	VkPipeline plasmaSimulationPipeline;

	VkCommandPool commandPoolCompute;

	VkDescriptorPool descriptorPoolCompute;
	std::array<VkDescriptorSet, 2> descriptorSetsCompute;

	std::array<VkCommandBuffer, 2> commandBuffersCompute;

	VkFence computeFence;

	VkBuffer uniformBufferCompute;
	VkDeviceMemory uniformBufferMemoryCompute;

	VkBuffer uniformVariablesBuffer;
	VkDeviceMemory uniformVariablesBufferMemory;

	bool framebufferResized = false;

	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
	{
		auto app = reinterpret_cast<BoidApplication*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	void initVulkan()
	{
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createRenderPass();
		createDescriptorSetLayout();
		createCommandPoolCompute();
		createCommandPool();
		createTextureImage();			//done
		createTextureImageView();		//done
		createTextureSampler();			//done
		createPlasmaVertexBuffer();
		createAtomsVertexBuffer();
		createSkyboxVertexBuffer();
		createVectorRodsVertexBuffer();
		createIndexBuffer(); //for cubemap done
		createUniformBuffers();
		//createStorageBuffers();
		createPlasmaBuffer();
		createAtomsBuffer();
		createGridBuffer();
		createVectorBuffer();
		createVectorRodsBuffer();
		createComputeDescriptorSet();
		createGridResetPipeline();
		createGridSetupPipeline();
		createGridFindNeighborPipeline();
		createVectorFieldPipeline();
		createPlasmaSimulationPipeline();
		createComputePipeline();
		createSkyboxPipeline();	//for cubemap
		//createGraphicsPipeline();
		createPlasmaPipeline();
		createAtomsPipeline();
		createVectorRodsPipeline();
		createFramebuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBufferCompute();
		createCommandBuffers();
		createSyncObjects();
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			drawFrame();
			//std::cout << "Current Frame: " << currentFrame << std::endl;
		}

		vkDeviceWaitIdle(device);
	}

	void cleanupSwapChain()
	{
		for (size_t i = 0; i < swapChainFramebuffers.size(); i++)
		{
			vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
		}

		vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

		vkDestroyPipeline(device, skyboxPipeline, nullptr);
		vkDestroyPipelineLayout(device, skyboxPipelineLayout, nullptr);
		vkDestroyPipeline(device, plasmaPipeline, nullptr);
		vkDestroyPipelineLayout(device, plasmaPipelineLayout, nullptr);
		vkDestroyPipeline(device, atomsPipeline, nullptr);
		vkDestroyPipelineLayout(device, atomsPipelineLayout, nullptr);
		vkDestroyPipeline(device, vectorRodsPipeline, nullptr);
		vkDestroyPipelineLayout(device, vectorRodsPipelineLayout, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);

		for (size_t i = 0; i < swapChainImageViews.size(); i++)
		{
			vkDestroyImageView(device, swapChainImageViews[i], nullptr);
		}

		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}

	void cleanup()
	{
		cleanupSwapChain();

		vkFreeCommandBuffers(device, commandPoolCompute, static_cast<uint32_t>(commandBuffersCompute.size()), commandBuffersCompute.data());

		vkDestroySampler(device, textureSampler, nullptr);
		vkDestroyImageView(device, textureImageView, nullptr);

		vkDestroyImage(device, textureImage, nullptr);
		vkFreeMemory(device, textureImageMemory, nullptr);

		vkDestroyPipeline(device, computePipeline, nullptr);
		vkDestroyPipeline(device, gridResetPipeline, nullptr);
		vkDestroyPipeline(device, gridSetupPipeline, nullptr);
		vkDestroyPipeline(device, gridFindNeighborPipeline, nullptr);
		vkDestroyPipeline(device, vectorFieldPipeline, nullptr);
		vkDestroyPipeline(device, plasmaSimulationPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayoutCompute, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayoutGridReset, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayoutGridSetup, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayoutGridFindNeighbor, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayoutVectorField, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayoutPlasmaSimulation, nullptr);

		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyDescriptorPool(device, descriptorPoolCompute, nullptr);

		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayoutCompute, nullptr);

		for (size_t i = 0; i < swapChainImages.size(); i++)
		{
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
		}

		vkDestroyBuffer(device, uniformVariablesBuffer, nullptr);
		vkFreeMemory(device, uniformVariablesBufferMemory, nullptr);

		vkDestroyBuffer(device, uniformBufferCompute, nullptr);
		vkFreeMemory(device, uniformBufferMemoryCompute, nullptr);

		vkDestroyBuffer(device, storageBuffers.grid_params, nullptr);
		vkDestroyBuffer(device, storageBuffers.grid_cell_counter, nullptr);
		vkDestroyBuffer(device, storageBuffers.grid_data, nullptr);
		vkDestroyBuffer(device, storageBuffers.grid_neighborhood_data, nullptr);
		vkDestroyBuffer(device, storageBuffers.grid_neighborhood_counter, nullptr);
		vkDestroyBuffer(device, storageBuffers.vector_field, nullptr);
		vkDestroyBuffer(device, storageBuffers.vector_rods, nullptr);
		vkFreeMemory(device, gridParamsBufferMemory, nullptr);
		vkFreeMemory(device, gridCounterBufferMemory, nullptr);
		vkFreeMemory(device, gridDataBufferMemory, nullptr);
		vkFreeMemory(device, gridNeighborhoodCounterBufferMemory, nullptr);
		vkFreeMemory(device, gridNeighborhoodDataBufferMemory, nullptr);
		vkFreeMemory(device, vectorFieldBufferMemory, nullptr);
		vkFreeMemory(device, vectorRodsBufferMemory, nullptr);

		vkDestroyBuffer(device, storageBuffers.atoms, nullptr);
		vkFreeMemory(device, atomsBufferMemory, nullptr);
	
		vkDestroyBuffer(device, storageBuffers.plasma, nullptr);
		vkFreeMemory(device, plasmaBufferMemory, nullptr);

		vkDestroyBuffer(device, plasmaVertexBuffer, nullptr);
		vkFreeMemory(device, plasmaVertexBufferMemory, nullptr);

		vkDestroyBuffer(device, atomsVertexBuffer, nullptr);
		vkFreeMemory(device, atomsVertexBufferMemory, nullptr);

		vkDestroyBuffer(device, skyboxVertexBuffer, nullptr);
		vkFreeMemory(device, skyboxVertexBufferMemory, nullptr);

		vkDestroyBuffer(device, vectorRodsVertexBuffer, nullptr);
		vkFreeMemory(device, vectorRodsVertexBufferMemory, nullptr);

		vkDestroyBuffer(device, indexBuffer, nullptr);
		vkFreeMemory(device, indexBufferMemory, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroySemaphore(device, renderFinishedSemaphore[i], nullptr);
			vkDestroySemaphore(device, imageAvailableSemaphore[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}

		vkDestroyFence(device, computeFence, nullptr);

		vkDestroyCommandPool(device, commandPool, nullptr);
		vkDestroyCommandPool(device, commandPoolCompute, nullptr);

		vkDestroyDevice(device, nullptr);

		if (enableValidationLayers)
		{
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();
	}

	void recreateSwapChain()
	{
		int width = 0, height = 0;
		while (width == 0 || height == 0)
		{
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(device);

		cleanupSwapChain();

		createSwapChain();
		createImageViews();
		createRenderPass();
		//createGraphicsPipeline();
		createPlasmaPipeline();
		createAtomsPipeline();
		createFramebuffers();
		createCommandBuffers();
	}

	void createInstance()
	{

		if (enableValidationLayers && !checkValidationLayerSupport())
		{
			throw std::runtime_error("validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Boids";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		auto extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else
		{
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create instance!");
		}
	}

	void setupDebugMessenger()
	{
		if (!enableValidationLayers) return;

		VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
		createInfo.pUserData = nullptr; //optional

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}

	void createSurface()
	{
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create window surface!");
		}
	}

	void pickPhysicalDevice()
	{
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		if (deviceCount == 0)
		{
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		for (const auto& device : devices)
		{
			if (isDeviceSuitable(device))
			{
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE)
		{
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	void createLogicalDevice()
	{
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.computeFamily.value(), indices.graphicsFamily.value(), indices.presentFamily.value() };

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies)
		{
			VkDeviceQueueCreateInfo queueCreateInfo = {};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures = {};
		deviceFeatures.wideLines = VK_TRUE;					//must be true for LineWidth < 1.0f
		deviceFeatures.samplerAnisotropy = VK_TRUE;

		VkDeviceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();

		createInfo.pEnabledFeatures = &deviceFeatures;

		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else
		{
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create logical device!");
		}

		vkGetDeviceQueue(device, indices.computeFamily.value(), 0, &computeQueue);
		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

	void createSwapChain()
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
		{
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;

		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		if (indices.graphicsFamily != indices.presentFamily)
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0; //optional
			createInfo.pQueueFamilyIndices = nullptr; //optional
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;

		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create swap chain!");
		}

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	void createImageViews()
	{
		swapChainImageViews.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++)
		{
			VkImageViewCreateInfo createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = swapChainImages[i];
			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			createInfo.format = swapChainImageFormat;
			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel = 0;
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount = 1;

			if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create image views!");
			}
		}
	}

	void createRenderPass()
	{
		VkAttachmentDescription colorAttachment = {};
		colorAttachment.format = swapChainImageFormat;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentRef = {};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		std::array<VkSubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &colorAttachment;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassInfo.pDependencies = dependencies.data();

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create render pass!");
		}
	}

	void createDescriptorSetLayout()
	{
		VkDescriptorSetLayoutBinding uboLayoutBinding = {};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		uboLayoutBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
		samplerLayoutBinding.binding = 1;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerLayoutBinding.pImmutableSamplers = nullptr;
		samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };
		VkDescriptorSetLayoutCreateInfo layoutInfo = {};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}

	void createComputeDescriptorSet()
	{
		//descriptorSetLayoutCompute
		std::vector<VkDescriptorSetLayoutBinding> storageLayoutBinding =
		{
			descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
			descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
			descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
			descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3),
			descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4),
			descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 5),
			descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 6),
			descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 7),
			descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 8),
			descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 9),
			descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 10),
		};

		VkDescriptorSetLayoutCreateInfo descriptorLayout = {};
		descriptorLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		descriptorLayout.bindingCount = static_cast<uint32_t>(storageLayoutBinding.size());
		descriptorLayout.pBindings = storageLayoutBinding.data();

		if (vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayoutCompute) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create descriptor set layout for compute!");
		}

		// descriptor pool compute

		std::vector<VkDescriptorPoolSize> poolSizes =
		{
			descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4),
			descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 27)  //ssbo count * 3
		};

		VkDescriptorPoolCreateInfo poolInfo = descriptorPoolCreateInfo(poolSizes, 4);

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPoolCompute) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create descriptor pool for compute!");
		}

		// descriptor sets compute

		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPoolCompute;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &descriptorSetLayoutCompute;

		if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSetsCompute[0]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate descriptor set for compute!");
		}

		if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSetsCompute[1]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate descriptor set for compute!");
		}

		VkDescriptorBufferInfo plasmaBufferInputInfo = {};
		plasmaBufferInputInfo.buffer = storageBuffers.plasma;
		plasmaBufferInputInfo.offset = 0;
		plasmaBufferInputInfo.range = sizeof(Plasma) * plasma_count;

		VkDescriptorBufferInfo atomsBufferInputInfo = {};
		atomsBufferInputInfo.buffer = storageBuffers.atoms;
		atomsBufferInputInfo.offset = 0;
		atomsBufferInputInfo.range = sizeof(Atom) * atoms_count;

		VkDescriptorBufferInfo gridParamsBufferInputInfo = {};
		gridParamsBufferInputInfo.buffer = storageBuffers.grid_params;
		gridParamsBufferInputInfo.offset = 0;
		gridParamsBufferInputInfo.range = sizeof(Uniform_grid_params);

		VkDescriptorBufferInfo gridBufferInputInfo = {};
		gridBufferInputInfo.buffer = storageBuffers.grid_data;
		gridBufferInputInfo.offset = 0;
		gridBufferInputInfo.range = grid_params.grid_capacity * sizeof(int);

		VkDescriptorBufferInfo gridCounterBufferInputInfo = {};
		gridCounterBufferInputInfo.buffer = storageBuffers.grid_cell_counter;
		gridCounterBufferInputInfo.offset = 0;
		gridCounterBufferInputInfo.range = grid_params.total_cell_count * sizeof(int);

		VkDescriptorBufferInfo gridVectorBufferInputInfo = {};
		gridVectorBufferInputInfo.buffer = storageBuffers.vector_field;
		gridVectorBufferInputInfo.offset = 0;
		gridVectorBufferInputInfo.range = grid_params.total_cell_count * sizeof(VectorField);

		VkDescriptorBufferInfo gridNeighborhoodDataBufferInputInfo = {};
		gridNeighborhoodDataBufferInputInfo.buffer = storageBuffers.grid_neighborhood_data;
		gridNeighborhoodDataBufferInputInfo.offset = 0;
		gridNeighborhoodDataBufferInputInfo.range = neighbor_amount * atoms_count * sizeof(int);

		VkDescriptorBufferInfo gridNeighborhoodCounterBufferInputInfo = {};
		gridNeighborhoodCounterBufferInputInfo.buffer = storageBuffers.grid_neighborhood_counter;
		gridNeighborhoodCounterBufferInputInfo.offset = 0;
		gridNeighborhoodCounterBufferInputInfo.range = atoms_count * sizeof(int);

		VkDescriptorBufferInfo uboBufferInfo = {};
		uboBufferInfo.buffer = uniformBufferCompute;
		uboBufferInfo.offset = 0;
		uboBufferInfo.range = sizeof(UBO);

		VkDescriptorBufferInfo uniformVariablesBufferInfo = {};
		uniformVariablesBufferInfo.buffer = uniformVariablesBuffer;
		uniformVariablesBufferInfo.offset = 0;
		uniformVariablesBufferInfo.range = sizeof(Uniform_Variables);

		VkDescriptorBufferInfo vectorRodsDataBufferInputInfo = {};
		vectorRodsDataBufferInputInfo.buffer = storageBuffers.vector_rods;
		vectorRodsDataBufferInputInfo.offset = 0;
		vectorRodsDataBufferInputInfo.range = sizeof(Vector_Rods) * grid_params.total_cell_count;

		std::vector<VkWriteDescriptorSet> descriptorWrite =
		{
			writeDescriptorSet(descriptorSetsCompute[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &plasmaBufferInputInfo),
			writeDescriptorSet(descriptorSetsCompute[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &atomsBufferInputInfo),
			writeDescriptorSet(descriptorSetsCompute[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &gridParamsBufferInputInfo),
			writeDescriptorSet(descriptorSetsCompute[0], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3, &uboBufferInfo),
			writeDescriptorSet(descriptorSetsCompute[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, &gridBufferInputInfo),
			writeDescriptorSet(descriptorSetsCompute[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5, &gridCounterBufferInputInfo),
			writeDescriptorSet(descriptorSetsCompute[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6, &gridVectorBufferInputInfo),
			writeDescriptorSet(descriptorSetsCompute[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7, &gridNeighborhoodDataBufferInputInfo),
			writeDescriptorSet(descriptorSetsCompute[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8, &gridNeighborhoodCounterBufferInputInfo),
			writeDescriptorSet(descriptorSetsCompute[0], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 9, &uniformVariablesBufferInfo),
			writeDescriptorSet(descriptorSetsCompute[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10, &vectorRodsDataBufferInputInfo),

			writeDescriptorSet(descriptorSetsCompute[1], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &plasmaBufferInputInfo),
			writeDescriptorSet(descriptorSetsCompute[1], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &atomsBufferInputInfo),
			writeDescriptorSet(descriptorSetsCompute[1], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &gridParamsBufferInputInfo),
			writeDescriptorSet(descriptorSetsCompute[1], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3, &uboBufferInfo),
			writeDescriptorSet(descriptorSetsCompute[1], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, &gridBufferInputInfo),
			writeDescriptorSet(descriptorSetsCompute[1], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5, &gridCounterBufferInputInfo),
			writeDescriptorSet(descriptorSetsCompute[1], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6, &gridVectorBufferInputInfo),
			writeDescriptorSet(descriptorSetsCompute[1], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7, &gridNeighborhoodDataBufferInputInfo),
			writeDescriptorSet(descriptorSetsCompute[1], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8, &gridNeighborhoodCounterBufferInputInfo),
			writeDescriptorSet(descriptorSetsCompute[1], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 9, &uniformVariablesBufferInfo),
			writeDescriptorSet(descriptorSetsCompute[1], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10, &vectorRodsDataBufferInputInfo),
		};

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrite.size()), descriptorWrite.data(), 0, nullptr);
	}

	void createGridResetPipeline()
	{
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		vkGetDeviceQueue(device, queueFamilyIndices.computeFamily.value(), 0, &computeQueue);

		auto compShaderCode = readFile("../src/shaders/simulation/grid_reset.comp.spv");

		VkShaderModule compShaderModule = createShaderModule(compShaderCode);

		VkPipelineShaderStageCreateInfo compShaderStageInfo = {};
		compShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		compShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		compShaderStageInfo.module = compShaderModule;
		compShaderStageInfo.pName = "main";


		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayoutCompute;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayoutGridReset) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create pipeline layout for compute!");
		}


		VkComputePipelineCreateInfo computeInfo = {};
		computeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		computeInfo.stage = compShaderStageInfo;
		computeInfo.layout = pipelineLayoutGridReset;
		computeInfo.flags = 0;

		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computeInfo, nullptr, &gridResetPipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create compute pipeline!");
		}

		vkDestroyShaderModule(device, compShaderModule, nullptr);
	}

	void createGridSetupPipeline()
	{
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		vkGetDeviceQueue(device, queueFamilyIndices.computeFamily.value(), 0, &computeQueue);

		auto compShaderCode = readFile("../src/shaders/simulation/grid_setup.comp.spv");

		VkShaderModule compShaderModule = createShaderModule(compShaderCode);

		VkPipelineShaderStageCreateInfo compShaderStageInfo = {};
		compShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		compShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		compShaderStageInfo.module = compShaderModule;
		compShaderStageInfo.pName = "main";


		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayoutCompute;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayoutGridSetup) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create pipeline layout for compute!");
		}


		VkComputePipelineCreateInfo computeInfo = {};
		computeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		computeInfo.stage = compShaderStageInfo;
		computeInfo.layout = pipelineLayoutGridSetup;
		computeInfo.flags = 0;

		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computeInfo, nullptr, &gridSetupPipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create compute pipeline!");
		}

		vkDestroyShaderModule(device, compShaderModule, nullptr);
	}

	void createGridFindNeighborPipeline()
	{
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		vkGetDeviceQueue(device, queueFamilyIndices.computeFamily.value(), 0, &computeQueue);

		auto compShaderCode = readFile("../src/shaders/simulation/grid_find_neighbors.comp.spv");

		VkShaderModule compShaderModule = createShaderModule(compShaderCode);

		VkPipelineShaderStageCreateInfo compShaderStageInfo = {};
		compShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		compShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		compShaderStageInfo.module = compShaderModule;
		compShaderStageInfo.pName = "main";


		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayoutCompute;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayoutGridFindNeighbor) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create pipeline layout for compute!");
		}


		VkComputePipelineCreateInfo computeInfo = {};
		computeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		computeInfo.stage = compShaderStageInfo;
		computeInfo.layout = pipelineLayoutGridFindNeighbor;
		computeInfo.flags = 0;

		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computeInfo, nullptr, &gridFindNeighborPipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create compute pipeline!");
		}

		vkDestroyShaderModule(device, compShaderModule, nullptr);
	}

	void createVectorFieldPipeline()
	{
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		vkGetDeviceQueue(device, queueFamilyIndices.computeFamily.value(), 0, &computeQueue);

		auto compShaderCode = readFile("../src/shaders/simulation/vector_field.comp.spv");

		VkShaderModule compShaderModule = createShaderModule(compShaderCode);

		VkPipelineShaderStageCreateInfo compShaderStageInfo = {};
		compShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		compShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		compShaderStageInfo.module = compShaderModule;
		compShaderStageInfo.pName = "main";


		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayoutCompute;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayoutVectorField) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create pipeline layout for compute!");
		}


		VkComputePipelineCreateInfo computeInfo = {};
		computeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		computeInfo.stage = compShaderStageInfo;
		computeInfo.layout = pipelineLayoutVectorField;
		computeInfo.flags = 0;

		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computeInfo, nullptr, &vectorFieldPipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create compute pipeline!");
		}

		vkDestroyShaderModule(device, compShaderModule, nullptr);
	}

	void createPlasmaSimulationPipeline()
	{
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		vkGetDeviceQueue(device, queueFamilyIndices.computeFamily.value(), 0, &computeQueue);

		auto compShaderCode = readFile("../src/shaders/simulation/plasma_movement.comp.spv");

		VkShaderModule compShaderModule = createShaderModule(compShaderCode);

		VkPipelineShaderStageCreateInfo compShaderStageInfo = {};
		compShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		compShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		compShaderStageInfo.module = compShaderModule;
		compShaderStageInfo.pName = "main";


		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayoutCompute;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayoutPlasmaSimulation) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create pipeline layout for compute!");
		}


		VkComputePipelineCreateInfo computeInfo = {};
		computeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		computeInfo.stage = compShaderStageInfo;
		computeInfo.layout = pipelineLayoutPlasmaSimulation;
		computeInfo.flags = 0;

		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computeInfo, nullptr, &plasmaSimulationPipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create compute pipeline!");
		}

		vkDestroyShaderModule(device, compShaderModule, nullptr);
	}

	void createComputePipeline()
	{
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		vkGetDeviceQueue(device, queueFamilyIndices.computeFamily.value(), 0, &computeQueue);

		auto compShaderCode = readFile("../src/shaders/simulation/shader.comp.spv");

		VkShaderModule compShaderModule = createShaderModule(compShaderCode);

		VkPipelineShaderStageCreateInfo compShaderStageInfo = {};
		compShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		compShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		compShaderStageInfo.module = compShaderModule;
		compShaderStageInfo.pName = "main";

		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayoutCompute;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayoutCompute) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create pipeline layout for compute!");
		}

		VkComputePipelineCreateInfo computeInfo = {};
		computeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		computeInfo.stage = compShaderStageInfo;
		computeInfo.layout = pipelineLayoutCompute;
		computeInfo.flags = 0;

		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computeInfo, nullptr, &computePipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create compute pipeline!");
		}

		VkFenceCreateInfo fenceInfo = {};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		if (vkCreateFence(device, &fenceInfo, nullptr, &computeFence) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create fence for compute!");
		}

		vkDestroyShaderModule(device, compShaderModule, nullptr);
	}

	void createSkyboxPipeline()
	{
		auto vertShaderCode = readFile("../src/shaders/simulation/skybox.vert.spv");
		auto fragShaderCode = readFile("../src/shaders/simulation/skybox.frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		auto bindingDescription = Skybox::getBindingDescription();
		auto attributeDescriptions = Skybox::getAttributeDescriptions();

		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;
		viewport.height = (float)swapChainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;

		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f; //optional
		rasterizer.depthBiasClamp = 0.0f; //optional
		rasterizer.depthBiasSlopeFactor = 0.0f; //optional

		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f;	//optional
		multisampling.pSampleMask = nullptr;	//optional
		multisampling.alphaToCoverageEnable = VK_FALSE;	//optional
		multisampling.alphaToOneEnable = VK_FALSE;	//optional

		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;	//optional
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; //optional
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;	//optional
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; //optional
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; //optional
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; //optional

		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;	//optional
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f; //optional
		colorBlending.blendConstants[1] = 0.0f; //optional
		colorBlending.blendConstants[2] = 0.0f; //optional
		colorBlending.blendConstants[3] = 0.0f; //optional

		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
		pipelineLayoutInfo.pushConstantRangeCount = 0; //optional //wirf das mal beim anderen raus, könnte richtig probleme machen du Spacko!!!!
		pipelineLayoutInfo.pPushConstantRanges = nullptr; //optional

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &skyboxPipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = nullptr; //optional
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = nullptr; //optional
		pipelineInfo.layout = skyboxPipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; //optional
		pipelineInfo.basePipelineIndex = -1; //optional

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &skyboxPipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
	}

	void createPlasmaPipeline()
	{
		auto vertShaderCode = readFile("../src/shaders/simulation/plasma.vert.spv");
		auto fragShaderCode = readFile("../src/shaders/simulation/plasma.frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		std::array<VkVertexInputBindingDescription, 2> bindingDescription = {};
		bindingDescription[0].binding = 0;
		bindingDescription[0].stride = sizeof(Vertex_Plasma);
		bindingDescription[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		bindingDescription[1].binding = 1;
		bindingDescription[1].stride = sizeof(Plasma);
		bindingDescription[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

		std::array<VkVertexInputAttributeDescription, 5 > attributeDescriptions = {};
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex_Plasma, position);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex_Plasma, normal);

		attributeDescriptions[2].binding = 1;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Plasma, pos);

		attributeDescriptions[3].binding = 1;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(Plasma, vel);

		attributeDescriptions[4].binding = 1;
		attributeDescriptions[4].location = 4;
		attributeDescriptions[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[4].offset = offsetof(Plasma, color);


		vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescription.size());
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexBindingDescriptions = bindingDescription.data();
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;
		viewport.height = (float)swapChainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;

		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 5.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f; //optional
		rasterizer.depthBiasClamp = 0.0f; //optional
		rasterizer.depthBiasSlopeFactor = 0.0f; //optional

		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f;	//optional
		multisampling.pSampleMask = nullptr;	//optional
		multisampling.alphaToCoverageEnable = VK_FALSE;	//optional
		multisampling.alphaToOneEnable = VK_FALSE;	//optional

		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;	//optional
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; //optional
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;	//optional
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; //optional
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; //optional
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; //optional

		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;	//optional
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f; //optional
		colorBlending.blendConstants[1] = 0.0f; //optional
		colorBlending.blendConstants[2] = 0.0f; //optional
		colorBlending.blendConstants[3] = 0.0f; //optional

		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
		pipelineLayoutInfo.pushConstantRangeCount = 0; //optional
		pipelineLayoutInfo.pPushConstantRanges = nullptr; //optional

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &plasmaPipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = nullptr; //optional
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = nullptr; //optional
		pipelineInfo.layout = plasmaPipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; //optional
		pipelineInfo.basePipelineIndex = -1; //optional

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &plasmaPipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
	}

	void createAtomsPipeline()
	{
		auto vertShaderCode = readFile("../src/shaders/simulation/atoms.vert.spv");
		auto fragShaderCode = readFile("../src/shaders/simulation/atoms.frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		std::array<VkVertexInputBindingDescription, 2> bindingDescription = {};
		bindingDescription[0].binding = 0;
		bindingDescription[0].stride = sizeof(Vertex_Atoms);
		bindingDescription[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		bindingDescription[1].binding = 1;
		bindingDescription[1].stride = sizeof(Atom);
		bindingDescription[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

		std::array<VkVertexInputAttributeDescription, 5 > attributeDescriptions = {};
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex_Atoms, position);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex_Atoms, normal);

		attributeDescriptions[2].binding = 1;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Atom, pos);

		attributeDescriptions[3].binding = 1;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(Atom, vel);

		attributeDescriptions[4].binding = 1;
		attributeDescriptions[4].location = 4;
		attributeDescriptions[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[4].offset = offsetof(Atom, color);


		vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescription.size());
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexBindingDescriptions = bindingDescription.data();
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;
		viewport.height = (float)swapChainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;

		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 5.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f; //optional
		rasterizer.depthBiasClamp = 0.0f; //optional
		rasterizer.depthBiasSlopeFactor = 0.0f; //optional

		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f;	//optional
		multisampling.pSampleMask = nullptr;	//optional
		multisampling.alphaToCoverageEnable = VK_FALSE;	//optional
		multisampling.alphaToOneEnable = VK_FALSE;	//optional

		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;	//optional
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; //optional
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;	//optional
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; //optional
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; //optional
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; //optional

		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;	//optional
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f; //optional
		colorBlending.blendConstants[1] = 0.0f; //optional
		colorBlending.blendConstants[2] = 0.0f; //optional
		colorBlending.blendConstants[3] = 0.0f; //optional

		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
		pipelineLayoutInfo.pushConstantRangeCount = 0; //optional
		pipelineLayoutInfo.pPushConstantRanges = nullptr; //optional

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &atomsPipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = nullptr; //optional
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = nullptr; //optional
		pipelineInfo.layout = atomsPipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; //optional
		pipelineInfo.basePipelineIndex = -1; //optional

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &atomsPipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
	}

	void createVectorRodsPipeline()
	{
		auto vertShaderCode = readFile("../src/shaders/simulation/rods.vert.spv");
		auto fragShaderCode = readFile("../src/shaders/simulation/rods.frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		std::array<VkVertexInputBindingDescription, 2> bindingDescription = {};
		bindingDescription[0].binding = 0;
		bindingDescription[0].stride = sizeof(Vertex_Vector_Rods);
		bindingDescription[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		bindingDescription[1].binding = 1;
		bindingDescription[1].stride = sizeof(Vector_Rods);
		bindingDescription[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

		std::array<VkVertexInputAttributeDescription, 4 > attributeDescriptions = {};
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex_Vector_Rods, position);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex_Vector_Rods, normal);

		attributeDescriptions[2].binding = 1;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vector_Rods, position);

		attributeDescriptions[3].binding = 1;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(Vector_Rods, velocity);


		vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescription.size());
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexBindingDescriptions = bindingDescription.data();
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;
		viewport.height = (float)swapChainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;

		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 5.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f; //optional
		rasterizer.depthBiasClamp = 0.0f; //optional
		rasterizer.depthBiasSlopeFactor = 0.0f; //optional

		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f;	//optional
		multisampling.pSampleMask = nullptr;	//optional
		multisampling.alphaToCoverageEnable = VK_FALSE;	//optional
		multisampling.alphaToOneEnable = VK_FALSE;	//optional

		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;	//optional
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; //optional
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;	//optional
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; //optional
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; //optional
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; //optional

		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;	//optional
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f; //optional
		colorBlending.blendConstants[1] = 0.0f; //optional
		colorBlending.blendConstants[2] = 0.0f; //optional
		colorBlending.blendConstants[3] = 0.0f; //optional

		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
		pipelineLayoutInfo.pushConstantRangeCount = 0; //optional
		pipelineLayoutInfo.pPushConstantRanges = nullptr; //optional

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &vectorRodsPipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = nullptr; //optional
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = nullptr; //optional
		pipelineInfo.layout = vectorRodsPipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; //optional
		pipelineInfo.basePipelineIndex = -1; //optional

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &vectorRodsPipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
	}

	void createFramebuffers()
	{
		swapChainFramebuffers.resize(swapChainImageViews.size());

		for (size_t i = 0; i < swapChainImageViews.size(); i++)
		{
			VkImageView attachments[] =
			{
				swapChainImageViews[i]
			};

			VkFramebufferCreateInfo framebufferInfo = {};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = 1;
			framebufferInfo.pAttachments = attachments;
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}

	void createCommandPoolCompute()
	{
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo poolInfoCompute = {};
		poolInfoCompute.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfoCompute.queueFamilyIndex = queueFamilyIndices.computeFamily.value();
		poolInfoCompute.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

		if (vkCreateCommandPool(device, &poolInfoCompute, nullptr, &commandPoolCompute) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create command pool for compute!");
		}
	}

	void createCommandPool()
	{
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
		poolInfo.flags = 0;

		if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create command pool!");
		}
	}

	void createTextureImage()
	{
		int texWidth, texHeight, texChannels;

		stbi_uc* pixels0 = stbi_load("../resources/skybox.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

		VkDeviceSize imageSize = texWidth * texHeight * 4;
		if (!pixels0)
		{
			throw std::runtime_error("failed to load texture image!");
		}

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels0, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);

		stbi_image_free(pixels0);

		createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
	{
		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
	}

	void createTextureImageView()
	{
		textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_UNORM);
	}

	void createTextureSampler()
	{
		VkSamplerCreateInfo samplerInfo = {};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = 16;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 0.0f;

		if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create texture sampler!");
		}
	}

	VkImageView createImageView(VkImage image, VkFormat format)
	{
		VkImageViewCreateInfo viewInfo = {};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView imageView;
		if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create texturee image view!");
		}

		return imageView;
	}

	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;

		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
		{
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else
		{
			throw std::invalid_argument("unsupported layout transition!");
		}

		vkCmdPipelineBarrier(
			commandBuffer,
			sourceStage, destinationStage,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		endSingleTimeCommands(commandBuffer);
	}

	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferImageCopy region = {};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = { 0, 0, 0 };
		region.imageExtent =
		{
			width,
			height,
			1
		};

		vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		endSingleTimeCommands(commandBuffer);
	}

	void createPlasmaVertexBuffer()
	{
		VkDeviceSize bufferSize = sizeof(plasma_vertices[0]) * plasma_vertices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, plasma_vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, plasmaVertexBuffer, plasmaVertexBufferMemory);

		copyBuffer(stagingBuffer, plasmaVertexBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createAtomsVertexBuffer()
	{
		VkDeviceSize bufferSize = sizeof(atoms_vertices[0]) * atoms_vertices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, atoms_vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, atomsVertexBuffer, atomsVertexBufferMemory);

		copyBuffer(stagingBuffer, atomsVertexBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createSkyboxVertexBuffer()
	{
		VkDeviceSize bufferSize = sizeof(sky_vertices[0]) * sky_vertices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, sky_vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, skyboxVertexBuffer, skyboxVertexBufferMemory);

		copyBuffer(stagingBuffer, skyboxVertexBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createVectorRodsVertexBuffer()
	{
		VkDeviceSize bufferSize = sizeof(rods_vertices[0]) * rods_vertices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, rods_vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vectorRodsVertexBuffer, vectorRodsVertexBufferMemory);

		copyBuffer(stagingBuffer, vectorRodsVertexBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createIndexBuffer()
	{
		VkDeviceSize bufferSize = sizeof(sky_indices[0]) * sky_indices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, sky_indices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}


	void createPlasmaBuffer()
	{
		std::vector<Plasma> plasmaBuffer(plasma_count);

		std::random_device rd; // obtain a random number from hardware
		std::mt19937 eng(rd()); // seed the generator
		std::uniform_real_distribution<> distr(-50, 50); // define the range

		for (auto& plasma : plasmaBuffer)
		{
			
			plasma.pos = glm::vec4(distr(eng), rand() % 20 + 70, distr(eng), 0.0f);
			plasma.vel = (glm::vec4(glm::vec3(0.01), 0.0f) - glm::vec4(0.5f, 0.5f, 0.5f, 0.0f));
			plasma.lifetime = 10.0f;
			plasma.max_lifetime = 0.0f;
			plasma.hit_counter = 30;
			plasma.color = glm::vec4(0.828f, 0.683f, 0.214f, 1.0f);
		}

		VkDeviceSize bufferSize = sizeof(Plasma) * plasma_count;

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, plasmaBuffer.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, storageBuffers.plasma, plasmaBufferMemory);

		copyBuffer(stagingBuffer, storageBuffers.plasma, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createAtomsBuffer()
	{
		std::vector<Atom> atomsBuffer(atoms_count);

		std::random_device rd; // obtain a random number from hardware
		std::mt19937 eng(rd()); // seed the generator
		std::uniform_real_distribution<> distr(-50, 50); // define the range

		for (auto& atom : atomsBuffer)
		{

			//atom.pos = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
			atom.vel = (glm::vec4(glm::vec3(0.01), 0.0f) - glm::vec4(0.5f, 0.5f, 0.5f, 0.0f));
			atom.color = glm::vec4(1.0f);
			atom.type = generate();
			atom.is_excited = 0;
			atom.is_dead = 0;
			atom.lifetime = 0.0f;
			//atom.max_lifetime = 0.0f;



			if (atom.type == 0)		//atomic oxygen(red)
			{
				atom.pos = glm::vec4(distr(eng), rand() % 30 + 20, distr(eng), 0.0f);
				atom.color = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
				atom.max_lifetime = 110.0f;
			}

			if (atom.type == 1)		//atomic oxygen(green)
			{
				atom.pos = glm::vec4(distr(eng), rand() % 40 + -30, distr(eng), 0.0f);
				atom.color = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
				atom.max_lifetime = 0.7f;
			}

			if (atom.type == 2)		//ionized nitrogen(blue)
			{
				atom.pos = glm::vec4(distr(eng), rand() % 40 + -30, distr(eng), 0.0f);
				atom.color = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
				atom.max_lifetime = 0.0001f;
			}
		}

		VkDeviceSize bufferSize = sizeof(Atom) * atoms_count;

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, atomsBuffer.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, storageBuffers.atoms, atomsBufferMemory);

		copyBuffer(stagingBuffer, storageBuffers.atoms, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createVectorRodsBuffer()
	{
		std::vector<Vector_Rods> rodsBuffer(grid_params.total_cell_count);

		float offset_x = grid_params.cell_size.x / 2.0f;
		float offset_y = grid_params.cell_size.y / 2.0f;
		float offset_z = grid_params.cell_size.z / 2.0f;
		int count = 0;

		for (int i = 0; i < grid_params.cell_count.x; i++)
		{
			for (int j = 0; j < grid_params.cell_count.y; j++)
			{
				for (int k = 0; k < grid_params.cell_count.z; k++)
				{
					if (count < grid_params.total_cell_count)
					{
						rodsBuffer[count].position = glm::vec4((grid_params.grid_min.x + offset_x) + (i * grid_params.cell_size.x),
							(grid_params.grid_min.y + offset_y) + (j * grid_params.cell_size.y),
							(grid_params.grid_min.z + offset_z) + (k * grid_params.cell_size.z), 0.0f);

						count++;
					}
					else
					{
						break;
					}
				}
			}
		}

		/*rodsBuffer[0].position = glm::vec4(-450.0f, 0.0f, 0.0f, 0.0f);
		rodsBuffer[1].position = glm::vec4(-450.0f, 0.0f, 0.0f, 0.0f);
		rodsBuffer[2].position = glm::vec4(-450.0f, 0.0f, 0.0f, 0.0f);
		rodsBuffer[3].position = glm::vec4(-450.0f, 0.0f, 0.0f, 0.0f);
		rodsBuffer[4].position = glm::vec4(-450.0f, 0.0f, 0.0f, 0.0f);

		rodsBuffer[5].position = glm::vec4(450.0f, 0.0f, 0.0f, 0.0f);
		rodsBuffer[6].position = glm::vec4(450.0f, 0.0f, 0.0f, 0.0f);
		rodsBuffer[7].position = glm::vec4(450.0f, 0.0f, 0.0f, 0.0f);
		rodsBuffer[8].position = glm::vec4(450.0f, 0.0f, 0.0f, 0.0f);
		rodsBuffer[9].position = glm::vec4(450.0f, 0.0f, 0.0f, 0.0f);

		rodsBuffer[10].position = glm::vec4(0.0f, 0.0f, -450.0f, 0.0f);
		rodsBuffer[11].position = glm::vec4(0.0f, 0.0f, -450.0f, 0.0f);
		rodsBuffer[12].position = glm::vec4(0.0f, 0.0f, -450.0f, 0.0f);
		rodsBuffer[13].position = glm::vec4(0.0f, 0.0f, -450.0f, 0.0f);
		rodsBuffer[14].position = glm::vec4(0.0f, 0.0f, -450.0f, 0.0f);

		rodsBuffer[15].position = glm::vec4(0.0f, 0.0f, 450.0f, 0.0f);
		rodsBuffer[16].position = glm::vec4(0.0f, 0.0f, 450.0f, 0.0f);
		rodsBuffer[17].position = glm::vec4(0.0f, 0.0f, 450.0f, 0.0f);
		rodsBuffer[18].position = glm::vec4(0.0f, 0.0f, 450.0f, 0.0f);
		rodsBuffer[19].position = glm::vec4(0.0f, 0.0f, 450.0f, 0.0f);

		for (int i = 20; i < grid_params.total_cell_count; i++)
		{
			rodsBuffer[i].position = glm::vec4(0.0f);
		}*/

		for (auto& rods : rodsBuffer)
		{
			rods.velocity = glm::vec4(0.0f);
			//rods.position = glm::vec4(0.0f);
		}


		VkDeviceSize bufferSize = sizeof(Vector_Rods) * grid_params.total_cell_count;

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, rodsBuffer.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, storageBuffers.vector_rods, vectorRodsBufferMemory);

		copyBuffer(stagingBuffer, storageBuffers.vector_rods, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createGridBuffer()
	{
		VkDeviceSize bufferSize = sizeof(grid_params);

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, &grid_params, (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, storageBuffers.grid_params, gridParamsBufferMemory);

		copyBuffer(stagingBuffer, storageBuffers.grid_params, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);


		std::vector<int> gridBuffer(grid_params.grid_capacity * sizeof(int));
		for (auto& ssbo : gridBuffer)
		{
			ssbo = -1;
		}

		VkDeviceSize gridBufferSize = grid_params.grid_capacity * sizeof(int);

		createBuffer(gridBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* grid_data;
		vkMapMemory(device, stagingBufferMemory, 0, gridBufferSize, 0, &grid_data);
		memcpy(grid_data, gridBuffer.data(), (size_t)gridBufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(grid_params.grid_capacity * sizeof(int), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, storageBuffers.grid_data, gridDataBufferMemory);

		copyBuffer(stagingBuffer, storageBuffers.grid_data, gridBufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);

		createBuffer(grid_params.total_cell_count * sizeof(int), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, storageBuffers.grid_cell_counter, gridCounterBufferMemory);

		VkDeviceSize neighborhood_data_size = neighbor_amount * atoms_count * sizeof(int);
		createBuffer(neighborhood_data_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, storageBuffers.grid_neighborhood_data, gridNeighborhoodDataBufferMemory);
		createBuffer(atoms_count * sizeof(int), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, storageBuffers.grid_neighborhood_counter, gridNeighborhoodCounterBufferMemory);

	}

	void createVectorBuffer()
	{
		std::vector<VectorField> vectorBuffer(grid_params.total_cell_count);

		float offset_x = grid_params.cell_size.x / 2.0f;
		float offset_y = grid_params.cell_size.y / 2.0f;
		float offset_z = grid_params.cell_size.z / 2.0f;
		int count = 0;

		for (int i = 0; i < grid_params.cell_count.x; i++)
		{
			for (int j = 0; j < grid_params.cell_count.y; j++)
			{
				for (int k = 0; k < grid_params.cell_count.z; k++)
				{
					if (count < grid_params.total_cell_count)
					{
						vectorBuffer[count].pos = glm::vec4((grid_params.grid_min.x + offset_x) + (i * grid_params.cell_size.x),
							(grid_params.grid_min.y + offset_y) + (j * grid_params.cell_size.y),
							(grid_params.grid_min.z + offset_z) + (k * grid_params.cell_size.z), 0.0f);

						count++;
					}
					else
					{
						break;
					}
				}
			}
		}

		VkDeviceSize bufferSize = sizeof(VectorField) * grid_params.total_cell_count;

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, vectorBuffer.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, storageBuffers.vector_field, vectorFieldBufferMemory);

		copyBuffer(stagingBuffer, storageBuffers.vector_field, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);

	}


	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
	{
		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate buffer memory!");
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}

	VkCommandBuffer beginSingleTimeCommands()
	{
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	void endSingleTimeCommands(VkCommandBuffer commandBuffer)
	{
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);

		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion = {};
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer);
	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
		{
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	void createUniformBuffers()
	{
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);

		uniformBuffers.resize(swapChainImages.size());
		uniformBuffersMemory.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++)
		{
			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
		}

		VkDeviceSize bufferSizeCompute = sizeof(UBO);

		createBuffer(bufferSizeCompute, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBufferCompute, uniformBufferMemoryCompute);

		VkDeviceSize bufferSizeVariables = sizeof(Uniform_Variables);

		createBuffer(bufferSizeVariables, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformVariablesBuffer, uniformVariablesBufferMemory);
	}

	void createDescriptorPool()
	{
		std::array<VkDescriptorPoolSize, 2> poolSizes = {};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(swapChainImages.size());


		VkDescriptorPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();;
		poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	void createDescriptorSets()
	{
		std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
		allocInfo.pSetLayouts = layouts.data();

		descriptorSets.resize(swapChainImages.size());
		if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < swapChainImages.size(); i++)
		{
			VkDescriptorBufferInfo bufferInfo = {};
			bufferInfo.buffer = uniformBuffers[i];
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(UniformBufferObject);

			VkDescriptorImageInfo imageInfo = {};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfo.imageView = textureImageView;
			imageInfo.sampler = textureSampler;

			std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = descriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &bufferInfo;

			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = descriptorSets[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pImageInfo = &imageInfo;
			//descriptorWrite.pImageInfo = nullptr; // Optional
			//descriptorWrite.pTexelBufferView = nullptr; // Optional

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}
	}

	void createCommandBufferCompute()
	{
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPoolCompute;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = 2;

		if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffersCompute[0]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocat command buffers!");
		}

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		for (size_t i = 0; i < 2; i++)
		{
			if (vkBeginCommandBuffer(commandBuffersCompute[i], &beginInfo) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to begin recording command buffer!");
			}

			VkBufferMemoryBarrier bufferBarrier = {};
			bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
			//bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			//bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			bufferBarrier.srcAccessMask = 0;
			bufferBarrier.dstAccessMask = 0;
			bufferBarrier.srcQueueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
			bufferBarrier.dstQueueFamilyIndex = queueFamilyIndices.computeFamily.value();
			bufferBarrier.size = VK_WHOLE_SIZE;

			std::vector<VkBufferMemoryBarrier> bufferBarriers;
			bufferBarrier.buffer = storageBuffers.plasma;
			bufferBarriers.push_back(bufferBarrier);
			bufferBarrier.buffer = storageBuffers.atoms;
			bufferBarriers.push_back(bufferBarrier);
			bufferBarrier.buffer = storageBuffers.grid_params;
			bufferBarriers.push_back(bufferBarrier);
			bufferBarrier.buffer = storageBuffers.grid_data;
			bufferBarriers.push_back(bufferBarrier);
			bufferBarrier.buffer = storageBuffers.grid_cell_counter;
			bufferBarriers.push_back(bufferBarrier);
			bufferBarrier.buffer = storageBuffers.vector_field;
			bufferBarriers.push_back(bufferBarrier);
			bufferBarrier.buffer = storageBuffers.grid_neighborhood_data;
			bufferBarriers.push_back(bufferBarrier);
			bufferBarrier.buffer = storageBuffers.grid_neighborhood_counter;
			bufferBarriers.push_back(bufferBarrier);
			bufferBarrier.buffer = storageBuffers.vector_rods;
			bufferBarriers.push_back(bufferBarrier);

			vkCmdPipelineBarrier(commandBuffersCompute[i], VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(), 0, nullptr);

			vkCmdBindDescriptorSets(commandBuffersCompute[i], VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayoutCompute, 0, 1, &descriptorSetsCompute[readSet], 0, 0); //must be binded only once as all pipelines use the same descriptor set
			std::cout << "Current Frame: " << currentFrame << std::endl;

			std::cout << "Read Set: " << readSet << std::endl;
			readSet = 1 - readSet;

			//bind compute pipeline for unifor grid reset
			vkCmdBindPipeline(commandBuffersCompute[i], VK_PIPELINE_BIND_POINT_COMPUTE, gridResetPipeline);
			vkCmdDispatch(commandBuffersCompute[i], total_particle_count / 128 + 1, 1, 1);

			//bind compute pipeline for unifor grid setup
			vkCmdBindPipeline(commandBuffersCompute[i], VK_PIPELINE_BIND_POINT_COMPUTE, gridSetupPipeline);
			vkCmdDispatch(commandBuffersCompute[i], total_particle_count / 128 + 1, 1, 1);

			//bind compute pipeline for unifor grid find_neighbors
			vkCmdBindPipeline(commandBuffersCompute[i], VK_PIPELINE_BIND_POINT_COMPUTE, gridFindNeighborPipeline);
			vkCmdDispatch(commandBuffersCompute[i], total_particle_count / 128 + 1, 1, 1);

			//bind compute pipeline for vector field initialization
			vkCmdBindPipeline(commandBuffersCompute[i], VK_PIPELINE_BIND_POINT_COMPUTE, vectorFieldPipeline);
			vkCmdDispatch(commandBuffersCompute[i], grid_params.total_cell_count / 128 + 1, 1, 1);

			//bind compute pipeline for plasma simulation
			vkCmdBindPipeline(commandBuffersCompute[i], VK_PIPELINE_BIND_POINT_COMPUTE, plasmaSimulationPipeline);
			vkCmdDispatch(commandBuffersCompute[i], total_particle_count / 128 + 1, 1, 1);

			//bind compute pipeline for simulation
			vkCmdBindPipeline(commandBuffersCompute[i], VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
			vkCmdDispatch(commandBuffersCompute[i], total_particle_count / 128 + 1, 1, 1);


			for (auto& barrier : bufferBarriers) //brauch ich das oder war das in dem Beispiel wegen dem iteration krams??????
			{
				barrier.srcAccessMask = 0;
				barrier.dstAccessMask = 0;
				barrier.srcQueueFamilyIndex = queueFamilyIndices.computeFamily.value();
				barrier.dstQueueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
			}

			vkCmdPipelineBarrier(commandBuffersCompute[i], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(), 0, nullptr);

			for (auto& barrier : bufferBarriers)
			{
				barrier.srcAccessMask = 0;
				barrier.dstAccessMask = 0;
				barrier.srcQueueFamilyIndex = queueFamilyIndices.computeFamily.value();
				barrier.dstQueueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
			}

			vkCmdPipelineBarrier(commandBuffersCompute[i], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(), 0, nullptr);

			vkEndCommandBuffer(commandBuffersCompute[i]);
		}
	}

	void createCommandBuffers()
	{
		commandBuffers.resize(swapChainFramebuffers.size());

		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate command buffers!");
		}

		for (size_t i = 0; i < commandBuffers.size(); i++)
		{
			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
			beginInfo.pInheritanceInfo = nullptr; //optional

			if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to begin recording command buffer!");
			}

			VkRenderPassBeginInfo renderPassInfo = {};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = renderPass;
			renderPassInfo.framebuffer = swapChainFramebuffers[i];
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent = swapChainExtent;

			VkClearValue clearColor = { 0.5f, 0.5f, 0.5f, 1.0f };
			renderPassInfo.clearValueCount = 1;
			renderPassInfo.pClearValues = &clearColor;

			vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			//pipeline for skybox rendering
			vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipeline);

			VkBuffer skyboxVertexBuffers[] = { skyboxVertexBuffer };
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, skyboxVertexBuffers, offsets);

			vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT16);

			vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

			vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(sky_indices.size()), 1, 0, 0, 0);

			//pipeline for the plasma particles
			vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, plasmaPipeline);

			VkBuffer plasmaVertexBuffers[] = { plasmaVertexBuffer };
			VkBuffer plasma_ssbos[] = { storageBuffers.plasma };
			//VkDeviceSize offsets[] = { 0 };

			vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, plasmaVertexBuffers, offsets);
			vkCmdBindVertexBuffers(commandBuffers[i], 1, 1, plasma_ssbos, offsets);

			vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, plasmaPipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

			vkCmdDraw(commandBuffers[i], static_cast<uint32_t>(plasma_vertices.size()), plasma_count, 0, 0);

			//pipeline for the atoms particles
			vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, atomsPipeline);

			VkBuffer atomVertexBuffers[] = { atomsVertexBuffer };
			VkBuffer atom_ssbos[] = { storageBuffers.atoms };
			//VkDeviceSize offsets[] = { 0 };

			vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, atomVertexBuffers, offsets);
			vkCmdBindVertexBuffers(commandBuffers[i], 1, 1, atom_ssbos, offsets);

			vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, atomsPipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

			vkCmdDraw(commandBuffers[i], static_cast<uint32_t>(atoms_vertices.size()), atoms_count, 0, 0);

			//pipeline for vector rods
			vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, vectorRodsPipeline);

			VkBuffer rodsVertexBuffers[] = { vectorRodsVertexBuffer };
			VkBuffer rods_ssbos[] = { storageBuffers.vector_rods };

			vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, rodsVertexBuffers, offsets);
			vkCmdBindVertexBuffers(commandBuffers[i], 1, 1, rods_ssbos, offsets);

			vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, vectorRodsPipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

			vkCmdDraw(commandBuffers[i], static_cast<uint32_t>(rods_vertices.size()), grid_params.total_cell_count, 0, 0);

			vkCmdEndRenderPass(commandBuffers[i]);

			if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to record command buffer!");
			}
		}
	}

	void createSyncObjects()
	{
		imageAvailableSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo = {};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo = {};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore[i]) != VK_SUCCESS || vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore[i]) != VK_SUCCESS || vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create synchronization objects for a frame!");
			}
		}

	}

	void updateComputeUBO()
	{
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		//float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		float time = glfwGetTime();

		UBO computeUBO = {};
		computeUBO.goal = glm::vec3(sinf(time * 0.34f), cosf(time * 0.29f), sinf(time * 0.12f) * cosf(time * 0.5f)) * glm::vec3(35.0f, 25.0f, 60.0f);

		void* data;
		vkMapMemory(device, uniformBufferMemoryCompute, 0, sizeof(computeUBO), 0, &data);
		memcpy(data, &computeUBO, sizeof(computeUBO));
		vkUnmapMemory(device, uniformBufferMemoryCompute);
	}

	void updateUniformVariables()
	{
		
		float t = glfwGetTime();
		float dt = (t - old_t) / 1000.0;
		old_t = t;

		Uniform_Variables variables = {};
		variables.delta_time = dt;
		variables.theta = t;

		void* data;
		vkMapMemory(device, uniformVariablesBufferMemory, 0, sizeof(variables), 0, &data);
		memcpy(data, &variables, sizeof(variables));
		vkUnmapMemory(device, uniformVariablesBufferMemory);
	}

	void updateUniformBuffer(uint32_t currentImage)
	{
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();


		UniformBufferObject ubo = {};
		ubo.model = glm::mat4(1.0f);
		//ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		//ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
		ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 300.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 3000.0f);

		ubo.view = camera.view();
		ubo.proj = camera.projection();
		ubo.proj[1][1] *= -1;

		void* data;
		vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
		memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
	}

	void drawFrame()
	{
		vkWaitForFences(device, 1, &computeFence, VK_TRUE, UINT64_MAX);
		vkResetFences(device, 1, &computeFence);

		updateComputeUBO();
		updateUniformVariables();

		VkSubmitInfo computeSubmitInfo = {};
		computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		computeSubmitInfo.commandBufferCount = 1;
		computeSubmitInfo.pCommandBuffers = &commandBuffersCompute[currentFrame ^ 1];

		if (vkQueueSubmit(computeQueue, 1, &computeSubmitInfo, computeFence) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to submit compute command buffer!");
		}

		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());
		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphore[currentFrame], VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
		{
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		float t = glfwGetTime();

		camera.update(window);
		updateUniformBuffer(imageIndex);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphore[currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphore[currentFrame] };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(presentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
		{
			framebufferResized = false;
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("failed to present swap chain image!");
		}

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	VkShaderModule createShaderModule(const std::vector<char>& code)
	{
		VkShaderModuleCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create shader module!");
		}

		return shaderModule;
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
	{
		if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED)
		{
			return{ VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
		}

		for (const auto& availableFormat : availableFormats)
		{
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			{
				return availableFormat;
			}
		}

		return availableFormats[0];
	}


	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes)
	{
		VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

		for (const auto& availablePresentMode : availablePresentModes)
		{
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				return availablePresentMode;
			}
			else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
			{
				bestMode = availablePresentMode;
			}
		}

		return bestMode;
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
	{
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		{
			return capabilities.currentExtent;
		}
		else
		{
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actualExtent =
			{
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			return actualExtent;
		}
	}

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
	{
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

		if (formatCount != 0)
		{
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

		if (presentModeCount != 0)
		{
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	bool isDeviceSuitable(VkPhysicalDevice device)
	{
		QueueFamilyIndices indices = findQueueFamilies(device);

		bool extensionsSupported = checkDeviceExtensionSupport(device);

		bool swapChainAdequate = false;
		if (extensionsSupported)
		{
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		return indices.isComplete() && extensionsSupported && swapChainAdequate;
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device)
	{
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions)
		{
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
	{
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies)
		{
			if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)
			{
				indices.computeFamily = i;
			}

			if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				indices.graphicsFamily = i;
			}

			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

			if (queueFamily.queueCount > 0 && presentSupport)
			{
				indices.presentFamily = i;
			}

			if (indices.isComplete())
			{
				break;
			}

			i++;
		}

		return indices;
	}

	std::vector<const char*> getRequiredExtensions()
	{
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enableValidationLayers)
		{
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	bool checkValidationLayerSupport()
	{
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers)
		{
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers)
			{
				if (strcmp(layerName, layerProperties.layerName) == 0)
				{
					layerFound = true;
					break;
				}
			}

			if (!layerFound)
			{
				return false;
			}
		}

		return true;
	}

	static std::vector<char> readFile(const std::string& filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open())
		{
			throw std::runtime_error("failed to open file!");
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);
		file.close();

		return buffer;
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
	{
		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}

	inline float random_float()
	{
		float res;
		unsigned int tmp;
		static unsigned int seed = 0xFFFF0C59;

		seed *= 16807;

		tmp = seed ^ (seed >> 4) ^ (seed << 15);

		*((unsigned int*)& res) = (tmp >> 9) | 0x3F800000;

		return (res - 1.0f);
	}

	/*float random(int min, int max)
	{
		srand(time(NULL));
		float random = static_cast<float>(rand() % max + min);

		return random;
	}*/

	glm::vec3 random_vec3(float minmag = 0.0f, float maxmag = 1.0f)
	{
		glm::vec3 randomvec(random_float() * 2.0f - 1.0f, random_float() * 2.0f - 1.0f, random_float() * 2.0f - 1.0f);
		randomvec = normalize(randomvec);
		randomvec *= (random_float() * (maxmag - minmag) + minmag);

		return randomvec;
	}

	inline VkDescriptorSetLayoutBinding descriptorSetLayoutBinding(VkDescriptorType type, VkShaderStageFlags stageFlags, uint32_t binding, uint32_t descriptorCount = 1)
	{
		VkDescriptorSetLayoutBinding setLayoutBinding{};
		setLayoutBinding.descriptorType = type;
		setLayoutBinding.stageFlags = stageFlags;
		setLayoutBinding.binding = binding;
		setLayoutBinding.descriptorCount = descriptorCount;
		return setLayoutBinding;
	}

	inline VkWriteDescriptorSet writeDescriptorSet(VkDescriptorSet dstSet, VkDescriptorType type, uint32_t binding, VkDescriptorBufferInfo* bufferInfo, uint32_t descriptorCount = 1)
	{
		VkWriteDescriptorSet writeDescriptorSet{};
		writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSet.dstSet = dstSet;
		writeDescriptorSet.descriptorType = type;
		writeDescriptorSet.dstBinding = binding;
		writeDescriptorSet.pBufferInfo = bufferInfo;
		writeDescriptorSet.descriptorCount = descriptorCount;
		return writeDescriptorSet;
	}

	inline VkDescriptorPoolSize descriptorPoolSize(VkDescriptorType type, uint32_t descriptorCount)
	{
		VkDescriptorPoolSize descriptorPoolSize{};
		descriptorPoolSize.type = type;
		descriptorPoolSize.descriptorCount = descriptorCount;
		return descriptorPoolSize;
	}

	inline VkDescriptorPoolCreateInfo descriptorPoolCreateInfo(const std::vector<VkDescriptorPoolSize>& poolSizes, uint32_t maxSets)
	{
		VkDescriptorPoolCreateInfo descriptorPoolInfo{};
		descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		descriptorPoolInfo.pPoolSizes = poolSizes.data();
		descriptorPoolInfo.maxSets = maxSets;
		return descriptorPoolInfo;
	}

	int random()
	{
		int random = rand();

		return (random % 2);
	}

	int generate()
	{
		int x = random();
		int y = random();

		return (x + y);
	}
};

int main()
{
	BoidApplication app;

	try
	{
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}