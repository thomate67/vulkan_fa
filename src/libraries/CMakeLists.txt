#libraries
find_package(assimp REQUIRED)
find_package(glm  REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

file(GLOB_RECURSE SOURCES "${BASE_INTERNAL_DIR}/*.cpp" "${BASE_INTERNAL_DIR}/*.c")

list(APPEND BASE_INCLUDE_DIRECTORIES ${BASE_INTERNAL_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG  "${BASE_INTERNAL_DIR}/Debug")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE  "${BASE_INTERNAL_DIR}/Release")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${BASE_INTERNAL_DIR}/RelWithDebInfo")

add_library(int_lib STATIC ${SOURCES})
target_link_libraries(int_lib PRIVATE glfw glm Vulkan::Vulkan ${ASSIMP_LIBRARIES})
target_include_directories(int_lib PUBLIC ${BASE_INCLUDE_DIRECTORIES})

include_directories(${BASE_INCLUDE_DIRECTORIES})