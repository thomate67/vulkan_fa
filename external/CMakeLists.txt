#external
file(GLOB_RECURSE SOURCES "${BASE_EXTERNAL_DIR}/*.cpp" "${BASE_EXTERNAL_DIR}/*.c")

find_package(assimp REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

list(APPEND BASE_INCLUDE_DIRECTORIES ${BASE_EXTERNAL_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG  "${BASE_EXTERNAL_DIR}/Debug")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE  "${BASE_EXTERNAL_DIR}/Release")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${BASE_EXTERNAL_DIR}/RelWithDebInfo")


add_library(ext_lib STATIC ${SOURCES})
target_link_libraries(ext_lib PUBLIC glfw glm Vulkan::Vulkan ${ASSIMP_LIBRARIES})
target_include_directories(ext_lib PUBLIC ${BASE_INCLUDE_DIRECTORIES})

include_directories(${BASE_INCLUDE_DIRECTORIES})