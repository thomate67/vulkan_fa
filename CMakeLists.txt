#Main
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(BASE VERSION 0.1.0.0 LANGUAGES CXX)

set(BASE_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(BASE_BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(BASE_EXTERNAL_DIR ${PROJECT_SOURCE_DIR}/external)
set(BASE_INTERNAL_DIR ${PROJECT_SOURCE_DIR}/src/libraries)

set(BASE_RESOURCES_PATH	    ${PROJECT_SOURCE_DIR}/resources)
set(BASE_SHADERS_PATH       ${PROJECT_SOURCE_DIR}/src/shaders)

if(MSVC)
	set(compile_options -MP - openmp -permissive- -Zc:twoPhase-)
else()
	set(compile_options -Wnon-template-friend -Wclass-memaccess -Wtautological-compare -Wconversion-null)
endif()

list(APPEND BASE_INTERNAL_DIRECTORIES "${BASE_INTERNAL_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  "${BASE_BIN_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "${BASE_BIN_DIR}")

add_subdirectory(src)
add_subdirectory(external)

install(DIRECTORY "resources" DESTINATION .)
install(DIRECTORY "shaders" DESTINATION .)