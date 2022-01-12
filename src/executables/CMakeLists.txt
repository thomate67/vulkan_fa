#executables
#find_package(assimp REQUIRED)
#find_package(glm 0.9.8 REQUIRED)
#find_package(glfw3 3.2.1 REQUIRED)
#find_package(Vulkan REQUIRED)

function(make_executable exe_name)
	file(GLOB_RECURSE EXE ${exe_name}/*.cpp)

	if(NOT TARGET ${exe_name})
		add_executable(${exe_name} ${EXE})
	endif()

	#target_compile_features(${exe_name} PRIVATE cxx_std_17)
	target_link_libraries(${exe_name} PRIVATE ext_lib)
	target_link_libraries(${exe_name} PRIVATE int_lib)
	#target_link_libraries(${exe_name} PUBLIC glfw Vulkan::Vulkan ${ASSIMP_LIBRARIES} glm)
	target_include_directories(${exe_name} PUBLIC ${BASE_EXTERNAL_DIR})
	target_include_directories(${exe_name} PUBLIC ${BASE_INTERNAL_DIR})

	set_target_properties(${exe_name} PROPERTIES
										LINKER_LANGUAGE CXX
										RUNTIME_OUTPUT_DIRECTORY "${BASE_BIN_DIR}"
										CXX_STANDARD 17
										CXX_STANDARD_REQUIRED ON)
endfunction(make_executable)

make_executable(01_boids)
make_executable(02_boids_points)
make_executable(03_ba_in_vulkan)
make_executable(04_imgui)
make_executable(05_new_particle)
make_executable(06_simulation)
make_executable(07_vector_fields)