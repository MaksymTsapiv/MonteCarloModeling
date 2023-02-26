cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME model)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(${PROJECT_NAME} LANGUAGES CUDA)

enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)

##########################################################
# User configurable options of the template
##########################################################
# Note: symbols like WARNINGS_AS_ERRORS in configuration are intentionally variables
# and not CMake options --using options creates too much problem for students.

#  but sometimes it creates too much trouble, so default is OFF.
set(PROJECT_SOURCES main.cu src/grid.cu src/particle.cu src/parse_config.cu
													src/array.cu src/utils.cu src/rot_methods.cu src/patch.cu)

set(PROJECT_HEADERS include/grid.cuh include/particle.cuh include/time_measurement.cuh
			include/parse_config.cuh include/d3.cuh include/array.cuh include/utils.cuh
			include/quat.cuh include/rot_methods.cuh include/patch.cuh)



if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif ()

# Warnings as errors should be imported here -- do not move this line
#include(cmake/CompilerWarnings.cmake)

##########################################################
# Project files, packages, libraries and so on
##########################################################

#! Project main executable source compilation
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC include)

#! Add external packages
#find_package(Threads REQUIRED)
#target_include_directories(${PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIR})

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries (${PROJECT_NAME} Eigen3::Eigen)

set_target_properties( ${PROJECT_NAME}
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
		CUDA_SEPARABLE_COMPILATION ON
		#		CUDA_ARCHITECTURES OFF
		)

INSTALL(PROGRAMS
		$<TARGET_FILE:${PROJECT_NAME}>	# ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}
		DESTINATION bin)

if (I_USE_ARCH_BTW)
	target_include_directories(${PROJECT_NAME} PUBLIC /usr/lib/gcc/x86_64-pc-linux-gnu/11.3.0/include/c++)
endif ()

# Define ALL_TARGETS variable to use in PVS and Sanitizers
#set(ALL_TARGETS ${PROJECT_NAME})
