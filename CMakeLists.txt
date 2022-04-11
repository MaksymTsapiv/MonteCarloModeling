cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME model)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)

##########################################################
# User configurable options of the template
##########################################################
# Note: symbols like WARNINGS_AS_ERRORS in configuration are intentionally variables
# and not CMake options --using options creates too much problem for students.

#  but sometimes it creates too much trouble, so default is OFF.
set(PROJECT_SOURCES main.cu grid.cu particle.cu parse_config.cu cell.cu)
set(PROJECT_HEADERS grid.cuh particle.cuh time_measurement.cuh parse_config.cuh cell.cuh d3.cuh)



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

#! Add external packages
#find_package(Threads REQUIRED)
#target_include_directories(${PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIR})

set_target_properties( ${PROJECT_NAME}
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
		CUDA_SEPARABLE_COMPILATION ON
		#		CUDA_ARCHITECTURES OFF
		)

INSTALL(PROGRAMS
		$<TARGET_FILE:${PROJECT_NAME}>	# ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}
		DESTINATION bin)

# Define ALL_TARGETS variable to use in PVS and Sanitizers
#set(ALL_TARGETS ${PROJECT_NAME})
