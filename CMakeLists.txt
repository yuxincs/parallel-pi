cmake_minimum_required(VERSION 2.8)
project(PI-Calculation)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
message("All executables will be saved to ${PROJECT_BINARY_DIR}/bin")

add_subdirectory(OpenMP)
add_subdirectory(PThread)
