cmake_minimum_required(VERSION 3.23)
project(lab_4 CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(lab_4 main.cu plane.cuh plane.cu MonteCarlo.cu MonteCarlo.cuh DeviceInfo.cuh DeviceInfo.cu)

set_target_properties(lab_4 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
