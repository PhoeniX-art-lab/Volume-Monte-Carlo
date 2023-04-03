cmake_minimum_required(VERSION 3.23)
project(lab_4 CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(lab_4 main.cu app/plane.cuh app/plane.cu app/MonteCarlo.cu app/MonteCarlo.cuh app/DeviceInfo.cuh app/DeviceInfo.cu app/planeGPU.cu app/planeGPU.cuh)

set_target_properties(lab_4 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
