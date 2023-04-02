#include "DeviceInfo.cuh"


void DeviceInfo::printCudaDeviceInfo(cudaDeviceProp &prop) {
    printf("Device name:                                        %s\n", prop.name);
    printf("Global memory available on device:                  %zu\n", prop.totalGlobalMem);
    printf("Shared memory available per block:                  %zu\n", prop.sharedMemPerBlock);
    printf("Count of 32-bit registers available per block:      %i\n", prop.regsPerBlock);
    printf("Warp size in threads:                               %i\n", prop.warpSize);
    printf("Maximum pitch in bytes allowed by memory copies:    %zu\n", prop.memPitch);
    printf("Maximum number of threads per block:                %i\n", prop.maxThreadsPerBlock);
    printf("Maximum size of each dimension of a block[0]:       %i\n", prop.maxThreadsDim[0]);
    printf("Maximum size of each dimension of a block[1]:       %i\n", prop.maxThreadsDim[1]);
    printf("Maximum size of each dimension of a block[2]:       %i\n", prop.maxThreadsDim[2]);
    printf("Maximum size of each dimension of a grid[0]:        %i\n", prop.maxGridSize[0]);
    printf("Maximum size of each dimension of a grid[1]:        %i\n", prop.maxGridSize[1]);
    printf("Maximum size of each dimension of a grid[2]:        %i\n", prop.maxGridSize[2]);
    printf("Clock frequency in kilohertz:                       %i\n", prop.clockRate);
    printf("totalConstMem:                                      %zu\n", prop.totalConstMem);
    printf("Major compute capability:                           %i\n", prop.major);
    printf("Minor compute capability:                           %i\n", prop.minor);
    printf("Number of multiprocessors on device:                %i\n", prop.multiProcessorCount);
    printf("---------------------------------------------------------------\n");
}


void DeviceInfo::getCudaDeviceInfo() {
    int count;
    cudaDeviceProp prop{};

    cudaGetDeviceCount(&count);
    printf("Count CUDA devices = %i\n", count);

    for (int i = 0; i < count; i++) {
        cudaGetDeviceProperties(&prop, i);
        printCudaDeviceInfo(prop);
    }
}

