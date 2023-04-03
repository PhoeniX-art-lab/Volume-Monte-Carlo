#include <iostream>
#include <random>
#include <chrono>
#include "app/MonteCarlo.cuh"
#include "app/DeviceInfo.cuh"


int main() {
    const long int n_iter = 10000000;
    DeviceInfo::getCudaDeviceInfo();

    Point points[] = {
            {-1, -1, -1},   // A
            {-1, 1, -1},    // B
            {1, -1, -1},    // C
            {1, 1, 1},      // E
            {-1, 1, 1},     // F
            {1, -1, 1}      // G
    };
    std::vector<Plane> planes = {
            getPlane(points[0], points[1], points[2]),  // ABC
            getPlane(points[0], points[1], points[4]),  // ABF
            getPlane(points[0], points[2], points[5]),  // ACG
            getPlane(points[4], points[3], points[5]),  // FEG
            getPlane(points[3], points[5], points[2]),  // EGC
            getPlane(points[3], points[4], points[1]),  // EFB
    };

    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_res = calculateVolumeCPU(planes, n_iter);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Time: " << ((std::chrono::duration<double>) (t2 - t1)).count() << std::endl;
    std::cout << "CPU Result: " << cpu_res << std::endl;

    double gpu_res = calculateVolumeGPU(planes, n_iter);
    std::cout << "GPU Result: " << gpu_res << std::endl;

    return 0;
}
