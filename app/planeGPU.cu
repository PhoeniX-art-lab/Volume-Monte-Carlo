#include "plane.cuh"


__device__ bool rayIntersectsPlaneGPU(const Plane plane, const Point point) {
    Point ray_direction = {1.0, 0.0, 0.0};
    double denominator = plane.a * ray_direction.x + plane.b * ray_direction.y + plane.c * ray_direction.z;
    if (std::abs(denominator) < 1e-6)
        return false;
    double numerator = -(plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d);

    return (numerator / denominator) >= 0;
}


__device__ bool isPointInsidePolyhedronGPU(const Plane* planes, Point point, int num_planes) {
    int cnt = 0;
    for (int i = 0; i < num_planes; i++) {
        const Plane &plane = planes[i];
        if (rayIntersectsPlaneGPU(plane, point))
            cnt += 1;
    }
    return cnt % 2 == 1;
}



__global__ void countInsidePointsKernel(int n_iter, const Plane *planes, int num_planes, int* N) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto stride = blockDim.x * gridDim.x;

    int cnt = 0;
    curandState state;
    curand_init(clock64(), idx, 0, &state);
    for (auto i = idx; i < n_iter; i += stride) {
        Point point = {
                2.0 * curand_uniform(&state) - 1.0,
                2.0 * curand_uniform(&state) - 1.0,
                2.0 * curand_uniform(&state) - 1.0
        };
        if (isPointInsidePolyhedronGPU(planes, point, num_planes))
            cnt += 1;
    }

    // Perform hierarchical reduction within the block using shared memory
    __shared__ int sdata[1024];
    auto tid = threadIdx.x;
    sdata[tid] = cnt;
    __syncthreads();

    for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Add the total count to the global sum using atomicAdd
    if (tid == 0) {
        atomicAdd(N, sdata[0]);
    }
}