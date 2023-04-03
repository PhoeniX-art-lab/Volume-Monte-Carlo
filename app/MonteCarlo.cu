#include "MonteCarlo.cuh"


double calculateVolumeCPU(std::vector<Plane> &planes, int n_iter) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distr(-1, 1);
    int N = 0;

    for (int i = 0; i < n_iter; i++) {
        Point point = {distr(gen), distr(gen), distr(gen)};

        if (isPointInsidePolyhedron(planes, point))
            N++;
    }

    double V_cube = 8.0;
    return V_cube * N / (double) n_iter;
}


double calculateVolumeGPU(std::vector<Plane> &planes, int n_iter) {
    Plane *dev_planes = nullptr;
    cudaMalloc(&dev_planes, sizeof(Plane) * planes.size());
    cudaMemcpy(dev_planes, planes.data(), sizeof(Plane) * planes.size(), cudaMemcpyHostToDevice);

    int *d_N;
    cudaMalloc(&d_N, sizeof(int));
    cudaMemset(d_N, 0, sizeof(int));

    dim3 threadsPerBlock(512);
    dim3 numBlocks((n_iter + threadsPerBlock.x - 1) / threadsPerBlock.x);
    auto t1 = std::chrono::high_resolution_clock::now();
    countInsidePointsKernel<<<numBlocks, threadsPerBlock>>>(n_iter, dev_planes, static_cast<int>(planes.size()), d_N);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CUDA Time: " << ((std::chrono::duration<double>) (t2 - t1)).count() << std::endl;

    int N;
    cudaMemcpy(&N, d_N, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_planes);

    double V_cube = 8.0;
    return V_cube * N / static_cast<double>(n_iter);
}
