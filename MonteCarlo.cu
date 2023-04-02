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