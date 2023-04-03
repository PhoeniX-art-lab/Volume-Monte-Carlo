#ifndef LAB_4_POINT_CUH
#define LAB_4_POINT_CUH

#include <random>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <curand_kernel.h>

struct Point {
    double x, y, z;
};

struct Plane {
    double a, b, c, d;
};

Plane getPlane(Point, Point, Point);
bool rayIntersectsPlane(Plane, Point);
bool isPointInsidePolyhedron(const std::vector<Plane> &planes, Point point);
__global__ void countInsidePointsKernel(int, const Plane *, int, int *);

#endif //LAB_4_POINT_CUH
