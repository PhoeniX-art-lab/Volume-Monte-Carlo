#ifndef LAB_4_POINT_CUH
#define LAB_4_POINT_CUH

#endif //LAB_4_POINT_CUH

#include <random>
#include <cmath>

struct Point {
    double x, y, z;
};

struct Plane {
    double a, b, c, d;
};

Plane getPlane(Point, Point, Point);
bool rayIntersectsPlane(Plane, Point);
bool isPointInsidePolyhedron(const std::vector<Plane>& planes, Point point);
