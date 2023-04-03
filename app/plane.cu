#include "plane.cuh"


Plane getPlane(Point p1, Point p2, Point p3) {
    Point v1 = {p2.x - p1.x, p2.y - p1.y, p2.z - p1.z};
    Point v2 = {p3.x - p1.x, p3.y - p1.y, p3.z - p1.z};

    Point n = {v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x};
    double d = -1 * (n.x * p1.x + n.y * p1.y + n.z * p1.z);

    return {n.x, n.y, n.z, d};
}


bool rayIntersectsPlane(const Plane plane, const Point point) {
    Point ray_direction = {1.0, 0.0, 0.0};
    double denominator = plane.a * ray_direction.x + plane.b * ray_direction.y + plane.c * ray_direction.z;
    if (std::abs(denominator) < 1e-6)
        return false;
    double numerator = -(plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d);

    return (numerator / denominator) >= 0;
}


bool isPointInsidePolyhedron(const std::vector<Plane> &planes, Point point) {
    int cnt = 0;
    for (const auto plane : planes) {
        if (rayIntersectsPlane(plane, point))
            cnt += 1;
    }

    return cnt % 2 == 1;
}
