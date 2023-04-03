#ifndef LAB_4_MONTECARLO_CUH
#define LAB_4_MONTECARLO_CUH

#include "plane.cuh"


double calculateVolumeCPU(std::vector<Plane> &, int);
double calculateVolumeGPU(std::vector<Plane> &, int);

#endif //LAB_4_MONTECARLO_CUH
