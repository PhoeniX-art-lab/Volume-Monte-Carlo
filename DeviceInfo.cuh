#ifndef LAB_4_DEVICEINFO_CUH
#define LAB_4_DEVICEINFO_CUH

#include <iostream>

class DeviceInfo {
public:
    static void printCudaDeviceInfo(cudaDeviceProp&);
    static void getCudaDeviceInfo();
};

#endif //LAB_4_DEVICEINFO_CUH
