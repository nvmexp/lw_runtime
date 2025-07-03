#ifndef _LWVS_LWVS_BusGrind_H_
#define _LWVS_LWVS_BusGrind_H_

#include <string>
#include <vector>
#include <iostream>
#include "lwml.h"
#include "Plugin.h"
#include "Gpu.h"

class BusGrind : public Plugin
{
public:
    BusGrind();
    ~BusGrind() {}
    
    void Go(TestParameters *testParameters, unsigned int)
    {
        /* Setting selfParallel doesn't work for some reason. Calling parallel */
        std::vector<unsigned int>gpuList;
        gpuList.push_back(0);
        
        Go(testParameters, gpuList);
    }

    void Go(TestParameters *testParameters)
    {
        throw std::runtime_error("Not implemented in this test.");
    }

    void Go(TestParameters *testParameters, const std::vector<unsigned int> &gpuList);
};



#endif // _LWVS_LWVS_BusGrind_H_
