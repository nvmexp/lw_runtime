#ifndef _LWVS_LWVS_Memory_H_
#define _LWVS_LWVS_Memory_H_

#include <string>
#include <vector>
#include <iostream>
#include "Plugin.h"
#include "TestParameters.h"
#include "PluginCommon.h"
#include "lwml.h"

class Memory : public Plugin
{
public:
    Memory();
    ~Memory() { }

    void Go(TestParameters *testParameters, unsigned int);
    void Go(TestParameters *testParameters, const std::vector<unsigned int> &gpuList)
    {
        throw std::runtime_error("Not implemented in this test.");        
    }
    void Go(TestParameters *testParameters)
    {
        throw std::runtime_error("Not implemented in this test.");
    }
    

private:
    //variables
    TestParameters * tp;

};



#endif // _LWVS_LWVS_Memory_H_
