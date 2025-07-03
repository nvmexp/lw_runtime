#ifndef MEMORYBANDWIDTHPLUGIN_H
#define MEMORYBANDWIDTHPLUGIN_H

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include "PluginCommon.h"
#include "Plugin.h"


class MemoryBandwidthPlugin : public Plugin
{
public:
    MemoryBandwidthPlugin();
    ~MemoryBandwidthPlugin() {};

    void Go(TestParameters *testParameters, const std::vector<unsigned int> &gpuList);

    // unimplemented pure virtual funcs
    void Go(TestParameters *testParameters)
    {
        Go(NULL, 0);
    }
    void Go(TestParameters *testParameters, unsigned int)
    {
        throw std::runtime_error("Not implemented in this test.");
    }
};

#endif // MEMORYBANDWIDTHPLUGIN_H
