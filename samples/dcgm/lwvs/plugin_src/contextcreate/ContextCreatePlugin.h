#ifndef CONTEXTCREATEPLUGIN_H
#define CONTEXTCREATEPLUGIN_H

#include "PluginCommon.h"
#include "Plugin.h"

#define CONTEXT_CREATE_PASS  0
#define CONTEXT_CREATE_FAIL -1
#define CONTEXT_CREATE_SKIP -2

class ContextCreatePlugin : public Plugin
{
public:
    ContextCreatePlugin();
    ~ContextCreatePlugin() {}

    void Go(TestParameters *testParameters, const std::vector<unsigned int> &gpuList);

    // Unused
    void Go(TestParameters *tp) 
    { 
        Go(NULL, 0);
    }

    // Unused
    void Go(TestParameters *tp, unsigned int i)
    {
        throw std::runtime_error("Not implemented for Context Create");
    }
};

#endif
