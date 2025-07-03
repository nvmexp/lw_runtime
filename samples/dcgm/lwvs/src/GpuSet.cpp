#include "GpuSet.h"
#include "PluginStrings.h"
#include "Test.h"

const int LWSTOM_TEST_OBJS      = 0;
const int SOFTWARE_TEST_OBJS    = 1;
const int HARDWARE_TEST_OBJS    = 2;
const int INTEGRATION_TEST_OBJS = 3;
const int PERFORMANCE_TEST_OBJS = 4;
    
GpuSet::GpuSet() : name(), properties(), testsRequested(), gpuObjs(), m_lwstomTestObjs(),
                   m_softwareTestObjs(), m_hardwareTestObjs(), m_integrationTestObjs(),
                   m_performanceTestObjs()
{
    properties.present = false;
}

int GpuSet::AddTestObject(int testClass, Test *test)
{
    if (!test)
        return -1;

    switch (testClass)
    {
        case LWSTOM_TEST_OBJS:
            m_lwstomTestObjs.push_back(test);
            break;

        case SOFTWARE_TEST_OBJS:
            m_softwareTestObjs.push_back(test);
            break;

        case HARDWARE_TEST_OBJS:
            m_hardwareTestObjs.push_back(test);
            break;

        case INTEGRATION_TEST_OBJS:
            m_integrationTestObjs.push_back(test);
            break;

        case PERFORMANCE_TEST_OBJS:
            m_performanceTestObjs.push_back(test);
            break;

        default:
            return -1;
            break;
    }

    return 0;
}
