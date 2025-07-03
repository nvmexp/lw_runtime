#ifndef _LWVS_LWVS_GpuSet_H_
#define _LWVS_LWVS_GpuSet_H_

#include <string>
#include <vector>
#include "Gpu.h"
#include "Test.h"

extern const int LWSTOM_TEST_OBJS;
extern const int SOFTWARE_TEST_OBJS;
extern const int HARDWARE_TEST_OBJS;
extern const int INTEGRATION_TEST_OBJS;
extern const int PERFORMANCE_TEST_OBJS;

class GpuSet
{

/***************************PUBLIC***********************************/
public:
    // Default initializers?
    GpuSet();
    ~GpuSet() {};

    std::string name;
    struct Props
    {
        bool present;
        std::string brand;
        std::vector<unsigned int> index;
        std::string name;
        std::string busid;
        std::string uuid;
    };

    Props properties;

    std::vector<std::map<std::string, std::string> > testsRequested;
    std::vector<Gpu *> gpuObjs; // corresponding GPU objects
    
    std::vector<Test *> m_lwstomTestObjs;     // user-specified test objects
    std::vector<Test *> m_softwareTestObjs;   // software-class test objects
    std::vector<Test *> m_hardwareTestObjs;   // hardware-class test objects
    std::vector<Test *> m_integrationTestObjs;// integration-class test objects
    std::vector<Test *> m_performanceTestObjs;// performance-class test objects

    int AddTestObject(int testClass, Test *test);

/***************************PRIVATE**********************************/
private:

/***************************PROTECTED********************************/
protected:
};

#endif //_LWVS_LWVS_GpuSet_H_

