/* 
 * File:   TestHealthMonitor.h
 */

#ifndef TESTHEALTHMONITOR_H
#define TESTHEALTHMONITOR_H

#include "TestLwcmModule.h"
#include "dcgm_structs.h"
#include "dcgm_agent.h"

class TestHealthMonitor : public TestLwcmModule {
public:
    TestHealthMonitor();
    virtual ~TestHealthMonitor();
    
    /*************************************************************************/
    /* Inherited methods from TestLwcmModule */
    int Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus);
    int Run();
    int Cleanup();
    std::string GetTag();  
    
private:
    int TestHMSet();
    int TestHMCheckPCIe();
    int TestHMCheckMemSbe();
    int TestHMCheckMemDbe();
    int TestHMCheckInforom();
    int TestHMCheckThermal();
    int TestHMCheckPower();
    int TestHMCheckLWLink();

    std::vector<test_lwcm_gpu_t>m_gpus; /* List of GPUs to run on, copied in Init() */
};

#endif  /* HM */
