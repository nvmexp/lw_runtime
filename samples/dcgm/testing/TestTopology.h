/* 
 * File:   TestTopology.h
 */

#ifndef TESTTOPOLOGY_H
#define TESTTOPOLOGY_H

#include "TestLwcmModule.h"
#include "dcgm_structs_internal.h"
#include "dcgm_structs.h"
#include "dcgm_agent.h"

class TestTopology : public TestLwcmModule {
public:
    TestTopology();
    virtual ~TestTopology();
    
    /*************************************************************************/
    /* Inherited methods from TestLwcmModule */
    int Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus);
    int Run();
    int Cleanup();
    std::string GetTag();  
    
private:
    int TestTopologyDevice();
    int TestTopologyGroup();

    std::vector<test_lwcm_gpu_t>m_gpus; /* List of GPUs to run on, copied in Init() */
};

#endif  /* TESTVERSIONING_H */
