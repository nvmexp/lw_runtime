/* 
 * File:   TestPolicyManager.h
 */

#ifndef TESTPOLICYMANAGER_H
#define TESTPOLICYMANAGER_H

#include "TestLwcmModule.h"
#include "dcgm_structs_internal.h"
#include "dcgm_structs.h"
#include "dcgm_agent.h"

class TestPolicyManager : public TestLwcmModule {
public:
    TestPolicyManager();
    virtual ~TestPolicyManager();
    
    /*************************************************************************/
    /* Inherited methods from TestLwcmModule */
    int Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus);
    int Run();
    int Cleanup();
    std::string GetTag();  
    
private:
    int TestPolicySetGet();
    int TestPolicyRegUnreg();
    int TestPolicyRegUnregXID();
};

#endif  /* TESTVERSIONING_H */
