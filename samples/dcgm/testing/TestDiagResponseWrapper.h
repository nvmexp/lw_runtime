/* 
 * File:   TestDiagResponseWrapper.h
 */

#ifndef TEST_DIAG_RESPONSE_WRAPPER_H
#define TEST_DIAG_RESPONSE_WRAPPER_H

#include "TestLwcmModule.h"
#include "dcgm_structs.h"

class TestDiagResponseWrapper : public TestLwcmModule
{
public:
    TestDiagResponseWrapper();
    virtual ~TestDiagResponseWrapper();
    
    /*************************************************************************/
    /* Inherited methods from TestLwcmModule */
    int Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus);
    int Run();
    int Cleanup();
    std::string GetTag();  
    
private:
    /*************************************************************************/
    /*
     * Actual test cases. These should return a status like below
     *
     * Returns 0 on success
     *        <0 on fatal error. Will abort entire framework
     *        >0 on non-fatal error
     *
     **/
    int TestInitializeDiagResponse();
    int TestSetPerGpuResponseState();
    int TestAddPerGpuMessage();
    int TestSetGpuIndex();
    int TestGetBasicTestResultIndex();
    int TestRecordSystemError();
    int TestAddErrorDetail();
};

#endif /* TESTDIAGMANAGER_H */
