/* 
 * File:   TestDiagManager.h
 */

#ifndef TESTDIAGMANAGER_H
#define TESTDIAGMANAGER_H

#include "TestLwcmModule.h"
#include "dcgm_structs.h"

class TestDiagManager : public TestLwcmModule {
public:
    TestDiagManager();
    virtual ~TestDiagManager();
    
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
    int TestPositiveDummyExelwtable();
    int TestNegativeDummyExelwtable();
    int TestCreateLwvsCommand();
    int TestPopulateRunDiag();
    int TestFillResponseStructure();
    int TestPerformExternalCommand();
    int TestErrorsFromLevelOne();
    void CreateDummyScript();
    void CreateDummyFailScript();
    void RemoveDummyScript();
};

#endif /* TESTDIAGMANAGER_H */
