
/* 
 * File:   TestActionManager.h
 */

#ifndef TESTDIAG_H
#define TESTDIAG_H

#include "TestDcgmiModule.h"
#include "dcgm_structs.h"
#include "json/json.h"

class TestDiag : public TestDcgmiModule {
public:
    TestDiag();
    virtual ~TestDiag();
    
    /*************************************************************************/
    /* Inherited methods from TestLwcmModule */
    int Run();
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
    int TestPopulateGpuList();
    int TestHelperGetPluginName();
    int TestHelperJsonAddResult();
    int TestHelperJsonAddBasicTests();
    int TestHelperJsonBuildOutput();
    int TestHelperJsonTestEntry(Json::Value &testEntry, int gpuIndex, const std::string &status, const std::string &warning);
    int TestGetFailureResult();
};

#endif /* TESTDIAG_H */
