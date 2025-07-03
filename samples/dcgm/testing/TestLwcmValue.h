#ifndef TESTLWCMVALUE_H
#define TESTLWCMVALUE_H

#include "TestLwcmModule.h"
#include "dcgm_structs.h"

class TestLwcmValue : public TestLwcmModule {
public:
    TestLwcmValue();
    virtual ~TestLwcmValue();

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
    int TestColwersions(void);

};

#endif  /* TESTLWCMVALUE_H */
