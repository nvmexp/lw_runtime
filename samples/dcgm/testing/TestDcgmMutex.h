#ifndef TESTDCGMMUTEX_H
#define TESTDCGMMUTEX_H

#include "TestLwcmModule.h"
#include "dcgm_fields.h"
#include "timelib.h"
#include "DcgmMutex.h"

/*****************************************************************************/
class TestDcgmMutex : public TestLwcmModule
{
public:
    TestDcgmMutex();
    ~TestDcgmMutex();

    /*************************************************************************/
    /* Inherited methods from TestLwcmModule */
    int Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus);
    int Run();
    int Cleanup();
    std::string GetTag();

    /*************************************************************************/
private:
    std::vector<test_lwcm_gpu_t>m_gpus; /* List of GPUs to run on, copied in Init() */

    /*************************************************************************/
    /*
     * Actual test cases. These should return a status like below
     *
     * Returns 0 on success
     *        <0 on fatal error. Will abort entire framework
     *        >0 on non-fatal error
     *
     **/
    int TestDoubleLock();
    int TestDoubleUnlock();
    int TestPerf();

    /*************************************************************************/
    /*
     * Helper for running a test.  Provided the test name, the return from running the
     * test, and a reference to the number of failed tests it will take care of printing
     * the appropriate pass/fail messages and increment the failed test count.
     *
     * An std::runtime_error is raised if the given testReturn was < 0, signifying a fatal error.
     */
    void CompleteTest(std::string testName, int testReturn, int &Nfailed);

    /*************************************************************************/
};

/*****************************************************************************/

#endif //TESTDCGMMUTEX_H
