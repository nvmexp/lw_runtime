#ifndef TESTLWCMMODULE_H
#define TESTLWCMMODULE_H

#include <vector>
#include <string>
#include "dcgm_structs.h"

/*****************************************************************************/
typedef struct test_lwcm_gpu_t
{
    unsigned int gpuId;         /* LWCM GPU id. use LwcmSettings to colwert to lwmlIndex */
    unsigned int lwmlIndex;     /* LWML index */
} test_lwcm_gpu_t, *test_lwcm_gpu_p;

/*****************************************************************************/
/*
 * Base class for a Lwcm test module
 *
 * Note: if you add an instance of this class, make sure it gets loaded in
 *       TestLwcmUnitTests::LoadModules() so that it runs
 */

class TestLwcmModule
{
public:
    /*
     * Placeholder constructor and destructor. You do not need to call these
     *
     */
    TestLwcmModule() {}
    virtual ~TestLwcmModule() {} /* Virtual to satisfy ancient GCC */

    /*************************************************************************/
    /*
     * Method child classes should implement to initialize an instance of a TestLwcmModule
     * class. When this method is done, Run() should be able to be called
     *
     * All initialization code of your module that depends on LWML or LWCM being
     * at least globally initialized should be in here, as those are not guaranteed
     * to be loaded when your class is instantiated
     *
     * argv represents all of the arguments to the test framework that weren't
     * already parsed globally and are thus assumed to be module parameters
     *
     * gpus represents which GPUs to try to run on as discovered/parsed from
     * the command line
     *
     * Returns 0 on success
     *        !0 on error
     */
    virtual int Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus) = 0;

    /*************************************************************************/
    /*
     * Run the tests of this module
     *
     * Returns  0 on success
     *         <0 on fatal failure. Will abort entire framework
     *         >0 on test failure. Will keep running other tests.
     *
     */
    virtual int Run() = 0;

    /*************************************************************************/
    /*
     * Tell this module to clean up after itself. Your destructor should
     * also call this method
     *
     */
    virtual int Cleanup() = 0;

    /*************************************************************************/
    /*
     * Get the tag for this test. This tag should be how the module is referenced
     * in both command line and logging
     */
    virtual std::string GetTag() = 0;

    /*************************************************************************/
    /*
     * Should this module be included in the default list of modules run?
     * 
     * Users can run all modules by passing -a to testdcgmunittests
     */
    virtual bool IncludeInDefaultList(void)
    {
        return true;
    }
    
    /*****************************************************************************/
    /**
     * Assign copy of DCGM Handle so that it can be used by the derived modules
     */
    void SetDcgmHandle(dcgmHandle_t dcgmHandle) {
        m_dcgmHandle = dcgmHandle;
    }
    
    /*****************************************************************************/
    /*
     * Get a boolean value as to whether this is a debug build of DCGM or not
     *
     * Returns: 0 if not a debug build
     *          1 if is a debug build
     *
     */
    int IsDebugBuild(void) {
#ifdef _DEBUG
        return 1;
#else
        return 0;
#endif
    }

    /*****************************************************************************/

protected:
    dcgmHandle_t m_dcgmHandle;

};


/*****************************************************************************/

#endif //TESTLWCMMODULE_H
