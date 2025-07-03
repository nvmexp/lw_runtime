/*
 * Defines the class structure for the highest level 
 * object in the Lwpu Validation Suite.
 *
 * Add other docs in here as methods get defined
 *
 */

#ifndef _LWVS_LWVS_LwidiaValidationSuite_H_
#define _LWVS_LWVS_LwidiaValidationSuite_H_

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include "Test.h"
#include "Gpu.h"
#include "ConfigFileParser_v2.h"
#include "TestFramework.h"
#include "TestParameters.h"
#include "Whitelist.h"
#include "common.h"
#include "LwvsSystemChecker.h"

#define LWVS_VERSION "1.7"
#define MIN_MAJOR_VERSION 346
#define MAX_MAJOR_VERSION 500

using namespace std;

class LwidiaValidationSuite
{
/***************************PUBLIC***********************************/
public:
    // ctor/dtor
    LwidiaValidationSuite();
    ~LwidiaValidationSuite();

    void go(int argc, char *argv[]);

/***************************PRIVATE**********************************/
private:

    // methods
    void processCommandLine(int argc, char *argv[]);
    void enumerateAllVisibleGpus();
    void enumerateAllVisibleTests();
    void fillGpuSetObjs(vector<GpuSet *> gpuSets);
    void banner();
    void CheckDriverVersion();
    vector<Gpu *> decipherProperties(GpuSet* set);
    vector<Test *>::iterator findTestName(std::string testName);
    void fillTestVectors(suiteNames_enum suite, Test::testClasses_enum testClass, GpuSet * set);
    void overrideParameters(TestParameters *tp, const std::string &lowerCaseTestName);
    void startTimer();
    void stopTimer();
    bool HasGenericSupport(const std::string &gpuBrand, uint64_t gpuArch);
    void InitializeAndCheckGpuObjs(std::vector<GpuSet *> &gpuSets);

    // vars
    bool logInit;
    std::vector<Gpu *> gpuVect;
    std::vector<Test *> testVect;
    std::vector<TestParameters *> tpVect;
    Whitelist * whitelist;
    FrameworkConfig * fwcfg;

    // classes
    ConfigFileParser_v2 *parser;
    TestFramework *tf;

    // parsing variables
    std::string configFile;
    std::string debugFile;
    std::string hwDiagLogFile;
    
    bool listTests;
    bool listGpus;
    bool appendMode;
    timer_t initTimer;
    struct sigaction restoreSigAction;
    unsigned int initWaitTime;
    bool loadedLwdaLibrary;
    LwvsSystemChecker m_sysCheck;

/***************************PROTECTED********************************/
protected:
};
#endif // _LWVS_LWVS_LwidiaValidationSuite_H_
