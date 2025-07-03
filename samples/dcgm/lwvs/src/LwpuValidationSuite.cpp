
#define LWML_INIT_UUID /* must only be defined once per linkage unit in order to call lwml internal APIs */

#include "LwidiaValidationSuite.h"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <cstdlib>
#include <tclap/CmdLine.h>
#include <setjmp.h>
#include <time.h>
#include <signal.h>
#include "PluginStrings.h"
#include "../../common/DcgmStringTokenize.h"
#include "json/json.h"
#include "LwvsJsonStrings.h"
#include "DcgmHandle.h"
#include "DcgmSystem.h"
#include "ParsingUtility.h"
#include "ParameterValidator.h"

extern "C" {
    #include "lwml.h"
    #include "lwml_internal.h"
    #include "lwca-loader.h"
}

DcgmHandle   dcgmHandle;
DcgmSystem   dcgmSystem;
LwvsCommon   lwvsCommon;
bool         initTimedOut = false;
jmp_buf      exitInitialization;

/*****************************************************************************/
LwidiaValidationSuite::LwidiaValidationSuite() : logInit(false), gpuVect(), testVect(), tpVect(), whitelist(0),
                                                 tf(0), configFile(), debugFile(), hwDiagLogFile(),
                                                 listTests(false), listGpus(false), appendMode(false),
                                                 initTimer(0), initWaitTime(120), loadedLwdaLibrary(false),
                                                 m_sysCheck()
{   
    fwcfg = new FrameworkConfig();
    parser = new ConfigFileParser_v2("/etc/lwpu-validation-suite/lwvs.conf", *fwcfg);

    // init globals
    lwvsCommon.Init();
}  

/*****************************************************************************/
LwidiaValidationSuite::~LwidiaValidationSuite()
{
    for (std::vector<Gpu *>::iterator it = gpuVect.begin(); it != gpuVect.end(); ++it)
    {
        delete (*it);  
    }

    for (std::vector<TestParameters *>:: iterator it = tpVect.begin(); it != tpVect.end(); ++it)
    {
        delete (*it);
    }
    delete tf;
    delete whitelist;
    delete parser;

    if (loadedLwdaLibrary == true)
        unloadLwdaLibrary();

    if (logInit)
        loggingShutdown();
    dcgmShutdown();
}

/*****************************************************************************/
void LwidiaValidationSuite::CheckDriverVersion()
{
    dcgmSystem.Init(dcgmHandle.GetHandle());
    dcgmDeviceAttributes_t deviceAttr;

    memset(&deviceAttr, 0, sizeof(deviceAttr));
    deviceAttr.version = dcgmDeviceAttributes_version1;
    dcgmReturn_t ret = dcgmSystem.GetDeviceAttributes(0, deviceAttr);
    unsigned int count = 0;
    std::stringstream additionalMsg;

    // Attempt re-connecting if we have trouble with our initial interaction with the hostengine.
    for (unsigned int count = 0; count < 3 && ret == DCGM_ST_CONNECTION_NOT_VALID; ++count)
    {
        dcgmReturn_t connectionRet = dcgmHandle.ConnectToDcgm(lwvsCommon.dcgmHostname);
        if (connectionRet == DCGM_ST_OK)
        {
            dcgmSystem.Init(dcgmHandle.GetHandle());
            ret = dcgmSystem.GetDeviceAttributes(0, deviceAttr);
        }
        else
        {
            additionalMsg.str("");
            additionalMsg << "Couldn't re-connect to hostengine after establishing an invalid connection: "
                          << dcgmHandle.RetToString(connectionRet);
        }
    }

    std::stringstream buf;
    std::string s_version;
    unsigned int i_version = 0;
    std::stringstream ss;

    if (ret != DCGM_ST_OK)
    {
        buf << "Unable to get the driver version: " << dcgmHandle.RetToString(ret);
        if (additionalMsg.rdbuf()->in_avail() == 0)
        {
            buf << ". Couldn't succeed despite " << count << " retries.";
        }
        else
        {
            buf << ". " << additionalMsg;
        }
        throw std::runtime_error(buf.str());
    }

    s_version = deviceAttr.identifiers.driverVersion;
    s_version = s_version.substr(0, s_version.find("."));
    ss << s_version;

    ss >> i_version;

    if (i_version < MIN_MAJOR_VERSION || i_version > MAX_MAJOR_VERSION)
    {
        std::stringstream exceptionSs;
        exceptionSs << "The installed driver is not between v" << MIN_MAJOR_VERSION << " and v" << MAX_MAJOR_VERSION; 
        throw std::runtime_error (exceptionSs.str());
    }
}

// Handler for SIGALRM as part of ensuring we can't stall indefinitely on DCGM init 
void alarm_handler(int sig)
{
    initTimedOut = true;
    longjmp(exitInitialization, 1);
}

/*
 * startTimer - Starts a timer to make sure we don't get stuck in the DCGM initialization and or loading
 * the LWCA library
 */
void LwidiaValidationSuite::startTimer()
{
    struct itimerspec value;
    struct sigaction act;
    
    memset(&value, 0, sizeof(value));
    memset(&act, 0, sizeof(act));

    // Set SIGALRM to use alarm_handler
    act.sa_handler = alarm_handler;
    sigemptyset(&act.sa_mask);
    sigaction(SIGALRM, &act, &this->restoreSigAction);

    // Set the timer for 20 seconds
    value.it_value.tv_sec = this->initWaitTime;
    timer_create(CLOCK_REALTIME, NULL, &this->initTimer);
    timer_settime(this->initTimer, 0, &value, NULL);
}

/*
 * stopTimer - disables the timer previously set to make sure we don't get stuck in the DCGM initialization
 * and or loading the LWCA library
 */
void LwidiaValidationSuite::stopTimer()
{
    struct itimerspec value;
    memset(&value, 0, sizeof(value));
    timer_settime(this->initTimer, 0, &value, NULL);

    // Restore any previous signal handler
    sigaction(SIGALRM, &this->restoreSigAction, NULL);
}


/*****************************************************************************/
void LwidiaValidationSuite::go(int argc, char *argv[])
{
    std::vector<unsigned int> gpuIndices;
    std::string               info;

    processCommandLine(argc, argv);
    if (lwvsCommon.quietMode)
    {
        std::cout.setstate(std::ios::failbit);
    }
    banner();

    loggingInit((char *)LWVS_ELW_DBG_LVL, (char *)LWVS_ELW_DBG_APPEND,
                (char *)LWVS_ELW_DBG_FILE);
    logInit = true;

    startTimer();
    
    // Mark this as the point to return too if DCGM's init takes too long
    setjmp(exitInitialization);

    // If initTimedOut is set, then we've timed out while trying to initialize DCGM and load the lwca library
    if (initTimedOut == true)
    {
        std::stringstream buf;
        buf << "We reached the " << this->initWaitTime << " second timeout while attempting to initialize DCGM";
        buf << " and load the LWCA library.";
        buf << "\nPlease check why these systems are unresponsive.\n";

        throw std::runtime_error(buf.str());
    }

    dcgmReturn_t ret = dcgmHandle.ConnectToDcgm(lwvsCommon.dcgmHostname);
    if (ret != DCGM_ST_OK)
    {
        std::stringstream buf;
        buf << "Unable to connect to DCGM: " << dcgmHandle.GetLastError();
        throw std::runtime_error (buf.str());
    }
    if (LWDA_LIBRARY_LOAD_SUCCESS != loadDefaultLwdaLibrary())
        throw std::runtime_error ("Unable to load the LWCA library, check for correct LD_LIBRARY_PATH");
    else
        loadedLwdaLibrary = true;
    
    stopTimer();

    CheckDriverVersion();

    if (configFile.size() > 0)
        parser->setConfigFile(configFile);
    if (!parser->Init() && !lwvsCommon.configless)
    {
        std::ostringstream out;
        out << std::endl << "Unable to open config file ";
        if (configFile.size() > 0)
        {
            out << configFile <<", please check path and try again." << std::endl;
            throw std::runtime_error(out.str());
        }
        else
        {
            // If they didn't specify a config file, just warn
            out << "/etc/lwpu-validation-suite/lwvs.conf. " << std::endl;
            out << "Please check the path or specify a config file via the \"-c\" command line option." << std::endl;
            out << "Add --configless to suppress this warning." << std::endl;
            info = out.str();

            // Force to true if we couldn't open a config file
            lwvsCommon.configless = true;
        }
    }

    // second process the config file
    if (!lwvsCommon.configless)
        parser->ParseGlobalsAndGpu();
    else
        parser->legacyGlobalStructHelper();

    vector<GpuSet *> gpuSets = parser->getGpuSetVec();

    enumerateAllVisibleGpus();

    if (listGpus)
    {
        std::cout << "Supported GPUs available:" << std::endl;

        for (vector<Gpu *>::iterator it = gpuVect.begin(); it != gpuVect.end(); ++it)
        {
            std::cout << "\t" << "[" << (*it)->getDevicePciBusId() << "] -- " << (*it)->getDeviceName() << std::endl;
        }
        std::cout << std::endl;
        return;
    }

    for (size_t i = 0; i < gpuSets.size(); i++) {
        for (size_t j = 0; j < gpuSets[i]->properties.index.size(); j++) {
            gpuIndices.push_back(gpuSets[i]->properties.index[j]);
        }
    }

    if (gpuIndices.size() == 0) {
        for (size_t i = 0; i < gpuVect.size(); i++) {
            gpuIndices.push_back(gpuVect[i]->getDeviceIndex());
        }
    }

    if (lwvsCommon.training && !lwvsCommon.forceTraining)
    {
        std::string sysError = m_sysCheck.CheckSystemInterference();

        if (sysError.size())
        {
            PRINT_WARNING("%s", "%s", sysError.c_str());
        }
    }

    // construct the test framework now that we have the GPU and test objects
    tf = new TestFramework(lwvsCommon.jsonOutput, gpuIndices);
    tf->loadPlugins();
    if (info.size() > 0)
    {
        tf->addInfoStatement(info);
    }

    enumerateAllVisibleTests();
    if (listTests)
    {
        std::cout << "Tests available:" << std::endl;

        for (vector<Test *>::iterator it = testVect.begin(); it != testVect.end(); ++it)
        {
            if (it + 1 != testVect.end()) // last object is a "skip" object that does not need to be displayed
                std::cout << "\t" << (*it)->GetTestName() << " -- " << (*it)->getTestDesc() << std::endl;
        }
        std::cout << std::endl;
        return;
    }

    if (lwvsCommon.training)
    {
        InitializeAndCheckGpuObjs(gpuSets);

        unsigned int iterations = lwvsCommon.trainingIterations;
        for (unsigned int i = 0; i < iterations; i++)
        {
            for (size_t setIndex = 0; setIndex < gpuSets.size(); setIndex++)
            {
                if (i == 0)
                    fillTestVectors(LWVS_SUITE_LONG, Test::LWVS_CLASS_SOFTWARE, gpuSets[setIndex]);

                fillTestVectors(LWVS_SUITE_LONG, Test::LWVS_CLASS_HARDWARE, gpuSets[setIndex]);
                fillTestVectors(LWVS_SUITE_LONG, Test::LWVS_CLASS_INTEGRATION, gpuSets[setIndex]);
                fillTestVectors(LWVS_SUITE_LONG, Test::LWVS_CLASS_PERFORMANCE, gpuSets[setIndex]);
            }
        
            tf->go(gpuSets);
       
            float pcnt = static_cast<float>(i + 1) / static_cast<float>(iterations); 
            if (lwvsCommon.jsonOutput == false)
            {
                std::cout << "Completed iteration " << i+1 << " of " << iterations << " : training is " 
                          << static_cast<int>(pcnt * 100) << "% complete." << std::endl;
            }
        }

        tf->CallwlateAndSaveGoldelwalues();
    }
    else
    {
        fillGpuSetObjs(gpuSets);

        // Execute the tests... let the TF catch all exceptions and decide 
        // whether to throw them higher.
        tf->go(gpuSets);
    }

    return;
}

/*****************************************************************************/
void LwidiaValidationSuite::banner()
{
    if (lwvsCommon.jsonOutput == false)
    {
        cout << endl << LWVS_NAME << " (version " << DRIVER_MAJOR_VERSION << ")" << endl << endl;
    }
}

/*****************************************************************************/
void LwidiaValidationSuite::enumerateAllVisibleTests()
{
    // for now just use the testVec stored in the Framework
    // but eventually obfuscate this some
    
    testVect = tf->getTests();
}

bool LwidiaValidationSuite::HasGenericSupport(const std::string &gpuBrand, uint64_t gpuArch)
{
    static const unsigned int MAJOR_MAXWELL_COMPAT = 5;
    static const unsigned int MAJOR_KEPLER_COMPAT = 3;

    if ((DCGM_LWDA_COMPUTE_CAPABILITY_MAJOR(gpuArch) >= MAJOR_MAXWELL_COMPAT) ||
        (gpuBrand == "Tesla" && DCGM_LWDA_COMPUTE_CAPABILITY_MAJOR(gpuArch) >= MAJOR_KEPLER_COMPAT))
        return true;

    return false;
}

/*****************************************************************************/
void LwidiaValidationSuite::enumerateAllVisibleGpus()
{
    unsigned int deviceCount;
    bool isWhitelisted;
    std::vector<unsigned int> gpuIds;
    dcgmReturn_t ret = dcgmSystem.GetAllSupportedDevices(gpuIds);

    if (ret != DCGM_ST_OK)
    {
        std::stringstream buf;
        buf << "Unable to retrieve device count: " << dcgmHandle.RetToString(ret);
        throw runtime_error(buf.str());
    }
    
    whitelist = new Whitelist();

    for (size_t i = 0; i < gpuIds.size(); i++)
    {
        Gpu *gpu = new Gpu(gpuIds[i]);

        /* Find out if this device is supported, which is any of the following:
           1. On the LWVS whitelist explicitly
           2. A Kepler or newer Tesla part 
           3. a Maxwell or newer part of any other brand (Lwdqro, VdChip, Titan, Grid)
        */
           
        isWhitelisted = whitelist->isWhitelisted(gpu->getDevicePciDeviceId());
        std::string gpuBrand = gpu->getDeviceBrandAsString();
        uint64_t gpuArch = gpu->getDeviceArchitecture();

        if (isWhitelisted)
        {
            PRINT_DEBUG("%u", "dcgmIndex %u is directly whitelisted.", gpuIds[i]);
            gpu->setDeviceIsSupported(true);
        }
        else if (HasGenericSupport(gpuBrand, gpuArch))
        {
            PRINT_DEBUG("%u %s %u", "lwmlIndex %u, brand %s, arch %u is supported", 
                        gpuIds[i], gpuBrand.c_str(), static_cast<unsigned int>(gpuArch));
            gpu->setDeviceIsSupported(true);
        }
        else
        {
            PRINT_DEBUG("%u %s %u", "lwmlIndex %u, brand %s, arch %u is NOT supported", 
                        gpuIds[i], gpuBrand.c_str(), static_cast<unsigned int>(gpuArch));
            gpu->setDeviceIsSupported(false);
        }
        

        if (gpu->getDeviceIsSupported())
            gpuVect.push_back(gpu);
        else 
        {
            std::stringstream ss;
            ss << "\t" << "[" << gpu->getDevicePciBusId() << "] -- " << gpu->getDeviceName() <<
                 " -- Not Supported";
            PRINT_INFO("%s", "%s", ss.str().c_str());
            delete gpu;
        }
    }

    /* Allow the whitelist to adjust itself now that GPUs have been read in */
    whitelist->postProcessWhitelist(gpuVect);


    return;
}

/*****************************************************************************/
void LwidiaValidationSuite::overrideParameters(TestParameters *tp, const std::string &lowerCaseTestName)
{
    if (lwvsCommon.parms.find(lowerCaseTestName) != lwvsCommon.parms.end()) {
        for (std::map<std::string, std::string>::iterator it = lwvsCommon.parms[lowerCaseTestName].begin();
             it != lwvsCommon.parms[lowerCaseTestName].end();
             it++) {
            tp->OverrideFromString(it->first, it->second);
        }
    }
}

/*****************************************************************************/
void LwidiaValidationSuite::InitializeAndCheckGpuObjs(std::vector<GpuSet *> &gpuSets)
{
    for (unsigned int i = 0; i < gpuSets.size(); i++)
    {
        if (!gpuSets[i]->properties.present) 
        {
            gpuSets[i]->gpuObjs = gpuVect;
        }
        else
        {
            gpuSets[i]->gpuObjs = decipherProperties(gpuSets[i]);
        }

        if (gpuSets[i]->gpuObjs.size() == 0)
        { // nothing matched
            std::ostringstream ss;
            ss << "Unable to match GPU set '" << gpuSets[i]->name << "' to any GPU(s) on the system.";
            PRINT_ERROR("%s", "%s", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }
    
        // ensure homogeneity
        std::string firstName = gpuSets[i]->gpuObjs[0]->getDeviceName();
        for (std::vector<Gpu *>::iterator gpuIt = gpuSets[i]->gpuObjs.begin(); gpuIt != gpuSets[i]->gpuObjs.end(); gpuIt++)
        {
            // no need to check the first but...
            if (firstName != (*gpuIt)->getDeviceName())
            {
                std::ostringstream ss;
                ss << "LWVS does not support running on non-homogeneous GPUs during a single run. ";
                ss << "Please use the -i option to specify a list of identical GPUs. ";
                ss << "Run lwvs -g to list the GPUs on the system. Run lwvs --help for additional usage info. ";
                PRINT_ERROR("%s", "%s", ss.str().c_str());
                throw std::runtime_error(ss.str());
            }
        }
    }
}

/*****************************************************************************/
// take our GPU sets vector and fill in the appropriate GPU objects that match that set
void LwidiaValidationSuite::fillGpuSetObjs(vector<GpuSet *> gpuSets)
{
    // The rules are:
    // a) the "properties" struct is exclusionary. If properties is empty (properties.present == false)
    //    then all available GPU objects are included in the set
    // b) the "tests" vector is also exclusionary. If tests.size() == 0 then all available test 
    //    objects are included in the set

    for (unsigned int i = 0; i < gpuSets.size(); i++)
    {
        bool first_pass = true;
        if (!gpuSets[i]->properties.present) 
        {
            gpuSets[i]->gpuObjs = gpuVect;
        }
        else
        {
            gpuSets[i]->gpuObjs = decipherProperties(gpuSets[i]);
        }
        if (gpuSets[i]->gpuObjs.size() == 0)
        { // nothing matched
            std::ostringstream ss;
            ss << "Unable to match GPU set '" << gpuSets[i]->name << "' to any GPU(s) on the system.";
            PRINT_ERROR("%s", "%s", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }
    
        // ensure homogeneity
        std::string firstName = gpuSets[i]->gpuObjs[0]->getDeviceName();
        for (std::vector<Gpu *>::iterator gpuIt = gpuSets[i]->gpuObjs.begin(); gpuIt != gpuSets[i]->gpuObjs.end(); gpuIt++)
        {
            // no need to check the first but...
            if (firstName != (*gpuIt)->getDeviceName())
            {
                std::ostringstream ss;
                ss << "LWVS does not support running on non-homogeneous GPUs during a single run. ";
                ss << "Please use the -i option to specify a list of identical GPUs. ";
                ss << "Run lwvs -g to list the GPUs on the system. Run lwvs --help for additional usage info. ";
                PRINT_ERROR("%s", "%s", ss.str().c_str());
                throw std::runtime_error(ss.str());
            }
        }

        // go through the vector of tests requested and try to match them with an actual test.
        // push a warning if no match found
        for (std::vector<std::map<std::string, std::string> >::iterator reqIt = gpuSets[i]->testsRequested.begin();
             reqIt != gpuSets[i]->testsRequested.end();
             ++reqIt)
        {
            bool found = false;
            std::string requestedTestName = (*reqIt)["name"];
            std::string compareTestName = requestedTestName;
            std::transform(compareTestName.begin(), compareTestName.end(), compareTestName.begin(), ::tolower);

            // first check the test suite names
            suiteNames_enum suite;
            if (compareTestName == "quick" || compareTestName == "short")
            {
                found = true;
                suite = LWVS_SUITE_QUICK;
            } else if (compareTestName == "medium")
            {
                found = true;
                suite = LWVS_SUITE_MEDIUM;
            } else if (compareTestName == "long")
            {
                found = true;
                suite = LWVS_SUITE_LONG;
            }

            if (found)
            {
                fillTestVectors (suite, Test::LWVS_CLASS_SOFTWARE, gpuSets[i]);
                fillTestVectors (suite, Test::LWVS_CLASS_HARDWARE, gpuSets[i]);
                fillTestVectors (suite, Test::LWVS_CLASS_INTEGRATION, gpuSets[i]);
                fillTestVectors (suite, Test::LWVS_CLASS_PERFORMANCE, gpuSets[i]);
            }
            // then check the test groups
            else
            {
                if (first_pass == true)
                {
                    fillTestVectors (LWVS_SUITE_LWSTOM, Test::LWVS_CLASS_SOFTWARE, gpuSets[i]);
                    first_pass = false;
                }
                std::map<std::string, std::vector<Test *> > groups = tf->getTestGroups();
                std::map<std::string, std::vector<Test *> >::iterator it = groups.find(requestedTestName);

                if (it != groups.end())
                {
                    found = true;

                    // Add each test from the list
                    for (size_t i = 0; i < groups[requestedTestName].size(); i++)
                        gpuSets[i]->AddTestObject(LWSTOM_TEST_OBJS, groups[requestedTestName][i]);
                }
                else // now check individual tests
                {
                    for (std::vector<Test *>::iterator testIt = testVect.begin();
                         testIt != testVect.end();
                         ++testIt)
                    {
                        // colwert everything to lower case for comparison
                        std::string compareTestName = (*testIt)->GetTestName();
                        std::transform(compareTestName.begin(), compareTestName.end(), 
                                       compareTestName.begin(), ::tolower);
                        std::string lowerCaseTestName = requestedTestName;
                        std::transform(lowerCaseTestName.begin(), lowerCaseTestName.end(), 
                                       lowerCaseTestName.begin(), ::tolower);

                        if (compareTestName == lowerCaseTestName)
                        {
                            found = true;
                            // Make a full copy of the test parameters
                            TestParameters * tp = new TestParameters(*(*testIt)->getTestParameters());
                            tpVect.push_back(tp); // purely for accounting when we go to cleanup


                            whitelist->getDefaultsByDeviceId(lowerCaseTestName, 
                                                             gpuSets[i]->gpuObjs[0]->getDevicePciDeviceId(), tp);

                            if (lwvsCommon.parms.size() > 0)
                            {
                                overrideParameters(tp, lowerCaseTestName);
                            }
                            else if (!lwvsCommon.configless)
                            {
                                parser->ParseTestOverrides(lowerCaseTestName, *tp);
                            }
                            tp->AddString(PS_PLUGIN_NAME, (*testIt)->GetTestName());
                            tp->AddString(PS_LOGFILE, (*testIt)->getFullLogFileName());
                            tp->AddDouble(PS_LOGFILE_TYPE, (double)lwvsCommon.logFileType,
                                          LWVS_LOGFILE_TYPE_JSON, LWVS_LOGFILE_TYPE_BINARY);

                            (*testIt)->pushArgVectorElement(Test::LWVS_CLASS_LWSTOM, tp);
                            gpuSets[i]->AddTestObject(LWSTOM_TEST_OBJS, (*testIt));
                            break;
                        }
                    }
                }
            }

            if (!found)
            {
                std::stringstream ss;
                ss << "Warning: requested test \"" << requestedTestName <<
                      "\" was not found among possible test choices." << std::endl;
                PRINT_ERROR("%s", "%s", ss.str().c_str());
                throw std::runtime_error(ss.str());
            }
        }
    }
}

/*****************************************************************************/
void LwidiaValidationSuite::fillTestVectors(suiteNames_enum suite, Test::testClasses_enum testClass, GpuSet * set)
{
    std::vector<Test *>::iterator itSkip  = findTestName("Skip");
    int type;
    std::vector <std::string> testNames;
    bool skipIsPushed = false;


//TODO: if these out based on suite
    switch (testClass)
    {
        case Test::LWVS_CLASS_SOFTWARE:
            testNames.push_back("Blacklist");
            testNames.push_back("LWML Library");
            testNames.push_back("LWCA Main Library");
            /* Now that we link statically against lwca from the plugins, there's no need for this check. Furthermore, having it
             * makes it so lw-hostengine has to have the lwca toolkit in its LD_LIBRARY_PATH
             */
            //testNames.push_back("LWCA Toolkit Libraries");
            testNames.push_back("Permissions and OS-related Blocks");
            testNames.push_back("Persistence Mode");
            testNames.push_back("Elwironmental Variables");
            testNames.push_back("Page Retirement");
            testNames.push_back("Graphics Processes");
            testNames.push_back("Inforom");
            type = SOFTWARE_TEST_OBJS;
            break;
        case Test::LWVS_CLASS_HARDWARE:
            if (suite == LWVS_SUITE_MEDIUM || suite == LWVS_SUITE_LONG) testNames.push_back("Memory");
            if (suite == LWVS_SUITE_LONG)
                testNames.push_back(GPUBURN_PLUGIN_NAME);
            type = HARDWARE_TEST_OBJS;
            break;
        case Test::LWVS_CLASS_INTEGRATION:
            if (suite == LWVS_SUITE_MEDIUM || suite == LWVS_SUITE_LONG) testNames.push_back("PCIe");
//            if (suite == LWVS_SUITE_MEDIUM || suite == LWVS_SUITE_LONG) testNames.push_back("GPU Direct");
//            if (suite == LWVS_SUITE_MEDIUM || suite == LWVS_SUITE_LONG) testNames.push_back("P and T Limits");
            type = INTEGRATION_TEST_OBJS;
            break;
        case Test::LWVS_CLASS_PERFORMANCE:
            if (suite == LWVS_SUITE_LONG) testNames.push_back(MEMBW_PLUGIN_NAME);
            if (suite == LWVS_SUITE_LONG) testNames.push_back("SM Stress");
            if (suite == LWVS_SUITE_LONG) testNames.push_back("Targeted Stress");
            if (suite == LWVS_SUITE_LONG) testNames.push_back("Targeted Power");
            type = PERFORMANCE_TEST_OBJS;
            break;
        default:
            throw std::runtime_error("Received a test class that is not valid.");
            break;
    }

    for (vector<std::string>::iterator it = testNames.begin(); it != testNames.end(); it++)
    {
        vector<Test *>::iterator testIt;
        if (testClass != Test::LWVS_CLASS_SOFTWARE)
            testIt = findTestName(*it);
        else 
        {
            testIt = findTestName("Deployment");
            if (testIt == itSkip)
                throw std::runtime_error("The software deployment program was not properly loaded. Please check the plugin path and that the plugins are valid.");
        }
        if (testIt != itSkip)
        {
            TestParameters * tp = new TestParameters(*(*testIt)->getTestParameters());  // clone the existing one
            tpVect.push_back(tp); // purely for accounting when we go to cleanup

            if (testClass != Test::LWVS_CLASS_SOFTWARE)
            {
                // for uniformity downstream
                std::string lowerCaseTestName = *it;
                std::transform(lowerCaseTestName.begin(), lowerCaseTestName.end(), lowerCaseTestName.begin(), ::tolower);
                
                // pull just the first GPU device ID since they are all meant to be the same at this point
                whitelist->getDefaultsByDeviceId(lowerCaseTestName, set->gpuObjs[0]->getDevicePciDeviceId(), tp);

                /*Update the EUD log file path 
                if(lowerCaseTestName == EUD_PLUGIN_LF_NAME && hwDiagLogFile != "")
                    tp->SetString(EUD_LOG_FILE_PATH, hwDiagLogFile);

                if (suite == LWVS_SUITE_MEDIUM && lowerCaseTestName == EUD_PLUGIN_LF_NAME)
                    tp->SetString(EUD_PLUGIN_RUN_MODE, "medium");*/
               
                // TODO based on the suite, modify the test_duration parm 
                if (lwvsCommon.parms.size() > 0)
                    overrideParameters(tp, lowerCaseTestName);
                else if (!lwvsCommon.configless)
                    parser->ParseTestOverrides(lowerCaseTestName, *tp);
            }

            tp->AddString(PS_PLUGIN_NAME, (*it));
            tp->AddString(PS_LOGFILE, (*testIt)->getFullLogFileName());
            tp->AddDouble(PS_LOGFILE_TYPE, (double)lwvsCommon.logFileType,
                          LWVS_LOGFILE_TYPE_JSON, LWVS_LOGFILE_TYPE_BINARY);


            (*testIt)->pushArgVectorElement(testClass, tp);
            set->AddTestObject(type, *testIt);
        }
        else
        {
            TestParameters * tp = new TestParameters();
            tpVect.push_back(tp); // purely for accounting when we go to cleanup

            tp->AddString(PS_PLUGIN_NAME, (*it));

            (*testIt)->pushArgVectorElement(testClass, tp);
        }

        if (testIt == itSkip && skipIsPushed == false)
        {
            skipIsPushed = true;
            set->AddTestObject(type, *itSkip);
        }

    }
}

/*****************************************************************************/
vector<Test *>::iterator LwidiaValidationSuite::findTestName(std::string testName)
{
    vector<Test *>::iterator it;
    for (it = testVect.begin(); it != testVect.end(); it++)
    {
        if ((*it)->GetTestName() == testName)
            return it;
    }

    return --it; // returns the skip test
}

/*****************************************************************************/
vector<Gpu *> LwidiaValidationSuite::decipherProperties(GpuSet * set)
{
    // exclusionary rules:
    // 1) If pci busID exists, ignore everything else 
    // 2) If UUID exists, ignore everything else
    // 3) If index exists check for brand/name exclusions
    // 4) Otherwise match only on brand/name combos
    vector<Gpu *> tempGpuVec;

    std::vector<Gpu *>::iterator it = gpuVect.begin();
    while (it != gpuVect.end())
    {
        bool brand = false, name = false;
        if (set->properties.brand.length() > 0) brand = true;
        if (set->properties.name.length()  > 0)
        { 
            name  = true;
            // kludge to handle special naming of K10
            if (set->properties.name == "Tesla K10")
                set->properties.name = "Tesla K10.G1.8GB";
        }

        if (set->properties.uuid.length() > 0)
        {
            if (set->properties.uuid == (*it)->getDeviceGpuUuid())
                tempGpuVec.push_back(*it);   
            ++it;
            continue; // skip everything else
        } else if (set->properties.busid.length() > 0)
        {
            if (set->properties.busid == (*it)->getDevicePciBusId())
                tempGpuVec.push_back(*it);
            ++it;
            continue; // skip everything else
        } else if (set->properties.index.size() > 0)
        {
            for (unsigned int i = 0; i < set->properties.index.size(); i++)
            {
                if (set->properties.index[i] == (*it)->getDeviceIndex())
                {
                    if (!brand && !name) 
                        tempGpuVec.push_back(*it);
                    if (brand && !name &&
                        set->properties.brand == (*it)->getDeviceBrandAsString())
                        tempGpuVec.push_back(*it);
                    if (name && !brand &&
                        set->properties.name == (*it)->getDeviceName())
                        tempGpuVec.push_back(*it);
                    if (brand && name &&
                        set->properties.brand == (*it)->getDeviceBrandAsString() &&
                        set->properties.name == (*it)->getDeviceName())
                        tempGpuVec.push_back(*it);
                }
            }
        } else if (brand || name)
        {
            if (brand && !name &&
                set->properties.brand == (*it)->getDeviceBrandAsString())
                tempGpuVec.push_back(*it);
            if (name && !brand &&
                set->properties.name == (*it)->getDeviceName())
                tempGpuVec.push_back(*it);
            if (brand && name &&
                set->properties.brand == (*it)->getDeviceBrandAsString() &&
                set->properties.name == (*it)->getDeviceName())
                tempGpuVec.push_back(*it);
        }
        ++it;
    }
    
    return tempGpuVec;
}

/*****************************************************************************/
void initializeDesiredTests(const std::string &specifiedTests)
{
    if (specifiedTests.size() > 0) {
        std::vector<std::string> testNames;
        tokenizeString(specifiedTests, ",", testNames);
        for (size_t i = 0; i < testNames.size(); i++) {
            lwvsCommon.desiredTest.insert(testNames[i]);
        }
    }
}

/*****************************************************************************/
void initializeParameters(const std::string &parms)
{
    ParameterValidator pv;
    std::stringstream buf;
    buf << "Invalid Parameter String: ";

    if (parms.size() > 0)
    {
        std::vector<std::string> parmsVec;
        tokenizeString(parms, ";", parmsVec);

        for (size_t i = 0; i < parmsVec.size(); i++)
        {
            std::string testName;
            std::string parmName;
            std::string parmValue;

            size_t dot = parmsVec[i].find('.');
            size_t equals = parmsVec[i].find('=');

            if (dot != std::string::npos && equals != std::string::npos)
            {
                testName = parmsVec[i].substr(0, dot);
                parmName = parmsVec[i].substr(dot + 1, equals - (dot + 1));
                parmValue = parmsVec[i].substr(equals + 1);

                // Make sure the name is lower case
                std::transform(testName.begin(), testName.end(), testName.begin(), ::tolower);
                std::transform(parmName.begin(), parmName.end(), parmName.begin(), ::tolower);

                if (pv.IsValidTestName(testName) == false)
                {
                    buf << "test '" << testName << "' does not exist.";
                    throw std::runtime_error(buf.str());
                }

                size_t subtestDot = parmName.find('.');
                if (subtestDot != std::string::npos)
                {
                    // Found a subtest
                    std::string subtest(parmName.substr(0, subtestDot));
                    std::string subtestParm(parmName.substr(subtestDot+1));

                    if (pv.IsValidSubtest(testName, subtest) == false)
                    {
                        buf << "test '" << testName << "' has no subtest '" << subtest << "'.";
                        throw std::runtime_error(buf.str());
                    }

                    if (pv.IsValidSubtestParameter(testName, subtest, subtestParm) == false)
                    {
                        buf << "test '" << testName << "' subtest '" << subtest << "' has no parameter '"
                            << subtestParm << "'.";
                        throw std::runtime_error(buf.str());
                    }
                }
                else if (pv.IsValidParameter(testName, parmName) == false)
                {
                    buf << "test '" << testName << "' has no parameter '" << parmName << "'.";
                    throw std::runtime_error(buf.str());
                }

                lwvsCommon.parms[testName][parmName] = parmValue;
            }
            else
            {
                buf << "unable to parse test, parameter name, and value from string '" << parmsVec[i]
                    << "'. Format should be <testname>[.<subtest>].<parameter name>=<parameter value>";
                throw std::runtime_error(buf.str());
            }
        }
    }
}

/*****************************************************************************/
/* Special class to handle custom output for CL
 */

class LWVSOutput : public TCLAP::StdOutput
{
public:
    virtual void usage (TCLAP::CmdLineInterface & _cmd)
    {
        TCLAP::StdOutput::usage(_cmd);

        std::cout << "Please email lwdatools@lwpu.com with any questions, bug reports, etc." << std::endl << std::endl;
    }

};

/*****************************************************************************/
/* Process command line arguments and use those arguments to override anything specified
 * by the config file. 
 */
void LwidiaValidationSuite::processCommandLine(int argc, char *argv[])
{
    std::string configFileArg;

    try
    {
        TCLAP::CmdLine cmd(LWVS_NAME, ' ', DRIVER_VERSION);
        LWVSOutput lwout;
        cmd.setOutput(&lwout);
        // add this so it displays as part of help but it is effectively ignored
        TCLAP::SwitchArg verboseArg("v", "verbose", "Enable verbose reporting for some plugins.", cmd, false);
        TCLAP::SwitchArg listTestsArg("t", "listTests", "List the tests available to be exelwted through LWVS.", cmd, false);
        TCLAP::SwitchArg statsOnFailArg("", "statsonfail", "Output statistic logs only if a test failure is encountered.", cmd, false);
        TCLAP::ValueArg<std::string> hwdiaglogfileArg("", "hwdiaglogfile", "Encrypted HW diagnostics log file. Append this to save the HW diagnostics logs at any specified location.\
            If path is not specified then \"lwpu-diagnostic.log\" is the default logfile .", false, "", "HW Diagnostics log file", cmd);
        TCLAP::ValueArg<std::string> specificTestArg("", "specifiedtest", "Run a specific test in a configless mode. Multiple word tests should be in quotes.",
            false, "", "specific test to run", cmd);
        TCLAP::SwitchArg parseArg("s", "scriptable", "Give output in colon-separated, more script-friendly format.", cmd, false);
        TCLAP::SwitchArg quietModeArg("", "quiet", "No console output given.  See logs and return code for errors.", cmd, false);
        TCLAP::ValueArg<std::string> pluginPathArg("p", "pluginpath", "Specify a custom path for the LWVS plugins.",
            false, "", "path to plugins", cmd);
        TCLAP::ValueArg<std::string> debugFileArg("l", "debugLogFile", "Encrypted logfile for debug information. If a debug level \
            has been specified then \"lwvs.log\" is the default logfile.", 
            false, "", "debug file", cmd);
        TCLAP::ValueArg<std::string> indexArg("i", "indexes", "Comma separated list of indexes to run LWVS on.", false, "", "indexes", cmd);
        TCLAP::SwitchArg listGpusArg("g", "listGpus", "List the GPUS available.", cmd, false);
        TCLAP::ValueArg<unsigned int> debugLevelArg("d", "debugLevel", "Debug level 0-5 with 5 being the most verbose.  The logfile \
            can be specified by the --debugLogFile parameter.", 
            false, 0, "debug level", cmd);
        TCLAP::SwitchArg configLessArg("", "configless", "Run LWVS in a configless mode.  Exelwtes a \"long\" test on all supported GPUs.", cmd, false);
        TCLAP::ValueArg<std::string> configArg("c", "config", "Specify a path to the configuration file.",
            false, "", "path to config file", cmd);
        TCLAP::SwitchArg appendModeArg("a", "appendLog", "Append this run to the current debug log file. See the --debugLogFile parameter.", cmd, false);
        TCLAP::ValueArg<std::string> statsPathArg("", "statspath", "Write the plugin statistics to a given path rather than the current directory.",
                                                  false, "", "plugin statistics path", cmd);
        TCLAP::ValueArg<std::string> parms("", "parameters", "Specify test parameters in a configless mode.", false, "", "parameters to set for tests", cmd);
        TCLAP::SwitchArg jsonArg("j", "jsonOutput", "Format output as json. Note: prevents progress updates.", cmd, false);
        TCLAP::ValueArg<unsigned int> initializationWaitTime("w", "initwaittime",
                "Number of seconds to wait before aborting DCGM initialization", false, 120,
                "initialization wait time", cmd);
        TCLAP::ValueArg<std::string> dcgmHost("", "dcgmHostname", "Specify the hostname where DCGM is running.", false, "", "DCGM hostname", cmd);
        TCLAP::SwitchArg fromDcgmArg("z", "from-dcgm",
                "Specify that this was launched by dcgmi diag and not from ilwoking lwvs directly", cmd, false);
        TCLAP::SwitchArg trainArg("", "train",
                                  "Train LWVS to generate golden values for this system's configuration.", cmd,
                                  false);
        TCLAP::SwitchArg forceTrainArg("", "force",
                                       "Train LWVS for golden values despite warnings to the contrary", cmd,
                                       false);
        TCLAP::ValueArg<unsigned int> trainingIterations("", "training-iterations", "The number of iterations to "
                                                             "use while training the diagnostic. The default is "
                                                             "4.", false, 4, "training iterations", cmd);
        TCLAP::ValueArg<unsigned int> trainingVariance("", "training-variance", "The amount of variance - after "
                                                           "normalizing the data - required to trust the data. "
                                                           "The default is 5", false, 5, "training variance", cmd);
        TCLAP::ValueArg<unsigned int> trainingTolerance("", "training-tolerance", "The percentage the golden "
                                                            "value should be scaled to allow some tolerance when "
                                                            "running the diagnostic later. For example, if the "
                                                            "callwlated golden value for a minimum bandwidth were "
                                                            "9000 and the tolerance were set to 5, then the "
                                                            "minimum bandwidth written to the configuration file "
                                                            "would be 8550, 95% of 9000. The default value is 5.",
                                                            false, 5, "training tolerance", cmd);
        TCLAP::ValueArg<std::string> goldelwaluesFile("", "golden-values-filename", "Specify the path where the "
                                                      "DCGM GPU diagnostic should save the golden values file "
                                                      "produced in training mode.", false,
                                                      "/tmp/golden_values.yml", "path to golden values file", cmd);

        TCLAP::ValueArg<std::string> throttleMask("", "throttle-mask", 
            "Specify which throttling reasons should be ignored. You can provide a comma separated list of reasons. "
            "For example, specifying 'HW_SLOWDOWN,SW_THERMAL' would ignore the HW_SLOWDOWN and SW_THERMAL throttling "
            "reasons. Alternatively, you can specify the integer value of the ignore bitmask. For the bitmask, "
            "multiple reasons may be specified by the sum of their bit masks. For "
            "example, specifying '40' would ignore the HW_SLOWDOWN and SW_THERMAL throttling reasons.\n"
            "Valid throttling reasons and their corresponding bitmasks (given in parentheses) are:\n"
            "HW_SLOWDOWN (8)\nSW_THERMAL (32)\nHW_THERMAL (64)\nHW_POWER_BRAKE (128)", false, "", 
            "throttle reasons to ignore", cmd);

        TCLAP::SwitchArg failEarly("", "fail-early",
            "Enable early failure checks for the Targeted Power, Targeted Stress, SM Stress, and Diagnostic tests. "
            "When enabled, these tests check for a failure once every 5 seconds (can be modified by the "
            "--check-interval parameter) while the test is running instead of a single check performed after the "
            "test is complete. Disabled by default.", cmd, false);

        TCLAP::ValueArg<unsigned int> failCheckInterval("", "check-interval",
            "Specify the interval (in seconds) at which the early failure checks should occur for the "
            "Targeted Power, Targeted Stress, SM Stress, and Diagnostic tests when early failure checks are enabled. "
            "Default is once every 5 seconds. Interval must be between 1 and 300", false, 5, "failure check interval",
            cmd);


        cmd.parse(argc, argv);

        configFileArg = configArg.getValue();
        if (configFileArg.size() > 0)
        {
            configFile = configFileArg;
        }

        listGpus = listGpusArg.getValue();
        listTests = listTestsArg.getValue();
        lwvsCommon.verbose = verboseArg.getValue();
        lwvsCommon.pluginPath = pluginPathArg.getValue();
        lwvsCommon.parse = parseArg.getValue();
        lwvsCommon.quietMode = quietModeArg.getValue();
        lwvsCommon.configless = configLessArg.getValue();
        lwvsCommon.statsOnlyOnFail = statsOnFailArg.getValue();
        lwvsCommon.indexString = indexArg.getValue();
        lwvsCommon.jsonOutput = jsonArg.getValue();
        lwvsCommon.dcgmHostname = dcgmHost.getValue();
        lwvsCommon.fromDcgm = fromDcgmArg.getValue();
        lwvsCommon.training = trainArg.getValue();
        lwvsCommon.forceTraining = forceTrainArg.getValue();
        lwvsCommon.trainingIterations = trainingIterations.getValue();
        lwvsCommon.trainingVariancePcnt = trainingVariance.getValue() / 100.0;
        lwvsCommon.trainingTolerancePcnt = trainingTolerance.getValue() / 100.0;
        lwvsCommon.goldelwaluesFile = goldelwaluesFile.getValue();
        lwvsCommon.SetStatsPath(statsPathArg.getValue());

        this->initWaitTime = initializationWaitTime.getValue();

        if (listGpus || listTests)
            lwvsCommon.configless = true;

        if (lwvsCommon.desiredTest.size() > 0)
            lwvsCommon.configless = true;

        initializeDesiredTests(specificTestArg.getValue());
        initializeParameters(parms.getValue());

        switch (debugLevelArg.getValue())
        {
            case LWML_DBG_CRITICAL:
                lwosSetElw(LWVS_ELW_DBG_LVL, "CRITICAL");
                break;
        
            case LWML_DBG_ERROR:
                lwosSetElw(LWVS_ELW_DBG_LVL, "ERROR");
                break;
        
            case LWML_DBG_WARNING:
                lwosSetElw(LWVS_ELW_DBG_LVL, "WARNING");
                break;
            case LWML_DBG_INFO:
                lwosSetElw(LWVS_ELW_DBG_LVL, "INFO");
                break;
            
            default:
            case LWML_DBG_DEBUG:
                lwosSetElw(LWVS_ELW_DBG_LVL, "DEBUG");
                break;
        }

        appendMode = appendModeArg.getValue();
        if(appendMode)
            lwosSetElw(LWVS_ELW_DBG_APPEND, "1");

        debugFile = debugFileArg.getValue();
        if(debugFile.size() > 0)
            lwosSetElw(LWVS_ELW_DBG_FILE, debugFile.c_str());
        
        if(hwdiaglogfileArg.isSet())
            hwDiagLogFile = hwdiaglogfileArg.getValue();

        if (debugLevelArg.getValue() > 0 && debugFile == "")
            debugFile = "lwvs.log";

        // Set bitmask for ignoring user specified throttling reasons
        if (throttleMask.isSet())
        {
            std::string reasonStr = throttleMask.getValue();
            // Make reasonStr lower case for parsing
            std::transform(reasonStr.begin(), reasonStr.end(), reasonStr.begin(), ::tolower);
            lwvsCommon.throttleIgnoreMask = GetThrottleIgnoreReasonMaskFromString(reasonStr);
        }

        // Enable early failure checks if requested
        if (failEarly.isSet())
        {
            lwvsCommon.failEarly = true;
            if (failCheckInterval.isSet())
            {
                lwvsCommon.failCheckInterval = failCheckInterval.getValue();
            }
        }

    }
    catch (TCLAP::ArgException &e)
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; 
        throw std::runtime_error ("An error oclwrred trying to parse the command line.");
    }
}
