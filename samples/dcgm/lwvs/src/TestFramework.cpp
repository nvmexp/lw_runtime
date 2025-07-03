#include "TestFramework.h"
#include "PluginStrings.h"
#include "Gpu.h"
#include <string>
#include <iostream>
#include <vector>
#include <list>
#include <stdexcept>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include "JsonOutput.h"
#include "common.h"
#include "DcgmSystem.h"
#include "DcgmHandle.h"

extern DcgmSystem dcgmSystem;
extern DcgmHandle dcgmHandle;

// globals
static unsigned int TEST_DL_BUF_SIZE=1024;
extern "C" {
    std::map<std::string, maker_t *, less<string> > factory;
}

/*****************************************************************************/
TestFramework::TestFramework() : testList(), testGroup(), dlList(), skipRest(false), m_goldelwalues(),
                                 m_validGpuId(0)
{
    output = new Output();
}

/*****************************************************************************/
TestFramework::TestFramework(bool jsonOutput, std::vector<unsigned int> &gpuIndices) : testList(), testGroup(),
                                                                                       dlList(), output(0),
                                                                                       skipRest(false)
{
    if (jsonOutput)
    {
        output = new JsonOutput(gpuIndices);
    }
    else
    {
        output = new Output();
    }

    if (!gpuIndices.empty())
    {
        m_validGpuId = gpuIndices[0];
    }
    else
    {
        // This should never happen
        m_validGpuId = 0;
    }
}

/*****************************************************************************/
TestFramework::~TestFramework()
{
    // close the plugins
    for (vector<Test *>::iterator itr = testList.begin(); itr != testList.end(); itr++)
        if (*itr) delete (*itr);

    // close the shared libs
    for (list<void*>::iterator itr = dlList.begin(); itr != dlList.end(); itr++)
        if (*itr) dlclose (*itr);

    delete output;
}

std::string TestFramework::GetPluginBaseDir()
{
    char szTmp[32] = {0};
    char buf[1024] = {0};
    std::string binaryPath;
    std::string pluginsPath;
    std::vector <std::string> searchPaths;
    std::vector <std::string>::iterator pathIt;
    struct stat statBuf = { 0 };

    snprintf(szTmp, sizeof(szTmp), "/proc/%d/exe", getpid());
    if (readlink(szTmp, buf, sizeof(buf) - 1) <= 0)
    {
        std::stringstream ss;
        ss << "Unable to read lwvs binary path from /proc: " << strerror(errno);
        throw std::runtime_error(ss.str());
    }

    // out starting point is the binary path... plugins should be in a relative path to this
    binaryPath = buf; 
    if (stat(buf, &statBuf))
    {
        std::stringstream errBuf;
        errBuf << "Cannot stat LWVS binary '" << buf << "' : '" << strerror(errno)
               << "', so we cannot selwrely load the plugins.";
        PRINT_ERROR("%s", "%s", errBuf.str().c_str());
        throw std::runtime_error(errBuf.str());
    }

    m_lwvsBinaryMode = statBuf.st_mode;
    m_lwvsOwnerUid = statBuf.st_uid;
    m_lwvsOwnerGid = statBuf.st_gid;

    if (lwvsCommon.pluginPath.size() == 0) 
    {
        binaryPath = binaryPath.substr(0, binaryPath.find_last_of("/"));
        
        searchPaths.push_back("/plugins");
        searchPaths.push_back("/../share/lwpu-validation-suite/plugins");

        for (pathIt = searchPaths.begin(); pathIt != searchPaths.end(); pathIt++)
        {
            pluginsPath = binaryPath + (*pathIt);
            PRINT_DEBUG("%s", "Searching %s for plugins.", pluginsPath.c_str());
            if  (access(pluginsPath.c_str(), 0) == 0)
            {
                struct stat status;
                stat(pluginsPath.c_str(), &status);

                if (!(status.st_mode & S_IFDIR)) // not a dir
                    continue;
                else
                    break;
            }
        }
        if (pathIt == searchPaths.end())
            throw std::runtime_error ("Plugins directory was not found.  Please check paths or use -p to set it.");
    }
    else 
        pluginsPath = lwvsCommon.pluginPath;

    return pluginsPath;
}

std::string TestFramework::GetPluginDirExtension() const
{
    const int UNIVERSAL_MIN_MAJOR_VERSION = 418;
    const int PARTIAL_MIN_MAJOR_VERSION = 410;
    const int PARTIAL_MIN_MINOR_VERSION = 72;
    static const std::string LWDA_10_EXTENSION("/lwda10/");
    static const std::string LWDA_9_EXTENSION("/lwda9/");

    // Check driver version
    dcgmDeviceAttributes_t attrs;
    memset(&attrs, 0, sizeof(attrs));
    attrs.version = dcgmDeviceAttributes_version1;
    dcgmReturn_t ret = dcgmSystem.GetDeviceAttributes(m_validGpuId, attrs);

    if (ret != DCGM_ST_OK)
    {
        char buf[1024];
        snprintf(buf, sizeof(buf), "Cannot determine GPU driver version: '%s'",
                 dcgmHandle.RetToString(ret).c_str());
        PRINT_ERROR("%s", "%s", buf);
        throw std::runtime_error(buf);
    }

    char *next;
    long minorVersion = 0;
    long majorVersion = strtol(attrs.identifiers.driverVersion, &next, 10);

    if (next != NULL && *next == '.')
    {
        next++;
        minorVersion = strtol(next, NULL, 10);
    }

    // Determine the extension based on the major / minor versions
    if (majorVersion >= UNIVERSAL_MIN_MAJOR_VERSION)
    {
        return LWDA_10_EXTENSION;
    }
    else if (majorVersion >= PARTIAL_MIN_MAJOR_VERSION && minorVersion >= PARTIAL_MIN_MINOR_VERSION)
    {
        return LWDA_10_EXTENSION;
    }
    else
    {
        return LWDA_9_EXTENSION;
    }
}

std::string TestFramework::GetPluginDir()
{
    return GetPluginBaseDir() + GetPluginDirExtension();
}

void TestFramework::LoadLibrary(const char *libraryPath, const char *libraryName)
{
    void *dlib = dlopen(libraryPath, RTLD_LAZY);
    if (dlib == NULL)
    {
        std::stringstream ss;
        std::string dlopen_error = dlerror();
        ss << "Unable to open plugin " << libraryName << " due to: " << dlopen_error << std::endl;
        PRINT_ERROR("%s", "%s", ss.str().c_str());
    }
    else
    {
        dlList.insert(dlList.end(), dlib);
        PRINT_DEBUG("%s", "Successfully loaded dlib %s", libraryName);
    }
}

/*****************************************************************************/
bool TestFramework::PluginPermissionsMatch(const std::string &pluginDir, const std::string &plugin)
{
    bool match = false;
    struct stat statBuf;

    if (stat(plugin.c_str(), &statBuf))
    {
        PRINT_ERROR("%s %s %s", "Not loading plugin '%s' in dir '%s' because I cannot stat the file : '%s'",
                    plugin.c_str(), pluginDir.c_str(), strerror(errno));
    }
    else if (m_lwvsBinaryMode != statBuf.st_mode)
    {
        PRINT_ERROR("%s %s %o %o", "Not loading plugin '%s' in dir '%s' because its permissions '%o' do not match "
                    "the diagnostic's : '%o'",
                    plugin.c_str(), pluginDir.c_str(), statBuf.st_mode, m_lwvsBinaryMode);
    }
    else if (m_lwvsOwnerUid != statBuf.st_uid)
    {
        PRINT_ERROR("%s %s %u %u", "Not loading plugin '%s' in dir '%s' because its owner uid '%u' does not match "
                    "the diagnostic's : '%u'",
                    plugin.c_str(), pluginDir.c_str(), statBuf.st_uid, m_lwvsOwnerUid);
    }
    else if (m_lwvsOwnerGid != statBuf.st_gid)
    {
        PRINT_ERROR("%s %s %o %o", "Not loading plugin '%s' in dir '%s' because its owner gid '%o' does not match "
                    "the diagnostic's : '%o'",
                    plugin.c_str(), pluginDir.c_str(), statBuf.st_gid, m_lwvsOwnerGid);
    }
    else
    {
        match = true;
    }

    return match;
}

/*****************************************************************************/
void TestFramework::loadPlugins()
{
    // plugin discovery
    char oldPath[2048] = {0};
    std::string pluginDir = GetPluginDir();
    struct dirent *dirent = 0;
    DIR           *dir;
    std::string    dotSo(".so");
    std::map<string, maker_t *, std::less<std::string> >::iterator fitr;
    std::stringstream errbuf;

    if (getcwd(oldPath, sizeof(oldPath)) == 0)
    {
        errbuf << "Cannot load plugins: unable to get current dir: '" << strerror(errno) << "'";
        PRINT_ERROR("%s", "%s", errbuf.str().c_str());
        throw std::runtime_error(errbuf.str());
    }

    if (chdir(pluginDir.c_str()))
    {
        errbuf << "Cannot load plugins: unable to change to the plugin dir '" << pluginDir
               << "': '" << strerror(errno) << "'";
        PRINT_ERROR("%s", "%s", errbuf.str().c_str());
        throw std::runtime_error(errbuf.str());
    }

    // Load the pluginCommon library first so that those libraries can find the appropriate symbols
    LoadLibrary("./libpluginCommon.so", "libpluginCommon.so");

    if ((dir = opendir(".")) == 0)
    {
        errbuf << "Cannot load plugins: unable to open the current dir '" << pluginDir
               << "': '" << strerror(errno) << "'";
        PRINT_ERROR("%s", "%s", errbuf.str().c_str());
        throw std::runtime_error(errbuf.str());
    }

    // Read the entire directory and get the .sos
    while ((dirent = readdir(dir)) != NULL)
    {
        // Skip files that don't end in .so
        char *dot = strrchr(dirent->d_name, '.');
        if (dot == NULL || dotSo != dot)
            continue;

        // Tell dlopen to look in this directory
        char buf[2048];
        snprintf(buf, sizeof(buf), "./%s", dirent->d_name);

        // Do not load any plugins with different permissions / owner than the diagnostic binary
        if (!PluginPermissionsMatch(pluginDir, buf))
            continue;

        LoadLibrary(buf, dirent->d_name);
    }

    chdir(oldPath);

    for (fitr = factory.begin(); fitr != factory.end(); fitr++)
    {
        Test * temp = new Test(factory[fitr->first]());
        temp->SetOutputObject(output);
        testList.insert(testList.end(), temp);
        insertIntoTestGroup(temp->getTestGroup(), temp);
    }

    Test * skipTest = new Test(NULL);
    testList.insert(testList.end(), skipTest); // put a dummy test for skipping at the end
}

/*****************************************************************************/
void TestFramework::insertIntoTestGroup(std::string groupName, Test * testObject)
{
    // groupName may be a CSV list
    std::istringstream ss(groupName);
    std::string token;

    while(std::getline(ss, token, ',')) {
        while (token[0] == ' ')
        {
            token.erase(token.begin());
        }
        testGroup[token].push_back(testObject);
    }
}

/*****************************************************************************/
void TestFramework::go(vector<GpuSet *> gpuset)
{
    // iterate through all GPU sets
    std::vector<GpuSet *>::iterator setItr;
    for (setItr = gpuset.begin(); setItr != gpuset.end(); setItr++)
    {
        std::vector<Test *> testList;
        std::vector<Gpu *>  gpuList = (*setItr)->gpuObjs;

        testList = (*setItr)->m_softwareTestObjs;
        if (testList.size() > 0)
            goList(Test::LWVS_CLASS_SOFTWARE, testList, gpuList);

        testList = (*setItr)->m_hardwareTestObjs;
        if (testList.size() > 0)
            goList(Test::LWVS_CLASS_HARDWARE, testList, gpuList);

        testList = (*setItr)->m_integrationTestObjs;
        if (testList.size() > 0)
            goList(Test::LWVS_CLASS_INTEGRATION, testList, gpuList);

        testList = (*setItr)->m_performanceTestObjs;
        if (testList.size() > 0)
            goList(Test::LWVS_CLASS_PERFORMANCE, testList, gpuList);

        testList = (*setItr)->m_lwstomTestObjs;
        if (testList.size() > 0)
            goList(Test::LWVS_CLASS_LWSTOM, testList, gpuList);
    }
    
    if (!lwvsCommon.training)
        output->print();
}

/*****************************************************************************/
void TestFramework::CallwlateAndSaveGoldelwalues()
{
    if (lwvsCommon.training)
    {
        // filename becomes a parameter in the next check in
        dcgmReturn_t ret = m_goldelwalues.CallwlateAndWriteGoldelwalues(lwvsCommon.goldelwaluesFile);
        char buf[512];

        if (ret == DCGM_ST_OK)
        {
            snprintf(buf, sizeof(buf), "Successfully trained the diagnostic. The golden values file is here %s\n",
                     lwvsCommon.goldelwaluesFile.c_str());
        }
        else
        {
            snprintf(buf, sizeof(buf), "ERROR in training : %s\n", errorString(ret));
        }

        output->AddTrainingResult(buf);
        output->print();
    }
}

/*****************************************************************************/
void TestFramework::ReportTrainingError(Test *test)
{
    std::stringstream msg;
    msg << "Unable to complete training due to failure in " << test->GetTestName() << ": \n";
    std::vector<std::string> warnings = test->GetWarnings();
    for (size_t i = 0; i < warnings.size(); i++)
        msg << warnings[i] << "\n";
    throw std::runtime_error (msg.str());
}

/*****************************************************************************/
void TestFramework::EvaluateTestTraining(Test *test)
{
    if (test->GetResult() == LWVS_RESULT_FAIL)
    {
        ReportTrainingError(test);
    }
    else
    {
        m_goldelwalues.RecordGoldelwalueInputs(test->GetTestName(), test->GetObservedMetrics());
    }
}

/*****************************************************************************/
void TestFramework::GetAndOutputHeader(Test::testClasses_enum classNum)
{
    // Don't do anything if we're in training mode or the legacy parse mode
    if (lwvsCommon.parse || lwvsCommon.training)
        return;

    std::string header;
    switch (classNum)
    {
        case Test::LWVS_CLASS_SOFTWARE:
            header = "Deployment";
            break;
        case Test::LWVS_CLASS_HARDWARE:
            header = "Hardware";
            break;
        case Test::LWVS_CLASS_INTEGRATION:
            header = "Integration";
            break;
        case Test::LWVS_CLASS_PERFORMANCE:
            header = "Stress";
            break;
        case Test::LWVS_CLASS_LWSTOM:
        default:
            header = "Custom";
            break;
    }
    output->header(header);
}

/*****************************************************************************/
void TestFramework::goList(Test::testClasses_enum classNum, std::vector<Test *> testList, std::vector<Gpu *> gpuList)
{
    GetAndOutputHeader(classNum);

    // iterate through all tests giving them the GPU objects needed
    for (std::vector<Test *>::iterator testItr = testList.begin(); testItr != testList.end(); testItr++)
    {
        Test * test = (*testItr); // readability

        unsigned int vecSize = test->getArgVectorSize(classNum);
        for (unsigned int i = 0; i < vecSize; i++)
        {
            TestParameters * tp = test->popArgVectorElement(classNum);
            std::string name = tp->GetString(PS_PLUGIN_NAME);

            if ((test->getTestParallel() && !lwvsCommon.serialize)
                || classNum == Test::LWVS_CLASS_SOFTWARE)
            {
                output->prep(name);
                if (!skipRest && !main_should_stop)
                {
                    if (classNum == Test::LWVS_CLASS_SOFTWARE)
                    {
                        if (!lwvsCommon.requirePersistenceMode)
                            tp->SetString(SW_STR_REQUIRE_PERSISTENCE , "False");
                        if (name == "Blacklist")
                            tp->SetString(SW_STR_DO_TEST, "blacklist");
                        else if (name == "LWML Library")
                            tp->SetString(SW_STR_DO_TEST, "libraries_lwml");
                        else if (name == "LWCA Main Library")
                            tp->SetString(SW_STR_DO_TEST, "libraries_lwda");
                        else if (name == "LWCA Toolkit Libraries")
                            tp->SetString(SW_STR_DO_TEST, "libraries_lwdatk");
                        else if (name == "Permissions and OS-related Blocks")
                            tp->SetString(SW_STR_DO_TEST, "permissions");
                        else if (name == "Persistence Mode")
                            tp->SetString(SW_STR_DO_TEST, "persistence_mode");
                        else if (name == "Elwironmental Variables")
                            tp->SetString(SW_STR_DO_TEST, "elw_variables");
                        else if (name == "Page Retirement")
                            tp->SetString(SW_STR_DO_TEST, "page_retirement");
                        else if (name == "Graphics Processes")
                            tp->SetString(SW_STR_DO_TEST, "graphics_processes");
                        else if (name == "Inforom")
                            tp->SetString(SW_STR_DO_TEST, "inforom");
                    }
                    test->go(tp, gpuList);
                    if (lwvsCommon.training)
                    {
                        EvaluateTestTraining(test);
                    }
                    else
                    {
                        output->Result(test->GetResults(), test->GetWarnings(), test->GetErrors(),
                                       test->GetGpuErrors(), test->GetVerboseInfo(), test->GetGpuVerboseInfo());
                    }
                }
                else
                {
                    /* If the test hasn't been run (test->go() was not called), test->GetResults() returns 
                     * empty results, which is treated as the test being skipped.
                     */
                    output->Result(test->GetResults(), test->GetWarnings(), test->GetErrors(),
                                   test->GetGpuErrors(), test->GetVerboseInfo(), test->GetGpuVerboseInfo());
                }
            }
            else
            {
                std::vector<Gpu *>::iterator gpuItr;
                for (gpuItr = gpuList.begin(); gpuItr != gpuList.end(); gpuItr++)
                {
                    std::ostringstream ss;  // no std::to_string()
                    ss << (*gpuItr)->getDeviceIndex();
                    std::string newName(name);

                    newName.append(" GPU");
                    newName.append(ss.str());
                    output->prep (newName);

                    if (!skipRest && !main_should_stop)
                    {
                        if (test->getTestParallel() && lwvsCommon.serialize)
                        {
                            std::vector<Gpu *> tempList;
                            tempList.push_back(*gpuItr);
                            test->go(tp, tempList);
                        }
                        else 
                            test->go(tp, *gpuItr);

                        if (lwvsCommon.training)
                        {
                            EvaluateTestTraining(test);
                        }
                        else
                        {
                            output->Result(test->GetResults(), test->GetWarnings(), test->GetErrors(),
                                           test->GetGpuErrors(), test->GetVerboseInfo(),
                                           test->GetGpuVerboseInfo());
                        }
                    }
                    else
                    {
                        /* If the test hasn't been run (test->go() was not called), test->GetResults() returns 
                         * empty results, which is treated as the test being skipped.
                         */
                        output->Result(test->GetResults(), test->GetWarnings(), test->GetErrors(),
                                       test->GetGpuErrors(), test->GetVerboseInfo(), test->GetGpuVerboseInfo());
                    }
                }
            }
            PRINT_DEBUG("%s %d %d", "test %s had overall result %d. configless is %d", 
                        name.c_str(), test->GetResult(), lwvsCommon.configless);
            if (test->GetResult() == LWVS_RESULT_FAIL && !lwvsCommon.configless)
            {
                skipRest = true;
            }
        }
    }
}

void TestFramework::addInfoStatement(const std::string &info)
{
    output->addInfoStatement(info);
}

