/* 
 * A base class responsible for looking for and loading all LWVS plugins.  Though
 * the Plugin objects are instantiated here through the factory interface, they are
 * destroyed as part of the Test destructor.  Test objects and the dynamic library
 * closing is done as part of this destructor.
 */
#ifndef _LWVS_LWVS_TestFramework_H
#define _LWVS_LWVS_TestFramework_H

#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "Test.h"
#include "GpuSet.h"
#include "Output.h"
#include "GoldelwalueCallwlator.h"

class TestFramework
{
/***************************PUBLIC***********************************/
public:
    TestFramework();
    TestFramework(bool jsonOutput, std::vector<unsigned int> &gpuIndices);
    ~TestFramework();   

    // methods
    void go(std::vector<GpuSet *> gpuset);
    void loadPlugins();
    void addInfoStatement(const std::string &info);
   
    // Getters
    vector<Test *> getTests() 
    {
        return testList;
    }

    std::map<std::string, std::vector<Test *> > getTestGroups()
    {
        return testGroup;
    }

    void CallwlateAndSaveGoldelwalues();

/***************************PRIVATE**********************************/
private:
    std::vector<Test *> testList;
    std::map<std::string, std::vector<Test *> > testGroup;
    std::list<void *> dlList;
    Output * output;
    bool skipRest;
    mode_t m_lwvsBinaryMode;
    uid_t  m_lwvsOwnerUid;
    gid_t  m_lwvsOwnerGid;
    // Recorded metrics from each test run - used only in training mode
    GoldelwalueCallwlator          m_goldelwalues;
    unsigned int m_validGpuId;

    //methods
    void insertIntoTestGroup(std::string, Test*);
    void goList(Test::testClasses_enum suite, std::vector<Test *> testList, std::vector <Gpu *> gpuList);
    void LoadLibrary(const char *libPath, const char *libName);
    void ReportTrainingError(Test *test);
    void EvaluateTestTraining(Test *test);
    void GetAndOutputHeader(Test::testClasses_enum classNum);
    
    /********************************************************************/
    std::string GetPluginDir();
    
    /********************************************************************/
    /*
     * Determines and returns the base directory for the plugins, and gets the 
     * appropriate permissions to check against later
     */
    std::string GetPluginBaseDir();

    /********************************************************************/
    /*
     * Checks the driver version and returns /lwda9/ or /lwda10/ to load the correct plugins
     */
    std::string GetPluginDirExtension() const;

    /********************************************************************/
    /*
     * Returns true if the plugin has the same permissions and owner as the LWVS binary
     */
    bool PluginPermissionsMatch(const std::string &pluginDir, const std::string &plugin);

/***************************PROTECTED********************************/
protected:
};

#endif //  _LWVS_LWVS_TestFramework_H


