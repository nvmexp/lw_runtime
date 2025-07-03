/* The Test class is a base interface to a plugin two primary goals:
 *    1   Obfuscate the plugin class from the rest of LWVS thus most
 *        calls are simply passed through to the corresponding plugin.
 *        But that allows plugins to be compiled completely separate.
 *    2   Catch exceptions thrown by the plugin and make sure they
 *        do not make it all the way up and kill all of LWVS.
 */
#ifndef _LWVS_LWVS_TEST_H
#define _LWVS_LWVS_TEST_H

#include "common.h"
#include "Gpu.h"
#include "Plugin.h"
#include "TestParameters.h"
#include <string>
#include <vector>
#include <list>
#include <iostream>

class Test
{
/***************************PUBLIC***********************************/
public:
    Test() {}
    Test(Plugin *);
    ~Test();

    enum testClasses_enum
    {
        LWVS_CLASS_SOFTWARE,
        LWVS_CLASS_HARDWARE,
        LWVS_CLASS_INTEGRATION,
        LWVS_CLASS_PERFORMANCE,
        LWVS_CLASS_LWSTOM,
        LWVS_CLASS_LAST
    };

    void go(TestParameters *);
    void go(TestParameters *, Gpu *);
    void go(TestParameters *, std::vector<Gpu *>);

    std::string GetTestName()
    {
        return m_infoStruct.name;
    }

    std::string getTestDesc()
    {
        return m_infoStruct.shortDescription;
    }

    std::string getTestGroup()
    {
        return m_infoStruct.testGroups;
    }

    bool getTestParallel()
    {
        return m_infoStruct.selfParallel;
    }

    TestParameters * getTestParameters()
    {
        return m_infoStruct.defaultTestParameters;
    }

    unsigned int getArgVectorSize(testClasses_enum num)
    {
        return (m_argMap[num]).size();
    }

    void pushArgVectorElement(testClasses_enum num, TestParameters *obj)
    {
        m_argMap[num].push_back(obj);
    }

    TestParameters * popArgVectorElement(testClasses_enum num)
    {
        if (m_argMap.find(num) == m_argMap.end())
            return 0;

        TestParameters *tp = m_argMap[num].front();
        if (tp)
            m_argMap[num].erase(m_argMap[num].begin());
        return tp;
    }

    // pass throughs to plugin
    void SetOutputObject(Output *obj)
    {
        m_plugin->SetOutputObject(obj);
    }

    lwvsPluginResult_t GetResult()
    {
        if (m_skipTest)
        {
            return LWVS_RESULT_SKIP;
        }
        if (m_plugin)
        {
            return m_plugin->GetResult();
        }
        return LWVS_RESULT_FAIL;
    }

    const lwvsPluginGpuResults_t& GetResults() const
    {
        if (m_skipTest || m_plugin == NULL)
        {
            return m_emptyGpuResults;
        }
        return m_plugin->GetGpuResults();
    }

    const std::vector<std::string>& GetWarnings() const
    {
        if (m_skipTest || m_plugin == NULL)
        {
            return m_emptyMessages;
        }
        return m_plugin->GetWarnings();
    }

    const lwvsPluginGpuMessages_t& GetGpuWarnings() const
    {
        if (m_skipTest || m_plugin == NULL)
        {
            return m_emptyGpuMessages;
        }
        return m_plugin->GetGpuWarnings();
    }

    const std::vector<DcgmError> &GetErrors() const
    {
        return m_plugin->GetErrors();
    }

    const lwvsPluginGpuErrors_t &GetGpuErrors() const
    {
        return m_plugin->GetGpuErrors();
    }

    const std::vector<std::string>& GetVerboseInfo() const
    {
        if (m_skipTest || m_plugin == NULL)
        {
            return m_emptyMessages;
        }
        return m_plugin->GetVerboseInfo();
    }

    const lwvsPluginGpuMessages_t& GetGpuVerboseInfo() const
    {
        if (m_skipTest || m_plugin == NULL)
        {
            return m_emptyGpuMessages;
        }
        return m_plugin->GetGpuVerboseInfo();
    }

    //Get per-test log file tag to distinguish tests' log files from each other
    std::string getLogFileTag()
    {
        return m_infoStruct.logFileTag;
    }

    observedMetrics_t GetObservedMetrics() const
    {
        return m_plugin->GetObservedMetrics();
    }

    //Get the full filename of the log file to write for this test
    std::string getFullLogFileName();

/***************************PRIVATE**********************************/
private:
    /* Methods */
    void getOut(std::string error);

    /* Variables */
    infoStruct_t                                                m_infoStruct;
    Plugin                                                     *m_plugin;
    std::map<testClasses_enum, std::vector<TestParameters *> >  m_argMap;
    bool                                                        m_skipTest;
    static const lwvsPluginGpuResults_t                         m_emptyGpuResults;
    static const lwvsPluginGpuMessages_t                        m_emptyGpuMessages;
    static const std::vector<string>                            m_emptyMessages;

/***************************PROTECTED********************************/
protected:
};

#endif //_LWVS_LWVS_TEST_H

