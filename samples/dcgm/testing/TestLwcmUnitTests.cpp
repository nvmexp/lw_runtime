#include "TestLwcmModule.h"
#include "TestCacheManager.h"
#include "TestFieldGroups.h"
#include "TestStatCollection.h"
#include "TestKeyedVector.h"
#include "TestProtobuf.h"
#include "TestVersioning.h"
#include "TestDiagManager.h"
#include "TestGroupManager.h"
#include "TestLwcmValue.h"
#include "TestPolicyManager.h"
#include "TestHealthMonitor.h"
#include "TestTopology.h"
#include "TestDcgmConnections.h"
#include "dcgm_agent_internal.h"
#include "TestDiagResponseWrapper.h"

#ifdef DCGM_BUILD_LWSWITCH_MODULE
    #include "TestDcgmModuleLwSwitch.h"
#endif

#ifdef DCGM_BUILD_VGPU_MODULE
    #include "TestDcgmModuleVgpu.h"
#endif

#include "LwcmSettings.h"
#include "dcgm_fields.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include "lwml.h"
#include "logging.h"
#include <vector>
#include <map>
#include <string>

/*****************************************************************************/
class TestLwcmUnitTests
{
public:
    /*************************************************************************/
    TestLwcmUnitTests()
    {
        m_embeddedStarted = false;
        m_runAllModules = false;
        m_gpus.clear();
        m_moduleArgs.clear();
        m_onlyModulesToRun.clear();
    }

    /*************************************************************************/
    ~TestLwcmUnitTests()
    {
        Cleanup();
    }

    /*************************************************************************/
    int LoadModules()
    {
        TestLwcmModule *module;
        std::string moduleTag;

        module = (TestLwcmModule *)(new TestGroupManager());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestLwcmModule *)(new TestKeyedVector());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestLwcmModule *)(new TestCacheManager());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestLwcmModule *)(new TestFieldGroups());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag = module->GetTag();
        m_modules[moduleTag] = module;
		
        module = (TestLwcmModule *)(new TestProtobuf());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag = module->GetTag();
        m_modules[moduleTag] = module;        

        module = (TestLwcmModule *)(new TestStatCollection());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestLwcmModule *)(new TestVersioning());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestLwcmModule *)(new TestDiagManager());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestLwcmModule *)(new TestLwcmValue());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestLwcmModule *)(new TestPolicyManager());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestLwcmModule *)(new TestHealthMonitor());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestLwcmModule *)(new TestTopology());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag = module->GetTag();
        m_modules[moduleTag] = module;
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestLwcmModule *)(new TestDcgmConnections());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag = module->GetTag();
        m_modules[moduleTag] = module;

        module = new TestDiagResponseWrapper();
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag = module->GetTag();
        m_modules[moduleTag] = module;

#ifdef DCGM_BUILD_LWSWITCH_MODULE
        module = (TestLwcmModule *)(new TestDcgmModuleLwSwitch());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag = module->GetTag();
        m_modules[moduleTag] = module;
#endif

#ifdef DCGM_BUILD_VGPU_MODULE
        module = (TestLwcmModule *)(new TestDcgmModuleVgpu());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag = module->GetTag();
        m_modules[moduleTag] = module;
#endif

        /* Copy above 4 lines to add your own module */
        return 0;
    }

    /*************************************************************************/
    int UnloadModules()
    {
        std::map<std::string, TestLwcmModule *>::iterator moduleIt;
        std::string moduleTag;
        TestLwcmModule *module;

        PRINT_DEBUG("%d", "Unloading %d modules", (int)m_modules.size());

        for(moduleIt = m_modules.begin(); moduleIt != m_modules.end(); moduleIt++)
        {
            moduleTag = (*moduleIt).first;
            module = (*moduleIt).second;

            delete(module);
        }

        /* All of the contained pointers are invalid. Clear the contents */
        m_modules.clear();

        return 0;
    }

    /*************************************************************************/
    void Cleanup()
    {
        UnloadModules();

        m_gpus.empty();
        m_moduleArgs.empty();
        dcgmShutdown();
    }

    /*************************************************************************/
    int StartAndConnectToRemoteDcgm(void)
    {
        int ret;
        const etblDCGMEngineInternal *pEtbl = NULL;

        ret = dcgmInternalGetExportTable((const void**)&pEtbl, &ETID_DCGMEngineInternal);
        if (DCGM_ST_OK  != ret) 
        {
            fprintf(stderr, "Err: Can't get the export table. Return: %d\n", ret);
            return -1;
        }

        ret = DCGM_CALL_ETBL(pEtbl, fpdcgmServerRun, (5555, (char *)"127.0.0.1", 1));
        if(ret)
        {
            fprintf(stderr, "fpdcgmServerRun returned %d\n", ret);
            return -1;
        }

        ret = dcgmConnect((char *)"127.0.0.1:5555", &m_dcgmHandle);
        if(ret)
        {
            fprintf(stderr, "dcgmConnect returned %d\n", ret);
            return -1;
        }

        return 0;
    }

    /*************************************************************************/
    int Init()
    {
        int st;

        /* Initialize DCGM */
        if (DCGM_ST_OK != dcgmInit())
        {
            fprintf(stderr, "DCGM could not initialize");
            return -1;
        }

        /* Embedded Mode */
        if (DCGM_ST_OK != dcgmStartEmbedded(DCGM_OPERATION_MODE_MANUAL, &m_dcgmHandle))
        {
            fprintf(stderr, "DCGM could not start the host engine");
            return -1;
        }
        m_embeddedStarted = true;
        
        /* Load all of the modules */
        st = LoadModules();
        if(st)
        {
            fprintf(stderr, "LoadModules() returned %d\n", st);
            return -1;
        }
        
        return 0;
    }

    /*************************************************************************/
    int ParseCommandLine(int argc, char *argv[])
    {
        int i;

        //argv[0] = program name. We don't care about it
        for(i = 1; i < argc; i++)
        {
            /* Parse any arguments. If they are processed here, continue
                     and don't put them into m_moduleArgs */
            if(!strcmp(argv[i], "-m"))
            {
                if(argc - i < 2)
                {
                    fprintf(stderr, "-m requires a 2nd parameter\n");
                    return -1;
                }
                i++; /* Move to the actual argument */

                printf("Adding %s to the list of modules to run\n", argv[i]);
                m_onlyModulesToRun[argv[i]] = 0;
                continue;
            }
            else if(!strcmp(argv[i], "-r"))
            {
                printf("Enabling a local remote DCGM.\n");
                m_startRemoteServer = true;
            }
            else if(!strcmp(argv[i], "-a"))
            {
                printf("Running all modules, including non-L0 ones.\n");
                m_runAllModules = true;
            }

            m_moduleArgs.push_back(std::string(argv[i]));
        }

        return 0;
    }

    /*************************************************************************/
    int FindAllGpus()
    {
        lwmlReturn_t lwmlReturn;
        unsigned int deviceCount = 0;
        unsigned int i;
        lwmlDevice_t lwmlDevice = 0;
        test_lwcm_gpu_t gpu;
        std::vector<unsigned int>gpuIds;
        std::vector<unsigned int>::iterator gpuIt;
        dcgmReturn_t dcgmReturn;

        /* Create a cache manager to get the GPU count */
        DcgmCacheManager *cacheManager = new DcgmCacheManager();
        dcgmReturn = cacheManager->Init(1, 3600.0);
        if(dcgmReturn != DCGM_ST_OK)
        {
            fprintf(stderr, "Failed to init cache manager. dcgmReturn %d", dcgmReturn);
            delete(cacheManager);
            return -1;
        }

        dcgmReturn = cacheManager->GetGpuIds(1, gpuIds);


        if(gpuIds.size() < 1)
        {
            fprintf(stderr, "No GPUs found. If you are testing on non-whitelisted GPUs, "
                    "set %s=1 in your environment", DCGM_ELW_WL_BYPASS);
            delete(cacheManager);
            return -1;
        }

        for(gpuIt = gpuIds.begin(); gpuIt != gpuIds.end(); gpuIt++)
        {
            /* Success. Record device */
            gpu.gpuId = *gpuIt;
            gpu.lwmlIndex = cacheManager->GpuIdToLwmlIndex(gpu.gpuId);
            printf("Using lwmlIndex %u. GpuId %u\n", gpu.lwmlIndex, gpu.gpuId);
            m_gpus.push_back(gpu);
        }

        delete(cacheManager);
        return 0;
    }

    /*************************************************************************/
    int RunOneModule(TestLwcmModule *module)
    {
        int st, runSt;

        st = module->Init(m_moduleArgs, m_gpus);
        if(st)
        {
            fprintf(stderr, "Module init for %s failed with %d\n",
                    module->GetTag().c_str(), st);
            return -1;
        }

        /* Actually run the module */
        runSt = module->Run();

        /* Clean-up unconditionally before dealing with the run status */
        module->Cleanup();

        if(runSt > 0)
        {
            fprintf(stderr, "Module %s had non-fatal failure.\n", module->GetTag().c_str());
            return 1;
        }
        else if(runSt < 0)
        {
            fprintf(stderr, "Module %s had FATAL failure st %d.\n",
                    module->GetTag().c_str(), runSt);
            return -1;
        }

        return 0;
    }

    /*************************************************************************/
    int RunModules(void)
    {
        int st;
        int Nfailed = 0;

        /* Run all modules */
        TestLwcmModule *module;
        std::string moduleTag;

        /* Should we only run certain modules? */
        if(m_onlyModulesToRun.size() > 0)
        {
            std::map<std::string,int>::iterator onlyModuleIt;
            std::map<std::string, TestLwcmModule *>::iterator moduleIt;

            printf("Running %d modules\n", (int)m_onlyModulesToRun.size());

            for(onlyModuleIt = m_onlyModulesToRun.begin(); onlyModuleIt != m_onlyModulesToRun.end(); onlyModuleIt++)
            {
                moduleTag = onlyModuleIt->first;

                moduleIt = m_modules.find(moduleTag);
                if(moduleIt == m_modules.end())
                {
                    fprintf(stderr, "%s is not a valid module name", moduleTag.c_str());
                    return 1;
                }

                module = (*moduleIt).second;

                st = RunOneModule(module);
                if(st)
                    Nfailed++;
                if(st < 0)
                    break; /* Fatal error */
            }
        }
        else /* Run all modules */
        {
            std::map<std::string, TestLwcmModule *>::iterator moduleIt;

            printf("Running %d modules\n", (int)m_modules.size());

            for(moduleIt = m_modules.begin(); moduleIt != m_modules.end(); moduleIt++)
            {
                moduleTag = (*moduleIt).first;
                module = (*moduleIt).second;

                if(!m_runAllModules && !module->IncludeInDefaultList())
                {
                    printf("Skipping module \"%s\" not included in default list. Pass -a to include all modules.\n", moduleTag.c_str());
                    continue;
                }

                st = RunOneModule(module);
                if(st)
                    Nfailed++;
                if(st < 0)
                    break; /* Fatal error */
            }
        }


        if(Nfailed > 0)
        {
            fprintf(stderr, "%d modules had test failures\n", Nfailed);
            return 1;
        }

        printf("All modules PASSED\n");
        return 0;
    }

    /*************************************************************************/
    /*
     * Main entry point for this class
     */
    int Run(int argc, char *argv[])
    {
        int st;

        /* Parse command line before discovering GPUs in case we specify gpus on
         * the command line
         */
        st = ParseCommandLine(argc, argv);
        if(st)
            return -1;
        
        /* Do we want to be remote to our DCGM? */
        if(m_startRemoteServer)
        {
            st = StartAndConnectToRemoteDcgm();
            if(st)
                return -1;
        }

        st = FindAllGpus();
        if(st)
            return -1;

        st = RunModules();
        if(st)
            return -1;

        return 0;
    }

    /*************************************************************************/

private:
    dcgmHandle_t m_dcgmHandle; /* Handle to our host engine. Only valid if m_embeddedStarted == 1 */
    bool m_embeddedStarted; /* Has an embedded host engine been started? 1=yes. 0=no */
    bool m_startRemoteServer; /* Has a TCP/IP serverbeen started? 1=yes. 0=no (pass -r to the program) */
    bool m_runAllModules; /* Should we run all modules discovered, even non-default modules? */
    std::vector<test_lwcm_gpu_t> m_gpus; /* GPUs to run on */
    std::vector<std::string> m_moduleArgs; /* ARGV[] array of args to pass to plugins */
    std::map<std::string, TestLwcmModule *>m_modules; /* Test modules to run, indexed by each
                                                         module's GetTag() */
    std::map<std::string,int>m_onlyModulesToRun; /* Map of 'moduletag'=>0 of modules we are supposed to run.
                                                    Empty = run all modules */
};


/*****************************************************************************/
int main(int argc, char *argv[])
{
    int st = 0;
    int retSt = 0;

    TestLwcmUnitTests *dcgmUnitTests = new TestLwcmUnitTests();

    /* Use fprintf(stderr) or printf() in this function because we're not sure if logging is initialized */

    st = dcgmUnitTests->Init();
    if(st)
    {
        fprintf(stderr, "dcgmUnitTests->Init() returned %d\n", st);
        delete dcgmUnitTests;
        return 1;
    }

    /* Actually run tests */
    retSt = dcgmUnitTests->Run(argc, argv);

    delete(dcgmUnitTests);
    dcgmUnitTests = 0;
    return retSt;
}
