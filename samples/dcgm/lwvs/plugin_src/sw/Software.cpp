#include "Software.h"
#include "lwml.h"
#include <iostream>
#include <stdexcept>
#include <dlfcn.h>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include "logging.h"
#include "DcgmRecorder.h"
#include "DcgmGPUHardwareLimits.h"
#include "DcgmError.h"
#include "dcgm_errors.h"

/* This module is admittedly entirely Linux centric.  lwml_loader and lwda_loader
 * use lwosLoadLibrary() which is much more platform agnostic but is part of the 
 * driver build and thus not really appropriate here.  This will have to be 
 * completely rewritten when/if we ever port LWVS to Windows.
 */

#ifdef __x86_64__
__asm__(".symver memcpy,memcpy@GLIBC_2.2.5");
#endif

Software::Software()
{
    m_infoStruct.name = "Deployment";
    m_infoStruct.shortDescription = "Software deployment checks plugin.";
    m_infoStruct.testGroups = "Software";
    m_infoStruct.selfParallel = true;
    m_infoStruct.logFileTag = SW_PLUGIN_LF_NAME;

    tp = new TestParameters();
    tp->AddString(PS_RUN_IF_GOM_ENABLED, "True");
    tp->AddString(SW_STR_DO_TEST, "None");
    tp->AddString(SW_STR_REQUIRE_PERSISTENCE, "True");
    m_infoStruct.defaultTestParameters = tp;
}

void Software::Go(TestParameters *testParameters, const std::vector<unsigned int> &gpuList)
{
    InitializeForGpuList(gpuList);

    if (testParameters->GetString(SW_STR_DO_TEST) == "blacklist")
        checkBlacklist();
    else if (testParameters->GetString(SW_STR_DO_TEST) == "permissions")
        checkPermissions();
    else if (testParameters->GetString(SW_STR_DO_TEST) == "libraries_lwml")
        checkLibraries(CHECK_LWML);
    else if (testParameters->GetString(SW_STR_DO_TEST) == "libraries_lwda")
        checkLibraries(CHECK_LWDA);
    else if (testParameters->GetString(SW_STR_DO_TEST) == "libraries_lwdatk")
        checkLibraries(CHECK_LWDATK);
    else if (testParameters->GetString(SW_STR_DO_TEST) == "persistence_mode")
    {
        int shouldCheckPersistence = testParameters->GetBoolFromString(SW_STR_REQUIRE_PERSISTENCE);

        if(!shouldCheckPersistence)
        {
            PRINT_INFO("", "Skipping persistence check");
            SetResult(LWVS_RESULT_SKIP);
        }
        else
        {
            checkPersistenceMode(gpuList);
        }
    }
    else if (testParameters->GetString(SW_STR_DO_TEST) == "elw_variables")
        checkForBadElwVaribles();
    else if (testParameters->GetString(SW_STR_DO_TEST) == "graphics_processes")
        checkForGraphicsProcesses(gpuList);
    else if (testParameters->GetString(SW_STR_DO_TEST) == "page_retirement")
        checkPageRetirement(gpuList);
    else if (testParameters->GetString(SW_STR_DO_TEST) == "inforom")
        checkInforom(gpuList);

    if (GetResult() == LWVS_RESULT_FAIL)
    {
        lwvsCommon.errorMask |= SW_ERR_FAIL;
    }
}

bool Software::checkPermissions()
{
    // check how many devices LWML is reporting and compare to 
    // the number of devices listed in /dev
    lwmlReturn_t lwmlResult;
    unsigned int lwmlCount = 0;
    unsigned int deviceCount = 0;

    DIR *dir;
    struct dirent * ent;
    int errorno;
    std::string dirName = "/dev";

    if (retrieveDeviceCount(&lwmlCount))
    {
        return true;
    }

    // everything below here is not necessarily a failure
    SetResult(LWVS_RESULT_PASS);
    dir = opendir(dirName.c_str());

    if (NULL == dir)
        return false;

    ent = readdir(dir); 
    errorno = errno;
    while (NULL != ent)
    {
        string entryName = ent->d_name;
        if (entryName.compare(0,  6, "lwpu") == 0 &&
            entryName.compare(0,  9, "lwidiactl") != 0 &&
            entryName.compare(0, 10, "lwpu-uvm") != 0 &&
            entryName.compare(0, 13, "lwpu-lwlink") != 0 &&
            entryName.compare(0, 14, "lwpu-modeset") != 0 &&
            entryName.compare(0, 15, "lwpu-lwswitch") != 0)
        {
            deviceCount++;
            std::stringstream ss;
            ss << dirName << "/" << entryName;
            if (access(ss.str().c_str(), R_OK) != 0)
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NO_ACCESS_TO_FILE, d, ss.str().c_str(), strerror(errno));
                AddError(d);
                SetResult(LWVS_RESULT_WARN);
            }
        }

        ent = readdir(dir);
        errorno = errno;
    }
    closedir(dir);

    if (deviceCount != lwmlCount)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_DEVICE_COUNT_MISMATCH, d);
        AddError(d);
        SetResult(LWVS_RESULT_WARN);
    }
    return false;
}

void *Software::LoadLwmlLib()
{
    const char *lwmlLib = "liblwidia-ml.so.1";
    void * lib_handle = dlopen (lwmlLib, RTLD_LAZY);

    if (!lib_handle)
    {
        std::string error = dlerror();
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CANNOT_OPEN_LIB, d, lwmlLib, error.c_str());
        AddError(d);
        SetResult(LWVS_RESULT_FAIL);
    }

    return lib_handle;
}

int Software::LoadAndCallLwmlInit(void *lib_handle)
{
    const char *lwml_init_name = "lwmlInit_v2";
    lwmlReturn_t (*fn)() = (lwmlReturn_t (*)()) dlsym(lib_handle, lwml_init_name);

    if (!fn)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_LIB_BAD, d, lwml_init_name);
        AddError(d);
        SetResult(LWVS_RESULT_FAIL);
        return 1;
    }
    
    lwmlReturn_t lwmlResult = (*fn)();
    if (LWML_SUCCESS != lwmlResult)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, lwml_init_name, lwmlErrorString(lwmlResult));
        AddError(d);
        SetResult(LWVS_RESULT_FAIL);
        return 1;
    }

    return 0;
}

int Software::LoadAndCallLwmlShutdown(void *lib_handle)
{
    const char *shutdown_name = "lwmlShutdown";
    lwmlReturn_t (*fnLwmlDestroy)() = (lwmlReturn_t (*)()) dlsym(lib_handle, shutdown_name);
    if(!fnLwmlDestroy)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_LIB_BAD, d, shutdown_name);
        AddError(d);
        SetResult(LWVS_RESULT_FAIL);
        return 1;
    }

    lwmlReturn_t lwmlResult = (*fnLwmlDestroy)();
    if (LWML_SUCCESS != lwmlResult)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, shutdown_name, lwmlErrorString(lwmlResult));
        AddError(d);
        SetResult(LWVS_RESULT_FAIL);
        return 1;
    }

    return 0;
}

int Software::retrieveDeviceCount(unsigned int *deviceCount)
{
    // If we are here then we've found LWML... go ahead and dlopen 
    // it and grab the device count.
    // In C, void * and args, can be dynamically typecast to anything without
    // explicitly doing it.  C++, not so much.

    lwmlReturn_t (*fn)();
    lwmlReturn_t (*fn2)(unsigned int *);
    unsigned int count = 0;
    void * lib_handle = NULL;
    lwmlReturn_t lwmlResult;

    if (deviceCount == NULL)
        return LWML_ERROR_ILWALID_ARGUMENT;

    lib_handle = LoadLwmlLib();
    if (!lib_handle)
    {
        return 1;
    }

    if (LoadAndCallLwmlInit(lib_handle) != 0)
    {
        return 1;
    }

    fn2 = (lwmlReturn_t (*)(unsigned int *)) dlsym(lib_handle, "lwmlDeviceGetCount_v2");
    lwmlResult = (*fn2)(&count);
    if (LWML_SUCCESS != lwmlResult)
        return lwmlResult;

    if (LoadAndCallLwmlShutdown(lib_handle) != 0)
    {
        return 1;
    }

    *deviceCount = count;
    return 0;
}

bool Software::checkLibraries(libraryCheck_t checkLib)
{
    // check whether the LWML, LWCA, and LWCA toolkit libraries can be found 
    // via default paths
    bool fail = false;
    std::vector<std::string> libs;

    switch (checkLib)
    {
        case CHECK_LWML:
            libs.push_back("liblwidia-ml.so.1");
            break;
        case CHECK_LWDA:
            libs.push_back("liblwda.so");
            break;
        case CHECK_LWDATK:
            libs.push_back("liblwdart.so");
            libs.push_back("liblwblas.so");
            break;
        default:
        {
            // should never get here
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_BAD_PARAMETER, d, __func__);
            AddError(d);
            SetResult(LWVS_RESULT_FAIL);
        }
    }

    for (std::vector<std::string>::iterator it = libs.begin(); it != libs.end(); it++)
    {
        std::string error;
        if (!findLib(*it, error))
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CANNOT_OPEN_LIB, d, it->c_str(), error.c_str());
            AddError(d);
            if (checkLib != CHECK_LWDATK)
            {
                SetResult(LWVS_RESULT_FAIL);
            }
            else 
            {
                SetResult(LWVS_RESULT_WARN);
            }
                
            fail = true;
        }
    }

    // The statements that follow are all classified as info statements because only messages directly tied
    // to failures are errors.
    if (checkLib == CHECK_LWDATK && fail == true)
    {
        AddInfo("The LWCA Toolkit libraries could not be found.");
        AddInfo("Is LD_LIBRARY_PATH set to the 64-bit library path? (usually /usr/local/lwca/lib64)");
        AddInfo("Some tests will not run.");
    }
    if (checkLib == CHECK_LWDA && fail == true)
    {
        AddInfo("The LWCA main library could not be found.");
        AddInfo("Skipping remainder of tests.");
    }
    if (checkLib == CHECK_LWML && fail == true)
    {
        AddInfo("The LWML main library could not be found in the default search paths.");
        AddInfo("Please check to see if it is installed or that LD_LIBRARY_PATH contains the path to liblwidia-ml.so.1.");
        AddInfo("Skipping remainder of tests.");
    }
    return fail;
}

bool Software::checkBlacklist()
{
    // check whether the nouveau driver is installed and if so, fail this test
    bool status = false;

    const std::string searchPaths[] = { "/sys/bus/pci/devices",
                                        "/sys/bus/pci_express/devices" };
    const std::string driverDirs[] =  { "driver", "subsystem/drivers" };

    const std::string blackList[] =   { "nouveau" };

    for (int i = 0; i < sizeof(searchPaths) / sizeof(searchPaths[0]); i++)
    {
        DIR *dir;
        struct dirent * ent;
        int errorno;

        dir = opendir(searchPaths[i].c_str());

        if (NULL == dir)
            continue;
        
        ent = readdir(dir);
        errorno = errno;
        while (NULL != ent)
        {
            if ((strcmp(ent->d_name, ".") == 0) ||
                (strcmp(ent->d_name, "..") == 0))
            {
                ent = readdir(dir);
                errorno = errno;
                continue;
            }
            for (int j = 0; j < sizeof(driverDirs) / sizeof(driverDirs[0]); j++)
            {
                std::string baseDir = searchPaths[i];
                std::stringstream testPath;
                testPath << baseDir << "/" << ent->d_name << "/" << driverDirs[j];
                if (checkDriverPathBlacklist(testPath.str(), blackList))
                {
                    SetResult(LWVS_RESULT_FAIL);
                    status = true;
                }
            }
            ent = readdir(dir);
            errorno = errno;
        }
        closedir(dir);
    }
    if (!status)
        SetResult(LWVS_RESULT_PASS);
    return status;
}

int Software::checkDriverPathBlacklist(std::string driverPath, const std::string *blackList)
{
    int ret;
    char symlinkTarget[1024];
    ret = readlink(driverPath.c_str(), symlinkTarget, sizeof(symlinkTarget));
    if (ret >= (signed int)sizeof(symlinkTarget))
    {   
        assert(0);
        return ENAMETOOLONG;
    }   
    else if (ret < 0)
    {   
        int errorno = errno;
    
        switch (errorno)
        {   
            case ENOENT:
                // driverPath does not exist, ignore it
                // this driver doesn't use this path format
                return 0;
            case EILWAL: // not a symlink
                return 0;

            case EACCES:
            case ENOTDIR:
            case ELOOP:
            case ENAMETOOLONG:
            case EIO:
            default:
                // Something bad happened
                return errorno;
        }   
    }   
    else
    {
        symlinkTarget[ret] = '\0';  // readlink doesn't null terminate
        for (int i = 0; i < sizeof(blackList) / sizeof(blackList[0]); i++)
        {
            if (strcmp(blackList[i].c_str(), basename(symlinkTarget)) == 0)
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_BLACKLISTED_DRIVER, d, blackList[i].c_str());
                AddError(d);
                return 1;
            }
        }
    }

    return 0;
}

bool Software::findLib(std::string library, std::string &error)
{
    void *handle;
    handle = dlopen(library.c_str(), RTLD_NOW);
    if (!handle)
    {
        error = dlerror();
        return false;
    }
    dlclose(handle);
    return true;
}

int Software::checkForGraphicsProcesses(const std::vector<unsigned int> &gpuList)
{
    lwmlReturn_t (*fnLwmlInit)();
    lwmlReturn_t (*fnGetDeviceById)(unsigned int index, lwmlDevice_t *device);
    lwmlReturn_t (*fnGetGraphicsRunningProcesses)(lwmlDevice_t device, unsigned int *infoCount, lwmlProcessInfo_t *infos);
    void * lib_handle = NULL;
    char errorBuf[256];
    lwmlReturn_t lwmlResult;
    std::vector<unsigned int>::const_iterator gpuIt;
    unsigned int graphicsProcesses = 0;
    bool graphicsBool = false;
    unsigned int lwmlGpuIndex;
    const char *lwmlLib = "liblwidia-ml.so.1";
    lwmlDevice_t lwmlDevice;

    errorBuf[255] = 0;

    lib_handle = LoadLwmlLib();
    if (!lib_handle)
    {
        return 1;
    }

    if (LoadAndCallLwmlInit(lib_handle))
    {
        return 1;
    }

    const char *device_get_name = "lwmlDeviceGetHandleByIndex_v2";
    fnGetDeviceById = (lwmlReturn_t (*)(unsigned int index, lwmlDevice_t *device)) dlsym(lib_handle, device_get_name);
    if(!fnGetDeviceById)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_LIB_BAD, d, device_get_name);
        AddError(d);
        SetResult(LWVS_RESULT_FAIL);
        return 1;
    }

    const char *get_processes_name = "lwmlDeviceGetGraphicsRunningProcesses";
    fnGetGraphicsRunningProcesses = (lwmlReturn_t (*)(lwmlDevice_t device, unsigned int *infoCount, lwmlProcessInfo_t *infos)) 
                                      dlsym (lib_handle, get_processes_name);
    if (!fnGetGraphicsRunningProcesses)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_LIB_BAD, d, device_get_name);
        AddError(d);
        SetResult(LWVS_RESULT_FAIL);
        return 1;
    }

    for (gpuIt = gpuList.begin(); gpuIt != gpuList.end(); gpuIt++)
    {
        graphicsProcesses = 0;
        lwmlGpuIndex = *gpuIt;

        lwmlResult = (*fnGetDeviceById)(lwmlGpuIndex, &lwmlDevice);
        if(lwmlResult != LWML_SUCCESS)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, device_get_name, lwmlErrorString(lwmlResult));
            AddError(d);
            SetResult(LWVS_RESULT_FAIL);
            break;
        }

        lwmlResult = (*fnGetGraphicsRunningProcesses)(lwmlDevice, &graphicsProcesses, NULL);
        if(lwmlResult != LWML_SUCCESS && lwmlResult != LWML_ERROR_INSUFFICIENT_SIZE)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, get_processes_name, lwmlErrorString(lwmlResult));
            AddError(d);
            SetResult(LWVS_RESULT_FAIL);
            continue;
        }

        if (graphicsProcesses > 0)
        {
            graphicsBool = true;
            break;
        }
    }

    if (graphicsBool)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_GRAPHICS_PROCESSES, d);
        AddError(d);
        SetResult(LWVS_RESULT_WARN);
    }

    if (LoadAndCallLwmlShutdown(lib_handle))
    {
        return 1;
    }

    return 0;
}

int Software::checkPersistenceMode(const std::vector<unsigned int> &gpuList)
{
    lwmlReturn_t (*fnLwmlInitDestroy)();
    lwmlReturn_t (*fnGetPersistenceMode)(lwmlDevice_t device, lwmlEnableState_t *mode);
    lwmlReturn_t (*fnGetDeviceById)(unsigned int index, lwmlDevice_t *device);
    void * lib_handle = NULL;
    lwmlReturn_t lwmlResult;
    char errorBuf[256];
    unsigned int lwmlGpuIndex;
    errorBuf[255] = 0;
    lwmlDevice_t lwmlDevice;
    lwmlEnableState_t persistenceEnabled;
    std::vector<unsigned int>::const_iterator gpuIt;

    lib_handle = LoadLwmlLib();
    if (!lib_handle)
    {
        return 1;
    }

    if (LoadAndCallLwmlInit(lib_handle))
    {
        return 1;
    }

    const char *get_device_name = "lwmlDeviceGetHandleByIndex_v2";
    fnGetDeviceById = (lwmlReturn_t (*)(unsigned int index, lwmlDevice_t *device)) dlsym(lib_handle, get_device_name);
    if (!fnGetDeviceById)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_LIB_BAD, d, get_device_name);
        AddError(d);
        SetResult(LWVS_RESULT_FAIL);
        return 1;
    }

    const char *persistence_mode_name = "lwmlDeviceGetPersistenceMode";
    fnGetPersistenceMode = (lwmlReturn_t (*)(lwmlDevice_t device, lwmlEnableState_t *mode)) dlsym(lib_handle, persistence_mode_name);
    if (!fnGetPersistenceMode)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_LIB_BAD, d, persistence_mode_name);
        AddError(d);
        SetResult(LWVS_RESULT_FAIL);
        return 1;
    }


    for (gpuIt = gpuList.begin(); gpuIt != gpuList.end(); gpuIt++)
    {
        lwmlGpuIndex = *gpuIt;

        lwmlResult = (*fnGetDeviceById)(lwmlGpuIndex, &lwmlDevice);
        if(lwmlResult != LWML_SUCCESS)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, get_device_name, lwmlErrorString(lwmlResult));
            AddError(d);
            SetResult(LWVS_RESULT_FAIL);
            break;
        }

        lwmlResult = (*fnGetPersistenceMode)(lwmlDevice, &persistenceEnabled);
        if(lwmlResult != LWML_SUCCESS)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, persistence_mode_name, lwmlErrorString(lwmlResult));
            AddError(d);
            SetResult(LWVS_RESULT_FAIL);
            continue;
        }

        if (persistenceEnabled != LWML_FEATURE_ENABLED)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PERSISTENCE_MODE, d, lwmlGpuIndex);
            AddError(d);
            SetResult(LWVS_RESULT_FAIL);
            continue;
        }
    }

    if (LoadAndCallLwmlShutdown(lib_handle))
    {
        return 1;
    }

    return 0;
}

int Software::checkPageRetirement(const std::vector<unsigned int> &gpuList)
{
    DcgmRecorder                                dcgmRecorder;
    std::string                                 dcgmInitError;
    char                                        errorBuf[256];
    unsigned int                                gpuId;
    std::vector<unsigned int>::const_iterator   gpuIt;
    dcgmFieldValue_v2                           pendingRetirementsFieldValue;
    dcgmFieldValue_v2                           dbeFieldValue;
    dcgmFieldValue_v2                           sbeFieldValue;
    dcgmReturn_t                                ret;
    int64_t                                     retiredPagesTotal;
    
    errorBuf[255] = '\0';
    dcgmInitError = dcgmRecorder.Init(lwvsCommon.dcgmHostname); // fails if there is an error connecting to host engine
    if (!dcgmInitError.empty())
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_HOSTENGINE_CONN, d, dcgmInitError.c_str());
        AddError(d);
        SetResult(LWVS_RESULT_FAIL);
        return 1;
    }

    for(gpuIt = gpuList.begin(); gpuIt != gpuList.end(); gpuIt++)
    {
        gpuId = *gpuIt;
        // Check for pending page retirements
        ret = dcgmRecorder.GetLwrrentFieldValue(gpuId, DCGM_FI_DEV_RETIRED_PENDING, 
                                                       pendingRetirementsFieldValue, 0);
        if (ret != DCGM_ST_OK)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "retired_pages_pending", gpuId);
            AddError(d);
            SetResult(LWVS_RESULT_FAIL);
            continue;
        }

        if (pendingRetirementsFieldValue.value.i64 > 0) 
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PENDING_PAGE_RETIREMENTS, d, gpuId);
            AddError(d);
            SetResult(LWVS_RESULT_FAIL);
            continue;
        }

        // Check total page retirement count
        // DBE retired pages
        ret = dcgmRecorder.GetLwrrentFieldValue(gpuId, DCGM_FI_DEV_RETIRED_DBE, dbeFieldValue, 0);
        if (ret != DCGM_ST_OK)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "retired_pages_dbe", gpuId);
            AddError(d);
            SetResult(LWVS_RESULT_FAIL);
            continue;
        }

        // SBE retired pages
        ret = dcgmRecorder.GetLwrrentFieldValue(gpuId, DCGM_FI_DEV_RETIRED_SBE, sbeFieldValue, 0);
        if (ret != DCGM_ST_OK)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "retired_pages_sbe", gpuId);
            AddError(d);
            SetResult(LWVS_RESULT_FAIL);
            continue;
        }

        retiredPagesTotal = dbeFieldValue.value.i64 + sbeFieldValue.value.i64;
        if (retiredPagesTotal >= DCGM_LIMIT_MAX_RETIRED_PAGES)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_RETIRED_PAGES_LIMIT, d, DCGM_LIMIT_MAX_RETIRED_PAGES, gpuId);
            AddError(d);
            SetResult(LWVS_RESULT_FAIL);
            continue;
        }
    }

    dcgmRecorder.Shutdown();
    return 0;
}

int Software::checkInforom(const std::vector<unsigned int> &gpuList)
{
    lwmlReturn_t (*fnLwmlInitDestroy)();
    lwmlReturn_t (*fnLwmlDeviceValidateInforom)(lwmlDevice_t device);
    lwmlReturn_t (*fnGetDeviceById)(unsigned int index, lwmlDevice_t *device);
    void * lib_handle = NULL;
    lwmlReturn_t lwmlResult;
    char errorBuf[256] = {0};
    unsigned int lwmlGpuIndex;
    lwmlDevice_t lwmlDevice;
    lwmlEnableState_t isPending;
    std::vector<unsigned int>::const_iterator gpuIt;

    lib_handle = LoadLwmlLib();
    if (!lib_handle)
    {
        return 1;
    }

    if (LoadAndCallLwmlInit(lib_handle))
    {
        return 1;
    }

    const char *get_device_name = "lwmlDeviceGetHandleByIndex_v2";
    fnGetDeviceById = (lwmlReturn_t (*)(unsigned int index, lwmlDevice_t *device)) dlsym(lib_handle, get_device_name);
    if (!fnGetDeviceById)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_LIB_BAD, d, get_device_name);
        AddError(d);
        SetResult(LWVS_RESULT_FAIL);
        return 1;
    }

    const char *validate_inforom_name = "lwmlDeviceValidateInforom";
    fnLwmlDeviceValidateInforom = (lwmlReturn_t (*)(lwmlDevice_t device)) dlsym(lib_handle, validate_inforom_name);
    if (!fnLwmlDeviceValidateInforom)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_LIB_BAD, d, get_device_name);
        AddError(d);
        SetResult(LWVS_RESULT_FAIL);
        return 1;
    }

    for (gpuIt = gpuList.begin(); gpuIt != gpuList.end(); gpuIt++)
    {
        lwmlGpuIndex = *gpuIt;

        lwmlResult = (*fnGetDeviceById)(lwmlGpuIndex, &lwmlDevice);
        if (lwmlResult != LWML_SUCCESS)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, get_device_name, lwmlErrorString(lwmlResult));
            AddError(d);
            SetResult(LWVS_RESULT_FAIL);
            break;
        }

        lwmlResult = (*fnLwmlDeviceValidateInforom)(lwmlDevice);
        if (lwmlResult == LWML_ERROR_NOT_SUPPORTED)
        {
            snprintf(errorBuf, sizeof(errorBuf)-1, "lwmlDeviceValidateInforom NOT SUPPORTED for GPU %u. Skipping check.",
                     lwmlGpuIndex);
            PRINT_INFO("%s", "%s", errorBuf);
            AddInfo(std::string(errorBuf)); // TODO: do we want to report this to the user as verbose info or a warning?
            continue;
        }

        if (lwmlResult == LWML_ERROR_CORRUPTED_INFOROM)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CORRUPT_INFOROM, d, lwmlGpuIndex);
            AddError(d);
            SetResult(LWVS_RESULT_FAIL);
            continue;
        }
        else if (lwmlResult != LWML_SUCCESS)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, validate_inforom_name, lwmlErrorString(lwmlResult));
            if (lwmlResult == LWML_ERROR_ILWALID_ARGUMENT)
            {
                d.AddDetail("Re-flash the InfoROM to clear this error. If it persists, verify the LWML library "
                        "and driver are installed correctly, and if it still persists then run a field diagnostic.");
            }
            AddError(d);
            SetResult(LWVS_RESULT_FAIL);
            continue;
        }
    }

    if (LoadAndCallLwmlShutdown(lib_handle))
    {
        return 1;
    }

    return 0;
}

int Software::checkForBadElwVaribles()
{
    int st;
    std::vector<std::string>checkKeys;
    std::vector<std::string>::iterator checkKeysIt;
    std::string checkKey;
    char elwBuf[64] = {0};
    char errorBuf[256] = {0};

    /* Elw variables to look for */
    checkKeys.push_back(std::string("NSIGHT_LWDA_DEBUGGER"));
    checkKeys.push_back(std::string("LWDA_INJECTION32_PATH"));
    checkKeys.push_back(std::string("LWDA_INJECTION64_PATH"));
    checkKeys.push_back(std::string("LWDA_AUTO_BOOST"));
    checkKeys.push_back(std::string("LWDA_ENABLE_COREDUMP_ON_EXCEPTION"));
    checkKeys.push_back(std::string("LWDA_COREDUMP_FILE"));
    checkKeys.push_back(std::string("LWDA_DEVICE_WAITS_ON_EXCEPTION"));
    checkKeys.push_back(std::string("LWDA_PROFILE"));
    checkKeys.push_back(std::string("COMPUTE_PROFILE"));
    checkKeys.push_back(std::string("OPENCL_PROFILE"));

    for (checkKeysIt = checkKeys.begin(); checkKeysIt != checkKeys.end(); checkKeysIt++)
    {
        checkKey = *checkKeysIt;

        /* Does the variable exist in the environment? */
        st = lwosGetElw(checkKey.c_str(), elwBuf, sizeof(elwBuf));
        if(st < 0)
        {
            PRINT_DEBUG("%s", "Elw Variable %s not found (GOOD)", checkKey.c_str());
            continue;
        }

        /* Variable found. Warn */
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_BAD_LWDA_ELW, d, checkKey.c_str());
        AddError(d);
        SetResult(LWVS_RESULT_WARN);
    }

    return 0;
}

extern "C" {
    Plugin *maker() {
        return new Software;
    }
    class proxy {
    public:
        proxy()
        {
            factory["Software"] = maker;
        }
    };    
    proxy p;
}                                            
