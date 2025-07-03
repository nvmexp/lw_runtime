#include <assert.h>
#include <string.h>
#include <lwos.h>
#include <lwca.h>
#include <lwml.h>
#include "memory_plugin.h"
#include "memtest_kernel.ptx.string"
#include "timelib.h"
#include "L1TagLwda.h"
#include <PluginCommon.h>

#define PCIE_ONE_BW 250.0f // PCIe 1.0 - 250 MB/s per lane
#define PCIE_TWO_BW 500.0f // PCIe 2.x - 500 MB/s per lane
#define PCIE_THREE_BW 1000.0f // PCIe 3.0 - 1 GB/s per lane
#define PCIE_FOUR_BW 2000.0f // PCIe 4.0 - 2 GB/s per lane
#define NUMELMS(x) (sizeof(x) / sizeof(x[0]))

//#define MEMCOPY_ITERATIONS 50
//#define MEMCOPY_SIZE (1 << 27) // 128M

#ifdef __x86_64__
__asm__(".symver memcpy,memcpy@GLIBC_2.2.5");
#endif

#define ROUND_UP(n, multiple)                                               \
    ( ((n) + (multiple-1)) - (((n) + (multiple-1)) % (multiple)) )

// colwert bytes / ms into MB/s
//#define inMBperSecond(bytes, ms)
//    ( ((bytes) * 1000.0f) / ((ms) * 1024.0f * 1024.0f) )

/*****************************************************************************/
/* For now, use a heap struct */
mem_globals_t g_memGlobals;

/*****************************************************************************/
// defs pulled from healthmon
// For double bit ECC error reporting
typedef enum dbeReportCode_e
{
    NO_DBE_DETECTED               = 0,
    DBE_DETECTED                  = 1,
    DBE_DETECTED_ALREADY_REPORTED = 2,
    DBE_QUERY_ERROR               = 3,
    DBE_QUERY_UNSUPPORTED         = 4,

    // Keep this last
    DBE_REPORT_CODE_COUNT
} dbeReportCode_t;

typedef enum testResult
{
    TEST_RESULT_SUCCESS,
    TEST_RESULT_ERROR_FATAL,
    TEST_RESULT_ERROR_NON_FATAL,
    TEST_RESULT_ERROR_FAILED_TO_RUN,
    TEST_RESULT_WARNING,
    TEST_RESULT_SKIPPED,
    TEST_RESULT_COUNT,
} testResult_t;

dbeReportCode_t checkForDBE(mem_globals_p memGlobals, timelib64_t startTime, timelib64_t endTime)
{
    std::vector<unsigned short> fieldIds;
    std::vector<DcgmError> errorList;
    fieldIds.push_back(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL);
    int st = memGlobals->m_dcgmRecorder->CheckErrorFields(fieldIds, 0, memGlobals->lwmlGpuIndex, errorList,
                                                          startTime, endTime);

    // Copy any error messages
    for (size_t i = 0; i < errorList.size(); i++)
    {
        memGlobals->memory->AddErrorForGpu(memGlobals->lwmlGpuIndex, errorList[i]);
    }

    if (st == DR_COMM_ERROR)
    {
        if (errorList.size())
            PRINT_ERROR("%s", "Unable to query for DBE errors: %s", errorList[0].GetMessage().c_str());
        else
            PRINT_ERROR("", "Unable to query for DBE errors.");
        return DBE_QUERY_ERROR;
    }
    else if (st == DR_VIOLATION)
    {
        lwvsCommon.errorMask |= MEM_ERR_DBE_FAIL;
        PRINT_ERROR("", "An uncorrectable double-bit ECC error has oclwrred.");
        return DBE_DETECTED;
    }

    return NO_DBE_DETECTED;
}



static lwvsPluginResult_t runTestDeviceMemory(mem_globals_p memGlobals, lwmlDevice_t device,
                                        LWdevice lwDevice, LWcontext ctx)
{
    int attr;
    unsigned int error_h;
    size_t total, free;
    size_t size;
    LWdeviceptr alloc, errors;
    LWmodule mod;
    LWfunction memsetval, memcheckval;
    LWresult lwRes;
    void* ptr;
    static char testVals[5] = {(char)0x00, (char)0xAA, (char)0x55, (char)0xFF, (char)0x00};
    static const float minMemoryTest = 0.75;
    unsigned int memoryMismatchOclwrred = 0;
    timelib64_t  startTime = timelib_usecSince1970();

    lwRes = lwDeviceGetAttribute(&attr, LW_DEVICE_ATTRIBUTE_ECC_ENABLED, lwDevice);
    if(LWDA_SUCCESS != lwRes) goto error_no_cleanup;

    if (!attr)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ECC_DISABLED, d, "Memory", memGlobals->lwmlGpuIndex);
        PRINT_INFO("%s", "%s", d.GetMessage().c_str());
        memGlobals->memory->AddErrorForGpu(memGlobals->lwmlGpuIndex, d);
        return LWVS_RESULT_SKIP;
    }
    lwRes = lwModuleLoadData(&mod, memtest_kernel);
    if(LWDA_SUCCESS != lwRes) goto error_no_cleanup;

    lwRes = lwModuleGetGlobal(&errors, NULL, mod, "errors");
    if(LWDA_SUCCESS != lwRes) goto error_no_cleanup;

    lwRes = lwModuleGetFunction(&memsetval, mod, "memsetval");
    if(LWDA_SUCCESS != lwRes) goto error_no_cleanup;

    lwRes = lwModuleGetFunction(&memcheckval, mod, "memcheckval");
    if(LWDA_SUCCESS != lwRes) goto error_no_cleanup;

    lwRes = lwMemGetInfo(&free, &total);
    if(LWDA_SUCCESS != lwRes) goto error_no_cleanup;

    // alloc as much memory as possible
    size = free;
    do
    {
        size = (size / 100) * 99;
        if (size < (total * minMemoryTest))
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_MEMORY_ALLOC, d, minMemoryTest * 100, memGlobals->lwmlGpuIndex);
            memGlobals->memory->AddErrorForGpu(memGlobals->lwmlGpuIndex, d);
            lwvsCommon.errorMask |= MEM_ERR_LWDA_ALLOC_FAIL;
            PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
            goto error_no_cleanup;
        }
        lwRes = lwMemAlloc(&alloc, size);
    } while (LWDA_ERROR_OUT_OF_MEMORY == lwRes);


    if (LWDA_SUCCESS != lwRes)
        goto error_no_cleanup;

    {
        std::stringstream ss;
        ss.setf(std::ios::fixed, std::ios::floatfield);
        ss.precision(1);
        ss << "Allocated " << size << " bytes (" << (float) (size * 100.0f / total) << "%)";
        memGlobals->memory->AddInfoVerboseForGpu(memGlobals->lwmlGpuIndex, ss.str());
    }

    // Everything after this point needs to clean up
    memGlobals->m_dcgmRecorder->SetGpuStat(memGlobals->lwmlGpuIndex, "bytes_copied_per_test", (long long)size);
    memGlobals->m_dcgmRecorder->SetGpuStat(memGlobals->lwmlGpuIndex, "num_tests", (long long)NUMELMS(testVals));

//    std::stringstream os;
//    os.setf(std::ios::fixed, std::ios::floatfield);
//    os.precision(1);
//    os << "Allocated " << size << " bytes (" << (float) (size * 100.0f / total) << "%";
//    memGlobals->memory->AddInfoVerbose(os.str());
//    PRINT_INFO("%zu %.1f", "Allocated %zu bytes (%.1f%%)", size, (float) (size * 100.0f / total));

    lwRes = lwFuncSetBlockShape(memsetval, 256, 1, 1);
    if (LWDA_SUCCESS != lwRes) goto cleanup;

    lwRes = lwFuncSetBlockShape(memcheckval, 256, 1, 1);
    if(LWDA_SUCCESS != lwRes) goto cleanup;

    {
        int j;
        for (j = 0; j < NUMELMS(testVals); ++j)
        {
            int offset = 0;
            ptr = (void*)alloc;

            lwRes = lwParamSetv(memsetval, offset, &ptr, sizeof(void*));
            if(LWDA_SUCCESS != lwRes) goto cleanup;

            offset = ROUND_UP(offset + sizeof(void*), sizeof(void*));

            lwRes = lwParamSetv(memsetval, offset, &size, sizeof(size_t));
            if (LWDA_SUCCESS != lwRes) goto cleanup;

            offset = ROUND_UP(offset + sizeof(size_t), sizeof(size_t));

            lwRes = lwParamSetv(memsetval, offset, &testVals[j], sizeof(char));
            if (LWDA_SUCCESS != lwRes) goto cleanup;

            offset += sizeof(char);

            lwRes = lwParamSetSize(memsetval, offset);
            if (LWDA_SUCCESS != lwRes) goto cleanup;

            lwRes = lwLaunchGridAsync(memsetval, 128, 1, 0);
            if (LWDA_SUCCESS != lwRes) goto cleanup;

            lwRes = lwCtxSynchronize();
            if (LWDA_SUCCESS != lwRes) goto cleanup;

            offset = 0;
            ptr = (void*)alloc;

            lwRes = lwParamSetv(memcheckval, offset, &ptr, sizeof(void*));
            if (LWDA_SUCCESS != lwRes) goto cleanup;

            offset = ROUND_UP(offset + sizeof(void*), sizeof(void*));

            lwRes = lwParamSetv(memcheckval, offset, &size, sizeof(size_t));
            if (LWDA_SUCCESS != lwRes) goto cleanup;

            offset = ROUND_UP(offset + sizeof(size_t), sizeof(size_t));

            lwRes = lwParamSetv(memcheckval, offset, &testVals[j], sizeof(char));
            if (LWDA_SUCCESS != lwRes) goto cleanup;

            offset += sizeof(char);

            lwRes = lwParamSetSize(memcheckval, offset);
            if (LWDA_SUCCESS != lwRes) goto cleanup;

            lwRes = lwLaunchGridAsync(memcheckval, 128, 1, 0);
            if (LWDA_SUCCESS != lwRes) goto cleanup;

            lwRes = lwCtxSynchronize();
            if (LWDA_SUCCESS != lwRes) goto cleanup;

            lwRes = lwMemcpyDtoH(&error_h, errors, sizeof(unsigned int));
            if (LWDA_SUCCESS != lwRes) goto cleanup;

            if (error_h)
            {
                //
                // This should be rare.  This will happen if:
                // - LWCA error containment failed to contain the DBE
                // - >2 bits were flipped, ECC didn't catch the error
                //
                memoryMismatchOclwrred = 1;
                goto cleanup;
            }
        }
    }

    lwRes = LWDA_SUCCESS;

cleanup:
    // Release resources
    if (LWDA_ERROR_ECC_UNCORRECTABLE == lwMemFree(alloc))
    {
        //
        // Ignore other errors, outside the scope of the memory test
        // But give LWCA a final chance to contain memory errors
        //
        lwRes = LWDA_ERROR_ECC_UNCORRECTABLE;
    }

error_no_cleanup:
    //
    // Remember, many LWCA calls may return errors from previous async launches
    // Check the last LWCA call's return code
    //
    if (LWDA_ERROR_ECC_UNCORRECTABLE == lwRes)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWDA_DBE, d, memGlobals->lwmlGpuIndex);
        memGlobals->memory->AddErrorForGpu(memGlobals->lwmlGpuIndex, d);
        memGlobals->memory->AddInfo("A DBE oclwrred, LWCA error containment reported issue.");
        //PRINT_ERROR(STR_DEVICE_MEMORY_TEST_ERROR_DBE);
        return LWVS_RESULT_FAIL;
    }

    {
        dbeReportCode_t dbeCode = checkForDBE(memGlobals, startTime, timelib_usecSince1970());
        lwvsPluginResult_t ret = LWVS_RESULT_PASS;
        char buf[1024];

        switch (dbeCode)
        {
            case NO_DBE_DETECTED:
            case DBE_QUERY_UNSUPPORTED:
                // To the best of our knowledge, no DBE has oclwrred
                break;

            case DBE_QUERY_ERROR:
                // record the error, but allow memory mismatch message to print
                ret = LWVS_RESULT_FAIL;
                break;

            case DBE_DETECTED:
                //addInfo
//                PRINT_DEBUG_TRACE("A DBE oclwrred, but LWCA error containment don't report the issue.");
                return LWVS_RESULT_FAIL;
                break;

            case DBE_DETECTED_ALREADY_REPORTED:
                // If we previously saw a DBE then we shouldn't be running the memory test
                assert(0);
                PRINT_INFO("", "Skipping test because an uncorrectable double-bit ECC error has oclwrred.");
                return LWVS_RESULT_SKIP;
                break;

            default:
                // Unexpected error code
                assert(0);
                return LWVS_RESULT_FAIL;
                break;
        }

        if (memoryMismatchOclwrred)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_MEMORY_MISMATCH, d, memGlobals->lwmlGpuIndex);
            memGlobals->memory->AddErrorForGpu(memGlobals->lwmlGpuIndex, d);
            return LWVS_RESULT_FAIL;
        }

        return ret;
    }
}

testResult_t testUtilGetLwDeviceByLwmlDevice(lwmlDevice_t lwmlDevice, LWdevice * lwDevice, std::stringstream &error)
{
    lwmlPciInfo_t pciInfo;
    LWresult lwRes;

    assert(lwDevice);

    lwmlReturn_t lwRet = lwmlDeviceGetPciInfo(lwmlDevice, &pciInfo);

    if (LWML_SUCCESS != lwRet)
    {
        error << "lwmlDeviceGetPciInfo failed: '" << lwmlErrorString(lwRet) << "'.";
        return TEST_RESULT_ERROR_FAILED_TO_RUN;
    }

    lwRes = lwDeviceGetByPCIBusId(lwDevice, pciInfo.busId);
    if (LWDA_SUCCESS != lwRes)
    {
        //
        // We have a GPU that LWML can see but LWCA cannot
        // I.E. LWDA_VISIBLE_DEVICES is hiding this GPU
        //
//        PRINT_DEBUG_TRACE("lwDeviceGetByPCIBusId -> %d", lwRes);
        const char *errorStr;
        lwGetErrorString(lwRes, &errorStr);
        if (errorStr != NULL)
        {
            error << "lwDeviceGetByPCIBusId failed: '" << errorStr << "'.";
        }
        else
        {
            error << "lwDeviceGetByPCIBudId failed with unknown error: " << lwRes << ".";
        }
        
        const char *lwdaVis = getelw("LWDA_VISIBLE_DEVICES");
        if (lwdaVis == NULL)
        {
            error << " LWDA_VISIBLE_DEVICES is not set.";
        }
        else
        {
            error << " LWDA_VISIBLE_DEVICES is set to '" << lwdaVis << "'.";
        }

        return TEST_RESULT_SKIPPED;
    }

    return TEST_RESULT_SUCCESS;
}

/*****************************************************************************/
void mem_cleanup(mem_globals_p memGlobals)
{
    LWresult lwRes;

    if (memGlobals->m_dcgmRecorder)
    {
        delete(memGlobals->m_dcgmRecorder);
        memGlobals->m_dcgmRecorder = 0;
    }

    if(memGlobals->lwCtxCreated)
    {
        lwRes = lwCtxDestroy(memGlobals->lwCtx);
        if (LWDA_SUCCESS != lwRes)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(memGlobals->memory, "lwCtxDestroy", lwRes, memGlobals->lwmlGpuIndex);
        }
    }
    memGlobals->lwCtxCreated = 0;

    if(memGlobals->lwvsDevice)
    {
        memGlobals->lwvsDevice->RestoreState();
        delete(memGlobals->lwvsDevice);
        memGlobals->lwvsDevice = 0;
    }

    if(memGlobals->lwmlInitialized)
    {
        lwmlShutdown();
    }
    memGlobals->lwmlInitialized = 0;
}

/*****************************************************************************/
int mem_init(mem_globals_p memGlobals, unsigned int lwmlGpuIndex)
{
    int st;
    LWresult lwRes;
    lwmlReturn_t lwmlSt;
    char buf[256];
    lwvsReturn_t lwvsReturn;

    memGlobals->lwmlGpuIndex = lwmlGpuIndex;

    if (LWML_SUCCESS != (lwmlSt = lwmlInit()))
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlInit", lwmlErrorString(lwmlSt));
        PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
        memGlobals->memory->AddError(d);
        return 1;
    }
    memGlobals->lwmlInitialized = 1;

    lwRes = lwInit(0);
    if (LWDA_SUCCESS != lwRes)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWDA_API, d, "lwInit");
        std::string error = AppendLwdaDriverError("Unable to initialize LWCA library", lwRes);
        d.AddDetail(error);
        memGlobals->memory->AddError(d);
        return 1;
    }
    memGlobals->lwdaInitialized = 1;

    memGlobals->lwvsDevice = new LwvsDevice(memGlobals->memory);
    st = memGlobals->lwvsDevice->Init(memGlobals->lwmlGpuIndex);
    if(st)
    {
        return 1;
    }

    memGlobals->lwmlDevice = memGlobals->lwvsDevice->GetLwmlDevice();
    std::stringstream error;

    if (TEST_RESULT_SUCCESS != testUtilGetLwDeviceByLwmlDevice(memGlobals->lwmlDevice,
                                                               &memGlobals->lwDevice, error))
    {
        int st;
        char elwBuf[32] = {0}; /* Doesn't need to be large since lwosGetElw will just return > 0
                                  if it isn't big enough */
        std::stringstream ss;

        st = lwosGetElw("LWDA_VISIBLE_DEVICES", elwBuf, sizeof(elwBuf));
        unsigned int lwmlDeviceIndex = 999; /* Initialize to a bad number so it's easy to spot problems */
        lwmlDeviceGetIndex(memGlobals->lwmlDevice, &lwmlDeviceIndex);

        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWDA_DEVICE, d, memGlobals->lwmlGpuIndex, error.str().c_str());

        if (st >= 0)
        {
            /* Found LWDA_VISIBLE_DEVICES */
            d.AddDetail("If you specify LWDA_VISIBLE_DEVICES in your environment, it must include all Lwca "\
                        "device indices that DCGM GPU Diagnostic will be run on.");
            memGlobals->memory->AddErrorForGpu(memGlobals->lwmlGpuIndex, d);
        }
        return 1;
    }

    lwRes = lwCtxCreate(&memGlobals->lwCtx, 0, memGlobals->lwDevice);
    if (LWDA_SUCCESS != lwRes)
    {
        std::string error = AppendLwdaDriverError("Unable to create LWCA context", lwRes);
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWDA_API, d, "lwCtxCreate");
        d.AddDetail(error);
        memGlobals->memory->AddErrorForGpu(memGlobals->lwmlGpuIndex, d);
        return 1;
    }
    memGlobals->lwCtxCreated = 1;

    return 0;
}

lwvsPluginResult_t combine_results(lwvsPluginResult_t mainResult, lwvsPluginResult_t subTestResult)
{
    lwvsPluginResult_t result = mainResult;
    switch (subTestResult)
    {
        case LWVS_RESULT_WARN:
            if (mainResult != LWVS_RESULT_FAIL)
            {
                result = subTestResult;
            }
            break;
        case LWVS_RESULT_FAIL:
            result = subTestResult;
            break;

        case LWVS_RESULT_PASS:
        case LWVS_RESULT_SKIP:
        default:
            /* NO-OP */
            break;
    }
    
    return result;
}

/*****************************************************************************/
int main_entry(unsigned int lwmlGpuIndex, Memory *memory, TestParameters *tp)
{
    lwvsPluginResult_t result;
    int st;
    lwmlDevice_t device;
    lwmlReturn_t lwmlResult;
    lwmlEnableState_t lwrrentEcc, pendingEcc;

    mem_globals_p memGlobals = &g_memGlobals;

    memset(memGlobals, 0, sizeof(*memGlobals));
    memGlobals->memory = memory;
    memGlobals->testParameters = tp;

    memGlobals->m_dcgmRecorder = new DcgmRecorder();
    memGlobals->m_dcgmRecorder->Init(lwvsCommon.dcgmHostname);

    char fieldGroupName[128];
    char groupName[128];
    snprintf(fieldGroupName, sizeof(fieldGroupName), "memory%d_field_group", lwmlGpuIndex);
    snprintf(groupName, sizeof(groupName), "memory%d_group", lwmlGpuIndex);

    std::vector<unsigned short> fieldIds;
    fieldIds.push_back(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL);

    std::vector<unsigned int> gpuIds;
    gpuIds.push_back(lwmlGpuIndex);

    // Allow 5 minutes for the test - that should be plenty
    memGlobals->m_dcgmRecorder->AddWatches(fieldIds, gpuIds, false, fieldGroupName, groupName, 300.0);

    st = mem_init(memGlobals, lwmlGpuIndex);
    if (st)
    {
        memGlobals->memory->SetResult(LWVS_RESULT_FAIL);
        mem_cleanup(memGlobals);
        return 1;
    }

    // check if this card supports ECC and be good about skipping/warning etc.
    if (LWML_SUCCESS != (lwmlResult = lwmlDeviceGetHandleByIndex(lwmlGpuIndex, &device)))
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceGetHandleByIndex", lwmlErrorString(lwmlResult));
        PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
        memGlobals->memory->AddError(d);
        memGlobals->memory->SetResult(LWVS_RESULT_FAIL);
        mem_cleanup(memGlobals);
        return 1;
    }

    if (LWML_SUCCESS != (lwmlResult = lwmlDeviceGetEccMode(device, &lwrrentEcc, &pendingEcc)))
    {
        if (lwmlResult == LWML_ERROR_NOT_SUPPORTED)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ECC_UNSUPPORTED, d);
            memGlobals->memory->AddErrorForGpu(lwmlGpuIndex, d);
            memGlobals->memory->SetResult(LWVS_RESULT_SKIP);
        }
        else
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceGetEccMode", lwmlErrorString(lwmlResult));
            PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
            memGlobals->memory->AddError(d);
            memGlobals->memory->SetResult(LWVS_RESULT_FAIL);
        }
        mem_cleanup(memGlobals);
        return 1;
    }

    if (lwrrentEcc != LWML_FEATURE_ENABLED)
    {
        std::stringstream ss;
        if (pendingEcc == LWML_FEATURE_ENABLED)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ECC_PENDING, d, lwmlGpuIndex);
            memGlobals->memory->AddErrorForGpu(lwmlGpuIndex, d);
        }
        else
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ECC_DISABLED, d, "Memory", lwmlGpuIndex);
            memGlobals->memory->AddErrorForGpu(lwmlGpuIndex, d);
        }
        memGlobals->memory->SetResult(LWVS_RESULT_SKIP);
        mem_cleanup(memGlobals);
        return 1;
    }

    try
    {
        result = runTestDeviceMemory(memGlobals, memGlobals->lwmlDevice, memGlobals->lwDevice,
                                     memGlobals->lwCtx);

        if (tp->GetBoolFromString(MEMORY_L1TAG_STR_IS_ALLOWED) && (result != LWVS_RESULT_SKIP))
        {
            // Run the cache subtest
            L1TagLwda ltc(memGlobals->memory, tp, memGlobals);
            lwvsPluginResult_t tmpResult = ltc.TestMain(memGlobals->lwmlGpuIndex);
            result = combine_results(result, tmpResult);
        }

        if (result != LWVS_RESULT_PASS)
        {
            if (result == LWVS_RESULT_SKIP)
            {
                memGlobals->memory->SetResult(LWVS_RESULT_SKIP);
            }
            else
            {
                memGlobals->memory->SetResult(LWVS_RESULT_FAIL);
            }
            
            mem_cleanup(memGlobals);
            return 1;
        }
    }
    catch (std::runtime_error &e)
    {
        PRINT_ERROR("%s", "Caught runtime_error %s", e.what());
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, e.what());
        memGlobals->memory->AddError(d);
        memGlobals->memory->SetResult(LWVS_RESULT_FAIL);
        mem_cleanup(memGlobals);
        throw;
    }

    if (main_should_stop)
    {
        memGlobals->memory->SetResult(LWVS_RESULT_SKIP);
    }
    else
    {
        memGlobals->memory->SetResult(LWVS_RESULT_PASS);
    }

    mem_cleanup(memGlobals);
    return 0;
}
