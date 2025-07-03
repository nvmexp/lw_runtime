#include "TestDcgmModuleVgpu.h"
#include "dcgm_vgpu_internal.h"

/*************************************************************************/
TestDcgmModuleVgpu::TestDcgmModuleVgpu()
{
}

/*************************************************************************/
TestDcgmModuleVgpu::~TestDcgmModuleVgpu()
{
}

/*************************************************************************/
int TestDcgmModuleVgpu::TestStart()
{
    dcgmReturn_t dcgmReturn;
    dcgm_vgpu_msg_start_t startMsg;

    memset(&startMsg, 0, sizeof(startMsg));
    startMsg.header.version = dcgm_vgpu_msg_start_version;

    dcgmReturn = DCGM_CALL_ETBL(m_pEtbl, fpVgpuStart, (m_dcgmHandle, &startMsg));
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "fpVgpuStart failed with %d\n", (int)dcgmReturn);
        return 1;
    }

    return 0;
}

/*************************************************************************/
int TestDcgmModuleVgpu::TestShutdown()
{
    dcgmReturn_t dcgmReturn;
    dcgm_vgpu_msg_shutdown_t shutdownMsg;

    memset(&shutdownMsg, 0, sizeof(shutdownMsg));
    shutdownMsg.header.version = dcgm_vgpu_msg_shutdown_version;

    dcgmReturn = DCGM_CALL_ETBL(m_pEtbl, fpVgpuShutdown, (m_dcgmHandle, &shutdownMsg));
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "fpVgpuShutdown failed with %d\n", (int)dcgmReturn);
        return 1;
    }

    return 0;
}

/*************************************************************************/
std::string TestDcgmModuleVgpu::GetTag()
{
    return std::string("vgpu");
}

/*************************************************************************/
int TestDcgmModuleVgpu::Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = dcgmInternalGetExportTable((const void**)&m_pEtbl, &ETID_DCGMVgpuInternal);
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmInternalGetExportTable failed with %d\n", (int)dcgmReturn);
        return -1;
    }

    return 0;
}

/*************************************************************************/
int TestDcgmModuleVgpu::Run()
{
    int st;
    int Nfailed = 0;

    st = TestStart();
    if(st)
    {
        Nfailed++;
        fprintf(stderr, "TestDcgmModuleVgpu::TestStart FAILED with %d\n", st);
        if(st < 0)
            return -1;
    }
    else
        printf("TestDcgmModuleVgpu::TestStart PASSED\n");


    st = TestShutdown();
    if(st)
    {
        Nfailed++;
        fprintf(stderr, "TestDcgmModuleVgpu::TestShutdown FAILED with %d\n", st);
        if(st < 0)
            return -1;
    }
    else
        printf("TestDcgmModuleVgpu::TestShutdown PASSED\n");

    return 0;
}

/*************************************************************************/
int TestDcgmModuleVgpu::Cleanup()
{
    return 0;
}

/*************************************************************************/
