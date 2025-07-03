
#include "TestDcgmModuleLwSwitch.h"
#include "dcgm_lwswitch_internal.h"

/*************************************************************************/
TestDcgmModuleLwSwitch::TestDcgmModuleLwSwitch()
{
}

/*************************************************************************/
TestDcgmModuleLwSwitch::~TestDcgmModuleLwSwitch()
{
}

/*************************************************************************/
int TestDcgmModuleLwSwitch::TestStart()
{
    dcgmReturn_t dcgmReturn;
    dcgm_lwswitch_msg_start_t startMsg;

    memset(&startMsg, 0, sizeof(startMsg));
    startMsg.header.version = dcgm_lwswitch_msg_start_version;
    startMsg.startLocal = 1;
    startMsg.startGlobal = 1;

    dcgmReturn = DCGM_CALL_ETBL(m_pEtbl, fpLwswitchStart, (m_dcgmHandle, &startMsg));
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "fpLwswitchStart failed with %d\n", (int)dcgmReturn);
        return 1;
    }

// let Global and Local Fabric Managers cycle for a while.  For product they will become daemons

    sleep(20);

    return 0;
}

/*************************************************************************/
int TestDcgmModuleLwSwitch::TestShutdown()
{
    dcgmReturn_t dcgmReturn;
    dcgm_lwswitch_msg_shutdown_t shutdownMsg;

    memset(&shutdownMsg, 0, sizeof(shutdownMsg));
    shutdownMsg.header.version = dcgm_lwswitch_msg_shutdown_version;

    dcgmReturn = DCGM_CALL_ETBL(m_pEtbl, fpLwswitchShutdown, (m_dcgmHandle, &shutdownMsg));
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "fpLwswitchShutdown failed with %d\n", (int)dcgmReturn);
        return 1;
    }

    return 0;
}

/*************************************************************************/
std::string TestDcgmModuleLwSwitch::GetTag()
{
    return std::string("lwswitch");
}

/*************************************************************************/
int TestDcgmModuleLwSwitch::Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = dcgmInternalGetExportTable((const void**)&m_pEtbl, &ETID_DCGMLwSwitchInternal);
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmInternalGetExportTable failed with %d\n", (int)dcgmReturn);
        return -1;
    }

    return 0;
}

/*************************************************************************/
int TestDcgmModuleLwSwitch::Run()
{
    int st;
    int Nfailed = 0;

    st = TestStart();
    if(st)
    {
        Nfailed++;
        fprintf(stderr, "TestDcgmModuleLwSwitch::TestStart FAILED with %d\n", st);
        if(st < 0)
            return -1;
    }
    else
        printf("TestDcgmModuleLwSwitch::TestStart PASSED\n");


    st = TestShutdown();
    if(st)
    {
        Nfailed++;
        fprintf(stderr, "TestDcgmModuleLwSwitch::TestShutdown FAILED with %d\n", st);
        if(st < 0)
            return -1;
    }
    else
        printf("TestDcgmModuleLwSwitch::TestShutdown PASSED\n");

    return 0;
}

/*************************************************************************/
int TestDcgmModuleLwSwitch::Cleanup()
{
    return 0;
}

/*************************************************************************/
