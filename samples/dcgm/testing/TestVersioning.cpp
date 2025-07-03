#include "TestVersioning.h"
#include <stddef.h>


TestVersioning::TestVersioning() {}
TestVersioning::~TestVersioning() {}

int TestVersioning::Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus)
{
    return 0;
}

int TestVersioning::Run()
{
    int st;
    int Nfailed = 0;

    st = TestBasicAPIVersions();
    if(st)
    {
        Nfailed++;
        fprintf(stderr, "TestVersioning::TestBasicAPIVersions FAILED with %d\n", st);
        if(st < 0)
            return -1;
    }
    
    printf("TestVersioning::TestBasicAPIVersions PASSED\n");

    if(Nfailed > 0)
    {
        fprintf(stderr, "%d tests FAILED\n", Nfailed);
        return 1;
    }

    return 0;    
}

int TestVersioning::Cleanup()
{
    return 0;
}

std::string TestVersioning::GetTag()
{
    return std::string("versioning");
}

int recognizeAPIVersion(dcgmVersionTest_t * dummyStruct)
{
    dummyStruct->a = 5;
    if (dummyStruct->version >= dcgmVersionTest_version2)
        dummyStruct->b = 10;
    if (dummyStruct->version > dcgmVersionTest_version2)
        return -1;

    return 0;
}

int TestVersioning::TestBasicAPIVersions()
{
    dcgmVersionTest_v1 api_v1;
    dcgmVersionTest_v2 api_v2;

    api_v1.version = dcgmVersionTest_version1;
    api_v2.version = dcgmVersionTest_version2;

    /* Test the positive */
    if (recognizeAPIVersion((dcgmVersionTest_t*)&api_v1) != 0)
        return -1;
    if (recognizeAPIVersion((dcgmVersionTest_t*)&api_v2) != 0)
        return -1;

    /* Test the negative */
    api_v2.version = dcgmVersionTest_version3;
    if (recognizeAPIVersion((dcgmVersionTest_t*)&api_v2) == 0)
        return -1;
    return 0;
}


