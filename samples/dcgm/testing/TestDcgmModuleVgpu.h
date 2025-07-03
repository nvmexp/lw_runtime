#ifndef TESTLWCMMODULEVGPU_H
#define TESTLWCMMODULEVGPU_H

#include "TestLwcmModule.h"
#include "dcgm_fields.h"
#include "timelib.h"
#include "LwcmGroup.h"
#include "dcgm_vgpu_internal.h"

class TestDcgmModuleVgpu : public TestLwcmModule {
public:
    TestDcgmModuleVgpu();
    virtual ~TestDcgmModuleVgpu();

    int Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus);
    int Run();
    int Cleanup();
    std::string GetTag();

private:

    etblDCGMVgpuInternal *m_pEtbl;

    int TestStart();
    int TestShutdown();
};

#endif //TESTLWCMMODULEVGPU_H
