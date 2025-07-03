#ifndef TESTLWCMMODULELWSWITCH_H
#define TESTLWCMMODULELWSWITCH_H

#include "TestLwcmModule.h"
#include "dcgm_fields.h"
#include "timelib.h"
#include "LwcmGroup.h"
#include "dcgm_lwswitch_internal.h"

class TestDcgmModuleLwSwitch : public TestLwcmModule {
public:
    TestDcgmModuleLwSwitch();
    virtual ~TestDcgmModuleLwSwitch();

    int Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus);
    int Run();
    int Cleanup();
    std::string GetTag();

private:

    etblDCGMLwSwitchInternal *m_pEtbl;

    int TestStart();
    int TestShutdown();
};

#endif //TESTLWCMMODULELWSWITCH_H
