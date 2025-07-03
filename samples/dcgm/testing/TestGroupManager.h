/* 
 * File:   TestGroupManager.h
 */

#ifndef TESTGROUPMANAGER_H
#define	TESTGROUPMANAGER_H

#include "TestLwcmModule.h"
#include "dcgm_fields.h"
#include "timelib.h"
#include "LwcmGroup.h"


class TestGroupManager : public TestLwcmModule {
public:
    TestGroupManager();
    virtual ~TestGroupManager();
    
    int Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus);
    int Run();
    int Cleanup();
    std::string GetTag();    
    
private:

    int TestGroupCreation();
    int TestGroupGetAllGrpIds();
    int TestGroupManageGpus();
    int TestGroupReportErrOnDuplicate();
    int TestDefaultGpusAreDynamic();
    int TestDefaultLwSwitchesAreDynamic();

    int HelperOperationsOnGroup(LwcmGroupManager *pLwcmGrpManager, unsigned int groupId, string groupName,
                                DcgmCacheManager *cacheManager);

    std::vector<test_lwcm_gpu_t>m_gpus; /* List of GPUs to run on, copied in Init() */
};

#endif	/* TESTGROUPMANAGER_H */

