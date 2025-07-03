#pragma once

#include "Command.h"
#include "DcgmiOutput.h"
#include "dcgm_structs.h"

class DcgmiProfile 
{
public:
    DcgmiProfile();
    virtual ~DcgmiProfile();
    dcgmReturn_t RunProfileList(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, bool outputAsJson);
    dcgmReturn_t RunProfileSetPause(dcgmHandle_t dcgmHandle, bool pause);
    
    dcgmReturn_t statusToStr(dcgmModuleStatus_t status, std::string& str);
private:
};

class DcgmiProfileList : public Command
{
public:
    DcgmiProfileList(const std::string& hostname, const std::string &gpuIdsStr, 
                     const std::string &groupIdStr, bool outputAsJson);
    virtual ~DcgmiProfileList();
    int Execute();

private:
    DcgmiProfile mProfileObj;
    std::string mGpuIdsStr;
    std::string mGroupIdStr;
    
    dcgmGpuGrp_t mGroupId;

    dcgmReturn_t ValidateOrCreateEntityGroup(void); /* Set mGroupId based on mGpuIdsStr and mGroupIdStr */
    dcgmReturn_t CreateEntityGroupFromEntityList(void); /* Helper called by ValidateOrCreateEntityGroup() */
};

class DcgmiProfileSetPause : public Command
{
public:
    DcgmiProfileSetPause(const std::string& hostname, bool pause);
    virtual ~DcgmiProfileSetPause();
    int Execute();

private:
    DcgmiProfile mProfileObj;
    bool m_pause; /* Should we pause (true) or resume (false) */
};
