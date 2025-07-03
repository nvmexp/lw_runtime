/*
 * Group.h
 *
 */

#ifndef GROUP_H_
#define GROUP_H_

#include "Command.h"
#include "DcgmiOutput.h"
#include <vector>

class Group {
public:
    Group();
    virtual ~Group();

    /*****************************************************************************
     * This method is used to list the groups on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunGroupList(dcgmHandle_t pDcgmHandle, bool json);

    /*****************************************************************************
     * This method is used to create groups on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunGroupCreate(dcgmHandle_t pDcgmHandle, dcgmGroupType_t groupType);

    /*****************************************************************************
     * This method is used to remove groups on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunGroupDestroy(dcgmHandle_t pDcgmHandle);

    /*****************************************************************************
     * This method is used to get the info for a group on the host-engine represented
     * by the DCGM handle
     * It is overloaded in order to allow reuse in RunGroupList
     *****************************************************************************/
    dcgmReturn_t RunGroupInfo(dcgmHandle_t pDcgmHandle, bool json);
    dcgmReturn_t RunGroupInfo(dcgmHandle_t mDcgmHandle, DcgmiOutputBoxer& outGroup);

    /*****************************************************************************
     * This method is used to add to or remove from a group on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunGroupManageDevice(dcgmHandle_t pDcgmHandle, bool add);

    /******************************************************************************
     * Getters and setters
     ******************************************************************************/
    void SetGroupId(unsigned int id);
    unsigned int GetGroupId();
    void SetGroupName(std::string name);
    std::string getGroupName();
    void SetGroupInfo(std::string name);
    std::string getGroupInfo();

private:
    dcgmGpuGrp_t groupId;
    std::string groupName;
    std::string groupInfo;
};


/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * List Group Ilwoker class
 */
class GroupList : public Command
{
public:
    GroupList(std::string hostname, bool json);
    virtual ~GroupList();

    int Execute();

private:
    Group groupObj;
};

/**
 * Create Group Ilwoker class
 */
class GroupCreate : public Command
{
public:
    GroupCreate(std::string hostname, Group &obj, dcgmGroupType_t groupType);
    virtual ~GroupCreate();

    int Execute();

private:
    Group groupObj;
    dcgmGroupType_t groupType;
};

/**
 * Destroy Group Ilwoker class
 */
class GroupDestroy : public Command
{
public:
    GroupDestroy(std::string hostname, Group &obj);
    virtual ~GroupDestroy();

    int Execute();

private:
    Group groupObj;
};

/**
 * Group Info Ilwoker class
 */
class GroupInfo : public Command
{
public:
    GroupInfo(std::string hostname, Group &obj, bool json);
    virtual ~GroupInfo();

    int Execute();

private:
    Group groupObj;
};

/**
 * Add to Group Ilwoker class
 */
class GroupAddTo : public Command
{
public:
    GroupAddTo(std::string hostname, Group &obj);
    virtual ~GroupAddTo();

    int Execute();

private:
    Group groupObj;
};

/**
 * Add to Group Ilwoker class
 */
class GroupDeleteFrom : public Command
{
public:
    GroupDeleteFrom(std::string hostname, Group &obj);
    virtual ~GroupDeleteFrom();

    int Execute();

private:
    Group groupObj;
};

#endif /* GROUP_H_ */
