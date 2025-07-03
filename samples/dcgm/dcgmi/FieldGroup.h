#ifndef FIELDGROUP_H
#define FIELDGROUP_H

#include "Command.h"
#include <vector>

class FieldGroup {
public:
    FieldGroup();
    virtual ~FieldGroup();

    /*****************************************************************************
     * This method is used to list the field groups on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunGroupListAll(dcgmHandle_t dcgmHandle, bool json);

    /*****************************************************************************
     * This method is used to query field group info on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunGroupInfo(dcgmHandle_t dcgmHandle, bool json);

    /*****************************************************************************
     * This method is used to create groups on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunGroupCreate(dcgmHandle_t dcgmHandle);

    /*****************************************************************************
     * This method is used to delete a group on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunGroupDestroy(dcgmHandle_t pLwcmHandle);

    /******************************************************************************
     * Getters and setters
     ******************************************************************************/
    void SetFieldGroupId(unsigned int id);
    unsigned int GetFieldGroupId();
    void SetFieldGroupName(std::string name);
    std::string GetFieldGroupName();
    void SetFieldIdsString(std::string fieldIdsStr);

private:
    /*****************************************************************************
     * helper method to take a string of form x,y,z... and
     * transform that into a list of field IDs
     *****************************************************************************/
    dcgmReturn_t HelperFieldIdStringParse(std::string input, std::vector<unsigned short>&fieldIds);

    unsigned int m_fieldGroupId;
    std::string m_fieldGroupName;
    std::string m_fieldIdsStr;
};


/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * List Group Ilwoker class
 */
class FieldGroupListAll : public Command
{
public:
    FieldGroupListAll(std::string hostname, bool json);
    virtual ~FieldGroupListAll();

    int Execute();

private:
    FieldGroup fieldGroupObj;
};

/**
 * Create Field Group Ilwoker class
 */
class FieldGroupCreate : public Command
{
public:
    FieldGroupCreate(std::string hostname, FieldGroup &obj);
    virtual ~FieldGroupCreate();

    int Execute();

private:
    FieldGroup fieldGroupObj;
};

/**
 * Destroy Field Group Ilwoker class
 */
class FieldGroupDestroy : public Command
{
public:
    FieldGroupDestroy(std::string hostname, FieldGroup &obj);
    virtual ~FieldGroupDestroy();

    int Execute();

private:
    FieldGroup fieldGroupObj;
};

/**
 * Field Group Info Ilwoker class
 */
class FieldGroupInfo : public Command
{
public:
    FieldGroupInfo(std::string hostname, FieldGroup &obj, bool json);
    virtual ~FieldGroupInfo();

    int Execute();

private:
    FieldGroup fieldGroupObj;
};


#endif /* GROUP_H_ */
