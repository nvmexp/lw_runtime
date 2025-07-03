/*
 *  DeviceMonitor.h
 */

#ifndef DEVICEMONITOR_H_
#define DEVICEMONITOR_H_

#include "Command.h"
#include "dcgm_structs.h"
#include <sstream>
#include <string.h>
#include "dcgm_structs.h"
#include "dcgm_agent.h"
#include "CommandOutputController.h"
#include <signal.h>
#include <iomanip>
#include <vector>
#include <map>
#include <climits>
#include <algorithm>
#include "dcgmi_common.h"

class DeviceMonitor
{
    public:
        DeviceMonitor();
        virtual ~DeviceMonitor();
};

/**
 * The structure stores details about the list options
 * of the dmon by being used as placeholder for
 * storing details like long name, short name and field id.
 */
struct Field_Details
{
    std::string  m_longName;   /* Long name of the field. It could be used to lookup the field id
                                 and short name with -l option.*/
    std::string  m_shortName;  /* Short name of the field. This short name can be used to identify the
                                 fields used to see the field details when listed using -l option. */
    unsigned int m_fieldId;   /* The Field Id. This could be used with -e option in dcgmi dmon. */
};

class DeviceInfo : public Command
{
public:
    DeviceInfo(std::string hostname, std::string requestedGPUIds, std::string grpIdStr, 
               std::string fldIds, int fieldGroupId, int dlay, int cnt, bool lst, bool watchFields);
    virtual ~DeviceInfo();
    
    int Execute();
        
private:
    
    std::map<dcgmi_entity_pair_t, std::vector<std::string> > m_entityStats; /* map that holds the entity ID and vector of values for fields that
                                                                            are queried in dmon are requested. Populated in callback of 
                                                                            dcgmGetLatestValues_v2. */
    dcgmHandle_t    m_dcgmHandle;                                        /* dcgmHandle for the field value retrieval.*/
    std::string     m_requestedEntityIds;                                /* The Entity Ids requested in the options for command.*/
    std::string     m_groupIdStr;                                        /* The group Id mentioned in command. */
    std::string     m_fieldIdsStr;                                       /* The Field Ids to query for. Mentioned in the options.  */ 
    std::vector<unsigned short>m_fieldIds;                               /* The actual field IDs queried, either parsed from m_fieldIdsStr or
                                                                            retrieved via m_fieldGrpId_int */
    int             m_fieldGrpId_int;                                    /* The integer colwerted value of FieldGroup Id 
                                                                            mentioned in the command.*/
    int             m_delay;                                             /* Delay after which the values to be updated.*/
    int             m_count;                                             /* Number of times to iterate before exiting.*/
    dcgmGpuGrp_t    m_myGroupId;                                         /* Gpu group id of the group created by dmon. */
    dcgmFieldGrp_t  m_fieldGroupId;                                      /* Field group id of the fieldgroup created by the dmon.*/
    bool            m_list;                                              /* Boolean value that states if list option is mentioned in the
                                                                            command line.*/
    bool            m_watchFields;                                       /* Should we (true) or not (false) create watches on the fields? */
    std::vector<Field_Details> m_fieldDetails;                           /* Vector or the field details structure for each field. Populated
                                                                            and used when list option is mentioned with command.  */
    std::string             m_header;                                    /* The header for formatting the output. */
    std::string             m_headerUnit;                                /* The unit for the header values. (Eg: C(elsius) for Temperature
                                                                            or W(att) for Power). */
    unsigned short          m_widthArray[UCHAR_MAX];                     /* The array that holds width for each header element in
                                                                            order of  their oclwrence in the output. Max number of width
                                                                            that header can accomodate is 255. Each value stands for width
                                                                            for next field id in the horizontal row for each gpu.*/

    
    /**********************************************************************
     *  ValidateOrCreateEntityGroup
     * 
     *  - Validates and creates group of entities mentioned if entityIds are mentioned.
     *  - Validates and fetches existing entity group if groupId is mentioned.
     *
     *  @return dcgmReturn_t : result type - success or error
     */
    dcgmReturn_t ValidateOrCreateEntityGroup(void);

    /**********************************************************************
     * ValidateOrCreateFieldGroup
     * 
     *  - Validates and creates group of m_fieldIds mentioned.
     *  - Validates and fetches existing FieldId group if fieldgroupId is mentioned.
     *
     *  @return dcgmReturn_t : result type - success or error
     */
    dcgmReturn_t ValidateOrCreateFieldGroup(void);

    dcgmReturn_t CreateFieldGroupAndGpuGroup(void);

    dcgmReturn_t CreateEntityGroupFromEntityList(void);

    dcgmReturn_t LockWatchAndUpdate();
    void SetHeaderForOutput(std::vector<std::string> ,int);
    void PrintHeaderForOutput();
    void SetHeaderForOutput(unsigned short a[], unsigned int b);
    void PopulateFieldDetails();
    dcgmReturn_t DisplayFieldDetails();
        
};
#endif /* DEVICEMONITOR_H_ */

