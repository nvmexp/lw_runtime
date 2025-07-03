#ifndef DCGMI_COMMON_H
#define DCGMI_COMMON_H

#include "dcgm_structs.h"
#include <vector>

/*****************************************************************************/
/* Defines */
#define _DCGMI_FORMAL_NAME "LWPU Datacenter GPU Management Interface"
#define _DCGMI_VERSION DCGM_VERSION_STRING

/*****************************************************************************/
/* Unique key for an entityGroupId:entityId combo */
typedef struct dcgmi_entity_pair_t
{
    dcgm_field_entity_group_t entityGroupId;   //!< Entity Group ID entity belongs to
    dcgm_field_eid_t entityId;                 //!< Entity ID of the entity

    /* == operator needed for using this struct in a std::map */
    bool operator==(const dcgmi_entity_pair_t& a) const
    {   
        return (entityGroupId == a.entityGroupId && entityId == a.entityId);
    }

    /* < operator needed for using this struct in a std::map */
    bool operator<(const dcgmi_entity_pair_t& a) const
    {   
        return (entityGroupId < a.entityGroupId || entityId < a.entityId);
    }
} dcgmi_entity_pair_t;

/*****************************************************************************/
/*
 * Parse an entity list string into an array of entity pairs
 * 
 * input       IN: Comma-separated list of entities like "0,1,2" or "gpu:0,lwswitch:996"
 * entityList OUT: List of entities parsed. This is only populated if the return is DCGM_ST_OK
 * 
 * Returns: DCGM_ST_OK if parsing was successful
 *          Other DCGM_ST_* error if a problem oclwrred. This will print an error to stdout
 * 
 */
dcgmReturn_t dcgmi_parse_entity_list_string(std::string input, std::vector<dcgmGroupEntityPair_t>&entityList);

/*****************************************************************************/
/*
 * Create an entity group with the entities in entityList as members
 * 
 * This helper can be used if you don't care about the name of your entity group
 * one will randomly be assigned
 * 
 * dcgmHandle     IN: Connection handle to the host engine
 * groupType      IN: Type of group to create (empty, all gpus, ...etc)
 * groupId       OUT: group ID handle of the group that was created. This is only 
 *                    valid if the return was DCGM_ST_OK
 * entityList     IN: List of entities to add to the group
 * 
 * Returns: DCGM_ST_OK if the group was successfully created
 *          Other DCGM_ST_* error if a problem oclwrred. This will print an error to stdout
 *                    
 */
dcgmReturn_t dcgmi_create_entity_group(dcgmHandle_t dcgmHandle, dcgmGroupType_t groupType,
                                       dcgmGpuGrp_t *groupId, 
                                       std::vector<dcgmGroupEntityPair_t>&entityList);

/*****************************************************************************/
/*
 * Parses an entity group ID string to see if it contains special group names
 * 
 * all_gpus and all_lwswitches are considered special group names and will result in
 * true being returned.
 * 
 * groupIdStr  IN: String of a group ID like "1" or "all_gpus"
 * groupType  OUT: Group type of the special group. Will be set if true is returned
 * groupId    OUT: Group ID. Will be set if true is returned
 * 
 * Returns true if groupIdStr contained a special group name. groupType will contain 
 *             the groupType that corresponds with groupIdStr
 *         false if groupIdStr did not contain a special group name
 */
bool dcgmi_entity_group_id_is_special(std::string &groupIdStr, dcgmGroupType_t *groupType,
                                      dcgmGpuGrp_t *groupId);

/*****************************************************************************/
/*
 * Parse a field ID list string into an array of field IDs
 * 
 * input       IN: Comma-separated list of field IDs like "0,1,2"
 * fieldIds   OUT: List of field IDs that was parsed. This is only populated if the return is DCGM_ST_OK
 * validate    IN: If true, each field ID will be looked up to see if it's actually valid
 * 
 * Returns: DCGM_ST_OK if parsing was successful
 *          Other DCGM_ST_* error if a problem oclwrred. This will print an error to stdout
 * 
 */
dcgmReturn_t dcgmi_parse_field_id_list_string(std::string input, 
                                              std::vector<unsigned short>&fieldIds, 
                                              bool validate);

/*****************************************************************************/
/*
 * Create a field group with the field IDs in fieldIds as members
 * 
 * This helper can be used if you don't care about the name of your field group.
 * one will randomly be assigned
 * 
 * dcgmHandle     IN: Connection handle to the host engine
 * groupId       OUT: group ID handle of the group that was created. This is only 
 *                    valid if the return was DCGM_ST_OK
 * fieldIds       IN: List of field IDs to add to the group
 * 
 * Returns: DCGM_ST_OK if the group was successfully created
 *          Other DCGM_ST_* error if a problem oclwrred. This will print an error to stdout
 *                    
 */
dcgmReturn_t dcgmi_create_field_group(dcgmHandle_t dcgmHandle, 
                                      dcgmFieldGrp_t *groupId, 
                                      std::vector<unsigned short>&fieldIds);

/*****************************************************************************/
/*
 * Parse a hostname string, returning a pointer to the start of the actual hostname
 *
 * hostName             IN: Hostname to parse. If this starts with "file:", then 
 *                          isUnixSocketAddress will be set to true
 * isUnixSocketAddress OUT: Whether (true) or not the given hostName appears to be
 *                          a unix socket filename
 * logOnError           IN: Whether or not to log to stdout on error
 * 
 * Returns: Pointer to hostname on success
 *          NULL on error. Will log to stdout if logOnError == true
 */
const char *dcgmi_parse_hostname_string(const char *hostName, bool *isUnixSocketAddress,
                                        bool logOnError);

/*****************************************************************************/

#endif // DCGMI_COMMON_H
