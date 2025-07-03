
#include <sstream>
#include <iostream>
#include <string.h>
#include "dcgmi_common.h"
#include "DcgmStringTokenize.h"
#include "dcgm_agent.h"
#include "dcgm_fields.h"
#include "lwos.h"

/*******************************************************************************/
dcgmReturn_t dcgmi_parse_entity_list_string(std::string input, std::vector<dcgmGroupEntityPair_t>&entityList)
{
    dcgmReturn_t result = DCGM_ST_OK;
    std::stringstream ss(input);
    unsigned int uintTemp;
    dcgmGroupEntityPair_t insertElem;
    std::string entityIdStr;
    std::vector<std::string>tokens;
    unsigned int i;

    /* Divide the string into a vector of substrings by comma */
    std::string delimStr = ",";

    /* This is expecting input to be strings like:
       "0,1,2" or
       "gpu:0,lwswitch:3,2" (gpu 0, lwswitch 3, gpu 2) */
    tokenizeString(input, delimStr, tokens);

    for(i = 0; i < tokens.size(); i++)
    {
        entityIdStr = tokens[i];
        if(entityIdStr.size() < 1)
        {
            std::cout << "Error: Comma without a value detected at token " << i+1 << " of " << input << std::endl;
            return DCGM_ST_BADPARAM;
        }
        
        /* Default to GPUs in the case that the entityGroupId isn't specified */
        insertElem.entityGroupId = DCGM_FE_GPU;

        /* See if this has an entityGroup on the front */
        size_t colonPos = entityIdStr.find_first_of(":");
        if(colonPos != std::string::npos)
        {
            switch(entityIdStr.at(0))
            {
                case 'g':
                case 'G':
                    insertElem.entityGroupId = DCGM_FE_GPU;
                    break;
                case 'n':
                case 'N':
                    insertElem.entityGroupId = DCGM_FE_SWITCH;
                    break;
                case 'v':
                case 'V':
                    insertElem.entityGroupId = DCGM_FE_VGPU;
                    break;
                default:
                    std::cout << "Error: invalid entity type: '" << entityIdStr 
                            << "'. Expected gpu/vgpu/lwswitch." << std::endl;
                    return DCGM_ST_BADPARAM;
            }
            
            /* Move past the colon */
            entityIdStr = entityIdStr.substr(colonPos+1);
        }

        if(entityIdStr.size() < 1)
        {
            std::cout << "Error: empty entityId detected in " << entityIdStr << std::endl;
            return DCGM_ST_BADPARAM;
        }

        /* Add an item */
        if(isdigit(entityIdStr.at(0)))
        {
            insertElem.entityId = atoi(entityIdStr.c_str());
            entityList.push_back(insertElem);
        }
        else
        {
            std::cout << "Error: Expected numerical entityId instead of " << entityIdStr << std::endl;
            return DCGM_ST_BADPARAM;
        }
    }

    return result;
}

/*******************************************************************************/
dcgmReturn_t dcgmi_create_entity_group(dcgmHandle_t dcgmHandle, dcgmGroupType_t groupType, 
                                       dcgmGpuGrp_t *groupId, 
                                       std::vector<dcgmGroupEntityPair_t>&entityList)
{
    dcgmReturn_t dcgmReturn;
    static int numGroupsCreated = 0;
    unsigned int myPid = (unsigned int)lwosProcessId();
    char groupName[32] = {0};
    unsigned int i;

    snprintf(groupName, sizeof(groupName)-1, "dcgmi_%u_%d", myPid, ++numGroupsCreated);

    dcgmReturn = dcgmGroupCreate(dcgmHandle, groupType, groupName, groupId);
    if(dcgmReturn != DCGM_ST_OK)
    {
        std::cout << "Got error while creating a GPU group: " << errorString(dcgmReturn) << std::endl;
        return dcgmReturn;
    }

    for(i = 0; i < entityList.size(); i++)
    {
        dcgmReturn = dcgmGroupAddEntity(dcgmHandle, *groupId, entityList[i].entityGroupId, 
                                        entityList[i].entityId);
        if(dcgmReturn != DCGM_ST_OK)
        {
            std::cout << "Error: Got error " << errorString(dcgmReturn) << " while "
                      << "adding " << DcgmFieldsGetEntityGroupString(entityList[i].entityGroupId)
                      << " " << entityList[i].entityId << " to our entity group." << std::endl;
            return dcgmReturn;
        }
    }

    return DCGM_ST_OK;
}

/*******************************************************************************/
bool dcgmi_entity_group_id_is_special(std::string &groupIdStr, dcgmGroupType_t *groupType,
                                      dcgmGpuGrp_t *groupId)
{
    if(groupIdStr == "all_gpus")
    {
        *groupType = DCGM_GROUP_DEFAULT;
        *groupId = (void *)(intptr_t)DCGM_GROUP_ALL_GPUS;
        return true;
    }
    else if(groupIdStr == "all_lwswitches")
    {
        *groupType = DCGM_GROUP_DEFAULT_LWSWITCHES;
        *groupId = (void *)(intptr_t)DCGM_GROUP_ALL_LWSWITCHES;
        return true;
    }
    else
    {
        *groupType = DCGM_GROUP_EMPTY;
        return false;
    }
}

/*******************************************************************************/
dcgmReturn_t dcgmi_parse_field_id_list_string(std::string input, 
                                              std::vector<unsigned short>&fieldIds, 
                                              bool validate)
{
    dcgmReturn_t result = DCGM_ST_OK;
    std::vector<std::string>tokens;
    unsigned int i; 
    unsigned short fieldId;
    dcgm_field_meta_p fieldMeta;

    /* Divide the string into a vector of substrings by comma */
    std::string delimStr = ",";
    tokenizeString(input, delimStr, tokens);

    /* Colwert each token into an unsigned integer */
    for(i = 0; i < tokens.size(); i++)
    {
        fieldId = atoi(tokens[i].c_str());
        if(!fieldId && (!tokens[i].size() || tokens[i].at(0) != '0'))
        {
            std::cout << "Error: Expected numerical fieldId. Got '" << tokens[i] << "' instead." << std::endl;
            return DCGM_ST_BADPARAM;
        }

        if(validate)
        {
            fieldMeta = DcgmFieldGetById(fieldId);
            if(!fieldMeta)
            {
                std::cout << "Error: Got invalid field ID. '" << fieldId 
                          << "'. See dcgm_fields.h for a list of valid field IDs." << std::endl;
                return DCGM_ST_BADPARAM;
            }
        }

        fieldIds.push_back(fieldId);
    }

    return DCGM_ST_OK;
}

/*******************************************************************************/
dcgmReturn_t dcgmi_create_field_group(dcgmHandle_t dcgmHandle, 
                                      dcgmFieldGrp_t *groupId, 
                                      std::vector<unsigned short>&fieldIds)
{
    dcgmReturn_t dcgmReturn;
    static int numGroupsCreated = 0;
    unsigned int myPid = (unsigned int)lwosProcessId();
    char groupName[32] = {0};

    snprintf(groupName, sizeof(groupName)-1, "dcgmi_%u_%d", myPid, ++numGroupsCreated);

    dcgmReturn = dcgmFieldGroupCreate(dcgmHandle, fieldIds.size(), &fieldIds[0], groupName, groupId);
    if(dcgmReturn != DCGM_ST_OK)
    {
        std::cout << "Got error while creating a Field Group: " << errorString(dcgmReturn) << std::endl;
    }

    return dcgmReturn;
}

/*******************************************************************************/
const char *dcgmi_parse_hostname_string(const char *hostName, bool *isUnixSocketAddress, bool logOnError)
{
    if(!strncmp(hostName, "unix://", 7))
    {
        *isUnixSocketAddress = true;
        /* Looks like a unix socket. Do some validation */
        if(strlen(hostName) < 8)
        {
            if(logOnError)
                std::cout << "Missing hostname after \"unix://\"." << std::endl;
            
            return NULL;
        }
        else
            return &hostName[7];
    }

    /* No "unix://". Treat like a regular hostname */
    *isUnixSocketAddress = false;
    return hostName;
}

/*******************************************************************************/
