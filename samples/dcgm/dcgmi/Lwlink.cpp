
#include "Lwlink.h"
#include <sstream>
#include <iostream>
#include <string.h>
#include <stdexcept>
#include "dcgm_structs.h"
#include "dcgm_agent.h"
#include "CommandOutputController.h"
#include "DcgmiOutput.h"
#include "dcgm_agent_internal.h"
#include "lwml.h"
#include <map>

using namespace std;

std::string DISPLAY_LWLINK_ERROR_COUNT_HEADER = "LWLINK Error Counts";

/************************************************************************************/
Lwlink::Lwlink(){
}

Lwlink::~Lwlink(){
}

std::string Lwlink::HelperGetLwlinkErrorCountType(unsigned short fieldId)
{
    // Return the Lwlink error type string based on the fieldId
    switch(fieldId)
    {
        case DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L0:
        case DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L1:
        case DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L2:
        case DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L3:
        case DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L4:
        case DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L5:
            return "CRC FLIT Error";
        case DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L0:
        case DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L1:
        case DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L2:
        case DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L3:
        case DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L4:
        case DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L5:
            return "CRC Data Error";
        case DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L0:
        case DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L1:
        case DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L2:
        case DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L3:
        case DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L4:
        case DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L5:
            return "Replay Error";
        case DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L0:
        case DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L1:
        case DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L2:
        case DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L3:
        case DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L4:
        case DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L5:
            return "Recovery Error";
        default:
            return "Unknown";
    }
}

dcgmReturn_t Lwlink::DisplayLwLinkErrorCountsForGpu(dcgmHandle_t mLwcmHandle, unsigned int gpuId, bool json)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmReturn_t returnResult = DCGM_ST_OK;
    DcgmiOutputTree outTree(30, 50);
    DcgmiOutputJson outJson;
    DcgmiOutput& out = json ? (DcgmiOutput&) outJson : (DcgmiOutput&) outTree;
    unsigned short fieldIds[LWML_LWLINK_ERROR_COUNT * LWML_LWLINK_MAX_LINKS] = {0};
    dcgmFieldValue_v1 values[LWML_LWLINK_ERROR_COUNT * LWML_LWLINK_MAX_LINKS];
    int numFieldIds = LWML_LWLINK_ERROR_COUNT * LWML_LWLINK_MAX_LINKS;
    stringstream ss;
    dcgmFieldGrp_t fieldGroupId;

    //Variable to get the fieldId in fieldIds array
    unsigned int fieldIdStart = 0;
    //Variable to track the count of the lwlink error types for each link
    unsigned int fieldIdCount = 0;
    unsigned int fieldId = 0;

    memset(&values[0], 0, sizeof(values));

    /* Various LWLink error counters to be displayed */
    fieldIds[0] = DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L0;
    fieldIds[1] = DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L0;
    fieldIds[2] = DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L0;
    fieldIds[3] = DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L0;
    fieldIds[4] = DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L1;
    fieldIds[5] = DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L1;
    fieldIds[6] = DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L1;
    fieldIds[7] = DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L1;
    fieldIds[8] = DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L2;
    fieldIds[9] = DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L2;
    fieldIds[10] = DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L2;
    fieldIds[11] = DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L2;
    fieldIds[12] = DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L3;
    fieldIds[13] = DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L3;
    fieldIds[14] = DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L3;
    fieldIds[15] = DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L3;

    fieldIds[16] = DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L4;
    fieldIds[17] = DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L4;
    fieldIds[18] = DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L4;
    fieldIds[19] = DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L4;

    fieldIds[20] = DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L5;
    fieldIds[21] = DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L5;
    fieldIds[22] = DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L5;
    fieldIds[23] = DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L5;
    
    fieldIds[24] = DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L6;
    fieldIds[25] = DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L6;
    fieldIds[26] = DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L6;
    fieldIds[27] = DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L6;
    
    fieldIds[28] = DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L7;
    fieldIds[29] = DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L7;
    fieldIds[30] = DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L7;
    fieldIds[31] = DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L7;
    
    fieldIds[32] = DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L8;
    fieldIds[33] = DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L8;
    fieldIds[34] = DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L8;
    fieldIds[35] = DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L8;
    
    fieldIds[36] = DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L9;
    fieldIds[37] = DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L9;
    fieldIds[38] = DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L9;
    fieldIds[39] = DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L9;
    
    fieldIds[40] = DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L10;
    fieldIds[41] = DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L10;
    fieldIds[42] = DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L10;
    fieldIds[43] = DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L10;
    
    fieldIds[44] = DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L11;
    fieldIds[45] = DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L11;
    fieldIds[46] = DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L11;
    fieldIds[47] = DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L11;
    /* Make sure to update the 2nd parameter to dcgmFieldGroupCreate below if you make this
     * list bigger
     */

    // Add a field group
    result = dcgmFieldGroupCreate(mLwcmHandle, numFieldIds, fieldIds, (char *)"dcgmi_lwlink", &fieldGroupId);
    if(result != DCGM_ST_OK)
    {
        std::cout<<"Error: Unable to add a lwlink field group. Return : "<<errorString(result)<<std::endl;
        PRINT_DEBUG("%d", "Error while adding field group - %d", result);
        return result;
    }

    // Add watch for lwlink error count fields
    result = dcgmWatchFields(mLwcmHandle, (dcgmGpuGrp_t)DCGM_GROUP_ALL_GPUS, fieldGroupId, 1000000, 300, 0);
    if(DCGM_ST_OK != result)
    {
        std::cout<<"Error: Unable to add watch for lwlink error field collections. Return : "<<errorString(result)<<std::endl;
        PRINT_DEBUG("%d", "Error while adding watch for lwlink error count field collection - %d", result);
        returnResult = result;
        goto CLEANUP;
    }

    // Wait for the fields to be updated  
    result = dcgmUpdateAllFields(mLwcmHandle, 1);
    if(DCGM_ST_OK != result)
    {
        std::cout<<"Error Updating the lwlink error count fields. Return: "<<errorString(result)<<std::endl;
        PRINT_DEBUG("%d", "Error while updating the lwlink error count fields - %d", result);
        returnResult = result;
        goto CLEANUP;
    }

    //Header Info
    out.addHeader(DISPLAY_LWLINK_ERROR_COUNT_HEADER);
    ss << "GPU " << gpuId;
    out.addHeader(ss.str());

    //Get the latest values of the fields for the requested gpu Id
    result = dcgmGetLatestValuesForFields(mLwcmHandle , gpuId, fieldIds, numFieldIds , values);
    if(DCGM_ST_OK != result)
    {
        std::cout<<"Error: Unable to retreive latest value for lwlink error counts. Return: "<<errorString(result)<<"."<<std::endl;
        PRINT_ERROR("%d", "Error retrieveing latest value for lwlink error counts : %d", result);
        returnResult = result;
        goto CLEANUP;
    }

    //Display the lwlink errors for each link
    for(unsigned int lwlink = 0;  lwlink < LWML_LWLINK_MAX_LINKS  ; lwlink++)
    {
        for(fieldId = fieldIdStart, fieldIdCount=0; fieldIdCount < LWML_LWLINK_ERROR_COUNT && fieldId < (LWML_LWLINK_ERROR_COUNT * LWML_LWLINK_MAX_LINKS);
               fieldIdCount++, fieldId++)
        {
            if(values[fieldId].status != DCGM_ST_OK)
            {
                std::cout<<"Warning: Unable to retrieve lwlink "<<HelperGetLwlinkErrorCountType(values[fieldId].fieldId)<<" count for link "<<lwlink<<" for gpuId "<<gpuId<<" - "<<errorString((dcgmReturn_t)values[fieldId].status)<<std::endl;
                PRINT_DEBUG("%s %d %d", "Unable to retrieve lwlink %s count for link %d, gpuId %d", HelperGetLwlinkErrorCountType(values[fieldId].fieldId).c_str(), lwlink, gpuId);                
            }
            else
            {
                ss.str("");
                ss << "Link " << lwlink;
                DcgmiOutputBoxer& outLink = out[ss.str()];
                outLink[HelperGetLwlinkErrorCountType(values[fieldId].fieldId)] =
                    (long long)values[fieldId].value.i64;
            }
        }
        
        fieldIdStart = fieldIdStart + LWML_LWLINK_ERROR_COUNT;
    }

    std::cout << out.str();

CLEANUP:
    result = dcgmFieldGroupDestroy(mLwcmHandle, fieldGroupId);
    if(result != DCGM_ST_OK)
    {
        std::cout<<"Error: Unable to remove a lwlink field group. Return : "<<errorString(result)<<std::endl;
        PRINT_ERROR("%d", "Error %d from dcgmFieldGroupDestroy", (int)result);
        /* In cleanup code already. Return retResult from above */
        if(returnResult == DCGM_ST_OK)
            returnResult = result;
    }

    return returnResult;
}

static char lwLinkStateToCharacter(dcgmLwLinkLinkState_t linkState)
{
    switch(linkState)
    {
        case DcgmLwLinkLinkStateDown:
            return 'D';
        case DcgmLwLinkLinkStateUp:
            return 'U';
        case DcgmLwLinkLinkStateDisabled:
            return 'X';
        default:
        case DcgmLwLinkLinkStateNotSupported:
            return '_';
    }
}

static std::string getIndentation(int numIndents)
{
    int i, j;
    std::string retStr;

    for(i = 0; i < numIndents; i++)
    {
        for(j = 0; j < 4; j++)
        {
            retStr.push_back(' ');
        }
    }

    return retStr;
}


dcgmReturn_t Lwlink::DisplayLwLinkLinkStatus(dcgmHandle_t dcgmHandle)
{
    dcgmReturn_t result;
    dcgmLwLinkStatus_v2 linkStatus;
    unsigned int i, j;

    memset(&linkStatus, 0, sizeof(linkStatus));
    linkStatus.version = dcgmLwLinkStatus_version2;

    result = dcgmGetLwLinkLinkStatus(dcgmHandle, &linkStatus);

    if(result != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to retrieve LwLink link status from DCGM. Return: " 
                  << errorString(result)<<"."<<std::endl;
        PRINT_ERROR("%d", "Unable to retrieve LwLink link status from DCGM. Return: %d", result);
        return result;
    }

    std::cout << "+----------------------+" << std::endl
              << "|  LwLink Link Status  |" << std::endl
              << "+----------------------+" << std::endl;

    std::cout << "GPUs:" << std::endl;

    if(linkStatus.numGpus < 1)
    {
        std::cout << getIndentation(1) << "No GPUs found." << std::endl;
    }
    else
    {
        for(i = 0; i < linkStatus.numGpus; i++)
        {
            std::cout << getIndentation(1) << "gpuId " << linkStatus.gpus[i].entityId << ":" << std::endl << "        ";
            for(j = 0; j < DCGM_LWLINK_MAX_LINKS_PER_GPU; j++)
            {
                if(j > 0)
                    std::cout << " ";
                std::cout << lwLinkStateToCharacter(linkStatus.gpus[i].linkState[j]);
            }
            std::cout << std::endl;
        }
    }

    std::cout << "LwSwitches:" << std::endl;

    if(linkStatus.numLwSwitches < 1)
    {
        std::cout << "    No LwSwitches found." << std::endl;
    }
    else
    {
        for(i = 0; i < linkStatus.numLwSwitches; i++)
        {
            std::cout << getIndentation(1) << "physicalId " << linkStatus.lwSwitches[i].entityId << ":" << std::endl << "        ";
            for(j = 0; j < DCGM_LWLINK_MAX_LINKS_PER_LWSWITCH; j++)
            {
                if(j > 0)
                    std::cout << " ";
                std::cout << lwLinkStateToCharacter(linkStatus.lwSwitches[i].linkState[j]);
            }
            std::cout << std::endl;
        }
    }

    std::cout << std::endl
              << "Key: Up=U, Down=D, Disabled=X, Not Supported=_" << std::endl;


    return DCGM_ST_OK;
}

GetGpuLwlinkErrorCounts::GetGpuLwlinkErrorCounts(std::string hostname, unsigned int gpuId, bool json)
{
    mHostName = hostname;
    mGpuId = gpuId;
    mJson = json;
}    

GetGpuLwlinkErrorCounts::~GetGpuLwlinkErrorCounts()
{
}

int GetGpuLwlinkErrorCounts::Execute() 
{
    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }

    return mLwlinkObj.DisplayLwLinkErrorCountsForGpu(mLwcmHandle, mGpuId, mJson);
}


GetLwLinkLinkStatuses::GetLwLinkLinkStatuses(std::string hostname)
{
    mHostName = hostname;
}    

GetLwLinkLinkStatuses::~GetLwLinkLinkStatuses()
{
}

int GetLwLinkLinkStatuses::Execute() 
{
    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    
    return mLwlinkObj.DisplayLwLinkLinkStatus(mLwcmHandle);
}
