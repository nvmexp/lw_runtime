
#include "Topo.h"
#include <sstream>
#include <iostream>
#include <string.h>
#include <stdexcept>
#include "dcgm_structs.h"
#include "dcgm_agent.h"
#include "CommandOutputController.h"
#include "DcgmiOutput.h"
#include <map>

using namespace std;


/* Device Info */
#define MAX_SIZE_OF_AFFINITY_STRING 54 // Used for overflow (ATTRIBUTE_DATA_TAG tag)

#define HEADER_NAME "Topology Information"

/*****************************************************************************************/

Topo::Topo() {
    // TODO Auto-generated constructor stub

}

Topo::~Topo() {
    // TODO Auto-generated destructor stub
}

/********************************************************************************/
dcgmReturn_t Topo::DisplayGPUTopology(dcgmHandle_t mLwcmHandle, unsigned int requestedGPUId, bool json)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmDeviceTopology_t gpuTopo;
    gpuTopo.version = dcgmDeviceTopology_version;
    DcgmiOutputTree outTree(20, 60);
    DcgmiOutputJson outJson;
    DcgmiOutput& out = json ? (DcgmiOutput&) outJson : (DcgmiOutput&) outTree;
    stringstream ss;

    // Get topology
    result = dcgmGetDeviceTopology(mLwcmHandle, requestedGPUId, &gpuTopo);
    if (result == DCGM_ST_NOT_SUPPORTED)
    {
        std::cout << "Getting topology is not supported for GPU " << requestedGPUId << std::endl;
        PRINT_INFO("%u", "Getting topology is not supported for GPU: %u", requestedGPUId);
        return result;
    }
    else if (result != DCGM_ST_OK)
    {
        std::cout << "Error: unable to get topology for GPU " << requestedGPUId << ". Return: " << errorString(result)
                  << "." << std::endl;
        PRINT_ERROR("%u %d", "Error gettting topology for gpu: %u. Return: %d", requestedGPUId, result);
        return result;
    }

    // Header
    ss << "GPU ID: " << requestedGPUId;
    out.addHeader(HEADER_NAME);
    out.addHeader(ss.str());

    // Affinity

    std::string strHold = HelperGetAffinity(gpuTopo.cpuAffinityMask);
    unsigned int p = 0;
    unsigned int start = 0;

    if (strHold.length() > MAX_SIZE_OF_AFFINITY_STRING){
        while (start < strHold.length()){
            p += MAX_SIZE_OF_AFFINITY_STRING;
            if (p >= strHold.length()) p = strHold.length() - 1;

            else { // Put pointer to last available digit
                while (isdigit(strHold.at(p))){
                    if (p + 1 < strHold.length() && !isdigit(strHold.at(p + 1))) break; //check if landed on end of a digit
                    p--;
                }
                while (!isdigit(strHold.at(p))){
                    p--;
                }
            }
            // p is now the index of a the last digit of a CPU

            // Comma case
            if (p + 1 < strHold.length() && strHold.at(p + 1) == ','){
                ss.str(strHold.substr(start, p - start + 2));
            } else {
            // Hyphen case
                ss.str(strHold.substr(start, p - start + 1));
            }

            // Need to only print CPU Core affinity in first line
            if (start == 0){
                out["CPU Core Affinity"] = ss.str();
            } else {
                out["CPU Core Affinity"].addOverflow(ss.str());
            }

            start = p + 2; // move ahead two characters
        }

    } else {
        out["CPU Core Affinity"] = strHold;
    }

    for (unsigned int i = 0; i < gpuTopo.numGpus; i++){
        ss.str(""); // clear
        ss << "To GPU "  << gpuTopo.gpuPaths[i].gpuId;
        out[ss.str()] = HelperGetPciPath(gpuTopo.gpuPaths[i].path);
        if (gpuTopo.gpuPaths[i].localLwLinkIds != 0){
            out[ss.str()].addOverflow(HelperGetLwLinkPath(gpuTopo.gpuPaths[i].path, gpuTopo.gpuPaths[i].localLwLinkIds));
        }
    }

    std::cout << out.str();

    return DCGM_ST_OK;
}


/********************************************************************************/
dcgmReturn_t Topo::DisplayGroupTopology(dcgmHandle_t mLwcmHandle, dcgmGpuGrp_t requestedGroupId, bool json)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmGroupTopology_t groupTopo;
    groupTopo.version = dcgmGroupTopology_version;
    DcgmiOutputTree outTree(20, 60);
    DcgmiOutputJson outJson;
    DcgmiOutput& out = json ? (DcgmiOutput&) outJson : (DcgmiOutput&) outTree;
    stringstream ss;
    dcgmGroupInfo_t stLwcmGroupInfo;

    // Get group name
    stLwcmGroupInfo.version = dcgmGroupInfo_version;
    result = dcgmGroupGetInfo(mLwcmHandle, requestedGroupId, &stLwcmGroupInfo);
    if (DCGM_ST_OK != result) {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED)? "The Group is not found" : errorString(result);
        cout << "Error: Unable to get group information. Return: "<< error << endl;
        PRINT_ERROR("%u,%d","Error: GroupGetInfo for GroupId: %u. Return: %d", (unsigned int)(uintptr_t)requestedGroupId, result);
        return DCGM_ST_GENERIC_ERROR;
    }

    // Get topology
    result = dcgmGetGroupTopology(mLwcmHandle, requestedGroupId, &groupTopo);
    if (result == DCGM_ST_NOT_SUPPORTED)
    {
        std::cout << "Getting topology is not supported for group " << requestedGroupId << std::endl;
        PRINT_INFO("%u",
                   "Getting topology is not supported for this configuration of group %u",
                   (unsigned int)(uintptr_t)requestedGroupId);
        return result;
    }
    else if (result != DCGM_ST_OK)
    {
        std::cout << "Error: unable to get topology for Group. Return: " << errorString(result) << "." << std::endl;
        PRINT_ERROR("%u %d",
                    "Error gettting topology for group: %u. Return: %d",
                    (unsigned int)(uintptr_t)requestedGroupId,
                    result);
        return result;
    }

    // Header
    out.addHeader(HEADER_NAME);
    ss << stLwcmGroupInfo.groupName;
    out.addHeader(ss.str());

    // Affinity

    std::string strHold = HelperGetAffinity(groupTopo.groupCpuAffinityMask);
    unsigned int p = 0;
    unsigned int start = 0;

    if (strHold.length() > MAX_SIZE_OF_AFFINITY_STRING){
        while (start < strHold.length()){
            p += MAX_SIZE_OF_AFFINITY_STRING;
            if (p >= strHold.length()) p = strHold.length() - 1;

            else { // Put pointer to last available digit
                while (isdigit(strHold.at(p))){
                    if (p + 1 < strHold.length() && !isdigit(strHold.at(p + 1))) break; //check if landed on end of a digit
                    p--;
                }
                while (!isdigit(strHold.at(p))){
                    p--;
                }
            }
            // p is now the index of a the last digit of a CPU

            // Comma case
            if (p + 1 < strHold.length() && strHold.at(p + 1) == ','){
                ss.str(strHold.substr(start, p - start + 2));
            } else {
            // Hyphen case
                ss.str(strHold.substr(start, p - start + 1));
            }

            // Need to only print CPU Core affinity in first line
            if (start == 0){
                out["CPU Core Affinity"] = ss.str();
            } else {
                out["CPU Core Affinity"].addOverflow(ss.str());
            }

            start = p + 2; // move ahead two characters
        }

    } else {
        out["CPU Core Affinity"] = strHold;
    }

    // Numa optimal

    out["NUMA Optimal"] = groupTopo.numaOptimalFlag? "True" : "False";

    // Worst path

    out["Worst Path"] = HelperGetPciPath(groupTopo.slowestPath);

    std::cout << out.str();

    return DCGM_ST_OK;
}

// **************************************************************************************************
std::string Topo::HelperGetAffinity(unsigned long *cpuAffinity){

    stringstream ss;

    bool lastBitOn = false;
    int leftmostBitInOnSequence = -1;

    for (int i = 0; i < DCGM_AFFINITY_BITMASK_ARRAY_SIZE - 1; i++){

        for (int index = 0; index < 32; index++){

            if (cpuAffinity[i] & (1 << index) ){

                if (lastBitOn){ // On Sequence continues
                    continue;
                } else {  // On Sequence begins
                    leftmostBitInOnSequence = index + (i * 32);
                    lastBitOn = true;
                }

            } else {
                if (lastBitOn){  // On Sequence ended
                    // We have a valid sequence. Append it to the string stream.

                    if (leftmostBitInOnSequence == (index + (i * 32) - 1)){ // Sequence was one bit long
                        ss << leftmostBitInOnSequence << ", ";
                    } else if (leftmostBitInOnSequence == (index + (i * 32) - 2)){ // Sequence was two bits long
                        ss << leftmostBitInOnSequence << ", " << (index + (i * 32) - 1) << ", ";
                    } else { // Sequence more than two bits long
                    ss << leftmostBitInOnSequence << " - " << (index + (i * 32) - 1) << ", ";
                    }

                    lastBitOn = false;

                } else {  // Off Sequence continues
                    lastBitOn = false; // Not needed (just for clarity)
                    continue;
                }
            }
        }
    }

    // Need to handle case where the very last bit is on
    if (lastBitOn){
        // We have a valid sequence. Append it to the string stream.
        ss << leftmostBitInOnSequence << " - " << 255 << ", ";
    }

    // Return string with last two characters (comma and whitespace) removed.

    return ss.str().substr(0, ss.str().length() - 2);
}



// **************************************************************************************************
std::string Topo::HelperGetPciPath(dcgmGpuTopologyLevel_t &path){

    dcgmGpuTopologyLevel_t pciPath = DCGM_TOPOLOGY_PATH_PCI(path);
    switch (pciPath)
    {
        case DCGM_TOPOLOGY_BOARD:
            return "Connected via an on-board PCIe switch";
        case DCGM_TOPOLOGY_SINGLE:
            return "Connected via a single PCIe switch";
        case DCGM_TOPOLOGY_MULTIPLE:
            return "Connected via multiple PCIe switches";
        case DCGM_TOPOLOGY_HOSTBRIDGE:
            return "Connected via a PCIe host bridge";
        case DCGM_TOPOLOGY_CPU:
            return "Connected via a CPU-level link";
        case DCGM_TOPOLOGY_SYSTEM:
            return "Connected via a CPU-level link";
        default:
            return "Unknown";
    }
}

// **************************************************************************************************
std::string Topo::HelperGetLwLinkPath(dcgmGpuTopologyLevel_t &path, unsigned int linkMask){
    
    std::stringstream pathSS;
    unsigned int maxLinks = DCGM_LWLINK_MAX_LINKS_PER_GPU;

    pathSS << "Connected via ";

    dcgmGpuTopologyLevel_t lwLinkPath = DCGM_TOPOLOGY_PATH_LWLINK(path);
    switch (lwLinkPath)
    {   
        case DCGM_TOPOLOGY_LWLINK1:
            pathSS << "one LWLINK ";
            break;
        case DCGM_TOPOLOGY_LWLINK2:
            pathSS << "two LWLINKs ";
            break;
        case DCGM_TOPOLOGY_LWLINK3:
            pathSS << "three LWLINKs ";
            break;
        case DCGM_TOPOLOGY_LWLINK4:
            pathSS << "four LWLINKs ";
            break;
        case DCGM_TOPOLOGY_LWLINK5:
            pathSS << "five LWLINKs ";
            break;
        case DCGM_TOPOLOGY_LWLINK6:
            pathSS << "six LWLINKs ";
            break;
        default:
            return "Unknown";
    }   

    if (lwLinkPath == DCGM_TOPOLOGY_LWLINK1)
        pathSS << "(Link: ";
    else
        pathSS << "(Links: ";

    bool startedLinkList = false;
    for (unsigned int i = 0; i < maxLinks; i++)
    {   
        unsigned int mask = 1 << i;

        if (mask & linkMask)
        {   
            if (startedLinkList)
            {   
                pathSS <<  ", ";
            }   
            pathSS << i;
            startedLinkList = true;
        }   
    }   
    
    pathSS << ")";
    return pathSS.str();
}

/*****************************************************************************
 *****************************************************************************
 * Get GPU Topology
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GetGPUTopo::GetGPUTopo(std::string hostname, unsigned int gpu, bool json) {
    mHostName = hostname;
    mGpuId = gpu;
    mJson = json;
}

/*****************************************************************************/
GetGPUTopo::~GetGPUTopo() {
}

/*****************************************************************************/
int GetGPUTopo::Execute() {
    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return topoObj.DisplayGPUTopology(mLwcmHandle, mGpuId, mJson);
}



/*****************************************************************************
 *****************************************************************************
 * Get Group Topology
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GetGroupTopo::GetGroupTopo(std::string hostname, unsigned int groupId, bool json) {
    mHostName = hostname;
    mGroupId = (dcgmGpuGrp_t)(long long) groupId;
    mJson = json;
}

/*****************************************************************************/
GetGroupTopo::~GetGroupTopo() {
}

/*****************************************************************************/
int GetGroupTopo::Execute() {
    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return topoObj.DisplayGroupTopology(mLwcmHandle, mGroupId, mJson);
}
