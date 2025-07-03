
#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include <vector>
#include <map>
#include <string.h>
#include <chrono>

#include "json/json.h"

#include "lw_fm_agent.h"
#include "sharedFabricTestParser.h"
#include "platformModelDelta.h"

using namespace std; 
using namespace std::chrono; 

extern unsigned int fabricMode;

/*
This is the default partition assignment for Delta systems. In bare metal environment,
this information can be generated based on the partition list returned by FM.

Note: This may require changes based on your system GPU PCI BDF assignments.
*/ 
/*
static std::map<int, std::vector<std::string> > paritionIdToGpuBdf =
{
    { 0, {"3b:00.0", "45:00.0", "41:00.0", "36:00.0", "63:00.0", "5d:00.0", "67:00.0", "59:00.0", "b9:00.0", "c8:00.0", "c4:00.0", "be:00.0", "e2:00.0", "f0:00.0", "eb:00.0", "e6:00.0"} },
    { 1, {"3b:00.0", "45:00.0", "41:00.0", "36:00.0", "63:00.0", "5d:00.0", "67:00.0", "59:00.0"} },
    { 2, {"b9:00.0", "c8:00.0", "c4:00.0", "be:00.0", "e2:00.0", "f0:00.0", "eb:00.0", "e6:00.0"} },
    { 3, {"3b:00.0", "45:00.0", "41:00.0", "36:00.0", "b9:00.0", "c8:00.0", "c4:00.0", "be:00.0"} },
    { 4, {"63:00.0", "5d:00.0", "67:00.0", "59:00.0", "e2:00.0", "f0:00.0", "eb:00.0", "e6:00.0"} },
    { 5, {"3b:00.0", "45:00.0", "41:00.0", "36:00.0", "e2:00.0", "f0:00.0", "eb:00.0", "e6:00.0"} },
    { 6, {"63:00.0", "5d:00.0", "67:00.0", "59:00.0", "b9:00.0", "c8:00.0", "c4:00.0", "be:00.0"} },
    { 7, {"3b:00.0", "45:00.0", "41:00.0", "36:00.0"} },
    { 8, {"63:00.0", "5d:00.0", "67:00.0", "59:00.0"} },
    { 9, {"b9:00.0", "c8:00.0", "c4:00.0", "be:00.0"} },
    { 10, {"e2:00.0", "f0:00.0", "eb:00.0", "e6:00.0"} },
    { 11, {"3b:00.0", "45:00.0"} },
    { 12, {"41:00.0", "36:00.0"} },
    { 13, {"63:00.0", "5d:00.0"} },
    { 14, {"67:00.0", "59:00.0"} },
    { 15, {"b9:00.0", "c8:00.0"} },
    { 16, {"c4:00.0", "be:00.0"} },
    { 17, {"e2:00.0", "f0:00.0"} },
    { 18, {"eb:00.0", "e6:00.0"} },
    { 19, {"3b:00.0"} },
    { 20, {"45:00.0"} },
    { 21, {"41:00.0"} },
    { 22, {"36:00.0"} },
    { 23, {"63:00.0"} },
    { 24, {"5d:00.0"} },
    { 25, {"67:00.0"} },
    { 26, {"59:00.0"} },
    { 27, {"b9:00.0"} },
    { 28, {"c8:00.0"} },
    { 29, {"c4:00.0"} },
    { 30, {"be:00.0"} },
    { 31, {"e2:00.0"} },
    { 32, {"f0:00.0"} },
    { 33, {"eb:00.0"} },
    { 34, {"e6:00.0"} }
};
*/
    

//Partition information based on single base board luna-por-23 system

static std::map<int, std::vector<std::string> > paritionIdToGpuBdf =
{
    { 2, {"27:00.0", "ae:00.0", "a7:00.0", "2e:00.0", "07:00.0", "90:00.0", "87:00.0", "10:00.0"} },
    { 9, {"27:00.0", "ae:00.0", "a7:00.0", "2e:00.0"} },
    { 10,{"07:00.0", "90:00.0", "87:00.0", "10:00.0"} },
    { 15, {"27:00.0", "ae:00.0"} },
    { 16, {"a7:00.0", "2e:00.0"} },
    { 17, {"07:00.0", "90:00.0"} },
    { 18, {"87:00.0", "10:00.0"} },
    { 27, {"27:00.0"} },
    { 28, {"ae:00.0"} },
    { 29, {"a7:00.0"} },
    { 30, {"2e:00.0"} },
    { 31, {"07:00.0"} },
    { 32, {"90:00.0"} },
    { 33, {"87:00.0"} },
    { 34, {"10:00.0"} }
};

/*
// Partition information based on single base board luna-por-19 system

static std::map<int, std::vector<std::string> > paritionIdToGpuBdf =
{
    { 2, {"27:00.0", "ad:00.0", "a7:00.0", "2e:00.0", "07:00.0", "90:00.0", "87:00.0", "0f:00.0"} },
    { 9, {"27:00.0", "ad:00.0", "a7:00.0", "2e:00.0"} },
    { 10,{"07:00.0", "90:00.0", "87:00.0", "0f:00.0"} },
    { 15, {"27:00.0", "ad:00.0"} },
    { 16, {"a7:00.0", "2e:00.0"} },
    { 17, {"07:00.0", "90:00.0"} },
    { 18, {"87:00.0", "0f:00.0"} },
    { 27, {"27:00.0"} },
    { 28, {"ad:00.0"} },
    { 29, {"a7:00.0"} },
    { 30, {"2e:00.0"} },
    { 31, {"07:00.0"} },
    { 32, {"90:00.0"} },
    { 33, {"87:00.0"} },
    { 34, {"0f:00.0"} }
};

*/

void
platformModelDelta::unbindAllPartitionGpus(void)
{
	// command format "echo 0000:3b:00.0 > /sys/bus/pci/drivers/lwpu/unbind > /dev/null"

    // take first partition which has the list of all the GPUs (largest GPU partition)
	std::map<int, std::vector<std::string> >::iterator firstPart = paritionIdToGpuBdf.begin();
    
	std::vector<std::string>::iterator it;
	std::vector<std::string> allGpuList = firstPart->second;
	for (it = allGpuList.begin(); it != allGpuList.end(); it++) {
		std::ostringstream cmd;
		cmd << "echo " << "0000:" << (*it) << " > /sys/bus/pci/drivers/lwpu/unbind";
		system(cmd.str().c_str());
	}
}

void
platformModelDelta::resetAndBindPartitionGpus(int partitionId)
{
	system("setpci -v -d 10de:20b0 0x71c.l=0x10 >/dev/null 2>&1");
	system("setpci -v -d 10de:1ae8 0x71c.l=0xf00011 >/dev/null 2>&1");

	std::vector<std::string>::iterator it;
	std::vector<std::string> gpuList;

	// command format echo 1 > /sys/bus/pci/devices/0000:3b:00.0/reset

	gpuList = paritionIdToGpuBdf[partitionId];
	for (it = gpuList.begin(); it != gpuList.end(); it++) {
		std::ostringstream cmd;
		cmd << "echo 1 > /sys/bus/pci/devices/0000:" << (*it) << "/reset";
		system(cmd.str().c_str());
	}

	// command format echo 0000:3b:00.0 > /sys/bus/pci/drivers/lwpu/bind"
	gpuList = paritionIdToGpuBdf[partitionId];
	for (it = gpuList.begin(); it != gpuList.end(); it++) {
		std::ostringstream cmd;
		cmd << "echo " << "0000:" << (*it) << " > /sys/bus/pci/drivers/lwpu/bind";
		system(cmd.str().c_str());
	}
}

void
platformModelDelta::unBindPartitionGpus(int partitionId)
{
	std::vector<std::string>::iterator it;
	std::vector<std::string> gpuList;
	// command format echo 0000:3b:00.0 > /sys/bus/pci/drivers/lwpu/unbind

	gpuList = paritionIdToGpuBdf[partitionId];
	for (it = gpuList.begin(); it != gpuList.end(); it++) {
		std::ostringstream cmd;
		cmd << "echo " << "0000:" << (*it) << " > /sys/bus/pci/drivers/lwpu/unbind";
		system(cmd.str().c_str());
	}
}

void
platformModelDelta::resetPartitionGpus(int partitionId)
{
	system("setpci -v -d 10de:20b0 0x71c.l=0x10 >/dev/null 2>&1");
	system("setpci -v -d 10de:1ae8 0x71c.l=0xf00011 >/dev/null 2>&1");

	std::vector<std::string>::iterator it;
	std::vector<std::string> gpuList;

	// command format echo 1 > /sys/bus/pci/devices/0000:3b:00.0/reset

	gpuList = paritionIdToGpuBdf[partitionId];
	for (it = gpuList.begin(); it != gpuList.end(); it++) {
		std::ostringstream cmd;
		cmd << "echo 1 > /sys/bus/pci/devices/0000:" << (*it) << "/reset";
		system(cmd.str().c_str());
	}
}

void
platformModelDelta::doSharedSwitchPartitionActivation(fmHandle_t pFmHandle, int partitionId)
{
    fmReturn_t fmReturn;

    fprintf(stderr, "Activating Shared LWSwitch Partition id %d\n", partitionId);
    
    // assuming FM is started in shared mode, before starting our test
    // unbind all the GPU from LWPU
    unbindAllPartitionGpus();

    // first reset and bind the GPUs belonging to the GPU
    resetAndBindPartitionGpus(partitionId);
   
    fmReturn = fmActivateFabricPartition(pFmHandle, partitionId);
        
    if (fmReturn != FM_ST_SUCCESS) {
        fprintf(stderr, "Error: Failed to activate Shared LWSwitch Partition id %d. Return: %d\n", partitionId, fmReturn);
        return;
    }
    
    unBindPartitionGpus(partitionId);

    fprintf(stderr, "Successfully activated Shared LWSwitch partition id %d\n", partitionId);
}

void
platformModelDelta::doVgpuPartitionActivation(fmHandle_t pFmHandle, int partitionId)
{
    fmPciDevice_t vfList[FM_MAX_NUM_GPUS];
    std::vector<std::string>::iterator it;
    std::vector<std::string> gpuList;
    std::string delimiter = ":";
    gpuList = paritionIdToGpuBdf[partitionId];
    fmReturn_t fmReturn;
    int i = 0;

    fprintf(stderr, "Activating vGPU Partition id %d\n", partitionId);
    
    for (i=0, it = gpuList.begin(); it != gpuList.end(); it++, i++) {
        FILE *fp;
        std::ostringstream cmd;
        std::string s = *it;
        char str[32];

        // VF domain number: mostly zero
        vfList[i].domain = 0;

        // VF bus number: Computed based on PF's bus number
        s = s.substr(0, s.find(delimiter));
        vfList[i].bus = std::stoul(s,nullptr,16);

        // VF device number: always zero
        vfList[i].device = 0;

        // VF function number: Computed based on SRIOV First VF offfset
        cmd << "cat /sys/bus/pci/devices/0000:" << (*it) << "/sriov_offset";
        fp = popen(cmd.str().c_str(), "r");
        if (!fp) {
            fprintf(stderr, "Error: popen() failed while activating vGPU partition id %d\n", partitionId);
            return;
        }

        fgets(str, sizeof(str), fp);
        vfList[i].function = std::stoul(str,nullptr,16);
        pclose(fp);

        fprintf(stderr, "Activating vGPU Partition id %d, VF%d 0x%x.%x.%x.%x\n",
                partitionId, i, vfList[i].domain, vfList[i].bus, vfList[i].device, vfList[i].function);
    }

    fprintf(stderr, "Activating vGPU Partition id %d, #VFs %d\n", partitionId, i);
    fmReturn = fmActivateFabricPartitionWithVFs(pFmHandle, partitionId, vfList, i);

    if (fmReturn != FM_ST_SUCCESS) {
        fprintf(stderr, "Error: Failed to activate vGPU Partition id %d. Return: %d\n", partitionId, fmReturn);
        return;
    }
    
    fprintf(stderr, "Successfully activated vGPU partition id %d\n", partitionId);
}

void
platformModelDelta::doPartitionActivationStreeTest(fmHandle_t pFmHandle)
{
    fmReturn_t fmReturn;
    int testCount;

    for (testCount = 0; testCount < MAX_PARTITION_ACTIVATION_COUNT; testCount++) {

        fprintf(stderr, "Partition Activation Stress Test Iteration %d \n", testCount);
        for (int partitionId = 0; partitionId < MAX_DELTA_PARTITIONS; partitionId++) {
            std::vector<std::string> gpuList = paritionIdToGpuBdf[partitionId];
            if (gpuList.empty()) {
                continue;
            }

            // Activate GPU Partition
            doPartitionActivation(pFmHandle, partitionId);

            // Deactivate GPU Partition
            doPartitionDeactivation(pFmHandle, partitionId);
        }
    }
}

void
platformModelDelta::doPartitionList(fmHandle_t pFmHandle)
{
    fmReturn_t fmReturn;
    fmFabricPartitionList_t partitionList;
    
    memset(&partitionList, 0, sizeof(fmFabricPartitionList_t));
    partitionList.version = fmFabricPartitionList_version;
    
    fmReturn = fmGetSupportedFabricPartitions(pFmHandle, &partitionList);
    if (fmReturn != FM_ST_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to list partition. Return: %d\n", fmReturn);
        return;
    }
    
    Json::Value jsonPartitionList(Json::objectValue);
    Json::Value jsonPartitionInfoList(Json::arrayValue);
    Json::Value jsonPartitionInfo(Json::objectValue);
    Json::Value jsonPartitionGpuInfoList(Json::arrayValue);
    Json::Value jsonPartitionGpuInfo(Json::objectValue);
    
    jsonPartitionList["version"] = partitionList.version;
    jsonPartitionList["numPartitions"] = partitionList.numPartitions;
    jsonPartitionInfoList.clear();
    
    for (unsigned int partIdx = 0; partIdx < partitionList.numPartitions; ++partIdx)
    {
        fmFabricPartitionInfo_t *partInfo = &partitionList.partitionInfo[partIdx];
        jsonPartitionGpuInfoList.clear();

        jsonPartitionInfo["partitionId"] = partInfo->partitionId;
        jsonPartitionInfo["isActive"] = partInfo->isActive;
        jsonPartitionInfo["numGpus"] = partInfo->numGpus;

        for (unsigned int gpuIdx = 0; gpuIdx < partInfo->numGpus; ++gpuIdx)
        {
            fmFabricPartitionGpuInfo_t *gpuInfo = &partInfo->gpuInfo[gpuIdx];
            jsonPartitionGpuInfo["physicalId"] = gpuInfo->physicalId;
            jsonPartitionGpuInfo["uuid"] = gpuInfo->uuid;
            jsonPartitionGpuInfo["pciBusId"] = gpuInfo->pciBusId;
            jsonPartitionGpuInfo["numLwLinksAvailable"] = gpuInfo->numLwLinksAvailable;
            jsonPartitionGpuInfo["lwlinkLineRateMBps"] = gpuInfo->lwlinkLineRateMBps;
            jsonPartitionGpuInfoList.append(jsonPartitionGpuInfo);
        }

        jsonPartitionInfo["gpuInfo"] = jsonPartitionGpuInfoList;
        jsonPartitionInfoList.append(jsonPartitionInfo);
    }

    jsonPartitionList["partitionInfo"] = jsonPartitionInfoList;

    Json::StyledWriter styledWriter;
    std::string sStyled = styledWriter.write(jsonPartitionList);

    FILE * fp;
    fp = fopen (PARTITION_LIST_DUMP_FILE_NAME, "w");
    fprintf(fp, "%s", sStyled.c_str());
    fclose(fp);

    fprintf(stderr, "partition information is dumped into file %s at current directory\n", PARTITION_LIST_DUMP_FILE_NAME);
}

void
platformModelDelta::doPartitionActivation(fmHandle_t pFmHandle, int partitionId)
{
    fprintf(stderr, "Activating Partition id %d\n", partitionId);
    
    // Shared LWSwitch mode
    if (fabricMode == 0x1) {
        doSharedSwitchPartitionActivation(pFmHandle, partitionId);
    } else {
        // vGPU mode
        doVgpuPartitionActivation(pFmHandle, partitionId);
    }
}

void
platformModelDelta::doPartitionDeactivation(fmHandle_t pFmHandle, int partitionId)
{
    fmReturn_t fmReturn;

    fmReturn = fmDeactivateFabricPartition(pFmHandle, partitionId);
    if (fmReturn != FM_ST_SUCCESS) {
        fprintf(stderr, "Error: Failed to deactivate partition id %d. Return: %d\n", partitionId, fmReturn);
        return;
    }

    // Shared LWSwitch mode
    if (fabricMode == 0x1) {
        resetPartitionGpus(partitionId);
    }

    fprintf(stderr, "partition id %d deactivated successfully\n", partitionId);
}

void
platformModelDelta::setActivatedPartitionList(fmHandle_t pFmHandle, SharedFabricCmdParser_t *pCmdParser)
{
    fmReturn_t fmReturn;
    fmActivatedFabricPartitionList_t partitionList;

    memset(&partitionList, 0, sizeof(fmActivatedFabricPartitionList_t));
    partitionList.version = fmActivatedFabricPartitionList_version;
    partitionList.numPartitions = pCmdParser->mNumPartitions;

    for (unsigned i = 0; i < pCmdParser->mNumPartitions; i++)
    {
        fprintf(stderr, "partition id %d \n", pCmdParser->mPartitionIds[i]);
        partitionList.partitionIds[i] = pCmdParser->mPartitionIds[i];
    }

    fmReturn = fmSetActivatedFabricPartitions(pFmHandle, &partitionList);
    if (fmReturn != FM_ST_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to set activated partitions. Return: %d\n", fmReturn);
    }

    return;
}
