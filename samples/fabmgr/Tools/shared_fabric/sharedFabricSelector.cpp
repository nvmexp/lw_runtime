#include <stdio.h>
#include <stdlib.h>
#include "commandline/commandline.h"
#include "sharedFabricSelectorParser.h"
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include "errno.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

#include "fabricmanagerHA.pb.h"

#include "string.h"
#include "json/json.h"

struct all_args sharedFabricSelectorArgs[] = {

        {
                SHARED_FABRIC_CMD_HELP,
                "-h",
                "--help",
                "\t\tDisplays help information",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                SHARED_FABRIC_CMD_LIST_PARTITION,
                "-l",
                "--list",
                "\t\tQuery all the supported fabric partitions",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                SHARED_FABRIC_CMD_ACTIVATE_PARTITION,
                "-a",
                "--activate",
                "\t\tActivate a supported fabric partition",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                SHARED_FABRIC_CMD_DEACTIVATE_PARTITION,
                "-d",
                "--deactivate",
                "\tDeactivate a previously activated fabric partition",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                SHARED_FABRIC_CMD_SET_ACTIVATED_PARTITION_LIST,
                "",
                "--set-activated-list",
                "\tSet a list of lwrrently activated fabric partitions",
                "\n\t",
                CMDLINE_OPTION_VALUE_OPTIONAL
        },
        {
                SHARED_FABRIC_CMD_GET_LWLINK_FAILED_DEVICES,
                "",
                "--get-lwlink-failed-devices",
                "\t\tQuery all LWLink failed devices",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                SHARED_FABRIC_CMD_LIST_UNSUPPORTED_PARTITION,
                "",
                "--list-unsupported-partitions",
                "\t\tQuery all the unsupported fabric partitions",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                SHARED_FABRIC_CMD_HOSTNAME,
                "",
                "--hostname",
                "\t\thostname or IP address of Fabric Manager. Default: 127.0.0.1.",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                SHARED_FABRIC_CMD_UNIX_DOMAIN_SOCKET,
                "",
                "--unix-domain-socket",
                "\t\tUnix domain socket path for Fabric Manager connection.",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                SHARED_FABRIC_CMD_COLWERT_BIN_TO_TXT,
                "-b",
                "--binary-statefile",
                "\t\tColwert a binary state file to text file.",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                SHARED_FABRIC_CMD_COLWERT_TXT_TO_BIN,
                "-t",
                "--text-statefile",
                "\t\tColwert a text state file to binary file.",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
};

/*****************************************************************************
 Method to Display Usage Info
 *****************************************************************************/
static void sharedFabricSelectorUsage(void * pCmdLine) {
    printf("\n\n");
    printf("    Usage: sharedFabricSelector [options]\n");
    printf("\n");
    printf("    Options include:\n");
    cmdline_printOptionsSummary(pCmdLine, 0);
    printf("\n");
    printf("    Please email lwdatools@lwpu.com with any questions,\n"
            "    bug reports, etc.");
    printf("\n\n");
    exit(0);
}

/*****************************************************************************
 Method to Display Help Message
 *****************************************************************************/
static void sharedFabricSelectorDisplayHelpMessage(void * pCmdLine) {
    printf("\n");
    printf("    HGX-2 Fabric Partition Selector for multitenancy\n");

    sharedFabricSelectorUsage(pCmdLine);
}

fmReturn_t listPartitions(SharedFabricCmdParser_t *pCmdParser, fmHandle_t fmHandle)
{
    fmReturn_t fmReturn;
    fmFabricPartitionList_t partitionList;

    memset(&partitionList, 0, sizeof(fmFabricPartitionList_t));
    partitionList.version = fmFabricPartitionList_version;

    fmReturn = fmGetSupportedFabricPartitions(fmHandle, &partitionList);
    if (fmReturn != FM_ST_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to list partition. Return: %d\n", fmReturn);
        return fmReturn;
    }

    Json::Value jsonPartitionList(Json::objectValue);
    Json::Value jsonPartitionInfoList(Json::arrayValue);
    Json::Value jsonPartitionInfo(Json::objectValue);
    Json::Value jsonPartitionGpuInfoList(Json::arrayValue);
    Json::Value jsonPartitionGpuInfo(Json::objectValue);

    jsonPartitionList["version"] = partitionList.version;
    jsonPartitionList["numPartitions"] = partitionList.numPartitions;
    jsonPartitionList["maxNumPartitions"] = partitionList.maxNumPartitions;
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
            jsonPartitionGpuInfo["maxNumLwLinks"] = gpuInfo->maxNumLwLinks;
            jsonPartitionGpuInfo["lwlinkLineRateMBps"] = gpuInfo->lwlinkLineRateMBps;
            jsonPartitionGpuInfoList.append(jsonPartitionGpuInfo);
        }

        jsonPartitionInfo["gpuInfo"] = jsonPartitionGpuInfoList;
        jsonPartitionInfoList.append(jsonPartitionInfo);
    }

    jsonPartitionList["partitionInfo"] = jsonPartitionInfoList;

    Json::StyledWriter styledWriter;
    std::string sStyled = styledWriter.write(jsonPartitionList);
    fprintf(stdout, "%s", sStyled.c_str());

    return fmReturn;
}

fmReturn_t listUnsupportedPartitions(SharedFabricCmdParser_t *pCmdParser, fmHandle_t fmHandle)
{
    fmReturn_t fmReturn;
    fmUnsupportedFabricPartitionList_t partitionList;

    memset(&partitionList, 0, sizeof(fmUnsupportedFabricPartitionList_t));
    partitionList.version = fmUnsupportedFabricPartitionList_version;

    fmReturn = fmGetUnsupportedFabricPartitions(fmHandle, &partitionList);
    if (fmReturn != FM_ST_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to list unsupported partition. Return: %d\n", fmReturn);
        return fmReturn;
    }

    Json::Value jsonPartitionList(Json::objectValue);
    Json::Value jsonPartitionInfoList(Json::arrayValue);
    Json::Value jsonPartitionInfo(Json::objectValue);
    Json::Value jsonPartitionGpuList(Json::arrayValue);

    jsonPartitionList["version"] = partitionList.version;
    jsonPartitionList["numPartitions"] = partitionList.numPartitions;
    jsonPartitionInfoList.clear();

    for (unsigned int partIdx = 0; partIdx < partitionList.numPartitions; ++partIdx)
    {
        fmUnsupportedFabricPartitionInfo_t *partInfo = &partitionList.partitionInfo[partIdx];
        jsonPartitionGpuList.clear();

        jsonPartitionInfo["partitionId"] = partInfo->partitionId;
        jsonPartitionInfo["numGpus"] = partInfo->numGpus;

        for (unsigned int gpuIdx = 0; gpuIdx < partInfo->numGpus; ++gpuIdx)
        {
            jsonPartitionGpuList.append(partInfo->gpuPhysicalIds[gpuIdx]);
        }

        jsonPartitionInfo["gpuPhysicalIds"] = jsonPartitionGpuList;
        jsonPartitionInfoList.append(jsonPartitionInfo);
    }

    jsonPartitionList["partitionInfo"] = jsonPartitionInfoList;

    Json::StyledWriter styledWriter;
    std::string sStyled = styledWriter.write(jsonPartitionList);
    fprintf(stdout, "%s", sStyled.c_str());

    return fmReturn;
}

fmReturn_t activatePartition(SharedFabricCmdParser_t *pCmdParser, fmHandle_t fmHandle)
{
    fmReturn_t fmReturn;

    fmReturn = fmActivateFabricPartition(fmHandle, pCmdParser->mPartitionId);
    if (fmReturn != FM_ST_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to activate partition. Return: %d\n", fmReturn);
    }

    return fmReturn;
}

fmReturn_t deactivatePartition(SharedFabricCmdParser_t *pCmdParser, fmHandle_t fmHandle)
{
    fmReturn_t fmReturn;

    fmReturn = fmDeactivateFabricPartition(fmHandle, pCmdParser->mPartitionId);
    if (fmReturn != FM_ST_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to deactivate partition. Return: %d\n", fmReturn);
    }

    return fmReturn;
}

fmReturn_t setActivatedPartitionList(SharedFabricCmdParser_t *pCmdParser, fmHandle_t fmHandle)
{
    fmReturn_t fmReturn;
    fmActivatedFabricPartitionList_t partitionList;

    memset(&partitionList, 0, sizeof(fmActivatedFabricPartitionList_t));
    partitionList.version = fmActivatedFabricPartitionList_version;
    partitionList.numPartitions = pCmdParser->mNumPartitions;

    for (unsigned i = 0; i < pCmdParser->mNumPartitions; i++)
    {
        partitionList.partitionIds[i] = pCmdParser->mPartitionIds[i];
    }

    fmReturn = fmSetActivatedFabricPartitions(fmHandle, &partitionList);
    if (fmReturn != FM_ST_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to set activated partitions. Return: %d\n", fmReturn);
    }

    return fmReturn;
}

fmReturn_t getLwlinkFailedDevices(SharedFabricCmdParser_t *pCmdParser, fmHandle_t fmHandle)
{
    fmReturn_t fmReturn;
    fmLwlinkFailedDevices_t lwlinkFailedDevice;

    memset(&lwlinkFailedDevice, 0, sizeof(fmLwlinkFailedDevices_t));
    lwlinkFailedDevice.version = fmLwlinkFailedDevices_version;

    fmReturn = fmGetLwlinkFailedDevices(fmHandle, &lwlinkFailedDevice);
    if (fmReturn != FM_ST_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to LWLink failed devices. Return: %d\n", fmReturn);
        return fmReturn;
    }

    Json::Value jsonLwlinkFailedDevices(Json::objectValue);
    Json::Value jsonLwlinkFailedGpuInfo(Json::objectValue);
    Json::Value jsonLwlinkFailedSwitchInfo(Json::objectValue);
    Json::Value jsonLwlinkFailedGpuInfoList(Json::arrayValue);
    Json::Value jsonLwlinkFailedSwitchInfoList(Json::arrayValue);
    Json::Value jsonPortList(Json::arrayValue);

    jsonLwlinkFailedDevices["version"] = lwlinkFailedDevice.version;
    jsonLwlinkFailedDevices["numGpus"] = lwlinkFailedDevice.numGpus;
    jsonLwlinkFailedDevices["numSwitches"] = lwlinkFailedDevice.numSwitches;
    jsonLwlinkFailedGpuInfoList.clear();
    jsonLwlinkFailedSwitchInfoList.clear();

    uint32_t i, j;
    for (i = 0; i < lwlinkFailedDevice.numGpus; i++) {
        fmLwlinkFailedDeviceInfo_t &gpuInfo = lwlinkFailedDevice.gpuInfo[i];
        jsonLwlinkFailedGpuInfo["uuid"] = gpuInfo.uuid;
        jsonLwlinkFailedGpuInfo["pciBusId"] = gpuInfo.pciBusId;
        jsonLwlinkFailedGpuInfo["numPorts"] = gpuInfo.numPorts;

        jsonPortList.clear();
        for (j = 0; j < gpuInfo.numPorts; j++) {
            jsonPortList.append(gpuInfo.portNum[j]);
        }
        jsonLwlinkFailedGpuInfo["portNum"] = jsonPortList;
        jsonLwlinkFailedGpuInfoList.append(jsonLwlinkFailedGpuInfo);
    }

    jsonLwlinkFailedDevices["gpuInfo"] = jsonLwlinkFailedGpuInfoList;

    for (i = 0; i < lwlinkFailedDevice.numSwitches; i++) {
        fmLwlinkFailedDeviceInfo_t &switchInfo = lwlinkFailedDevice.switchInfo[i];
        jsonLwlinkFailedSwitchInfo["uuid"] = switchInfo.uuid;
        jsonLwlinkFailedSwitchInfo["pciBusId"] = switchInfo.pciBusId;
        jsonLwlinkFailedSwitchInfo["numPorts"] = switchInfo.numPorts;

        jsonPortList.clear();
        for (j = 0; j < switchInfo.numPorts; j++) {
            jsonPortList.append(switchInfo.portNum[j]);
        }
        jsonLwlinkFailedSwitchInfo["portNum"] = jsonPortList;
        jsonLwlinkFailedSwitchInfoList.append(jsonLwlinkFailedSwitchInfo);
    }
    jsonLwlinkFailedDevices["switchInfo"] = jsonLwlinkFailedSwitchInfoList;

    Json::StyledWriter styledWriter;
    std::string sStyled = styledWriter.write(jsonLwlinkFailedDevices);
    fprintf(stdout, "%s", sStyled.c_str());

    return fmReturn;
}

fmReturn_t colwertToBinStatefile(SharedFabricCmdParser_t *pCmdParser, fmHandle_t fmHandle)
{
    fabricmanagerHA::fmHaState haState;
    std::string  stateText;
    FILE        *outFile;

    outFile = fopen( pCmdParser->mOutFileName, "w+" );
    if ( outFile == NULL )
    {
        fprintf(stderr, "Failed to open output file %s, error is %s.\n",
                pCmdParser->mOutFileName, strerror(errno));
        return FM_ST_GENERIC_ERROR;
    }

    // Read the protobuf binary file.
    std::fstream input(pCmdParser->mInFileName, std::ios::in | std::ios::binary);
    if ( !input )
    {
        fprintf(stderr, "File %s is not Found. \n", pCmdParser->mInFileName);
        fclose( outFile );
        return FM_ST_GENERIC_ERROR;

    }
    else if ( !haState.ParseFromIstream(&input) )
    {
        fprintf(stderr, "Failed to parse file %s. \n", pCmdParser->mInFileName);
        fclose( outFile );
        return FM_ST_GENERIC_ERROR;
    }

    google::protobuf::TextFormat::PrintToString(haState, &stateText);
    fwrite( stateText.c_str(), 1, (int)stateText.length(), outFile);
    fclose( outFile );
    return FM_ST_SUCCESS;
}

fmReturn_t colwertToTxtStatefile(SharedFabricCmdParser_t *pCmdParser, fmHandle_t fmHandle)
{
    fabricmanagerHA::fmHaState haState;
    int       inFileFd;
    FILE     *outFile;

    inFileFd = open(pCmdParser->mInFileName, O_RDONLY);
    if ( inFileFd < 0 )
    {
        fprintf(stderr, "Failed to open input file %s, error is %s.\n",
                pCmdParser->mInFileName, strerror(errno));
        return FM_ST_GENERIC_ERROR;
    }

    outFile = fopen( pCmdParser->mOutFileName, "w+" );
    if ( outFile == NULL )
    {
        fprintf(stderr, "Failed to open output file %s, error is %s.\n",
                pCmdParser->mOutFileName, strerror(errno));
        close( inFileFd );
        return FM_ST_GENERIC_ERROR;
    }

    google::protobuf::io::FileInputStream fileInput(inFileFd);
    google::protobuf::TextFormat::Parse(&fileInput, &haState);

    // write the binary state file
    int   fileLength = haState.ByteSize();
    int   bytesWritten;
    char *bufToWrite = new char[fileLength];

    if ( bufToWrite == NULL )
    {
        close( inFileFd );
        fclose ( outFile );
        return FM_ST_GENERIC_ERROR;
    }

    haState.SerializeToArray( bufToWrite, fileLength );
    fwrite( bufToWrite, 1, fileLength, outFile );
    close( inFileFd );
    fclose ( outFile );
    delete[] bufToWrite;
    return FM_ST_SUCCESS;
}

int cleanup(SharedFabricCmdParser_t *pCmdParser, fmHandle_t fmHandle, int status)
{
    (void)sharedFabricCmdParserDestroy(pCmdParser);
    // need to do fmDisconnect only if fmConnect was successful
    if (status != -1)
        fmDisconnect(fmHandle);
    fmLibShutdown();

    return status;
}

int main(int argc, char **argv)
{
    fmReturn_t ret;
    SharedFabricCmdParser_t *pCmdParser;
    fmHandle_t fmHandle;

    pCmdParser = sharedFabricCmdParserInit(argc, argv, sharedFabricSelectorArgs,
                                           SHARED_FABRIC_CMD_COUNT, sharedFabricSelectorUsage,
                                           sharedFabricSelectorDisplayHelpMessage);
    if (NULL == pCmdParser) {
        return FM_ST_BADPARAM;
    }

    ret = sharedFabricCmdProcessing(pCmdParser);
    if (FM_ST_SUCCESS != ret) {
        if (ret == FM_ST_BADPARAM){
            fprintf(stderr, "Unable to process command: bad command line parameter. \n");
        } else {
            fprintf(stderr, "Unable to process command: generic error. \n");
        }
        sharedFabricCmdParserDestroy(pCmdParser);
        return ret;
    }

    /* Initialize FM lib */
    ret = fmLibInit();
    if (FM_ST_SUCCESS != ret) {
        // assume that error message has already been printed
        fprintf(stderr, "Error: failed to initialize fabric manager interface library");
        cleanup(pCmdParser, fmHandle, -1);
        return ret;
    }

    fmConnectParams_t connectParams;
    connectParams.timeoutMs = 1000; // in milliseconds
    connectParams.version = fmConnectParams_version;

    memset(connectParams.addressInfo, 0, sizeof(connectParams.addressInfo));
    if (strnlen(pCmdParser->mUnixSockPath, MAX_PATH_LEN) > 0) {
        strncpy(connectParams.addressInfo, pCmdParser->mUnixSockPath, MAX_PATH_LEN);
        connectParams.addressIsUnixSocket = 1;

    } else {
        strncpy(connectParams.addressInfo, pCmdParser->mHostname, MAX_PATH_LEN);
        connectParams.addressIsUnixSocket = 0;
    }

    ret = fmConnect(&connectParams, &fmHandle);
    if (ret != FM_ST_SUCCESS){
        fprintf(stderr, "Error connecting to Fabric Manager. Return: %d \n", ret);
        cleanup(pCmdParser, fmHandle, -1);
        return ret;
    }

    try {
        if (pCmdParser->mListPartition) {
            ret = listPartitions(pCmdParser, fmHandle);
        }
        else if (pCmdParser->mActivatePartition) {
            ret = activatePartition(pCmdParser, fmHandle);
        }
        else if (pCmdParser->mDeactivatePartition) {
            ret = deactivatePartition(pCmdParser, fmHandle);
        }
        else if (pCmdParser->mSetActivatedPartitions) {
            ret = setActivatedPartitionList(pCmdParser, fmHandle);
        }
        else if (pCmdParser->mGetLwlinkFailedDevices) {
            ret = getLwlinkFailedDevices(pCmdParser, fmHandle);
        }
        else if (pCmdParser->mBinToTxt) {
            ret = colwertToBinStatefile(pCmdParser, fmHandle);
        }
        else if (pCmdParser->mTxtToBin) {
            ret = colwertToTxtStatefile(pCmdParser, fmHandle);
        }
        else if (pCmdParser->mListUnsupportedPartition) {
            ret = listUnsupportedPartitions(pCmdParser, fmHandle);
        }
        else {
            fprintf(stderr, "Error unknown command. \n");
            ret = FM_ST_BADPARAM;
        }
    } catch (const std::exception &e) {
        fprintf(stderr, "%s\n", e.what());
    }

    cleanup(pCmdParser, fmHandle, 0);
    return ret;
}
