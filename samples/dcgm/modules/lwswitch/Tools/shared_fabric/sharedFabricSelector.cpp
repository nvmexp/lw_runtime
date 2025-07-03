#include <stdio.h>
#include <stdlib.h>
#include "commandline/commandline.h"
#include "sharedFabricSelectorParser.h"
#include "lwos.h"
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
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include "dcgm_module_fm_internal.h"
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
                SHARED_FABRIC_CMD_HOSTNAME,
                "",
                "--hostname",
                "\t\thostname or IP address of Fabric Manager. Default: 127.0.0.1.",
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

dcgmReturn_t listPartition(SharedFabricCmdParser_t *pCmdParser, dcgmHandle_t dcgmHandle)
{
    dcgmReturn_t dcgmReturn;
    dcgmFabricPartitionList_t partitionList;
    const etblDCGMModuleFMInternal *pEtbl = NULL;

    memset(&partitionList, 0, sizeof(dcgmFabricPartitionList_t));
    partitionList.version = dcgmFabricPartitionList_version;
    dcgmReturn = dcgmInternalGetExportTable((const void**)&pEtbl, &ETID_DCGMModuleFMInternal);
    if (DCGM_ST_OK  != dcgmReturn)
    {
        fprintf(stderr, "Error: Can't get the Fabric Manager export table. Return: %d\n", dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = DCGM_CALL_ETBL(pEtbl, fpdcgmGetSupportedFabricPartitions,
                                (dcgmHandle, &partitionList));
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "Error: Failed to list partition. Return: %d\n", dcgmReturn);
        return dcgmReturn;
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
        dcgmFabricPartitionInfo_t *partInfo = &partitionList.partitionInfo[partIdx];
        jsonPartitionGpuInfoList.clear();

        jsonPartitionInfo["partitionId"] = partInfo->partitionId;
        jsonPartitionInfo["isActive"] = partInfo->isActive;
        jsonPartitionInfo["numGpus"] = partInfo->numGpus;

        for (unsigned int gpuIdx = 0; gpuIdx < partInfo->numGpus; ++gpuIdx)
        {
            dcgmFabricPartitionGpuInfo_t *gpuInfo = &partInfo->gpuInfo[gpuIdx];
            jsonPartitionGpuInfo["physicalId"] = gpuInfo->physicalId;
            jsonPartitionGpuInfo["uuid"] = gpuInfo->uuid;
            jsonPartitionGpuInfo["pciBusId"] = gpuInfo->pciBusId;
            jsonPartitionGpuInfoList.append(jsonPartitionGpuInfo);
        }

        jsonPartitionInfo["gpuInfo"] = jsonPartitionGpuInfoList;
        jsonPartitionInfoList.append(jsonPartitionInfo);
    }

    jsonPartitionList["partitionInfo"] = jsonPartitionInfoList;

    Json::StyledWriter styledWriter;
    std::string sStyled = styledWriter.write(jsonPartitionList);
    fprintf(stdout, "%s", sStyled.c_str());

    return dcgmReturn;
}

dcgmReturn_t activatePartition(SharedFabricCmdParser_t *pCmdParser, dcgmHandle_t dcgmHandle)
{
    dcgmReturn_t dcgmReturn;
    const etblDCGMModuleFMInternal *pEtbl = NULL;

    dcgmReturn = dcgmInternalGetExportTable((const void**)&pEtbl, &ETID_DCGMModuleFMInternal);
    if (DCGM_ST_OK  != dcgmReturn) {
        fprintf(stderr, "Error: Can't get the Fabric Manager export table. Return: %d\n", dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = DCGM_CALL_ETBL(pEtbl, fpdcgmActivateFabricPartition,
                                (dcgmHandle, pCmdParser->mPartitionId));
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "Error: Failed to activate partition. Return: %d\n", dcgmReturn);
    }

    return dcgmReturn;
}

dcgmReturn_t deactivatePartition(SharedFabricCmdParser_t *pCmdParser, dcgmHandle_t dcgmHandle)
{
    dcgmReturn_t dcgmReturn;
    const etblDCGMModuleFMInternal *pEtbl = NULL;

    dcgmReturn = dcgmInternalGetExportTable((const void**)&pEtbl, &ETID_DCGMModuleFMInternal);
    if (DCGM_ST_OK  != dcgmReturn) {
        fprintf(stderr, "Error: Can't get the Fabric Manager export table. Return: %d\n", dcgmReturn);

        return dcgmReturn;
    }

    dcgmReturn = DCGM_CALL_ETBL(pEtbl, fpdcgmDeactivateFabricPartition,
                                (dcgmHandle, pCmdParser->mPartitionId));
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "Error: Failed to deactivate partition. Return: %d\n", dcgmReturn);
    }

    return dcgmReturn;
}

dcgmReturn_t setActivatedPartitionList(SharedFabricCmdParser_t *pCmdParser, dcgmHandle_t dcgmHandle)
{
    dcgmReturn_t dcgmReturn;
    const etblDCGMModuleFMInternal *pEtbl = NULL;
    dcgmActivatedFabricPartitionList_t partitionList;

    memset(&partitionList, 0, sizeof(dcgmActivatedFabricPartitionList_t));
    partitionList.version = dcgmActivatedFabricPartitionList_version;
    partitionList.numPartitions = pCmdParser->mNumPartitions;

    for (unsigned i = 0; i < pCmdParser->mNumPartitions; i++)
    {
        partitionList.partitionIds[i] = pCmdParser->mPartitionIds[i];
    }

    dcgmReturn = dcgmInternalGetExportTable((const void**)&pEtbl, &ETID_DCGMModuleFMInternal);
    if (DCGM_ST_OK  != dcgmReturn)
    {
        fprintf(stderr, "Error: Can't get the Fabric Manager export table. Return: %d\n", dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = dcgmInternalGetExportTable((const void**)&pEtbl, &ETID_DCGMModuleFMInternal);
    if (DCGM_ST_OK  != dcgmReturn) {
        fprintf(stderr, "Error: Can't get the Fabric Manager export table. Return: %d\n", dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = DCGM_CALL_ETBL(pEtbl, fpdcgmSetActivatedFabricPartitions,
                                (dcgmHandle, &partitionList));
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "Error: Failed to set activated partitions. Return: %d\n", dcgmReturn);
    }

    return dcgmReturn;
}

dcgmReturn_t colwertToBinStatefile(SharedFabricCmdParser_t *pCmdParser, dcgmHandle_t dcgmHandle)
{
    fabricmanagerHA::fmHaState haState;
    std::string  stateText;
    FILE        *outFile;

    outFile = fopen( pCmdParser->mOutFileName, "w+" );
    if ( outFile == NULL )
    {
        fprintf(stderr, "Failed to open output file %s, error is %s.\n",
                pCmdParser->mOutFileName, strerror(errno));
        return DCGM_ST_GENERIC_ERROR;
    }

    // Read the protobuf binary file.
    std::fstream input(pCmdParser->mInFileName, std::ios::in | std::ios::binary);
    if ( !input )
    {
        fprintf(stderr, "File %s is not Found. \n", pCmdParser->mInFileName);
        fclose( outFile );
        return DCGM_ST_GENERIC_ERROR;

    }
    else if ( !haState.ParseFromIstream(&input) )
    {
        fprintf(stderr, "Failed to parse file %s. \n", pCmdParser->mInFileName);
        fclose( outFile );
        return DCGM_ST_GENERIC_ERROR;
    }

    google::protobuf::TextFormat::PrintToString(haState, &stateText);
    fwrite( stateText.c_str(), 1, (int)stateText.length(), outFile);
    fclose( outFile );
    return DCGM_ST_OK;
}

dcgmReturn_t colwertToTxtStatefile(SharedFabricCmdParser_t *pCmdParser, dcgmHandle_t dcgmHandle)
{
    fabricmanagerHA::fmHaState haState;
    int       inFileFd;
    FILE     *outFile;

    inFileFd = open(pCmdParser->mInFileName, O_RDONLY);
    if ( inFileFd < 0 )
    {
        fprintf(stderr, "Failed to open input file %s, error is %s.\n",
                pCmdParser->mInFileName, strerror(errno));
        return DCGM_ST_GENERIC_ERROR;
    }

    outFile = fopen( pCmdParser->mOutFileName, "w+" );
    if ( outFile == NULL )
    {
        fprintf(stderr, "Failed to open output file %s, error is %s.\n",
                pCmdParser->mOutFileName, strerror(errno));
        close( inFileFd );
        return DCGM_ST_GENERIC_ERROR;
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
        return DCGM_ST_GENERIC_ERROR;
    }

    haState.SerializeToArray( bufToWrite, fileLength );
    fwrite( bufToWrite, 1, fileLength, outFile );
    close( inFileFd );
    fclose ( outFile );
    return DCGM_ST_OK;
}

int cleanup(SharedFabricCmdParser_t *pCmdParser, dcgmHandle_t dcgmHandle, int status)
{
    (void)sharedFabricCmdParserDestroy(pCmdParser);
    dcgmDisconnect(dcgmHandle);
    dcgmShutdown();

    return status;
}

int main(int argc, char **argv)
{
    dcgmReturn_t ret;
    SharedFabricCmdParser_t *pCmdParser;
    dcgmHandle_t dcgmHandle;

    pCmdParser = sharedFabricCmdParserInit(argc, argv, sharedFabricSelectorArgs,
                                           SHARED_FABRIC_CMD_COUNT, sharedFabricSelectorUsage,
                                           sharedFabricSelectorDisplayHelpMessage);
    if (NULL == pCmdParser) {
        return DCGM_ST_BADPARAM;
    }

    ret = sharedFabricCmdProcessing(pCmdParser);
    if (DCGM_ST_OK != ret) {
        if (ret == DCGM_ST_BADPARAM){
            fprintf(stderr, "Unable to process command: bad command line parameter. \n");
        } else {
            fprintf(stderr, "Unable to process command: generic error. \n");
        }
        sharedFabricCmdParserDestroy(pCmdParser);
        return ret;
    }

    /* Initialize DCGM */
    ret = dcgmInit();
    if (DCGM_ST_OK != ret) {
        // assume that error message has already been printed
        fprintf(stderr, "Error: DCGM failed to initialize");
        cleanup(pCmdParser, dcgmHandle, -1);
        return ret;
    }

    ret = dcgmConnect(pCmdParser->mHostname, &dcgmHandle);
    if (ret != DCGM_ST_OK){
        fprintf(stderr, "Error connecting to Fabric Manager. Return: %s \n", errorString(ret));
        cleanup(pCmdParser, dcgmHandle, -1);
        return ret;
    }

    if (pCmdParser->mListPartition) {
        ret = listPartition(pCmdParser, dcgmHandle);
    }
    else if (pCmdParser->mActivatePartition) {
        ret = activatePartition(pCmdParser, dcgmHandle);
    }
    else if (pCmdParser->mDeactivatePartition) {
        ret = deactivatePartition(pCmdParser, dcgmHandle);
    }
    else if (pCmdParser->mSetActivatedPartitions) {
        ret = setActivatedPartitionList(pCmdParser, dcgmHandle);
    }
    else if (pCmdParser->mBinToTxt) {
        ret = colwertToBinStatefile(pCmdParser, dcgmHandle);
    }
    else if (pCmdParser->mTxtToBin) {
        ret = colwertToTxtStatefile(pCmdParser, dcgmHandle);
    }
    else {
        fprintf(stderr, "Error unknown command. \n");
        ret = DCGM_ST_BADPARAM;
    }

    cleanup(pCmdParser, dcgmHandle, 0);
    return ret;
}
