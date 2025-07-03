/*
 * DcgmiTest.cpp
 *
 */

#include "DcgmiTest.h"
#include <sstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <ctype.h>
#include <ctime>
#include "dcgm_structs.h"
#include "dcgm_agent.h"
#include "dcgm_agent_internal.h"
#include "dcgm_client_internal.h"
#include "CommandOutputController.h"

using namespace std;



//const etblDCGMClientTestInternal *t_pEtbl;
extern const etblDCGMClientInternal *g_pEtblClient;
extern const etblDCGMEngineTestInternal *g_pEtblAgentInternal;
const etblDCGMEngineTestInternal *t_pEtbl;

/***************************************************************************************/

char TEST_DATA[] =
        " <DATA_NAME              > : <DATA_INFO                                   > \n";

#define DATA_NAME_TAG "<DATA_NAME"
#define DATA_INFO_TAG "<DATA_INFO"


/**************************************************************************************/

DcgmiTest::DcgmiTest() {
    // TODO Auto-generated constructor stub

}

DcgmiTest::~DcgmiTest() {
    // TODO Auto-generated destructor stub
}

dcgmReturn_t DcgmiTest::CacheFileLoadSave(dcgmHandle_t mDcgmHandle, std::string filename, bool save)
{
    dcgmReturn_t result = DCGM_ST_OK;
    if (save)
        result = DCGM_CALL_ETBL(g_pEtblClient, fpClientSaveCacheManagerStats, (mDcgmHandle, filename.c_str(), DCGM_STATS_FILE_TYPE_JSON));
    else
        result = DCGM_CALL_ETBL(g_pEtblClient, fpClientLoadCacheManagerStats, (mDcgmHandle, filename.c_str(), DCGM_STATS_FILE_TYPE_JSON));

    if (DCGM_ST_OK != result) {
        std::cout << "Error: Cannot " << (save ? "save" : "load") << " cache information from the remote node. Return: " << errorString(result) << std::endl;
        return result;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmiTest::IntrospectCache(dcgmHandle_t mDcgmHandle, unsigned int gId, std::string fieldId, bool isGroup)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmCacheManagerFieldInfo_t fieldInfo;
    dcgmGroupInfo_t stLwcmGroupInfo;
    unsigned int gpuIds[DCGM_MAX_NUM_DEVICES];
    unsigned int i, numGpus = 0;

    // fetch gpus for group
    if (isGroup)
    {
        stLwcmGroupInfo.version = dcgmGroupInfo_version;
        result = dcgmGroupGetInfo(mDcgmHandle, (dcgmGpuGrp_t)(long long) gId, &stLwcmGroupInfo);
        if (DCGM_ST_OK != result) 
        {
            std::string error = (result == DCGM_ST_NOT_CONFIGURED)? "The Group is not found" : errorString(result);
            std::cout << "Error: Unable to retrieve information about group " << gId << ". Return: " << error << "."<< std::endl;
            return result;
        }
        
        for(i = 0; i < stLwcmGroupInfo.count; i++)
        {
            if(stLwcmGroupInfo.entityList[i].entityGroupId == DCGM_FE_GPU)
            {
                gpuIds[numGpus] = stLwcmGroupInfo.entityList[i].entityId;
                numGpus++;
            }   
        }

    } 
    else 
    {
        gpuIds[0] =  gId;
        numGpus++;
    }

    // fetch export table
    result = dcgmInternalGetExportTable((const void**)&t_pEtbl, &ETID_DCGMEngineTestInternal);

    if (result != DCGM_ST_OK)
    {
        std::cout << "Error: get the export table. Return: " << errorString(result) << std::endl;
        return result;
    }

    // get field info
    DcgmFieldsInit();
    memset(&fieldInfo,0,sizeof(dcgmCacheManagerFieldInfo_t));
    fieldInfo.version = dcgmCacheManagerFieldInfo_version;
    result = HelperParseForFieldId(fieldId, fieldInfo.fieldId, mDcgmHandle);

    if (result != DCGM_ST_OK){
        cout << "Bad parameter passed to function. ";
        return result;
    }

    for (unsigned int i = 0; i < numGpus ; i++)
    {
        fieldInfo.gpuId = (unsigned int)(uintptr_t) gpuIds[i];

        result = DCGM_CALL_ETBL(t_pEtbl, fpdcgmGetCacheManagerFieldInfo, (mDcgmHandle, &fieldInfo));

        if (DCGM_ST_OK != result) 
        {
            std::cout << "Error: Unable to get field info for GPU ID: " << fieldInfo.gpuId << ". Return: " << errorString(result) << std::endl;
            return result;
        }

        HelperDisplayField(fieldInfo);
    }

    //std::cout << "Successfully retrieved cache field info." << std::endl;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmiTest::InjectCache(dcgmHandle_t mDcgmHandle, unsigned int gpuId, std::string fieldId, unsigned int pTime, std::string &injectValue)
{

    dcgmReturn_t result = DCGM_ST_OK;
    dcgmInjectFieldValue_t injectFieldValue;

    // fetch export table
    result = dcgmInternalGetExportTable((const void**)&t_pEtbl, &ETID_DCGMEngineTestInternal);

    if (result != DCGM_ST_OK){
        std::cout << "Error: get the export table. Return: " << errorString(result) << std::endl;
        return result;
    }

    DcgmFieldsInit();

    // get current time
    time_t  timev;
    time(&timev);

    unsigned short fieldIdsd;
    injectFieldValue.ts = (pTime + timev)*1000000 + 4000000; /// adding default of 4 microseconds into the future
    HelperParseForFieldId(fieldId,(unsigned short&) injectFieldValue.fieldId, mDcgmHandle);
    HelperInitFieldValue(injectFieldValue, injectValue);

    // inject field value
    result = DCGM_CALL_ETBL(t_pEtbl, fpdcgmInjectFieldValue, (mDcgmHandle, gpuId, &injectFieldValue));

    if (DCGM_ST_OK != result) {
        std::cout << "Error: Unable to inject info. Return: " << errorString(result) << std::endl;
        return result;
    }

    std::cout << "Successfully injected field info." << std::endl;

    return DCGM_ST_OK;
}

void DcgmiTest::HelperDisplayField(dcgmCacheManagerFieldInfo_t &fieldInfo){
    CommandOutputController cmdView = CommandOutputController();
    dcgm_field_meta_p fieldMeta;
    long long buffer = 0;

    fieldMeta = DcgmFieldGetById(fieldInfo.fieldId);

    cmdView.setDisplayStencil(TEST_DATA);

    cmdView.addDisplayParameter(DATA_NAME_TAG,   "Field ID");
    cmdView.addDisplayParameter(DATA_INFO_TAG,   fieldMeta->tag);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG,   "Flags");
    cmdView.addDisplayParameter(DATA_INFO_TAG, fieldInfo.flags);
    cmdView.display();

    //buffer = fieldInfo.lastStatus;
    cmdView.addDisplayParameter(DATA_NAME_TAG,   "Last Status");
    cmdView.addDisplayParameter(DATA_INFO_TAG, fieldInfo.lastStatus);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG,   "Number of Samples");
    cmdView.addDisplayParameter(DATA_INFO_TAG, fieldInfo.numSamples);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG,   "Newest Timestamp");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatTimestamp(fieldInfo.newestTimestamp));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG,   "Oldest Timestamp");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatTimestamp(fieldInfo.oldestTimestamp));
    cmdView.display();

    //buffer = fieldInfo.monitorFrequencyUsec / 1000000;
    cmdView.addDisplayParameter(DATA_NAME_TAG,   "Monitor Frequency");
    cmdView.addDisplayParameter(DATA_INFO_TAG, fieldInfo.monitorFrequencyUsec / 1000000);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG,   "Max Age (sec)");
    cmdView.addDisplayParameter(DATA_INFO_TAG, fieldInfo.maxAgeUsec / 1000000);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG,   "Fetch Count");
    cmdView.addDisplayParameter(DATA_INFO_TAG, fieldInfo.fetchCount);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG,   "Total Fetch Usec");
    cmdView.addDisplayParameter(DATA_INFO_TAG, fieldInfo.execTimeUsec);
    cmdView.display();

    double usecPerFetch = 0.0;
    if(fieldInfo.fetchCount != 0)
    {
        usecPerFetch = (double)fieldInfo.execTimeUsec / (double)fieldInfo.fetchCount;
    }
    cmdView.addDisplayParameter(DATA_NAME_TAG,   "Usec Per Fetch");
    cmdView.addDisplayParameter(DATA_INFO_TAG, usecPerFetch);
    cmdView.display();

    std::cout << endl;

}

dcgmReturn_t DcgmiTest::HelperInitFieldValue(dcgmInjectFieldValue_t &injectFieldValue, std::string &injectValue){

    dcgm_field_meta_p fieldMeta;

    // get meta data
    fieldMeta = DcgmFieldGetById(injectFieldValue.fieldId);

    injectFieldValue.version = dcgmInjectFieldValue_version;
    injectFieldValue.fieldType = fieldMeta->fieldType;
    injectFieldValue.status = DCGM_ST_OK;

    // wrap in try catch
    switch (injectFieldValue.fieldType){
    case DCGM_FT_TIMESTAMP:
    case DCGM_FT_INT64:
        injectFieldValue.value.i64 = atol(injectValue.c_str());
        break;
    case DCGM_FT_STRING:
        strncpy(injectFieldValue.value.str, injectValue.c_str(), 255);
        break;
    case DCGM_FT_DOUBLE:
        injectFieldValue.value.dbl = atof(injectValue.c_str());
        break;
    case DCGM_FT_BINARY:
        //Not Supported
        //injectFieldValue.value.blob
        injectFieldValue.value.i64 = 0;
        break;
    default:
        break;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmiTest::HelperParseForFieldId(std::string str, unsigned short &fieldId,
                                         dcgmHandle_t dcgmHandle){

    // find , or :

    dcgmFieldGrp_t fieldGroupId;
    dcgmReturn_t dcgmReturn;
    int index;
    dcgmFieldGroupInfo_t fieldGroupInfo;

    if (str.find(',') != string::npos){
        index = str.find(',');
    } else {
        fieldId = atoi(str.c_str());
        return DCGM_ST_OK;
    }

    fieldGroupId = (dcgmFieldGrp_t)(intptr_t)atoi(str.substr(0,1).c_str());

    index = atoi(str.substr(index + 1).c_str());

    memset(&fieldGroupInfo, 0, sizeof(fieldGroupInfo));
    fieldGroupInfo.version = dcgmFieldGroupInfo_version;
    fieldGroupInfo.fieldGroupId = fieldGroupId;

    dcgmReturn = dcgmFieldGroupGetInfo(dcgmHandle, &fieldGroupInfo);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d %llu", "dcgmFieldGroupGetInfo returned %d for fieldGrpId %llu",
                    (int)dcgmReturn, (unsigned long long)fieldGroupId);
        return dcgmReturn;
    }

    if (index >= (int)fieldGroupInfo.numFieldIds){
        return DCGM_ST_BADPARAM;
    }

    fieldId = fieldGroupInfo.fieldIds[index];
    return DCGM_ST_OK;
}

std::string DcgmiTest::HelperFormatTimestamp(long long timestamp){
    stringstream ss;
    long long temp = timestamp/1000000;
    std::string str = ctime((long*)&temp);

    // Remove returned next line character
    str = str.substr(0, str.length() - 1);

    ss << str; //<< ":" << std::setw(4) << std::setfill('0') <<timestamp % 1000000;

    return ss.str();
}

/*****************************************************************************
 *****************************************************************************
 *Cache LoadSave Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
LoadSaveCache::LoadSaveCache(std::string hostname, std::string fileName, bool save) {
    mHostName = hostname;
    this->fileName = fileName;
    this->save = save;
}

/*****************************************************************************/
LoadSaveCache::~LoadSaveCache() {
}

/*****************************************************************************/
int LoadSaveCache::Execute() {
    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return adminObj.CacheFileLoadSave(mLwcmHandle, fileName, save);
}

/*****************************************************************************
 *****************************************************************************
 *Cache Introspect Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
IntrospectCache::IntrospectCache(std::string hostname, unsigned int groupId, std::string fieldId , bool isGroup) {
    mHostName = hostname;
    mGpuId = groupId;
    mFieldId = fieldId;
    mIDisGroup = isGroup;
}

/*****************************************************************************/
IntrospectCache::~IntrospectCache() {
}

/*****************************************************************************/
int IntrospectCache::Execute() {
    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return adminObj.IntrospectCache(mLwcmHandle, mGpuId, mFieldId, mIDisGroup);
}

/*****************************************************************************
 *****************************************************************************
 *Cache Inject Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
InjectCache::InjectCache(std::string hostname, unsigned int gpuId, std::string fieldId, unsigned int pTime, std::string injectValue) {
    mHostName = hostname;
    mGId = gpuId;
    mFieldId = fieldId;
    mTime = pTime;
    mInjectValue = injectValue;
}

/*****************************************************************************/
InjectCache::~InjectCache() {
}

/*****************************************************************************/
int InjectCache::Execute() {
    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return adminObj.InjectCache(mLwcmHandle, mGId, mFieldId, mTime, mInjectValue);
}
