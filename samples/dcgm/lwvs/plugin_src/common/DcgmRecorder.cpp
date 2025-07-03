#include <cstdlib>
#include <ctime>
#include <sstream>
#include <vector>
#include <fstream>
#include <iostream>
#include <errno.h>
#include <stdexcept>

#include "DcgmRecorder.h"
#include "dcgm_agent.h"
#include "dcgm_fields.h"
#include "common.h"
#include "logging.h"
#include "timelib.h"

const long long defaultFrequency = 1000000; // update each field every second (a million microseconds)

DcgmRecorder::DcgmRecorder() : m_fieldIds(), m_gpuIds(), m_fieldGroupId(0), m_gpuData(), m_groupedData(),
                               m_groupSingleData(), m_dcgmHandle(), m_dcgmGroup(), m_dcgmSystem(),
                               m_valuesHolder(), m_groupedDataMutex(0), m_gpuDataMutex(0),
                               m_groupedSingleDataMutex(0), m_nextValuesSinceTs(0)

{
}

DcgmRecorder::~DcgmRecorder()
{
    Shutdown();
}

std::string DcgmRecorder::CreateGroup(const std::vector<unsigned int> &gpuIds, bool allGpus,
                                      const std::string &groupName)
{
    dcgmReturn_t ret;
    char         randomName[100];
    std::string  errStr;

    if (m_dcgmHandle.GetHandle() == 0)
        return "Must connect to DCGM before creating a group";

    ret = m_dcgmGroup.Init(m_dcgmHandle.GetHandle(), groupName, gpuIds);

    if (ret != DCGM_ST_OK)
        errStr = m_dcgmHandle.RetToString(ret);

    return errStr;
}

std::string DcgmRecorder::AddWatches(const std::vector<unsigned short> &fieldIds,
                                     const std::vector<unsigned int> &gpuIds, bool  allGpus,
                                     const std::string &fieldGroupName, const std::string &groupName,
                                     double testDuration)

{
    dcgmReturn_t ret;
    char         randomName[100];
    std::string  errStr;
    m_fieldIds = fieldIds;
    m_gpuIds = gpuIds;
    unsigned short fieldIdArray[DCGM_FI_MAX_FIELDS];
    int numFieldIds;

    if (fieldIds.size() == 0 || fieldIds.size() > DCGM_FI_MAX_FIELDS)
        return "Field Ids must contain at least 1 field id";

    if (gpuIds.size() == 0)
        return "Gpu Ids must contain at least 1 gpu id";

    errStr = CreateGroup(gpuIds, allGpus, groupName);
    if (errStr.size() != 0)
        return errStr;

    numFieldIds = fieldIds.size();

    for (size_t i = 0; i < fieldIds.size(); i++)
        fieldIdArray[i] = fieldIds[i];

    ret = m_dcgmGroup.FieldGroupCreate(fieldIds, fieldGroupName);
    if (ret != DCGM_ST_OK)
    {
        GetErrorString(ret, errStr);
        return errStr;
    }

    ret = m_dcgmGroup.WatchFields(defaultFrequency, testDuration + 30);
    if (ret != DCGM_ST_OK)
    {
        GetErrorString(ret, errStr);
    }

    return errStr;
}

void DcgmRecorder::GetErrorString(dcgmReturn_t ret, std::string &err)
{
    std::stringstream err_stream;
    const char *tmp = errorString(ret);

    if (tmp == NULL)
    {
        err_stream << "Unknown error from DCGM: " << ret;
        err = err_stream.str();
    }
    else
        err = tmp;
}

std::string DcgmRecorder::Init(const std::string &hostname)
{
    std::string errStr;
    dcgmReturn_t ret;

    ret = m_dcgmHandle.ConnectToDcgm(hostname);

    if (ret != DCGM_ST_OK)
        errStr = m_dcgmHandle.GetLastError();
    else
        m_dcgmSystem.Init(m_dcgmHandle.GetHandle());

    return errStr;
}

std::string DcgmRecorder::Shutdown()
{
    std::string errStr;

    if (m_dcgmHandle.GetHandle() == 0)
    {
        m_fieldGroupId = 0;
        return errStr;
    }

    if (m_fieldGroupId != 0)
    {
        // Ignore errors
        dcgmFieldGroupDestroy(m_dcgmHandle.GetHandle(), m_fieldGroupId);
        m_fieldGroupId = 0;
    }

    m_dcgmGroup.Cleanup();

    return errStr;
}

void DcgmRecorder::GetTagFromFieldId(unsigned short fieldId, std::string &tag)
{
    dcgm_field_meta_p fm = DcgmFieldGetById(fieldId);

    if (fm == 0)
    {
        std::stringstream tmp;
        tmp << fieldId;
        tag = tmp.str();
    }
    else
        tag = fm->tag;
}

void DcgmRecorder::ClearLwstomData()
{
    DcgmLockGuard lock(&m_gpuDataMutex);
    m_gpuData.clear();
}

void DcgmRecorder::InsertLwstomData(const std::string &groupName, const std::string &name, dcgmTimeseriesInfo_t &data)
{
    DcgmLockGuard lock(&m_groupedDataMutex);
    m_groupedData[groupName][name].push_back(data);
}


void DcgmRecorder::InsertLwstomData(unsigned int gpuId, const std::string &name, dcgmTimeseriesInfo_t &data)
{
    DcgmLockGuard lock(&m_gpuDataMutex);
    m_gpuData[gpuId][name].push_back(data);
}

void DcgmRecorder::SetGroupedStat(const std::string &groupName, const std::string &name, double value)
{
    dcgmTimeseriesInfo_t data;
    data.val.fp64 = value;
    data.isInt = false;
    data.timestamp = timelib_usecSince1970();

    InsertLwstomData(groupName, name, data);
}

void DcgmRecorder::SetGroupedStat(const std::string &groupName, const std::string &name, long long value)
{
    dcgmTimeseriesInfo_t data;
    data.val.i64 = value;
    data.isInt = true;
    data.timestamp = timelib_usecSince1970();

    InsertLwstomData(groupName, name, data);
}

void DcgmRecorder::SetGpuStat(unsigned int gpuId, const std::string &name, double value)
{
    dcgmTimeseriesInfo_t data;
    data.val.fp64 = value;
    data.isInt = false;
    data.timestamp = timelib_usecSince1970();

    InsertLwstomData(gpuId, name, data);
}

void DcgmRecorder::SetGpuStat(unsigned int gpuId, const std::string &name, long long value)
{
    dcgmTimeseriesInfo_t data;
    data.val.i64 = static_cast<uint64_t>(value);
    data.isInt = true;
    data.timestamp = timelib_usecSince1970();

    InsertLwstomData(gpuId, name, data);
}

int storeValues(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId, dcgmFieldValue_v1 *values,
                     int numValues, void *userData)
{
    if (userData == 0)
        return static_cast<int>(DCGM_ST_BADPARAM);

    DcgmValuesSinceHolder *dvsh = static_cast<DcgmValuesSinceHolder *>(userData);
    for (int i = 0; i < numValues; i++)
    {
        // Skip bad values
        if (values[i].status != DCGM_ST_OK)
            continue;

        dvsh->AddValue(entityGroupId, entityId, values[i].fieldId, values[i]);
    }

    return 0;
}

dcgmReturn_t DcgmRecorder::GetFieldValuesSince(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId,
                                               unsigned short fieldId, long long ts, bool force)
{
    // Use the timestamp to prevent asking for something that we've already grabbed, unless force is true
    if (force == true)
    {
        m_valuesHolder.ClearCache();
    }
    else if (ts < m_nextValuesSinceTs)
    {
        ts = m_nextValuesSinceTs;
    }

    dcgmReturn_t ret = m_dcgmGroup.GetValuesSince(ts, storeValues, &m_valuesHolder, &m_nextValuesSinceTs);

    return ret;
}
    
unsigned int DcgmRecorder::GpuIdToJsonStatsIndex(unsigned int gpuId)
{
    for (size_t i = 0; i < m_gpuIds.size(); i++)
    {
        if (m_gpuIds[i] == gpuId)
            return i;
    }

    return m_gpuIds.size();
}

void DcgmRecorder::AddLwstomTimeseriesVector(Json::Value &jv, std::vector<dcgmTimeseriesInfo_t> &vec)
{
    Json::ArrayIndex next = 0;
    for (size_t i = 0; i < vec.size(); i++, next++)
    {
        if (vec[i].isInt)
            jv[next]["value"] = static_cast<Json::Value::Int64>(vec[i].val.i64);
        else
            jv[next]["value"] = vec[i].val.fp64;
        jv[next]["timestamp"] = vec[i].timestamp;
    }
}

void DcgmRecorder::AddGpuDataToJson(Json::Value &jv)
{
    DcgmLockGuard lock(&m_gpuDataMutex);
    std::map<unsigned int, std::map<std::string, std::vector<dcgmTimeseriesInfo_t> > >::iterator mapMapIt;
    for (mapMapIt = m_gpuData.begin(); mapMapIt != m_gpuData.end(); mapMapIt++)
    {
        unsigned int jsonIndex = GpuIdToJsonStatsIndex(mapMapIt->first);

        std::map<std::string, std::vector<dcgmTimeseriesInfo_t> >::iterator vecMapIt;
        for (vecMapIt = mapMapIt->second.begin(); vecMapIt != mapMapIt->second.end(); ++vecMapIt)
            AddLwstomTimeseriesVector(jv[GPUS][jsonIndex][vecMapIt->first], vecMapIt->second);
    }
}

void DcgmRecorder::AddNonTimeseriesDataToJson(Json::Value &jv)
{
    DcgmLockGuard lock(&m_groupedSingleDataMutex);
    std::map<std::string, std::map<std::string, std::string> >::iterator sMapMapIt;
    for (sMapMapIt = m_groupSingleData.begin(); sMapMapIt != m_groupSingleData.end(); ++sMapMapIt)
    {
        std::string name = sMapMapIt->first;
        std::map<std::string, std::string>::iterator mapIt;
        for (mapIt = sMapMapIt->second.begin(); mapIt != sMapMapIt->second.end(); ++mapIt)
            jv[name][mapIt->first] = mapIt->second;
    }
}

void DcgmRecorder::AddGroupedDataToJson(Json::Value &jv)
{
    DcgmLockGuard lock(&m_groupedDataMutex);
    std::map<std::string, std::map<std::string, std::vector<dcgmTimeseriesInfo_t> > >::iterator gMapMapIt;
    for (gMapMapIt = m_groupedData.begin(); gMapMapIt != m_groupedData.end(); ++gMapMapIt)
    {
        std::string groupName = gMapMapIt->first;

        std::map<std::string, std::vector<dcgmTimeseriesInfo_t> >::iterator vecMapIt;
        for (vecMapIt = gMapMapIt->second.begin(); vecMapIt != gMapMapIt->second.end(); ++vecMapIt)
            AddLwstomTimeseriesVector(jv[groupName][vecMapIt->first], vecMapIt->second);
    }
}

void DcgmRecorder::AddLwstomData(Json::Value &jv)
{
    AddGpuDataToJson(jv);

    AddNonTimeseriesDataToJson(jv);

    AddGroupedDataToJson(jv);
}

std::string DcgmRecorder::GetWatchedFieldsAsJson(Json::Value &jv, long long ts)
{
    std::string  errStr;
    dcgmReturn_t ret = DCGM_ST_OK;

    // Make sure we have all of our values queried
    for (size_t i = 0; i < m_gpuIds.size(); i++)
    {
        for (size_t j = 0; j < m_fieldIds.size(); j++)
        {
            ret = GetFieldValuesSince(DCGM_FE_GPU, m_gpuIds[i], m_fieldIds[j], ts, false);
            
            if (ret != DCGM_ST_OK)
            {
                GetErrorString(ret, errStr);
                return errStr;
            }
        }
    }

    m_valuesHolder.AddToJson(jv);
    AddLwstomData(jv);

    return errStr;
}

/*
 * GPUs Json is in the format:
 *
 *  jv[GPUS] is an array of gpu ids
 *  jv[GPUS][gpuId] is a map of atributes
 *  jv[GPUS][gpuId][attrname] is an array of objects with timestamp and value
 */
std::string DcgmRecorder::GetWatchedFieldsAsString(std::string &output, long long ts)
{
    Json::Value jv;
    std::string errStr = GetWatchedFieldsAsJson(jv, ts);
    if (errStr.size() > 0)
        return errStr;

    std::stringstream buf;
    buf << "GPU Collections\n";

    Json::Value &gpuArray = jv[GPUS];
    for (unsigned int gpuIndex = 0; gpuIndex < gpuArray.size(); gpuIndex++)
    {
        buf << "\tLwml Idx " << gpuIndex << "\n";

        for (Json::Value::iterator it = gpuArray[gpuIndex].begin(); it != gpuArray[gpuIndex].end(); ++it)
        {
            std::string attrName = it.key().asString();
            Json::Value &attrArray = gpuArray[gpuIndex][attrName];

            for (unsigned int attrIndex = 0; attrIndex < attrArray.size(); attrIndex++)
            {
                buf << "\t\t" << attrName << ": timestamp " << attrArray[attrIndex]["timestamp"];
                buf << ", val " << attrArray[attrIndex]["value"] << "\n";
            }
        }
    }

    output = buf.str();

    return errStr;
}

int DcgmRecorder::WriteToFile(const std::string &filename, int logFileType, long long testStart)
{
    std::ofstream f;
    f.open(filename.c_str());

    if (f.fail())
    {
        PRINT_ERROR("%s %s", "Unable to open file %s: '%s'", filename.c_str(), strerror(errno));
        return -1;
    }


    switch (logFileType)
    {
        case LWVS_LOGFILE_TYPE_TEXT:
            {
                std::string output;
                std::string error = GetWatchedFieldsAsString(output, testStart);
                if (error.size() == 0)
                    f << output;
                else
                    f << error;

                break;
            }

        case LWVS_LOGFILE_TYPE_JSON:
        default:
            {
                Json::Value jv;
                std::string error = GetWatchedFieldsAsJson(jv, testStart);

                if (error.size() == 0)
                    f << jv.toStyledString();
                else
                    f << error;
            }

            break;
    }

    f.close();
    return 0;
}

dcgmReturn_t DcgmRecorder::GetFieldSummary(dcgmFieldSummaryRequest_t &request)
{
    dcgmReturn_t ret;
    std::string  errStr;

    request.version = dcgmFieldSummaryRequest_version1;
    ret = dcgmGetFieldSummary(m_dcgmHandle.GetHandle(), &request);

    if (ret == DCGM_ST_NO_DATA)
    {
        // Lack of data is not an error
        ret = DCGM_ST_OK;
    }

    return ret;
}

int DcgmRecorder::GetValueIndex(unsigned short fieldId)
{
    // Default to index 0 for DCGM_SUMMARY_MAX
    int index = 0;

    switch (fieldId)
    {
        case DCGM_FI_DEV_ECC_SBE_VOL_TOTAL:
        case DCGM_FI_DEV_ECC_DBE_VOL_TOTAL:
        case DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L0:
        case DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L1:
        case DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L2:
        case DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L3:
        case DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L4:
        case DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L5:
        case DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_TOTAL:
        case DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L0:
        case DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L1:
        case DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L2:
        case DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L3:
        case DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L4:
        case DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L5:
        case DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_TOTAL:
        case DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL:
        case DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_TOTAL:
        case DCGM_FI_DEV_PCIE_REPLAY_COUNTER:

            // All of these should use DCGM_SUMMARY_DIFF
            index = 1;
            break;
    }

    return index;
}

int DcgmRecorder::CheckErrorFields(std::vector<unsigned short> &fieldIds,
                                   const std::vector<dcgmTimeseriesInfo_t> *failureThresholds, unsigned int gpuId,
                                   std::vector<DcgmError> &errorList, timelib64_t startTime,
                                   timelib64_t &endTime)
{
    char buf[256] = { 0 };
    int  st = DR_SUCCESS;

    dcgmFieldSummaryRequest_t fsr;
    std::string error;
    memset(&fsr, 0, sizeof(fsr));
    fsr.entityGroupId = DCGM_FE_GPU;
    fsr.entityId = gpuId;
    fsr.summaryTypeMask = DCGM_SUMMARY_MAX | DCGM_SUMMARY_DIFF;
    fsr.startTime = startTime;
    fsr.endTime = endTime;

    for (size_t i = 0; i < fieldIds.size(); i++)
    {
        dcgm_field_meta_p fm = DcgmFieldGetById(fieldIds[i]);
        if (fm == 0)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CANNOT_GET_FIELD_TAG, d, fieldIds[i]);
            errorList.push_back(d);
            return DR_COMM_ERROR;
        }

        memset(&fsr.response, 0, sizeof(fsr.response));
        fsr.fieldId = fieldIds[i];
        dcgmReturn_t ret = GetFieldSummary(fsr);

        if (ret != DCGM_ST_OK)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, fm->tag, gpuId);
            errorList.push_back(d);
            return DR_COMM_ERROR;
        }

        int valueIndex = GetValueIndex(fieldIds[i]);

        // Check for failure detection
        if (fm->fieldType == DCGM_FT_INT64)
        {
            if (failureThresholds == 0 && fsr.response.values[valueIndex].i64 > 0 &&
                DCGM_INT64_IS_BLANK(fsr.response.values[valueIndex].i64) == 0)
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FIELD_VIOLATION, d,
                                                  fsr.response.values[valueIndex].i64, fm->tag, gpuId);
                errorList.push_back(d);
                st = DR_VIOLATION;
            }
            else if (failureThresholds != 0 && fsr.response.values[valueIndex].i64 >
                    (*failureThresholds)[i].val.i64 &&
                    DCGM_INT64_IS_BLANK(fsr.response.values[valueIndex].i64) == 0)
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FIELD_THRESHOLD, d,
                                                  fsr.response.values[valueIndex].i64, fm->tag, gpuId,
                                                  (*failureThresholds)[i].val.i64);
                errorList.push_back(d);
                st = DR_VIOLATION;
            }
        }
        else if (fm->fieldType == DCGM_FT_DOUBLE)
        {
            if (failureThresholds == 0 && fsr.response.values[valueIndex].fp64 > 0.0 &&
                DCGM_FP64_IS_BLANK(fsr.response.values[valueIndex].fp64) == 0)
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FIELD_VIOLATION_DBL, d,
                                                  fsr.response.values[valueIndex].fp64, fm->tag, gpuId);
                errorList.push_back(d);
                st = DR_VIOLATION;
            }
            else if (failureThresholds != 0 && fsr.response.values[valueIndex].fp64 >
                    (*failureThresholds)[i].val.fp64 &&
                    DCGM_FP64_IS_BLANK(fsr.response.values[valueIndex].fp64) == 0)
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FIELD_THRESHOLD, d,
                                                  fsr.response.values[valueIndex].fp64, fm->tag, gpuId,
                                                  (*failureThresholds)[i].val.fp64);
                errorList.push_back(d);
                st = DR_VIOLATION;
            }
        }
        else
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_UNSUPPORTED_FIELD_TYPE, d, fm->tag);
            errorList.push_back(d);
            st = DR_VIOLATION;
        }
    }

    return st;
}
    
dcgmReturn_t DcgmRecorder::CheckPerSecondErrorConditions(const std::vector<unsigned short> &fieldIds,
                                                         const std::vector<dcgmFieldValue_v1> &failureThreshold,
                                                         unsigned int gpuId, std::vector<DcgmError> &errorList,
                                                         timelib64_t startTime)
{
    dcgmReturn_t st = DCGM_ST_OK;

    if (fieldIds.size() != failureThreshold.size())
    {
        PRINT_ERROR("", "One failure threshold must be specified for each field id");
        return DCGM_ST_BADPARAM;
    }

    for (size_t i = 0; i < fieldIds.size(); i++)
    {
        std::string tag;
        GetTagFromFieldId(fieldIds[i], tag);

        // Make sure we have the timeseries data for these fields
        st = GetFieldValuesSince(DCGM_FE_GPU, gpuId, fieldIds[i], startTime, true);
        if (st != DCGM_ST_OK)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, st, tag.c_str(), gpuId);
            errorList.push_back(d);
            return st;
        }

        // On error the values holder will appened to errorList
        if (m_valuesHolder.DoesValuePassPerSecondThreshold(fieldIds[i], failureThreshold[i], gpuId, tag.c_str(),
                    errorList, startTime))
        {
            st = DCGM_ST_DIAG_THRESHOLD_EXCEEDED;
        }
    }

    return st;
}

dcgmHandle_t DcgmRecorder::GetHandle()
{
    return m_dcgmHandle.GetHandle();
}

void DcgmRecorder::SetSingleGroupStat(const std::string &gpuId, const std::string &name, const std::string &value)
{

    DcgmLockGuard lock(&m_groupedSingleDataMutex);
    m_groupSingleData[name][gpuId] = value;
}

std::vector<dcgmTimeseriesInfo_t> DcgmRecorder::GetLwstomGpuStat(unsigned int gpuId, const std::string &name)
{
    DcgmLockGuard lock(&m_gpuDataMutex);
    return m_gpuData[gpuId][name];
}

int DcgmRecorder::CheckThermalViolations(unsigned int gpuId, std::vector<DcgmError> &errorList,
                                         timelib64_t startTime, timelib64_t endTime)
{
    int  st = DR_SUCCESS;
    dcgmFieldSummaryRequest_t fsr;
    memset(&fsr, 0, sizeof(fsr));
    fsr.fieldId = DCGM_FI_DEV_THERMAL_VIOLATION;
    fsr.entityGroupId = DCGM_FE_GPU;
    fsr.entityId = gpuId;
    fsr.summaryTypeMask = DCGM_SUMMARY_SUM;
    fsr.startTime = startTime;
    fsr.endTime = endTime;

    char buf[256] = {0};

    dcgmReturn_t ret = GetFieldSummary(fsr);

    if (ret != DCGM_ST_OK)
    {
        // This may be null since we only expose thermal violations with the setting of an elwironmental variable
        return ret;
    }

    if (fsr.response.values[0].i64 > 0 &&
        !DCGM_INT64_IS_BLANK(fsr.response.values[0].i64))
    {
        dcgmReturn_t ret = GetFieldValuesSince(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_THERMAL_VIOLATION, startTime, true);
        dcgmFieldValue_v1 dfv;

        if (ret == DCGM_ST_OK)
        {
            m_valuesHolder.GetFirstNonZero(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_CLOCK_THROTTLE_REASONS, dfv, 0);
        }

        if (dfv.ts != 0) // the field value timestamp will be 0 if we couldn't find one
        {
            double timeDiff = (dfv.ts - startTime) / 1000000.0;
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_THERMAL_VIOLATIONS_TS, d, fsr.response.values[0].i64,
                                              timeDiff, gpuId);
            errorList.push_back(d);
        }
        else
        {
            double timeDiff = (dfv.ts - startTime) / 1000000.0;
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_THERMAL_VIOLATIONS, d, fsr.response.values[0].i64, gpuId);
            errorList.push_back(d);
        }

        // Thermal violations were found so, make the return indicate we found a violation
        st = DR_VIOLATION;
    }

    return st;
}

int DcgmRecorder::CheckGpuTemperature(unsigned int gpuId, std::vector<DcgmError> &errorList, long long maxTemp,
                                      std::string &infoMsg, timelib64_t startTime, timelib64_t endTime,
                                      long long &highTemp)

{
    int st = DR_SUCCESS;
    char buf[256] = {0};
    dcgmFieldSummaryRequest_t fsr;
    memset(&fsr, 0, sizeof(fsr));
    fsr.fieldId = DCGM_FI_DEV_GPU_TEMP;
    fsr.entityGroupId = DCGM_FE_GPU;
    fsr.entityId = gpuId;
    fsr.summaryTypeMask = DCGM_SUMMARY_MAX | DCGM_SUMMARY_AVG;
    fsr.startTime = startTime;
    fsr.endTime = endTime;

    dcgmReturn_t ret = GetFieldSummary(fsr);

    if (ret != DCGM_ST_OK)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "gpu temperature", gpuId);
        errorList.push_back(d);
        highTemp = 0;
        return DR_COMM_ERROR;
    }

    highTemp = fsr.response.values[0].i64;
    if (DCGM_INT64_IS_BLANK(fsr.response.values[0].i64))
    {
        highTemp = 0;
    }

    if (highTemp > maxTemp)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEMP_VIOLATION, d, highTemp, gpuId, maxTemp);
        errorList.push_back(d);
        st = DR_VIOLATION;
    }

    double avg = fsr.response.values[1].i64;
    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(0);
    ss << "GPU " << gpuId << " temperature average:\t" << avg << " C";
    infoMsg = ss.str();

    return st;
}
    
int DcgmRecorder::CheckForThrottling(unsigned int gpuId, timelib64_t startTime,
                                      std::vector<DcgmError> &errorList)
{
    // mask for the failures we're evaluating
    static const uint64_t failureMask = DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN | 
                                        DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL | 
                                        DCGM_CLOCKS_THROTTLE_REASON_HW_THERMAL |
                                        DCGM_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE;
    uint64_t mask = failureMask;
    
    // Update mask to ignore throttle reasons given by the ignoreMask
    if (lwvsCommon.throttleIgnoreMask != DCGM_INT64_BLANK && lwvsCommon.throttleIgnoreMask > 0)
    {
        mask &= ~lwvsCommon.throttleIgnoreMask;
    }

    dcgmFieldValue_v1 dfv;
    dcgmReturn_t st = GetFieldValuesSince(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_CLOCK_THROTTLE_REASONS, startTime, true);
    int rc = DR_SUCCESS;

    std::stringstream buf;

    if (st != DCGM_ST_OK)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, st, "clock throttling", gpuId);
        errorList.push_back(d);
        return DR_COMM_ERROR;
    }
    
    m_valuesHolder.GetFirstNonZero(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_CLOCK_THROTTLE_REASONS, dfv, mask);
    int64_t maskedResult = dfv.value.i64 & mask;
    
    if (maskedResult)
    {
        const char *detail = NULL;
        double timeDiff = (dfv.ts - startTime) / 1000000.0;

        if ((maskedResult & DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN))
        {
            detail = "clocks_throttle_reason_hw_slowdown: either the temperature is too high or there is a "\
                     "power supply problem (the power brake assertion has been tripped).";
        }
        else if ((maskedResult & DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL))
        {
            detail = "clocks_throttle_reason_sw_thermal_slowdown: the GPU or its memory have reached unsafe "\
                     "temperatures.";
        }
        else if ((maskedResult & DCGM_CLOCKS_THROTTLE_REASON_HW_THERMAL))
        {
            detail = "clocks_throttle_reason_hw_thermal_slowdown: the GPU or its memory have reached unsafe "\
                     "temperatures.";
        }
        else if ((maskedResult & DCGM_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE))
        {
            detail = "clocks_throttle_reason_hw_power_brake_slowdown: the power brake assertion has triggered. "\
                     "Please check the power supply.";
        }

        if (detail != NULL)
        {
            rc = DR_VIOLATION;
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_THROTTLING_VIOLATION, d, gpuId, timeDiff, detail);
            errorList.push_back(d);
        }
    }

    return rc;
}

dcgmReturn_t DcgmRecorder::GetLwrrentFieldValue(unsigned int gpuId, unsigned short fieldId,
                                                dcgmFieldValue_v2 &value, unsigned int flags)
{
    memset(&value, 0, sizeof(value));

    return m_dcgmSystem.GetGpuLatestValue(gpuId, fieldId, flags, value);
}

int DcgmRecorder::GetLatestValuesForWatchedFields(unsigned int flags, std::vector<DcgmError> &errorList)
{
    dcgmReturn_t ret = m_dcgmSystem.GetLatestValuesForGpus(m_gpuIds, m_fieldIds, flags, storeValues,
                                                           &m_valuesHolder);

    if (ret != DCGM_ST_OK)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "all watched fields", m_gpuIds[0]);
        errorList.push_back(d);
        return DR_COMM_ERROR;
    }

    return DR_SUCCESS;
}
    
std::string DcgmRecorder::GetGpuUtilizationNote(unsigned int gpuId, timelib64_t startTime, timelib64_t endTime)
{
    static const int UTILIZATION_THRESHOLD = 75;
    std::stringstream msg;
    int  st = DR_SUCCESS;
    dcgmFieldSummaryRequest_t fsr;
    memset(&fsr, 0, sizeof(fsr));
    fsr.fieldId = DCGM_FI_DEV_GPU_UTIL;
    fsr.entityGroupId = DCGM_FE_GPU;
    fsr.entityId = gpuId;
    fsr.summaryTypeMask = DCGM_SUMMARY_MAX;
    fsr.startTime = startTime;
    fsr.endTime = endTime;
    
    dcgmReturn_t ret = GetFieldSummary(fsr);

    if (ret != DCGM_ST_OK)
    {
        std::string error;
        GetErrorString(ret, error);
        PRINT_ERROR("%s %u", "unable to query for gpu temperature: %s for GPU %u", error.c_str(), gpuId);
        return error;
    }

    if (fsr.response.values[0].i64 < UTILIZATION_THRESHOLD)
    {
        msg << "NOTE: GPU usage was only " << fsr.response.values[0].i64 << " for GPU " << gpuId
            << ". This may have caused the failure.";
    }

    return msg.str();
}

dcgmReturn_t DcgmRecorder::GetDeviceAttributes(unsigned int gpuId, dcgmDeviceAttributes_t &attributes)
{
    memset(&attributes, 0, sizeof(attributes));
    attributes.version = dcgmDeviceAttributes_version1;
    return m_dcgmSystem.GetDeviceAttributes(gpuId, attributes);
}

