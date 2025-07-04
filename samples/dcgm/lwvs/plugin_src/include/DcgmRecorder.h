#ifndef DCGM_RECORDER_H
#define DCGM_RECORDER_H

#include <vector>
#include <string>
#include <map>

#include "DcgmHandle.h"
#include "DcgmGroup.h"
#include "DcgmSystem.h"
#include "json/json.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include "timelib.h"
#include "DcgmValuesSinceHolder.h"
#include "DcgmMutex.h"
#include "DcgmError.h"

#define GPUS    "GPUS"
#define VALUE   "value"

#define DR_SUCCESS     0
#define DR_COMM_ERROR -1
#define DR_VIOLATION  -2
#define DR_THROTTLING -3

typedef struct
{
    union
    {
        uint64_t i64;
        double   fp64;
    } val;

    unsigned short fieldId;
    bool           isInt;
    long long      timestamp;
} dcgmTimeseriesInfo_t;

class DcgmRecorder
{
public:
    DcgmRecorder();
    ~DcgmRecorder();

    /*
     * Adds watches to the specified field list by creating field groups and gpu groups with the specified
     * names.
     *
     * Returns an empty string on SUCCESS or a string with an error message in it on failure
     */
    std::string AddWatches(const std::vector<unsigned short> &fieldIds, const std::vector<unsigned int> &gpuIds,
                           bool allGpus, const std::string &fieldGroupName, const std::string &groupName,
                           double testDuration);

    /*
     */
    std::string Init(const std::string &hostname);

    /*
     */
    std::string Shutdown();

    /*
     * Iterate over data
     *
     * @param fieldId - the fieldId we want to query from DCGM
     * @param ts - timestamp that we want values since
     *
     * @return DCGM_ST_* as appropriate
     */
    dcgmReturn_t CheckFieldValuesSince(unsigned short fieldId, long long ts);

    /*
     * Writes the data for the field id group to the specified file name in the specified format
     */
    int WriteToFile(const std::string &filename, int logFileType, long long testStart);

    /*
     */
    dcgmReturn_t GetFieldSummary(dcgmFieldSummaryRequest_t &request);

    /*
     * Populate tag with the field name for the specified field id
     */
    static void GetTagFromFieldId(unsigned short fieldId, std::string &tag);

    /*
     * Adds a custom timeseries statistic for GPU gpuId with the specified name and value
     */
    void SetGpuStat(unsigned int gpuId, const std::string &name, double value);

    /*
     * Adds a custom timeseries statistic for GPU gpuId with the specified name and value
     */
    void SetGpuStat(unsigned int gpuId, const std::string &name, long long value);

    /*
     * Get the data associated with teh custom gpu stat recorded
     */
    std::vector<dcgmTimeseriesInfo_t> GetLwstomGpuStat(unsigned int gpuId, const std::string &name);

    /*
     * Adds a custom field fo GPU gpuId
     */
    void SetSingleGroupStat(const std::string &gpuId, const std::string &name, const std::string &value);

    /*
     * Adds a custom statistic for group groupName with the specified name and value
     */
    void SetGroupedStat(const std::string &groupName, const std::string &name, double value);

    /*
     * Adds a custom statistic for group groupName with the specified name and value
     */
    void SetGroupedStat(const std::string &groupName, const std::string &name, long long value);

    /*
     * Clears the custom data lwrrently stored in this object
     */
    void ClearLwstomData();

    /*
     * Makes a stateful query for the specified entity's fieldId since ts. If we have already stored the data 
     * for the specified entity's fieldId in m_valuesHolder, then no work is done, unless force is set.
     *
     * @return:
     *
     * DCGM_ST_OK     : on success
     * DCGM_ST_*      : to call out specific errors
     */
    dcgmReturn_t GetFieldValuesSince(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId,
                                     unsigned short fieldId, long long ts, bool force);

    /*
     * Checks for any non-zero entries for any of the fields specified in fieldIds between startTime and
     * endTime for GPU gpuId, and adds an error message to errorList if any violate the failureThresholds
     * or warningThresholds. warningThresholds causes a warning message to be printed even if a field
     * value does not meet the value in failureThresholds
     *
     * @return:
     *
     * DR_SUCCESS      : on success
     * DR_COMM_ERROR   : if we couldn't get the information from DCGM
     * DR_VIOLATION    : if a value was found above a failure threshold
     */
    int CheckErrorFields(std::vector<unsigned short> &fieldIds,
                         const std::vector<dcgmTimeseriesInfo_t> *failureThreshold, unsigned int gpuId,
                         std::vector<DcgmError> &errorList, timelib64_t startTime, timelib64_t &endTime);

    /*
     * Iterates through the stored values to check if there is ever a jump that is greater than or equal to
     * the specified threshold, and records an error in errorList if one is present.
     *
     * @return
     * DCGM_ST_OK                       : on success
     * DCGM_ST_BADPARAM                 : if not enough thresholds are offered
     * DCGM_ST_DIAG_THRESHOLD_EXCEEDED  : if a value was found above a failure threshold
     */
    dcgmReturn_t CheckPerSecondErrorConditions(const std::vector<unsigned short> &fieldIds,
                                               const std::vector<dcgmFieldValue_v1> &failureThreshold,
                                               unsigned int gpuId, std::vector<DcgmError> &errorList,
                                               timelib64_t startTime);

    /*
     * Determines the index of our dcgmFieldSummaryRequest_t we should look at for this fieldId.
     * Lwrrently, the options are 0 for DCGM_SUMMARY_MAX and 1 for DCGM_SUMMARY_DIFF.
     *
     * Note: this may need updating as additional fields are added if those fields need to look at a different
     * summary option
     *
     * @return:
     *
     * 0        : look at DCGM_SUMMARY_MAX
     * 1        : look at DCGM_SUMMARY_DIFF
     */
    int GetValueIndex(unsigned short fieldId);

    /*
     * Checks for any thermal violations in the specified time period, and adds an error message if they are
     * present.
     *
     * @return:
     *
     * DR_SUCCESS     : on success
     * DR_COMM_ERROR  : if we couldn't get the information from DCGM
     * DR_VIOLATION   : if a there was a thermal violation
     */
    int CheckThermalViolations(unsigned int gpuId, std::vector<DcgmError> &errorList, timelib64_t startTime,
                               timelib64_t endTime);

    /*
     * Checks if the GPU temperature was about the maxTemp at any point in the specified time period, and adds
     * an error message if the temperature was too high. 
     * Populates infoMsg with the average temperature
     * Populates highTemp with the highest temperature recorded during the test; set to 0 on error.
     *
     * @return:
     *
     * DR_SUCCESS     : on success
     * DR_COMM_ERROR  : if we couldn't get the information from DCGM
     * DR_VIOLATION   : if the temperature was ever too high
     */
    int CheckGpuTemperature(unsigned int gpuId, std::vector<DcgmError> &errorList, long long maxTemp,
                            std::string &infoMsg, timelib64_t startTime, timelib64_t endTime, long long &highTemp);

    dcgmHandle_t GetHandle();

    /*
     * Checks if the specified GPU has throttling and sets an appropriate error message if so.
     *
     * @return:
     *
     * DR_SUCCESS     : No throttling is happening
     * DR_COMM_ERROR  : if we couldn't get the information from DCGM
     * DR_VIOLATION   : throttling is happening
     */
    int CheckForThrottling(unsigned int gpuId, timelib64_t startTime, std::vector<DcgmError> &errorList);

    /*
     * Populates dcgmTimeseriesInfo with the current value of the specified field for the specified GPU
     * Adds an appropriate warning if we can't get the fieldId
     *
     * @return:
     *
     * DCGM_ST_OK     : on success
     * DCGM_ST_*      : if we couldn't get the information from DCGM
     */
    dcgmReturn_t GetLwrrentFieldValue(unsigned int gpuId, unsigned short fieldId, dcgmFieldValue_v2 &value,
                                      unsigned int flags);
    
    /*
     * Retrieves the latest values for the watched fields (added via AddWatches). 
     * @param flags - Set to 0 to get latest cached values from DCGM
     *              - Set to DCGM_FV_FLAG_LIVE_DATA to get live values from the driver
     *
     * @return:
     *
     * DR_SUCCESS     : on success
     * DR_COMM_ERROR  : if we couldn't get the information from DCGM
     */
    int GetLatestValuesForWatchedFields(unsigned int flags, std::vector<DcgmError> &errorList);

    /*
     * Translates the gpuId to the index it should be in the Json stats file
     * They are usually the same, but if we are running on non-conselwtive GPUs then they will not agree.
     *
     * @return:
     *
     * The corresponding index if found
     * The number of valid GPUs if not found
     */
    unsigned int GpuIdToJsonStatsIndex(unsigned int gpuId);

    /*
     * If gpu utilization is below 75% for this GPU, then return a note prompting the user to investigate.
     * Otherwise return an empty string
     */
    std::string GetGpuUtilizationNote(unsigned int gpuId, timelib64_t startTime, timelib64_t endTime);

    /*
     * Get the attributes for the specified device
     */
    dcgmReturn_t GetDeviceAttributes(unsigned int gpuId, dcgmDeviceAttributes_t &attributes);

private:
    std::vector<unsigned short> m_fieldIds;
    std::vector<unsigned int> m_gpuIds;
    dcgmFieldGrp_t m_fieldGroupId;
    std::map<unsigned int, std::map<std::string, std::vector<dcgmTimeseriesInfo_t> > > m_gpuData;
    std::map<std::string, std::map<std::string, std::vector<dcgmTimeseriesInfo_t> > > m_groupedData;
    std::map<std::string, std::map<std::string, std::string> > m_groupSingleData;

    DcgmHandle     m_dcgmHandle;
    DcgmGroup      m_dcgmGroup;
    DcgmSystem     m_dcgmSystem;

    DcgmValuesSinceHolder m_valuesHolder;

    DcgmMutex m_groupedDataMutex;
    DcgmMutex m_gpuDataMutex;
    DcgmMutex m_groupedSingleDataMutex;

    long long m_nextValuesSinceTs;

    /*
     * Helper method to get the watched fields as a string
     */
    std::string GetWatchedFieldsAsString(std::string &output, long long ts);

    /*
     * Helper method to get the watched fields as a json object
     */
    std::string GetWatchedFieldsAsJson(Json::Value &jv, long long ts);

    /*
     * Helper method to create a group in DCGM
     */
    std::string CreateGroup(const std::vector<unsigned int> &gpuIds, bool allGpus, const std::string &groupName);

    /*
     * Helper method to get an error string from a dcgmReturn_t
     */
    void GetErrorString(dcgmReturn_t ret, std::string &err);

    /*
     * Helper method for recording custom data that isn't tracked by DCGM
     * Threadsafe
     */
    void InsertLwstomData(unsigned int gpuId, const std::string &name, dcgmTimeseriesInfo_t &data);

    /*
     * Helper method for recording custom data that isn't tracked by DCGM
     * Threadsafe
     */
    void InsertLwstomData(const std::string &groupName, const std::string &name, dcgmTimeseriesInfo_t &data);

    /*
     * Helper method for adding the custom data to the json object
     */
    void AddLwstomData(Json::Value &jv);

    /*
     * Helper method to add the grouped data from m_groupedData to a json object
     */
    void AddGroupedDataToJson(Json::Value &jv);

    /*
     * Helper method to add the gpu data from m_gpuData to a json object
     */
    void AddGpuDataToJson(Json::Value &jv);

    /*
     * Helper method to add the non-timeseries data from m_groupSingleData to a json object
     */
    void AddNonTimeseriesDataToJson(Json::Value &jv);

    /*
     * Helper method for adding a vector of timeseries data to a json object
     */
    void AddLwstomTimeseriesVector(Json::Value &jv, std::vector<dcgmTimeseriesInfo_t> &vec);
};


#endif
