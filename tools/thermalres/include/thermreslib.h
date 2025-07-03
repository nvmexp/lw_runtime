/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2021-2022 by LWPU Corporation. All rights reserved.
 * All information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#pragma once

#include <vector>
#include <map>
#include <array>
#include <string>
#include <limits>
#include <cstdint>

#define ILWALID_THERMRES_VAL_UINT32    UINT32_MAX
#define ILWALID_THERMRES_VAL_FLOAT     std::numeric_limits<float>::max()
#define ILWALID_THERMRES_TEMP_DEGREE   (-1.0)
#define ILWALID_THERMRES_POWER_MW      (-1.0)
#define ILWALID_THERMRES_R_VALUE       (-1.0)
#define ILWALID_THERMRES_CHANNEL_NAME  "ILWALID_CHANNEL_NAME"

#define CHECK_THERMRES_RC(f)                                 \
    do                                                       \
    {                                                        \
        if (ThermalResLib::ThermResRC::OK != (rc = f))       \
            return rc;                                       \
    } while (0)                                              \

#define THERMRES_FIRST_RC(f)                                 \
    do {                                                     \
         rc = (f);                                           \
         if (ThermalResLib::ThermResRC::OK == firstRc)       \
            firstRc = rc;                                    \
     } while (0)


#ifdef LW_MODS
void ThermResPrintf(const char* format, ...);
#else
#define ThermResPrintf printf
#endif

using ChannelNameToSensorDataIdMap = std::map<std::string, uint32_t>;
using MonitorIdList = std::vector<uint32_t>;

class ThermalResLib
{
public:
    enum ChannelType
    {
        POWER,
        TEMP
    };

    enum DeviceType
    {
        GPU,
        CPU,
        LWSWITCH,
        SOC
    };

    enum Unit
    {
        MILLIWATT,
        DEGREE_C
    };

    struct SensorData
    {
        uint32_t    sensorInfoIdx;
        float       value;
        uint64_t    timestamp;
    };

    struct SensorInfo
    {
        std::string device;
        std::string chName;
        DeviceType  deviceType;
        ChannelType chType;
        float       iir;
        uint32_t    maxIdleValue;
        Unit        unit;

        SensorInfo(const std::string& device,
            const std::string& chName,
            DeviceType  deviceType,
            ChannelType chType,
            float iir,
            Unit unit) :
            device(device)
            ,chName(chName)
            ,deviceType(deviceType)
            ,chType(chType)
            ,iir(iir)
            ,maxIdleValue(ILWALID_THERMRES_VAL_UINT32)
            ,unit(unit) {}
    };

    struct RConfig
    {
        std::string pwrChName;
        std::string tempChName;
        std::vector<std::pair<uint32_t, uint32_t>> pwrLimits;
        std::vector<std::pair<float, float>> rLimits;
        std::vector<uint64_t> checkTimeMs;
    };

    ThermalResLib(const std::vector<RConfig>& rConfigList,
        const std::vector<SensorInfo>& sensorInfoList) :
        m_RConfigList(rConfigList)
        ,m_SensorInfoList(sensorInfoList) {};
    ~ThermalResLib() = default;

    struct SensorDataStat
    {
        uint32_t    sensorInfoIdx;
        // The most recent sample
        float       value;
        // The most recent sample after an IIR filter has been applied
        float       filteredValue;
        uint32_t    numSamples;
        uint64_t    timestamp;

        explicit SensorDataStat(uint32_t sensorInfoIdx) :
            sensorInfoIdx(sensorInfoIdx)
            ,value(ILWALID_THERMRES_VAL_FLOAT)
            ,filteredValue(0.0)
            ,numSamples(0)
            ,timestamp(0) {}
    };

    struct Monitor
    {
        std::string device;
        uint32_t    rIndex;
        uint32_t    pwrSensorIndex;
        uint32_t    tempSensorIndex;
        uint32_t    checkTimeIndex;
        // The average temperature/power as baseline
        float       baseTemp_degC;
        float       basePower_mW;
        float       r;
        float       filteredR;
        float       lwmulativePower;
        float       avgPower_mW;

        explicit Monitor(const std::string& device) :
            device(device)
            ,rIndex(0)
            ,pwrSensorIndex(0)
            ,tempSensorIndex(0)
            ,checkTimeIndex(0)
            ,baseTemp_degC(ILWALID_THERMRES_TEMP_DEGREE)
            ,basePower_mW(ILWALID_THERMRES_POWER_MW)
            ,r(ILWALID_THERMRES_R_VALUE)
            ,filteredR(ILWALID_THERMRES_R_VALUE)
            ,lwmulativePower(0.0)
            ,avgPower_mW(0.0) {}
    };

    enum ThermResRC
    {
        OK = 0,
        POWER_TOO_LOW,
        POWER_TOO_HIGH,
        TEMP_TOO_LOW,
        TEMP_TOO_HIGH,
        THERMAL_RESISTANCE_TOO_LOW,
        THERMAL_RESISTANCE_TOO_HIGH,
        BAD_PARAMETER,
        RCONFIG_DOES_NOT_EXIST,
        ILWALID_TEMP_VALUE,
        ILWALID_POWER_VALUE
    };

    ThermResRC StoreSensorData(const SensorData &sensorData, uint32_t *pId);
    ThermResRC CreateMonitor(const std::string& device, const RConfig& rConfig, uint32_t *pId);
    const SensorDataStat& GetSensorDataStat(uint32_t id) const;
    const Monitor& GetMonitor(uint32_t id) const;
    const std::vector<SensorDataStat>& GetAllSensorDataStat() const;
    const std::vector<Monitor>& GetAllMonitor() const;
    ThermResRC ValidateRAndPowerLimits(const std::string& device, 
        uint64_t lwrrTimeStamp, 
        bool* isPeformChecked);
    ThermResRC ValidateIdleLimits(const std::string& device);
    ThermResRC ComputeRValue(const std::string& device);
    ThermResRC UpdateMonitorBaselineValue(const std::string& device);
    ThermResRC ResetMonitorPower(uint32_t monIdx);
    ThermResRC ForceSetPowerSensorDataStatValue(uint32_t sensorIdx, float filteredVal);

private:
    const std::vector<RConfig>&                         m_RConfigList;
    const std::vector<SensorInfo>&                      m_SensorInfoList;
    std::vector<SensorDataStat>                         m_SensorDataStatList;
    std::vector<Monitor>                                m_MonitorList;
    //Mapping of (device name, channel name) to sensor data ID
    std::map<std::string, ChannelNameToSensorDataIdMap> m_SensorDataStatMap;
    //Mapping of device name to monitor ID list
    std::map<std::string, MonitorIdList>                m_DeviceToMonitorIdListMap;
    //Mapping of sensor data ID to monitor ID list
    std::map<uint32_t, MonitorIdList>                   m_SensorDataIdToMonitorIdListMap;
    const static std::array<uint64_t, 11>               s_pow10;

    uint32_t        SearchRconfig(const RConfig& rConfig) const;
    uint32_t        GetSensorInfoIdxByName(const std::string& device, const std::string& chName) const;
    ThermResRC      ValidateSensorData(const SensorData& sensorData) const;
    ThermResRC      ValidatePowerSensorDataStat(const SensorDataStat& pwrSensorData) const;
    ThermResRC      ValidateTempSensorDataStat(const SensorDataStat& tempSensorData) const;
    static uint64_t CallwlateEpsilon(float r);
};
