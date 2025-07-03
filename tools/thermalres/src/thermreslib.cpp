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

#include "thermreslib.h"
#include <cmath>
#include <assert.h>

#ifdef LW_MODS
#include "modsdrv.h"
void ThermResPrintf(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    ModsDrvVPrintf(PRI_NORMAL, format, args);
    va_end(args);
}
#endif

const std::array<uint64_t, 11> ThermalResLib::s_pow10 = {
                                                            1ULL,
                                                            10ULL,
                                                            100ULL,
                                                            1000ULL,
                                                            10000ULL,
                                                            100000ULL,
                                                            1000000ULL,
                                                            10000000ULL,
                                                            100000000ULL,
                                                            1000000000ULL,
                                                            10000000000ULL
                                                        };

uint32_t ThermalResLib::SearchRconfig(const RConfig& rConfig) const
{
    uint32_t index = ILWALID_THERMRES_VAL_UINT32;
    for (uint32_t i = 0; i < m_RConfigList.size(); i++)
    {
        if (m_RConfigList[i].pwrChName == rConfig.pwrChName &&
            m_RConfigList[i].tempChName == rConfig.tempChName)
        {
            index = i;
            break;
        }
    }
    return index;
}

uint32_t ThermalResLib::GetSensorInfoIdxByName
(
    const std::string& device,
    const std::string& chName
) const
{
    uint32_t index = ILWALID_THERMRES_VAL_UINT32;
    for (uint32_t i = 0; i < m_SensorInfoList.size(); i++)
    {
        if (m_SensorInfoList[i].device == device && m_SensorInfoList[i].chName == chName)
        {
            index = i;
            break;
        }
    }
    return index;
}

ThermalResLib::ThermResRC ThermalResLib::ResetMonitorPower(uint32_t monIdx)
{
    ThermResRC rc = ThermResRC::OK;
    assert(monIdx < m_MonitorList.size());
    uint32_t sensorIndex = m_MonitorList[monIdx].pwrSensorIndex;
    CHECK_THERMRES_RC(ValidatePowerSensorDataStat(m_SensorDataStatList[sensorIndex]));

    m_MonitorList[monIdx].lwmulativePower = 0.0f;
    m_MonitorList[monIdx].avgPower_mW = 0.0f;
    m_SensorDataStatList[sensorIndex].numSamples = 0;

    return rc;
}

ThermalResLib::ThermResRC ThermalResLib::ForceSetPowerSensorDataStatValue
(
    uint32_t sensorIdx,
    float filteredVal
)
{
    ThermResRC rc = ThermResRC::OK;
    assert(sensorIdx < m_SensorDataStatList.size());
    CHECK_THERMRES_RC(ValidatePowerSensorDataStat(m_SensorDataStatList[sensorIdx]));
    m_SensorDataStatList[sensorIdx].filteredValue = filteredVal;
    return rc;
}

ThermalResLib::ThermResRC ThermalResLib::CreateMonitor
(
    const std::string& device,
    const RConfig& rConfig,
    uint32_t *pId
)
{
    ThermResRC rc = ThermResRC::OK;
    uint32_t rIndex = SearchRconfig(rConfig);
    if (rIndex == ILWALID_THERMRES_VAL_UINT32)
    {
        return ThermResRC::RCONFIG_DOES_NOT_EXIST;
    }

    Monitor newMon(device);
    newMon.rIndex = rIndex;
    if (m_SensorDataStatMap[device].find(rConfig.pwrChName) == \
        m_SensorDataStatMap[device].end())
    {
        uint32_t pwrSensorInfoIdx = GetSensorInfoIdxByName(device, rConfig.pwrChName);
        assert(pwrSensorInfoIdx != ILWALID_THERMRES_VAL_UINT32);
        m_SensorDataStatList.emplace_back(pwrSensorInfoIdx);
        m_SensorDataStatMap[device][rConfig.pwrChName] = \
            static_cast<uint32_t>(m_SensorDataStatList.size() - 1);
    }

    if (m_SensorDataStatMap[device].find(rConfig.tempChName) == \
        m_SensorDataStatMap[device].end())
    {
        uint32_t tempSensorInfoIdx = GetSensorInfoIdxByName(device, rConfig.tempChName);
        assert(tempSensorInfoIdx != ILWALID_THERMRES_VAL_UINT32);
        m_SensorDataStatList.emplace_back(tempSensorInfoIdx);
        m_SensorDataStatMap[device][rConfig.tempChName] = \
            static_cast<uint32_t>(m_SensorDataStatList.size() - 1);
    }

    newMon.pwrSensorIndex = m_SensorDataStatMap[device][rConfig.pwrChName];
    newMon.tempSensorIndex = m_SensorDataStatMap[device][rConfig.tempChName];
    m_MonitorList.push_back(newMon);

    uint32_t monId = static_cast<uint32_t>(m_MonitorList.size() - 1);
    *pId = monId;

    m_SensorDataIdToMonitorIdListMap[newMon.pwrSensorIndex].push_back(monId);
    m_SensorDataIdToMonitorIdListMap[newMon.tempSensorIndex].push_back(monId);
    m_DeviceToMonitorIdListMap[device].push_back(monId);

    return rc;
}

ThermalResLib::ThermResRC ThermalResLib::ValidateSensorData(const SensorData& sensorData) const
{
    const SensorInfo& sensorInfo = m_SensorInfoList[sensorData.sensorInfoIdx];
    if (sensorInfo.unit != ThermalResLib::Unit::MILLIWATT &&
        sensorInfo.unit != ThermalResLib::Unit::DEGREE_C)
    {
        ThermResPrintf("Sensor data for channel %s has an invalid unit\n",
            sensorInfo.chName.c_str());
        return ThermResRC::BAD_PARAMETER;
    }
    return ThermResRC::OK;
}

ThermalResLib::ThermResRC ThermalResLib::ValidatePowerSensorDataStat
(
    const SensorDataStat& pwrSensorData
) const
{
    const SensorInfo& pwrSensorInfo = m_SensorInfoList[pwrSensorData.sensorInfoIdx];

    if (pwrSensorInfo.chType != ChannelType::POWER)
    {
        ThermResPrintf("Power sensor data for channel %s has an invalid channel type\n",
            pwrSensorInfo.chName.c_str());
        return ThermResRC::BAD_PARAMETER;
    }

    if (pwrSensorInfo.unit != Unit::MILLIWATT)
    {
        ThermResPrintf("Power sensor data for channel %s has an invalid unit %d\n",
            pwrSensorInfo.chName.c_str(),
            pwrSensorInfo.unit);
        return ThermResRC::BAD_PARAMETER;
    }

    return ThermResRC::OK;
}

ThermalResLib::ThermResRC ThermalResLib::ValidateTempSensorDataStat
(
    const SensorDataStat& tempSensorData
) const
{
    const SensorInfo& tempSensorInfo = m_SensorInfoList[tempSensorData.sensorInfoIdx];
    if (tempSensorInfo.chType != ChannelType::TEMP)
    {
        ThermResPrintf("Temperature sensor data for channel %s has an invalid channel type\n",
            tempSensorInfo.chName.c_str());
        return ThermResRC::BAD_PARAMETER;
    }

    if (tempSensorInfo.unit != Unit::DEGREE_C)
    {
        ThermResPrintf("Temperature sensor data for channel %s has an invalid unit %d\n",
            tempSensorInfo.chName.c_str(),
            tempSensorInfo.unit);
        return ThermResRC::BAD_PARAMETER;
    }

    return ThermResRC::OK;
}

ThermalResLib::ThermResRC ThermalResLib::StoreSensorData
(
    const SensorData& sensorData,
    uint32_t* pId
)
{
    ThermResRC rc = ThermResRC::OK;
    assert(pId);
    CHECK_THERMRES_RC(ValidateSensorData(sensorData));
    const SensorInfo& sensorInfo = m_SensorInfoList[sensorData.sensorInfoIdx];

    const std::string& device = sensorInfo.device;
    const std::string& chName = sensorInfo.chName;

    uint32_t sensorIndex;
    if (m_SensorDataStatMap[device].find(chName) == m_SensorDataStatMap[device].end())
    {
        m_SensorDataStatList.emplace_back(sensorData.sensorInfoIdx);
        sensorIndex = static_cast<uint32_t>(m_SensorDataStatList.size() - 1);
        m_SensorDataStatMap[device][chName] = sensorIndex;
    }
    else
    {
        sensorIndex = m_SensorDataStatMap[device][chName];
    }

    *pId = sensorIndex;
    m_SensorDataStatList[sensorIndex].value = sensorData.value;
    m_SensorDataStatList[sensorIndex].timestamp = sensorData.timestamp;
    m_SensorDataStatList[sensorIndex].filteredValue += \
        sensorInfo.iir * (sensorData.value - m_SensorDataStatList[sensorIndex].filteredValue);
    m_SensorDataStatList[sensorIndex].numSamples++;

    if (sensorInfo.chType == ChannelType::POWER)
    {
        for (const auto& monIndex : m_SensorDataIdToMonitorIdListMap[sensorIndex])
        {
            m_MonitorList[monIndex].lwmulativePower += sensorData.value;
            m_MonitorList[monIndex].avgPower_mW = m_MonitorList[monIndex].lwmulativePower / \
                m_SensorDataStatList[sensorIndex].numSamples;
        }
    }

    return rc;
}

ThermalResLib::ThermResRC ThermalResLib::UpdateMonitorBaselineValue
(
    const std::string& device
)
{
    ThermResRC rc = ThermResRC::OK;
    for (const auto& monIndex : m_DeviceToMonitorIdListMap[device])
    {
        Monitor& lwrrMonitor = m_MonitorList[monIndex];
        uint32_t pwrSensorIndex = lwrrMonitor.pwrSensorIndex;
        uint32_t tempSensorIndex = lwrrMonitor.tempSensorIndex;
        assert(pwrSensorIndex < m_SensorDataStatList.size());
        assert(tempSensorIndex < m_SensorDataStatList.size());
        CHECK_THERMRES_RC(ValidatePowerSensorDataStat(m_SensorDataStatList[pwrSensorIndex]));
        CHECK_THERMRES_RC(ValidateTempSensorDataStat(m_SensorDataStatList[tempSensorIndex]));

        lwrrMonitor.baseTemp_degC = m_SensorDataStatList[tempSensorIndex].filteredValue;
        lwrrMonitor.basePower_mW = m_SensorDataStatList[pwrSensorIndex].filteredValue;
    }

    return rc;
}

ThermalResLib::ThermResRC ThermalResLib::ComputeRValue
(
    const std::string& device
)
{
    ThermResRC rc = ThermResRC::OK;
    for (const auto& monIndex : m_DeviceToMonitorIdListMap[device])
    {
        Monitor& lwrrMonitor = m_MonitorList[monIndex];
        uint32_t pwrSensorIndex = lwrrMonitor.pwrSensorIndex;
        uint32_t tempSensorIndex = lwrrMonitor.tempSensorIndex;
        assert(pwrSensorIndex < m_SensorDataStatList.size());
        assert(tempSensorIndex < m_SensorDataStatList.size());
        CHECK_THERMRES_RC(ValidatePowerSensorDataStat(m_SensorDataStatList[pwrSensorIndex]));
        CHECK_THERMRES_RC(ValidateTempSensorDataStat(m_SensorDataStatList[tempSensorIndex]));

        if (lwrrMonitor.baseTemp_degC != ILWALID_THERMRES_TEMP_DEGREE &&
            lwrrMonitor.basePower_mW != ILWALID_THERMRES_POWER_MW)
        {
            lwrrMonitor.r =
                (m_SensorDataStatList[tempSensorIndex].value - lwrrMonitor.baseTemp_degC) / \
                (m_SensorDataStatList[pwrSensorIndex].value - lwrrMonitor.basePower_mW);
            lwrrMonitor.filteredR =
                (m_SensorDataStatList[tempSensorIndex].filteredValue - lwrrMonitor.baseTemp_degC) / \
                (m_SensorDataStatList[pwrSensorIndex].filteredValue - lwrrMonitor.basePower_mW);
        }
    }
    return rc;
}

const ThermalResLib::SensorDataStat& ThermalResLib::GetSensorDataStat
(
    uint32_t id
) const
{
    assert(id < m_SensorDataStatList.size());
    return m_SensorDataStatList[id];
}

const ThermalResLib::Monitor& ThermalResLib::GetMonitor
(
    uint32_t id
) const
{
    assert(id < m_MonitorList.size());
    return m_MonitorList[id];
}

const std::vector<ThermalResLib::SensorDataStat>& ThermalResLib::GetAllSensorDataStat() const
{
    return m_SensorDataStatList;
}

const std::vector<ThermalResLib::Monitor>& ThermalResLib::GetAllMonitor() const
{
    return m_MonitorList;
}

ThermalResLib::ThermResRC ThermalResLib::ValidateRAndPowerLimits
(
    const std::string& device,
    uint64_t lwrrTimeStamp,
    bool*    isPeformChecked
)
{
    ThermResRC rc = ThermResRC::OK;
    ThermResRC firstRc = ThermResRC::OK;
    *isPeformChecked = false;
    for (const auto& monIndex : m_DeviceToMonitorIdListMap[device])
    {
        Monitor &lwrrMonitor = m_MonitorList[monIndex];
        const RConfig &rConfig = m_RConfigList[lwrrMonitor.rIndex];

        if (rConfig.checkTimeMs.size() > 0 &&
            lwrrMonitor.checkTimeIndex < rConfig.checkTimeMs.size() &&
            lwrrTimeStamp >= rConfig.checkTimeMs[lwrrMonitor.checkTimeIndex])
        {
            if (rConfig.pwrLimits.size() > 0)
            {
                uint32_t minPwr = rConfig.pwrLimits[lwrrMonitor.checkTimeIndex].first;
                uint32_t maxPwr = rConfig.pwrLimits[lwrrMonitor.checkTimeIndex].second;
                assert(lwrrMonitor.pwrSensorIndex < m_SensorDataStatList.size());
                const SensorDataStat& pwrSensor = m_SensorDataStatList[lwrrMonitor.pwrSensorIndex];
                CHECK_THERMRES_RC(ValidatePowerSensorDataStat(pwrSensor));
                const SensorInfo& pwrSensorInfo = m_SensorInfoList[pwrSensor.sensorInfoIdx];

                uint32_t filteredPwr = static_cast<uint32_t>(round(pwrSensor.filteredValue));
                if (minPwr != ILWALID_THERMRES_VAL_UINT32 && filteredPwr < minPwr)
                {
                    ThermResPrintf("Monitor %u: %s power (%umW) is lower than allowed (%umW)\n",
                        monIndex,
                        pwrSensorInfo.chName.c_str(),
                        filteredPwr,
                        minPwr);
                    
                    THERMRES_FIRST_RC(ThermResRC::POWER_TOO_LOW);
                }

                if (maxPwr != ILWALID_THERMRES_VAL_UINT32 && filteredPwr > maxPwr)
                {
                    ThermResPrintf("Monitor %u: %s power (%umW) is higher than allowed (%umW)\n",
                        monIndex,
                        pwrSensorInfo.chName.c_str(),
                        filteredPwr,
                        maxPwr);
    
                    THERMRES_FIRST_RC(ThermResRC::POWER_TOO_HIGH);
                }
            }

            float minR = rConfig.rLimits[lwrrMonitor.checkTimeIndex].first;
            float maxR = rConfig.rLimits[lwrrMonitor.checkTimeIndex].second;
            uint64_t minREpsilon = ThermalResLib::CallwlateEpsilon(minR);
            uint64_t maxREpsilon = ThermalResLib::CallwlateEpsilon(maxR);

            assert(lwrrMonitor.filteredR != ILWALID_THERMRES_R_VALUE);
            if (minR != ILWALID_THERMRES_R_VALUE &&
                static_cast<uint64_t>(minR * s_pow10[10]) >= \
                    minREpsilon + static_cast<uint64_t>(lwrrMonitor.filteredR * s_pow10[10]))
            {
                ThermResPrintf("Monitor %u (%s, %s) R value (%.3e) is lower than allowed (%.3e)\n",
                    monIndex,
                    rConfig.tempChName.c_str(),
                    rConfig.pwrChName.c_str(),
                    lwrrMonitor.filteredR,
                    minR);
                
                THERMRES_FIRST_RC(ThermResRC::THERMAL_RESISTANCE_TOO_LOW);
            }

            if (maxR != ILWALID_THERMRES_R_VALUE &&
                static_cast<uint64_t>(lwrrMonitor.filteredR * s_pow10[10]) >= \
                    maxREpsilon + static_cast<uint64_t>(maxR * s_pow10[10]))
            {
                ThermResPrintf("Monitor %u (%s, %s) R value (%.3e) is higher than allowed (%.3e)\n",
                    monIndex,
                    rConfig.tempChName.c_str(),
                    rConfig.pwrChName.c_str(),
                    lwrrMonitor.filteredR,
                    maxR);
                
                THERMRES_FIRST_RC(ThermResRC::THERMAL_RESISTANCE_TOO_HIGH);
            }
            lwrrMonitor.checkTimeIndex++;
            *isPeformChecked = true;
        }
    }

    return firstRc;
}

//should only be called at the end of the idle phase
ThermalResLib::ThermResRC ThermalResLib::ValidateIdleLimits(const std::string& device)
{
    ThermResRC rc = ThermResRC::OK;
    ThermResRC firstRc = ThermResRC::OK;

    MonitorIdList monIndexList = m_DeviceToMonitorIdListMap[device];
    ThermResPrintf("Verifying average power and temperature readings at idle\n");

    for (const auto& monIndex : monIndexList)
    {
        const Monitor& lwrrMonitor = m_MonitorList[monIndex];

        assert(lwrrMonitor.pwrSensorIndex < m_SensorDataStatList.size());
        const SensorDataStat& pwrSensorDataStat = m_SensorDataStatList[lwrrMonitor.pwrSensorIndex];
        CHECK_THERMRES_RC(ValidatePowerSensorDataStat(pwrSensorDataStat));
        const SensorInfo& pwrSensorInfo = m_SensorInfoList[pwrSensorDataStat.sensorInfoIdx];

        uint32_t maxIdlePwr = pwrSensorInfo.maxIdleValue;
        float idlePower_mW = pwrSensorDataStat.filteredValue;
        if (idlePower_mW == 0)
        {
            ThermResPrintf("Unable to get valid idle power value\n");
            return ThermResRC::ILWALID_POWER_VALUE;
        }
        if (maxIdlePwr != ILWALID_THERMRES_VAL_UINT32 &&
            idlePower_mW > static_cast<float>(maxIdlePwr))
        {
            ThermResPrintf("Device %s : %s power is too high at idle (observed=%.2fmW, maxAllowed=%umW)\n",
                device.c_str(), pwrSensorInfo.chName.c_str(), idlePower_mW, maxIdlePwr);
            
            THERMRES_FIRST_RC(ThermResRC::POWER_TOO_HIGH);
        }

        assert(lwrrMonitor.tempSensorIndex < m_SensorDataStatList.size());
        const SensorDataStat& tempSensorDataStat = \
            m_SensorDataStatList[lwrrMonitor.tempSensorIndex];
        CHECK_THERMRES_RC(ValidateTempSensorDataStat(tempSensorDataStat));
        const SensorInfo& tempSensorInfo = m_SensorInfoList[tempSensorDataStat.sensorInfoIdx];
        uint32_t maxIdleTemp = tempSensorInfo.maxIdleValue;
        float idleTemp_degC = tempSensorDataStat.filteredValue;
        if (idleTemp_degC == 0)
        {
            ThermResPrintf("Unable to get valid idle temp value\n");
            return ThermResRC::ILWALID_TEMP_VALUE;
        }
        if (maxIdleTemp != ILWALID_THERMRES_VAL_UINT32 &&
            idleTemp_degC - static_cast<float>(maxIdleTemp) > 0.01f)
        {
            ThermResPrintf("Device %s : %s is too hot at idle (observed=%.2fdegC, maxAllowed=%.2fdegC)\n",
                device.c_str(),
                tempSensorInfo.chName.c_str(),
                idleTemp_degC,
                static_cast<float>(maxIdleTemp));

            THERMRES_FIRST_RC(ThermResRC::TEMP_TOO_HIGH);
        }
    }

    return firstRc;
}

// We report R values using scientific notation with four significant digits
// (e.g. 1.234e-05). When we compare the measured R values to MinR and MaxR,
// we want to do so only using the first four digits. We do not want a scenario
// where MaxR and FilteredR are both reported as 1.234e-05 but we fail the
// check because FilteredR is actually 1.2349e-05. In this example, we want
// FilteredR to be at least 0.001e-05 (or 1e-8) greater than MaxR. This
// threshold value is referred to as "epsilon".
// This function returns the "epsilon" value using integer math based on
// the fact that the range of R Limit is [1e-6, 1e-3].
uint64_t ThermalResLib::CallwlateEpsilon(float r)
{
    uint64_t colwertedEpsilon = 1;
    uint64_t colwertedR = static_cast<uint64_t>(r * s_pow10[10]);
    if (colwertedR >= s_pow10[6])
    {
        colwertedEpsilon = s_pow10[3];
    }
    else if (colwertedR >= s_pow10[5])
    {
        colwertedEpsilon = s_pow10[2];
    }
    else if (colwertedR >= s_pow10[4])
    {
        colwertedEpsilon = s_pow10[1];
    }
    else
    {
        colwertedEpsilon = s_pow10[0];
    }

    return colwertedEpsilon;

}

