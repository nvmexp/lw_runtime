#pragma once

#include <lwca.h>
#include "gtest/gtest.h"
#include "utils/utils.h"
#include <vector>
#include <memory>
#include <ostream>
#include <functional>

#define EXPECT_LWOS(x) EXPECT_EQ(x, 0)
#define ASSERT_LWOS(x) ASSERT_EQ(x, 0)

#define EXPECT_DRV(x) EXPECT_EQ(x, LWDA_SUCCESS)
#define ASSERT_DRV(x) ASSERT_EQ(x, LWDA_SUCCESS)

#define GET_VERSIONED_EXPORT_TABLE(etbl, member) \
    std::unique_ptr<const LWetbl ## etbl> etbl; \
    ::lwca::getLwdaExportTable<LWetbl ## etbl>(&LW_ETID_ ## etbl, etbl, offsetof(LWetbl ## etbl, member) + sizeof(((LWetbl ## etbl *)0)->member));

#define EXPECT_VERSIONED_EXPORT_TABLE(etbl, member) \
    GET_VERSIONED_EXPORT_TABLE(etbl, member);       \
    if (etbl.get() == nullptr) GTEST_SUCCEED() << "Waiving, failed to find etbl " << #etbl;

#define EXPECT_LWDA_EXPORT_TABLE(etbl) EXPECT_LWDA_VERSIONED_EXPORT_TABLE(etbl, sizeof(LWetbl ## etbl))

/* GTest helper for printing error names */
void PrintTo(const LWresult& status, std::ostream *os);
/* Helper for LWuuid */
bool operator<(const LWuuid &rhs, const LWuuid& lhs);

namespace lwca
{

template<typename Etbl>
void getLwdaExportTable(const LWuuid *uuid, std::unique_ptr<const Etbl>& etbl,
                        size_t minimalVersion = sizeof(Etbl))
{
    const Etbl *orig_etbl = nullptr;
    LWresult status = lwGetExportTable((const void **)&orig_etbl, uuid);
    if (status == LWDA_SUCCESS || etbl->struct_size > minimalVersion) {
        Etbl *versioned_etbl = new Etbl;
        memcpy(versioned_etbl, orig_etbl, minimalVersion);
        memset((char *)versioned_etbl + minimalVersion, 0, sizeof(Etbl) - minimalVersion);
        etbl.reset(versioned_etbl);
    }
}

/* Device list functions */
typedef std::pair<int, int> DevicePair;
typedef std::function<bool(int)> DevicePredicate;
typedef std::vector<int> DeviceVector;
typedef std::vector<DevicePair> DevicePairVector;

DeviceVector allDevices(void);
DevicePairVector getPeerPairs(const DeviceVector& devs = allDevices());

DeviceVector uniqueDevicesByArch(const DeviceVector& devs = allDevices());

DeviceVector filterDevices(DevicePredicate pred, const DeviceVector& devs = allDevices());

/****************************
 * Common device predicates *
 ****************************/

#define ANDPredicate(a,b) std::bind(std::logical_and<bool>(),              \
                                    std::bind((a), std::placeholders::_1), \
                                    std::bind((b), std::placeholders::_1))
#define ORPredicate(a,b) std::bind(std::logical_or<bool>(),               \
                                   std::bind((a), std::placeholders::_1), \
                                   std::bind((b), std::placeholders::_1))
#define NOTPredicate(a) std::bind(std::logical_not<bool>(),              \
                                  std::bind((a), std::placeholders::_1))

bool isMpsDevice(int dev);
bool isLwdaIpcEnabledDevice(int dev);

/* Device attribute predicates */
bool hasDeviceAttributeEqualTo(int dev, LWdevice_attribute attr, int val);

template<LWdevice_attribute attr, int val>
bool hasDeviceAttributeEqualTo(int dev)
{
    return hasDeviceAttributeEqualTo(dev, attr, val);
}

/* Device attribute predicate aliases */
#define isCilpEnabledDevice                        hasDeviceAttributeEqualTo<LW_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, 1>
#define isMobileDevice                             hasDeviceAttributeEqualTo<LW_DEVICE_ATTRIBUTE_INTEGRATED, 1>
#define isCooperativeGroupEnabledDevice            hasDeviceAttributeEqualTo<LW_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, 1>
#define isCooperativeGroupMultiDeviceEnabledDevice hasDeviceAttributeEqualTo<LW_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH, 1>
#define isUvmFullEnabledDevice                     hasDeviceAttributeEqualTo<LW_DEVICE_ATTRIBUTE_CONLWRRENT_MANAGED_ACCESS, 1>
#define isAtsEnabledDevice                         hasDeviceAttributeEqualTo<LW_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, 1>
#define isUvmEnabledDevice                         hasDeviceAttributeEqualTo<LW_DEVICE_ATTRIBUTE_MANAGED_MEMORY, 1>

/* Compute mode predicates */
bool isComputeModeCompatibleDevice(int dev, unsigned min_major, unsigned min_minor,
                                   unsigned max_major, unsigned max_minor);

template<unsigned min_major, unsigned min_minor = 0, unsigned max_major = 999, unsigned max_minor = 999>
bool isComputeModeCompatibleDevice(int dev)
{
    return isComputeModeCompatibleDevice(dev, min_major, min_minor, max_major, max_minor);
}

/* Compute mode predicate aliases */
#define isKeplerPlusDevice  isComputeModeCompatibleDevice<3>
#define isMaxwellPlusDevice isComputeModeCompatibleDevice<5>
#define isPascalPlusDevice  isComputeModeCompatibleDevice<6>
#define isVoltaPlusDevice   isComputeModeCompatibleDevice<7>
#define isTuringPlusDevice  isComputeModeCompatibleDevice<7, 5>
#define isAmperePlusDevice  isComputeModeCompatibleDevice<8>

#define isCnpEnabledDevice  isKeplerPlusDevice
/******************************
 * LWCA utilities environment *
 ******************************/
class LwdaElwironment : public ::utils::BaseElwironment {
public:
    LwdaElwironment();
    ~LwdaElwironment();
    void SetUp() override;
    void TearDown() override;
    void parseArguments(int argc, char **argv) override;
    static const LwdaElwironment *getElw();
private:
    GTEST_DISALLOW_COPY_AND_ASSIGN_(LwdaElwironment);
};

/*******************************************
 * LWCA Specific result / output utilities *
 *******************************************/

class LwdaCleanupListener : public ::testing::EmptyTestEventListener
{
public:
    explicit LwdaCleanupListener();
    virtual ~LwdaCleanupListener();

    void OnTestEnd(const ::testing::TestInfo& test_info) override;

private:
    /* We disallow copying EventListeners */
    GTEST_DISALLOW_COPY_AND_ASSIGN_(LwdaCleanupListener);
};

}
