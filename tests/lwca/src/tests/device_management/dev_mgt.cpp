#include <lwos_platform.h>
#include "gtest/gtest.h"
#include "lwda_utils/lwda_utils.h"

#include <set>

namespace
{

TEST(DeviceManagement, DeviceCount)
{
    int count = 0;
    // Negative tests
    EXPECT_EQ(lwDeviceGetCount(nullptr), LWDA_ERROR_ILWALID_VALUE) << "Null check";

    EXPECT_DRV(lwDeviceGetCount(&count));
    EXPECT_GT(count, 0) << "Successful query returned incorrect device count";
}

TEST(DeviceManagement, DeviceGet)
{
    LWdevice dev = LW_DEVICE_ILWALID;
    int count = 0;
    ASSERT_DRV(lwDeviceGetCount(&count));

    // Negative tests
    EXPECT_EQ(lwDeviceGet(nullptr, 0), LWDA_ERROR_ILWALID_VALUE) << "Null check";
    EXPECT_EQ(lwDeviceGet(&dev, count), LWDA_ERROR_ILWALID_DEVICE) << "Invalid ordinal";
    EXPECT_EQ(lwDeviceGet(&dev, -1), LWDA_ERROR_ILWALID_DEVICE) << "Invalid ordinal";
    EXPECT_EQ(lwDeviceGet(&dev, LW_DEVICE_ILWALID), LWDA_ERROR_ILWALID_DEVICE) << "Invalid ordinal";

    // Unique-ness test
    std::set<LWdevice> devices;
    for (int i = 0; i < count; i++) {
        SCOPED_TRACE_STREAM("Device " << i);

        EXPECT_DRV(lwDeviceGet(&dev, i)) << "Failed to get a lwdevice for device " << i;
        devices.insert(dev);
    }
    EXPECT_EQ(devices.size(), count) << "Some returned devices are not unique";
}

TEST(DeviceManagement, DeviceGetName)
{
    int count = 0;
    char name[1024] = {0};
    LWdevice dev = LW_DEVICE_ILWALID;

    ASSERT_DRV(lwDeviceGet(&dev, 0));
    ASSERT_DRV(lwDeviceGetCount(&count));

    // Negative tests
    EXPECT_EQ(lwDeviceGetName(nullptr, sizeof(name), dev), LWDA_ERROR_ILWALID_VALUE) << "Null check";
    EXPECT_EQ(lwDeviceGetName(name, sizeof(name), count), LWDA_ERROR_ILWALID_DEVICE) << "Invalid device ordinal";
    EXPECT_EQ(lwDeviceGetName(name, sizeof(name), -1), LWDA_ERROR_ILWALID_DEVICE) << "Invalid device ordinal";
    EXPECT_EQ(lwDeviceGetName(name, sizeof(name), LW_DEVICE_ILWALID), LWDA_ERROR_ILWALID_DEVICE) << "Invalid device ordinal";

    // Name checking
    for (int i = 0; i < count; i++) {
        SCOPED_TRACE_STREAM("Device " << i);

        ASSERT_DRV(lwDeviceGet(&dev, i));
        EXPECT_DRV(lwDeviceGetName(name, sizeof(name), dev));
        size_t sz = strnlen(name, sizeof(name));
        EXPECT_NE(sz, 0);
        EXPECT_LE(sz, sizeof(name));
        for (size_t j = 0; j < sz; j++) {
            EXPECT_PRED1(isascii, name[i]) << "Character " << j << " is not ascii";
        }
    }
}

TEST(DeviceManagement, DeviceGetUuid)
{
    LWuuid uuid;
    LWdevice dev = LW_DEVICE_ILWALID;
    int count = 0;
    ASSERT_DRV(lwDeviceGet(&dev, 0));

    // Negative tests
    EXPECT_EQ(lwDeviceGetUuid(nullptr, dev), LWDA_ERROR_ILWALID_VALUE) << "Null check";
    EXPECT_EQ(lwDeviceGetUuid(&uuid, LW_DEVICE_ILWALID), LWDA_ERROR_ILWALID_DEVICE) << "Invalid device";

    // Unique-ness test
    std::set<LWuuid> uuids;
    ASSERT_DRV(lwDeviceGetCount(&count));
    for (int i = 0; i < count; i++) {
        SCOPED_TRACE_STREAM("Device " << i);

        EXPECT_DRV(lwDeviceGet(&dev, i));
        EXPECT_DRV(lwDeviceGetUuid(&uuid, dev));
        uuids.insert(uuid);
    }
    EXPECT_EQ(uuids.size(), count) << "Some returned uuids are not unique";
}

TEST(DeviceManagement, DeviceTotalMem)
{
    size_t bytes = 0;
    LWdevice dev = LW_DEVICE_ILWALID;
    int count = 0;

    ASSERT_DRV(lwDeviceGetCount(&count));
    ASSERT_DRV(lwDeviceGet(&dev, 0));
    
    // Negative tests
    EXPECT_EQ(lwDeviceTotalMem(nullptr, dev), LWDA_ERROR_ILWALID_VALUE) << "Null check";
    EXPECT_EQ(lwDeviceTotalMem(&bytes, LW_DEVICE_ILWALID), LWDA_ERROR_ILWALID_DEVICE) << "Invalid device";

    for (int i = 0; i < count; i++) {
        SCOPED_TRACE_STREAM("Device " << i);
        EXPECT_DRV(lwDeviceTotalMem(&bytes, dev));
        EXPECT_NE(bytes, 0);
    }
}

TEST(DeviceManagement, DeviceGetAttribute)
{
    int prop = 0;
    LWdevice dev = LW_DEVICE_ILWALID;
    int devCount = 0;

    ASSERT_DRV(lwDeviceGetCount(&devCount));
    ASSERT_DRV(lwDeviceGet(&dev, 0));

    // Negative tests
    EXPECT_EQ(lwDeviceGetAttribute(nullptr, LW_DEVICE_ATTRIBUTE_WARP_SIZE, dev), LWDA_ERROR_ILWALID_VALUE);
    EXPECT_EQ(lwDeviceGetAttribute(&prop, (LWdevice_attribute)0, dev), LWDA_ERROR_ILWALID_VALUE);
    EXPECT_EQ(lwDeviceGetAttribute(&prop, LW_DEVICE_ATTRIBUTE_WARP_SIZE, LW_DEVICE_ILWALID), LWDA_ERROR_ILWALID_DEVICE);

    // Good path tests - we can't actually validate the values here without
    // internal knowledge of each device and platform, so just validate that
    // something is returned successfully for all attributes on all devices
    for (int d = 0; d < devCount; d++) {
        SCOPED_TRACE_STREAM("Device " << d);
        ASSERT_DRV(lwDeviceGet(&dev, d));
        for (size_t j = (size_t)LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK; j < (size_t)LW_DEVICE_ATTRIBUTE_MAX; j++) {
#if !LWCFG(GLOBAL_ARCH_HOPPER)
            // Skip LW_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH. This attribute is not defined in lwca.h when LWCFG(GLOBAL_ARCH_HOPPER) is not enabled.
            if (j == 120 ) {
                continue;
            }
#endif
            SCOPED_TRACE_STREAM("Attribute " << j);
            EXPECT_DRV(lwDeviceGetAttribute(&prop, static_cast<LWdevice_attribute>(j), dev));
        }
    }
}

}
