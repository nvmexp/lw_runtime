#include <lwos_platform.h>

#include "lwda_utils/lwda_utils.h"
#include <string.h>

#define LW_INIT_UUID
#include <lwda_etbl/unittest.h>
#include <lwda_etbl/tools_device.h>
#include <lwda_etbl/tools_driver.h>
#undef LW_INIT_UUID

void PrintTo(const LWresult& status, std::ostream *os)
{
    const char *name;
    lwGetErrorName(status, &name);
    *os << (name == nullptr ? "<unknown>" : name) << " (" << static_cast<unsigned long>(status) << ')';
}

bool operator<(const LWuuid &rhs, const LWuuid& lhs)
{
    return memcmp(&rhs, &lhs, sizeof(rhs)) > 0;
}

namespace lwca
{

static LwdaElwironment *lwrrent_elw = nullptr;

/*********************************
 * Device filter implementations *
 *********************************/

bool isMpsDevice(int)
{
    GET_VERSIONED_EXPORT_TABLE(UnitTest, IsMpsClient);
    return !!UnitTest && !!UnitTest->IsMpsClient();
}

bool isLwdaIpcEnabledDevice(int dev)
{
    LWdevice lwdev;
    LWtools_driver_type driverType;

    GET_VERSIONED_EXPORT_TABLE(ToolsDevice, DeviceGetDriverType);
    if (!!ToolsDevice) {
        return false;
    }

    if (lwDeviceGet(&lwdev, dev) != LWDA_SUCCESS) {
        return false;
    }
    if (ToolsDevice->DeviceGetDriverType(&driverType, lwdev) != LWDA_SUCCESS) {
        return false;
    }

    switch (driverType) {
    case LW_TOOLS_DRIVER_TYPE_RM:
    case LW_TOOLS_DRIVER_TYPE_MPS:
        // LwdaIPC is only enabled on RM & MPS, and only if uva is enabled.
        return hasDeviceAttributeEqualTo(dev, LW_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, 1);
    case LW_TOOLS_DRIVER_TYPE_WDDM:
    case LW_TOOLS_DRIVER_TYPE_AMODEL:
    case LW_TOOLS_DRIVER_TYPE_MRM:
    default:
        break;
    }

    return false;
}

bool hasDeviceAttributeEqualTo(int dev, LWdevice_attribute attr, int val)
{
    LWdevice lwdev;
    int attr_val = 0;
    if (lwDeviceGet(&lwdev, dev) != LWDA_SUCCESS) {
        return false;
    }
    if (lwDeviceGetAttribute(&attr_val, attr, lwdev) != LWDA_SUCCESS) {
        return false;
    }
    return attr_val == val;
}

bool isComputeModeCompatibleDevice(int dev, unsigned min_major, unsigned min_minor,
                                   unsigned max_major, unsigned max_minor)
{
    int major = 0, minor = 0;
    LWdevice lwdev;

    if (lwDeviceGet(&lwdev, dev) != LWDA_SUCCESS) {
        return false;
    }
    if (lwDeviceGetAttribute(&major, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                             lwdev) != LWDA_SUCCESS) {
        return false;
    }
    if (lwDeviceGetAttribute(&minor, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                             lwdev) != LWDA_SUCCESS) {
        return false;
    }

    return (min_major * 100 + min_minor) <= static_cast<unsigned>(major * 100 + minor)
           && (max_major * 100 + max_minor) >= static_cast<unsigned>(major * 100 + minor);
}

DeviceVector allDevices()
{
    int count;
    LWresult status = lwDeviceGetCount(&count);
    if (status != LWDA_SUCCESS || count == 0) {
        return DeviceVector();
    }
    DeviceVector devs(count);
    for (int i = 0; i < count; i++) {
        devs[i] = i;
    }
    return devs;
}

DevicePairVector
getPeerPairs(const DeviceVector& devs)
{
    DevicePairVector ret;
    LWdevice devA, devB;
    for (size_t i = 0; i < devs.size(); i++) {
        lwDeviceGet(&devA, devs[i]);
        for (size_t j = i; j < devs.size(); j++) {
            int accessAB = 0, accessBA = 0;
            lwDeviceGet(&devB, devs[j]);
            lwDeviceGetP2PAttribute(&accessAB, LW_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED, devA, devB);
            lwDeviceGetP2PAttribute(&accessBA, LW_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED, devB, devA);
            if (accessAB && accessBA) {
                ret.push_back(std::make_pair(devA, devB));
            }
        }
    }
    return ret;
}

DeviceVector uniqueDevicesByArch(const DeviceVector& devs)
{
    DeviceVector v;
    LWdevice devA, devB;
    int majorA = 0, minorA = 0, majorB = 0, minorB = 0;
    for (size_t i = 0; i < devs.size(); i++) {
        bool found = false;
        lwDeviceGet(&devA, devs[i]);
        lwDeviceGetAttribute(&majorA, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, devA);
        lwDeviceGetAttribute(&minorA, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, devA);
        for (size_t j = 0; j < v.size(); j++) {
            lwDeviceGet(&devB, v[j]);
            lwDeviceGetAttribute(&majorB, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, devB);
            lwDeviceGetAttribute(&minorB, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, devB);
            if ((majorA * 100 + minorA) == (majorB * 100 + minorB)) {
                found = true;
                break;
            }
        }
        if (!found) {
            v.push_back(devs[i]);
        }
    }

    return v;
}

DeviceVector filterDevices(DevicePredicate pred, const DeviceVector& input)
{
    DeviceVector ret(input);
    ret.erase(std::remove_if(ret.begin(), ret.end(), NOTPredicate(pred)), ret.end());
    return ret;
}

/********************
 * LWCA Environment *
 ********************/

LwdaElwironment::LwdaElwironment()
{
    lwrrent_elw = this;
}

LwdaElwironment::~LwdaElwironment()
{
}

void LwdaElwironment::SetUp()
{
    // Initialize the driver for the rest of the tests
    ASSERT_DRV(lwInit(0)) << "Driver failed to initialize.\n"
                          << "\tIf you are not a LWCA driver engineer, please verify your system configuration is correct";
    // TODO : Conditionally retain all the primary device contexts so all the
    // tests can use them
}

void LwdaElwironment::TearDown()
{
    // TODO : Potentially call lwiFinalize() here to verify leak handling, etc
}

void LwdaElwironment::parseArguments(int argc, char **argv)
{
}

const LwdaElwironment * LwdaElwironment::getElw()
{
    return lwrrent_elw;
}

LwdaCleanupListener::LwdaCleanupListener()
{

}

LwdaCleanupListener::~LwdaCleanupListener()
{

}

// Helper callback for LwdaCleanupListener::OnTestEnd to collect all the current contexts
static void LWDAAPI enumerateContextsCallback(void *pUserData, LWtoolsEnumContextCallbackData *contextData)
{
    std::vector<LWcontext> *ctxs = static_cast<std::vector<LWcontext> *>(pUserData);
    ctxs->push_back(contextData->ctx);
}

void LwdaCleanupListener::OnTestEnd(const ::testing::TestInfo& test_info)
{
    GET_VERSIONED_EXPORT_TABLE(ToolsDriver, EnumerateContexts);
    int device_count = 0;
    std::vector<LWcontext> ctxs;
    LWtoolsEnumContextData cb_data = {};
    cb_data.pUserData = static_cast<void *>(&ctxs);
    cb_data.struct_size = sizeof(cb_data);
    cb_data.pfnCallback = enumerateContextsCallback;

    // We assume that, if the test passed, it cleaned itself up
    if (!test_info.result()->Failed()) {
        return;
    }

    if (!ToolsDriver) {
        return;
    }

    if (ToolsDriver->EnumerateContexts(&cb_data) != LWDA_SUCCESS) {
        return;
    }

    // Destroy any and all created non-primary contexts
    // Primary contexts will be enumerated here too, but lwCtxDestroy will
    // return LWDA_ERROR_ILWALID_CONTEXT, which we'll ignore
    for (size_t i = 0; i < ctxs.size(); i++) {
        lwCtxDestroy(ctxs[i]);
    }

    lwDeviceGetCount(&device_count);

    // Now, run through all the devices and release their primary contexts
    for (int i = 0; i < device_count; i++) {
        LWdevice lwdev;
        unsigned int flags;
        int active;

        lwDeviceGet(&lwdev, i);
        lwDevicePrimaryCtxGetState(lwdev, &flags, &active);

        while (active) {
            lwDevicePrimaryCtxRelease(lwdev);
            lwDevicePrimaryCtxGetState(lwdev, &flags, &active);
        }
    }
    // TODO : optionally re-retain the primary contexts for the next set of tests

    // At this point, the driver should be in a reasonably pristine state
}

}

