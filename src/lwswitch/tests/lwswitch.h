/*******************************************************************************
    Copyright (c) 2013-2021 LWPU Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#ifndef _LWLINK_TESTS_LWSWITCH_H_
#define _LWLINK_TESTS_LWSWITCH_H_

#include "gtest/gtest.h"
#include "lwstatus.h"
#include "UtilOS.h"

#include <iostream>
#include <fstream>

#include <lw32.h>

extern "C"
{
    #include "ioctl_lwswitch.h"
    #include "ioctl_dev_lwswitch.h"
    #include "ioctl_dev_internal_lwswitch.h"

    #include "lwswitch_user_api.h"
    #include "lwlink_user_api.h"

    #include "lwlink_lib_ctrl.h"
    #include "lwlink_lib_ioctl.h"
}

extern LwU32 g_instance;

using namespace std;
#define LWSWITCH_DEVICE_DBDF_SIZE_MAX               13
#define LWSWITCH_ILWALID_PORT               0xffffffff

#ifdef _WIN32
    #define LW_SNPRINTF sprintf_s
#else
    #define LW_SNPRINTF snprintf
#endif

namespace lwswitch
{

typedef struct
{
    LwS32 instance;                                 // Device instance
    LwUuid uuid;                                    // Device UUID
    char dbdf[LWSWITCH_DEVICE_DBDF_SIZE_MAX];       // "%04x:%02x:%02x.0" + '\0'
} LWSwitchDevice;

class LWSwitchDevicePrinter : public ::testing::EmptyTestEventListener
{
    private:
        LwS32 instance;

    protected:
        // Called after a failed assertion or a SUCCEED() invocation.
        virtual void OnTestPartResult(const ::testing::TestPartResult& test_part_result)
        {
            if (instance >= 0)
                printf("Failure on lwswitch %d\n", instance);
        }

    public:
        LWSwitchDevicePrinter()
        {
            instance = -1;
        }

        void setInstance(LwS32 inst)
        {
            instance = inst;
        }

        void clearInstance(void)
        {
            instance = -1;
        }
};

extern LWSwitchDevicePrinter *g_listener;
extern LwBool verbose;

class LWSwitchDeviceTestBase
{
    private:
        LWSwitchDevice dev;
        lwswitch_device *device = NULL;
        lwlink_session *lwlinkDevice = NULL;
        LwU64 errorIdxs[LWSWITCH_ERROR_SEVERITY_MAX];
        LwU32 arch;
        LwU32 platform;
        LwU32 deviceID;
        LwU64 linkMask;
        LwU64 initializedLinkMask;
        LwU32 vcCount;
        LwU32 linkCount;
        LwU32 rlanTableSize;
        LwU32 ridTableSize;
        LwU32 remapTableSize;
        LwU32 remapTableExtASize;
        LwU32 remapTableExtBSize;
        LwU32 remapTableMCSize;
        LwBool bUnbindOnTeardown;
        LwU32 i2cAdapters[LWSWITCH_CTRL_NUM_I2C_PORTS];
        LwU32 i2cAdapterPortNum[LWSWITCH_CTRL_NUM_I2C_PORTS];
        LwU32 numI2cAdapters;
        LwBool inforomLWLSupported;
        LwBool inforomBBXSupported;
        static LwBool bRegWriteAccessChecked;
        static LwBool bRegReadAccessChecked;
        static LwBool bRegWriteAccess;
        static LwBool bRegReadAccess;

        LW_STATUS getDevices(LWSWITCH_GET_DEVICES_V2_PARAMS *pParams)
        {
            LW_STATUS status;

            status = lwswitch_api_get_devices(pParams);
            if (status != LW_OK)
            {
                printf("Cannot find lwswitch control device\n");\
                return status;
            }

            return LW_OK;
        }

        LW_STATUS getDeviceFromInstance(LwU32 instance, LWSwitchDevice *pDev)
        {
            LWSWITCH_GET_DEVICES_V2_PARAMS params;
            LwU32 iter;

            LW_STATUS status = getDevices(&params);
            if (status != LW_OK)
            {
                return status;
            }

            for (iter = 0; iter < params.deviceCount; iter++)
            {
                if (instance == params.info[iter].deviceInstance)
                    break;
            }

            if (iter < params.deviceCount)
            {
                pDev->instance = instance;
                memcpy(&pDev->uuid, &params.info[iter].uuid, sizeof(pDev->uuid));
                LW_SNPRINTF(pDev->dbdf, LWSWITCH_DEVICE_DBDF_SIZE_MAX, "%04x:%02x:%02x.0", params.info[iter].pciDomain,
                        params.info[iter].pciBus, params.info[iter].pciDevice);

                return LW_OK;
            }

            printf("Cannot find device with instance number: %u\n", instance);

            return LW_ERR_ILWALID_ARGUMENT;
        }

    protected:

    LWSwitchDeviceTestBase()
    {
        bUnbindOnTeardown = LW_FALSE;
    }

    virtual ~LWSwitchDeviceTestBase()
    {
        // You can do clean-up work that doesn't throw exceptions here.

        if (bUnbindOnTeardown)
        {
            // Destroy and re-create device after every test. This serves two
            // purposes -
            // 1. Tests driver load/unload sequence prior to running a new state.
            // 2. Re-inits driver's sticky state such as disabled interrupts on
            //    fatal errors.
            unbindRebindDevice();
        }
    }

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    virtual void SetUp()
    {
        LW_STATUS status;

        ASSERT_EQ(getDeviceFromInstance(g_instance, &dev), 0);

        g_listener->setInstance(dev.instance);
        ASSERT_GE(dev.instance, 0);
        LWSWITCH_GET_INFO infoParams;
        LWSWITCH_GET_LWLINK_STATUS_PARAMS linkStatusParams = {0};

        status = lwswitch_api_create_device(&dev.uuid, &device);
        ASSERT_EQ(status, LW_OK) << "Device open failed!";

        memset(errorIdxs, 0, sizeof(errorIdxs));
        readAllErrors();

        memset(i2cAdapters, 0, sizeof(i2cAdapters));
        memset(i2cAdapterPortNum, 0, sizeof(i2cAdapterPortNum));
        numI2cAdapters = 0;
        osAddLWSwitchI2CAdapter(i2cAdapters, i2cAdapterPortNum, LWSWITCH_CTRL_NUM_I2C_PORTS, string(dev.dbdf), &numI2cAdapters);

        //
        // TO-DO:
        // Combine the below GET_INFO IOCTLs into one IOCTL call to the driver
        //

        // Get ARCH
        memset(&infoParams, 0, sizeof(infoParams));
        infoParams.index[0] = LWSWITCH_GET_INFO_INDEX_ARCH;
        infoParams.count = 1;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &infoParams, sizeof(infoParams));
        ASSERT_EQ(status, LW_OK);
        arch = infoParams.info[0];

        // Get Platform
        memset(&infoParams, 0, sizeof(infoParams));
        infoParams.index[0] = LWSWITCH_GET_INFO_INDEX_PLATFORM;
        infoParams.count = 1;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &infoParams, sizeof(infoParams));
        ASSERT_EQ(status, LW_OK);
        platform = infoParams.info[0];
        ASSERT_NE(platform, LWSWITCH_GET_INFO_INDEX_PLATFORM_UNKNOWN);

        // Get Device ID
        memset(&infoParams, 0, sizeof(infoParams));
        infoParams.index[0] = LWSWITCH_GET_INFO_INDEX_DEVICE_ID;
        infoParams.count = 1;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &infoParams, sizeof(infoParams));
        ASSERT_EQ(status, LW_OK);
        deviceID = infoParams.info[0];

        // Get link caps
        memset(&infoParams, 0, sizeof(infoParams));
        infoParams.index[0] = LWSWITCH_GET_INFO_INDEX_ENABLED_PORTS_MASK_31_0;
        infoParams.index[1] = LWSWITCH_GET_INFO_INDEX_ENABLED_PORTS_MASK_63_32;
        infoParams.count = 2;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &infoParams, sizeof(infoParams));
        ASSERT_EQ(status, LW_OK);
        linkMask = infoParams.info[0] | ((LwU64) infoParams.info[1] << 32);
        ASSERT_NE(linkMask, 0);

        // Get link count
        memset(&infoParams, 0, sizeof(infoParams));
        infoParams.index[0] = LWSWITCH_GET_INFO_INDEX_NUM_PORTS;
        infoParams.count = 1;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &infoParams, sizeof(infoParams));
        ASSERT_EQ(status, LW_OK);
        linkCount = infoParams.info[0];

        // Get vc count
        memset(&infoParams, 0, sizeof(infoParams));
        infoParams.index[0] = LWSWITCH_GET_INFO_INDEX_NUM_VCS;
        infoParams.count = 1;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &infoParams, sizeof(infoParams));
        ASSERT_EQ(status, LW_OK);
        vcCount = infoParams.info[0];

        // Get rlanTableSize
        memset(&infoParams, 0, sizeof(infoParams));
        infoParams.index[0] = LWSWITCH_GET_INFO_INDEX_ROUTING_LAN_TABLE_SIZE;
        infoParams.count = 1;
        infoParams.info[0] = 0;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &infoParams, sizeof(infoParams));
        ASSERT_EQ(status, LW_OK);
        rlanTableSize = infoParams.info[0];

        // Get ridTableSize
        memset(&infoParams, 0, sizeof(infoParams));
        infoParams.index[0] = LWSWITCH_GET_INFO_INDEX_ROUTING_ID_TABLE_SIZE;
        infoParams.count = 1;
        infoParams.info[0] = 0;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &infoParams, sizeof(infoParams));
        ASSERT_EQ(status, LW_OK);
        ridTableSize = infoParams.info[0];

        // Get remapTableSize
        memset(&infoParams, 0, sizeof(infoParams));
        infoParams.index[0] = LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_TABLE_SIZE;
        infoParams.count = 1;
        infoParams.info[0] = 0;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &infoParams, sizeof(infoParams));
        ASSERT_EQ(status, LW_OK);
        remapTableSize = infoParams.info[0];

        // Get remapExtASize
        memset(&infoParams, 0, sizeof(infoParams));
        infoParams.index[0] = LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_EXTA_TABLE_SIZE;
        infoParams.count = 1;
        infoParams.info[0] = 0;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &infoParams, sizeof(infoParams));
        ASSERT_EQ(status, LW_OK);
        remapTableExtASize = infoParams.info[0];

        // Get remapTableExtBSize
        memset(&infoParams, 0, sizeof(infoParams));
        infoParams.index[0] = LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_EXTB_TABLE_SIZE;
        infoParams.count = 1;
        infoParams.info[0] = 0;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &infoParams, sizeof(infoParams));
        ASSERT_EQ(status, LW_OK);
        remapTableExtBSize = infoParams.info[0];

        // Get remapTableMCSize
        memset(&infoParams, 0, sizeof(infoParams));
        infoParams.index[0] = LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_MULTICAST_TABLE_SIZE;
        infoParams.count = 1;
        infoParams.info[0] = 0;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &infoParams, sizeof(infoParams));
        ASSERT_EQ(status, LW_OK);
        remapTableMCSize = infoParams.info[0];

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_LWLINK_STATUS, &linkStatusParams, sizeof(linkStatusParams));
        ASSERT_EQ(status, LW_OK);

        // get inforomLWLSupported
        memset(&infoParams, 0, sizeof(infoParams));
        infoParams.index[0] = LWSWITCH_GET_INFO_INDEX_INFOROM_LWL_SUPPORTED;
        infoParams.count = 1;
        infoParams.info[0] = 0;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &infoParams, sizeof(infoParams));
        ASSERT_EQ(status, LW_OK);
        inforomLWLSupported = infoParams.info[0] ? LW_TRUE : LW_FALSE;

        // get inforomBBXSupported
        memset(&infoParams, 0, sizeof(infoParams));
        infoParams.index[0] = LWSWITCH_GET_INFO_INDEX_INFOROM_BBX_SUPPORTED;
        infoParams.count = 1;
        infoParams.info[0] = 0;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &infoParams, sizeof(infoParams));
        ASSERT_EQ(status, LW_OK);
        inforomBBXSupported = infoParams.info[0] ? LW_TRUE : LW_FALSE;

        for (LwU32 iter = 0; iter < LWSWITCH_LWLINK_MAX_LINKS; iter++)
        {
            if ((linkStatusParams.linkInfo[iter].linkState == LWSWITCH_LWLINK_STATUS_LINK_STATE_INIT) ||
                (linkStatusParams.linkInfo[iter].linkState == LWSWITCH_LWLINK_STATUS_LINK_STATE_ILWALID))
            {
                initializedLinkMask &= ~(1 << iter);
            }
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
            if (linkStatusParams.linkInfo[iter].bIsRepeaterMode)
            {
                linkMask &= ~(1 << iter);
            }
#endif
        }

        // Set the starting point for rand()
        srand((unsigned int)time(0));

        // LWLink device node setup

        status = lwlink_api_init();
        ASSERT_EQ(status, LW_OK) << "Cannot create lwlink node" << endl;

        status = lwlink_api_create_session(&lwlinkDevice);
        ASSERT_EQ(status, LW_OK) << "Cannot open lwlink session" << endl;
    }

    virtual void TearDown()
    {
        // Code here will be called immediately after each test (right
        // before the destructor).

        lwswitch_api_free_device(&device);

        lwlink_api_free_session(&lwlinkDevice);
    }

public:
    void unbindRebindDevice()
    {
        // Restart LWSwitch device to re-enable the interrupts.
        osUpdateLWSwitchDeviceState(LWSWITCH_DEVICE_STATE_UPDATE_RESTART_DEVICE, deviceID, string(dev.dbdf));
    }

    lwswitch_device* getDevice()
    {
        return device;
    }

    lwlink_session* getLwlinkSession()
    {
        return lwlinkDevice;
    }

    LwU32 getArch()
    {
        return arch;
    }

    std::string getArchString()
    {
        switch(arch)
        {
            case LWSWITCH_GET_INFO_INDEX_ARCH_SV10:
                return "Willow";

            case LWSWITCH_GET_INFO_INDEX_ARCH_LR10:
                return "Limerock";
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
            case LWSWITCH_GET_INFO_INDEX_ARCH_LS10:
                return "Laguna Seca";
#endif
            default:
                return "Unknown";
        }
    }

    LwBool isArchSv10()
    {
        return (arch == LWSWITCH_GET_INFO_INDEX_ARCH_SV10);
    }

    LwBool isArchLr10()
    {
        return (arch == LWSWITCH_GET_INFO_INDEX_ARCH_LR10);
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool isArchLs10()
    {
        return (arch == LWSWITCH_GET_INFO_INDEX_ARCH_LS10);
    }
#endif

    LwBool isFmodel()
    {
        return (platform == LWSWITCH_GET_INFO_INDEX_PLATFORM_FMODEL);
    }

    LwU32 getLinkCount()
    {
        return linkCount;
    }

    LwU32 getvcCount()
    {
        return vcCount;
    }

    LwU32 getRlanTableSize()
    {
        return rlanTableSize;
    }

    LwU32 getRidTableSize()
    {
        return ridTableSize;
    }

    LwU32 getRemapTableSize(LWSWITCH_TABLE_SELECT_REMAP tableSelect)
    {
        if (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_PRIMARY)
        {
            return remapTableSize;
        }
        else if (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_EXTA)
        {
            return remapTableExtASize;
        }
        else if (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_EXTB)
        {
            return remapTableExtBSize;
        }
        else if (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_MULTICAST)
        {
            return remapTableMCSize;
        }
        else 
        {
            return 0;
        }
    }

    LWSWITCH_TABLE_SELECT_REMAP getRemapTableNext(LWSWITCH_TABLE_SELECT_REMAP tableSelect)
    {
        switch (tableSelect)
        {
            case LWSWITCH_TABLE_SELECT_REMAP_PRIMARY:
                return LWSWITCH_TABLE_SELECT_REMAP_EXTA;
                break;
            case LWSWITCH_TABLE_SELECT_REMAP_EXTA:
                return LWSWITCH_TABLE_SELECT_REMAP_EXTB;
                break;
            case LWSWITCH_TABLE_SELECT_REMAP_EXTB:
                return LWSWITCH_TABLE_SELECT_REMAP_MULTICAST;
                break;
            case LWSWITCH_TABLE_SELECT_REMAP_MULTICAST:
                return (LWSWITCH_TABLE_SELECT_REMAP) 0xFF;
                break;
            default:
                return (LWSWITCH_TABLE_SELECT_REMAP) ~0;
                break;
        }
    }

    LwU64 getLinkMask()
    {
        return linkMask;
    }

    LwU64 getLinkInitializedMask()
    {
        return initializedLinkMask;
    }

    void getI2cAdapters(LwU32 *adapters, LwU32 arrLen)
    {
        LwU32 i;

        if (arrLen < numI2cAdapters)
        {
            printf("Invalid array size.");
            return;
        }

        for (i = 0; i < numI2cAdapters; i++)
        {
            adapters[i] = i2cAdapters[i];
        }
    }

    void getI2cPortNums(LwU32 *adapterPortNum, LwU32 arrLen)
    {
        LwU32 i;

        if (arrLen < numI2cAdapters)
        {
            printf("Invalid array size.");
            return;
        }

        for (i = 0; i < numI2cAdapters; i++)
        {
            adapterPortNum[i] = i2cAdapterPortNum[i];
        }
    }

    LwU32 getNumI2cAdapters()
    {
        return numI2cAdapters;
    }

    LwBool isInforomLWLSupported()
    {
        return inforomLWLSupported;
    }

    LwBool isInforomBBXSupported()
    {
        return inforomBBXSupported;
    }

    void validPort(LwU32 *port)
    {
        LwU64 linkMask = getLinkMask();
        LwU32 linkCount = getLinkCount();
        LwU32 i;

        for (i = 0; i < linkCount; i++)
        {
            if ((LwU64)(1 << i) & linkMask)
            {
                *port = i;
                return;
            }
        }
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    void repeaterPort(LwU32 *port)
    {
        LW_STATUS status;
        LwU64 linkMask = getLinkMask();
        LwU32 linkCount = getLinkCount();
        lwswitch_device *device = getDevice();
        LWSWITCH_GET_LWLINK_STATUS_PARAMS link_status_params;
        LwU32 i;

        memset(&link_status_params, 0, sizeof(link_status_params));

        status = lwswitch_api_control(device,
                                      IOCTL_LWSWITCH_GET_LWLINK_STATUS,
                                      &link_status_params,
                                      sizeof(link_status_params));
        ASSERT_EQ(status, LW_OK);

        for (i = 0; i < linkCount; i++)
        {
            if ((((LwU64)(1 << i) & linkMask) == 0) &&
                link_status_params.linkInfo[i].bIsRepeaterMode)
            {
                *port = i;
                return;
            }
        }
    }

    void getValidPortOrRepeaterPort(LwU32 *valid_port, LwBool *is_repeater_port)
    {
        LwU32 port = LWSWITCH_ILWALID_PORT;
        LwBool bIsRepeaterPort = LW_FALSE;

        validPort(&port);
        if (port == LWSWITCH_ILWALID_PORT)
        {
            if (!isArchLs10())
            {
                FAIL() << "Found no valid ports.";
            }
            else
            {
                //
                // On LS10, ports may be disabled because of Repeater Mode.
                // If there are links in Repeater Mode, it should be tested
                // seperately. If there are no ports in Repeater Mode,
                // something is wrong.
                //
                repeaterPort(&port);
                if (port == LWSWITCH_ILWALID_PORT)
                {
                    FAIL() << "Found no ports in Repeater Mode.";
                }
                else
                {
                    bIsRepeaterPort = LW_TRUE;
                }
            }
        }

        *valid_port = port;
        if (is_repeater_port != NULL)
        {
            *is_repeater_port = bIsRepeaterPort;
        }
    }
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    // The driver only supports this functionality on Debug/Develop builds.
    LwBool isRegWritePermitted()
    {
        LWSWITCH_REGISTER_WRITE wr;
        lwswitch_device *device = getDevice();
        LW_STATUS status;

        if (!bRegWriteAccessChecked)
        {
            // Try to write PMC_BOOT_0 (offset 0)
            memset(&wr, 0, sizeof(wr));
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_REGISTER_WRITE, &wr, sizeof(IOCTL_LWSWITCH_REGISTER_WRITE));

            if (status == LW_ERR_NOT_SUPPORTED)
            {
                bRegWriteAccess = false;
            }
            else
            {
                bRegWriteAccess = true;
            }

            bRegWriteAccessChecked = true;
        }

        return bRegWriteAccess;
    }

    // The driver only supports this functionality on Debug/Develop builds.
    LwBool isRegReadPermitted()
    {
        LWSWITCH_REGISTER_READ rd;
        lwswitch_device *device = getDevice();
        LW_STATUS status;

        if (!bRegReadAccessChecked)
        {
            // Try to read PMC_BOOT_0 (offset 0)
            memset(&rd, 0, sizeof(rd));
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_REGISTER_READ, &rd, sizeof(IOCTL_LWSWITCH_REGISTER_READ));

            if (status == LW_ERR_NOT_SUPPORTED)
            {
                bRegReadAccess = LW_FALSE;
            }
            else
            {
                bRegReadAccess = LW_TRUE;
            }

            bRegReadAccessChecked = LW_TRUE;
        }

        return bRegReadAccess;
    }

    void regWrite(LwU32 engine, LwU32 instance, LwU32 offset, LwU32 val)
    {
        LWSWITCH_REGISTER_WRITE wr;

        memset(&wr, 0, sizeof(wr));
        wr.engine = engine;
        wr.instance = instance;
        wr.offset = offset;
        wr.bcast = 0;
        wr.val = val;

        doIoctl(IOCTL_LWSWITCH_REGISTER_WRITE, &wr, sizeof(wr));
    }

    void regRead(LwU32 engine, LwU32 instance, LwU32 offset, LwU32 *val)
    {
        LWSWITCH_REGISTER_READ rd;

        memset(&rd, 0, sizeof(rd));
        rd.engine = engine;
        rd.instance = instance;
        rd.offset = offset;

        doIoctl(IOCTL_LWSWITCH_REGISTER_READ, &rd, sizeof(rd));

        *val = rd.val;
    }

    void doIoctl(LwU32 cmd, void *params, LwU32 param_size)
    {
        LW_STATUS status;
        lwswitch_device *device = getDevice();

        status = lwswitch_api_control(device, cmd, params, param_size);
        ASSERT_EQ(status, LW_OK);
    }

    void getErrors(LWSWITCH_GET_ERRORS_PARAMS *p)
    {
        ASSERT_LT(p->errorType, (LwU32) LWSWITCH_ERROR_SEVERITY_MAX);

        p->errorIndex = errorIdxs[p->errorType];
        doIoctl(IOCTL_LWSWITCH_GET_ERRORS, p, sizeof(LWSWITCH_GET_ERRORS_PARAMS));
        if (p->errorCount > 0)
        {
            errorIdxs[p->errorType] = p->errorIndex;
        }
    }

    void readAllErrors()
    {
        LwU32 errorType;
        LWSWITCH_GET_ERRORS_PARAMS data;

        for (errorType = 0; errorType < LWSWITCH_ERROR_SEVERITY_MAX; errorType++)
        {
            memset(&data, 0, sizeof(data));
            data.errorType = errorType;
            getErrors(&data);
            errorIdxs[errorType] = data.nextErrorIndex;
        }
    }

    void resetPorts(LwU64 mask)
    {
        LWSWITCH_RESET_AND_DRAIN_LINKS_PARAMS dp_params;

        memset(&dp_params, 0, sizeof(dp_params));
        dp_params.linkMask = mask;

        doIoctl(IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS, &dp_params, sizeof(dp_params));
    }

    void createEvent(LwU32 *events, LwU32 numEvents, lwswitch_event **event)
    {
        LW_STATUS status;

        lwswitch_device *device = getDevice();

        status = lwswitch_api_create_event(device, events, numEvents, event);
        ASSERT_EQ(status, LW_OK);
    }

    void waitForEvent(lwswitch_event **event, const LwU32 timeout = 100, LwBool bTimeoutExpected = LW_FALSE)
    {
        LW_STATUS status;

        status = lwswitch_api_wait_events(event, 1, timeout);

        ASSERT_EQ(status, bTimeoutExpected ? LW_ERR_TIMEOUT : LW_OK);

        lwswitch_api_free_event(event);
    }

    void setUnbindOnTeardown()
    {
        bUnbindOnTeardown = LW_TRUE;
    }

    void skipUnbindOnTeardown()
    {
        bUnbindOnTeardown = LW_FALSE;
    }

    void initLinks_SV10()
    {
        LW_STATUS status;
        lwlink_session *session = getLwlinkSession();

        // Init phase1
        lwlink_initphase1 initphase1 = {0};
        status = lwlink_api_control(session, IOCTL_LWLINK_INITPHASE1, &initphase1, sizeof(initphase1));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(initphase1.status, 0) << "status: " << initphase1.status << endl;

        // Init RX Termination
        lwlink_rx_init_term rxInitTerm = {0};
        status = lwlink_api_control(session, IOCTL_LWLINK_RX_INIT_TERM , &rxInitTerm, sizeof(rxInitTerm));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(rxInitTerm.status, 0) << "status: " << rxInitTerm.status << endl;

        // Set RX detect
        lwlink_set_rx_detect setRxDetect = {0};
        status = lwlink_api_control(session, IOCTL_LWLINK_SET_RX_DETECT , &setRxDetect, sizeof(setRxDetect));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(setRxDetect.status, 0) << "status: " << setRxDetect.status << endl;

        // Get RX detect
        lwlink_get_rx_detect getRxDetect = {0};
        status = lwlink_api_control(session, IOCTL_LWLINK_GET_RX_DETECT , &getRxDetect, sizeof(getRxDetect));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(getRxDetect.status, 0) << "status: " << getRxDetect.status << endl;

        // Set TX common mode
        lwlink_set_tx_common_mode setTxCommonMode = {0};
        setTxCommonMode.commMode = LW_TRUE;
        status = lwlink_api_control(session, IOCTL_LWLINK_SET_TX_COMMON_MODE, &setTxCommonMode, sizeof(setTxCommonMode));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(setTxCommonMode.status, 0) << "status: " << setTxCommonMode.status << endl;
    }

    void initLinks_LR10()
    {
        LW_STATUS status;
        lwlink_session *session = getLwlinkSession();
        LWSWITCH_GET_DEVICES_V2_PARAMS params;
        lwlink_pci_dev_info pciInfo = {0};
        LwU32 i;

        // Get PCI Info
        status = getDevices(&params);
        ASSERT_EQ(status, LW_OK);
        pciInfo.domain   = static_cast<LwU16>(params.info[g_instance].pciDomain);
        pciInfo.bus      = static_cast<LwU8>(params.info[g_instance].pciBus);
        pciInfo.device   = static_cast<LwU8>(params.info[g_instance].pciDevice);
        pciInfo.function = static_cast<LwU8>(params.info[g_instance].pciFunction);

        // Init phase1
        lwlink_initphase1 initphase1 = {0};
        status = lwlink_api_control(session, IOCTL_LWLINK_INITPHASE1, &initphase1, sizeof(initphase1));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(initphase1.status, 0) << "status: " << initphase1.status << endl;

        // Init RX Termination
        lwlink_rx_init_term rxInitTerm = {0};
        status = lwlink_api_control(session, IOCTL_LWLINK_RX_INIT_TERM , &rxInitTerm, sizeof(rxInitTerm));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(rxInitTerm.status, 0) << "status: " << rxInitTerm.status << endl;

        // Set RX detect
        lwlink_set_rx_detect setRxDetect = {0};
        status = lwlink_api_control(session, IOCTL_LWLINK_SET_RX_DETECT , &setRxDetect, sizeof(setRxDetect));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(setRxDetect.status, 0) << "status: " << setRxDetect.status << endl;

        // Get RX detect
        lwlink_get_rx_detect getRxDetect = {0};
        status = lwlink_api_control(session, IOCTL_LWLINK_GET_RX_DETECT , &getRxDetect, sizeof(getRxDetect));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(getRxDetect.status, 0) << "status: " << getRxDetect.status << endl;

        // Set TX common mode
        lwlink_set_tx_common_mode setTxCommonMode = {0};
        setTxCommonMode.commMode = LW_TRUE;
        status = lwlink_api_control(session, IOCTL_LWLINK_SET_TX_COMMON_MODE, &setTxCommonMode, sizeof(setTxCommonMode));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(setTxCommonMode.status, 0) << "status: " << setTxCommonMode.status << endl;

        // Initiate Rx Calibration
        lwlink_calibrate calibrate = {0};
        status = lwlink_api_control(session, IOCTL_LWLINK_CALIBRATE, &calibrate, sizeof(calibrate));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(calibrate.status, 0) << "status: " << calibrate.status << endl;

        // Set TX common mode
        setTxCommonMode = {0};
        setTxCommonMode.commMode = LW_FALSE;
        status = lwlink_api_control(session, IOCTL_LWLINK_SET_TX_COMMON_MODE, &setTxCommonMode, sizeof(setTxCommonMode));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(setTxCommonMode.status, 0) << "status: " << setTxCommonMode.status << endl;

        // Enable TX data
        lwlink_enable_data enableData = {0};
        status = lwlink_api_control(session, IOCTL_LWLINK_ENABLE_DATA, &enableData, sizeof(enableData));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(enableData.status, 0) << "status: " << enableData.status << endl;

        // Init links
        lwlink_link_init_async initLinks = {0};
        status = lwlink_api_control(session, IOCTL_CTRL_LWLINK_LINK_INIT_ASYNC, &initLinks, sizeof(initLinks));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(initLinks.status, 0) << "status: " << initLinks.status << endl;

        // Check link status
        lwlink_device_link_init_status linkInitStatus = {{0}};
        linkInitStatus.devInfo.nodeId = LW_U16_MAX;
        linkInitStatus.devInfo.pciInfo = pciInfo;
        status = lwlink_api_control(session, IOCTL_CTRL_LWLINK_DEVICE_LINK_INIT_STATUS, &linkInitStatus, sizeof(linkInitStatus));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(linkInitStatus.status, 0) << "status: " << linkInitStatus.status << endl;
        ASSERT_EQ(linkInitStatus.linkStatus[0].initStatus, LW_TRUE) << "initStatus: " << linkInitStatus.linkStatus[0].initStatus << endl;

        // Send INITNEGOTIATE request to minion
        lwlink_initnegotiate initnegotiate = {0};
        status = lwlink_api_control(session, IOCTL_LWLINK_INITNEGOTIATE, &initnegotiate, sizeof(initnegotiate));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(initnegotiate.status, 0) << "status: " << initnegotiate.status << endl;

        // Initiate an lwlink connection discovery
        lwlink_discover_intranode_conns discovery = {0};
        status = lwlink_api_control(session, IOCTL_LWLINK_DISCOVER_INTRANODE_CONNS, &discovery, sizeof(discovery));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(discovery.status, 0) << "status: " << discovery.status << endl;

        // Get intranode connections
        lwlink_device_get_intranode_conns connsInfo = {{0}};
        connsInfo.devInfo.nodeId = LW_U16_MAX;
        connsInfo.devInfo.pciInfo = pciInfo;
        status = lwlink_api_control(session, IOCTL_LWLINK_DEVICE_GET_INTRANODE_CONNS, &connsInfo, sizeof(connsInfo));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(connsInfo.status, 0) << "status: " << connsInfo.status << endl;

        // Train links to Active.
        lwlink_train_intranode_conns_parallel trainLinks = {0};
        trainLinks.trainTo = lwlink_train_conn_swcfg_to_active;
        trainLinks.endPointPairsCount = connsInfo.numConnections;

        for(i = 0; i < connsInfo.numConnections; i++)
        {
            trainLinks.endPointPairs[i].src = connsInfo.conn[i].srcEndPoint;
            trainLinks.endPointPairs[i].dst = connsInfo.conn[i].dstEndPoint;   
        }

        status = lwlink_api_control(session, IOCTL_LWLINK_TRAIN_INTRANODE_CONNS_PARALLEL, &trainLinks, sizeof(trainLinks));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(trainLinks.status, 0) << "status: " << trainLinks.status << endl;

        for(i = 0; i < connsInfo.numConnections; i++)
        {
            ASSERT_EQ(trainLinks.endpointPairsStates[i].srcEnd.linkMode, lwlink_link_mode_active) << "srcLinkMode: " << trainLinks.endpointPairsStates[i].srcEnd.linkMode << endl;
            ASSERT_EQ(trainLinks.endpointPairsStates[i].dstEnd.linkMode, lwlink_link_mode_active) << "dstLinkMode: " << trainLinks.endpointPairsStates[i].dstEnd.linkMode << endl;
            ASSERT_EQ(trainLinks.endpointPairsStates[i].srcEnd.txSubLinkMode, lwlink_tx_sublink_mode_hs) << "srcTxSubLinkMode: " << trainLinks.endpointPairsStates[i].srcEnd.txSubLinkMode << endl;
            ASSERT_EQ(trainLinks.endpointPairsStates[i].dstEnd.txSubLinkMode, lwlink_tx_sublink_mode_hs) << "dstTxSubLinkMode: " << trainLinks.endpointPairsStates[i].dstEnd.txSubLinkMode << endl;
            ASSERT_EQ(trainLinks.endpointPairsStates[i].srcEnd.rxSubLinkMode, lwlink_rx_sublink_mode_hs) << "srcRxSubLinkMode: " << trainLinks.endpointPairsStates[i].srcEnd.rxSubLinkMode << endl;
            ASSERT_EQ(trainLinks.endpointPairsStates[i].dstEnd.rxSubLinkMode, lwlink_rx_sublink_mode_hs) << "dstRxSubLinkMode: " << trainLinks.endpointPairsStates[i].dstEnd.rxSubLinkMode << endl;
        }
    }

    void initLinks()
    {
        if (isArchSv10())
        {
            initLinks_SV10();
        }
        else
        {
            initLinks_LR10();
        }
    }

    void shutdownLinks()
    {
        LW_STATUS status;
        lwlink_session *session = getLwlinkSession();
        LWSWITCH_GET_DEVICES_V2_PARAMS params;
        lwlink_pci_dev_info pciInfo = {0};
        LwU32 i;

        // Get PCI Info
        status = getDevices(&params);
        ASSERT_EQ(status, LW_OK);
        pciInfo.domain   = static_cast<LwU16>(params.info[g_instance].pciDomain);
        pciInfo.bus      = static_cast<LwU8>(params.info[g_instance].pciBus);
        pciInfo.device   = static_cast<LwU8>(params.info[g_instance].pciDevice);
        pciInfo.function = static_cast<LwU8>(params.info[g_instance].pciFunction);

        // Initiate an lwlink connection discovery
        lwlink_discover_intranode_conns discovery = {0};
        status = lwlink_api_control(session, IOCTL_LWLINK_DISCOVER_INTRANODE_CONNS, &discovery, sizeof(discovery));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(discovery.status, 0) << "status: " << discovery.status << endl;

        // Get intranode connections
        lwlink_device_get_intranode_conns connsInfo = {{0}};
        connsInfo.devInfo.nodeId = LW_U16_MAX;
        connsInfo.devInfo.pciInfo = pciInfo;
        status = lwlink_api_control(session, IOCTL_LWLINK_DEVICE_GET_INTRANODE_CONNS, &connsInfo, sizeof(connsInfo));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(connsInfo.status, 0) << "status: " << connsInfo.status << endl;

        // Shutdown links
        lwlink_train_intranode_conns_parallel trainLinks = {0};
        trainLinks.trainTo = lwlink_train_conn_to_off;
        trainLinks.endPointPairsCount = connsInfo.numConnections;
        
        for(i = 0; i < connsInfo.numConnections; i++)
        {
            trainLinks.endPointPairs[i].src = connsInfo.conn[i].srcEndPoint;
            trainLinks.endPointPairs[i].dst = connsInfo.conn[i].dstEndPoint;
        }

        status = lwlink_api_control(session, IOCTL_LWLINK_TRAIN_INTRANODE_CONNS_PARALLEL, &trainLinks, sizeof(trainLinks));
        ASSERT_EQ(status, LW_OK) << "LW_STATUS: 0x" << hex << status << endl;
        ASSERT_EQ(trainLinks.status, 0) << "status: " << trainLinks.status << endl;

        for(i = 0; i < connsInfo.numConnections; i++)
        {
            ASSERT_EQ(trainLinks.endpointPairsStates[i].srcEnd.linkMode,  lwlink_link_mode_reset) << "srcLinkMode: " << trainLinks.endpointPairsStates[i].srcEnd.linkMode << endl;
            ASSERT_EQ(trainLinks.endpointPairsStates[i].dstEnd.linkMode,  lwlink_link_mode_reset) << "dstLinkMode: " << trainLinks.endpointPairsStates[i].dstEnd.linkMode << endl;
            ASSERT_EQ(trainLinks.endpointPairsStates[i].srcEnd.txSubLinkMode, lwlink_tx_sublink_mode_off) << "srcTxSubLinkMode: " << trainLinks.endpointPairsStates[i].srcEnd.txSubLinkMode << endl;
            ASSERT_EQ(trainLinks.endpointPairsStates[i].dstEnd.txSubLinkMode, lwlink_tx_sublink_mode_off) << "dstTxSubLinkMode: " << trainLinks.endpointPairsStates[i].dstEnd.txSubLinkMode << endl;
            ASSERT_EQ(trainLinks.endpointPairsStates[i].srcEnd.rxSubLinkMode, lwlink_rx_sublink_mode_off) << "srcRxSubLinkMode: " << trainLinks.endpointPairsStates[i].srcEnd.rxSubLinkMode << endl;
            ASSERT_EQ(trainLinks.endpointPairsStates[i].dstEnd.rxSubLinkMode, lwlink_rx_sublink_mode_off) << "dstRxSubLinkMode: " << trainLinks.endpointPairsStates[i].dstEnd.rxSubLinkMode << endl;
        }
    }
};

class LWSwitchDeviceTest : public LWSwitchDeviceTestBase, public ::testing::Test
{
public:
    void SetUp()
    {
        LWSwitchDeviceTestBase::SetUp();
    }

    void TearDown()
    {
        LWSwitchDeviceTestBase::TearDown();
    }
};

} // namespace

#endif // _LWLINK_TESTS_LWSWITCH_H_
