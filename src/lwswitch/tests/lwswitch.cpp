/*******************************************************************************
    Copyright (c) 2013-2020 LWPU Corporation

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

#include "lwswitch.h"
#include "commandline/commandline.h"

using namespace lwswitch;

class LWSwitchDevicePrinter *lwswitch::g_listener;

LwBool lwswitch::verbose = 0;
LwU32  g_instance = 0;

enum
{
    ARGNAME_HELP,
    ARGNAME_LIST,
    ARGNAME_ID,
    ARGNAME__COUNT
};

struct all_args allArgs[] =
{
    {
        ARGNAME_HELP,
        "-h",
        "--help",
        "Display the help messages",
        "\n\t"
        "Display a detailed usage description and exit.",
        CMDLINE_OPTION_NO_VALUE_ALLOWED,
    },
    {
        ARGNAME_LIST,
        "-l",
        "--list",
        "List all LWSwitch devices",
        "\n\t"
        "List all LWSwitch devices on this computer.",
        CMDLINE_OPTION_NO_VALUE_ALLOWED,
    },
    {
        ARGNAME_ID,
        "-i",
        "--index",
        "Id of LWSwitch to use (zero-based)",
        "\n\t"
        "Id of LWSwitch to use. They are numbered starting at zero. Defaults to 0 "
        "if this argument is not provided\n\t",
        CMDLINE_OPTION_VALUE_REQUIRED,
    },
};

static LW_STATUS listDevices()
{
    LWSWITCH_GET_DEVICES_V2_PARAMS params;
    LW_STATUS status;
    LwU32 i;
    char uuid_string[LWSWITCH_UUID_STRING_LENGTH] = { 0 };

    status = lwswitch_api_get_devices(&params);
    if (status != LW_OK)
    {
        printf("Cannot find lwswitch control device\n");
        return status;
    }

    for (i = 0; i < params.deviceCount; i++)
    {
        lwswitch_uuid_to_string(&params.info[i].uuid, uuid_string, LWSWITCH_UUID_STRING_LENGTH);

        printf("LWSwitch Device ID: %03d \n\t UUID: %s \n\t PCI: %04x:%02x:%02x.0\n",
                    params.info[i].deviceInstance, uuid_string, params.info[i].pciDomain,
                    params.info[i].pciBus, params.info[i].pciDevice);
        printf("\t Physical ID: %08x\n", params.info[i].physId);
        printf("\t Driver State %02x, Fabric State %02x, Reason %02x\n",
               params.info[i].driverState, params.info[i].deviceState, params.info[i].deviceReason);
    }

    return LW_OK;
}

static LW_STATUS validateDevice(unsigned deviceInstance)
{
    LWSWITCH_GET_DEVICES_V2_PARAMS params;
    LW_STATUS status;
    unsigned int idx;

    status = lwswitch_api_get_devices(&params);
    if (status != LW_OK)
    {
        printf("Cannot find lwswitch control device\n");
        return LW_ERR_OPERATING_SYSTEM;
    }

    if (deviceInstance >= params.deviceCount)
    {
        return LW_ERR_ILWALID_ARGUMENT;
    }

    for (idx = 0; idx < params.deviceCount; idx++)
    {
        if (params.info[idx].deviceInstance == deviceInstance &&
            params.info[idx].deviceState == LWSWITCH_DEVICE_FABRIC_STATE_BLACKLISTED)
        {
            printf("Cannot test blacklisted device\n");
            return LW_ERR_ILWALID_ARGUMENT;
        }
    }

    return LW_OK;
}

// Initialize static members of the classes
LwBool LWSwitchDeviceTestBase::bRegReadAccessChecked = LW_FALSE;
LwBool LWSwitchDeviceTestBase::bRegWriteAccessChecked = LW_FALSE;
LwBool LWSwitchDeviceTestBase::bRegReadAccess = LW_FALSE;
LwBool LWSwitchDeviceTestBase::bRegWriteAccess = LW_FALSE;

int main(int argc, char **argv)
{
    void *pCmdLine = NULL;
    int status = 0;
    LwBool runTests = LW_FALSE;

    status = cmdline_init(argc, argv, allArgs, ARGNAME__COUNT, &pCmdLine);
    if (status != 0)
    {
        printf("Command line init failed 0x%x\n", status);
        cmdline_printOptionsSummary(pCmdLine, 1);
        goto done;
    }

    // Help
    if (cmdline_exists(pCmdLine, ARGNAME_HELP))
    {
        cmdline_printOptionsSummary(pCmdLine, 1);
        goto done;
    }

    // List all devices
    if (cmdline_exists(pCmdLine, ARGNAME_LIST))
    {
        status = listDevices();
        goto done;
    }

    // Index
    if (cmdline_exists(pCmdLine, ARGNAME_ID))
    {
        unsigned long long val;

        g_instance = cmdline_getIntegerVal(pCmdLine, ARGNAME_ID, &val) ? (unsigned)val : 0;
        status = validateDevice(g_instance);
        if (status != 0)
        {
            printf("Invalid device ID!\n");
            goto done;
        }
    }

    // Devices loaded, now ready to init and run!
    runTests = LW_TRUE;

done:
    testing::InitGoogleTest(&argc, argv);

    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    lwswitch::g_listener = new LWSwitchDevicePrinter();
    listeners.Append(lwswitch::g_listener);

    if (runTests)
    {
        status = RUN_ALL_TESTS();
    }

    cmdline_destroy(pCmdLine);
    return status;
}
