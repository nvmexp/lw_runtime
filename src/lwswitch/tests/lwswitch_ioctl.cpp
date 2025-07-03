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

#include "lwswitch.h"
#include "lwmisc.h"
#include "UtilOS.h"

#include "lr10/dev_ingress_ip.h"
#include "lr10/dev_nport_ip.h"

#ifdef __linux__
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif // __linux__

using namespace lwswitch;

int main(int argc, char **argv);

#define INVALID 0xdeadbeef

/*
 * Colwert LwTemp to 32 bit Float
 */
#define LW_TYPES_LW_TEMP_TO_F32(fxp)  \
    (LwF32) fxp / (1 << DRF_SIZE(LW_TYPES_FXP_FRACTIONAL(24, 8)))

TEST_F(LWSwitchDeviceTest, IoctlSetSwitchPortConfig)
{
    LWSWITCH_SET_SWITCH_PORT_CONFIG p;
    LwU64 linkMask = getLinkMask();
    LwU32 linkCount = getLinkCount();
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 i;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwU32 valid_port;
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    for (i = 0; i < linkCount; i++)
    {
        if (LWBIT64(i) & linkMask)
        {
            p.portNum = i;
            p.type = CONNECT_ACCESS_GPU;
            p.requesterLinkID = 0xf;
            p.requesterLanID  = 0xf;
            p.count           = CONNECT_COUNT_512;
            p.acCoupled = LW_FALSE;
            p.enableVC1 = LW_FALSE;
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG, &p, sizeof(p));
            ASSERT_EQ(status, LW_OK);

            p.type = CONNECT_ACCESS_CPU;
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG, &p, sizeof(p));
            ASSERT_EQ(status, LW_OK);

            p.type = CONNECT_TRUNK_SWITCH;
            p.enableVC1 = LW_TRUE;
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG, &p, sizeof(p));
            ASSERT_EQ(status, LW_OK);

            p.type = CONNECT_ACCESS_SWITCH;
            p.enableVC1 = LW_FALSE;
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG, &p, sizeof(p));
            ASSERT_EQ(status, LW_OK);
        }
    }
}

TEST_F(LWSwitchDeviceTest, IoctlSetSwitchPortConfigRepeaterMode)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LWSWITCH_SET_SWITCH_PORT_CONFIG p;
    LWSWITCH_GET_LWLINK_STATUS_PARAMS link_status_params;
    LwU64 linkMask = getLinkMask();
    LwU32 linkCount = getLinkCount();
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 i;
    LwBool bHasRepeaterLink = LW_FALSE;

    if (!isArchLs10())
    {
        return;
    }

    memset(&link_status_params, 0, sizeof(link_status_params));

    // Check if link is in Repeater Mode
    status = lwswitch_api_control(device,
                                  IOCTL_LWSWITCH_GET_LWLINK_STATUS,
                                  &link_status_params,
                                  sizeof(link_status_params));
    ASSERT_EQ(status, LW_OK);

    for (i = 0; i < linkCount; i++)
    {
        if (((LWBIT64(i) & linkMask) == 0) &&
            link_status_params.linkInfo[i].bIsRepeaterMode)
        {
            bHasRepeaterLink = LW_TRUE;
            p.portNum = i;
            p.type = CONNECT_ACCESS_GPU;
            p.requesterLinkID = 0xf;
            p.requesterLanID  = 0xf;
            p.count           = CONNECT_COUNT_512;
            p.acCoupled = LW_FALSE;
            p.enableVC1 = LW_FALSE;

            status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG, &p, sizeof(p));
            ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

            p.type = CONNECT_ACCESS_CPU;
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG, &p, sizeof(p));
            ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

            p.type = CONNECT_TRUNK_SWITCH;
            p.enableVC1 = LW_TRUE;
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG, &p, sizeof(p));
            ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

            p.type = CONNECT_ACCESS_SWITCH;
            p.enableVC1 = LW_FALSE;
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG, &p, sizeof(p));
            ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
        }
    }

    if (!bHasRepeaterLink)
    {
        printf("[ SKIPPED  ] No port in Repeater Mode.\n");
    }
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
}

TEST_F(LWSwitchDeviceTest, IoctlSetSwitchPortConfigBadInput)
{
    LWSWITCH_SET_SWITCH_PORT_CONFIG p;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    // Invalid port
    p.portNum = 192;
    p.type = CONNECT_ACCESS_GPU;
    p.requesterLinkID = 1;
    p.requesterLanID  = p.requesterLinkID;
    p.count           = CONNECT_COUNT_512;
    p.acCoupled = LW_FALSE;
    p.enableVC1 = LW_FALSE;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Invalid type
    p.portNum = valid_port;
    p.type = (LWSWITCH_CONNECTION_TYPE)0xdeadbeef;
    p.requesterLinkID = 1;
    p.requesterLanID  = p.requesterLinkID;
    p.count           = CONNECT_COUNT_512;
    p.acCoupled = LW_FALSE;
    p.enableVC1 = LW_FALSE;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Invalid requesterLinkID
    p.portNum = valid_port;
    p.type = CONNECT_ACCESS_GPU;
    p.requesterLinkID = 0xffffffff;
    p.requesterLanID  = 1;
    p.count           = CONNECT_COUNT_512;
    p.acCoupled = LW_FALSE;
    p.enableVC1 = LW_FALSE;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Invalid requesterLanID
    if (!isArchSv10())
    {
        p.portNum = valid_port;
        p.type = CONNECT_ACCESS_GPU;
        p.requesterLinkID = 1;
        p.requesterLanID  = 0xffffffff;
        p.count           = CONNECT_COUNT_512;
        p.acCoupled = LW_FALSE;
        p.enableVC1 = LW_FALSE;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG, &p, sizeof(p));
        ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
    }

    // Invalid VC enable
    p.portNum = valid_port;
    p.type = CONNECT_ACCESS_GPU;
    p.requesterLinkID = 1;
    p.requesterLanID  = p.requesterLinkID;
    p.count           = CONNECT_COUNT_512;
    p.acCoupled = LW_FALSE;
    p.enableVC1 = LW_TRUE;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
}

//
// Callwlate some valid offset/counts for up to max entries.
//
struct first_num {
    LwU32 first;
    LwU32 num;
};

static LwU32 gen_first_num(struct first_num *inputs, LwU32 max)
{
    LwU32 count = 0;
    LwU32 f, n;

    for (f=0; f < max; f = (f ? (f<<1) : 1))
    {
        for (n=0; (f + (1<<n)) <= max; n++)
        {
            inputs[count].first = f;
            inputs[count].num = 1 << n;
            count++;
        }
    }

    return count;
}

#define ROUTING_TABLE_ENTRIES    8192

TEST_F(LWSwitchDeviceTest, IoctlGetIngressResponseTable)
{
    LWSWITCH_GET_INGRESS_RESPONSE_TABLE_PARAMS get_table_params;
    LWSWITCH_SET_INGRESS_RESPONSE_TABLE set_table_params;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 i;

    validPort(&valid_port);
    if (valid_port == LWSWITCH_ILWALID_PORT)
    {
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        //
        // Ports can only be invalid on LS10 because of Repeater Mode
        //
        if (!isArchLs10())
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        {
            FAIL() << "Found no valid ports.";
        }
    }

    get_table_params.firstIndex = 0;
    get_table_params.portNum = valid_port;

    /* this ioctl is only supported on sv10 */
    if (!isArchSv10())
    {
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INGRESS_RESPONSE_TABLE, &get_table_params, sizeof(get_table_params));
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
        return;
    }

    memset(&set_table_params, 0, sizeof(LWSWITCH_SET_INGRESS_RESPONSE_TABLE));

    /* write a pattern */
    for (i = 0; i < LWSWITCH_INGRESS_RESPONSE_ENTRIES_MAX - 1; i+=2) 
    {
        set_table_params.entries[i].vcModeValid7_0 = 0x55;
        set_table_params.entries[i].vcModeValid15_8 = 0x2200;
        set_table_params.entries[i].vcModeValid17_16 = 1;
        set_table_params.entries[i].routePolicy = 1;
        set_table_params.entries[i].entryValid = 1;
    }

    set_table_params.portNum = valid_port;
    set_table_params.numEntries = LWSWITCH_INGRESS_RESPONSE_ENTRIES_MAX;

    for (i = 0; i < ROUTING_TABLE_ENTRIES / LWSWITCH_INGRESS_RESPONSE_ENTRIES_MAX; i++)
    {
        set_table_params.firstIndex = i * LWSWITCH_INGRESS_RESPONSE_ENTRIES_MAX;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE, &set_table_params, sizeof(set_table_params));
        ASSERT_EQ(status, LW_OK);
    }

    /* read the tables */
    get_table_params.nextIndex = 0;

    while (get_table_params.nextIndex < ROUTING_TABLE_ENTRIES)
    {
        /* get up to 256 nonzero entries at a time */
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INGRESS_RESPONSE_TABLE, &get_table_params, sizeof(get_table_params));
        ASSERT_EQ(status, LW_OK);

        /* check that we read back what we wrote */
        for (i = 0; i < get_table_params.numEntries; i++)
        {
            /* we wrote only even entries, the rest should be zero */
            ASSERT_FALSE(get_table_params.entries[i].idx % 2);

            ASSERT_EQ(set_table_params.entries[0].vcModeValid7_0,
                      get_table_params.entries[i].entry.vcModeValid7_0);
            ASSERT_EQ(set_table_params.entries[0].vcModeValid15_8,
                      get_table_params.entries[i].entry.vcModeValid15_8);
            ASSERT_EQ(set_table_params.entries[0].vcModeValid17_16,
                      get_table_params.entries[i].entry.vcModeValid17_16);
            ASSERT_EQ(set_table_params.entries[0].routePolicy,
                     get_table_params.entries[i].entry.routePolicy);
            ASSERT_EQ(set_table_params.entries[0].entryValid,
                     get_table_params.entries[i].entry.entryValid);
        }

        get_table_params.firstIndex = get_table_params.nextIndex;
    }
}

TEST_F(LWSwitchDeviceTest, IoctlGetIngressResponseTableBadInput)
{
    LWSWITCH_GET_INGRESS_RESPONSE_TABLE_PARAMS t;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    validPort(&valid_port);
    if (valid_port == LWSWITCH_ILWALID_PORT)
    {
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        //
        // Ports can only be invalid on LS10 because of Repeater Mode
        //
        if (!isArchLs10())
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        {
            FAIL() << "Found no valid ports.";
        }
    }

    /* this ioctl is only supported on sv10 */
    if (!isArchSv10())
    {
        t.firstIndex = 0;
        t.portNum = valid_port;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INGRESS_RESPONSE_TABLE, &t, sizeof(t));
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
        return;
    }

    /* Bad portNum */
    t.portNum = 0xffffffff;
    t.firstIndex = 0;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INGRESS_RESPONSE_TABLE, &t, sizeof(t));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    /* Bad first Index */
    t.portNum = valid_port;
    t.firstIndex = ROUTING_TABLE_ENTRIES;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INGRESS_RESPONSE_TABLE, &t, sizeof(t));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

}

TEST_F(LWSwitchDeviceTest, IoctlSetIngressResponseTable)
{
    LWSWITCH_INGRESS_RESPONSE_ENTRY e[ROUTING_TABLE_ENTRIES];
    LWSWITCH_SET_INGRESS_RESPONSE_TABLE t;
    struct first_num inputs[ROUTING_TABLE_ENTRIES];
    LwU64 linkMask = getLinkMask();
    LwU32 linkCount = getLinkCount();
    lwswitch_device *device = getDevice();
    LwU32 count;
    LW_STATUS status;
    LwU32 i, j, k;

    memset(&t, 0, sizeof(t));

    /* this ioctl is only supported on sv10 */
    if (!isArchSv10())
    {
        t.portNum = 0;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE, &t, sizeof(t));
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
        return;
    }

    for (i = 0; i < ROUTING_TABLE_ENTRIES; i++)
    {
        e[i].vcModeValid7_0 = 0;
        e[i].vcModeValid15_8 = 0;
        e[i].vcModeValid17_16 = 0;
        e[i].routePolicy = 1;
        e[i].entryValid = 1;
    }

    count = gen_first_num(inputs, ROUTING_TABLE_ENTRIES);

    for (i = 0; i < linkCount; i++)
    {
        if (LWBIT64(i) & linkMask)
        {
            // Loop over all first/num inputs
            for (j=0; j < count; j++)
            {
                t.portNum = i;
                t.firstIndex = inputs[j].first;

                if (verbose)
                    printf("\ttest %d: %d %d\n", j, inputs[j].first, inputs[j].num);

                do
                {
                    t.numEntries = LW_MIN(
                        LWSWITCH_INGRESS_RESPONSE_ENTRIES_MAX,
                        inputs[j].first + inputs[j].num - t.firstIndex);
                    if (t.numEntries > 0)
                    {
                        for (k=0; k < t.numEntries; k++)
                        {
                            t.entries[k] = e[t.firstIndex + k];
                        }
                        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE, &t, sizeof(t));
                        ASSERT_EQ(status, LW_OK);

                        t.firstIndex += t.numEntries;
                    }
                }
                while (t.numEntries > 0);
            }
        }
    }
}

TEST_F(LWSwitchDeviceTest, IoctlSetIngressResponseTableBadInput)
{
    LWSWITCH_SET_INGRESS_RESPONSE_TABLE t;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    LwU32 tmp;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 i;

    /* this ioctl is only supported on sv10 */
    if (!isArchSv10())
    {
        return;
    }

    validPort(&valid_port);
    if (valid_port == LWSWITCH_ILWALID_PORT)
    {
        FAIL() << "Found no valid ports.";
    }

    for (i = 0; i < LWSWITCH_INGRESS_RESPONSE_ENTRIES_MAX; i++)
    {
        t.entries[i].vcModeValid7_0 = 0;
        t.entries[i].vcModeValid15_8 = 0;
        t.entries[i].vcModeValid17_16 = 0;
        t.entries[i].routePolicy = 1;
        t.entries[i].entryValid = 1;
    }

    // Bad portNum
    t.portNum = 0xffffffff;
    t.firstIndex = 0;
    t.numEntries = 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE, &t, sizeof(t));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Bad first Index
    t.portNum = valid_port;
    t.firstIndex = ROUTING_TABLE_ENTRIES;
    t.numEntries = 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE, &t, sizeof(t));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Bad last index
    t.portNum = valid_port;
    t.firstIndex = ROUTING_TABLE_ENTRIES - 1;
    t.numEntries = 2;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE, &t, sizeof(t));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Too many entries
    t.portNum = valid_port;
    t.firstIndex = 0;
    t.numEntries = LWSWITCH_INGRESS_RESPONSE_ENTRIES_MAX + 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE, &t, sizeof(t));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Set up "t" for testing bad entry[0]
    t.portNum = valid_port;
    t.firstIndex = 0;
    t.numEntries = 1;

    // Bad route policy - not checked by kernel.
    tmp = t.entries[0].routePolicy;
    t.entries[0].routePolicy = 0xdead;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE, &t, sizeof(t));
    ASSERT_EQ(status, LW_OK);
    t.entries[0].routePolicy = tmp;

    // vcModeVald* are full register values and don't need to test them.
}

TEST_F(LWSwitchDeviceTest, IoctlGetIngressRequestTable)
{
    LWSWITCH_GET_INGRESS_REQUEST_TABLE_PARAMS get_table_params;
    LWSWITCH_SET_INGRESS_REQUEST_TABLE set_table_params;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 i;

    validPort(&valid_port);
    if (valid_port == LWSWITCH_ILWALID_PORT)
    {
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        //
        // Ports can only be invalid on LS10 because of Repeater Mode
        //
        if (!isArchLs10())
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        {
            FAIL() << "Found no valid ports.";
        }
    }

    get_table_params.firstIndex = 0;
    get_table_params.portNum = valid_port;

    /* this ioctl is only supported on sv10 */
    if (!isArchSv10())
    {
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INGRESS_REQUEST_TABLE, &get_table_params, sizeof(get_table_params));
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
        return;
    }

    memset(&set_table_params, 0, sizeof(LWSWITCH_SET_INGRESS_REQUEST_TABLE));

    /* write a pattern */
    for (i = 0; i < LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX; i+=2)
    {
        set_table_params.entries[i].vcModeValid7_0 = 0x55;
        set_table_params.entries[i].vcModeValid15_8 = 0x2200;
        set_table_params.entries[i].vcModeValid17_16 = 1;
        set_table_params.entries[i].mappedAddress = 0xaa5;
        set_table_params.entries[i].routePolicy = 1;
        set_table_params.entries[i].entryValid = 1;
    }

    set_table_params.portNum = valid_port;
    set_table_params.numEntries = LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX;

    for (i = 0; i < ROUTING_TABLE_ENTRIES / LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX; i++)
    {
        set_table_params.firstIndex = i * LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE, &set_table_params, sizeof(set_table_params));
        ASSERT_EQ(status, LW_OK);
    }

    /* read the table */
    get_table_params.nextIndex = 0;

    while (get_table_params.nextIndex < ROUTING_TABLE_ENTRIES)
    {
        /* get up to 256 nonzero entries at a time */
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INGRESS_REQUEST_TABLE, &get_table_params, sizeof(get_table_params));
        ASSERT_EQ(status, LW_OK);

        /* check that we read back what we wrote */
        for (i = 0; i < get_table_params.numEntries; i++)
        {
            /* we wrote only even entries, the rest should be zero */
            ASSERT_FALSE(get_table_params.entries[i].idx % 2);

            ASSERT_EQ(set_table_params.entries[0].vcModeValid7_0,
                      get_table_params.entries[i].entry.vcModeValid7_0);
            ASSERT_EQ(set_table_params.entries[0].vcModeValid15_8,
                      get_table_params.entries[i].entry.vcModeValid15_8);
            ASSERT_EQ(set_table_params.entries[0].vcModeValid17_16,
                      get_table_params.entries[i].entry.vcModeValid17_16);
            ASSERT_EQ(set_table_params.entries[0].mappedAddress,
                      get_table_params.entries[i].entry.mappedAddress);
            ASSERT_EQ(set_table_params.entries[0].routePolicy,
                     get_table_params.entries[i].entry.routePolicy);
            ASSERT_EQ(set_table_params.entries[0].entryValid,
                     get_table_params.entries[i].entry.entryValid);
        }

        get_table_params.firstIndex = get_table_params.nextIndex;
    }
}

TEST_F(LWSwitchDeviceTest, IoctlGetIngressRequestTableBadInput)
{
    LWSWITCH_GET_INGRESS_REQUEST_TABLE_PARAMS t;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    validPort(&valid_port);
    if (valid_port == LWSWITCH_ILWALID_PORT)
    {
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        //
        // Ports can only be invalid on LS10 because of Repeater Mode
        //
        if (!isArchLs10())
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        {
            FAIL() << "Found no valid ports.";
        }
    }

    /* this ioctl is only supported on sv10 */
    if (!isArchSv10())
    {
        t.portNum = valid_port;
        t.firstIndex = 0;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INGRESS_REQUEST_TABLE, &t, sizeof(t));
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
        return;
    }

    /* Bad portNum */
    t.portNum = 0xffffffff;
    t.firstIndex = 0;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INGRESS_REQUEST_TABLE, &t, sizeof(t));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    /* Bad first Index */
    t.portNum = valid_port;
    t.firstIndex = ROUTING_TABLE_ENTRIES;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INGRESS_REQUEST_TABLE, &t, sizeof(t));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
}

TEST_F(LWSwitchDeviceTest, IoctlSetIngressRequestTable)
{
    LWSWITCH_INGRESS_REQUEST_ENTRY e[ROUTING_TABLE_ENTRIES];
    LWSWITCH_SET_INGRESS_REQUEST_TABLE t;
    struct first_num inputs[ROUTING_TABLE_ENTRIES];
    LwU64 linkMask = getLinkMask();
    LwU32 linkCount = getLinkCount();
    LwU32 count;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 i, j, k;

    memset(&t, 0, sizeof(t));

    /* this ioctl is only supported on sv10 */
    if (!isArchSv10())
    {
        t.portNum = 0;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE, &t, sizeof(t));
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
        return;
    }

    for (i = 0; i < ROUTING_TABLE_ENTRIES; i++)
    {
        e[i].vcModeValid7_0 = 0;
        e[i].vcModeValid15_8 = 0;
        e[i].vcModeValid17_16 = 0;
        e[i].mappedAddress = 0;
        e[i].routePolicy = 0;
        e[i].entryValid = 1;
    }

    count = gen_first_num(inputs, ROUTING_TABLE_ENTRIES);

    for (i = 0; i < linkCount; i++)
    {
        if (LWBIT64(i) & linkMask)
        {
            // Loop over all first/num inputs
            for (j=0; j < count; j++)
            {
                t.portNum = i;
                t.firstIndex = inputs[j].first;

                if (verbose)
                    printf("\ttest %d: %d %d %d\n", j, i, inputs[j].first, inputs[j].num);

                do
                {
                    t.numEntries = LW_MIN(
                        LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX,
                        inputs[j].first + inputs[j].num - t.firstIndex);
                    if (t.numEntries > 0)
                    {
                        for (k=0; k < t.numEntries; k++)
                        {
                            t.entries[k] = e[t.firstIndex + k];
                        }
                        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE, &t, sizeof(t));
                        ASSERT_EQ(status, LW_OK);

                        t.firstIndex += t.numEntries;
                    }
                }
                while (t.numEntries > 0);
            }
        }
    }
}

TEST_F(LWSwitchDeviceTest, IoctlSetIngressRequestTableBadInput)
{
    LWSWITCH_SET_INGRESS_REQUEST_TABLE t;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    LwU32 tmp;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 i;

    /* this ioctl is only supported on sv10 */
    if (!isArchSv10())
    {
        return;
    }

    validPort(&valid_port);
    if (valid_port == LWSWITCH_ILWALID_PORT)
    {
        FAIL() << "Found no valid port.";
    }

    for (i = 0; i < LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX; i++)
    {
        t.entries[i].vcModeValid7_0 = 0;
        t.entries[i].vcModeValid15_8 = 0;
        t.entries[i].vcModeValid17_16 = 0;
        t.entries[i].mappedAddress = 0;
        t.entries[i].routePolicy = 0;
        t.entries[i].entryValid = 1;
    }

    // Bad portNum
    t.portNum = 0xffffffff;
    t.firstIndex = 0;
    t.numEntries = 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE, &t, sizeof(t));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Bad first Index
    t.portNum = valid_port;
    t.firstIndex = ROUTING_TABLE_ENTRIES;
    t.numEntries = 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE, &t, sizeof(t));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Bad last index
    t.portNum = valid_port;
    t.firstIndex = ROUTING_TABLE_ENTRIES - 1;
    t.numEntries = 2;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE, &t, sizeof(t));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Too many entries
    t.portNum = valid_port;
    t.firstIndex = 0;
    t.numEntries = LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX + 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE, &t, sizeof(t));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Set up "t" for testing bad entry[0]
    t.portNum = valid_port;
    t.firstIndex = 0;
    t.numEntries = 1;

    // Bad route policy - not checked by kernel.
    tmp = t.entries[0].routePolicy;
    t.entries[0].routePolicy = 0xdead;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE, &t, sizeof(t));
    ASSERT_EQ(status, LW_OK);
    t.entries[0].routePolicy = tmp;

    // Bad mappedAddress - not checked by kernel.
    tmp = t.entries[0].mappedAddress;
    t.entries[0].mappedAddress = 0xdeadbeef;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE, &t, sizeof(t));
    ASSERT_EQ(status, LW_OK);
    t.entries[0].mappedAddress = tmp;

    // vcModeVald* are full register values and don't need to test them.
}

TEST_F(LWSwitchDeviceTest, IoctlSetIngressRequestValid)
{
    LwBool e[ROUTING_TABLE_ENTRIES];
    LWSWITCH_SET_INGRESS_REQUEST_VALID p;
    struct first_num inputs[ROUTING_TABLE_ENTRIES];
    LwU64 linkMask = getLinkMask();;
    LwU32 linkCount = getLinkCount();
    LwU32 count;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 i, j, k;

    memset(&p, 0, sizeof(p));

    /* this ioctl is only supported on sv10 */
    if (!isArchSv10())
    {
        p.portNum = 0;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_REQUEST_VALID, &p, sizeof(p));
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
        return;
    }

    // Alternate valid/invalid
    for (i=0; i < ROUTING_TABLE_ENTRIES; i++)
    {
        e[i] = i & 1;
    }

    count = gen_first_num(inputs, ROUTING_TABLE_ENTRIES);

    for (i = 0; i < linkCount; i++)
    {
        if (LWBIT64(i) & linkMask)
        {
            // Loop over all first/num inputs
            for (j=0; j < count; j++)
            {
                p.portNum = i;
                p.firstIndex = inputs[j].first;

                if (verbose)
                    printf("\ttest %d: %d %d %d\n", j, i, p.firstIndex, p.numEntries);

                do
                {
                    p.numEntries = LW_MIN(
                        LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX,
                        inputs[j].first + inputs[j].num - p.firstIndex);
                    if (p.numEntries > 0)
                    {
                        for (k=0; k < p.numEntries; k++)
                        {
                            p.entryValid[k] = e[p.firstIndex + k];
                        }
                        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_REQUEST_VALID, &p, sizeof(p));
                        ASSERT_EQ(status, LW_OK);

                        p.firstIndex += p.numEntries;
                    }
                }
                while (p.numEntries > 0);
            }
        }
    }
}

TEST_F(LWSwitchDeviceTest, IoctlSetIngressRequestValidBadInput)
{
    LWSWITCH_SET_INGRESS_REQUEST_VALID p;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 i;

    if (!isArchSv10())
    {
        return;
    }

    validPort(&valid_port);
    if (valid_port == LWSWITCH_ILWALID_PORT)
    {
        FAIL() << "Found no valid port.";
    }

    memset(&p, 0, sizeof(p));

    // Alternate valid/invalid
    for (i=0; i < LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX; i++)
    {
        p.entryValid[i] = i & 1;
    }

    // Invalid port
    p.portNum = 0xbaddad;
    p.firstIndex = 0;
    p.numEntries = 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_REQUEST_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Bad first index
    p.portNum = valid_port;
    p.firstIndex = ROUTING_TABLE_ENTRIES;
    p.numEntries = 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_REQUEST_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Bad last index
    p.portNum = valid_port;
    p.firstIndex = ROUTING_TABLE_ENTRIES - 1;
    p.numEntries = 2;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_REQUEST_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Too many entries
    p.portNum = valid_port;
    p.firstIndex = 0;
    p.numEntries = LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX + 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_INGRESS_REQUEST_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
}

TEST_F(LWSwitchDeviceTest, IoctlSetGangedLinkTable)
{
    LWSWITCH_SET_GANGED_LINK_TABLE p;
    LwU64 linkMask = getLinkMask();
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    memset(&p, 0, sizeof(p));
    p.link_mask = (LwU32) linkMask;
    // p.entries = 0 isn't valid but driver doesn't know that.

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_GANGED_LINK_TABLE, &p, sizeof(p));
    if (isArchSv10())
    {
        ASSERT_EQ(status, LW_OK);
    }
    else
    {
        // Unsupported
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
    }
}

TEST_F(LWSwitchDeviceTest, IoctlSetGangedLinkTableBadInput)
{
    LWSWITCH_SET_GANGED_LINK_TABLE p;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (!isArchSv10())
    {
        return;
    }

    // Invalid link mask - unchecked by driver lwrrently
    memset(&p, 0, sizeof(p));
    p.link_mask = 0xdead0000;
    // p.entries = 0 isn't valid but driver doesn't know that.

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_GANGED_LINK_TABLE, &p, sizeof(p));
    ASSERT_NE(status, LW_OK);
}

TEST_F(LWSwitchDeviceTest, IoctlSetRemapPolicy)
{
    LWSWITCH_SET_REMAP_POLICY p;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LWSWITCH_TABLE_SELECT_REMAP tableSelect;
    LwU32 remap_policy_table_size;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    for (tableSelect = LWSWITCH_TABLE_SELECT_REMAP_PRIMARY;
         tableSelect <= LWSWITCH_TABLE_SELECT_REMAP_MULTICAST;
         tableSelect = getRemapTableNext(tableSelect))
    {
        remap_policy_table_size = getRemapTableSize(tableSelect);

        printf("[ SUBTEST ] Remap: %s (size 0x%x)\n",
             (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_PRIMARY ? "PRIMARY" :
             (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_EXTA    ? "EXTA" :
             (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_EXTB    ? "EXTB" :
             (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_MULTICAST ? "MULTICAST" :
             "ERROR:Unknown REMAP table")))),
             remap_policy_table_size);

        //
        // Confirm per-arch expected REMAP table support
        //
        if (isArchSv10())
        {
            ASSERT_EQ(remap_policy_table_size, 0);
        }
        else if (isArchLr10())
        {
            if (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_PRIMARY)
            {
                ASSERT_NE(remap_policy_table_size, 0);
            }
            else
            {
                ASSERT_EQ(remap_policy_table_size, 0);
            }
        }
        else
        {
            ASSERT_NE(remap_policy_table_size, 0);
        }

        memset(&p, 0, sizeof(p));

        p.portNum = valid_port;
        p.firstIndex = 0;
        p.numEntries = LWSWITCH_REMAP_POLICY_ENTRIES_MAX;

        p.tableSelect = tableSelect;
        remap_policy_table_size = getRemapTableSize(p.tableSelect);

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));

        if (remap_policy_table_size == 0)
        {
            // Unsupported
            ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
        }
        else
        {
            ASSERT_EQ(status, LW_OK);
        }
    }
}

TEST_F(LWSwitchDeviceTest, IoctlSetRemapPolicyRepeaterMode)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LWSWITCH_SET_REMAP_POLICY p;
    LwU32 repeater_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (!isArchLs10())
    {
        return;
    }

    repeaterPort(&repeater_port);
    if (repeater_port == LWSWITCH_ILWALID_PORT)
    {
        printf("[ SKIPPED  ] No port in Repeater Mode.\n");
        return;
    }

    memset(&p, 0, sizeof(p));

    p.portNum = repeater_port;
    p.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_PRIMARY;
    p.firstIndex = 0;
    p.numEntries = LWSWITCH_REMAP_POLICY_ENTRIES_MAX;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
}

TEST_F(LWSwitchDeviceTest, IoctlSetRemapPolicyBadInput)
{
    LWSWITCH_SET_REMAP_POLICY p;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    LwU32 remap_policy_table_size;
    LWSWITCH_TABLE_SELECT_REMAP tableSelect;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (isArchSv10())
    {
        return;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    memset(&p, 0, sizeof(p));

    // Invalid table
    p.portNum = valid_port;
    p.tableSelect = (LWSWITCH_TABLE_SELECT_REMAP) ~0;
    p.firstIndex = 0;
    p.numEntries = 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);

    for (tableSelect = LWSWITCH_TABLE_SELECT_REMAP_PRIMARY;
         tableSelect <= LWSWITCH_TABLE_SELECT_REMAP_MULTICAST;
         tableSelect = getRemapTableNext(tableSelect))
    {
         remap_policy_table_size = getRemapTableSize(tableSelect);

         printf("[ SUBTEST ] Remap: %s (size 0x%x)\n",
              (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_PRIMARY ? "PRIMARY" :
              (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_EXTA    ? "EXTA" :
              (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_EXTB    ? "EXTB" :
              (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_MULTICAST ? "MULTICAST" :
              "ERROR:Unknown REMAP table")))),
              remap_policy_table_size);

        memset(&p, 0, sizeof(p));

        // Invalid port
        p.portNum = 0xbaddad;
        p.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_PRIMARY;
        p.firstIndex = 0;
        p.numEntries = 1;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_NE(status, LW_OK);

        // Bad first index
        p.portNum = valid_port;
        p.tableSelect = tableSelect;
        p.firstIndex = remap_policy_table_size;
        p.numEntries = 1;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_NE(status, LW_OK);

        // Bad last index
        p.portNum = valid_port;
        p.tableSelect = tableSelect;
        p.firstIndex = remap_policy_table_size - 1;
        p.numEntries = 2;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_NE(status, LW_OK);

        // Too many entries
        p.portNum = valid_port;
        p.tableSelect = tableSelect;
        p.firstIndex = 0;
        p.numEntries = LWSWITCH_REMAP_POLICY_ENTRIES_MAX + 1;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_NE(status, LW_OK);

        // Invalid flags
        p.portNum = valid_port;
        p.tableSelect = tableSelect;
        p.firstIndex = 0;
        p.numEntries = 1;
        p.remapPolicy[0].flags = ~0;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_NE(status, LW_OK);

        // Invalid target ID
        p.portNum = valid_port;
        p.tableSelect = tableSelect;
        p.firstIndex = 0;
        p.numEntries = 1;
        p.remapPolicy[0].flags = 0;
        p.remapPolicy[0].targetId = ~0;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_NE(status, LW_OK);

        // Invalid IRL select
        p.portNum = valid_port;
        p.tableSelect = tableSelect;
        p.firstIndex = 0;
        p.numEntries = 1;
        p.remapPolicy[0].targetId = 0;
        p.remapPolicy[0].irlSelect = ~0;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_NE(status, LW_OK);

        // Invalid remap address
        p.portNum = valid_port;
        p.tableSelect = tableSelect;
        p.firstIndex = 0;
        p.numEntries = 1;
        p.remapPolicy[0].flags = LWSWITCH_REMAP_POLICY_FLAGS_REMAP_ADDR;
        p.remapPolicy[0].irlSelect = 0;
        p.remapPolicy[0].address = ~0;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_NE(status, LW_OK);

        // Invalid request ctxt mask
        p.portNum = valid_port;
        p.tableSelect = tableSelect;
        p.firstIndex = 0;
        p.numEntries = 1;
        p.remapPolicy[0].flags = LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_CHECK;
        p.remapPolicy[0].address = 0;
        p.remapPolicy[0].reqCtxMask = ~0;
        p.remapPolicy[0].reqCtxChk = 0;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_NE(status, LW_OK);

        // Invalid request ctxt check
        p.portNum = valid_port;
        p.tableSelect = tableSelect;
        p.firstIndex = 0;
        p.numEntries = 1;
        p.remapPolicy[0].flags = LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_CHECK;
        p.remapPolicy[0].reqCtxMask = 0;
        p.remapPolicy[0].reqCtxChk = ~0;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_NE(status, LW_OK);

        // Invalid request ctxt replace
        p.portNum = valid_port;
        p.tableSelect = tableSelect;
        p.firstIndex = 0;
        p.numEntries = 1;
        p.remapPolicy[0].flags = LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_REPLACE;
        p.remapPolicy[0].reqCtxChk = 0;
        p.remapPolicy[0].reqCtxRep = ~0;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_NE(status, LW_OK);

        // Invalid address check base
        p.portNum = valid_port;
        p.tableSelect = tableSelect;
        p.firstIndex = 0;
        p.numEntries = 1;
        p.remapPolicy[0].flags = LWSWITCH_REMAP_POLICY_FLAGS_ADR_BASE;
        p.remapPolicy[0].reqCtxRep = 0;
        p.remapPolicy[0].addressBase = ~0;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_NE(status, LW_OK);

        // Invalid address check limit
        p.portNum = valid_port;
        p.tableSelect = tableSelect;
        p.firstIndex = 0;
        p.numEntries = 1;
        p.remapPolicy[0].flags = LWSWITCH_REMAP_POLICY_FLAGS_ADR_BASE;
        p.remapPolicy[0].addressBase = 0;
        p.remapPolicy[0].addressLimit = ~0;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_NE(status, LW_OK);

        // Invalid address check range
        p.portNum = valid_port;
        p.tableSelect = tableSelect;
        p.firstIndex = 0;
        p.numEntries = 1;
        p.remapPolicy[0].flags = LWSWITCH_REMAP_POLICY_FLAGS_ADR_BASE;
        p.remapPolicy[0].addressBase = 0x100000;
        p.remapPolicy[0].addressLimit = 0;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_NE(status, LW_OK);

        // Invalid to enable offset without range check
        p.portNum = valid_port;
        p.tableSelect = tableSelect;
        p.firstIndex = 0;
        p.numEntries = 1;
        p.remapPolicy[0].flags = LWSWITCH_REMAP_POLICY_FLAGS_ADR_OFFSET;
        p.remapPolicy[0].addressBase = 0;
        p.remapPolicy[0].addressOffset = 0;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_NE(status, LW_OK);

        // Invalid address check address
        p.portNum = valid_port;
        p.tableSelect = tableSelect;
        p.firstIndex = 0;
        p.numEntries = 1;
        p.remapPolicy[0].flags =
            LWSWITCH_REMAP_POLICY_FLAGS_ADR_BASE |
            LWSWITCH_REMAP_POLICY_FLAGS_ADR_OFFSET;
        p.remapPolicy[0].addressOffset = ~0;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_NE(status, LW_OK);

        // Invalid offset relative to limit overflows 64G bounds
        // Address offset function deprecated post-LR10
        p.portNum = valid_port;
        p.tableSelect = tableSelect;
        p.firstIndex = 0;
        p.numEntries = 1;
        p.remapPolicy[0].flags = 
            LWSWITCH_REMAP_POLICY_FLAGS_ADR_OFFSET |
            LWSWITCH_REMAP_POLICY_FLAGS_ADR_BASE;
        p.remapPolicy[0].addressBase = 0x0;
        p.remapPolicy[0].addressLimit = 0x0FFFF00000;
        p.remapPolicy[0].addressOffset = 0x100000;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_NE(status, LW_OK);

        if (!isArchLr10())
        {
            // Address offset deprecated post-LR10
            p.portNum = valid_port;
            p.tableSelect = tableSelect;
            p.firstIndex = 0;
            p.numEntries = 1;
            p.remapPolicy[0].flags = 0;
            p.remapPolicy[0].addressBase = 0x0;
            p.remapPolicy[0].addressLimit = 0x0;
            p.remapPolicy[0].addressOffset = 0x100000;

            status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
            ASSERT_NE(status, LW_OK);

            // Reflective mapping only allowed on MULTICAST REMAP
            if (tableSelect != LWSWITCH_TABLE_SELECT_REMAP_MULTICAST)
            {
                p.portNum = valid_port;
                p.tableSelect = tableSelect;
                p.firstIndex = 0;
                p.numEntries = 1;
                p.remapPolicy[0].flags = LWSWITCH_REMAP_POLICY_FLAGS_REFLECTIVE;
                p.remapPolicy[0].addressOffset = 0x0;

                status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &p, sizeof(p));
                ASSERT_NE(status, LW_OK);
            }
        }
    }
}

TEST_F(LWSwitchDeviceTest, IoctlGetRemapPolicy)
{
    LWSWITCH_GET_REMAP_POLICY_PARAMS get_table_params;
    LWSWITCH_SET_REMAP_POLICY set_table_params;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    LwU32 remap_num_entries = LWSWITCH_REMAP_POLICY_ENTRIES_MAX;
    LwU32 remap_policy_table_size;
    LWSWITCH_TABLE_SELECT_REMAP tableSelect;
    LwU64 address;
    LwU64 addressOffset;
    LwU64 addressBase;
    LwU64 addressLimit;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 i;
    LwU32 count;
    LwU32 flags =
                LWSWITCH_REMAP_POLICY_FLAGS_REMAP_ADDR |
                LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_CHECK |
                LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_REPLACE |
                LWSWITCH_REMAP_POLICY_FLAGS_ADR_BASE;

    if (isArchLr10())
    {
        address = 0x7ff000000000;
        addressBase = 0xf00000;
        addressLimit = 0xfff00000;
        // Only LR10 supports AddressOffset
        addressOffset = 0xff00000;
        flags |= LWSWITCH_REMAP_POLICY_FLAGS_ADR_OFFSET;
    }
    else
    {
        address = 0xfff8000000000;
        addressBase = 0x1e00000;
        addressLimit = 0x7fffe00000;
        // LS10 ignores AddressOffset and does not support _ADR_OFFSET
        addressOffset = 0x0000000;
        flags |= LWSWITCH_REMAP_POLICY_FLAGS_ADDR_TYPE;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    memset(&get_table_params, 0, sizeof(LWSWITCH_GET_REMAP_POLICY_PARAMS));

    get_table_params.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_PRIMARY;

    /* this ioctl is only supported on lr10 */
    if (isArchSv10())
    {
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_REMAP_POLICY, &get_table_params, sizeof(get_table_params));
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
        return;
    }
    /* lr10 fmodel does not support table readback. Hence, test only the ioctl */
    if (isArchLr10() && isFmodel())
    {
        get_table_params.firstIndex = 0;
        get_table_params.portNum = valid_port;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_REMAP_POLICY, &get_table_params, sizeof(get_table_params));
        ASSERT_EQ(status, LW_OK);
        return;
    }

    for (tableSelect = LWSWITCH_TABLE_SELECT_REMAP_PRIMARY;
         tableSelect <= LWSWITCH_TABLE_SELECT_REMAP_MULTICAST;
         tableSelect = getRemapTableNext(tableSelect))
    {
        remap_policy_table_size = getRemapTableSize(tableSelect);
        printf("[ SUBTEST ] Remap: %s (size 0x%x)\n",
            (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_PRIMARY ? "PRIMARY" :
            (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_EXTA    ? "EXTA" :
            (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_EXTB    ? "EXTB" :
            (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_MULTICAST ? "MULTICAST" :
            "ERROR:Unknown REMAP table")))),
            remap_policy_table_size);

        memset(&get_table_params, 0, sizeof(LWSWITCH_GET_REMAP_POLICY_PARAMS));

        get_table_params.tableSelect = tableSelect;

        if (remap_policy_table_size == 0)
        {
            printf("[ SKIPPED  ] REMAP %s not supported.\n",
                (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_PRIMARY ? "PRIMARY" :
                (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_EXTA    ? "EXTA" :
                (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_EXTB    ? "EXTB" :
                (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_MULTICAST ? "MULTICAST" :
                "ERROR:Unknown REMAP table")))));
            continue;
        }

        // Set random table size
        count = remap_policy_table_size / remap_num_entries;
        remap_policy_table_size = ((rand() % count) + 1) * remap_num_entries;

        memset(&set_table_params, 0, sizeof(LWSWITCH_SET_REMAP_POLICY));

        // Set random entries in REMAP Policy Table.
        for (i = 0; i < remap_num_entries; i++)
        {
            set_table_params.remapPolicy[i].irlSelect =
                (rand() % DRF_MASK(LW_INGRESS_REMAPTABDATA0_IRL_SEL));
            set_table_params.remapPolicy[i].reqCtxMask =
                (rand() % DRF_MASK(LW_INGRESS_REMAPTABDATA1_REQCTXT_MSK));
            set_table_params.remapPolicy[i].reqCtxChk =
                (rand() % DRF_MASK(LW_INGRESS_REMAPTABDATA1_REQCTXT_CHK));
            set_table_params.remapPolicy[i].reqCtxRep =
                (rand() % DRF_MASK(LW_INGRESS_REMAPTABDATA2_REQCTXT_REP));
            if (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_MULTICAST)
            {
                //
                // TODO: Bug #3252746: figure out how to handle LR10 vs LS10 
                // REMAP register differences
                //
                set_table_params.remapPolicy[i].targetId =
                    (rand() % DRF_MASK(6:0));
            }
            else
            {
                set_table_params.remapPolicy[i].targetId =
                    (rand() % DRF_MASK(LW_INGRESS_REMAPTABDATA4_TGTID));
            }

            set_table_params.remapPolicy[i].entryValid = rand() % 2;
            set_table_params.remapPolicy[i].flags = flags;
            set_table_params.remapPolicy[i].address = address;
            set_table_params.remapPolicy[i].addressOffset = addressOffset;
            set_table_params.remapPolicy[i].addressBase = addressBase;
            set_table_params.remapPolicy[i].addressLimit = addressLimit;
        }

        set_table_params.portNum = valid_port;
        set_table_params.tableSelect = tableSelect;
        set_table_params.numEntries = remap_num_entries;

        for (i = 0; i < remap_policy_table_size / remap_num_entries; i++)
        {
            set_table_params.firstIndex = i * remap_num_entries;
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &set_table_params, sizeof(set_table_params));
            ASSERT_EQ(status, LW_OK);
        }

        /* read the tables */
        get_table_params.tableSelect = tableSelect;
        get_table_params.firstIndex = 0;
        get_table_params.nextIndex = 0;
        get_table_params.portNum = valid_port;

        while (get_table_params.nextIndex < remap_policy_table_size)
        {
            /* get up to 64 nonzero entries at a time */
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_REMAP_POLICY, &get_table_params, sizeof(get_table_params));

            /* check that we read back what we wrote */
            ASSERT_EQ(set_table_params.numEntries, get_table_params.numEntries);

            for (i = 0; i < get_table_params.numEntries; i++)
            {
                ASSERT_EQ(memcmp(&set_table_params.remapPolicy[i],
                    &get_table_params.entry[i], sizeof(LWSWITCH_REMAP_POLICY_ENTRY)), 0);
            }

            get_table_params.firstIndex = get_table_params.nextIndex;
        }
    }
}

TEST_F(LWSwitchDeviceTest, IoctlGetRemapPolicyRepeaterMode)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LW_STATUS status;
    LwU32 repeater_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LWSWITCH_GET_REMAP_POLICY_PARAMS p;

    if (!isArchLs10())
    {
        return;
    }

    repeaterPort(&repeater_port);
    if (repeater_port == LWSWITCH_ILWALID_PORT)
    {
        printf("[ SKIPPED  ] No port in Repeater Mode.\n");
        return;
    }

    memset(&p, 0, sizeof(p));

    p.firstIndex = 0;
    p.portNum = repeater_port;
    p.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_PRIMARY;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_REMAP_POLICY, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
}

TEST_F(LWSwitchDeviceTest, IoctlGetRemapPolicyBadInput)
{
    LWSWITCH_GET_REMAP_POLICY_PARAMS p;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    LwU32 remap_policy_table_size;
    LWSWITCH_TABLE_SELECT_REMAP tableSelect;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (isArchSv10())
    {
        return;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    // Invalid table
    p.portNum = valid_port;
    p.tableSelect = (LWSWITCH_TABLE_SELECT_REMAP) ~0;
    p.firstIndex = 0;
    p.numEntries = 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_REMAP_POLICY, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);

    for (tableSelect = LWSWITCH_TABLE_SELECT_REMAP_PRIMARY;
         tableSelect <= LWSWITCH_TABLE_SELECT_REMAP_MULTICAST;
         tableSelect = getRemapTableNext(tableSelect))
    {
        remap_policy_table_size = getRemapTableSize(tableSelect);
        printf("[ SUBTEST ] Remap: %s (size 0x%x)\n",
            (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_PRIMARY ? "PRIMARY" :
            (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_EXTA    ? "EXTA" :
            (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_EXTB    ? "EXTB" :
            (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_MULTICAST ? "MULTICAST" :
            "ERROR:Unknown REMAP table")))),
            remap_policy_table_size);

        if (remap_policy_table_size == 0)
        {
            printf("[ SKIPPED  ] REMAP %s not supported.\n",
                (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_PRIMARY ? "PRIMARY" :
                (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_EXTA    ? "EXTA" :
                (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_EXTB    ? "EXTB" :
                (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_MULTICAST ? "MULTICAST" :
                "ERROR:Unknown REMAP table")))));
            continue;
        }

        // Invalid port
        p.portNum = 0xbaddad;
        p.tableSelect = tableSelect;
        p.firstIndex = 0;
        p.numEntries = 1;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

        // Bad first Index
        p.portNum = valid_port;
        p.tableSelect = tableSelect;
        p.firstIndex = remap_policy_table_size;
        p.numEntries = 1;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_REMAP_POLICY, &p, sizeof(p));
        ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
    }
}

/* 
 * Test procedure
 *
 * Fill REMAP Policy Table with random data using SET_REMAP_POLICY call.
 * Set the valid bits using SET_REMAP_POLICY_VALID call.
 * GET REMAP entries using GET_REMAP_POLICY call.
 * Compare valid bits against previously set values.
 * Verify that SET_REMAP_POLICY_VALID API doesn't disturb other data fields.
 */

TEST_F(LWSwitchDeviceTest, IoctlSetRemapPolicyRequestValid)
{
    LWSWITCH_SET_REMAP_POLICY_VALID set_valid_params;
    LWSWITCH_SET_REMAP_POLICY set_table_params;
    LWSWITCH_GET_REMAP_POLICY_PARAMS get_table_params;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    LwU32 remap_policy_table_size;
    LWSWITCH_TABLE_SELECT_REMAP tableSelect;
    LwU32 remap_num_entries = LWSWITCH_REMAP_POLICY_ENTRIES_MAX;
    LwU64 address;
    LwU64 addressOffset;
    LwU64 addressBase;
    LwU64 addressLimit;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 i;
    LwU32 count;
    LwU32 flags =
                LWSWITCH_REMAP_POLICY_FLAGS_REMAP_ADDR |
                LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_CHECK |
                LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_REPLACE |
                LWSWITCH_REMAP_POLICY_FLAGS_ADR_BASE;

    if (isArchLr10())
    {
        address = 0x7ff000000000;
        addressBase = 0xf00000;
        addressLimit = 0xfff00000;
        // Only LR10 supports AddressOffset
        addressOffset = 0xff00000;
        flags |= LWSWITCH_REMAP_POLICY_FLAGS_ADR_OFFSET;
    }
    else
    {
        address = 0xfff8000000000;
        addressBase = 0x1e00000;
        addressLimit = 0x7fffe00000;
        // LS10 ignores AddressOffset and does not support _ADR_OFFSET
        addressOffset = 0x0000000;
        flags |= LWSWITCH_REMAP_POLICY_FLAGS_ADDR_TYPE;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    for (tableSelect = LWSWITCH_TABLE_SELECT_REMAP_PRIMARY;
         tableSelect <= LWSWITCH_TABLE_SELECT_REMAP_MULTICAST;
         tableSelect = getRemapTableNext(tableSelect))
    {
        remap_policy_table_size = getRemapTableSize(tableSelect);

        printf("[ SUBTEST ] Remap: %s (size 0x%x)\n",
             (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_PRIMARY ? "PRIMARY" :
             (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_EXTA    ? "EXTA" :
             (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_EXTB    ? "EXTB" :
             (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_MULTICAST ? "MULTICAST" :
             "ERROR:Unknown REMAP table")))),
             remap_policy_table_size);

        memset(&set_valid_params, 0, sizeof(LWSWITCH_SET_REMAP_POLICY_VALID));

        set_table_params.tableSelect = tableSelect;

        /* this ioctl is only supported on lr10 */
        if (isArchSv10())
        {
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY_VALID, &set_valid_params, sizeof(set_valid_params));
            ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
            return;
        }

        /* lr10 fmodel does not support table readback. Hence, test only the ioctl */
        if (isArchLr10() && isFmodel())
        {
            set_valid_params.portNum = valid_port;
            set_valid_params.numEntries = remap_num_entries;

            status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY_VALID, &set_valid_params, sizeof(set_valid_params));
            ASSERT_EQ(status, LW_OK);
            return;
        }

        if (remap_policy_table_size == 0)
        {
            printf("[ SKIPPED  ] REMAP %s not supported.\n",
                (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_PRIMARY ? "PRIMARY" :
                (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_EXTA    ? "EXTA" :
                (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_EXTB    ? "EXTB" :
                (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_MULTICAST ? "MULTICAST" :
                "ERROR:Unknown REMAP table")))));
            continue;
        }

        // Set random table size
        count = remap_policy_table_size / remap_num_entries;
        remap_policy_table_size = ((rand() % count) + 1) * remap_num_entries;

        memset(&set_table_params, 0, sizeof(LWSWITCH_SET_REMAP_POLICY));

        // Set random entries in REMAP Policy Table
        for (i = 0; i < remap_num_entries; i++)
        {
            set_table_params.remapPolicy[i].irlSelect =
                (rand() % DRF_MASK(LW_INGRESS_REMAPTABDATA0_IRL_SEL));
            set_table_params.remapPolicy[i].reqCtxMask =
                (rand() % DRF_MASK(LW_INGRESS_REMAPTABDATA1_REQCTXT_MSK));
            set_table_params.remapPolicy[i].reqCtxChk =
                (rand() % DRF_MASK(LW_INGRESS_REMAPTABDATA1_REQCTXT_CHK));
            set_table_params.remapPolicy[i].reqCtxRep =
                (rand() % DRF_MASK(LW_INGRESS_REMAPTABDATA2_REQCTXT_REP));
            if (tableSelect == LWSWITCH_TABLE_SELECT_REMAP_MULTICAST)
            {
                //
                // TODO: Bug #3252746: figure out how to handle LR10 vs LS10 
                // REMAP register differences
                //
                set_table_params.remapPolicy[i].targetId =
                    (rand() % DRF_MASK(6:0));
            }
            else
            {
                set_table_params.remapPolicy[i].targetId =
                    (rand() % DRF_MASK(LW_INGRESS_REMAPTABDATA4_TGTID));
            }

            set_table_params.remapPolicy[i].entryValid = rand() % 2;
            set_table_params.remapPolicy[i].flags = flags;
            set_table_params.remapPolicy[i].address = address;
            set_table_params.remapPolicy[i].addressOffset = addressOffset;
            set_table_params.remapPolicy[i].addressBase = addressBase;
            set_table_params.remapPolicy[i].addressLimit = addressLimit;
        }

        set_table_params.portNum = valid_port;
        set_table_params.tableSelect = tableSelect;
        set_table_params.numEntries = remap_num_entries;

        for (i = 0; i < remap_policy_table_size / remap_num_entries; i++)
        {
            set_table_params.firstIndex = i * remap_num_entries;
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY, &set_table_params, sizeof(set_table_params));
            ASSERT_EQ(status, LW_OK);
        }

        // Set valid bits in the REMAP Policy table.
        for (i = 0; i < remap_num_entries; i++)
        {
            set_valid_params.entryValid[i] = rand() % 2; // set 0 or 1
        }

        set_valid_params.portNum = valid_port;
        set_valid_params.tableSelect = tableSelect;
        set_valid_params.numEntries = remap_num_entries;

        for (i = 0; i < remap_policy_table_size / remap_num_entries; i++)
        {
            set_valid_params.firstIndex = i * remap_num_entries;
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY_VALID, &set_valid_params, sizeof(set_valid_params));
            ASSERT_EQ(status, LW_OK);
        }

        /* read the tables */
        memset(&get_table_params, 0, sizeof(LWSWITCH_GET_REMAP_POLICY_PARAMS));

        get_table_params.firstIndex = 0;
        get_table_params.nextIndex = 0;
        get_table_params.portNum = valid_port;
        get_table_params.tableSelect = tableSelect;

        while (get_table_params.nextIndex < remap_policy_table_size)
        {
            /* get up to 64 nonzero entries at a time */
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_REMAP_POLICY, &get_table_params, sizeof(get_table_params));
            ASSERT_EQ(status, LW_OK);
            ASSERT_EQ(set_valid_params.numEntries, get_table_params.numEntries);

            for (i = 0; i < get_table_params.numEntries; i++)
            {
                /* verify valid entries */
                ASSERT_EQ(set_valid_params.entryValid[i],
                    get_table_params.entry[i].entryValid);

                /* verify that SET_REMAP_POLICY_VALID call didn't disturb other data */
                set_table_params.remapPolicy[i].entryValid = set_valid_params.entryValid[i];

                ASSERT_EQ(memcmp(&set_table_params.remapPolicy[i],
                    &get_table_params.entry[i], sizeof(LWSWITCH_REMAP_POLICY_ENTRY)), 0);
            }

            get_table_params.firstIndex = get_table_params.nextIndex;
        }
    }
}

TEST_F(LWSwitchDeviceTest, IoctlSetRemapPolicyRequestValidRepeaterMode)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LWSWITCH_SET_REMAP_POLICY_VALID p;
    LwU32 repeater_port = LWSWITCH_ILWALID_PORT;
    LwU32 remap_num_entries = LWSWITCH_REMAP_POLICY_ENTRIES_MAX;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (!isArchLs10())
    {
        return;
    }

    repeaterPort(&repeater_port);
    if (repeater_port == LWSWITCH_ILWALID_PORT)
    {
        printf("[ SKIPPED  ] No port in Repeater Mode.\n");
        return;
    }

    memset(&p, 0, sizeof(p));

    p.portNum = repeater_port;
    p.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_PRIMARY;
    p.numEntries = remap_num_entries;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
}

TEST_F(LWSwitchDeviceTest, IoctlSetRemapPolicyRequestValidBadInput)
{
    LWSWITCH_SET_REMAP_POLICY_VALID p;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    LwU32 remap_policy_table_size = getRemapTableSize(LWSWITCH_TABLE_SELECT_REMAP_PRIMARY);
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (isArchSv10())
    {
        return;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    // Invalid port
    p.portNum = 0xbaddad;
    p.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_PRIMARY;
    p.firstIndex = 0;
    p.numEntries = 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Invalid table
    p.portNum = valid_port;
    p.tableSelect = (LWSWITCH_TABLE_SELECT_REMAP) ~0;
    p.firstIndex = 0;
    p.numEntries = 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);

    // Bad first Index
    p.portNum = valid_port;
    p.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_PRIMARY;
    p.firstIndex = remap_policy_table_size;
    p.numEntries = 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

   // Bad last Index
    p.portNum = valid_port;
    p.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_PRIMARY;
    p.firstIndex = remap_policy_table_size - 1;
    p.numEntries = 2;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Too many entries
    p.portNum = valid_port;
    p.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_PRIMARY;
    p.firstIndex = 0;
    p.numEntries = LWSWITCH_REMAP_POLICY_ENTRIES_MAX + 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_REMAP_POLICY_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
}

TEST_F(LWSwitchDeviceTest, IoctlSetRoutingID)
{
    LWSWITCH_SET_ROUTING_ID p;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    LwU32 i;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    memset(&p, 0, sizeof(p));

    p.portNum = valid_port;
    p.firstIndex = 0;
    p.numEntries = LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX;

    for (i=0; i < p.numEntries; i++)
    {
        p.routingId[i].numEntries = 1;
    }

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID, &p, sizeof(p));
    if (isArchSv10())
    {
        // Unsupported
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
    }
    else
    {
        ASSERT_EQ(status, LW_OK);
    }
}

TEST_F(LWSwitchDeviceTest, IoctlSetRoutingIDRepeaterMode)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LWSWITCH_SET_ROUTING_ID p;
    LwU32 repeater_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (!isArchLs10())
    {
        return;
    }

    repeaterPort(&repeater_port);
    if (repeater_port == LWSWITCH_ILWALID_PORT)
    {
        printf("[ SKIPPED  ] No port in Repeater Mode.\n");
        return;
    }

    memset(&p, 0, sizeof(p));

    p.portNum = repeater_port;
    p.firstIndex = 0;
    p.numEntries = LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
}

TEST_F(LWSwitchDeviceTest, IoctlSetRoutingIDBadInput)
{
    LWSWITCH_SET_ROUTING_ID p;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    LwU32 routing_id_table_size = getRidTableSize();
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (isArchSv10())
    {
        return;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    memset(&p, 0, sizeof(p));

    // Invalid port
    p.portNum = 0xbaddad;
    p.firstIndex = 0;
    p.numEntries = 1;
    p.routingId[0].numEntries = 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID, &p, sizeof(p));
    ASSERT_NE(status, LW_OK);

    // Bad first index
    p.portNum = valid_port;
    p.firstIndex = routing_id_table_size;
    p.numEntries = 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID, &p, sizeof(p));
    ASSERT_NE(status, LW_OK);

    // Bad last index
    p.portNum = valid_port;
    p.firstIndex = routing_id_table_size - 1;
    p.numEntries = 2;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID, &p, sizeof(p));
    ASSERT_NE(status, LW_OK);

    // Too many entries
    p.portNum = valid_port;
    p.firstIndex = 0;
    p.numEntries = LWSWITCH_ROUTING_ID_ENTRIES_MAX + 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID, &p, sizeof(p));
    ASSERT_NE(status, LW_OK);

    // Too few port list entries
    p.portNum = valid_port;
    p.firstIndex = 0;
    p.numEntries = 1;
    p.routingId[0].numEntries = 0;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID, &p, sizeof(p));
    ASSERT_NE(status, LW_OK);

    // Too many port list entries
    p.portNum = valid_port;
    p.firstIndex = 0;
    p.numEntries = 1;
    p.routingId[0].numEntries = LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX + 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID, &p, sizeof(p));
    ASSERT_NE(status, LW_OK);

    // Invalid port list VC map entry
    p.portNum = valid_port;
    p.firstIndex = 0;
    p.numEntries = 1;
    p.routingId[0].numEntries = 1;
    p.routingId[0].portList[0].vcMap = ~0;
    p.routingId[0].portList[0].destPortNum = 0;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID, &p, sizeof(p));
    ASSERT_NE(status, LW_OK);

    // Invalid port list destination port entry
    p.portNum = valid_port;
    p.firstIndex = 0;
    p.numEntries = 1;
    p.routingId[0].numEntries = 1;
    p.routingId[0].portList[0].vcMap = LWSWITCH_ROUTING_ID_VCMAP_SAME;
    p.routingId[0].portList[0].destPortNum = 0xbaddad;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID, &p, sizeof(p));
    ASSERT_NE(status, LW_OK);
}

TEST_F(LWSwitchDeviceTest, IoctlGetRoutingId)
{
    LWSWITCH_GET_ROUTING_ID_PARAMS get_table_params;
    LWSWITCH_SET_ROUTING_ID set_table_params;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    LwU32 rid_num_entries = LWSWITCH_ROUTING_ID_ENTRIES_MAX;
    LwU32 rid_table_size = getRidTableSize();
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 i, j;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    memset(&get_table_params, 0, sizeof(LWSWITCH_GET_ROUTING_ID_PARAMS));

    /* this ioctl is only supported on lr10 */
    if (isArchSv10())
    {
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_ROUTING_ID, &get_table_params, sizeof(get_table_params));
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
        return;
    }

    /* lr10 fmodel does not support table readback. Hence, test only the ioctl */
    if (isArchLr10() && isFmodel())
    {
        get_table_params.firstIndex = 0;
        get_table_params.portNum = valid_port;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_ROUTING_ID, &get_table_params, sizeof(get_table_params));
        ASSERT_EQ(status, LW_OK);
        return;
    }

    memset(&set_table_params, 0, sizeof(LWSWITCH_SET_ROUTING_ID));

    for (i = 0; i < rid_num_entries; i++)
    {
        for (j = 0; j < LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX; j++)
        {
            set_table_params.routingId[i].portList[j].destPortNum =
                rand() % LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX;
            set_table_params.routingId[i].portList[j].vcMap =
                rand() % LWSWITCH_ROUTING_ID_VC_MODE_MAX;
        }

        set_table_params.routingId[i].numEntries =
            LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX;
        set_table_params.routingId[i].entryValid = rand() % 2;
        set_table_params.routingId[i].enableIrlErrResponse = rand() % 2;
        set_table_params.routingId[i].useRoutingLan = rand() % 2;
    }

    set_table_params.portNum = valid_port;
    set_table_params.numEntries = rid_num_entries;

    for (i = 0; i < rid_table_size / rid_num_entries; i++)
    {
        set_table_params.firstIndex = i * rid_num_entries;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID, &set_table_params, sizeof(set_table_params));
        ASSERT_EQ(status, LW_OK);
    }

    /* read the tables */
    get_table_params.firstIndex = 0;
    get_table_params.portNum = valid_port;
    get_table_params.nextIndex = 0;

    while (get_table_params.nextIndex < rid_table_size)
    {
        /* get up to 64 nonzero entries at a time */
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_ROUTING_ID, &get_table_params, sizeof(get_table_params));

        /* check that we read back what we wrote */
        ASSERT_EQ(set_table_params.numEntries, get_table_params.numEntries);

        for (i = 0; i < get_table_params.numEntries; i++)
        {
            ASSERT_EQ(memcmp(&set_table_params.routingId[i],
                &get_table_params.entries[i].entry, sizeof(LWSWITCH_ROUTING_ID_ENTRY)), 0);
        }

        get_table_params.firstIndex = get_table_params.nextIndex;
    }
}

TEST_F(LWSwitchDeviceTest, IoctlGetRoutingIdRepeaterMode)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LWSWITCH_GET_ROUTING_ID_PARAMS p;
    LwU32 repeater_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (!isArchLs10())
    {
        return;
    }

    repeaterPort(&repeater_port);
    if (repeater_port == LWSWITCH_ILWALID_PORT)
    {
        printf("[ SKIPPED  ] No port in Repeater Mode.\n");
        return;
    }

    memset(&p, 0, sizeof(p));

    p.firstIndex = 0;
    p.portNum = repeater_port;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_ROUTING_ID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
}

TEST_F(LWSwitchDeviceTest, IoctlGetRoutingIdBadInput)
{
    LWSWITCH_GET_ROUTING_ID_PARAMS p;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    LwU32 rid_table_size = getRidTableSize();
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (isArchSv10())
    {
        return;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    // Invalid port
    p.portNum = 0xbaddad;
    p.firstIndex = 0;
    p.numEntries = 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_ROUTING_ID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Bad first Index
    p.portNum = valid_port;
    p.firstIndex = rid_table_size;
    p.numEntries = 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_ROUTING_ID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
}

/* 
 * Test procedure
 *
 * Fill RID RAM with portList data using SET_ROUTING_ID call.
 * Set the valid bits using SET_ROUTING_ID_VALID call.
 * GET RID entries using GET_ROUTING_ID call.
 * Compare valid bits against previously set values.
 * Verify that SET_ROUTING_ID_VALID API doesn't disturb portList data.
 */

TEST_F(LWSwitchDeviceTest, IoctlSetRidRequestValid)
{
    LWSWITCH_SET_ROUTING_ID_VALID set_valid_params;
    LWSWITCH_SET_ROUTING_ID set_table_params;
    LWSWITCH_GET_ROUTING_ID_PARAMS get_table_params;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    LwU32 rid_table_size = getRidTableSize();
    LwU32 rid_num_entries = LWSWITCH_ROUTING_ID_ENTRIES_MAX;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 i, j;
    LwU32 count;

    memset(&set_valid_params, 0, sizeof(LWSWITCH_SET_ROUTING_ID_VALID));

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    /* this ioctl is only supported on lr10 */
    if (isArchSv10())
    {
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID_VALID, &set_valid_params, sizeof(set_valid_params));
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
        return;
    }

    /* lr10 fmodel does not support table readback. Hence, test only the ioctl */
    if (isArchLr10() && isFmodel())
    {
        set_valid_params.portNum = valid_port;
        set_valid_params.numEntries = rid_num_entries;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID_VALID, &set_valid_params, sizeof(set_valid_params));
        ASSERT_EQ(status, LW_OK);
        return;
    }

    memset(&set_table_params, 0, sizeof(LWSWITCH_SET_ROUTING_ID));

    // Set random table size
    count = rid_table_size / rid_num_entries;
    rid_table_size = ((rand() % count) + 1) * rid_num_entries;

    /* Set random entries in the RID RAM */
    for (i = 0; i < rid_num_entries; i++)
    {
        set_table_params.routingId[i].entryValid = 1;
        for (j = 0; j < LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX; j++)
        {
            set_table_params.routingId[i].portList[j].destPortNum =
                rand() % LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX;
            set_table_params.routingId[i].portList[j].vcMap =
                rand() % LWSWITCH_ROUTING_ID_VC_MODE_MAX;
        }

        set_table_params.routingId[i].numEntries =
            LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX;
    }

    set_table_params.portNum = valid_port;
    set_table_params.numEntries = rid_num_entries;

    for (i = 0; i < rid_table_size / rid_num_entries; i++)
    {
        set_table_params.firstIndex = i * rid_num_entries;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID, &set_table_params, sizeof(set_table_params));
        ASSERT_EQ(status, LW_OK);
    }

    for (i = 0; i < rid_num_entries; i++)
    {
        set_valid_params.entryValid[i] = rand() % 2; // set 0 or 1
    }

    set_valid_params.portNum = valid_port;
    set_valid_params.numEntries = rid_num_entries;

    for (i = 0; i < rid_table_size / rid_num_entries; i++)
    {
        set_valid_params.firstIndex = i * rid_num_entries;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID_VALID, &set_valid_params, sizeof(set_valid_params));
        ASSERT_EQ(status, LW_OK);
    }

    /* read the tables */
    memset(&get_table_params, 0, sizeof(LWSWITCH_GET_ROUTING_ID_PARAMS));

    get_table_params.firstIndex = 0;
    get_table_params.portNum = valid_port;
    get_table_params.nextIndex = 0;

    while (get_table_params.nextIndex < rid_table_size)
    {
        /* get up to 64 nonzero entries at a time */
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_ROUTING_ID, &get_table_params, sizeof(get_table_params));
        ASSERT_EQ(status, LW_OK);

        ASSERT_EQ(set_valid_params.numEntries, get_table_params.numEntries);

        /* verify valid entries */
        for (i = 0; i < get_table_params.numEntries; i++)
        {
            ASSERT_EQ(set_valid_params.entryValid[i], get_table_params.entries[i].entry.entryValid);

            /* verify that SET_ROUTING_ID_VALID call doesn't disturb portList data */
            ASSERT_EQ(memcmp(set_table_params.routingId[i].portList,
                get_table_params.entries[i].entry.portList, sizeof(LWSWITCH_ROUTING_ID_DEST_PORT_LIST)), 0);
        }

        get_table_params.firstIndex = get_table_params.nextIndex;
    }
}

TEST_F(LWSwitchDeviceTest, IoctlSetRidRequestValidRepeaterMode)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LWSWITCH_SET_ROUTING_ID_VALID p;
    LwU32 repeater_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LwU32 rid_num_entries = LWSWITCH_ROUTING_ID_ENTRIES_MAX;
    LW_STATUS status;

    if (!isArchLs10())
    {
        return;
    }

    repeaterPort(&repeater_port);
    if (repeater_port == LWSWITCH_ILWALID_PORT)
    {
        printf("[ SKIPPED  ] No port in Repeater Mode.\n");
        return;
    }

    memset(&p, 0, sizeof(p));

    p.portNum = repeater_port;
    p.numEntries = rid_num_entries;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
}

TEST_F(LWSwitchDeviceTest, IoctlSetRidRequestValidBadInput)
{
    LWSWITCH_SET_ROUTING_ID_VALID p;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    LwU32 rid_table_size = getRidTableSize();
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (isArchSv10())
    {
        return;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    // Invalid port
    p.portNum = 0xbaddad;
    p.firstIndex = 0;
    p.numEntries = 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Bad first Index
    p.portNum = valid_port;
    p.firstIndex = rid_table_size;
    p.numEntries = 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

   // Bad last Index
    p.portNum = valid_port;
    p.firstIndex = rid_table_size - 1;
    p.numEntries = 2;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Too many entries
    p.portNum = valid_port;
    p.firstIndex = 0;
    p.numEntries = LWSWITCH_ROUTING_ID_ENTRIES_MAX + 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_ID_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
}

TEST_F(LWSwitchDeviceTest, IoctlSetRoutingLAN)
{
    LWSWITCH_SET_ROUTING_LAN p;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 i;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    memset(&p, 0, sizeof(p));

    p.portNum = valid_port;
    p.firstIndex = 0;
    p.numEntries = LWSWITCH_ROUTING_LAN_ENTRIES_MAX;
    for (i = 0; i < p.numEntries; i++)
    {
        p.routingLan[i].numEntries = 1;
        p.routingLan[i].portList[0].groupSize = 1;
    }

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN, &p, sizeof(p));
    if (isArchSv10())
    {
        // Unsupported
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
    }
    else
    {
        ASSERT_EQ(status, LW_OK);
    }
}

TEST_F(LWSwitchDeviceTest, IoctlSetRoutingLANRepeaterMode)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LWSWITCH_SET_ROUTING_LAN p;
    LwU32 repeater_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 i;

    if (!isArchLs10())
    {
        return;
    }

    repeaterPort(&repeater_port);
    if (repeater_port == LWSWITCH_ILWALID_PORT)
    {
        printf("[ SKIPPED  ] No port in Repeater Mode.\n");
        return;
    }

    memset(&p, 0, sizeof(p));

    p.portNum = repeater_port;
    p.firstIndex = 0;
    p.numEntries = LWSWITCH_ROUTING_LAN_ENTRIES_MAX;
    for (i = 0; i < p.numEntries; i++)
    {
        p.routingLan[i].numEntries = 1;
        p.routingLan[i].portList[0].groupSize = 1;
    }

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
}

TEST_F(LWSwitchDeviceTest, IoctlSetRoutingLANBadInput)
{
    LWSWITCH_SET_ROUTING_LAN p;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    LwU32 rlan_table_size = getRlanTableSize();
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 i;

    if (isArchSv10())
    {
        return;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    memset(&p, 0, sizeof(p));

    // Invalid port
    p.portNum = 0xbaddad;
    p.firstIndex = 0;
    p.numEntries = 1;
    for (i = 0; i < p.numEntries; i++)
    {
        p.routingLan[i].numEntries = 1;
        p.routingLan[i].portList[0].groupSize = 1;
    }

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN, &p, sizeof(p));
    ASSERT_NE(status, LW_OK);

    // Bad first index
    p.portNum = valid_port;
    p.firstIndex = rlan_table_size;
    p.numEntries = 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN, &p, sizeof(p));
    ASSERT_NE(status, LW_OK);

    // Bad last index
    p.portNum = valid_port;
    p.firstIndex = rlan_table_size - 1;
    p.numEntries = 2;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN, &p, sizeof(p));
    ASSERT_NE(status, LW_OK);

    // Too many entries
    p.portNum = valid_port;
    p.firstIndex = 0;
    p.numEntries = LWSWITCH_ROUTING_LAN_ENTRIES_MAX + 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN, &p, sizeof(p));
    ASSERT_NE(status, LW_OK);

    // Too many port list entries
    p.portNum = valid_port;
    p.firstIndex = 0;
    p.numEntries = 1;
    p.routingLan[0].numEntries = LWSWITCH_ROUTING_LAN_GROUP_SEL_MAX + 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN, &p, sizeof(p));
    ASSERT_NE(status, LW_OK);

    // Invalid port list group select entry
    p.portNum = valid_port;
    p.firstIndex = 0;
    p.numEntries = 1;
    p.routingLan[0].numEntries = 1;
    p.routingLan[0].portList[0].groupSelect = ~0;
    p.routingLan[0].portList[0].groupSize = 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN, &p, sizeof(p));
    ASSERT_NE(status, LW_OK);

    // Invalid port list group size entry (0)
    p.portNum = valid_port;
    p.firstIndex = 0;
    p.numEntries = 1;
    p.routingLan[0].numEntries = 1;
    p.routingLan[0].portList[0].groupSelect = 0;
    p.routingLan[0].portList[0].groupSize = 0;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN, &p, sizeof(p));
    ASSERT_NE(status, LW_OK);

    // Invalid port list group size entry (out of range)
    p.portNum = valid_port;
    p.firstIndex = 0;
    p.numEntries = 1;
    p.routingLan[0].numEntries = 1;
    p.routingLan[0].portList[0].groupSelect = 0;
    p.routingLan[0].portList[0].groupSize = ~0;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN, &p, sizeof(p));
    ASSERT_NE(status, LW_OK);
}

TEST_F(LWSwitchDeviceTest, IoctlGetRoutingLan)
{
    LWSWITCH_GET_ROUTING_LAN_PARAMS get_table_params;
    LWSWITCH_SET_ROUTING_LAN set_table_params;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    LwU32 rlan_num_entries = LWSWITCH_ROUTING_LAN_ENTRIES_MAX;
    LwU32 rlan_table_size = getRlanTableSize();
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 i, j;

    memset(&get_table_params, 0, sizeof(LWSWITCH_GET_ROUTING_LAN_PARAMS));

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    /* this ioctl is only supported on lr10 */
    if (isArchSv10())
    {
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_ROUTING_LAN, &get_table_params, sizeof(get_table_params));
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
        return;
    }

    /* lr10 fmodel does not support table readback. Hence, test only the ioctl */
    if (isArchLr10() && isFmodel())
    {
        get_table_params.firstIndex = 0;
        get_table_params.portNum = valid_port;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_ROUTING_LAN, &get_table_params, sizeof(get_table_params));
        ASSERT_EQ(status, LW_OK);
        return;
    }

    memset(&set_table_params, 0, sizeof(LWSWITCH_SET_ROUTING_LAN));

    for (i = 0; i < rlan_num_entries; i++)
    {
        for (j = 0; j < LWSWITCH_ROUTING_LAN_GROUP_SEL_MAX; j++)
        {
            set_table_params.routingLan[i].portList[j].groupSelect =
                rand() % LWSWITCH_ROUTING_LAN_GROUP_SEL_MAX;
            set_table_params.routingLan[i].portList[j].groupSize =
                (rand() % LWSWITCH_ROUTING_LAN_GROUP_SIZE_MAX) + 1;
        }

        set_table_params.routingLan[i].numEntries =
            LWSWITCH_ROUTING_LAN_GROUP_SEL_MAX;
        set_table_params.routingLan[i].entryValid = rand() % 2;  // set 0 or 1

    }

    set_table_params.portNum = valid_port;
    set_table_params.numEntries = rlan_num_entries;

    for (i = 0; i < rlan_table_size / rlan_num_entries; i++)
    {
        set_table_params.firstIndex = i * rlan_num_entries;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN, &set_table_params, sizeof(set_table_params));
        ASSERT_EQ(status, LW_OK);
    }

    /* read the tables */
    get_table_params.firstIndex = 0;
    get_table_params.portNum = valid_port;
    get_table_params.nextIndex = 0;

    while (get_table_params.nextIndex < rlan_table_size)
    {
        /* get up to 64 nonzero entries at a time */
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_ROUTING_LAN, &get_table_params, sizeof(get_table_params));

        /* check that we read back what we wrote */
        ASSERT_EQ(set_table_params.numEntries, get_table_params.numEntries);
        
        for (i = 0; i < get_table_params.numEntries; i++)
        {
            ASSERT_EQ(memcmp(&set_table_params.routingLan[i],
                      &get_table_params.entries[i].entry, sizeof(LWSWITCH_ROUTING_LAN_ENTRY)), 0);
        }

        get_table_params.firstIndex = get_table_params.nextIndex;
    }
}

TEST_F(LWSwitchDeviceTest, IoctlGetRoutingLanRepeaterMode)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LWSWITCH_GET_ROUTING_LAN_PARAMS p;
    LwU32 repeater_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (!isArchLs10())
    {
        return;
    }

    repeaterPort(&repeater_port);
    if (repeater_port == LWSWITCH_ILWALID_PORT)
    {
        printf("[ SKIPPED  ] No port in Repeater Mode.\n");
        return;
    }

    memset(&p, 0, sizeof(p));

    p.firstIndex = 0;
    p.portNum = repeater_port;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_ROUTING_LAN, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
}

TEST_F(LWSwitchDeviceTest, IoctlGetRoutingLanBadInput)
{
    LWSWITCH_GET_ROUTING_LAN_PARAMS p;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    LwU32 rlan_table_size = getRlanTableSize();
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (isArchSv10())
    {
        return;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    // Invalid port
    p.portNum = 0xbaddad;
    p.firstIndex = 0;
    p.numEntries = 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_ROUTING_LAN, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Bad first Index
    p.portNum = valid_port;
    p.firstIndex = rlan_table_size;
    p.numEntries = 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_ROUTING_LAN, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
}

/* 
 * Test procedure
 *
 * Fill RLAN RAM with portList data using SET_ROUTING_LAN call.
 * Set the valid bits using SET_ROUTING_LAN_VALID call.
 * GET RLAN entries using GET_ROUTING_LAN call.
 * Compare valid bits against previously set values.
 * Verify that SET_ROUTING_LAN_VALID API doesn't disturb other entries.
 */

TEST_F(LWSwitchDeviceTest, IoctlSetRlanRequestValid)
{
    LWSWITCH_SET_ROUTING_LAN_VALID set_valid_params;
    LWSWITCH_SET_ROUTING_LAN set_table_params;
    LWSWITCH_GET_ROUTING_LAN_PARAMS get_table_params;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    LwU32 rlan_table_size = getRlanTableSize();
    LwU32 rlan_num_entries = LWSWITCH_ROUTING_LAN_ENTRIES_MAX;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 i, j;
    LwU32 count;

    memset(&set_valid_params, 0, sizeof(LWSWITCH_SET_ROUTING_LAN_VALID));

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    /* this ioctl is only supported on lr10 */
    if (isArchSv10())
    {
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN_VALID, &set_valid_params, sizeof(set_valid_params));
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
        return;
    }

    /* lr10 fmodel does not support table readback. Hence, test only the ioctl */
    if (isArchLr10() && isFmodel())
    {
        set_valid_params.portNum = valid_port;
        set_valid_params.numEntries = rlan_num_entries;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN_VALID, &set_valid_params, sizeof(set_valid_params));
        ASSERT_EQ(status, LW_OK);
        return;
    }

    memset(&set_table_params, 0, sizeof(LWSWITCH_SET_ROUTING_LAN));

    // Set random table size
    count = rlan_table_size / rlan_num_entries;
    rlan_table_size = ((rand() % count) + 1) * rlan_num_entries;

    /* Set random entries in the RLAN RAM */
    for (i = 0; i < rlan_num_entries; i++)
    {
        set_table_params.routingLan[i].entryValid = 1;
        for (j = 0; j < LWSWITCH_ROUTING_LAN_GROUP_SEL_MAX; j++)
        {
            set_table_params.routingLan[i].portList[j].groupSelect =
                rand() % LWSWITCH_ROUTING_LAN_GROUP_SEL_MAX;
            set_table_params.routingLan[i].portList[j].groupSize =
                (rand() % LWSWITCH_ROUTING_LAN_GROUP_SIZE_MAX) + 1;
        }

        set_table_params.routingLan[i].numEntries =
            LWSWITCH_ROUTING_LAN_GROUP_SEL_MAX;
    }

    set_table_params.portNum = valid_port;
    set_table_params.numEntries = rlan_num_entries;

    for (i = 0; i < rlan_table_size / rlan_num_entries; i++)
    {
        set_table_params.firstIndex = i * rlan_num_entries;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN, &set_table_params, sizeof(set_table_params));
        ASSERT_EQ(status, LW_OK);
    }

    for (i = 0; i < rlan_num_entries; i++)
    {
        set_valid_params.entryValid[i] = rand() % 2; // set 0 or 1
    }

    set_valid_params.portNum = valid_port;
    set_valid_params.numEntries = rlan_num_entries;

    for (i = 0; i < rlan_table_size / rlan_num_entries; i++)
    {
        set_valid_params.firstIndex = i * rlan_num_entries;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN_VALID, &set_valid_params, sizeof(set_valid_params));
        ASSERT_EQ(status, LW_OK);
    }

    /* read the tables */
    memset(&get_table_params, 0, sizeof(LWSWITCH_GET_ROUTING_LAN_PARAMS));

    get_table_params.firstIndex = 0;
    get_table_params.portNum = valid_port;
    get_table_params.nextIndex = 0;

    while (get_table_params.nextIndex < rlan_table_size)
    {
        /* get up to 64 nonzero entries at a time */
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_ROUTING_LAN, &get_table_params, sizeof(get_table_params));
        ASSERT_EQ(status, LW_OK);

        ASSERT_EQ(set_valid_params.numEntries, get_table_params.numEntries);

        /* verify valid entries */
        for (i = 0; i < get_table_params.numEntries; i++)
        {
            ASSERT_EQ(set_valid_params.entryValid[i], get_table_params.entries[i].entry.entryValid);

            /* verify that SET_ROUTING_LAN_VALID call doesn't disturb portList data */
            ASSERT_EQ(memcmp(set_table_params.routingLan[i].portList,
                      get_table_params.entries[i].entry.portList, sizeof(LWSWITCH_ROUTING_LAN_PORT_SELECT)), 0);
        }

        get_table_params.firstIndex = get_table_params.nextIndex;
    }
}

TEST_F(LWSwitchDeviceTest, IoctlSetRlanRequestValidRepeaterMode)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LWSWITCH_SET_ROUTING_LAN_VALID p;
    LwU32 repeater_port = LWSWITCH_ILWALID_PORT;
    LwU32 rlan_num_entries = LWSWITCH_ROUTING_LAN_ENTRIES_MAX;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (!isArchLs10())
    {
        return;
    }

    repeaterPort(&repeater_port);
    if (repeater_port == LWSWITCH_ILWALID_PORT)
    {
        printf("[ SKIPPED  ] No port in Repeater Mode.\n");
        return;
    }

    memset(&p, 0, sizeof(p));

    p.portNum = repeater_port;
    p.numEntries = rlan_num_entries;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
}

TEST_F(LWSwitchDeviceTest, IoctlSetRlanRequestValidBadInput)
{
    LWSWITCH_SET_ROUTING_LAN_VALID p;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    LwU32 rlan_table_size = getRlanTableSize();
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (isArchSv10())
    {
        return;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    // Invalid port
    p.portNum = 0xbaddad;
    p.firstIndex = 0;
    p.numEntries = 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Bad first Index
    p.portNum = valid_port;
    p.firstIndex = rlan_table_size;
    p.numEntries = 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

   // Bad last Index
    p.portNum = valid_port;
    p.firstIndex = rlan_table_size - 1;
    p.numEntries = 2;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Too many entries
    p.portNum = valid_port;
    p.firstIndex = 0;
    p.numEntries = LWSWITCH_ROUTING_LAN_ENTRIES_MAX + 1;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_ROUTING_LAN_VALID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
}

TEST_F(LWSwitchDeviceTest, IoctlSetLatencyBins)
{
    LWSWITCH_SET_LATENCY_BINS p;
    lwswitch_device *device = getDevice();
    LwU32 vcCount = getvcCount();
    LW_STATUS status;
    LwU32 i;

    for (i = 0; i < vcCount; i++)
    {
        p.bin[i].lowThreshold = 10;
        p.bin[i].medThreshold = 11;
        p.bin[i].hiThreshold = 12;
    }
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_LATENCY_BINS, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);

    for (i = 0; i < vcCount; i++)
    {
        p.bin[i].lowThreshold = 120;
        p.bin[i].medThreshold = 200;
        p.bin[i].hiThreshold = 10000;
    }
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_LATENCY_BINS, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);

    for (i = 0; i < vcCount; i++)
    {
        p.bin[i].lowThreshold = 120;
        p.bin[i].medThreshold = 200;
        p.bin[i].hiThreshold = 1000;
    }
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_LATENCY_BINS, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);
}

TEST_F(LWSwitchDeviceTest, IoctlSetLatencyBinsBadInput)
{
    LWSWITCH_SET_LATENCY_BINS p;
    LwU32 tmp;
    lwswitch_device *device = getDevice();
    LwU32 vcCount = getvcCount();
    LW_STATUS status;
    LwU32 i;

    for (i = 0; i < vcCount; i++)
    {
        p.bin[i].lowThreshold = 120;
        p.bin[i].medThreshold = 200;
        p.bin[i].hiThreshold = 1000;
    }

    // low less than minimum value
    for (i = 0; i < vcCount; i++)
    {
        tmp = p.bin[i].lowThreshold;
        p.bin[i].lowThreshold = 1;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_LATENCY_BINS, &p, sizeof(p));
        ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
        p.bin[i].lowThreshold = tmp;
    }

    // med < low
    for (i = 0; i < vcCount; i++)
    {
        tmp = p.bin[i].medThreshold;
        p.bin[i].medThreshold = p.bin[i].lowThreshold - 1;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_LATENCY_BINS, &p, sizeof(p));
        ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
        p.bin[i].medThreshold = tmp;
    }

    // hi < med
    for (i = 0; i < vcCount; i++)
    {
        tmp = p.bin[i].hiThreshold;
        p.bin[i].hiThreshold = p.bin[i].medThreshold - 1;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_LATENCY_BINS, &p, sizeof(p));
        ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
        p.bin[i].hiThreshold = tmp;
    }

    // low > max (but also > mid/high)
    for (i = 0; i < vcCount; i++)
    {
        tmp = p.bin[i].lowThreshold;
        p.bin[i].lowThreshold = 0xffffff;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_LATENCY_BINS, &p, sizeof(p));
        ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
        p.bin[i].lowThreshold = tmp;
    }
}

//
// Test get error inputs. For this test we don't know the state
// of errors in the system so it is challenging to fetch errors.
// This will be tested more in the interrupt and poll tests.
//
TEST_F(LWSwitchDeviceTest, IoctlGetErrors)
{
    LWSWITCH_GET_ERRORS_PARAMS p;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    p.errorType = LWSWITCH_ERROR_SEVERITY_FATAL;
    p.errorCount = INVALID;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_ERRORS, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);
    ASSERT_NE(p.errorCount, INVALID);

    p.errorType = LWSWITCH_ERROR_SEVERITY_NONFATAL;
    p.errorCount = INVALID;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_ERRORS, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);
    ASSERT_NE(p.errorCount, INVALID);
}

TEST_F(LWSwitchDeviceTest, IoctlGetErrorsBadInput)
{
    LWSWITCH_GET_ERRORS_PARAMS p;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    p.errorType = 0xdeaddead;
    p.errorCount = INVALID;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_ERRORS, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
}

TEST_F(LWSwitchDeviceTest, IoctlGetInternalLatency)
{
    // This call is pretty slow on simulation...
    LWSWITCH_GET_INTERNAL_LATENCY p, q;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    for (LwU32 i = 0; i < getvcCount(); i++)
    {
        memset(&p, 0xff, sizeof(p));
        p.vc_selector = i;
        q = p;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INTERNAL_LATENCY, &p, sizeof(p));
        ASSERT_EQ(status, LW_OK);
        ASSERT_NE(memcmp(&p, &q, sizeof(p)), 0);
    }
}

TEST_F(LWSwitchDeviceTest, IoctlGetInternalLatencyBadInput)
{
    LWSWITCH_GET_INTERNAL_LATENCY p;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 ilwalid_vc = getvcCount();

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INTERNAL_LATENCY, NULL, sizeof(IOCTL_LWSWITCH_GET_INTERNAL_LATENCY));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    memset(&p, 0xff, sizeof(p));
    p.vc_selector = ilwalid_vc;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INTERNAL_LATENCY, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
}

TEST_F(LWSwitchDeviceTest, IoctlGetInfo)
{
    static const LWSWITCH_GET_INFO_INDEX idxs[] = {
        LWSWITCH_GET_INFO_INDEX_ARCH,
        LWSWITCH_GET_INFO_INDEX_IMPL,
        LWSWITCH_GET_INFO_INDEX_CHIPID,
        LWSWITCH_GET_INFO_INDEX_REVISION_MAJOR,
        LWSWITCH_GET_INFO_INDEX_REVISION_MINOR,
        LWSWITCH_GET_INFO_INDEX_REVISION_MINOR_EXT,
        LWSWITCH_GET_INFO_INDEX_FOUNDRY,
        LWSWITCH_GET_INFO_INDEX_FAB,
        LWSWITCH_GET_INFO_INDEX_LOT_CODE_0,
        LWSWITCH_GET_INFO_INDEX_LOT_CODE_1,
        LWSWITCH_GET_INFO_INDEX_WAFER,
        LWSWITCH_GET_INFO_INDEX_XCOORD,
        LWSWITCH_GET_INFO_INDEX_YCOORD,
        LWSWITCH_GET_INFO_INDEX_SPEEDO_REV,
        LWSWITCH_GET_INFO_INDEX_SPEEDO0,
        LWSWITCH_GET_INFO_INDEX_SPEEDO1,
        LWSWITCH_GET_INFO_INDEX_SPEEDO2,
        LWSWITCH_GET_INFO_INDEX_IDDQ,
        LWSWITCH_GET_INFO_INDEX_IDDQ_REV,
        LWSWITCH_GET_INFO_INDEX_ATE_REV,
        LWSWITCH_GET_INFO_INDEX_VENDOR_CODE,
        LWSWITCH_GET_INFO_INDEX_OPS_RESERVED,
        LWSWITCH_GET_INFO_INDEX_DEVICE_ID,

        LWSWITCH_GET_INFO_INDEX_NUM_PORTS,
        LWSWITCH_GET_INFO_INDEX_ENABLED_PORTS_MASK_31_0,
        LWSWITCH_GET_INFO_INDEX_ENABLED_PORTS_MASK_63_32,
        LWSWITCH_GET_INFO_INDEX_NUM_VCS,
        LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_TABLE_SIZE,
        LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_EXTA_TABLE_SIZE,
        LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_EXTB_TABLE_SIZE,
        LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_MULTICAST_TABLE_SIZE,
        LWSWITCH_GET_INFO_INDEX_ROUTING_ID_TABLE_SIZE,
        LWSWITCH_GET_INFO_INDEX_ROUTING_LAN_TABLE_SIZE,

        LWSWITCH_GET_INFO_INDEX_FREQ_KHZ,
        LWSWITCH_GET_INFO_INDEX_VCOFREQ_KHZ,
        LWSWITCH_GET_INFO_INDEX_VOLTAGE_MVOLT,
        LWSWITCH_GET_INFO_INDEX_PHYSICAL_ID,

        LWSWITCH_GET_INFO_INDEX_PCI_DOMAIN,
        LWSWITCH_GET_INFO_INDEX_PCI_BUS,
        LWSWITCH_GET_INFO_INDEX_PCI_DEVICE,
        LWSWITCH_GET_INFO_INDEX_PCI_FUNCTION,

        (LWSWITCH_GET_INFO_INDEX)INVALID
    };
    LWSWITCH_GET_INFO get_info, q;
    LwU32 i;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    // Call each one at a time
    for (i=0; idxs[i] != INVALID; i++)
    {
        get_info.index[0] = idxs[i];
        get_info.count = 1;
        get_info.info[0] = 0xdeadbeef;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &get_info, sizeof(get_info));
        ASSERT_EQ(status, LW_OK);
        ASSERT_NE(get_info.info[0], 0xdeadbeef);
    }

    // All together
    for (i=0; (idxs[i] != INVALID) && (i < LWSWITCH_GET_INFO_COUNT_MAX); i++)
    {
        get_info.index[i] = idxs[i];
        get_info.info[i] = 0xdeadbeef;
        ASSERT_LT(i, (LwU32)LWSWITCH_GET_INFO_COUNT_MAX);
    }
    get_info.count = i;
    q = get_info;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &get_info, sizeof(get_info));
    ASSERT_EQ(status, LW_OK);
    for (i=0; i < get_info.count; i++)
    {
        ASSERT_NE(get_info.info[i], 0xdeadbeef);
    }

    // Max # of gets
    for (i = 0; i < LWSWITCH_GET_INFO_COUNT_MAX; i++)
    {
        get_info.index[i] = LWSWITCH_GET_INFO_INDEX_IMPL;
    }
    get_info.count = i;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &get_info, sizeof(get_info));
    ASSERT_EQ(status, LW_OK);

    // Zero count is useless but should be a nop.
    get_info.count = 0;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &get_info, sizeof(get_info));
    ASSERT_EQ(status, LW_OK);
}

TEST_F(LWSwitchDeviceTest, IoctlGetInfoBadInput)
{
    LWSWITCH_GET_INFO get_info;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    // Bad index
    get_info.index[0] = (LWSWITCH_GET_INFO_INDEX)0xcafe;
    get_info.count = 1;
    get_info.info[0] = 0xdeadbeef;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &get_info, sizeof(get_info));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
    ASSERT_EQ(get_info.info[0], 0xdeadbeef);

    // Count too large
    get_info.index[0] = LWSWITCH_GET_INFO_INDEX_IMPL;
    get_info.count = LWSWITCH_GET_INFO_COUNT_MAX + 1;
    get_info.info[0] = 0xdeadbeef;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &get_info, sizeof(get_info));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
    ASSERT_EQ(get_info.info[0], 0xdeadbeef);
}

TEST_F(LWSwitchDeviceTest, IoctlGetIngressReqLinkId)
{
    LWSWITCH_GET_INGRESS_REQLINKID_PARAMS id_params;
    LWSWITCH_SET_SWITCH_PORT_CONFIG pc_params;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 expectedReqLinkID;
    LwU32 rlan_shift = DRF_SHIFT_RT(LW_NPORT_REQLINKID_REQROUTINGID) + 1;
    LwU32 count[] = {CONNECT_COUNT_512, CONNECT_COUNT_1024, CONNECT_COUNT_2048};
    LwU32 i;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwBool bIsRepeaterPort;
    getValidPortOrRepeaterPort(&valid_port, &bIsRepeaterPort);
    if (bIsRepeaterPort)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    id_params.portNum = valid_port;

    pc_params.portNum = valid_port;
    pc_params.type = CONNECT_ACCESS_GPU;
    pc_params.requesterLinkID = 3;
    pc_params.acCoupled = LW_FALSE;
    pc_params.enableVC1 = LW_FALSE;

    if (isArchSv10())
    {
        // write known value
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG, &pc_params, sizeof(pc_params));
        ASSERT_EQ(status, LW_OK);

        // read back and compare
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INGRESS_REQLINKID, &id_params, sizeof(id_params));
        ASSERT_EQ(status, LW_OK);
        ASSERT_EQ(id_params.requesterLinkID, pc_params.requesterLinkID);
    }
    else if(isArchLr10())
    {
        pc_params.requesterLanID = 12;

        for (i = 0; i < 3; i++)
        {
            pc_params.count = count[i];
            expectedReqLinkID = pc_params.requesterLinkID;

            // write known value
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG, &pc_params, sizeof(pc_params));
            ASSERT_EQ(status, LW_OK);

            // read back and compare
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INGRESS_REQLINKID, &id_params, sizeof(id_params));
            ASSERT_EQ(status, LW_OK);

            if (pc_params.count ==  CONNECT_COUNT_1024)
            {
                // The upper bit in REQROUTINGLAN becomes REQROUTINGID[9]
                expectedReqLinkID |= (pc_params.requesterLanID >> 3) << rlan_shift;
            }

            if (pc_params.count ==  CONNECT_COUNT_2048)
            {
                // The upper two bits in REQROUTINGLAN becomes REQROUTINGID[10:9]
                expectedReqLinkID |= (pc_params.requesterLanID >> 2) << rlan_shift;
            }

            ASSERT_EQ(id_params.requesterLinkID, expectedReqLinkID);
        }
    }
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    else if(isArchLs10())
    {
        pc_params.requesterLanID = 12;

        expectedReqLinkID = pc_params.requesterLinkID;

        // write known value
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG, &pc_params, sizeof(pc_params));
        ASSERT_EQ(status, LW_OK);

        // read back and compare
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INGRESS_REQLINKID, &id_params, sizeof(id_params));
        ASSERT_EQ(status, LW_OK);

        ASSERT_EQ(id_params.requesterLinkID, expectedReqLinkID);
    }
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    else
    {
        assert(0); // Unknown Arch
    }
}

TEST_F(LWSwitchDeviceTest, IoctlGetIngressReqLinkIdRepeaterMode)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LWSWITCH_GET_INGRESS_REQLINKID_PARAMS p;
    LwU32 repeater_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (!isArchLs10())
    {
        return;
    }

    repeaterPort(&repeater_port);
    if (repeater_port == LWSWITCH_ILWALID_PORT)
    {
        printf("[ SKIPPED  ] No port in Repeater Mode.\n");
        return;
    }

    memset(&p, 0, sizeof(p));

    p.portNum = repeater_port;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INGRESS_REQLINKID, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
}

TEST_F(LWSwitchDeviceTest, IoctlGetIngressReqLinkIdBadInput)
{
    LWSWITCH_GET_INGRESS_REQLINKID_PARAMS params;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    // Bad portNum
    params.portNum = 0xffffffff;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INGRESS_REQLINKID, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
}

TEST_F(LWSwitchDeviceTest, IoctlUnregisterLink)
{
    LWSWITCH_UNREGISTER_LINK_PARAMS ul_params;
    LWSWITCH_GET_INFO info_params;
    LwU64 port_mask;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (getLinkInitializedMask() != 0)
    {
        printf("[ SKIPPED ] The links are not in INIT or INVALID (RESET) state. "
               "This indicates that some other fabric management software is "
               "running and has already acquired.\n");
        return;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    getValidPortOrRepeaterPort(&valid_port, NULL);
#else
    validPort(&valid_port);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    // unregister link
    memset(&info_params, 0, sizeof(info_params));
    ul_params.portNum = valid_port;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_UNREGISTER_LINK, &ul_params, sizeof(ul_params));
    ASSERT_EQ(status, LW_OK);

    // Get link caps and make sure the unregistered link doesn't exist.
    memset(&info_params, 0, sizeof(info_params));
    info_params.index[0] = LWSWITCH_GET_INFO_INDEX_ENABLED_PORTS_MASK_31_0;
    info_params.index[1] = LWSWITCH_GET_INFO_INDEX_ENABLED_PORTS_MASK_63_32;
    info_params.count = 2;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &info_params, sizeof(info_params));
    ASSERT_EQ(status, LW_OK);
    port_mask = info_params.info[0] | ((LwU64) info_params.info[1] << 32);
    ASSERT_TRUE(!(port_mask & LWBIT64(valid_port)));
}

TEST_F(LWSwitchDeviceTest, IoctlUnregisterLinkBadInput)
{
    LWSWITCH_UNREGISTER_LINK_PARAMS ul_params;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    // Bad portNum
    memset(&ul_params, 0, sizeof(ul_params));
    ul_params.portNum = 0xffffffff;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_UNREGISTER_LINK, &ul_params, sizeof(ul_params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
}

TEST_F(LWSwitchDeviceTest, IoctlResetAndDrainLinks)
{
    LWSWITCH_RESET_AND_DRAIN_LINKS_PARAMS dp_params;
    LwU32 valid_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    getValidPortOrRepeaterPort(&valid_port, NULL);
#else
    validPort(&valid_port);
#endif

    memset(&dp_params, 0, sizeof(dp_params));

    /* On SV10, the link-pair must be reset together */
    if (isArchSv10())
    {
        dp_params.linkMask = LWBIT(valid_port >> 1) | LWBIT((valid_port >> 1) + 1);
    }
    else if (isArchLr10())
    {
        dp_params.linkMask = isFmodel() ? LWBIT64(valid_port) : getLinkMask();
    }
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    else if (isArchLs10())
    {
        printf("[ SKIPPED ] This functionality is not yet enabled on Laguna Seca (Bug #2867809)\n");
        return;
    }
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    else
    {
        assert(0); // Unknown Architecture
    }

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS, &dp_params, sizeof(dp_params));
    ASSERT_EQ(status, LW_OK);
}

TEST_F(LWSwitchDeviceTest, IoctlResetAndDrainLinksBadInput)
{
    LWSWITCH_RESET_AND_DRAIN_LINKS_PARAMS dp_params;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    memset(&dp_params, 0, sizeof(dp_params));

    /* Zero link mask */
    dp_params.linkMask = 0;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS, &dp_params, sizeof(dp_params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    /* Bad link mask */
    dp_params.linkMask = ~0;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS, &dp_params, sizeof(dp_params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    /* On SV10, resetting a single link is not allowed */
    if (isArchSv10())
    {
        dp_params.linkMask = 0x1;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS, &dp_params, sizeof(dp_params));
        ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
    }
}

TEST_F(LWSwitchDeviceTest, IoctlAfterDeviceUnbind)
{
    LWSWITCH_GET_INFO get_info;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    get_info.index[0] = LWSWITCH_GET_INFO_INDEX_ARCH;
    get_info.count = 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &get_info, sizeof(get_info));
    ASSERT_EQ(status, LW_OK);

    // Unbind the device and make sure the current device is stale.
    unbindRebindDevice();

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_INFO, &get_info, sizeof(get_info));
    ASSERT_EQ(status, LW_ERR_OPERATING_SYSTEM);
}

TEST_F(LWSwitchDeviceTest, IoctlAcquireCapability)
{
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    status = lwswitch_api_acquire_capability(device, LWSWITCH_CAP_FABRIC_MANAGEMENT);
    if (status == LW_ERR_NOT_SUPPORTED)
    {
        printf("[ SKIPPED  ] capabilities are not supported on this OS\n");
    }
    else
    {
        ASSERT_EQ(status, LW_OK);
    }
}

TEST_F(LWSwitchDeviceTest, IoctlAcquireCapabilityBadInput)
{
#ifdef __linux__
    LWSWITCH_ACQUIRE_CAPABILITY_PARAMS p;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    int fd;

    fd = open("/dev/null", O_RDONLY);
    ASSERT_NE(fd, -1) << "error opening /dev/null";

    p.capDescriptor = fd;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_ACQUIRE_CAPABILITY, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_GENERIC);
#else
    printf("[ SKIPPED  ] capabilities are not supported on this OS\n");
#endif // __linux__
}

TEST_F(LWSwitchDeviceTest, IoctlGetTemperature)
{
    LWSWITCH_CTRL_GET_TEMPERATURE_PARAMS p;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 numChannels;
    LwU32 i;

    memset(&p, 0, sizeof(p));

    if (isArchSv10())
    {
        printf("[ SKIPPED ] This test is not supported on Willow\n");
        return;
    }
    else if (isArchLr10()
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
             || isArchLs10()
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
           )
    {
        numChannels = LWSWITCH_NUM_CHANNELS_LR10;
        for (i = 0; i < numChannels; i++)
            p.channelMask |= LWBIT(i);
    }
    else
    {
        assert(0); // unknown Architecture
    }
 
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_TEMPERATURE, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);

    if (verbose)
    {
        for (i = 0; i < numChannels; i++)
        {
            if (p.status[i] == 0)
            {
                printf("Temperature of channel %d = %f\n",
                    i, LW_TYPES_LW_TEMP_TO_F32(p.temperature[i]));
            }
            else
            {
                printf("Temperature reading failed for channel %d. rc:%d\n",
                    i, p.status[i]);
            }
        }
    }
}

TEST_F(LWSwitchDeviceTest, IoctlGetTemperatureBadInput)
{
    LWSWITCH_CTRL_GET_TEMPERATURE_PARAMS p;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (isArchSv10())
    {
        printf("[ SKIPPED ] This test is not supported on Willow\n");
        return;
    }
    else if (isArchLr10()
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
             || isArchLs10()
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
            )
    {
        // No sensor mask
        p.channelMask = 0x0;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_TEMPERATURE, &p, sizeof(p));
        ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

        // OOB sensor mask
        if (isArchLr10())
        {
            p.channelMask = (1 << LWSWITCH_NUM_CHANNELS_LR10);
        }
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        else
        {
            p.channelMask = (1 << LWSWITCH_NUM_CHANNELS_LS10);
        }
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_TEMPERATURE, &p, sizeof(p));
        ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
    }
    else
    {
        assert(0); // Unknown architecture
    }
}

TEST_F(LWSwitchDeviceTest, IoctlGetTemperatureLimit)
{
    LWSWITCH_CTRL_GET_TEMPERATURE_LIMIT_PARAMS p;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwF32 readTemp;

    memset(&p, 0, sizeof(p));

    if (isArchSv10())
    {
        printf("[ SKIPPED ] This test is not supported on Willow\n");
        return;
    }

    p.thermalEventId = LWSWITCH_CTRL_THERMAL_EVENT_ID_WARN;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_TEMPERATURE_LIMIT, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);

    readTemp = LW_TYPES_LW_TEMP_TO_F32(p.temperatureLimit);

    /* skip this check on fmodel */
    if (!isFmodel())
    {
        ASSERT_GT(readTemp, 0);
    }

    if (verbose)
    {
        printf("%f\n", readTemp);
    }

    p.thermalEventId = LWSWITCH_CTRL_THERMAL_EVENT_ID_OVERT;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_TEMPERATURE_LIMIT, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);

    readTemp = LW_TYPES_LW_TEMP_TO_F32(p.temperatureLimit);

    /* skip this check on fmodel */
    if (!isFmodel())
    {
        ASSERT_GT(readTemp, 0);
    }

    if (verbose)
    {
        printf("%f\n", readTemp);
    }
}

TEST_F(LWSwitchDeviceTest, IoctlGetTemperatureLimitBadInput)
{
    LWSWITCH_CTRL_GET_TEMPERATURE_LIMIT_PARAMS p;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    memset(&p, 0, sizeof(p));

    if (isArchSv10())
    {
        printf("[ SKIPPED ] This test is not supported on Willow\n");
        return;
    }

    p.thermalEventId = 0xffffffff;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_TEMPERATURE_LIMIT, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
}


TEST_F(LWSwitchDeviceTest, IoctlGetBiosVersion)
{
    //TODO: Enable once the implementation in RM is done.
/*
    LWSWITCH_GET_BIOS_INFO_PARAMS p = {0};
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (isArchSv10())
    {
        printf("[ SKIPPED ] This test is not supported on Willow\n");
        return;
    }
    else if (isArchLr10()
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
             || isArchLs10()
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
            )
    {
        memset(&p, 0, sizeof(p));
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_BIOS_INFO, &p, sizeof(p));
        ASSERT_EQ(status, LWL_ERR_NOT_IMPLEMENTED );
    }
    else
    {
        assert(0); // Unknown architecture
    }
*/
}

TEST_F(LWSwitchDeviceTest, IoctlGetThroughputCounters)
{
    LWSWITCH_GET_THROUGHPUT_COUNTERS_PARAMS p, q;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (isArchSv10())
    {
        printf("[ SKIPPED ] This test is not supported on Willow\n");
        return;
    }

    /* skip this test on fmodel */
    else if (isFmodel())
    {
        printf("[ SKIPPED ] This test is not supported on fmodel\n");
        return;
    }

    else if (isArchLr10()
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
             || isArchLs10()
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
            )
    {
        // Unlikely to have a all 0xfff real value...
        memset(&p, 0xff, sizeof(p));
        q = p;

        p.counterMask = LWSWITCH_THROUGHPUT_COUNTERS_TYPE_DATA_TX |
                        LWSWITCH_THROUGHPUT_COUNTERS_TYPE_DATA_RX |
                        LWSWITCH_THROUGHPUT_COUNTERS_TYPE_RAW_TX |
                        LWSWITCH_THROUGHPUT_COUNTERS_TYPE_RAW_RX;
        p.linkMask = getLinkMask();

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_THROUGHPUT_COUNTERS,
                                      &p, sizeof(p));
        ASSERT_EQ(status, LW_OK);
        ASSERT_NE(memcmp(&p, &q, sizeof(p)), 0);
    }
    else
    {
        assert(0); // Unknown architecture
    }
}

static LW_STATUS
_lwswitch_get_fabric_state(
    int instance,
    LWSWITCH_DRIVER_FABRIC_STATE *driverState,
    LWSWITCH_DEVICE_FABRIC_STATE *deviceState,
    LWSWITCH_DEVICE_BLACKLIST_REASON *deviceReason
)
{
    LWSWITCH_GET_DEVICES_V2_PARAMS params;
    LW_STATUS status;
    LwU32 idx;

    status = lwswitch_api_get_devices(&params);
    if (status != LW_OK)
    {
        printf("Error accessing lwswitch control device\n");
        return status;
    }

    for (idx = 0; idx < params.deviceCount; idx++)
    {
        if (params.info[idx].deviceInstance == instance)
        {
            if (driverState != NULL)
                *driverState = params.info[idx].driverState;
            if (deviceState != NULL)
                *deviceState = params.info[idx].deviceState;
            if (deviceReason != NULL)
                *deviceReason = params.info[idx].deviceReason;
            return LW_OK;
        }
    }

    return LW_ERR_ILWALID_ARGUMENT;
}

TEST_F(LWSwitchDeviceTest, IoctlBlacklistDevice)
{
    LWSWITCH_BLACKLIST_DEVICE_PARAMS p;
    LWSWITCH_DEVICE_FABRIC_STATE deviceState;
    LWSWITCH_DEVICE_BLACKLIST_REASON deviceReason;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    lwlink_get_devices_info linkInfo;
    lwlink_session *session = getLwlinkSession();
    char name[20];
    int i;

    // If the device is already blacklisted then open should have failed
    status = _lwswitch_get_fabric_state(g_instance, NULL, &deviceState, NULL);
    ASSERT_EQ(status, LW_OK);
    ASSERT_NE(deviceState, LWSWITCH_DEVICE_FABRIC_STATE_BLACKLISTED) <<
        "Open of blacklisted device should have failed!";

    // Blacklist the device
    p.deviceReason = LWSWITCH_DEVICE_BLACKLIST_REASON_ACCESS_LINK_FAILURE;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_BLACKLIST_DEVICE, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);

    // Read the fabric state and verify the device was blacklisted
    status = _lwswitch_get_fabric_state(g_instance, NULL, &deviceState, &deviceReason);
    ASSERT_EQ(status, LW_OK);
    ASSERT_EQ(deviceState, LWSWITCH_DEVICE_FABRIC_STATE_BLACKLISTED);
    ASSERT_EQ(deviceReason, LWSWITCH_DEVICE_BLACKLIST_REASON_ACCESS_LINK_FAILURE);

    // Try changing the blacklist reason, and expect an error
    p.deviceReason = LWSWITCH_DEVICE_BLACKLIST_REASON_UNSPEC_DEVICE_FAILURE;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_BLACKLIST_DEVICE, &p, sizeof(p));
    ASSERT_NE(status, LW_OK);

    // Verify the fabric state was not changed by the error case
    status = _lwswitch_get_fabric_state(g_instance, NULL, &deviceState, &deviceReason);
    ASSERT_EQ(status, LW_OK);
    ASSERT_EQ(deviceState, LWSWITCH_DEVICE_FABRIC_STATE_BLACKLISTED);
    ASSERT_EQ(deviceReason, LWSWITCH_DEVICE_BLACKLIST_REASON_ACCESS_LINK_FAILURE);

    // Read link info from LWLinkCoreLib and verify the device was removed
    memset(&linkInfo, 0, sizeof(linkInfo));
    status = lwlink_api_control(session, IOCTL_LWLINK_GET_DEVICES_INFO, 
                                &linkInfo, sizeof(linkInfo));
    ASSERT_EQ(status, LW_OK);
    ASSERT_EQ(linkInfo.status, LW_OK);

    LW_SNPRINTF(name, sizeof name, "lwswitch%d", g_instance);
    // Search LWLinkCoreLib for the same device
    for (i = 0; i < (int)linkInfo.numDevice; i++)
        if (!strcmp(name, linkInfo.devInfo[i].deviceName))
            break;
    // If the device is found then it must have zero links
    if (i < (int)linkInfo.numDevice)
    {
        ASSERT_EQ(linkInfo.devInfo[i].numLinks, 0) << "Links are still registered";
    }

    // unbind/rebind after the test to clear the blacklist setting
    setUnbindOnTeardown();
}

TEST_F(LWSwitchDeviceTest, IoctlSetFMDriverState)
{
    LWSWITCH_SET_FM_DRIVER_STATE_PARAMS p;
    LWSWITCH_DRIVER_FABRIC_STATE driverState;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    // Set the driver state to Configured
    p.driverState = LWSWITCH_DRIVER_FABRIC_STATE_CONFIGURED;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_FM_DRIVER_STATE, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);

    // Verify the driver state is set as intended
    status = _lwswitch_get_fabric_state(g_instance, &driverState, NULL, NULL);
    ASSERT_EQ(status, LW_OK);
    ASSERT_EQ(driverState, LWSWITCH_DRIVER_FABRIC_STATE_CONFIGURED);

    // Change the driver state to Standby
    p.driverState = LWSWITCH_DRIVER_FABRIC_STATE_STANDBY;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_FM_DRIVER_STATE, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);

    // Verify the driver state
    status = _lwswitch_get_fabric_state(g_instance, &driverState, NULL, NULL);
    ASSERT_EQ(status, LW_OK);
    ASSERT_EQ(driverState, LWSWITCH_DRIVER_FABRIC_STATE_STANDBY);
}

TEST_F(LWSwitchDeviceTest, IoctlSetDeviceFabricState)
{
    LWSWITCH_SET_DEVICE_FABRIC_STATE_PARAMS p;
    LWSWITCH_DEVICE_FABRIC_STATE deviceState;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    // Set the device state to Configured
    p.deviceState = LWSWITCH_DEVICE_FABRIC_STATE_CONFIGURED;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_DEVICE_FABRIC_STATE, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);

    // Verify the device state
    status = _lwswitch_get_fabric_state(g_instance, NULL, &deviceState, NULL);
    ASSERT_EQ(status, LW_OK);
    ASSERT_EQ(deviceState, LWSWITCH_DEVICE_FABRIC_STATE_CONFIGURED);

    // Change the device state to Standby
    p.deviceState = LWSWITCH_DEVICE_FABRIC_STATE_STANDBY;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_DEVICE_FABRIC_STATE, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);

    // Verify the device state
    status = _lwswitch_get_fabric_state(g_instance, NULL, &deviceState, NULL);
    ASSERT_EQ(status, LW_OK);
    ASSERT_EQ(deviceState, LWSWITCH_DEVICE_FABRIC_STATE_STANDBY);
}

TEST_F(LWSwitchDeviceTest, IoctlSetFMTimeout)
{
    LWSWITCH_SET_FM_HEARTBEAT_TIMEOUT_PARAMS p;
    LWSWITCH_DRIVER_FABRIC_STATE driverState;
    LWSWITCH_DEVICE_FABRIC_STATE deviceState;
    LWSWITCH_SET_FM_DRIVER_STATE_PARAMS fm;
    LWSWITCH_SET_DEVICE_FABRIC_STATE_PARAMS ds;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    int i;

    // Set timeout to 2 seconds
    p.fmTimeout = 2000;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_FM_HEARTBEAT_TIMEOUT, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);

    // Set driver state to Configured
    fm.driverState = LWSWITCH_DRIVER_FABRIC_STATE_CONFIGURED;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_FM_DRIVER_STATE, &fm, sizeof(fm));
    ASSERT_EQ(status, LW_OK);

    // Verify driver state
    status = _lwswitch_get_fabric_state(g_instance, &driverState, NULL, NULL);
    ASSERT_EQ(status, LW_OK);
    ASSERT_EQ(driverState, LWSWITCH_DRIVER_FABRIC_STATE_CONFIGURED);

    // Send heartbeat updates for 4 seconds, and we should not see a timeout
    for (i = 0; i < 4; i++)
    {
        // Send heartbeat with device state to Configured
        ds.deviceState = LWSWITCH_DEVICE_FABRIC_STATE_CONFIGURED;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_DEVICE_FABRIC_STATE,
                                      &ds, sizeof(ds));
        ASSERT_EQ(status, LW_OK);

        osSleep(1000);

        // Verify the device state
        status = _lwswitch_get_fabric_state(g_instance, NULL, &deviceState, NULL);
        ASSERT_EQ(status, LW_OK);
        ASSERT_EQ(deviceState, LWSWITCH_DEVICE_FABRIC_STATE_CONFIGURED);
    }

    // Wait long enough to see a timeout
    osSleep(3000);

    // Verify the timeout happened
    status = _lwswitch_get_fabric_state(g_instance, &driverState, NULL, NULL);
    ASSERT_EQ(status, LW_OK);
    ASSERT_EQ(driverState, LWSWITCH_DRIVER_FABRIC_STATE_MANAGER_TIMEOUT);

    // Set timeout to a normal value
    p.fmTimeout = 10000;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_FM_HEARTBEAT_TIMEOUT, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);

    // Send a heartbeat, which should change the state from Timeout to Configured
    ds.deviceState = LWSWITCH_DEVICE_FABRIC_STATE_CONFIGURED;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_DEVICE_FABRIC_STATE,
                                  &ds, sizeof(ds));
    ASSERT_EQ(status, LW_OK);

    // Verify the state no longer shows a timeout
    status = _lwswitch_get_fabric_state(g_instance, &driverState, &deviceState, NULL);
    ASSERT_EQ(status, LW_OK);
    ASSERT_EQ(deviceState, LWSWITCH_DEVICE_FABRIC_STATE_CONFIGURED);
    ASSERT_EQ(driverState, LWSWITCH_DRIVER_FABRIC_STATE_CONFIGURED);

    // Exit the test with driver state as Standby
    fm.driverState = LWSWITCH_DRIVER_FABRIC_STATE_STANDBY;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_FM_DRIVER_STATE, &fm, sizeof(fm));
    ASSERT_EQ(status, LW_OK);

    // Verify the state
    status = _lwswitch_get_fabric_state(g_instance, &driverState, NULL, NULL);
    ASSERT_EQ(status, LW_OK);
    ASSERT_EQ(driverState, LWSWITCH_DRIVER_FABRIC_STATE_STANDBY);
}

TEST_F(LWSwitchDeviceTest, IoctlRegisterEvents)
{
#ifdef __linux__
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LWSWITCH_REGISTER_EVENTS_PARAMS reg;
    LWSWITCH_UNREGISTER_EVENTS_PARAMS unreg;
    LwU32 i;

    reg.osDescriptor = NULL;
    unreg.osDescriptor = NULL;

    reg.eventIds[0]  = LWSWITCH_DEVICE_EVENT_FATAL;
    reg.eventIds[1]  = LWSWITCH_DEVICE_EVENT_NONFATAL;
    reg.numEvents    = 1;

    // Run register->unregister sequence twice to ensure that behavior of both ioctls is correct
    for (i = 0; i < 2; i++)
    {
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_REGISTER_EVENTS, &reg, sizeof(reg));
        ASSERT_EQ(status, LW_OK);

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_UNREGISTER_EVENTS, &unreg, sizeof(unreg));
        ASSERT_EQ(status, LW_OK);
    }

    reg.numEvents = 2;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_REGISTER_EVENTS, &reg, sizeof(reg));

    // REGISTER_EVENTS only supports numEvents = 1 on Linux
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // osDescriptor should be NULL for Linux
    reg.osDescriptor = (void *) 0xDEADBEEF;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_REGISTER_EVENTS, &reg, sizeof(reg));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
#endif // __linux__
}

// TODO: Read these from GET_INFO (JIRA LWSDRV-422)
#define MCRIDTAB_DEPTH_LS10        0x0000007f
#define MCRIDEXTTAB_DEPTH_LS10     0x0000000f

struct lwswitch_mc_port_vchop
{
    LwU32 port;
    LwU32 vcHop;
};

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
static int port_vchop_compare
(
    const void *arg1,
    const void *arg2
)
{
    struct lwswitch_mc_port_vchop *a = *((struct lwswitch_mc_port_vchop **)arg1);
    struct lwswitch_mc_port_vchop *b = *((struct lwswitch_mc_port_vchop **)arg2);

    if (a->port < b->port)
    {
        return -1;
    }

    if (a->port > b->port)
    {
        return 1;
    }

    return 0;
}

//
// Helper function used to verify multicast table entries in the tests below.
//
// Port order is not guaranteed to be preserved within a given spray group,
// so this function sorts the get_params ports for comparison, assuming the
// set_params ports are already sorted as part of the test.
//
// Returns 0 if the entries are equivalent, 1 otherwise.
//
static int lwswitchCompareMCRIDEntries
(
    LWSWITCH_SET_MC_RID_TABLE_PARAMS *set_params,
    LWSWITCH_GET_MC_RID_TABLE_PARAMS *get_params
)
{
    LwU32 i, j;
    LwU32 ret = 0;
    LwU32 sg_offset[LWSWITCH_MC_MAX_SPRAYGROUPS] = { 0 };
    struct lwswitch_mc_port_vchop *port_vchop_ptrs[LWSWITCH_MC_MAX_PORTS] = { 0 };

    // verify fields
    if (set_params->entryValid != get_params->entryValid)
    {
        printf("set_params->entryValid (%d) != get_params->entryValid (%d)\n",
                set_params->entryValid, get_params->entryValid);
        return 1;
    }

    if (!set_params->entryValid)
    {
        // no need to compare anything else for invalid entry
        return 0;
    }

    if (set_params->mcSize != get_params->mcSize)
    {
        printf("set_params->mcSize (%d) != get_params->mcSize (%d)\n",
                set_params->mcSize, get_params->mcSize);
        return 1;
    }

    if (set_params->numSprayGroups != get_params->numSprayGroups)
    {
        printf("set_params->numSprayGroups (%d) != get_params->numSprayGroups (%d)\n",
                set_params->numSprayGroups, get_params->numSprayGroups);
        return 1;
    }

    if (set_params->extendedValid != get_params->extendedValid)
    {
        printf("set_params->extendedValid (%d) != get_params->extendedValid (%d)\n",
                set_params->extendedValid, get_params->extendedValid);
        return 1;
    }

    if (set_params->extendedValid)
    {
        if (set_params->extendedPtr != get_params->extendedPtr)
        {
            printf("set_params->extendedPtr (%d) != get_params->extendedPtr (%d)\n",
                    set_params->extendedPtr, get_params->extendedPtr);
            return 1;
        }
    }

    if (set_params->noDynRsp != get_params->noDynRsp)
    {
        printf("set_params->noDynRsp (%d) != get_params->noDynRsp (%d)\n",
                set_params->noDynRsp, get_params->noDynRsp);
        return 1;
    }

    // callwlate sg offsets
    sg_offset[0] = 0;
    for (i = 0; i < get_params->numSprayGroups - 1; i++)
        sg_offset[i + 1] = get_params->portsPerSprayGroup[i];

    // compare spray groups
    for (i = 0; i < get_params->numSprayGroups; i++)
    {
        if (set_params->portsPerSprayGroup[i] != get_params->portsPerSprayGroup[i])
        {
            printf(
            "set_params->portsPerSprayGroup[%d] (%d) != get_params->portsPerSprayGroup[%d] (%d)\n",
            i, set_params->portsPerSprayGroup[i], i, get_params->portsPerSprayGroup[i]);

            return 1;
        }

        if (set_params->replicaValid[i] != get_params->replicaValid[i])
        {
            printf("set_params->replicaValid[%d] (%d) != get_params->replicaValid[%d] (%d)\n",
                    i, set_params->replicaValid[i], i, get_params->replicaValid[i]);

            return 1;
        }

        // replicaOffset is the offset within the spray group to the primary replica port.
        // so it will be at sg_offset[i] + replicaOffset[i]
        if (set_params->replicaValid[i])
        {
            if (set_params->ports[sg_offset[i] + set_params->replicaOffset[i]] !=
                get_params->ports[sg_offset[i] + get_params->replicaOffset[i]])
            {
                printf("set_params primaryReplica (%d) != get_params primaryReplica (%d)\n",
                        set_params->ports[sg_offset[i] + set_params->replicaOffset[i]],
                        get_params->ports[sg_offset[i] + get_params->replicaOffset[i]]);
                return 1;
            }
        }

        // build a list of pointers to port/vchop pairs
        for (j = 0; j < set_params->portsPerSprayGroup[i]; j++)
        {
            port_vchop_ptrs[j] =
                (struct lwswitch_mc_port_vchop *)malloc(sizeof(struct lwswitch_mc_port_vchop));
            if (port_vchop_ptrs[j] == NULL)
            {
                for (i = 0; i < j; i++)
                    free(port_vchop_ptrs[i]);

                printf("Out of memory");
                return 1;
            }

            port_vchop_ptrs[j]->port = get_params->ports[sg_offset[i] + j];
            port_vchop_ptrs[j]->vcHop =  get_params->vcHop[sg_offset[i] + j];
        }

        // the input ports are assumed to be sorted, so we only sort the output ports
        qsort(port_vchop_ptrs, get_params->portsPerSprayGroup[i],
                sizeof(struct lwswitch_mc_port_vchop *), port_vchop_compare);

        // verify
        for (j = 0; j < set_params->portsPerSprayGroup[i]; j++)
        {
            if (port_vchop_ptrs[j]->port != set_params->ports[sg_offset[i] + j])
            {
                printf("port_vchop_ptrs[%d]->port (%d) != set_params->ports[%d] (%d)\n",
                        j, port_vchop_ptrs[j]->port, sg_offset[i] + j,
                        set_params->ports[sg_offset[i] + j]);
                ret = 1;
                break;
            }

            if (port_vchop_ptrs[j]->vcHop != set_params->vcHop[sg_offset[i] + j])
            {
                printf("port_vchop_ptrs[%d]->vcHop (%d) != set_params->vcHop[%d] (%d)\n",
                        j, port_vchop_ptrs[j]->vcHop, sg_offset[i] + j,
                        set_params->vcHop[sg_offset[i] + j]);
                ret = 1;
                break;
            }

        }

        // clean up
        for (j = 0; j < set_params->portsPerSprayGroup[i]; j++)
        {
            free(port_vchop_ptrs[j]);
        }

        if (ret != 0)
            break;
    }

    return ret;
}
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

TEST_F(LWSwitchDeviceTest, IoctlSetMCRIDTable)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwU32 i, j, compare_result;
    LwU32 valid_port;
    LwBool is_repeater_port;
    LWSWITCH_SET_MC_RID_TABLE_PARAMS params;
    LWSWITCH_GET_MC_RID_TABLE_PARAMS get_params;
    lwswitch_device *device = getDevice();
    LwU32 mc_port_list[LWSWITCH_MC_MAX_PORTS] = { 0 };
    LwU8 vchops[LWSWITCH_MC_MAX_PORTS] = { 0 };
    LW_STATUS status;

    memset(&params, 0, sizeof(params));
    memset(&get_params, 0, sizeof(get_params));

    if (!isArchLs10())
    {
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params,
                                        sizeof(params));
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);

        return;
    }

    getValidPortOrRepeaterPort(&valid_port, &is_repeater_port);

    if (is_repeater_port)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }

    // main table

    //
    // 1 group of 64 ports, no replica offset
    //
    for (i = 0; i < LWSWITCH_MC_MAX_PORTS; i++)
    {
        mc_port_list[i] = i;
        vchops[i] = i % 4;
    }

    params.portNum = valid_port;
    params.index = 0;
    params.extendedTable = 0;
    memcpy(params.ports, mc_port_list, (sizeof(LwU32) * LWSWITCH_MC_MAX_PORTS));
    params.mcSize = LWSWITCH_MC_MAX_PORTS;
    params.numSprayGroups = 1;
    params.entryValid = 1;
    params.portsPerSprayGroup[0] = LWSWITCH_MC_MAX_PORTS;
    params.replicaValid[0] = LW_FALSE;
    memcpy(params.vcHop, vchops, sizeof(LwU8) * LWSWITCH_MC_MAX_PORTS);

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_OK);

    // verify
    get_params.portNum = valid_port;
    get_params.index = 0;
    get_params.extendedTable = 0;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_MC_RID_TABLE, &get_params,
                                    sizeof(get_params));
    ASSERT_EQ(status, LW_OK);

    compare_result = lwswitchCompareMCRIDEntries(&params, &get_params);
    ASSERT_EQ(compare_result, 0);

    //
    // 1 group of 64 ports, replica offset provided
    //
    params.replicaOffset[0] = 5;
    params.replicaValid[0] = LW_TRUE;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_OK);

    // verify
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_MC_RID_TABLE, &get_params,
                                    sizeof(get_params));
    ASSERT_EQ(status, LW_OK);

    compare_result = lwswitchCompareMCRIDEntries(&params, &get_params);
    ASSERT_EQ(compare_result, 0);

    params.replicaOffset[0] = 0xFFFF;
    params.replicaValid[0] = 0;

    //
    // 1 group, even columns only
    //
    memset(params.ports, 0, sizeof(params.ports));
    for (i = 0; i < 32; i++)
    {
        params.ports[i] = i;
    }
    params.portsPerSprayGroup[0] = 32;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_OK);

    // verify
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_MC_RID_TABLE, &get_params,
                                    sizeof(get_params));
    ASSERT_EQ(status, LW_OK);

    compare_result = lwswitchCompareMCRIDEntries(&params, &get_params);
    ASSERT_EQ(compare_result, 0);

    //
    // 1 group, odd columns only
    //
    memset(params.ports, 0, sizeof(params.ports));
    for (i = 0, j = 32; i < 32; i++)
    {
        params.ports[i] = j++;
    }
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_OK);

    // verify
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_MC_RID_TABLE, &get_params,
                                    sizeof(get_params));
    ASSERT_EQ(status, LW_OK);

    compare_result = lwswitchCompareMCRIDEntries(&params, &get_params);
    ASSERT_EQ(compare_result, 0);

    //
    // 8 groups, 4 ports each, in overlapping tcps
    //
    params.numSprayGroups = 8;
    for (i = 0; i < 8; i++)
    {
        params.portsPerSprayGroup[i] = 4;
    }

    for (i = 0; i < 32; i++)
    {
        params.ports[i] = i;
    }

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_OK);

    // verify
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_MC_RID_TABLE, &get_params,
                                    sizeof(get_params));
    ASSERT_EQ(status, LW_OK);

    compare_result = lwswitchCompareMCRIDEntries(&params, &get_params);
    ASSERT_EQ(compare_result, 0);

    //
    // 16 groups, 4 ports each, all in the same tcp
    //
    params.numSprayGroups = 16;
    for (i = 0; i < 16; i++)
        params.portsPerSprayGroup[i] = 4;

    for (i = 0; i < LWSWITCH_MC_MAX_PORTS; i += 4)
    {
        mc_port_list[i] = 48;
        mc_port_list[i + 1] = 49;
        mc_port_list[i + 2] = 50;
        mc_port_list[i + 3] = 51;
    }
    memcpy(params.ports, mc_port_list, (sizeof(LwU32) * LWSWITCH_MC_MAX_PORTS));
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_OK);

    // verify
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_MC_RID_TABLE, &get_params,
                                    sizeof(get_params));
    ASSERT_EQ(status, LW_OK);

    compare_result = lwswitchCompareMCRIDEntries(&params, &get_params);
    ASSERT_EQ(compare_result, 0);

    //
    // program the full main table
    //
    for (i = 0; i < LWSWITCH_MC_MAX_PORTS; i++)
    {
        mc_port_list[i] = i;
        vchops[i] = i % 3;
    }
    params.numSprayGroups = 1;
    params.portsPerSprayGroup[0] = LWSWITCH_MC_MAX_PORTS;
    memcpy(params.ports, mc_port_list, (sizeof(LwU32) * LWSWITCH_MC_MAX_PORTS));

    for (i = 0; i <= MCRIDTAB_DEPTH_LS10; i++)
    {
        params.index = i;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params,
                                      sizeof(params));
        ASSERT_EQ(status, LW_OK);
    }

    // verify
    for (i = 0; i <= MCRIDTAB_DEPTH_LS10; i++)
    {
        get_params.index = i;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_MC_RID_TABLE, &get_params,
                                    sizeof(get_params));
        ASSERT_EQ(status, LW_OK);

        compare_result = lwswitchCompareMCRIDEntries(&params, &get_params);
        ASSERT_EQ(compare_result, 0);
    }

    params.index = 0;

    //
    // clear the full main table
    //
    params.entryValid = 0;

    for (i = 0; i <= MCRIDTAB_DEPTH_LS10; i++)
    {
        params.index = i;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params,
                                      sizeof(params));
        ASSERT_EQ(status, LW_OK);
    }

    // verify
    for (i = 0; i <= MCRIDTAB_DEPTH_LS10; i++)
    {
        get_params.index = i;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_MC_RID_TABLE, &get_params,
                                    sizeof(get_params));
        ASSERT_EQ(status, LW_OK);
        ASSERT_EQ(get_params.entryValid, 0);
    }

    params.entryValid = 1;
    params.index = 0;

    //
    // program full extended table
    //
    params.extendedTable = 1;

    for (i = 0; i <= MCRIDEXTTAB_DEPTH_LS10; i++)
    {
        params.index = i;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params,
                                      sizeof(params));
        ASSERT_EQ(status, LW_OK);
    }

    // verify

    get_params.extendedTable = 1;

    for (i = 0; i <= MCRIDEXTTAB_DEPTH_LS10; i++)
    {
        get_params.index = i;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_MC_RID_TABLE, &get_params,
                                    sizeof(get_params));
        ASSERT_EQ(status, LW_OK);

        compare_result = lwswitchCompareMCRIDEntries(&params, &get_params);
        ASSERT_EQ(compare_result, 0);
    }

    params.extendedTable = 0;
    params.index = 0;

    // clear full extended table

    params.extendedTable = 1;
    params.entryValid = 0;

    for (i = 0; i <= MCRIDEXTTAB_DEPTH_LS10; i++)
    {
        params.index = i;

        status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params,
                                      sizeof(params));
        ASSERT_EQ(status, LW_OK);
    }

    // verify
    for (i = 0; i <= MCRIDEXTTAB_DEPTH_LS10; i++)
    {
        get_params.index = i;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_MC_RID_TABLE, &get_params,
                                    sizeof(get_params));
        ASSERT_EQ(status, LW_OK);

        ASSERT_EQ(get_params.entryValid, params.entryValid);
    }

    params.extendedTable = 0;
    params.entryValid = 1;
    params.index = 0;

    //
    // 1 group, 1 port, main table
    //
    memset(&params, 0, sizeof(params));
    params.portNum = valid_port;
    params.mcSize = 1;
    params.numSprayGroups = 1;
    params.portsPerSprayGroup[0] = 1;
    params.entryValid = 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params,
                                      sizeof(params));
    ASSERT_EQ(status, LW_OK);

    // verify
    memset(&get_params, 0, sizeof(get_params));
    get_params.portNum = valid_port;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_MC_RID_TABLE, &get_params,
                                    sizeof(get_params));
    ASSERT_EQ(status, LW_OK);

    compare_result = lwswitchCompareMCRIDEntries(&params, &get_params);
    ASSERT_EQ(compare_result, 0);

    //
    // 1 group, 1 port, extended table
    //
    memset(&params, 0, sizeof(params));
    params.portNum = valid_port;
    params.extendedTable = 1;
    params.mcSize = 1;
    params.numSprayGroups = 1;
    params.portsPerSprayGroup[0] = 1;
    params.entryValid = 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params,
                                      sizeof(params));
    ASSERT_EQ(status, LW_OK);

    // verify
    memset(&get_params, 0, sizeof(get_params));
    get_params.portNum = valid_port;
    get_params.extendedTable = 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_MC_RID_TABLE, &get_params,
                                    sizeof(get_params));
    ASSERT_EQ(status, LW_OK);

    compare_result = lwswitchCompareMCRIDEntries(&params, &get_params);
    ASSERT_EQ(compare_result, 0);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
}

TEST_F(LWSwitchDeviceTest, IoctlSetMCRIDTableRepeaterMode)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LWSWITCH_SET_MC_RID_TABLE_PARAMS params;
    LwU32 repeater_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (!isArchLs10())
    {
        printf("[ SKIPPED ] This test is only supported on LS10\n");
        return;
    }

    repeaterPort(&repeater_port);
    if (repeater_port == LWSWITCH_ILWALID_PORT)
    {
        printf("[ SKIPPED  ] No port in Repeater Mode.\n");
        return;
    }

    memset(&params, 0, sizeof(params));

    params.portNum = repeater_port;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
}

TEST_F(LWSwitchDeviceTest, IoctlSetMCRIDTableBadInput)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwU32 i;
    LwU32 valid_port;
    LwBool is_repeater_port;
    LWSWITCH_SET_MC_RID_TABLE_PARAMS params;
    lwswitch_device *device = getDevice();
    LwU32 mc_port_list[LWSWITCH_MC_MAX_PORTS] = { 0 };
    LW_STATUS status;

    if (!isArchLs10())
    {
        printf("[ SKIPPED ] This test is only supported on LS10\n");
        return;
    }

    getValidPortOrRepeaterPort(&valid_port, &is_repeater_port);

    if (is_repeater_port)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }

    // main table

    memset(&params, 0, sizeof(params));

    // initial common settings
    memset(&params, 0, sizeof(params));
    params.mcSize = 4;
    params.numSprayGroups = 1;
    params.portsPerSprayGroup[0] = 4;
    params.ports[1] = 1;
    params.ports[2] = 2;
    params.ports[3] = 3;
    params.entryValid = 1;

    // invalid port
    params.portNum = LWSWITCH_ILWALID_PORT;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.portNum = valid_port;

    // index out of range for valid entry - main table
    params.index = MCRIDTAB_DEPTH_LS10 + 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.index = 0;

    // index out of range for valid entry - extended table
    params.extendedTable = 1;
    params.index = MCRIDEXTTAB_DEPTH_LS10 + 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.index = 0;
    params.extendedTable = 0;

    // index out of range for invalid entry - main table
    params.entryValid = 0;
    params.index = MCRIDTAB_DEPTH_LS10 + 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.entryValid = 1;
    params.index = 0;

    // index out of range for invalid entry - extended table
    params.extendedTable = 1;
    params.entryValid = 0;
    params.index = MCRIDEXTTAB_DEPTH_LS10 + 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.entryValid = 1;
    params.index = 0;
    params.extendedTable = 0;

    // extended ptr specified for extended table
    params.extendedPtr = 0;
    params.extendedValid = 1;
    params.extendedTable = 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.extendedValid = 0;
    params.extendedTable = 0;

    // extended ptr out of range
    params.extendedPtr = MCRIDEXTTAB_DEPTH_LS10 + 1;
    params.extendedValid = 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.extendedValid = 0;
    params.extendedPtr = 0;

    // port number out of range in ports array - main table
    params.ports[3] = LWSWITCH_MC_MAX_PORTS;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.ports[3] = 3;

    // port number out of range in ports array - extended table
    params.extendedTable = 1;
    params.ports[3] = LWSWITCH_MC_MAX_PORTS;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.extendedTable = 0;
    params.ports[3] = 3;

    // invalid vchop value
    params.vcHop[2] = 5;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.vcHop[2] = 0;

    // mcsize over max
    params.mcSize = LWSWITCH_MC_MAX_PORTS + 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.mcSize = 4;

    // empty mcsize (mcsize only checked for entryValid)
    params.mcSize = 0;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.mcSize = 4;

    // invalid numspraygroups value
    params.numSprayGroups = LWSWITCH_MC_MAX_SPRAYGROUPS + 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.numSprayGroups = 1;

    // numSprayGroups must be nonzero
    params.numSprayGroups = 0;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.numSprayGroups = 1;

    // portsPerSprayGroup over max
    params.portsPerSprayGroup[0] = LWSWITCH_MC_MAX_PORTS + 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.portsPerSprayGroup[0] = 4;

    // portsPerSprayGroup under min
    params.portsPerSprayGroup[0] = 0;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.portsPerSprayGroup[0] = 4;

    // replicaOffset out of range
    params.replicaOffset[0] = params.portsPerSprayGroup[0];
    params.replicaValid[0] = 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.replicaOffset[0] = 0;
    params.replicaValid[0] = 0;

    // duplicate port number in spray group
    params.ports[2] = 3;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.ports[2] = 2;

    // 16 groups, 4 ports each, overflow due to TCP overlap
    memset(&params, 0, sizeof(params));
    for (i = 0; i < LWSWITCH_MC_MAX_PORTS; i++)
        mc_port_list[i] = i;

    params.numSprayGroups = LWSWITCH_MC_MAX_SPRAYGROUPS;
    for (i = 0; i < LWSWITCH_MC_MAX_SPRAYGROUPS; i++)
        params.portsPerSprayGroup[i] = 4;

    memcpy(params.ports, mc_port_list, (sizeof(LwU32) * LWSWITCH_MC_MAX_PORTS));

    params.portNum = valid_port;
    params.mcSize = LWSWITCH_MC_MAX_PORTS;
    params.entryValid = 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_MORE_PROCESSING_REQUIRED);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
}

TEST_F(LWSwitchDeviceTest, IoctlGetMCRIDTableRepeaterMode)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LWSWITCH_GET_MC_RID_TABLE_PARAMS params;
    LwU32 repeater_port = LWSWITCH_ILWALID_PORT;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (!isArchLs10())
    {
        printf("[ SKIPPED ] This test is only supported on LS10\n");
        return;
    }

    repeaterPort(&repeater_port);
    if (repeater_port == LWSWITCH_ILWALID_PORT)
    {
        printf("[ SKIPPED  ] No port in Repeater Mode.\n");
        return;
    }

    memset(&params, 0, sizeof(params));

    params.portNum = repeater_port;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
}

TEST_F(LWSwitchDeviceTest, IoctlGetMCRIDTableBadInput)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LwU32 valid_port;
    LwBool is_repeater_port;
    LWSWITCH_GET_MC_RID_TABLE_PARAMS params;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (!isArchLs10())
    {
        printf("[ SKIPPED ] This test is only supported on LS10\n");
        return;
    }

    getValidPortOrRepeaterPort(&valid_port, &is_repeater_port);
    if (is_repeater_port)
    {
        printf("[ SKIPPED  ] All ports in Repeater Mode.\n");
        return;
    }

    memset(&params, 0, sizeof(params));

    // invalid port
    params.portNum = LWSWITCH_ILWALID_PORT;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.portNum = valid_port;

    // index out of range for main table
    params.index = MCRIDTAB_DEPTH_LS10 + 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    params.index = 0;

    // index out of range for extended table
    params.extendedTable = 1;
    params.index = MCRIDEXTTAB_DEPTH_LS10 + 1;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_MC_RID_TABLE, &params, sizeof(params));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
}

TEST_F(LWSwitchDeviceTest, IoctlGetLWLinkCounters)
{
    LwU64 linkMask = getLinkMask();
    LwU32 linkCount = getLinkCount();
    LWSWITCH_LWLINK_GET_COUNTERS_PARAMS params;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU8 linkId;
    LwU32 counterMask;

    /* skip this test on fmodel */
    if (isFmodel())
    {
        printf("[ SKIPPED ] This test is not supported on fmodel\n");
        return;
    }

    memset(&params, 0, sizeof(params));

    counterMask = LWSWITCH_LWLINK_COUNTER_TL_TX0 | 
                  LWSWITCH_LWLINK_COUNTER_TL_TX1 | 
                  LWSWITCH_LWLINK_COUNTER_TL_RX0 | 
                  LWSWITCH_LWLINK_COUNTER_TL_RX1 | 
                  LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT | 
                  LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L0 | 
                  LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L1 | 
                  LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L2 | 
                  LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L3 | 
                  LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L4 | 
                  LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L5 | 
                  LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L6 | 
                  LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L7 | 
                  LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY | 
                  LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY | 
                  LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_REPLAY |
                  LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_PASS |
                  LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_FAIL;

    for (linkId = 0; linkId < linkCount; linkId++)
    {
        if (LWBIT64(linkId) & linkMask)
        {
            params.linkId = linkId;
            params.counterMask = counterMask;

            status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_LWLINK_COUNTERS, &params, sizeof(params));
            ASSERT_EQ(status, LW_OK);

            if (verbose)
            {
                printf(" LinkID = %d\n", linkId);
                
                printf("\t TL_TX0 = %llu\n", params.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_TL_TX0)]);
                printf("\t TL_TX1 = %llu\n", params.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_TL_TX1)]);
                printf("\t TL_RX0 = %llu\n", params.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_TL_RX0)]);
                printf("\t TL_rX1 = %llu\n", params.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_TL_RX1)]);
                printf("\t CRC_FLIT = %llu\n", params.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT)]);
                printf("\t CRC_LANE_L0 = %llu\n", params.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L0)]);
                printf("\t CRC_LANE_L1 = %llu\n", params.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L1)]);
                printf("\t CRC_LANE_L2 = %llu\n", params.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L2)]);
                printf("\t CRC_LANE_L3 = %llu\n", params.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L3)]);
                printf("\t CRC_LANE_L4 = %llu\n", params.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L4)]);
                printf("\t CRC_LANE_L5 = %llu\n", params.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L5)]);
                printf("\t CRC_LANE_L6 = %llu\n", params.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L6)]);
                printf("\t CRC_LANE_L7 = %llu\n", params.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L7)]);
                printf("\t TX_ERR_REPLAY = %llu\n", params.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY)]);
                printf("\t TX_ERR_RECOVERY = %llu\n", params.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY)]);
                printf("\t RX_ERR_REPLAY = %llu\n", params.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_REPLAY)]);
                printf("\t PHY PASS = %llu\n", params.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_PASS)]);
                printf("\t PHY FAIL = %llu\n", params.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_FAIL)]);
            }
        }
    }
}

TEST_F(LWSwitchDeviceTest, IoctlGetLWLinkEccErrors)
{
    LwU64 linkMask = getLinkMask();
    LwU32 linkCount = getLinkCount();
    LWSWITCH_GET_LWLINK_ECC_ERRORS_PARAMS params;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU8 linkId, laneId;

    memset(&params, 0, sizeof(params));

    params.linkMask = linkMask;

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_LWLINK_ECC_ERRORS, &params, sizeof(params));
    if (isArchSv10())
    {
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
        return;
    }
    else
    {
        ASSERT_EQ(status, LW_OK);
    }

    if (verbose)
    {
        for (linkId = 0; linkId < linkCount; linkId++)
        {
            if (LWBIT64(linkId) & linkMask)
            {
                printf(" LinkID = %d\n", linkId);
                printf("\t EccDecFailed = %d; eccDecFailedOverflowed = %d\n", 
                params.errorLink[linkId].eccDecFailed, 
                params.errorLink[linkId].eccDecFailedOverflowed);

                for (laneId = 0; laneId < LWSWITCH_LWLINK_MAX_LANES; laneId++)
                {
                    printf("\t Lane%d: valid = %d; overflowed = %d; eccErrorValue = %d\n", 
                           laneId, 
                           params.errorLink[linkId].errorLane[laneId].valid, 
                           params.errorLink[linkId].errorLane[laneId].overflowed, 
                           params.errorLink[linkId].errorLane[laneId].eccErrorValue);
                }
            }
        }
    }
}

TEST_F(LWSwitchDeviceTest, IoctlCciCmisPresence)
{
    LWSWITCH_CCI_CMIS_PRESENCE_PARAMS params;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    memset(&params, 0, sizeof(params));

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_CMIS_PRESENCE, &params, sizeof(params));

    ASSERT_EQ(status, LW_OK);

    if (verbose)
    {
        if ((params.cagesMask & params.modulesMask) != params.modulesMask)
        {
            printf(" Error: Plugged in module not in cages mask\n");
        }

        printf(" Cages Mask = 0x%x, Modules Mask = 0x%x\n", params.cagesMask, params.modulesMask); 
    }
    ASSERT_TRUE((params.cagesMask & params.modulesMask) == params.modulesMask);
}

TEST_F(LWSwitchDeviceTest, IoctlGetInforomLWLinkMaxCorrectableErrorRate)
{
    LwU64 linkMask = getLinkMask();
    LwU32 linkCount = getLinkCount();
    LWSWITCH_GET_LWLINK_MAX_CORRECTABLE_ERROR_RATES_PARAMS params;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU8 linkId;

    /* skip this test on fmodel */
    if (isFmodel())
    {
        printf("[ SKIPPED ] This test is not supported on fmodel\n");
        return;
    }

    memset(&params, 0, sizeof(params));

    for (linkId = 0; linkId < linkCount; linkId++)
    {
        if (LWBIT64(linkId) & linkMask)
        {
            params.linkId = linkId;

            status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_LWLINK_MAX_ERROR_RATES, &params, sizeof(params));
            if (isArchSv10() || !isInforomLWLSupported())
            {
                ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
                return;
            }
            else
            {
                ASSERT_EQ(status, LW_OK);
            }

            if (verbose)
            {
                printf("\n linkID = %d", linkId);
                printf("\n\t DailyMaxCorrectableErrorRates");
                for (int i = 0; i < LWSWITCH_LWLINK_MAX_CORRECTABLE_ERROR_DAYS; i++)
                {
                    printf("\n\t\t lastUpdated = %d", params.dailyMaxCorrectableErrorRates[i].lastUpdated);
                    printf("\n\t\t flitCrcErrorsPerMinute = %d", params.dailyMaxCorrectableErrorRates[i].flitCrcErrorsPerMinute);
                    for (int j = 0; j < LWSWITCH_LWLINK_MAX_LANES; j++)
                    {
                        printf("\n\t\t laneCrcErrorsPerMinute[%d] = %d", j, params.dailyMaxCorrectableErrorRates[i].laneCrcErrorsPerMinute[j]);
                    }
                }
                
                printf("\n\t MonthlyMaxCorrectableErrorRates");
                for (int i = 0; i < LWSWITCH_LWLINK_MAX_CORRECTABLE_ERROR_MONTHS; i++)
                {
                    printf("\n\t\t lastUpdated = %d", params.monthlyMaxCorrectableErrorRates[i].lastUpdated);
                    printf("\n\t\t flitCrcErrorsPerMinute = %d", params.monthlyMaxCorrectableErrorRates[i].flitCrcErrorsPerMinute);
                    for (int j = 0; j < LWSWITCH_LWLINK_MAX_LANES; j++)
                    {
                        printf("\n\t\t laneCrcErrorsPerMinute[%d] = %d", j, params.monthlyMaxCorrectableErrorRates[i].laneCrcErrorsPerMinute[j]);
                    }
                }
                printf("\n");
            }
        }
    }
}

TEST_F(LWSwitchDeviceTest, IoctlGetInforomLWLinkErrors)
{
    LWSWITCH_GET_LWLINK_ERROR_COUNTS_PARAMS params;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    /* skip this test on fmodel */
    if (isFmodel())
    {
        printf("[ SKIPPED ] This test is not supported on fmodel\n");
        return;
    }

    memset(&params, 0, sizeof(params));

    do
    {
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_LWLINK_ERROR_COUNTS, &params, sizeof(params));
        if (isArchSv10() || !isInforomLWLSupported())
        {
            ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
            return;
        }
        else
        {
            ASSERT_EQ(status, LW_OK);
        }

        if (verbose)
        {
            printf("\n === errorCount = %d ===", params.errorCount);
            for (LwU32 i = 0; i < params.errorCount; i++)
            {
                printf("\n\t linkId = %d, error = %d, timeStamp = %d, count = %lld",
                    params.errorLog[i].instance,
                    params.errorLog[i].error,
                    params.errorLog[i].timeStamp,
                    params.errorLog[i].count);
            }
            printf("\n");
        }
    } while(params.errorCount > 0);
}

TEST_F(LWSwitchDeviceTest, IoctlGetInforomEccErrors)
{
    LWSWITCH_GET_ECC_ERROR_COUNTS_PARAMS params;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    /* skip this test on fmodel */
    if (isFmodel())
    {
        printf("[ SKIPPED ] This test is not supported on fmodel\n");
        return;
    }

    memset(&params, 0, sizeof(params));

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_ECC_ERROR_COUNTS, &params, sizeof(params));
    if (isArchSv10())
    {
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
        return;
    }
    else
    {
        ASSERT_EQ(status, LW_OK);
    }

    if (verbose)
    {
        printf("\n === errorCount = %d, uncorrectedTotal = %lld, correctedTotal = %lld ===", params.errorCount, params.uncorrectedTotal, params.correctedTotal);
        for (LwU32 i = 0; i < params.errorCount; i++)
        {
            printf("\n\t sxid = %d, linkId = %d, lastErrorTimestamp = %d, bAddressValid=%d, address=0x%x, correctedCount = %d, uncorrectedCount = %d",
                params.errorLog[i].sxid,
                params.errorLog[i].linkId,
                params.errorLog[i].lastErrorTimestamp,
                params.errorLog[i].bAddressValid,
                params.errorLog[i].address,
                params.errorLog[i].correctedCount,
                params.errorLog[i].uncorrectedCount);
        }
        printf("\n");
    }
}

TEST_F(LWSwitchDeviceTest, IoctlGetInforomSxidErrors)
{
    LWSWITCH_GET_SXIDS_PARAMS params;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    /* skip this test on fmodel */
    if (isFmodel())
    {
        printf("[ SKIPPED ] This test is not supported on fmodel\n");
        return;
    }

    memset(&params, 0, sizeof(params));

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_SXIDS, &params, sizeof(params));
    if (isArchSv10() || !isInforomBBXSupported())
    {
        ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
        return;
    }
    else
    {
        ASSERT_EQ(status, LW_OK);
    }

    if (verbose)
    {
        printf("\n === sxidCount = %d ===", params.sxidCount);

        for (int i = 0; i < LWSWITCH_SXID_ENTRIES_NUM; i++)
        {
            if (params.sxidFirst[i].timestamp == 0)
            {
                break; //reach the end
            }

            printf("\n\t\t number = %d, timestamp = %d",
                params.sxidFirst[i].sxid,
                params.sxidFirst[i].timestamp);
        }

        for (int i = 0; i < LWSWITCH_SXID_ENTRIES_NUM; i++)
        {
            if (params.sxidLast[i].timestamp == 0)
            {
                break; //reach the end
            }

            printf("\n\t\t number = %d, timestamp = %d",
                params.sxidLast[i].sxid,
                params.sxidLast[i].timestamp);
        }
        printf("\n");
    }
}

TEST_F(LWSwitchDeviceTest, IoctlCciCmisMapping)
{
    LWSWITCH_CCI_CMIS_MEMORY_ACCESS_READ_PARAMS readParams;
    LWSWITCH_CCI_CMIS_LWLINK_MAPPING_PARAMS mappingParams;
    LWSWITCH_CCI_CMIS_PRESENCE_PARAMS presenceParams;
    lwswitch_device *device = getDevice();
    LwU64 linkMask;
    LwU64 encodedValue;
    LwU8 laneMask;
    LwU8 linkId;
    LW_STATUS status;
    LwU8 cageIndex;

    memset(&readParams, 0, sizeof(readParams));
    memset(&presenceParams, 0, sizeof(presenceParams));

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_CMIS_MEMORY_ACCESS_READ, &readParams, sizeof(readParams));

    if (status == LW_ERR_NOT_SUPPORTED)
    {
        printf("[ SKIPPED  ] CCI not supported.\n");
        return;
    }

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_CMIS_PRESENCE, &presenceParams, sizeof(presenceParams));
    ASSERT_EQ(status, LW_OK);

    FOR_EACH_INDEX_IN_MASK(32, cageIndex, presenceParams.cagesMask)
    {
        if (verbose)
        {
            printf("Cage index = %d\n", cageIndex); 
        }

        memset(&mappingParams, 0, sizeof(mappingParams));

        mappingParams.cageIndex = cageIndex;
        status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_CMIS_LWLINK_MAPPING, &mappingParams, sizeof(mappingParams));

        if (status == LW_ERR_NOT_SUPPORTED)
        {
            printf("[ SKIPPED  ] CCI not supported.\n");
            return;
        }

        ASSERT_EQ(status, LW_OK);

        if (verbose)
        {
            linkMask = mappingParams.linkMask;
            encodedValue = mappingParams.encodedValue;
            printf("\tLink Mask = 0x%llx\n", linkMask); 
            FOR_EACH_INDEX_IN_MASK(64, linkId, linkMask)
            {    
                LWSWITCH_CCI_CMIS_LWLINK_MAPPING_GET_OSFP_LANE_MASK(laneMask, linkId, encodedValue); 
                printf("\t\tLink %d, OSFP Lane Mask = 0x%x\n", linkId, laneMask); 
            }
            FOR_EACH_INDEX_IN_MASK_END;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;
}

struct lwswitch_cci_cmis_config
{
    LwU8 type;
    char name[20];
    LwU8 bank;
    LwU8 page;
    LwU8 address;
    LwU8 count;
};

enum CMIS_TYPE {
    CMIS_IDENTIFIER,
    CMIS_MODULE_TYPE,
    CMIS_VENDOR_NAME,
    CMIS_CABLE_LENGTH,
    CMIS_VENDOR_PN,
    CMIS_VENDOR_RN,
    CMIS_VENDOR_SN,
    CMIS_TX_POWER,
    CMIS_RX_POWER,
    CMIS_TEMP,
    CMIS_SUPPLY_VOLT,
    CMIS_TEC_LWRR,
    CMIS_TYPE_MAX,
};

struct lwswitch_cci_cmis_config cmisCfg[CMIS_TYPE_MAX] =
{
    /* type, bank, page, byte offset, byte length */
    {CMIS_IDENTIFIER,   "CMIS_IDENTIFIER",  0, 0,    0,   1},
    {CMIS_MODULE_TYPE,  "CMIS_MODULE_TYPE", 0, 0,    85,  1},  /* Page 89, Table 8-12 Byte 85 Module Media Type Encodings */
    {CMIS_VENDOR_NAME,  "CMIS_VENDOR_NAME", 0, 0,    129, 16}, /* ASCII */
    {CMIS_CABLE_LENGTH, "CMIS_CABLE_LENGTH",0, 0,    202, 1},
    {CMIS_VENDOR_PN,    "CMIS_VENDOR_PN",   0, 0,    148, 16}, /* Vendor part number ASCII */
    {CMIS_VENDOR_RN,    "CMIS_VENDOR_RN",   0, 0,    164, 2},  /* Vendor revision number ASCII */
    {CMIS_VENDOR_SN,    "CMIS_VENDOR_SN",   0, 0,    166, 16}, /* Serial number ASCII */
    {CMIS_TX_POWER,     "CMIS_TX_POWER",    0, 0x11, 154, 16}, /* TX lane 1-8 Power */
    {CMIS_RX_POWER,     "CMIS_RX_POWER",    0, 0x11, 186, 16}, /* RX lane 1-8 Power */
    {CMIS_TEMP,         "CMIS_TEMP",        0, 0,    14,  2},  /* Table 8-6 Modules Monitors */
    {CMIS_SUPPLY_VOLT,  "CMIS_SUPPLY_VOLT", 0, 0,    16,  2},  /* Table 8-6 */
    {CMIS_TEC_LWRR,     "CMIS_TEC_LWRR",    0, 0,    18,  2},  /* Table 8-6 */
};

static double pow(LwU8 base, LwU8 power)
{
    double ret = 1;
    LwU8 i;

    for (i = 0; i < power; i++)
    {
        ret = ret * base;
    }

    return ret;
}

static void print_cmis_info(LwU8 type, LwU8 *data)
{
    LwU8 multiplier, length, lane;
    LwU16 power;
    LwS16 tmp;

    printf("\t\t%s: ", cmisCfg[type].name);
    switch (type)
    {
        case CMIS_VENDOR_NAME:
        case CMIS_VENDOR_PN:
        case CMIS_VENDOR_RN:
        case CMIS_VENDOR_SN:
        {
            printf("%s\n", data);
            break;
        }
        case CMIS_IDENTIFIER:
        case CMIS_MODULE_TYPE:
        {
            printf("0x%x\n", data[0]);
            break;
        }
        case CMIS_CABLE_LENGTH:
        {
            multiplier = data[0] >> 6;
            length = data[0] & 0x3F;
            printf("%.2f\n", pow(10, multiplier) * length / 10);
            break;
        }
        case CMIS_TX_POWER:
        case CMIS_RX_POWER:
        {
            for (lane = 0; lane < 8; lane++)
            {
                power = ((LwU16)(data[2 * lane]) << 8) + data[2 * lane + 1];
                printf("%d\t", power);
            }
            printf("\n");
            break;
        }
        case CMIS_TEMP:
        case CMIS_SUPPLY_VOLT:
        case CMIS_TEC_LWRR:
        {
            tmp = ((LwS16)(data[0]) << 8) + data[1];
            printf("%d\n", tmp);
            break;
        }
        default:
        {
            printf("\n");
            break;
        }
    }
}

TEST_F(LWSwitchDeviceTest, IoctlCciCmisRead)
{
    LWSWITCH_CCI_CMIS_MEMORY_ACCESS_READ_PARAMS readParams;
    LWSWITCH_CCI_CMIS_MEMORY_ACCESS_WRITE_PARAMS writeParams;
    LWSWITCH_CCI_CMIS_PRESENCE_PARAMS presenceParams;
    lwswitch_device *device = getDevice();
    LwU8 cageIndex, type;
    LW_STATUS status;

    memset(&readParams, 0, sizeof(readParams));
    memset(&presenceParams, 0, sizeof(presenceParams));

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_CMIS_MEMORY_ACCESS_READ, &readParams, sizeof(readParams));

    if (status == LW_ERR_NOT_SUPPORTED)
    {
        printf("[ SKIPPED  ] CCI not supported.\n");
        return;
    }

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_CMIS_PRESENCE, &presenceParams, sizeof(presenceParams));
    ASSERT_EQ(status, LW_OK);

    FOR_EACH_INDEX_IN_MASK(32, cageIndex, presenceParams.cagesMask)
    {
        if (verbose)
        {
            printf(" Cage index = %d\n", cageIndex); 
        }

        memset(&readParams, 0, sizeof(readParams));
        memset(&writeParams, 0, sizeof(writeParams));
        
        // Testing whether bank and page remain the same after an access
        if (presenceParams.modulesMask & LWBIT(cageIndex))
        {     
            //       
            // manually set bank and page
            // bank 0, page 11h is required for pages modules
            // see section 8.8 pg 136 http://www.qsfp-dd.com/wp-content/uploads/2019/05/QSFP-DD-CMIS-rev4p0.pdf
            // no need to provide bank and page params since address is < 0x80
            //
            writeParams.cageIndex = cageIndex;
            writeParams.address = 126;
            writeParams.count = 2;
            writeParams.data[0] = 0x0; // bank
            writeParams.data[1] =  0x11; // page
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_CMIS_MEMORY_ACCESS_WRITE, &writeParams, sizeof(writeParams));
            ASSERT_EQ(status, LW_OK);

            //
            // read vendor name 
            // see section 8.3 pg 92 http://www.qsfp-dd.com/wp-content/uploads/2019/05/QSFP-DD-CMIS-rev4p0.pdf 
            //
            readParams.cageIndex = cageIndex;
            readParams.bank = 0x0;
            readParams.page = 0x0;
            readParams.address = 129;
            readParams.count = 16;
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_CMIS_MEMORY_ACCESS_READ, &readParams, sizeof(readParams));
            ASSERT_EQ(status, LW_OK);

            if (verbose)
            {
                printf(" Vendor name: %s\n", readParams.data); 
            }

            for (type = 0; type < CMIS_TYPE_MAX; type++)
            {
                memset(&readParams, 0, sizeof(readParams));
                readParams.cageIndex = cageIndex;
                readParams.bank = cmisCfg[type].bank;
                readParams.page = cmisCfg[type].page;
                readParams.address = cmisCfg[type].address;
                readParams.count = cmisCfg[type].count;
                status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_CMIS_MEMORY_ACCESS_READ, &readParams, sizeof(readParams));
                ASSERT_EQ(status, LW_OK);

                if (verbose)
                {
                    print_cmis_info(type, readParams.data);
                }
            }

            readParams.address = 126;
            readParams.count = 2;
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_CMIS_MEMORY_ACCESS_READ, &readParams, sizeof(readParams));
            ASSERT_EQ(status, LW_OK);

            // bank and page should remain the same
            ASSERT_EQ(readParams.data[0], 0x0);
            ASSERT_EQ(readParams.data[1], 0x11);
        }
        else
        {
            // reads from unplugged in modules should fail
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_CMIS_MEMORY_ACCESS_READ, &readParams, sizeof(readParams));
            ASSERT_TRUE(status != LW_OK);
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;
}

TEST_F(LWSwitchDeviceTest, IoctlCciCmisWrite)
{
    LWSWITCH_CCI_CMIS_MEMORY_ACCESS_READ_PARAMS readParams;
    LWSWITCH_CCI_CMIS_MEMORY_ACCESS_WRITE_PARAMS writeParams;
    LWSWITCH_CCI_CMIS_PRESENCE_PARAMS presenceParams;
    lwswitch_device *device = getDevice();
    LwU8 cageIndex;
    LW_STATUS status;

    memset(&writeParams, 0, sizeof(writeParams));
    memset(&presenceParams, 0, sizeof(presenceParams));

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_CMIS_MEMORY_ACCESS_WRITE, &writeParams, sizeof(writeParams));

    if (status == LW_ERR_NOT_SUPPORTED)
    {
        printf("[ SKIPPED  ] CCI not supported.\n");
        return;
    }

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_CMIS_PRESENCE, &presenceParams, sizeof(presenceParams));
    ASSERT_EQ(status, LW_OK);

    FOR_EACH_INDEX_IN_MASK(32, cageIndex, presenceParams.cagesMask)
    {
        if (verbose)
        {
            printf(" Cage index = %d\n", cageIndex); 
        }

        memset(&readParams, 0, sizeof(readParams));
        memset(&writeParams, 0, sizeof(writeParams));

        // Testing whether bank and page remain the same after an access
        if (presenceParams.modulesMask & LWBIT(cageIndex))
        {
            //       
            // manually set bank and page
            // bank 0, page 11h is required for pages modules
            // see section 8.8 pg 136 http://www.qsfp-dd.com/wp-content/uploads/2019/05/QSFP-DD-CMIS-rev4p0.pdf
            // no need to provide bank and page params since address is < 0x80
            //
            writeParams.cageIndex = cageIndex;
            writeParams.address = 126;
            writeParams.count = 2;
            writeParams.data[0] = 0x0; // bank
            writeParams.data[1] =  0x11; // page
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_CMIS_MEMORY_ACCESS_WRITE, &writeParams, sizeof(writeParams));
            ASSERT_EQ(status, LW_OK);

            //
            // read vendor name 
            // see section 8.3 pg 92 http://www.qsfp-dd.com/wp-content/uploads/2019/05/QSFP-DD-CMIS-rev4p0.pdf 
            //
            readParams.cageIndex = cageIndex;
            readParams.bank = 0x0;
            readParams.page = 0x0;
            readParams.address = 129;
            readParams.count = 16;
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_CMIS_MEMORY_ACCESS_READ, &readParams, sizeof(readParams));
            ASSERT_EQ(status, LW_OK);

            if (verbose)
            {
                printf(" Vendor name: %s\n", readParams.data); 
            }

            readParams.address = 126;
            readParams.count = 2;
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_CMIS_MEMORY_ACCESS_READ, &readParams, sizeof(readParams));
            ASSERT_EQ(status, LW_OK);

            // bank and page should remain the same
            ASSERT_EQ(readParams.data[0], 0x0);
            ASSERT_EQ(readParams.data[1], 0x11);
        }
        else
        {
            // writes to unplugged in modules should fail
            status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_CMIS_MEMORY_ACCESS_WRITE, &writeParams, sizeof(writeParams));
            ASSERT_TRUE(status != LW_OK);
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;
}

TEST_F(LWSwitchDeviceTest, IoctlCciCmisCageBezelMarking)
{
    LWSWITCH_CCI_CMIS_CAGE_BEZEL_MARKING_PARAMS params;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    memset(&params, 0, sizeof(params));

    params.cageIndex = 0;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_CMIS_CAGE_BEZEL_MARKING, &params, sizeof(params));

    if (status == LW_ERR_NOT_SUPPORTED)
    {
        printf("[ SKIPPED  ] CCI not supported.\n");
        return;
    }

    ASSERT_EQ(status, LW_OK);

    if (verbose)
    {
        printf(" Bezel Info: %s\n", params.bezelMarking); 
    }
}

TEST_F(LWSwitchDeviceTest, IoctlCciGetGradingValues)
{
    LWSWITCH_CCI_GET_GRADING_VALUES_PARAMS params;
    lwswitch_device *device = getDevice();
    LwU8 laneNum;
    LW_STATUS status;

    memset(&params, 0, sizeof(params));

    params.linkId = 0;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_CCI_GET_GRADING_VALUES, &params, sizeof(params));

    if (status == LW_ERR_NOT_SUPPORTED)
    {
        printf("[ SKIPPED  ] CCI not supported.\n");
        return;
    }

    ASSERT_EQ(status, LW_OK);

    if (verbose)
    {
        printf(" Link %d TX-Input Initial Tuning BER:", params.linkId);
        FOR_EACH_INDEX_IN_MASK(8, laneNum, params.laneMask)
        {
            printf(" %d,", params.grading.tx_init[laneNum]);
        }
        FOR_EACH_INDEX_IN_MASK_END;
        printf("\n");

        printf(" Link %d RX-Input Initial Tuning BER:", params.linkId);
        FOR_EACH_INDEX_IN_MASK(8, laneNum, params.laneMask)
        {
            printf(" %d,", params.grading.rx_init[laneNum]);
        }
        FOR_EACH_INDEX_IN_MASK_END;
        printf("\n\n");

        printf(" Link %d TX-Input Maintenance BER:", params.linkId);
        FOR_EACH_INDEX_IN_MASK(8, laneNum, params.laneMask)
        {
            printf(" %d,", params.grading.tx_maint[laneNum]);
        }
        FOR_EACH_INDEX_IN_MASK_END;
        printf("\n");

        printf(" Link %d RX-Input Maintenance BER:", params.linkId);
        FOR_EACH_INDEX_IN_MASK(8, laneNum, params.laneMask)
        {
            printf(" %d,", params.grading.rx_maint[laneNum]);
        }
        FOR_EACH_INDEX_IN_MASK_END;
        printf("\n\n");
    }
}

TEST_F(LWSwitchDeviceTest, IoctlGetFomValues)
{
    LWSWITCH_GET_FOM_VALUES_PARAMS params;
    lwswitch_device *device = getDevice();
    LwU8 i;
    LW_STATUS status;

    memset(&params, 0, sizeof(params));

    if (isArchSv10())
    {
        printf("[ SKIPPED ] This test is not supported on Willow\n");
        return;
    }

    params.linkId = 0;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_FOM_VALUES, &params, sizeof(params));

    ASSERT_EQ(status, LW_OK);

    if (verbose)
    {
        printf(" Link %d FOM values:", params.linkId);
        for (i = 0; i < params.numLanes; i++)
        {
            printf(" %d,", params.figureOfMeritValues[i]);
        }
        printf("\n");
    }
}

TEST_F(LWSwitchDeviceTest, IoctlGetLpCounters)
{
    LWSWITCH_GET_LWLINK_LP_COUNTERS_PARAMS params;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    memset(&params, 0, sizeof(params));

    params.linkId = 0;
    params.counterValidMask |= LWBIT(CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_COUNT_TX_LWHS);
    params.counterValidMask |= LWBIT(CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_COUNT_TX_OTHER);
    params.counterValidMask |= LWBIT(CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_NUM_TX_LP_ENTER);
    params.counterValidMask |= LWBIT(CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_NUM_TX_LP_EXIT);
    params.counterValidMask |= LWBIT(CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_COUNT_TX_SLEEP);

    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_LWLINK_LP_COUNTERS, &params, sizeof(params));
    if (status == LW_ERR_NOT_SUPPORTED)
    {
        printf("[ SKIPPED  ] This test is not supported on current arch\n");
        return;
    }

    ASSERT_EQ(status, LW_OK);

    if (verbose)
    {
        if (params.counterValidMask & LWBIT(CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_COUNT_TX_LWHS))
        {
            printf(" COUNT_TX_LWHS: %u\n", params.counterValues[CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_COUNT_TX_LWHS]);
        }

        if (params.counterValidMask & LWBIT(CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_COUNT_TX_OTHER))
        {
            printf(" COUNT_TX_OTHER: %u\n", params.counterValues[CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_COUNT_TX_OTHER]);
        }

        if (params.counterValidMask & LWBIT(CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_NUM_TX_LP_ENTER))
        {
            printf(" NUM_TX_LP_ENTER: %u\n", params.counterValues[CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_NUM_TX_LP_ENTER]);
        }

        if (params.counterValidMask & LWBIT(CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_NUM_TX_LP_EXIT))
        {
            printf(" NUM_TX_LP_EXIT: %u\n", params.counterValues[CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_NUM_TX_LP_EXIT]);
        }

        if (params.counterValidMask & LWBIT(CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_COUNT_TX_SLEEP))
        {
            printf(" COUNT_TX_SLEEP: %u\n", params.counterValues[CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_COUNT_TX_SLEEP]);
        }
    }
}

TEST_F(LWSwitchDeviceTest, IoctlSetResidencyBins)
{
    LWSWITCH_SET_RESIDENCY_BINS p;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (isArchSv10() || isArchLr10())
    {
        printf("[ SKIPPED ] This test is not supported on pre-Laguna\n");
        return;
    }

    p.table_select = LWSWITCH_TABLE_SELECT_MULTICAST;
    p.bin.lowThreshold = 10;
    p.bin.hiThreshold = 12;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_RESIDENCY_BINS, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);

    p.table_select = LWSWITCH_TABLE_SELECT_MULTICAST;
    p.bin.lowThreshold = 120;
    p.bin.hiThreshold = 10000;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_RESIDENCY_BINS, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);

    p.table_select = LWSWITCH_TABLE_SELECT_MULTICAST;
    p.bin.lowThreshold = 120;
    p.bin.hiThreshold = 1000;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_RESIDENCY_BINS, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);

    p.table_select = LWSWITCH_TABLE_SELECT_REDUCTION;
    p.bin.lowThreshold = 10;
    p.bin.hiThreshold = 12;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_RESIDENCY_BINS, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);

    p.table_select = LWSWITCH_TABLE_SELECT_REDUCTION;
    p.bin.lowThreshold = 120;
    p.bin.hiThreshold = 10000;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_RESIDENCY_BINS, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);

    p.table_select = LWSWITCH_TABLE_SELECT_REDUCTION;
    p.bin.lowThreshold = 120;
    p.bin.hiThreshold = 1000;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_RESIDENCY_BINS, &p, sizeof(p));
    ASSERT_EQ(status, LW_OK);
}

TEST_F(LWSwitchDeviceTest, IoctlSetResidencyBinsBadInput)
{
    LWSWITCH_SET_RESIDENCY_BINS p;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (isArchSv10() || isArchLr10())
    {
        printf("[ SKIPPED ] This test is not supported on pre-Laguna\n");
        return;
    }

    // Invalid table
    p.table_select = ~0;
    p.bin.lowThreshold = 120;
    p.bin.hiThreshold = 1000;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_RESIDENCY_BINS, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);

    // Low greater than High
    p.table_select = LWSWITCH_TABLE_SELECT_REDUCTION;
    p.bin.hiThreshold = 120;
    p.bin.lowThreshold = 1000;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_RESIDENCY_BINS, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Threshold out of range
    p.table_select = LWSWITCH_TABLE_SELECT_REDUCTION;
    p.bin.lowThreshold = 120;
    p.bin.hiThreshold = 0x01000000ll * 1333 / 1000;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_SET_RESIDENCY_BINS, &p, sizeof(p));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);
}

TEST_F(LWSwitchDeviceTest, IoctlGetResidencyBins)
{
    LwU64 linkMask = getLinkMask();
    LwU32 linkCount = getLinkCount();
    LWSWITCH_GET_RESIDENCY_BINS p, q;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 link;

    if (isArchSv10() || isArchLr10())
    {
        printf("[ SKIPPED ] This test is not supported on pre-Laguna\n");
        return;
    }

    for (link = 0; link < linkCount; link++)
    {
        if (LWBIT64(link) & linkMask)
        {
            memset(&p, 0xff, sizeof(p));
            p.link = link;
            p.table_select = LWSWITCH_TABLE_SELECT_MULTICAST;
            q = p;

            status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_RESIDENCY_BINS, &p, sizeof(p));
            ASSERT_EQ(status, LW_OK);
            ASSERT_NE(memcmp(&p, &q, sizeof(p)), 0);

            memset(&p, 0xff, sizeof(p));
            p.link = link;
            p.table_select = LWSWITCH_TABLE_SELECT_REDUCTION;
            q = p;

            status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_RESIDENCY_BINS, &p, sizeof(p));
            ASSERT_EQ(status, LW_OK);
            ASSERT_NE(memcmp(&p, &q, sizeof(p)), 0);
        }
    }
}

TEST_F(LWSwitchDeviceTest, IoctlGetResidencyBinsBadInput)
{
    LWSWITCH_GET_RESIDENCY_BINS p;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (isArchSv10() || isArchLr10())
    {
        printf("[ SKIPPED ] This test is not supported on pre-Laguna\n");
        return;
    }

    // Null parameters
    p.link = 0;
    p.table_select = LWSWITCH_TABLE_SELECT_MULTICAST;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_RESIDENCY_BINS, NULL, sizeof(IOCTL_LWSWITCH_GET_RESIDENCY_BINS));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);


    // Invalid port
    p.link = ~0;
    p.table_select = LWSWITCH_TABLE_SELECT_MULTICAST;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_RESIDENCY_BINS, &p, sizeof(IOCTL_LWSWITCH_GET_RESIDENCY_BINS));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Invalid table
    p.link = 0;
    p.table_select = ~0;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_RESIDENCY_BINS, &p, sizeof(IOCTL_LWSWITCH_GET_RESIDENCY_BINS));
    ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
}

TEST_F(LWSwitchDeviceTest, IoctlGetRBStallBusy)
{
    LwU64 linkMask = getLinkMask();
    LwU32 linkCount = getLinkCount();
    LWSWITCH_GET_RB_STALL_BUSY p, q;
    lwswitch_device *device = getDevice();
    LW_STATUS status;
    LwU32 link;

    if (isArchSv10() || isArchLr10())
    {
        printf("[ SKIPPED ] This test is not supported on pre-Laguna\n");
        return;
    }

    for (link = 0; link < linkCount; link++)
    {
        if (LWBIT64(link) & linkMask)
        {
            memset(&p, 0xff, sizeof(p));
            p.link = link;
            p.table_select = LWSWITCH_TABLE_SELECT_MULTICAST;
            q = p;

            status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_RB_STALL_BUSY, &p, sizeof(p));
            ASSERT_EQ(status, LW_OK);
            ASSERT_NE(memcmp(&p, &q, sizeof(p)), 0);

            memset(&p, 0xff, sizeof(p));
            p.link = link;
            p.table_select = LWSWITCH_TABLE_SELECT_REDUCTION;
            q = p;

            status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_RB_STALL_BUSY, &p, sizeof(p));
            ASSERT_EQ(status, LW_OK);
            ASSERT_NE(memcmp(&p, &q, sizeof(p)), 0);
        }
    }
}

TEST_F(LWSwitchDeviceTest, IoctlGetRBStallBusyBadInput)
{
    LWSWITCH_GET_RB_STALL_BUSY p;
    lwswitch_device *device = getDevice();
    LW_STATUS status;

    if (isArchSv10() || isArchLr10())
    {
        printf("[ SKIPPED ] This test is not supported on pre-Laguna\n");
        return;
    }

    // Null parameters
    p.link = 0;
    p.table_select = LWSWITCH_TABLE_SELECT_MULTICAST;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_RB_STALL_BUSY, NULL, sizeof(IOCTL_LWSWITCH_GET_RB_STALL_BUSY));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Invalid port
    p.link = ~0;
    p.table_select = LWSWITCH_TABLE_SELECT_MULTICAST;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_RB_STALL_BUSY, &p, sizeof(IOCTL_LWSWITCH_GET_RB_STALL_BUSY));
    ASSERT_EQ(status, LW_ERR_ILWALID_ARGUMENT);

    // Invalid table
    p.link = 0;
    p.table_select = ~0;
    status = lwswitch_api_control(device, IOCTL_LWSWITCH_GET_RB_STALL_BUSY, &p, sizeof(IOCTL_LWSWITCH_GET_RB_STALL_BUSY));
    ASSERT_EQ(status, LW_ERR_NOT_SUPPORTED);
}

