/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include "common_lwswitch.h"
#include "lr10/lr10.h"

#include "lwswitch/lr10/dev_lws_top.h"
#include "lwswitch/lr10/lwlinkip_discovery.h"
#include "lwswitch/lr10/npgip_discovery.h"
#include "lwswitch/lr10/nxbar_discovery.h"

#define VERBOSE_MMIO_DISCOVERY      0

#define MAKE_DISCOVERY_LR10(device, _chip, _engine) \
    {                                               \
        #_engine,                                   \
        NUM_##_engine##_ENGINE_##_chip,             \
        LW_SWPTOP_ENUM_DEVICE_##_engine,            \
        chip_device->eng##_engine                   \
    }

typedef struct
{
    const char *engname;
    LwU32 engcount_max;
    LwU32 discovery_id;
    ENGINE_DESCRIPTOR_TYPE_LR10    *engine;
}
DISCOVERY_TABLE_TYPE_LR10;

#define LWSWITCH_DISCOVERY_ENTRY_ILWALID    0x0
#define LWSWITCH_DISCOVERY_ENTRY_ENUM       0x1
#define LWSWITCH_DISCOVERY_ENTRY_DATA1      0x2
#define LWSWITCH_DISCOVERY_ENTRY_DATA2      0x3

typedef struct
{
    void (*parse_entry)(lwswitch_device *device, LwU32 entry, LwU32 *entry_type, LwBool *entry_chain);
    void (*parse_enum)(lwswitch_device *device, LwU32 entry, LwU32 *entry_device, LwU32 *entry_id, LwU32 *entry_version);
    void (*handle_data1)(lwswitch_device *device, LwU32 entry, ENGINE_DESCRIPTOR_TYPE_LR10 *engine, LwU32 entry_device, LwU32 *discovery_list_size);
    void (*handle_data2)(lwswitch_device *device, LwU32 entry, ENGINE_DESCRIPTOR_TYPE_LR10 *engine, LwU32 entry_device);
}
LWSWITCH_DISCOVERY_HANDLERS_LR10;

#define DISCOVERY_DUMP_ENGINE_LR10(_device,  _engine, _bcast)   \
    _discovery_dump_eng_lr10(_device, #_engine, LWSWITCH_GET_CHIP_DEVICE_LR10(_device)->eng##_engine##_bcast, NUM_##_engine##_bcast##_ENGINE_LR10);

static void
_discovery_dump_eng_lr10
(
    lwswitch_device *device,
    const char *eng_name,
    ENGINE_DESCRIPTOR_TYPE_LR10 *engine,
    LwU32 count
)
{
    LwU32 i;

    if (VERBOSE_MMIO_DISCOVERY)
    {
        for (i = 0; i < count; i++)
        {
            if (engine[i].valid)
            {
                if (engine[i].disc_type == DISCOVERY_TYPE_DISCOVERY)
                {
                    LWSWITCH_PRINT(device, SETUP,
                        "%-24s[%2d]:V:%1x %s:%6x                      CL/ID:%2x/%2x\n",
                        eng_name, i,
                        engine[i].version,
                        "DI",
                        engine[i].info.top.discovery,
                        engine[i].info.top.cluster, engine[i].info.top.cluster_id);
                }
                else if (engine[i].disc_type == DISCOVERY_TYPE_UNICAST)
                {
                    LWSWITCH_PRINT(device, SETUP,
                        "%-24s[%2d]:V:%1x %s:%6x\n",
                        eng_name, i,
                        engine[i].version,
                        "UC",
                        engine[i].info.uc.uc_addr);
                }
                else if (engine[i].disc_type == DISCOVERY_TYPE_BROADCAST)
                {
                    LWSWITCH_PRINT(device, SETUP,
                        "%-24s[%2d]:V:%1x %s:%6x %s:%6x/%6x/%6x\n",
                        eng_name, i,
                        engine[i].version,
                        "BC",
                        engine[i].info.bc.bc_addr,
                        "MC",
                        engine[i].info.bc.mc_addr[0],
                        engine[i].info.bc.mc_addr[1],
                        engine[i].info.bc.mc_addr[2]);
                }
                else
                {
                    LWSWITCH_PRINT(device, SETUP,
                        "%-24s[%2d]:V:%1x UNDEFINED\n",
                        eng_name, i,
                        engine[i].version);
                }
            }
            else
            {
                LWSWITCH_PRINT(device, SETUP,
                    "%-24s[%2d]: INVALID\n",
                    eng_name, i);
            }
        }
    }
}

static LwlStatus
_lwswitch_device_discovery_lr10
(
    lwswitch_device *device,
    LwU32   discovery_offset,
    DISCOVERY_TABLE_TYPE_LR10 *discovery_table,
    LwU32 discovery_table_size,
    LWSWITCH_DISCOVERY_HANDLERS_LR10 *discovery_handlers
)
{
    ENGINE_DESCRIPTOR_TYPE_LR10 *engine = NULL;
    LwU32                   entry_type = LWSWITCH_DISCOVERY_ENTRY_ILWALID;
    LwBool                  entry_chain = LW_FALSE;
    LwU32                   entry = 0;
    LwU32                   entry_device = 0;
    LwU32                   entry_id = 0;
    LwU32                   entry_version = 0;
    LwU32                   entry_count = 0;
    LwBool                  done = LW_FALSE;
    LwlStatus               retval = LWL_SUCCESS;

    //
    // Must be at least two entries.  We'll fix it up later when we find the length in the table
    //
    LwU32                   discovery_list_size = 2;

    if (VERBOSE_MMIO_DISCOVERY)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: LwSwitch Engine discovery table @%x\n",
            __FUNCTION__,
            discovery_offset);
    }

    while ((!done) && (entry_count < discovery_list_size))
    {
        entry = LWSWITCH_OFF_RD32(device, discovery_offset);
        discovery_handlers->parse_entry(device, entry, &entry_type, &entry_chain);

        switch (entry_type)
        {
            case LWSWITCH_DISCOVERY_ENTRY_ENUM:
                LWSWITCH_ASSERT(engine == NULL);
                discovery_handlers->parse_enum(device, entry, &entry_device, &entry_id, &entry_version);

                {
                    LwU32 i;

                    for(i = 0; i < discovery_table_size; i++)
                    {
                        if (entry_device == discovery_table[i].discovery_id)
                        {
                            if (discovery_table[i].engine == NULL)
                            {
                                LWSWITCH_PRINT(device, ERROR,
                                    "%s:_ENUM: ERROR: %s:device=%x id=%x version=%x not supported!\n",
                                    __FUNCTION__,
                                    discovery_table[i].engname,
                                    entry_device, entry_id, entry_version);
                                LWSWITCH_ASSERT(0);
                                continue;
                            }

                            if (entry_id < discovery_table[i].engcount_max)
                            {
                                engine = &(discovery_table[i].engine[entry_id]);
                                break;
                            }
                            else
                            {
                                LWSWITCH_PRINT(device, ERROR,
                                    "%s:_ENUM: ERROR: %s[%d] out of engine range %d..%d\n",
                                    __FUNCTION__,
                                    discovery_table[i].engname,
                                    entry_id,
                                    0, discovery_table[i].engcount_max-1);
                            }
                        }
                    }

                    if (engine == NULL)
                    {
                        LWSWITCH_PRINT(device, ERROR,
                            "%s:_ENUM: ERROR: device=%x id=%x version=%x not recognized!\n",
                            __FUNCTION__,
                            entry_device, entry_id, entry_version);
                    }
                }

                if (engine != NULL)
                {
                    if ((engine->valid == LW_TRUE) && 
                        (engine->disc_type != DISCOVERY_TYPE_DISCOVERY))
                    {
                        LWSWITCH_PRINT(device, WARN,
                            "%s:_ENUM: WARNING: device=%x id=%x previously discovered!\n",
                            __FUNCTION__,
                            entry_device, entry_id);
                    }
                    engine->valid = LW_TRUE;
                    engine->version  = entry_version;
                }

                break;

            case LWSWITCH_DISCOVERY_ENTRY_DATA1:
                discovery_handlers->handle_data1(device, entry, engine, entry_device, &discovery_list_size);
                break;

            case LWSWITCH_DISCOVERY_ENTRY_DATA2:
                if (engine == NULL)
                {
                    LWSWITCH_PRINT(device, ERROR,
                        "%s:DATA2:engine == NULL.  Skipping processing\n",
                        __FUNCTION__);
                }
                else
                {
                    discovery_handlers->handle_data2(device, entry, engine, entry_device);
                }
                break;

            default:
                LWSWITCH_PRINT(device, ERROR,
                    "%s:Unknown (%d)\n",
                    __FUNCTION__, entry_type);
                LWSWITCH_ASSERT(0);
                // Deliberate fallthrough
            case LWSWITCH_DISCOVERY_ENTRY_ILWALID:
                // Invalid entry.  Just ignore it
                LWSWITCH_PRINT(device, SETUP,
                    "%s:_ILWALID -- skip 0x%08x\n",
                    __FUNCTION__, entry);
                break;
        }

        if (!entry_chain)
        {
            // End of chain.  Close the active engine
            engine = NULL;
            entry_device  = 0;      // Mark invalid
            entry_id      = ~0;
            entry_version = ~0;
        }

        discovery_offset += sizeof(LwU32);
        entry_count++;
    }

    if (entry_chain)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s:Discovery list incorrectly terminated: chain end(%d)\n",
            __FUNCTION__,
            entry_chain);
        LWSWITCH_ASSERT(!entry_chain);
    }

    return retval;
}

static void
_lwswitch_ptop_parse_entry_lr10
(
    lwswitch_device *device,
    LwU32   entry,
    LwU32   *entry_type,
    LwBool  *entry_chain
)
{
    LwU32 entry_type_lr10;

    entry_type_lr10 = DRF_VAL(_SWPTOP, _, ENTRY, entry);
    *entry_chain = FLD_TEST_DRF(_SWPTOP, _, CHAIN, _ENABLE, entry);

    switch (entry_type_lr10)
    {
        case LW_SWPTOP_ENTRY_ENUM:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_ENUM;
            break;
        case LW_SWPTOP_ENTRY_DATA1:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_DATA1;
            break;
        case LW_SWPTOP_ENTRY_DATA2:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_DATA2;
            break;
        default:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_ILWALID;
            break;
    }
}

static void
_lwswitch_ptop_parse_enum_lr10
(
    lwswitch_device *device,
    LwU32   entry,
    LwU32   *entry_device,
    LwU32   *entry_id,
    LwU32   *entry_version
)
{
    *entry_device  = DRF_VAL(_SWPTOP, _, ENUM_DEVICE, entry);
    *entry_id      = DRF_VAL(_SWPTOP, _, ENUM_ID, entry);
    *entry_version = DRF_VAL(_SWPTOP, _, ENUM_VERSION, entry);
}

static void
_lwswitch_ptop_handle_data1_lr10
(
    lwswitch_device *device,
    LwU32 entry,
    ENGINE_DESCRIPTOR_TYPE_LR10 *engine,
    LwU32 entry_device,
    LwU32 *discovery_list_size
)
{
    if (LW_SWPTOP_ENUM_DEVICE_PTOP == entry_device)
    {
        *discovery_list_size = DRF_VAL(_SWPTOP, _DATA1, _PTOP_LENGTH, entry);
        return;
    }
    else
    {
        LWSWITCH_ASSERT(DRF_VAL(_SWPTOP, _DATA1, _RESERVED, entry) == 0);
    }

    if (engine == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s:DATA1:engine == NULL.  Skipping processing\n",
            __FUNCTION__);
        return;
    }

    engine->info.top.cluster    = DRF_VAL(_SWPTOP, _DATA1, _CLUSTER_TYPE, entry);
    engine->info.top.cluster_id = DRF_VAL(_SWPTOP, _DATA1, _CLUSTER_NUMBER, entry);
    engine->disc_type = DISCOVERY_TYPE_DISCOVERY;
}

static void
_lwswitch_ptop_handle_data2_lr10
(
    lwswitch_device *device,
    LwU32 entry,
    ENGINE_DESCRIPTOR_TYPE_LR10 *engine,
    LwU32 entry_device
)
{
    LwU32 data2_type = DRF_VAL(_SWPTOP, _DATA2, _TYPE, entry);
    LwU32 data2_addr = DRF_VAL(_SWPTOP, _DATA2, _ADDR, entry);

    switch(data2_type)
    {
        case LW_SWPTOP_DATA2_TYPE_DISCOVERY:
            // Parse sub-discovery table
            engine->disc_type = DISCOVERY_TYPE_DISCOVERY;
            engine->info.top.discovery = data2_addr*sizeof(LwU32);
            break;
        case LW_SWPTOP_DATA2_TYPE_UNICAST:
            engine->disc_type = DISCOVERY_TYPE_UNICAST;
            engine->info.uc.uc_addr = data2_addr*sizeof(LwU32);
            break;
        case LW_SWPTOP_DATA2_TYPE_BROADCAST:
            engine->disc_type = DISCOVERY_TYPE_BROADCAST;
            engine->info.bc.bc_addr = data2_addr*sizeof(LwU32);
            break;
        case LW_SWPTOP_DATA2_TYPE_MULTICAST0:
        case LW_SWPTOP_DATA2_TYPE_MULTICAST1:
        case LW_SWPTOP_DATA2_TYPE_MULTICAST2:
            {
                LwU32 mc_idx = data2_type - LW_SWPTOP_DATA2_TYPE_MULTICAST0;
                engine->disc_type = DISCOVERY_TYPE_BROADCAST;
                engine->info.bc.mc_addr[mc_idx] = data2_addr*sizeof(LwU32);
            }
            break;
        case LW_SWPTOP_DATA2_TYPE_ILWALID:
            LWSWITCH_PRINT(device, SETUP,
                "%s:_DATA2: %s=%6x\n",
                __FUNCTION__,
                "_ILWALID", data2_addr);
            engine->disc_type = DISCOVERY_TYPE_UNDEFINED;
            break;
        default:
            LWSWITCH_PRINT(device, SETUP,
                "%s:_DATA2: Unknown type 0x%x (0x%08x)!\n",
                __FUNCTION__, data2_type, entry);
            engine->disc_type = DISCOVERY_TYPE_UNDEFINED;
            LWSWITCH_ASSERT(0);
            break;
    }
}

void
lwswitch_lwlw_parse_entry_lr10
(
    lwswitch_device *device,
    LwU32   entry,
    LwU32   *entry_type,
    LwBool  *entry_chain
)
{
    LwU32 entry_type_lwlw;

    entry_type_lwlw = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _ENTRY, entry);
    *entry_chain = FLD_TEST_DRF(_LWLINKIP, _DISCOVERY_COMMON, _CHAIN, _ENABLE, entry);

    switch (entry_type_lwlw)
    {
        case LW_LWLINKIP_DISCOVERY_COMMON_ENTRY_ENUM:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_ENUM;
            break;
        case LW_LWLINKIP_DISCOVERY_COMMON_ENTRY_DATA1:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_DATA1;
            break;
        case LW_LWLINKIP_DISCOVERY_COMMON_ENTRY_DATA2:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_DATA2;
            break;
        default:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_ILWALID;
            break;
    }
}

void
lwswitch_lwlw_parse_enum_lr10
(
    lwswitch_device *device,
    LwU32   entry,
    LwU32   *entry_device,
    LwU32   *entry_id,
    LwU32   *entry_version
)
{
    LwU32 entry_reserved;

    *entry_device  = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DEVICE, entry);
    *entry_id      = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _ID, entry);
    *entry_version = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _VERSION, entry);

    entry_reserved = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _RESERVED, entry);
    LWSWITCH_ASSERT(entry_reserved == 0);

    if (*entry_version != LW_LWLINKIP_DISCOVERY_COMMON_VERSION_LWLINK30)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s:_LWLINKIP, _DISCOVERY_COMMON, _VERSION = %x but expected %x (_LWLINK30).\n",
            __FUNCTION__, *entry_version, LW_LWLINKIP_DISCOVERY_COMMON_VERSION_LWLINK30);
    }
}

void
lwswitch_lwlw_handle_data1_lr10
(
    lwswitch_device *device,
    LwU32 entry,
    ENGINE_DESCRIPTOR_TYPE_LR10 *engine,
    LwU32 entry_device,
    LwU32 *discovery_list_size
)
{
    if ((LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_IOCTRL == entry_device) ||
        (LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_SIOCTRL == entry_device) ||
        (LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_TIOCTRL == entry_device) ||
        (LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_LWLW == entry_device))
    {
        *discovery_list_size = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DATA1_IOCTRL_LENGTH, entry);
    }

    if (engine == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s:DATA1:engine == NULL.  Skipping processing\n",
            __FUNCTION__);
        return;
    }

    if ((LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_IOCTRL != entry_device) &&
        (LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_SIOCTRL != entry_device) &&
        (LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_TIOCTRL != entry_device) &&
        (LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_LWLW != entry_device))
    {
        // Nothing specific needed to handle
        if (0 != DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DATA1_RESERVED, entry))
        {
            LWSWITCH_PRINT(device, WARN,
                "%s:WARNING:IOCTRL _RESERVED field != 0 (entry %x -> %x)\n",
                __FUNCTION__,
                entry, DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DATA1_RESERVED, entry));
        }
    }

    if (0 != DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DATA1_RESERVED2, entry))
    {
        LWSWITCH_PRINT(device, WARN,
            "%s:WARNING:IOCTRL _RESERVED2 field != 0 (entry %x -> %x)\n",
            __FUNCTION__,
            entry, DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DATA1_RESERVED2, entry));
    }
}

void
lwswitch_lwlw_handle_data2_lr10
(
    lwswitch_device *device,
    LwU32 entry,
    ENGINE_DESCRIPTOR_TYPE_LR10 *engine,
    LwU32 entry_device
)
{
    LwU32 data2_type = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON_DATA2, _TYPE, entry);
    LwU32 data2_addr = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON_DATA2, _ADDR, entry);

    switch(data2_type)
    {

        case LW_LWLINKIP_DISCOVERY_COMMON_DATA2_TYPE_DISCOVERY:
            // Parse sub-discovery table

            //
            // Lwrrently _DISCOVERY is not used in the second
            // level discovery.
            //
            LWSWITCH_ASSERT(0);

            break;

        case LW_LWLINKIP_DISCOVERY_COMMON_DATA2_TYPE_UNICAST:
            engine->disc_type = DISCOVERY_TYPE_UNICAST;
            engine->info.uc.uc_addr = data2_addr*sizeof(LwU32);
            break;

        case LW_LWLINKIP_DISCOVERY_COMMON_DATA2_TYPE_BROADCAST:
            engine->disc_type = DISCOVERY_TYPE_BROADCAST;
            engine->info.bc.bc_addr = data2_addr*sizeof(LwU32);
            break;

        case LW_LWLINKIP_DISCOVERY_COMMON_DATA2_TYPE_MULTICAST0:
        case LW_LWLINKIP_DISCOVERY_COMMON_DATA2_TYPE_MULTICAST1:
        case LW_LWLINKIP_DISCOVERY_COMMON_DATA2_TYPE_MULTICAST2:
            {
                LwU32 mc_idx = data2_type - LW_LWLINKIP_DISCOVERY_COMMON_DATA2_TYPE_MULTICAST0;
                engine->disc_type = DISCOVERY_TYPE_BROADCAST;
                engine->info.bc.mc_addr[mc_idx] = data2_addr*sizeof(LwU32);
            }
            break;

        case LW_LWLINKIP_DISCOVERY_COMMON_DATA2_TYPE_ILWALID:
            LWSWITCH_PRINT(device, ERROR,
                "%s:_DATA2: %s=%6x\n",
                __FUNCTION__,
                "_ILWALID", data2_addr);
            engine->disc_type = DISCOVERY_TYPE_UNDEFINED;
            break;

        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s:_DATA2: Unknown!\n",
                __FUNCTION__);
            LWSWITCH_ASSERT(0);
            break;
    }
}

static void
_lwswitch_npg_parse_entry_lr10
(
    lwswitch_device *device,
    LwU32   entry,
    LwU32   *entry_type,
    LwBool  *entry_chain
)
{
    LwU32 entry_type_npg;

    entry_type_npg = DRF_VAL(_NPG, _DISCOVERY, _ENTRY, entry);
    *entry_chain = FLD_TEST_DRF(_NPG, _DISCOVERY, _CHAIN, _ENABLE, entry);

    switch (entry_type_npg)
    {
        case LW_NPG_DISCOVERY_ENTRY_ENUM:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_ENUM;
            break;
        case LW_NPG_DISCOVERY_ENTRY_DATA1:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_DATA1;
            break;
        case LW_NPG_DISCOVERY_ENTRY_DATA2:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_DATA2;
            break;
        default:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_ILWALID;
            break;
    }
}

static void
_lwswitch_npg_parse_enum_lr10
(
    lwswitch_device *device,
    LwU32   entry,
    LwU32   *entry_device,
    LwU32   *entry_id,
    LwU32   *entry_version
)
{
    *entry_device  = DRF_VAL(_NPG, _DISCOVERY, _ENUM_DEVICE, entry);
    *entry_id      = DRF_VAL(_NPG, _DISCOVERY, _ENUM_ID, entry);
    *entry_version = DRF_VAL(_NPG, _DISCOVERY, _ENUM_VERSION, entry);
    LWSWITCH_ASSERT(DRF_VAL(_NPG, _DISCOVERY, _ENUM_RESERVED, entry) == 0);

    if (*entry_version != LW_NPG_DISCOVERY_ENUM_VERSION_2)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s:_NPG_DISCOVERY_ENUM_VERSION = %x but expected %x (_VERSION_2).\n",
            __FUNCTION__, *entry_version, LW_NPG_DISCOVERY_ENUM_VERSION_2);
    }
}

static void
_lwswitch_npg_handle_data1_lr10
(
    lwswitch_device *device,
    LwU32 entry,
    ENGINE_DESCRIPTOR_TYPE_LR10 *engine,
    LwU32 entry_device,
    LwU32 *discovery_list_size
)
{
    if (LW_NPG_DISCOVERY_ENUM_DEVICE_NPG == entry_device)
    {
        *discovery_list_size = DRF_VAL(_NPG, _DISCOVERY, _DATA1_NPG_LENGTH, entry);
    }
    else
    {
        LWSWITCH_ASSERT(0 == DRF_VAL(_NPG, _DISCOVERY, _DATA1_RESERVED, entry));
    }

    if (engine == NULL)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s:DATA1:engine == NULL.  Skipping processing\n",
            __FUNCTION__);
        return;
    }
}

static void
_lwswitch_npg_handle_data2_lr10
(
    lwswitch_device *device,
    LwU32 entry,
    ENGINE_DESCRIPTOR_TYPE_LR10 *engine,
    LwU32 entry_device
)
{
    LwU32 data2_type = DRF_VAL(_NPG, _DISCOVERY_DATA2, _TYPE, entry);
    LwU32 data2_addr = DRF_VAL(_NPG, _DISCOVERY_DATA2, _ADDR, entry);

    switch(data2_type)
    {
        case LW_NPG_DISCOVERY_DATA2_TYPE_DISCOVERY:
            // Parse sub-discovery table

            //
            // Lwrrently _DISCOVERY is not used in the second
            // level discovery.
            //
            LWSWITCH_ASSERT(0);

            break;

        case LW_NPG_DISCOVERY_DATA2_TYPE_UNICAST:
            engine->disc_type = DISCOVERY_TYPE_UNICAST;
            engine->info.uc.uc_addr = data2_addr*sizeof(LwU32);
            break;

        case LW_NPG_DISCOVERY_DATA2_TYPE_BROADCAST:
            engine->disc_type = DISCOVERY_TYPE_BROADCAST;
            engine->info.bc.bc_addr = data2_addr*sizeof(LwU32);
            break;

        case LW_NPG_DISCOVERY_DATA2_TYPE_MULTICAST0:
        case LW_NPG_DISCOVERY_DATA2_TYPE_MULTICAST1:
        case LW_NPG_DISCOVERY_DATA2_TYPE_MULTICAST2:
            {
                LwU32 mc_idx = data2_type - LW_NPG_DISCOVERY_DATA2_TYPE_MULTICAST0;
                engine->disc_type = DISCOVERY_TYPE_BROADCAST;
                engine->info.bc.mc_addr[mc_idx] = data2_addr*sizeof(LwU32);
            }
            break;

        case LW_NPG_DISCOVERY_DATA2_TYPE_ILWALID:
            LWSWITCH_PRINT(device, SETUP,
                "%s:_DATA2: %s=%6x\n",
                __FUNCTION__,
                "_ILWALID", data2_addr);
            engine->disc_type = DISCOVERY_TYPE_UNDEFINED;
            break;

        default:
            LWSWITCH_PRINT(device, SETUP,
                "%s:_DATA2: Unknown!\n",
                __FUNCTION__);
            LWSWITCH_ASSERT(0);
            break;
    }
}

void
lwswitch_nxbar_parse_entry_lr10
(
    lwswitch_device *device,
    LwU32   entry,
    LwU32   *entry_type,
    LwBool  *entry_chain
)
{
    LwU32 entry_type_nxbar;

    entry_type_nxbar = DRF_VAL(_NXBAR, _DISCOVERY, _ENTRY, entry);
    *entry_chain = FLD_TEST_DRF(_NXBAR, _DISCOVERY, _CHAIN, _ENABLE, entry);

    switch (entry_type_nxbar)
    {
        case LW_NXBAR_DISCOVERY_ENTRY_ENUM:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_ENUM;
            break;
        case LW_NXBAR_DISCOVERY_ENTRY_DATA1:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_DATA1;
            break;
        case LW_NXBAR_DISCOVERY_ENTRY_DATA2:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_DATA2;
            break;
        default:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_ILWALID;
            break;
    }
}

void
lwswitch_nxbar_parse_enum_lr10
(
    lwswitch_device *device,
    LwU32   entry,
    LwU32   *entry_device,
    LwU32   *entry_id,
    LwU32   *entry_version
)
{
    LwU32 entry_reserved;

    *entry_device  = DRF_VAL(_NXBAR, _DISCOVERY, _ENUM_DEVICE, entry);
    *entry_id      = DRF_VAL(_NXBAR, _DISCOVERY, _ENUM_ID, entry);
    *entry_version = DRF_VAL(_NXBAR, _DISCOVERY, _ENUM_VERSION, entry);

    entry_reserved = DRF_VAL(_NXBAR, _DISCOVERY, _ENUM_RESERVED, entry);
    LWSWITCH_ASSERT(entry_reserved == 0);

    if (*entry_version != LW_NXBAR_DISCOVERY_ENUM_VERSION_2)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s:_NXBAR_DISCOVERY_ENUM_VERSION = %x but expected %x (_VERSION_2).\n",
            __FUNCTION__, *entry_version, LW_NXBAR_DISCOVERY_ENUM_VERSION_2);
    }
}

void
lwswitch_nxbar_handle_data1_lr10
(
    lwswitch_device *device,
    LwU32 entry,
    ENGINE_DESCRIPTOR_TYPE_LR10 *engine,
    LwU32 entry_device,
    LwU32 *discovery_list_size
)
{
    if (LW_NXBAR_DISCOVERY_ENUM_DEVICE_NXBAR == entry_device)
    {
        *discovery_list_size = DRF_VAL(_NXBAR, _DISCOVERY, _DATA1_NXBAR_LENGTH, entry);
    }

    if (engine == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s:DATA1:engine == NULL.  Skipping processing\n",
            __FUNCTION__);
        return;
    }

    if (LW_NXBAR_DISCOVERY_ENUM_DEVICE_NXBAR != entry_device)
    {
        LWSWITCH_ASSERT(DRF_VAL(_NXBAR, _DISCOVERY, _DATA1_RESERVED, entry) == 0);
    }
}

void
lwswitch_nxbar_handle_data2_lr10
(
    lwswitch_device *device,
    LwU32 entry,
    ENGINE_DESCRIPTOR_TYPE_LR10 *engine,
    LwU32 entry_device
)
{
    LwU32 data2_type = DRF_VAL(_NXBAR, _DISCOVERY_DATA2, _TYPE, entry);
    LwU32 data2_addr = DRF_VAL(_NXBAR, _DISCOVERY_DATA2, _ADDR, entry);

    switch(data2_type)
    {
        case LW_NXBAR_DISCOVERY_DATA2_TYPE_DISCOVERY:
            // Parse sub-discovery table

            //
            // Lwrrently _DISCOVERY is not used in the second
            // level discovery.
            //
            LWSWITCH_ASSERT(0);

            break;

        case LW_NXBAR_DISCOVERY_DATA2_TYPE_UNICAST:
            engine->disc_type = DISCOVERY_TYPE_UNICAST;
            engine->info.uc.uc_addr = data2_addr*sizeof(LwU32);
            break;

        case LW_NXBAR_DISCOVERY_DATA2_TYPE_BROADCAST:
            engine->disc_type = DISCOVERY_TYPE_BROADCAST;
            engine->info.bc.bc_addr = data2_addr*sizeof(LwU32);
            engine->info.bc.mc_addr[0] = 0;
            engine->info.bc.mc_addr[1] = 0;
            engine->info.bc.mc_addr[2] = 0;
            break;

        case LW_NXBAR_DISCOVERY_DATA2_TYPE_MULTICAST0:
        case LW_NXBAR_DISCOVERY_DATA2_TYPE_MULTICAST1:
        case LW_NXBAR_DISCOVERY_DATA2_TYPE_MULTICAST2:
            {
                LwU32 mc_idx = data2_type - LW_NXBAR_DISCOVERY_DATA2_TYPE_MULTICAST0;
                engine->info.bc.mc_addr[mc_idx] = data2_addr*sizeof(LwU32);
                LWSWITCH_PRINT(device, ERROR,
                    "%s:_DATA2: NXBAR MULTICAST%d=0x%x but should be unused!\n",
                    __FUNCTION__, mc_idx, engine->info.bc.mc_addr[mc_idx]);
                LWSWITCH_ASSERT(0);
            }
            break;

        case LW_NXBAR_DISCOVERY_DATA2_TYPE_ILWALID:
            LWSWITCH_PRINT(device, ERROR,
                "%s:_DATA2: %s=%6x\n",
                __FUNCTION__,
                "_ILWALID", data2_addr);
            engine->disc_type = DISCOVERY_TYPE_UNDEFINED;
            break;

        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s:_DATA2: Unknown!\n",
                __FUNCTION__);
            LWSWITCH_ASSERT(0);
            break;
    }
}

#define MAKE_DISCOVERY_LWLINK_LR10(_eng, _bcast)    \
    {                                               \
        #_eng#_bcast,                               \
        NUM_##_eng##_bcast##_ENGINE_LR10,           \
        LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_##_eng, \
        chip_device->eng##_eng##_bcast              \
    }

#define MAKE_DISCOVERY_NPG_LR10(_eng, _bcast)       \
    {                                               \
        #_eng#_bcast,                               \
        NUM_##_eng##_bcast##_ENGINE_LR10,           \
        LW_NPG_DISCOVERY_ENUM_DEVICE_##_eng,        \
        chip_device->eng##_eng##_bcast              \
    }

#define MAKE_DISCOVERY_NXBAR_LR10(_eng, _bcast)     \
    {                                               \
        #_eng#_bcast,                               \
        NUM_##_eng##_bcast##_ENGINE_LR10,           \
        LW_NXBAR_DISCOVERY_ENUM_DEVICE_##_eng,      \
        chip_device->eng##_eng##_bcast              \
    }

static
LWSWITCH_DISCOVERY_HANDLERS_LR10 discovery_handlers_ptop_lr10 =
{
    &_lwswitch_ptop_parse_entry_lr10,
    &_lwswitch_ptop_parse_enum_lr10,
    &_lwswitch_ptop_handle_data1_lr10,
    &_lwswitch_ptop_handle_data2_lr10
};

static
LWSWITCH_DISCOVERY_HANDLERS_LR10 discovery_handlers_npg_lr10 =
{
    &_lwswitch_npg_parse_entry_lr10,
    &_lwswitch_npg_parse_enum_lr10,
    &_lwswitch_npg_handle_data1_lr10,
    &_lwswitch_npg_handle_data2_lr10
};

static
LWSWITCH_DISCOVERY_HANDLERS_LR10 discovery_handlers_lwlw_lr10 =
{
    &lwswitch_lwlw_parse_entry_lr10,
    &lwswitch_lwlw_parse_enum_lr10,
    &lwswitch_lwlw_handle_data1_lr10,
    &lwswitch_lwlw_handle_data2_lr10
};

static
LWSWITCH_DISCOVERY_HANDLERS_LR10 discovery_handlers_nxbar_lr10 =
{
    &lwswitch_nxbar_parse_entry_lr10,
    &lwswitch_nxbar_parse_enum_lr10,
    &lwswitch_nxbar_handle_data1_lr10,
    &lwswitch_nxbar_handle_data2_lr10
};

//
// Parse top level PTOP engine discovery information to identify MMIO, interrupt, and
// reset information
//

LwlStatus
lwswitch_device_discovery_lr10
(
    lwswitch_device *device,
    LwU32   discovery_offset
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);

    DISCOVERY_TABLE_TYPE_LR10 discovery_table_lr10[] =
    {
        MAKE_DISCOVERY_LR10(device, LR10, PTOP),
        MAKE_DISCOVERY_LR10(device, LR10, NPG),
        MAKE_DISCOVERY_LR10(device, LR10, NPG_BCAST),
        MAKE_DISCOVERY_LR10(device, LR10, CLKS),
        MAKE_DISCOVERY_LR10(device, LR10, FUSE),
        MAKE_DISCOVERY_LR10(device, LR10, JTAG),
        MAKE_DISCOVERY_LR10(device, LR10, PMGR),
        MAKE_DISCOVERY_LR10(device, LR10, SAW),
        MAKE_DISCOVERY_LR10(device, LR10, XP3G),
        MAKE_DISCOVERY_LR10(device, LR10, XVE),
        MAKE_DISCOVERY_LR10(device, LR10, ROM),
        MAKE_DISCOVERY_LR10(device, LR10, EXTDEV),
        MAKE_DISCOVERY_LR10(device, LR10, PRIVMAIN),
        MAKE_DISCOVERY_LR10(device, LR10, PRIVLOC),
        MAKE_DISCOVERY_LR10(device, LR10, PTIMER),
        MAKE_DISCOVERY_LR10(device, LR10, SOE),
        MAKE_DISCOVERY_LR10(device, LR10, SMR),
        MAKE_DISCOVERY_LR10(device, LR10, I2C),
        MAKE_DISCOVERY_LR10(device, LR10, SE),
        MAKE_DISCOVERY_LR10(device, LR10, LWLW),
        MAKE_DISCOVERY_LR10(device, LR10, LWLW_BCAST),
        MAKE_DISCOVERY_LR10(device, LR10, NXBAR),
        MAKE_DISCOVERY_LR10(device, LR10, NXBAR_BCAST),
        MAKE_DISCOVERY_LR10(device, LR10, THERM)
    };
    LwU32 discovery_table_lr10_size = LW_ARRAY_ELEMENTS(discovery_table_lr10);
    LwU32 i;
    LwlStatus   status;

    status = _lwswitch_device_discovery_lr10(
        device, discovery_offset, discovery_table_lr10, discovery_table_lr10_size, 
        &discovery_handlers_ptop_lr10);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "MMIO discovery failed\n");
        return status;
    }

    if (VERBOSE_MMIO_DISCOVERY)
    {
        LWSWITCH_PRINT(device, SETUP,
            "PTOP Discovery\n");

        DISCOVERY_DUMP_ENGINE_LR10(device, PTOP, );
        DISCOVERY_DUMP_ENGINE_LR10(device, NPG, );
        DISCOVERY_DUMP_ENGINE_LR10(device, NPG, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, CLKS, );
        DISCOVERY_DUMP_ENGINE_LR10(device, FUSE, );
        DISCOVERY_DUMP_ENGINE_LR10(device, JTAG, );
        DISCOVERY_DUMP_ENGINE_LR10(device, PMGR, );
        DISCOVERY_DUMP_ENGINE_LR10(device, SAW, );
        DISCOVERY_DUMP_ENGINE_LR10(device, XP3G, );
        DISCOVERY_DUMP_ENGINE_LR10(device, XVE, );
        DISCOVERY_DUMP_ENGINE_LR10(device, ROM, );
        DISCOVERY_DUMP_ENGINE_LR10(device, EXTDEV, );
        DISCOVERY_DUMP_ENGINE_LR10(device, PRIVMAIN, );
        DISCOVERY_DUMP_ENGINE_LR10(device, PRIVLOC, );
        DISCOVERY_DUMP_ENGINE_LR10(device, PTIMER, );
        DISCOVERY_DUMP_ENGINE_LR10(device, SOE, );
        DISCOVERY_DUMP_ENGINE_LR10(device, SMR, );
        DISCOVERY_DUMP_ENGINE_LR10(device, I2C, );
        DISCOVERY_DUMP_ENGINE_LR10(device, SE, );
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLW, );
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLW, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, NXBAR, );
        DISCOVERY_DUMP_ENGINE_LR10(device, NXBAR, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, THERM, );
    }

    for (i = 0; i < NUM_LWLW_ENGINE_LR10; i++)
    {
        if (LWSWITCH_ENG_VALID_LR10(device, LWLW, i) && 
            (chip_device->engLWLW[i].info.top.discovery != 0))
        {
            DISCOVERY_TABLE_TYPE_LR10 discovery_table_lwlw[] =
            {
                MAKE_DISCOVERY_LWLINK_LR10(MINION, ),
                MAKE_DISCOVERY_LWLINK_LR10(LWLIPT, ),
                MAKE_DISCOVERY_LWLINK_LR10(TX_PERFMON, ),
                MAKE_DISCOVERY_LWLINK_LR10(RX_PERFMON, ),
                MAKE_DISCOVERY_LWLINK_LR10(TX_PERFMON_MULTICAST, ),
                MAKE_DISCOVERY_LWLINK_LR10(RX_PERFMON_MULTICAST, ),
                MAKE_DISCOVERY_LWLINK_LR10(LWLTLC, ),
                MAKE_DISCOVERY_LWLINK_LR10(LWLTLC_MULTICAST, ),
                MAKE_DISCOVERY_LWLINK_LR10(LWLIPT_SYS_PERFMON, ),
                MAKE_DISCOVERY_LWLINK_LR10(LWLW, ),
                MAKE_DISCOVERY_LWLINK_LR10(PLL, ),
                MAKE_DISCOVERY_LWLINK_LR10(LWLW_PERFMON, ),
                MAKE_DISCOVERY_LWLINK_LR10(LWLDL_MULTICAST, ),
                MAKE_DISCOVERY_LWLINK_LR10(LWLIPT_LNK_MULTICAST, ),
                MAKE_DISCOVERY_LWLINK_LR10(SYS_PERFMON_MULTICAST, ),
                MAKE_DISCOVERY_LWLINK_LR10(LWLDL, ),
                MAKE_DISCOVERY_LWLINK_LR10(LWLIPT_LNK, ),
                MAKE_DISCOVERY_LWLINK_LR10(SYS_PERFMON, )
            };
            LwU32 discovery_table_lwlw_size = LW_ARRAY_ELEMENTS(discovery_table_lwlw);

            status = _lwswitch_device_discovery_lr10(
                device, chip_device->engLWLW[i].info.top.discovery, discovery_table_lwlw, 
                discovery_table_lwlw_size, &discovery_handlers_lwlw_lr10);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "LWLW[%d] discovery failed\n", i);
                return status;
            }
        }
    }
    if (VERBOSE_MMIO_DISCOVERY)
    {
        LWSWITCH_PRINT(device, SETUP,
            "LWLW[0..%d] Discovery\n",
             NUM_LWLW_ENGINE_LR10-1);
        DISCOVERY_DUMP_ENGINE_LR10(device, MINION, );
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLIPT, );
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLTLC, );
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLTLC_MULTICAST, );
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLIPT_SYS_PERFMON, );
        DISCOVERY_DUMP_ENGINE_LR10(device, TX_PERFMON_MULTICAST, );
        DISCOVERY_DUMP_ENGINE_LR10(device, RX_PERFMON_MULTICAST, );
        DISCOVERY_DUMP_ENGINE_LR10(device, TX_PERFMON, );
        DISCOVERY_DUMP_ENGINE_LR10(device, RX_PERFMON, );
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLW, );
        DISCOVERY_DUMP_ENGINE_LR10(device, PLL, );
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLW_PERFMON, );
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLDL_MULTICAST, );
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLIPT_LNK_MULTICAST, );
        DISCOVERY_DUMP_ENGINE_LR10(device, SYS_PERFMON_MULTICAST, );
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLDL, );
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLIPT_LNK, );
        DISCOVERY_DUMP_ENGINE_LR10(device, SYS_PERFMON, );
    }

    for (i = 0; i < NUM_LWLW_BCAST_ENGINE_LR10; i++)
    {
        if (LWSWITCH_ENG_VALID_LR10(device, LWLW_BCAST, i) && 
            (chip_device->engLWLW_BCAST[i].info.top.discovery != 0))
        {
            DISCOVERY_TABLE_TYPE_LR10 discovery_table_lwlw[] =
            {
                MAKE_DISCOVERY_LWLINK_LR10(MINION, _BCAST),
                MAKE_DISCOVERY_LWLINK_LR10(LWLIPT, _BCAST),
                MAKE_DISCOVERY_LWLINK_LR10(LWLTLC, _BCAST),
                MAKE_DISCOVERY_LWLINK_LR10(LWLTLC_MULTICAST, _BCAST),
                MAKE_DISCOVERY_LWLINK_LR10(LWLIPT_SYS_PERFMON, _BCAST),
                MAKE_DISCOVERY_LWLINK_LR10(TX_PERFMON_MULTICAST, _BCAST),
                MAKE_DISCOVERY_LWLINK_LR10(RX_PERFMON_MULTICAST, _BCAST),
                MAKE_DISCOVERY_LWLINK_LR10(TX_PERFMON, _BCAST),
                MAKE_DISCOVERY_LWLINK_LR10(RX_PERFMON, _BCAST),
                MAKE_DISCOVERY_LWLINK_LR10(LWLW, _BCAST),
                MAKE_DISCOVERY_LWLINK_LR10(PLL, _BCAST),
                MAKE_DISCOVERY_LWLINK_LR10(LWLW_PERFMON, _BCAST),
                MAKE_DISCOVERY_LWLINK_LR10(LWLDL_MULTICAST, _BCAST),
                MAKE_DISCOVERY_LWLINK_LR10(LWLIPT_LNK_MULTICAST, _BCAST),
                MAKE_DISCOVERY_LWLINK_LR10(SYS_PERFMON_MULTICAST, _BCAST),
                MAKE_DISCOVERY_LWLINK_LR10(LWLDL, _BCAST),
                MAKE_DISCOVERY_LWLINK_LR10(LWLIPT_LNK, _BCAST),
                MAKE_DISCOVERY_LWLINK_LR10(SYS_PERFMON, _BCAST)
            };
            LwU32 discovery_table_lwlw_size = LW_ARRAY_ELEMENTS(discovery_table_lwlw);

            status = _lwswitch_device_discovery_lr10(
                device, chip_device->engLWLW_BCAST[i].info.top.discovery, discovery_table_lwlw, 
                discovery_table_lwlw_size, &discovery_handlers_lwlw_lr10);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "LWLW_BCAST[%d] discovery failed\n", i);
                return status;
            }
        }
    }
    if (VERBOSE_MMIO_DISCOVERY)
    {
        LWSWITCH_PRINT(device, SETUP,
            "LWLW_BCAST[0..%d] Discovery\n",
             NUM_LWLW_BCAST_ENGINE_LR10-1);
        DISCOVERY_DUMP_ENGINE_LR10(device, MINION, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLIPT, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLTLC, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLTLC_MULTICAST, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLIPT_SYS_PERFMON, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, TX_PERFMON_MULTICAST, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, RX_PERFMON_MULTICAST, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, TX_PERFMON, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, RX_PERFMON, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLW, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, PLL, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLW_PERFMON, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLDL_MULTICAST, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLIPT_LNK_MULTICAST, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, SYS_PERFMON_MULTICAST, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLDL, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, LWLIPT_LNK, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, SYS_PERFMON, _BCAST);
    }

    for (i = 0; i < NUM_NPG_ENGINE_LR10; i++)
    {
        if (LWSWITCH_ENG_VALID_LR10(device, NPG, i) && 
            (chip_device->engNPG[i].info.top.discovery != 0))
        {
            DISCOVERY_TABLE_TYPE_LR10 discovery_table_npg[] =
            {
                MAKE_DISCOVERY_NPG_LR10(NPG, ),
                MAKE_DISCOVERY_NPG_LR10(NPORT, ),
                MAKE_DISCOVERY_NPG_LR10(NPORT_MULTICAST, ),
                MAKE_DISCOVERY_NPG_LR10(NPG_PERFMON, ),
                MAKE_DISCOVERY_NPG_LR10(NPORT_PERFMON, ),
                MAKE_DISCOVERY_NPG_LR10(NPORT_PERFMON_MULTICAST, )
            };
            LwU32 discovery_table_npg_size = LW_ARRAY_ELEMENTS(discovery_table_npg);

            status = _lwswitch_device_discovery_lr10(
                device, chip_device->engNPG[i].info.top.discovery, discovery_table_npg, 
                discovery_table_npg_size, &discovery_handlers_npg_lr10);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "NPG[%d] discovery failed\n", i);
                return status;
            }
        }
    }
    if (VERBOSE_MMIO_DISCOVERY)
    {
        LWSWITCH_PRINT(device, SETUP,
            "NPG[0..%d] Discovery\n",
             NUM_NPG_ENGINE_LR10-1);
        DISCOVERY_DUMP_ENGINE_LR10(device, NPG, );
        DISCOVERY_DUMP_ENGINE_LR10(device, NPORT, );
        DISCOVERY_DUMP_ENGINE_LR10(device, NPORT_MULTICAST, );
        DISCOVERY_DUMP_ENGINE_LR10(device, NPG_PERFMON, );
        DISCOVERY_DUMP_ENGINE_LR10(device, NPORT_PERFMON, );
        DISCOVERY_DUMP_ENGINE_LR10(device, NPORT_PERFMON_MULTICAST, );
    }

    for (i = 0; i < NUM_NPG_BCAST_ENGINE_LR10; i++)
    {
        if (LWSWITCH_ENG_VALID_LR10(device, NPG_BCAST, i) && 
            (chip_device->engNPG_BCAST[i].info.top.discovery != 0))
        {
            DISCOVERY_TABLE_TYPE_LR10 discovery_table_npg[] =
            {
                MAKE_DISCOVERY_NPG_LR10(NPG, _BCAST),
                MAKE_DISCOVERY_NPG_LR10(NPORT, _BCAST),
                MAKE_DISCOVERY_NPG_LR10(NPORT_MULTICAST, _BCAST),
                MAKE_DISCOVERY_NPG_LR10(NPG_PERFMON, _BCAST),
                MAKE_DISCOVERY_NPG_LR10(NPORT_PERFMON, _BCAST),
                MAKE_DISCOVERY_NPG_LR10(NPORT_PERFMON_MULTICAST, _BCAST)
            };
            LwU32 discovery_table_npg_size = LW_ARRAY_ELEMENTS(discovery_table_npg);

            status = _lwswitch_device_discovery_lr10(
                device, chip_device->engNPG_BCAST[i].info.top.discovery, discovery_table_npg, 
                discovery_table_npg_size, &discovery_handlers_npg_lr10);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "NPG_BCAST[%d] discovery failed\n", i);
                return status;
            }
        }
    }
    if (VERBOSE_MMIO_DISCOVERY)
    {
        LWSWITCH_PRINT(device, SETUP,
            "NPG_BCAST[%d] Discovery\n",
            NUM_NPG_BCAST_ENGINE_LR10-1);
        DISCOVERY_DUMP_ENGINE_LR10(device, NPG, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, NPORT, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, NPORT_MULTICAST, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, NPG_PERFMON, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, NPORT_PERFMON, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, NPORT_PERFMON_MULTICAST, _BCAST);
    }

    for (i = 0; i < NUM_NXBAR_ENGINE_LR10; i++)
    {
        if (LWSWITCH_ENG_VALID_LR10(device, NXBAR, i) && 
            (chip_device->engNXBAR[i].info.top.discovery != 0))
        {
            DISCOVERY_TABLE_TYPE_LR10 discovery_table_nxbar[] =
            {
                MAKE_DISCOVERY_NXBAR_LR10(NXBAR, ),
                MAKE_DISCOVERY_NXBAR_LR10(TILE, ),
                MAKE_DISCOVERY_NXBAR_LR10(TILE_MULTICAST, ),
                MAKE_DISCOVERY_NXBAR_LR10(NXBAR_PERFMON, ),
                MAKE_DISCOVERY_NXBAR_LR10(TILE_PERFMON, ),
                MAKE_DISCOVERY_NXBAR_LR10(TILE_PERFMON_MULTICAST, )
            };
            LwU32 discovery_table_nxbar_size = LW_ARRAY_ELEMENTS(discovery_table_nxbar);

            status = _lwswitch_device_discovery_lr10(
                device, chip_device->engNXBAR[i].info.top.discovery, 
                discovery_table_nxbar, discovery_table_nxbar_size, 
                &discovery_handlers_nxbar_lr10);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "NXBAR[%d] discovery failed\n", i);
                return status;
            }
        }
    }
    if (VERBOSE_MMIO_DISCOVERY)
    {
        LWSWITCH_PRINT(device, SETUP,
            "NXBAR[0..%d] Discovery\n",
            NUM_NXBAR_ENGINE_LR10-1);
        DISCOVERY_DUMP_ENGINE_LR10(device, NXBAR, );
        DISCOVERY_DUMP_ENGINE_LR10(device, TILE, );
        DISCOVERY_DUMP_ENGINE_LR10(device, TILE_MULTICAST, );
        DISCOVERY_DUMP_ENGINE_LR10(device, NXBAR_PERFMON, );
        DISCOVERY_DUMP_ENGINE_LR10(device, TILE_PERFMON, );
        DISCOVERY_DUMP_ENGINE_LR10(device, TILE_PERFMON_MULTICAST, );
    }

    for (i = 0; i < NUM_NXBAR_BCAST_ENGINE_LR10; i++)
    {
        if (LWSWITCH_ENG_VALID_LR10(device, NXBAR_BCAST, i) && 
            (chip_device->engNXBAR_BCAST[i].info.top.discovery != 0))
        {
            DISCOVERY_TABLE_TYPE_LR10 discovery_table_nxbar[] =
            {
                MAKE_DISCOVERY_NXBAR_LR10(NXBAR, _BCAST),
                MAKE_DISCOVERY_NXBAR_LR10(TILE, _BCAST),
                MAKE_DISCOVERY_NXBAR_LR10(TILE_MULTICAST, _BCAST),
                MAKE_DISCOVERY_NXBAR_LR10(NXBAR_PERFMON, _BCAST),
                MAKE_DISCOVERY_NXBAR_LR10(TILE_PERFMON, _BCAST),
                MAKE_DISCOVERY_NXBAR_LR10(TILE_PERFMON_MULTICAST, _BCAST)
            };
            LwU32 discovery_table_nxbar_size = LW_ARRAY_ELEMENTS(discovery_table_nxbar);

            status = _lwswitch_device_discovery_lr10(
                device, chip_device->engNXBAR_BCAST[i].info.top.discovery, 
                discovery_table_nxbar, discovery_table_nxbar_size, 
                &discovery_handlers_nxbar_lr10);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "NXBAR_BCAST[%d] discovery failed\n", i);
                return status;
            }
        }
    }
    if (VERBOSE_MMIO_DISCOVERY)
    {
        LWSWITCH_PRINT(device, SETUP,
            "NXBAR_BCAST[0..%d] Discovery\n",
            NUM_NXBAR_BCAST_ENGINE_LR10-1);
        DISCOVERY_DUMP_ENGINE_LR10(device, NXBAR, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, TILE, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, TILE_MULTICAST, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, NXBAR_PERFMON, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, TILE_PERFMON, _BCAST);
        DISCOVERY_DUMP_ENGINE_LR10(device, TILE_PERFMON_MULTICAST, _BCAST);
    }

    return status;
}

//
// Filter engine discovery information to handle platform-specific differences.
//
// Emulation and RTL sims have devices that show up in the discovery table but
// are actually tied off and not present.  On GPU the engine enables and
// floorsweeping info are used to disable devices that are not present.
// But a similar mechanism does not exist in LwSwitch.
// So instead we ilwalidate the devices that are known to be not-present on a
// given platform.
//

void
lwswitch_filter_discovery_lr10
(
    lwswitch_device *device
)
{
    return;
}

#define LWSWITCH_PROCESS_DISCOVERY(_lwrrent, _engine, _multicast)           \
    {                                                                       \
        LwU32 i;                                                            \
        ct_assert(NUM_##_engine##_ENGINE_LR10 <= LWSWITCH_ENGINE_DESCRIPTOR_UC_SIZE); \
                                                                            \
        _lwrrent->eng_name = #_engine;                                      \
        _lwrrent->eng_id = LWSWITCH_ENGINE_ID_##_engine;                    \
        _lwrrent->eng_count = NUM_##_engine##_ENGINE_LR10;                  \
                                                                            \
        for (i = 0; i < NUM_##_engine##_ENGINE_LR10; i++)                   \
        {                                                                   \
            if (chip_device->eng##_engine[i].valid)                         \
            {                                                               \
                _lwrrent->uc_addr[i] =                                      \
                    chip_device->eng##_engine[i].info.uc.uc_addr;           \
            }                                                               \
        }                                                                   \
                                                                            \
        if (chip_device->eng##_engine##_multicast[0].valid)                 \
        {                                                                   \
            _lwrrent->bc_addr =                                             \
                chip_device->eng##_engine##_multicast[0].info.bc.bc_addr;   \
        }                                                                   \
                                                                            \
        _lwrrent->mc_addr_count = 0;                                        \
    }                                                                       \

#define LWSWITCH_PROCESS_COMMON(_engine, _multicast)                        \
    {                                                                       \
        LWSWITCH_ENGINE_DESCRIPTOR_TYPE *current;                           \
        ct_assert(LWSWITCH_ENGINE_ID_##_engine < LWSWITCH_ENGINE_ID_SIZE);  \
                                                                            \
        current = &chip_device->io.common[LWSWITCH_ENGINE_ID_##_engine];    \
        LWSWITCH_PROCESS_DISCOVERY(current, _engine, _multicast)            \
    }

//
// Process engine discovery information to associate engines
//

LwlStatus
lwswitch_process_discovery_lr10
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32       i, j;
    LwlStatus   retval = LWL_SUCCESS;
    LwU64       link_enable_mask;

    //
    // Process per-link information
    //
    for (i = 0; i < LWSWITCH_NUM_LINKS_LR10; i++)
    {
        device->link[i].valid = 
            LWSWITCH_ENG_VALID_LR10(device, NPORT, LWSWITCH_GET_LINK_ENG_INST(device, i, NPORT)) &&
            LWSWITCH_ENG_VALID_LR10(device, LWLTLC, LWSWITCH_GET_LINK_ENG_INST(device, i, LWLTLC)) &&
            LWSWITCH_ENG_VALID_LR10(device, LWLDL, LWSWITCH_GET_LINK_ENG_INST(device, i, LWLDL)) &&
            LWSWITCH_ENG_VALID_LR10(device, LWLIPT_LNK, LWSWITCH_GET_LINK_ENG_INST(device, i, LWLIPT_LNK)) &&
            LWSWITCH_ENG_VALID_LR10(device, LWLW, LWSWITCH_GET_LINK_ENG_INST(device, i, LWLW)) &&
            LWSWITCH_ENG_VALID_LR10(device, MINION, LWSWITCH_GET_LINK_ENG_INST(device, i, MINION)) &&
            LWSWITCH_ENG_VALID_LR10(device, LWLIPT, LWSWITCH_GET_LINK_ENG_INST(device, i, LWLIPT));
    }

    //
    // Disable engines requested by regkey "LinkEnableMask".
    // All the links are enabled by default.
    //
    link_enable_mask = ((LwU64)device->regkeys.link_enable_mask2 << 32 |
        (LwU64)device->regkeys.link_enable_mask);

    for (i = 0; i < LWSWITCH_NUM_LINKS_LR10; i++)
    {
        if ((LWBIT64(i) & link_enable_mask) == 0)
        {
            LWSWITCH_PRINT(device, SETUP,
                "%s: Disable link #%d\n",
                __FUNCTION__, i);
            device->link[i].valid                  = LW_FALSE;
            chip_device->engNPORT[i].valid         = LW_FALSE;
            chip_device->engNPORT_PERFMON[i].valid = LW_FALSE;
            chip_device->engLWLTLC[i].valid        = LW_FALSE;
            chip_device->engTX_PERFMON[i].valid    = LW_FALSE;
            chip_device->engRX_PERFMON[i].valid    = LW_FALSE;
        }
    }

    //
    // Process common engine information
    //

    // Mark all entries as invalid
    for (i = 0; i < LWSWITCH_ENGINE_ID_SIZE; i++)
    {
        chip_device->io.common[i].eng_name = "";
        chip_device->io.common[i].eng_id = LWSWITCH_ENGINE_ID_SIZE; // Out of range
        chip_device->io.common[i].eng_count = 0;
        for (j = 0; j < LWSWITCH_ENGINE_DESCRIPTOR_UC_SIZE; j++)
        {
            chip_device->io.common[i].uc_addr[j] = LWSWITCH_BASE_ADDR_ILWALID;
        }
        chip_device->io.common[i].bc_addr = LWSWITCH_BASE_ADDR_ILWALID;
        for (j = 0; j < LWSWITCH_ENGINE_DESCRIPTOR_MC_SIZE; j++)
        {
            chip_device->io.common[i].mc_addr[j] = LWSWITCH_BASE_ADDR_ILWALID;
        }
        chip_device->io.common[i].mc_addr_count = 0;
    }

    LWSWITCH_LIST_LR10_ENGINES(LWSWITCH_PROCESS_COMMON)

    return retval;
}
