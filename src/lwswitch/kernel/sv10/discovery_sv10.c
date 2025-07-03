/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2016-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "common_lwswitch.h"
#include "sv10/sv10.h"

#include "lwswitch/svnp01/dev_lws_top.h"
#include "lwswitch/svnp01/npgip_discovery.h"
#include "lwswitch/svnp01/lwlinkip_discovery.h"
#include "lwswitch/svnp01/swxip_discovery.h"

#define VERBOSE_MMIO_DISCOVERY      0

#define NUM_SIOCTRL_SUBENGINE_SV10               (NUM_SIOCTRL_ENGINE_SV10+NUM_SIOCTRL_BCAST_ENGINE_SV10)
#define NUM_LWLTLC_SUBENGINE_SV10                (NUM_SIOCTRL_SUBENGINE_SV10*NUM_LWLTLC_INSTANCES_SV10)
#define NUM_DLPL_SUBENGINE_SV10                  (NUM_SIOCTRL_SUBENGINE_SV10*NUM_DLPL_INSTANCES_SV10)
#define NUM_TX_PERFMON_SUBENGINE_SV10            (NUM_SIOCTRL_SUBENGINE_SV10*NUM_TX_PERFMON_INSTANCES_SV10)
#define NUM_RX_PERFMON_SUBENGINE_SV10            (NUM_SIOCTRL_SUBENGINE_SV10*NUM_RX_PERFMON_INSTANCES_SV10)
#define NUM_IOCTRL_SUBENGINE_SV10                0   // GPU only
#define NUM_LWLTL_SUBENGINE_SV10                 0   // GP100 only
#define NUM_LWLINK_SUBENGINE_SV10                0
#define NUM_MINION_SUBENGINE_SV10                (NUM_SIOCTRL_SUBENGINE_SV10)
#define NUM_LWLIPT_SUBENGINE_SV10                (NUM_SIOCTRL_SUBENGINE_SV10)
#define NUM_IOCTRLMIF_SUBENGINE_SV10             0   // GV100 only
#define NUM_DLPL_MULTICAST_SUBENGINE_SV10        (NUM_SIOCTRL_SUBENGINE_SV10)
#define NUM_LWLTLC_MULTICAST_SUBENGINE_SV10      (NUM_SIOCTRL_SUBENGINE_SV10)
#define NUM_IOCTRLMIF_MULTICAST_SUBENGINE_SV10   0   // GV100 only
#define NUM_TIOCTRL_SUBENGINE_SV10               0   // CheetAh only
#define NUM_SIOCTRL_PERFMON_SUBENGINE_SV10       NUM_SIOCTRL_SUBENGINE_SV10
#define NUM_LWLIPT_SYS_PERFMON_SUBENGINE_SV10    NUM_SIOCTRL_SUBENGINE_SV10
#define NUM_TX_PERFMON_MULTICAST_SUBENGINE_SV10  (NUM_SIOCTRL_SUBENGINE_SV10)
#define NUM_RX_PERFMON_MULTICAST_SUBENGINE_SV10  (NUM_SIOCTRL_SUBENGINE_SV10)

#define MAKE_DISCOVERY_SV10(device, chip, engine)   \
    {                                               \
        #engine,                                    \
        NUM_##engine##_ENG_INSTANCES_##chip,        \
        NUM_##engine##_ENGINE_##chip,               \
        LW_SWPTOP_ENUM_DEVICE_##engine,             \
        chip_device->eng##engine,                   \
        &chip_device->num##engine                   \
    }

#define CLEAR_ENGINE_SV10(device, chip, engine)     \
    chip_device->num##engine = 0;                   \
    lwswitch_os_memset(&(chip_device->eng##engine[0]),   \
            0, NUM_##engine##_ENGINE_##chip*sizeof(ENGINE_DESCRIPTOR_TYPE_SV10))

//
// Engine discovery lookup table used when parsing PTOP, SWX, LWLINK, and NPG
// discovery information
//

typedef struct
{
    const char *engname;
    LwU32 instcount_max;
    LwU32 engcount_max;
    LwU32 discovery_id;
    ENGINE_DESCRIPTOR_TYPE_SV10    *engine;
    LwU32 *instcount;
}
DISCOVERY_TABLE_TYPE_SV10;

#define LWSWITCH_DISCOVERY_ENTRY_ILWALID    0x0
#define LWSWITCH_DISCOVERY_ENTRY_ENUM       0x1
#define LWSWITCH_DISCOVERY_ENTRY_DATA1      0x2
#define LWSWITCH_DISCOVERY_ENTRY_DATA2      0x3

typedef struct
{
    void (*parse_entry)(lwswitch_device *device, LwU32 entry, LwU32 *entry_type, LwBool *entry_chain);
    void (*parse_enum)(lwswitch_device *device, LwU32 entry, LwU32 *entry_device, LwU32 *entry_id, LwU32 *entry_version);
    void (*handle_data1)(lwswitch_device *device, LwU32 entry, ENGINE_DESCRIPTOR_TYPE_SV10 *engine, LwU32 entry_device, LwU32 *discovery_list_size);
    void (*handle_data2)(lwswitch_device *device, LwU32 entry, ENGINE_DESCRIPTOR_TYPE_SV10 *engine, LwU32 entry_device);
}
LWSWITCH_DISCOVERY_HANDLERS_SV10;

//
// Debug function for dumping the parsed engine discovery information
//

static void
_discovery_dump_eng_sv10
(
    lwswitch_device *device,
    const char *eng_name,
    ENGINE_DESCRIPTOR_TYPE_SV10 *engine,
    LwU32   count
)
{
    LwU32 i;

    if (VERBOSE_MMIO_DISCOVERY)
    {
        for (i = 0; i < count; i++)
        {
            if (engine[i].valid)
            {
                LWSWITCH_PRINT(device, SETUP,
                    "%-24s[%2d]:%2x[%2d]:V:%1x UC:%6x BC:%6x"
                    " RST:%6x[%2x] INT:%6x[%2x] CL/ID:%2x/%2x DISC:%6x\n",
                    eng_name, i,
                    engine[i].engine, engine[i].instance,
                    engine[i].version,
                    engine[i].uc_addr, engine[i].bc_addr,
                    engine[i].reset_addr, engine[i].reset_bit,
                    engine[i].intr_addr, engine[i].intr_bit,
                    engine[i].cluster, engine[i].cluster_id,
                    engine[i].discovery);
            }
        }
    }
}

#define DISCOVERY_DUMP_ENGINE_SV10(_device,  _engine)          \
    _discovery_dump_eng_sv10(_device, #_engine, &(LWSWITCH_GET_CHIP_DEVICE_SV10(_device)->eng##_engine[0]), LWSWITCH_GET_CHIP_DEVICE_SV10(_device)->num##_engine);

#define DISCOVERY_DUMP_SUBENGINE_SV10(_device, _engine)       \
    _discovery_dump_eng_sv10(_device, #_engine, &(engine->subeng##_engine[0]), engine->num##_engine);

//
// Parse engine discovery information to identify MMIO, interrupt, and
// reset information
//

static LwlStatus
_lwswitch_device_discovery_sv10
(
    lwswitch_device *device,
    LwU32   discovery_offset,
    DISCOVERY_TABLE_TYPE_SV10 *discovery_table,
    LwU32 discovery_table_size,
    LWSWITCH_DISCOVERY_HANDLERS_SV10 *discovery_handlers
)
{
    ENGINE_DESCRIPTOR_TYPE_SV10 *engine = NULL;
    LwU32                   entry_type = LWSWITCH_DISCOVERY_ENTRY_ILWALID;
    LwBool                  entry_chain = LW_FALSE;
    LwU32                   entry = 0;
    LwU32                   entry_device = 0;
    LwU32                   entry_id = 0;
    LwU32                   entry_version = 0;
    LwU32                   entry_count = 0;
    LwBool                  done = LW_FALSE;
    LwlStatus               retval = LWL_SUCCESS;
    LwU32                   eng_inst_index = 0;

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
                                if (*(discovery_table[i].instcount) < discovery_table[i].instcount_max)
                                {
                                    eng_inst_index = *(discovery_table[i].instcount);
                                    engine = &(discovery_table[i].engine[eng_inst_index]);
                                    (*discovery_table[i].instcount)++;
                                    break;
                                }
                                else
                                {
                                    LWSWITCH_PRINT(device, ERROR,
                                        "%s:_ENUM: ERROR: %s[%d] out of instance range %d..%d\n",
                                        __FUNCTION__,
                                        discovery_table[i].engname,
                                        entry_id,
                                        0, discovery_table[i].instcount_max-1);
                                }
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
                    if (engine->valid == LW_TRUE)
                    {
                        LWSWITCH_PRINT(device, WARN,
                            "%s:_ENUM: WARNING: device=%x id=%x previously discovered!\n",
                            __FUNCTION__,
                            entry_device, entry_id);
                    }
                    LWSWITCH_ASSERT(engine->valid != LW_TRUE);
                    engine->valid = LW_TRUE;

                    engine->engine   = entry_device;
                    engine->instance = entry_id;
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
_lwswitch_ptop_parse_entry_sv10
(
    lwswitch_device *device,
    LwU32   entry,
    LwU32   *entry_type,
    LwBool  *entry_chain
)
{
    LwU32 entry_type_sv10;

    entry_type_sv10 = DRF_VAL(_SWPTOP, _, ENTRY, entry);
    *entry_chain = FLD_TEST_DRF(_SWPTOP, _, CHAIN, _ENABLE, entry);

    switch (entry_type_sv10)
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
_lwswitch_ptop_parse_enum_sv10
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
_lwswitch_ptop_handle_data1_sv10
(
    lwswitch_device *device,
    LwU32 entry,
    ENGINE_DESCRIPTOR_TYPE_SV10 *engine,
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

    engine->reset_bit  = DRF_VAL(_SWPTOP, _DATA1, _RESET, entry);
    engine->intr_bit   = DRF_VAL(_SWPTOP, _DATA1, _INTR, entry);
    engine->cluster    = DRF_VAL(_SWPTOP, _DATA1, _CLUSTER_TYPE, entry);
    engine->cluster_id = DRF_VAL(_SWPTOP, _DATA1, _CLUSTER_NUMBER, entry);
}

static void
_lwswitch_ptop_handle_data2_sv10
(
    lwswitch_device *device,
    LwU32 entry,
    ENGINE_DESCRIPTOR_TYPE_SV10 *engine,
    LwU32 entry_device
)
{
    LwU32 data2_type = DRF_VAL(_SWPTOP, _DATA2, _TYPE, entry);
    LwU32 data2_addr = DRF_VAL(_SWPTOP, _DATA2, _ADDR, entry);

    switch(data2_type)
    {
        case LW_SWPTOP_DATA2_TYPE_RESETREG:
            engine->reset_addr = data2_addr*sizeof(LwU32);
            break;
        case LW_SWPTOP_DATA2_TYPE_INTRREG:
            engine->intr_addr = data2_addr*sizeof(LwU32);
            break;
        case LW_SWPTOP_DATA2_TYPE_DISCOVERY:
            // Parse sub-discovery table
            engine->discovery = data2_addr*sizeof(LwU32);
            break;
        case LW_SWPTOP_DATA2_TYPE_UNICAST:
            engine->uc_addr = data2_addr*sizeof(LwU32);
            break;
        case LW_SWPTOP_DATA2_TYPE_BROADCAST:
            engine->bc_addr = data2_addr*sizeof(LwU32);
            break;
        case LW_SWPTOP_DATA2_TYPE_ILWALID:
            LWSWITCH_PRINT(device, SETUP,
                "%s:_DATA2: %s=%6x\n",
                __FUNCTION__,
                "_ILWALID", data2_addr);
            break;
        default:
            LWSWITCH_PRINT(device, SETUP,
                "%s:_DATA2: Unknown type 0x%x (0x%08x)!\n",
                __FUNCTION__, data2_type, entry);
            LWSWITCH_ASSERT(0);
            break;
    }
}


static void
_lwswitch_sioctrl_parse_entry_sv10
(
    lwswitch_device *device,
    LwU32   entry,
    LwU32   *entry_type,
    LwBool  *entry_chain
)
{
    LwU32 entry_type_sioctrl;

    entry_type_sioctrl = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _ENTRY, entry);
    *entry_chain = FLD_TEST_DRF(_LWLINKIP, _DISCOVERY_COMMON, _CHAIN, _ENABLE, entry);

    switch (entry_type_sioctrl)
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

static void
_lwswitch_sioctrl_parse_enum_sv10
(
    lwswitch_device *device,
    LwU32   entry,
    LwU32   *entry_device,
    LwU32   *entry_id,
    LwU32   *entry_version
)
{
    *entry_device  = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DEVICE, entry);
    *entry_id      = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _ID, entry);
    *entry_version = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _VERSION, entry);
    LWSWITCH_ASSERT(DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _RESERVED, entry) == 0);

    if (*entry_version != LW_LWLINKIP_DISCOVERY_COMMON_VERSION_3)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s:_LWLINKIP, _DISCOVERY_COMMON, _VERSION = %x but expected %x (_VERSION_3).\n",
            __FUNCTION__, *entry_version, LW_LWLINKIP_DISCOVERY_COMMON_VERSION_3);
    }
}

static void
_lwswitch_map_dlpl_to_sioctrl
(
    lwswitch_device *device,
    LwU32 entry_device,
    LwU32 dlpl_instance,
    LWLINK_SUBENGINE_DESCRIPTOR_TYPE_SV10 **subengSIOCTRL,
    LwU32 *idx_dlpl
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 idx_sioctrl;

    if (LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_DLPL == entry_device)
    {
        *idx_dlpl = dlpl_instance % NUM_DLPL_INSTANCES_SV10;
        idx_sioctrl = dlpl_instance / NUM_DLPL_INSTANCES_SV10;
    }
    else
    {
        *idx_dlpl = dlpl_instance % NUM_DLPL_MULTICAST_INSTANCES_SV10;
        idx_sioctrl = dlpl_instance / NUM_DLPL_MULTICAST_INSTANCES_SV10;
    }

    if (idx_sioctrl < NUM_SIOCTRL_ENGINE_SV10)
    {
        *subengSIOCTRL = &chip_device->subengSIOCTRL[idx_sioctrl];
    }
    else if (idx_sioctrl < NUM_SIOCTRL_SUBENGINE_SV10)
    {
        *subengSIOCTRL = &chip_device->subengSIOCTRL_BCAST[idx_sioctrl - NUM_SIOCTRL_ENGINE_SV10];
    }
    else
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s:Processing DLPL[%d] into SIOCTRL failed\n",
            __FUNCTION__,
            dlpl_instance);
        *subengSIOCTRL = NULL;
        *idx_dlpl = 0;
    }
}

static void
_lwswitch_sioctrl_handle_data1_sv10
(
    lwswitch_device *device,
    LwU32 entry,
    ENGINE_DESCRIPTOR_TYPE_SV10 *engine,
    LwU32 entry_device,
    LwU32 *discovery_list_size
)
{
    if ((LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_IOCTRL == entry_device) ||
        (LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_SIOCTRL == entry_device) ||
        (LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_TIOCTRL == entry_device))
    {
        *discovery_list_size = 
            DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DATA1_IOCTRL_LENGTH, entry);
    }

    if (engine == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s:DATA1:engine == NULL.  Skipping processing\n",
            __FUNCTION__);
        return;
    }

    engine->reset_bit  = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _RESET, entry);
    engine->intr_bit   = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _INTR, entry);

    if (LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_LWLIPT == entry_device)
    {
        LWSWITCH_ASSERT(0 == DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _LWLIPT_RESERVED, entry));
    }
    else if ((LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_DLPL == entry_device) ||
        (LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_DLPL_MULTICAST == entry_device))
    {
        LWLINK_SUBENGINE_DESCRIPTOR_TYPE_SV10 *subengSIOCTRL;
        LwU32 num_tx = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DLPL_NUM_TX, entry);
        LwU32 num_rx = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DLPL_NUM_RX, entry);
        LwU32 idx_dlpl;

        LWSWITCH_ASSERT(0 == DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DLPL_RESERVED, entry));

        _lwswitch_map_dlpl_to_sioctrl(device, entry_device, engine->instance, &subengSIOCTRL, &idx_dlpl);
        if (subengSIOCTRL)
        {
            if (LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_DLPL == entry_device)
            {
                subengSIOCTRL->dlpl_info[idx_dlpl].num_tx = num_tx;
                subengSIOCTRL->dlpl_info[idx_dlpl].num_rx = num_rx;
            }
            else
            {
                subengSIOCTRL->dlpl_info_multicast.num_tx = num_tx;
                subengSIOCTRL->dlpl_info_multicast.num_rx = num_rx;
            }
        }
        else
        {
            LWSWITCH_PRINT(device, SETUP,
                "%s:Failed to associate DLPL[%d] with an SIOCTRL\n",
                __FUNCTION__,
                engine->instance);
        }
    }
    else if ((LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_IOCTRL != entry_device) &&
        (LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_SIOCTRL != entry_device) &&
        (LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_TIOCTRL != entry_device))
    {
        LWSWITCH_ASSERT(0 == DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DATA1_RESERVED, entry));
    }
}

static void
_lwswitch_sioctrl_handle_data2_sv10
(
    lwswitch_device *device,
    LwU32 entry,
    ENGINE_DESCRIPTOR_TYPE_SV10 *engine,
    LwU32 entry_device
)
{
    LWLINK_SUBENGINE_DESCRIPTOR_TYPE_SV10 *subengSIOCTRL;
    LwU32 idx_dlpl;
    LwU32 data2_type = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON_DATA2, _TYPE, entry);
    LwU32 data2_addr = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON_DATA2, _ADDR, entry);

    switch(data2_type)
    {
        case LW_LWLINKIP_DISCOVERY_COMMON_DATA2_TYPE_PLLCONTROL:

            _lwswitch_map_dlpl_to_sioctrl(
                device, entry_device, engine->instance, &subengSIOCTRL, &idx_dlpl);
            if (subengSIOCTRL)
            {
                if (LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_DLPL == entry_device)
                {
                    subengSIOCTRL->dlpl_info[idx_dlpl].master =
                        DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DLPL_DATA2_MASTER, entry);
                    subengSIOCTRL->dlpl_info[idx_dlpl].master_id =
                        DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DLPL_DATA2_MASTERID, entry);
                }
                else
                {
                    subengSIOCTRL->dlpl_info_multicast.master =
                        DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DLPL_DATA2_MASTER, entry);
                    subengSIOCTRL->dlpl_info_multicast.master_id =
                        DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DLPL_DATA2_MASTERID, entry);
                }
            }
            else
            {
                LWSWITCH_PRINT(device, SETUP,
                    "%s:Failed to associate DLPL[%d] PLLCONTROL with an SIOCTRL\n",
                    __FUNCTION__,
                    engine->instance);
            }

            LWSWITCH_ASSERT(LW_LWLINKIP_DISCOVERY_COMMON_DLPL_DATA2_TYPE_PLLCONTROL ==
                DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DLPL_DATA2_TYPE, entry));
            LWSWITCH_ASSERT(0 == DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DLPL_DATA2_RESERVED, entry));
            LWSWITCH_ASSERT(0 == DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DLPL_DATA2_RESERVED2, entry));

            break;

        case LW_LWLINKIP_DISCOVERY_COMMON_DATA2_TYPE_RESETREG:
            engine->reset_addr = data2_addr*sizeof(LwU32);
            break;

        case LW_LWLINKIP_DISCOVERY_COMMON_DATA2_TYPE_INTRREG:
            engine->intr_addr = data2_addr*sizeof(LwU32);
            break;

        case LW_LWLINKIP_DISCOVERY_COMMON_DATA2_TYPE_DISCOVERY:
            // Parse sub-discovery table
            engine->discovery = data2_addr*sizeof(LwU32);

            //
            // Lwrrently _DISCOVERY is not used in the second
            // level discovery.
            //
            LWSWITCH_ASSERT(0);

            break;

        case LW_LWLINKIP_DISCOVERY_COMMON_DATA2_TYPE_UNICAST:
            if (engine->version <= LW_LWLINKIP_DISCOVERY_COMMON_VERSION_LWLINK20)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s:_DATA2 _UNICAST may conflict with _PRI_BASE\n",
                    __FUNCTION__);
            }
            engine->uc_addr = data2_addr*sizeof(LwU32);
            break;

        case LW_LWLINKIP_DISCOVERY_COMMON_DATA2_TYPE_BROADCAST:
            engine->bc_addr = data2_addr*sizeof(LwU32);
            break;

        case LW_LWLINKIP_DISCOVERY_COMMON_DATA2_TYPE_ILWALID:
            LWSWITCH_PRINT(device, SETUP,
                "%s:_DATA2: %s=%6x\n",
                __FUNCTION__,
                "_ILWALID", data2_addr);
            break;

        default:
            LWSWITCH_PRINT(device, SETUP,
                "%s:_DATA2: Unknown!\n",
                __FUNCTION__);
            LWSWITCH_ASSERT(0);
            break;
    }
}

static void
_lwswitch_npg_parse_entry_sv10
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
_lwswitch_npg_parse_enum_sv10
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

    if (*entry_version != LW_NPG_DISCOVERY_ENUM_VERSION_1)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s:_NPG_DISCOVERY_ENUM_VERSION = %x but expected %x (_VERSION_1).\n",
            __FUNCTION__, *entry_version, LW_NPG_DISCOVERY_ENUM_VERSION_1);
    }
}

static void
_lwswitch_npg_handle_data1_sv10
(
    lwswitch_device *device,
    LwU32 entry,
    ENGINE_DESCRIPTOR_TYPE_SV10 *engine,
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

    engine->reset_bit  = DRF_VAL(_NPG, _DISCOVERY, _DATA1_RESET, entry);
    engine->intr_bit   = DRF_VAL(_NPG, _DISCOVERY, _DATA1_INTR, entry);
}

static void
_lwswitch_npg_handle_data2_sv10
(
    lwswitch_device *device,
    LwU32 entry,
    ENGINE_DESCRIPTOR_TYPE_SV10 *engine,
    LwU32 entry_device
)
{
    LwU32 data2_type = DRF_VAL(_NPG, _DISCOVERY_DATA2, _TYPE, entry);
    LwU32 data2_addr = DRF_VAL(_NPG, _DISCOVERY_DATA2, _ADDR, entry);

    switch(data2_type)
    {
        case LW_NPG_DISCOVERY_DATA2_TYPE_RESETREG:
            engine->reset_addr = data2_addr*sizeof(LwU32);
            break;

        case LW_NPG_DISCOVERY_DATA2_TYPE_INTRREG:
            engine->intr_addr = data2_addr*sizeof(LwU32);
            break;

        case LW_NPG_DISCOVERY_DATA2_TYPE_DISCOVERY:
            // Parse sub-discovery table
            engine->discovery = data2_addr*sizeof(LwU32);

            //
            // Lwrrently _DISCOVERY is not used in the second
            // level discovery.
            //
            LWSWITCH_ASSERT(0);

            break;

        case LW_NPG_DISCOVERY_DATA2_TYPE_UNICAST:
            engine->uc_addr = data2_addr*sizeof(LwU32);
            break;

        case LW_NPG_DISCOVERY_DATA2_TYPE_BROADCAST:
            engine->bc_addr = data2_addr*sizeof(LwU32);
            break;

        case LW_NPG_DISCOVERY_DATA2_TYPE_ILWALID:
            LWSWITCH_PRINT(device, SETUP,
                "%s:_DATA2: %s=%6x\n",
                __FUNCTION__,
                "_ILWALID", data2_addr);
            break;

        default:
            LWSWITCH_PRINT(device, SETUP,
                "%s:_DATA2: Unknown!\n",
                __FUNCTION__);
            LWSWITCH_ASSERT(0);
            break;
    }
}

static void
_lwswitch_swx_parse_entry_sv10
(
    lwswitch_device *device,
    LwU32   entry,
    LwU32   *entry_type,
    LwBool  *entry_chain
)
{
    LwU32 entry_type_npg;

    entry_type_npg = DRF_VAL(_SWX, _DISCOVERY, _ENTRY, entry);
    *entry_chain = FLD_TEST_DRF(_SWX, _DISCOVERY, _CHAIN, _ENABLE, entry);

    switch (entry_type_npg)
    {
        case LW_SWX_DISCOVERY_ENTRY_ENUM:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_ENUM;
            break;
        case LW_SWX_DISCOVERY_ENTRY_DATA1:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_DATA1;
            break;
        case LW_SWX_DISCOVERY_ENTRY_DATA2:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_DATA2;
            break;
        default:
            *entry_type = LWSWITCH_DISCOVERY_ENTRY_ILWALID;
            break;
    }
}

static void
_lwswitch_swx_parse_enum_sv10
(
    lwswitch_device *device,
    LwU32   entry,
    LwU32   *entry_device,
    LwU32   *entry_id,
    LwU32   *entry_version
)
{
    LwU32 entry_reserved;

    *entry_device  = DRF_VAL(_SWX, _DISCOVERY, _ENUM_DEVICE, entry);
    *entry_id      = DRF_VAL(_SWX, _DISCOVERY, _ENUM_ID, entry);
    *entry_version = DRF_VAL(_SWX, _DISCOVERY, _ENUM_VERSION, entry);

    entry_reserved = DRF_VAL(_SWX, _DISCOVERY, _ENUM_RESERVED, entry);
    LWSWITCH_ASSERT(entry_reserved == 0);

    if (*entry_version != LW_SWX_DISCOVERY_ENUM_VERSION_1)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s:_SWX_DISCOVERY_ENUM_VERSION = %x but expected %x (_VERSION_1).\n",
            __FUNCTION__, *entry_version, LW_SWX_DISCOVERY_ENUM_VERSION_1);
    }
}

static void
_lwswitch_swx_handle_data1_sv10
(
    lwswitch_device *device,
    LwU32 entry,
    ENGINE_DESCRIPTOR_TYPE_SV10 *engine,
    LwU32 entry_device,
    LwU32 *discovery_list_size
)
{
    if (LW_SWX_DISCOVERY_ENUM_DEVICE_SWX == entry_device)
    {
        *discovery_list_size = DRF_VAL(_SWX, _DISCOVERY, _DATA1_SWX_LENGTH, entry);
    }
    else
    {
        LWSWITCH_ASSERT(DRF_VAL(_SWX, _DISCOVERY, _DATA1_RESERVED, entry) == 0);
    }

    if (engine == NULL)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s:DATA1:engine == NULL.  Skipping processing\n",
            __FUNCTION__);
        return;
    }

    engine->reset_bit  = DRF_VAL(_SWX, _DISCOVERY, _DATA1_RESET, entry);
    engine->intr_bit   = DRF_VAL(_SWX, _DISCOVERY, _DATA1_INTR, entry);
}

static void
_lwswitch_swx_handle_data2_sv10
(
    lwswitch_device *device,
    LwU32 entry,
    ENGINE_DESCRIPTOR_TYPE_SV10 *engine,
    LwU32 entry_device
)
{
    LwU32 data2_type = DRF_VAL(_SWX, _DISCOVERY_DATA2, _TYPE, entry);
    LwU32 data2_addr = DRF_VAL(_SWX, _DISCOVERY_DATA2, _ADDR, entry);

    switch(data2_type)
    {
        case LW_SWX_DISCOVERY_DATA2_TYPE_RESETREG:
            engine->reset_addr = data2_addr*sizeof(LwU32);
            break;

        case LW_SWX_DISCOVERY_DATA2_TYPE_INTRREG:
            engine->intr_addr = data2_addr*sizeof(LwU32);
            break;

        case LW_SWX_DISCOVERY_DATA2_TYPE_DISCOVERY:
            // Parse sub-discovery table
            engine->discovery = data2_addr*sizeof(LwU32);

            //
            // Lwrrently _DISCOVERY is not used in the second
            // level discovery.
            //
            LWSWITCH_ASSERT(0);

            break;

        case LW_SWX_DISCOVERY_DATA2_TYPE_UNICAST:
            engine->uc_addr = data2_addr*sizeof(LwU32);
            break;

        case LW_SWX_DISCOVERY_DATA2_TYPE_BROADCAST:
            engine->bc_addr = data2_addr*sizeof(LwU32);
            break;

        case LW_SWX_DISCOVERY_DATA2_TYPE_ILWALID:
            LWSWITCH_PRINT(device, ERROR,
                "%s:_DATA2: %s=%6x\n",
                __FUNCTION__,
                "_ILWALID", data2_addr);
            break;

        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s:_DATA2: Unknown!\n",
                __FUNCTION__);
            LWSWITCH_ASSERT(0);
            break;
    }
}

#define MAKE_DISCOVERY_SV10_LWLINK(eng)             \
    {                                               \
        #eng,                                       \
        NUM_##eng##_INSTANCES_SV10,                 \
        NUM_##eng##_SUBENGINE_SV10,                 \
        LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_##eng,  \
        engine->subeng##eng,                        \
        &engine->num##eng                           \
    }

#define MAKE_DISCOVERY_SV10_NPG(eng)        \
    {                                       \
        #eng,                               \
        NUM_##eng##_INSTANCES_SV10,         \
        NUM_##eng##_SUBENGINE_SV10,         \
        LW_NPG_DISCOVERY_ENUM_DEVICE_##eng, \
        engine->subeng##eng,                \
        &engine->num##eng                   \
    }

#define MAKE_DISCOVERY_SV10_SWX(eng)        \
    {                                       \
        #eng,                               \
        NUM_##eng##_INSTANCES_SV10,         \
        NUM_##eng##_SUBENGINE_SV10,         \
        LW_SWX_DISCOVERY_ENUM_DEVICE_##eng, \
        engine->subeng##eng,                \
        &engine->num##eng                   \
    }

static
LWSWITCH_DISCOVERY_HANDLERS_SV10 discovery_handlers_ptop =
{
    &_lwswitch_ptop_parse_entry_sv10,
    &_lwswitch_ptop_parse_enum_sv10,
    &_lwswitch_ptop_handle_data1_sv10,
    &_lwswitch_ptop_handle_data2_sv10
};

static
LWSWITCH_DISCOVERY_HANDLERS_SV10 discovery_handlers_sioctrl =
{
    &_lwswitch_sioctrl_parse_entry_sv10,
    &_lwswitch_sioctrl_parse_enum_sv10,
    &_lwswitch_sioctrl_handle_data1_sv10,
    &_lwswitch_sioctrl_handle_data2_sv10
};

static
LWSWITCH_DISCOVERY_HANDLERS_SV10 discovery_handlers_npg =
{
    &_lwswitch_npg_parse_entry_sv10,
    &_lwswitch_npg_parse_enum_sv10,
    &_lwswitch_npg_handle_data1_sv10,
    &_lwswitch_npg_handle_data2_sv10
};

static
LWSWITCH_DISCOVERY_HANDLERS_SV10 discovery_handlers_swx =
{
    &_lwswitch_swx_parse_entry_sv10,
    &_lwswitch_swx_parse_enum_sv10,
    &_lwswitch_swx_handle_data1_sv10,
    &_lwswitch_swx_handle_data2_sv10
};

LwlStatus
lwswitch_device_discovery_sv10
(
    lwswitch_device *device,
    LwU32   discovery_offset
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    DISCOVERY_TABLE_TYPE_SV10 discovery_table_sv10[] =
    {
        MAKE_DISCOVERY_SV10(device, SV10, PTOP),
        MAKE_DISCOVERY_SV10(device, SV10, SIOCTRL),
        MAKE_DISCOVERY_SV10(device, SV10, SIOCTRL_BCAST),
        MAKE_DISCOVERY_SV10(device, SV10, NPG),
        MAKE_DISCOVERY_SV10(device, SV10, NPG_BCAST),
        MAKE_DISCOVERY_SV10(device, SV10, SWX),
        MAKE_DISCOVERY_SV10(device, SV10, SWX_BCAST),
        MAKE_DISCOVERY_SV10(device, SV10, CLKS),
        MAKE_DISCOVERY_SV10(device, SV10, FUSE),
        MAKE_DISCOVERY_SV10(device, SV10, JTAG),
        MAKE_DISCOVERY_SV10(device, SV10, PMGR),
        MAKE_DISCOVERY_SV10(device, SV10, SAW),
        MAKE_DISCOVERY_SV10(device, SV10, XP3G),
        MAKE_DISCOVERY_SV10(device, SV10, XVE),
        MAKE_DISCOVERY_SV10(device, SV10, ROM),
        MAKE_DISCOVERY_SV10(device, SV10, EXTDEV),
        MAKE_DISCOVERY_SV10(device, SV10, PRIVMAIN),
        MAKE_DISCOVERY_SV10(device, SV10, PRIVLOC)
    };
    LwU32 discovery_table_sv10_size = LW_ARRAY_ELEMENTS(discovery_table_sv10);

    LwU32 i;
    LwlStatus   status;

    CLEAR_ENGINE_SV10(device, SV10, PTOP);
    CLEAR_ENGINE_SV10(device, SV10, SIOCTRL);
    CLEAR_ENGINE_SV10(device, SV10, SIOCTRL_BCAST);
    CLEAR_ENGINE_SV10(device, SV10, NPG);
    CLEAR_ENGINE_SV10(device, SV10, NPG_BCAST);
    CLEAR_ENGINE_SV10(device, SV10, SWX);
    CLEAR_ENGINE_SV10(device, SV10, SWX_BCAST);
    CLEAR_ENGINE_SV10(device, SV10, CLKS);
    CLEAR_ENGINE_SV10(device, SV10, FUSE);
    CLEAR_ENGINE_SV10(device, SV10, JTAG);
    CLEAR_ENGINE_SV10(device, SV10, PMGR);
    CLEAR_ENGINE_SV10(device, SV10, SAW);
    CLEAR_ENGINE_SV10(device, SV10, XP3G);
    CLEAR_ENGINE_SV10(device, SV10, XVE);
    CLEAR_ENGINE_SV10(device, SV10, ROM);
    CLEAR_ENGINE_SV10(device, SV10, EXTDEV);
    CLEAR_ENGINE_SV10(device, SV10, PRIVMAIN);
    CLEAR_ENGINE_SV10(device, SV10, PRIVLOC);

    status = _lwswitch_device_discovery_sv10(
        device, discovery_offset, discovery_table_sv10, discovery_table_sv10_size,
        &discovery_handlers_ptop);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "MMIO discovery failed\n");
        return status;
    }

    if (VERBOSE_MMIO_DISCOVERY)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s:PTOP Discovery\n",
            __FUNCTION__);
        DISCOVERY_DUMP_ENGINE_SV10(device, PTOP);
        DISCOVERY_DUMP_ENGINE_SV10(device, SIOCTRL);
        DISCOVERY_DUMP_ENGINE_SV10(device, SIOCTRL_BCAST);
        DISCOVERY_DUMP_ENGINE_SV10(device, NPG);
        DISCOVERY_DUMP_ENGINE_SV10(device, NPG_BCAST);
        DISCOVERY_DUMP_ENGINE_SV10(device, SWX);
        DISCOVERY_DUMP_ENGINE_SV10(device, SWX_BCAST);
        DISCOVERY_DUMP_ENGINE_SV10(device, CLKS);
        DISCOVERY_DUMP_ENGINE_SV10(device, FUSE);
        DISCOVERY_DUMP_ENGINE_SV10(device, JTAG);
        DISCOVERY_DUMP_ENGINE_SV10(device, PMGR);
        DISCOVERY_DUMP_ENGINE_SV10(device, SAW);
        DISCOVERY_DUMP_ENGINE_SV10(device, XP3G);
        DISCOVERY_DUMP_ENGINE_SV10(device, XVE);
        DISCOVERY_DUMP_ENGINE_SV10(device, ROM);
        DISCOVERY_DUMP_ENGINE_SV10(device, EXTDEV);
        DISCOVERY_DUMP_ENGINE_SV10(device, PRIVMAIN);
        DISCOVERY_DUMP_ENGINE_SV10(device, PRIVLOC);
    }

    for (i = 0; i < NUM_SIOCTRL_ENGINE_SV10; i++)
    {
        if (chip_device->engSIOCTRL[i].valid && 
            (chip_device->engSIOCTRL[i].discovery != 0))
        {
            LWLINK_SUBENGINE_DESCRIPTOR_TYPE_SV10 *engine = &(chip_device->subengSIOCTRL[i]);
            DISCOVERY_TABLE_TYPE_SV10 discovery_table_sioctrl[] =
            {
                MAKE_DISCOVERY_SV10_LWLINK(MINION),
                MAKE_DISCOVERY_SV10_LWLINK(LWLIPT),
                MAKE_DISCOVERY_SV10_LWLINK(TX_PERFMON),
                MAKE_DISCOVERY_SV10_LWLINK(RX_PERFMON),
                MAKE_DISCOVERY_SV10_LWLINK(TX_PERFMON_MULTICAST),
                MAKE_DISCOVERY_SV10_LWLINK(RX_PERFMON_MULTICAST),
                MAKE_DISCOVERY_SV10_LWLINK(LWLTLC),
                MAKE_DISCOVERY_SV10_LWLINK(DLPL_MULTICAST),
                MAKE_DISCOVERY_SV10_LWLINK(LWLTLC_MULTICAST),
                MAKE_DISCOVERY_SV10_LWLINK(DLPL),
                MAKE_DISCOVERY_SV10_LWLINK(SIOCTRL),
                MAKE_DISCOVERY_SV10_LWLINK(SIOCTRL_PERFMON),
                MAKE_DISCOVERY_SV10_LWLINK(LWLIPT_SYS_PERFMON)
            };
            LwU32 discovery_table_sioctrl_size = LW_ARRAY_ELEMENTS(discovery_table_sioctrl);

            status = _lwswitch_device_discovery_sv10(
                device, chip_device->engSIOCTRL[i].discovery, discovery_table_sioctrl,
                discovery_table_sioctrl_size, &discovery_handlers_sioctrl);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "SIOCTRL[%d] discovery failed\n", i);
                return status;
            }
            if (VERBOSE_MMIO_DISCOVERY)
            {
                LWSWITCH_PRINT(device, SETUP,
                    "%s:SIOCTRL[%d] Discovery\n",
                    __FUNCTION__, i);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, MINION);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, LWLIPT);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, LWLTLC);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, DLPL_MULTICAST);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, LWLTLC_MULTICAST);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, DLPL);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, SIOCTRL);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, SIOCTRL_PERFMON);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, LWLIPT_SYS_PERFMON);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, TX_PERFMON_MULTICAST);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, RX_PERFMON_MULTICAST);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, TX_PERFMON);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, RX_PERFMON);

                LWSWITCH_PRINT(device, SETUP,
                    "%s: DLPL[0]: tx=%x rx=%x master=%d master_id=%d\n",
                    __FUNCTION__,
                    engine->dlpl_info[0].num_tx, engine->dlpl_info[0].num_rx,
                    engine->dlpl_info[0].master, engine->dlpl_info[0].master_id);
                LWSWITCH_PRINT(device, SETUP,
                    "%s: DLPL[1]: tx=%x rx=%x master=%d master_id=%d\n",
                    __FUNCTION__,
                    engine->dlpl_info[1].num_tx, engine->dlpl_info[1].num_rx,
                    engine->dlpl_info[1].master, engine->dlpl_info[1].master_id);
                LWSWITCH_PRINT(device, SETUP,
                    "%s: DLPL MC: tx=%x rx=%x master=%d master_id=%d\n",
                    __FUNCTION__,
                    engine->dlpl_info_multicast.num_tx, engine->dlpl_info_multicast.num_rx,
                    engine->dlpl_info_multicast.master, engine->dlpl_info_multicast.master_id);
            }
        }
    }

    for (i = 0; i < NUM_SIOCTRL_BCAST_ENGINE_SV10; i++)
    {
        if (chip_device->engSIOCTRL_BCAST[i].valid && 
            (chip_device->engSIOCTRL_BCAST[i].discovery != 0))
        {
            LWLINK_SUBENGINE_DESCRIPTOR_TYPE_SV10 *engine = &(chip_device->subengSIOCTRL_BCAST[i]);
            DISCOVERY_TABLE_TYPE_SV10 discovery_table_sioctrl[] =
            {
                MAKE_DISCOVERY_SV10_LWLINK(MINION),
                MAKE_DISCOVERY_SV10_LWLINK(LWLIPT),
                MAKE_DISCOVERY_SV10_LWLINK(LWLTLC),
                MAKE_DISCOVERY_SV10_LWLINK(DLPL_MULTICAST),
                MAKE_DISCOVERY_SV10_LWLINK(LWLTLC_MULTICAST),
                MAKE_DISCOVERY_SV10_LWLINK(DLPL),
                MAKE_DISCOVERY_SV10_LWLINK(SIOCTRL),
                MAKE_DISCOVERY_SV10_LWLINK(SIOCTRL_PERFMON),
                MAKE_DISCOVERY_SV10_LWLINK(LWLIPT_SYS_PERFMON),
                MAKE_DISCOVERY_SV10_LWLINK(TX_PERFMON_MULTICAST),
                MAKE_DISCOVERY_SV10_LWLINK(RX_PERFMON_MULTICAST),
                MAKE_DISCOVERY_SV10_LWLINK(TX_PERFMON),
                MAKE_DISCOVERY_SV10_LWLINK(RX_PERFMON)
            };
            LwU32 discovery_table_sioctrl_size = LW_ARRAY_ELEMENTS(discovery_table_sioctrl);

            status = _lwswitch_device_discovery_sv10(
                device, chip_device->engSIOCTRL_BCAST[i].discovery,
                discovery_table_sioctrl, discovery_table_sioctrl_size,
                &discovery_handlers_sioctrl);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "SIOCTRL_BCAST[%d] discovery failed\n", i);
                return status;
            }
            if (VERBOSE_MMIO_DISCOVERY)
            {
                LWSWITCH_PRINT(device, SETUP,
                    "%s:SIOCTRL_BCAST[%d] Discovery\n",
                    __FUNCTION__, i);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, MINION);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, LWLIPT);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, LWLTLC);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, DLPL_MULTICAST);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, LWLTLC_MULTICAST);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, DLPL);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, SIOCTRL);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, SIOCTRL_PERFMON);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, LWLIPT_SYS_PERFMON);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, TX_PERFMON_MULTICAST);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, RX_PERFMON_MULTICAST);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, TX_PERFMON);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, RX_PERFMON);

                LWSWITCH_PRINT(device, SETUP,
                    "%s: DLPL[0]: tx=%x rx=%x master=%d master_id=%d\n",
                    __FUNCTION__,
                    engine->dlpl_info[0].num_tx, engine->dlpl_info[0].num_rx,
                    engine->dlpl_info[0].master, engine->dlpl_info[0].master_id);
                LWSWITCH_PRINT(device, SETUP,
                    "%s: DLPL[1]: tx=%x rx=%x master=%d master_id=%d\n",
                    __FUNCTION__,
                    engine->dlpl_info[1].num_tx, engine->dlpl_info[1].num_rx,
                    engine->dlpl_info[1].master, engine->dlpl_info[1].master_id);
                LWSWITCH_PRINT(device, SETUP,
                    "%s: DLPL MC: tx=%x rx=%x master=%d master_id=%d\n",
                    __FUNCTION__,
                    engine->dlpl_info_multicast.num_tx, engine->dlpl_info_multicast.num_rx,
                    engine->dlpl_info_multicast.master, engine->dlpl_info_multicast.master_id);
            }
        }
    }

    for (i = 0; i < NUM_NPG_ENGINE_SV10; i++)
    {
        if (chip_device->engNPG[i].valid &&
            (chip_device->engNPG[i].discovery != 0))
        {
            NPG_SUBENGINE_DESCRIPTOR_TYPE_SV10 *engine = &(chip_device->subengNPG[i]);
            DISCOVERY_TABLE_TYPE_SV10 discovery_table_npg[] =
            {
                MAKE_DISCOVERY_SV10_NPG(NPG),
                MAKE_DISCOVERY_SV10_NPG(NPORT),
                MAKE_DISCOVERY_SV10_NPG(NPORT_MULTICAST),
                MAKE_DISCOVERY_SV10_NPG(NPG_PERFMON),
                MAKE_DISCOVERY_SV10_NPG(NPORT_PERFMON),
                MAKE_DISCOVERY_SV10_NPG(NPORT_PERFMON_MULTICAST)
            };
            LwU32 discovery_table_npg_size = LW_ARRAY_ELEMENTS(discovery_table_npg);

            status = _lwswitch_device_discovery_sv10(
                device, chip_device->engNPG[i].discovery, discovery_table_npg,
                discovery_table_npg_size, &discovery_handlers_npg);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "NPG[%d] discovery failed\n", i);
                return status;
            }

            if (VERBOSE_MMIO_DISCOVERY)
            {
                LWSWITCH_PRINT(device, SETUP,
                    "%s:NPG[%d] Discovery\n",
                    __FUNCTION__, i);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, NPG);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, NPORT);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, NPORT_MULTICAST);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, NPG_PERFMON);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, NPORT_PERFMON);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, NPORT_PERFMON_MULTICAST);
            }
        }
    }

    for (i = 0; i < NUM_NPG_BCAST_ENGINE_SV10; i++)
    {
        if (chip_device->engNPG_BCAST[i].valid &&
            (chip_device->engNPG_BCAST[i].discovery != 0))
        {
            NPG_SUBENGINE_DESCRIPTOR_TYPE_SV10 *engine = &(chip_device->subengNPG_BCAST[i]);
            DISCOVERY_TABLE_TYPE_SV10 discovery_table_npg[] =
            {
                MAKE_DISCOVERY_SV10_NPG(NPG),
                MAKE_DISCOVERY_SV10_NPG(NPORT),
                MAKE_DISCOVERY_SV10_NPG(NPORT_MULTICAST),
                MAKE_DISCOVERY_SV10_NPG(NPG_PERFMON),
                MAKE_DISCOVERY_SV10_NPG(NPORT_PERFMON),
                MAKE_DISCOVERY_SV10_NPG(NPORT_PERFMON_MULTICAST)
            };
            LwU32 discovery_table_npg_size = LW_ARRAY_ELEMENTS(discovery_table_npg);

            status = _lwswitch_device_discovery_sv10(
                device, chip_device->engNPG_BCAST[i].discovery, discovery_table_npg,
                discovery_table_npg_size, &discovery_handlers_npg);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "NPG_BCAST[%d] discovery failed\n", i);
                return status;
            }

            if (VERBOSE_MMIO_DISCOVERY)
            {
                LWSWITCH_PRINT(device, SETUP,
                    "%s:NPG_BCAST[%d] Discovery\n",
                    __FUNCTION__, i);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, NPG);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, NPORT);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, NPORT_MULTICAST);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, NPG_PERFMON);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, NPORT_PERFMON);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, NPORT_PERFMON_MULTICAST);
            }
        }
    }

    for (i = 0; i < NUM_SWX_ENGINE_SV10; i++)
    {
        if (chip_device->engSWX[i].valid &&
            (chip_device->engSWX[i].discovery != 0))
        {
            SWX_SUBENGINE_DESCRIPTOR_TYPE_SV10 *engine = &(chip_device->subengSWX[i]);
            DISCOVERY_TABLE_TYPE_SV10 discovery_table_swx[] =
            {
                MAKE_DISCOVERY_SV10_SWX(SWX),
                MAKE_DISCOVERY_SV10_SWX(AFS),
                MAKE_DISCOVERY_SV10_SWX(AFS_MULTICAST),
                MAKE_DISCOVERY_SV10_SWX(SWX_PERFMON),
                MAKE_DISCOVERY_SV10_SWX(AFS_PERFMON),
                MAKE_DISCOVERY_SV10_SWX(AFS_PERFMON_MULTICAST)
            };
            LwU32 discovery_table_swx_size = LW_ARRAY_ELEMENTS(discovery_table_swx);

            status = _lwswitch_device_discovery_sv10(device, chip_device->engSWX[i].discovery,
                discovery_table_swx, discovery_table_swx_size, &discovery_handlers_swx);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "SWX[%d] discovery failed\n", i);
                return status;
            }

            if (VERBOSE_MMIO_DISCOVERY)
            {
                LWSWITCH_PRINT(device, SETUP,
                    "%s:SWX[%d] Discovery\n",
                    __FUNCTION__, i);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, SWX);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, AFS);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, AFS_MULTICAST);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, SWX_PERFMON);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, AFS_PERFMON);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, AFS_PERFMON_MULTICAST);
            }
        }
    }

    for (i = 0; i < NUM_SWX_BCAST_ENGINE_SV10; i++)
    {
        if (chip_device->engSWX_BCAST[i].valid &&
            (chip_device->engSWX_BCAST[i].discovery != 0))
        {
            SWX_SUBENGINE_DESCRIPTOR_TYPE_SV10 *engine = &(chip_device->subengSWX_BCAST[i]);
            DISCOVERY_TABLE_TYPE_SV10 discovery_table_swx[] =
            {
                MAKE_DISCOVERY_SV10_SWX(SWX),
                MAKE_DISCOVERY_SV10_SWX(AFS),
                MAKE_DISCOVERY_SV10_SWX(AFS_MULTICAST),
                MAKE_DISCOVERY_SV10_SWX(SWX_PERFMON),
                MAKE_DISCOVERY_SV10_SWX(AFS_PERFMON),
                MAKE_DISCOVERY_SV10_SWX(AFS_PERFMON_MULTICAST)
            };
            LwU32 discovery_table_swx_size = LW_ARRAY_ELEMENTS(discovery_table_swx);

            status = _lwswitch_device_discovery_sv10(
                device, chip_device->engSWX_BCAST[i].discovery, discovery_table_swx,
                discovery_table_swx_size, &discovery_handlers_swx);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "SWX_BCAST[%d] discovery failed\n", i);
                return status;
            }

            if (VERBOSE_MMIO_DISCOVERY)
            {
                LWSWITCH_PRINT(device, SETUP,
                    "%s:SWX_BCAST[%d] Discovery\n",
                    __FUNCTION__, i);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, SWX);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, AFS);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, AFS_MULTICAST);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, SWX_PERFMON);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, AFS_PERFMON);
                DISCOVERY_DUMP_SUBENGINE_SV10(device, AFS_PERFMON_MULTICAST);
            }
        }
    }

    return status;
}

static void
_filter_discovery_ilwalidate_sioctrl
(
    lwswitch_device *device,
    LWLINK_SUBENGINE_DESCRIPTOR_TYPE_SV10 *engSIOCTRL
)
{
    LwU32 j;

    LWSWITCH_ASSERT(NULL != engSIOCTRL);

    engSIOCTRL->subengMINION[0].valid = LW_FALSE;
    engSIOCTRL->subengLWLIPT[0].valid = LW_FALSE;
    for (j=0; j<engSIOCTRL->numLWLTLC; j++)
        engSIOCTRL->subengLWLTLC[j].valid = LW_FALSE;
    engSIOCTRL->subengDLPL_MULTICAST[0].valid = LW_FALSE;
    engSIOCTRL->subengLWLTLC_MULTICAST[0].valid = LW_FALSE;
    for (j=0; j<engSIOCTRL->numDLPL; j++)
        engSIOCTRL->subengDLPL[j].valid = LW_FALSE;
    engSIOCTRL->subengSIOCTRL[0].valid = LW_FALSE;
    engSIOCTRL->subengSIOCTRL_PERFMON[0].valid = LW_FALSE;
    engSIOCTRL->subengLWLIPT_SYS_PERFMON[0].valid = LW_FALSE;
    engSIOCTRL->subengTX_PERFMON_MULTICAST[0].valid = LW_FALSE;
    engSIOCTRL->subengRX_PERFMON_MULTICAST[0].valid = LW_FALSE;
    for (j=0; j<engSIOCTRL->numTX_PERFMON; j++)
        engSIOCTRL->subengTX_PERFMON[j].valid = LW_FALSE;
    for (j=0; j<engSIOCTRL->numRX_PERFMON; j++)
        engSIOCTRL->subengRX_PERFMON[j].valid = LW_FALSE;
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
lwswitch_filter_discovery_sv10
(
    lwswitch_device *device
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    if (IS_RTLSIM(device))
    {
        LwU32   i;

        // No SIOCTRL are present
        for (i = 0; i < chip_device->numSIOCTRL; i++)
        {
            LWSWITCH_PRINT(device, SETUP,
                "%s: ilwalidate SIOCTRL[%d] and all its subengines\n",
                __FUNCTION__, i);
            chip_device->engSIOCTRL[i].valid = LW_FALSE;
            _filter_discovery_ilwalidate_sioctrl(device, &(chip_device->subengSIOCTRL[i]));
        }

        // No BCAST SIOCTRL are present
        for (i = 0; i < chip_device->numSIOCTRL_BCAST; i++)
        {
            LWSWITCH_PRINT(device, SETUP,
                "%s: ilwalidate BCAST SIOCTRL[%d] and all its subengines\n",
                __FUNCTION__, i);
            chip_device->engSIOCTRL_BCAST[i].valid = LW_FALSE;
            _filter_discovery_ilwalidate_sioctrl(device, &(chip_device->subengSIOCTRL_BCAST[i]));
        }
    }

    //
    // The SWX AFS interrupt bit mapping described in discovery tables doesn't
    // match reality.  The bits are swizzled as described in bug #200241882.
    //
    if (chip_device->overrides.WAR_Bug_200241882_AFS_interrupt_bits)
    {
        static const LwU32 afs_interrupt_bits[NUM_AFS_INSTANCES_SV10] = { 0, 2, 4, 6, 7, 8, 5, 3, 1 };
        LwU32 i, j;

        LWSWITCH_PRINT(device, SETUP,
            "%s: Applying SWX AFS interrupt bit swizzle\n",
            __FUNCTION__);
        for (i = 0; i < chip_device->numSWX; i++)
        {
            if (chip_device->engSWX[i].valid)
            {
                for (j = 0; j < chip_device->subengSWX[i].numAFS; j++)
                {
                    if (chip_device->subengSWX[i].subengAFS[j].valid)
                    {
                        LWSWITCH_ASSERT(chip_device->engSWX[i].instance < NUM_SWX_ENGINE_SV10);
                        LWSWITCH_ASSERT(chip_device->subengSWX[i].subengAFS[j].instance >=
                                chip_device->engSWX[i].instance * NUM_AFS_INSTANCES_SV10);
                        LWSWITCH_ASSERT(chip_device->subengSWX[i].subengAFS[j].instance <
                                (chip_device->engSWX[i].instance+1) * NUM_AFS_INSTANCES_SV10);
                        chip_device->subengSWX[i].subengAFS[j].intr_bit =
                            afs_interrupt_bits[chip_device->subengSWX[i].subengAFS[j].instance %
                            NUM_AFS_INSTANCES_SV10];
                    }
                }
            }
        }

        if (chip_device->engSWX_BCAST[0].valid)
        {
            for (j = 0; j < chip_device->subengSWX_BCAST[0].numAFS; j++)
            {
                if (chip_device->subengSWX_BCAST[0].subengAFS[j].valid)
                {
                    chip_device->subengSWX_BCAST[0].subengAFS[j].intr_bit =
                        afs_interrupt_bits[chip_device->subengSWX_BCAST[0].subengAFS[j].instance %
                        NUM_AFS_INSTANCES_SV10];
                }
            }
        }
    }
}

//
// Process engine discovery information to associate engines
//

LwlStatus
lwswitch_process_discovery_sv10
(
    lwswitch_device *device
)
{
    LwU32       i,j;
    LwlStatus   retval = LWL_SUCCESS;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    //
    // Cross reference the per-link engine information to collect per physical link info
    //
#define COLLECT_PER_LINK_SV10(_eng, _subeng)                                 \
    for (j=0; j < NUM_## _subeng ## _INSTANCES_SV10; j++)                    \
    {                                                                   \
        if (chip_device->subeng## _eng [i].subeng## _subeng [j].valid)  \
        {                                                               \
            LWSWITCH_ASSERT(chip_device->subeng## _eng [i].subeng## _subeng [j].instance < LWSWITCH_NUM_LINKS_SV10); \
            chip_device->link[chip_device->subeng## _eng [i].subeng## _subeng [j].instance].eng## _subeng   \
                = &chip_device->subeng## _eng [i].subeng## _subeng [j]; \
        }                                                               \
    }

    // SIOCTRL
    for (i=0; i < chip_device->numSIOCTRL; i++)
    {
        LWSWITCH_ASSERT(chip_device->engSIOCTRL[i].instance < NUM_SIOCTRL_ENGINE_SV10);

        chip_device->link[2*chip_device->engSIOCTRL[i].instance  ].valid      =
            chip_device->engSIOCTRL[i].valid;
        // Shared
        chip_device->link[2*chip_device->engSIOCTRL[i].instance  ].engSIOCTRL =
            &chip_device->subengSIOCTRL[i].subengSIOCTRL[0];
        chip_device->link[2*chip_device->engSIOCTRL[i].instance  ].engMINION  =
            &chip_device->subengSIOCTRL[i].subengMINION[0];
        chip_device->link[2*chip_device->engSIOCTRL[i].instance  ].engLWLIPT  =
            &chip_device->subengSIOCTRL[i].subengLWLIPT[0];

        chip_device->link[2*chip_device->engSIOCTRL[i].instance+1].valid      =
            chip_device->engSIOCTRL[i].valid;
        // Shared
        chip_device->link[2*chip_device->engSIOCTRL[i].instance+1].engSIOCTRL =
            &chip_device->subengSIOCTRL[i].subengSIOCTRL[0];
        chip_device->link[2*chip_device->engSIOCTRL[i].instance+1].engMINION  =
            &chip_device->subengSIOCTRL[i].subengMINION[0];
        chip_device->link[2*chip_device->engSIOCTRL[i].instance+1].engLWLIPT  =
            &chip_device->subengSIOCTRL[i].subengLWLIPT[0];

        COLLECT_PER_LINK_SV10(SIOCTRL, DLPL)
        COLLECT_PER_LINK_SV10(SIOCTRL, LWLTLC)
        COLLECT_PER_LINK_SV10(SIOCTRL, TX_PERFMON)
        COLLECT_PER_LINK_SV10(SIOCTRL, RX_PERFMON)
    }

    // NPG
    for (i=0; i < chip_device->numNPG; i++)
    {
        LWSWITCH_ASSERT(chip_device->engNPG[i].valid);
        LWSWITCH_ASSERT(chip_device->engNPG[i].instance < NUM_NPG_ENGINE_SV10);

        COLLECT_PER_LINK_SV10(NPG, NPORT)
        COLLECT_PER_LINK_SV10(NPG, NPORT_PERFMON)
    }

    //
    // Mark all the engines associated with disabled links as not valid
    //

    for (i=0; i < LWSWITCH_NUM_LINKS_SV10; i++)
    {
        if (chip_device->link[i].valid)
        {
            if ((LWBIT(i) & device->regkeys.link_enable_mask) == 0)
            {
                LWSWITCH_PRINT(device, SETUP,
                    "%s: Disable link #%d\n",
                    __FUNCTION__,
                    i);
                chip_device->link[i].valid                   = LW_FALSE;

                chip_device->link[i].engNPORT->valid         = LW_FALSE;
                chip_device->link[i].engNPORT_PERFMON->valid = LW_FALSE;
                chip_device->link[i].engDLPL->valid          = LW_FALSE;
                chip_device->link[i].engLWLTLC->valid        = LW_FALSE;
                chip_device->link[i].engTX_PERFMON->valid    = LW_FALSE;
                chip_device->link[i].engRX_PERFMON->valid    = LW_FALSE;
            }
        }
    }

    {
        LwU32 idx_sioctrl;
        LwU32 idx_dlpl;
        LwBool valid;

        // Mark SIOCTRLs invalid if all their links are disabled
        for (idx_sioctrl = 0; idx_sioctrl < NUM_SIOCTRL_ENGINE_SV10; idx_sioctrl++)
        {
            if (chip_device->engSIOCTRL[idx_sioctrl].valid &&
                chip_device->subengSIOCTRL[idx_sioctrl].subengSIOCTRL[0].valid)
            {
                valid = LW_FALSE;

                for (idx_dlpl = 0; idx_dlpl < NUM_DLPL_INSTANCES_SV10; idx_dlpl++)
                {
                    valid |= chip_device->subengSIOCTRL[idx_sioctrl].subengDLPL[idx_dlpl].valid;
                }

                if (!valid)
                {
                    chip_device->subengSIOCTRL[idx_sioctrl].subengSIOCTRL[0].valid = LW_FALSE;
                    chip_device->subengSIOCTRL[idx_sioctrl].subengSIOCTRL_PERFMON[0].valid = LW_FALSE;
                    chip_device->subengSIOCTRL[idx_sioctrl].subengMINION[0].valid = LW_FALSE;
                    chip_device->subengSIOCTRL[idx_sioctrl].subengLWLIPT[0].valid = LW_FALSE;
                    chip_device->subengSIOCTRL[idx_sioctrl].subengLWLTLC[0].valid = LW_FALSE;
                    chip_device->subengSIOCTRL[idx_sioctrl].subengDLPL_MULTICAST[0].valid = LW_FALSE;
                    chip_device->subengSIOCTRL[idx_sioctrl].subengLWLTLC_MULTICAST[0].valid = LW_FALSE;
                    chip_device->subengSIOCTRL[idx_sioctrl].subengTX_PERFMON[0].valid = LW_FALSE;
                    chip_device->subengSIOCTRL[idx_sioctrl].subengRX_PERFMON[0].valid = LW_FALSE;
                    chip_device->subengSIOCTRL[idx_sioctrl].subengTX_PERFMON_MULTICAST[0].valid = LW_FALSE;
                    chip_device->subengSIOCTRL[idx_sioctrl].subengRX_PERFMON_MULTICAST[0].valid = LW_FALSE;
                    chip_device->subengSIOCTRL[idx_sioctrl].subengLWLIPT_SYS_PERFMON[0].valid = LW_FALSE;
                }
            }
        }
    }

    {
        LwU32 idx_npg;
        LwU32 idx_nport;
        LwBool valid;

        // Mark NPGs invalid if all their NPORTs (links) are disabled
        for (idx_npg = 0; idx_npg < NUM_NPG_ENGINE_SV10; idx_npg++)
        {
            if (chip_device->engNPG[idx_npg].valid &&
                chip_device->subengNPG[idx_npg].subengNPG[0].valid)
            {
                valid = LW_FALSE;

                for (idx_nport = 0; idx_nport < NUM_NPORT_INSTANCES_SV10; idx_nport++)
                {
                    valid |= chip_device->subengNPG[idx_npg].subengNPORT[idx_nport].valid;
                }

                if (!valid)
                {
                    chip_device->subengNPG[idx_npg].subengNPG[0].valid = LW_FALSE;
                    chip_device->subengNPG[idx_npg].subengNPG_PERFMON[0].valid = LW_FALSE;
                    chip_device->subengNPG[idx_npg].subengNPORT_MULTICAST[0].valid = LW_FALSE;
                    chip_device->subengNPG[idx_npg].subengNPORT_PERFMON_MULTICAST[0].valid = LW_FALSE;
                }
            }
        }
    }

    return retval;
}
