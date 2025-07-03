/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include "common_lwswitch.h"
#include "ls10/ls10.h"
#include "ls10/multicast_ls10.h"

#include "lwswitch/ls10/dev_route_ip.h"

// Source: IAS Table 44. Laguna NXbar TileCol Port Mapping
static const LWSWITCH_COLUMN_PORT_OFFSET_LS10 lwswitch_portmap_ls10[LWSWITCH_NUM_LINKS_LS10] = {
    // ports 0 - 10
    { 0,  0 }, { 0,  1 }, { 0,  2 }, { 0,  3 },
    { 0,  4 }, { 0,  5 }, { 0,  6 }, { 0,  7 },
    { 0,  8 }, { 0,  9 }, { 0, 10 },
    // ports 11 - 16
    { 2,  0 }, { 2,  3 }, { 2,  4 }, { 2,  5 },
    { 2,  8 },
    //ports 16 - 26
    { 4, 10 }, { 4,  9 }, { 4,  8 }, { 4,  7 },
    { 4,  6 }, { 4,  5 }, { 4,  4 }, { 4,  3 },
    { 4,  2 }, { 4,  1 }, { 4,  0 },
    // ports 27 - 31
    { 2,  9 }, { 2,  7 }, { 2,  6 }, { 2,  2 },
    { 2,  1 },
    // ports 32 - 42
    { 1,  0 }, { 1,  1 }, { 1,  2 }, { 1,  3 },
    { 1,  4 }, { 1,  5 }, { 1,  6 }, { 1,  7 },
    { 1,  8 }, { 1,  9 }, { 1, 10 },
    // ports 43 - 47
    { 3,  0 }, { 3,  3 }, { 3,  4 }, { 3,  5 },
    { 3,  8 },
    // ports 48 - 58
    { 5, 10 }, { 5,  9 }, { 5,  8 }, { 5,  7 },
    { 5,  6 }, { 5,  5 }, { 5,  4 }, { 5,  3 },
    { 5,  2 }, { 5,  1 }, { 5,  0 },
    // ports 59 - 63
    { 3,  9 }, { 3,  7 }, { 3,  6 },  { 3,  2 },
    { 3,  1 }
};

static LwlStatus
_lwswitch_get_column_port_offset_ls10
(
    LwU32 port,
    LWSWITCH_COLUMN_PORT_OFFSET_LS10 *column_port_offset
)
{
    if (port >=  LWSWITCH_NUM_LINKS_LS10)
        return -LWL_BAD_ARGS;

    *column_port_offset = lwswitch_portmap_ls10[port];

    return LWL_SUCCESS;
}

#if defined(LWSWITCH_MC_DEBUG) || defined(LWSWITCH_MC_TRACE)
static void
_lwswitch_mc_print_directive
(
    lwswitch_device *device,
    LWSWITCH_TCP_DIRECTIVE_LS10 *mc_directive
)
{
    if (!mc_directive)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: null directive pointer\n", __FUNCTION__);
        return;
    }

    LWSWITCH_PRINT(device, INFO, "TCP:      %4d ", mc_directive->tcp);

    // pretty-print null ports
    if (mc_directive->tcpEPort == LWSWITCH_MC_NULL_PORT_LS10)
    {
        LWSWITCH_PRINT(device, INFO, "EPort:       X         OPort: %4d",
                        mc_directive->tcpOPort);
    }
    else if (mc_directive->tcpOPort == LWSWITCH_MC_NULL_PORT_LS10)
    {
         LWSWITCH_PRINT(device, INFO, "EPort:    %4d         OPort:    X",
                         mc_directive->tcpEPort);
    }
    else
    {
        LWSWITCH_PRINT(device, INFO, "EPort:    %4d         OPort: %4d",
                    mc_directive->tcpEPort,
                    mc_directive->tcpOPort);
    }

    LWSWITCH_PRINT(device, INFO, "EAltPath: %4d      OAltPath: %4d",
                    mc_directive->tcpEAltPath,
                    mc_directive->tcpOAltPath);
    LWSWITCH_PRINT(device, INFO, "EVCHop:   %4d      OVCHop:   %4d",
                    mc_directive->tcpEVCHop,
                    mc_directive->tcpOVCHop);
    LWSWITCH_PRINT(device, INFO, "portFlag: %4d continueRound: %4d lastRound: %4d ",
                    mc_directive->portFlag,
                    mc_directive->continueRound,
                    mc_directive->lastRound);
    LWSWITCH_PRINT(device, INFO, "\n");
}

static void
_lwswitch_mc_print_directives
(
    lwswitch_device *device,
    LWSWITCH_TCP_DIRECTIVE_LS10 *mcp_list,
    LwU32 entries_used,
    LwU8 *spray_group_ptrs,
    LwU32 num_spray_groups
)
{
    LwU32 i, spray_group_offset, round, spray_group_idx, lwr_entry_idx, entries_printed;
    LwBool spray_group_done = LW_FALSE;

    if (num_spray_groups == 0)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: No spray groups specified\n", __FUNCTION__);
        return;
    }

    if (num_spray_groups > LWSWITCH_MC_MAX_SPRAY_LS10)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Too many spray groups specified: %d\n",
                        __FUNCTION__, num_spray_groups);
        return;
    }

    if (entries_used > LWSWITCH_MC_TCP_LIST_SIZE_LS10)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Too many entries specified: %d\n",
                        __FUNCTION__, entries_used);
        return;
    }


    LWSWITCH_PRINT(device, INFO, "Total spray groups %d\n", num_spray_groups);

    entries_printed = 0;

    // loop through spray groups
    for (spray_group_idx = 0; spray_group_idx < num_spray_groups; spray_group_idx++)
    {
        spray_group_done = LW_FALSE;
        spray_group_offset = spray_group_ptrs[spray_group_idx];
        lwr_entry_idx = spray_group_offset;
        round = 0;

        LWSWITCH_PRINT(device, INFO, "Spray group %d offset %d\n", spray_group_idx,
                        spray_group_offset);

        while (!spray_group_done)
        {
            if (entries_printed >= LWSWITCH_MC_TCP_LIST_SIZE_LS10)
            {
                LWSWITCH_PRINT(device, ERROR, "%s: Overflow of mcplist. Entries printed: %d\n",
                                __FUNCTION__, entries_printed);
                return;
            }

            LWSWITCH_PRINT(device, INFO, "Round %d, First mc_plist Index %d round size %d\n",
                            round, lwr_entry_idx, mcp_list[lwr_entry_idx].roundSize);

            for (i = 0; i < mcp_list[lwr_entry_idx].roundSize; i++)
            {
                if ((i + lwr_entry_idx) > LWSWITCH_MC_TCP_LIST_SIZE_LS10)
                {
                    LWSWITCH_PRINT(device, ERROR, "%s: Overflow of mcplist. %d\n",
                    __FUNCTION__, i + lwr_entry_idx);
                }

                _lwswitch_mc_print_directive(device, &mcp_list[i + lwr_entry_idx]);
                entries_printed++;

                if (mcp_list[i + lwr_entry_idx].lastRound)
                {
                    LWSWITCH_PRINT(device, INFO, "Last round of spray group found at offset %d\n",
                        i + lwr_entry_idx);
                    spray_group_done = LW_TRUE;
                }
            }

            round++;
            lwr_entry_idx += i;
        }

    }
}
#endif

//
// Build column-port bitmap. Each 32-bit portmap in the array represents a column.
// Each bit set in the portmap represents the column-relative port offset.
//
static LwlStatus
_lwswitch_mc_build_cpb
(
    lwswitch_device *device,
    LwU32 num_ports,
    LwU32 *spray_group,
    LwU32 num_columns,
    LwU32 *cpb,
    LwU8 *vchop_array_sg,
    LwU8 vchop_map[LWSWITCH_MC_NUM_COLUMNS_LS10][LWSWITCH_MC_PORTS_PER_COLUMN_LS10]
)
{
    LwU32 i, ret;
    LWSWITCH_COLUMN_PORT_OFFSET_LS10 cpo;

    if ((spray_group == NULL) || (cpb == NULL) || (num_ports == 0) ||
        (num_ports > LWSWITCH_NUM_LINKS_LS10))
    {
        LWSWITCH_PRINT(device, ERROR, "%s: invalid arguments\n", __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    lwswitch_os_memset(cpb, 0, sizeof(*cpb) * num_columns);
    lwswitch_os_memset(vchop_map, 0, sizeof(LwU8) *
                        LWSWITCH_MC_NUM_COLUMNS_LS10 * LWSWITCH_MC_PORTS_PER_COLUMN_LS10);

    for (i = 0; i < num_ports; i++)
    {
        ret = _lwswitch_get_column_port_offset_ls10(spray_group[i], &cpo);
        if (ret != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR, "%s: error getting column-port offset\n", __FUNCTION__);
            return ret;
        }

        if (lwswitch_test_flags(cpb[cpo.column], LWBIT(cpo.port_offset)))
        {
            LWSWITCH_PRINT(device, ERROR, "%s: duplicate port specified: %d\n", __FUNCTION__,
                            spray_group[i]);
            return -LWL_BAD_ARGS;
        }

        lwswitch_set_flags(&cpb[cpo.column], LWBIT(cpo.port_offset));

        if (vchop_array_sg[i] > LWSWITCH_MC_VCHOP_FORCE1)
        {
            LWSWITCH_PRINT(device, ERROR, "%s: vchop value out of range: %d\n", __FUNCTION__,
                           vchop_array_sg[i]);
            return -LWL_BAD_ARGS;
        }


        vchop_map[cpo.column][cpo.port_offset] = vchop_array_sg[i];
    }

    return LWL_SUCCESS;
}

//
// Determine whether the given column/offset pair matches the given absolute
// primary_replica port number.
//
static LwBool
_is_primary_replica
(
    LwU32 col,
    LwU32 offset,
    LwU32 primary_replica
)
{
    LWSWITCH_COLUMN_PORT_OFFSET_LS10 cpo;

    if (primary_replica == LWSWITCH_MC_ILWALID)
        return LW_FALSE;

    if (_lwswitch_get_column_port_offset_ls10(primary_replica, &cpo) != LWL_SUCCESS)
        return LW_FALSE;

    if ((cpo.column == col) && (cpo.port_offset == offset))
        return LW_TRUE;

    return LW_FALSE;
}

//
// This function compacts the directive list and updates port_list_size
//
static LwlStatus
_lwswitch_mc_compact_portlist
(
    lwswitch_device *device,
    LWSWITCH_TCP_DIRECTIVE_LS10 *port_list,
    LwU32 *port_list_size
)
{
    LwU32 lwr_portlist_pos, new_portlist_pos;
    LWSWITCH_TCP_DIRECTIVE_LS10 *lwr_dir, *old_list;

    if (port_list_size == NULL)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: port list size ptr is null\n", __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    if ((port_list == NULL) || (*port_list_size == 0))
        return LWL_SUCCESS;

    if ((*port_list_size) > LWSWITCH_MC_TCP_LIST_SIZE_LS10)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: port list size out of range\n", __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

#ifdef LWSWITCH_MC_DEBUG
    LWSWITCH_PRINT(device, INFO, "%s: old size: %d\n", __FUNCTION__, *port_list_size);
#endif

    // create temporary directive list
    old_list = lwswitch_os_malloc(sizeof(LWSWITCH_TCP_DIRECTIVE_LS10) * (*port_list_size));

    if (!old_list)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: error allocating temporary portlist\n", __FUNCTION__);
        return -LWL_NO_MEM;
    }

    lwswitch_os_memcpy(old_list, port_list, sizeof(LWSWITCH_TCP_DIRECTIVE_LS10) * (*port_list_size));

    // rebuild list using only valid entries
    new_portlist_pos = 0;

    for (lwr_portlist_pos = 0; lwr_portlist_pos < (*port_list_size); lwr_portlist_pos++)
    {
        lwr_dir = &old_list[lwr_portlist_pos];

        if (lwr_dir->tcp != LWSWITCH_MC_ILWALID)
        {
#ifdef LWSWITCH_MC_TRACE
            LWSWITCH_PRINT(device, INFO, "%s: valid directive:\n", __FUNCTION__);
            _lwswitch_mc_print_directive(device, &old_list[lwr_portlist_pos]);
#endif
            lwswitch_os_memcpy(&port_list[new_portlist_pos], &old_list[lwr_portlist_pos],
                    sizeof(LWSWITCH_TCP_DIRECTIVE_LS10));
            new_portlist_pos++;
        }
    }

    lwswitch_os_free(old_list);

#ifdef LWSWITCH_MC_DEBUG
    LWSWITCH_PRINT(device, INFO, "%s: new size:  %d\n", __FUNCTION__, new_portlist_pos);
#endif

    *port_list_size = new_portlist_pos;

    return LWL_SUCCESS;
}

//
// Set the round flags to indicate the size of each multicast round.
// See IAS section "6.12. Consistent MC Semantics" for more info.
//
static void
_lwswitch_mc_set_round_flags
(
    LWSWITCH_TCP_DIRECTIVE_LS10 *port_list,
    LwU32 port_list_size
)
{
    LwU32 lwr_portlist_pos, round_size, round_start, round_end;
    LWSWITCH_TCP_DIRECTIVE_LS10 *lwr_dir, *next_dir;

    if ((port_list == NULL) || (port_list_size == 0))
        return;

    round_start = 0;
    round_end = 0;

    for (lwr_portlist_pos = 0; lwr_portlist_pos < port_list_size; lwr_portlist_pos++)
    {
        lwr_dir = &port_list[lwr_portlist_pos];

        // special case: last element: end of round and last round
        if (lwr_portlist_pos == port_list_size - 1)
        {
            lwr_dir->continueRound = LW_FALSE;
            lwr_dir->lastRound = LW_TRUE;

            round_end = lwr_portlist_pos;
            round_size = round_end - round_start + 1;

            // set the round size in the first directive
            lwr_dir = &port_list[round_start];
            lwr_dir->roundSize = (LwU8)round_size;
        }
        else
        {
            // if next tcp is less than or equal to the current, then current is end of round
            next_dir = &port_list[lwr_portlist_pos + 1];
            if (next_dir->tcp <= lwr_dir->tcp)
            {
                lwr_dir->continueRound = LW_FALSE;

                round_end = lwr_portlist_pos;
                round_size = round_end - round_start + 1;

                // set the round size in the first directive
                lwr_dir = &port_list[round_start];
                lwr_dir->roundSize = (LwU8)round_size;

                // advance round_start
                round_start = lwr_portlist_pos + 1;
            }
        }
    }
}

//
// Set the port flags to indicate primary replica port location.
// See IAS section "6.12. Consistent MC Semantics" for more info.
//
static void
_lwswitch_mc_set_port_flags
(
        LWSWITCH_TCP_DIRECTIVE_LS10 *port_list,
        LwU32 port_list_size
)
{
    LwU32 lwr_portlist_pos;
    LWSWITCH_TCP_DIRECTIVE_LS10 *lwr_dir, *next_dir;

    if ((port_list == NULL) || (port_list_size == 0))
        return;

    for (lwr_portlist_pos = 0; lwr_portlist_pos < port_list_size; lwr_portlist_pos++)
    {
        lwr_dir = &port_list[lwr_portlist_pos];

        if (lwr_dir->primaryReplica != PRIMARY_REPLICA_NONE)
        {
            if (lwr_dir->lastRound)
            {
                lwr_dir->continueRound = LW_TRUE;

                if (lwr_dir->primaryReplica == PRIMARY_REPLICA_EVEN)
                    lwr_dir->portFlag = 0;
                if (lwr_dir->primaryReplica == PRIMARY_REPLICA_ODD)
                    lwr_dir->portFlag = 1;
            }
            else
            {
                // primary replica is in this directive, next directive specifies even or odd
                lwr_dir->portFlag = 1;

                if (lwr_portlist_pos + 1 >= port_list_size)
                {
                    LWSWITCH_ASSERT(0);
                    return;
                }

                next_dir = &port_list[lwr_portlist_pos + 1];

                if (lwr_dir->primaryReplica == PRIMARY_REPLICA_EVEN)
                    next_dir->portFlag = 0;
                if (lwr_dir->primaryReplica == PRIMARY_REPLICA_ODD)
                    next_dir->portFlag = 1;
            }
        }
    }
}

//
// This function "pops" the next port offset from the portlist bitmap.
//
static LW_INLINE LwU8
_lwswitch_mc_get_next_port
(
    LwU32 *portmap
)
{
    LwU32 port;

    if (!portmap)
    {
        LWSWITCH_ASSERT(0);
        return LWSWITCH_MC_NULL_PORT_LS10;
    }

    //
    // We have to do some gymnastics here because LOWESTBITIDX_32 is
    // destructive on the input variable, and the result is not assignable.
    //
    port = *portmap;
    LOWESTBITIDX_32(port);
    lwswitch_clear_flags(portmap, LWBIT(port));

    if (port >= LWSWITCH_MC_PORTS_PER_COLUMN_LS10)
    {
        LWSWITCH_ASSERT(0);
        return LWSWITCH_MC_NULL_PORT_LS10;
    }

    return (LwU8)port;
}

//
// This helper function generates a map of directive list offsets indexed by tile/column pair
// port offsets. This is used during construction of the directive list to point to where each
// newly constructed directive will be placed in the list. This process has to account for the
// fact that the middle two columns contain 10 ports each, while the rest have 11, all mapping
// into a 32-entry directive list.
//
static LW_INLINE void
_lwswitch_mc_build_mcplist_position_map
(
    LwU32 port_offsets_by_tcp[LWSWITCH_MC_NUM_COLUMN_PAIRS_LS10][LWSWITCH_MC_PORTS_PER_COLUMN_LS10]
)
{
    LwU32 i, j, tcp;

    if (!port_offsets_by_tcp)
    {
        LWSWITCH_ASSERT(0);
        return;
    }

    for (tcp = 0; tcp < LWSWITCH_MC_NUM_COLUMN_PAIRS_LS10; tcp++)
    {
        if (tcp == 0)
        {
            j = 0;
            for (i = 0; i < LWSWITCH_MC_PORTS_PER_COLUMN_LS10; i++)
            {
                port_offsets_by_tcp[tcp][i] = j;
                j += LWSWITCH_MC_NUM_COLUMN_PAIRS_LS10;
            }
        }

        if (tcp == 1)
        {
            j = 1;
            for (i = 0; i < LWSWITCH_MC_PORTS_PER_COLUMN_LS10 - 1; i++)
            {
                port_offsets_by_tcp[tcp][i] = j;
                j += LWSWITCH_MC_NUM_COLUMN_PAIRS_LS10;
            }
        }

        if (tcp == 2)
        {
            j = 2;
            for (i = 0; i < LWSWITCH_MC_PORTS_PER_COLUMN_LS10; i++)
            {
                port_offsets_by_tcp[tcp][i] = (j == LWSWITCH_MC_TCP_LIST_SIZE_LS10) ?
                                                    (LWSWITCH_MC_TCP_LIST_SIZE_LS10 - 1) : j;
                j += LWSWITCH_MC_NUM_COLUMN_PAIRS_LS10;
            }
        }
    }
}

//
// Wrapper for the NUMSETBITS_32 macro, which is destructive on input.
//
static LW_INLINE LwU32
_lwswitch_mc_get_pop_count
(
    LwU32 i
)
{
    LwU32 tmp = i;

    NUMSETBITS_32(tmp);

    return tmp;
}

//
// Build a list of TCP directives. This is the main colwersion function which is used to build a
// TCP directive list for each spray group from a given column/port bitmap.
//
// @param device                [in]  pointer to the lwswitch device struct
// @param cpb                   [in]  pointer to the column/port bitmap used to build directive list
// @param primary_replica       [in]  the primary replica port for this spray group, if specified
// @param vchop_map             [in]  array containing per-port vchop values in column/port format
// @param port_list             [out] array where the newly built directive list is written
// @param entries_used          [out] pointer to an int where the size of resulting list is written
//
static LwlStatus
_lwswitch_mc_build_portlist
(
    lwswitch_device *device,
    LwU32 *cpb,
    LwU32 primary_replica,
    LwU8 vchop_map[LWSWITCH_MC_NUM_COLUMNS_LS10][LWSWITCH_MC_PORTS_PER_COLUMN_LS10],
    LWSWITCH_TCP_DIRECTIVE_LS10 *port_list,
    LwU32 *entries_used
)
{
    LwU32 ecol_idx, ocol_idx, ecol_portcount, ocol_portcount, ecol_portmap, ocol_portmap;
    LwU32 lwr_portlist_pos, j, lwr_portlist_slot, last_portlist_pos;
    LwU8 lwr_eport, lwr_oport, i;
    LwS32 extra_ports;
    LwU32 port_offsets_by_tcp[LWSWITCH_MC_NUM_COLUMN_PAIRS_LS10][LWSWITCH_MC_PORTS_PER_COLUMN_LS10];
    LWSWITCH_TCP_DIRECTIVE_LS10 *lwr_dir;

    if ((cpb == NULL) || (port_list == NULL))
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Invalid arguments\n", __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    _lwswitch_mc_build_mcplist_position_map(port_offsets_by_tcp);

    //
    // process columns pairwise. if one column is larger than the other by 2 or more entries,
    // set the port as alt path
    //

    lwr_portlist_pos = 0;
    last_portlist_pos = 0;
    lwr_portlist_slot = 0;

    for ( i = 0; i < LWSWITCH_MC_NUM_COLUMN_PAIRS_LS10;  i++ )
    {
        ecol_idx = 2 * i;
        ocol_idx = 2 * i + 1;

        ecol_portmap = cpb[ecol_idx];
        ocol_portmap = cpb[ocol_idx];

        ecol_portcount = _lwswitch_mc_get_pop_count(ecol_portmap);
        ocol_portcount = _lwswitch_mc_get_pop_count(ocol_portmap);

        extra_ports = ecol_portcount - ocol_portcount;

        // Start current portlist position on column offset of the current column
        lwr_portlist_slot = 0;
        lwr_portlist_pos = port_offsets_by_tcp[i][lwr_portlist_slot];

        if ( extra_ports >= 0 )
        {
            //
            // even column has more ports or both columns have an equal number
            // iterate on odd column port count to go through both columns
            //
            for (j = 0; j < ocol_portcount; j++, lwr_portlist_slot++)
            {
                lwr_eport = _lwswitch_mc_get_next_port(&ecol_portmap);
                lwr_oport = _lwswitch_mc_get_next_port(&ocol_portmap);
                if ((lwr_eport == LWSWITCH_MC_NULL_PORT_LS10) ||
                    (lwr_oport == LWSWITCH_MC_NULL_PORT_LS10))
                {
                    return -LWL_ERR_GENERIC;
                }

                // assign the ports to the current directive
                lwr_portlist_pos = port_offsets_by_tcp[i][lwr_portlist_slot];
                lwr_dir = &port_list[lwr_portlist_pos];
                lwr_dir->tcpEPort = lwr_eport;
                lwr_dir->tcpOPort = lwr_oport;

                lwr_dir->tcpEVCHop = vchop_map[ecol_idx][lwr_eport];
                lwr_dir->tcpOVCHop = vchop_map[ocol_idx][lwr_oport];

                lwr_dir->tcp = i;

#ifdef LWSWITCH_MC_TRACE
                LWSWITCH_PRINT(device, INFO, "%s: tcp: %d, extra: %d, lwr_eport: %d, lwr_oport %d\n",
                                __FUNCTION__, i, extra_ports, lwr_eport, lwr_oport);
                LWSWITCH_PRINT(device, INFO, "%s: lwr_portlist_pos: %d\n", __FUNCTION__,
                                lwr_portlist_pos);
#endif
                // set primary replica
                if (_is_primary_replica(ocol_idx, lwr_oport, primary_replica))
                    lwr_dir->primaryReplica = PRIMARY_REPLICA_ODD;

                if (_is_primary_replica(ecol_idx, lwr_eport, primary_replica))
                    lwr_dir->primaryReplica = PRIMARY_REPLICA_EVEN;

            }

            // if both columns had the same number of ports, move on to the next column pair
            if (!extra_ports)
            {
                last_portlist_pos = LW_MAX(last_portlist_pos, lwr_portlist_pos);
                continue;
            }

            //
            // otherwise, handle remaining ports in even column
            // for the first extra port, assign it directly
            // lwr_portlist_slot is incremented by the last iteration, or 0
            //
            lwr_eport = _lwswitch_mc_get_next_port(&ecol_portmap);
            if (lwr_eport == LWSWITCH_MC_NULL_PORT_LS10)
            {
                return -LWL_ERR_GENERIC;
            }

            lwr_portlist_pos = port_offsets_by_tcp[i][lwr_portlist_slot];
            lwr_dir = &port_list[lwr_portlist_pos];
            lwr_dir->tcpEPort = lwr_eport;

            lwr_dir->tcpEVCHop = vchop_map[ecol_idx][lwr_eport];

            lwr_dir->tcp = i;

#ifdef LWSWITCH_MC_TRACE
            LWSWITCH_PRINT(device, INFO, "%s: tcp: %d, extra: %d, lwr_eport: %d\n",
                            __FUNCTION__, i, extra_ports, lwr_eport);
            LWSWITCH_PRINT(device, INFO, "%s: lwr_portlist_pos: %d\n", __FUNCTION__,
                            lwr_portlist_pos);
#endif

            // if this is the primary replica port, mark it
            if (_is_primary_replica(ecol_idx, lwr_eport, primary_replica))
                lwr_dir->primaryReplica = PRIMARY_REPLICA_EVEN;

            extra_ports--;

            // if there are more, assign to altpath
            while (extra_ports)
            {
                // get next port from even column
                lwr_eport = _lwswitch_mc_get_next_port(&ecol_portmap);
                if (lwr_eport == LWSWITCH_MC_NULL_PORT_LS10)
                {
                    return -LWL_ERR_GENERIC;
                }

                // assign it to odd port in current directive (altpath)
                lwr_dir->tcpOPort = lwr_eport;
                lwr_dir->tcpOAltPath = LW_TRUE;

                lwr_dir->tcpOVCHop = vchop_map[ecol_idx][lwr_eport];

#ifdef LWSWITCH_MC_TRACE
                LWSWITCH_PRINT(device, INFO, "%s: tcp: %d, extra: %d, lwr_eport: %d (alt)\n",
                                __FUNCTION__, i, extra_ports, lwr_eport);
                LWSWITCH_PRINT(device, INFO, "%s: lwr_portlist_pos: %d\n", __FUNCTION__,
                                lwr_portlist_pos);
#endif
                // if this is the primary replica port, mark it
                if (_is_primary_replica(ecol_idx, lwr_eport, primary_replica))
                    lwr_dir->primaryReplica = PRIMARY_REPLICA_ODD;

                extra_ports--;

                // if there are more ports remaining, start the next entry
                if (extra_ports)
                {
                    // advance the portlist entry
                    lwr_portlist_slot++;
                    lwr_portlist_pos = port_offsets_by_tcp[i][lwr_portlist_slot];
                    lwr_dir = &port_list[lwr_portlist_pos];

                    lwr_eport = _lwswitch_mc_get_next_port(&ecol_portmap);
                    if (lwr_eport == LWSWITCH_MC_NULL_PORT_LS10)
                    {
                        return -LWL_ERR_GENERIC;
                    }

                    lwr_dir->tcpEPort = lwr_eport;

                    lwr_dir->tcpEVCHop = vchop_map[ecol_idx][lwr_eport];

                    lwr_dir->tcp = i;

#ifdef LWSWITCH_MC_TRACE
                    LWSWITCH_PRINT(device, INFO, "%s: tcp: %d, extra: %d, lwr_eport: %d\n",
                                    __FUNCTION__, i, extra_ports, lwr_eport);
                    LWSWITCH_PRINT(device, INFO, "%s: lwr_portlist_pos: %d\n", __FUNCTION__,
                                    lwr_portlist_pos);
#endif

                    // if this is the primary replica port, mark it
                    if (_is_primary_replica(ecol_idx, lwr_eport, primary_replica))
                        lwr_dir->primaryReplica = PRIMARY_REPLICA_EVEN;

                    extra_ports--;
                }
            }
        }
        else
        {
            // odd column has more ports
            extra_ports = -extra_ports;

            // iterate over even column to go through port pairs
            for (j = 0; j < ecol_portcount; j++, lwr_portlist_slot++)
            {
                lwr_eport = _lwswitch_mc_get_next_port(&ecol_portmap);
                lwr_oport = _lwswitch_mc_get_next_port(&ocol_portmap);
                if ((lwr_eport == LWSWITCH_MC_NULL_PORT_LS10) ||
                    (lwr_oport == LWSWITCH_MC_NULL_PORT_LS10))
                {
                    return -LWL_ERR_GENERIC;
                }

                // assign the ports to the current directive
                lwr_portlist_pos = port_offsets_by_tcp[i][lwr_portlist_slot];
                lwr_dir = &port_list[lwr_portlist_pos];
                lwr_dir->tcpEPort = lwr_eport;
                lwr_dir->tcpOPort = lwr_oport;

                lwr_dir->tcpEVCHop = vchop_map[ecol_idx][lwr_eport];
                lwr_dir->tcpOVCHop = vchop_map[ocol_idx][lwr_oport];

                lwr_dir->tcp = i;

#ifdef LWSWITCH_MC_TRACE
                LWSWITCH_PRINT(device, INFO, "%s: tcp: %d, extra: %d, lwr_eport: %d, lwr_oport %d\n",
                                __FUNCTION__, i, extra_ports, lwr_eport, lwr_oport);
                LWSWITCH_PRINT(device, INFO, "%s: lwr_portlist_pos: %d\n", __FUNCTION__,
                                lwr_portlist_pos);
#endif
                if (_is_primary_replica(ocol_idx, lwr_oport, primary_replica))
                    lwr_dir->primaryReplica = PRIMARY_REPLICA_EVEN;
                if (_is_primary_replica(ecol_idx, lwr_eport, primary_replica))
                    lwr_dir->primaryReplica = PRIMARY_REPLICA_ODD;

            }

            // handle the leftover ports in odd column
            lwr_oport = _lwswitch_mc_get_next_port(&ocol_portmap);
            if (lwr_oport == LWSWITCH_MC_NULL_PORT_LS10)
            {
                return -LWL_ERR_GENERIC;
            }

            // lwr_portlist_slot is incremented by the last iteration, or 0
            lwr_portlist_pos = port_offsets_by_tcp[i][lwr_portlist_slot];
            lwr_dir = &port_list[lwr_portlist_pos];

            lwr_dir->tcpOPort = lwr_oport;

            lwr_dir->tcpOVCHop = vchop_map[ocol_idx][lwr_oport];

            lwr_dir->tcp = i;

#ifdef LWSWITCH_MC_TRACE
             LWSWITCH_PRINT(device, INFO, "%s: tcp: %d, extra: %d, lwr_oport %d\n",
                            __FUNCTION__, i, extra_ports, lwr_oport);
             LWSWITCH_PRINT(device, INFO, "%s: lwr_portlist_pos: %d\n", __FUNCTION__,
                            lwr_portlist_pos);
#endif

            if (_is_primary_replica(ocol_idx, lwr_oport, primary_replica))
                lwr_dir->primaryReplica = PRIMARY_REPLICA_ODD;

            extra_ports--;

            // process any remaining ports in odd column
            while (extra_ports)
            {
                // get next odd port
                lwr_oport = _lwswitch_mc_get_next_port(&ocol_portmap);
                if (lwr_oport == LWSWITCH_MC_NULL_PORT_LS10)
                {
                    return -LWL_ERR_GENERIC;
                }

                // set it as even altpath port in current directive
                lwr_dir->tcpEPort = lwr_oport;
                lwr_dir->tcpEAltPath = LW_TRUE;

                lwr_dir->tcpEVCHop = vchop_map[ocol_idx][lwr_oport];

#ifdef LWSWITCH_MC_TRACE
                LWSWITCH_PRINT(device, INFO, "%s: tcp: %d, extra: %d, lwr_oport %d (alt)\n",
                                __FUNCTION__, i, extra_ports, lwr_oport);
                LWSWITCH_PRINT(device, INFO, "%s: lwr_portlist_pos: %d\n", __FUNCTION__,
                                lwr_portlist_pos);
#endif

                if (_is_primary_replica(ocol_idx, lwr_oport, primary_replica))
                    lwr_dir->primaryReplica = PRIMARY_REPLICA_EVEN;

                extra_ports--;

                // if there is another port, it goes in the next directive
                if (extra_ports)
                {
                    lwr_portlist_slot++;
                    lwr_portlist_pos = port_offsets_by_tcp[i][lwr_portlist_slot];
                    lwr_dir = &port_list[lwr_portlist_pos];

                    lwr_oport = _lwswitch_mc_get_next_port(&ocol_portmap);
                    if (lwr_oport == LWSWITCH_MC_NULL_PORT_LS10)
                    {
                        return -LWL_ERR_GENERIC;
                    }

                    lwr_dir->tcpOPort = lwr_oport;

                    lwr_dir->tcpOVCHop = vchop_map[ocol_idx][lwr_oport];

                    lwr_dir->tcp = i;

#ifdef LWSWITCH_MC_TRACE
                    LWSWITCH_PRINT(device, INFO, "%s: tcp: %d, extra: %d, lwr_oport %d\n",
                                    __FUNCTION__, i, extra_ports, lwr_oport);
                    LWSWITCH_PRINT(device, INFO, "%s: lwr_portlist_pos: %d\n", __FUNCTION__,
                                    lwr_portlist_pos);
#endif

                    if (_is_primary_replica(ocol_idx, lwr_oport, primary_replica))
                        lwr_dir->primaryReplica = PRIMARY_REPLICA_EVEN;

                    extra_ports--;
                }
            }
        }

        last_portlist_pos = LW_MAX(last_portlist_pos, lwr_portlist_pos);
    }

    // set the lastRound flag for the last entry in the spray string
    lwr_dir = &port_list[last_portlist_pos];
    lwr_dir->lastRound = LW_TRUE;

    *entries_used = last_portlist_pos + 1;

#ifdef LWSWITCH_MC_DEBUG
    LWSWITCH_PRINT(device, INFO,
                    "%s: entries_used: %d, lwr_portlist_pos: %d last_portlist_pos: %d\n",
                    __FUNCTION__, *entries_used, lwr_portlist_pos, last_portlist_pos);
#endif

    return LWL_SUCCESS;
}

//
// Helper that initializes a given directive list to some base values.
//
static LW_INLINE LwlStatus
lwswitch_init_portlist_ls10
(
    lwswitch_device *device,
    LWSWITCH_TCP_DIRECTIVE_LS10 *mcp_list,
    LwU32 mcp_list_size
)
{
    LwU32 i;

    if (mcp_list_size > LWSWITCH_MC_TCP_LIST_SIZE_LS10)
    {
         LWSWITCH_PRINT(device, ERROR, "%s: mcp_list_size out of range (%d)\n",
                        __FUNCTION__, mcp_list_size);
        return -LWL_BAD_ARGS;
    }

    lwswitch_os_memset(mcp_list, 0,
                        sizeof(LWSWITCH_TCP_DIRECTIVE_LS10) * mcp_list_size);

    //
    // initialize port list with invalid values
    // continueRound will be fixed up when processing round flags
    //
    for ( i = 0; i < mcp_list_size; i ++ )
    {
        mcp_list[i].tcp = LWSWITCH_MC_ILWALID;
        mcp_list[i].continueRound = LW_TRUE;
        mcp_list[i].tcpEPort = LWSWITCH_MC_NULL_PORT_LS10;
        mcp_list[i].tcpOPort = LWSWITCH_MC_NULL_PORT_LS10;
    }

    return LWL_SUCCESS;
}


//
// Helper to traverse list of directives given in src and copy only valid entries to dst starting
// at dst_list_offset.
//
// This is used when building the final directive list from individual per-spray-group lists,
// ensuring that no invalid entries sneak in, as well as checking for a nontrivial corner case
// where a configuration of input spray groups can result in a directive list larger than the
// 32-entry space allowed in the table. This returns -LWL_MORE_PROCESSING_REQUIRED which is
// then propagated to the caller to adjust the input parameters and try again.
//
static LW_INLINE LwlStatus
_lwswitch_mc_copy_valid_entries_ls10
(
    lwswitch_device *device,
    LWSWITCH_TCP_DIRECTIVE_LS10 *dst,
    LWSWITCH_TCP_DIRECTIVE_LS10 *src,
    LwU32 num_valid_entries,
    LwU32 dst_list_offset
)
{
    LwU32 i;

    if (num_valid_entries + dst_list_offset > LWSWITCH_MC_TCP_LIST_SIZE_LS10)
    {
        LWSWITCH_PRINT(device, ERROR,
                        "%s: Overflow of mcplist. num_valid_entries: %d, dst_list_offset: %d\n",
                        __FUNCTION__, num_valid_entries, dst_list_offset);
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    for (i = 0; i < num_valid_entries; i++)
    {
        if (src[i].tcp == LWSWITCH_MC_ILWALID)
        {
            LWSWITCH_PRINT(device, ERROR, "%s: invalid entry at offset %d\n", __FUNCTION__, i);
            return -LWL_ERR_GENERIC;
        }

#ifdef LWSWITCH_MC_TRACE
            LWSWITCH_PRINT(device, INFO, "%s: copying entry from src[%d] to dst[%d]\n",
                            __FUNCTION__, i, dst_list_offset + i);
            _lwswitch_mc_print_directive(device, &src[i]);
#endif

        lwswitch_os_memcpy(&dst[dst_list_offset + i], &src[i], sizeof(LWSWITCH_TCP_DIRECTIVE_LS10));
    }

    return LWL_SUCCESS;
}


//
// Build multicast directive list using the inputs given.
//
// @param device                [in] pointer to the lwswitch device struct
// @param port_list             [in] array of ports for all spray groups
// @param ports_per_spray_group [in] array specifying the size of each spray group
// @param pri_replica_offsets   [in] array, offsets of primary replica ports for each spray group
// @param replica_valid_array   [in] array, specifies which pri_replica_offsets are valid
// @param vchop_array           [in] array of vchop values for each port given in port_list
// @param table_entry           [out] pointer to table entry where directive list will be written
// @param entries_used          [out] pointer, number of valid entries produced is written here
//
LwlStatus
lwswitch_mc_build_mcp_list_ls10
(
    lwswitch_device *device,
    LwU32 *port_list,
    LwU32 *ports_per_spray_group,
    LwU32 *pri_replica_offsets,
    LwBool *replica_valid_array,
    LwU8 *vchop_array,
    LWSWITCH_MC_RID_ENTRY_LS10 *table_entry,
    LwU32 *entries_used
)
{
    LwU32 i, spray_group_idx, spray_group_size, num_spray_groups, ret;
    LwU8 *spray_group_ptrs;
    LwU32 spray_group_offset = 0;
    LwU32 primary_replica_port = LWSWITCH_MC_ILWALID;
    LwU32 dir_entries_used_sg = 0;
    LwU32 dir_entries_used = 0;
    LwU32 mcplist_offset = 0;
    LwU32 cpb[LWSWITCH_MC_NUM_COLUMNS_LS10] = { 0 };
    LwU8 vchop_map[LWSWITCH_MC_NUM_COLUMNS_LS10][LWSWITCH_MC_PORTS_PER_COLUMN_LS10];
    LWSWITCH_TCP_DIRECTIVE_LS10 tmp_mcp_list[LWSWITCH_MC_TCP_LIST_SIZE_LS10];
    LWSWITCH_TCP_DIRECTIVE_LS10 *mcp_list;

    LwU32 j;

    if ((device == NULL) || (port_list == NULL) || (ports_per_spray_group == NULL) ||
        (pri_replica_offsets == NULL) || (replica_valid_array == NULL) || (vchop_array == NULL) ||
        (table_entry == NULL) || (entries_used == NULL))
    {
        return -LWL_BAD_ARGS;
    }

    num_spray_groups = table_entry->num_spray_groups;
    spray_group_ptrs = table_entry->spray_group_ptrs;
    mcp_list = table_entry->directives;

    if (num_spray_groups == 0)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: No spray groups specified\n", __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    if (num_spray_groups > LWSWITCH_MC_MAX_SPRAYGROUPS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Too many spray groups specified: %d\n",
                        __FUNCTION__, num_spray_groups);
        return -LWL_BAD_ARGS;
    }

    for (i = 0, j = 0; i < num_spray_groups; i++)
    {
        if (ports_per_spray_group[i] < LWSWITCH_MC_MIN_PORTS_PER_GROUP_LS10)
        {
            LWSWITCH_PRINT(device, ERROR, "%s: Too few ports in spray group %d\n",
                            __FUNCTION__, i);
            return -LWL_BAD_ARGS;
        }

        if (ports_per_spray_group[i] > LWSWITCH_NUM_LINKS_LS10)
        {
            LWSWITCH_PRINT(device, ERROR, "%s: Too many ports in spray group %d\n",
                            __FUNCTION__, i);
            return -LWL_BAD_ARGS;
        }

        j += ports_per_spray_group[i];
    }

    if (j > LWSWITCH_NUM_LINKS_LS10)
    {
         LWSWITCH_PRINT(device, ERROR, "%s: Too many ports specified in total spray groups: %d\n",
                        __FUNCTION__, j);
        return -LWL_BAD_ARGS;
    }

    ret = lwswitch_init_portlist_ls10(device, mcp_list, LWSWITCH_MC_TCP_LIST_SIZE_LS10);
    if (ret != LWL_SUCCESS)
        return ret;

    // build spray strings for each spray group
    for ( spray_group_idx = 0; spray_group_idx < num_spray_groups; spray_group_idx++ )
    {
        spray_group_size = ports_per_spray_group[spray_group_idx];

#ifdef LWSWITCH_MC_DEBUG
        LWSWITCH_PRINT(device, INFO, "%s: processing spray group %d size %d of %d total groups\n",
                        __FUNCTION__, spray_group_idx, spray_group_size, num_spray_groups);
#endif

        ret = lwswitch_init_portlist_ls10(device, tmp_mcp_list, LWSWITCH_MC_TCP_LIST_SIZE_LS10);
        if (ret != LWL_SUCCESS)
            return ret;

        ret = _lwswitch_mc_build_cpb(device, spray_group_size, &port_list[spray_group_offset],
                                        LWSWITCH_MC_NUM_COLUMNS_LS10, cpb,
                                        &vchop_array[spray_group_offset], vchop_map);

        if (ret != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                            "%s: error building column-port bitmap for spray group %d: %d\n",
                            __FUNCTION__, spray_group_idx, ret);
            return ret;
        }

        // Set the offset to this spray group in the mcp list.
        spray_group_ptrs[spray_group_idx] = (LwU8)dir_entries_used;

#ifdef LWSWITCH_MC_TRACE
        LWSWITCH_PRINT(device, INFO, "%s: spray group offset for group %d is %d\n",
                       __FUNCTION__, spray_group_idx, dir_entries_used);

        for (i = 0; i < LWSWITCH_MC_NUM_COLUMNS_LS10; i++)
        {
            LWSWITCH_PRINT(device, INFO, "%d Relative ports in column %d\n",
                                        _lwswitch_mc_get_pop_count(cpb[i]), i);

            for ( j = 0; j < 32; j++ )
            {
                if (lwswitch_test_flags(cpb[i], LWBIT(j)))
                {
                    LWSWITCH_PRINT(device, INFO, "%4d", j);
                }
            }
            LWSWITCH_PRINT(device, INFO, "\n");
        }
#endif
        // if primary replica is specified for this spray group, find the port number
        if (replica_valid_array[spray_group_idx])
        {
            if (pri_replica_offsets[spray_group_idx] >= spray_group_size)
            {
                LWSWITCH_PRINT(device, ERROR,
                            "%s: primary replica offset %d is out of range for spray group %d\n",
                            __FUNCTION__, pri_replica_offsets[spray_group_idx], spray_group_idx);
                return -LWL_BAD_ARGS;
            }

            for (i = 0; i < spray_group_size; i++)
            {
                if (pri_replica_offsets[spray_group_idx] == i)
                {
                    primary_replica_port = port_list[spray_group_offset + i];
#ifdef LWSWITCH_MC_DEBUG
                    LWSWITCH_PRINT(device, INFO, "Primary replica port in spray group %d is %d\n",
                                    spray_group_idx, primary_replica_port);
#endif
                }
            }
        }
#ifdef LWSWITCH_MC_DEBUG
        if (primary_replica_port == LWSWITCH_MC_ILWALID)
            LWSWITCH_PRINT(device, INFO, "%s: No primary replica specified for spray group %d\n",
                           __FUNCTION__, spray_group_idx);
#endif

        // process columns into spray group of multicast directives
        mcplist_offset = dir_entries_used;

        if (mcplist_offset >= LWSWITCH_MC_TCP_LIST_SIZE_LS10)
        {
            LWSWITCH_PRINT(device, ERROR, "%s: Overflow: mcplist_offset is %d\n",
                            __FUNCTION__, mcplist_offset);
            return -LWL_ERR_GENERIC;
        }

#ifdef LWSWITCH_MC_DEBUG
        LWSWITCH_PRINT(device, INFO, "%s: building tmp mc portlist at mcp offset %d, size %d\n",
                        __FUNCTION__, mcplist_offset, spray_group_size);
#endif

        ret = _lwswitch_mc_build_portlist(device, cpb, primary_replica_port, vchop_map,
                                          tmp_mcp_list, &dir_entries_used_sg);

        if (ret != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR, "%s: error building MC portlist\n", __FUNCTION__);
            return ret;
        }

#ifdef LWSWITCH_MC_DEBUG
        LWSWITCH_PRINT(device, INFO, "%s: entries used after building portlist: %d\n",
                       __FUNCTION__, dir_entries_used_sg);
#endif

        ret = _lwswitch_mc_compact_portlist(device, tmp_mcp_list, &dir_entries_used_sg);
        if (ret != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR, "%s: error compacting MC portlist\n", __FUNCTION__);
            return ret;
        }

        _lwswitch_mc_set_round_flags(tmp_mcp_list, dir_entries_used_sg);

        _lwswitch_mc_set_port_flags(tmp_mcp_list, dir_entries_used_sg);

        //copy spray group entries into final portlist
        ret = _lwswitch_mc_copy_valid_entries_ls10(device, mcp_list, tmp_mcp_list,
                                                   dir_entries_used_sg, mcplist_offset);
        if (ret != LWL_SUCCESS)
            return ret;

        dir_entries_used += dir_entries_used_sg;

        // increment position in the input port list
        spray_group_offset += spray_group_size;
    }

    *entries_used = dir_entries_used;

#ifdef LWSWITCH_MC_TRACE
    _lwswitch_mc_print_directives(device, mcp_list, *entries_used, spray_group_ptrs,
                                  num_spray_groups);
#endif

    return LWL_SUCCESS;
}

static LwU32
_lwswitch_col_offset_to_port_ls10
(
    LwU32 col,
    LwU32 offset
)
{
    LwU32 i;
    LWSWITCH_COLUMN_PORT_OFFSET_LS10 cpo;

    if ((col > LWSWITCH_MC_NUM_COLUMNS_LS10) || (offset > LWSWITCH_MC_PORTS_PER_COLUMN_LS10))
        return LWSWITCH_MC_ILWALID;

    for (i = 0; i < LWSWITCH_NUM_LINKS_LS10; i++)
    {
        cpo = lwswitch_portmap_ls10[i];

        if ((cpo.column == col) && (cpo.port_offset == offset))
            return i;
    }

    return LWSWITCH_MC_ILWALID;
}

LwlStatus
lwswitch_mc_unwind_directives_ls10
(
    lwswitch_device *device,
    LWSWITCH_TCP_DIRECTIVE_LS10 directives[LWSWITCH_MC_TCP_LIST_SIZE_LS10],
    LwU32 ports[LWSWITCH_MC_MAX_PORTS],
    LwU8 vc_hop[LWSWITCH_MC_MAX_PORTS],
    LwU32 ports_per_spray_group[LWSWITCH_MC_MAX_SPRAYGROUPS],
    LwU32 replica_offset[LWSWITCH_MC_MAX_SPRAYGROUPS],
    LwBool replica_valid[LWSWITCH_MC_MAX_SPRAYGROUPS]
)
{
    LwU32 ret = LWL_SUCCESS;
    LwU32 i, port_idx, lwr_sg, ports_in_lwr_sg, port, primary_replica;
    LWSWITCH_TCP_DIRECTIVE_LS10 lwr_dir, prev_dir;

    lwr_sg = 0;
    port_idx = 0;
    ports_in_lwr_sg = 0;

    for (i = 0; i < LWSWITCH_MC_TCP_LIST_SIZE_LS10; i++)
    {
        lwr_dir = directives[i];
        if (lwr_dir.tcp == LWSWITCH_MC_ILWALID)
        {
#ifdef LWSWITCH_MC_DEBUG
            LWSWITCH_PRINT(device, INFO, "%s: reached end of directive list (element %d)\n",
                            __FUNCTION__, i);
#endif
            break;
        }

        //
        // Find primary replica.
        // For more info, see: IAS 6.12. Consistent MC Semantics
        //

        primary_replica = PRIMARY_REPLICA_NONE;

        //
        // If lastRound = 1 and continueRound = 1, primary replica is in
        // this TCP directive and portFlag = 0/1 selects even/odd port.
        //
        if ((lwr_dir.lastRound) && (lwr_dir.continueRound))
        {
            if (lwr_dir.portFlag)
                primary_replica = PRIMARY_REPLICA_ODD;
            else
                primary_replica = PRIMARY_REPLICA_EVEN;

        }
        //
        // If the previous TCP directive's portFlag = 0, and if it was not
        // used to select the even or odd port of its predecessor, and this
        // directive's portFlag == 1, this TCP directive contains the
        // primary replica, and the next TCP directive's portFlag = 0/1
        // selects the even/odd port of this TCP directive.
        //

        // If we don't have the first or last directive and portFlag == 1
        else if ((i < (LWSWITCH_MC_TCP_LIST_SIZE_LS10 - 1)) && (i > 0) && (lwr_dir.portFlag == 1))
        {
            prev_dir = directives[i - 1];

            // Is the previous directive in the same sg and is the portFlag == 0?
            if ((prev_dir.lastRound == 0) && (prev_dir.portFlag == 0))
            {
                // Check if there is no predecessor, or if the predecessor's portFlag == 0
                if ((i < 2) || (directives[i - 2].portFlag == 0))
                {
                    // The next directive's portFlags specify even or odd
                    if (directives[i + 1].portFlag)
                        primary_replica = PRIMARY_REPLICA_ODD;
                    else
                        primary_replica = PRIMARY_REPLICA_EVEN;
                }
            }
        }

        if (lwr_dir.tcpEPort != LWSWITCH_MC_NULL_PORT_LS10)
        {
            ports_in_lwr_sg++;

            if (lwr_dir.tcpEAltPath)
            {
                port = _lwswitch_col_offset_to_port_ls10(lwr_dir.tcp * 2 + 1, lwr_dir.tcpEPort);
            }
            else
            {
                port = _lwswitch_col_offset_to_port_ls10(lwr_dir.tcp * 2, lwr_dir.tcpEPort);
            }

            if (port == LWSWITCH_MC_ILWALID)
            {
                // if we get here, there's a bug when colwerting from col/offset to port number
                LWSWITCH_ASSERT(0);
                return -LWL_ERR_GENERIC;
            }

            if (port_idx >= LWSWITCH_MC_MAX_PORTS)
            {
                // if we get here, there's a bug when incrementing the port index
                LWSWITCH_ASSERT(0);
                return -LWL_ERR_GENERIC;
            }

            vc_hop[port_idx] = lwr_dir.tcpEVCHop;
            ports[port_idx] = port;

            if (primary_replica == PRIMARY_REPLICA_EVEN)
            {
                replica_offset[lwr_sg] = port_idx;
                replica_valid[lwr_sg] = LW_TRUE;
#ifdef LWSWITCH_MC_TRACE
                LWSWITCH_PRINT(device, INFO, "%s: primary replica is port %d, offset %d in sg %d\n",
                                __FUNCTION__, port, port_idx, lwr_sg);
#endif
            }

            port_idx++;
        }

        if (lwr_dir.tcpOPort != LWSWITCH_MC_NULL_PORT_LS10)
        {
            ports_in_lwr_sg++;

            if (lwr_dir.tcpOAltPath)
            {
                port = _lwswitch_col_offset_to_port_ls10(lwr_dir.tcp * 2, lwr_dir.tcpOPort);
            }
            else
            {
                port = _lwswitch_col_offset_to_port_ls10(lwr_dir.tcp * 2 + 1, lwr_dir.tcpOPort);
            }

            if (port == LWSWITCH_MC_ILWALID)
            {
                // if we get here, there's a bug when colwerting from col/offset to port number
                LWSWITCH_ASSERT(0);
                return -LWL_ERR_GENERIC;
            }

            if (port_idx >= LWSWITCH_MC_MAX_PORTS)
            {
                // if we get here, there's a bug when incrementing the port index
                LWSWITCH_ASSERT(0);
                return -LWL_ERR_GENERIC;
            }

            vc_hop[port_idx] = lwr_dir.tcpOVCHop;
            ports[port_idx] = port;

            if (primary_replica == PRIMARY_REPLICA_ODD)
            {
                replica_offset[lwr_sg] = port_idx;
                replica_valid[lwr_sg] = LW_TRUE;
#ifdef LWSWITCH_MC_TRACE
                LWSWITCH_PRINT(device, INFO, "%s: primary replica is port %d, offset %d in sg %d\n",
                                __FUNCTION__, port, port_idx, lwr_sg);
#endif
            }

            port_idx++;
        }

        if (lwr_dir.lastRound)
        {
#ifdef LWSWITCH_MC_TRACE
            LWSWITCH_PRINT(device, INFO, "%s: reached end of spray group %d, %d total ports\n",
                            __FUNCTION__, lwr_sg, ports_in_lwr_sg);
#endif
            ports_per_spray_group[lwr_sg] = ports_in_lwr_sg;
            ports_in_lwr_sg = 0;
            lwr_sg++;
        }
    }

    return ret;
}

//
// Ilwalidate an MCRID table entry.
//
// @param device                [in] pointer to the lwswitch device struct
// @param port                  [in] port for which to ilwalidate the table entry
// @param index                 [in] index into the MCRID table
// @param use_extended_table    [in] specifies whether to use the extended table, or main table
// @param zero                  [in] specifies whether to zero the entry as well as ilwalidate
//
LwlStatus
lwswitch_mc_ilwalidate_mc_rid_entry_ls10
(
    lwswitch_device *device,
    LwU32 port,
    LwU32 index,
    LwBool use_extended_table,
    LwBool zero
)
{
    LwU32 reg, i;

    if ((device == NULL) || (!lwswitch_is_link_valid(device, port)))
        return -LWL_BAD_ARGS;

    if (use_extended_table && (index > LW_ROUTE_RIDTABADDR_INDEX_MCRIDEXTTAB_DEPTH))
    {
        LWSWITCH_PRINT(device, ERROR, "%s: index %d out of range for extended table\n",
                        __FUNCTION__, index);
        return -LWL_BAD_ARGS;
    }

    if (index > LW_ROUTE_RIDTABADDR_INDEX_MCRIDTAB_DEPTH)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: index %d out of range for main table\n",
                       __FUNCTION__, index);
        return -LWL_BAD_ARGS;
    }

    if (use_extended_table)
        reg = FLD_SET_DRF(_ROUTE, _RIDTABADDR, _RAM_SEL, _SELECTSEXTMCRIDROUTERAM, 0);
    else
        reg = FLD_SET_DRF(_ROUTE, _RIDTABADDR, _RAM_SEL, _SELECTSMCRIDROUTERAM, 0);

    reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABADDR, _INDEX, index, reg);
    LWSWITCH_NPORT_WR32_LS10(device, port, _ROUTE, _RIDTABADDR, reg);

    reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA0, _VALID, 0, 0);

    if (!zero)
    {
        LWSWITCH_NPORT_WR32_LS10(device, port, _ROUTE, _RIDTABDATA0, reg);
        return LWL_SUCCESS;
    }

    for (i = 0; i < 4; i++)
    {
        LWSWITCH_NPORT_WR32_LS10(device, port, _ROUTE, _RIDTABDATA2, 0);
    }

    for (i = 0; i < 32; i++)
    {
        LWSWITCH_NPORT_WR32_LS10(device, port, _ROUTE, _RIDTABDATA1, 0);
    }

    LWSWITCH_NPORT_WR32_LS10(device, port, _ROUTE, _RIDTABDATA0, 0);

    return LWL_SUCCESS;
}

//
// Program an MCRID table entry.
//
// @param device                [in] pointer to the lwswitch device struct
// @param port                  [in] port for which to write the table entry
// @param table_entry           [in] pointer to the table entry to write
// @param directive_list_size   [in] size of the directive list contained in table_entry
//
LwlStatus
lwswitch_mc_program_mc_rid_entry_ls10
(
    lwswitch_device *device,
    LwU32 port,
    LWSWITCH_MC_RID_ENTRY_LS10 *table_entry,
    LwU32 directive_list_size
)
{
    LwU32 i, reg;
    LWSWITCH_TCP_DIRECTIVE_LS10 *lwr_dir;

    if ((device == NULL) || (!lwswitch_is_link_valid(device, port)) || (table_entry == NULL))
    {
        return -LWL_BAD_ARGS;
    }

    if (table_entry->use_extended_table &&
        (table_entry->index > LW_ROUTE_RIDTABADDR_INDEX_MCRIDEXTTAB_DEPTH))
    {
        LWSWITCH_PRINT(device, ERROR, "%s: index %d out of range for extended table\n",
                        __FUNCTION__, table_entry->index);
        return -LWL_BAD_ARGS;
    }

    if (table_entry->index > LW_ROUTE_RIDTABADDR_INDEX_MCRIDTAB_DEPTH)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: index %d out of range for main table\n",
                       __FUNCTION__, table_entry->index);
        return -LWL_BAD_ARGS;
    }

    if (directive_list_size > LWSWITCH_MC_TCP_LIST_SIZE_LS10)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: directive_list_size out of range\n", __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    if ((table_entry->num_spray_groups > LWSWITCH_MC_MAX_SPRAY_LS10) ||
        (table_entry->num_spray_groups == 0))
    {
        LWSWITCH_PRINT(device, ERROR, "%s: num_spray_groups out of range\n", __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    if ((table_entry->mcpl_size > LWSWITCH_NUM_LINKS_LS10) ||
        (table_entry->mcpl_size == 0))
    {
        LWSWITCH_PRINT(device, ERROR, "%s: mcpl_size out of range\n", __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    if (table_entry->ext_ptr_valid &&
        (table_entry->ext_ptr > LW_ROUTE_RIDTABADDR_INDEX_MCRIDEXTTAB_DEPTH))
    {
        LWSWITCH_PRINT(device, ERROR, "%s: extended_ptr out of range\n", __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    if (table_entry->use_extended_table)
        reg = FLD_SET_DRF(_ROUTE, _RIDTABADDR, _RAM_SEL, _SELECTSEXTMCRIDROUTERAM, 0);
    else
        reg = FLD_SET_DRF(_ROUTE, _RIDTABADDR, _RAM_SEL, _SELECTSMCRIDROUTERAM, 0);

    reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABADDR, _INDEX, table_entry->index, reg);
    LWSWITCH_NPORT_WR32_LS10(device, port, _ROUTE, _RIDTABADDR, reg);

    //
    // Write register 2. Each time this register is written it causes the
    // mcpl_str_ptr index to increment by 4 (4 entries are written at a time).
    //
    i = 0;
    while (i < table_entry->num_spray_groups)
    {

#ifdef LWSWITCH_MC_DEBUG
        LWSWITCH_PRINT(device, INFO, "%s: writing offset %d for spray group %d\n",
                                __FUNCTION__, table_entry->spray_group_ptrs[i], i);
#endif
        reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA2, _MCPL_STR_PTR0,
                              table_entry->spray_group_ptrs[i], 0);
        i++;

        if (i < table_entry->num_spray_groups)
        {
#ifdef LWSWITCH_MC_DEBUG
            LWSWITCH_PRINT(device, INFO, "%s: writing offset %d for spray group %d\n",
                                __FUNCTION__, table_entry->spray_group_ptrs[i], i);
#endif
            reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA2, _MCPL_STR_PTR1,
                                  table_entry->spray_group_ptrs[i], reg);
            i++;
        }

        if (i < table_entry->num_spray_groups)
        {
#ifdef LWSWITCH_MC_DEBUG
            LWSWITCH_PRINT(device, INFO, "%s: writing offset %d for spray group %d\n",
                                __FUNCTION__, table_entry->spray_group_ptrs[i], i);
#endif
            reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA2, _MCPL_STR_PTR2,
                                  table_entry->spray_group_ptrs[i], reg);
            i++;
        }

        if (i < table_entry->num_spray_groups)
        {
#ifdef LWSWITCH_MC_DEBUG
            LWSWITCH_PRINT(device, INFO, "%s: writing offset %d for spray group %d\n",
                                __FUNCTION__, table_entry->spray_group_ptrs[i], i);
#endif
            reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA2, _MCPL_STR_PTR3,
                                  table_entry->spray_group_ptrs[i], reg);
            i++;
        }

        LWSWITCH_NPORT_WR32_LS10(device, port, _ROUTE, _RIDTABDATA2, reg);
    }


    //
    // Write register 1. Each time this register is written it causes the mcpl_directive
    // index to increment by 1.
    //

    for (i = 0; i < directive_list_size; i++)
    {
        lwr_dir = &table_entry->directives[i];

        reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA1, _MCPL_E_PORT, lwr_dir->tcpEPort, 0);
        reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA1, _MCPL_E_ALTPATH, lwr_dir->tcpEAltPath, reg);
        reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA1, _MCPL_E_REQ_VCHOP, lwr_dir->tcpEVCHop, reg);
        reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA1, _MCPL_O_PORT, lwr_dir->tcpOPort, reg);
        reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA1, _MCPL_O_ALTPATH, lwr_dir->tcpOAltPath, reg);
        reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA1, _MCPL_O_REQ_VCHOP, lwr_dir->tcpOVCHop, reg);
        reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA1, _MCPL_TCP, lwr_dir->tcp, reg);
        reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA1, _MCPL_PORT_FLAG, lwr_dir->portFlag, reg);
        reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA1, _MCPL_RND_CONTINUE, lwr_dir->continueRound, reg);
        reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA1, _MCPL_LAST_RND, lwr_dir->lastRound, reg);

        LWSWITCH_NPORT_WR32_LS10(device, port, _ROUTE, _RIDTABDATA1, reg);
    }

    //
    // Write register 0.
    //
    // Due to size limitations in HW, _MCPL_SIZE must be adjusted by one here.
    // From the reference manuals:
    //
    // The number of expected responses at this switch hop for this MCID is MCPL_SIZE+1.
    //
    // For _MCPL_SPRAY_SIZE the value 0 represents 16. This requires no adjustment
    // when writing, since 16 is truncated to 0 due to field width.
    //
    // The input parameters for both of these are guaranteed to be nonzero values.
    //
    reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA0, _MCPL_SIZE, table_entry->mcpl_size - 1, 0);
    reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA0, _MCPL_SPRAY_SIZE, table_entry->num_spray_groups,
                          reg);
    reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA0, _MCPL_RID_EXT_PTR, table_entry->ext_ptr, reg);
    reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA0, _MCPL_RID_EXT_PTR_VAL, table_entry->ext_ptr_valid,
                          reg);
    reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA0, _VALID, 1, reg);

    if (!table_entry->use_extended_table)
         reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABDATA0, _MCPL_NO_DYN_RSP, table_entry->no_dyn_rsp, reg);

    LWSWITCH_NPORT_WR32_LS10(device, port, _ROUTE, _RIDTABDATA0, reg);

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_mc_read_mc_rid_entry_ls10
(
    lwswitch_device *device,
    LwU32 port,
    LWSWITCH_MC_RID_ENTRY_LS10 *table_entry
)
{
    LwU32 i, reg;

    if ((device == NULL) || (table_entry == NULL))
        return -LWL_BAD_ARGS;

    if (table_entry->use_extended_table &&
        (table_entry->index > LW_ROUTE_RIDTABADDR_INDEX_MCRIDEXTTAB_DEPTH))
    {
        LWSWITCH_PRINT(device, ERROR, "%s: index %d out of range for extended table\n",
                        __FUNCTION__, table_entry->index);
        return -LWL_BAD_ARGS;
    }

    if (table_entry->index > LW_ROUTE_RIDTABADDR_INDEX_MCRIDTAB_DEPTH)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: index %d out of range for main table\n",
                       __FUNCTION__, table_entry->index);
        return -LWL_BAD_ARGS;
    }

    // set the address
    if (table_entry->use_extended_table)
        reg = FLD_SET_DRF(_ROUTE, _RIDTABADDR, _RAM_SEL, _SELECTSEXTMCRIDROUTERAM, 0);
    else
        reg = FLD_SET_DRF(_ROUTE, _RIDTABADDR, _RAM_SEL, _SELECTSMCRIDROUTERAM, 0);

    reg = FLD_SET_DRF_NUM(_ROUTE, _RIDTABADDR, _INDEX, table_entry->index, reg);
    LWSWITCH_NPORT_WR32_LS10(device, port, _ROUTE, _RIDTABADDR, reg);

    // read in the entry
    reg = LWSWITCH_NPORT_RD32_LS10(device, port, _ROUTE, _RIDTABDATA0);

    // parse DATA0
    table_entry->valid = DRF_VAL(_ROUTE, _RIDTABDATA0, _VALID, reg);

    // if the entry is invalid, we're done
    if (!table_entry->valid)
    {
        return LWL_SUCCESS;
    }

    //
    // Due to size limitations in HW, _MCPL_SIZE must be adjusted by one here.
    // From the reference manuals:
    //
    // The number of expected responses at this switch hop for this MCID is MCPL_SIZE+1.
    //
    // For _MCPL_SPRAY_SIZE, the value 0 represents 16, so we need to adjust for that here.
    //
    table_entry->mcpl_size = DRF_VAL(_ROUTE, _RIDTABDATA0, _MCPL_SIZE, reg) + 1;
    table_entry->num_spray_groups = DRF_VAL(_ROUTE, _RIDTABDATA0, _MCPL_SPRAY_SIZE, reg);

    if (table_entry->num_spray_groups == 0)
        table_entry->num_spray_groups = 16;

    if (!table_entry->use_extended_table)
    {
        table_entry->ext_ptr = DRF_VAL(_ROUTE, _RIDTABDATA0, _MCPL_RID_EXT_PTR, reg);
        table_entry->ext_ptr_valid = DRF_VAL(_ROUTE, _RIDTABDATA0, _MCPL_RID_EXT_PTR_VAL, reg);
    }

    table_entry->no_dyn_rsp = DRF_VAL(_ROUTE, _RIDTABDATA0, _MCPL_NO_DYN_RSP, reg);

    // DATA1 contains the directives

    for (i = 0; i < LWSWITCH_MC_TCP_LIST_SIZE_LS10; i++)
    {
        reg = LWSWITCH_NPORT_RD32_LS10(device, port, _ROUTE, _RIDTABDATA1);

        table_entry->directives[i].tcpEPort = DRF_VAL(_ROUTE, _RIDTABDATA1, _MCPL_E_PORT, reg);
        table_entry->directives[i].tcpEAltPath = DRF_VAL(_ROUTE, _RIDTABDATA1, _MCPL_E_ALTPATH, reg);
        table_entry->directives[i].tcpEVCHop = DRF_VAL(_ROUTE, _RIDTABDATA1, _MCPL_E_REQ_VCHOP, reg);
        table_entry->directives[i].tcpOPort = DRF_VAL(_ROUTE, _RIDTABDATA1, _MCPL_O_PORT, reg);
        table_entry->directives[i].tcpOAltPath = DRF_VAL(_ROUTE, _RIDTABDATA1, _MCPL_O_ALTPATH, reg);
        table_entry->directives[i].tcpOVCHop = DRF_VAL(_ROUTE, _RIDTABDATA1, _MCPL_O_REQ_VCHOP, reg);
        table_entry->directives[i].tcp = DRF_VAL(_ROUTE, _RIDTABDATA1, _MCPL_TCP, reg);
        table_entry->directives[i].portFlag = DRF_VAL(_ROUTE, _RIDTABDATA1, _MCPL_PORT_FLAG, reg);
        table_entry->directives[i].continueRound = DRF_VAL(_ROUTE, _RIDTABDATA1,
                                                           _MCPL_RND_CONTINUE, reg);
        table_entry->directives[i].lastRound = DRF_VAL(_ROUTE, _RIDTABDATA1, _MCPL_LAST_RND, reg);
    }

    // DATA2 contains the spray group pointers. This register loads the next 4 pointers on each read.
    i = 0;
    while (i < table_entry->num_spray_groups)
    {
        reg = LWSWITCH_NPORT_RD32_LS10(device, port, _ROUTE, _RIDTABDATA2);

        table_entry->spray_group_ptrs[i] = DRF_VAL(_ROUTE, _RIDTABDATA2, _MCPL_STR_PTR0, reg);
        i++;

        if (i < table_entry->num_spray_groups)
        {
            table_entry->spray_group_ptrs[i] = DRF_VAL(_ROUTE, _RIDTABDATA2, _MCPL_STR_PTR1, reg);
            i++;
        }

        if (i < table_entry->num_spray_groups)
        {
            table_entry->spray_group_ptrs[i] = DRF_VAL(_ROUTE, _RIDTABDATA2, _MCPL_STR_PTR2, reg);
            i++;
        }

        if (i < table_entry->num_spray_groups)
        {
            table_entry->spray_group_ptrs[i] = DRF_VAL(_ROUTE, _RIDTABDATA2, _MCPL_STR_PTR3, reg);
            i++;
        }
    }

    return LWL_SUCCESS;
}
