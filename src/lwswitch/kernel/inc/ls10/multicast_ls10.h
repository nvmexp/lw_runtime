/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _MULTICAST_LS10_H_
#define _MULTICAST_LS10_H_

#define LWSWITCH_MC_TCP_LIST_SIZE_LS10          LWSWITCH_NUM_LINKS_LS10 / 2
#define LWSWITCH_MC_MAX_SPRAY_LS10              16
#define LWSWITCH_MC_NUM_COLUMNS_LS10            6
#define LWSWITCH_MC_NUM_COLUMN_PAIRS_LS10       LWSWITCH_MC_NUM_COLUMNS_LS10 / 2
#define LWSWITCH_MC_PORTS_PER_COLUMN_LS10       11
#define LWSWITCH_MC_MIN_PORTS_PER_GROUP_LS10    1

#define PRIMARY_REPLICA_NONE                    0
#define PRIMARY_REPLICA_EVEN                    1
#define PRIMARY_REPLICA_ODD                     2

#define LWSWITCH_MC_ILWALID                     0xFF

#define LWSWITCH_MC_NULL_PORT_LS10              0xF

//
// Debug and trace print toggles
// To enable tracing, define LWSWITCH_MC_TRACE
//
#if defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
#define LWSWITCH_MC_DEBUG                       1
#endif

typedef struct {
    LwU32 column;
    LwU32 port_offset;
} LWSWITCH_COLUMN_PORT_OFFSET_LS10;

typedef struct {
    LwU8      tcp;           // Tile column pair
    LwU8      tcpEPort;      // Port index within even column
    LwU8      tcpEVCHop;     // VC selection
    LwU8      tcpOPort;      // Port index within odd column
    LwU8      tcpOVCHop;     // VC selection
    LwU8      roundSize;     // This is no longer part of the hardware structure. We retain it here
                             // because it is useful in various loops
    LwU8      primaryReplica;// This field is not in hardware. This code uses it to
                             // track which port should be primary, so that it can make a pass over
                             // the assembled tcp directive list and adjust portFlag and
                             // continueRound as needed to indicate primary replica
                             // valid values are:
                             //         PRIMARY_REPLICA_NONE (0b00): no primary replica in tcp
                             //         PRIMARY_REPLICA_EVEN (0b01): even (0) port is primary replica
                             //         PRIMARY_REPLICA_ODD  (0b10): odd  (1) port is primary replica
    LwBool    tcpEAltPath :1;// Alternative to select from odd column
    LwBool    tcpOAltPath :1;// Alternative to select from even column
    LwBool    lastRound   :1;// last TCP directive of the last round in this multicast string
                             // could be multiple strings in case of spray
    LwBool    continueRound:1;// dual meaning:
                             // 1) if lastRound = 1 and continueRound = 1, primary replica is in
                             // this TCP directive and portFlag = 0/1 selects even/odd port
                             // 2) if lastRound = 0 there are more TCP directives for this round.
    LwBool    portFlag    :1;// triple meaning:
                             // 1) if lastRound = 1 and continueRound = 1, primary replica is in
                             // this TCP directive and portFlag = 0/1 selects even/odd port
                             // 2) If the previous TCP directive was not used to select the even/odd
                             // port of its predecessor, and if portFlag of the previous TCP
                             // directive = 1, portFlag of this TCP directive = 0/1 selects
                             // the even/odd port of its predecessor
                             // 3) if the previous TCP directive's portFlag = 0, and if it was not
                             // used to select the even or odd port of its predecessor, this TCP
                             // directive's portFlag == 1, this TCP directive contains the
                             // primary replica, and the next TCP directive's portFlag = 0/1
                             // selects the even/odd port of this TCP directive
} LWSWITCH_TCP_DIRECTIVE_LS10;

typedef struct {
    LwU8        index;
    LwBool      use_extended_table;
    LwU8        mcpl_size;
    LwU8        num_spray_groups;
    LwU8        ext_ptr;
    LwBool      no_dyn_rsp;
    LwBool      ext_ptr_valid;
    LwBool      valid;
    LWSWITCH_TCP_DIRECTIVE_LS10 directives[LWSWITCH_MC_TCP_LIST_SIZE_LS10];
    LwU8        spray_group_ptrs[LWSWITCH_MC_MAX_SPRAY_LS10];
} LWSWITCH_MC_RID_ENTRY_LS10;

LwlStatus lwswitch_mc_build_mcp_list_ls10(lwswitch_device *device, LwU32 *port_list,
                                            LwU32 *ports_per_spray_string,
                                            LwU32 *pri_replica_offsets, LwBool *replica_valid_array,
                                            LwU8 *vchop_array,
                                            LWSWITCH_MC_RID_ENTRY_LS10 *table_entry,
                                            LwU32 *entries_used);

LwlStatus lwswitch_mc_unwind_directives_ls10(lwswitch_device *device,
                                                LWSWITCH_TCP_DIRECTIVE_LS10* directives,
                                                LwU32 *ports, LwU8 *vc_hop,
                                                LwU32 *ports_per_spray_group, LwU32 *replica_offset,
                                                LwBool *replica_valid);

LwlStatus lwswitch_mc_ilwalidate_mc_rid_entry_ls10(lwswitch_device *device, LwU32 port, LwU32 index,
                                                    LwBool use_extended_table, LwBool zero);

LwlStatus lwswitch_mc_program_mc_rid_entry_ls10(lwswitch_device *device, LwU32 port,
                                                LWSWITCH_MC_RID_ENTRY_LS10 *table_entry,
                                                LwU32 directive_list_size);

LwlStatus lwswitch_mc_read_mc_rid_entry_ls10(lwswitch_device *device, LwU32 port,
                                             LWSWITCH_MC_RID_ENTRY_LS10 *table_entry);
#endif //_MULTICAST_LS10_H_
