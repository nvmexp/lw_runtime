/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2016 - 2018 Cavium Inc.
 * All rights reserved.
 * www.cavium.com
 */

#ifndef __ECORE_HSI_INIT_FUNC__
#define __ECORE_HSI_INIT_FUNC__
/********************************/
/* HSI Init Functions constants */
/********************************/

/* Number of VLAN priorities */
#define NUM_OF_VLAN_PRIORITIES			8

/* Size of CRC8 lookup table */
#ifndef LINUX_REMOVE
#define CRC8_TABLE_SIZE					256
#endif

/*
 * BRB RAM init requirements
 */
struct init_brb_ram_req {
	u32 guranteed_per_tc /* guaranteed size per TC, in bytes */;
	u32 headroom_per_tc /* headroom size per TC, in bytes */;
	u32 min_pkt_size /* min packet size, in bytes */;
	u32 max_ports_per_engine /* min packet size, in bytes */;
	u8 num_active_tcs[MAX_NUM_PORTS] /* number of active TCs per port */;
};


/*
 * ETS per-TC init requirements
 */
struct init_ets_tc_req {
/* if set, this TC participates in the arbitration with a strict priority
 * (the priority is equal to the TC ID)
 */
	u8 use_sp;
/* if set, this TC participates in the arbitration with a WFQ weight
 * (indicated by the weight field)
 */
	u8 use_wfq;
	u16 weight /* An arbitration weight. Valid only if use_wfq is set. */;
};

/*
 * ETS init requirements
 */
struct init_ets_req {
	u32 mtu /* Max packet size (in bytes) */;
/* ETS initialization requirements per TC. */
	struct init_ets_tc_req tc_req[NUM_OF_TCS];
};



/*
 * NIG LB RL init requirements
 */
struct init_nig_lb_rl_req {
/* Global MAC+LB RL rate (in Mbps). If set to 0, the RL will be disabled. */
	u16 lb_mac_rate;
/* Global LB RL rate (in Mbps). If set to 0, the RL will be disabled. */
	u16 lb_rate;
	u32 mtu /* Max packet size (in bytes) */;
/* RL rate per physical TC (in Mbps). If set to 0, the RL will be disabled. */
	u16 tc_rate[NUM_OF_PHYS_TCS];
};


/*
 * NIG TC mapping for each priority
 */
struct init_nig_pri_tc_map_entry {
	u8 tc_id /* the mapped TC ID */;
	u8 valid /* indicates if the mapping entry is valid */;
};


/*
 * NIG priority to TC map init requirements
 */
struct init_nig_pri_tc_map_req {
	struct init_nig_pri_tc_map_entry pri[NUM_OF_VLAN_PRIORITIES];
};


/*
 * QM per global RL init parameters
 */
struct init_qm_global_rl_params {
/* Rate limit in Mb/sec units. If set to zero, the link speed is uwsed
 * instead.
 */
	u32 rate_limit;
};


/*
 * QM per port init parameters
 */
struct init_qm_port_params {
	u8 active /* Indicates if this port is active */;
/* Vector of valid bits for active TCs used by this port */
	u8 active_phys_tcs;
/* number of PBF command lines that can be used by this port */
	u16 num_pbf_cmd_lines;
/* number of BTB blocks that can be used by this port */
	u16 num_btb_blocks;
	u16 reserved;
};


/*
 * QM per-PQ init parameters
 */
struct init_qm_pq_params {
	u8 vport_id /* VPORT ID */;
	u8 tc_id /* TC ID */;
	u8 wrr_group /* WRR group */;
/* Indicates if a rate limiter should be allocated for the PQ (0/1) */
	u8 rl_valid;
	u16 rl_id /* RL ID, valid only if rl_valid is true */;
	u8 port_id /* Port ID */;
	u8 reserved;
};


/*
 * QM per VPORT init parameters
 */
struct init_qm_vport_params {
/* WFQ weight. A value of 0 means dont configure. ignored if VPORT WFQ is
 * globally disabled.
 */
	u16 wfq;
/* the first Tx PQ ID associated with this VPORT for each TC. */
	u16 first_tx_pq_id[NUM_OF_TCS];
};

#endif /* __ECORE_HSI_INIT_FUNC__ */
