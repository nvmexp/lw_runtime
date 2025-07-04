/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2014-2020 Broadcom
 * All rights reserved.
 */

#include "ulp_template_db_enum.h"
#include "ulp_template_db_field.h"
#include "ulp_template_struct.h"
#include "ulp_rte_parser.h"

uint16_t ulp_act_sig_tbl[BNXT_ULP_ACT_SIG_TBL_MAX_SZ] = {
	[BNXT_ULP_ACT_HID_015a] = 1,
	[BNXT_ULP_ACT_HID_00eb] = 2,
	[BNXT_ULP_ACT_HID_0043] = 3,
	[BNXT_ULP_ACT_HID_03d8] = 4,
	[BNXT_ULP_ACT_HID_02c1] = 5,
	[BNXT_ULP_ACT_HID_015e] = 6,
	[BNXT_ULP_ACT_HID_00ef] = 7,
	[BNXT_ULP_ACT_HID_0047] = 8,
	[BNXT_ULP_ACT_HID_03dc] = 9,
	[BNXT_ULP_ACT_HID_02c5] = 10,
	[BNXT_ULP_ACT_HID_025b] = 11,
	[BNXT_ULP_ACT_HID_01ec] = 12,
	[BNXT_ULP_ACT_HID_0144] = 13,
	[BNXT_ULP_ACT_HID_04d9] = 14,
	[BNXT_ULP_ACT_HID_03c2] = 15,
	[BNXT_ULP_ACT_HID_025f] = 16,
	[BNXT_ULP_ACT_HID_01f0] = 17,
	[BNXT_ULP_ACT_HID_0148] = 18,
	[BNXT_ULP_ACT_HID_04dd] = 19,
	[BNXT_ULP_ACT_HID_03c6] = 20,
	[BNXT_ULP_ACT_HID_0000] = 21,
	[BNXT_ULP_ACT_HID_0002] = 22,
	[BNXT_ULP_ACT_HID_0800] = 23,
	[BNXT_ULP_ACT_HID_0101] = 24,
	[BNXT_ULP_ACT_HID_0020] = 25,
	[BNXT_ULP_ACT_HID_0901] = 26,
	[BNXT_ULP_ACT_HID_0121] = 27,
	[BNXT_ULP_ACT_HID_0004] = 28,
	[BNXT_ULP_ACT_HID_0006] = 29,
	[BNXT_ULP_ACT_HID_0804] = 30,
	[BNXT_ULP_ACT_HID_0105] = 31,
	[BNXT_ULP_ACT_HID_0024] = 32,
	[BNXT_ULP_ACT_HID_0905] = 33,
	[BNXT_ULP_ACT_HID_0125] = 34,
	[BNXT_ULP_ACT_HID_0001] = 35,
	[BNXT_ULP_ACT_HID_0005] = 36,
	[BNXT_ULP_ACT_HID_0009] = 37,
	[BNXT_ULP_ACT_HID_000d] = 38,
	[BNXT_ULP_ACT_HID_0021] = 39,
	[BNXT_ULP_ACT_HID_0029] = 40,
	[BNXT_ULP_ACT_HID_0025] = 41,
	[BNXT_ULP_ACT_HID_002d] = 42,
	[BNXT_ULP_ACT_HID_0801] = 43,
	[BNXT_ULP_ACT_HID_0809] = 44,
	[BNXT_ULP_ACT_HID_0805] = 45,
	[BNXT_ULP_ACT_HID_080d] = 46,
	[BNXT_ULP_ACT_HID_0c15] = 47,
	[BNXT_ULP_ACT_HID_0c19] = 48,
	[BNXT_ULP_ACT_HID_02f6] = 49,
	[BNXT_ULP_ACT_HID_04f8] = 50,
	[BNXT_ULP_ACT_HID_01df] = 51,
	[BNXT_ULP_ACT_HID_07e5] = 52,
	[BNXT_ULP_ACT_HID_06ce] = 53,
	[BNXT_ULP_ACT_HID_02fa] = 54,
	[BNXT_ULP_ACT_HID_04fc] = 55,
	[BNXT_ULP_ACT_HID_01e3] = 56,
	[BNXT_ULP_ACT_HID_07e9] = 57,
	[BNXT_ULP_ACT_HID_06d2] = 58,
	[BNXT_ULP_ACT_HID_03f7] = 59,
	[BNXT_ULP_ACT_HID_05f9] = 60,
	[BNXT_ULP_ACT_HID_02e0] = 61,
	[BNXT_ULP_ACT_HID_08e6] = 62,
	[BNXT_ULP_ACT_HID_07cf] = 63,
	[BNXT_ULP_ACT_HID_03fb] = 64,
	[BNXT_ULP_ACT_HID_05fd] = 65,
	[BNXT_ULP_ACT_HID_02e4] = 66,
	[BNXT_ULP_ACT_HID_08ea] = 67,
	[BNXT_ULP_ACT_HID_07d3] = 68,
	[BNXT_ULP_ACT_HID_040d] = 69,
	[BNXT_ULP_ACT_HID_040f] = 70,
	[BNXT_ULP_ACT_HID_0413] = 71,
	[BNXT_ULP_ACT_HID_0567] = 72,
	[BNXT_ULP_ACT_HID_0a49] = 73,
	[BNXT_ULP_ACT_HID_050e] = 74,
	[BNXT_ULP_ACT_HID_0668] = 75,
	[BNXT_ULP_ACT_HID_0b4a] = 76,
	[BNXT_ULP_ACT_HID_0411] = 77,
	[BNXT_ULP_ACT_HID_056b] = 78,
	[BNXT_ULP_ACT_HID_0a4d] = 79,
	[BNXT_ULP_ACT_HID_0512] = 80,
	[BNXT_ULP_ACT_HID_066c] = 81,
	[BNXT_ULP_ACT_HID_0b4e] = 82
};

struct bnxt_ulp_act_match_info ulp_act_match_list[] = {
	[1] = {
	.act_hid = BNXT_ULP_ACT_HID_015a,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[2] = {
	.act_hid = BNXT_ULP_ACT_HID_00eb,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[3] = {
	.act_hid = BNXT_ULP_ACT_HID_0043,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[4] = {
	.act_hid = BNXT_ULP_ACT_HID_03d8,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[5] = {
	.act_hid = BNXT_ULP_ACT_HID_02c1,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[6] = {
	.act_hid = BNXT_ULP_ACT_HID_015e,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[7] = {
	.act_hid = BNXT_ULP_ACT_HID_00ef,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[8] = {
	.act_hid = BNXT_ULP_ACT_HID_0047,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[9] = {
	.act_hid = BNXT_ULP_ACT_HID_03dc,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[10] = {
	.act_hid = BNXT_ULP_ACT_HID_02c5,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[11] = {
	.act_hid = BNXT_ULP_ACT_HID_025b,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[12] = {
	.act_hid = BNXT_ULP_ACT_HID_01ec,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[13] = {
	.act_hid = BNXT_ULP_ACT_HID_0144,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[14] = {
	.act_hid = BNXT_ULP_ACT_HID_04d9,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[15] = {
	.act_hid = BNXT_ULP_ACT_HID_03c2,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[16] = {
	.act_hid = BNXT_ULP_ACT_HID_025f,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[17] = {
	.act_hid = BNXT_ULP_ACT_HID_01f0,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[18] = {
	.act_hid = BNXT_ULP_ACT_HID_0148,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[19] = {
	.act_hid = BNXT_ULP_ACT_HID_04dd,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[20] = {
	.act_hid = BNXT_ULP_ACT_HID_03c6,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 1
	},
	[21] = {
	.act_hid = BNXT_ULP_ACT_HID_0000,
	.act_sig = { .bits =
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 2
	},
	[22] = {
	.act_hid = BNXT_ULP_ACT_HID_0002,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DROP |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 2
	},
	[23] = {
	.act_hid = BNXT_ULP_ACT_HID_0800,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_POP_VLAN |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 2
	},
	[24] = {
	.act_hid = BNXT_ULP_ACT_HID_0101,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 2
	},
	[25] = {
	.act_hid = BNXT_ULP_ACT_HID_0020,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_VXLAN_DECAP |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 2
	},
	[26] = {
	.act_hid = BNXT_ULP_ACT_HID_0901,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_POP_VLAN |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 2
	},
	[27] = {
	.act_hid = BNXT_ULP_ACT_HID_0121,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_VXLAN_DECAP |
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 2
	},
	[28] = {
	.act_hid = BNXT_ULP_ACT_HID_0004,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 2
	},
	[29] = {
	.act_hid = BNXT_ULP_ACT_HID_0006,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_DROP |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 2
	},
	[30] = {
	.act_hid = BNXT_ULP_ACT_HID_0804,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_POP_VLAN |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 2
	},
	[31] = {
	.act_hid = BNXT_ULP_ACT_HID_0105,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 2
	},
	[32] = {
	.act_hid = BNXT_ULP_ACT_HID_0024,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_VXLAN_DECAP |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 2
	},
	[33] = {
	.act_hid = BNXT_ULP_ACT_HID_0905,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_POP_VLAN |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 2
	},
	[34] = {
	.act_hid = BNXT_ULP_ACT_HID_0125,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_VXLAN_DECAP |
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 2
	},
	[35] = {
	.act_hid = BNXT_ULP_ACT_HID_0001,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_MARK |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 3
	},
	[36] = {
	.act_hid = BNXT_ULP_ACT_HID_0005,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_MARK |
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 3
	},
	[37] = {
	.act_hid = BNXT_ULP_ACT_HID_0009,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_MARK |
		BNXT_ULP_ACTION_BIT_RSS |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 3
	},
	[38] = {
	.act_hid = BNXT_ULP_ACT_HID_000d,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_MARK |
		BNXT_ULP_ACTION_BIT_RSS |
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 3
	},
	[39] = {
	.act_hid = BNXT_ULP_ACT_HID_0021,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_MARK |
		BNXT_ULP_ACTION_BIT_VXLAN_DECAP |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 3
	},
	[40] = {
	.act_hid = BNXT_ULP_ACT_HID_0029,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_MARK |
		BNXT_ULP_ACTION_BIT_RSS |
		BNXT_ULP_ACTION_BIT_VXLAN_DECAP |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 3
	},
	[41] = {
	.act_hid = BNXT_ULP_ACT_HID_0025,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_MARK |
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_VXLAN_DECAP |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 3
	},
	[42] = {
	.act_hid = BNXT_ULP_ACT_HID_002d,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_MARK |
		BNXT_ULP_ACTION_BIT_RSS |
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_VXLAN_DECAP |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 3
	},
	[43] = {
	.act_hid = BNXT_ULP_ACT_HID_0801,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_MARK |
		BNXT_ULP_ACTION_BIT_POP_VLAN |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 3
	},
	[44] = {
	.act_hid = BNXT_ULP_ACT_HID_0809,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_MARK |
		BNXT_ULP_ACTION_BIT_RSS |
		BNXT_ULP_ACTION_BIT_POP_VLAN |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 3
	},
	[45] = {
	.act_hid = BNXT_ULP_ACT_HID_0805,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_MARK |
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_POP_VLAN |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 3
	},
	[46] = {
	.act_hid = BNXT_ULP_ACT_HID_080d,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_MARK |
		BNXT_ULP_ACTION_BIT_RSS |
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_POP_VLAN |
		BNXT_ULP_FLOW_DIR_BITMASK_ING },
	.act_tid = 3
	},
	[47] = {
	.act_hid = BNXT_ULP_ACT_HID_0c15,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_VXLAN_ENCAP |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 4
	},
	[48] = {
	.act_hid = BNXT_ULP_ACT_HID_0c19,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_VXLAN_ENCAP |
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 4
	},
	[49] = {
	.act_hid = BNXT_ULP_ACT_HID_02f6,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[50] = {
	.act_hid = BNXT_ULP_ACT_HID_04f8,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[51] = {
	.act_hid = BNXT_ULP_ACT_HID_01df,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[52] = {
	.act_hid = BNXT_ULP_ACT_HID_07e5,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[53] = {
	.act_hid = BNXT_ULP_ACT_HID_06ce,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[54] = {
	.act_hid = BNXT_ULP_ACT_HID_02fa,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[55] = {
	.act_hid = BNXT_ULP_ACT_HID_04fc,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[56] = {
	.act_hid = BNXT_ULP_ACT_HID_01e3,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[57] = {
	.act_hid = BNXT_ULP_ACT_HID_07e9,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[58] = {
	.act_hid = BNXT_ULP_ACT_HID_06d2,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[59] = {
	.act_hid = BNXT_ULP_ACT_HID_03f7,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[60] = {
	.act_hid = BNXT_ULP_ACT_HID_05f9,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[61] = {
	.act_hid = BNXT_ULP_ACT_HID_02e0,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[62] = {
	.act_hid = BNXT_ULP_ACT_HID_08e6,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[63] = {
	.act_hid = BNXT_ULP_ACT_HID_07cf,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[64] = {
	.act_hid = BNXT_ULP_ACT_HID_03fb,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[65] = {
	.act_hid = BNXT_ULP_ACT_HID_05fd,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[66] = {
	.act_hid = BNXT_ULP_ACT_HID_02e4,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[67] = {
	.act_hid = BNXT_ULP_ACT_HID_08ea,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[68] = {
	.act_hid = BNXT_ULP_ACT_HID_07d3,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_IPV4_SRC |
		BNXT_ULP_ACTION_BIT_SET_IPV4_DST |
		BNXT_ULP_ACTION_BIT_SET_TP_SRC |
		BNXT_ULP_ACTION_BIT_SET_TP_DST |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 5
	},
	[69] = {
	.act_hid = BNXT_ULP_ACT_HID_040d,
	.act_sig = { .bits =
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 6
	},
	[70] = {
	.act_hid = BNXT_ULP_ACT_HID_040f,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DROP |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 6
	},
	[71] = {
	.act_hid = BNXT_ULP_ACT_HID_0413,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DROP |
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 6
	},
	[72] = {
	.act_hid = BNXT_ULP_ACT_HID_0567,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_SET_VLAN_PCP |
		BNXT_ULP_ACTION_BIT_SET_VLAN_VID |
		BNXT_ULP_ACTION_BIT_PUSH_VLAN |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 6
	},
	[73] = {
	.act_hid = BNXT_ULP_ACT_HID_0a49,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_SET_VLAN_VID |
		BNXT_ULP_ACTION_BIT_PUSH_VLAN |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 6
	},
	[74] = {
	.act_hid = BNXT_ULP_ACT_HID_050e,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 6
	},
	[75] = {
	.act_hid = BNXT_ULP_ACT_HID_0668,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_SET_VLAN_PCP |
		BNXT_ULP_ACTION_BIT_SET_VLAN_VID |
		BNXT_ULP_ACTION_BIT_PUSH_VLAN |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 6
	},
	[76] = {
	.act_hid = BNXT_ULP_ACT_HID_0b4a,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_SET_VLAN_VID |
		BNXT_ULP_ACTION_BIT_PUSH_VLAN |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 6
	},
	[77] = {
	.act_hid = BNXT_ULP_ACT_HID_0411,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 6
	},
	[78] = {
	.act_hid = BNXT_ULP_ACT_HID_056b,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_VLAN_PCP |
		BNXT_ULP_ACTION_BIT_SET_VLAN_VID |
		BNXT_ULP_ACTION_BIT_PUSH_VLAN |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 6
	},
	[79] = {
	.act_hid = BNXT_ULP_ACT_HID_0a4d,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_SET_VLAN_VID |
		BNXT_ULP_ACTION_BIT_PUSH_VLAN |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 6
	},
	[80] = {
	.act_hid = BNXT_ULP_ACT_HID_0512,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 6
	},
	[81] = {
	.act_hid = BNXT_ULP_ACT_HID_066c,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_SET_VLAN_PCP |
		BNXT_ULP_ACTION_BIT_SET_VLAN_VID |
		BNXT_ULP_ACTION_BIT_PUSH_VLAN |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 6
	},
	[82] = {
	.act_hid = BNXT_ULP_ACT_HID_0b4e,
	.act_sig = { .bits =
		BNXT_ULP_ACTION_BIT_COUNT |
		BNXT_ULP_ACTION_BIT_DEC_TTL |
		BNXT_ULP_ACTION_BIT_SET_VLAN_VID |
		BNXT_ULP_ACTION_BIT_PUSH_VLAN |
		BNXT_ULP_FLOW_DIR_BITMASK_EGR },
	.act_tid = 6
	}
};
