/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2017 Intel Corporation
 */

#ifndef _CMDLINE_TM_H_
#define _CMDLINE_TM_H_

 /* Traffic Management CLI */
extern cmdline_parse_inst_t cmd_show_port_tm_cap;
extern cmdline_parse_inst_t cmd_show_port_tm_level_cap;
extern cmdline_parse_inst_t cmd_show_port_tm_node_cap;
extern cmdline_parse_inst_t cmd_show_port_tm_node_type;
extern cmdline_parse_inst_t cmd_show_port_tm_node_stats;
extern cmdline_parse_inst_t cmd_add_port_tm_node_shaper_profile;
extern cmdline_parse_inst_t cmd_del_port_tm_node_shaper_profile;
extern cmdline_parse_inst_t cmd_add_port_tm_node_shared_shaper;
extern cmdline_parse_inst_t cmd_del_port_tm_node_shared_shaper;
extern cmdline_parse_inst_t cmd_add_port_tm_node_wred_profile;
extern cmdline_parse_inst_t cmd_del_port_tm_node_wred_profile;
extern cmdline_parse_inst_t cmd_set_port_tm_node_shaper_profile;
extern cmdline_parse_inst_t cmd_add_port_tm_nonleaf_node;
extern cmdline_parse_inst_t cmd_add_port_tm_nonleaf_node_pmode;
extern cmdline_parse_inst_t cmd_add_port_tm_leaf_node;
extern cmdline_parse_inst_t cmd_del_port_tm_node;
extern cmdline_parse_inst_t cmd_set_port_tm_node_parent;
extern cmdline_parse_inst_t cmd_suspend_port_tm_node;
extern cmdline_parse_inst_t cmd_resume_port_tm_node;
extern cmdline_parse_inst_t cmd_port_tm_hierarchy_commit;
extern cmdline_parse_inst_t cmd_port_tm_mark_vlan_dei;
extern cmdline_parse_inst_t cmd_port_tm_mark_ip_ecn;
extern cmdline_parse_inst_t cmd_port_tm_mark_ip_dscp;

#endif /* _CMDLINE_TM_H_ */
