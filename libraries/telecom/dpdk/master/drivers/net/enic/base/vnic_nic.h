/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright 2008-2017 Cisco Systems, Inc.  All rights reserved.
 * Copyright 2007 Nuova Systems, Inc.  All rights reserved.
 */

#ifndef _VNIC_NIC_H_
#define _VNIC_NIC_H_

#define NIC_CFG_RSS_DEFAULT_CPU_MASK_FIELD	0xffUL
#define NIC_CFG_RSS_DEFAULT_CPU_SHIFT		0
#define NIC_CFG_RSS_HASH_TYPE			(0xffUL << 8)
#define NIC_CFG_RSS_HASH_TYPE_MASK_FIELD	0xffUL
#define NIC_CFG_RSS_HASH_TYPE_SHIFT		8
#define NIC_CFG_RSS_HASH_BITS			(7UL << 16)
#define NIC_CFG_RSS_HASH_BITS_MASK_FIELD	7UL
#define NIC_CFG_RSS_HASH_BITS_SHIFT		16
#define NIC_CFG_RSS_BASE_CPU			(7UL << 19)
#define NIC_CFG_RSS_BASE_CPU_MASK_FIELD		7UL
#define NIC_CFG_RSS_BASE_CPU_SHIFT		19
#define NIC_CFG_RSS_ENABLE			(1UL << 22)
#define NIC_CFG_RSS_ENABLE_MASK_FIELD		1UL
#define NIC_CFG_RSS_ENABLE_SHIFT		22
#define NIC_CFG_TSO_IPID_SPLIT_EN		(1UL << 23)
#define NIC_CFG_TSO_IPID_SPLIT_EN_MASK_FIELD	1UL
#define NIC_CFG_TSO_IPID_SPLIT_EN_SHIFT		23
#define NIC_CFG_IG_VLAN_STRIP_EN		(1UL << 24)
#define NIC_CFG_IG_VLAN_STRIP_EN_MASK_FIELD	1UL
#define NIC_CFG_IG_VLAN_STRIP_EN_SHIFT		24

#define NIC_CFG_RSS_HASH_TYPE_UDP_IPV4		(1 << 0)
#define NIC_CFG_RSS_HASH_TYPE_IPV4		(1 << 1)
#define NIC_CFG_RSS_HASH_TYPE_TCP_IPV4		(1 << 2)
#define NIC_CFG_RSS_HASH_TYPE_IPV6		(1 << 3)
#define NIC_CFG_RSS_HASH_TYPE_TCP_IPV6		(1 << 4)
#define NIC_CFG_RSS_HASH_TYPE_RSVD1		(1 << 5)
#define NIC_CFG_RSS_HASH_TYPE_RSVD2		(1 << 6)
#define NIC_CFG_RSS_HASH_TYPE_UDP_IPV6		(1 << 7)

static inline void vnic_set_nic_cfg(uint32_t *nic_cfg,
	uint8_t rss_default_cpu, uint8_t rss_hash_type,
	uint8_t rss_hash_bits, uint8_t rss_base_cpu,
	uint8_t rss_enable, uint8_t tso_ipid_split_en,
	uint8_t ig_vlan_strip_en)
{
	*nic_cfg = (rss_default_cpu & NIC_CFG_RSS_DEFAULT_CPU_MASK_FIELD) |
		((rss_hash_type & NIC_CFG_RSS_HASH_TYPE_MASK_FIELD)
			<< NIC_CFG_RSS_HASH_TYPE_SHIFT) |
		((rss_hash_bits & NIC_CFG_RSS_HASH_BITS_MASK_FIELD)
			<< NIC_CFG_RSS_HASH_BITS_SHIFT) |
		((rss_base_cpu & NIC_CFG_RSS_BASE_CPU_MASK_FIELD)
			<< NIC_CFG_RSS_BASE_CPU_SHIFT) |
		((rss_enable & NIC_CFG_RSS_ENABLE_MASK_FIELD)
			<< NIC_CFG_RSS_ENABLE_SHIFT) |
		((tso_ipid_split_en & NIC_CFG_TSO_IPID_SPLIT_EN_MASK_FIELD)
			<< NIC_CFG_TSO_IPID_SPLIT_EN_SHIFT) |
		((ig_vlan_strip_en & NIC_CFG_IG_VLAN_STRIP_EN_MASK_FIELD)
			<< NIC_CFG_IG_VLAN_STRIP_EN_SHIFT);
}

#endif /* _VNIC_NIC_H_ */
