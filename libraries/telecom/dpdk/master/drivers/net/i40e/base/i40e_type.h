/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2001-2020 Intel Corporation
 */

#ifndef _I40E_TYPE_H_
#define _I40E_TYPE_H_

#include "i40e_status.h"
#include "i40e_osdep.h"
#include "i40e_register.h"
#include "i40e_adminq.h"
#include "i40e_hmc.h"
#include "i40e_lan_hmc.h"
#include "i40e_devids.h"

#define UNREFERENCED_XPARAMETER
#define UNREFERENCED_1PARAMETER(_p) (_p);
#define UNREFERENCED_2PARAMETER(_p, _q) (_p); (_q);
#define UNREFERENCED_3PARAMETER(_p, _q, _r) (_p); (_q); (_r);
#define UNREFERENCED_4PARAMETER(_p, _q, _r, _s) (_p); (_q); (_r); (_s);
#define UNREFERENCED_5PARAMETER(_p, _q, _r, _s, _t) (_p); (_q); (_r); (_s); (_t);

#ifndef LINUX_MACROS
#ifndef BIT
#define BIT(a) (1UL << (a))
#endif /* BIT */
#ifndef BIT_ULL
#define BIT_ULL(a) (1ULL << (a))
#endif /* BIT_ULL */
#endif /* LINUX_MACROS */

#ifndef I40E_MASK
/* I40E_MASK is a macro used on 32 bit registers */
#define I40E_MASK(mask, shift) (mask << shift)
#endif

#define I40E_MAX_PF			16
#define I40E_MAX_PF_VSI			64
#define I40E_MAX_PF_QP			128
#define I40E_MAX_VSI_QP			16
#define I40E_MAX_VF_VSI			4
#define I40E_MAX_CHAINED_RX_BUFFERS	5
#define I40E_MAX_PF_UDP_OFFLOAD_PORTS	16

/* something less than 1 minute */
#define I40E_HEARTBEAT_TIMEOUT		(HZ * 50)

/* Max default timeout in ms, */
#define I40E_MAX_LWM_TIMEOUT		18000

/* Max timeout in ms for the phy to respond */
#define I40E_MAX_PHY_TIMEOUT		500

/* Check whether address is multicast. */
#define I40E_IS_MULTICAST(address) (bool)(((u8 *)(address))[0] & ((u8)0x01))

/* Check whether an address is broadcast. */
#define I40E_IS_BROADCAST(address)	\
	((((u8 *)(address))[0] == ((u8)0xff)) && \
	(((u8 *)(address))[1] == ((u8)0xff)))

/* Switch from ms to the 1usec global time (this is the GTIME resolution) */
#define I40E_MS_TO_GTIME(time)		((time) * 1000)

/* forward declaration */
struct i40e_hw;
typedef void (*I40E_ADMINQ_CALLBACK)(struct i40e_hw *, struct i40e_aq_desc *);

#ifndef ETH_ALEN
#define ETH_ALEN	6
#endif
/* Data type manipulation macros. */
#define I40E_HI_DWORD(x)	((u32)((((x) >> 16) >> 16) & 0xFFFFFFFF))
#define I40E_LO_DWORD(x)	((u32)((x) & 0xFFFFFFFF))

#define I40E_HI_WORD(x)		((u16)(((x) >> 16) & 0xFFFF))
#define I40E_LO_WORD(x)		((u16)((x) & 0xFFFF))

#define I40E_HI_BYTE(x)		((u8)(((x) >> 8) & 0xFF))
#define I40E_LO_BYTE(x)		((u8)((x) & 0xFF))

/* Number of Transmit Descriptors must be a multiple of 32. */
#define I40E_REQ_TX_DESCRIPTOR_MULTIPLE	32
/* Number of Receive Descriptors must be a multiple of 32 if
 * the number of descriptors is greater than 32.
 */
#define I40E_REQ_RX_DESCRIPTOR_MULTIPLE	32

#define I40E_DESC_UNUSED(R)	\
	((((R)->next_to_clean > (R)->next_to_use) ? 0 : (R)->count) + \
	(R)->next_to_clean - (R)->next_to_use - 1)

/* bitfields for Tx queue mapping in QTX_CTL */
#define I40E_QTX_CTL_VF_QUEUE	0x0
#define I40E_QTX_CTL_VM_QUEUE	0x1
#define I40E_QTX_CTL_PF_QUEUE	0x2

/* debug masks - set these bits in hw->debug_mask to control output */
enum i40e_debug_mask {
	I40E_DEBUG_INIT			= 0x00000001,
	I40E_DEBUG_RELEASE		= 0x00000002,

	I40E_DEBUG_LINK			= 0x00000010,
	I40E_DEBUG_PHY			= 0x00000020,
	I40E_DEBUG_HMC			= 0x00000040,
	I40E_DEBUG_LWM			= 0x00000080,
	I40E_DEBUG_LAN			= 0x00000100,
	I40E_DEBUG_FLOW			= 0x00000200,
	I40E_DEBUG_DCB			= 0x00000400,
	I40E_DEBUG_DIAG			= 0x00000800,
	I40E_DEBUG_FD			= 0x00001000,
	I40E_DEBUG_PACKAGE		= 0x00002000,

	I40E_DEBUG_AQ_MESSAGE		= 0x01000000,
	I40E_DEBUG_AQ_DESCRIPTOR	= 0x02000000,
	I40E_DEBUG_AQ_DESC_BUFFER	= 0x04000000,
	I40E_DEBUG_AQ_COMMAND		= 0x06000000,
	I40E_DEBUG_AQ			= 0x0F000000,

	I40E_DEBUG_USER			= 0xF0000000,

	I40E_DEBUG_ALL			= 0xFFFFFFFF
};

/* PCI Bus Info */
#define I40E_PCI_LINK_STATUS		0xB2
#define I40E_PCI_LINK_WIDTH		0x3F0
#define I40E_PCI_LINK_WIDTH_1		0x10
#define I40E_PCI_LINK_WIDTH_2		0x20
#define I40E_PCI_LINK_WIDTH_4		0x40
#define I40E_PCI_LINK_WIDTH_8		0x80
#define I40E_PCI_LINK_SPEED		0xF
#define I40E_PCI_LINK_SPEED_2500	0x1
#define I40E_PCI_LINK_SPEED_5000	0x2
#define I40E_PCI_LINK_SPEED_8000	0x3

#define I40E_MDIO_CLAUSE22_STCODE_MASK	I40E_MASK(1, \
						  I40E_GLGEN_MSCA_STCODE_SHIFT)
#define I40E_MDIO_CLAUSE22_OPCODE_WRITE_MASK	I40E_MASK(1, \
						  I40E_GLGEN_MSCA_OPCODE_SHIFT)
#define I40E_MDIO_CLAUSE22_OPCODE_READ_MASK	I40E_MASK(2, \
						  I40E_GLGEN_MSCA_OPCODE_SHIFT)

#define I40E_MDIO_CLAUSE45_STCODE_MASK	I40E_MASK(0, \
						  I40E_GLGEN_MSCA_STCODE_SHIFT)
#define I40E_MDIO_CLAUSE45_OPCODE_ADDRESS_MASK	I40E_MASK(0, \
						  I40E_GLGEN_MSCA_OPCODE_SHIFT)
#define I40E_MDIO_CLAUSE45_OPCODE_WRITE_MASK	I40E_MASK(1, \
						  I40E_GLGEN_MSCA_OPCODE_SHIFT)
#define I40E_MDIO_CLAUSE45_OPCODE_READ_INC_ADDR_MASK	I40E_MASK(2, \
						  I40E_GLGEN_MSCA_OPCODE_SHIFT)
#define I40E_MDIO_CLAUSE45_OPCODE_READ_MASK	I40E_MASK(3, \
						  I40E_GLGEN_MSCA_OPCODE_SHIFT)

#define I40E_PHY_COM_REG_PAGE			0x1E
#define I40E_PHY_LED_LINK_MODE_MASK		0xF0
#define I40E_PHY_LED_MANUAL_ON			0x100
#define I40E_PHY_LED_PROV_REG_1			0xC430
#define I40E_PHY_LED_MODE_MASK			0xFFFF
#define I40E_PHY_LED_MODE_ORIG			0x80000000

/* Memory types */
enum i40e_memset_type {
	I40E_NONDMA_MEM = 0,
	I40E_DMA_MEM
};

/* Memcpy types */
enum i40e_memcpy_type {
	I40E_NONDMA_TO_NONDMA = 0,
	I40E_NONDMA_TO_DMA,
	I40E_DMA_TO_DMA,
	I40E_DMA_TO_NONDMA
};

/* These are structs for managing the hardware information and the operations.
 * The structures of function pointers are filled out at init time when we
 * know for sure exactly which hardware we're working with.  This gives us the
 * flexibility of using the same main driver code but adapting to slightly
 * different hardware needs as new parts are developed.  For this architecture,
 * the Firmware and AdminQ are intended to insulate the driver from most of the
 * future changes, but these structures will also do part of the job.
 */
enum i40e_mac_type {
	I40E_MAC_UNKNOWN = 0,
	I40E_MAC_XL710,
	I40E_MAC_VF,
	I40E_MAC_X722,
	I40E_MAC_X722_VF,
	I40E_MAC_GENERIC,
};

enum i40e_media_type {
	I40E_MEDIA_TYPE_UNKNOWN = 0,
	I40E_MEDIA_TYPE_FIBER,
	I40E_MEDIA_TYPE_BASET,
	I40E_MEDIA_TYPE_BACKPLANE,
	I40E_MEDIA_TYPE_CX4,
	I40E_MEDIA_TYPE_DA,
	I40E_MEDIA_TYPE_VIRTUAL
};

enum i40e_fc_mode {
	I40E_FC_NONE = 0,
	I40E_FC_RX_PAUSE,
	I40E_FC_TX_PAUSE,
	I40E_FC_FULL,
	I40E_FC_PFC,
	I40E_FC_DEFAULT
};

enum i40e_set_fc_aq_failures {
	I40E_SET_FC_AQ_FAIL_NONE = 0,
	I40E_SET_FC_AQ_FAIL_GET = 1,
	I40E_SET_FC_AQ_FAIL_SET = 2,
	I40E_SET_FC_AQ_FAIL_UPDATE = 4,
	I40E_SET_FC_AQ_FAIL_SET_UPDATE = 6
};

enum i40e_vsi_type {
	I40E_VSI_MAIN	= 0,
	I40E_VSI_VMDQ1	= 1,
	I40E_VSI_VMDQ2	= 2,
	I40E_VSI_CTRL	= 3,
	I40E_VSI_FCOE	= 4,
	I40E_VSI_MIRROR	= 5,
	I40E_VSI_SRIOV	= 6,
	I40E_VSI_FDIR	= 7,
	I40E_VSI_TYPE_UNKNOWN
};

enum i40e_queue_type {
	I40E_QUEUE_TYPE_RX = 0,
	I40E_QUEUE_TYPE_TX,
	I40E_QUEUE_TYPE_PE_CEQ,
	I40E_QUEUE_TYPE_UNKNOWN
};

struct i40e_link_status {
	enum i40e_aq_phy_type phy_type;
	enum i40e_aq_link_speed link_speed;
	u8 link_info;
	u8 an_info;
	u8 req_fec_info;
	u8 fec_info;
	u8 ext_info;
	u8 loopback;
	/* is Link Status Event notification to SW enabled */
	bool lse_enable;
	u16 max_frame_size;
	bool crc_enable;
	u8 pacing;
	u8 requested_speeds;
	u8 module_type[3];
	/* 1st byte: module identifier */
#define I40E_MODULE_TYPE_SFP		0x03
#define I40E_MODULE_TYPE_QSFP		0x0D
	/* 2nd byte: ethernet compliance codes for 10/40G */
#define I40E_MODULE_TYPE_40G_ACTIVE	0x01
#define I40E_MODULE_TYPE_40G_LR4	0x02
#define I40E_MODULE_TYPE_40G_SR4	0x04
#define I40E_MODULE_TYPE_40G_CR4	0x08
#define I40E_MODULE_TYPE_10G_BASE_SR	0x10
#define I40E_MODULE_TYPE_10G_BASE_LR	0x20
#define I40E_MODULE_TYPE_10G_BASE_LRM	0x40
#define I40E_MODULE_TYPE_10G_BASE_ER	0x80
	/* 3rd byte: ethernet compliance codes for 1G */
#define I40E_MODULE_TYPE_1000BASE_SX	0x01
#define I40E_MODULE_TYPE_1000BASE_LX	0x02
#define I40E_MODULE_TYPE_1000BASE_CX	0x04
#define I40E_MODULE_TYPE_1000BASE_T	0x08
};

struct i40e_phy_info {
	struct i40e_link_status link_info;
	struct i40e_link_status link_info_old;
	bool get_link_info;
	enum i40e_media_type media_type;
	/* all the phy types the LWM is capable of */
	u64 phy_types;
};

#define I40E_CAP_PHY_TYPE_SGMII BIT_ULL(I40E_PHY_TYPE_SGMII)
#define I40E_CAP_PHY_TYPE_1000BASE_KX BIT_ULL(I40E_PHY_TYPE_1000BASE_KX)
#define I40E_CAP_PHY_TYPE_10GBASE_KX4 BIT_ULL(I40E_PHY_TYPE_10GBASE_KX4)
#define I40E_CAP_PHY_TYPE_10GBASE_KR BIT_ULL(I40E_PHY_TYPE_10GBASE_KR)
#define I40E_CAP_PHY_TYPE_40GBASE_KR4 BIT_ULL(I40E_PHY_TYPE_40GBASE_KR4)
#define I40E_CAP_PHY_TYPE_XAUI BIT_ULL(I40E_PHY_TYPE_XAUI)
#define I40E_CAP_PHY_TYPE_XFI BIT_ULL(I40E_PHY_TYPE_XFI)
#define I40E_CAP_PHY_TYPE_SFI BIT_ULL(I40E_PHY_TYPE_SFI)
#define I40E_CAP_PHY_TYPE_XLAUI BIT_ULL(I40E_PHY_TYPE_XLAUI)
#define I40E_CAP_PHY_TYPE_XLPPI BIT_ULL(I40E_PHY_TYPE_XLPPI)
#define I40E_CAP_PHY_TYPE_40GBASE_CR4_LW BIT_ULL(I40E_PHY_TYPE_40GBASE_CR4_LW)
#define I40E_CAP_PHY_TYPE_10GBASE_CR1_LW BIT_ULL(I40E_PHY_TYPE_10GBASE_CR1_LW)
#define I40E_CAP_PHY_TYPE_10GBASE_AOC BIT_ULL(I40E_PHY_TYPE_10GBASE_AOC)
#define I40E_CAP_PHY_TYPE_40GBASE_AOC BIT_ULL(I40E_PHY_TYPE_40GBASE_AOC)
#define I40E_CAP_PHY_TYPE_100BASE_TX BIT_ULL(I40E_PHY_TYPE_100BASE_TX)
#define I40E_CAP_PHY_TYPE_1000BASE_T BIT_ULL(I40E_PHY_TYPE_1000BASE_T)
#define I40E_CAP_PHY_TYPE_10GBASE_T BIT_ULL(I40E_PHY_TYPE_10GBASE_T)
#define I40E_CAP_PHY_TYPE_10GBASE_SR BIT_ULL(I40E_PHY_TYPE_10GBASE_SR)
#define I40E_CAP_PHY_TYPE_10GBASE_LR BIT_ULL(I40E_PHY_TYPE_10GBASE_LR)
#define I40E_CAP_PHY_TYPE_10GBASE_SFPP_LW BIT_ULL(I40E_PHY_TYPE_10GBASE_SFPP_LW)
#define I40E_CAP_PHY_TYPE_10GBASE_CR1 BIT_ULL(I40E_PHY_TYPE_10GBASE_CR1)
#define I40E_CAP_PHY_TYPE_40GBASE_CR4 BIT_ULL(I40E_PHY_TYPE_40GBASE_CR4)
#define I40E_CAP_PHY_TYPE_40GBASE_SR4 BIT_ULL(I40E_PHY_TYPE_40GBASE_SR4)
#define I40E_CAP_PHY_TYPE_40GBASE_LR4 BIT_ULL(I40E_PHY_TYPE_40GBASE_LR4)
#define I40E_CAP_PHY_TYPE_1000BASE_SX BIT_ULL(I40E_PHY_TYPE_1000BASE_SX)
#define I40E_CAP_PHY_TYPE_1000BASE_LX BIT_ULL(I40E_PHY_TYPE_1000BASE_LX)
#define I40E_CAP_PHY_TYPE_1000BASE_T_OPTICAL \
				BIT_ULL(I40E_PHY_TYPE_1000BASE_T_OPTICAL)
#define I40E_CAP_PHY_TYPE_20GBASE_KR2 BIT_ULL(I40E_PHY_TYPE_20GBASE_KR2)
/*
 * Defining the macro I40E_TYPE_OFFSET to implement a bit shift for some
 * PHY types. There is an unused bit (31) in the I40E_CAP_PHY_TYPE_* bit
 * fields but no corresponding gap in the i40e_aq_phy_type enumeration. So,
 * a shift is needed to adjust for this with values larger than 31. The
 * only affected values are I40E_PHY_TYPE_25GBASE_*.
 */
#define I40E_PHY_TYPE_OFFSET 1
#define I40E_CAP_PHY_TYPE_25GBASE_KR BIT_ULL(I40E_PHY_TYPE_25GBASE_KR + \
					     I40E_PHY_TYPE_OFFSET)
#define I40E_CAP_PHY_TYPE_25GBASE_CR BIT_ULL(I40E_PHY_TYPE_25GBASE_CR + \
					     I40E_PHY_TYPE_OFFSET)
#define I40E_CAP_PHY_TYPE_25GBASE_SR BIT_ULL(I40E_PHY_TYPE_25GBASE_SR + \
					     I40E_PHY_TYPE_OFFSET)
#define I40E_CAP_PHY_TYPE_25GBASE_LR BIT_ULL(I40E_PHY_TYPE_25GBASE_LR + \
					     I40E_PHY_TYPE_OFFSET)
#define I40E_CAP_PHY_TYPE_25GBASE_AOC BIT_ULL(I40E_PHY_TYPE_25GBASE_AOC + \
					     I40E_PHY_TYPE_OFFSET)
#define I40E_CAP_PHY_TYPE_25GBASE_ACC BIT_ULL(I40E_PHY_TYPE_25GBASE_ACC + \
					     I40E_PHY_TYPE_OFFSET)
/* Offset for 2.5G/5G PHY Types value to bit number colwersion */
#define I40E_PHY_TYPE_OFFSET2 (-10)
#define I40E_CAP_PHY_TYPE_2_5GBASE_T BIT_ULL(I40E_PHY_TYPE_2_5GBASE_T + \
					     I40E_PHY_TYPE_OFFSET2)
#define I40E_CAP_PHY_TYPE_5GBASE_T BIT_ULL(I40E_PHY_TYPE_5GBASE_T + \
					     I40E_PHY_TYPE_OFFSET2)
#define I40E_HW_CAP_MAX_GPIO			30
#define I40E_HW_CAP_MDIO_PORT_MODE_MDIO		0
#define I40E_HW_CAP_MDIO_PORT_MODE_I2C		1

enum i40e_acpi_programming_method {
	I40E_ACPI_PROGRAMMING_METHOD_HW_FVL = 0,
	I40E_ACPI_PROGRAMMING_METHOD_AQC_FPK = 1
};

#define I40E_WOL_SUPPORT_MASK			0x1
#define I40E_ACPI_PROGRAMMING_METHOD_MASK	0x2
#define I40E_PROXY_SUPPORT_MASK			0x4

/* Capabilities of a PF or a VF or the whole device */
struct i40e_hw_capabilities {
	u32  switch_mode;
#define I40E_LWM_IMAGE_TYPE_EVB		0x0
#define I40E_LWM_IMAGE_TYPE_CLOUD	0x2
#define I40E_LWM_IMAGE_TYPE_UDP_CLOUD	0x3

	/* Cloud filter modes:
	 * Mode1: Filter on L4 port only
	 * Mode2: Filter for non-tunneled traffic
	 * Mode3: Filter for tunnel traffic
	 */
#define I40E_CLOUD_FILTER_MODE1	0x6
#define I40E_CLOUD_FILTER_MODE2	0x7
#define I40E_CLOUD_FILTER_MODE3	0x8
#define I40E_SWITCH_MODE_MASK	0xF

	u32  management_mode;
	u32  mng_protocols_over_mctp;
#define I40E_MNG_PROTOCOL_PLDM		0x2
#define I40E_MNG_PROTOCOL_OEM_COMMANDS	0x4
#define I40E_MNG_PROTOCOL_NCSI		0x8
	u32  npar_enable;
	u32  os2bmc;
	u32  valid_functions;
	bool sr_iov_1_1;
	bool vmdq;
	bool evb_802_1_qbg; /* Edge Virtual Bridging */
	bool evb_802_1_qbh; /* Bridge Port Extension */
	bool dcb;
	bool fcoe;
	bool iscsi; /* Indicates iSCSI enabled */
	bool flex10_enable;
	bool flex10_capable;
	u32  flex10_mode;
#define I40E_FLEX10_MODE_UNKNOWN	0x0
#define I40E_FLEX10_MODE_DCC		0x1
#define I40E_FLEX10_MODE_DCI		0x2

	u32 flex10_status;
#define I40E_FLEX10_STATUS_DCC_ERROR	0x1
#define I40E_FLEX10_STATUS_VC_MODE	0x2

	bool sec_rev_disabled;
	bool update_disabled;
#define I40E_LWM_MGMT_SEC_REV_DISABLED	0x1
#define I40E_LWM_MGMT_UPDATE_DISABLED	0x2

	bool mgmt_cem;
	bool ieee_1588;
	bool iwarp;
	bool fd;
	u32 fd_filters_guaranteed;
	u32 fd_filters_best_effort;
	bool rss;
	u32 rss_table_size;
	u32 rss_table_entry_width;
	bool led[I40E_HW_CAP_MAX_GPIO];
	bool sdp[I40E_HW_CAP_MAX_GPIO];
	u32 lwm_image_type;
	u32 num_flow_director_filters;
	u32 num_vfs;
	u32 vf_base_id;
	u32 num_vsis;
	u32 num_rx_qp;
	u32 num_tx_qp;
	u32 base_queue;
	u32 num_msix_vectors;
	u32 num_msix_vectors_vf;
	u32 led_pin_num;
	u32 sdp_pin_num;
	u32 mdio_port_num;
	u32 mdio_port_mode;
	u8 rx_buf_chain_len;
	u32 enabled_tcmap;
	u32 maxtc;
	u64 wr_csr_prot;
	bool dis_unused_ports;
	bool apm_wol_support;
	enum i40e_acpi_programming_method acpi_prog_method;
	bool proxy_support;
};

struct i40e_mac_info {
	enum i40e_mac_type type;
	u8 addr[ETH_ALEN];
	u8 perm_addr[ETH_ALEN];
	u8 san_addr[ETH_ALEN];
	u8 port_addr[ETH_ALEN];
	u16 max_fcoeq;
};

enum i40e_aq_resources_ids {
	I40E_LWM_RESOURCE_ID = 1
};

enum i40e_aq_resource_access_type {
	I40E_RESOURCE_READ = 1,
	I40E_RESOURCE_WRITE
};

struct i40e_lwm_info {
	u64 hw_semaphore_timeout; /* usec global time (GTIME resolution) */
	u32 timeout;              /* [ms] */
	u16 sr_size;              /* Shadow RAM size in words */
	bool blank_lwm_mode;      /* is LWM empty (no FW present)*/
	u16 version;              /* LWM package version */
	u32 eetrack;              /* LWM data version */
	u32 oem_ver;              /* OEM version info */
};

/* definitions used in LWM update support */

enum i40e_lwmupd_cmd {
	I40E_LWMUPD_ILWALID,
	I40E_LWMUPD_READ_CON,
	I40E_LWMUPD_READ_SNT,
	I40E_LWMUPD_READ_LCB,
	I40E_LWMUPD_READ_SA,
	I40E_LWMUPD_WRITE_ERA,
	I40E_LWMUPD_WRITE_CON,
	I40E_LWMUPD_WRITE_SNT,
	I40E_LWMUPD_WRITE_LCB,
	I40E_LWMUPD_WRITE_SA,
	I40E_LWMUPD_CSUM_CON,
	I40E_LWMUPD_CSUM_SA,
	I40E_LWMUPD_CSUM_LCB,
	I40E_LWMUPD_STATUS,
	I40E_LWMUPD_EXEC_AQ,
	I40E_LWMUPD_GET_AQ_RESULT,
	I40E_LWMUPD_GET_AQ_EVENT,
	I40E_LWMUPD_FEATURES,
};

enum i40e_lwmupd_state {
	I40E_LWMUPD_STATE_INIT,
	I40E_LWMUPD_STATE_READING,
	I40E_LWMUPD_STATE_WRITING,
	I40E_LWMUPD_STATE_INIT_WAIT,
	I40E_LWMUPD_STATE_WRITE_WAIT,
	I40E_LWMUPD_STATE_ERROR
};

/* lwm_access definition and its masks/shifts need to be accessible to
 * application, core driver, and shared code.  Where is the right file?
 */
#define I40E_LWM_READ	0xB
#define I40E_LWM_WRITE	0xC

#define I40E_LWM_MOD_PNT_MASK 0xFF

#define I40E_LWM_TRANS_SHIFT			8
#define I40E_LWM_TRANS_MASK			(0xf << I40E_LWM_TRANS_SHIFT)
#define I40E_LWM_PRESERVATION_FLAGS_SHIFT	12
#define I40E_LWM_PRESERVATION_FLAGS_MASK \
				(0x3 << I40E_LWM_PRESERVATION_FLAGS_SHIFT)
#define I40E_LWM_PRESERVATION_FLAGS_SELECTED	0x01
#define I40E_LWM_PRESERVATION_FLAGS_ALL		0x02
#define I40E_LWM_CON				0x0
#define I40E_LWM_SNT				0x1
#define I40E_LWM_LCB				0x2
#define I40E_LWM_SA				(I40E_LWM_SNT | I40E_LWM_LCB)
#define I40E_LWM_ERA				0x4
#define I40E_LWM_CSUM				0x8
#define I40E_LWM_AQE				0xe
#define I40E_LWM_EXEC				0xf

#define I40E_LWM_EXEC_GET_AQ_RESULT		0x0
#define I40E_LWM_EXEC_FEATURES			0xe
#define I40E_LWM_EXEC_STATUS			0xf

#define I40E_LWM_ADAPT_SHIFT	16
#define I40E_LWM_ADAPT_MASK	(0xffffULL << I40E_LWM_ADAPT_SHIFT)

#define I40E_LWMUPD_MAX_DATA	4096
#define I40E_LWMUPD_IFACE_TIMEOUT 2 /* seconds */

struct i40e_lwm_access {
	u32 command;
	u32 config;
	u32 offset;	/* in bytes */
	u32 data_size;	/* in bytes */
	u8 data[1];
};

/* LWMUpdate features API */
#define I40E_LWMUPD_FEATURES_API_VER_MAJOR		0
#define I40E_LWMUPD_FEATURES_API_VER_MINOR		14
#define I40E_LWMUPD_FEATURES_API_FEATURES_ARRAY_LEN	12

#define I40E_LWMUPD_FEATURE_FLAT_LWM_SUPPORT		BIT(0)

struct i40e_lwmupd_features {
	u8 major;
	u8 minor;
	u16 size;
	u8 features[I40E_LWMUPD_FEATURES_API_FEATURES_ARRAY_LEN];
};

/* (Q)SFP module access definitions */
#define I40E_I2C_EEPROM_DEV_ADDR	0xA0
#define I40E_I2C_EEPROM_DEV_ADDR2	0xA2
#define I40E_MODULE_TYPE_ADDR		0x00
#define I40E_MODULE_REVISION_ADDR	0x01
#define I40E_MODULE_SFF_8472_COMP	0x5E
#define I40E_MODULE_SFF_8472_SWAP	0x5C
#define I40E_MODULE_SFF_ADDR_MODE	0x04
#define I40E_MODULE_SFF_DIAG_CAPAB	0x40
#define I40E_MODULE_TYPE_QSFP_PLUS	0x0D
#define I40E_MODULE_TYPE_QSFP28		0x11
#define I40E_MODULE_QSFP_MAX_LEN	640

/* PCI bus types */
enum i40e_bus_type {
	i40e_bus_type_unknown = 0,
	i40e_bus_type_pci,
	i40e_bus_type_pcix,
	i40e_bus_type_pci_express,
	i40e_bus_type_reserved
};

/* PCI bus speeds */
enum i40e_bus_speed {
	i40e_bus_speed_unknown	= 0,
	i40e_bus_speed_33	= 33,
	i40e_bus_speed_66	= 66,
	i40e_bus_speed_100	= 100,
	i40e_bus_speed_120	= 120,
	i40e_bus_speed_133	= 133,
	i40e_bus_speed_2500	= 2500,
	i40e_bus_speed_5000	= 5000,
	i40e_bus_speed_8000	= 8000,
	i40e_bus_speed_reserved
};

/* PCI bus widths */
enum i40e_bus_width {
	i40e_bus_width_unknown	= 0,
	i40e_bus_width_pcie_x1	= 1,
	i40e_bus_width_pcie_x2	= 2,
	i40e_bus_width_pcie_x4	= 4,
	i40e_bus_width_pcie_x8	= 8,
	i40e_bus_width_32	= 32,
	i40e_bus_width_64	= 64,
	i40e_bus_width_reserved
};

/* Bus parameters */
struct i40e_bus_info {
	enum i40e_bus_speed speed;
	enum i40e_bus_width width;
	enum i40e_bus_type type;

	u16 func;
	u16 device;
	u16 lan_id;
	u16 bus_id;
};

/* Flow control (FC) parameters */
struct i40e_fc_info {
	enum i40e_fc_mode lwrrent_mode; /* FC mode in effect */
	enum i40e_fc_mode requested_mode; /* FC mode requested by caller */
};

#define I40E_MAX_TRAFFIC_CLASS		8
#define I40E_MAX_USER_PRIORITY		8
#define I40E_DCBX_MAX_APPS		32
#define I40E_LLDPDU_SIZE		1500
#define I40E_TLV_STATUS_OPER		0x1
#define I40E_TLV_STATUS_SYNC		0x2
#define I40E_TLV_STATUS_ERR		0x4
#define I40E_CEE_OPER_MAX_APPS		3
#define I40E_APP_PROTOID_FCOE		0x8906
#define I40E_APP_PROTOID_ISCSI		0x0cbc
#define I40E_APP_PROTOID_FIP		0x8914
#define I40E_APP_SEL_ETHTYPE		0x1
#define I40E_APP_SEL_TCPIP		0x2
#define I40E_CEE_APP_SEL_ETHTYPE	0x0
#define I40E_CEE_APP_SEL_TCPIP		0x1

/* CEE or IEEE 802.1Qaz ETS Configuration data */
struct i40e_dcb_ets_config {
	u8 willing;
	u8 cbs;
	u8 maxtcs;
	u8 prioritytable[I40E_MAX_TRAFFIC_CLASS];
	u8 tcbwtable[I40E_MAX_TRAFFIC_CLASS];
	u8 tsatable[I40E_MAX_TRAFFIC_CLASS];
};

/* CEE or IEEE 802.1Qaz PFC Configuration data */
struct i40e_dcb_pfc_config {
	u8 willing;
	u8 mbc;
	u8 pfccap;
	u8 pfcenable;
};

/* CEE or IEEE 802.1Qaz Application Priority data */
struct i40e_dcb_app_priority_table {
	u8  priority;
	u8  selector;
	u16 protocolid;
};

struct i40e_dcbx_config {
	u8  dcbx_mode;
#define I40E_DCBX_MODE_CEE	0x1
#define I40E_DCBX_MODE_IEEE	0x2
	u8  app_mode;
#define I40E_DCBX_APPS_NON_WILLING	0x1
	u32 numapps;
	u32 tlv_status; /* CEE mode TLV status */
	struct i40e_dcb_ets_config etscfg;
	struct i40e_dcb_ets_config etsrec;
	struct i40e_dcb_pfc_config pfc;
	struct i40e_dcb_app_priority_table app[I40E_DCBX_MAX_APPS];
};

/* Port hardware description */
struct i40e_hw {
	u8 *hw_addr;
	void *back;

	/* subsystem structs */
	struct i40e_phy_info phy;
	struct i40e_mac_info mac;
	struct i40e_bus_info bus;
	struct i40e_lwm_info lwm;
	struct i40e_fc_info fc;

	/* switch device is used to get link status when i40e is in ipn3ke */
	struct rte_eth_dev *switch_dev;

	/* pci info */
	u16 device_id;
	u16 vendor_id;
	u16 subsystem_device_id;
	u16 subsystem_vendor_id;
	u8 revision_id;
	u8 port;
	bool adapter_stopped;
	bool adapter_closed;

	/* capabilities for entire device and PCI func */
	struct i40e_hw_capabilities dev_caps;
	struct i40e_hw_capabilities func_caps;

	/* Flow Director shared filter space */
	u16 fdir_shared_filter_count;

	/* device profile info */
	u8  pf_id;
	u16 main_vsi_seid;

	/* for multi-function MACs */
	u16 partition_id;
	u16 num_partitions;
	u16 num_ports;

	/* Closest numa node to the device */
	u16 numa_node;

	/* Admin Queue info */
	struct i40e_adminq_info aq;

	/* state of lwm update process */
	enum i40e_lwmupd_state lwmupd_state;
	struct i40e_aq_desc lwm_wb_desc;
	struct i40e_aq_desc lwm_aq_event_desc;
	struct i40e_virt_mem lwm_buff;
	bool lwm_release_on_done;
	u16 lwm_wait_opcode;

	/* HMC info */
	struct i40e_hmc_info hmc; /* HMC info struct */

	/* LLDP/DCBX Status */
	u16 dcbx_status;

	/* DCBX info */
	struct i40e_dcbx_config local_dcbx_config; /* Oper/Local Cfg */
	struct i40e_dcbx_config remote_dcbx_config; /* Peer Cfg */
	struct i40e_dcbx_config desired_dcbx_config; /* CEE Desired Cfg */

	/* WoL and proxy support */
	u16 num_wol_proxy_filters;
	u16 wol_proxy_vsi_seid;

#define I40E_HW_FLAG_AQ_SRCTL_ACCESS_ENABLE BIT_ULL(0)
#define I40E_HW_FLAG_802_1AD_CAPABLE        BIT_ULL(1)
#define I40E_HW_FLAG_AQ_PHY_ACCESS_CAPABLE  BIT_ULL(2)
#define I40E_HW_FLAG_LWM_READ_REQUIRES_LOCK BIT_ULL(3)
#define I40E_HW_FLAG_FW_LLDP_STOPPABLE	    BIT_ULL(4)
#define I40E_HW_FLAG_FW_LLDP_PERSISTENT     BIT_ULL(5)
#define I40E_HW_FLAG_AQ_PHY_ACCESS_EXTENDED BIT_ULL(6)
#define I40E_HW_FLAG_DROP_MODE		    BIT_ULL(7)
#define I40E_HW_FLAG_X722_FEC_REQUEST_CAPABLE BIT_ULL(8)
	u64 flags;

	/* Used in set switch config AQ command */
	u16 switch_tag;
	u16 first_tag;
	u16 second_tag;

	/* LWMUpdate features */
	struct i40e_lwmupd_features lwmupd_features;

	/* debug mask */
	u32 debug_mask;
	char err_str[16];
};

STATIC INLINE bool i40e_is_vf(struct i40e_hw *hw)
{
	return (hw->mac.type == I40E_MAC_VF ||
		hw->mac.type == I40E_MAC_X722_VF);
}

struct i40e_driver_version {
	u8 major_version;
	u8 minor_version;
	u8 build_version;
	u8 subbuild_version;
	u8 driver_string[32];
};

/* RX Descriptors */
union i40e_16byte_rx_desc {
	struct {
		__le64 pkt_addr; /* Packet buffer address */
		__le64 hdr_addr; /* Header buffer address */
	} read;
	struct {
		struct {
			struct {
				union {
					__le16 mirroring_status;
					__le16 fcoe_ctx_id;
				} mirr_fcoe;
				__le16 l2tag1;
			} lo_dword;
			union {
				__le32 rss; /* RSS Hash */
				__le32 fd_id; /* Flow director filter id */
				__le32 fcoe_param; /* FCoE DDP Context id */
			} hi_dword;
		} qword0;
		struct {
			/* ext status/error/pktype/length */
			__le64 status_error_len;
		} qword1;
	} wb;  /* writeback */
};

union i40e_32byte_rx_desc {
	struct {
		__le64  pkt_addr; /* Packet buffer address */
		__le64  hdr_addr; /* Header buffer address */
			/* bit 0 of hdr_buffer_addr is DD bit */
		__le64  rsvd1;
		__le64  rsvd2;
	} read;
	struct {
		struct {
			struct {
				union {
					__le16 mirroring_status;
					__le16 fcoe_ctx_id;
				} mirr_fcoe;
				__le16 l2tag1;
			} lo_dword;
			union {
				__le32 rss; /* RSS Hash */
				__le32 fcoe_param; /* FCoE DDP Context id */
				/* Flow director filter id in case of
				 * Programming status desc WB
				 */
				__le32 fd_id;
			} hi_dword;
		} qword0;
		struct {
			/* status/error/pktype/length */
			__le64 status_error_len;
		} qword1;
		struct {
			__le16 ext_status; /* extended status */
			__le16 rsvd;
			__le16 l2tag2_1;
			__le16 l2tag2_2;
		} qword2;
		struct {
			union {
				__le32 flex_bytes_lo;
				__le32 pe_status;
			} lo_dword;
			union {
				__le32 flex_bytes_hi;
				__le32 fd_id;
			} hi_dword;
		} qword3;
	} wb;  /* writeback */
};

#define I40E_RXD_QW0_MIRROR_STATUS_SHIFT	8
#define I40E_RXD_QW0_MIRROR_STATUS_MASK	(0x3FUL << \
					 I40E_RXD_QW0_MIRROR_STATUS_SHIFT)
#define I40E_RXD_QW0_FCOEINDX_SHIFT	0
#define I40E_RXD_QW0_FCOEINDX_MASK	(0xFFFUL << \
					 I40E_RXD_QW0_FCOEINDX_SHIFT)

enum i40e_rx_desc_status_bits {
	/* Note: These are predefined bit offsets */
	I40E_RX_DESC_STATUS_DD_SHIFT		= 0,
	I40E_RX_DESC_STATUS_EOF_SHIFT		= 1,
	I40E_RX_DESC_STATUS_L2TAG1P_SHIFT	= 2,
	I40E_RX_DESC_STATUS_L3L4P_SHIFT		= 3,
	I40E_RX_DESC_STATUS_CRCP_SHIFT		= 4,
	I40E_RX_DESC_STATUS_TSYNINDX_SHIFT	= 5, /* 2 BITS */
	I40E_RX_DESC_STATUS_TSYLWALID_SHIFT	= 7,
	I40E_RX_DESC_STATUS_EXT_UDP_0_SHIFT	= 8,

	I40E_RX_DESC_STATUS_UMBCAST_SHIFT	= 9, /* 2 BITS */
	I40E_RX_DESC_STATUS_FLM_SHIFT		= 11,
	I40E_RX_DESC_STATUS_FLTSTAT_SHIFT	= 12, /* 2 BITS */
	I40E_RX_DESC_STATUS_LPBK_SHIFT		= 14,
	I40E_RX_DESC_STATUS_IPV6EXADD_SHIFT	= 15,
	I40E_RX_DESC_STATUS_RESERVED2_SHIFT	= 16, /* 2 BITS */
	I40E_RX_DESC_STATUS_INT_UDP_0_SHIFT	= 18,
	I40E_RX_DESC_STATUS_LAST /* this entry must be last!!! */
};

#define I40E_RXD_QW1_STATUS_SHIFT	0
#define I40E_RXD_QW1_STATUS_MASK	((BIT(I40E_RX_DESC_STATUS_LAST) - 1) << \
					 I40E_RXD_QW1_STATUS_SHIFT)

#define I40E_RXD_QW1_STATUS_TSYNINDX_SHIFT   I40E_RX_DESC_STATUS_TSYNINDX_SHIFT
#define I40E_RXD_QW1_STATUS_TSYNINDX_MASK	(0x3UL << \
					     I40E_RXD_QW1_STATUS_TSYNINDX_SHIFT)

#define I40E_RXD_QW1_STATUS_TSYLWALID_SHIFT  I40E_RX_DESC_STATUS_TSYLWALID_SHIFT
#define I40E_RXD_QW1_STATUS_TSYLWALID_MASK   BIT_ULL(I40E_RXD_QW1_STATUS_TSYLWALID_SHIFT)

#define I40E_RXD_QW1_STATUS_UMBCAST_SHIFT	I40E_RX_DESC_STATUS_UMBCAST
#define I40E_RXD_QW1_STATUS_UMBCAST_MASK	(0x3UL << \
					 I40E_RXD_QW1_STATUS_UMBCAST_SHIFT)

enum i40e_rx_desc_fltstat_values {
	I40E_RX_DESC_FLTSTAT_NO_DATA	= 0,
	I40E_RX_DESC_FLTSTAT_RSV_FD_ID	= 1, /* 16byte desc? FD_ID : RSV */
	I40E_RX_DESC_FLTSTAT_RSV	= 2,
	I40E_RX_DESC_FLTSTAT_RSS_HASH	= 3,
};

#define I40E_RXD_PACKET_TYPE_UNICAST	0
#define I40E_RXD_PACKET_TYPE_MULTICAST	1
#define I40E_RXD_PACKET_TYPE_BROADCAST	2
#define I40E_RXD_PACKET_TYPE_MIRRORED	3

#define I40E_RXD_QW1_ERROR_SHIFT	19
#define I40E_RXD_QW1_ERROR_MASK		(0xFFUL << I40E_RXD_QW1_ERROR_SHIFT)

enum i40e_rx_desc_error_bits {
	/* Note: These are predefined bit offsets */
	I40E_RX_DESC_ERROR_RXE_SHIFT		= 0,
	I40E_RX_DESC_ERROR_RECIPE_SHIFT		= 1,
	I40E_RX_DESC_ERROR_HBO_SHIFT		= 2,
	I40E_RX_DESC_ERROR_L3L4E_SHIFT		= 3, /* 3 BITS */
	I40E_RX_DESC_ERROR_IPE_SHIFT		= 3,
	I40E_RX_DESC_ERROR_L4E_SHIFT		= 4,
	I40E_RX_DESC_ERROR_EIPE_SHIFT		= 5,
	I40E_RX_DESC_ERROR_OVERSIZE_SHIFT	= 6,
	I40E_RX_DESC_ERROR_PPRS_SHIFT		= 7
};

enum i40e_rx_desc_error_l3l4e_fcoe_masks {
	I40E_RX_DESC_ERROR_L3L4E_NONE		= 0,
	I40E_RX_DESC_ERROR_L3L4E_PROT		= 1,
	I40E_RX_DESC_ERROR_L3L4E_FC		= 2,
	I40E_RX_DESC_ERROR_L3L4E_DMAC_ERR	= 3,
	I40E_RX_DESC_ERROR_L3L4E_DMAC_WARN	= 4
};

#define I40E_RXD_QW1_PTYPE_SHIFT	30
#define I40E_RXD_QW1_PTYPE_MASK		(0xFFULL << I40E_RXD_QW1_PTYPE_SHIFT)

/* Packet type non-ip values */
enum i40e_rx_l2_ptype {
	I40E_RX_PTYPE_L2_RESERVED			= 0,
	I40E_RX_PTYPE_L2_MAC_PAY2			= 1,
	I40E_RX_PTYPE_L2_TIMESYNC_PAY2			= 2,
	I40E_RX_PTYPE_L2_FIP_PAY2			= 3,
	I40E_RX_PTYPE_L2_OUI_PAY2			= 4,
	I40E_RX_PTYPE_L2_MACCNTRL_PAY2			= 5,
	I40E_RX_PTYPE_L2_LLDP_PAY2			= 6,
	I40E_RX_PTYPE_L2_ECP_PAY2			= 7,
	I40E_RX_PTYPE_L2_EVB_PAY2			= 8,
	I40E_RX_PTYPE_L2_QCN_PAY2			= 9,
	I40E_RX_PTYPE_L2_EAPOL_PAY2			= 10,
	I40E_RX_PTYPE_L2_ARP				= 11,
	I40E_RX_PTYPE_L2_FCOE_PAY3			= 12,
	I40E_RX_PTYPE_L2_FCOE_FCDATA_PAY3		= 13,
	I40E_RX_PTYPE_L2_FCOE_FCRDY_PAY3		= 14,
	I40E_RX_PTYPE_L2_FCOE_FCRSP_PAY3		= 15,
	I40E_RX_PTYPE_L2_FCOE_FCOTHER_PA		= 16,
	I40E_RX_PTYPE_L2_FCOE_VFT_PAY3			= 17,
	I40E_RX_PTYPE_L2_FCOE_VFT_FCDATA		= 18,
	I40E_RX_PTYPE_L2_FCOE_VFT_FCRDY			= 19,
	I40E_RX_PTYPE_L2_FCOE_VFT_FCRSP			= 20,
	I40E_RX_PTYPE_L2_FCOE_VFT_FCOTHER		= 21,
	I40E_RX_PTYPE_GRENAT4_MAC_PAY3			= 58,
	I40E_RX_PTYPE_GRENAT4_MACVLAN_IPV6_ICMP_PAY4	= 87,
	I40E_RX_PTYPE_GRENAT6_MAC_PAY3			= 124,
	I40E_RX_PTYPE_GRENAT6_MACVLAN_IPV6_ICMP_PAY4	= 153,
	I40E_RX_PTYPE_PARSER_ABORTED			= 255
};

struct i40e_rx_ptype_decoded {
	u32 ptype:8;
	u32 known:1;
	u32 outer_ip:1;
	u32 outer_ip_ver:1;
	u32 outer_frag:1;
	u32 tunnel_type:3;
	u32 tunnel_end_prot:2;
	u32 tunnel_end_frag:1;
	u32 inner_prot:4;
	u32 payload_layer:3;
};

enum i40e_rx_ptype_outer_ip {
	I40E_RX_PTYPE_OUTER_L2	= 0,
	I40E_RX_PTYPE_OUTER_IP	= 1
};

enum i40e_rx_ptype_outer_ip_ver {
	I40E_RX_PTYPE_OUTER_NONE	= 0,
	I40E_RX_PTYPE_OUTER_IPV4	= 0,
	I40E_RX_PTYPE_OUTER_IPV6	= 1
};

enum i40e_rx_ptype_outer_fragmented {
	I40E_RX_PTYPE_NOT_FRAG	= 0,
	I40E_RX_PTYPE_FRAG	= 1
};

enum i40e_rx_ptype_tunnel_type {
	I40E_RX_PTYPE_TUNNEL_NONE		= 0,
	I40E_RX_PTYPE_TUNNEL_IP_IP		= 1,
	I40E_RX_PTYPE_TUNNEL_IP_GRENAT		= 2,
	I40E_RX_PTYPE_TUNNEL_IP_GRENAT_MAC	= 3,
	I40E_RX_PTYPE_TUNNEL_IP_GRENAT_MAC_VLAN	= 4,
};

enum i40e_rx_ptype_tunnel_end_prot {
	I40E_RX_PTYPE_TUNNEL_END_NONE	= 0,
	I40E_RX_PTYPE_TUNNEL_END_IPV4	= 1,
	I40E_RX_PTYPE_TUNNEL_END_IPV6	= 2,
};

enum i40e_rx_ptype_inner_prot {
	I40E_RX_PTYPE_INNER_PROT_NONE		= 0,
	I40E_RX_PTYPE_INNER_PROT_UDP		= 1,
	I40E_RX_PTYPE_INNER_PROT_TCP		= 2,
	I40E_RX_PTYPE_INNER_PROT_SCTP		= 3,
	I40E_RX_PTYPE_INNER_PROT_ICMP		= 4,
	I40E_RX_PTYPE_INNER_PROT_TIMESYNC	= 5
};

enum i40e_rx_ptype_payload_layer {
	I40E_RX_PTYPE_PAYLOAD_LAYER_NONE	= 0,
	I40E_RX_PTYPE_PAYLOAD_LAYER_PAY2	= 1,
	I40E_RX_PTYPE_PAYLOAD_LAYER_PAY3	= 2,
	I40E_RX_PTYPE_PAYLOAD_LAYER_PAY4	= 3,
};

#define I40E_RX_PTYPE_BIT_MASK		0x0FFFFFFF
#define I40E_RX_PTYPE_SHIFT		56

#define I40E_RXD_QW1_LENGTH_PBUF_SHIFT	38
#define I40E_RXD_QW1_LENGTH_PBUF_MASK	(0x3FFFULL << \
					 I40E_RXD_QW1_LENGTH_PBUF_SHIFT)

#define I40E_RXD_QW1_LENGTH_HBUF_SHIFT	52
#define I40E_RXD_QW1_LENGTH_HBUF_MASK	(0x7FFULL << \
					 I40E_RXD_QW1_LENGTH_HBUF_SHIFT)

#define I40E_RXD_QW1_LENGTH_SPH_SHIFT	63
#define I40E_RXD_QW1_LENGTH_SPH_MASK	BIT_ULL(I40E_RXD_QW1_LENGTH_SPH_SHIFT)

#define I40E_RXD_QW1_NEXTP_SHIFT	38
#define I40E_RXD_QW1_NEXTP_MASK		(0x1FFFULL << I40E_RXD_QW1_NEXTP_SHIFT)

#define I40E_RXD_QW2_EXT_STATUS_SHIFT	0
#define I40E_RXD_QW2_EXT_STATUS_MASK	(0xFFFFFUL << \
					 I40E_RXD_QW2_EXT_STATUS_SHIFT)

enum i40e_rx_desc_ext_status_bits {
	/* Note: These are predefined bit offsets */
	I40E_RX_DESC_EXT_STATUS_L2TAG2P_SHIFT	= 0,
	I40E_RX_DESC_EXT_STATUS_L2TAG3P_SHIFT	= 1,
	I40E_RX_DESC_EXT_STATUS_FLEXBL_SHIFT	= 2, /* 2 BITS */
	I40E_RX_DESC_EXT_STATUS_FLEXBH_SHIFT	= 4, /* 2 BITS */
	I40E_RX_DESC_EXT_STATUS_FDLONGB_SHIFT	= 9,
	I40E_RX_DESC_EXT_STATUS_FCOELONGB_SHIFT	= 10,
	I40E_RX_DESC_EXT_STATUS_PELONGB_SHIFT	= 11,
};

#define I40E_RXD_QW2_L2TAG2_SHIFT	0
#define I40E_RXD_QW2_L2TAG2_MASK	(0xFFFFUL << I40E_RXD_QW2_L2TAG2_SHIFT)

#define I40E_RXD_QW2_L2TAG3_SHIFT	16
#define I40E_RXD_QW2_L2TAG3_MASK	(0xFFFFUL << I40E_RXD_QW2_L2TAG3_SHIFT)

enum i40e_rx_desc_pe_status_bits {
	/* Note: These are predefined bit offsets */
	I40E_RX_DESC_PE_STATUS_QPID_SHIFT	= 0, /* 18 BITS */
	I40E_RX_DESC_PE_STATUS_L4PORT_SHIFT	= 0, /* 16 BITS */
	I40E_RX_DESC_PE_STATUS_IPINDEX_SHIFT	= 16, /* 8 BITS */
	I40E_RX_DESC_PE_STATUS_QPIDHIT_SHIFT	= 24,
	I40E_RX_DESC_PE_STATUS_APBVTHIT_SHIFT	= 25,
	I40E_RX_DESC_PE_STATUS_PORTV_SHIFT	= 26,
	I40E_RX_DESC_PE_STATUS_URG_SHIFT	= 27,
	I40E_RX_DESC_PE_STATUS_IPFRAG_SHIFT	= 28,
	I40E_RX_DESC_PE_STATUS_IPOPT_SHIFT	= 29
};

#define I40E_RX_PROG_STATUS_DESC_LENGTH_SHIFT		38
#define I40E_RX_PROG_STATUS_DESC_LENGTH			0x2000000

#define I40E_RX_PROG_STATUS_DESC_QW1_PROGID_SHIFT	2
#define I40E_RX_PROG_STATUS_DESC_QW1_PROGID_MASK	(0x7UL << \
				I40E_RX_PROG_STATUS_DESC_QW1_PROGID_SHIFT)

#define I40E_RX_PROG_STATUS_DESC_QW1_STATUS_SHIFT	0
#define I40E_RX_PROG_STATUS_DESC_QW1_STATUS_MASK	(0x7FFFUL << \
				I40E_RX_PROG_STATUS_DESC_QW1_STATUS_SHIFT)

#define I40E_RX_PROG_STATUS_DESC_QW1_ERROR_SHIFT	19
#define I40E_RX_PROG_STATUS_DESC_QW1_ERROR_MASK		(0x3FUL << \
				I40E_RX_PROG_STATUS_DESC_QW1_ERROR_SHIFT)

enum i40e_rx_prog_status_desc_status_bits {
	/* Note: These are predefined bit offsets */
	I40E_RX_PROG_STATUS_DESC_DD_SHIFT	= 0,
	I40E_RX_PROG_STATUS_DESC_PROG_ID_SHIFT	= 2 /* 3 BITS */
};

enum i40e_rx_prog_status_desc_prog_id_masks {
	I40E_RX_PROG_STATUS_DESC_FD_FILTER_STATUS	= 1,
	I40E_RX_PROG_STATUS_DESC_FCOE_CTXT_PROG_STATUS	= 2,
	I40E_RX_PROG_STATUS_DESC_FCOE_CTXT_ILWL_STATUS	= 4,
};

enum i40e_rx_prog_status_desc_error_bits {
	/* Note: These are predefined bit offsets */
	I40E_RX_PROG_STATUS_DESC_FD_TBL_FULL_SHIFT	= 0,
	I40E_RX_PROG_STATUS_DESC_NO_FD_ENTRY_SHIFT	= 1,
	I40E_RX_PROG_STATUS_DESC_FCOE_TBL_FULL_SHIFT	= 2,
	I40E_RX_PROG_STATUS_DESC_FCOE_CONFLICT_SHIFT	= 3
};

#define I40E_TWO_BIT_MASK	0x3
#define I40E_THREE_BIT_MASK	0x7
#define I40E_FOUR_BIT_MASK	0xF
#define I40E_EIGHTEEN_BIT_MASK	0x3FFFF

/* TX Descriptor */
struct i40e_tx_desc {
	__le64 buffer_addr; /* Address of descriptor's data buf */
	__le64 cmd_type_offset_bsz;
};

#define I40E_TXD_QW1_DTYPE_SHIFT	0
#define I40E_TXD_QW1_DTYPE_MASK		(0xFUL << I40E_TXD_QW1_DTYPE_SHIFT)

enum i40e_tx_desc_dtype_value {
	I40E_TX_DESC_DTYPE_DATA		= 0x0,
	I40E_TX_DESC_DTYPE_NOP		= 0x1, /* same as Context desc */
	I40E_TX_DESC_DTYPE_CONTEXT	= 0x1,
	I40E_TX_DESC_DTYPE_FCOE_CTX	= 0x2,
	I40E_TX_DESC_DTYPE_FILTER_PROG	= 0x8,
	I40E_TX_DESC_DTYPE_DDP_CTX	= 0x9,
	I40E_TX_DESC_DTYPE_FLEX_DATA	= 0xB,
	I40E_TX_DESC_DTYPE_FLEX_CTX_1	= 0xC,
	I40E_TX_DESC_DTYPE_FLEX_CTX_2	= 0xD,
	I40E_TX_DESC_DTYPE_DESC_DONE	= 0xF
};

#define I40E_TXD_QW1_CMD_SHIFT	4
#define I40E_TXD_QW1_CMD_MASK	(0x3FFUL << I40E_TXD_QW1_CMD_SHIFT)

enum i40e_tx_desc_cmd_bits {
	I40E_TX_DESC_CMD_EOP			= 0x0001,
	I40E_TX_DESC_CMD_RS			= 0x0002,
	I40E_TX_DESC_CMD_ICRC			= 0x0004,
	I40E_TX_DESC_CMD_IL2TAG1		= 0x0008,
	I40E_TX_DESC_CMD_DUMMY			= 0x0010,
	I40E_TX_DESC_CMD_IIPT_NONIP		= 0x0000, /* 2 BITS */
	I40E_TX_DESC_CMD_IIPT_IPV6		= 0x0020, /* 2 BITS */
	I40E_TX_DESC_CMD_IIPT_IPV4		= 0x0040, /* 2 BITS */
	I40E_TX_DESC_CMD_IIPT_IPV4_CSUM		= 0x0060, /* 2 BITS */
	I40E_TX_DESC_CMD_FCOET			= 0x0080,
	I40E_TX_DESC_CMD_L4T_EOFT_UNK		= 0x0000, /* 2 BITS */
	I40E_TX_DESC_CMD_L4T_EOFT_TCP		= 0x0100, /* 2 BITS */
	I40E_TX_DESC_CMD_L4T_EOFT_SCTP		= 0x0200, /* 2 BITS */
	I40E_TX_DESC_CMD_L4T_EOFT_UDP		= 0x0300, /* 2 BITS */
	I40E_TX_DESC_CMD_L4T_EOFT_EOF_N		= 0x0000, /* 2 BITS */
	I40E_TX_DESC_CMD_L4T_EOFT_EOF_T		= 0x0100, /* 2 BITS */
	I40E_TX_DESC_CMD_L4T_EOFT_EOF_NI	= 0x0200, /* 2 BITS */
	I40E_TX_DESC_CMD_L4T_EOFT_EOF_A		= 0x0300, /* 2 BITS */
};

#define I40E_TXD_QW1_OFFSET_SHIFT	16
#define I40E_TXD_QW1_OFFSET_MASK	(0x3FFFFULL << \
					 I40E_TXD_QW1_OFFSET_SHIFT)

enum i40e_tx_desc_length_fields {
	/* Note: These are predefined bit offsets */
	I40E_TX_DESC_LENGTH_MACLEN_SHIFT	= 0, /* 7 BITS */
	I40E_TX_DESC_LENGTH_IPLEN_SHIFT		= 7, /* 7 BITS */
	I40E_TX_DESC_LENGTH_L4_FC_LEN_SHIFT	= 14 /* 4 BITS */
};

#define I40E_TXD_QW1_MACLEN_MASK (0x7FUL << I40E_TX_DESC_LENGTH_MACLEN_SHIFT)
#define I40E_TXD_QW1_IPLEN_MASK  (0x7FUL << I40E_TX_DESC_LENGTH_IPLEN_SHIFT)
#define I40E_TXD_QW1_L4LEN_MASK  (0xFUL << I40E_TX_DESC_LENGTH_L4_FC_LEN_SHIFT)
#define I40E_TXD_QW1_FCLEN_MASK  (0xFUL << I40E_TX_DESC_LENGTH_L4_FC_LEN_SHIFT)

#define I40E_TXD_QW1_TX_BUF_SZ_SHIFT	34
#define I40E_TXD_QW1_TX_BUF_SZ_MASK	(0x3FFFULL << \
					 I40E_TXD_QW1_TX_BUF_SZ_SHIFT)

#define I40E_TXD_QW1_L2TAG1_SHIFT	48
#define I40E_TXD_QW1_L2TAG1_MASK	(0xFFFFULL << I40E_TXD_QW1_L2TAG1_SHIFT)

/* Context descriptors */
struct i40e_tx_context_desc {
	__le32 tunneling_params;
	__le16 l2tag2;
	__le16 rsvd;
	__le64 type_cmd_tso_mss;
};

#define I40E_TXD_CTX_QW1_DTYPE_SHIFT	0
#define I40E_TXD_CTX_QW1_DTYPE_MASK	(0xFUL << I40E_TXD_CTX_QW1_DTYPE_SHIFT)

#define I40E_TXD_CTX_QW1_CMD_SHIFT	4
#define I40E_TXD_CTX_QW1_CMD_MASK	(0xFFFFUL << I40E_TXD_CTX_QW1_CMD_SHIFT)

enum i40e_tx_ctx_desc_cmd_bits {
	I40E_TX_CTX_DESC_TSO		= 0x01,
	I40E_TX_CTX_DESC_TSYN		= 0x02,
	I40E_TX_CTX_DESC_IL2TAG2	= 0x04,
	I40E_TX_CTX_DESC_IL2TAG2_IL2H	= 0x08,
	I40E_TX_CTX_DESC_SWTCH_NOTAG	= 0x00,
	I40E_TX_CTX_DESC_SWTCH_UPLINK	= 0x10,
	I40E_TX_CTX_DESC_SWTCH_LOCAL	= 0x20,
	I40E_TX_CTX_DESC_SWTCH_VSI	= 0x30,
	I40E_TX_CTX_DESC_SWPE		= 0x40
};

#define I40E_TXD_CTX_QW1_TSO_LEN_SHIFT	30
#define I40E_TXD_CTX_QW1_TSO_LEN_MASK	(0x3FFFFULL << \
					 I40E_TXD_CTX_QW1_TSO_LEN_SHIFT)

#define I40E_TXD_CTX_QW1_MSS_SHIFT	50
#define I40E_TXD_CTX_QW1_MSS_MASK	(0x3FFFULL << \
					 I40E_TXD_CTX_QW1_MSS_SHIFT)

#define I40E_TXD_CTX_QW1_VSI_SHIFT	50
#define I40E_TXD_CTX_QW1_VSI_MASK	(0x1FFULL << I40E_TXD_CTX_QW1_VSI_SHIFT)

#define I40E_TXD_CTX_QW0_EXT_IP_SHIFT	0
#define I40E_TXD_CTX_QW0_EXT_IP_MASK	(0x3ULL << \
					 I40E_TXD_CTX_QW0_EXT_IP_SHIFT)

enum i40e_tx_ctx_desc_eipt_offload {
	I40E_TX_CTX_EXT_IP_NONE		= 0x0,
	I40E_TX_CTX_EXT_IP_IPV6		= 0x1,
	I40E_TX_CTX_EXT_IP_IPV4_NO_CSUM	= 0x2,
	I40E_TX_CTX_EXT_IP_IPV4		= 0x3
};

#define I40E_TXD_CTX_QW0_EXT_IPLEN_SHIFT	2
#define I40E_TXD_CTX_QW0_EXT_IPLEN_MASK	(0x3FULL << \
					 I40E_TXD_CTX_QW0_EXT_IPLEN_SHIFT)

#define I40E_TXD_CTX_QW0_NATT_SHIFT	9
#define I40E_TXD_CTX_QW0_NATT_MASK	(0x3ULL << I40E_TXD_CTX_QW0_NATT_SHIFT)

#define I40E_TXD_CTX_UDP_TUNNELING	BIT_ULL(I40E_TXD_CTX_QW0_NATT_SHIFT)
#define I40E_TXD_CTX_GRE_TUNNELING	(0x2ULL << I40E_TXD_CTX_QW0_NATT_SHIFT)

#define I40E_TXD_CTX_QW0_EIP_NOINC_SHIFT	11
#define I40E_TXD_CTX_QW0_EIP_NOINC_MASK	BIT_ULL(I40E_TXD_CTX_QW0_EIP_NOINC_SHIFT)

#define I40E_TXD_CTX_EIP_NOINC_IPID_CONST	I40E_TXD_CTX_QW0_EIP_NOINC_MASK

#define I40E_TXD_CTX_QW0_NATLEN_SHIFT	12
#define I40E_TXD_CTX_QW0_NATLEN_MASK	(0X7FULL << \
					 I40E_TXD_CTX_QW0_NATLEN_SHIFT)

#define I40E_TXD_CTX_QW0_DECTTL_SHIFT	19
#define I40E_TXD_CTX_QW0_DECTTL_MASK	(0xFULL << \
					 I40E_TXD_CTX_QW0_DECTTL_SHIFT)

#define I40E_TXD_CTX_QW0_L4T_CS_SHIFT	23
#define I40E_TXD_CTX_QW0_L4T_CS_MASK	BIT_ULL(I40E_TXD_CTX_QW0_L4T_CS_SHIFT)
struct i40e_nop_desc {
	__le64 rsvd;
	__le64 dtype_cmd;
};

#define I40E_TXD_NOP_QW1_DTYPE_SHIFT	0
#define I40E_TXD_NOP_QW1_DTYPE_MASK	(0xFUL << I40E_TXD_NOP_QW1_DTYPE_SHIFT)

#define I40E_TXD_NOP_QW1_CMD_SHIFT	4
#define I40E_TXD_NOP_QW1_CMD_MASK	(0x7FUL << I40E_TXD_NOP_QW1_CMD_SHIFT)

enum i40e_tx_nop_desc_cmd_bits {
	/* Note: These are predefined bit offsets */
	I40E_TX_NOP_DESC_EOP_SHIFT	= 0,
	I40E_TX_NOP_DESC_RS_SHIFT	= 1,
	I40E_TX_NOP_DESC_RSV_SHIFT	= 2 /* 5 bits */
};

struct i40e_filter_program_desc {
	__le32 qindex_flex_ptype_vsi;
	__le32 rsvd;
	__le32 dtype_cmd_cntindex;
	__le32 fd_id;
};
#define I40E_TXD_FLTR_QW0_QINDEX_SHIFT	0
#define I40E_TXD_FLTR_QW0_QINDEX_MASK	(0x7FFUL << \
					 I40E_TXD_FLTR_QW0_QINDEX_SHIFT)
#define I40E_TXD_FLTR_QW0_FLEXOFF_SHIFT	11
#define I40E_TXD_FLTR_QW0_FLEXOFF_MASK	(0x7UL << \
					 I40E_TXD_FLTR_QW0_FLEXOFF_SHIFT)
#define I40E_TXD_FLTR_QW0_PCTYPE_SHIFT	17
#define I40E_TXD_FLTR_QW0_PCTYPE_MASK	(0x3FUL << \
					 I40E_TXD_FLTR_QW0_PCTYPE_SHIFT)

/* Packet Classifier Types for filters */
enum i40e_filter_pctype {
	/* Note: Values 0-28 are reserved for future use.
	 * Value 29, 30, 32 are not supported on XL710 and X710.
	 */
	I40E_FILTER_PCTYPE_NONF_UNICAST_IPV4_UDP	= 29,
	I40E_FILTER_PCTYPE_NONF_MULTICAST_IPV4_UDP	= 30,
	I40E_FILTER_PCTYPE_NONF_IPV4_UDP		= 31,
	I40E_FILTER_PCTYPE_NONF_IPV4_TCP_SYN_NO_ACK	= 32,
	I40E_FILTER_PCTYPE_NONF_IPV4_TCP		= 33,
	I40E_FILTER_PCTYPE_NONF_IPV4_SCTP		= 34,
	I40E_FILTER_PCTYPE_NONF_IPV4_OTHER		= 35,
	I40E_FILTER_PCTYPE_FRAG_IPV4			= 36,
	/* Note: Values 37-38 are reserved for future use.
	 * Value 39, 40, 42 are not supported on XL710 and X710.
	 */
	I40E_FILTER_PCTYPE_NONF_UNICAST_IPV6_UDP	= 39,
	I40E_FILTER_PCTYPE_NONF_MULTICAST_IPV6_UDP	= 40,
	I40E_FILTER_PCTYPE_NONF_IPV6_UDP		= 41,
	I40E_FILTER_PCTYPE_NONF_IPV6_TCP_SYN_NO_ACK	= 42,
	I40E_FILTER_PCTYPE_NONF_IPV6_TCP		= 43,
	I40E_FILTER_PCTYPE_NONF_IPV6_SCTP		= 44,
	I40E_FILTER_PCTYPE_NONF_IPV6_OTHER		= 45,
	I40E_FILTER_PCTYPE_FRAG_IPV6			= 46,
	/* Note: Value 47 is reserved for future use */
	I40E_FILTER_PCTYPE_FCOE_OX			= 48,
	I40E_FILTER_PCTYPE_FCOE_RX			= 49,
	I40E_FILTER_PCTYPE_FCOE_OTHER			= 50,
	/* Note: Values 51-62 are reserved for future use */
	I40E_FILTER_PCTYPE_L2_PAYLOAD			= 63,
};

enum i40e_filter_program_desc_dest {
	I40E_FILTER_PROGRAM_DESC_DEST_DROP_PACKET		= 0x0,
	I40E_FILTER_PROGRAM_DESC_DEST_DIRECT_PACKET_QINDEX	= 0x1,
	I40E_FILTER_PROGRAM_DESC_DEST_DIRECT_PACKET_OTHER	= 0x2,
};

enum i40e_filter_program_desc_fd_status {
	I40E_FILTER_PROGRAM_DESC_FD_STATUS_NONE			= 0x0,
	I40E_FILTER_PROGRAM_DESC_FD_STATUS_FD_ID		= 0x1,
	I40E_FILTER_PROGRAM_DESC_FD_STATUS_FD_ID_4FLEX_BYTES	= 0x2,
	I40E_FILTER_PROGRAM_DESC_FD_STATUS_8FLEX_BYTES		= 0x3,
};

#define I40E_TXD_FLTR_QW0_DEST_VSI_SHIFT	23
#define I40E_TXD_FLTR_QW0_DEST_VSI_MASK	(0x1FFUL << \
					 I40E_TXD_FLTR_QW0_DEST_VSI_SHIFT)

#define I40E_TXD_FLTR_QW1_DTYPE_SHIFT	0
#define I40E_TXD_FLTR_QW1_DTYPE_MASK	(0xFUL << I40E_TXD_FLTR_QW1_DTYPE_SHIFT)

#define I40E_TXD_FLTR_QW1_CMD_SHIFT	4
#define I40E_TXD_FLTR_QW1_CMD_MASK	(0xFFFFULL << \
					 I40E_TXD_FLTR_QW1_CMD_SHIFT)

#define I40E_TXD_FLTR_QW1_PCMD_SHIFT	(0x0ULL + I40E_TXD_FLTR_QW1_CMD_SHIFT)
#define I40E_TXD_FLTR_QW1_PCMD_MASK	(0x7ULL << I40E_TXD_FLTR_QW1_PCMD_SHIFT)

enum i40e_filter_program_desc_pcmd {
	I40E_FILTER_PROGRAM_DESC_PCMD_ADD_UPDATE	= 0x1,
	I40E_FILTER_PROGRAM_DESC_PCMD_REMOVE		= 0x2,
};

#define I40E_TXD_FLTR_QW1_DEST_SHIFT	(0x3ULL + I40E_TXD_FLTR_QW1_CMD_SHIFT)
#define I40E_TXD_FLTR_QW1_DEST_MASK	(0x3ULL << I40E_TXD_FLTR_QW1_DEST_SHIFT)

#define I40E_TXD_FLTR_QW1_CNT_ENA_SHIFT	(0x7ULL + I40E_TXD_FLTR_QW1_CMD_SHIFT)
#define I40E_TXD_FLTR_QW1_CNT_ENA_MASK	BIT_ULL(I40E_TXD_FLTR_QW1_CNT_ENA_SHIFT)

#define I40E_TXD_FLTR_QW1_FD_STATUS_SHIFT	(0x9ULL + \
						 I40E_TXD_FLTR_QW1_CMD_SHIFT)
#define I40E_TXD_FLTR_QW1_FD_STATUS_MASK (0x3ULL << \
					  I40E_TXD_FLTR_QW1_FD_STATUS_SHIFT)

#define I40E_TXD_FLTR_QW1_ATR_SHIFT	(0xEULL + \
					 I40E_TXD_FLTR_QW1_CMD_SHIFT)
#define I40E_TXD_FLTR_QW1_ATR_MASK	BIT_ULL(I40E_TXD_FLTR_QW1_ATR_SHIFT)

#define I40E_TXD_FLTR_QW1_CNTINDEX_SHIFT 20
#define I40E_TXD_FLTR_QW1_CNTINDEX_MASK	(0x1FFUL << \
					 I40E_TXD_FLTR_QW1_CNTINDEX_SHIFT)

enum i40e_filter_type {
	I40E_FLOW_DIRECTOR_FLTR = 0,
	I40E_PE_QUAD_HASH_FLTR = 1,
	I40E_ETHERTYPE_FLTR,
	I40E_FCOE_CTX_FLTR,
	I40E_MAC_VLAN_FLTR,
	I40E_HASH_FLTR
};

struct i40e_vsi_context {
	u16 seid;
	u16 uplink_seid;
	u16 vsi_number;
	u16 vsis_allocated;
	u16 vsis_unallocated;
	u16 flags;
	u8 pf_num;
	u8 vf_num;
	u8 connection_type;
	struct i40e_aqc_vsi_properties_data info;
};

struct i40e_veb_context {
	u16 seid;
	u16 uplink_seid;
	u16 veb_number;
	u16 vebs_allocated;
	u16 vebs_unallocated;
	u16 flags;
	struct i40e_aqc_get_veb_parameters_completion info;
};

/* Statistics collected by each port, VSI, VEB, and S-channel */
struct i40e_eth_stats {
	u64 rx_bytes;			/* gorc */
	u64 rx_unicast;			/* uprc */
	u64 rx_multicast;		/* mprc */
	u64 rx_broadcast;		/* bprc */
	u64 rx_discards;		/* rdpc */
	u64 rx_unknown_protocol;	/* rupp */
	u64 tx_bytes;			/* gotc */
	u64 tx_unicast;			/* uptc */
	u64 tx_multicast;		/* mptc */
	u64 tx_broadcast;		/* bptc */
	u64 tx_discards;		/* tdpc */
	u64 tx_errors;			/* tepc */
};

/* Statistics collected per VEB per TC */
struct i40e_veb_tc_stats {
	u64 tc_rx_packets[I40E_MAX_TRAFFIC_CLASS];
	u64 tc_rx_bytes[I40E_MAX_TRAFFIC_CLASS];
	u64 tc_tx_packets[I40E_MAX_TRAFFIC_CLASS];
	u64 tc_tx_bytes[I40E_MAX_TRAFFIC_CLASS];
};

/* Statistics collected per function for FCoE */
struct i40e_fcoe_stats {
	u64 rx_fcoe_packets;		/* fcoeprc */
	u64 rx_fcoe_dwords;		/* focedwrc */
	u64 rx_fcoe_dropped;		/* fcoerpdc */
	u64 tx_fcoe_packets;		/* fcoeptc */
	u64 tx_fcoe_dwords;		/* focedwtc */
	u64 fcoe_bad_fccrc;		/* fcoecrc */
	u64 fcoe_last_error;		/* fcoelast */
	u64 fcoe_ddp_count;		/* fcoeddpc */
};

/* offset to per function FCoE statistics block */
#define I40E_FCOE_VF_STAT_OFFSET	0
#define I40E_FCOE_PF_STAT_OFFSET	128
#define I40E_FCOE_STAT_MAX		(I40E_FCOE_PF_STAT_OFFSET + I40E_MAX_PF)

/* Statistics collected by the MAC */
struct i40e_hw_port_stats {
	/* eth stats collected by the port */
	struct i40e_eth_stats eth;

	/* additional port specific stats */
	u64 tx_dropped_link_down;	/* tdold */
	u64 crc_errors;			/* crcerrs */
	u64 illegal_bytes;		/* illerrc */
	u64 error_bytes;		/* errbc */
	u64 mac_local_faults;		/* mlfc */
	u64 mac_remote_faults;		/* mrfc */
	u64 rx_length_errors;		/* rlec */
	u64 link_xon_rx;		/* lxonrxc */
	u64 link_xoff_rx;		/* lxoffrxc */
	u64 priority_xon_rx[8];		/* pxonrxc[8] */
	u64 priority_xoff_rx[8];	/* pxoffrxc[8] */
	u64 link_xon_tx;		/* lxontxc */
	u64 link_xoff_tx;		/* lxofftxc */
	u64 priority_xon_tx[8];		/* pxontxc[8] */
	u64 priority_xoff_tx[8];	/* pxofftxc[8] */
	u64 priority_xon_2_xoff[8];	/* pxon2offc[8] */
	u64 rx_size_64;			/* prc64 */
	u64 rx_size_127;		/* prc127 */
	u64 rx_size_255;		/* prc255 */
	u64 rx_size_511;		/* prc511 */
	u64 rx_size_1023;		/* prc1023 */
	u64 rx_size_1522;		/* prc1522 */
	u64 rx_size_big;		/* prc9522 */
	u64 rx_undersize;		/* ruc */
	u64 rx_fragments;		/* rfc */
	u64 rx_oversize;		/* roc */
	u64 rx_jabber;			/* rjc */
	u64 tx_size_64;			/* ptc64 */
	u64 tx_size_127;		/* ptc127 */
	u64 tx_size_255;		/* ptc255 */
	u64 tx_size_511;		/* ptc511 */
	u64 tx_size_1023;		/* ptc1023 */
	u64 tx_size_1522;		/* ptc1522 */
	u64 tx_size_big;		/* ptc9522 */
	u64 mac_short_packet_dropped;	/* mspdc */
	u64 checksum_error;		/* xec */
	/* flow director stats */
	u64 fd_atr_match;
	u64 fd_sb_match;
	u64 fd_atr_tunnel_match;
	u32 fd_atr_status;
	u32 fd_sb_status;
	/* EEE LPI */
	u32 tx_lpi_status;
	u32 rx_lpi_status;
	u64 tx_lpi_count;		/* etlpic */
	u64 rx_lpi_count;		/* erlpic */
	u64 tx_lpi_duration;
	u64 rx_lpi_duration;
};

/* Checksum and Shadow RAM pointers */
#define I40E_SR_LWM_CONTROL_WORD		0x00
#define I40E_SR_PCIE_ANALOG_CONFIG_PTR		0x03
#define I40E_SR_PHY_ANALOG_CONFIG_PTR		0x04
#define I40E_SR_OPTION_ROM_PTR			0x05
#define I40E_SR_RO_PCIR_REGS_AUTO_LOAD_PTR	0x06
#define I40E_SR_AUTO_GENERATED_POINTERS_PTR	0x07
#define I40E_SR_PCIR_REGS_AUTO_LOAD_PTR		0x08
#define I40E_SR_EMP_GLOBAL_MODULE_PTR		0x09
#define I40E_SR_RO_PCIE_LCB_PTR			0x0A
#define I40E_SR_EMP_IMAGE_PTR			0x0B
#define I40E_SR_PE_IMAGE_PTR			0x0C
#define I40E_SR_CSR_PROTECTED_LIST_PTR		0x0D
#define I40E_SR_MNG_CONFIG_PTR			0x0E
#define I40E_EMP_MODULE_PTR			0x0F
#define I40E_SR_EMP_MODULE_PTR			0x48
#define I40E_SR_PBA_FLAGS			0x15
#define I40E_SR_PBA_BLOCK_PTR			0x16
#define I40E_SR_BOOT_CONFIG_PTR			0x17
#define I40E_LWM_OEM_VER_OFF			0x83
#define I40E_SR_LWM_DEV_STARTER_VERSION		0x18
#define I40E_SR_LWM_WAKE_ON_LAN			0x19
#define I40E_SR_ALTERNATE_SAN_MAC_ADDRESS_PTR	0x27
#define I40E_SR_PERMANENT_SAN_MAC_ADDRESS_PTR	0x28
#define I40E_SR_LWM_MAP_VERSION			0x29
#define I40E_SR_LWM_IMAGE_VERSION		0x2A
#define I40E_SR_LWM_STRUCTURE_VERSION		0x2B
#define I40E_SR_LWM_EETRACK_LO			0x2D
#define I40E_SR_LWM_EETRACK_HI			0x2E
#define I40E_SR_VPD_PTR				0x2F
#define I40E_SR_PXE_SETUP_PTR			0x30
#define I40E_SR_PXE_CONFIG_LWST_OPTIONS_PTR	0x31
#define I40E_SR_LWM_ORIGINAL_EETRACK_LO		0x34
#define I40E_SR_LWM_ORIGINAL_EETRACK_HI		0x35
#define I40E_SR_SW_ETHERNET_MAC_ADDRESS_PTR	0x37
#define I40E_SR_POR_REGS_AUTO_LOAD_PTR		0x38
#define I40E_SR_EMPR_REGS_AUTO_LOAD_PTR		0x3A
#define I40E_SR_GLOBR_REGS_AUTO_LOAD_PTR	0x3B
#define I40E_SR_CORER_REGS_AUTO_LOAD_PTR	0x3C
#define I40E_SR_PHY_ACTIVITY_LIST_PTR		0x3D
#define I40E_SR_PCIE_ALT_AUTO_LOAD_PTR		0x3E
#define I40E_SR_SW_CHECKSUM_WORD		0x3F
#define I40E_SR_1ST_FREE_PROVISION_AREA_PTR	0x40
#define I40E_SR_4TH_FREE_PROVISION_AREA_PTR	0x42
#define I40E_SR_3RD_FREE_PROVISION_AREA_PTR	0x44
#define I40E_SR_2ND_FREE_PROVISION_AREA_PTR	0x46
#define I40E_SR_EMP_SR_SETTINGS_PTR		0x48
#define I40E_SR_FEATURE_CONFIGURATION_PTR	0x49
#define I40E_SR_CONFIGURATION_METADATA_PTR	0x4D
#define I40E_SR_IMMEDIATE_VALUES_PTR		0x4E

/* Auxiliary field, mask and shift definition for Shadow RAM and LWM Flash */
#define I40E_SR_VPD_MODULE_MAX_SIZE		1024
#define I40E_SR_PCIE_ALT_MODULE_MAX_SIZE	1024
#define I40E_SR_CONTROL_WORD_1_SHIFT		0x06
#define I40E_SR_CONTROL_WORD_1_MASK	(0x03 << I40E_SR_CONTROL_WORD_1_SHIFT)
#define I40E_SR_CONTROL_WORD_1_LWM_BANK_VALID	BIT(5)
#define I40E_SR_LWM_MAP_STRUCTURE_TYPE		BIT(12)
#define I40E_PTR_TYPE				BIT(15)
#define I40E_SR_OCP_CFG_WORD0			0x2B
#define I40E_SR_OCP_ENABLED			BIT(15)

/* Shadow RAM related */
#define I40E_SR_SECTOR_SIZE_IN_WORDS	0x800
#define I40E_SR_BUF_ALIGNMENT		4096
#define I40E_SR_WORDS_IN_1KB		512
/* Checksum should be callwlated such that after adding all the words,
 * including the checksum word itself, the sum should be 0xBABA.
 */
#define I40E_SR_SW_CHECKSUM_BASE	0xBABA

#define I40E_SRRD_SRCTL_ATTEMPTS	100000

/* FCoE Tx context descriptor - Use the i40e_tx_context_desc struct */

enum i40E_fcoe_tx_ctx_desc_cmd_bits {
	I40E_FCOE_TX_CTX_DESC_OPCODE_SINGLE_SEND	= 0x00, /* 4 BITS */
	I40E_FCOE_TX_CTX_DESC_OPCODE_TSO_FC_CLASS2	= 0x01, /* 4 BITS */
	I40E_FCOE_TX_CTX_DESC_OPCODE_TSO_FC_CLASS3	= 0x05, /* 4 BITS */
	I40E_FCOE_TX_CTX_DESC_OPCODE_ETSO_FC_CLASS2	= 0x02, /* 4 BITS */
	I40E_FCOE_TX_CTX_DESC_OPCODE_ETSO_FC_CLASS3	= 0x06, /* 4 BITS */
	I40E_FCOE_TX_CTX_DESC_OPCODE_DWO_FC_CLASS2	= 0x03, /* 4 BITS */
	I40E_FCOE_TX_CTX_DESC_OPCODE_DWO_FC_CLASS3	= 0x07, /* 4 BITS */
	I40E_FCOE_TX_CTX_DESC_OPCODE_DDP_CTX_ILWL	= 0x08, /* 4 BITS */
	I40E_FCOE_TX_CTX_DESC_OPCODE_DWO_CTX_ILWL	= 0x09, /* 4 BITS */
	I40E_FCOE_TX_CTX_DESC_RELOFF			= 0x10,
	I40E_FCOE_TX_CTX_DESC_CLRSEQ			= 0x20,
	I40E_FCOE_TX_CTX_DESC_DIFENA			= 0x40,
	I40E_FCOE_TX_CTX_DESC_IL2TAG2			= 0x80
};

/* FCoE DIF/DIX Context descriptor */
struct i40e_fcoe_difdix_context_desc {
	__le64 flags_buff0_buff1_ref;
	__le64 difapp_msk_bias;
};

#define I40E_FCOE_DIFDIX_CTX_QW0_FLAGS_SHIFT	0
#define I40E_FCOE_DIFDIX_CTX_QW0_FLAGS_MASK	(0xFFFULL << \
					I40E_FCOE_DIFDIX_CTX_QW0_FLAGS_SHIFT)

enum i40e_fcoe_difdix_ctx_desc_flags_bits {
	/* 2 BITS */
	I40E_FCOE_DIFDIX_CTX_DESC_RSVD				= 0x0000,
	/* 1 BIT  */
	I40E_FCOE_DIFDIX_CTX_DESC_APPTYPE_TAGCHK		= 0x0000,
	/* 1 BIT  */
	I40E_FCOE_DIFDIX_CTX_DESC_APPTYPE_TAGNOTCHK		= 0x0004,
	/* 2 BITS */
	I40E_FCOE_DIFDIX_CTX_DESC_GTYPE_OPAQUE			= 0x0000,
	/* 2 BITS */
	I40E_FCOE_DIFDIX_CTX_DESC_GTYPE_CHKINTEGRITY		= 0x0008,
	/* 2 BITS */
	I40E_FCOE_DIFDIX_CTX_DESC_GTYPE_CHKINTEGRITY_APPTAG	= 0x0010,
	/* 2 BITS */
	I40E_FCOE_DIFDIX_CTX_DESC_GTYPE_CHKINTEGRITY_APPREFTAG	= 0x0018,
	/* 2 BITS */
	I40E_FCOE_DIFDIX_CTX_DESC_REFTYPE_CNST			= 0x0000,
	/* 2 BITS */
	I40E_FCOE_DIFDIX_CTX_DESC_REFTYPE_INC1BLK		= 0x0020,
	/* 2 BITS */
	I40E_FCOE_DIFDIX_CTX_DESC_REFTYPE_APPTAG		= 0x0040,
	/* 2 BITS */
	I40E_FCOE_DIFDIX_CTX_DESC_REFTYPE_RSVD			= 0x0060,
	/* 1 BIT  */
	I40E_FCOE_DIFDIX_CTX_DESC_DIXMODE_XSUM			= 0x0000,
	/* 1 BIT  */
	I40E_FCOE_DIFDIX_CTX_DESC_DIXMODE_CRC			= 0x0080,
	/* 2 BITS */
	I40E_FCOE_DIFDIX_CTX_DESC_DIFHOST_UNTAG			= 0x0000,
	/* 2 BITS */
	I40E_FCOE_DIFDIX_CTX_DESC_DIFHOST_BUF			= 0x0100,
	/* 2 BITS */
	I40E_FCOE_DIFDIX_CTX_DESC_DIFHOST_RSVD			= 0x0200,
	/* 2 BITS */
	I40E_FCOE_DIFDIX_CTX_DESC_DIFHOST_EMBDTAGS		= 0x0300,
	/* 1 BIT  */
	I40E_FCOE_DIFDIX_CTX_DESC_DIFLAN_UNTAG			= 0x0000,
	/* 1 BIT  */
	I40E_FCOE_DIFDIX_CTX_DESC_DIFLAN_TAG			= 0x0400,
	/* 1 BIT */
	I40E_FCOE_DIFDIX_CTX_DESC_DIFBLK_512B			= 0x0000,
	/* 1 BIT */
	I40E_FCOE_DIFDIX_CTX_DESC_DIFBLK_4K			= 0x0800
};

#define I40E_FCOE_DIFDIX_CTX_QW0_BUFF0_SHIFT	12
#define I40E_FCOE_DIFDIX_CTX_QW0_BUFF0_MASK	(0x3FFULL << \
					I40E_FCOE_DIFDIX_CTX_QW0_BUFF0_SHIFT)

#define I40E_FCOE_DIFDIX_CTX_QW0_BUFF1_SHIFT	22
#define I40E_FCOE_DIFDIX_CTX_QW0_BUFF1_MASK	(0x3FFULL << \
					I40E_FCOE_DIFDIX_CTX_QW0_BUFF1_SHIFT)

#define I40E_FCOE_DIFDIX_CTX_QW0_REF_SHIFT	32
#define I40E_FCOE_DIFDIX_CTX_QW0_REF_MASK	(0xFFFFFFFFULL << \
					I40E_FCOE_DIFDIX_CTX_QW0_REF_SHIFT)

#define I40E_FCOE_DIFDIX_CTX_QW1_APP_SHIFT	0
#define I40E_FCOE_DIFDIX_CTX_QW1_APP_MASK	(0xFFFFULL << \
					I40E_FCOE_DIFDIX_CTX_QW1_APP_SHIFT)

#define I40E_FCOE_DIFDIX_CTX_QW1_APP_MSK_SHIFT	16
#define I40E_FCOE_DIFDIX_CTX_QW1_APP_MSK_MASK	(0xFFFFULL << \
					I40E_FCOE_DIFDIX_CTX_QW1_APP_MSK_SHIFT)

#define I40E_FCOE_DIFDIX_CTX_QW1_REF_BIAS_SHIFT	32
#define I40E_FCOE_DIFDIX_CTX_QW0_REF_BIAS_MASK	(0xFFFFFFFFULL << \
					I40E_FCOE_DIFDIX_CTX_QW1_REF_BIAS_SHIFT)

/* FCoE DIF/DIX Buffers descriptor */
struct i40e_fcoe_difdix_buffers_desc {
	__le64 buff_addr0;
	__le64 buff_addr1;
};

/* FCoE DDP Context descriptor */
struct i40e_fcoe_ddp_context_desc {
	__le64 rsvd;
	__le64 type_cmd_foff_lsize;
};

#define I40E_FCOE_DDP_CTX_QW1_DTYPE_SHIFT	0
#define I40E_FCOE_DDP_CTX_QW1_DTYPE_MASK	(0xFULL << \
					I40E_FCOE_DDP_CTX_QW1_DTYPE_SHIFT)

#define I40E_FCOE_DDP_CTX_QW1_CMD_SHIFT	4
#define I40E_FCOE_DDP_CTX_QW1_CMD_MASK	(0xFULL << \
					 I40E_FCOE_DDP_CTX_QW1_CMD_SHIFT)

enum i40e_fcoe_ddp_ctx_desc_cmd_bits {
	I40E_FCOE_DDP_CTX_DESC_BSIZE_512B	= 0x00, /* 2 BITS */
	I40E_FCOE_DDP_CTX_DESC_BSIZE_4K		= 0x01, /* 2 BITS */
	I40E_FCOE_DDP_CTX_DESC_BSIZE_8K		= 0x02, /* 2 BITS */
	I40E_FCOE_DDP_CTX_DESC_BSIZE_16K	= 0x03, /* 2 BITS */
	I40E_FCOE_DDP_CTX_DESC_DIFENA		= 0x04, /* 1 BIT  */
	I40E_FCOE_DDP_CTX_DESC_LASTSEQH		= 0x08, /* 1 BIT  */
};

#define I40E_FCOE_DDP_CTX_QW1_FOFF_SHIFT	16
#define I40E_FCOE_DDP_CTX_QW1_FOFF_MASK	(0x3FFFULL << \
					 I40E_FCOE_DDP_CTX_QW1_FOFF_SHIFT)

#define I40E_FCOE_DDP_CTX_QW1_LSIZE_SHIFT	32
#define I40E_FCOE_DDP_CTX_QW1_LSIZE_MASK	(0x3FFFULL << \
					I40E_FCOE_DDP_CTX_QW1_LSIZE_SHIFT)

/* FCoE DDP/DWO Queue Context descriptor */
struct i40e_fcoe_queue_context_desc {
	__le64 dmaindx_fbase;           /* 0:11 DMAINDX, 12:63 FBASE */
	__le64 flen_tph;                /* 0:12 FLEN, 13:15 TPH */
};

#define I40E_FCOE_QUEUE_CTX_QW0_DMAINDX_SHIFT	0
#define I40E_FCOE_QUEUE_CTX_QW0_DMAINDX_MASK	(0xFFFULL << \
					I40E_FCOE_QUEUE_CTX_QW0_DMAINDX_SHIFT)

#define I40E_FCOE_QUEUE_CTX_QW0_FBASE_SHIFT	12
#define I40E_FCOE_QUEUE_CTX_QW0_FBASE_MASK	(0xFFFFFFFFFFFFFULL << \
					I40E_FCOE_QUEUE_CTX_QW0_FBASE_SHIFT)

#define I40E_FCOE_QUEUE_CTX_QW1_FLEN_SHIFT	0
#define I40E_FCOE_QUEUE_CTX_QW1_FLEN_MASK	(0x1FFFULL << \
					I40E_FCOE_QUEUE_CTX_QW1_FLEN_SHIFT)

#define I40E_FCOE_QUEUE_CTX_QW1_TPH_SHIFT	13
#define I40E_FCOE_QUEUE_CTX_QW1_TPH_MASK	(0x7ULL << \
					I40E_FCOE_QUEUE_CTX_QW1_FLEN_SHIFT)

enum i40e_fcoe_queue_ctx_desc_tph_bits {
	I40E_FCOE_QUEUE_CTX_DESC_TPHRDESC	= 0x1,
	I40E_FCOE_QUEUE_CTX_DESC_TPHDATA	= 0x2
};

#define I40E_FCOE_QUEUE_CTX_QW1_RECIPE_SHIFT	30
#define I40E_FCOE_QUEUE_CTX_QW1_RECIPE_MASK	(0x3ULL << \
					I40E_FCOE_QUEUE_CTX_QW1_RECIPE_SHIFT)

/* FCoE DDP/DWO Filter Context descriptor */
struct i40e_fcoe_filter_context_desc {
	__le32 param;
	__le16 seqn;

	/* 48:51(0:3) RSVD, 52:63(4:15) DMAINDX */
	__le16 rsvd_dmaindx;

	/* 0:7 FLAGS, 8:52 RSVD, 53:63 LANQ */
	__le64 flags_rsvd_lanq;
};

#define I40E_FCOE_FILTER_CTX_QW0_DMAINDX_SHIFT	4
#define I40E_FCOE_FILTER_CTX_QW0_DMAINDX_MASK	(0xFFF << \
					I40E_FCOE_FILTER_CTX_QW0_DMAINDX_SHIFT)

enum i40e_fcoe_filter_ctx_desc_flags_bits {
	I40E_FCOE_FILTER_CTX_DESC_CTYP_DDP	= 0x00,
	I40E_FCOE_FILTER_CTX_DESC_CTYP_DWO	= 0x01,
	I40E_FCOE_FILTER_CTX_DESC_ENODE_INIT	= 0x00,
	I40E_FCOE_FILTER_CTX_DESC_ENODE_RSP	= 0x02,
	I40E_FCOE_FILTER_CTX_DESC_FC_CLASS2	= 0x00,
	I40E_FCOE_FILTER_CTX_DESC_FC_CLASS3	= 0x04
};

#define I40E_FCOE_FILTER_CTX_QW1_FLAGS_SHIFT	0
#define I40E_FCOE_FILTER_CTX_QW1_FLAGS_MASK	(0xFFULL << \
					I40E_FCOE_FILTER_CTX_QW1_FLAGS_SHIFT)

#define I40E_FCOE_FILTER_CTX_QW1_PCTYPE_SHIFT     8
#define I40E_FCOE_FILTER_CTX_QW1_PCTYPE_MASK      (0x3FULL << \
			I40E_FCOE_FILTER_CTX_QW1_PCTYPE_SHIFT)

#define I40E_FCOE_FILTER_CTX_QW1_LANQINDX_SHIFT     53
#define I40E_FCOE_FILTER_CTX_QW1_LANQINDX_MASK      (0x7FFULL << \
			I40E_FCOE_FILTER_CTX_QW1_LANQINDX_SHIFT)

enum i40e_switch_element_types {
	I40E_SWITCH_ELEMENT_TYPE_MAC	= 1,
	I40E_SWITCH_ELEMENT_TYPE_PF	= 2,
	I40E_SWITCH_ELEMENT_TYPE_VF	= 3,
	I40E_SWITCH_ELEMENT_TYPE_EMP	= 4,
	I40E_SWITCH_ELEMENT_TYPE_BMC	= 6,
	I40E_SWITCH_ELEMENT_TYPE_PE	= 16,
	I40E_SWITCH_ELEMENT_TYPE_VEB	= 17,
	I40E_SWITCH_ELEMENT_TYPE_PA	= 18,
	I40E_SWITCH_ELEMENT_TYPE_VSI	= 19,
};

/* Supported EtherType filters */
enum i40e_ether_type_index {
	I40E_ETHER_TYPE_1588		= 0,
	I40E_ETHER_TYPE_FIP		= 1,
	I40E_ETHER_TYPE_OUI_EXTENDED	= 2,
	I40E_ETHER_TYPE_MAC_CONTROL	= 3,
	I40E_ETHER_TYPE_LLDP		= 4,
	I40E_ETHER_TYPE_EVB_PROTOCOL1	= 5,
	I40E_ETHER_TYPE_EVB_PROTOCOL2	= 6,
	I40E_ETHER_TYPE_QCN_CNM		= 7,
	I40E_ETHER_TYPE_8021X		= 8,
	I40E_ETHER_TYPE_ARP		= 9,
	I40E_ETHER_TYPE_RSV1		= 10,
	I40E_ETHER_TYPE_RSV2		= 11,
};

/* Filter context base size is 1K */
#define I40E_HASH_FILTER_BASE_SIZE	1024
/* Supported Hash filter values */
enum i40e_hash_filter_size {
	I40E_HASH_FILTER_SIZE_1K	= 0,
	I40E_HASH_FILTER_SIZE_2K	= 1,
	I40E_HASH_FILTER_SIZE_4K	= 2,
	I40E_HASH_FILTER_SIZE_8K	= 3,
	I40E_HASH_FILTER_SIZE_16K	= 4,
	I40E_HASH_FILTER_SIZE_32K	= 5,
	I40E_HASH_FILTER_SIZE_64K	= 6,
	I40E_HASH_FILTER_SIZE_128K	= 7,
	I40E_HASH_FILTER_SIZE_256K	= 8,
	I40E_HASH_FILTER_SIZE_512K	= 9,
	I40E_HASH_FILTER_SIZE_1M	= 10,
};

/* DMA context base size is 0.5K */
#define I40E_DMA_CNTX_BASE_SIZE		512
/* Supported DMA context values */
enum i40e_dma_cntx_size {
	I40E_DMA_CNTX_SIZE_512		= 0,
	I40E_DMA_CNTX_SIZE_1K		= 1,
	I40E_DMA_CNTX_SIZE_2K		= 2,
	I40E_DMA_CNTX_SIZE_4K		= 3,
	I40E_DMA_CNTX_SIZE_8K		= 4,
	I40E_DMA_CNTX_SIZE_16K		= 5,
	I40E_DMA_CNTX_SIZE_32K		= 6,
	I40E_DMA_CNTX_SIZE_64K		= 7,
	I40E_DMA_CNTX_SIZE_128K		= 8,
	I40E_DMA_CNTX_SIZE_256K		= 9,
};

/* Supported Hash look up table (LUT) sizes */
enum i40e_hash_lut_size {
	I40E_HASH_LUT_SIZE_128		= 0,
	I40E_HASH_LUT_SIZE_512		= 1,
};

/* Structure to hold a per PF filter control settings */
struct i40e_filter_control_settings {
	/* number of PE Quad Hash filter buckets */
	enum i40e_hash_filter_size pe_filt_num;
	/* number of PE Quad Hash contexts */
	enum i40e_dma_cntx_size pe_cntx_num;
	/* number of FCoE filter buckets */
	enum i40e_hash_filter_size fcoe_filt_num;
	/* number of FCoE DDP contexts */
	enum i40e_dma_cntx_size fcoe_cntx_num;
	/* size of the Hash LUT */
	enum i40e_hash_lut_size	hash_lut_size;
	/* enable FDIR filters for PF and its VFs */
	bool enable_fdir;
	/* enable Ethertype filters for PF and its VFs */
	bool enable_ethtype;
	/* enable MAC/VLAN filters for PF and its VFs */
	bool enable_macvlan;
};

/* Structure to hold device level control filter counts */
struct i40e_control_filter_stats {
	u16 mac_etype_used;   /* Used perfect match MAC/EtherType filters */
	u16 etype_used;       /* Used perfect EtherType filters */
	u16 mac_etype_free;   /* Un-used perfect match MAC/EtherType filters */
	u16 etype_free;       /* Un-used perfect EtherType filters */
};

enum i40e_reset_type {
	I40E_RESET_POR		= 0,
	I40E_RESET_CORER	= 1,
	I40E_RESET_GLOBR	= 2,
	I40E_RESET_EMPR		= 3,
};

/* IEEE 802.1AB LLDP Agent Variables from LWM */
#define I40E_LWM_LLDP_CFG_PTR   0x06
#define I40E_SR_LLDP_CFG_PTR    0x31
struct i40e_lldp_variables {
	u16 length;
	u16 adminstatus;
	u16 msgfasttx;
	u16 msgtxinterval;
	u16 txparams;
	u16 timers;
	u16 crc8;
};

/* Offsets into Alternate Ram */
#define I40E_ALT_STRUCT_FIRST_PF_OFFSET		0   /* in dwords */
#define I40E_ALT_STRUCT_DWORDS_PER_PF		64   /* in dwords */
#define I40E_ALT_STRUCT_OUTER_VLAN_TAG_OFFSET	0xD  /* in dwords */
#define I40E_ALT_STRUCT_USER_PRIORITY_OFFSET	0xC  /* in dwords */
#define I40E_ALT_STRUCT_MIN_BW_OFFSET		0xE  /* in dwords */
#define I40E_ALT_STRUCT_MAX_BW_OFFSET		0xF  /* in dwords */

/* Alternate Ram Bandwidth Masks */
#define I40E_ALT_BW_VALUE_MASK		0xFF
#define I40E_ALT_BW_RELATIVE_MASK	0x40000000
#define I40E_ALT_BW_VALID_MASK		0x80000000

/* RSS Hash Table Size */
#define I40E_PFQF_CTL_0_HASHLUTSIZE_512	0x00010000

/* INPUT SET MASK for RSS, flow director, and flexible payload */
#define I40E_L3_SRC_SHIFT		47
#define I40E_L3_SRC_MASK		(0x3ULL << I40E_L3_SRC_SHIFT)
#define I40E_L3_V6_SRC_SHIFT		43
#define I40E_L3_V6_SRC_MASK		(0xFFULL << I40E_L3_V6_SRC_SHIFT)
#define I40E_L3_DST_SHIFT		35
#define I40E_L3_DST_MASK		(0x3ULL << I40E_L3_DST_SHIFT)
#define I40E_L3_V6_DST_SHIFT		35
#define I40E_L3_V6_DST_MASK		(0xFFULL << I40E_L3_V6_DST_SHIFT)
#define I40E_L4_SRC_SHIFT		34
#define I40E_L4_SRC_MASK		(0x1ULL << I40E_L4_SRC_SHIFT)
#define I40E_L4_DST_SHIFT		33
#define I40E_L4_DST_MASK		(0x1ULL << I40E_L4_DST_SHIFT)
#define I40E_VERIFY_TAG_SHIFT		31
#define I40E_VERIFY_TAG_MASK		(0x3ULL << I40E_VERIFY_TAG_SHIFT)

#define I40E_FLEX_50_SHIFT		13
#define I40E_FLEX_50_MASK		(0x1ULL << I40E_FLEX_50_SHIFT)
#define I40E_FLEX_51_SHIFT		12
#define I40E_FLEX_51_MASK		(0x1ULL << I40E_FLEX_51_SHIFT)
#define I40E_FLEX_52_SHIFT		11
#define I40E_FLEX_52_MASK		(0x1ULL << I40E_FLEX_52_SHIFT)
#define I40E_FLEX_53_SHIFT		10
#define I40E_FLEX_53_MASK		(0x1ULL << I40E_FLEX_53_SHIFT)
#define I40E_FLEX_54_SHIFT		9
#define I40E_FLEX_54_MASK		(0x1ULL << I40E_FLEX_54_SHIFT)
#define I40E_FLEX_55_SHIFT		8
#define I40E_FLEX_55_MASK		(0x1ULL << I40E_FLEX_55_SHIFT)
#define I40E_FLEX_56_SHIFT		7
#define I40E_FLEX_56_MASK		(0x1ULL << I40E_FLEX_56_SHIFT)
#define I40E_FLEX_57_SHIFT		6
#define I40E_FLEX_57_MASK		(0x1ULL << I40E_FLEX_57_SHIFT)

/* Version format for Dynamic Device Personalization(DDP) */
struct i40e_ddp_version {
	u8 major;
	u8 minor;
	u8 update;
	u8 draft;
};

#define I40E_DDP_NAME_SIZE	32

/* Package header */
struct i40e_package_header {
	struct i40e_ddp_version version;
	u32 segment_count;
	u32 segment_offset[1];
};

/* Generic segment header */
struct i40e_generic_seg_header {
#define SEGMENT_TYPE_METADATA	0x00000001
#define SEGMENT_TYPE_NOTES	0x00000002
#define SEGMENT_TYPE_I40E	0x00000011
#define SEGMENT_TYPE_X722	0x00000012
	u32 type;
	struct i40e_ddp_version version;
	u32 size;
	char name[I40E_DDP_NAME_SIZE];
};

struct i40e_metadata_segment {
	struct i40e_generic_seg_header header;
	struct i40e_ddp_version version;
#define I40E_DDP_TRACKID_RDONLY		0
#define I40E_DDP_TRACKID_ILWALID	0xFFFFFFFF
	u32 track_id;
	char name[I40E_DDP_NAME_SIZE];
};

struct i40e_device_id_entry {
	u32 vendor_dev_id;
	u32 sub_vendor_dev_id;
};

struct i40e_profile_segment {
	struct i40e_generic_seg_header header;
	struct i40e_ddp_version version;
	char name[I40E_DDP_NAME_SIZE];
	u32 device_table_count;
	struct i40e_device_id_entry device_table[1];
};

struct i40e_section_table {
	u32 section_count;
	u32 section_offset[1];
};

struct i40e_profile_section_header {
	u16 tbl_size;
	u16 data_end;
	struct {
#define SECTION_TYPE_INFO	0x00000010
#define SECTION_TYPE_MMIO	0x00000800
#define SECTION_TYPE_RB_MMIO	0x00001800
#define SECTION_TYPE_AQ		0x00000801
#define SECTION_TYPE_RB_AQ	0x00001801
#define SECTION_TYPE_NOTE	0x80000000
#define SECTION_TYPE_NAME	0x80000001
#define SECTION_TYPE_PROTO	0x80000002
#define SECTION_TYPE_PCTYPE	0x80000003
#define SECTION_TYPE_PTYPE	0x80000004
		u32 type;
		u32 offset;
		u32 size;
	} section;
};

struct i40e_profile_tlv_section_record {
	u8 rtype;
	u8 type;
	u16 len;
	u8 data[12];
};

/* Generic AQ section in proflie */
struct i40e_profile_aq_section {
	u16 opcode;
	u16 flags;
	u8  param[16];
	u16 datalen;
	u8  data[1];
};

struct i40e_profile_info {
	u32 track_id;
	struct i40e_ddp_version version;
	u8 op;
#define I40E_DDP_ADD_TRACKID		0x01
#define I40E_DDP_REMOVE_TRACKID	0x02
	u8 reserved[7];
	u8 name[I40E_DDP_NAME_SIZE];
};

#define I40E_BCM_PHY_PCS_STATUS1_PAGE	0x3
#define I40E_BCM_PHY_PCS_STATUS1_REG	0x0001
#define I40E_BCM_PHY_PCS_STATUS1_RX_LPI	BIT(8)
#define I40E_BCM_PHY_PCS_STATUS1_TX_LPI	BIT(9)

#endif /* _I40E_TYPE_H_ */
