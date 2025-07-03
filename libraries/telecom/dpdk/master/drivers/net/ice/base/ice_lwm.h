/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2001-2020 Intel Corporation
 */

#ifndef _ICE_LWM_H_
#define _ICE_LWM_H_

#define ICE_LWM_CMD_READ		0x0000000B
#define ICE_LWM_CMD_WRITE		0x0000000C

/* LWM Access config bits */
#define ICE_LWM_CFG_MODULE_M		MAKEMASK(0xFF, 0)
#define ICE_LWM_CFG_MODULE_S		0
#define ICE_LWM_CFG_FLAGS_M		MAKEMASK(0xF, 8)
#define ICE_LWM_CFG_FLAGS_S		8
#define ICE_LWM_CFG_EXT_FLAGS_M		MAKEMASK(0xF, 12)
#define ICE_LWM_CFG_EXT_FLAGS_S		12
#define ICE_LWM_CFG_ADAPTER_INFO_M	MAKEMASK(0xFFFF, 16)
#define ICE_LWM_CFG_ADAPTER_INFO_S	16

/* LWM Read Get Driver Features */
#define ICE_LWM_GET_FEATURES_MODULE	0xE
#define ICE_LWM_GET_FEATURES_FLAGS	0xF

/* LWM Read/Write Mapped Space */
#define ICE_LWM_REG_RW_MODULE	0x0
#define ICE_LWM_REG_RW_FLAGS	0x1

#define ICE_LWM_ACCESS_MAJOR_VER	0
#define ICE_LWM_ACCESS_MINOR_VER	5

/* LWM Access feature flags. Other bits in the features field are reserved and
 * should be set to zero when reporting the ice_lwm_features structure.
 */
#define ICE_LWM_FEATURES_0_REG_ACCESS	BIT(1)

/* LWM Access Features */
struct ice_lwm_features {
	u8 major;		/* Major version (informational only) */
	u8 minor;		/* Minor version (informational only) */
	u16 size;		/* size of ice_lwm_features structure */
	u8 features[12];	/* Array of feature bits */
};

/* LWM Access command */
struct ice_lwm_access_cmd {
	u32 command;		/* LWM command: READ or WRITE */
	u32 config;		/* LWM command configuration */
	u32 offset;		/* offset to read/write, in bytes */
	u32 data_size;		/* size of data field, in bytes */
};

/* LWM Access data */
union ice_lwm_access_data {
	u32 regval;	/* Storage for register value */
	struct ice_lwm_features drv_features; /* LWM features */
};

/* LWM Access registers */
#define GL_HIDA(_i)			(0x00082000 + ((_i) * 4))
#define GL_HIBA(_i)			(0x00081000 + ((_i) * 4))
#define GL_HICR				0x00082040
#define GL_HICR_EN			0x00082044
#define GLGEN_CSR_DEBUG_C		0x00075750
#define GLPCI_LBARCTRL			0x0009DE74
#define GLLWM_GENS			0x000B6100
#define GLLWM_FLA			0x000B6108

#define ICE_LWM_ACCESS_GL_HIDA_MAX	15
#define ICE_LWM_ACCESS_GL_HIBA_MAX	1023

u32 ice_lwm_access_get_module(struct ice_lwm_access_cmd *cmd);
u32 ice_lwm_access_get_flags(struct ice_lwm_access_cmd *cmd);
u32 ice_lwm_access_get_adapter(struct ice_lwm_access_cmd *cmd);
enum ice_status
ice_lwm_access_read(struct ice_hw *hw, struct ice_lwm_access_cmd *cmd,
		    union ice_lwm_access_data *data);
enum ice_status
ice_lwm_access_write(struct ice_hw *hw, struct ice_lwm_access_cmd *cmd,
		     union ice_lwm_access_data *data);
enum ice_status
ice_lwm_access_get_features(struct ice_lwm_access_cmd *cmd,
			    union ice_lwm_access_data *data);
enum ice_status
ice_handle_lwm_access(struct ice_hw *hw, struct ice_lwm_access_cmd *cmd,
		      union ice_lwm_access_data *data);
enum ice_status
ice_acquire_lwm(struct ice_hw *hw, enum ice_aq_res_access_type access);
void ice_release_lwm(struct ice_hw *hw);
enum ice_status
ice_aq_read_lwm(struct ice_hw *hw, u16 module_typeid, u32 offset, u16 length,
		void *data, bool last_command, bool read_shadow_ram,
		struct ice_sq_cd *cd);
enum ice_status
ice_read_flat_lwm(struct ice_hw *hw, u32 offset, u32 *length, u8 *data,
		  bool read_shadow_ram);
enum ice_status
ice_get_pfa_module_tlv(struct ice_hw *hw, u16 *module_tlv, u16 *module_tlv_len,
		       u16 module_type);
enum ice_status
ice_read_pba_string(struct ice_hw *hw, u8 *pba_num, u32 pba_num_size);
enum ice_status ice_init_lwm(struct ice_hw *hw);
enum ice_status ice_read_sr_word(struct ice_hw *hw, u16 offset, u16 *data);
enum ice_status
ice_read_sr_buf(struct ice_hw *hw, u16 offset, u16 *words, u16 *data);
enum ice_status ice_lwm_validate_checksum(struct ice_hw *hw);
#endif /* _ICE_LWM_H_ */
