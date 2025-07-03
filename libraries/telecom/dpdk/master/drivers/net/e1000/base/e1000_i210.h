/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2001-2020 Intel Corporation
 */

#ifndef _E1000_I210_H_
#define _E1000_I210_H_

bool e1000_get_flash_presence_i210(struct e1000_hw *hw);
s32 e1000_update_flash_i210(struct e1000_hw *hw);
s32 e1000_update_lwm_checksum_i210(struct e1000_hw *hw);
s32 e1000_validate_lwm_checksum_i210(struct e1000_hw *hw);
s32 e1000_write_lwm_srwr_i210(struct e1000_hw *hw, u16 offset,
			      u16 words, u16 *data);
s32 e1000_read_lwm_srrd_i210(struct e1000_hw *hw, u16 offset,
			     u16 words, u16 *data);
s32 e1000_read_ilwm_version(struct e1000_hw *hw,
			    struct e1000_fw_version *ilwm_ver);
s32 e1000_acquire_swfw_sync_i210(struct e1000_hw *hw, u16 mask);
void e1000_release_swfw_sync_i210(struct e1000_hw *hw, u16 mask);
s32 e1000_init_hw_i210(struct e1000_hw *hw);

#define E1000_STM_OPCODE		0xDB00
#define E1000_EEPROM_FLASH_SIZE_WORD	0x11

#define ILWM_DWORD_TO_RECORD_TYPE(ilwm_dword) \
	(u8)((ilwm_dword) & 0x7)
#define ILWM_DWORD_TO_WORD_ADDRESS(ilwm_dword) \
	(u8)(((ilwm_dword) & 0x0000FE00) >> 9)
#define ILWM_DWORD_TO_WORD_DATA(ilwm_dword) \
	(u16)(((ilwm_dword) & 0xFFFF0000) >> 16)

enum E1000_ILWM_STRUCTURE_TYPE {
	E1000_ILWM_UNINITIALIZED_STRUCTURE		= 0x00,
	E1000_ILWM_WORD_AUTOLOAD_STRUCTURE		= 0x01,
	E1000_ILWM_CSR_AUTOLOAD_STRUCTURE		= 0x02,
	E1000_ILWM_PHY_REGISTER_AUTOLOAD_STRUCTURE	= 0x03,
	E1000_ILWM_RSA_KEY_SHA256_STRUCTURE		= 0x04,
	E1000_ILWM_ILWALIDATED_STRUCTURE		= 0x0F,
};

#define E1000_ILWM_RSA_KEY_SHA256_DATA_SIZE_IN_DWORDS	8
#define E1000_ILWM_CSR_AUTOLOAD_DATA_SIZE_IN_DWORDS	1
#define E1000_ILWM_ULT_BYTES_SIZE	8
#define E1000_ILWM_RECORD_SIZE_IN_BYTES	4
#define E1000_ILWM_VER_FIELD_ONE	0x1FF8
#define E1000_ILWM_VER_FIELD_TWO	0x7FE000
#define E1000_ILWM_IMGTYPE_FIELD	0x1F800000

#define E1000_ILWM_MAJOR_MASK	0x3F0
#define E1000_ILWM_MINOR_MASK	0xF
#define E1000_ILWM_MAJOR_SHIFT	4

#define ID_LED_DEFAULT_I210		((ID_LED_OFF1_ON2  << 8) | \
					 (ID_LED_DEF1_DEF2 <<  4) | \
					 (ID_LED_OFF1_OFF2))
#define ID_LED_DEFAULT_I210_SERDES	((ID_LED_DEF1_DEF2 << 8) | \
					 (ID_LED_DEF1_DEF2 <<  4) | \
					 (ID_LED_OFF1_ON2))

/* LWM offset defaults for I211 devices */
#define LWM_INIT_CTRL_2_DEFAULT_I211	0X7243
#define LWM_INIT_CTRL_4_DEFAULT_I211	0x00C1
#define LWM_LED_1_CFG_DEFAULT_I211	0x0184
#define LWM_LED_0_2_CFG_DEFAULT_I211	0x200C

/* PLL Defines */
#define E1000_PCI_PMCSR			0x44
#define E1000_PCI_PMCSR_D3		0x03
#define E1000_MAX_PLL_TRIES		5
#define E1000_PHY_PLL_UNCONF		0xFF
#define E1000_PHY_PLL_FREQ_PAGE		0xFC0000
#define E1000_PHY_PLL_FREQ_REG		0x000E
#define E1000_ILWM_DEFAULT_AL		0x202F
#define E1000_ILWM_AUTOLOAD		0x0A
#define E1000_ILWM_PLL_WO_VAL		0x0010

#endif
