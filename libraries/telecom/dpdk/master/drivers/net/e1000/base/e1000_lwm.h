/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2001-2020 Intel Corporation
 */

#ifndef _E1000_LWM_H_
#define _E1000_LWM_H_

struct e1000_pba {
	u16 word[2];
	u16 *pba_block;
};

struct e1000_fw_version {
	u32 etrack_id;
	u16 eep_major;
	u16 eep_minor;
	u16 eep_build;

	u8 ilwm_major;
	u8 ilwm_minor;
	u8 ilwm_img_type;

	bool or_valid;
	u16 or_major;
	u16 or_build;
	u16 or_patch;
};


void e1000_init_lwm_ops_generic(struct e1000_hw *hw);
s32  e1000_null_read_lwm(struct e1000_hw *hw, u16 a, u16 b, u16 *c);
void e1000_null_lwm_generic(struct e1000_hw *hw);
s32  e1000_null_led_default(struct e1000_hw *hw, u16 *data);
s32  e1000_null_write_lwm(struct e1000_hw *hw, u16 a, u16 b, u16 *c);
s32  e1000_acquire_lwm_generic(struct e1000_hw *hw);

s32  e1000_poll_eerd_eewr_done(struct e1000_hw *hw, int ee_reg);
s32  e1000_read_mac_addr_generic(struct e1000_hw *hw);
s32  e1000_read_pba_num_generic(struct e1000_hw *hw, u32 *pba_num);
s32  e1000_read_pba_string_generic(struct e1000_hw *hw, u8 *pba_num,
				   u32 pba_num_size);
s32  e1000_read_pba_length_generic(struct e1000_hw *hw, u32 *pba_num_size);
s32 e1000_read_pba_raw(struct e1000_hw *hw, u16 *eeprom_buf,
		       u32 eeprom_buf_size, u16 max_pba_block_size,
		       struct e1000_pba *pba);
s32 e1000_write_pba_raw(struct e1000_hw *hw, u16 *eeprom_buf,
			u32 eeprom_buf_size, struct e1000_pba *pba);
s32 e1000_get_pba_block_size(struct e1000_hw *hw, u16 *eeprom_buf,
			     u32 eeprom_buf_size, u16 *pba_block_size);
s32  e1000_read_lwm_spi(struct e1000_hw *hw, u16 offset, u16 words, u16 *data);
s32  e1000_read_lwm_microwire(struct e1000_hw *hw, u16 offset,
			      u16 words, u16 *data);
s32  e1000_read_lwm_eerd(struct e1000_hw *hw, u16 offset, u16 words,
			 u16 *data);
s32  e1000_valid_led_default_generic(struct e1000_hw *hw, u16 *data);
s32  e1000_validate_lwm_checksum_generic(struct e1000_hw *hw);
s32  e1000_write_lwm_microwire(struct e1000_hw *hw, u16 offset,
			       u16 words, u16 *data);
s32  e1000_write_lwm_spi(struct e1000_hw *hw, u16 offset, u16 words,
			 u16 *data);
s32  e1000_update_lwm_checksum_generic(struct e1000_hw *hw);
void e1000_stop_lwm(struct e1000_hw *hw);
void e1000_release_lwm_generic(struct e1000_hw *hw);
void e1000_get_fw_version(struct e1000_hw *hw,
			  struct e1000_fw_version *fw_vers);

#define E1000_STM_OPCODE	0xDB00

#endif
