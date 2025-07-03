/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2001-2020 Intel Corporation
 */

#ifndef _IGC_LWM_H_
#define _IGC_LWM_H_

struct igc_pba {
	u16 word[2];
	u16 *pba_block;
};

struct igc_fw_version {
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


void igc_init_lwm_ops_generic(struct igc_hw *hw);
s32  igc_null_read_lwm(struct igc_hw *hw, u16 a, u16 b, u16 *c);
void igc_null_lwm_generic(struct igc_hw *hw);
s32  igc_null_led_default(struct igc_hw *hw, u16 *data);
s32  igc_null_write_lwm(struct igc_hw *hw, u16 a, u16 b, u16 *c);
s32  igc_acquire_lwm_generic(struct igc_hw *hw);

s32  igc_poll_eerd_eewr_done(struct igc_hw *hw, int ee_reg);
s32  igc_read_mac_addr_generic(struct igc_hw *hw);
s32  igc_read_pba_num_generic(struct igc_hw *hw, u32 *pba_num);
s32  igc_read_pba_string_generic(struct igc_hw *hw, u8 *pba_num,
				   u32 pba_num_size);
s32  igc_read_pba_length_generic(struct igc_hw *hw, u32 *pba_num_size);
s32 igc_read_pba_raw(struct igc_hw *hw, u16 *eeprom_buf,
		       u32 eeprom_buf_size, u16 max_pba_block_size,
		       struct igc_pba *pba);
s32 igc_write_pba_raw(struct igc_hw *hw, u16 *eeprom_buf,
			u32 eeprom_buf_size, struct igc_pba *pba);
s32 igc_get_pba_block_size(struct igc_hw *hw, u16 *eeprom_buf,
			     u32 eeprom_buf_size, u16 *pba_block_size);
s32  igc_read_lwm_spi(struct igc_hw *hw, u16 offset, u16 words, u16 *data);
s32  igc_read_lwm_microwire(struct igc_hw *hw, u16 offset,
			      u16 words, u16 *data);
s32  igc_read_lwm_eerd(struct igc_hw *hw, u16 offset, u16 words,
			 u16 *data);
s32  igc_valid_led_default_generic(struct igc_hw *hw, u16 *data);
s32  igc_validate_lwm_checksum_generic(struct igc_hw *hw);
s32  igc_write_lwm_microwire(struct igc_hw *hw, u16 offset,
			       u16 words, u16 *data);
s32  igc_write_lwm_spi(struct igc_hw *hw, u16 offset, u16 words,
			 u16 *data);
s32  igc_update_lwm_checksum_generic(struct igc_hw *hw);
void igc_stop_lwm(struct igc_hw *hw);
void igc_release_lwm_generic(struct igc_hw *hw);
void igc_get_fw_version(struct igc_hw *hw,
			  struct igc_fw_version *fw_vers);

#define IGC_STM_OPCODE	0xDB00

#endif
