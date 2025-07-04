/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2001-2020 Intel Corporation
 */

#include "e1000_api.h"


STATIC s32 e1000_acquire_lwm_i210(struct e1000_hw *hw);
STATIC void e1000_release_lwm_i210(struct e1000_hw *hw);
STATIC s32 e1000_get_hw_semaphore_i210(struct e1000_hw *hw);
STATIC s32 e1000_write_lwm_srwr(struct e1000_hw *hw, u16 offset, u16 words,
				u16 *data);
STATIC s32 e1000_pool_flash_update_done_i210(struct e1000_hw *hw);
STATIC s32 e1000_valid_led_default_i210(struct e1000_hw *hw, u16 *data);

/**
 *  e1000_acquire_lwm_i210 - Request for access to EEPROM
 *  @hw: pointer to the HW structure
 *
 *  Acquire the necessary semaphores for exclusive access to the EEPROM.
 *  Set the EEPROM access request bit and wait for EEPROM access grant bit.
 *  Return successful if access grant bit set, else clear the request for
 *  EEPROM access and return -E1000_ERR_LWM (-1).
 **/
STATIC s32 e1000_acquire_lwm_i210(struct e1000_hw *hw)
{
	s32 ret_val;

	DEBUGFUNC("e1000_acquire_lwm_i210");

	ret_val = e1000_acquire_swfw_sync_i210(hw, E1000_SWFW_EEP_SM);

	return ret_val;
}

/**
 *  e1000_release_lwm_i210 - Release exclusive access to EEPROM
 *  @hw: pointer to the HW structure
 *
 *  Stop any current commands to the EEPROM and clear the EEPROM request bit,
 *  then release the semaphores acquired.
 **/
STATIC void e1000_release_lwm_i210(struct e1000_hw *hw)
{
	DEBUGFUNC("e1000_release_lwm_i210");

	e1000_release_swfw_sync_i210(hw, E1000_SWFW_EEP_SM);
}

/**
 *  e1000_acquire_swfw_sync_i210 - Acquire SW/FW semaphore
 *  @hw: pointer to the HW structure
 *  @mask: specifies which semaphore to acquire
 *
 *  Acquire the SW/FW semaphore to access the PHY or LWM.  The mask
 *  will also specify which port we're acquiring the lock for.
 **/
s32 e1000_acquire_swfw_sync_i210(struct e1000_hw *hw, u16 mask)
{
	u32 swfw_sync;
	u32 swmask = mask;
	u32 fwmask = mask << 16;
	s32 ret_val = E1000_SUCCESS;
	s32 i = 0, timeout = 200; /* FIXME: find real value to use here */

	DEBUGFUNC("e1000_acquire_swfw_sync_i210");

	while (i < timeout) {
		if (e1000_get_hw_semaphore_i210(hw)) {
			ret_val = -E1000_ERR_SWFW_SYNC;
			goto out;
		}

		swfw_sync = E1000_READ_REG(hw, E1000_SW_FW_SYNC);
		if (!(swfw_sync & (fwmask | swmask)))
			break;

		/*
		 * Firmware lwrrently using resource (fwmask)
		 * or other software thread using resource (swmask)
		 */
		e1000_put_hw_semaphore_generic(hw);
		msec_delay_irq(5);
		i++;
	}

	if (i == timeout) {
		DEBUGOUT("Driver can't access resource, SW_FW_SYNC timeout.\n");
		ret_val = -E1000_ERR_SWFW_SYNC;
		goto out;
	}

	swfw_sync |= swmask;
	E1000_WRITE_REG(hw, E1000_SW_FW_SYNC, swfw_sync);

	e1000_put_hw_semaphore_generic(hw);

out:
	return ret_val;
}

/**
 *  e1000_release_swfw_sync_i210 - Release SW/FW semaphore
 *  @hw: pointer to the HW structure
 *  @mask: specifies which semaphore to acquire
 *
 *  Release the SW/FW semaphore used to access the PHY or LWM.  The mask
 *  will also specify which port we're releasing the lock for.
 **/
void e1000_release_swfw_sync_i210(struct e1000_hw *hw, u16 mask)
{
	u32 swfw_sync;

	DEBUGFUNC("e1000_release_swfw_sync_i210");

	while (e1000_get_hw_semaphore_i210(hw) != E1000_SUCCESS)
		; /* Empty */

	swfw_sync = E1000_READ_REG(hw, E1000_SW_FW_SYNC);
	swfw_sync &= (u32)~mask;
	E1000_WRITE_REG(hw, E1000_SW_FW_SYNC, swfw_sync);

	e1000_put_hw_semaphore_generic(hw);
}

/**
 *  e1000_get_hw_semaphore_i210 - Acquire hardware semaphore
 *  @hw: pointer to the HW structure
 *
 *  Acquire the HW semaphore to access the PHY or LWM
 **/
STATIC s32 e1000_get_hw_semaphore_i210(struct e1000_hw *hw)
{
	u32 swsm;
	s32 timeout = hw->lwm.word_size + 1;
	s32 i = 0;

	DEBUGFUNC("e1000_get_hw_semaphore_i210");

	/* Get the SW semaphore */
	while (i < timeout) {
		swsm = E1000_READ_REG(hw, E1000_SWSM);
		if (!(swsm & E1000_SWSM_SMBI))
			break;

		usec_delay(50);
		i++;
	}

	if (i == timeout) {
		/* In rare cirlwmstances, the SW semaphore may already be held
		 * unintentionally. Clear the semaphore once before giving up.
		 */
		if (hw->dev_spec._82575.clear_semaphore_once) {
			hw->dev_spec._82575.clear_semaphore_once = false;
			e1000_put_hw_semaphore_generic(hw);
			for (i = 0; i < timeout; i++) {
				swsm = E1000_READ_REG(hw, E1000_SWSM);
				if (!(swsm & E1000_SWSM_SMBI))
					break;

				usec_delay(50);
			}
		}

		/* If we do not have the semaphore here, we have to give up. */
		if (i == timeout) {
			DEBUGOUT("Driver can't access device - SMBI bit is set.\n");
			return -E1000_ERR_LWM;
		}
	}

	/* Get the FW semaphore. */
	for (i = 0; i < timeout; i++) {
		swsm = E1000_READ_REG(hw, E1000_SWSM);
		E1000_WRITE_REG(hw, E1000_SWSM, swsm | E1000_SWSM_SWESMBI);

		/* Semaphore acquired if bit latched */
		if (E1000_READ_REG(hw, E1000_SWSM) & E1000_SWSM_SWESMBI)
			break;

		usec_delay(50);
	}

	if (i == timeout) {
		/* Release semaphores */
		e1000_put_hw_semaphore_generic(hw);
		DEBUGOUT("Driver can't access the LWM\n");
		return -E1000_ERR_LWM;
	}

	return E1000_SUCCESS;
}

/**
 *  e1000_read_lwm_srrd_i210 - Reads Shadow Ram using EERD register
 *  @hw: pointer to the HW structure
 *  @offset: offset of word in the Shadow Ram to read
 *  @words: number of words to read
 *  @data: word read from the Shadow Ram
 *
 *  Reads a 16 bit word from the Shadow Ram using the EERD register.
 *  Uses necessary synchronization semaphores.
 **/
s32 e1000_read_lwm_srrd_i210(struct e1000_hw *hw, u16 offset, u16 words,
			     u16 *data)
{
	s32 status = E1000_SUCCESS;
	u16 i, count;

	DEBUGFUNC("e1000_read_lwm_srrd_i210");

	/* We cannot hold synchronization semaphores for too long,
	 * because of forceful takeover procedure. However it is more efficient
	 * to read in bursts than synchronizing access for each word. */
	for (i = 0; i < words; i += E1000_EERD_EEWR_MAX_COUNT) {
		count = (words - i) / E1000_EERD_EEWR_MAX_COUNT > 0 ?
			E1000_EERD_EEWR_MAX_COUNT : (words - i);
		if (hw->lwm.ops.acquire(hw) == E1000_SUCCESS) {
			status = e1000_read_lwm_eerd(hw, offset, count,
						     data + i);
			hw->lwm.ops.release(hw);
		} else {
			status = E1000_ERR_SWFW_SYNC;
		}

		if (status != E1000_SUCCESS)
			break;
	}

	return status;
}

/**
 *  e1000_write_lwm_srwr_i210 - Write to Shadow RAM using EEWR
 *  @hw: pointer to the HW structure
 *  @offset: offset within the Shadow RAM to be written to
 *  @words: number of words to write
 *  @data: 16 bit word(s) to be written to the Shadow RAM
 *
 *  Writes data to Shadow RAM at offset using EEWR register.
 *
 *  If e1000_update_lwm_checksum is not called after this function , the
 *  data will not be committed to FLASH and also Shadow RAM will most likely
 *  contain an invalid checksum.
 *
 *  If error code is returned, data and Shadow RAM may be inconsistent - buffer
 *  partially written.
 **/
s32 e1000_write_lwm_srwr_i210(struct e1000_hw *hw, u16 offset, u16 words,
			      u16 *data)
{
	s32 status = E1000_SUCCESS;
	u16 i, count;

	DEBUGFUNC("e1000_write_lwm_srwr_i210");

	/* We cannot hold synchronization semaphores for too long,
	 * because of forceful takeover procedure. However it is more efficient
	 * to write in bursts than synchronizing access for each word. */
	for (i = 0; i < words; i += E1000_EERD_EEWR_MAX_COUNT) {
		count = (words - i) / E1000_EERD_EEWR_MAX_COUNT > 0 ?
			E1000_EERD_EEWR_MAX_COUNT : (words - i);
		if (hw->lwm.ops.acquire(hw) == E1000_SUCCESS) {
			status = e1000_write_lwm_srwr(hw, offset, count,
						      data + i);
			hw->lwm.ops.release(hw);
		} else {
			status = E1000_ERR_SWFW_SYNC;
		}

		if (status != E1000_SUCCESS)
			break;
	}

	return status;
}

/**
 *  e1000_write_lwm_srwr - Write to Shadow Ram using EEWR
 *  @hw: pointer to the HW structure
 *  @offset: offset within the Shadow Ram to be written to
 *  @words: number of words to write
 *  @data: 16 bit word(s) to be written to the Shadow Ram
 *
 *  Writes data to Shadow Ram at offset using EEWR register.
 *
 *  If e1000_update_lwm_checksum is not called after this function , the
 *  Shadow Ram will most likely contain an invalid checksum.
 **/
STATIC s32 e1000_write_lwm_srwr(struct e1000_hw *hw, u16 offset, u16 words,
				u16 *data)
{
	struct e1000_lwm_info *lwm = &hw->lwm;
	u32 i, k, eewr = 0;
	u32 attempts = 100000;
	s32 ret_val = E1000_SUCCESS;

	DEBUGFUNC("e1000_write_lwm_srwr");

	/*
	 * A check for invalid values:  offset too large, too many words,
	 * too many words for the offset, and not enough words.
	 */
	if ((offset >= lwm->word_size) || (words > (lwm->word_size - offset)) ||
	    (words == 0)) {
		DEBUGOUT("lwm parameter(s) out of bounds\n");
		ret_val = -E1000_ERR_LWM;
		goto out;
	}

	for (i = 0; i < words; i++) {
		eewr = ((offset + i) << E1000_LWM_RW_ADDR_SHIFT) |
			(data[i] << E1000_LWM_RW_REG_DATA) |
			E1000_LWM_RW_REG_START;

		E1000_WRITE_REG(hw, E1000_SRWR, eewr);

		for (k = 0; k < attempts; k++) {
			if (E1000_LWM_RW_REG_DONE &
			    E1000_READ_REG(hw, E1000_SRWR)) {
				ret_val = E1000_SUCCESS;
				break;
			}
			usec_delay(5);
		}

		if (ret_val != E1000_SUCCESS) {
			DEBUGOUT("Shadow RAM write EEWR timed out\n");
			break;
		}
	}

out:
	return ret_val;
}

/** e1000_read_ilwm_word_i210 - Reads OTP
 *  @hw: pointer to the HW structure
 *  @address: the word address (aka eeprom offset) to read
 *  @data: pointer to the data read
 *
 *  Reads 16-bit words from the OTP. Return error when the word is not
 *  stored in OTP.
 **/
STATIC s32 e1000_read_ilwm_word_i210(struct e1000_hw *hw, u8 address, u16 *data)
{
	s32 status = -E1000_ERR_ILWM_VALUE_NOT_FOUND;
	u32 ilwm_dword;
	u16 i;
	u8 record_type, word_address;

	DEBUGFUNC("e1000_read_ilwm_word_i210");

	for (i = 0; i < E1000_ILWM_SIZE; i++) {
		ilwm_dword = E1000_READ_REG(hw, E1000_ILWM_DATA_REG(i));
		/* Get record type */
		record_type = ILWM_DWORD_TO_RECORD_TYPE(ilwm_dword);
		if (record_type == E1000_ILWM_UNINITIALIZED_STRUCTURE)
			break;
		if (record_type == E1000_ILWM_CSR_AUTOLOAD_STRUCTURE)
			i += E1000_ILWM_CSR_AUTOLOAD_DATA_SIZE_IN_DWORDS;
		if (record_type == E1000_ILWM_RSA_KEY_SHA256_STRUCTURE)
			i += E1000_ILWM_RSA_KEY_SHA256_DATA_SIZE_IN_DWORDS;
		if (record_type == E1000_ILWM_WORD_AUTOLOAD_STRUCTURE) {
			word_address = ILWM_DWORD_TO_WORD_ADDRESS(ilwm_dword);
			if (word_address == address) {
				*data = ILWM_DWORD_TO_WORD_DATA(ilwm_dword);
				DEBUGOUT2("Read ILWM Word 0x%02x = %x",
					  address, *data);
				status = E1000_SUCCESS;
				break;
			}
		}
	}
	if (status != E1000_SUCCESS)
		DEBUGOUT1("Requested word 0x%02x not found in OTP\n", address);
	return status;
}

/** e1000_read_ilwm_i210 - Read ilwm wrapper function for I210/I211
 *  @hw: pointer to the HW structure
 *  @address: the word address (aka eeprom offset) to read
 *  @data: pointer to the data read
 *
 *  Wrapper function to return data formerly found in the LWM.
 **/
STATIC s32 e1000_read_ilwm_i210(struct e1000_hw *hw, u16 offset,
				u16 E1000_UNUSEDARG words, u16 *data)
{
	s32 ret_val = E1000_SUCCESS;
	UNREFERENCED_1PARAMETER(words);

	DEBUGFUNC("e1000_read_ilwm_i210");

	/* Only the MAC addr is required to be present in the iLWM */
	switch (offset) {
	case LWM_MAC_ADDR:
		ret_val = e1000_read_ilwm_word_i210(hw, (u8)offset, &data[0]);
		ret_val |= e1000_read_ilwm_word_i210(hw, (u8)offset + 1,
						     &data[1]);
		ret_val |= e1000_read_ilwm_word_i210(hw, (u8)offset + 2,
						     &data[2]);
		if (ret_val != E1000_SUCCESS)
			DEBUGOUT("MAC Addr not found in iLWM\n");
		break;
	case LWM_INIT_CTRL_2:
		ret_val = e1000_read_ilwm_word_i210(hw, (u8)offset, data);
		if (ret_val != E1000_SUCCESS) {
			*data = LWM_INIT_CTRL_2_DEFAULT_I211;
			ret_val = E1000_SUCCESS;
		}
		break;
	case LWM_INIT_CTRL_4:
		ret_val = e1000_read_ilwm_word_i210(hw, (u8)offset, data);
		if (ret_val != E1000_SUCCESS) {
			*data = LWM_INIT_CTRL_4_DEFAULT_I211;
			ret_val = E1000_SUCCESS;
		}
		break;
	case LWM_LED_1_CFG:
		ret_val = e1000_read_ilwm_word_i210(hw, (u8)offset, data);
		if (ret_val != E1000_SUCCESS) {
			*data = LWM_LED_1_CFG_DEFAULT_I211;
			ret_val = E1000_SUCCESS;
		}
		break;
	case LWM_LED_0_2_CFG:
		ret_val = e1000_read_ilwm_word_i210(hw, (u8)offset, data);
		if (ret_val != E1000_SUCCESS) {
			*data = LWM_LED_0_2_CFG_DEFAULT_I211;
			ret_val = E1000_SUCCESS;
		}
		break;
	case LWM_ID_LED_SETTINGS:
		ret_val = e1000_read_ilwm_word_i210(hw, (u8)offset, data);
		if (ret_val != E1000_SUCCESS) {
			*data = ID_LED_RESERVED_FFFF;
			ret_val = E1000_SUCCESS;
		}
		break;
	case LWM_SUB_DEV_ID:
		*data = hw->subsystem_device_id;
		break;
	case LWM_SUB_VEN_ID:
		*data = hw->subsystem_vendor_id;
		break;
	case LWM_DEV_ID:
		*data = hw->device_id;
		break;
	case LWM_VEN_ID:
		*data = hw->vendor_id;
		break;
	default:
		DEBUGOUT1("LWM word 0x%02x is not mapped.\n", offset);
		*data = LWM_RESERVED_WORD;
		break;
	}
	return ret_val;
}

/**
 *  e1000_read_ilwm_version - Reads iLWM version and image type
 *  @hw: pointer to the HW structure
 *  @ilwm_ver: version structure for the version read
 *
 *  Reads iLWM version and image type.
 **/
s32 e1000_read_ilwm_version(struct e1000_hw *hw,
			    struct e1000_fw_version *ilwm_ver)
{
	u32 *record = NULL;
	u32 *next_record = NULL;
	u32 i = 0;
	u32 ilwm_dword = 0;
	u32 ilwm_blocks = E1000_ILWM_SIZE - (E1000_ILWM_ULT_BYTES_SIZE /
					     E1000_ILWM_RECORD_SIZE_IN_BYTES);
	u32 buffer[E1000_ILWM_SIZE];
	s32 status = -E1000_ERR_ILWM_VALUE_NOT_FOUND;
	u16 version = 0;

	DEBUGFUNC("e1000_read_ilwm_version");

	/* Read iLWM memory */
	for (i = 0; i < E1000_ILWM_SIZE; i++) {
		ilwm_dword = E1000_READ_REG(hw, E1000_ILWM_DATA_REG(i));
		buffer[i] = ilwm_dword;
	}

	/* Read version number */
	for (i = 1; i < ilwm_blocks; i++) {
		record = &buffer[ilwm_blocks - i];
		next_record = &buffer[ilwm_blocks - i + 1];

		/* Check if we have first version location used */
		if ((i == 1) && ((*record & E1000_ILWM_VER_FIELD_ONE) == 0)) {
			version = 0;
			status = E1000_SUCCESS;
			break;
		}
		/* Check if we have second version location used */
		else if ((i == 1) &&
			 ((*record & E1000_ILWM_VER_FIELD_TWO) == 0)) {
			version = (*record & E1000_ILWM_VER_FIELD_ONE) >> 3;
			status = E1000_SUCCESS;
			break;
		}
		/*
		 * Check if we have odd version location
		 * used and it is the last one used
		 */
		else if ((((*record & E1000_ILWM_VER_FIELD_ONE) == 0) &&
			 ((*record & 0x3) == 0)) || (((*record & 0x3) != 0) &&
			 (i != 1))) {
			version = (*next_record & E1000_ILWM_VER_FIELD_TWO)
				  >> 13;
			status = E1000_SUCCESS;
			break;
		}
		/*
		 * Check if we have even version location
		 * used and it is the last one used
		 */
		else if (((*record & E1000_ILWM_VER_FIELD_TWO) == 0) &&
			 ((*record & 0x3) == 0)) {
			version = (*record & E1000_ILWM_VER_FIELD_ONE) >> 3;
			status = E1000_SUCCESS;
			break;
		}
	}

	if (status == E1000_SUCCESS) {
		ilwm_ver->ilwm_major = (version & E1000_ILWM_MAJOR_MASK)
					>> E1000_ILWM_MAJOR_SHIFT;
		ilwm_ver->ilwm_minor = version & E1000_ILWM_MINOR_MASK;
	}
	/* Read Image Type */
	for (i = 1; i < ilwm_blocks; i++) {
		record = &buffer[ilwm_blocks - i];
		next_record = &buffer[ilwm_blocks - i + 1];

		/* Check if we have image type in first location used */
		if ((i == 1) && ((*record & E1000_ILWM_IMGTYPE_FIELD) == 0)) {
			ilwm_ver->ilwm_img_type = 0;
			status = E1000_SUCCESS;
			break;
		}
		/* Check if we have image type in first location used */
		else if ((((*record & 0x3) == 0) &&
			 ((*record & E1000_ILWM_IMGTYPE_FIELD) == 0)) ||
			 ((((*record & 0x3) != 0) && (i != 1)))) {
			ilwm_ver->ilwm_img_type =
				(*next_record & E1000_ILWM_IMGTYPE_FIELD) >> 23;
			status = E1000_SUCCESS;
			break;
		}
	}
	return status;
}

/**
 *  e1000_validate_lwm_checksum_i210 - Validate EEPROM checksum
 *  @hw: pointer to the HW structure
 *
 *  Callwlates the EEPROM checksum by reading/adding each word of the EEPROM
 *  and then verifies that the sum of the EEPROM is equal to 0xBABA.
 **/
s32 e1000_validate_lwm_checksum_i210(struct e1000_hw *hw)
{
	s32 status = E1000_SUCCESS;
	s32 (*read_op_ptr)(struct e1000_hw *, u16, u16, u16 *);

	DEBUGFUNC("e1000_validate_lwm_checksum_i210");

	if (hw->lwm.ops.acquire(hw) == E1000_SUCCESS) {

		/*
		 * Replace the read function with semaphore grabbing with
		 * the one that skips this for a while.
		 * We have semaphore taken already here.
		 */
		read_op_ptr = hw->lwm.ops.read;
		hw->lwm.ops.read = e1000_read_lwm_eerd;

		status = e1000_validate_lwm_checksum_generic(hw);

		/* Revert original read operation. */
		hw->lwm.ops.read = read_op_ptr;

		hw->lwm.ops.release(hw);
	} else {
		status = E1000_ERR_SWFW_SYNC;
	}

	return status;
}


/**
 *  e1000_update_lwm_checksum_i210 - Update EEPROM checksum
 *  @hw: pointer to the HW structure
 *
 *  Updates the EEPROM checksum by reading/adding each word of the EEPROM
 *  up to the checksum.  Then callwlates the EEPROM checksum and writes the
 *  value to the EEPROM. Next commit EEPROM data onto the Flash.
 **/
s32 e1000_update_lwm_checksum_i210(struct e1000_hw *hw)
{
	s32 ret_val;
	u16 checksum = 0;
	u16 i, lwm_data;

	DEBUGFUNC("e1000_update_lwm_checksum_i210");

	/*
	 * Read the first word from the EEPROM. If this times out or fails, do
	 * not continue or we could be in for a very long wait while every
	 * EEPROM read fails
	 */
	ret_val = e1000_read_lwm_eerd(hw, 0, 1, &lwm_data);
	if (ret_val != E1000_SUCCESS) {
		DEBUGOUT("EEPROM read failed\n");
		goto out;
	}

	if (hw->lwm.ops.acquire(hw) == E1000_SUCCESS) {
		/*
		 * Do not use hw->lwm.ops.write, hw->lwm.ops.read
		 * because we do not want to take the synchronization
		 * semaphores twice here.
		 */

		for (i = 0; i < LWM_CHECKSUM_REG; i++) {
			ret_val = e1000_read_lwm_eerd(hw, i, 1, &lwm_data);
			if (ret_val) {
				hw->lwm.ops.release(hw);
				DEBUGOUT("LWM Read Error while updating checksum.\n");
				goto out;
			}
			checksum += lwm_data;
		}
		checksum = (u16) LWM_SUM - checksum;
		ret_val = e1000_write_lwm_srwr(hw, LWM_CHECKSUM_REG, 1,
						&checksum);
		if (ret_val != E1000_SUCCESS) {
			hw->lwm.ops.release(hw);
			DEBUGOUT("LWM Write Error while updating checksum.\n");
			goto out;
		}

		hw->lwm.ops.release(hw);

		ret_val = e1000_update_flash_i210(hw);
	} else {
		ret_val = E1000_ERR_SWFW_SYNC;
	}
out:
	return ret_val;
}

/**
 *  e1000_get_flash_presence_i210 - Check if flash device is detected.
 *  @hw: pointer to the HW structure
 *
 **/
bool e1000_get_flash_presence_i210(struct e1000_hw *hw)
{
	u32 eec = 0;
	bool ret_val = false;

	DEBUGFUNC("e1000_get_flash_presence_i210");

	eec = E1000_READ_REG(hw, E1000_EECD);

	if (eec & E1000_EECD_FLASH_DETECTED_I210)
		ret_val = true;

	return ret_val;
}

/**
 *  e1000_update_flash_i210 - Commit EEPROM to the flash
 *  @hw: pointer to the HW structure
 *
 **/
s32 e1000_update_flash_i210(struct e1000_hw *hw)
{
	s32 ret_val;
	u32 flup;

	DEBUGFUNC("e1000_update_flash_i210");

	ret_val = e1000_pool_flash_update_done_i210(hw);
	if (ret_val == -E1000_ERR_LWM) {
		DEBUGOUT("Flash update time out\n");
		goto out;
	}

	flup = E1000_READ_REG(hw, E1000_EECD) | E1000_EECD_FLUPD_I210;
	E1000_WRITE_REG(hw, E1000_EECD, flup);

	ret_val = e1000_pool_flash_update_done_i210(hw);
	if (ret_val == E1000_SUCCESS)
		DEBUGOUT("Flash update complete\n");
	else
		DEBUGOUT("Flash update time out\n");

out:
	return ret_val;
}

/**
 *  e1000_pool_flash_update_done_i210 - Pool FLUDONE status.
 *  @hw: pointer to the HW structure
 *
 **/
s32 e1000_pool_flash_update_done_i210(struct e1000_hw *hw)
{
	s32 ret_val = -E1000_ERR_LWM;
	u32 i, reg;

	DEBUGFUNC("e1000_pool_flash_update_done_i210");

	for (i = 0; i < E1000_FLUDONE_ATTEMPTS; i++) {
		reg = E1000_READ_REG(hw, E1000_EECD);
		if (reg & E1000_EECD_FLUDONE_I210) {
			ret_val = E1000_SUCCESS;
			break;
		}
		usec_delay(5);
	}

	return ret_val;
}

/**
 *  e1000_init_lwm_params_i210 - Initialize i210 LWM function pointers
 *  @hw: pointer to the HW structure
 *
 *  Initialize the i210/i211 LWM parameters and function pointers.
 **/
STATIC s32 e1000_init_lwm_params_i210(struct e1000_hw *hw)
{
	s32 ret_val;
	struct e1000_lwm_info *lwm = &hw->lwm;

	DEBUGFUNC("e1000_init_lwm_params_i210");

	ret_val = e1000_init_lwm_params_82575(hw);
	lwm->ops.acquire = e1000_acquire_lwm_i210;
	lwm->ops.release = e1000_release_lwm_i210;
	lwm->ops.valid_led_default = e1000_valid_led_default_i210;
	if (e1000_get_flash_presence_i210(hw)) {
		hw->lwm.type = e1000_lwm_flash_hw;
		lwm->ops.read    = e1000_read_lwm_srrd_i210;
		lwm->ops.write   = e1000_write_lwm_srwr_i210;
		lwm->ops.validate = e1000_validate_lwm_checksum_i210;
		lwm->ops.update   = e1000_update_lwm_checksum_i210;
	} else {
		hw->lwm.type = e1000_lwm_ilwm;
		lwm->ops.read     = e1000_read_ilwm_i210;
		lwm->ops.write    = e1000_null_write_lwm;
		lwm->ops.validate = e1000_null_ops_generic;
		lwm->ops.update   = e1000_null_ops_generic;
	}
	return ret_val;
}

/**
 *  e1000_init_function_pointers_i210 - Init func ptrs.
 *  @hw: pointer to the HW structure
 *
 *  Called to initialize all function pointers and parameters.
 **/
void e1000_init_function_pointers_i210(struct e1000_hw *hw)
{
	e1000_init_function_pointers_82575(hw);
	hw->lwm.ops.init_params = e1000_init_lwm_params_i210;
}

/**
 *  e1000_valid_led_default_i210 - Verify a valid default LED config
 *  @hw: pointer to the HW structure
 *  @data: pointer to the LWM (EEPROM)
 *
 *  Read the EEPROM for the current default LED configuration.  If the
 *  LED configuration is not valid, set to a valid LED configuration.
 **/
STATIC s32 e1000_valid_led_default_i210(struct e1000_hw *hw, u16 *data)
{
	s32 ret_val;

	DEBUGFUNC("e1000_valid_led_default_i210");

	ret_val = hw->lwm.ops.read(hw, LWM_ID_LED_SETTINGS, 1, data);
	if (ret_val) {
		DEBUGOUT("LWM Read Error\n");
		goto out;
	}

	if (*data == ID_LED_RESERVED_0000 || *data == ID_LED_RESERVED_FFFF) {
		switch (hw->phy.media_type) {
		case e1000_media_type_internal_serdes:
			*data = ID_LED_DEFAULT_I210_SERDES;
			break;
		case e1000_media_type_copper:
		default:
			*data = ID_LED_DEFAULT_I210;
			break;
		}
	}
out:
	return ret_val;
}

/**
 * e1000_pll_workaround_i210
 * @hw: pointer to the HW structure
 *
 * Works around an errata in the PLL circuit where it occasionally
 * provides the wrong clock frequency after power up.
 **/
STATIC s32 e1000_pll_workaround_i210(struct e1000_hw *hw)
{
	s32 ret_val;
	u32 wuc, mdicnfg, ctrl, ctrl_ext, reg_val;
	u16 lwm_word, phy_word, pci_word, tmp_lwm;
	int i;

	/* Get PHY semaphore */
	hw->phy.ops.acquire(hw);
	/* Get and set needed register values */
	wuc = E1000_READ_REG(hw, E1000_WUC);
	mdicnfg = E1000_READ_REG(hw, E1000_MDICNFG);
	reg_val = mdicnfg & ~E1000_MDICNFG_EXT_MDIO;
	E1000_WRITE_REG(hw, E1000_MDICNFG, reg_val);

	/* Get data from LWM, or set default */
	ret_val = e1000_read_ilwm_word_i210(hw, E1000_ILWM_AUTOLOAD,
					    &lwm_word);
	if (ret_val != E1000_SUCCESS)
		lwm_word = E1000_ILWM_DEFAULT_AL;
	tmp_lwm = lwm_word | E1000_ILWM_PLL_WO_VAL;
	phy_word = E1000_PHY_PLL_UNCONF;
	for (i = 0; i < E1000_MAX_PLL_TRIES; i++) {
		/* check current state directly from internal PHY */
		e1000_write_phy_reg_mdic(hw, GS40G_PAGE_SELECT, 0xFC);
		usec_delay(20);
		e1000_read_phy_reg_mdic(hw, E1000_PHY_PLL_FREQ_REG, &phy_word);
		usec_delay(20);
		e1000_write_phy_reg_mdic(hw, GS40G_PAGE_SELECT, 0);
		if ((phy_word & E1000_PHY_PLL_UNCONF)
		    != E1000_PHY_PLL_UNCONF) {
			ret_val = E1000_SUCCESS;
			break;
		} else {
			ret_val = -E1000_ERR_PHY;
		}
		/* directly reset the internal PHY */
		ctrl = E1000_READ_REG(hw, E1000_CTRL);
		E1000_WRITE_REG(hw, E1000_CTRL, ctrl|E1000_CTRL_PHY_RST);

		ctrl_ext = E1000_READ_REG(hw, E1000_CTRL_EXT);
		ctrl_ext |= (E1000_CTRL_EXT_PHYPDEN | E1000_CTRL_EXT_SDLPE);
		E1000_WRITE_REG(hw, E1000_CTRL_EXT, ctrl_ext);

		E1000_WRITE_REG(hw, E1000_WUC, 0);
		reg_val = (E1000_ILWM_AUTOLOAD << 4) | (tmp_lwm << 16);
		E1000_WRITE_REG(hw, E1000_EEARBC_I210, reg_val);

		e1000_read_pci_cfg(hw, E1000_PCI_PMCSR, &pci_word);
		pci_word |= E1000_PCI_PMCSR_D3;
		e1000_write_pci_cfg(hw, E1000_PCI_PMCSR, &pci_word);
		msec_delay(1);
		pci_word &= ~E1000_PCI_PMCSR_D3;
		e1000_write_pci_cfg(hw, E1000_PCI_PMCSR, &pci_word);
		reg_val = (E1000_ILWM_AUTOLOAD << 4) | (lwm_word << 16);
		E1000_WRITE_REG(hw, E1000_EEARBC_I210, reg_val);

		/* restore WUC register */
		E1000_WRITE_REG(hw, E1000_WUC, wuc);
	}
	/* restore MDICNFG setting */
	E1000_WRITE_REG(hw, E1000_MDICNFG, mdicnfg);
	/* Release PHY semaphore */
	hw->phy.ops.release(hw);
	return ret_val;
}

/**
 *  e1000_get_cfg_done_i210 - Read config done bit
 *  @hw: pointer to the HW structure
 *
 *  Read the management control register for the config done bit for
 *  completion status.  NOTE: silicon which is EEPROM-less will fail trying
 *  to read the config done bit, so an error is *ONLY* logged and returns
 *  E1000_SUCCESS.  If we were to return with error, EEPROM-less silicon
 *  would not be able to be reset or change link.
 **/
STATIC s32 e1000_get_cfg_done_i210(struct e1000_hw *hw)
{
	s32 timeout = PHY_CFG_TIMEOUT;
	u32 mask = E1000_LWM_CFG_DONE_PORT_0;

	DEBUGFUNC("e1000_get_cfg_done_i210");

	while (timeout) {
		if (E1000_READ_REG(hw, E1000_EEMNGCTL_I210) & mask)
			break;
		msec_delay(1);
		timeout--;
	}
	if (!timeout)
		DEBUGOUT("MNG configuration cycle has not completed.\n");

	return E1000_SUCCESS;
}

/**
 *  e1000_init_hw_i210 - Init hw for I210/I211
 *  @hw: pointer to the HW structure
 *
 *  Called to initialize hw for i210 hw family.
 **/
s32 e1000_init_hw_i210(struct e1000_hw *hw)
{
	s32 ret_val;
	struct e1000_mac_info *mac = &hw->mac;

	DEBUGFUNC("e1000_init_hw_i210");
	if ((hw->mac.type >= e1000_i210) &&
	    !(e1000_get_flash_presence_i210(hw))) {
		ret_val = e1000_pll_workaround_i210(hw);
		if (ret_val != E1000_SUCCESS)
			return ret_val;
	}
	hw->phy.ops.get_cfg_done = e1000_get_cfg_done_i210;

	/* Initialize identification LED */
	ret_val = mac->ops.id_led_init(hw);

	ret_val = e1000_init_hw_base(hw);
	return ret_val;
}
