/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2001-2020 Intel Corporation
 */

#include "igc_api.h"

static s32 igc_init_lwm_params_i225(struct igc_hw *hw);
static s32 igc_init_mac_params_i225(struct igc_hw *hw);
static s32 igc_init_phy_params_i225(struct igc_hw *hw);
static s32 igc_reset_hw_i225(struct igc_hw *hw);
static s32 igc_acquire_lwm_i225(struct igc_hw *hw);
static void igc_release_lwm_i225(struct igc_hw *hw);
static s32 igc_get_hw_semaphore_i225(struct igc_hw *hw);
static s32 __igc_write_lwm_srwr(struct igc_hw *hw, u16 offset, u16 words,
				  u16 *data);
static s32 igc_pool_flash_update_done_i225(struct igc_hw *hw);
static s32 igc_valid_led_default_i225(struct igc_hw *hw, u16 *data);

/**
 *  igc_init_lwm_params_i225 - Init LWM func ptrs.
 *  @hw: pointer to the HW structure
 **/
static s32 igc_init_lwm_params_i225(struct igc_hw *hw)
{
	struct igc_lwm_info *lwm = &hw->lwm;
	u32 eecd = IGC_READ_REG(hw, IGC_EECD);
	u16 size;

	DEBUGFUNC("igc_init_lwm_params_i225");

	size = (u16)((eecd & IGC_EECD_SIZE_EX_MASK) >>
		     IGC_EECD_SIZE_EX_SHIFT);
	/*
	 * Added to a constant, "size" becomes the left-shift value
	 * for setting word_size.
	 */
	size += LWM_WORD_SIZE_BASE_SHIFT;

	/* Just in case size is out of range, cap it to the largest
	 * EEPROM size supported
	 */
	if (size > 15)
		size = 15;

	lwm->word_size = 1 << size;
	lwm->opcode_bits = 8;
	lwm->delay_usec = 1;
	lwm->type = igc_lwm_eeprom_spi;


	lwm->page_size = eecd & IGC_EECD_ADDR_BITS ? 32 : 8;
	lwm->address_bits = eecd & IGC_EECD_ADDR_BITS ?
			    16 : 8;

	if (lwm->word_size == (1 << 15))
		lwm->page_size = 128;

	lwm->ops.acquire = igc_acquire_lwm_i225;
	lwm->ops.release = igc_release_lwm_i225;
	lwm->ops.valid_led_default = igc_valid_led_default_i225;
	if (igc_get_flash_presence_i225(hw)) {
		hw->lwm.type = igc_lwm_flash_hw;
		lwm->ops.read    = igc_read_lwm_srrd_i225;
		lwm->ops.write   = igc_write_lwm_srwr_i225;
		lwm->ops.validate = igc_validate_lwm_checksum_i225;
		lwm->ops.update   = igc_update_lwm_checksum_i225;
	} else {
		hw->lwm.type = igc_lwm_ilwm;
		lwm->ops.write    = igc_null_write_lwm;
		lwm->ops.validate = igc_null_ops_generic;
		lwm->ops.update   = igc_null_ops_generic;
	}

	return IGC_SUCCESS;
}

/**
 *  igc_init_mac_params_i225 - Init MAC func ptrs.
 *  @hw: pointer to the HW structure
 **/
static s32 igc_init_mac_params_i225(struct igc_hw *hw)
{
	struct igc_mac_info *mac = &hw->mac;
	struct igc_dev_spec_i225 *dev_spec = &hw->dev_spec._i225;

	DEBUGFUNC("igc_init_mac_params_i225");

	/* Initialize function pointer */
	igc_init_mac_ops_generic(hw);

	/* Set media type */
	hw->phy.media_type = igc_media_type_copper;
	/* Set mta register count */
	mac->mta_reg_count = 128;
	/* Set rar entry count */
	mac->rar_entry_count = IGC_RAR_ENTRIES_BASE;

	/* reset */
	mac->ops.reset_hw = igc_reset_hw_i225;
	/* hw initialization */
	mac->ops.init_hw = igc_init_hw_i225;
	/* link setup */
	mac->ops.setup_link = igc_setup_link_generic;
	/* check for link */
	mac->ops.check_for_link = igc_check_for_link_i225;
	/* link info */
	mac->ops.get_link_up_info = igc_get_speed_and_duplex_copper_generic;
	/* acquire SW_FW sync */
	mac->ops.acquire_swfw_sync = igc_acquire_swfw_sync_i225;
	/* release SW_FW sync */
	mac->ops.release_swfw_sync = igc_release_swfw_sync_i225;

	/* Allow a single clear of the SW semaphore on I225 */
	dev_spec->clear_semaphore_once = true;
	mac->ops.setup_physical_interface = igc_setup_copper_link_i225;

	/* Set if part includes ASF firmware */
	mac->asf_firmware_present = true;

	/* multicast address update */
	mac->ops.update_mc_addr_list = igc_update_mc_addr_list_generic;

	mac->ops.write_vfta = igc_write_vfta_generic;

	return IGC_SUCCESS;
}

/**
 *  igc_init_phy_params_i225 - Init PHY func ptrs.
 *  @hw: pointer to the HW structure
 **/
static s32 igc_init_phy_params_i225(struct igc_hw *hw)
{
	struct igc_phy_info *phy = &hw->phy;
	s32 ret_val = IGC_SUCCESS;
	u32 ctrl_ext;

	DEBUGFUNC("igc_init_phy_params_i225");

	phy->ops.read_i2c_byte = igc_read_i2c_byte_generic;
	phy->ops.write_i2c_byte = igc_write_i2c_byte_generic;

	if (hw->phy.media_type != igc_media_type_copper) {
		phy->type = igc_phy_none;
		goto out;
	}

	phy->ops.power_up   = igc_power_up_phy_copper;
	phy->ops.power_down = igc_power_down_phy_copper_base;

	phy->autoneg_mask = AUTONEG_ADVERTISE_SPEED_DEFAULT_2500;

	phy->reset_delay_us	= 100;

	phy->ops.acquire	= igc_acquire_phy_base;
	phy->ops.check_reset_block = igc_check_reset_block_generic;
	phy->ops.commit		= igc_phy_sw_reset_generic;
	phy->ops.release	= igc_release_phy_base;

	ctrl_ext = IGC_READ_REG(hw, IGC_CTRL_EXT);

	/* Make sure the PHY is in a good state. Several people have reported
	 * firmware leaving the PHY's page select register set to something
	 * other than the default of zero, which causes the PHY ID read to
	 * access something other than the intended register.
	 */
	ret_val = hw->phy.ops.reset(hw);
	if (ret_val)
		goto out;

	IGC_WRITE_REG(hw, IGC_CTRL_EXT, ctrl_ext);
	phy->ops.read_reg = igc_read_phy_reg_gpy;
	phy->ops.write_reg = igc_write_phy_reg_gpy;

	ret_val = igc_get_phy_id(hw);
	/* Verify phy id and set remaining function pointers */
	switch (phy->id) {
	case I225_I_PHY_ID:
		phy->type		= igc_phy_i225;
		phy->ops.set_d0_lplu_state = igc_set_d0_lplu_state_i225;
		phy->ops.set_d3_lplu_state = igc_set_d3_lplu_state_i225;
		/* TODO - complete with GPY PHY information */
		break;
	default:
		ret_val = -IGC_ERR_PHY;
		goto out;
	}

out:
	return ret_val;
}

/**
 *  igc_reset_hw_i225 - Reset hardware
 *  @hw: pointer to the HW structure
 *
 *  This resets the hardware into a known state.
 **/
static s32 igc_reset_hw_i225(struct igc_hw *hw)
{
	u32 ctrl;
	s32 ret_val;

	DEBUGFUNC("igc_reset_hw_i225");

	/*
	 * Prevent the PCI-E bus from sticking if there is no TLP connection
	 * on the last TLP read/write transaction when MAC is reset.
	 */
	ret_val = igc_disable_pcie_master_generic(hw);
	if (ret_val)
		DEBUGOUT("PCI-E Master disable polling has failed.\n");

	DEBUGOUT("Masking off all interrupts\n");
	IGC_WRITE_REG(hw, IGC_IMC, 0xffffffff);

	IGC_WRITE_REG(hw, IGC_RCTL, 0);
	IGC_WRITE_REG(hw, IGC_TCTL, IGC_TCTL_PSP);
	IGC_WRITE_FLUSH(hw);

	msec_delay(10);

	ctrl = IGC_READ_REG(hw, IGC_CTRL);

	DEBUGOUT("Issuing a global reset to MAC\n");
	IGC_WRITE_REG(hw, IGC_CTRL, ctrl | IGC_CTRL_RST);

	ret_val = igc_get_auto_rd_done_generic(hw);
	if (ret_val) {
		/*
		 * When auto config read does not complete, do not
		 * return with an error. This can happen in situations
		 * where there is no eeprom and prevents getting link.
		 */
		DEBUGOUT("Auto Read Done did not complete\n");
	}

	/* Clear any pending interrupt events. */
	IGC_WRITE_REG(hw, IGC_IMC, 0xffffffff);
	IGC_READ_REG(hw, IGC_ICR);

	/* Install any alternate MAC address into RAR0 */
	ret_val = igc_check_alt_mac_addr_generic(hw);

	return ret_val;
}

/* igc_acquire_lwm_i225 - Request for access to EEPROM
 * @hw: pointer to the HW structure
 *
 * Acquire the necessary semaphores for exclusive access to the EEPROM.
 * Set the EEPROM access request bit and wait for EEPROM access grant bit.
 * Return successful if access grant bit set, else clear the request for
 * EEPROM access and return -IGC_ERR_LWM (-1).
 */
static s32 igc_acquire_lwm_i225(struct igc_hw *hw)
{
	s32 ret_val;

	DEBUGFUNC("igc_acquire_lwm_i225");

	ret_val = igc_acquire_swfw_sync_i225(hw, IGC_SWFW_EEP_SM);

	return ret_val;
}

/* igc_release_lwm_i225 - Release exclusive access to EEPROM
 * @hw: pointer to the HW structure
 *
 * Stop any current commands to the EEPROM and clear the EEPROM request bit,
 * then release the semaphores acquired.
 */
static void igc_release_lwm_i225(struct igc_hw *hw)
{
	DEBUGFUNC("igc_release_lwm_i225");

	igc_release_swfw_sync_i225(hw, IGC_SWFW_EEP_SM);
}

/* igc_acquire_swfw_sync_i225 - Acquire SW/FW semaphore
 * @hw: pointer to the HW structure
 * @mask: specifies which semaphore to acquire
 *
 * Acquire the SW/FW semaphore to access the PHY or LWM.  The mask
 * will also specify which port we're acquiring the lock for.
 */
s32 igc_acquire_swfw_sync_i225(struct igc_hw *hw, u16 mask)
{
	u32 swfw_sync;
	u32 swmask = mask;
	u32 fwmask = mask << 16;
	s32 ret_val = IGC_SUCCESS;
	s32 i = 0, timeout = 200; /* FIXME: find real value to use here */

	DEBUGFUNC("igc_acquire_swfw_sync_i225");

	while (i < timeout) {
		if (igc_get_hw_semaphore_i225(hw)) {
			ret_val = -IGC_ERR_SWFW_SYNC;
			goto out;
		}

		swfw_sync = IGC_READ_REG(hw, IGC_SW_FW_SYNC);
		if (!(swfw_sync & (fwmask | swmask)))
			break;

		/* Firmware lwrrently using resource (fwmask)
		 * or other software thread using resource (swmask)
		 */
		igc_put_hw_semaphore_generic(hw);
		msec_delay_irq(5);
		i++;
	}

	if (i == timeout) {
		DEBUGOUT("Driver can't access resource, SW_FW_SYNC timeout.\n");
		ret_val = -IGC_ERR_SWFW_SYNC;
		goto out;
	}

	swfw_sync |= swmask;
	IGC_WRITE_REG(hw, IGC_SW_FW_SYNC, swfw_sync);

	igc_put_hw_semaphore_generic(hw);

out:
	return ret_val;
}

/* igc_release_swfw_sync_i225 - Release SW/FW semaphore
 * @hw: pointer to the HW structure
 * @mask: specifies which semaphore to acquire
 *
 * Release the SW/FW semaphore used to access the PHY or LWM.  The mask
 * will also specify which port we're releasing the lock for.
 */
void igc_release_swfw_sync_i225(struct igc_hw *hw, u16 mask)
{
	u32 swfw_sync;

	DEBUGFUNC("igc_release_swfw_sync_i225");

	while (igc_get_hw_semaphore_i225(hw) != IGC_SUCCESS)
		; /* Empty */

	swfw_sync = IGC_READ_REG(hw, IGC_SW_FW_SYNC);
	swfw_sync &= ~mask;
	IGC_WRITE_REG(hw, IGC_SW_FW_SYNC, swfw_sync);

	igc_put_hw_semaphore_generic(hw);
}

/*
 * igc_setup_copper_link_i225 - Configure copper link settings
 * @hw: pointer to the HW structure
 *
 * Configures the link for auto-neg or forced speed and duplex.  Then we check
 * for link, once link is established calls to configure collision distance
 * and flow control are called.
 */
s32 igc_setup_copper_link_i225(struct igc_hw *hw)
{
	u32 phpm_reg;
	s32 ret_val;
	u32 ctrl;

	DEBUGFUNC("igc_setup_copper_link_i225");

	ctrl = IGC_READ_REG(hw, IGC_CTRL);
	ctrl |= IGC_CTRL_SLU;
	ctrl &= ~(IGC_CTRL_FRCSPD | IGC_CTRL_FRCDPX);
	IGC_WRITE_REG(hw, IGC_CTRL, ctrl);

	phpm_reg = IGC_READ_REG(hw, IGC_I225_PHPM);
	phpm_reg &= ~IGC_I225_PHPM_GO_LINKD;
	IGC_WRITE_REG(hw, IGC_I225_PHPM, phpm_reg);

	ret_val = igc_setup_copper_link_generic(hw);

	return ret_val;
}

/* igc_get_hw_semaphore_i225 - Acquire hardware semaphore
 * @hw: pointer to the HW structure
 *
 * Acquire the HW semaphore to access the PHY or LWM
 */
static s32 igc_get_hw_semaphore_i225(struct igc_hw *hw)
{
	u32 swsm;
	s32 timeout = hw->lwm.word_size + 1;
	s32 i = 0;

	DEBUGFUNC("igc_get_hw_semaphore_i225");

	/* Get the SW semaphore */
	while (i < timeout) {
		swsm = IGC_READ_REG(hw, IGC_SWSM);
		if (!(swsm & IGC_SWSM_SMBI))
			break;

		usec_delay(50);
		i++;
	}

	if (i == timeout) {
		/* In rare cirlwmstances, the SW semaphore may already be held
		 * unintentionally. Clear the semaphore once before giving up.
		 */
		if (hw->dev_spec._i225.clear_semaphore_once) {
			hw->dev_spec._i225.clear_semaphore_once = false;
			igc_put_hw_semaphore_generic(hw);
			for (i = 0; i < timeout; i++) {
				swsm = IGC_READ_REG(hw, IGC_SWSM);
				if (!(swsm & IGC_SWSM_SMBI))
					break;

				usec_delay(50);
			}
		}

		/* If we do not have the semaphore here, we have to give up. */
		if (i == timeout) {
			DEBUGOUT("Driver can't access device -\n");
			DEBUGOUT("SMBI bit is set.\n");
			return -IGC_ERR_LWM;
		}
	}

	/* Get the FW semaphore. */
	for (i = 0; i < timeout; i++) {
		swsm = IGC_READ_REG(hw, IGC_SWSM);
		IGC_WRITE_REG(hw, IGC_SWSM, swsm | IGC_SWSM_SWESMBI);

		/* Semaphore acquired if bit latched */
		if (IGC_READ_REG(hw, IGC_SWSM) & IGC_SWSM_SWESMBI)
			break;

		usec_delay(50);
	}

	if (i == timeout) {
		/* Release semaphores */
		igc_put_hw_semaphore_generic(hw);
		DEBUGOUT("Driver can't access the LWM\n");
		return -IGC_ERR_LWM;
	}

	return IGC_SUCCESS;
}

/* igc_read_lwm_srrd_i225 - Reads Shadow Ram using EERD register
 * @hw: pointer to the HW structure
 * @offset: offset of word in the Shadow Ram to read
 * @words: number of words to read
 * @data: word read from the Shadow Ram
 *
 * Reads a 16 bit word from the Shadow Ram using the EERD register.
 * Uses necessary synchronization semaphores.
 */
s32 igc_read_lwm_srrd_i225(struct igc_hw *hw, u16 offset, u16 words,
			     u16 *data)
{
	s32 status = IGC_SUCCESS;
	u16 i, count;

	DEBUGFUNC("igc_read_lwm_srrd_i225");

	/* We cannot hold synchronization semaphores for too long,
	 * because of forceful takeover procedure. However it is more efficient
	 * to read in bursts than synchronizing access for each word.
	 */
	for (i = 0; i < words; i += IGC_EERD_EEWR_MAX_COUNT) {
		count = (words - i) / IGC_EERD_EEWR_MAX_COUNT > 0 ?
			IGC_EERD_EEWR_MAX_COUNT : (words - i);
		if (hw->lwm.ops.acquire(hw) == IGC_SUCCESS) {
			status = igc_read_lwm_eerd(hw, offset, count,
						     data + i);
			hw->lwm.ops.release(hw);
		} else {
			status = IGC_ERR_SWFW_SYNC;
		}

		if (status != IGC_SUCCESS)
			break;
	}

	return status;
}

/* igc_write_lwm_srwr_i225 - Write to Shadow RAM using EEWR
 * @hw: pointer to the HW structure
 * @offset: offset within the Shadow RAM to be written to
 * @words: number of words to write
 * @data: 16 bit word(s) to be written to the Shadow RAM
 *
 * Writes data to Shadow RAM at offset using EEWR register.
 *
 * If igc_update_lwm_checksum is not called after this function , the
 * data will not be committed to FLASH and also Shadow RAM will most likely
 * contain an invalid checksum.
 *
 * If error code is returned, data and Shadow RAM may be inconsistent - buffer
 * partially written.
 */
s32 igc_write_lwm_srwr_i225(struct igc_hw *hw, u16 offset, u16 words,
			      u16 *data)
{
	s32 status = IGC_SUCCESS;
	u16 i, count;

	DEBUGFUNC("igc_write_lwm_srwr_i225");

	/* We cannot hold synchronization semaphores for too long,
	 * because of forceful takeover procedure. However it is more efficient
	 * to write in bursts than synchronizing access for each word.
	 */
	for (i = 0; i < words; i += IGC_EERD_EEWR_MAX_COUNT) {
		count = (words - i) / IGC_EERD_EEWR_MAX_COUNT > 0 ?
			IGC_EERD_EEWR_MAX_COUNT : (words - i);
		if (hw->lwm.ops.acquire(hw) == IGC_SUCCESS) {
			status = __igc_write_lwm_srwr(hw, offset, count,
							data + i);
			hw->lwm.ops.release(hw);
		} else {
			status = IGC_ERR_SWFW_SYNC;
		}

		if (status != IGC_SUCCESS)
			break;
	}

	return status;
}

/* __igc_write_lwm_srwr - Write to Shadow Ram using EEWR
 * @hw: pointer to the HW structure
 * @offset: offset within the Shadow Ram to be written to
 * @words: number of words to write
 * @data: 16 bit word(s) to be written to the Shadow Ram
 *
 * Writes data to Shadow Ram at offset using EEWR register.
 *
 * If igc_update_lwm_checksum is not called after this function , the
 * Shadow Ram will most likely contain an invalid checksum.
 */
static s32 __igc_write_lwm_srwr(struct igc_hw *hw, u16 offset, u16 words,
				  u16 *data)
{
	struct igc_lwm_info *lwm = &hw->lwm;
	u32 i, k, eewr = 0;
	u32 attempts = 100000;
	s32 ret_val = IGC_SUCCESS;

	DEBUGFUNC("__igc_write_lwm_srwr");

	/* A check for invalid values:  offset too large, too many words,
	 * too many words for the offset, and not enough words.
	 */
	if (offset >= lwm->word_size || words > (lwm->word_size - offset) ||
			words == 0) {
		DEBUGOUT("lwm parameter(s) out of bounds\n");
		ret_val = -IGC_ERR_LWM;
		goto out;
	}

	for (i = 0; i < words; i++) {
		eewr = ((offset + i) << IGC_LWM_RW_ADDR_SHIFT) |
			(data[i] << IGC_LWM_RW_REG_DATA) |
			IGC_LWM_RW_REG_START;

		IGC_WRITE_REG(hw, IGC_SRWR, eewr);

		for (k = 0; k < attempts; k++) {
			if (IGC_LWM_RW_REG_DONE &
			    IGC_READ_REG(hw, IGC_SRWR)) {
				ret_val = IGC_SUCCESS;
				break;
			}
			usec_delay(5);
		}

		if (ret_val != IGC_SUCCESS) {
			DEBUGOUT("Shadow RAM write EEWR timed out\n");
			break;
		}
	}

out:
	return ret_val;
}

/* igc_read_ilwm_version_i225 - Reads iLWM version and image type
 * @hw: pointer to the HW structure
 * @ilwm_ver: version structure for the version read
 *
 * Reads iLWM version and image type.
 */
s32 igc_read_ilwm_version_i225(struct igc_hw *hw,
				 struct igc_fw_version *ilwm_ver)
{
	u32 *record = NULL;
	u32 *next_record = NULL;
	u32 i = 0;
	u32 ilwm_dword = 0;
	u32 ilwm_blocks = IGC_ILWM_SIZE - (IGC_ILWM_ULT_BYTES_SIZE /
					     IGC_ILWM_RECORD_SIZE_IN_BYTES);
	u32 buffer[IGC_ILWM_SIZE];
	s32 status = -IGC_ERR_ILWM_VALUE_NOT_FOUND;
	u16 version = 0;

	DEBUGFUNC("igc_read_ilwm_version_i225");

	/* Read iLWM memory */
	for (i = 0; i < IGC_ILWM_SIZE; i++) {
		ilwm_dword = IGC_READ_REG(hw, IGC_ILWM_DATA_REG(i));
		buffer[i] = ilwm_dword;
	}

	/* Read version number */
	for (i = 1; i < ilwm_blocks; i++) {
		record = &buffer[ilwm_blocks - i];
		next_record = &buffer[ilwm_blocks - i + 1];

		/* Check if we have first version location used */
		if (i == 1 && (*record & IGC_ILWM_VER_FIELD_ONE) == 0) {
			version = 0;
			status = IGC_SUCCESS;
			break;
		}
		/* Check if we have second version location used */
		else if ((i == 1) &&
			 ((*record & IGC_ILWM_VER_FIELD_TWO) == 0)) {
			version = (*record & IGC_ILWM_VER_FIELD_ONE) >> 3;
			status = IGC_SUCCESS;
			break;
		}
		/* Check if we have odd version location
		 * used and it is the last one used
		 */
		else if ((((*record & IGC_ILWM_VER_FIELD_ONE) == 0) &&
			  ((*record & 0x3) == 0)) || (((*record & 0x3) != 0) &&
			   (i != 1))) {
			version = (*next_record & IGC_ILWM_VER_FIELD_TWO)
				  >> 13;
			status = IGC_SUCCESS;
			break;
		}
		/* Check if we have even version location
		 * used and it is the last one used
		 */
		else if (((*record & IGC_ILWM_VER_FIELD_TWO) == 0) &&
			 ((*record & 0x3) == 0)) {
			version = (*record & IGC_ILWM_VER_FIELD_ONE) >> 3;
			status = IGC_SUCCESS;
			break;
		}
	}

	if (status == IGC_SUCCESS) {
		ilwm_ver->ilwm_major = (version & IGC_ILWM_MAJOR_MASK)
					>> IGC_ILWM_MAJOR_SHIFT;
		ilwm_ver->ilwm_minor = version & IGC_ILWM_MINOR_MASK;
	}
	/* Read Image Type */
	for (i = 1; i < ilwm_blocks; i++) {
		record = &buffer[ilwm_blocks - i];
		next_record = &buffer[ilwm_blocks - i + 1];

		/* Check if we have image type in first location used */
		if (i == 1 && (*record & IGC_ILWM_IMGTYPE_FIELD) == 0) {
			ilwm_ver->ilwm_img_type = 0;
			status = IGC_SUCCESS;
			break;
		}
		/* Check if we have image type in first location used */
		else if ((((*record & 0x3) == 0) &&
			  ((*record & IGC_ILWM_IMGTYPE_FIELD) == 0)) ||
			    ((((*record & 0x3) != 0) && (i != 1)))) {
			ilwm_ver->ilwm_img_type =
				(*next_record & IGC_ILWM_IMGTYPE_FIELD) >> 23;
			status = IGC_SUCCESS;
			break;
		}
	}
	return status;
}

/* igc_validate_lwm_checksum_i225 - Validate EEPROM checksum
 * @hw: pointer to the HW structure
 *
 * Callwlates the EEPROM checksum by reading/adding each word of the EEPROM
 * and then verifies that the sum of the EEPROM is equal to 0xBABA.
 */
s32 igc_validate_lwm_checksum_i225(struct igc_hw *hw)
{
	s32 status = IGC_SUCCESS;
	s32 (*read_op_ptr)(struct igc_hw *hw, u16 offset,
			u16 count, u16 *data);

	DEBUGFUNC("igc_validate_lwm_checksum_i225");

	if (hw->lwm.ops.acquire(hw) == IGC_SUCCESS) {
		/* Replace the read function with semaphore grabbing with
		 * the one that skips this for a while.
		 * We have semaphore taken already here.
		 */
		read_op_ptr = hw->lwm.ops.read;
		hw->lwm.ops.read = igc_read_lwm_eerd;

		status = igc_validate_lwm_checksum_generic(hw);

		/* Revert original read operation. */
		hw->lwm.ops.read = read_op_ptr;

		hw->lwm.ops.release(hw);
	} else {
		status = IGC_ERR_SWFW_SYNC;
	}

	return status;
}

/* igc_update_lwm_checksum_i225 - Update EEPROM checksum
 * @hw: pointer to the HW structure
 *
 * Updates the EEPROM checksum by reading/adding each word of the EEPROM
 * up to the checksum.  Then callwlates the EEPROM checksum and writes the
 * value to the EEPROM. Next commit EEPROM data onto the Flash.
 */
s32 igc_update_lwm_checksum_i225(struct igc_hw *hw)
{
	s32 ret_val;
	u16 checksum = 0;
	u16 i, lwm_data;

	DEBUGFUNC("igc_update_lwm_checksum_i225");

	/* Read the first word from the EEPROM. If this times out or fails, do
	 * not continue or we could be in for a very long wait while every
	 * EEPROM read fails
	 */
	ret_val = igc_read_lwm_eerd(hw, 0, 1, &lwm_data);
	if (ret_val != IGC_SUCCESS) {
		DEBUGOUT("EEPROM read failed\n");
		goto out;
	}

	if (hw->lwm.ops.acquire(hw) == IGC_SUCCESS) {
		/* Do not use hw->lwm.ops.write, hw->lwm.ops.read
		 * because we do not want to take the synchronization
		 * semaphores twice here.
		 */

		for (i = 0; i < LWM_CHECKSUM_REG; i++) {
			ret_val = igc_read_lwm_eerd(hw, i, 1, &lwm_data);
			if (ret_val) {
				hw->lwm.ops.release(hw);
				DEBUGOUT("LWM Read Error while updating\n");
				DEBUGOUT("checksum.\n");
				goto out;
			}
			checksum += lwm_data;
		}
		checksum = (u16)LWM_SUM - checksum;
		ret_val = __igc_write_lwm_srwr(hw, LWM_CHECKSUM_REG, 1,
						 &checksum);
		if (ret_val != IGC_SUCCESS) {
			hw->lwm.ops.release(hw);
			DEBUGOUT("LWM Write Error while updating checksum.\n");
			goto out;
		}

		hw->lwm.ops.release(hw);

		ret_val = igc_update_flash_i225(hw);
	} else {
		ret_val = IGC_ERR_SWFW_SYNC;
	}
out:
	return ret_val;
}

/* igc_get_flash_presence_i225 - Check if flash device is detected.
 * @hw: pointer to the HW structure
 */
bool igc_get_flash_presence_i225(struct igc_hw *hw)
{
	u32 eec = 0;
	bool ret_val = false;

	DEBUGFUNC("igc_get_flash_presence_i225");

	eec = IGC_READ_REG(hw, IGC_EECD);

	if (eec & IGC_EECD_FLASH_DETECTED_I225)
		ret_val = true;

	return ret_val;
}

/* igc_set_flsw_flash_burst_counter_i225 - sets FLSW LWM Burst
 * Counter in FLSWCNT register.
 *
 * @hw: pointer to the HW structure
 * @burst_counter: size in bytes of the Flash burst to read or write
 */
s32 igc_set_flsw_flash_burst_counter_i225(struct igc_hw *hw,
					    u32 burst_counter)
{
	s32 ret_val = IGC_SUCCESS;

	DEBUGFUNC("igc_set_flsw_flash_burst_counter_i225");

	/* Validate input data */
	if (burst_counter < IGC_I225_SHADOW_RAM_SIZE) {
		/* Write FLSWCNT - burst counter */
		IGC_WRITE_REG(hw, IGC_I225_FLSWCNT, burst_counter);
	} else {
		ret_val = IGC_ERR_ILWALID_ARGUMENT;
	}

	return ret_val;
}

/* igc_write_erase_flash_command_i225 - write/erase to a sector
 * region on a given address.
 *
 * @hw: pointer to the HW structure
 * @opcode: opcode to be used for the write command
 * @address: the offset to write into the FLASH image
 */
s32 igc_write_erase_flash_command_i225(struct igc_hw *hw, u32 opcode,
					 u32 address)
{
	u32 flswctl = 0;
	s32 timeout = IGC_LWM_GRANT_ATTEMPTS;
	s32 ret_val = IGC_SUCCESS;

	DEBUGFUNC("igc_write_erase_flash_command_i225");

	flswctl = IGC_READ_REG(hw, IGC_I225_FLSWCTL);
	/* Polling done bit on FLSWCTL register */
	while (timeout) {
		if (flswctl & IGC_FLSWCTL_DONE)
			break;
		usec_delay(5);
		flswctl = IGC_READ_REG(hw, IGC_I225_FLSWCTL);
		timeout--;
	}

	if (!timeout) {
		DEBUGOUT("Flash transaction was not done\n");
		return -IGC_ERR_LWM;
	}

	/* Build and issue command on FLSWCTL register */
	flswctl = address | opcode;
	IGC_WRITE_REG(hw, IGC_I225_FLSWCTL, flswctl);

	/* Check if issued command is valid on FLSWCTL register */
	flswctl = IGC_READ_REG(hw, IGC_I225_FLSWCTL);
	if (!(flswctl & IGC_FLSWCTL_CMDV)) {
		DEBUGOUT("Write flash command failed\n");
		ret_val = IGC_ERR_ILWALID_ARGUMENT;
	}

	return ret_val;
}

/* igc_update_flash_i225 - Commit EEPROM to the flash
 * if fw_valid_bit is set, FW is active. setting FLUPD bit in EEC
 * register makes the FW load the internal shadow RAM into the flash.
 * Otherwise, fw_valid_bit is 0. if FL_SELW.block_prtotected_sw = 0
 * then FW is not active so the SW is responsible shadow RAM dump.
 *
 * @hw: pointer to the HW structure
 */
s32 igc_update_flash_i225(struct igc_hw *hw)
{
	u16 lwrrent_offset_data = 0;
	u32 block_sw_protect = 1;
	u16 base_address = 0x0;
	u32 i, fw_valid_bit;
	u16 lwrrent_offset;
	s32 ret_val = 0;
	u32 flup;

	DEBUGFUNC("igc_update_flash_i225");

	block_sw_protect = IGC_READ_REG(hw, IGC_I225_FLSELW) &
					  IGC_FLSELW_BLK_SW_ACCESS_I225;
	fw_valid_bit = IGC_READ_REG(hw, IGC_FWSM) &
				      IGC_FWSM_FW_VALID_I225;
	if (fw_valid_bit) {
		ret_val = igc_pool_flash_update_done_i225(hw);
		if (ret_val == -IGC_ERR_LWM) {
			DEBUGOUT("Flash update time out\n");
			goto out;
		}

		flup = IGC_READ_REG(hw, IGC_EECD) | IGC_EECD_FLUPD_I225;
		IGC_WRITE_REG(hw, IGC_EECD, flup);

		ret_val = igc_pool_flash_update_done_i225(hw);
		if (ret_val == IGC_SUCCESS)
			DEBUGOUT("Flash update complete\n");
		else
			DEBUGOUT("Flash update time out\n");
	} else if (!block_sw_protect) {
		/* FW is not active and security protection is disabled.
		 * therefore, SW is in charge of shadow RAM dump.
		 * Check which sector is valid. if sector 0 is valid,
		 * base address remains 0x0. otherwise, sector 1 is
		 * valid and it's base address is 0x1000
		 */
		if (IGC_READ_REG(hw, IGC_EECD) & IGC_EECD_SEC1VAL_I225)
			base_address = 0x1000;

		/* Valid sector erase */
		ret_val = igc_write_erase_flash_command_i225(hw,
						  IGC_I225_ERASE_CMD_OPCODE,
						  base_address);
		if (!ret_val) {
			DEBUGOUT("Sector erase failed\n");
			goto out;
		}

		lwrrent_offset = base_address;

		/* Write */
		for (i = 0; i < IGC_I225_SHADOW_RAM_SIZE / 2; i++) {
			/* Set burst write length */
			ret_val = igc_set_flsw_flash_burst_counter_i225(hw,
									  0x2);
			if (ret_val != IGC_SUCCESS)
				break;

			/* Set address and opcode */
			ret_val = igc_write_erase_flash_command_i225(hw,
						IGC_I225_WRITE_CMD_OPCODE,
						2 * lwrrent_offset);
			if (ret_val != IGC_SUCCESS)
				break;

			ret_val = igc_read_lwm_eerd(hw, lwrrent_offset,
						      1, &lwrrent_offset_data);
			if (ret_val) {
				DEBUGOUT("Failed to read from EEPROM\n");
				goto out;
			}

			/* Write LwrrentOffseData to FLSWDATA register */
			IGC_WRITE_REG(hw, IGC_I225_FLSWDATA,
					lwrrent_offset_data);
			lwrrent_offset++;

			/* Wait till operation has finished */
			ret_val = igc_poll_eerd_eewr_done(hw,
						IGC_LWM_POLL_READ);
			if (ret_val)
				break;

			usec_delay(1000);
		}
	}
out:
	return ret_val;
}

/* igc_pool_flash_update_done_i225 - Pool FLUDONE status.
 * @hw: pointer to the HW structure
 */
s32 igc_pool_flash_update_done_i225(struct igc_hw *hw)
{
	s32 ret_val = -IGC_ERR_LWM;
	u32 i, reg;

	DEBUGFUNC("igc_pool_flash_update_done_i225");

	for (i = 0; i < IGC_FLUDONE_ATTEMPTS; i++) {
		reg = IGC_READ_REG(hw, IGC_EECD);
		if (reg & IGC_EECD_FLUDONE_I225) {
			ret_val = IGC_SUCCESS;
			break;
		}
		usec_delay(5);
	}

	return ret_val;
}

/* igc_set_ltr_i225 - Set Latency Tolerance Reporting thresholds.
 * @hw: pointer to the HW structure
 * @link: bool indicating link status
 *
 * Set the LTR thresholds based on the link speed (Mbps), EEE, and DMAC
 * settings, otherwise specify that there is no LTR requirement.
 */
static s32 igc_set_ltr_i225(struct igc_hw *hw, bool link)
{
	u16 speed, duplex;
	u32 tw_system, ltrc, ltrv, ltr_min, ltr_max, scale_min, scale_max;
	s32 size;

	DEBUGFUNC("igc_set_ltr_i225");

	/* If we do not have link, LTR thresholds are zero. */
	if (link) {
		hw->mac.ops.get_link_up_info(hw, &speed, &duplex);

		/* Check if using copper interface with EEE enabled or if the
		 * link speed is 10 Mbps.
		 */
		if (hw->phy.media_type == igc_media_type_copper &&
				!hw->dev_spec._i225.eee_disable &&
				speed != SPEED_10) {
			/* EEE enabled, so send LTRMAX threshold. */
			ltrc = IGC_READ_REG(hw, IGC_LTRC) |
				IGC_LTRC_EEEMS_EN;
			IGC_WRITE_REG(hw, IGC_LTRC, ltrc);

			/* Callwlate tw_system (nsec). */
			if (speed == SPEED_100)
				tw_system = ((IGC_READ_REG(hw, IGC_EEE_SU) &
					IGC_TW_SYSTEM_100_MASK) >>
					IGC_TW_SYSTEM_100_SHIFT) * 500;
			else
				tw_system = (IGC_READ_REG(hw, IGC_EEE_SU) &
					IGC_TW_SYSTEM_1000_MASK) * 500;
		} else {
			tw_system = 0;
		}

		/* Get the Rx packet buffer size. */
		size = IGC_READ_REG(hw, IGC_RXPBS) &
			IGC_RXPBS_SIZE_I225_MASK;

		/* Callwlations vary based on DMAC settings. */
		if (IGC_READ_REG(hw, IGC_DMACR) & IGC_DMACR_DMAC_EN) {
			size -= (IGC_READ_REG(hw, IGC_DMACR) &
				 IGC_DMACR_DMACTHR_MASK) >>
				 IGC_DMACR_DMACTHR_SHIFT;
			/* Colwert size to bits. */
			size *= 1024 * 8;
		} else {
			/* Colwert size to bytes, subtract the MTU, and then
			 * colwert the size to bits.
			 */
			size *= 1024;
			size -= hw->dev_spec._i225.mtu;
			size *= 8;
		}

		if (size < 0) {
			DEBUGOUT1("Invalid effective Rx buffer size %d\n",
				  size);
			return -IGC_ERR_CONFIG;
		}

		/* Callwlate the thresholds. Since speed is in Mbps, simplify
		 * the callwlation by multiplying size/speed by 1000 for result
		 * to be in nsec before dividing by the scale in nsec. Set the
		 * scale such that the LTR threshold fits in the register.
		 */
		ltr_min = (1000 * size) / speed;
		ltr_max = ltr_min + tw_system;
		scale_min = (ltr_min / 1024) < 1024 ? IGC_LTRMILW_SCALE_1024 :
			    IGC_LTRMILW_SCALE_32768;
		scale_max = (ltr_max / 1024) < 1024 ? IGC_LTRMAXV_SCALE_1024 :
			    IGC_LTRMAXV_SCALE_32768;
		ltr_min /= scale_min == IGC_LTRMILW_SCALE_1024 ? 1024 : 32768;
		ltr_max /= scale_max == IGC_LTRMAXV_SCALE_1024 ? 1024 : 32768;

		/* Only write the LTR thresholds if they differ from before. */
		ltrv = IGC_READ_REG(hw, IGC_LTRMILW);
		if (ltr_min != (ltrv & IGC_LTRMILW_LTRV_MASK)) {
			ltrv = IGC_LTRMILW_LSNP_REQ | ltr_min |
			      (scale_min << IGC_LTRMILW_SCALE_SHIFT);
			IGC_WRITE_REG(hw, IGC_LTRMILW, ltrv);
		}

		ltrv = IGC_READ_REG(hw, IGC_LTRMAXV);
		if (ltr_max != (ltrv & IGC_LTRMAXV_LTRV_MASK)) {
			ltrv = IGC_LTRMAXV_LSNP_REQ | ltr_max |
			      (scale_min << IGC_LTRMAXV_SCALE_SHIFT);
			IGC_WRITE_REG(hw, IGC_LTRMAXV, ltrv);
		}
	}

	return IGC_SUCCESS;
}

/* igc_check_for_link_i225 - Check for link
 * @hw: pointer to the HW structure
 *
 * Checks to see of the link status of the hardware has changed.  If a
 * change in link status has been detected, then we read the PHY registers
 * to get the current speed/duplex if link exists.
 */
s32 igc_check_for_link_i225(struct igc_hw *hw)
{
	struct igc_mac_info *mac = &hw->mac;
	s32 ret_val;
	bool link = false;

	DEBUGFUNC("igc_check_for_link_i225");

	/* We only want to go out to the PHY registers to see if
	 * Auto-Neg has completed and/or if our link status has
	 * changed.  The get_link_status flag is set upon receiving
	 * a Link Status Change or Rx Sequence Error interrupt.
	 */
	if (!mac->get_link_status) {
		ret_val = IGC_SUCCESS;
		goto out;
	}

	/* First we want to see if the MII Status Register reports
	 * link.  If so, then we want to get the current speed/duplex
	 * of the PHY.
	 */
	ret_val = igc_phy_has_link_generic(hw, 1, 0, &link);
	if (ret_val)
		goto out;

	if (!link)
		goto out; /* No link detected */

	mac->get_link_status = false;

	/* Check if there was DownShift, must be checked
	 * immediately after link-up
	 */
	igc_check_downshift_generic(hw);

	/* If we are forcing speed/duplex, then we simply return since
	 * we have already determined whether we have link or not.
	 */
	if (!mac->autoneg)
		goto out;

	/* Auto-Neg is enabled.  Auto Speed Detection takes care
	 * of MAC speed/duplex configuration.  So we only need to
	 * configure Collision Distance in the MAC.
	 */
	mac->ops.config_collision_dist(hw);

	/* Configure Flow Control now that Auto-Neg has completed.
	 * First, we need to restore the desired flow control
	 * settings because we may have had to re-autoneg with a
	 * different link partner.
	 */
	ret_val = igc_config_fc_after_link_up_generic(hw);
	if (ret_val)
		DEBUGOUT("Error configuring flow control\n");
out:
	/* Now that we are aware of our link settings, we can set the LTR
	 * thresholds.
	 */
	ret_val = igc_set_ltr_i225(hw, link);

	return ret_val;
}

/* igc_init_function_pointers_i225 - Init func ptrs.
 * @hw: pointer to the HW structure
 *
 * Called to initialize all function pointers and parameters.
 */
void igc_init_function_pointers_i225(struct igc_hw *hw)
{
	igc_init_mac_ops_generic(hw);
	igc_init_phy_ops_generic(hw);
	igc_init_lwm_ops_generic(hw);
	hw->mac.ops.init_params = igc_init_mac_params_i225;
	hw->lwm.ops.init_params = igc_init_lwm_params_i225;
	hw->phy.ops.init_params = igc_init_phy_params_i225;
}

/* igc_valid_led_default_i225 - Verify a valid default LED config
 * @hw: pointer to the HW structure
 * @data: pointer to the LWM (EEPROM)
 *
 * Read the EEPROM for the current default LED configuration.  If the
 * LED configuration is not valid, set to a valid LED configuration.
 */
static s32 igc_valid_led_default_i225(struct igc_hw *hw, u16 *data)
{
	s32 ret_val;

	DEBUGFUNC("igc_valid_led_default_i225");

	ret_val = hw->lwm.ops.read(hw, LWM_ID_LED_SETTINGS, 1, data);
	if (ret_val) {
		DEBUGOUT("LWM Read Error\n");
		goto out;
	}

	if (*data == ID_LED_RESERVED_0000 || *data == ID_LED_RESERVED_FFFF) {
		switch (hw->phy.media_type) {
		case igc_media_type_internal_serdes:
			*data = ID_LED_DEFAULT_I225_SERDES;
			break;
		case igc_media_type_copper:
		default:
			*data = ID_LED_DEFAULT_I225;
			break;
		}
	}
out:
	return ret_val;
}

/* igc_get_cfg_done_i225 - Read config done bit
 * @hw: pointer to the HW structure
 *
 * Read the management control register for the config done bit for
 * completion status.  NOTE: silicon which is EEPROM-less will fail trying
 * to read the config done bit, so an error is *ONLY* logged and returns
 * IGC_SUCCESS.  If we were to return with error, EEPROM-less silicon
 * would not be able to be reset or change link.
 */
static s32 igc_get_cfg_done_i225(struct igc_hw *hw)
{
	s32 timeout = PHY_CFG_TIMEOUT;
	u32 mask = IGC_LWM_CFG_DONE_PORT_0;

	DEBUGFUNC("igc_get_cfg_done_i225");

	while (timeout) {
		if (IGC_READ_REG(hw, IGC_EEMNGCTL_I225) & mask)
			break;
		msec_delay(1);
		timeout--;
	}
	if (!timeout)
		DEBUGOUT("MNG configuration cycle has not completed.\n");

	return IGC_SUCCESS;
}

/* igc_init_hw_i225 - Init hw for I225
 * @hw: pointer to the HW structure
 *
 * Called to initialize hw for i225 hw family.
 */
s32 igc_init_hw_i225(struct igc_hw *hw)
{
	s32 ret_val;

	DEBUGFUNC("igc_init_hw_i225");

	hw->phy.ops.get_cfg_done = igc_get_cfg_done_i225;
	ret_val = igc_init_hw_base(hw);
	return ret_val;
}

/*
 * igc_set_d0_lplu_state_i225 - Set Low-Power-Link-Up (LPLU) D0 state
 * @hw: pointer to the HW structure
 * @active: true to enable LPLU, false to disable
 *
 * Note: since I225 does not actually support LPLU, this function
 * simply enables/disables 1G and 2.5G speeds in D0.
 */
s32 igc_set_d0_lplu_state_i225(struct igc_hw *hw, bool active)
{
	u32 data;

	DEBUGFUNC("igc_set_d0_lplu_state_i225");

	data = IGC_READ_REG(hw, IGC_I225_PHPM);

	if (active) {
		data |= IGC_I225_PHPM_DIS_1000;
		data |= IGC_I225_PHPM_DIS_2500;
	} else {
		data &= ~IGC_I225_PHPM_DIS_1000;
		data &= ~IGC_I225_PHPM_DIS_2500;
	}

	IGC_WRITE_REG(hw, IGC_I225_PHPM, data);
	return IGC_SUCCESS;
}

/*
 * igc_set_d3_lplu_state_i225 - Set Low-Power-Link-Up (LPLU) D3 state
 * @hw: pointer to the HW structure
 * @active: true to enable LPLU, false to disable
 *
 * Note: since I225 does not actually support LPLU, this function
 * simply enables/disables 100M, 1G and 2.5G speeds in D3.
 */
s32 igc_set_d3_lplu_state_i225(struct igc_hw *hw, bool active)
{
	u32 data;

	DEBUGFUNC("igc_set_d3_lplu_state_i225");

	data = IGC_READ_REG(hw, IGC_I225_PHPM);

	if (active) {
		data |= IGC_I225_PHPM_DIS_100_D3;
		data |= IGC_I225_PHPM_DIS_1000_D3;
		data |= IGC_I225_PHPM_DIS_2500_D3;
	} else {
		data &= ~IGC_I225_PHPM_DIS_100_D3;
		data &= ~IGC_I225_PHPM_DIS_1000_D3;
		data &= ~IGC_I225_PHPM_DIS_2500_D3;
	}

	IGC_WRITE_REG(hw, IGC_I225_PHPM, data);
	return IGC_SUCCESS;
}

/**
 *  igc_set_eee_i225 - Enable/disable EEE support
 *  @hw: pointer to the HW structure
 *  @adv2p5G: boolean flag enabling 2.5G EEE advertisement
 *  @adv1G: boolean flag enabling 1G EEE advertisement
 *  @adv100M: boolean flag enabling 100M EEE advertisement
 *
 *  Enable/disable EEE based on setting in dev_spec structure.
 *
 **/
s32 igc_set_eee_i225(struct igc_hw *hw, bool adv2p5G, bool adv1G,
		       bool adv100M)
{
	u32 ipcnfg, eeer;

	DEBUGFUNC("igc_set_eee_i225");

	if (hw->mac.type != igc_i225 ||
	    hw->phy.media_type != igc_media_type_copper)
		goto out;
	ipcnfg = IGC_READ_REG(hw, IGC_IPCNFG);
	eeer = IGC_READ_REG(hw, IGC_EEER);

	/* enable or disable per user setting */
	if (!(hw->dev_spec._i225.eee_disable)) {
		u32 eee_su = IGC_READ_REG(hw, IGC_EEE_SU);

		if (adv100M)
			ipcnfg |= IGC_IPCNFG_EEE_100M_AN;
		else
			ipcnfg &= ~IGC_IPCNFG_EEE_100M_AN;

		if (adv1G)
			ipcnfg |= IGC_IPCNFG_EEE_1G_AN;
		else
			ipcnfg &= ~IGC_IPCNFG_EEE_1G_AN;

		if (adv2p5G)
			ipcnfg |= IGC_IPCNFG_EEE_2_5G_AN;
		else
			ipcnfg &= ~IGC_IPCNFG_EEE_2_5G_AN;

		eeer |= (IGC_EEER_TX_LPI_EN | IGC_EEER_RX_LPI_EN |
			IGC_EEER_LPI_FC);

		/* This bit should not be set in normal operation. */
		if (eee_su & IGC_EEE_SU_LPI_CLK_STP)
			DEBUGOUT("LPI Clock Stop Bit should not be set!\n");
	} else {
		ipcnfg &= ~(IGC_IPCNFG_EEE_2_5G_AN | IGC_IPCNFG_EEE_1G_AN |
			IGC_IPCNFG_EEE_100M_AN);
		eeer &= ~(IGC_EEER_TX_LPI_EN | IGC_EEER_RX_LPI_EN |
			IGC_EEER_LPI_FC);
	}
	IGC_WRITE_REG(hw, IGC_IPCNFG, ipcnfg);
	IGC_WRITE_REG(hw, IGC_EEER, eeer);
	IGC_READ_REG(hw, IGC_IPCNFG);
	IGC_READ_REG(hw, IGC_EEER);
out:

	return IGC_SUCCESS;
}
