/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2001-2020 Intel Corporation
 */

#include "igc_api.h"

static void igc_reload_lwm_generic(struct igc_hw *hw);

/**
 *  igc_init_lwm_ops_generic - Initialize LWM function pointers
 *  @hw: pointer to the HW structure
 *
 *  Setups up the function pointers to no-op functions
 **/
void igc_init_lwm_ops_generic(struct igc_hw *hw)
{
	struct igc_lwm_info *lwm = &hw->lwm;
	DEBUGFUNC("igc_init_lwm_ops_generic");

	/* Initialize function pointers */
	lwm->ops.init_params = igc_null_ops_generic;
	lwm->ops.acquire = igc_null_ops_generic;
	lwm->ops.read = igc_null_read_lwm;
	lwm->ops.release = igc_null_lwm_generic;
	lwm->ops.reload = igc_reload_lwm_generic;
	lwm->ops.update = igc_null_ops_generic;
	lwm->ops.valid_led_default = igc_null_led_default;
	lwm->ops.validate = igc_null_ops_generic;
	lwm->ops.write = igc_null_write_lwm;
}

/**
 *  igc_null_lwm_read - No-op function, return 0
 *  @hw: pointer to the HW structure
 *  @a: dummy variable
 *  @b: dummy variable
 *  @c: dummy variable
 **/
s32 igc_null_read_lwm(struct igc_hw IGC_UNUSEDARG * hw,
			u16 IGC_UNUSEDARG a, u16 IGC_UNUSEDARG b,
			u16 IGC_UNUSEDARG * c)
{
	DEBUGFUNC("igc_null_read_lwm");
	UNREFERENCED_4PARAMETER(hw, a, b, c);
	return IGC_SUCCESS;
}

/**
 *  igc_null_lwm_generic - No-op function, return void
 *  @hw: pointer to the HW structure
 **/
void igc_null_lwm_generic(struct igc_hw IGC_UNUSEDARG * hw)
{
	DEBUGFUNC("igc_null_lwm_generic");
	UNREFERENCED_1PARAMETER(hw);
}

/**
 *  igc_null_led_default - No-op function, return 0
 *  @hw: pointer to the HW structure
 *  @data: dummy variable
 **/
s32 igc_null_led_default(struct igc_hw IGC_UNUSEDARG * hw,
			   u16 IGC_UNUSEDARG * data)
{
	DEBUGFUNC("igc_null_led_default");
	UNREFERENCED_2PARAMETER(hw, data);
	return IGC_SUCCESS;
}

/**
 *  igc_null_write_lwm - No-op function, return 0
 *  @hw: pointer to the HW structure
 *  @a: dummy variable
 *  @b: dummy variable
 *  @c: dummy variable
 **/
s32 igc_null_write_lwm(struct igc_hw IGC_UNUSEDARG * hw,
			 u16 IGC_UNUSEDARG a, u16 IGC_UNUSEDARG b,
			 u16 IGC_UNUSEDARG * c)
{
	DEBUGFUNC("igc_null_write_lwm");
	UNREFERENCED_4PARAMETER(hw, a, b, c);
	return IGC_SUCCESS;
}

/**
 *  igc_raise_eec_clk - Raise EEPROM clock
 *  @hw: pointer to the HW structure
 *  @eecd: pointer to the EEPROM
 *
 *  Enable/Raise the EEPROM clock bit.
 **/
static void igc_raise_eec_clk(struct igc_hw *hw, u32 *eecd)
{
	*eecd = *eecd | IGC_EECD_SK;
	IGC_WRITE_REG(hw, IGC_EECD, *eecd);
	IGC_WRITE_FLUSH(hw);
	usec_delay(hw->lwm.delay_usec);
}

/**
 *  igc_lower_eec_clk - Lower EEPROM clock
 *  @hw: pointer to the HW structure
 *  @eecd: pointer to the EEPROM
 *
 *  Clear/Lower the EEPROM clock bit.
 **/
static void igc_lower_eec_clk(struct igc_hw *hw, u32 *eecd)
{
	*eecd = *eecd & ~IGC_EECD_SK;
	IGC_WRITE_REG(hw, IGC_EECD, *eecd);
	IGC_WRITE_FLUSH(hw);
	usec_delay(hw->lwm.delay_usec);
}

/**
 *  igc_shift_out_eec_bits - Shift data bits our to the EEPROM
 *  @hw: pointer to the HW structure
 *  @data: data to send to the EEPROM
 *  @count: number of bits to shift out
 *
 *  We need to shift 'count' bits out to the EEPROM.  So, the value in the
 *  "data" parameter will be shifted out to the EEPROM one bit at a time.
 *  In order to do this, "data" must be broken down into bits.
 **/
static void igc_shift_out_eec_bits(struct igc_hw *hw, u16 data, u16 count)
{
	struct igc_lwm_info *lwm = &hw->lwm;
	u32 eecd = IGC_READ_REG(hw, IGC_EECD);
	u32 mask;

	DEBUGFUNC("igc_shift_out_eec_bits");

	mask = 0x01 << (count - 1);
	if (lwm->type == igc_lwm_eeprom_microwire)
		eecd &= ~IGC_EECD_DO;
	else if (lwm->type == igc_lwm_eeprom_spi)
		eecd |= IGC_EECD_DO;

	do {
		eecd &= ~IGC_EECD_DI;

		if (data & mask)
			eecd |= IGC_EECD_DI;

		IGC_WRITE_REG(hw, IGC_EECD, eecd);
		IGC_WRITE_FLUSH(hw);

		usec_delay(lwm->delay_usec);

		igc_raise_eec_clk(hw, &eecd);
		igc_lower_eec_clk(hw, &eecd);

		mask >>= 1;
	} while (mask);

	eecd &= ~IGC_EECD_DI;
	IGC_WRITE_REG(hw, IGC_EECD, eecd);
}

/**
 *  igc_shift_in_eec_bits - Shift data bits in from the EEPROM
 *  @hw: pointer to the HW structure
 *  @count: number of bits to shift in
 *
 *  In order to read a register from the EEPROM, we need to shift 'count' bits
 *  in from the EEPROM.  Bits are "shifted in" by raising the clock input to
 *  the EEPROM (setting the SK bit), and then reading the value of the data out
 *  "DO" bit.  During this "shifting in" process the data in "DI" bit should
 *  always be clear.
 **/
static u16 igc_shift_in_eec_bits(struct igc_hw *hw, u16 count)
{
	u32 eecd;
	u32 i;
	u16 data;

	DEBUGFUNC("igc_shift_in_eec_bits");

	eecd = IGC_READ_REG(hw, IGC_EECD);

	eecd &= ~(IGC_EECD_DO | IGC_EECD_DI);
	data = 0;

	for (i = 0; i < count; i++) {
		data <<= 1;
		igc_raise_eec_clk(hw, &eecd);

		eecd = IGC_READ_REG(hw, IGC_EECD);

		eecd &= ~IGC_EECD_DI;
		if (eecd & IGC_EECD_DO)
			data |= 1;

		igc_lower_eec_clk(hw, &eecd);
	}

	return data;
}

/**
 *  igc_poll_eerd_eewr_done - Poll for EEPROM read/write completion
 *  @hw: pointer to the HW structure
 *  @ee_reg: EEPROM flag for polling
 *
 *  Polls the EEPROM status bit for either read or write completion based
 *  upon the value of 'ee_reg'.
 **/
s32 igc_poll_eerd_eewr_done(struct igc_hw *hw, int ee_reg)
{
	u32 attempts = 100000;
	u32 i, reg = 0;

	DEBUGFUNC("igc_poll_eerd_eewr_done");

	for (i = 0; i < attempts; i++) {
		if (ee_reg == IGC_LWM_POLL_READ)
			reg = IGC_READ_REG(hw, IGC_EERD);
		else
			reg = IGC_READ_REG(hw, IGC_EEWR);

		if (reg & IGC_LWM_RW_REG_DONE)
			return IGC_SUCCESS;

		usec_delay(5);
	}

	return -IGC_ERR_LWM;
}

/**
 *  igc_acquire_lwm_generic - Generic request for access to EEPROM
 *  @hw: pointer to the HW structure
 *
 *  Set the EEPROM access request bit and wait for EEPROM access grant bit.
 *  Return successful if access grant bit set, else clear the request for
 *  EEPROM access and return -IGC_ERR_LWM (-1).
 **/
s32 igc_acquire_lwm_generic(struct igc_hw *hw)
{
	u32 eecd = IGC_READ_REG(hw, IGC_EECD);
	s32 timeout = IGC_LWM_GRANT_ATTEMPTS;

	DEBUGFUNC("igc_acquire_lwm_generic");

	IGC_WRITE_REG(hw, IGC_EECD, eecd | IGC_EECD_REQ);
	eecd = IGC_READ_REG(hw, IGC_EECD);

	while (timeout) {
		if (eecd & IGC_EECD_GNT)
			break;
		usec_delay(5);
		eecd = IGC_READ_REG(hw, IGC_EECD);
		timeout--;
	}

	if (!timeout) {
		eecd &= ~IGC_EECD_REQ;
		IGC_WRITE_REG(hw, IGC_EECD, eecd);
		DEBUGOUT("Could not acquire LWM grant\n");
		return -IGC_ERR_LWM;
	}

	return IGC_SUCCESS;
}

/**
 *  igc_standby_lwm - Return EEPROM to standby state
 *  @hw: pointer to the HW structure
 *
 *  Return the EEPROM to a standby state.
 **/
static void igc_standby_lwm(struct igc_hw *hw)
{
	struct igc_lwm_info *lwm = &hw->lwm;
	u32 eecd = IGC_READ_REG(hw, IGC_EECD);

	DEBUGFUNC("igc_standby_lwm");

	if (lwm->type == igc_lwm_eeprom_microwire) {
		eecd &= ~(IGC_EECD_CS | IGC_EECD_SK);
		IGC_WRITE_REG(hw, IGC_EECD, eecd);
		IGC_WRITE_FLUSH(hw);
		usec_delay(lwm->delay_usec);

		igc_raise_eec_clk(hw, &eecd);

		/* Select EEPROM */
		eecd |= IGC_EECD_CS;
		IGC_WRITE_REG(hw, IGC_EECD, eecd);
		IGC_WRITE_FLUSH(hw);
		usec_delay(lwm->delay_usec);

		igc_lower_eec_clk(hw, &eecd);
	} else if (lwm->type == igc_lwm_eeprom_spi) {
		/* Toggle CS to flush commands */
		eecd |= IGC_EECD_CS;
		IGC_WRITE_REG(hw, IGC_EECD, eecd);
		IGC_WRITE_FLUSH(hw);
		usec_delay(lwm->delay_usec);
		eecd &= ~IGC_EECD_CS;
		IGC_WRITE_REG(hw, IGC_EECD, eecd);
		IGC_WRITE_FLUSH(hw);
		usec_delay(lwm->delay_usec);
	}
}

/**
 *  igc_stop_lwm - Terminate EEPROM command
 *  @hw: pointer to the HW structure
 *
 *  Terminates the current command by ilwerting the EEPROM's chip select pin.
 **/
void igc_stop_lwm(struct igc_hw *hw)
{
	u32 eecd;

	DEBUGFUNC("igc_stop_lwm");

	eecd = IGC_READ_REG(hw, IGC_EECD);
	if (hw->lwm.type == igc_lwm_eeprom_spi) {
		/* Pull CS high */
		eecd |= IGC_EECD_CS;
		igc_lower_eec_clk(hw, &eecd);
	} else if (hw->lwm.type == igc_lwm_eeprom_microwire) {
		/* CS on Microwire is active-high */
		eecd &= ~(IGC_EECD_CS | IGC_EECD_DI);
		IGC_WRITE_REG(hw, IGC_EECD, eecd);
		igc_raise_eec_clk(hw, &eecd);
		igc_lower_eec_clk(hw, &eecd);
	}
}

/**
 *  igc_release_lwm_generic - Release exclusive access to EEPROM
 *  @hw: pointer to the HW structure
 *
 *  Stop any current commands to the EEPROM and clear the EEPROM request bit.
 **/
void igc_release_lwm_generic(struct igc_hw *hw)
{
	u32 eecd;

	DEBUGFUNC("igc_release_lwm_generic");

	igc_stop_lwm(hw);

	eecd = IGC_READ_REG(hw, IGC_EECD);
	eecd &= ~IGC_EECD_REQ;
	IGC_WRITE_REG(hw, IGC_EECD, eecd);
}

/**
 *  igc_ready_lwm_eeprom - Prepares EEPROM for read/write
 *  @hw: pointer to the HW structure
 *
 *  Setups the EEPROM for reading and writing.
 **/
static s32 igc_ready_lwm_eeprom(struct igc_hw *hw)
{
	struct igc_lwm_info *lwm = &hw->lwm;
	u32 eecd = IGC_READ_REG(hw, IGC_EECD);
	u8 spi_stat_reg;

	DEBUGFUNC("igc_ready_lwm_eeprom");

	if (lwm->type == igc_lwm_eeprom_microwire) {
		/* Clear SK and DI */
		eecd &= ~(IGC_EECD_DI | IGC_EECD_SK);
		IGC_WRITE_REG(hw, IGC_EECD, eecd);
		/* Set CS */
		eecd |= IGC_EECD_CS;
		IGC_WRITE_REG(hw, IGC_EECD, eecd);
	} else if (lwm->type == igc_lwm_eeprom_spi) {
		u16 timeout = LWM_MAX_RETRY_SPI;

		/* Clear SK and CS */
		eecd &= ~(IGC_EECD_CS | IGC_EECD_SK);
		IGC_WRITE_REG(hw, IGC_EECD, eecd);
		IGC_WRITE_FLUSH(hw);
		usec_delay(1);

		/* Read "Status Register" repeatedly until the LSB is cleared.
		 * The EEPROM will signal that the command has been completed
		 * by clearing bit 0 of the internal status register.  If it's
		 * not cleared within 'timeout', then error out.
		 */
		while (timeout) {
			igc_shift_out_eec_bits(hw, LWM_RDSR_OPCODE_SPI,
						 hw->lwm.opcode_bits);
			spi_stat_reg = (u8)igc_shift_in_eec_bits(hw, 8);
			if (!(spi_stat_reg & LWM_STATUS_RDY_SPI))
				break;

			usec_delay(5);
			igc_standby_lwm(hw);
			timeout--;
		}

		if (!timeout) {
			DEBUGOUT("SPI LWM Status error\n");
			return -IGC_ERR_LWM;
		}
	}

	return IGC_SUCCESS;
}

/**
 *  igc_read_lwm_spi - Read EEPROM's using SPI
 *  @hw: pointer to the HW structure
 *  @offset: offset of word in the EEPROM to read
 *  @words: number of words to read
 *  @data: word read from the EEPROM
 *
 *  Reads a 16 bit word from the EEPROM.
 **/
s32 igc_read_lwm_spi(struct igc_hw *hw, u16 offset, u16 words, u16 *data)
{
	struct igc_lwm_info *lwm = &hw->lwm;
	u32 i = 0;
	s32 ret_val;
	u16 word_in;
	u8 read_opcode = LWM_READ_OPCODE_SPI;

	DEBUGFUNC("igc_read_lwm_spi");

	/* A check for invalid values:  offset too large, too many words,
	 * and not enough words.
	 */
	if (offset >= lwm->word_size || words > (lwm->word_size - offset) ||
			words == 0) {
		DEBUGOUT("lwm parameter(s) out of bounds\n");
		return -IGC_ERR_LWM;
	}

	ret_val = lwm->ops.acquire(hw);
	if (ret_val)
		return ret_val;

	ret_val = igc_ready_lwm_eeprom(hw);
	if (ret_val)
		goto release;

	igc_standby_lwm(hw);

	if (lwm->address_bits == 8 && offset >= 128)
		read_opcode |= LWM_A8_OPCODE_SPI;

	/* Send the READ command (opcode + addr) */
	igc_shift_out_eec_bits(hw, read_opcode, lwm->opcode_bits);
	igc_shift_out_eec_bits(hw, (u16)(offset * 2), lwm->address_bits);

	/* Read the data.  SPI LWMs increment the address with each byte
	 * read and will roll over if reading beyond the end.  This allows
	 * us to read the whole LWM from any offset
	 */
	for (i = 0; i < words; i++) {
		word_in = igc_shift_in_eec_bits(hw, 16);
		data[i] = (word_in >> 8) | (word_in << 8);
	}

release:
	lwm->ops.release(hw);

	return ret_val;
}

/**
 *  igc_read_lwm_microwire - Reads EEPROM's using microwire
 *  @hw: pointer to the HW structure
 *  @offset: offset of word in the EEPROM to read
 *  @words: number of words to read
 *  @data: word read from the EEPROM
 *
 *  Reads a 16 bit word from the EEPROM.
 **/
s32 igc_read_lwm_microwire(struct igc_hw *hw, u16 offset, u16 words,
			     u16 *data)
{
	struct igc_lwm_info *lwm = &hw->lwm;
	u32 i = 0;
	s32 ret_val;
	u8 read_opcode = LWM_READ_OPCODE_MICROWIRE;

	DEBUGFUNC("igc_read_lwm_microwire");

	/* A check for invalid values:  offset too large, too many words,
	 * and not enough words.
	 */
	if (offset >= lwm->word_size || words > (lwm->word_size - offset) ||
			words == 0) {
		DEBUGOUT("lwm parameter(s) out of bounds\n");
		return -IGC_ERR_LWM;
	}

	ret_val = lwm->ops.acquire(hw);
	if (ret_val)
		return ret_val;

	ret_val = igc_ready_lwm_eeprom(hw);
	if (ret_val)
		goto release;

	for (i = 0; i < words; i++) {
		/* Send the READ command (opcode + addr) */
		igc_shift_out_eec_bits(hw, read_opcode, lwm->opcode_bits);
		igc_shift_out_eec_bits(hw, (u16)(offset + i),
					lwm->address_bits);

		/* Read the data.  For microwire, each word requires the
		 * overhead of setup and tear-down.
		 */
		data[i] = igc_shift_in_eec_bits(hw, 16);
		igc_standby_lwm(hw);
	}

release:
	lwm->ops.release(hw);

	return ret_val;
}

/**
 *  igc_read_lwm_eerd - Reads EEPROM using EERD register
 *  @hw: pointer to the HW structure
 *  @offset: offset of word in the EEPROM to read
 *  @words: number of words to read
 *  @data: word read from the EEPROM
 *
 *  Reads a 16 bit word from the EEPROM using the EERD register.
 **/
s32 igc_read_lwm_eerd(struct igc_hw *hw, u16 offset, u16 words, u16 *data)
{
	struct igc_lwm_info *lwm = &hw->lwm;
	u32 i, eerd = 0;
	s32 ret_val = IGC_SUCCESS;

	DEBUGFUNC("igc_read_lwm_eerd");

	/* A check for invalid values:  offset too large, too many words,
	 * too many words for the offset, and not enough words.
	 */
	if (offset >= lwm->word_size || words > (lwm->word_size - offset) ||
			words == 0) {
		DEBUGOUT("lwm parameter(s) out of bounds\n");
		return -IGC_ERR_LWM;
	}

	for (i = 0; i < words; i++) {
		eerd = ((offset + i) << IGC_LWM_RW_ADDR_SHIFT) +
		       IGC_LWM_RW_REG_START;

		IGC_WRITE_REG(hw, IGC_EERD, eerd);
		ret_val = igc_poll_eerd_eewr_done(hw, IGC_LWM_POLL_READ);
		if (ret_val)
			break;

		data[i] = (IGC_READ_REG(hw, IGC_EERD) >>
			   IGC_LWM_RW_REG_DATA);
	}

	if (ret_val)
		DEBUGOUT1("LWM read error: %d\n", ret_val);

	return ret_val;
}

/**
 *  igc_write_lwm_spi - Write to EEPROM using SPI
 *  @hw: pointer to the HW structure
 *  @offset: offset within the EEPROM to be written to
 *  @words: number of words to write
 *  @data: 16 bit word(s) to be written to the EEPROM
 *
 *  Writes data to EEPROM at offset using SPI interface.
 *
 *  If igc_update_lwm_checksum is not called after this function , the
 *  EEPROM will most likely contain an invalid checksum.
 **/
s32 igc_write_lwm_spi(struct igc_hw *hw, u16 offset, u16 words, u16 *data)
{
	struct igc_lwm_info *lwm = &hw->lwm;
	s32 ret_val = -IGC_ERR_LWM;
	u16 widx = 0;

	DEBUGFUNC("igc_write_lwm_spi");

	/* A check for invalid values:  offset too large, too many words,
	 * and not enough words.
	 */
	if (offset >= lwm->word_size || words > (lwm->word_size - offset) ||
			words == 0) {
		DEBUGOUT("lwm parameter(s) out of bounds\n");
		return -IGC_ERR_LWM;
	}

	while (widx < words) {
		u8 write_opcode = LWM_WRITE_OPCODE_SPI;

		ret_val = lwm->ops.acquire(hw);
		if (ret_val)
			return ret_val;

		ret_val = igc_ready_lwm_eeprom(hw);
		if (ret_val) {
			lwm->ops.release(hw);
			return ret_val;
		}

		igc_standby_lwm(hw);

		/* Send the WRITE ENABLE command (8 bit opcode) */
		igc_shift_out_eec_bits(hw, LWM_WREN_OPCODE_SPI,
					 lwm->opcode_bits);

		igc_standby_lwm(hw);

		/* Some SPI eeproms use the 8th address bit embedded in the
		 * opcode
		 */
		if (lwm->address_bits == 8 && offset >= 128)
			write_opcode |= LWM_A8_OPCODE_SPI;

		/* Send the Write command (8-bit opcode + addr) */
		igc_shift_out_eec_bits(hw, write_opcode, lwm->opcode_bits);
		igc_shift_out_eec_bits(hw, (u16)((offset + widx) * 2),
					 lwm->address_bits);

		/* Loop to allow for up to whole page write of eeprom */
		while (widx < words) {
			u16 word_out = data[widx];
			word_out = (word_out >> 8) | (word_out << 8);
			igc_shift_out_eec_bits(hw, word_out, 16);
			widx++;

			if ((((offset + widx) * 2) % lwm->page_size) == 0) {
				igc_standby_lwm(hw);
				break;
			}
		}
		msec_delay(10);
		lwm->ops.release(hw);
	}

	return ret_val;
}

/**
 *  igc_write_lwm_microwire - Writes EEPROM using microwire
 *  @hw: pointer to the HW structure
 *  @offset: offset within the EEPROM to be written to
 *  @words: number of words to write
 *  @data: 16 bit word(s) to be written to the EEPROM
 *
 *  Writes data to EEPROM at offset using microwire interface.
 *
 *  If igc_update_lwm_checksum is not called after this function , the
 *  EEPROM will most likely contain an invalid checksum.
 **/
s32 igc_write_lwm_microwire(struct igc_hw *hw, u16 offset, u16 words,
			      u16 *data)
{
	struct igc_lwm_info *lwm = &hw->lwm;
	s32  ret_val;
	u32 eecd;
	u16 words_written = 0;
	u16 widx = 0;

	DEBUGFUNC("igc_write_lwm_microwire");

	/* A check for invalid values:  offset too large, too many words,
	 * and not enough words.
	 */
	if (offset >= lwm->word_size || words > (lwm->word_size - offset) ||
			words == 0) {
		DEBUGOUT("lwm parameter(s) out of bounds\n");
		return -IGC_ERR_LWM;
	}

	ret_val = lwm->ops.acquire(hw);
	if (ret_val)
		return ret_val;

	ret_val = igc_ready_lwm_eeprom(hw);
	if (ret_val)
		goto release;

	igc_shift_out_eec_bits(hw, LWM_EWEN_OPCODE_MICROWIRE,
				 (u16)(lwm->opcode_bits + 2));

	igc_shift_out_eec_bits(hw, 0, (u16)(lwm->address_bits - 2));

	igc_standby_lwm(hw);

	while (words_written < words) {
		igc_shift_out_eec_bits(hw, LWM_WRITE_OPCODE_MICROWIRE,
					 lwm->opcode_bits);

		igc_shift_out_eec_bits(hw, (u16)(offset + words_written),
					 lwm->address_bits);

		igc_shift_out_eec_bits(hw, data[words_written], 16);

		igc_standby_lwm(hw);

		for (widx = 0; widx < 200; widx++) {
			eecd = IGC_READ_REG(hw, IGC_EECD);
			if (eecd & IGC_EECD_DO)
				break;
			usec_delay(50);
		}

		if (widx == 200) {
			DEBUGOUT("LWM Write did not complete\n");
			ret_val = -IGC_ERR_LWM;
			goto release;
		}

		igc_standby_lwm(hw);

		words_written++;
	}

	igc_shift_out_eec_bits(hw, LWM_EWDS_OPCODE_MICROWIRE,
				 (u16)(lwm->opcode_bits + 2));

	igc_shift_out_eec_bits(hw, 0, (u16)(lwm->address_bits - 2));

release:
	lwm->ops.release(hw);

	return ret_val;
}

/**
 *  igc_read_pba_string_generic - Read device part number
 *  @hw: pointer to the HW structure
 *  @pba_num: pointer to device part number
 *  @pba_num_size: size of part number buffer
 *
 *  Reads the product board assembly (PBA) number from the EEPROM and stores
 *  the value in pba_num.
 **/
s32 igc_read_pba_string_generic(struct igc_hw *hw, u8 *pba_num,
				  u32 pba_num_size)
{
	s32 ret_val;
	u16 lwm_data;
	u16 pba_ptr;
	u16 offset;
	u16 length;

	DEBUGFUNC("igc_read_pba_string_generic");

	if (pba_num == NULL) {
		DEBUGOUT("PBA string buffer was null\n");
		return -IGC_ERR_ILWALID_ARGUMENT;
	}

	ret_val = hw->lwm.ops.read(hw, LWM_PBA_OFFSET_0, 1, &lwm_data);
	if (ret_val) {
		DEBUGOUT("LWM Read Error\n");
		return ret_val;
	}

	ret_val = hw->lwm.ops.read(hw, LWM_PBA_OFFSET_1, 1, &pba_ptr);
	if (ret_val) {
		DEBUGOUT("LWM Read Error\n");
		return ret_val;
	}

	/* if lwm_data is not ptr guard the PBA must be in legacy format which
	 * means pba_ptr is actually our second data word for the PBA number
	 * and we can decode it into an ascii string
	 */
	if (lwm_data != LWM_PBA_PTR_GUARD) {
		DEBUGOUT("LWM PBA number is not stored as string\n");

		/* make sure callers buffer is big enough to store the PBA */
		if (pba_num_size < IGC_PBANUM_LENGTH) {
			DEBUGOUT("PBA string buffer too small\n");
			return IGC_ERR_NO_SPACE;
		}

		/* extract hex string from data and pba_ptr */
		pba_num[0] = (lwm_data >> 12) & 0xF;
		pba_num[1] = (lwm_data >> 8) & 0xF;
		pba_num[2] = (lwm_data >> 4) & 0xF;
		pba_num[3] = lwm_data & 0xF;
		pba_num[4] = (pba_ptr >> 12) & 0xF;
		pba_num[5] = (pba_ptr >> 8) & 0xF;
		pba_num[6] = '-';
		pba_num[7] = 0;
		pba_num[8] = (pba_ptr >> 4) & 0xF;
		pba_num[9] = pba_ptr & 0xF;

		/* put a null character on the end of our string */
		pba_num[10] = '\0';

		/* switch all the data but the '-' to hex char */
		for (offset = 0; offset < 10; offset++) {
			if (pba_num[offset] < 0xA)
				pba_num[offset] += '0';
			else if (pba_num[offset] < 0x10)
				pba_num[offset] += 'A' - 0xA;
		}

		return IGC_SUCCESS;
	}

	ret_val = hw->lwm.ops.read(hw, pba_ptr, 1, &length);
	if (ret_val) {
		DEBUGOUT("LWM Read Error\n");
		return ret_val;
	}

	if (length == 0xFFFF || length == 0) {
		DEBUGOUT("LWM PBA number section invalid length\n");
		return -IGC_ERR_LWM_PBA_SECTION;
	}
	/* check if pba_num buffer is big enough */
	if (pba_num_size < (((u32)length * 2) - 1)) {
		DEBUGOUT("PBA string buffer too small\n");
		return -IGC_ERR_NO_SPACE;
	}

	/* trim pba length from start of string */
	pba_ptr++;
	length--;

	for (offset = 0; offset < length; offset++) {
		ret_val = hw->lwm.ops.read(hw, pba_ptr + offset, 1, &lwm_data);
		if (ret_val) {
			DEBUGOUT("LWM Read Error\n");
			return ret_val;
		}
		pba_num[offset * 2] = (u8)(lwm_data >> 8);
		pba_num[(offset * 2) + 1] = (u8)(lwm_data & 0xFF);
	}
	pba_num[offset * 2] = '\0';

	return IGC_SUCCESS;
}

/**
 *  igc_read_pba_length_generic - Read device part number length
 *  @hw: pointer to the HW structure
 *  @pba_num_size: size of part number buffer
 *
 *  Reads the product board assembly (PBA) number length from the EEPROM and
 *  stores the value in pba_num_size.
 **/
s32 igc_read_pba_length_generic(struct igc_hw *hw, u32 *pba_num_size)
{
	s32 ret_val;
	u16 lwm_data;
	u16 pba_ptr;
	u16 length;

	DEBUGFUNC("igc_read_pba_length_generic");

	if (pba_num_size == NULL) {
		DEBUGOUT("PBA buffer size was null\n");
		return -IGC_ERR_ILWALID_ARGUMENT;
	}

	ret_val = hw->lwm.ops.read(hw, LWM_PBA_OFFSET_0, 1, &lwm_data);
	if (ret_val) {
		DEBUGOUT("LWM Read Error\n");
		return ret_val;
	}

	ret_val = hw->lwm.ops.read(hw, LWM_PBA_OFFSET_1, 1, &pba_ptr);
	if (ret_val) {
		DEBUGOUT("LWM Read Error\n");
		return ret_val;
	}

	 /* if data is not ptr guard the PBA must be in legacy format */
	if (lwm_data != LWM_PBA_PTR_GUARD) {
		*pba_num_size = IGC_PBANUM_LENGTH;
		return IGC_SUCCESS;
	}

	ret_val = hw->lwm.ops.read(hw, pba_ptr, 1, &length);
	if (ret_val) {
		DEBUGOUT("LWM Read Error\n");
		return ret_val;
	}

	if (length == 0xFFFF || length == 0) {
		DEBUGOUT("LWM PBA number section invalid length\n");
		return -IGC_ERR_LWM_PBA_SECTION;
	}

	/* Colwert from length in u16 values to u8 chars, add 1 for NULL,
	 * and subtract 2 because length field is included in length.
	 */
	*pba_num_size = ((u32)length * 2) - 1;

	return IGC_SUCCESS;
}

/**
 *  igc_read_pba_num_generic - Read device part number
 *  @hw: pointer to the HW structure
 *  @pba_num: pointer to device part number
 *
 *  Reads the product board assembly (PBA) number from the EEPROM and stores
 *  the value in pba_num.
 **/
s32 igc_read_pba_num_generic(struct igc_hw *hw, u32 *pba_num)
{
	s32 ret_val;
	u16 lwm_data;

	DEBUGFUNC("igc_read_pba_num_generic");

	ret_val = hw->lwm.ops.read(hw, LWM_PBA_OFFSET_0, 1, &lwm_data);
	if (ret_val) {
		DEBUGOUT("LWM Read Error\n");
		return ret_val;
	} else if (lwm_data == LWM_PBA_PTR_GUARD) {
		DEBUGOUT("LWM Not Supported\n");
		return -IGC_NOT_IMPLEMENTED;
	}
	*pba_num = (u32)(lwm_data << 16);

	ret_val = hw->lwm.ops.read(hw, LWM_PBA_OFFSET_1, 1, &lwm_data);
	if (ret_val) {
		DEBUGOUT("LWM Read Error\n");
		return ret_val;
	}
	*pba_num |= lwm_data;

	return IGC_SUCCESS;
}


/**
 *  igc_read_pba_raw
 *  @hw: pointer to the HW structure
 *  @eeprom_buf: optional pointer to EEPROM image
 *  @eeprom_buf_size: size of EEPROM image in words
 *  @max_pba_block_size: PBA block size limit
 *  @pba: pointer to output PBA structure
 *
 *  Reads PBA from EEPROM image when eeprom_buf is not NULL.
 *  Reads PBA from physical EEPROM device when eeprom_buf is NULL.
 *
 **/
s32 igc_read_pba_raw(struct igc_hw *hw, u16 *eeprom_buf,
		       u32 eeprom_buf_size, u16 max_pba_block_size,
		       struct igc_pba *pba)
{
	s32 ret_val;
	u16 pba_block_size;

	if (pba == NULL)
		return -IGC_ERR_PARAM;

	if (eeprom_buf == NULL) {
		ret_val = igc_read_lwm(hw, LWM_PBA_OFFSET_0, 2,
					 &pba->word[0]);
		if (ret_val)
			return ret_val;
	} else {
		if (eeprom_buf_size > LWM_PBA_OFFSET_1) {
			pba->word[0] = eeprom_buf[LWM_PBA_OFFSET_0];
			pba->word[1] = eeprom_buf[LWM_PBA_OFFSET_1];
		} else {
			return -IGC_ERR_PARAM;
		}
	}

	if (pba->word[0] == LWM_PBA_PTR_GUARD) {
		if (pba->pba_block == NULL)
			return -IGC_ERR_PARAM;

		ret_val = igc_get_pba_block_size(hw, eeprom_buf,
						   eeprom_buf_size,
						   &pba_block_size);
		if (ret_val)
			return ret_val;

		if (pba_block_size > max_pba_block_size)
			return -IGC_ERR_PARAM;

		if (eeprom_buf == NULL) {
			ret_val = igc_read_lwm(hw, pba->word[1],
						 pba_block_size,
						 pba->pba_block);
			if (ret_val)
				return ret_val;
		} else {
			if (eeprom_buf_size > (u32)(pba->word[1] +
					      pba_block_size)) {
				memcpy(pba->pba_block,
				       &eeprom_buf[pba->word[1]],
				       pba_block_size * sizeof(u16));
			} else {
				return -IGC_ERR_PARAM;
			}
		}
	}

	return IGC_SUCCESS;
}

/**
 *  igc_write_pba_raw
 *  @hw: pointer to the HW structure
 *  @eeprom_buf: optional pointer to EEPROM image
 *  @eeprom_buf_size: size of EEPROM image in words
 *  @pba: pointer to PBA structure
 *
 *  Writes PBA to EEPROM image when eeprom_buf is not NULL.
 *  Writes PBA to physical EEPROM device when eeprom_buf is NULL.
 *
 **/
s32 igc_write_pba_raw(struct igc_hw *hw, u16 *eeprom_buf,
			u32 eeprom_buf_size, struct igc_pba *pba)
{
	s32 ret_val;

	if (pba == NULL)
		return -IGC_ERR_PARAM;

	if (eeprom_buf == NULL) {
		ret_val = igc_write_lwm(hw, LWM_PBA_OFFSET_0, 2,
					  &pba->word[0]);
		if (ret_val)
			return ret_val;
	} else {
		if (eeprom_buf_size > LWM_PBA_OFFSET_1) {
			eeprom_buf[LWM_PBA_OFFSET_0] = pba->word[0];
			eeprom_buf[LWM_PBA_OFFSET_1] = pba->word[1];
		} else {
			return -IGC_ERR_PARAM;
		}
	}

	if (pba->word[0] == LWM_PBA_PTR_GUARD) {
		if (pba->pba_block == NULL)
			return -IGC_ERR_PARAM;

		if (eeprom_buf == NULL) {
			ret_val = igc_write_lwm(hw, pba->word[1],
						  pba->pba_block[0],
						  pba->pba_block);
			if (ret_val)
				return ret_val;
		} else {
			if (eeprom_buf_size > (u32)(pba->word[1] +
					      pba->pba_block[0])) {
				memcpy(&eeprom_buf[pba->word[1]],
				       pba->pba_block,
				       pba->pba_block[0] * sizeof(u16));
			} else {
				return -IGC_ERR_PARAM;
			}
		}
	}

	return IGC_SUCCESS;
}

/**
 *  igc_get_pba_block_size
 *  @hw: pointer to the HW structure
 *  @eeprom_buf: optional pointer to EEPROM image
 *  @eeprom_buf_size: size of EEPROM image in words
 *  @pba_data_size: pointer to output variable
 *
 *  Returns the size of the PBA block in words. Function operates on EEPROM
 *  image if the eeprom_buf pointer is not NULL otherwise it accesses physical
 *  EEPROM device.
 *
 **/
s32 igc_get_pba_block_size(struct igc_hw *hw, u16 *eeprom_buf,
			     u32 eeprom_buf_size, u16 *pba_block_size)
{
	s32 ret_val;
	u16 pba_word[2];
	u16 length;

	DEBUGFUNC("igc_get_pba_block_size");

	if (eeprom_buf == NULL) {
		ret_val = igc_read_lwm(hw, LWM_PBA_OFFSET_0, 2, &pba_word[0]);
		if (ret_val)
			return ret_val;
	} else {
		if (eeprom_buf_size > LWM_PBA_OFFSET_1) {
			pba_word[0] = eeprom_buf[LWM_PBA_OFFSET_0];
			pba_word[1] = eeprom_buf[LWM_PBA_OFFSET_1];
		} else {
			return -IGC_ERR_PARAM;
		}
	}

	if (pba_word[0] == LWM_PBA_PTR_GUARD) {
		if (eeprom_buf == NULL) {
			ret_val = igc_read_lwm(hw, pba_word[1] + 0, 1,
						 &length);
			if (ret_val)
				return ret_val;
		} else {
			if (eeprom_buf_size > pba_word[1])
				length = eeprom_buf[pba_word[1] + 0];
			else
				return -IGC_ERR_PARAM;
		}

		if (length == 0xFFFF || length == 0)
			return -IGC_ERR_LWM_PBA_SECTION;
	} else {
		/* PBA number in legacy format, there is no PBA Block. */
		length = 0;
	}

	if (pba_block_size != NULL)
		*pba_block_size = length;

	return IGC_SUCCESS;
}

/**
 *  igc_read_mac_addr_generic - Read device MAC address
 *  @hw: pointer to the HW structure
 *
 *  Reads the device MAC address from the EEPROM and stores the value.
 *  Since devices with two ports use the same EEPROM, we increment the
 *  last bit in the MAC address for the second port.
 **/
s32 igc_read_mac_addr_generic(struct igc_hw *hw)
{
	u32 rar_high;
	u32 rar_low;
	u16 i;

	rar_high = IGC_READ_REG(hw, IGC_RAH(0));
	rar_low = IGC_READ_REG(hw, IGC_RAL(0));

	for (i = 0; i < IGC_RAL_MAC_ADDR_LEN; i++)
		hw->mac.perm_addr[i] = (u8)(rar_low >> (i * 8));

	for (i = 0; i < IGC_RAH_MAC_ADDR_LEN; i++)
		hw->mac.perm_addr[i + 4] = (u8)(rar_high >> (i * 8));

	for (i = 0; i < ETH_ADDR_LEN; i++)
		hw->mac.addr[i] = hw->mac.perm_addr[i];

	return IGC_SUCCESS;
}

/**
 *  igc_validate_lwm_checksum_generic - Validate EEPROM checksum
 *  @hw: pointer to the HW structure
 *
 *  Callwlates the EEPROM checksum by reading/adding each word of the EEPROM
 *  and then verifies that the sum of the EEPROM is equal to 0xBABA.
 **/
s32 igc_validate_lwm_checksum_generic(struct igc_hw *hw)
{
	s32 ret_val;
	u16 checksum = 0;
	u16 i, lwm_data;

	DEBUGFUNC("igc_validate_lwm_checksum_generic");

	for (i = 0; i < (LWM_CHECKSUM_REG + 1); i++) {
		ret_val = hw->lwm.ops.read(hw, i, 1, &lwm_data);
		if (ret_val) {
			DEBUGOUT("LWM Read Error\n");
			return ret_val;
		}
		checksum += lwm_data;
	}

	if (checksum != (u16)LWM_SUM) {
		DEBUGOUT("LWM Checksum Invalid\n");
		return -IGC_ERR_LWM;
	}

	return IGC_SUCCESS;
}

/**
 *  igc_update_lwm_checksum_generic - Update EEPROM checksum
 *  @hw: pointer to the HW structure
 *
 *  Updates the EEPROM checksum by reading/adding each word of the EEPROM
 *  up to the checksum.  Then callwlates the EEPROM checksum and writes the
 *  value to the EEPROM.
 **/
s32 igc_update_lwm_checksum_generic(struct igc_hw *hw)
{
	s32 ret_val;
	u16 checksum = 0;
	u16 i, lwm_data;

	DEBUGFUNC("igc_update_lwm_checksum");

	for (i = 0; i < LWM_CHECKSUM_REG; i++) {
		ret_val = hw->lwm.ops.read(hw, i, 1, &lwm_data);
		if (ret_val) {
			DEBUGOUT("LWM Read Error while updating checksum.\n");
			return ret_val;
		}
		checksum += lwm_data;
	}
	checksum = (u16)LWM_SUM - checksum;
	ret_val = hw->lwm.ops.write(hw, LWM_CHECKSUM_REG, 1, &checksum);
	if (ret_val)
		DEBUGOUT("LWM Write Error while updating checksum.\n");

	return ret_val;
}

/**
 *  igc_reload_lwm_generic - Reloads EEPROM
 *  @hw: pointer to the HW structure
 *
 *  Reloads the EEPROM by setting the "Reinitialize from EEPROM" bit in the
 *  extended control register.
 **/
static void igc_reload_lwm_generic(struct igc_hw *hw)
{
	u32 ctrl_ext;

	DEBUGFUNC("igc_reload_lwm_generic");

	usec_delay(10);
	ctrl_ext = IGC_READ_REG(hw, IGC_CTRL_EXT);
	ctrl_ext |= IGC_CTRL_EXT_EE_RST;
	IGC_WRITE_REG(hw, IGC_CTRL_EXT, ctrl_ext);
	IGC_WRITE_FLUSH(hw);
}

/**
 *  igc_get_fw_version - Get firmware version information
 *  @hw: pointer to the HW structure
 *  @fw_vers: pointer to output version structure
 *
 *  unsupported/not present features return 0 in version structure
 **/
void igc_get_fw_version(struct igc_hw *hw, struct igc_fw_version *fw_vers)
{
	u16 eeprom_verh, eeprom_verl, etrack_test, fw_version;
	u8 q, hval, rem, result;
	u16 comb_verh, comb_verl, comb_offset;

	memset(fw_vers, 0, sizeof(struct igc_fw_version));

	/*
	 * basic eeprom version numbers, bits used vary by part and by tool
	 * used to create the lwm images. Check which data format we have.
	 */
	switch (hw->mac.type) {
	case igc_i225:
		hw->lwm.ops.read(hw, LWM_ETRACK_HIWORD, 1, &etrack_test);
		/* find combo image version */
		hw->lwm.ops.read(hw, LWM_COMB_VER_PTR, 1, &comb_offset);
		if (comb_offset && comb_offset != LWM_VER_ILWALID) {
			hw->lwm.ops.read(hw, LWM_COMB_VER_OFF + comb_offset + 1,
					1, &comb_verh);
			hw->lwm.ops.read(hw, LWM_COMB_VER_OFF + comb_offset,
					1, &comb_verl);

			/* get Option Rom version if it exists and is valid */
			if (comb_verh && comb_verl &&
					comb_verh != LWM_VER_ILWALID &&
					comb_verl != LWM_VER_ILWALID) {
				fw_vers->or_valid = true;
				fw_vers->or_major = comb_verl >>
						LWM_COMB_VER_SHFT;
				fw_vers->or_build = (comb_verl <<
						LWM_COMB_VER_SHFT) |
						(comb_verh >>
						LWM_COMB_VER_SHFT);
				fw_vers->or_patch = comb_verh &
						LWM_COMB_VER_MASK;
			}
		}
		break;
	default:
		hw->lwm.ops.read(hw, LWM_ETRACK_HIWORD, 1, &etrack_test);
		return;
	}
	hw->lwm.ops.read(hw, LWM_VERSION, 1, &fw_version);
	fw_vers->eep_major = (fw_version & LWM_MAJOR_MASK)
			      >> LWM_MAJOR_SHIFT;

	/* check for old style version format in newer images*/
	if ((fw_version & LWM_NEW_DEC_MASK) == 0x0) {
		eeprom_verl = (fw_version & LWM_COMB_VER_MASK);
	} else {
		eeprom_verl = (fw_version & LWM_MINOR_MASK)
				>> LWM_MINOR_SHIFT;
	}
	/* Colwert minor value to hex before assigning to output struct
	 * Val to be colwerted will not be higher than 99, per tool output
	 */
	q = eeprom_verl / LWM_HEX_COLW;
	hval = q * LWM_HEX_TENS;
	rem = eeprom_verl % LWM_HEX_COLW;
	result = hval + rem;
	fw_vers->eep_minor = result;

	if ((etrack_test &  LWM_MAJOR_MASK) == LWM_ETRACK_VALID) {
		hw->lwm.ops.read(hw, LWM_ETRACK_WORD, 1, &eeprom_verl);
		hw->lwm.ops.read(hw, (LWM_ETRACK_WORD + 1), 1, &eeprom_verh);
		fw_vers->etrack_id = (eeprom_verh << LWM_ETRACK_SHIFT)
			| eeprom_verl;
	} else if ((etrack_test & LWM_ETRACK_VALID) == 0) {
		hw->lwm.ops.read(hw, LWM_ETRACK_WORD, 1, &eeprom_verh);
		hw->lwm.ops.read(hw, (LWM_ETRACK_WORD + 1), 1, &eeprom_verl);
		fw_vers->etrack_id = (eeprom_verh << LWM_ETRACK_SHIFT) |
				     eeprom_verl;
	}
}
