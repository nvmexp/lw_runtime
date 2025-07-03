/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2001-2020 Intel Corporation
 */

#include "e1000_api.h"

STATIC void e1000_reload_lwm_generic(struct e1000_hw *hw);

/**
 *  e1000_init_lwm_ops_generic - Initialize LWM function pointers
 *  @hw: pointer to the HW structure
 *
 *  Setups up the function pointers to no-op functions
 **/
void e1000_init_lwm_ops_generic(struct e1000_hw *hw)
{
	struct e1000_lwm_info *lwm = &hw->lwm;
	DEBUGFUNC("e1000_init_lwm_ops_generic");

	/* Initialize function pointers */
	lwm->ops.init_params = e1000_null_ops_generic;
	lwm->ops.acquire = e1000_null_ops_generic;
	lwm->ops.read = e1000_null_read_lwm;
	lwm->ops.release = e1000_null_lwm_generic;
	lwm->ops.reload = e1000_reload_lwm_generic;
	lwm->ops.update = e1000_null_ops_generic;
	lwm->ops.valid_led_default = e1000_null_led_default;
	lwm->ops.validate = e1000_null_ops_generic;
	lwm->ops.write = e1000_null_write_lwm;
}

/**
 *  e1000_null_lwm_read - No-op function, return 0
 *  @hw: pointer to the HW structure
 *  @a: dummy variable
 *  @b: dummy variable
 *  @c: dummy variable
 **/
s32 e1000_null_read_lwm(struct e1000_hw E1000_UNUSEDARG *hw,
			u16 E1000_UNUSEDARG a, u16 E1000_UNUSEDARG b,
			u16 E1000_UNUSEDARG *c)
{
	DEBUGFUNC("e1000_null_read_lwm");
	UNREFERENCED_4PARAMETER(hw, a, b, c);
	return E1000_SUCCESS;
}

/**
 *  e1000_null_lwm_generic - No-op function, return void
 *  @hw: pointer to the HW structure
 **/
void e1000_null_lwm_generic(struct e1000_hw E1000_UNUSEDARG *hw)
{
	DEBUGFUNC("e1000_null_lwm_generic");
	UNREFERENCED_1PARAMETER(hw);
	return;
}

/**
 *  e1000_null_led_default - No-op function, return 0
 *  @hw: pointer to the HW structure
 *  @data: dummy variable
 **/
s32 e1000_null_led_default(struct e1000_hw E1000_UNUSEDARG *hw,
			   u16 E1000_UNUSEDARG *data)
{
	DEBUGFUNC("e1000_null_led_default");
	UNREFERENCED_2PARAMETER(hw, data);
	return E1000_SUCCESS;
}

/**
 *  e1000_null_write_lwm - No-op function, return 0
 *  @hw: pointer to the HW structure
 *  @a: dummy variable
 *  @b: dummy variable
 *  @c: dummy variable
 **/
s32 e1000_null_write_lwm(struct e1000_hw E1000_UNUSEDARG *hw,
			 u16 E1000_UNUSEDARG a, u16 E1000_UNUSEDARG b,
			 u16 E1000_UNUSEDARG *c)
{
	DEBUGFUNC("e1000_null_write_lwm");
	UNREFERENCED_4PARAMETER(hw, a, b, c);
	return E1000_SUCCESS;
}

/**
 *  e1000_raise_eec_clk - Raise EEPROM clock
 *  @hw: pointer to the HW structure
 *  @eecd: pointer to the EEPROM
 *
 *  Enable/Raise the EEPROM clock bit.
 **/
STATIC void e1000_raise_eec_clk(struct e1000_hw *hw, u32 *eecd)
{
	*eecd = *eecd | E1000_EECD_SK;
	E1000_WRITE_REG(hw, E1000_EECD, *eecd);
	E1000_WRITE_FLUSH(hw);
	usec_delay(hw->lwm.delay_usec);
}

/**
 *  e1000_lower_eec_clk - Lower EEPROM clock
 *  @hw: pointer to the HW structure
 *  @eecd: pointer to the EEPROM
 *
 *  Clear/Lower the EEPROM clock bit.
 **/
STATIC void e1000_lower_eec_clk(struct e1000_hw *hw, u32 *eecd)
{
	*eecd = *eecd & ~E1000_EECD_SK;
	E1000_WRITE_REG(hw, E1000_EECD, *eecd);
	E1000_WRITE_FLUSH(hw);
	usec_delay(hw->lwm.delay_usec);
}

/**
 *  e1000_shift_out_eec_bits - Shift data bits our to the EEPROM
 *  @hw: pointer to the HW structure
 *  @data: data to send to the EEPROM
 *  @count: number of bits to shift out
 *
 *  We need to shift 'count' bits out to the EEPROM.  So, the value in the
 *  "data" parameter will be shifted out to the EEPROM one bit at a time.
 *  In order to do this, "data" must be broken down into bits.
 **/
STATIC void e1000_shift_out_eec_bits(struct e1000_hw *hw, u16 data, u16 count)
{
	struct e1000_lwm_info *lwm = &hw->lwm;
	u32 eecd = E1000_READ_REG(hw, E1000_EECD);
	u32 mask;

	DEBUGFUNC("e1000_shift_out_eec_bits");

	mask = 0x01 << (count - 1);
	if (lwm->type == e1000_lwm_eeprom_microwire)
		eecd &= ~E1000_EECD_DO;
	else
	if (lwm->type == e1000_lwm_eeprom_spi)
		eecd |= E1000_EECD_DO;

	do {
		eecd &= ~E1000_EECD_DI;

		if (data & mask)
			eecd |= E1000_EECD_DI;

		E1000_WRITE_REG(hw, E1000_EECD, eecd);
		E1000_WRITE_FLUSH(hw);

		usec_delay(lwm->delay_usec);

		e1000_raise_eec_clk(hw, &eecd);
		e1000_lower_eec_clk(hw, &eecd);

		mask >>= 1;
	} while (mask);

	eecd &= ~E1000_EECD_DI;
	E1000_WRITE_REG(hw, E1000_EECD, eecd);
}

/**
 *  e1000_shift_in_eec_bits - Shift data bits in from the EEPROM
 *  @hw: pointer to the HW structure
 *  @count: number of bits to shift in
 *
 *  In order to read a register from the EEPROM, we need to shift 'count' bits
 *  in from the EEPROM.  Bits are "shifted in" by raising the clock input to
 *  the EEPROM (setting the SK bit), and then reading the value of the data out
 *  "DO" bit.  During this "shifting in" process the data in "DI" bit should
 *  always be clear.
 **/
STATIC u16 e1000_shift_in_eec_bits(struct e1000_hw *hw, u16 count)
{
	u32 eecd;
	u32 i;
	u16 data;

	DEBUGFUNC("e1000_shift_in_eec_bits");

	eecd = E1000_READ_REG(hw, E1000_EECD);

	eecd &= ~(E1000_EECD_DO | E1000_EECD_DI);
	data = 0;

	for (i = 0; i < count; i++) {
		data <<= 1;
		e1000_raise_eec_clk(hw, &eecd);

		eecd = E1000_READ_REG(hw, E1000_EECD);

		eecd &= ~E1000_EECD_DI;
		if (eecd & E1000_EECD_DO)
			data |= 1;

		e1000_lower_eec_clk(hw, &eecd);
	}

	return data;
}

/**
 *  e1000_poll_eerd_eewr_done - Poll for EEPROM read/write completion
 *  @hw: pointer to the HW structure
 *  @ee_reg: EEPROM flag for polling
 *
 *  Polls the EEPROM status bit for either read or write completion based
 *  upon the value of 'ee_reg'.
 **/
s32 e1000_poll_eerd_eewr_done(struct e1000_hw *hw, int ee_reg)
{
	u32 attempts = 100000;
	u32 i, reg = 0;

	DEBUGFUNC("e1000_poll_eerd_eewr_done");

	for (i = 0; i < attempts; i++) {
		if (ee_reg == E1000_LWM_POLL_READ)
			reg = E1000_READ_REG(hw, E1000_EERD);
		else
			reg = E1000_READ_REG(hw, E1000_EEWR);

		if (reg & E1000_LWM_RW_REG_DONE)
			return E1000_SUCCESS;

		usec_delay(5);
	}

	return -E1000_ERR_LWM;
}

/**
 *  e1000_acquire_lwm_generic - Generic request for access to EEPROM
 *  @hw: pointer to the HW structure
 *
 *  Set the EEPROM access request bit and wait for EEPROM access grant bit.
 *  Return successful if access grant bit set, else clear the request for
 *  EEPROM access and return -E1000_ERR_LWM (-1).
 **/
s32 e1000_acquire_lwm_generic(struct e1000_hw *hw)
{
	u32 eecd = E1000_READ_REG(hw, E1000_EECD);
	s32 timeout = E1000_LWM_GRANT_ATTEMPTS;

	DEBUGFUNC("e1000_acquire_lwm_generic");

	E1000_WRITE_REG(hw, E1000_EECD, eecd | E1000_EECD_REQ);
	eecd = E1000_READ_REG(hw, E1000_EECD);

	while (timeout) {
		if (eecd & E1000_EECD_GNT)
			break;
		usec_delay(5);
		eecd = E1000_READ_REG(hw, E1000_EECD);
		timeout--;
	}

	if (!timeout) {
		eecd &= ~E1000_EECD_REQ;
		E1000_WRITE_REG(hw, E1000_EECD, eecd);
		DEBUGOUT("Could not acquire LWM grant\n");
		return -E1000_ERR_LWM;
	}

	return E1000_SUCCESS;
}

/**
 *  e1000_standby_lwm - Return EEPROM to standby state
 *  @hw: pointer to the HW structure
 *
 *  Return the EEPROM to a standby state.
 **/
STATIC void e1000_standby_lwm(struct e1000_hw *hw)
{
	struct e1000_lwm_info *lwm = &hw->lwm;
	u32 eecd = E1000_READ_REG(hw, E1000_EECD);

	DEBUGFUNC("e1000_standby_lwm");

	if (lwm->type == e1000_lwm_eeprom_microwire) {
		eecd &= ~(E1000_EECD_CS | E1000_EECD_SK);
		E1000_WRITE_REG(hw, E1000_EECD, eecd);
		E1000_WRITE_FLUSH(hw);
		usec_delay(lwm->delay_usec);

		e1000_raise_eec_clk(hw, &eecd);

		/* Select EEPROM */
		eecd |= E1000_EECD_CS;
		E1000_WRITE_REG(hw, E1000_EECD, eecd);
		E1000_WRITE_FLUSH(hw);
		usec_delay(lwm->delay_usec);

		e1000_lower_eec_clk(hw, &eecd);
	} else if (lwm->type == e1000_lwm_eeprom_spi) {
		/* Toggle CS to flush commands */
		eecd |= E1000_EECD_CS;
		E1000_WRITE_REG(hw, E1000_EECD, eecd);
		E1000_WRITE_FLUSH(hw);
		usec_delay(lwm->delay_usec);
		eecd &= ~E1000_EECD_CS;
		E1000_WRITE_REG(hw, E1000_EECD, eecd);
		E1000_WRITE_FLUSH(hw);
		usec_delay(lwm->delay_usec);
	}
}

/**
 *  e1000_stop_lwm - Terminate EEPROM command
 *  @hw: pointer to the HW structure
 *
 *  Terminates the current command by ilwerting the EEPROM's chip select pin.
 **/
void e1000_stop_lwm(struct e1000_hw *hw)
{
	u32 eecd;

	DEBUGFUNC("e1000_stop_lwm");

	eecd = E1000_READ_REG(hw, E1000_EECD);
	if (hw->lwm.type == e1000_lwm_eeprom_spi) {
		/* Pull CS high */
		eecd |= E1000_EECD_CS;
		e1000_lower_eec_clk(hw, &eecd);
	} else if (hw->lwm.type == e1000_lwm_eeprom_microwire) {
		/* CS on Microwire is active-high */
		eecd &= ~(E1000_EECD_CS | E1000_EECD_DI);
		E1000_WRITE_REG(hw, E1000_EECD, eecd);
		e1000_raise_eec_clk(hw, &eecd);
		e1000_lower_eec_clk(hw, &eecd);
	}
}

/**
 *  e1000_release_lwm_generic - Release exclusive access to EEPROM
 *  @hw: pointer to the HW structure
 *
 *  Stop any current commands to the EEPROM and clear the EEPROM request bit.
 **/
void e1000_release_lwm_generic(struct e1000_hw *hw)
{
	u32 eecd;

	DEBUGFUNC("e1000_release_lwm_generic");

	e1000_stop_lwm(hw);

	eecd = E1000_READ_REG(hw, E1000_EECD);
	eecd &= ~E1000_EECD_REQ;
	E1000_WRITE_REG(hw, E1000_EECD, eecd);
}

/**
 *  e1000_ready_lwm_eeprom - Prepares EEPROM for read/write
 *  @hw: pointer to the HW structure
 *
 *  Setups the EEPROM for reading and writing.
 **/
STATIC s32 e1000_ready_lwm_eeprom(struct e1000_hw *hw)
{
	struct e1000_lwm_info *lwm = &hw->lwm;
	u32 eecd = E1000_READ_REG(hw, E1000_EECD);
	u8 spi_stat_reg;

	DEBUGFUNC("e1000_ready_lwm_eeprom");

	if (lwm->type == e1000_lwm_eeprom_microwire) {
		/* Clear SK and DI */
		eecd &= ~(E1000_EECD_DI | E1000_EECD_SK);
		E1000_WRITE_REG(hw, E1000_EECD, eecd);
		/* Set CS */
		eecd |= E1000_EECD_CS;
		E1000_WRITE_REG(hw, E1000_EECD, eecd);
	} else if (lwm->type == e1000_lwm_eeprom_spi) {
		u16 timeout = LWM_MAX_RETRY_SPI;

		/* Clear SK and CS */
		eecd &= ~(E1000_EECD_CS | E1000_EECD_SK);
		E1000_WRITE_REG(hw, E1000_EECD, eecd);
		E1000_WRITE_FLUSH(hw);
		usec_delay(1);

		/* Read "Status Register" repeatedly until the LSB is cleared.
		 * The EEPROM will signal that the command has been completed
		 * by clearing bit 0 of the internal status register.  If it's
		 * not cleared within 'timeout', then error out.
		 */
		while (timeout) {
			e1000_shift_out_eec_bits(hw, LWM_RDSR_OPCODE_SPI,
						 hw->lwm.opcode_bits);
			spi_stat_reg = (u8)e1000_shift_in_eec_bits(hw, 8);
			if (!(spi_stat_reg & LWM_STATUS_RDY_SPI))
				break;

			usec_delay(5);
			e1000_standby_lwm(hw);
			timeout--;
		}

		if (!timeout) {
			DEBUGOUT("SPI LWM Status error\n");
			return -E1000_ERR_LWM;
		}
	}

	return E1000_SUCCESS;
}

/**
 *  e1000_read_lwm_spi - Read EEPROM's using SPI
 *  @hw: pointer to the HW structure
 *  @offset: offset of word in the EEPROM to read
 *  @words: number of words to read
 *  @data: word read from the EEPROM
 *
 *  Reads a 16 bit word from the EEPROM.
 **/
s32 e1000_read_lwm_spi(struct e1000_hw *hw, u16 offset, u16 words, u16 *data)
{
	struct e1000_lwm_info *lwm = &hw->lwm;
	u32 i = 0;
	s32 ret_val;
	u16 word_in;
	u8 read_opcode = LWM_READ_OPCODE_SPI;

	DEBUGFUNC("e1000_read_lwm_spi");

	/* A check for invalid values:  offset too large, too many words,
	 * and not enough words.
	 */
	if ((offset >= lwm->word_size) || (words > (lwm->word_size - offset)) ||
	    (words == 0)) {
		DEBUGOUT("lwm parameter(s) out of bounds\n");
		return -E1000_ERR_LWM;
	}

	ret_val = lwm->ops.acquire(hw);
	if (ret_val)
		return ret_val;

	ret_val = e1000_ready_lwm_eeprom(hw);
	if (ret_val)
		goto release;

	e1000_standby_lwm(hw);

	if ((lwm->address_bits == 8) && (offset >= 128))
		read_opcode |= LWM_A8_OPCODE_SPI;

	/* Send the READ command (opcode + addr) */
	e1000_shift_out_eec_bits(hw, read_opcode, lwm->opcode_bits);
	e1000_shift_out_eec_bits(hw, (u16)(offset*2), lwm->address_bits);

	/* Read the data.  SPI LWMs increment the address with each byte
	 * read and will roll over if reading beyond the end.  This allows
	 * us to read the whole LWM from any offset
	 */
	for (i = 0; i < words; i++) {
		word_in = e1000_shift_in_eec_bits(hw, 16);
		data[i] = (word_in >> 8) | (word_in << 8);
	}

release:
	lwm->ops.release(hw);

	return ret_val;
}

/**
 *  e1000_read_lwm_microwire - Reads EEPROM's using microwire
 *  @hw: pointer to the HW structure
 *  @offset: offset of word in the EEPROM to read
 *  @words: number of words to read
 *  @data: word read from the EEPROM
 *
 *  Reads a 16 bit word from the EEPROM.
 **/
s32 e1000_read_lwm_microwire(struct e1000_hw *hw, u16 offset, u16 words,
			     u16 *data)
{
	struct e1000_lwm_info *lwm = &hw->lwm;
	u32 i = 0;
	s32 ret_val;
	u8 read_opcode = LWM_READ_OPCODE_MICROWIRE;

	DEBUGFUNC("e1000_read_lwm_microwire");

	/* A check for invalid values:  offset too large, too many words,
	 * and not enough words.
	 */
	if ((offset >= lwm->word_size) || (words > (lwm->word_size - offset)) ||
	    (words == 0)) {
		DEBUGOUT("lwm parameter(s) out of bounds\n");
		return -E1000_ERR_LWM;
	}

	ret_val = lwm->ops.acquire(hw);
	if (ret_val)
		return ret_val;

	ret_val = e1000_ready_lwm_eeprom(hw);
	if (ret_val)
		goto release;

	for (i = 0; i < words; i++) {
		/* Send the READ command (opcode + addr) */
		e1000_shift_out_eec_bits(hw, read_opcode, lwm->opcode_bits);
		e1000_shift_out_eec_bits(hw, (u16)(offset + i),
					lwm->address_bits);

		/* Read the data.  For microwire, each word requires the
		 * overhead of setup and tear-down.
		 */
		data[i] = e1000_shift_in_eec_bits(hw, 16);
		e1000_standby_lwm(hw);
	}

release:
	lwm->ops.release(hw);

	return ret_val;
}

/**
 *  e1000_read_lwm_eerd - Reads EEPROM using EERD register
 *  @hw: pointer to the HW structure
 *  @offset: offset of word in the EEPROM to read
 *  @words: number of words to read
 *  @data: word read from the EEPROM
 *
 *  Reads a 16 bit word from the EEPROM using the EERD register.
 **/
s32 e1000_read_lwm_eerd(struct e1000_hw *hw, u16 offset, u16 words, u16 *data)
{
	struct e1000_lwm_info *lwm = &hw->lwm;
	u32 i, eerd = 0;
	s32 ret_val = E1000_SUCCESS;

	DEBUGFUNC("e1000_read_lwm_eerd");

	/* A check for invalid values:  offset too large, too many words,
	 * too many words for the offset, and not enough words.
	 */
	if ((offset >= lwm->word_size) || (words > (lwm->word_size - offset)) ||
	    (words == 0)) {
		DEBUGOUT("lwm parameter(s) out of bounds\n");
		return -E1000_ERR_LWM;
	}

	for (i = 0; i < words; i++) {
		eerd = ((offset + i) << E1000_LWM_RW_ADDR_SHIFT) +
		       E1000_LWM_RW_REG_START;

		E1000_WRITE_REG(hw, E1000_EERD, eerd);
		ret_val = e1000_poll_eerd_eewr_done(hw, E1000_LWM_POLL_READ);
		if (ret_val)
			break;

		data[i] = (E1000_READ_REG(hw, E1000_EERD) >>
			   E1000_LWM_RW_REG_DATA);
	}

	if (ret_val)
		DEBUGOUT1("LWM read error: %d\n", ret_val);

	return ret_val;
}

/**
 *  e1000_write_lwm_spi - Write to EEPROM using SPI
 *  @hw: pointer to the HW structure
 *  @offset: offset within the EEPROM to be written to
 *  @words: number of words to write
 *  @data: 16 bit word(s) to be written to the EEPROM
 *
 *  Writes data to EEPROM at offset using SPI interface.
 *
 *  If e1000_update_lwm_checksum is not called after this function , the
 *  EEPROM will most likely contain an invalid checksum.
 **/
s32 e1000_write_lwm_spi(struct e1000_hw *hw, u16 offset, u16 words, u16 *data)
{
	struct e1000_lwm_info *lwm = &hw->lwm;
	s32 ret_val = -E1000_ERR_LWM;
	u16 widx = 0;

	DEBUGFUNC("e1000_write_lwm_spi");

	/* A check for invalid values:  offset too large, too many words,
	 * and not enough words.
	 */
	if ((offset >= lwm->word_size) || (words > (lwm->word_size - offset)) ||
	    (words == 0)) {
		DEBUGOUT("lwm parameter(s) out of bounds\n");
		return -E1000_ERR_LWM;
	}

	while (widx < words) {
		u8 write_opcode = LWM_WRITE_OPCODE_SPI;

		ret_val = lwm->ops.acquire(hw);
		if (ret_val)
			return ret_val;

		ret_val = e1000_ready_lwm_eeprom(hw);
		if (ret_val) {
			lwm->ops.release(hw);
			return ret_val;
		}

		e1000_standby_lwm(hw);

		/* Send the WRITE ENABLE command (8 bit opcode) */
		e1000_shift_out_eec_bits(hw, LWM_WREN_OPCODE_SPI,
					 lwm->opcode_bits);

		e1000_standby_lwm(hw);

		/* Some SPI eeproms use the 8th address bit embedded in the
		 * opcode
		 */
		if ((lwm->address_bits == 8) && (offset >= 128))
			write_opcode |= LWM_A8_OPCODE_SPI;

		/* Send the Write command (8-bit opcode + addr) */
		e1000_shift_out_eec_bits(hw, write_opcode, lwm->opcode_bits);
		e1000_shift_out_eec_bits(hw, (u16)((offset + widx) * 2),
					 lwm->address_bits);

		/* Loop to allow for up to whole page write of eeprom */
		while (widx < words) {
			u16 word_out = data[widx];
			word_out = (word_out >> 8) | (word_out << 8);
			e1000_shift_out_eec_bits(hw, word_out, 16);
			widx++;

			if ((((offset + widx) * 2) % lwm->page_size) == 0) {
				e1000_standby_lwm(hw);
				break;
			}
		}
		msec_delay(10);
		lwm->ops.release(hw);
	}

	return ret_val;
}

/**
 *  e1000_write_lwm_microwire - Writes EEPROM using microwire
 *  @hw: pointer to the HW structure
 *  @offset: offset within the EEPROM to be written to
 *  @words: number of words to write
 *  @data: 16 bit word(s) to be written to the EEPROM
 *
 *  Writes data to EEPROM at offset using microwire interface.
 *
 *  If e1000_update_lwm_checksum is not called after this function , the
 *  EEPROM will most likely contain an invalid checksum.
 **/
s32 e1000_write_lwm_microwire(struct e1000_hw *hw, u16 offset, u16 words,
			      u16 *data)
{
	struct e1000_lwm_info *lwm = &hw->lwm;
	s32  ret_val;
	u32 eecd;
	u16 words_written = 0;
	u16 widx = 0;

	DEBUGFUNC("e1000_write_lwm_microwire");

	/* A check for invalid values:  offset too large, too many words,
	 * and not enough words.
	 */
	if ((offset >= lwm->word_size) || (words > (lwm->word_size - offset)) ||
	    (words == 0)) {
		DEBUGOUT("lwm parameter(s) out of bounds\n");
		return -E1000_ERR_LWM;
	}

	ret_val = lwm->ops.acquire(hw);
	if (ret_val)
		return ret_val;

	ret_val = e1000_ready_lwm_eeprom(hw);
	if (ret_val)
		goto release;

	e1000_shift_out_eec_bits(hw, LWM_EWEN_OPCODE_MICROWIRE,
				 (u16)(lwm->opcode_bits + 2));

	e1000_shift_out_eec_bits(hw, 0, (u16)(lwm->address_bits - 2));

	e1000_standby_lwm(hw);

	while (words_written < words) {
		e1000_shift_out_eec_bits(hw, LWM_WRITE_OPCODE_MICROWIRE,
					 lwm->opcode_bits);

		e1000_shift_out_eec_bits(hw, (u16)(offset + words_written),
					 lwm->address_bits);

		e1000_shift_out_eec_bits(hw, data[words_written], 16);

		e1000_standby_lwm(hw);

		for (widx = 0; widx < 200; widx++) {
			eecd = E1000_READ_REG(hw, E1000_EECD);
			if (eecd & E1000_EECD_DO)
				break;
			usec_delay(50);
		}

		if (widx == 200) {
			DEBUGOUT("LWM Write did not complete\n");
			ret_val = -E1000_ERR_LWM;
			goto release;
		}

		e1000_standby_lwm(hw);

		words_written++;
	}

	e1000_shift_out_eec_bits(hw, LWM_EWDS_OPCODE_MICROWIRE,
				 (u16)(lwm->opcode_bits + 2));

	e1000_shift_out_eec_bits(hw, 0, (u16)(lwm->address_bits - 2));

release:
	lwm->ops.release(hw);

	return ret_val;
}

/**
 *  e1000_read_pba_string_generic - Read device part number
 *  @hw: pointer to the HW structure
 *  @pba_num: pointer to device part number
 *  @pba_num_size: size of part number buffer
 *
 *  Reads the product board assembly (PBA) number from the EEPROM and stores
 *  the value in pba_num.
 **/
s32 e1000_read_pba_string_generic(struct e1000_hw *hw, u8 *pba_num,
				  u32 pba_num_size)
{
	s32 ret_val;
	u16 lwm_data;
	u16 pba_ptr;
	u16 offset;
	u16 length;

	DEBUGFUNC("e1000_read_pba_string_generic");

	if ((hw->mac.type == e1000_i210 ||
	     hw->mac.type == e1000_i211) &&
	     !e1000_get_flash_presence_i210(hw)) {
		DEBUGOUT("Flashless no PBA string\n");
		return -E1000_ERR_LWM_PBA_SECTION;
	}

	if (pba_num == NULL) {
		DEBUGOUT("PBA string buffer was null\n");
		return -E1000_ERR_ILWALID_ARGUMENT;
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
		if (pba_num_size < E1000_PBANUM_LENGTH) {
			DEBUGOUT("PBA string buffer too small\n");
			return E1000_ERR_NO_SPACE;
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

		return E1000_SUCCESS;
	}

	ret_val = hw->lwm.ops.read(hw, pba_ptr, 1, &length);
	if (ret_val) {
		DEBUGOUT("LWM Read Error\n");
		return ret_val;
	}

	if (length == 0xFFFF || length == 0) {
		DEBUGOUT("LWM PBA number section invalid length\n");
		return -E1000_ERR_LWM_PBA_SECTION;
	}
	/* check if pba_num buffer is big enough */
	if (pba_num_size < (((u32)length * 2) - 1)) {
		DEBUGOUT("PBA string buffer too small\n");
		return -E1000_ERR_NO_SPACE;
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

	return E1000_SUCCESS;
}

/**
 *  e1000_read_pba_length_generic - Read device part number length
 *  @hw: pointer to the HW structure
 *  @pba_num_size: size of part number buffer
 *
 *  Reads the product board assembly (PBA) number length from the EEPROM and
 *  stores the value in pba_num_size.
 **/
s32 e1000_read_pba_length_generic(struct e1000_hw *hw, u32 *pba_num_size)
{
	s32 ret_val;
	u16 lwm_data;
	u16 pba_ptr;
	u16 length;

	DEBUGFUNC("e1000_read_pba_length_generic");

	if (pba_num_size == NULL) {
		DEBUGOUT("PBA buffer size was null\n");
		return -E1000_ERR_ILWALID_ARGUMENT;
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
		*pba_num_size = E1000_PBANUM_LENGTH;
		return E1000_SUCCESS;
	}

	ret_val = hw->lwm.ops.read(hw, pba_ptr, 1, &length);
	if (ret_val) {
		DEBUGOUT("LWM Read Error\n");
		return ret_val;
	}

	if (length == 0xFFFF || length == 0) {
		DEBUGOUT("LWM PBA number section invalid length\n");
		return -E1000_ERR_LWM_PBA_SECTION;
	}

	/* Colwert from length in u16 values to u8 chars, add 1 for NULL,
	 * and subtract 2 because length field is included in length.
	 */
	*pba_num_size = ((u32)length * 2) - 1;

	return E1000_SUCCESS;
}

/**
 *  e1000_read_pba_num_generic - Read device part number
 *  @hw: pointer to the HW structure
 *  @pba_num: pointer to device part number
 *
 *  Reads the product board assembly (PBA) number from the EEPROM and stores
 *  the value in pba_num.
 **/
s32 e1000_read_pba_num_generic(struct e1000_hw *hw, u32 *pba_num)
{
	s32 ret_val;
	u16 lwm_data;

	DEBUGFUNC("e1000_read_pba_num_generic");

	ret_val = hw->lwm.ops.read(hw, LWM_PBA_OFFSET_0, 1, &lwm_data);
	if (ret_val) {
		DEBUGOUT("LWM Read Error\n");
		return ret_val;
	} else if (lwm_data == LWM_PBA_PTR_GUARD) {
		DEBUGOUT("LWM Not Supported\n");
		return -E1000_NOT_IMPLEMENTED;
	}
	*pba_num = (u32)(lwm_data << 16);

	ret_val = hw->lwm.ops.read(hw, LWM_PBA_OFFSET_1, 1, &lwm_data);
	if (ret_val) {
		DEBUGOUT("LWM Read Error\n");
		return ret_val;
	}
	*pba_num |= lwm_data;

	return E1000_SUCCESS;
}


/**
 *  e1000_read_pba_raw
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
s32 e1000_read_pba_raw(struct e1000_hw *hw, u16 *eeprom_buf,
		       u32 eeprom_buf_size, u16 max_pba_block_size,
		       struct e1000_pba *pba)
{
	s32 ret_val;
	u16 pba_block_size;

	if (pba == NULL)
		return -E1000_ERR_PARAM;

	if (eeprom_buf == NULL) {
		ret_val = e1000_read_lwm(hw, LWM_PBA_OFFSET_0, 2,
					 &pba->word[0]);
		if (ret_val)
			return ret_val;
	} else {
		if (eeprom_buf_size > LWM_PBA_OFFSET_1) {
			pba->word[0] = eeprom_buf[LWM_PBA_OFFSET_0];
			pba->word[1] = eeprom_buf[LWM_PBA_OFFSET_1];
		} else {
			return -E1000_ERR_PARAM;
		}
	}

	if (pba->word[0] == LWM_PBA_PTR_GUARD) {
		if (pba->pba_block == NULL)
			return -E1000_ERR_PARAM;

		ret_val = e1000_get_pba_block_size(hw, eeprom_buf,
						   eeprom_buf_size,
						   &pba_block_size);
		if (ret_val)
			return ret_val;

		if (pba_block_size > max_pba_block_size)
			return -E1000_ERR_PARAM;

		if (eeprom_buf == NULL) {
			ret_val = e1000_read_lwm(hw, pba->word[1],
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
				return -E1000_ERR_PARAM;
			}
		}
	}

	return E1000_SUCCESS;
}

/**
 *  e1000_write_pba_raw
 *  @hw: pointer to the HW structure
 *  @eeprom_buf: optional pointer to EEPROM image
 *  @eeprom_buf_size: size of EEPROM image in words
 *  @pba: pointer to PBA structure
 *
 *  Writes PBA to EEPROM image when eeprom_buf is not NULL.
 *  Writes PBA to physical EEPROM device when eeprom_buf is NULL.
 *
 **/
s32 e1000_write_pba_raw(struct e1000_hw *hw, u16 *eeprom_buf,
			u32 eeprom_buf_size, struct e1000_pba *pba)
{
	s32 ret_val;

	if (pba == NULL)
		return -E1000_ERR_PARAM;

	if (eeprom_buf == NULL) {
		ret_val = e1000_write_lwm(hw, LWM_PBA_OFFSET_0, 2,
					  &pba->word[0]);
		if (ret_val)
			return ret_val;
	} else {
		if (eeprom_buf_size > LWM_PBA_OFFSET_1) {
			eeprom_buf[LWM_PBA_OFFSET_0] = pba->word[0];
			eeprom_buf[LWM_PBA_OFFSET_1] = pba->word[1];
		} else {
			return -E1000_ERR_PARAM;
		}
	}

	if (pba->word[0] == LWM_PBA_PTR_GUARD) {
		if (pba->pba_block == NULL)
			return -E1000_ERR_PARAM;

		if (eeprom_buf == NULL) {
			ret_val = e1000_write_lwm(hw, pba->word[1],
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
				return -E1000_ERR_PARAM;
			}
		}
	}

	return E1000_SUCCESS;
}

/**
 *  e1000_get_pba_block_size
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
s32 e1000_get_pba_block_size(struct e1000_hw *hw, u16 *eeprom_buf,
			     u32 eeprom_buf_size, u16 *pba_block_size)
{
	s32 ret_val;
	u16 pba_word[2];
	u16 length;

	DEBUGFUNC("e1000_get_pba_block_size");

	if (eeprom_buf == NULL) {
		ret_val = e1000_read_lwm(hw, LWM_PBA_OFFSET_0, 2, &pba_word[0]);
		if (ret_val)
			return ret_val;
	} else {
		if (eeprom_buf_size > LWM_PBA_OFFSET_1) {
			pba_word[0] = eeprom_buf[LWM_PBA_OFFSET_0];
			pba_word[1] = eeprom_buf[LWM_PBA_OFFSET_1];
		} else {
			return -E1000_ERR_PARAM;
		}
	}

	if (pba_word[0] == LWM_PBA_PTR_GUARD) {
		if (eeprom_buf == NULL) {
			ret_val = e1000_read_lwm(hw, pba_word[1] + 0, 1,
						 &length);
			if (ret_val)
				return ret_val;
		} else {
			if (eeprom_buf_size > pba_word[1])
				length = eeprom_buf[pba_word[1] + 0];
			else
				return -E1000_ERR_PARAM;
		}

		if (length == 0xFFFF || length == 0)
			return -E1000_ERR_LWM_PBA_SECTION;
	} else {
		/* PBA number in legacy format, there is no PBA Block. */
		length = 0;
	}

	if (pba_block_size != NULL)
		*pba_block_size = length;

	return E1000_SUCCESS;
}

/**
 *  e1000_read_mac_addr_generic - Read device MAC address
 *  @hw: pointer to the HW structure
 *
 *  Reads the device MAC address from the EEPROM and stores the value.
 *  Since devices with two ports use the same EEPROM, we increment the
 *  last bit in the MAC address for the second port.
 **/
s32 e1000_read_mac_addr_generic(struct e1000_hw *hw)
{
	u32 rar_high;
	u32 rar_low;
	u16 i;

	rar_high = E1000_READ_REG(hw, E1000_RAH(0));
	rar_low = E1000_READ_REG(hw, E1000_RAL(0));

	for (i = 0; i < E1000_RAL_MAC_ADDR_LEN; i++)
		hw->mac.perm_addr[i] = (u8)(rar_low >> (i*8));

	for (i = 0; i < E1000_RAH_MAC_ADDR_LEN; i++)
		hw->mac.perm_addr[i+4] = (u8)(rar_high >> (i*8));

	for (i = 0; i < ETH_ADDR_LEN; i++)
		hw->mac.addr[i] = hw->mac.perm_addr[i];

	return E1000_SUCCESS;
}

/**
 *  e1000_validate_lwm_checksum_generic - Validate EEPROM checksum
 *  @hw: pointer to the HW structure
 *
 *  Callwlates the EEPROM checksum by reading/adding each word of the EEPROM
 *  and then verifies that the sum of the EEPROM is equal to 0xBABA.
 **/
s32 e1000_validate_lwm_checksum_generic(struct e1000_hw *hw)
{
	s32 ret_val;
	u16 checksum = 0;
	u16 i, lwm_data;

	DEBUGFUNC("e1000_validate_lwm_checksum_generic");

	for (i = 0; i < (LWM_CHECKSUM_REG + 1); i++) {
		ret_val = hw->lwm.ops.read(hw, i, 1, &lwm_data);
		if (ret_val) {
			DEBUGOUT("LWM Read Error\n");
			return ret_val;
		}
		checksum += lwm_data;
	}

	if (checksum != (u16) LWM_SUM) {
		DEBUGOUT("LWM Checksum Invalid\n");
		return -E1000_ERR_LWM;
	}

	return E1000_SUCCESS;
}

/**
 *  e1000_update_lwm_checksum_generic - Update EEPROM checksum
 *  @hw: pointer to the HW structure
 *
 *  Updates the EEPROM checksum by reading/adding each word of the EEPROM
 *  up to the checksum.  Then callwlates the EEPROM checksum and writes the
 *  value to the EEPROM.
 **/
s32 e1000_update_lwm_checksum_generic(struct e1000_hw *hw)
{
	s32 ret_val;
	u16 checksum = 0;
	u16 i, lwm_data;

	DEBUGFUNC("e1000_update_lwm_checksum");

	for (i = 0; i < LWM_CHECKSUM_REG; i++) {
		ret_val = hw->lwm.ops.read(hw, i, 1, &lwm_data);
		if (ret_val) {
			DEBUGOUT("LWM Read Error while updating checksum.\n");
			return ret_val;
		}
		checksum += lwm_data;
	}
	checksum = (u16) LWM_SUM - checksum;
	ret_val = hw->lwm.ops.write(hw, LWM_CHECKSUM_REG, 1, &checksum);
	if (ret_val)
		DEBUGOUT("LWM Write Error while updating checksum.\n");

	return ret_val;
}

/**
 *  e1000_reload_lwm_generic - Reloads EEPROM
 *  @hw: pointer to the HW structure
 *
 *  Reloads the EEPROM by setting the "Reinitialize from EEPROM" bit in the
 *  extended control register.
 **/
STATIC void e1000_reload_lwm_generic(struct e1000_hw *hw)
{
	u32 ctrl_ext;

	DEBUGFUNC("e1000_reload_lwm_generic");

	usec_delay(10);
	ctrl_ext = E1000_READ_REG(hw, E1000_CTRL_EXT);
	ctrl_ext |= E1000_CTRL_EXT_EE_RST;
	E1000_WRITE_REG(hw, E1000_CTRL_EXT, ctrl_ext);
	E1000_WRITE_FLUSH(hw);
}

/**
 *  e1000_get_fw_version - Get firmware version information
 *  @hw: pointer to the HW structure
 *  @fw_vers: pointer to output version structure
 *
 *  unsupported/not present features return 0 in version structure
 **/
void e1000_get_fw_version(struct e1000_hw *hw, struct e1000_fw_version *fw_vers)
{
	u16 eeprom_verh, eeprom_verl, etrack_test, fw_version;
	u8 q, hval, rem, result;
	u16 comb_verh, comb_verl, comb_offset;

	memset(fw_vers, 0, sizeof(struct e1000_fw_version));

	/* basic eeprom version numbers, bits used vary by part and by tool
	 * used to create the lwm images */
	/* Check which data format we have */
	switch (hw->mac.type) {
	case e1000_i211:
		e1000_read_ilwm_version(hw, fw_vers);
		return;
	case e1000_82575:
	case e1000_82576:
	case e1000_82580:
	case e1000_i354:
		hw->lwm.ops.read(hw, LWM_ETRACK_HIWORD, 1, &etrack_test);
		/* Use this format, unless EETRACK ID exists,
		 * then use alternate format
		 */
		if ((etrack_test &  LWM_MAJOR_MASK) != LWM_ETRACK_VALID) {
			hw->lwm.ops.read(hw, LWM_VERSION, 1, &fw_version);
			fw_vers->eep_major = (fw_version & LWM_MAJOR_MASK)
					      >> LWM_MAJOR_SHIFT;
			fw_vers->eep_minor = (fw_version & LWM_MINOR_MASK)
					      >> LWM_MINOR_SHIFT;
			fw_vers->eep_build = (fw_version & LWM_IMAGE_ID_MASK);
			goto etrack_id;
		}
		break;
	case e1000_i210:
		if (!(e1000_get_flash_presence_i210(hw))) {
			e1000_read_ilwm_version(hw, fw_vers);
			return;
		}
		/* fall through */
	case e1000_i350:
		hw->lwm.ops.read(hw, LWM_ETRACK_HIWORD, 1, &etrack_test);
		/* find combo image version */
		hw->lwm.ops.read(hw, LWM_COMB_VER_PTR, 1, &comb_offset);
		if ((comb_offset != 0x0) &&
		    (comb_offset != LWM_VER_ILWALID)) {

			hw->lwm.ops.read(hw, (LWM_COMB_VER_OFF + comb_offset
					 + 1), 1, &comb_verh);
			hw->lwm.ops.read(hw, (LWM_COMB_VER_OFF + comb_offset),
					 1, &comb_verl);

			/* get Option Rom version if it exists and is valid */
			if ((comb_verh && comb_verl) &&
			    ((comb_verh != LWM_VER_ILWALID) &&
			     (comb_verl != LWM_VER_ILWALID))) {

				fw_vers->or_valid = true;
				fw_vers->or_major =
					comb_verl >> LWM_COMB_VER_SHFT;
				fw_vers->or_build =
					(comb_verl << LWM_COMB_VER_SHFT)
					| (comb_verh >> LWM_COMB_VER_SHFT);
				fw_vers->or_patch =
					comb_verh & LWM_COMB_VER_MASK;
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

etrack_id:
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


