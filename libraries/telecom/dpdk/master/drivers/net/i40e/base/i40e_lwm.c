/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2001-2020 Intel Corporation
 */

#include <inttypes.h>

#include "i40e_prototype.h"

/**
 * i40e_init_lwm_ops - Initialize LWM function pointers
 * @hw: pointer to the HW structure
 *
 * Setup the function pointers and the LWM info structure. Should be called
 * once per LWM initialization, e.g. inside the i40e_init_shared_code().
 * Please notice that the LWM term is used here (& in all methods covered
 * in this file) as an equivalent of the FLASH part mapped into the SR.
 * We are accessing FLASH always through the Shadow RAM.
 **/
enum i40e_status_code i40e_init_lwm(struct i40e_hw *hw)
{
	struct i40e_lwm_info *lwm = &hw->lwm;
	enum i40e_status_code ret_code = I40E_SUCCESS;
	u32 fla, gens;
	u8 sr_size;

	DEBUGFUNC("i40e_init_lwm");

	/* The SR size is stored regardless of the lwm programming mode
	 * as the blank mode may be used in the factory line.
	 */
	gens = rd32(hw, I40E_GLLWM_GENS);
	sr_size = ((gens & I40E_GLLWM_GENS_SR_SIZE_MASK) >>
			   I40E_GLLWM_GENS_SR_SIZE_SHIFT);
	/* Switching to words (sr_size contains power of 2KB) */
	lwm->sr_size = BIT(sr_size) * I40E_SR_WORDS_IN_1KB;

	/* Check if we are in the normal or blank LWM programming mode */
	fla = rd32(hw, I40E_GLLWM_FLA);
	if (fla & I40E_GLLWM_FLA_LOCKED_MASK) { /* Normal programming mode */
		/* Max LWM timeout */
		lwm->timeout = I40E_MAX_LWM_TIMEOUT;
		lwm->blank_lwm_mode = false;
	} else { /* Blank programming mode */
		lwm->blank_lwm_mode = true;
		ret_code = I40E_ERR_LWM_BLANK_MODE;
		i40e_debug(hw, I40E_DEBUG_LWM, "LWM init error: unsupported blank mode.\n");
	}

	return ret_code;
}

/**
 * i40e_acquire_lwm - Generic request for acquiring the LWM ownership
 * @hw: pointer to the HW structure
 * @access: LWM access type (read or write)
 *
 * This function will request LWM ownership for reading
 * via the proper Admin Command.
 **/
enum i40e_status_code i40e_acquire_lwm(struct i40e_hw *hw,
				       enum i40e_aq_resource_access_type access)
{
	enum i40e_status_code ret_code = I40E_SUCCESS;
	u64 gtime, timeout;
	u64 time_left = 0;

	DEBUGFUNC("i40e_acquire_lwm");

	if (hw->lwm.blank_lwm_mode)
		goto i40e_i40e_acquire_lwm_exit;

	ret_code = i40e_aq_request_resource(hw, I40E_LWM_RESOURCE_ID, access,
					    0, &time_left, NULL);
	/* Reading the Global Device Timer */
	gtime = rd32(hw, I40E_GLVFGEN_TIMER);

	/* Store the timeout */
	hw->lwm.hw_semaphore_timeout = I40E_MS_TO_GTIME(time_left) + gtime;

	if (ret_code)
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "LWM acquire type %d failed time_left=%" PRIu64 " ret=%d aq_err=%d\n",
			   access, time_left, ret_code, hw->aq.asq_last_status);

	if (ret_code && time_left) {
		/* Poll until the current LWM owner timeouts */
		timeout = I40E_MS_TO_GTIME(I40E_MAX_LWM_TIMEOUT) + gtime;
		while ((gtime < timeout) && time_left) {
			i40e_msec_delay(10);
			gtime = rd32(hw, I40E_GLVFGEN_TIMER);
			ret_code = i40e_aq_request_resource(hw,
							I40E_LWM_RESOURCE_ID,
							access, 0, &time_left,
							NULL);
			if (ret_code == I40E_SUCCESS) {
				hw->lwm.hw_semaphore_timeout =
					    I40E_MS_TO_GTIME(time_left) + gtime;
				break;
			}
		}
		if (ret_code != I40E_SUCCESS) {
			hw->lwm.hw_semaphore_timeout = 0;
			i40e_debug(hw, I40E_DEBUG_LWM,
				   "LWM acquire timed out, wait %" PRIu64 " ms before trying again. status=%d aq_err=%d\n",
				   time_left, ret_code, hw->aq.asq_last_status);
		}
	}

i40e_i40e_acquire_lwm_exit:
	return ret_code;
}

/**
 * i40e_release_lwm - Generic request for releasing the LWM ownership
 * @hw: pointer to the HW structure
 *
 * This function will release LWM resource via the proper Admin Command.
 **/
void i40e_release_lwm(struct i40e_hw *hw)
{
	enum i40e_status_code ret_code = I40E_SUCCESS;
	u32 total_delay = 0;

	DEBUGFUNC("i40e_release_lwm");

	if (hw->lwm.blank_lwm_mode)
		return;

	ret_code = i40e_aq_release_resource(hw, I40E_LWM_RESOURCE_ID, 0, NULL);

	/* there are some rare cases when trying to release the resource
	 * results in an admin Q timeout, so handle them correctly
	 */
	while ((ret_code == I40E_ERR_ADMIN_QUEUE_TIMEOUT) &&
	       (total_delay < hw->aq.asq_cmd_timeout)) {
			i40e_msec_delay(1);
			ret_code = i40e_aq_release_resource(hw,
						I40E_LWM_RESOURCE_ID, 0, NULL);
			total_delay++;
	}
}

/**
 * i40e_poll_sr_srctl_done_bit - Polls the GLLWM_SRCTL done bit
 * @hw: pointer to the HW structure
 *
 * Polls the SRCTL Shadow RAM register done bit.
 **/
static enum i40e_status_code i40e_poll_sr_srctl_done_bit(struct i40e_hw *hw)
{
	enum i40e_status_code ret_code = I40E_ERR_TIMEOUT;
	u32 srctl, wait_cnt;

	DEBUGFUNC("i40e_poll_sr_srctl_done_bit");

	/* Poll the I40E_GLLWM_SRCTL until the done bit is set */
	for (wait_cnt = 0; wait_cnt < I40E_SRRD_SRCTL_ATTEMPTS; wait_cnt++) {
		srctl = rd32(hw, I40E_GLLWM_SRCTL);
		if (srctl & I40E_GLLWM_SRCTL_DONE_MASK) {
			ret_code = I40E_SUCCESS;
			break;
		}
		i40e_usec_delay(5);
	}
	if (ret_code == I40E_ERR_TIMEOUT)
		i40e_debug(hw, I40E_DEBUG_LWM, "Done bit in GLLWM_SRCTL not set");
	return ret_code;
}

/**
 * i40e_read_lwm_word_srctl - Reads Shadow RAM via SRCTL register
 * @hw: pointer to the HW structure
 * @offset: offset of the Shadow RAM word to read (0x000000 - 0x001FFF)
 * @data: word read from the Shadow RAM
 *
 * Reads one 16 bit word from the Shadow RAM using the GLLWM_SRCTL register.
 **/
STATIC enum i40e_status_code i40e_read_lwm_word_srctl(struct i40e_hw *hw,
						      u16 offset,
						      u16 *data)
{
	enum i40e_status_code ret_code = I40E_ERR_TIMEOUT;
	u32 sr_reg;

	DEBUGFUNC("i40e_read_lwm_word_srctl");

	if (offset >= hw->lwm.sr_size) {
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "LWM read error: Offset %d beyond Shadow RAM limit %d\n",
			   offset, hw->lwm.sr_size);
		ret_code = I40E_ERR_PARAM;
		goto read_lwm_exit;
	}

	/* Poll the done bit first */
	ret_code = i40e_poll_sr_srctl_done_bit(hw);
	if (ret_code == I40E_SUCCESS) {
		/* Write the address and start reading */
		sr_reg = ((u32)offset << I40E_GLLWM_SRCTL_ADDR_SHIFT) |
			 BIT(I40E_GLLWM_SRCTL_START_SHIFT);
		wr32(hw, I40E_GLLWM_SRCTL, sr_reg);

		/* Poll I40E_GLLWM_SRCTL until the done bit is set */
		ret_code = i40e_poll_sr_srctl_done_bit(hw);
		if (ret_code == I40E_SUCCESS) {
			sr_reg = rd32(hw, I40E_GLLWM_SRDATA);
			*data = (u16)((sr_reg &
				       I40E_GLLWM_SRDATA_RDDATA_MASK)
				    >> I40E_GLLWM_SRDATA_RDDATA_SHIFT);
		}
	}
	if (ret_code != I40E_SUCCESS)
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "LWM read error: Couldn't access Shadow RAM address: 0x%x\n",
			   offset);

read_lwm_exit:
	return ret_code;
}

/**
 * i40e_read_lwm_aq - Read Shadow RAM.
 * @hw: pointer to the HW structure.
 * @module_pointer: module pointer location in words from the LWM beginning
 * @offset: offset in words from module start
 * @words: number of words to write
 * @data: buffer with words to write to the Shadow RAM
 * @last_command: tells the AdminQ that this is the last command
 *
 * Writes a 16 bit words buffer to the Shadow RAM using the admin command.
 **/
STATIC enum i40e_status_code i40e_read_lwm_aq(struct i40e_hw *hw,
					      u8 module_pointer, u32 offset,
					      u16 words, void *data,
					      bool last_command)
{
	enum i40e_status_code ret_code = I40E_ERR_LWM;
	struct i40e_asq_cmd_details cmd_details;

	DEBUGFUNC("i40e_read_lwm_aq");

	memset(&cmd_details, 0, sizeof(cmd_details));
	cmd_details.wb_desc = &hw->lwm_wb_desc;

	/* Here we are checking the SR limit only for the flat memory model.
	 * We cannot do it for the module-based model, as we did not acquire
	 * the LWM resource yet (we cannot get the module pointer value).
	 * Firmware will check the module-based model.
	 */
	if ((offset + words) > hw->lwm.sr_size)
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "LWM write error: offset %d beyond Shadow RAM limit %d\n",
			   (offset + words), hw->lwm.sr_size);
	else if (words > I40E_SR_SECTOR_SIZE_IN_WORDS)
		/* We can write only up to 4KB (one sector), in one AQ write */
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "LWM write fail error: tried to write %d words, limit is %d.\n",
			   words, I40E_SR_SECTOR_SIZE_IN_WORDS);
	else if (((offset + (words - 1)) / I40E_SR_SECTOR_SIZE_IN_WORDS)
		 != (offset / I40E_SR_SECTOR_SIZE_IN_WORDS))
		/* A single write cannot spread over two sectors */
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "LWM write error: cannot spread over two sectors in a single write offset=%d words=%d\n",
			   offset, words);
	else
		ret_code = i40e_aq_read_lwm(hw, module_pointer,
					    2 * offset,  /*bytes*/
					    2 * words,   /*bytes*/
					    data, last_command, &cmd_details);

	return ret_code;
}

/**
 * i40e_read_lwm_word_aq - Reads Shadow RAM via AQ
 * @hw: pointer to the HW structure
 * @offset: offset of the Shadow RAM word to read (0x000000 - 0x001FFF)
 * @data: word read from the Shadow RAM
 *
 * Reads one 16 bit word from the Shadow RAM using the AdminQ
 **/
STATIC enum i40e_status_code i40e_read_lwm_word_aq(struct i40e_hw *hw, u16 offset,
						   u16 *data)
{
	enum i40e_status_code ret_code = I40E_ERR_TIMEOUT;

	DEBUGFUNC("i40e_read_lwm_word_aq");

	ret_code = i40e_read_lwm_aq(hw, 0x0, offset, 1, data, true);
	*data = LE16_TO_CPU(*(__le16 *)data);

	return ret_code;
}

/**
 * __i40e_read_lwm_word - Reads LWM word, assumes caller does the locking
 * @hw: pointer to the HW structure
 * @offset: offset of the Shadow RAM word to read (0x000000 - 0x001FFF)
 * @data: word read from the Shadow RAM
 *
 * Reads one 16 bit word from the Shadow RAM.
 *
 * Do not use this function except in cases where the lwm lock is already
 * taken via i40e_acquire_lwm().
 **/
enum i40e_status_code __i40e_read_lwm_word(struct i40e_hw *hw,
					   u16 offset,
					   u16 *data)
{

	if (hw->flags & I40E_HW_FLAG_AQ_SRCTL_ACCESS_ENABLE)
		return i40e_read_lwm_word_aq(hw, offset, data);

	return i40e_read_lwm_word_srctl(hw, offset, data);
}

/**
 * i40e_read_lwm_word - Reads LWM word, acquires lock if necessary
 * @hw: pointer to the HW structure
 * @offset: offset of the Shadow RAM word to read (0x000000 - 0x001FFF)
 * @data: word read from the Shadow RAM
 *
 * Reads one 16 bit word from the Shadow RAM.
 **/
enum i40e_status_code i40e_read_lwm_word(struct i40e_hw *hw, u16 offset,
					 u16 *data)
{
	enum i40e_status_code ret_code = I40E_SUCCESS;

	if (hw->flags & I40E_HW_FLAG_LWM_READ_REQUIRES_LOCK)
		ret_code = i40e_acquire_lwm(hw, I40E_RESOURCE_READ);

	if (ret_code)
		return ret_code;
	ret_code = __i40e_read_lwm_word(hw, offset, data);

	if (hw->flags & I40E_HW_FLAG_LWM_READ_REQUIRES_LOCK)
		i40e_release_lwm(hw);
	return ret_code;
}

/**
 * i40e_read_lwm_module_data - Reads LWM Buffer to specified memory location
 * @hw: Pointer to the HW structure
 * @module_ptr: Pointer to module in words with respect to LWM beginning
 * @module_offset: Offset in words from module start
 * @data_offset: Offset in words from reading data area start
 * @words_data_size: Words to read from LWM
 * @data_ptr: Pointer to memory location where resulting buffer will be stored
 **/
enum i40e_status_code
i40e_read_lwm_module_data(struct i40e_hw *hw, u8 module_ptr, u16 module_offset,
			  u16 data_offset, u16 words_data_size, u16 *data_ptr)
{
	enum i40e_status_code status;
	u16 specific_ptr = 0;
	u16 ptr_value = 0;
	u16 offset = 0;

	if (module_ptr != 0) {
		status = i40e_read_lwm_word(hw, module_ptr, &ptr_value);
		if (status != I40E_SUCCESS) {
			i40e_debug(hw, I40E_DEBUG_ALL,
				   "Reading lwm word failed.Error code: %d.\n",
				   status);
			return I40E_ERR_LWM;
		}
	}
#define I40E_LWM_ILWALID_PTR_VAL 0x7FFF
#define I40E_LWM_ILWALID_VAL 0xFFFF

	/* Pointer not initialized */
	if (ptr_value == I40E_LWM_ILWALID_PTR_VAL ||
	    ptr_value == I40E_LWM_ILWALID_VAL) {
		i40e_debug(hw, I40E_DEBUG_ALL, "Pointer not initialized.\n");
		return I40E_ERR_BAD_PTR;
	}

	/* Check whether the module is in SR mapped area or outside */
	if (ptr_value & I40E_PTR_TYPE) {
		/* Pointer points outside of the Shared RAM mapped area */
		i40e_debug(hw, I40E_DEBUG_ALL,
			   "Reading lwm data failed. Pointer points outside of the Shared RAM mapped area.\n");

		return I40E_ERR_PARAM;
	} else {
		/* Read from the Shadow RAM */

		status = i40e_read_lwm_word(hw, ptr_value + module_offset,
					    &specific_ptr);
		if (status != I40E_SUCCESS) {
			i40e_debug(hw, I40E_DEBUG_ALL,
				   "Reading lwm word failed.Error code: %d.\n",
				   status);
			return I40E_ERR_LWM;
		}

		offset = ptr_value + module_offset + specific_ptr +
			data_offset;

		status = i40e_read_lwm_buffer(hw, offset, &words_data_size,
					      data_ptr);
		if (status != I40E_SUCCESS) {
			i40e_debug(hw, I40E_DEBUG_ALL,
				   "Reading lwm buffer failed.Error code: %d.\n",
				   status);
		}
	}

	return status;
}

/**
 * i40e_read_lwm_buffer_srctl - Reads Shadow RAM buffer via SRCTL register
 * @hw: pointer to the HW structure
 * @offset: offset of the Shadow RAM word to read (0x000000 - 0x001FFF).
 * @words: (in) number of words to read; (out) number of words actually read
 * @data: words read from the Shadow RAM
 *
 * Reads 16 bit words (data buffer) from the SR using the i40e_read_lwm_srrd()
 * method. The buffer read is preceded by the LWM ownership take
 * and followed by the release.
 **/
STATIC enum i40e_status_code i40e_read_lwm_buffer_srctl(struct i40e_hw *hw, u16 offset,
							u16 *words, u16 *data)
{
	enum i40e_status_code ret_code = I40E_SUCCESS;
	u16 index, word;

	DEBUGFUNC("i40e_read_lwm_buffer_srctl");

	/* Loop through the selected region */
	for (word = 0; word < *words; word++) {
		index = offset + word;
		ret_code = i40e_read_lwm_word_srctl(hw, index, &data[word]);
		if (ret_code != I40E_SUCCESS)
			break;
	}

	/* Update the number of words read from the Shadow RAM */
	*words = word;

	return ret_code;
}

/**
 * i40e_read_lwm_buffer_aq - Reads Shadow RAM buffer via AQ
 * @hw: pointer to the HW structure
 * @offset: offset of the Shadow RAM word to read (0x000000 - 0x001FFF).
 * @words: (in) number of words to read; (out) number of words actually read
 * @data: words read from the Shadow RAM
 *
 * Reads 16 bit words (data buffer) from the SR using the i40e_read_lwm_aq()
 * method. The buffer read is preceded by the LWM ownership take
 * and followed by the release.
 **/
STATIC enum i40e_status_code i40e_read_lwm_buffer_aq(struct i40e_hw *hw, u16 offset,
						     u16 *words, u16 *data)
{
	enum i40e_status_code ret_code;
	u16 read_size = *words;
	bool last_cmd = false;
	u16 words_read = 0;
	u16 i = 0;

	DEBUGFUNC("i40e_read_lwm_buffer_aq");

	do {
		/* Callwlate number of bytes we should read in this step.
		 * FVL AQ do not allow to read more than one page at a time or
		 * to cross page boundaries.
		 */
		if (offset % I40E_SR_SECTOR_SIZE_IN_WORDS)
			read_size = min(*words,
					(u16)(I40E_SR_SECTOR_SIZE_IN_WORDS -
				      (offset % I40E_SR_SECTOR_SIZE_IN_WORDS)));
		else
			read_size = min((*words - words_read),
					I40E_SR_SECTOR_SIZE_IN_WORDS);

		/* Check if this is last command, if so set proper flag */
		if ((words_read + read_size) >= *words)
			last_cmd = true;

		ret_code = i40e_read_lwm_aq(hw, 0x0, offset, read_size,
					    data + words_read, last_cmd);
		if (ret_code != I40E_SUCCESS)
			goto read_lwm_buffer_aq_exit;

		/* Increment counter for words already read and move offset to
		 * new read location
		 */
		words_read += read_size;
		offset += read_size;
	} while (words_read < *words);

	for (i = 0; i < *words; i++)
		data[i] = LE16_TO_CPU(((__le16 *)data)[i]);

read_lwm_buffer_aq_exit:
	*words = words_read;
	return ret_code;
}

/**
 * __i40e_read_lwm_buffer - Reads LWM buffer, caller must acquire lock
 * @hw: pointer to the HW structure
 * @offset: offset of the Shadow RAM word to read (0x000000 - 0x001FFF).
 * @words: (in) number of words to read; (out) number of words actually read
 * @data: words read from the Shadow RAM
 *
 * Reads 16 bit words (data buffer) from the SR using the i40e_read_lwm_srrd()
 * method.
 **/
enum i40e_status_code __i40e_read_lwm_buffer(struct i40e_hw *hw,
					     u16 offset,
					     u16 *words, u16 *data)
{
	if (hw->flags & I40E_HW_FLAG_AQ_SRCTL_ACCESS_ENABLE)
		return i40e_read_lwm_buffer_aq(hw, offset, words, data);

	return i40e_read_lwm_buffer_srctl(hw, offset, words, data);
}

/**
 * i40e_read_lwm_buffer - Reads Shadow RAM buffer and acquire lock if necessary
 * @hw: pointer to the HW structure
 * @offset: offset of the Shadow RAM word to read (0x000000 - 0x001FFF).
 * @words: (in) number of words to read; (out) number of words actually read
 * @data: words read from the Shadow RAM
 *
 * Reads 16 bit words (data buffer) from the SR using the i40e_read_lwm_srrd()
 * method. The buffer read is preceded by the LWM ownership take
 * and followed by the release.
 **/
enum i40e_status_code i40e_read_lwm_buffer(struct i40e_hw *hw, u16 offset,
					   u16 *words, u16 *data)
{
	enum i40e_status_code ret_code = I40E_SUCCESS;

	if (hw->flags & I40E_HW_FLAG_AQ_SRCTL_ACCESS_ENABLE) {
		ret_code = i40e_acquire_lwm(hw, I40E_RESOURCE_READ);
		if (!ret_code) {
			ret_code = i40e_read_lwm_buffer_aq(hw, offset, words,
							 data);
			i40e_release_lwm(hw);
		}
	} else {
		ret_code = i40e_read_lwm_buffer_srctl(hw, offset, words, data);
	}

	return ret_code;
}

/**
 * i40e_write_lwm_aq - Writes Shadow RAM.
 * @hw: pointer to the HW structure.
 * @module_pointer: module pointer location in words from the LWM beginning
 * @offset: offset in words from module start
 * @words: number of words to write
 * @data: buffer with words to write to the Shadow RAM
 * @last_command: tells the AdminQ that this is the last command
 *
 * Writes a 16 bit words buffer to the Shadow RAM using the admin command.
 **/
enum i40e_status_code i40e_write_lwm_aq(struct i40e_hw *hw, u8 module_pointer,
					u32 offset, u16 words, void *data,
					bool last_command)
{
	enum i40e_status_code ret_code = I40E_ERR_LWM;
	struct i40e_asq_cmd_details cmd_details;

	DEBUGFUNC("i40e_write_lwm_aq");

	memset(&cmd_details, 0, sizeof(cmd_details));
	cmd_details.wb_desc = &hw->lwm_wb_desc;

	/* Here we are checking the SR limit only for the flat memory model.
	 * We cannot do it for the module-based model, as we did not acquire
	 * the LWM resource yet (we cannot get the module pointer value).
	 * Firmware will check the module-based model.
	 */
	if ((offset + words) > hw->lwm.sr_size)
		DEBUGOUT("LWM write error: offset beyond Shadow RAM limit.\n");
	else if (words > I40E_SR_SECTOR_SIZE_IN_WORDS)
		/* We can write only up to 4KB (one sector), in one AQ write */
		DEBUGOUT("LWM write fail error: cannot write more than 4KB in a single write.\n");
	else if (((offset + (words - 1)) / I40E_SR_SECTOR_SIZE_IN_WORDS)
		 != (offset / I40E_SR_SECTOR_SIZE_IN_WORDS))
		/* A single write cannot spread over two sectors */
		DEBUGOUT("LWM write error: cannot spread over two sectors in a single write.\n");
	else
		ret_code = i40e_aq_update_lwm(hw, module_pointer,
					      2 * offset,  /*bytes*/
					      2 * words,   /*bytes*/
					      data, last_command, 0,
					      &cmd_details);

	return ret_code;
}

/**
 * __i40e_write_lwm_word - Writes Shadow RAM word
 * @hw: pointer to the HW structure
 * @offset: offset of the Shadow RAM word to write
 * @data: word to write to the Shadow RAM
 *
 * Writes a 16 bit word to the SR using the i40e_write_lwm_aq() method.
 * LWM ownership have to be acquired and released (on ARQ completion event
 * reception) by caller. To commit SR to LWM update checksum function
 * should be called.
 **/
enum i40e_status_code __i40e_write_lwm_word(struct i40e_hw *hw, u32 offset,
					    void *data)
{
	DEBUGFUNC("i40e_write_lwm_word");

	*((__le16 *)data) = CPU_TO_LE16(*((u16 *)data));

	/* Value 0x00 below means that we treat SR as a flat mem */
	return i40e_write_lwm_aq(hw, 0x00, offset, 1, data, false);
}

/**
 * __i40e_write_lwm_buffer - Writes Shadow RAM buffer
 * @hw: pointer to the HW structure
 * @module_pointer: module pointer location in words from the LWM beginning
 * @offset: offset of the Shadow RAM buffer to write
 * @words: number of words to write
 * @data: words to write to the Shadow RAM
 *
 * Writes a 16 bit words buffer to the Shadow RAM using the admin command.
 * LWM ownership must be acquired before calling this function and released
 * on ARQ completion event reception by caller. To commit SR to LWM update
 * checksum function should be called.
 **/
enum i40e_status_code __i40e_write_lwm_buffer(struct i40e_hw *hw,
					      u8 module_pointer, u32 offset,
					      u16 words, void *data)
{
	__le16 *le_word_ptr = (__le16 *)data;
	u16 *word_ptr = (u16 *)data;
	u32 i = 0;

	DEBUGFUNC("i40e_write_lwm_buffer");

	for (i = 0; i < words; i++)
		le_word_ptr[i] = CPU_TO_LE16(word_ptr[i]);

	/* Here we will only write one buffer as the size of the modules
	 * mirrored in the Shadow RAM is always less than 4K.
	 */
	return i40e_write_lwm_aq(hw, module_pointer, offset, words,
				 data, false);
}

/**
 * i40e_calc_lwm_checksum - Callwlates and returns the checksum
 * @hw: pointer to hardware structure
 * @checksum: pointer to the checksum
 *
 * This function callwlates SW Checksum that covers the whole 64kB shadow RAM
 * except the VPD and PCIe ALT Auto-load modules. The structure and size of VPD
 * is customer specific and unknown. Therefore, this function skips all maximum
 * possible size of VPD (1kB).
 **/
enum i40e_status_code i40e_calc_lwm_checksum(struct i40e_hw *hw, u16 *checksum)
{
	enum i40e_status_code ret_code = I40E_SUCCESS;
	struct i40e_virt_mem vmem;
	u16 pcie_alt_module = 0;
	u16 checksum_local = 0;
	u16 vpd_module = 0;
	u16 *data;
	u16 i = 0;

	DEBUGFUNC("i40e_calc_lwm_checksum");

	ret_code = i40e_allocate_virt_mem(hw, &vmem,
				    I40E_SR_SECTOR_SIZE_IN_WORDS * sizeof(u16));
	if (ret_code)
		goto i40e_calc_lwm_checksum_exit;
	data = (u16 *)vmem.va;

	/* read pointer to VPD area */
	ret_code = __i40e_read_lwm_word(hw, I40E_SR_VPD_PTR, &vpd_module);
	if (ret_code != I40E_SUCCESS) {
		ret_code = I40E_ERR_LWM_CHECKSUM;
		goto i40e_calc_lwm_checksum_exit;
	}

	/* read pointer to PCIe Alt Auto-load module */
	ret_code = __i40e_read_lwm_word(hw, I40E_SR_PCIE_ALT_AUTO_LOAD_PTR,
					&pcie_alt_module);
	if (ret_code != I40E_SUCCESS) {
		ret_code = I40E_ERR_LWM_CHECKSUM;
		goto i40e_calc_lwm_checksum_exit;
	}

	/* Callwlate SW checksum that covers the whole 64kB shadow RAM
	 * except the VPD and PCIe ALT Auto-load modules
	 */
	for (i = 0; i < hw->lwm.sr_size; i++) {
		/* Read SR page */
		if ((i % I40E_SR_SECTOR_SIZE_IN_WORDS) == 0) {
			u16 words = I40E_SR_SECTOR_SIZE_IN_WORDS;

			ret_code = __i40e_read_lwm_buffer(hw, i, &words, data);
			if (ret_code != I40E_SUCCESS) {
				ret_code = I40E_ERR_LWM_CHECKSUM;
				goto i40e_calc_lwm_checksum_exit;
			}
		}

		/* Skip Checksum word */
		if (i == I40E_SR_SW_CHECKSUM_WORD)
			continue;
		/* Skip VPD module (colwert byte size to word count) */
		if ((i >= (u32)vpd_module) &&
		    (i < ((u32)vpd_module +
		     (I40E_SR_VPD_MODULE_MAX_SIZE / 2)))) {
			continue;
		}
		/* Skip PCIe ALT module (colwert byte size to word count) */
		if ((i >= (u32)pcie_alt_module) &&
		    (i < ((u32)pcie_alt_module +
		     (I40E_SR_PCIE_ALT_MODULE_MAX_SIZE / 2)))) {
			continue;
		}

		checksum_local += data[i % I40E_SR_SECTOR_SIZE_IN_WORDS];
	}

	*checksum = (u16)I40E_SR_SW_CHECKSUM_BASE - checksum_local;

i40e_calc_lwm_checksum_exit:
	i40e_free_virt_mem(hw, &vmem);
	return ret_code;
}

/**
 * i40e_update_lwm_checksum - Updates the LWM checksum
 * @hw: pointer to hardware structure
 *
 * LWM ownership must be acquired before calling this function and released
 * on ARQ completion event reception by caller.
 * This function will commit SR to LWM.
 **/
enum i40e_status_code i40e_update_lwm_checksum(struct i40e_hw *hw)
{
	enum i40e_status_code ret_code = I40E_SUCCESS;
	u16 checksum;
	__le16 le_sum;

	DEBUGFUNC("i40e_update_lwm_checksum");

	ret_code = i40e_calc_lwm_checksum(hw, &checksum);
	le_sum = CPU_TO_LE16(checksum);
	if (ret_code == I40E_SUCCESS)
		ret_code = i40e_write_lwm_aq(hw, 0x00, I40E_SR_SW_CHECKSUM_WORD,
					     1, &le_sum, true);

	return ret_code;
}

/**
 * i40e_validate_lwm_checksum - Validate EEPROM checksum
 * @hw: pointer to hardware structure
 * @checksum: callwlated checksum
 *
 * Performs checksum callwlation and validates the LWM SW checksum. If the
 * caller does not need checksum, the value can be NULL.
 **/
enum i40e_status_code i40e_validate_lwm_checksum(struct i40e_hw *hw,
						 u16 *checksum)
{
	enum i40e_status_code ret_code = I40E_SUCCESS;
	u16 checksum_sr = 0;
	u16 checksum_local = 0;

	DEBUGFUNC("i40e_validate_lwm_checksum");

	/* We must acquire the LWM lock in order to correctly synchronize the
	 * LWM accesses across multiple PFs. Without doing so it is possible
	 * for one of the PFs to read invalid data potentially indicating that
	 * the checksum is invalid.
	 */
	ret_code = i40e_acquire_lwm(hw, I40E_RESOURCE_READ);
	if (ret_code)
		return ret_code;
	ret_code = i40e_calc_lwm_checksum(hw, &checksum_local);
	__i40e_read_lwm_word(hw, I40E_SR_SW_CHECKSUM_WORD, &checksum_sr);
	i40e_release_lwm(hw);
	if (ret_code)
		return ret_code;

	/* Verify read checksum from EEPROM is the same as
	 * callwlated checksum
	 */
	if (checksum_local != checksum_sr)
		ret_code = I40E_ERR_LWM_CHECKSUM;

	/* If the user cares, return the callwlated checksum */
	if (checksum)
		*checksum = checksum_local;

	return ret_code;
}

STATIC enum i40e_status_code i40e_lwmupd_state_init(struct i40e_hw *hw,
						    struct i40e_lwm_access *cmd,
						    u8 *bytes, int *perrno);
STATIC enum i40e_status_code i40e_lwmupd_state_reading(struct i40e_hw *hw,
						    struct i40e_lwm_access *cmd,
						    u8 *bytes, int *perrno);
STATIC enum i40e_status_code i40e_lwmupd_state_writing(struct i40e_hw *hw,
						    struct i40e_lwm_access *cmd,
						    u8 *bytes, int *perrno);
STATIC enum i40e_lwmupd_cmd i40e_lwmupd_validate_command(struct i40e_hw *hw,
						    struct i40e_lwm_access *cmd,
						    int *perrno);
STATIC enum i40e_status_code i40e_lwmupd_lwm_erase(struct i40e_hw *hw,
						   struct i40e_lwm_access *cmd,
						   int *perrno);
STATIC enum i40e_status_code i40e_lwmupd_lwm_write(struct i40e_hw *hw,
						   struct i40e_lwm_access *cmd,
						   u8 *bytes, int *perrno);
STATIC enum i40e_status_code i40e_lwmupd_lwm_read(struct i40e_hw *hw,
						  struct i40e_lwm_access *cmd,
						  u8 *bytes, int *perrno);
STATIC enum i40e_status_code i40e_lwmupd_exec_aq(struct i40e_hw *hw,
						 struct i40e_lwm_access *cmd,
						 u8 *bytes, int *perrno);
STATIC enum i40e_status_code i40e_lwmupd_get_aq_result(struct i40e_hw *hw,
						    struct i40e_lwm_access *cmd,
						    u8 *bytes, int *perrno);
STATIC enum i40e_status_code i40e_lwmupd_get_aq_event(struct i40e_hw *hw,
						    struct i40e_lwm_access *cmd,
						    u8 *bytes, int *perrno);
STATIC INLINE u8 i40e_lwmupd_get_module(u32 val)
{
	return (u8)(val & I40E_LWM_MOD_PNT_MASK);
}
STATIC INLINE u8 i40e_lwmupd_get_transaction(u32 val)
{
	return (u8)((val & I40E_LWM_TRANS_MASK) >> I40E_LWM_TRANS_SHIFT);
}

STATIC INLINE u8 i40e_lwmupd_get_preservation_flags(u32 val)
{
	return (u8)((val & I40E_LWM_PRESERVATION_FLAGS_MASK) >>
		    I40E_LWM_PRESERVATION_FLAGS_SHIFT);
}

STATIC const char *i40e_lwm_update_state_str[] = {
	"I40E_LWMUPD_ILWALID",
	"I40E_LWMUPD_READ_CON",
	"I40E_LWMUPD_READ_SNT",
	"I40E_LWMUPD_READ_LCB",
	"I40E_LWMUPD_READ_SA",
	"I40E_LWMUPD_WRITE_ERA",
	"I40E_LWMUPD_WRITE_CON",
	"I40E_LWMUPD_WRITE_SNT",
	"I40E_LWMUPD_WRITE_LCB",
	"I40E_LWMUPD_WRITE_SA",
	"I40E_LWMUPD_CSUM_CON",
	"I40E_LWMUPD_CSUM_SA",
	"I40E_LWMUPD_CSUM_LCB",
	"I40E_LWMUPD_STATUS",
	"I40E_LWMUPD_EXEC_AQ",
	"I40E_LWMUPD_GET_AQ_RESULT",
	"I40E_LWMUPD_GET_AQ_EVENT",
	"I40E_LWMUPD_GET_FEATURES",
};

/**
 * i40e_lwmupd_command - Process an LWM update command
 * @hw: pointer to hardware structure
 * @cmd: pointer to lwm update command
 * @bytes: pointer to the data buffer
 * @perrno: pointer to return error code
 *
 * Dispatches command depending on what update state is current
 **/
enum i40e_status_code i40e_lwmupd_command(struct i40e_hw *hw,
					  struct i40e_lwm_access *cmd,
					  u8 *bytes, int *perrno)
{
	enum i40e_status_code status;
	enum i40e_lwmupd_cmd upd_cmd;

	DEBUGFUNC("i40e_lwmupd_command");

	/* assume success */
	*perrno = 0;

	/* early check for status command and debug msgs */
	upd_cmd = i40e_lwmupd_validate_command(hw, cmd, perrno);

	i40e_debug(hw, I40E_DEBUG_LWM, "%s state %d lwm_release_on_hold %d opc 0x%04x cmd 0x%08x config 0x%08x offset 0x%08x data_size 0x%08x\n",
		   i40e_lwm_update_state_str[upd_cmd],
		   hw->lwmupd_state,
		   hw->lwm_release_on_done, hw->lwm_wait_opcode,
		   cmd->command, cmd->config, cmd->offset, cmd->data_size);

	if (upd_cmd == I40E_LWMUPD_ILWALID) {
		*perrno = -EFAULT;
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "i40e_lwmupd_validate_command returns %d errno %d\n",
			   upd_cmd, *perrno);
	}

	/* a status request returns immediately rather than
	 * going into the state machine
	 */
	if (upd_cmd == I40E_LWMUPD_STATUS) {
		if (!cmd->data_size) {
			*perrno = -EFAULT;
			return I40E_ERR_BUF_TOO_SHORT;
		}

		bytes[0] = hw->lwmupd_state;

		if (cmd->data_size >= 4) {
			bytes[1] = 0;
			*((u16 *)&bytes[2]) = hw->lwm_wait_opcode;
		}

		/* Clear error status on read */
		if (hw->lwmupd_state == I40E_LWMUPD_STATE_ERROR)
			hw->lwmupd_state = I40E_LWMUPD_STATE_INIT;

		return I40E_SUCCESS;
	}

	/*
	 * A supported features request returns immediately
	 * rather than going into state machine
	 */
	if (upd_cmd == I40E_LWMUPD_FEATURES) {
		if (cmd->data_size < hw->lwmupd_features.size) {
			*perrno = -EFAULT;
			return I40E_ERR_BUF_TOO_SHORT;
		}

		/*
		 * If buffer is bigger than i40e_lwmupd_features structure,
		 * make sure the trailing bytes are set to 0x0.
		 */
		if (cmd->data_size > hw->lwmupd_features.size)
			i40e_memset(bytes + hw->lwmupd_features.size, 0x0,
				    cmd->data_size - hw->lwmupd_features.size,
				    I40E_NONDMA_MEM);

		i40e_memcpy(bytes, &hw->lwmupd_features,
			    hw->lwmupd_features.size, I40E_NONDMA_MEM);

		return I40E_SUCCESS;
	}

	/* Clear status even it is not read and log */
	if (hw->lwmupd_state == I40E_LWMUPD_STATE_ERROR) {
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "Clearing I40E_LWMUPD_STATE_ERROR state without reading\n");
		hw->lwmupd_state = I40E_LWMUPD_STATE_INIT;
	}

	/* Acquire lock to prevent race condition where adminq_task
	 * can execute after i40e_lwmupd_lwm_read/write but before state
	 * variables (lwm_wait_opcode, lwm_release_on_done) are updated.
	 *
	 * During LWMUpdate, it is observed that lock could be held for
	 * ~5ms for most commands. However lock is held for ~60ms for
	 * LWMUPD_CSUM_LCB command.
	 */
	i40e_acquire_spinlock(&hw->aq.arq_spinlock);
	switch (hw->lwmupd_state) {
	case I40E_LWMUPD_STATE_INIT:
		status = i40e_lwmupd_state_init(hw, cmd, bytes, perrno);
		break;

	case I40E_LWMUPD_STATE_READING:
		status = i40e_lwmupd_state_reading(hw, cmd, bytes, perrno);
		break;

	case I40E_LWMUPD_STATE_WRITING:
		status = i40e_lwmupd_state_writing(hw, cmd, bytes, perrno);
		break;

	case I40E_LWMUPD_STATE_INIT_WAIT:
	case I40E_LWMUPD_STATE_WRITE_WAIT:
		/* if we need to stop waiting for an event, clear
		 * the wait info and return before doing anything else
		 */
		if (cmd->offset == 0xffff) {
			i40e_lwmupd_clear_wait_state(hw);
			status = I40E_SUCCESS;
			break;
		}

		status = I40E_ERR_NOT_READY;
		*perrno = -EBUSY;
		break;

	default:
		/* invalid state, should never happen */
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "LWMUPD: no such state %d\n", hw->lwmupd_state);
		status = I40E_NOT_SUPPORTED;
		*perrno = -ESRCH;
		break;
	}

	i40e_release_spinlock(&hw->aq.arq_spinlock);
	return status;
}

/**
 * i40e_lwmupd_state_init - Handle LWM update state Init
 * @hw: pointer to hardware structure
 * @cmd: pointer to lwm update command buffer
 * @bytes: pointer to the data buffer
 * @perrno: pointer to return error code
 *
 * Process legitimate commands of the Init state and conditionally set next
 * state. Reject all other commands.
 **/
STATIC enum i40e_status_code i40e_lwmupd_state_init(struct i40e_hw *hw,
						    struct i40e_lwm_access *cmd,
						    u8 *bytes, int *perrno)
{
	enum i40e_status_code status = I40E_SUCCESS;
	enum i40e_lwmupd_cmd upd_cmd;

	DEBUGFUNC("i40e_lwmupd_state_init");

	upd_cmd = i40e_lwmupd_validate_command(hw, cmd, perrno);

	switch (upd_cmd) {
	case I40E_LWMUPD_READ_SA:
		status = i40e_acquire_lwm(hw, I40E_RESOURCE_READ);
		if (status) {
			*perrno = i40e_aq_rc_to_posix(status,
						     hw->aq.asq_last_status);
		} else {
			status = i40e_lwmupd_lwm_read(hw, cmd, bytes, perrno);
			i40e_release_lwm(hw);
		}
		break;

	case I40E_LWMUPD_READ_SNT:
		status = i40e_acquire_lwm(hw, I40E_RESOURCE_READ);
		if (status) {
			*perrno = i40e_aq_rc_to_posix(status,
						     hw->aq.asq_last_status);
		} else {
			status = i40e_lwmupd_lwm_read(hw, cmd, bytes, perrno);
			if (status)
				i40e_release_lwm(hw);
			else
				hw->lwmupd_state = I40E_LWMUPD_STATE_READING;
		}
		break;

	case I40E_LWMUPD_WRITE_ERA:
		status = i40e_acquire_lwm(hw, I40E_RESOURCE_WRITE);
		if (status) {
			*perrno = i40e_aq_rc_to_posix(status,
						     hw->aq.asq_last_status);
		} else {
			status = i40e_lwmupd_lwm_erase(hw, cmd, perrno);
			if (status) {
				i40e_release_lwm(hw);
			} else {
				hw->lwm_release_on_done = true;
				hw->lwm_wait_opcode = i40e_aqc_opc_lwm_erase;
				hw->lwmupd_state = I40E_LWMUPD_STATE_INIT_WAIT;
			}
		}
		break;

	case I40E_LWMUPD_WRITE_SA:
		status = i40e_acquire_lwm(hw, I40E_RESOURCE_WRITE);
		if (status) {
			*perrno = i40e_aq_rc_to_posix(status,
						     hw->aq.asq_last_status);
		} else {
			status = i40e_lwmupd_lwm_write(hw, cmd, bytes, perrno);
			if (status) {
				i40e_release_lwm(hw);
			} else {
				hw->lwm_release_on_done = true;
				hw->lwm_wait_opcode = i40e_aqc_opc_lwm_update;
				hw->lwmupd_state = I40E_LWMUPD_STATE_INIT_WAIT;
			}
		}
		break;

	case I40E_LWMUPD_WRITE_SNT:
		status = i40e_acquire_lwm(hw, I40E_RESOURCE_WRITE);
		if (status) {
			*perrno = i40e_aq_rc_to_posix(status,
						     hw->aq.asq_last_status);
		} else {
			status = i40e_lwmupd_lwm_write(hw, cmd, bytes, perrno);
			if (status) {
				i40e_release_lwm(hw);
			} else {
				hw->lwm_wait_opcode = i40e_aqc_opc_lwm_update;
				hw->lwmupd_state = I40E_LWMUPD_STATE_WRITE_WAIT;
			}
		}
		break;

	case I40E_LWMUPD_CSUM_SA:
		status = i40e_acquire_lwm(hw, I40E_RESOURCE_WRITE);
		if (status) {
			*perrno = i40e_aq_rc_to_posix(status,
						     hw->aq.asq_last_status);
		} else {
			status = i40e_update_lwm_checksum(hw);
			if (status) {
				*perrno = hw->aq.asq_last_status ?
				   i40e_aq_rc_to_posix(status,
						       hw->aq.asq_last_status) :
				   -EIO;
				i40e_release_lwm(hw);
			} else {
				hw->lwm_release_on_done = true;
				hw->lwm_wait_opcode = i40e_aqc_opc_lwm_update;
				hw->lwmupd_state = I40E_LWMUPD_STATE_INIT_WAIT;
			}
		}
		break;

	case I40E_LWMUPD_EXEC_AQ:
		status = i40e_lwmupd_exec_aq(hw, cmd, bytes, perrno);
		break;

	case I40E_LWMUPD_GET_AQ_RESULT:
		status = i40e_lwmupd_get_aq_result(hw, cmd, bytes, perrno);
		break;

	case I40E_LWMUPD_GET_AQ_EVENT:
		status = i40e_lwmupd_get_aq_event(hw, cmd, bytes, perrno);
		break;

	default:
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "LWMUPD: bad cmd %s in init state\n",
			   i40e_lwm_update_state_str[upd_cmd]);
		status = I40E_ERR_LWM;
		*perrno = -ESRCH;
		break;
	}
	return status;
}

/**
 * i40e_lwmupd_state_reading - Handle LWM update state Reading
 * @hw: pointer to hardware structure
 * @cmd: pointer to lwm update command buffer
 * @bytes: pointer to the data buffer
 * @perrno: pointer to return error code
 *
 * LWM ownership is already held.  Process legitimate commands and set any
 * change in state; reject all other commands.
 **/
STATIC enum i40e_status_code i40e_lwmupd_state_reading(struct i40e_hw *hw,
						    struct i40e_lwm_access *cmd,
						    u8 *bytes, int *perrno)
{
	enum i40e_status_code status = I40E_SUCCESS;
	enum i40e_lwmupd_cmd upd_cmd;

	DEBUGFUNC("i40e_lwmupd_state_reading");

	upd_cmd = i40e_lwmupd_validate_command(hw, cmd, perrno);

	switch (upd_cmd) {
	case I40E_LWMUPD_READ_SA:
	case I40E_LWMUPD_READ_CON:
		status = i40e_lwmupd_lwm_read(hw, cmd, bytes, perrno);
		break;

	case I40E_LWMUPD_READ_LCB:
		status = i40e_lwmupd_lwm_read(hw, cmd, bytes, perrno);
		i40e_release_lwm(hw);
		hw->lwmupd_state = I40E_LWMUPD_STATE_INIT;
		break;

	default:
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "LWMUPD: bad cmd %s in reading state.\n",
			   i40e_lwm_update_state_str[upd_cmd]);
		status = I40E_NOT_SUPPORTED;
		*perrno = -ESRCH;
		break;
	}
	return status;
}

/**
 * i40e_lwmupd_state_writing - Handle LWM update state Writing
 * @hw: pointer to hardware structure
 * @cmd: pointer to lwm update command buffer
 * @bytes: pointer to the data buffer
 * @perrno: pointer to return error code
 *
 * LWM ownership is already held.  Process legitimate commands and set any
 * change in state; reject all other commands
 **/
STATIC enum i40e_status_code i40e_lwmupd_state_writing(struct i40e_hw *hw,
						    struct i40e_lwm_access *cmd,
						    u8 *bytes, int *perrno)
{
	enum i40e_status_code status = I40E_SUCCESS;
	enum i40e_lwmupd_cmd upd_cmd;
	bool retry_attempt = false;

	DEBUGFUNC("i40e_lwmupd_state_writing");

	upd_cmd = i40e_lwmupd_validate_command(hw, cmd, perrno);

retry:
	switch (upd_cmd) {
	case I40E_LWMUPD_WRITE_CON:
		status = i40e_lwmupd_lwm_write(hw, cmd, bytes, perrno);
		if (!status) {
			hw->lwm_wait_opcode = i40e_aqc_opc_lwm_update;
			hw->lwmupd_state = I40E_LWMUPD_STATE_WRITE_WAIT;
		}
		break;

	case I40E_LWMUPD_WRITE_LCB:
		status = i40e_lwmupd_lwm_write(hw, cmd, bytes, perrno);
		if (status) {
			*perrno = hw->aq.asq_last_status ?
				   i40e_aq_rc_to_posix(status,
						       hw->aq.asq_last_status) :
				   -EIO;
			hw->lwmupd_state = I40E_LWMUPD_STATE_INIT;
		} else {
			hw->lwm_release_on_done = true;
			hw->lwm_wait_opcode = i40e_aqc_opc_lwm_update;
			hw->lwmupd_state = I40E_LWMUPD_STATE_INIT_WAIT;
		}
		break;

	case I40E_LWMUPD_CSUM_CON:
		/* Assumes the caller has acquired the lwm */
		status = i40e_update_lwm_checksum(hw);
		if (status) {
			*perrno = hw->aq.asq_last_status ?
				   i40e_aq_rc_to_posix(status,
						       hw->aq.asq_last_status) :
				   -EIO;
			hw->lwmupd_state = I40E_LWMUPD_STATE_INIT;
		} else {
			hw->lwm_wait_opcode = i40e_aqc_opc_lwm_update;
			hw->lwmupd_state = I40E_LWMUPD_STATE_WRITE_WAIT;
		}
		break;

	case I40E_LWMUPD_CSUM_LCB:
		/* Assumes the caller has acquired the lwm */
		status = i40e_update_lwm_checksum(hw);
		if (status) {
			*perrno = hw->aq.asq_last_status ?
				   i40e_aq_rc_to_posix(status,
						       hw->aq.asq_last_status) :
				   -EIO;
			hw->lwmupd_state = I40E_LWMUPD_STATE_INIT;
		} else {
			hw->lwm_release_on_done = true;
			hw->lwm_wait_opcode = i40e_aqc_opc_lwm_update;
			hw->lwmupd_state = I40E_LWMUPD_STATE_INIT_WAIT;
		}
		break;

	default:
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "LWMUPD: bad cmd %s in writing state.\n",
			   i40e_lwm_update_state_str[upd_cmd]);
		status = I40E_NOT_SUPPORTED;
		*perrno = -ESRCH;
		break;
	}

	/* In some cirlwmstances, a multi-write transaction takes longer
	 * than the default 3 minute timeout on the write semaphore.  If
	 * the write failed with an EBUSY status, this is likely the problem,
	 * so here we try to reacquire the semaphore then retry the write.
	 * We only do one retry, then give up.
	 */
	if (status && (hw->aq.asq_last_status == I40E_AQ_RC_EBUSY) &&
	    !retry_attempt) {
		enum i40e_status_code old_status = status;
		u32 old_asq_status = hw->aq.asq_last_status;
		u32 gtime;

		gtime = rd32(hw, I40E_GLVFGEN_TIMER);
		if (gtime >= hw->lwm.hw_semaphore_timeout) {
			i40e_debug(hw, I40E_DEBUG_ALL,
				   "LWMUPD: write semaphore expired (%d >= %" PRIu64 "), retrying\n",
				   gtime, hw->lwm.hw_semaphore_timeout);
			i40e_release_lwm(hw);
			status = i40e_acquire_lwm(hw, I40E_RESOURCE_WRITE);
			if (status) {
				i40e_debug(hw, I40E_DEBUG_ALL,
					   "LWMUPD: write semaphore reacquire failed aq_err = %d\n",
					   hw->aq.asq_last_status);
				status = old_status;
				hw->aq.asq_last_status = old_asq_status;
			} else {
				retry_attempt = true;
				goto retry;
			}
		}
	}

	return status;
}

/**
 * i40e_lwmupd_clear_wait_state - clear wait state on hw
 * @hw: pointer to the hardware structure
 **/
void i40e_lwmupd_clear_wait_state(struct i40e_hw *hw)
{
	i40e_debug(hw, I40E_DEBUG_LWM,
		   "LWMUPD: clearing wait on opcode 0x%04x\n",
		   hw->lwm_wait_opcode);

	if (hw->lwm_release_on_done) {
		i40e_release_lwm(hw);
		hw->lwm_release_on_done = false;
	}
	hw->lwm_wait_opcode = 0;

	if (hw->aq.arq_last_status) {
		hw->lwmupd_state = I40E_LWMUPD_STATE_ERROR;
		return;
	}

	switch (hw->lwmupd_state) {
	case I40E_LWMUPD_STATE_INIT_WAIT:
		hw->lwmupd_state = I40E_LWMUPD_STATE_INIT;
		break;

	case I40E_LWMUPD_STATE_WRITE_WAIT:
		hw->lwmupd_state = I40E_LWMUPD_STATE_WRITING;
		break;

	default:
		break;
	}
}

/**
 * i40e_lwmupd_check_wait_event - handle LWM update operation events
 * @hw: pointer to the hardware structure
 * @opcode: the event that just happened
 * @desc: AdminQ descriptor
 **/
void i40e_lwmupd_check_wait_event(struct i40e_hw *hw, u16 opcode,
				  struct i40e_aq_desc *desc)
{
	u32 aq_desc_len = sizeof(struct i40e_aq_desc);

	if (opcode == hw->lwm_wait_opcode) {
		i40e_memcpy(&hw->lwm_aq_event_desc, desc,
			    aq_desc_len, I40E_NONDMA_TO_NONDMA);
		i40e_lwmupd_clear_wait_state(hw);
	}
}

/**
 * i40e_lwmupd_validate_command - Validate given command
 * @hw: pointer to hardware structure
 * @cmd: pointer to lwm update command buffer
 * @perrno: pointer to return error code
 *
 * Return one of the valid command types or I40E_LWMUPD_ILWALID
 **/
STATIC enum i40e_lwmupd_cmd i40e_lwmupd_validate_command(struct i40e_hw *hw,
						    struct i40e_lwm_access *cmd,
						    int *perrno)
{
	enum i40e_lwmupd_cmd upd_cmd;
	u8 module, transaction;

	DEBUGFUNC("i40e_lwmupd_validate_command\n");

	/* anything that doesn't match a recognized case is an error */
	upd_cmd = I40E_LWMUPD_ILWALID;

	transaction = i40e_lwmupd_get_transaction(cmd->config);
	module = i40e_lwmupd_get_module(cmd->config);

	/* limits on data size */
	if ((cmd->data_size < 1) ||
	    (cmd->data_size > I40E_LWMUPD_MAX_DATA)) {
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "i40e_lwmupd_validate_command data_size %d\n",
			   cmd->data_size);
		*perrno = -EFAULT;
		return I40E_LWMUPD_ILWALID;
	}

	switch (cmd->command) {
	case I40E_LWM_READ:
		switch (transaction) {
		case I40E_LWM_CON:
			upd_cmd = I40E_LWMUPD_READ_CON;
			break;
		case I40E_LWM_SNT:
			upd_cmd = I40E_LWMUPD_READ_SNT;
			break;
		case I40E_LWM_LCB:
			upd_cmd = I40E_LWMUPD_READ_LCB;
			break;
		case I40E_LWM_SA:
			upd_cmd = I40E_LWMUPD_READ_SA;
			break;
		case I40E_LWM_EXEC:
			switch (module) {
			case I40E_LWM_EXEC_GET_AQ_RESULT:
				upd_cmd = I40E_LWMUPD_GET_AQ_RESULT;
				break;
			case I40E_LWM_EXEC_FEATURES:
				upd_cmd = I40E_LWMUPD_FEATURES;
				break;
			case I40E_LWM_EXEC_STATUS:
				upd_cmd = I40E_LWMUPD_STATUS;
				break;
			default:
				*perrno = -EFAULT;
				return I40E_LWMUPD_ILWALID;
			}
			break;
		case I40E_LWM_AQE:
			upd_cmd = I40E_LWMUPD_GET_AQ_EVENT;
			break;
		}
		break;

	case I40E_LWM_WRITE:
		switch (transaction) {
		case I40E_LWM_CON:
			upd_cmd = I40E_LWMUPD_WRITE_CON;
			break;
		case I40E_LWM_SNT:
			upd_cmd = I40E_LWMUPD_WRITE_SNT;
			break;
		case I40E_LWM_LCB:
			upd_cmd = I40E_LWMUPD_WRITE_LCB;
			break;
		case I40E_LWM_SA:
			upd_cmd = I40E_LWMUPD_WRITE_SA;
			break;
		case I40E_LWM_ERA:
			upd_cmd = I40E_LWMUPD_WRITE_ERA;
			break;
		case I40E_LWM_CSUM:
			upd_cmd = I40E_LWMUPD_CSUM_CON;
			break;
		case (I40E_LWM_CSUM|I40E_LWM_SA):
			upd_cmd = I40E_LWMUPD_CSUM_SA;
			break;
		case (I40E_LWM_CSUM|I40E_LWM_LCB):
			upd_cmd = I40E_LWMUPD_CSUM_LCB;
			break;
		case I40E_LWM_EXEC:
			if (module == 0)
				upd_cmd = I40E_LWMUPD_EXEC_AQ;
			break;
		}
		break;
	}

	return upd_cmd;
}

/**
 * i40e_lwmupd_exec_aq - Run an AQ command
 * @hw: pointer to hardware structure
 * @cmd: pointer to lwm update command buffer
 * @bytes: pointer to the data buffer
 * @perrno: pointer to return error code
 *
 * cmd structure contains identifiers and data buffer
 **/
STATIC enum i40e_status_code i40e_lwmupd_exec_aq(struct i40e_hw *hw,
						 struct i40e_lwm_access *cmd,
						 u8 *bytes, int *perrno)
{
	struct i40e_asq_cmd_details cmd_details;
	enum i40e_status_code status;
	struct i40e_aq_desc *aq_desc;
	u32 buff_size = 0;
	u8 *buff = NULL;
	u32 aq_desc_len;
	u32 aq_data_len;

	i40e_debug(hw, I40E_DEBUG_LWM, "LWMUPD: %s\n", __func__);
	if (cmd->offset == 0xffff)
		return I40E_SUCCESS;

	memset(&cmd_details, 0, sizeof(cmd_details));
	cmd_details.wb_desc = &hw->lwm_wb_desc;

	aq_desc_len = sizeof(struct i40e_aq_desc);
	memset(&hw->lwm_wb_desc, 0, aq_desc_len);

	/* get the aq descriptor */
	if (cmd->data_size < aq_desc_len) {
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "LWMUPD: not enough aq desc bytes for exec, size %d < %d\n",
			   cmd->data_size, aq_desc_len);
		*perrno = -EILWAL;
		return I40E_ERR_PARAM;
	}
	aq_desc = (struct i40e_aq_desc *)bytes;

	/* if data buffer needed, make sure it's ready */
	aq_data_len = cmd->data_size - aq_desc_len;
	buff_size = max(aq_data_len, (u32)LE16_TO_CPU(aq_desc->datalen));
	if (buff_size) {
		if (!hw->lwm_buff.va) {
			status = i40e_allocate_virt_mem(hw, &hw->lwm_buff,
							hw->aq.asq_buf_size);
			if (status)
				i40e_debug(hw, I40E_DEBUG_LWM,
					   "LWMUPD: i40e_allocate_virt_mem for exec buff failed, %d\n",
					   status);
		}

		if (hw->lwm_buff.va) {
			buff = hw->lwm_buff.va;
			i40e_memcpy(buff, &bytes[aq_desc_len], aq_data_len,
				I40E_NONDMA_TO_NONDMA);
		}
	}

	if (cmd->offset)
		memset(&hw->lwm_aq_event_desc, 0, aq_desc_len);

	/* and away we go! */
	status = i40e_asq_send_command(hw, aq_desc, buff,
				       buff_size, &cmd_details);
	if (status) {
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "i40e_lwmupd_exec_aq err %s aq_err %s\n",
			   i40e_stat_str(hw, status),
			   i40e_aq_str(hw, hw->aq.asq_last_status));
		*perrno = i40e_aq_rc_to_posix(status, hw->aq.asq_last_status);
		return status;
	}

	/* should we wait for a followup event? */
	if (cmd->offset) {
		hw->lwm_wait_opcode = cmd->offset;
		hw->lwmupd_state = I40E_LWMUPD_STATE_INIT_WAIT;
	}

	return status;
}

/**
 * i40e_lwmupd_get_aq_result - Get the results from the previous exec_aq
 * @hw: pointer to hardware structure
 * @cmd: pointer to lwm update command buffer
 * @bytes: pointer to the data buffer
 * @perrno: pointer to return error code
 *
 * cmd structure contains identifiers and data buffer
 **/
STATIC enum i40e_status_code i40e_lwmupd_get_aq_result(struct i40e_hw *hw,
						    struct i40e_lwm_access *cmd,
						    u8 *bytes, int *perrno)
{
	u32 aq_total_len;
	u32 aq_desc_len;
	int remainder;
	u8 *buff;

	i40e_debug(hw, I40E_DEBUG_LWM, "LWMUPD: %s\n", __func__);

	aq_desc_len = sizeof(struct i40e_aq_desc);
	aq_total_len = aq_desc_len + LE16_TO_CPU(hw->lwm_wb_desc.datalen);

	/* check offset range */
	if (cmd->offset > aq_total_len) {
		i40e_debug(hw, I40E_DEBUG_LWM, "%s: offset too big %d > %d\n",
			   __func__, cmd->offset, aq_total_len);
		*perrno = -EILWAL;
		return I40E_ERR_PARAM;
	}

	/* check copylength range */
	if (cmd->data_size > (aq_total_len - cmd->offset)) {
		int new_len = aq_total_len - cmd->offset;

		i40e_debug(hw, I40E_DEBUG_LWM, "%s: copy length %d too big, trimming to %d\n",
			   __func__, cmd->data_size, new_len);
		cmd->data_size = new_len;
	}

	remainder = cmd->data_size;
	if (cmd->offset < aq_desc_len) {
		u32 len = aq_desc_len - cmd->offset;

		len = min(len, cmd->data_size);
		i40e_debug(hw, I40E_DEBUG_LWM, "%s: aq_desc bytes %d to %d\n",
			   __func__, cmd->offset, cmd->offset + len);

		buff = ((u8 *)&hw->lwm_wb_desc) + cmd->offset;
		i40e_memcpy(bytes, buff, len, I40E_NONDMA_TO_NONDMA);

		bytes += len;
		remainder -= len;
		buff = hw->lwm_buff.va;
	} else {
		buff = (u8 *)hw->lwm_buff.va + (cmd->offset - aq_desc_len);
	}

	if (remainder > 0) {
		int start_byte = buff - (u8 *)hw->lwm_buff.va;

		i40e_debug(hw, I40E_DEBUG_LWM, "%s: databuf bytes %d to %d\n",
			   __func__, start_byte, start_byte + remainder);
		i40e_memcpy(bytes, buff, remainder, I40E_NONDMA_TO_NONDMA);
	}

	return I40E_SUCCESS;
}

/**
 * i40e_lwmupd_get_aq_event - Get the Admin Queue event from previous exec_aq
 * @hw: pointer to hardware structure
 * @cmd: pointer to lwm update command buffer
 * @bytes: pointer to the data buffer
 * @perrno: pointer to return error code
 *
 * cmd structure contains identifiers and data buffer
 **/
STATIC enum i40e_status_code i40e_lwmupd_get_aq_event(struct i40e_hw *hw,
						    struct i40e_lwm_access *cmd,
						    u8 *bytes, int *perrno)
{
	u32 aq_total_len;
	u32 aq_desc_len;

	i40e_debug(hw, I40E_DEBUG_LWM, "LWMUPD: %s\n", __func__);

	aq_desc_len = sizeof(struct i40e_aq_desc);
	aq_total_len = aq_desc_len + LE16_TO_CPU(hw->lwm_aq_event_desc.datalen);

	/* check copylength range */
	if (cmd->data_size > aq_total_len) {
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "%s: copy length %d too big, trimming to %d\n",
			   __func__, cmd->data_size, aq_total_len);
		cmd->data_size = aq_total_len;
	}

	i40e_memcpy(bytes, &hw->lwm_aq_event_desc, cmd->data_size,
		    I40E_NONDMA_TO_NONDMA);

	return I40E_SUCCESS;
}

/**
 * i40e_lwmupd_lwm_read - Read LWM
 * @hw: pointer to hardware structure
 * @cmd: pointer to lwm update command buffer
 * @bytes: pointer to the data buffer
 * @perrno: pointer to return error code
 *
 * cmd structure contains identifiers and data buffer
 **/
STATIC enum i40e_status_code i40e_lwmupd_lwm_read(struct i40e_hw *hw,
						  struct i40e_lwm_access *cmd,
						  u8 *bytes, int *perrno)
{
	struct i40e_asq_cmd_details cmd_details;
	enum i40e_status_code status;
	u8 module, transaction;
	bool last;

	transaction = i40e_lwmupd_get_transaction(cmd->config);
	module = i40e_lwmupd_get_module(cmd->config);
	last = (transaction == I40E_LWM_LCB) || (transaction == I40E_LWM_SA);

	memset(&cmd_details, 0, sizeof(cmd_details));
	cmd_details.wb_desc = &hw->lwm_wb_desc;

	status = i40e_aq_read_lwm(hw, module, cmd->offset, (u16)cmd->data_size,
				  bytes, last, &cmd_details);
	if (status) {
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "i40e_lwmupd_lwm_read mod 0x%x  off 0x%x  len 0x%x\n",
			   module, cmd->offset, cmd->data_size);
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "i40e_lwmupd_lwm_read status %d aq %d\n",
			   status, hw->aq.asq_last_status);
		*perrno = i40e_aq_rc_to_posix(status, hw->aq.asq_last_status);
	}

	return status;
}

/**
 * i40e_lwmupd_lwm_erase - Erase an LWM module
 * @hw: pointer to hardware structure
 * @cmd: pointer to lwm update command buffer
 * @perrno: pointer to return error code
 *
 * module, offset, data_size and data are in cmd structure
 **/
STATIC enum i40e_status_code i40e_lwmupd_lwm_erase(struct i40e_hw *hw,
						   struct i40e_lwm_access *cmd,
						   int *perrno)
{
	enum i40e_status_code status = I40E_SUCCESS;
	struct i40e_asq_cmd_details cmd_details;
	u8 module, transaction;
	bool last;

	transaction = i40e_lwmupd_get_transaction(cmd->config);
	module = i40e_lwmupd_get_module(cmd->config);
	last = (transaction & I40E_LWM_LCB);

	memset(&cmd_details, 0, sizeof(cmd_details));
	cmd_details.wb_desc = &hw->lwm_wb_desc;

	status = i40e_aq_erase_lwm(hw, module, cmd->offset, (u16)cmd->data_size,
				   last, &cmd_details);
	if (status) {
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "i40e_lwmupd_lwm_erase mod 0x%x  off 0x%x len 0x%x\n",
			   module, cmd->offset, cmd->data_size);
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "i40e_lwmupd_lwm_erase status %d aq %d\n",
			   status, hw->aq.asq_last_status);
		*perrno = i40e_aq_rc_to_posix(status, hw->aq.asq_last_status);
	}

	return status;
}

/**
 * i40e_lwmupd_lwm_write - Write LWM
 * @hw: pointer to hardware structure
 * @cmd: pointer to lwm update command buffer
 * @bytes: pointer to the data buffer
 * @perrno: pointer to return error code
 *
 * module, offset, data_size and data are in cmd structure
 **/
STATIC enum i40e_status_code i40e_lwmupd_lwm_write(struct i40e_hw *hw,
						   struct i40e_lwm_access *cmd,
						   u8 *bytes, int *perrno)
{
	enum i40e_status_code status = I40E_SUCCESS;
	struct i40e_asq_cmd_details cmd_details;
	u8 module, transaction;
	u8 preservation_flags;
	bool last;

	transaction = i40e_lwmupd_get_transaction(cmd->config);
	module = i40e_lwmupd_get_module(cmd->config);
	last = (transaction & I40E_LWM_LCB);
	preservation_flags = i40e_lwmupd_get_preservation_flags(cmd->config);

	memset(&cmd_details, 0, sizeof(cmd_details));
	cmd_details.wb_desc = &hw->lwm_wb_desc;

	status = i40e_aq_update_lwm(hw, module, cmd->offset,
				    (u16)cmd->data_size, bytes, last,
				    preservation_flags, &cmd_details);
	if (status) {
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "i40e_lwmupd_lwm_write mod 0x%x off 0x%x len 0x%x\n",
			   module, cmd->offset, cmd->data_size);
		i40e_debug(hw, I40E_DEBUG_LWM,
			   "i40e_lwmupd_lwm_write status %d aq %d\n",
			   status, hw->aq.asq_last_status);
		*perrno = i40e_aq_rc_to_posix(status, hw->aq.asq_last_status);
	}

	return status;
}
