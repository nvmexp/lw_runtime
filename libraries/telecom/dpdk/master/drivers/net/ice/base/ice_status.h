/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2001-2020 Intel Corporation
 */

#ifndef _ICE_STATUS_H_
#define _ICE_STATUS_H_

/* Error Codes */
enum ice_status {
	ICE_SUCCESS				= 0,

	/* Generic codes : Range -1..-49 */
	ICE_ERR_PARAM				= -1,
	ICE_ERR_NOT_IMPL			= -2,
	ICE_ERR_NOT_READY			= -3,
	ICE_ERR_NOT_SUPPORTED			= -4,
	ICE_ERR_BAD_PTR				= -5,
	ICE_ERR_ILWAL_SIZE			= -6,
	ICE_ERR_DEVICE_NOT_SUPPORTED		= -8,
	ICE_ERR_RESET_FAILED			= -9,
	ICE_ERR_FW_API_VER			= -10,
	ICE_ERR_NO_MEMORY			= -11,
	ICE_ERR_CFG				= -12,
	ICE_ERR_OUT_OF_RANGE			= -13,
	ICE_ERR_ALREADY_EXISTS			= -14,
	ICE_ERR_DOES_NOT_EXIST			= -15,
	ICE_ERR_IN_USE				= -16,
	ICE_ERR_MAX_LIMIT			= -17,
	ICE_ERR_RESET_ONGOING			= -18,
	ICE_ERR_HW_TABLE			= -19,
	ICE_ERR_FW_DDP_MISMATCH			= -20,

	/* LWM specific error codes: Range -50..-59 */
	ICE_ERR_LWM				= -50,
	ICE_ERR_LWM_CHECKSUM			= -51,
	ICE_ERR_BUF_TOO_SHORT			= -52,
	ICE_ERR_LWM_BLANK_MODE			= -53,

	/* ARQ/ASQ specific error codes. Range -100..-109 */
	ICE_ERR_AQ_ERROR			= -100,
	ICE_ERR_AQ_TIMEOUT			= -101,
	ICE_ERR_AQ_FULL				= -102,
	ICE_ERR_AQ_NO_WORK			= -103,
	ICE_ERR_AQ_EMPTY			= -104,
	ICE_ERR_AQ_FW_CRITICAL			= -105,
};

#endif /* _ICE_STATUS_H_ */
