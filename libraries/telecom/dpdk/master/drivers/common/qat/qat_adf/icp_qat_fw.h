/* SPDX-License-Identifier: (BSD-3-Clause OR GPL-2.0)
 * Copyright(c) 2015-2018 Intel Corporation
 */
#ifndef _ICP_QAT_FW_H_
#define _ICP_QAT_FW_H_
#include <sys/types.h>
#include "icp_qat_hw.h"

#define QAT_FIELD_SET(flags, val, bitpos, mask) \
{ (flags) = (((flags) & (~((mask) << (bitpos)))) | \
		(((val) & (mask)) << (bitpos))) ; }

#define QAT_FIELD_GET(flags, bitpos, mask) \
	(((flags) >> (bitpos)) & (mask))

#define ICP_QAT_FW_REQ_DEFAULT_SZ 128
#define ICP_QAT_FW_RESP_DEFAULT_SZ 32
#define ICP_QAT_FW_COMN_ONE_BYTE_SHIFT 8
#define ICP_QAT_FW_COMN_SINGLE_BYTE_MASK 0xFF
#define ICP_QAT_FW_NUM_LONGWORDS_1 1
#define ICP_QAT_FW_NUM_LONGWORDS_2 2
#define ICP_QAT_FW_NUM_LONGWORDS_3 3
#define ICP_QAT_FW_NUM_LONGWORDS_4 4
#define ICP_QAT_FW_NUM_LONGWORDS_5 5
#define ICP_QAT_FW_NUM_LONGWORDS_6 6
#define ICP_QAT_FW_NUM_LONGWORDS_7 7
#define ICP_QAT_FW_NUM_LONGWORDS_10 10
#define ICP_QAT_FW_NUM_LONGWORDS_13 13
#define ICP_QAT_FW_NULL_REQ_SERV_ID 1

enum icp_qat_fw_comn_resp_serv_id {
	ICP_QAT_FW_COMN_RESP_SERV_NULL,
	ICP_QAT_FW_COMN_RESP_SERV_CPM_FW,
	ICP_QAT_FW_COMN_RESP_SERV_DELIMITER
};

enum icp_qat_fw_comn_request_id {
	ICP_QAT_FW_COMN_REQ_NULL = 0,
	ICP_QAT_FW_COMN_REQ_CPM_FW_PKE = 3,
	ICP_QAT_FW_COMN_REQ_CPM_FW_LA = 4,
	ICP_QAT_FW_COMN_REQ_CPM_FW_DMA = 7,
	ICP_QAT_FW_COMN_REQ_CPM_FW_COMP = 9,
	ICP_QAT_FW_COMN_REQ_DELIMITER
};

struct icp_qat_fw_comn_req_hdr_cd_pars {
	union {
		struct {
			uint64_t content_desc_addr;
			uint16_t content_desc_resrvd1;
			uint8_t content_desc_params_sz;
			uint8_t content_desc_hdr_resrvd2;
			uint32_t content_desc_resrvd3;
		} s;
		struct {
			uint32_t serv_specif_fields[4];
		} s1;
	} u;
};

struct icp_qat_fw_comn_req_mid {
	uint64_t opaque_data;
	uint64_t src_data_addr;
	uint64_t dest_data_addr;
	uint32_t src_length;
	uint32_t dst_length;
};

struct icp_qat_fw_comn_req_cd_ctrl {
	uint32_t content_desc_ctrl_lw[ICP_QAT_FW_NUM_LONGWORDS_5];
};

struct icp_qat_fw_comn_req_hdr {
	uint8_t resrvd1;
	uint8_t service_cmd_id;
	uint8_t service_type;
	uint8_t hdr_flags;
	uint16_t serv_specif_flags;
	uint16_t comn_req_flags;
};

struct icp_qat_fw_comn_req_rqpars {
	uint32_t serv_specif_rqpars_lw[ICP_QAT_FW_NUM_LONGWORDS_13];
};

struct icp_qat_fw_comn_req {
	struct icp_qat_fw_comn_req_hdr comn_hdr;
	struct icp_qat_fw_comn_req_hdr_cd_pars cd_pars;
	struct icp_qat_fw_comn_req_mid comn_mid;
	struct icp_qat_fw_comn_req_rqpars serv_specif_rqpars;
	struct icp_qat_fw_comn_req_cd_ctrl cd_ctrl;
};

struct icp_qat_fw_comn_error {
	uint8_t xlat_err_code;
	uint8_t cmp_err_code;
};

struct icp_qat_fw_comn_resp_hdr {
	uint8_t resrvd1;
	uint8_t service_id;
	uint8_t response_type;
	uint8_t hdr_flags;
	struct icp_qat_fw_comn_error comn_error;
	uint8_t comn_status;
	uint8_t cmd_id;
};

struct icp_qat_fw_comn_resp {
	struct icp_qat_fw_comn_resp_hdr comn_hdr;
	uint64_t opaque_data;
	uint32_t resrvd[ICP_QAT_FW_NUM_LONGWORDS_4];
};

#define ICP_QAT_FW_COMN_REQ_FLAG_SET 1
#define ICP_QAT_FW_COMN_REQ_FLAG_CLR 0
#define ICP_QAT_FW_COMN_VALID_FLAG_BITPOS 7
#define ICP_QAT_FW_COMN_VALID_FLAG_MASK 0x1
#define ICP_QAT_FW_COMN_HDR_RESRVD_FLD_MASK 0x7F
#define ICP_QAT_FW_COMN_CLW_FLAG_BITPOS 6
#define ICP_QAT_FW_COMN_CLW_FLAG_MASK 0x1
#define ICP_QAT_FW_COMN_CLWNR_FLAG_BITPOS 5
#define ICP_QAT_FW_COMN_CLWNR_FLAG_MASK 0x1
#define ICP_QAT_FW_COMN_NULL_VERSION_FLAG_BITPOS 0
#define ICP_QAT_FW_COMN_NULL_VERSION_FLAG_MASK 0x1

#define ICP_QAT_FW_COMN_OV_SRV_TYPE_GET(icp_qat_fw_comn_req_hdr_t) \
	icp_qat_fw_comn_req_hdr_t.service_type

#define ICP_QAT_FW_COMN_OV_SRV_TYPE_SET(icp_qat_fw_comn_req_hdr_t, val) \
	icp_qat_fw_comn_req_hdr_t.service_type = val

#define ICP_QAT_FW_COMN_OV_SRV_CMD_ID_GET(icp_qat_fw_comn_req_hdr_t) \
	icp_qat_fw_comn_req_hdr_t.service_cmd_id

#define ICP_QAT_FW_COMN_OV_SRV_CMD_ID_SET(icp_qat_fw_comn_req_hdr_t, val) \
	icp_qat_fw_comn_req_hdr_t.service_cmd_id = val

#define ICP_QAT_FW_COMN_HDR_VALID_FLAG_GET(hdr_t) \
	ICP_QAT_FW_COMN_VALID_FLAG_GET(hdr_t.hdr_flags)

#define ICP_QAT_FW_COMN_HDR_CLWNR_FLAG_GET(hdr_flags) \
	QAT_FIELD_GET(hdr_flags, \
		ICP_QAT_FW_COMN_CLWNR_FLAG_BITPOS, \
		ICP_QAT_FW_COMN_CLWNR_FLAG_MASK)

#define ICP_QAT_FW_COMN_HDR_CLW_FLAG_GET(hdr_flags) \
	QAT_FIELD_GET(hdr_flags, \
		ICP_QAT_FW_COMN_CLW_FLAG_BITPOS, \
		ICP_QAT_FW_COMN_CLW_FLAG_MASK)

#define ICP_QAT_FW_COMN_HDR_VALID_FLAG_SET(hdr_t, val) \
	ICP_QAT_FW_COMN_VALID_FLAG_SET(hdr_t, val)

#define ICP_QAT_FW_COMN_VALID_FLAG_GET(hdr_flags) \
	QAT_FIELD_GET(hdr_flags, \
	ICP_QAT_FW_COMN_VALID_FLAG_BITPOS, \
	ICP_QAT_FW_COMN_VALID_FLAG_MASK)

#define ICP_QAT_FW_COMN_HDR_RESRVD_FLD_GET(hdr_flags) \
	(hdr_flags & ICP_QAT_FW_COMN_HDR_RESRVD_FLD_MASK)

#define ICP_QAT_FW_COMN_VALID_FLAG_SET(hdr_t, val) \
	QAT_FIELD_SET((hdr_t.hdr_flags), (val), \
	ICP_QAT_FW_COMN_VALID_FLAG_BITPOS, \
	ICP_QAT_FW_COMN_VALID_FLAG_MASK)

#define ICP_QAT_FW_COMN_HDR_FLAGS_BUILD(valid) \
	(((valid) & ICP_QAT_FW_COMN_VALID_FLAG_MASK) << \
	 ICP_QAT_FW_COMN_VALID_FLAG_BITPOS)

#define QAT_COMN_PTR_TYPE_BITPOS 0
#define QAT_COMN_PTR_TYPE_MASK 0x1
#define QAT_COMN_CD_FLD_TYPE_BITPOS 1
#define QAT_COMN_CD_FLD_TYPE_MASK 0x1
#define QAT_COMN_PTR_TYPE_FLAT 0x0
#define QAT_COMN_PTR_TYPE_SGL 0x1
#define QAT_COMN_CD_FLD_TYPE_64BIT_ADR 0x0
#define QAT_COMN_CD_FLD_TYPE_16BYTE_DATA 0x1
#define QAT_COMN_EXT_FLAGS_BITPOS 8
#define QAT_COMN_EXT_FLAGS_MASK 0x1
#define QAT_COMN_EXT_FLAGS_USED 0x1

#define ICP_QAT_FW_COMN_FLAGS_BUILD(cdt, ptr) \
	((((cdt) & QAT_COMN_CD_FLD_TYPE_MASK) << QAT_COMN_CD_FLD_TYPE_BITPOS) \
	 | (((ptr) & QAT_COMN_PTR_TYPE_MASK) << QAT_COMN_PTR_TYPE_BITPOS))

#define ICP_QAT_FW_COMN_PTR_TYPE_GET(flags) \
	QAT_FIELD_GET(flags, QAT_COMN_PTR_TYPE_BITPOS, QAT_COMN_PTR_TYPE_MASK)

#define ICP_QAT_FW_COMN_CD_FLD_TYPE_GET(flags) \
	QAT_FIELD_GET(flags, QAT_COMN_CD_FLD_TYPE_BITPOS, \
			QAT_COMN_CD_FLD_TYPE_MASK)

#define ICP_QAT_FW_COMN_PTR_TYPE_SET(flags, val) \
	QAT_FIELD_SET(flags, val, QAT_COMN_PTR_TYPE_BITPOS, \
			QAT_COMN_PTR_TYPE_MASK)

#define ICP_QAT_FW_COMN_CD_FLD_TYPE_SET(flags, val) \
	QAT_FIELD_SET(flags, val, QAT_COMN_CD_FLD_TYPE_BITPOS, \
			QAT_COMN_CD_FLD_TYPE_MASK)

#define ICP_QAT_FW_COMN_NEXT_ID_BITPOS 4
#define ICP_QAT_FW_COMN_NEXT_ID_MASK 0xF0
#define ICP_QAT_FW_COMN_LWRR_ID_BITPOS 0
#define ICP_QAT_FW_COMN_LWRR_ID_MASK 0x0F

#define ICP_QAT_FW_COMN_NEXT_ID_GET(cd_ctrl_hdr_t) \
	((((cd_ctrl_hdr_t)->next_lwrr_id) & ICP_QAT_FW_COMN_NEXT_ID_MASK) \
	>> (ICP_QAT_FW_COMN_NEXT_ID_BITPOS))

#define ICP_QAT_FW_COMN_NEXT_ID_SET(cd_ctrl_hdr_t, val) \
	{ ((cd_ctrl_hdr_t)->next_lwrr_id) = ((((cd_ctrl_hdr_t)->next_lwrr_id) \
	& ICP_QAT_FW_COMN_LWRR_ID_MASK) | \
	((val << ICP_QAT_FW_COMN_NEXT_ID_BITPOS) \
	 & ICP_QAT_FW_COMN_NEXT_ID_MASK)); }

#define ICP_QAT_FW_COMN_LWRR_ID_GET(cd_ctrl_hdr_t) \
	(((cd_ctrl_hdr_t)->next_lwrr_id) & ICP_QAT_FW_COMN_LWRR_ID_MASK)

#define ICP_QAT_FW_COMN_LWRR_ID_SET(cd_ctrl_hdr_t, val) \
	{ ((cd_ctrl_hdr_t)->next_lwrr_id) = ((((cd_ctrl_hdr_t)->next_lwrr_id) \
	& ICP_QAT_FW_COMN_NEXT_ID_MASK) | \
	((val) & ICP_QAT_FW_COMN_LWRR_ID_MASK)); }

#define ICP_QAT_FW_COMN_NEXT_ID_SET_2(next_lwrr_id, val)                       \
	do {                                                                   \
		(next_lwrr_id) =                                               \
		    (((next_lwrr_id) & ICP_QAT_FW_COMN_LWRR_ID_MASK) |         \
		     (((val) << ICP_QAT_FW_COMN_NEXT_ID_BITPOS) &              \
		      ICP_QAT_FW_COMN_NEXT_ID_MASK))                           \
	} while (0)

#define ICP_QAT_FW_COMN_LWRR_ID_SET_2(next_lwrr_id, val)                       \
	do {                                                                   \
		(next_lwrr_id) =                                               \
		    (((next_lwrr_id) & ICP_QAT_FW_COMN_NEXT_ID_MASK) |         \
		     ((val) & ICP_QAT_FW_COMN_LWRR_ID_MASK))                   \
	} while (0)

#define QAT_COMN_RESP_CRYPTO_STATUS_BITPOS 7
#define QAT_COMN_RESP_CRYPTO_STATUS_MASK 0x1
#define QAT_COMN_RESP_PKE_STATUS_BITPOS 6
#define QAT_COMN_RESP_PKE_STATUS_MASK 0x1
#define QAT_COMN_RESP_CMP_STATUS_BITPOS 5
#define QAT_COMN_RESP_CMP_STATUS_MASK 0x1
#define QAT_COMN_RESP_XLAT_STATUS_BITPOS 4
#define QAT_COMN_RESP_XLAT_STATUS_MASK 0x1
#define QAT_COMN_RESP_CMP_END_OF_LAST_BLK_BITPOS 3
#define QAT_COMN_RESP_CMP_END_OF_LAST_BLK_MASK 0x1
#define QAT_COMN_RESP_UNSUPPORTED_REQUEST_BITPOS 2
#define QAT_COMN_RESP_UNSUPPORTED_REQUEST_MASK 0x1
#define QAT_COMN_RESP_XLT_WA_APPLIED_BITPOS 0
#define QAT_COMN_RESP_XLT_WA_APPLIED_MASK 0x1

#define ICP_QAT_FW_COMN_RESP_CRYPTO_STAT_GET(status) \
	QAT_FIELD_GET(status, QAT_COMN_RESP_CRYPTO_STATUS_BITPOS, \
	QAT_COMN_RESP_CRYPTO_STATUS_MASK)

#define ICP_QAT_FW_COMN_RESP_PKE_STAT_GET(status) \
	QAT_FIELD_GET(status, QAT_COMN_RESP_PKE_STATUS_BITPOS, \
	QAT_COMN_RESP_PKE_STATUS_MASK)

#define ICP_QAT_FW_COMN_RESP_CMP_STAT_GET(status) \
	QAT_FIELD_GET(status, QAT_COMN_RESP_CMP_STATUS_BITPOS, \
	QAT_COMN_RESP_CMP_STATUS_MASK)

#define ICP_QAT_FW_COMN_RESP_XLAT_STAT_GET(status) \
	QAT_FIELD_GET(status, QAT_COMN_RESP_XLAT_STATUS_BITPOS, \
	QAT_COMN_RESP_XLAT_STATUS_MASK)

#define ICP_QAT_FW_COMN_RESP_XLT_WA_APPLIED_GET(status) \
	QAT_FIELD_GET(status, QAT_COMN_RESP_XLT_WA_APPLIED_BITPOS, \
	QAT_COMN_RESP_XLT_WA_APPLIED_MASK)

#define ICP_QAT_FW_COMN_RESP_CMP_END_OF_LAST_BLK_FLAG_GET(status) \
	QAT_FIELD_GET(status, QAT_COMN_RESP_CMP_END_OF_LAST_BLK_BITPOS, \
	QAT_COMN_RESP_CMP_END_OF_LAST_BLK_MASK)

#define ICP_QAT_FW_COMN_RESP_UNSUPPORTED_REQUEST_STAT_GET(status) \
	QAT_FIELD_GET(status, QAT_COMN_RESP_UNSUPPORTED_REQUEST_BITPOS, \
	QAT_COMN_RESP_UNSUPPORTED_REQUEST_MASK)

#define ICP_QAT_FW_COMN_STATUS_FLAG_OK 0
#define ICP_QAT_FW_COMN_STATUS_FLAG_ERROR 1
#define ICP_QAT_FW_COMN_STATUS_CMP_END_OF_LAST_BLK_FLAG_CLR 0
#define ICP_QAT_FW_COMN_STATUS_CMP_END_OF_LAST_BLK_FLAG_SET 1
#define ERR_CODE_NO_ERROR 0
#define ERR_CODE_ILWALID_BLOCK_TYPE -1
#define ERR_CODE_NO_MATCH_ONES_COMP -2
#define ERR_CODE_TOO_MANY_LEN_OR_DIS -3
#define ERR_CODE_INCOMPLETE_LEN -4
#define ERR_CODE_RPT_LEN_NO_FIRST_LEN -5
#define ERR_CODE_RPT_GT_SPEC_LEN -6
#define ERR_CODE_ILW_LIT_LEN_CODE_LEN -7
#define ERR_CODE_ILW_DIS_CODE_LEN -8
#define ERR_CODE_ILW_LIT_LEN_DIS_IN_BLK -9
#define ERR_CODE_DIS_TOO_FAR_BACK -10
#define ERR_CODE_OVERFLOW_ERROR -11
#define ERR_CODE_SOFT_ERROR -12
#define ERR_CODE_FATAL_ERROR -13
#define ERR_CODE_COMP_OUTPUT_CORRUPTION -14
#define ERR_CODE_HW_INCOMPLETE_FILE -15
#define ERR_CODE_SSM_ERROR -16
#define ERR_CODE_ENDPOINT_ERROR -17
#define ERR_CODE_CLW_ERROR -18
#define ERR_CODE_EMPTY_DYM_BLOCK -19
#define ERR_CODE_KPT_CRYPTO_SERVICE_FAIL_ILWALID_HANDLE -20
#define ERR_CODE_KPT_CRYPTO_SERVICE_FAIL_HMAC_FAILED -21
#define ERR_CODE_KPT_CRYPTO_SERVICE_FAIL_ILWALID_WRAPPING_ALGO -22
#define ERR_CODE_KPT_DRNG_SEED_NOT_LOAD -23

enum icp_qat_fw_slice {
	ICP_QAT_FW_SLICE_NULL = 0,
	ICP_QAT_FW_SLICE_CIPHER = 1,
	ICP_QAT_FW_SLICE_AUTH = 2,
	ICP_QAT_FW_SLICE_DRAM_RD = 3,
	ICP_QAT_FW_SLICE_DRAM_WR = 4,
	ICP_QAT_FW_SLICE_COMP = 5,
	ICP_QAT_FW_SLICE_XLAT = 6,
	ICP_QAT_FW_SLICE_DELIMITER
};
#endif
