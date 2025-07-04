/* SPDX-License-Identifier: (BSD-3-Clause OR GPL-2.0)
 * Copyright(c) 2015-2018 Intel Corporation
 */
#ifndef _ICP_QAT_HW_H_
#define _ICP_QAT_HW_H_

enum icp_qat_hw_ae_id {
	ICP_QAT_HW_AE_0 = 0,
	ICP_QAT_HW_AE_1 = 1,
	ICP_QAT_HW_AE_2 = 2,
	ICP_QAT_HW_AE_3 = 3,
	ICP_QAT_HW_AE_4 = 4,
	ICP_QAT_HW_AE_5 = 5,
	ICP_QAT_HW_AE_6 = 6,
	ICP_QAT_HW_AE_7 = 7,
	ICP_QAT_HW_AE_8 = 8,
	ICP_QAT_HW_AE_9 = 9,
	ICP_QAT_HW_AE_10 = 10,
	ICP_QAT_HW_AE_11 = 11,
	ICP_QAT_HW_AE_DELIMITER = 12
};

enum icp_qat_hw_qat_id {
	ICP_QAT_HW_QAT_0 = 0,
	ICP_QAT_HW_QAT_1 = 1,
	ICP_QAT_HW_QAT_2 = 2,
	ICP_QAT_HW_QAT_3 = 3,
	ICP_QAT_HW_QAT_4 = 4,
	ICP_QAT_HW_QAT_5 = 5,
	ICP_QAT_HW_QAT_DELIMITER = 6
};

enum icp_qat_hw_auth_algo {
	ICP_QAT_HW_AUTH_ALGO_NULL = 0,
	ICP_QAT_HW_AUTH_ALGO_SHA1 = 1,
	ICP_QAT_HW_AUTH_ALGO_MD5 = 2,
	ICP_QAT_HW_AUTH_ALGO_SHA224 = 3,
	ICP_QAT_HW_AUTH_ALGO_SHA256 = 4,
	ICP_QAT_HW_AUTH_ALGO_SHA384 = 5,
	ICP_QAT_HW_AUTH_ALGO_SHA512 = 6,
	ICP_QAT_HW_AUTH_ALGO_AES_XCBC_MAC = 7,
	ICP_QAT_HW_AUTH_ALGO_AES_CBC_MAC = 8,
	ICP_QAT_HW_AUTH_ALGO_AES_F9 = 9,
	ICP_QAT_HW_AUTH_ALGO_GALOIS_128 = 10,
	ICP_QAT_HW_AUTH_ALGO_GALOIS_64 = 11,
	ICP_QAT_HW_AUTH_ALGO_KASUMI_F9 = 12,
	ICP_QAT_HW_AUTH_ALGO_SNOW_3G_UIA2 = 13,
	ICP_QAT_HW_AUTH_ALGO_ZUC_3G_128_EIA3 = 14,
	ICP_QAT_HW_AUTH_RESERVED_1 = 15,
	ICP_QAT_HW_AUTH_RESERVED_2 = 16,
	ICP_QAT_HW_AUTH_ALGO_SHA3_256 = 17,
	ICP_QAT_HW_AUTH_RESERVED_3 = 18,
	ICP_QAT_HW_AUTH_ALGO_SHA3_512 = 19,
	ICP_QAT_HW_AUTH_ALGO_DELIMITER = 20
};

enum icp_qat_hw_auth_mode {
	ICP_QAT_HW_AUTH_MODE0 = 0,
	ICP_QAT_HW_AUTH_MODE1 = 1,
	ICP_QAT_HW_AUTH_MODE2 = 2,
	ICP_QAT_HW_AUTH_MODE_DELIMITER = 3
};

struct icp_qat_hw_auth_config {
	uint32_t config;
	uint32_t reserved;
};

#define QAT_AUTH_MODE_BITPOS 4
#define QAT_AUTH_MODE_MASK 0xF
#define QAT_AUTH_ALGO_BITPOS 0
#define QAT_AUTH_ALGO_MASK 0xF
#define QAT_AUTH_CMP_BITPOS 8
#define QAT_AUTH_CMP_MASK 0x7F
#define QAT_AUTH_SHA3_PADDING_DISABLE_BITPOS 16
#define QAT_AUTH_SHA3_PADDING_DISABLE_MASK 0x1
#define QAT_AUTH_SHA3_PADDING_OVERRIDE_BITPOS 17
#define QAT_AUTH_SHA3_PADDING_OVERRIDE_MASK 0x1
#define QAT_AUTH_ALGO_SHA3_BITPOS 22
#define QAT_AUTH_ALGO_SHA3_MASK 0x3
#define QAT_AUTH_SHA3_PROG_PADDING_POSTFIX_BITPOS 16
#define QAT_AUTH_SHA3_PROG_PADDING_POSTFIX_MASK 0xF
#define QAT_AUTH_SHA3_PROG_PADDING_PREFIX_BITPOS 24
#define QAT_AUTH_SHA3_PROG_PADDING_PREFIX_MASK 0xFF
#define QAT_AUTH_SHA3_HW_PADDING_ENABLE 0
#define QAT_AUTH_SHA3_HW_PADDING_DISABLE 1
#define QAT_AUTH_SHA3_PADDING_DISABLE_USE_DEFAULT 0
#define QAT_AUTH_SHA3_PADDING_OVERRIDE_USE_DEFAULT 0
#define QAT_AUTH_SHA3_PADDING_OVERRIDE_PROGRAMMABLE 1
#define QAT_AUTH_SHA3_PROG_PADDING_POSTFIX_RESERVED 0
#define QAT_AUTH_SHA3_PROG_PADDING_PREFIX_RESERVED 0

#define ICP_QAT_HW_AUTH_CONFIG_BUILD(mode, algo, cmp_len)                      \
	((((mode) & QAT_AUTH_MODE_MASK) << QAT_AUTH_MODE_BITPOS) |             \
	 (((algo) & QAT_AUTH_ALGO_MASK) << QAT_AUTH_ALGO_BITPOS) |             \
	 (((algo >> 4) & QAT_AUTH_ALGO_SHA3_MASK)                              \
			<< QAT_AUTH_ALGO_SHA3_BITPOS) |                        \
	 (((QAT_AUTH_SHA3_PADDING_DISABLE_USE_DEFAULT) &                       \
			QAT_AUTH_SHA3_PADDING_DISABLE_MASK)                    \
			<< QAT_AUTH_SHA3_PADDING_DISABLE_BITPOS) |             \
	 (((QAT_AUTH_SHA3_PADDING_OVERRIDE_USE_DEFAULT) &                      \
			QAT_AUTH_SHA3_PADDING_OVERRIDE_MASK)                   \
			<< QAT_AUTH_SHA3_PADDING_OVERRIDE_BITPOS) |            \
	 (((cmp_len) & QAT_AUTH_CMP_MASK) << QAT_AUTH_CMP_BITPOS))

#define ICP_QAT_HW_AUTH_CONFIG_BUILD_UPPER                                     \
	((((QAT_AUTH_SHA3_PROG_PADDING_POSTFIX_RESERVED) &                     \
		QAT_AUTH_SHA3_PROG_PADDING_POSTFIX_MASK)                       \
		<< QAT_AUTH_SHA3_PROG_PADDING_POSTFIX_BITPOS) |                \
	 (((QAT_AUTH_SHA3_PROG_PADDING_PREFIX_RESERVED) &                      \
		QAT_AUTH_SHA3_PROG_PADDING_PREFIX_MASK)                        \
		<< QAT_AUTH_SHA3_PROG_PADDING_PREFIX_BITPOS))

struct icp_qat_hw_auth_counter {
	uint32_t counter;
	uint32_t reserved;
};

#define QAT_AUTH_COUNT_MASK 0xFFFFFFFF
#define QAT_AUTH_COUNT_BITPOS 0
#define ICP_QAT_HW_AUTH_COUNT_BUILD(val) \
	(((val) & QAT_AUTH_COUNT_MASK) << QAT_AUTH_COUNT_BITPOS)

struct icp_qat_hw_auth_setup {
	struct icp_qat_hw_auth_config auth_config;
	struct icp_qat_hw_auth_counter auth_counter;
};

#define QAT_HW_DEFAULT_ALIGNMENT 8
#define QAT_HW_ROUND_UP(val, n) (((val) + ((n) - 1)) & (~(n - 1)))
#define ICP_QAT_HW_NULL_STATE1_SZ 32
#define ICP_QAT_HW_MD5_STATE1_SZ 16
#define ICP_QAT_HW_SHA1_STATE1_SZ 20
#define ICP_QAT_HW_SHA224_STATE1_SZ 32
#define ICP_QAT_HW_SHA3_224_STATE1_SZ 28
#define ICP_QAT_HW_SHA256_STATE1_SZ 32
#define ICP_QAT_HW_SHA3_256_STATE1_SZ 32
#define ICP_QAT_HW_SHA384_STATE1_SZ 64
#define ICP_QAT_HW_SHA3_384_STATE1_SZ 48
#define ICP_QAT_HW_SHA512_STATE1_SZ 64
#define ICP_QAT_HW_SHA3_512_STATE1_SZ 64
#define ICP_QAT_HW_AES_XCBC_MAC_STATE1_SZ 16
#define ICP_QAT_HW_AES_CBC_MAC_STATE1_SZ 16
#define ICP_QAT_HW_AES_F9_STATE1_SZ 32
#define ICP_QAT_HW_KASUMI_F9_STATE1_SZ 16
#define ICP_QAT_HW_GALOIS_128_STATE1_SZ 16
#define ICP_QAT_HW_SNOW_3G_UIA2_STATE1_SZ 8
#define ICP_QAT_HW_ZUC_3G_EIA3_STATE1_SZ 8

#define ICP_QAT_HW_NULL_STATE2_SZ 32
#define ICP_QAT_HW_MD5_STATE2_SZ 16
#define ICP_QAT_HW_SHA1_STATE2_SZ 20
#define ICP_QAT_HW_SHA224_STATE2_SZ 32
#define ICP_QAT_HW_SHA3_224_STATE2_SZ 0
#define ICP_QAT_HW_SHA256_STATE2_SZ 32
#define ICP_QAT_HW_SHA3_256_STATE2_SZ 0
#define ICP_QAT_HW_SHA384_STATE2_SZ 64
#define ICP_QAT_HW_SHA3_384_STATE2_SZ 0
#define ICP_QAT_HW_SHA512_STATE2_SZ 64
#define ICP_QAT_HW_SHA3_512_STATE2_SZ 0
#define ICP_QAT_HW_AES_XCBC_MAC_STATE2_SZ 48
#define ICP_QAT_HW_AES_XCBC_MAC_KEY_SZ 16
#define ICP_QAT_HW_AES_CBC_MAC_KEY_SZ 16
#define ICP_QAT_HW_AES_CCM_CBC_E_CTR0_SZ 16
#define ICP_QAT_HW_F9_IK_SZ 16
#define ICP_QAT_HW_F9_FK_SZ 16
#define ICP_QAT_HW_KASUMI_F9_STATE2_SZ (ICP_QAT_HW_F9_IK_SZ + \
	ICP_QAT_HW_F9_FK_SZ)
#define ICP_QAT_HW_AES_F9_STATE2_SZ ICP_QAT_HW_KASUMI_F9_STATE2_SZ
#define ICP_QAT_HW_SNOW_3G_UIA2_STATE2_SZ 24
#define ICP_QAT_HW_ZUC_3G_EIA3_STATE2_SZ 32
#define ICP_QAT_HW_GALOIS_H_SZ 16
#define ICP_QAT_HW_GALOIS_LEN_A_SZ 8
#define ICP_QAT_HW_GALOIS_E_CTR0_SZ 16

struct icp_qat_hw_auth_sha512 {
	struct icp_qat_hw_auth_setup inner_setup;
	uint8_t state1[ICP_QAT_HW_SHA512_STATE1_SZ];
	struct icp_qat_hw_auth_setup outer_setup;
	uint8_t state2[ICP_QAT_HW_SHA512_STATE2_SZ];
};

struct icp_qat_hw_auth_sha3_512 {
	struct icp_qat_hw_auth_setup inner_setup;
	uint8_t state1[ICP_QAT_HW_SHA3_512_STATE1_SZ];
	struct icp_qat_hw_auth_setup outer_setup;
};

struct icp_qat_hw_auth_algo_blk {
	struct icp_qat_hw_auth_sha512 sha;
};

#define ICP_QAT_HW_GALOIS_LEN_A_BITPOS 0
#define ICP_QAT_HW_GALOIS_LEN_A_MASK 0xFFFFFFFF

enum icp_qat_hw_cipher_algo {
	ICP_QAT_HW_CIPHER_ALGO_NULL = 0,
	ICP_QAT_HW_CIPHER_ALGO_DES = 1,
	ICP_QAT_HW_CIPHER_ALGO_3DES = 2,
	ICP_QAT_HW_CIPHER_ALGO_AES128 = 3,
	ICP_QAT_HW_CIPHER_ALGO_AES192 = 4,
	ICP_QAT_HW_CIPHER_ALGO_AES256 = 5,
	ICP_QAT_HW_CIPHER_ALGO_ARC4 = 6,
	ICP_QAT_HW_CIPHER_ALGO_KASUMI = 7,
	ICP_QAT_HW_CIPHER_ALGO_SNOW_3G_UEA2 = 8,
	ICP_QAT_HW_CIPHER_ALGO_ZUC_3G_128_EEA3 = 9,
	ICP_QAT_HW_CIPHER_ALGO_SM4 = 10,
	ICP_QAT_HW_CIPHER_ALGO_CHACHA20_POLY1305 = 11,
	ICP_QAT_HW_CIPHER_DELIMITER = 12
};

enum icp_qat_hw_cipher_mode {
	ICP_QAT_HW_CIPHER_ECB_MODE = 0,
	ICP_QAT_HW_CIPHER_CBC_MODE = 1,
	ICP_QAT_HW_CIPHER_CTR_MODE = 2,
	ICP_QAT_HW_CIPHER_F8_MODE = 3,
	ICP_QAT_HW_CIPHER_AEAD_MODE = 4,
	ICP_QAT_HW_CIPHER_XTS_MODE = 6,
	ICP_QAT_HW_CIPHER_MODE_DELIMITER = 7
};

struct icp_qat_hw_cipher_config {
	uint32_t val;
	uint32_t reserved;
};

enum icp_qat_hw_cipher_dir {
	ICP_QAT_HW_CIPHER_ENCRYPT = 0,
	ICP_QAT_HW_CIPHER_DECRYPT = 1,
};

enum icp_qat_hw_auth_op {
	ICP_QAT_HW_AUTH_VERIFY = 0,
	ICP_QAT_HW_AUTH_GENERATE = 1,
};

enum icp_qat_hw_cipher_colwert {
	ICP_QAT_HW_CIPHER_NO_COLWERT = 0,
	ICP_QAT_HW_CIPHER_KEY_COLWERT = 1,
};

#define QAT_CIPHER_MODE_BITPOS 4
#define QAT_CIPHER_MODE_MASK 0xF
#define QAT_CIPHER_ALGO_BITPOS 0
#define QAT_CIPHER_ALGO_MASK 0xF
#define QAT_CIPHER_COLWERT_BITPOS 9
#define QAT_CIPHER_COLWERT_MASK 0x1
#define QAT_CIPHER_DIR_BITPOS 8
#define QAT_CIPHER_DIR_MASK 0x1
#define QAT_CIPHER_AEAD_HASH_CMP_LEN_BITPOS 10
#define QAT_CIPHER_AEAD_HASH_CMP_LEN_MASK 0x1F
#define QAT_CIPHER_MODE_F8_KEY_SZ_MULT 2
#define QAT_CIPHER_MODE_XTS_KEY_SZ_MULT 2
#define ICP_QAT_HW_CIPHER_CONFIG_BUILD(mode, algo, colwert, dir) \
	(((mode & QAT_CIPHER_MODE_MASK) << QAT_CIPHER_MODE_BITPOS) | \
	((algo & QAT_CIPHER_ALGO_MASK) << QAT_CIPHER_ALGO_BITPOS) | \
	((colwert & QAT_CIPHER_COLWERT_MASK) << QAT_CIPHER_COLWERT_BITPOS) | \
	((dir & QAT_CIPHER_DIR_MASK) << QAT_CIPHER_DIR_BITPOS))

#define QAT_CIPHER_AEAD_AAD_LOWER_SHIFT 24
#define QAT_CIPHER_AEAD_AAD_UPPER_SHIFT 8
#define QAT_CIPHER_AEAD_AAD_SIZE_LOWER_MASK 0xFF
#define QAT_CIPHER_AEAD_AAD_SIZE_UPPER_MASK 0x3F
#define QAT_CIPHER_AEAD_AAD_SIZE_BITPOS 16
#define ICP_QAT_HW_CIPHER_CONFIG_BUILD_UPPER(aad_size) \
	({ \
	typeof(aad_size) aad_size1 = aad_size; \
	(((((aad_size1) >> QAT_CIPHER_AEAD_AAD_UPPER_SHIFT) & \
	QAT_CIPHER_AEAD_AAD_SIZE_UPPER_MASK) << \
	QAT_CIPHER_AEAD_AAD_SIZE_BITPOS) | \
	(((aad_size1) & QAT_CIPHER_AEAD_AAD_SIZE_LOWER_MASK) << \
	QAT_CIPHER_AEAD_AAD_LOWER_SHIFT)); \
	})

#define ICP_QAT_HW_DES_BLK_SZ 8
#define ICP_QAT_HW_3DES_BLK_SZ 8
#define ICP_QAT_HW_NULL_BLK_SZ 8
#define ICP_QAT_HW_AES_BLK_SZ 16
#define ICP_QAT_HW_KASUMI_BLK_SZ 8
#define ICP_QAT_HW_SNOW_3G_BLK_SZ 8
#define ICP_QAT_HW_ZUC_3G_BLK_SZ 8
#define ICP_QAT_HW_NULL_KEY_SZ 256
#define ICP_QAT_HW_DES_KEY_SZ 8
#define ICP_QAT_HW_3DES_KEY_SZ 24
#define ICP_QAT_HW_AES_128_KEY_SZ 16
#define ICP_QAT_HW_AES_192_KEY_SZ 24
#define ICP_QAT_HW_AES_256_KEY_SZ 32
#define ICP_QAT_HW_AES_128_F8_KEY_SZ (ICP_QAT_HW_AES_128_KEY_SZ * \
	QAT_CIPHER_MODE_F8_KEY_SZ_MULT)
#define ICP_QAT_HW_AES_192_F8_KEY_SZ (ICP_QAT_HW_AES_192_KEY_SZ * \
	QAT_CIPHER_MODE_F8_KEY_SZ_MULT)
#define ICP_QAT_HW_AES_256_F8_KEY_SZ (ICP_QAT_HW_AES_256_KEY_SZ * \
	QAT_CIPHER_MODE_F8_KEY_SZ_MULT)
#define ICP_QAT_HW_AES_128_XTS_KEY_SZ (ICP_QAT_HW_AES_128_KEY_SZ * \
	QAT_CIPHER_MODE_XTS_KEY_SZ_MULT)
#define ICP_QAT_HW_AES_256_XTS_KEY_SZ (ICP_QAT_HW_AES_256_KEY_SZ * \
	QAT_CIPHER_MODE_XTS_KEY_SZ_MULT)
#define ICP_QAT_HW_KASUMI_KEY_SZ 16
#define ICP_QAT_HW_KASUMI_F8_KEY_SZ (ICP_QAT_HW_KASUMI_KEY_SZ * \
	QAT_CIPHER_MODE_F8_KEY_SZ_MULT)
#define ICP_QAT_HW_AES_128_XTS_KEY_SZ (ICP_QAT_HW_AES_128_KEY_SZ * \
	QAT_CIPHER_MODE_XTS_KEY_SZ_MULT)
#define ICP_QAT_HW_AES_256_XTS_KEY_SZ (ICP_QAT_HW_AES_256_KEY_SZ * \
	QAT_CIPHER_MODE_XTS_KEY_SZ_MULT)
#define ICP_QAT_HW_ARC4_KEY_SZ 256
#define ICP_QAT_HW_SNOW_3G_UEA2_KEY_SZ 16
#define ICP_QAT_HW_SNOW_3G_UEA2_IV_SZ 16
#define ICP_QAT_HW_ZUC_3G_EEA3_KEY_SZ 16
#define ICP_QAT_HW_ZUC_3G_EEA3_IV_SZ 16
#define ICP_QAT_HW_MODE_F8_NUM_REG_TO_CLEAR 2
#define ICP_QAT_HW_CHACHAPOLY_KEY_SZ 32
#define ICP_QAT_HW_CHACHAPOLY_IV_SZ 12
#define ICP_QAT_HW_CHACHAPOLY_BLK_SZ 64
#define ICP_QAT_HW_SPC_CTR_SZ 16
#define ICP_QAT_HW_CHACHAPOLY_ICV_SZ 16
#define ICP_QAT_HW_CHACHAPOLY_AAD_MAX_LOG 14

#define ICP_QAT_HW_CIPHER_MAX_KEY_SZ ICP_QAT_HW_AES_256_F8_KEY_SZ

/* These defines describe position of the bit-fields
 * in the flags byte in B0
 */
#define ICP_QAT_HW_CCM_B0_FLAGS_ADATA_SHIFT      6
#define ICP_QAT_HW_CCM_B0_FLAGS_T_SHIFT          3

#define ICP_QAT_HW_CCM_BUILD_B0_FLAGS(Adata, t, q)                  \
	((((Adata) > 0 ? 1 : 0) << ICP_QAT_HW_CCM_B0_FLAGS_ADATA_SHIFT) \
	| ((((t) - 2) >> 1) << ICP_QAT_HW_CCM_B0_FLAGS_T_SHIFT) \
	| ((q) - 1))

#define ICP_QAT_HW_CCM_NQ_CONST 15
#define ICP_QAT_HW_CCM_AAD_B0_LEN 16
#define ICP_QAT_HW_CCM_AAD_LEN_INFO 2
#define ICP_QAT_HW_CCM_AAD_DATA_OFFSET (ICP_QAT_HW_CCM_AAD_B0_LEN + \
		ICP_QAT_HW_CCM_AAD_LEN_INFO)
#define ICP_QAT_HW_CCM_AAD_ALIGNMENT 16
#define ICP_QAT_HW_CCM_MSG_LEN_MAX_FIELD_SIZE 4
#define ICP_QAT_HW_CCM_NONCE_OFFSET 1

struct icp_qat_hw_cipher_algo_blk {
	struct icp_qat_hw_cipher_config cipher_config;
	uint8_t key[ICP_QAT_HW_CIPHER_MAX_KEY_SZ];
} __rte_cache_aligned;

/* ========================================================================= */
/*                COMPRESSION SLICE                                          */
/* ========================================================================= */

enum icp_qat_hw_compression_direction {
	ICP_QAT_HW_COMPRESSION_DIR_COMPRESS = 0,
	ICP_QAT_HW_COMPRESSION_DIR_DECOMPRESS = 1,
	ICP_QAT_HW_COMPRESSION_DIR_DELIMITER = 2
};

enum icp_qat_hw_compression_delayed_match {
	ICP_QAT_HW_COMPRESSION_DELAYED_MATCH_DISABLED = 0,
	ICP_QAT_HW_COMPRESSION_DELAYED_MATCH_ENABLED = 1,
	ICP_QAT_HW_COMPRESSION_DELAYED_MATCH_DELIMITER = 2
};

enum icp_qat_hw_compression_algo {
	ICP_QAT_HW_COMPRESSION_ALGO_DEFLATE = 0,
	ICP_QAT_HW_COMPRESSION_ALGO_LZS = 1,
	ICP_QAT_HW_COMPRESSION_ALGO_DELIMITER = 2
};


enum icp_qat_hw_compression_depth {
	ICP_QAT_HW_COMPRESSION_DEPTH_1 = 0,
	ICP_QAT_HW_COMPRESSION_DEPTH_4 = 1,
	ICP_QAT_HW_COMPRESSION_DEPTH_8 = 2,
	ICP_QAT_HW_COMPRESSION_DEPTH_16 = 3,
	ICP_QAT_HW_COMPRESSION_DEPTH_DELIMITER = 4
};

enum icp_qat_hw_compression_file_type {
	ICP_QAT_HW_COMPRESSION_FILE_TYPE_0 = 0,
	ICP_QAT_HW_COMPRESSION_FILE_TYPE_1 = 1,
	ICP_QAT_HW_COMPRESSION_FILE_TYPE_2 = 2,
	ICP_QAT_HW_COMPRESSION_FILE_TYPE_3 = 3,
	ICP_QAT_HW_COMPRESSION_FILE_TYPE_4 = 4,
	ICP_QAT_HW_COMPRESSION_FILE_TYPE_DELIMITER = 5
};

struct icp_qat_hw_compression_config {
	uint32_t val;
	uint32_t reserved;
};

#define QAT_COMPRESSION_DIR_BITPOS 4
#define QAT_COMPRESSION_DIR_MASK 0x7
#define QAT_COMPRESSION_DELAYED_MATCH_BITPOS 16
#define QAT_COMPRESSION_DELAYED_MATCH_MASK 0x1
#define QAT_COMPRESSION_ALGO_BITPOS 31
#define QAT_COMPRESSION_ALGO_MASK 0x1
#define QAT_COMPRESSION_DEPTH_BITPOS 28
#define QAT_COMPRESSION_DEPTH_MASK 0x7
#define QAT_COMPRESSION_FILE_TYPE_BITPOS 24
#define QAT_COMPRESSION_FILE_TYPE_MASK 0xF

#define ICP_QAT_HW_COMPRESSION_CONFIG_BUILD(                                   \
	dir, delayed, algo, depth, filetype)                                   \
	((((dir) & QAT_COMPRESSION_DIR_MASK) << QAT_COMPRESSION_DIR_BITPOS) |  \
	 (((delayed) & QAT_COMPRESSION_DELAYED_MATCH_MASK)                     \
	  << QAT_COMPRESSION_DELAYED_MATCH_BITPOS) |                           \
	 (((algo) & QAT_COMPRESSION_ALGO_MASK)                                 \
	  << QAT_COMPRESSION_ALGO_BITPOS) |                                    \
	 (((depth) & QAT_COMPRESSION_DEPTH_MASK)                               \
	  << QAT_COMPRESSION_DEPTH_BITPOS) |                                   \
	 (((filetype) & QAT_COMPRESSION_FILE_TYPE_MASK)                        \
	  << QAT_COMPRESSION_FILE_TYPE_BITPOS))

#endif
