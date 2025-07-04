/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2020 Broadcom
 * All rights reserved.
 */

#include <rte_cryptodev.h>

#include "bcmfs_sym_capabilities.h"

static const struct rte_cryptodev_capabilities bcmfs_sym_capabilities[] = {
	{
		/* SHA1 */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_SHA1,
				.block_size = 64,
				.key_size = {
					.min = 0,
					.max = 0,
					.increment = 0
				},
				.digest_size = {
					.min = 20,
					.max = 20,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* MD5 */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_MD5,
				.block_size = 64,
				.key_size = {
					.min = 0,
					.max = 0,
					.increment = 0
				},
				.digest_size = {
					.min = 16,
					.max = 16,
					.increment = 0
				},
			}, }
		}, }
	},
	{
		/* SHA224 */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_SHA224,
				.block_size = 64,
				.key_size = {
					.min = 0,
					.max = 0,
					.increment = 0
				},
				.digest_size = {
					.min = 28,
					.max = 28,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* SHA256 */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_SHA256,
				.block_size = 64,
				.key_size = {
					.min = 0,
					.max = 0,
					.increment = 0
				},
				.digest_size = {
					.min = 32,
					.max = 32,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* SHA384 */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_SHA384,
				.block_size = 64,
				.key_size = {
					.min = 0,
					.max = 0,
					.increment = 0
				},
				.digest_size = {
					.min = 48,
					.max = 48,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* SHA512 */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_SHA512,
				.block_size = 64,
				.key_size = {
					.min = 0,
					.max = 0,
					.increment = 0
				},
				.digest_size = {
					.min = 64,
					.max = 64,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* SHA3_224 */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_SHA3_224,
				.block_size = 144,
				.key_size = {
					.min = 0,
					.max = 0,
					.increment = 0
				},
				.digest_size = {
					.min = 28,
					.max = 28,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* SHA3_256 */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_SHA3_256,
				.block_size = 136,
				.key_size = {
					.min = 0,
					.max = 0,
					.increment = 0
				},
				.digest_size = {
					.min = 32,
					.max = 32,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* SHA3_384 */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_SHA3_384,
				.block_size = 104,
				.key_size = {
					.min = 0,
					.max = 0,
					.increment = 0
				},
				.digest_size = {
					.min = 48,
					.max = 48,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* SHA3_512 */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_SHA3_512,
				.block_size = 72,
				.key_size = {
					.min = 0,
					.max = 0,
					.increment = 0
				},
				.digest_size = {
					.min = 64,
					.max = 64,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* SHA1 HMAC */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_SHA1_HMAC,
				.block_size = 64,
				.key_size = {
					.min = 1,
					.max = 64,
					.increment = 1
				},
				.digest_size = {
					.min = 20,
					.max = 20,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* MD5 HMAC */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_MD5_HMAC,
				.block_size = 64,
				.key_size = {
					.min = 1,
					.max = 64,
					.increment = 1
				},
				.digest_size = {
					.min = 16,
					.max = 16,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* SHA224 HMAC */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_SHA224_HMAC,
				.block_size = 64,
				.key_size = {
					.min = 1,
					.max = 64,
					.increment = 1
				},
				.digest_size = {
					.min = 28,
					.max = 28,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* SHA256 HMAC */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_SHA256_HMAC,
				.block_size = 64,
				.key_size = {
					.min = 1,
					.max = 64,
					.increment = 1
				},
				.digest_size = {
					.min = 32,
					.max = 32,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* SHA384 HMAC */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_SHA384_HMAC,
				.block_size = 128,
				.key_size = {
					.min = 1,
					.max = 128,
					.increment = 1
				},
				.digest_size = {
					.min = 48,
					.max = 48,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* SHA512 HMAC*/
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_SHA512_HMAC,
				.block_size = 128,
				.key_size = {
					.min = 1,
					.max = 128,
					.increment = 1
				},
				.digest_size = {
					.min = 64,
					.max = 64,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* SHA3_224 HMAC */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_SHA3_224_HMAC,
				.block_size = 144,
				.key_size = {
					.min = 1,
					.max = 144,
					.increment = 1
				},
				.digest_size = {
					.min = 28,
					.max = 28,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* SHA3_256 HMAC */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_SHA3_256_HMAC,
				.block_size = 136,
				.key_size = {
					.min = 1,
					.max = 136,
					.increment = 1
				},
				.digest_size = {
					.min = 32,
					.max = 32,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* SHA3_384 HMAC */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_SHA3_384_HMAC,
				.block_size = 104,
				.key_size = {
					.min = 1,
					.max = 104,
					.increment = 1
				},
				.digest_size = {
					.min = 48,
					.max = 48,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* SHA3_512 HMAC */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_SHA3_512_HMAC,
				.block_size = 72,
				.key_size = {
					.min = 1,
					.max = 72,
					.increment = 1
				},
				.digest_size = {
					.min = 64,
					.max = 64,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* AES XCBC MAC */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_AES_XCBC_MAC,
				.block_size = 16,
				.key_size = {
					.min = 1,
					.max = 16,
					.increment = 1
				},
				.digest_size = {
					.min = 16,
					.max = 16,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* AES GMAC */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_AES_GMAC,
				.block_size = 16,
				.key_size = {
					.min = 16,
					.max = 32,
					.increment = 8
				},
				.digest_size = {
					.min = 16,
					.max = 16,
					.increment = 0
				},
				.aad_size = {
					.min = 0,
					.max = 65535,
					.increment = 1
				},
				.iv_size = {
					.min = 12,
					.max = 16,
					.increment = 4
				},
			}, }
		}, }
	},
	{
		/* AES CMAC */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_AES_CMAC,
				.block_size = 16,
				.key_size = {
					.min = 1,
					.max = 16,
					.increment = 1
				},
				.digest_size = {
					.min = 16,
					.max = 16,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* AES CBC MAC */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AUTH,
			{.auth = {
				.algo = RTE_CRYPTO_AUTH_AES_CBC_MAC,
				.block_size = 16,
				.key_size = {
					.min = 1,
					.max = 16,
					.increment = 1
				},
				.digest_size = {
					.min = 16,
					.max = 16,
					.increment = 0
				},
				.aad_size = { 0 }
			}, }
		}, }
	},
	{
		/* AES ECB */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_CIPHER,
			{.cipher = {
				.algo = RTE_CRYPTO_CIPHER_AES_ECB,
				.block_size = 16,
				.key_size = {
					.min = 16,
					.max = 32,
					.increment = 8
				},
				.iv_size = {
					.min = 0,
					.max = 0,
					.increment = 0
				}
			}, }
		}, }
	},
	{
		/* AES CBC */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_CIPHER,
			{.cipher = {
				.algo = RTE_CRYPTO_CIPHER_AES_CBC,
				.block_size = 16,
				.key_size = {
					.min = 16,
					.max = 32,
					.increment = 8
				},
				.iv_size = {
					.min = 16,
					.max = 16,
					.increment = 0
				}
			}, }
		}, }
	},
	{
		/* AES CTR */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_CIPHER,
			{.cipher = {
				.algo = RTE_CRYPTO_CIPHER_AES_CTR,
				.block_size = 16,
				.key_size = {
					.min = 16,
					.max = 32,
					.increment = 8
				},
				.iv_size = {
					.min = 16,
					.max = 16,
					.increment = 0
				}
			}, }
		}, }
	},
	{
		/* AES XTS */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_CIPHER,
			{.cipher = {
				.algo = RTE_CRYPTO_CIPHER_AES_XTS,
				.block_size = 16,
				.key_size = {
					.min = 32,
					.max = 64,
					.increment = 32
				},
				.iv_size = {
					.min = 16,
					.max = 16,
					.increment = 0
				}
			}, }
		}, }
	},
	{
		/* DES CBC */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_CIPHER,
			{.cipher = {
				.algo = RTE_CRYPTO_CIPHER_DES_CBC,
				.block_size = 8,
				.key_size = {
					.min = 8,
					.max = 8,
					.increment = 0
				},
				.iv_size = {
					.min = 16,
					.max = 16,
					.increment = 0
				}
			}, }
		}, }
	},
	{
		/* 3DES CBC */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_CIPHER,
			{.cipher = {
				.algo = RTE_CRYPTO_CIPHER_3DES_CBC,
				.block_size = 8,
				.key_size = {
					.min = 24,
					.max = 24,
					.increment = 0
				},
				.iv_size = {
					.min = 16,
					.max = 16,
					.increment = 0
				}
			}, }
		}, }
	},
	{
		/* 3DES ECB */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_CIPHER,
			{.cipher = {
				.algo = RTE_CRYPTO_CIPHER_3DES_ECB,
				.block_size = 8,
				.key_size = {
					.min = 24,
					.max = 24,
					.increment = 0
				},
				.iv_size = {
					.min = 0,
					.max = 0,
					.increment = 0
				}
			}, }
		}, }
	},
	{
		/* AES GCM */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AEAD,
			{.aead = {
				.algo = RTE_CRYPTO_AEAD_AES_GCM,
				.block_size = 16,
				.key_size = {
					.min = 16,
					.max = 32,
					.increment = 8
				},
				.digest_size = {
					.min = 16,
					.max = 16,
					.increment = 0
				},
				.aad_size = {
					.min = 0,
					.max = 65535,
					.increment = 1
				},
				.iv_size = {
					.min = 12,
					.max = 16,
					.increment = 4
				},
			}, }
		}, }
	},
	{
		/* AES CCM */
		.op = RTE_CRYPTO_OP_TYPE_SYMMETRIC,
		{.sym = {
			.xform_type = RTE_CRYPTO_SYM_XFORM_AEAD,
			{.aead = {
				.algo = RTE_CRYPTO_AEAD_AES_CCM,
				.block_size = 16,
				.key_size = {
					.min = 16,
					.max = 32,
					.increment = 8
				},
				.digest_size = {
					.min = 4,
					.max = 16,
					.increment = 2
				},
				.aad_size = {
					.min = 0,
					.max = 65535,
					.increment = 1
				},
				.iv_size = {
					.min = 7,
					.max = 13,
					.increment = 1
				},
			}, }
		}, }
	},

	RTE_CRYPTODEV_END_OF_CAPABILITIES_LIST()
};

const struct rte_cryptodev_capabilities *
bcmfs_sym_get_capabilities(void)
{
	return bcmfs_sym_capabilities;
}
