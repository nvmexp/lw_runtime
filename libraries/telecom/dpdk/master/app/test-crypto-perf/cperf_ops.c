/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2016-2017 Intel Corporation
 */

#include <rte_cryptodev.h>
#include <rte_ether.h>

#include "cperf_ops.h"
#include "cperf_test_vectors.h"

#ifdef RTE_LIB_SELWRITY
static int
cperf_set_ops_selwrity(struct rte_crypto_op **ops,
		uint32_t src_buf_offset __rte_unused,
		uint32_t dst_buf_offset __rte_unused,
		uint16_t nb_ops, struct rte_cryptodev_sym_session *sess,
		const struct cperf_options *options __rte_unused,
		const struct cperf_test_vector *test_vector __rte_unused,
		uint16_t iv_offset __rte_unused, uint32_t *imix_idx)
{
	uint16_t i;

	for (i = 0; i < nb_ops; i++) {
		struct rte_crypto_sym_op *sym_op = ops[i]->sym;
		struct rte_selwrity_session *sec_sess =
			(struct rte_selwrity_session *)sess;
		uint32_t buf_sz;

		uint32_t *per_pkt_hfn = rte_crypto_op_ctod_offset(ops[i],
					uint32_t *, iv_offset);
		*per_pkt_hfn = options->pdcp_ses_hfn_en ? 0 : PDCP_DEFAULT_HFN;

		ops[i]->status = RTE_CRYPTO_OP_STATUS_NOT_PROCESSED;
		rte_selwrity_attach_session(ops[i], sec_sess);
		sym_op->m_src = (struct rte_mbuf *)((uint8_t *)ops[i] +
							src_buf_offset);

		if (options->op_type == CPERF_PDCP) {
			sym_op->m_src->buf_len = options->segment_sz;
			sym_op->m_src->data_len = options->test_buffer_size;
			sym_op->m_src->pkt_len = sym_op->m_src->data_len;
		}

		if (options->op_type == CPERF_DOCSIS) {
			if (options->imix_distribution_count) {
				buf_sz = options->imix_buffer_sizes[*imix_idx];
				*imix_idx = (*imix_idx + 1) % options->pool_sz;
			} else
				buf_sz = options->test_buffer_size;

			sym_op->m_src->buf_len = options->segment_sz;
			sym_op->m_src->data_len = buf_sz;
			sym_op->m_src->pkt_len = buf_sz;

			/* DOCSIS header is not CRC'ed */
			sym_op->auth.data.offset = options->docsis_hdr_sz;
			sym_op->auth.data.length = buf_sz -
				sym_op->auth.data.offset - RTE_ETHER_CRC_LEN;
			/*
			 * DOCSIS header and SRC and DST MAC addresses are not
			 * ciphered
			 */
			sym_op->cipher.data.offset = sym_op->auth.data.offset +
				RTE_ETHER_HDR_LEN - RTE_ETHER_TYPE_LEN;
			sym_op->cipher.data.length = buf_sz -
				sym_op->cipher.data.offset;
		}

		/* Set dest mbuf to NULL if out-of-place (dst_buf_offset = 0) */
		if (dst_buf_offset == 0)
			sym_op->m_dst = NULL;
		else
			sym_op->m_dst = (struct rte_mbuf *)((uint8_t *)ops[i] +
							dst_buf_offset);
	}

	return 0;
}
#endif

static int
cperf_set_ops_null_cipher(struct rte_crypto_op **ops,
		uint32_t src_buf_offset, uint32_t dst_buf_offset,
		uint16_t nb_ops, struct rte_cryptodev_sym_session *sess,
		const struct cperf_options *options,
		const struct cperf_test_vector *test_vector __rte_unused,
		uint16_t iv_offset __rte_unused, uint32_t *imix_idx)
{
	uint16_t i;

	for (i = 0; i < nb_ops; i++) {
		struct rte_crypto_sym_op *sym_op = ops[i]->sym;

		ops[i]->status = RTE_CRYPTO_OP_STATUS_NOT_PROCESSED;
		rte_crypto_op_attach_sym_session(ops[i], sess);

		sym_op->m_src = (struct rte_mbuf *)((uint8_t *)ops[i] +
							src_buf_offset);

		/* Set dest mbuf to NULL if out-of-place (dst_buf_offset = 0) */
		if (dst_buf_offset == 0)
			sym_op->m_dst = NULL;
		else
			sym_op->m_dst = (struct rte_mbuf *)((uint8_t *)ops[i] +
							dst_buf_offset);

		/* cipher parameters */
		if (options->imix_distribution_count) {
			sym_op->cipher.data.length =
				options->imix_buffer_sizes[*imix_idx];
			*imix_idx = (*imix_idx + 1) % options->pool_sz;
		} else
			sym_op->cipher.data.length = options->test_buffer_size;
		sym_op->cipher.data.offset = 0;
	}

	return 0;
}

static int
cperf_set_ops_null_auth(struct rte_crypto_op **ops,
		uint32_t src_buf_offset, uint32_t dst_buf_offset,
		uint16_t nb_ops, struct rte_cryptodev_sym_session *sess,
		const struct cperf_options *options,
		const struct cperf_test_vector *test_vector __rte_unused,
		uint16_t iv_offset __rte_unused, uint32_t *imix_idx)
{
	uint16_t i;

	for (i = 0; i < nb_ops; i++) {
		struct rte_crypto_sym_op *sym_op = ops[i]->sym;

		ops[i]->status = RTE_CRYPTO_OP_STATUS_NOT_PROCESSED;
		rte_crypto_op_attach_sym_session(ops[i], sess);

		sym_op->m_src = (struct rte_mbuf *)((uint8_t *)ops[i] +
							src_buf_offset);

		/* Set dest mbuf to NULL if out-of-place (dst_buf_offset = 0) */
		if (dst_buf_offset == 0)
			sym_op->m_dst = NULL;
		else
			sym_op->m_dst = (struct rte_mbuf *)((uint8_t *)ops[i] +
							dst_buf_offset);

		/* auth parameters */
		if (options->imix_distribution_count) {
			sym_op->auth.data.length =
				options->imix_buffer_sizes[*imix_idx];
			*imix_idx = (*imix_idx + 1) % options->pool_sz;
		} else
			sym_op->auth.data.length = options->test_buffer_size;
		sym_op->auth.data.offset = 0;
	}

	return 0;
}

static int
cperf_set_ops_cipher(struct rte_crypto_op **ops,
		uint32_t src_buf_offset, uint32_t dst_buf_offset,
		uint16_t nb_ops, struct rte_cryptodev_sym_session *sess,
		const struct cperf_options *options,
		const struct cperf_test_vector *test_vector,
		uint16_t iv_offset, uint32_t *imix_idx)
{
	uint16_t i;

	for (i = 0; i < nb_ops; i++) {
		struct rte_crypto_sym_op *sym_op = ops[i]->sym;

		ops[i]->status = RTE_CRYPTO_OP_STATUS_NOT_PROCESSED;
		rte_crypto_op_attach_sym_session(ops[i], sess);

		sym_op->m_src = (struct rte_mbuf *)((uint8_t *)ops[i] +
							src_buf_offset);

		/* Set dest mbuf to NULL if out-of-place (dst_buf_offset = 0) */
		if (dst_buf_offset == 0)
			sym_op->m_dst = NULL;
		else
			sym_op->m_dst = (struct rte_mbuf *)((uint8_t *)ops[i] +
							dst_buf_offset);

		/* cipher parameters */
		if (options->imix_distribution_count) {
			sym_op->cipher.data.length =
				options->imix_buffer_sizes[*imix_idx];
			*imix_idx = (*imix_idx + 1) % options->pool_sz;
		} else
			sym_op->cipher.data.length = options->test_buffer_size;

		if (options->cipher_algo == RTE_CRYPTO_CIPHER_SNOW3G_UEA2 ||
				options->cipher_algo == RTE_CRYPTO_CIPHER_KASUMI_F8 ||
				options->cipher_algo == RTE_CRYPTO_CIPHER_ZUC_EEA3)
			sym_op->cipher.data.length <<= 3;

		sym_op->cipher.data.offset = 0;
	}

	if (options->test == CPERF_TEST_TYPE_VERIFY) {
		for (i = 0; i < nb_ops; i++) {
			uint8_t *iv_ptr = rte_crypto_op_ctod_offset(ops[i],
					uint8_t *, iv_offset);

			memcpy(iv_ptr, test_vector->cipher_iv.data,
					test_vector->cipher_iv.length);

		}
	}

	return 0;
}

static int
cperf_set_ops_auth(struct rte_crypto_op **ops,
		uint32_t src_buf_offset, uint32_t dst_buf_offset,
		uint16_t nb_ops, struct rte_cryptodev_sym_session *sess,
		const struct cperf_options *options,
		const struct cperf_test_vector *test_vector,
		uint16_t iv_offset, uint32_t *imix_idx)
{
	uint16_t i;

	for (i = 0; i < nb_ops; i++) {
		struct rte_crypto_sym_op *sym_op = ops[i]->sym;

		ops[i]->status = RTE_CRYPTO_OP_STATUS_NOT_PROCESSED;
		rte_crypto_op_attach_sym_session(ops[i], sess);

		sym_op->m_src = (struct rte_mbuf *)((uint8_t *)ops[i] +
							src_buf_offset);

		/* Set dest mbuf to NULL if out-of-place (dst_buf_offset = 0) */
		if (dst_buf_offset == 0)
			sym_op->m_dst = NULL;
		else
			sym_op->m_dst = (struct rte_mbuf *)((uint8_t *)ops[i] +
							dst_buf_offset);

		if (test_vector->auth_iv.length) {
			uint8_t *iv_ptr = rte_crypto_op_ctod_offset(ops[i],
								uint8_t *,
								iv_offset);
			memcpy(iv_ptr, test_vector->auth_iv.data,
					test_vector->auth_iv.length);
		}

		/* authentication parameters */
		if (options->auth_op == RTE_CRYPTO_AUTH_OP_VERIFY) {
			sym_op->auth.digest.data = test_vector->digest.data;
			sym_op->auth.digest.phys_addr =
					test_vector->digest.phys_addr;
		} else {

			uint32_t offset = options->test_buffer_size;
			struct rte_mbuf *buf, *tbuf;

			if (options->out_of_place) {
				buf = sym_op->m_dst;
			} else {
				tbuf = sym_op->m_src;
				while ((tbuf->next != NULL) &&
						(offset >= tbuf->data_len)) {
					offset -= tbuf->data_len;
					tbuf = tbuf->next;
				}
				/*
				 * If there is not enough room in segment,
				 * place the digest in the next segment
				 */
				if ((tbuf->data_len - offset) < options->digest_sz) {
					tbuf = tbuf->next;
					offset = 0;
				}
				buf = tbuf;
			}

			sym_op->auth.digest.data = rte_pktmbuf_mtod_offset(buf,
					uint8_t *, offset);
			sym_op->auth.digest.phys_addr =
					rte_pktmbuf_iova_offset(buf, offset);

		}

		if (options->imix_distribution_count) {
			sym_op->auth.data.length =
				options->imix_buffer_sizes[*imix_idx];
			*imix_idx = (*imix_idx + 1) % options->pool_sz;
		} else
			sym_op->auth.data.length = options->test_buffer_size;

		if (options->auth_algo == RTE_CRYPTO_AUTH_SNOW3G_UIA2 ||
				options->auth_algo == RTE_CRYPTO_AUTH_KASUMI_F9 ||
				options->auth_algo == RTE_CRYPTO_AUTH_ZUC_EIA3)
			sym_op->auth.data.length <<= 3;

		sym_op->auth.data.offset = 0;
	}

	if (options->test == CPERF_TEST_TYPE_VERIFY) {
		if (test_vector->auth_iv.length) {
			for (i = 0; i < nb_ops; i++) {
				uint8_t *iv_ptr = rte_crypto_op_ctod_offset(ops[i],
						uint8_t *, iv_offset);

				memcpy(iv_ptr, test_vector->auth_iv.data,
						test_vector->auth_iv.length);
			}
		}
	}
	return 0;
}

static int
cperf_set_ops_cipher_auth(struct rte_crypto_op **ops,
		uint32_t src_buf_offset, uint32_t dst_buf_offset,
		uint16_t nb_ops, struct rte_cryptodev_sym_session *sess,
		const struct cperf_options *options,
		const struct cperf_test_vector *test_vector,
		uint16_t iv_offset, uint32_t *imix_idx)
{
	uint16_t i;

	for (i = 0; i < nb_ops; i++) {
		struct rte_crypto_sym_op *sym_op = ops[i]->sym;

		ops[i]->status = RTE_CRYPTO_OP_STATUS_NOT_PROCESSED;
		rte_crypto_op_attach_sym_session(ops[i], sess);

		sym_op->m_src = (struct rte_mbuf *)((uint8_t *)ops[i] +
							src_buf_offset);

		/* Set dest mbuf to NULL if out-of-place (dst_buf_offset = 0) */
		if (dst_buf_offset == 0)
			sym_op->m_dst = NULL;
		else
			sym_op->m_dst = (struct rte_mbuf *)((uint8_t *)ops[i] +
							dst_buf_offset);

		/* cipher parameters */
		if (options->imix_distribution_count) {
			sym_op->cipher.data.length =
				options->imix_buffer_sizes[*imix_idx];
			*imix_idx = (*imix_idx + 1) % options->pool_sz;
		} else
			sym_op->cipher.data.length = options->test_buffer_size;

		if (options->cipher_algo == RTE_CRYPTO_CIPHER_SNOW3G_UEA2 ||
				options->cipher_algo == RTE_CRYPTO_CIPHER_KASUMI_F8 ||
				options->cipher_algo == RTE_CRYPTO_CIPHER_ZUC_EEA3)
			sym_op->cipher.data.length <<= 3;

		sym_op->cipher.data.offset = 0;

		/* authentication parameters */
		if (options->auth_op == RTE_CRYPTO_AUTH_OP_VERIFY) {
			sym_op->auth.digest.data = test_vector->digest.data;
			sym_op->auth.digest.phys_addr =
					test_vector->digest.phys_addr;
		} else {

			uint32_t offset = options->test_buffer_size;
			struct rte_mbuf *buf, *tbuf;

			if (options->out_of_place) {
				buf = sym_op->m_dst;
			} else {
				tbuf = sym_op->m_src;
				while ((tbuf->next != NULL) &&
						(offset >= tbuf->data_len)) {
					offset -= tbuf->data_len;
					tbuf = tbuf->next;
				}
				/*
				 * If there is not enough room in segment,
				 * place the digest in the next segment
				 */
				if ((tbuf->data_len - offset) < options->digest_sz) {
					tbuf = tbuf->next;
					offset = 0;
				}
				buf = tbuf;
			}

			sym_op->auth.digest.data = rte_pktmbuf_mtod_offset(buf,
					uint8_t *, offset);
			sym_op->auth.digest.phys_addr =
					rte_pktmbuf_iova_offset(buf, offset);
		}

		if (options->imix_distribution_count) {
			sym_op->auth.data.length =
				options->imix_buffer_sizes[*imix_idx];
			*imix_idx = (*imix_idx + 1) % options->pool_sz;
		} else
			sym_op->auth.data.length = options->test_buffer_size;

		if (options->auth_algo == RTE_CRYPTO_AUTH_SNOW3G_UIA2 ||
				options->auth_algo == RTE_CRYPTO_AUTH_KASUMI_F9 ||
				options->auth_algo == RTE_CRYPTO_AUTH_ZUC_EIA3)
			sym_op->auth.data.length <<= 3;

		sym_op->auth.data.offset = 0;
	}

	if (options->test == CPERF_TEST_TYPE_VERIFY) {
		for (i = 0; i < nb_ops; i++) {
			uint8_t *iv_ptr = rte_crypto_op_ctod_offset(ops[i],
					uint8_t *, iv_offset);

			memcpy(iv_ptr, test_vector->cipher_iv.data,
					test_vector->cipher_iv.length);
			if (test_vector->auth_iv.length) {
				/*
				 * Copy IV after the crypto operation and
				 * the cipher IV
				 */
				iv_ptr += test_vector->cipher_iv.length;
				memcpy(iv_ptr, test_vector->auth_iv.data,
						test_vector->auth_iv.length);
			}
		}

	}

	return 0;
}

static int
cperf_set_ops_aead(struct rte_crypto_op **ops,
		uint32_t src_buf_offset, uint32_t dst_buf_offset,
		uint16_t nb_ops, struct rte_cryptodev_sym_session *sess,
		const struct cperf_options *options,
		const struct cperf_test_vector *test_vector,
		uint16_t iv_offset, uint32_t *imix_idx)
{
	uint16_t i;
	/* AAD is placed after the IV */
	uint16_t aad_offset = iv_offset +
			RTE_ALIGN_CEIL(test_vector->aead_iv.length, 16);

	for (i = 0; i < nb_ops; i++) {
		struct rte_crypto_sym_op *sym_op = ops[i]->sym;

		ops[i]->status = RTE_CRYPTO_OP_STATUS_NOT_PROCESSED;
		rte_crypto_op_attach_sym_session(ops[i], sess);

		sym_op->m_src = (struct rte_mbuf *)((uint8_t *)ops[i] +
							src_buf_offset);

		/* Set dest mbuf to NULL if out-of-place (dst_buf_offset = 0) */
		if (dst_buf_offset == 0)
			sym_op->m_dst = NULL;
		else
			sym_op->m_dst = (struct rte_mbuf *)((uint8_t *)ops[i] +
							dst_buf_offset);

		/* AEAD parameters */
		if (options->imix_distribution_count) {
			sym_op->aead.data.length =
				options->imix_buffer_sizes[*imix_idx];
			*imix_idx = (*imix_idx + 1) % options->pool_sz;
		} else
			sym_op->aead.data.length = options->test_buffer_size;
		sym_op->aead.data.offset = 0;

		sym_op->aead.aad.data = rte_crypto_op_ctod_offset(ops[i],
					uint8_t *, aad_offset);
		sym_op->aead.aad.phys_addr = rte_crypto_op_ctophys_offset(ops[i],
					aad_offset);

		if (options->aead_op == RTE_CRYPTO_AEAD_OP_DECRYPT) {
			sym_op->aead.digest.data = test_vector->digest.data;
			sym_op->aead.digest.phys_addr =
					test_vector->digest.phys_addr;
		} else {

			uint32_t offset = sym_op->aead.data.length +
						sym_op->aead.data.offset;
			struct rte_mbuf *buf, *tbuf;

			if (options->out_of_place) {
				buf = sym_op->m_dst;
			} else {
				tbuf = sym_op->m_src;
				while ((tbuf->next != NULL) &&
						(offset >= tbuf->data_len)) {
					offset -= tbuf->data_len;
					tbuf = tbuf->next;
				}
				/*
				 * If there is not enough room in segment,
				 * place the digest in the next segment
				 */
				if ((tbuf->data_len - offset) < options->digest_sz) {
					tbuf = tbuf->next;
					offset = 0;
				}
				buf = tbuf;
			}

			sym_op->aead.digest.data = rte_pktmbuf_mtod_offset(buf,
					uint8_t *, offset);
			sym_op->aead.digest.phys_addr =
					rte_pktmbuf_iova_offset(buf, offset);
		}
	}

	if (options->test == CPERF_TEST_TYPE_VERIFY) {
		for (i = 0; i < nb_ops; i++) {
			uint8_t *iv_ptr = rte_crypto_op_ctod_offset(ops[i],
					uint8_t *, iv_offset);

			/*
			 * If doing AES-CCM, nonce is copied one byte
			 * after the start of IV field, and AAD is copied
			 * 18 bytes after the start of the AAD field.
			 */
			if (options->aead_algo == RTE_CRYPTO_AEAD_AES_CCM) {
				memcpy(iv_ptr + 1, test_vector->aead_iv.data,
					test_vector->aead_iv.length);

				memcpy(ops[i]->sym->aead.aad.data + 18,
					test_vector->aad.data,
					test_vector->aad.length);
			} else {
				memcpy(iv_ptr, test_vector->aead_iv.data,
					test_vector->aead_iv.length);

				memcpy(ops[i]->sym->aead.aad.data,
					test_vector->aad.data,
					test_vector->aad.length);
			}
		}
	}

	return 0;
}

static struct rte_cryptodev_sym_session *
cperf_create_session(struct rte_mempool *sess_mp,
	struct rte_mempool *priv_mp,
	uint8_t dev_id,
	const struct cperf_options *options,
	const struct cperf_test_vector *test_vector,
	uint16_t iv_offset)
{
	struct rte_crypto_sym_xform cipher_xform;
	struct rte_crypto_sym_xform auth_xform;
	struct rte_crypto_sym_xform aead_xform;
	struct rte_cryptodev_sym_session *sess = NULL;

#ifdef RTE_LIB_SELWRITY
	/*
	 * security only
	 */
	if (options->op_type == CPERF_PDCP) {
		/* Setup Cipher Parameters */
		cipher_xform.type = RTE_CRYPTO_SYM_XFORM_CIPHER;
		cipher_xform.next = NULL;
		cipher_xform.cipher.algo = options->cipher_algo;
		cipher_xform.cipher.op = options->cipher_op;
		cipher_xform.cipher.iv.offset = iv_offset;
		cipher_xform.cipher.iv.length = 4;

		/* cipher different than null */
		if (options->cipher_algo != RTE_CRYPTO_CIPHER_NULL) {
			cipher_xform.cipher.key.data = test_vector->cipher_key.data;
			cipher_xform.cipher.key.length = test_vector->cipher_key.length;
		} else {
			cipher_xform.cipher.key.data = NULL;
			cipher_xform.cipher.key.length = 0;
		}

		/* Setup Auth Parameters */
		if (options->auth_algo != 0) {
			auth_xform.type = RTE_CRYPTO_SYM_XFORM_AUTH;
			auth_xform.next = NULL;
			auth_xform.auth.algo = options->auth_algo;
			auth_xform.auth.op = options->auth_op;
			auth_xform.auth.iv.offset = iv_offset +
				cipher_xform.cipher.iv.length;

			/* auth different than null */
			if (options->auth_algo != RTE_CRYPTO_AUTH_NULL) {
				auth_xform.auth.digest_length = options->digest_sz;
				auth_xform.auth.key.length = test_vector->auth_key.length;
				auth_xform.auth.key.data = test_vector->auth_key.data;
				auth_xform.auth.iv.length = test_vector->auth_iv.length;
			} else {
				auth_xform.auth.digest_length = 0;
				auth_xform.auth.key.length = 0;
				auth_xform.auth.key.data = NULL;
				auth_xform.auth.iv.length = 0;
			}

			cipher_xform.next = &auth_xform;
		} else {
			cipher_xform.next = NULL;
		}

		struct rte_selwrity_session_conf sess_conf = {
			.action_type = RTE_SELWRITY_ACTION_TYPE_LOOKASIDE_PROTOCOL,
			.protocol = RTE_SELWRITY_PROTOCOL_PDCP,
			{.pdcp = {
				.bearer = 0x16,
				.domain = options->pdcp_domain,
				.pkt_dir = 0,
				.sn_size = options->pdcp_sn_sz,
				.hfn = options->pdcp_ses_hfn_en ?
					PDCP_DEFAULT_HFN : 0,
				.hfn_threshold = 0x70C0A,
				.hfn_ovrd = !(options->pdcp_ses_hfn_en),
			} },
			.crypto_xform = &cipher_xform
		};

		struct rte_selwrity_ctx *ctx = (struct rte_selwrity_ctx *)
					rte_cryptodev_get_sec_ctx(dev_id);

		/* Create security session */
		return (void *)rte_selwrity_session_create(ctx,
					&sess_conf, sess_mp, priv_mp);
	}
	if (options->op_type == CPERF_DOCSIS) {
		enum rte_selwrity_docsis_direction direction;

		cipher_xform.type = RTE_CRYPTO_SYM_XFORM_CIPHER;
		cipher_xform.next = NULL;
		cipher_xform.cipher.algo = options->cipher_algo;
		cipher_xform.cipher.op = options->cipher_op;
		cipher_xform.cipher.iv.offset = iv_offset;
		if (options->cipher_algo != RTE_CRYPTO_CIPHER_NULL) {
			cipher_xform.cipher.key.data =
				test_vector->cipher_key.data;
			cipher_xform.cipher.key.length =
				test_vector->cipher_key.length;
			cipher_xform.cipher.iv.length =
				test_vector->cipher_iv.length;
		} else {
			cipher_xform.cipher.key.data = NULL;
			cipher_xform.cipher.key.length = 0;
			cipher_xform.cipher.iv.length = 0;
		}
		cipher_xform.next = NULL;

		if (options->cipher_op == RTE_CRYPTO_CIPHER_OP_ENCRYPT)
			direction = RTE_SELWRITY_DOCSIS_DOWNLINK;
		else
			direction = RTE_SELWRITY_DOCSIS_UPLINK;

		struct rte_selwrity_session_conf sess_conf = {
			.action_type =
				RTE_SELWRITY_ACTION_TYPE_LOOKASIDE_PROTOCOL,
			.protocol = RTE_SELWRITY_PROTOCOL_DOCSIS,
			{.docsis = {
				.direction = direction,
			} },
			.crypto_xform = &cipher_xform
		};
		struct rte_selwrity_ctx *ctx = (struct rte_selwrity_ctx *)
					rte_cryptodev_get_sec_ctx(dev_id);

		/* Create security session */
		return (void *)rte_selwrity_session_create(ctx,
					&sess_conf, sess_mp, priv_mp);
	}
#endif
	sess = rte_cryptodev_sym_session_create(sess_mp);
	/*
	 * cipher only
	 */
	if (options->op_type == CPERF_CIPHER_ONLY) {
		cipher_xform.type = RTE_CRYPTO_SYM_XFORM_CIPHER;
		cipher_xform.next = NULL;
		cipher_xform.cipher.algo = options->cipher_algo;
		cipher_xform.cipher.op = options->cipher_op;
		cipher_xform.cipher.iv.offset = iv_offset;

		/* cipher different than null */
		if (options->cipher_algo != RTE_CRYPTO_CIPHER_NULL) {
			cipher_xform.cipher.key.data =
					test_vector->cipher_key.data;
			cipher_xform.cipher.key.length =
					test_vector->cipher_key.length;
			cipher_xform.cipher.iv.length =
					test_vector->cipher_iv.length;
		} else {
			cipher_xform.cipher.key.data = NULL;
			cipher_xform.cipher.key.length = 0;
			cipher_xform.cipher.iv.length = 0;
		}
		/* create crypto session */
		rte_cryptodev_sym_session_init(dev_id, sess, &cipher_xform,
				priv_mp);
	/*
	 *  auth only
	 */
	} else if (options->op_type == CPERF_AUTH_ONLY) {
		auth_xform.type = RTE_CRYPTO_SYM_XFORM_AUTH;
		auth_xform.next = NULL;
		auth_xform.auth.algo = options->auth_algo;
		auth_xform.auth.op = options->auth_op;
		auth_xform.auth.iv.offset = iv_offset;

		/* auth different than null */
		if (options->auth_algo != RTE_CRYPTO_AUTH_NULL) {
			auth_xform.auth.digest_length =
					options->digest_sz;
			auth_xform.auth.key.length =
					test_vector->auth_key.length;
			auth_xform.auth.key.data = test_vector->auth_key.data;
			auth_xform.auth.iv.length =
					test_vector->auth_iv.length;
		} else {
			auth_xform.auth.digest_length = 0;
			auth_xform.auth.key.length = 0;
			auth_xform.auth.key.data = NULL;
			auth_xform.auth.iv.length = 0;
		}
		/* create crypto session */
		rte_cryptodev_sym_session_init(dev_id, sess, &auth_xform,
				priv_mp);
	/*
	 * cipher and auth
	 */
	} else if (options->op_type == CPERF_CIPHER_THEN_AUTH
			|| options->op_type == CPERF_AUTH_THEN_CIPHER) {
		/*
		 * cipher
		 */
		cipher_xform.type = RTE_CRYPTO_SYM_XFORM_CIPHER;
		cipher_xform.next = NULL;
		cipher_xform.cipher.algo = options->cipher_algo;
		cipher_xform.cipher.op = options->cipher_op;
		cipher_xform.cipher.iv.offset = iv_offset;

		/* cipher different than null */
		if (options->cipher_algo != RTE_CRYPTO_CIPHER_NULL) {
			cipher_xform.cipher.key.data =
					test_vector->cipher_key.data;
			cipher_xform.cipher.key.length =
					test_vector->cipher_key.length;
			cipher_xform.cipher.iv.length =
					test_vector->cipher_iv.length;
		} else {
			cipher_xform.cipher.key.data = NULL;
			cipher_xform.cipher.key.length = 0;
			cipher_xform.cipher.iv.length = 0;
		}

		/*
		 * auth
		 */
		auth_xform.type = RTE_CRYPTO_SYM_XFORM_AUTH;
		auth_xform.next = NULL;
		auth_xform.auth.algo = options->auth_algo;
		auth_xform.auth.op = options->auth_op;
		auth_xform.auth.iv.offset = iv_offset +
			cipher_xform.cipher.iv.length;

		/* auth different than null */
		if (options->auth_algo != RTE_CRYPTO_AUTH_NULL) {
			auth_xform.auth.digest_length = options->digest_sz;
			auth_xform.auth.iv.length = test_vector->auth_iv.length;
			auth_xform.auth.key.length =
					test_vector->auth_key.length;
			auth_xform.auth.key.data =
					test_vector->auth_key.data;
		} else {
			auth_xform.auth.digest_length = 0;
			auth_xform.auth.key.length = 0;
			auth_xform.auth.key.data = NULL;
			auth_xform.auth.iv.length = 0;
		}

		/* cipher then auth */
		if (options->op_type == CPERF_CIPHER_THEN_AUTH) {
			cipher_xform.next = &auth_xform;
			/* create crypto session */
			rte_cryptodev_sym_session_init(dev_id,
					sess, &cipher_xform, priv_mp);
		} else { /* auth then cipher */
			auth_xform.next = &cipher_xform;
			/* create crypto session */
			rte_cryptodev_sym_session_init(dev_id,
					sess, &auth_xform, priv_mp);
		}
	} else { /* options->op_type == CPERF_AEAD */
		aead_xform.type = RTE_CRYPTO_SYM_XFORM_AEAD;
		aead_xform.next = NULL;
		aead_xform.aead.algo = options->aead_algo;
		aead_xform.aead.op = options->aead_op;
		aead_xform.aead.iv.offset = iv_offset;

		aead_xform.aead.key.data =
					test_vector->aead_key.data;
		aead_xform.aead.key.length =
					test_vector->aead_key.length;
		aead_xform.aead.iv.length = test_vector->aead_iv.length;

		aead_xform.aead.digest_length = options->digest_sz;
		aead_xform.aead.aad_length =
					options->aead_aad_sz;

		/* Create crypto session */
		rte_cryptodev_sym_session_init(dev_id,
					sess, &aead_xform, priv_mp);
	}

	return sess;
}

int
cperf_get_op_functions(const struct cperf_options *options,
		struct cperf_op_fns *op_fns)
{
	memset(op_fns, 0, sizeof(struct cperf_op_fns));

	op_fns->sess_create = cperf_create_session;

	if (options->op_type == CPERF_AEAD) {
		op_fns->populate_ops = cperf_set_ops_aead;
		return 0;
	}

	if (options->op_type == CPERF_AUTH_THEN_CIPHER
			|| options->op_type == CPERF_CIPHER_THEN_AUTH) {
		op_fns->populate_ops = cperf_set_ops_cipher_auth;
		return 0;
	}
	if (options->op_type == CPERF_AUTH_ONLY) {
		if (options->auth_algo == RTE_CRYPTO_AUTH_NULL)
			op_fns->populate_ops = cperf_set_ops_null_auth;
		else
			op_fns->populate_ops = cperf_set_ops_auth;
		return 0;
	}
	if (options->op_type == CPERF_CIPHER_ONLY) {
		if (options->cipher_algo == RTE_CRYPTO_CIPHER_NULL)
			op_fns->populate_ops = cperf_set_ops_null_cipher;
		else
			op_fns->populate_ops = cperf_set_ops_cipher;
		return 0;
	}
#ifdef RTE_LIB_SELWRITY
	if (options->op_type == CPERF_PDCP) {
		op_fns->populate_ops = cperf_set_ops_selwrity;
		return 0;
	}
	if (options->op_type == CPERF_DOCSIS) {
		op_fns->populate_ops = cperf_set_ops_selwrity;
		return 0;
	}
#endif
	return -1;
}
