/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2018-2019 Intel Corporation
 */

#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_hexdump.h>
#include <rte_comp.h>
#include <rte_bus_pci.h>
#include <rte_byteorder.h>
#include <rte_memcpy.h>
#include <rte_common.h>
#include <rte_spinlock.h>
#include <rte_log.h>
#include <rte_malloc.h>
#include <rte_memzone.h>

#include "qat_logs.h"
#include "qat_comp.h"
#include "qat_comp_pmd.h"

static void
qat_comp_fallback_to_fixed(struct icp_qat_fw_comp_req *comp_req)
{
	QAT_DP_LOG(DEBUG, "QAT PMD: fallback to fixed compression!");

	comp_req->comn_hdr.service_cmd_id =
			ICP_QAT_FW_COMP_CMD_STATIC;

	ICP_QAT_FW_COMN_NEXT_ID_SET(
			&comp_req->comp_cd_ctrl,
			ICP_QAT_FW_SLICE_DRAM_WR);

	ICP_QAT_FW_COMN_NEXT_ID_SET(
			&comp_req->u2.xlt_cd_ctrl,
			ICP_QAT_FW_SLICE_NULL);
	ICP_QAT_FW_COMN_LWRR_ID_SET(
			&comp_req->u2.xlt_cd_ctrl,
			ICP_QAT_FW_SLICE_NULL);
}

void
qat_comp_free_split_op_memzones(struct qat_comp_op_cookie *cookie,
				unsigned int nb_children)
{
	unsigned int i;

	/* free all memzones allocated for child descriptors */
	for (i = 0; i < nb_children; i++)
		rte_memzone_free(cookie->dst_memzones[i]);

	/* and free the pointer table */
	rte_free(cookie->dst_memzones);
	cookie->dst_memzones = NULL;
}

static int
qat_comp_allocate_split_op_memzones(struct qat_comp_op_cookie *cookie,
				    unsigned int nb_descriptors_needed)
{
	struct qat_queue *txq = &(cookie->qp->tx_q);
	char dst_memz_name[RTE_MEMZONE_NAMESIZE];
	unsigned int i;

	/* allocate the array of memzone pointers */
	cookie->dst_memzones = rte_zmalloc_socket("qat PMD im buf mz pointers",
			(nb_descriptors_needed - 1) *
				sizeof(const struct rte_memzone *),
			RTE_CACHE_LINE_SIZE, cookie->socket_id);

	if (cookie->dst_memzones == NULL) {
		QAT_DP_LOG(ERR,
			"QAT PMD: failed to allocate im buf mz pointers");
		return -ENOMEM;
	}

	for (i = 0; i < nb_descriptors_needed - 1; i++) {
		snprintf(dst_memz_name,
				sizeof(dst_memz_name),
				"dst_%u_%u_%u_%u_%u",
				cookie->qp->qat_dev->qat_dev_id,
				txq->hw_bundle_number, txq->hw_queue_number,
				cookie->cookie_index, i);

		cookie->dst_memzones[i] = rte_memzone_reserve_aligned(
				dst_memz_name, RTE_PMD_QAT_COMP_IM_BUFFER_SIZE,
				cookie->socket_id, RTE_MEMZONE_IOVA_CONTIG,
				RTE_CACHE_LINE_SIZE);

		if (cookie->dst_memzones[i] == NULL) {
			QAT_DP_LOG(ERR,
				"QAT PMD: failed to allocate dst buffer memzone");

			/* let's free all memzones allocated up to now */
			qat_comp_free_split_op_memzones(cookie, i);

			return -ENOMEM;
		}
	}

	return 0;
}

int
qat_comp_build_request(void *in_op, uint8_t *out_msg,
		       void *op_cookie,
		       enum qat_device_gen qat_dev_gen __rte_unused)
{
	struct rte_comp_op *op = in_op;
	struct qat_comp_op_cookie *cookie =
			(struct qat_comp_op_cookie *)op_cookie;
	struct qat_comp_stream *stream;
	struct qat_comp_xform *qat_xform;
	const uint8_t *tmpl;
	struct icp_qat_fw_comp_req *comp_req =
	    (struct icp_qat_fw_comp_req *)out_msg;

	if (op->op_type == RTE_COMP_OP_STATEFUL) {
		stream = op->stream;
		qat_xform = &stream->qat_xform;
		if (unlikely(qat_xform->qat_comp_request_type !=
			     QAT_COMP_REQUEST_DECOMPRESS)) {
			QAT_DP_LOG(ERR, "QAT PMD does not support stateful compression");
			op->status = RTE_COMP_OP_STATUS_ILWALID_ARGS;
			return -EILWAL;
		}
		if (unlikely(stream->op_in_progress)) {
			QAT_DP_LOG(ERR, "QAT PMD does not support running multiple stateful operations on the same stream at once");
			op->status = RTE_COMP_OP_STATUS_ILWALID_STATE;
			return -EILWAL;
		}
		stream->op_in_progress = 1;
	} else {
		stream = NULL;
		qat_xform = op->private_xform;
	}
	tmpl = (uint8_t *)&qat_xform->qat_comp_req_tmpl;

	rte_mov128(out_msg, tmpl);
	comp_req->comn_mid.opaque_data = (uint64_t)(uintptr_t)op;

	if (likely(qat_xform->qat_comp_request_type ==
			QAT_COMP_REQUEST_DYNAMIC_COMP_STATELESS)) {

		if (unlikely(op->src.length > QAT_FALLBACK_THLD)) {
			/* the operation must be split into pieces */
			if (qat_xform->checksum_type !=
					RTE_COMP_CHECKSUM_NONE) {
				/* fallback to fixed compression in case any
				 * checksum callwlation was requested
				 */
				qat_comp_fallback_to_fixed(comp_req);
			} else {
				/* callwlate num. of descriptors for split op */
				unsigned int nb_descriptors_needed =
					op->src.length / QAT_FALLBACK_THLD + 1;
				/* allocate memzone for output data */
				if (qat_comp_allocate_split_op_memzones(
					       cookie, nb_descriptors_needed)) {
					/* out of memory, fallback to fixed */
					qat_comp_fallback_to_fixed(comp_req);
				} else {
					QAT_DP_LOG(DEBUG,
							"Input data is too big, op must be split into %u descriptors",
							nb_descriptors_needed);
					return (int) nb_descriptors_needed;
				}
			}
		}

		/* set BFINAL bit according to flush_flag */
		comp_req->comp_pars.req_par_flags =
			ICP_QAT_FW_COMP_REQ_PARAM_FLAGS_BUILD(
				ICP_QAT_FW_COMP_SOP,
				ICP_QAT_FW_COMP_EOP,
				op->flush_flag == RTE_COMP_FLUSH_FINAL ?
					ICP_QAT_FW_COMP_BFINAL
					: ICP_QAT_FW_COMP_NOT_BFINAL,
				ICP_QAT_FW_COMP_CLW,
				ICP_QAT_FW_COMP_CLW_RECOVERY);

	} else if (op->op_type == RTE_COMP_OP_STATEFUL) {

		comp_req->comp_pars.req_par_flags =
			ICP_QAT_FW_COMP_REQ_PARAM_FLAGS_BUILD(
				(stream->start_of_packet) ?
					ICP_QAT_FW_COMP_SOP
				      : ICP_QAT_FW_COMP_NOT_SOP,
				(op->flush_flag == RTE_COMP_FLUSH_FULL ||
				 op->flush_flag == RTE_COMP_FLUSH_FINAL) ?
					ICP_QAT_FW_COMP_EOP
				      : ICP_QAT_FW_COMP_NOT_EOP,
				ICP_QAT_FW_COMP_NOT_BFINAL,
				ICP_QAT_FW_COMP_NO_CLW,
				ICP_QAT_FW_COMP_NO_CLW_RECOVERY);
	}

	/* common for sgl and flat buffers */
	comp_req->comp_pars.comp_len = op->src.length;
	comp_req->comp_pars.out_buffer_sz = rte_pktmbuf_pkt_len(op->m_dst) -
			op->dst.offset;

	if (op->m_src->next != NULL || op->m_dst->next != NULL) {
		/* sgl */
		int ret = 0;

		ICP_QAT_FW_COMN_PTR_TYPE_SET(comp_req->comn_hdr.comn_req_flags,
				QAT_COMN_PTR_TYPE_SGL);

		if (unlikely(op->m_src->nb_segs > cookie->src_nb_elems)) {
			/* we need to allocate more elements in SGL*/
			void *tmp;

			tmp = rte_realloc_socket(cookie->qat_sgl_src_d,
					  sizeof(struct qat_sgl) +
					  sizeof(struct qat_flat_buf) *
					  op->m_src->nb_segs, 64,
					  cookie->socket_id);

			if (unlikely(tmp == NULL)) {
				QAT_DP_LOG(ERR, "QAT PMD can't allocate memory"
					   " for %d elements of SGL",
					   op->m_src->nb_segs);
				op->status = RTE_COMP_OP_STATUS_ERROR;
				/* clear op-in-progress flag */
				if (stream)
					stream->op_in_progress = 0;
				return -ENOMEM;
			}
			/* new SGL is valid now */
			cookie->qat_sgl_src_d = (struct qat_sgl *)tmp;
			cookie->src_nb_elems = op->m_src->nb_segs;
			cookie->qat_sgl_src_phys_addr =
				rte_malloc_virt2iova(cookie->qat_sgl_src_d);
		}

		ret = qat_sgl_fill_array(op->m_src,
				op->src.offset,
				cookie->qat_sgl_src_d,
				op->src.length,
				cookie->src_nb_elems);
		if (ret) {
			QAT_DP_LOG(ERR, "QAT PMD Cannot fill source sgl array");
			op->status = RTE_COMP_OP_STATUS_ILWALID_ARGS;
			/* clear op-in-progress flag */
			if (stream)
				stream->op_in_progress = 0;
			return ret;
		}

		if (unlikely(op->m_dst->nb_segs > cookie->dst_nb_elems)) {
			/* we need to allocate more elements in SGL*/
			struct qat_sgl *tmp;

			tmp = rte_realloc_socket(cookie->qat_sgl_dst_d,
					  sizeof(struct qat_sgl) +
					  sizeof(struct qat_flat_buf) *
					  op->m_dst->nb_segs, 64,
					  cookie->socket_id);

			if (unlikely(tmp == NULL)) {
				QAT_DP_LOG(ERR, "QAT PMD can't allocate memory"
					   " for %d elements of SGL",
					   op->m_dst->nb_segs);
				op->status = RTE_COMP_OP_STATUS_ERROR;
				/* clear op-in-progress flag */
				if (stream)
					stream->op_in_progress = 0;
				return -ENOMEM;
			}
			/* new SGL is valid now */
			cookie->qat_sgl_dst_d = (struct qat_sgl *)tmp;
			cookie->dst_nb_elems = op->m_dst->nb_segs;
			cookie->qat_sgl_dst_phys_addr =
				rte_malloc_virt2iova(cookie->qat_sgl_dst_d);
		}

		ret = qat_sgl_fill_array(op->m_dst,
				op->dst.offset,
				cookie->qat_sgl_dst_d,
				comp_req->comp_pars.out_buffer_sz,
				cookie->dst_nb_elems);
		if (ret) {
			QAT_DP_LOG(ERR, "QAT PMD Cannot fill dest. sgl array");
			op->status = RTE_COMP_OP_STATUS_ILWALID_ARGS;
			/* clear op-in-progress flag */
			if (stream)
				stream->op_in_progress = 0;
			return ret;
		}

		comp_req->comn_mid.src_data_addr =
				cookie->qat_sgl_src_phys_addr;
		comp_req->comn_mid.dest_data_addr =
				cookie->qat_sgl_dst_phys_addr;
		comp_req->comn_mid.src_length = 0;
		comp_req->comn_mid.dst_length = 0;

	} else {
		/* flat aka linear buffer */
		ICP_QAT_FW_COMN_PTR_TYPE_SET(comp_req->comn_hdr.comn_req_flags,
				QAT_COMN_PTR_TYPE_FLAT);
		comp_req->comn_mid.src_length = op->src.length;
		comp_req->comn_mid.dst_length =
				comp_req->comp_pars.out_buffer_sz;

		comp_req->comn_mid.src_data_addr =
		    rte_pktmbuf_iova_offset(op->m_src, op->src.offset);
		comp_req->comn_mid.dest_data_addr =
		    rte_pktmbuf_iova_offset(op->m_dst, op->dst.offset);
	}

	if (unlikely(rte_pktmbuf_pkt_len(op->m_dst) < QAT_MIN_OUT_BUF_SIZE)) {
		/* QAT doesn't support dest. buffer lower
		 * than QAT_MIN_OUT_BUF_SIZE. Propagate error mark
		 * by colwerting this request to the null one
		 * and check the status in the response.
		 */
		QAT_DP_LOG(WARNING, "QAT destination buffer too small - resend with larger buffer");
		comp_req->comn_hdr.service_type = ICP_QAT_FW_COMN_REQ_NULL;
		comp_req->comn_hdr.service_cmd_id = ICP_QAT_FW_NULL_REQ_SERV_ID;
		cookie->error = RTE_COMP_OP_STATUS_OUT_OF_SPACE_TERMINATED;
	}

#if RTE_LOG_DP_LEVEL >= RTE_LOG_DEBUG
	QAT_DP_LOG(DEBUG, "Direction: %s",
	    qat_xform->qat_comp_request_type == QAT_COMP_REQUEST_DECOMPRESS ?
			    "decompression" : "compression");
	QAT_DP_HEXDUMP_LOG(DEBUG, "qat compression message:", comp_req,
		    sizeof(struct icp_qat_fw_comp_req));
#endif
	return 0;
}

static inline uint32_t adf_modulo(uint32_t data, uint32_t modulo_mask)
{
	return data & modulo_mask;
}

static inline void
qat_comp_mbuf_skip(struct rte_mbuf **mbuf, uint32_t *offset, uint32_t len)
{
	while (*offset + len >= rte_pktmbuf_data_len(*mbuf)) {
		len -= (rte_pktmbuf_data_len(*mbuf) - *offset);
		*mbuf = (*mbuf)->next;
		*offset = 0;
	}
	*offset = len;
}

int
qat_comp_build_multiple_requests(void *in_op, struct qat_qp *qp,
				 uint32_t parent_tail, int nb_descr)
{
	struct rte_comp_op op_backup;
	struct rte_mbuf dst_mbuf;
	struct rte_comp_op *op = in_op;
	struct qat_queue *txq = &(qp->tx_q);
	uint8_t *base_addr = (uint8_t *)txq->base_addr;
	uint8_t *out_msg = base_addr + parent_tail;
	uint32_t tail = parent_tail;
	struct icp_qat_fw_comp_req *comp_req =
			(struct icp_qat_fw_comp_req *)out_msg;
	struct qat_comp_op_cookie *parent_cookie =
			(struct qat_comp_op_cookie *)
			qp->op_cookies[parent_tail / txq->msg_size];
	struct qat_comp_op_cookie *child_cookie;
	uint16_t dst_data_size =
			RTE_MIN(RTE_PMD_QAT_COMP_IM_BUFFER_SIZE, 65535);
	uint32_t data_to_enqueue = op->src.length - QAT_FALLBACK_THLD;
	int num_descriptors_built = 1;
	int ret;

	QAT_DP_LOG(DEBUG, "op %p, parent_cookie %p", op, parent_cookie);

	/* copy original op to the local variable for restoring later */
	rte_memcpy(&op_backup, op, sizeof(op_backup));

	parent_cookie->nb_child_responses = 0;
	parent_cookie->nb_children = 0;
	parent_cookie->split_op = 1;
	parent_cookie->dst_data = op->m_dst;
	parent_cookie->dst_data_offset = op->dst.offset;

	op->src.length = QAT_FALLBACK_THLD;
	op->flush_flag = RTE_COMP_FLUSH_FULL;

	QAT_DP_LOG(DEBUG, "parent op src len %u dst len %u",
			op->src.length, op->m_dst->pkt_len);

	ret = qat_comp_build_request(in_op, out_msg, parent_cookie,
			qp->qat_dev_gen);
	if (ret != 0) {
		/* restore op and clear cookie */
		QAT_DP_LOG(WARNING, "Failed to build parent descriptor");
		op->src.length = op_backup.src.length;
		op->flush_flag = op_backup.flush_flag;
		parent_cookie->split_op = 0;
		return ret;
	}

	/* prepare local dst mbuf */
	rte_memcpy(&dst_mbuf, op->m_dst, sizeof(dst_mbuf));
	rte_pktmbuf_reset(&dst_mbuf);
	dst_mbuf.buf_len = dst_data_size;
	dst_mbuf.data_len = dst_data_size;
	dst_mbuf.pkt_len = dst_data_size;
	dst_mbuf.data_off = 0;

	/* update op for the child operations */
	op->m_dst = &dst_mbuf;
	op->dst.offset = 0;

	while (data_to_enqueue) {
		const struct rte_memzone *mz =
			parent_cookie->dst_memzones[num_descriptors_built - 1];
		uint32_t src_data_size = RTE_MIN(data_to_enqueue,
				QAT_FALLBACK_THLD);
		uint32_t cookie_index;

		/* update params for the next op */
		op->src.offset += QAT_FALLBACK_THLD;
		op->src.length = src_data_size;
		op->flush_flag = (src_data_size == data_to_enqueue) ?
			op_backup.flush_flag : RTE_COMP_FLUSH_FULL;

		/* update dst mbuf for the next op (use memzone for dst data) */
		dst_mbuf.buf_addr = mz->addr;
		dst_mbuf.buf_iova = mz->iova;

		/* move the tail and callwlate next cookie index */
		tail = adf_modulo(tail + txq->msg_size, txq->modulo_mask);
		cookie_index = tail / txq->msg_size;
		child_cookie = (struct qat_comp_op_cookie *)
				qp->op_cookies[cookie_index];
		comp_req = (struct icp_qat_fw_comp_req *)(base_addr + tail);

		/* update child cookie */
		child_cookie->split_op = 1; /* must be set for child as well */
		child_cookie->parent_cookie = parent_cookie; /* same as above */
		child_cookie->nb_children = 0;
		child_cookie->dest_buffer = mz->addr;

		QAT_DP_LOG(DEBUG,
				"cookie_index %u, child_cookie %p, comp_req %p",
				cookie_index, child_cookie, comp_req);
		QAT_DP_LOG(DEBUG,
				"data_to_enqueue %u, num_descriptors_built %d",
				data_to_enqueue, num_descriptors_built);
		QAT_DP_LOG(DEBUG, "child op src len %u dst len %u",
				op->src.length, op->m_dst->pkt_len);

		/* build the request */
		ret = qat_comp_build_request(op, (uint8_t *)comp_req,
				child_cookie, qp->qat_dev_gen);
		if (ret < 0) {
			QAT_DP_LOG(WARNING, "Failed to build child descriptor");
			/* restore op and clear cookie */
			rte_memcpy(op, &op_backup, sizeof(op_backup));
			parent_cookie->split_op = 0;
			parent_cookie->nb_children = 0;
			return ret;
		}

		data_to_enqueue -= src_data_size;
		num_descriptors_built++;
	}

	/* restore backed up original op */
	rte_memcpy(op, &op_backup, sizeof(op_backup));

	if (nb_descr != num_descriptors_built)
		QAT_DP_LOG(ERR, "split op. expected %d, built %d",
				nb_descr, num_descriptors_built);

	parent_cookie->nb_children = num_descriptors_built - 1;
	return num_descriptors_built;
}

static inline void
qat_comp_response_data_copy(struct qat_comp_op_cookie *cookie,
		       struct rte_comp_op *rx_op)
{
	struct qat_comp_op_cookie *pc = cookie->parent_cookie;
	struct rte_mbuf *sgl_buf = pc->dst_data;
	void *op_dst_addr = rte_pktmbuf_mtod_offset(sgl_buf, uint8_t *,
						    pc->dst_data_offset);

	/* number of bytes left in the current segment */
	uint32_t left_in_lwrrent = rte_pktmbuf_data_len(sgl_buf) -
			pc->dst_data_offset;

	uint32_t prod, sent;

	if (rx_op->produced <= left_in_lwrrent) {
		rte_memcpy(op_dst_addr, cookie->dest_buffer,
				rx_op->produced);
		/* callwlate dst mbuf and offset for the next child op */
		if (rx_op->produced == left_in_lwrrent) {
			pc->dst_data = sgl_buf->next;
			pc->dst_data_offset = 0;
		} else
			pc->dst_data_offset += rx_op->produced;
	} else {
		rte_memcpy(op_dst_addr, cookie->dest_buffer,
				left_in_lwrrent);
		sgl_buf = sgl_buf->next;
		prod = rx_op->produced - left_in_lwrrent;
		sent = left_in_lwrrent;
		while (prod > rte_pktmbuf_data_len(sgl_buf)) {
			op_dst_addr = rte_pktmbuf_mtod_offset(sgl_buf,
					uint8_t *, 0);

			rte_memcpy(op_dst_addr,
					((uint8_t *)cookie->dest_buffer) +
					sent,
					rte_pktmbuf_data_len(sgl_buf));

			prod -= rte_pktmbuf_data_len(sgl_buf);
			sent += rte_pktmbuf_data_len(sgl_buf);

			sgl_buf = sgl_buf->next;
		}

		op_dst_addr = rte_pktmbuf_mtod_offset(sgl_buf, uint8_t *, 0);

		rte_memcpy(op_dst_addr,
				((uint8_t *)cookie->dest_buffer) + sent,
				prod);

		/* callwlate dst mbuf and offset for the next child op */
		if (prod == rte_pktmbuf_data_len(sgl_buf)) {
			pc->dst_data = sgl_buf->next;
			pc->dst_data_offset = 0;
		} else {
			pc->dst_data = sgl_buf;
			pc->dst_data_offset = prod;
		}
	}
}

int
qat_comp_process_response(void **op, uint8_t *resp, void *op_cookie,
			  uint64_t *dequeue_err_count)
{
	struct icp_qat_fw_comp_resp *resp_msg =
			(struct icp_qat_fw_comp_resp *)resp;
	struct qat_comp_op_cookie *cookie =
			(struct qat_comp_op_cookie *)op_cookie;

	struct icp_qat_fw_resp_comp_pars *comp_resp1 =
	  (struct icp_qat_fw_resp_comp_pars *)&resp_msg->comp_resp_pars;

	QAT_DP_LOG(DEBUG, "input counter = %u, output counter = %u",
		   comp_resp1->input_byte_counter,
		   comp_resp1->output_byte_counter);

	struct rte_comp_op *rx_op = (struct rte_comp_op *)(uintptr_t)
			(resp_msg->opaque_data);
	struct qat_comp_stream *stream;
	struct qat_comp_xform *qat_xform;
	int err = resp_msg->comn_resp.comn_status &
			((1 << QAT_COMN_RESP_CMP_STATUS_BITPOS) |
			 (1 << QAT_COMN_RESP_XLAT_STATUS_BITPOS));

	if (rx_op->op_type == RTE_COMP_OP_STATEFUL) {
		stream = rx_op->stream;
		qat_xform = &stream->qat_xform;
		/* clear op-in-progress flag */
		stream->op_in_progress = 0;
	} else {
		stream = NULL;
		qat_xform = rx_op->private_xform;
	}

#if RTE_LOG_DP_LEVEL >= RTE_LOG_DEBUG
	QAT_DP_LOG(DEBUG, "Direction: %s",
	    qat_xform->qat_comp_request_type == QAT_COMP_REQUEST_DECOMPRESS ?
	    "decompression" : "compression");
	QAT_DP_HEXDUMP_LOG(DEBUG,  "qat_response:", (uint8_t *)resp_msg,
			sizeof(struct icp_qat_fw_comp_resp));
#endif

	if (unlikely(cookie->error)) {
		rx_op->status = cookie->error;
		cookie->error = 0;
		++(*dequeue_err_count);
		rx_op->debug_status = 0;
		rx_op->consumed = 0;
		rx_op->produced = 0;
		*op = (void *)rx_op;
		/* also in this case number of returned ops */
		/* must be equal to one, */
		/* appropriate status (error) must be set as well */
		return 1;
	}

	if (likely(qat_xform->qat_comp_request_type
			!= QAT_COMP_REQUEST_DECOMPRESS)) {
		if (unlikely(ICP_QAT_FW_COMN_HDR_CLW_FLAG_GET(
				resp_msg->comn_resp.hdr_flags)
					== ICP_QAT_FW_COMP_NO_CLW)) {
			rx_op->status = RTE_COMP_OP_STATUS_ERROR;
			rx_op->debug_status = ERR_CODE_QAT_COMP_WRONG_FW;
			*op = (void *)rx_op;
			QAT_DP_LOG(ERR, "QAT has wrong firmware");
			++(*dequeue_err_count);
			return 1;
		}
	}

	if (err) {
		if (unlikely((err & (1 << QAT_COMN_RESP_XLAT_STATUS_BITPOS))
			     &&	(qat_xform->qat_comp_request_type
				 == QAT_COMP_REQUEST_DYNAMIC_COMP_STATELESS))) {
			QAT_DP_LOG(ERR, "QAT intermediate buffer may be too "
			    "small for output, try configuring a larger size");
		}

		int8_t cmp_err_code =
			(int8_t)resp_msg->comn_resp.comn_error.cmp_err_code;
		int8_t xlat_err_code =
			(int8_t)resp_msg->comn_resp.comn_error.xlat_err_code;

		/* handle recoverable out-of-buffer condition in stateful
		 * decompression scenario
		 */
		if (cmp_err_code == ERR_CODE_OVERFLOW_ERROR && !xlat_err_code
				&& qat_xform->qat_comp_request_type
					== QAT_COMP_REQUEST_DECOMPRESS
				&& rx_op->op_type == RTE_COMP_OP_STATEFUL) {
			struct icp_qat_fw_resp_comp_pars *comp_resp =
					&resp_msg->comp_resp_pars;
			rx_op->status =
				RTE_COMP_OP_STATUS_OUT_OF_SPACE_RECOVERABLE;
			rx_op->consumed = comp_resp->input_byte_counter;
			rx_op->produced = comp_resp->output_byte_counter;
			stream->start_of_packet = 0;
		} else if ((cmp_err_code == ERR_CODE_OVERFLOW_ERROR
			  && !xlat_err_code)
				||
		    (!cmp_err_code && xlat_err_code == ERR_CODE_OVERFLOW_ERROR)
				||
		    (cmp_err_code == ERR_CODE_OVERFLOW_ERROR &&
		     xlat_err_code == ERR_CODE_OVERFLOW_ERROR)){

			struct icp_qat_fw_resp_comp_pars *comp_resp =
					(struct icp_qat_fw_resp_comp_pars *)
					&resp_msg->comp_resp_pars;

			/* handle recoverable out-of-buffer condition
			 * in stateless compression scenario
			 */
			if (comp_resp->input_byte_counter) {
				if ((qat_xform->qat_comp_request_type
				== QAT_COMP_REQUEST_FIXED_COMP_STATELESS) ||
				    (qat_xform->qat_comp_request_type
				== QAT_COMP_REQUEST_DYNAMIC_COMP_STATELESS)) {

					rx_op->status =
				RTE_COMP_OP_STATUS_OUT_OF_SPACE_RECOVERABLE;
					rx_op->consumed =
						comp_resp->input_byte_counter;
					rx_op->produced =
						comp_resp->output_byte_counter;
				} else
					rx_op->status =
				RTE_COMP_OP_STATUS_OUT_OF_SPACE_TERMINATED;
			} else
				rx_op->status =
				RTE_COMP_OP_STATUS_OUT_OF_SPACE_TERMINATED;
		} else
			rx_op->status = RTE_COMP_OP_STATUS_ERROR;

		++(*dequeue_err_count);
		rx_op->debug_status =
			*((uint16_t *)(&resp_msg->comn_resp.comn_error));
	} else {
		struct icp_qat_fw_resp_comp_pars *comp_resp =
		  (struct icp_qat_fw_resp_comp_pars *)&resp_msg->comp_resp_pars;

		rx_op->status = RTE_COMP_OP_STATUS_SUCCESS;
		rx_op->consumed = comp_resp->input_byte_counter;
		rx_op->produced = comp_resp->output_byte_counter;
		if (stream)
			stream->start_of_packet = 0;

		if (qat_xform->checksum_type != RTE_COMP_CHECKSUM_NONE) {
			if (qat_xform->checksum_type == RTE_COMP_CHECKSUM_CRC32)
				rx_op->output_chksum = comp_resp->lwrr_crc32;
			else if (qat_xform->checksum_type ==
					RTE_COMP_CHECKSUM_ADLER32)
				rx_op->output_chksum = comp_resp->lwrr_adler_32;
			else
				rx_op->output_chksum = comp_resp->lwrr_chksum;
		}
	}
	QAT_DP_LOG(DEBUG, "About to check for split op :cookies: %p %p, split:%u",
		cookie, cookie->parent_cookie, cookie->split_op);

	if (cookie->split_op) {
		*op = NULL;
		struct qat_comp_op_cookie *pc = cookie->parent_cookie;

		if (cookie->nb_children > 0) {
			QAT_DP_LOG(DEBUG, "Parent");
			/* parent - don't return until all children
			 * responses are collected
			 */
			cookie->total_consumed = rx_op->consumed;
			cookie->total_produced = rx_op->produced;
			if (err) {
				cookie->error = rx_op->status;
				rx_op->status = RTE_COMP_OP_STATUS_SUCCESS;
			} else {
				/* callwlate dst mbuf and offset for child op */
				qat_comp_mbuf_skip(&cookie->dst_data,
						&cookie->dst_data_offset,
						rx_op->produced);
			}
		} else {
			QAT_DP_LOG(DEBUG, "Child");
			if (pc->error == RTE_COMP_OP_STATUS_SUCCESS) {
				if (err)
					pc->error = rx_op->status;
				if (rx_op->produced) {
					/* this covers both SUCCESS and
					 * OUT_OF_SPACE_RECOVERABLE cases
					 */
					qat_comp_response_data_copy(cookie,
							rx_op);
					pc->total_consumed += rx_op->consumed;
					pc->total_produced += rx_op->produced;
				}
			}
			rx_op->status = RTE_COMP_OP_STATUS_SUCCESS;

			pc->nb_child_responses++;

			/* (child) cookie fields have to be reset
			 * to avoid problems with reusability -
			 * rx and tx queue starting from index zero
			 */
			cookie->nb_children = 0;
			cookie->split_op = 0;
			cookie->nb_child_responses = 0;
			cookie->dest_buffer = NULL;

			if (pc->nb_child_responses == pc->nb_children) {
				uint8_t child_resp;

				/* parent should be included as well */
				child_resp = pc->nb_child_responses + 1;

				rx_op->status = pc->error;
				rx_op->consumed = pc->total_consumed;
				rx_op->produced = pc->total_produced;
				*op = (void *)rx_op;

				/* free memzones used for dst data */
				qat_comp_free_split_op_memzones(pc,
						pc->nb_children);

				/* (parent) cookie fields have to be reset
				 * to avoid problems with reusability -
				 * rx and tx queue starting from index zero
				 */
				pc->nb_children = 0;
				pc->split_op = 0;
				pc->nb_child_responses = 0;
				pc->error = RTE_COMP_OP_STATUS_SUCCESS;

				return child_resp;
			}
		}
		return 0;
	}

	*op = (void *)rx_op;
	return 1;
}

unsigned int
qat_comp_xform_size(void)
{
	return RTE_ALIGN_CEIL(sizeof(struct qat_comp_xform), 8);
}

unsigned int
qat_comp_stream_size(void)
{
	return RTE_ALIGN_CEIL(sizeof(struct qat_comp_stream), 8);
}

static void qat_comp_create_req_hdr(struct icp_qat_fw_comn_req_hdr *header,
				    enum qat_comp_request_type request)
{
	if (request == QAT_COMP_REQUEST_FIXED_COMP_STATELESS)
		header->service_cmd_id = ICP_QAT_FW_COMP_CMD_STATIC;
	else if (request == QAT_COMP_REQUEST_DYNAMIC_COMP_STATELESS)
		header->service_cmd_id = ICP_QAT_FW_COMP_CMD_DYNAMIC;
	else if (request == QAT_COMP_REQUEST_DECOMPRESS)
		header->service_cmd_id = ICP_QAT_FW_COMP_CMD_DECOMPRESS;

	header->service_type = ICP_QAT_FW_COMN_REQ_CPM_FW_COMP;
	header->hdr_flags =
	    ICP_QAT_FW_COMN_HDR_FLAGS_BUILD(ICP_QAT_FW_COMN_REQ_FLAG_SET);

	header->comn_req_flags = ICP_QAT_FW_COMN_FLAGS_BUILD(
	    QAT_COMN_CD_FLD_TYPE_16BYTE_DATA, QAT_COMN_PTR_TYPE_FLAT);
}

static int qat_comp_create_templates(struct qat_comp_xform *qat_xform,
			const struct rte_memzone *interm_buff_mz,
			const struct rte_comp_xform *xform,
			const struct qat_comp_stream *stream,
			enum rte_comp_op_type op_type)
{
	struct icp_qat_fw_comp_req *comp_req;
	int comp_level, algo;
	uint32_t req_par_flags;
	int direction = ICP_QAT_HW_COMPRESSION_DIR_COMPRESS;

	if (unlikely(qat_xform == NULL)) {
		QAT_LOG(ERR, "Session was not created for this device");
		return -EILWAL;
	}

	if (op_type == RTE_COMP_OP_STATEFUL) {
		if (unlikely(stream == NULL)) {
			QAT_LOG(ERR, "Stream must be non null for stateful op");
			return -EILWAL;
		}
		if (unlikely(qat_xform->qat_comp_request_type !=
			     QAT_COMP_REQUEST_DECOMPRESS)) {
			QAT_LOG(ERR, "QAT PMD does not support stateful compression");
			return -ENOTSUP;
		}
	}

	if (qat_xform->qat_comp_request_type == QAT_COMP_REQUEST_DECOMPRESS) {
		direction = ICP_QAT_HW_COMPRESSION_DIR_DECOMPRESS;
		comp_level = ICP_QAT_HW_COMPRESSION_DEPTH_1;
		req_par_flags = ICP_QAT_FW_COMP_REQ_PARAM_FLAGS_BUILD(
				ICP_QAT_FW_COMP_SOP, ICP_QAT_FW_COMP_EOP,
				ICP_QAT_FW_COMP_BFINAL,
				ICP_QAT_FW_COMP_CLW,
				ICP_QAT_FW_COMP_CLW_RECOVERY);
	} else {
		if (xform->compress.level == RTE_COMP_LEVEL_PMD_DEFAULT)
			comp_level = ICP_QAT_HW_COMPRESSION_DEPTH_8;
		else if (xform->compress.level == 1)
			comp_level = ICP_QAT_HW_COMPRESSION_DEPTH_1;
		else if (xform->compress.level == 2)
			comp_level = ICP_QAT_HW_COMPRESSION_DEPTH_4;
		else if (xform->compress.level == 3)
			comp_level = ICP_QAT_HW_COMPRESSION_DEPTH_8;
		else if (xform->compress.level >= 4 &&
			 xform->compress.level <= 9)
			comp_level = ICP_QAT_HW_COMPRESSION_DEPTH_16;
		else {
			QAT_LOG(ERR, "compression level not supported");
			return -EILWAL;
		}
		req_par_flags = ICP_QAT_FW_COMP_REQ_PARAM_FLAGS_BUILD(
				ICP_QAT_FW_COMP_SOP, ICP_QAT_FW_COMP_EOP,
				ICP_QAT_FW_COMP_BFINAL, ICP_QAT_FW_COMP_CLW,
				ICP_QAT_FW_COMP_CLW_RECOVERY);
	}

	switch (xform->compress.algo) {
	case RTE_COMP_ALGO_DEFLATE:
		algo = ICP_QAT_HW_COMPRESSION_ALGO_DEFLATE;
		break;
	case RTE_COMP_ALGO_LZS:
	default:
		/* RTE_COMP_NULL */
		QAT_LOG(ERR, "compression algorithm not supported");
		return -EILWAL;
	}

	comp_req = &qat_xform->qat_comp_req_tmpl;

	/* Initialize header */
	qat_comp_create_req_hdr(&comp_req->comn_hdr,
					qat_xform->qat_comp_request_type);

	if (op_type == RTE_COMP_OP_STATEFUL) {
		comp_req->comn_hdr.serv_specif_flags =
				ICP_QAT_FW_COMP_FLAGS_BUILD(
			ICP_QAT_FW_COMP_STATEFUL_SESSION,
			ICP_QAT_FW_COMP_NOT_AUTO_SELECT_BEST,
			ICP_QAT_FW_COMP_NOT_ENH_AUTO_SELECT_BEST,
			ICP_QAT_FW_COMP_NOT_DISABLE_TYPE0_ENH_AUTO_SELECT_BEST,
			ICP_QAT_FW_COMP_ENABLE_SELWRE_RAM_USED_AS_INTMD_BUF);

		/* Decompression state registers */
		comp_req->comp_cd_ctrl.comp_state_addr =
				stream->state_registers_decomp_phys;

		/* Enable A, B, C, D, and E (CAMs). */
		comp_req->comp_cd_ctrl.ram_bank_flags =
			ICP_QAT_FW_COMP_RAM_FLAGS_BUILD(
				ICP_QAT_FW_COMP_BANK_DISABLED, /* Bank I */
				ICP_QAT_FW_COMP_BANK_DISABLED, /* Bank H */
				ICP_QAT_FW_COMP_BANK_DISABLED, /* Bank G */
				ICP_QAT_FW_COMP_BANK_DISABLED, /* Bank F */
				ICP_QAT_FW_COMP_BANK_ENABLED,  /* Bank E */
				ICP_QAT_FW_COMP_BANK_ENABLED,  /* Bank D */
				ICP_QAT_FW_COMP_BANK_ENABLED,  /* Bank C */
				ICP_QAT_FW_COMP_BANK_ENABLED,  /* Bank B */
				ICP_QAT_FW_COMP_BANK_ENABLED); /* Bank A */

		comp_req->comp_cd_ctrl.ram_banks_addr =
				stream->inflate_context_phys;
	} else {
		comp_req->comn_hdr.serv_specif_flags =
				ICP_QAT_FW_COMP_FLAGS_BUILD(
			ICP_QAT_FW_COMP_STATELESS_SESSION,
			ICP_QAT_FW_COMP_NOT_AUTO_SELECT_BEST,
			ICP_QAT_FW_COMP_NOT_ENH_AUTO_SELECT_BEST,
			ICP_QAT_FW_COMP_NOT_DISABLE_TYPE0_ENH_AUTO_SELECT_BEST,
			ICP_QAT_FW_COMP_ENABLE_SELWRE_RAM_USED_AS_INTMD_BUF);
	}

	comp_req->cd_pars.sl.comp_slice_cfg_word[0] =
	    ICP_QAT_HW_COMPRESSION_CONFIG_BUILD(
		direction,
		/* In CPM 1.6 only valid mode ! */
		ICP_QAT_HW_COMPRESSION_DELAYED_MATCH_ENABLED, algo,
		/* Translate level to depth */
		comp_level, ICP_QAT_HW_COMPRESSION_FILE_TYPE_0);

	comp_req->comp_pars.initial_adler = 1;
	comp_req->comp_pars.initial_crc32 = 0;
	comp_req->comp_pars.req_par_flags = req_par_flags;


	if (qat_xform->qat_comp_request_type ==
			QAT_COMP_REQUEST_FIXED_COMP_STATELESS ||
	    qat_xform->qat_comp_request_type == QAT_COMP_REQUEST_DECOMPRESS) {
		ICP_QAT_FW_COMN_NEXT_ID_SET(&comp_req->comp_cd_ctrl,
					    ICP_QAT_FW_SLICE_DRAM_WR);
		ICP_QAT_FW_COMN_LWRR_ID_SET(&comp_req->comp_cd_ctrl,
					    ICP_QAT_FW_SLICE_COMP);
	} else if (qat_xform->qat_comp_request_type ==
			QAT_COMP_REQUEST_DYNAMIC_COMP_STATELESS) {

		ICP_QAT_FW_COMN_NEXT_ID_SET(&comp_req->comp_cd_ctrl,
				ICP_QAT_FW_SLICE_XLAT);
		ICP_QAT_FW_COMN_LWRR_ID_SET(&comp_req->comp_cd_ctrl,
				ICP_QAT_FW_SLICE_COMP);

		ICP_QAT_FW_COMN_NEXT_ID_SET(&comp_req->u2.xlt_cd_ctrl,
				ICP_QAT_FW_SLICE_DRAM_WR);
		ICP_QAT_FW_COMN_LWRR_ID_SET(&comp_req->u2.xlt_cd_ctrl,
				ICP_QAT_FW_SLICE_XLAT);

		comp_req->u1.xlt_pars.inter_buff_ptr =
				interm_buff_mz->iova;
	}

#if RTE_LOG_DP_LEVEL >= RTE_LOG_DEBUG
	QAT_DP_HEXDUMP_LOG(DEBUG, "qat compression message template:", comp_req,
		    sizeof(struct icp_qat_fw_comp_req));
#endif
	return 0;
}

/**
 * Create driver private_xform data.
 *
 * @param dev
 *   Compressdev device
 * @param xform
 *   xform data from application
 * @param private_xform
 *   ptr where handle of pmd's private_xform data should be stored
 * @return
 *  - if successful returns 0
 *    and valid private_xform handle
 *  - <0 in error cases
 *  - Returns -EILWAL if input parameters are invalid.
 *  - Returns -ENOTSUP if comp device does not support the comp transform.
 *  - Returns -ENOMEM if the private_xform could not be allocated.
 */
int
qat_comp_private_xform_create(struct rte_compressdev *dev,
			      const struct rte_comp_xform *xform,
			      void **private_xform)
{
	struct qat_comp_dev_private *qat = dev->data->dev_private;

	if (unlikely(private_xform == NULL)) {
		QAT_LOG(ERR, "QAT: private_xform parameter is NULL");
		return -EILWAL;
	}
	if (unlikely(qat->xformpool == NULL)) {
		QAT_LOG(ERR, "QAT device has no private_xform mempool");
		return -ENOMEM;
	}
	if (rte_mempool_get(qat->xformpool, private_xform)) {
		QAT_LOG(ERR, "Couldn't get object from qat xform mempool");
		return -ENOMEM;
	}

	struct qat_comp_xform *qat_xform =
			(struct qat_comp_xform *)*private_xform;

	if (xform->type == RTE_COMP_COMPRESS) {

		if (xform->compress.deflate.huffman == RTE_COMP_HUFFMAN_FIXED ||
		  ((xform->compress.deflate.huffman == RTE_COMP_HUFFMAN_DEFAULT)
				   && qat->interm_buff_mz == NULL))
			qat_xform->qat_comp_request_type =
					QAT_COMP_REQUEST_FIXED_COMP_STATELESS;

		else if ((xform->compress.deflate.huffman ==
				RTE_COMP_HUFFMAN_DYNAMIC ||
				xform->compress.deflate.huffman ==
						RTE_COMP_HUFFMAN_DEFAULT) &&
				qat->interm_buff_mz != NULL)

			qat_xform->qat_comp_request_type =
					QAT_COMP_REQUEST_DYNAMIC_COMP_STATELESS;

		else {
			QAT_LOG(ERR,
					"IM buffers needed for dynamic deflate. Set size in config file");
			return -EILWAL;
		}

		qat_xform->checksum_type = xform->compress.chksum;

	} else {
		qat_xform->qat_comp_request_type = QAT_COMP_REQUEST_DECOMPRESS;
		qat_xform->checksum_type = xform->decompress.chksum;
	}

	if (qat_comp_create_templates(qat_xform, qat->interm_buff_mz, xform,
				      NULL, RTE_COMP_OP_STATELESS)) {
		QAT_LOG(ERR, "QAT: Problem with setting compression");
		return -EILWAL;
	}
	return 0;
}

/**
 * Free driver private_xform data.
 *
 * @param dev
 *   Compressdev device
 * @param private_xform
 *   handle of pmd's private_xform data
 * @return
 *  - 0 if successful
 *  - <0 in error cases
 *  - Returns -EILWAL if input parameters are invalid.
 */
int
qat_comp_private_xform_free(struct rte_compressdev *dev __rte_unused,
			    void *private_xform)
{
	struct qat_comp_xform *qat_xform =
			(struct qat_comp_xform *)private_xform;

	if (qat_xform) {
		memset(qat_xform, 0, qat_comp_xform_size());
		struct rte_mempool *mp = rte_mempool_from_obj(qat_xform);

		rte_mempool_put(mp, qat_xform);
		return 0;
	}
	return -EILWAL;
}

/**
 * Reset stream state for the next use.
 *
 * @param stream
 *   handle of pmd's private stream data
 */
static void
qat_comp_stream_reset(struct qat_comp_stream *stream)
{
	if (stream) {
		memset(&stream->qat_xform, 0, sizeof(struct qat_comp_xform));
		stream->start_of_packet = 1;
		stream->op_in_progress = 0;
	}
}

/**
 * Create driver private stream data.
 *
 * @param dev
 *   Compressdev device
 * @param xform
 *   xform data
 * @param stream
 *   ptr where handle of pmd's private stream data should be stored
 * @return
 *  - Returns 0 if private stream structure has been created successfully.
 *  - Returns -EILWAL if input parameters are invalid.
 *  - Returns -ENOTSUP if comp device does not support STATEFUL operations.
 *  - Returns -ENOTSUP if comp device does not support the comp transform.
 *  - Returns -ENOMEM if the private stream could not be allocated.
 */
int
qat_comp_stream_create(struct rte_compressdev *dev,
		       const struct rte_comp_xform *xform,
		       void **stream)
{
	struct qat_comp_dev_private *qat = dev->data->dev_private;
	struct qat_comp_stream *ptr;

	if (unlikely(stream == NULL)) {
		QAT_LOG(ERR, "QAT: stream parameter is NULL");
		return -EILWAL;
	}
	if (unlikely(xform->type == RTE_COMP_COMPRESS)) {
		QAT_LOG(ERR, "QAT: stateful compression not supported");
		return -ENOTSUP;
	}
	if (unlikely(qat->streampool == NULL)) {
		QAT_LOG(ERR, "QAT device has no stream mempool");
		return -ENOMEM;
	}
	if (rte_mempool_get(qat->streampool, stream)) {
		QAT_LOG(ERR, "Couldn't get object from qat stream mempool");
		return -ENOMEM;
	}

	ptr = (struct qat_comp_stream *) *stream;
	qat_comp_stream_reset(ptr);
	ptr->qat_xform.qat_comp_request_type = QAT_COMP_REQUEST_DECOMPRESS;
	ptr->qat_xform.checksum_type = xform->decompress.chksum;

	if (qat_comp_create_templates(&ptr->qat_xform, qat->interm_buff_mz,
				      xform, ptr, RTE_COMP_OP_STATEFUL)) {
		QAT_LOG(ERR, "QAT: problem with creating descriptor template for stream");
		rte_mempool_put(qat->streampool, *stream);
		*stream = NULL;
		return -EILWAL;
	}

	return 0;
}

/**
 * Free driver private stream data.
 *
 * @param dev
 *   Compressdev device
 * @param stream
 *   handle of pmd's private stream data
 * @return
 *  - 0 if successful
 *  - <0 in error cases
 *  - Returns -EILWAL if input parameters are invalid.
 *  - Returns -ENOTSUP if comp device does not support STATEFUL operations.
 *  - Returns -EBUSY if can't free stream as there are inflight operations
 */
int
qat_comp_stream_free(struct rte_compressdev *dev, void *stream)
{
	if (stream) {
		struct qat_comp_dev_private *qat = dev->data->dev_private;
		qat_comp_stream_reset((struct qat_comp_stream *) stream);
		rte_mempool_put(qat->streampool, stream);
		return 0;
	}
	return -EILWAL;
}
