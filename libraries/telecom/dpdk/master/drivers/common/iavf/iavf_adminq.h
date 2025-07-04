/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2001-2020 Intel Corporation
 */

#ifndef _IAVF_ADMINQ_H_
#define _IAVF_ADMINQ_H_

#include "iavf_osdep.h"
#include "iavf_status.h"
#include "iavf_adminq_cmd.h"

#define IAVF_ADMINQ_DESC(R, i)   \
	(&(((struct iavf_aq_desc *)((R).desc_buf.va))[i]))

#define IAVF_ADMINQ_DESC_ALIGNMENT 4096

struct iavf_adminq_ring {
	struct iavf_virt_mem dma_head;	/* space for dma structures */
	struct iavf_dma_mem desc_buf;	/* descriptor ring memory */
	struct iavf_virt_mem cmd_buf;	/* command buffer memory */

	union {
		struct iavf_dma_mem *asq_bi;
		struct iavf_dma_mem *arq_bi;
	} r;

	u16 count;		/* Number of descriptors */
	u16 rx_buf_len;		/* Admin Receive Queue buffer length */

	/* used for interrupt processing */
	u16 next_to_use;
	u16 next_to_clean;

	/* used for queue tracking */
	u32 head;
	u32 tail;
	u32 len;
	u32 bah;
	u32 bal;
};

/* ASQ transaction details */
struct iavf_asq_cmd_details {
	void *callback; /* cast from type IAVF_ADMINQ_CALLBACK */
	u64 cookie;
	u16 flags_ena;
	u16 flags_dis;
	bool async;
	bool postpone;
	struct iavf_aq_desc *wb_desc;
};

#define IAVF_ADMINQ_DETAILS(R, i)   \
	(&(((struct iavf_asq_cmd_details *)((R).cmd_buf.va))[i]))

/* ARQ event information */
struct iavf_arq_event_info {
	struct iavf_aq_desc desc;
	u16 msg_len;
	u16 buf_len;
	u8 *msg_buf;
};

/* Admin Queue information */
struct iavf_adminq_info {
	struct iavf_adminq_ring arq;    /* receive queue */
	struct iavf_adminq_ring asq;    /* send queue */
	u32 asq_cmd_timeout;            /* send queue cmd write back timeout*/
	u16 num_arq_entries;            /* receive queue depth */
	u16 num_asq_entries;            /* send queue depth */
	u16 arq_buf_size;               /* receive queue buffer size */
	u16 asq_buf_size;               /* send queue buffer size */
	u16 fw_maj_ver;                 /* firmware major version */
	u16 fw_min_ver;                 /* firmware minor version */
	u32 fw_build;                   /* firmware build number */
	u16 api_maj_ver;                /* api major version */
	u16 api_min_ver;                /* api minor version */

	struct iavf_spinlock asq_spinlock; /* Send queue spinlock */
	struct iavf_spinlock arq_spinlock; /* Receive queue spinlock */

	/* last status values on send and receive queues */
	enum iavf_admin_queue_err asq_last_status;
	enum iavf_admin_queue_err arq_last_status;
};

/* general information */
#define IAVF_AQ_LARGE_BUF	512
#define IAVF_ASQ_CMD_TIMEOUT	250000  /* usecs */

void iavf_fill_default_direct_cmd_desc(struct iavf_aq_desc *desc,
				       u16 opcode);

#endif /* _IAVF_ADMINQ_H_ */
