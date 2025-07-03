/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2018 Microsoft Corp.
 * Copyright (c) 2010-2012 Citrix Inc.
 * Copyright (c) 2012 NetApp Inc.
 * All rights reserved.
 */

/*
 * Network Virtualization Service.
 */


#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>

#include <rte_ethdev.h>
#include <rte_string_fns.h>
#include <rte_memzone.h>
#include <rte_malloc.h>
#include <rte_atomic.h>
#include <rte_branch_prediction.h>
#include <rte_ether.h>
#include <rte_common.h>
#include <rte_errno.h>
#include <rte_cycles.h>
#include <rte_memory.h>
#include <rte_eal.h>
#include <rte_dev.h>
#include <rte_bus_vmbus.h>

#include "hn_logs.h"
#include "hn_var.h"
#include "hn_lws.h"

static const uint32_t hn_lws_version[] = {
	LWS_VERSION_61,
	LWS_VERSION_6,
	LWS_VERSION_5,
	LWS_VERSION_4,
	LWS_VERSION_2,
	LWS_VERSION_1
};

static int hn_lws_req_send(struct hn_data *hv,
			   void *req, uint32_t reqlen)
{
	return rte_vmbus_chan_send(hn_primary_chan(hv),
				   VMBUS_CHANPKT_TYPE_INBAND,
				   req, reqlen, 0,
				   VMBUS_CHANPKT_FLAG_NONE, NULL);
}

static int
__hn_lws_exelwte(struct hn_data *hv,
	       void *req, uint32_t reqlen,
	       void *resp, uint32_t resplen,
	       uint32_t type)
{
	struct vmbus_channel *chan = hn_primary_chan(hv);
	char buffer[LWS_RESPSIZE_MAX];
	const struct hn_lws_hdr *hdr;
	uint64_t xactid;
	uint32_t len;
	int ret;

	/* Send request to ring buffer */
	ret = rte_vmbus_chan_send(chan, VMBUS_CHANPKT_TYPE_INBAND,
				  req, reqlen, 0,
				  VMBUS_CHANPKT_FLAG_RC, NULL);

	if (ret) {
		PMD_DRV_LOG(ERR, "send request failed: %d", ret);
		return ret;
	}

 retry:
	len = sizeof(buffer);
	ret = rte_vmbus_chan_recv(chan, buffer, &len, &xactid);
	if (ret == -EAGAIN) {
		rte_delay_us(HN_CHAN_INTERVAL_US);
		goto retry;
	}

	if (ret < 0) {
		PMD_DRV_LOG(ERR, "recv response failed: %d", ret);
		return ret;
	}

	if (len < sizeof(*hdr)) {
		PMD_DRV_LOG(ERR, "response missing LWS header");
		return -EILWAL;
	}

	hdr = (struct hn_lws_hdr *)buffer;

	/* Silently drop received packets while waiting for response */
	if (hdr->type == LWS_TYPE_RNDIS) {
		hn_lws_ack_rxbuf(chan, xactid);
		goto retry;
	}

	if (hdr->type != type) {
		PMD_DRV_LOG(ERR, "unexpected LWS resp %#x, expect %#x",
			    hdr->type, type);
		return -EILWAL;
	}

	if (len < resplen) {
		PMD_DRV_LOG(ERR,
			    "invalid LWS resp len %u (expect %u)",
			    len, resplen);
		return -EILWAL;
	}

	memcpy(resp, buffer, resplen);

	/* All pass! */
	return 0;
}


/*
 * Execute one control command and get the response.
 * Only one command can be active on a channel at once
 * Unlike BSD, DPDK does not have an interrupt context
 * so the polling is required to wait for response.
 */
static int
hn_lws_exelwte(struct hn_data *hv,
	       void *req, uint32_t reqlen,
	       void *resp, uint32_t resplen,
	       uint32_t type)
{
	struct hn_rx_queue *rxq = hv->primary;
	int ret;

	rte_spinlock_lock(&rxq->ring_lock);
	ret = __hn_lws_exelwte(hv, req, reqlen, resp, resplen, type);
	rte_spinlock_unlock(&rxq->ring_lock);

	return ret;
}

static int
hn_lws_doinit(struct hn_data *hv, uint32_t lws_ver)
{
	struct hn_lws_init init;
	struct hn_lws_init_resp resp;
	uint32_t status;
	int error;

	memset(&init, 0, sizeof(init));
	init.type = LWS_TYPE_INIT;
	init.ver_min = lws_ver;
	init.ver_max = lws_ver;

	error = hn_lws_exelwte(hv, &init, sizeof(init),
			       &resp, sizeof(resp),
			       LWS_TYPE_INIT_RESP);
	if (error)
		return error;

	status = resp.status;
	if (status != LWS_STATUS_OK) {
		/* Not fatal, try other versions */
		PMD_INIT_LOG(DEBUG, "lws init failed for ver 0x%x",
			     lws_ver);
		return -EILWAL;
	}

	return 0;
}

static int
hn_lws_conn_rxbuf(struct hn_data *hv)
{
	struct hn_lws_rxbuf_conn conn;
	struct hn_lws_rxbuf_connresp resp;
	uint32_t status;
	int error;

	/* Kernel has already setup RXBUF on primary channel. */

	/*
	 * Connect RXBUF to LWS.
	 */
	conn.type = LWS_TYPE_RXBUF_CONN;
	conn.gpadl = hv->rxbuf_res->phys_addr;
	conn.sig = LWS_RXBUF_SIG;
	PMD_DRV_LOG(DEBUG, "connect rxbuff va=%p gpad=%#" PRIx64,
		    hv->rxbuf_res->addr,
		    hv->rxbuf_res->phys_addr);

	error = hn_lws_exelwte(hv, &conn, sizeof(conn),
			       &resp, sizeof(resp),
			       LWS_TYPE_RXBUF_CONNRESP);
	if (error) {
		PMD_DRV_LOG(ERR,
			    "exec lws rxbuf conn failed: %d",
			    error);
		return error;
	}

	status = resp.status;
	if (status != LWS_STATUS_OK) {
		PMD_DRV_LOG(ERR,
			    "lws rxbuf conn failed: %x", status);
		return -EIO;
	}
	if (resp.nsect != 1) {
		PMD_DRV_LOG(ERR,
			    "lws rxbuf response num sections %u != 1",
			    resp.nsect);
		return -EIO;
	}

	PMD_DRV_LOG(INFO,
		    "receive buffer size %u count %u",
		    resp.lws_sect[0].slotsz,
		    resp.lws_sect[0].slotcnt);
	hv->rxbuf_section_cnt = resp.lws_sect[0].slotcnt;

	/*
	 * Pimary queue's rxbuf_info is not allocated at creation time.
	 * Now we can allocate it after we figure out the slotcnt.
	 */
	hv->primary->rxbuf_info = rte_calloc("HN_RXBUF_INFO",
			hv->rxbuf_section_cnt,
			sizeof(*hv->primary->rxbuf_info),
			RTE_CACHE_LINE_SIZE);
	if (!hv->primary->rxbuf_info) {
		PMD_DRV_LOG(ERR,
			    "could not allocate rxbuf info");
		return -ENOMEM;
	}

	return 0;
}

static void
hn_lws_disconn_rxbuf(struct hn_data *hv)
{
	struct hn_lws_rxbuf_disconn disconn;
	int error;

	/*
	 * Disconnect RXBUF from LWS.
	 */
	memset(&disconn, 0, sizeof(disconn));
	disconn.type = LWS_TYPE_RXBUF_DISCONN;
	disconn.sig = LWS_RXBUF_SIG;

	/* NOTE: No response. */
	error = hn_lws_req_send(hv, &disconn, sizeof(disconn));
	if (error) {
		PMD_DRV_LOG(ERR,
			    "send lws rxbuf disconn failed: %d",
			    error);
	}

	/*
	 * Linger long enough for LWS to disconnect RXBUF.
	 */
	rte_delay_ms(200);
}

static void
hn_lws_disconn_chim(struct hn_data *hv)
{
	int error;

	if (hv->chim_cnt != 0) {
		struct hn_lws_chim_disconn disconn;

		/* Disconnect chimney sending buffer from LWS. */
		memset(&disconn, 0, sizeof(disconn));
		disconn.type = LWS_TYPE_CHIM_DISCONN;
		disconn.sig = LWS_CHIM_SIG;

		/* NOTE: No response. */
		error = hn_lws_req_send(hv, &disconn, sizeof(disconn));

		if (error) {
			PMD_DRV_LOG(ERR,
				    "send lws chim disconn failed: %d", error);
		}

		hv->chim_cnt = 0;
		/*
		 * Linger long enough for LWS to disconnect chimney
		 * sending buffer.
		 */
		rte_delay_ms(200);
	}
}

static int
hn_lws_conn_chim(struct hn_data *hv)
{
	struct hn_lws_chim_conn chim;
	struct hn_lws_chim_connresp resp;
	uint32_t sectsz;
	unsigned long len = hv->chim_res->len;
	int error;

	/* Connect chimney sending buffer to LWS */
	memset(&chim, 0, sizeof(chim));
	chim.type = LWS_TYPE_CHIM_CONN;
	chim.gpadl = hv->chim_res->phys_addr;
	chim.sig = LWS_CHIM_SIG;
	PMD_DRV_LOG(DEBUG, "connect send buf va=%p gpad=%#" PRIx64,
		    hv->chim_res->addr,
		    hv->chim_res->phys_addr);

	error = hn_lws_exelwte(hv, &chim, sizeof(chim),
			       &resp, sizeof(resp),
			       LWS_TYPE_CHIM_CONNRESP);
	if (error) {
		PMD_DRV_LOG(ERR, "exec lws chim conn failed");
		return error;
	}

	if (resp.status != LWS_STATUS_OK) {
		PMD_DRV_LOG(ERR, "lws chim conn failed: %x",
			    resp.status);
		return -EIO;
	}

	sectsz = resp.sectsz;
	if (sectsz == 0 || sectsz & (sizeof(uint32_t) - 1)) {
		/* Can't use chimney sending buffer; done! */
		PMD_DRV_LOG(NOTICE,
			    "invalid chimney sending buffer section size: %u",
			    sectsz);
		error = -EILWAL;
		goto cleanup;
	}

	hv->chim_szmax = sectsz;
	hv->chim_cnt = len / sectsz;

	PMD_DRV_LOG(INFO, "send buffer %lu section size:%u, count:%u",
		    len, hv->chim_szmax, hv->chim_cnt);

	/* Done! */
	return 0;

cleanup:
	hn_lws_disconn_chim(hv);
	return error;
}

/*
 * Configure MTU and enable VLAN.
 */
static int
hn_lws_conf_ndis(struct hn_data *hv, unsigned int mtu)
{
	struct hn_lws_ndis_conf conf;
	int error;

	memset(&conf, 0, sizeof(conf));
	conf.type = LWS_TYPE_NDIS_CONF;
	conf.mtu = mtu + RTE_ETHER_HDR_LEN;
	conf.caps = LWS_NDIS_CONF_VLAN;

	/* enable SRIOV */
	if (hv->lws_ver >= LWS_VERSION_5)
		conf.caps |= LWS_NDIS_CONF_SRIOV;

	/* NOTE: No response. */
	error = hn_lws_req_send(hv, &conf, sizeof(conf));
	if (error) {
		PMD_DRV_LOG(ERR,
			    "send lws ndis conf failed: %d", error);
		return error;
	}

	return 0;
}

static int
hn_lws_init_ndis(struct hn_data *hv)
{
	struct hn_lws_ndis_init ndis;
	int error;

	memset(&ndis, 0, sizeof(ndis));
	ndis.type = LWS_TYPE_NDIS_INIT;
	ndis.ndis_major = NDIS_VERSION_MAJOR(hv->ndis_ver);
	ndis.ndis_minor = NDIS_VERSION_MINOR(hv->ndis_ver);

	/* NOTE: No response. */
	error = hn_lws_req_send(hv, &ndis, sizeof(ndis));
	if (error)
		PMD_DRV_LOG(ERR,
			    "send lws ndis init failed: %d", error);

	return error;
}

static int
hn_lws_init(struct hn_data *hv)
{
	unsigned int i;
	int error;

	/*
	 * Find the supported LWS version and set NDIS version accordingly.
	 */
	for (i = 0; i < RTE_DIM(hn_lws_version); ++i) {
		error = hn_lws_doinit(hv, hn_lws_version[i]);
		if (error) {
			PMD_INIT_LOG(DEBUG, "version %#x error %d",
				     hn_lws_version[i], error);
			continue;
		}

		hv->lws_ver = hn_lws_version[i];

		/* Set NDIS version according to LWS version. */
		hv->ndis_ver = NDIS_VERSION_6_30;
		if (hv->lws_ver <= LWS_VERSION_4)
			hv->ndis_ver = NDIS_VERSION_6_1;

		PMD_INIT_LOG(DEBUG,
			     "LWS version %#x, NDIS version %u.%u",
			     hv->lws_ver, NDIS_VERSION_MAJOR(hv->ndis_ver),
			     NDIS_VERSION_MINOR(hv->ndis_ver));
		return 0;
	}

	PMD_DRV_LOG(ERR,
		    "no LWS compatible version available");
	return -ENXIO;
}

int
hn_lws_attach(struct hn_data *hv, unsigned int mtu)
{
	int error;

	/*
	 * Initialize LWS.
	 */
	error = hn_lws_init(hv);
	if (error)
		return error;

	/** Configure NDIS before initializing it. */
	if (hv->lws_ver >= LWS_VERSION_2) {
		error = hn_lws_conf_ndis(hv, mtu);
		if (error)
			return error;
	}

	/*
	 * Initialize NDIS.
	 */
	error = hn_lws_init_ndis(hv);
	if (error)
		return error;

	/*
	 * Connect RXBUF.
	 */
	error = hn_lws_conn_rxbuf(hv);
	if (error)
		return error;

	/*
	 * Connect chimney sending buffer.
	 */
	error = hn_lws_conn_chim(hv);
	if (error) {
		hn_lws_disconn_rxbuf(hv);
		return error;
	}

	return 0;
}

void
hn_lws_detach(struct hn_data *hv __rte_unused)
{
	PMD_INIT_FUNC_TRACE();

	/* NOTE: there are no requests to stop the LWS. */
	hn_lws_disconn_rxbuf(hv);
	hn_lws_disconn_chim(hv);
}

/*
 * Ack the consumed RXBUF associated w/ this channel packet,
 * so that this RXBUF can be recycled by the hypervisor.
 */
void
hn_lws_ack_rxbuf(struct vmbus_channel *chan, uint64_t tid)
{
	unsigned int retries = 0;
	struct hn_lws_rndis_ack ack = {
		.type = LWS_TYPE_RNDIS_ACK,
		.status = LWS_STATUS_OK,
	};
	int error;

	PMD_RX_LOG(DEBUG, "ack RX id %" PRIu64, tid);

 again:
	error = rte_vmbus_chan_send(chan, VMBUS_CHANPKT_TYPE_COMP,
				    &ack, sizeof(ack), tid,
				    VMBUS_CHANPKT_FLAG_NONE, NULL);

	if (error == 0)
		return;

	if (error == -EAGAIN) {
		/*
		 * NOTE:
		 * This should _not_ happen in real world, since the
		 * consumption of the TX bufring from the TX path is
		 * controlled.
		 */
		PMD_RX_LOG(NOTICE, "RXBUF ack retry");
		if (++retries < 10) {
			rte_delay_ms(1);
			goto again;
		}
	}
	/* RXBUF leaks! */
	PMD_DRV_LOG(ERR, "RXBUF ack failed");
}

int
hn_lws_alloc_subchans(struct hn_data *hv, uint32_t *nsubch)
{
	struct hn_lws_subch_req req;
	struct hn_lws_subch_resp resp;
	int error;

	memset(&req, 0, sizeof(req));
	req.type = LWS_TYPE_SUBCH_REQ;
	req.op = LWS_SUBCH_OP_ALLOC;
	req.nsubch = *nsubch;

	error = hn_lws_exelwte(hv, &req, sizeof(req),
			       &resp, sizeof(resp),
			       LWS_TYPE_SUBCH_RESP);
	if (error)
		return error;

	if (resp.status != LWS_STATUS_OK) {
		PMD_INIT_LOG(ERR,
			     "lws subch alloc failed: %#x",
			     resp.status);
		return -EIO;
	}

	if (resp.nsubch > *nsubch) {
		PMD_INIT_LOG(NOTICE,
			     "%u subchans are allocated, requested %u",
			     resp.nsubch, *nsubch);
	}
	*nsubch = resp.nsubch;

	return 0;
}

void
hn_lws_set_datapath(struct hn_data *hv, uint32_t path)
{
	struct hn_lws_datapath dp;
	int error;

	PMD_DRV_LOG(DEBUG, "set datapath %s",
		    path ? "VF" : "Synthetic");

	memset(&dp, 0, sizeof(dp));
	dp.type = LWS_TYPE_SET_DATAPATH;
	dp.active_path = path;

	error = hn_lws_req_send(hv, &dp, sizeof(dp));
	if (error) {
		PMD_DRV_LOG(ERR,
			    "send set datapath failed: %d",
			    error);
	}
}
