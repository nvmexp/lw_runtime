/* SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright(c) 2019-2020 Xilinx, Inc.
 * Copyright(c) 2012-2019 Solarflare Communications Inc.
 */

#include "efx.h"
#include "efx_impl.h"
#if EFSYS_OPT_MON_STATS
#include "mcdi_mon.h"
#endif

#if EFX_OPTS_EF10()

/*
 * Non-interrupting event queue requires interrrupting event queue to
 * refer to for wake-up events even if wake ups are never used.
 * It could be even non-allocated event queue.
 */
#define	EFX_EF10_ALWAYS_INTERRUPTING_EVQ_INDEX	(0)

static	__checkReturn	boolean_t
ef10_ev_rx(
	__in		efx_evq_t *eep,
	__in		efx_qword_t *eqp,
	__in		const efx_ev_callbacks_t *eecp,
	__in_opt	void *arg);

static	__checkReturn	boolean_t
ef10_ev_tx(
	__in		efx_evq_t *eep,
	__in		efx_qword_t *eqp,
	__in		const efx_ev_callbacks_t *eecp,
	__in_opt	void *arg);

static	__checkReturn	boolean_t
ef10_ev_driver(
	__in		efx_evq_t *eep,
	__in		efx_qword_t *eqp,
	__in		const efx_ev_callbacks_t *eecp,
	__in_opt	void *arg);

static	__checkReturn	boolean_t
ef10_ev_drv_gen(
	__in		efx_evq_t *eep,
	__in		efx_qword_t *eqp,
	__in		const efx_ev_callbacks_t *eecp,
	__in_opt	void *arg);


static	__checkReturn	efx_rc_t
efx_mcdi_set_evq_tmr(
	__in		efx_nic_t *enp,
	__in		uint32_t instance,
	__in		uint32_t mode,
	__in		uint32_t timer_ns)
{
	efx_mcdi_req_t req;
	EFX_MCDI_DECLARE_BUF(payload, MC_CMD_SET_EVQ_TMR_IN_LEN,
		MC_CMD_SET_EVQ_TMR_OUT_LEN);
	efx_rc_t rc;

	req.emr_cmd = MC_CMD_SET_EVQ_TMR;
	req.emr_in_buf = payload;
	req.emr_in_length = MC_CMD_SET_EVQ_TMR_IN_LEN;
	req.emr_out_buf = payload;
	req.emr_out_length = MC_CMD_SET_EVQ_TMR_OUT_LEN;

	MCDI_IN_SET_DWORD(req, SET_EVQ_TMR_IN_INSTANCE, instance);
	MCDI_IN_SET_DWORD(req, SET_EVQ_TMR_IN_TMR_LOAD_REQ_NS, timer_ns);
	MCDI_IN_SET_DWORD(req, SET_EVQ_TMR_IN_TMR_RELOAD_REQ_NS, timer_ns);
	MCDI_IN_SET_DWORD(req, SET_EVQ_TMR_IN_TMR_MODE, mode);

	efx_mcdi_exelwte(enp, &req);

	if (req.emr_rc != 0) {
		rc = req.emr_rc;
		goto fail1;
	}

	if (req.emr_out_length_used < MC_CMD_SET_EVQ_TMR_OUT_LEN) {
		rc = EMSGSIZE;
		goto fail2;
	}

	return (0);

fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}


	__checkReturn	efx_rc_t
ef10_ev_init(
	__in		efx_nic_t *enp)
{
	_NOTE(ARGUNUSED(enp))
	return (0);
}

			void
ef10_ev_fini(
	__in		efx_nic_t *enp)
{
	_NOTE(ARGUNUSED(enp))
}

	__checkReturn	efx_rc_t
ef10_ev_qcreate(
	__in		efx_nic_t *enp,
	__in		unsigned int index,
	__in		efsys_mem_t *esmp,
	__in		size_t ndescs,
	__in		uint32_t id,
	__in		uint32_t us,
	__in		uint32_t flags,
	__in		efx_evq_t *eep)
{
	efx_nic_cfg_t *encp = &(enp->en_nic_cfg);
	uint32_t irq;
	efx_rc_t rc;
	boolean_t low_latency;

	_NOTE(ARGUNUSED(id))	/* buftbl id managed by MC */

	EFSYS_ASSERT((flags & EFX_EVQ_FLAGS_EXTENDED_WIDTH) == 0);

	/*
	 * NO_CONT_EV mode is only requested from the firmware when creating
	 * receive queues, but here it needs to be specified at event queue
	 * creation, as the event handler needs to know which format is in use.
	 *
	 * If EFX_EVQ_FLAGS_NO_CONT_EV is specified, all receive queues for this
	 * event queue will be created in NO_CONT_EV mode.
	 *
	 * See SF-109306-TC 5.11 "Events for RXQs in NO_CONT_EV mode".
	 */
	if (flags & EFX_EVQ_FLAGS_NO_CONT_EV) {
		if (enp->en_nic_cfg.enc_no_cont_ev_mode_supported == B_FALSE) {
			rc = EILWAL;
			goto fail1;
		}
	}

	/* Set up the handler table */
	eep->ee_rx	= ef10_ev_rx;
	eep->ee_tx	= ef10_ev_tx;
	eep->ee_driver	= ef10_ev_driver;
	eep->ee_drv_gen	= ef10_ev_drv_gen;
	eep->ee_mcdi	= ef10_ev_mcdi;

	/* Set up the event queue */
	/* INIT_EVQ expects function-relative vector number */
	if ((flags & EFX_EVQ_FLAGS_NOTIFY_MASK) ==
	    EFX_EVQ_FLAGS_NOTIFY_INTERRUPT) {
		irq = index;
	} else if (index == EFX_EF10_ALWAYS_INTERRUPTING_EVQ_INDEX) {
		irq = index;
		flags = (flags & ~EFX_EVQ_FLAGS_NOTIFY_MASK) |
		    EFX_EVQ_FLAGS_NOTIFY_INTERRUPT;
	} else {
		irq = EFX_EF10_ALWAYS_INTERRUPTING_EVQ_INDEX;
	}

	/*
	 * Interrupts may be raised for events immediately after the queue is
	 * created. See bug58606.
	 */

	/*
	 * On Huntington we need to specify the settings to use.
	 * If event queue type in flags is auto, we favour throughput
	 * if the adapter is running virtualization supporting firmware
	 * (i.e. the full featured firmware variant)
	 * and latency otherwise. The Ethernet Virtual Bridging
	 * capability is used to make this decision. (Note though that
	 * the low latency firmware variant is also best for
	 * throughput and corresponding type should be specified
	 * to choose it.)
	 *
	 * If FW supports EvQ types (e.g. on Medford and Medford2) the
	 * type which is specified in flags is passed to FW to make the
	 * decision and low_latency hint is ignored.
	 */
	low_latency = encp->enc_datapath_cap_evb ? 0 : 1;
	rc = efx_mcdi_init_evq(enp, index, esmp, ndescs, irq, us, flags,
	    low_latency);
	if (rc != 0)
		goto fail2;

	return (0);

fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

			void
ef10_ev_qdestroy(
	__in		efx_evq_t *eep)
{
	efx_nic_t *enp = eep->ee_enp;

	EFSYS_ASSERT(EFX_FAMILY_IS_EF10(enp));

	(void) efx_mcdi_fini_evq(enp, eep->ee_index);
}

	__checkReturn	efx_rc_t
ef10_ev_qprime(
	__in		efx_evq_t *eep,
	__in		unsigned int count)
{
	efx_nic_t *enp = eep->ee_enp;
	uint32_t rptr;
	efx_dword_t dword;

	rptr = count & eep->ee_mask;

	if (enp->en_nic_cfg.enc_bug35388_workaround) {
		EFX_STATIC_ASSERT(EF10_EVQ_MINNEVS >
		    (1 << ERF_DD_EVQ_IND_RPTR_WIDTH));
		EFX_STATIC_ASSERT(EF10_EVQ_MAXNEVS <
		    (1 << 2 * ERF_DD_EVQ_IND_RPTR_WIDTH));

		EFX_POPULATE_DWORD_2(dword,
		    ERF_DD_EVQ_IND_RPTR_FLAGS,
		    EFE_DD_EVQ_IND_RPTR_FLAGS_HIGH,
		    ERF_DD_EVQ_IND_RPTR,
		    (rptr >> ERF_DD_EVQ_IND_RPTR_WIDTH));
		EFX_BAR_VI_WRITED(enp, ER_DD_EVQ_INDIRECT, eep->ee_index,
		    &dword, B_FALSE);

		EFX_POPULATE_DWORD_2(dword,
		    ERF_DD_EVQ_IND_RPTR_FLAGS,
		    EFE_DD_EVQ_IND_RPTR_FLAGS_LOW,
		    ERF_DD_EVQ_IND_RPTR,
		    rptr & ((1 << ERF_DD_EVQ_IND_RPTR_WIDTH) - 1));
		EFX_BAR_VI_WRITED(enp, ER_DD_EVQ_INDIRECT, eep->ee_index,
		    &dword, B_FALSE);
	} else {
		EFX_POPULATE_DWORD_1(dword, ERF_DZ_EVQ_RPTR, rptr);
		EFX_BAR_VI_WRITED(enp, ER_DZ_EVQ_RPTR_REG, eep->ee_index,
		    &dword, B_FALSE);
	}

	return (0);
}

static	__checkReturn	efx_rc_t
efx_mcdi_driver_event(
	__in		efx_nic_t *enp,
	__in		uint32_t evq,
	__in		efx_qword_t data)
{
	efx_mcdi_req_t req;
	EFX_MCDI_DECLARE_BUF(payload, MC_CMD_DRIVER_EVENT_IN_LEN,
		MC_CMD_DRIVER_EVENT_OUT_LEN);
	efx_rc_t rc;

	req.emr_cmd = MC_CMD_DRIVER_EVENT;
	req.emr_in_buf = payload;
	req.emr_in_length = MC_CMD_DRIVER_EVENT_IN_LEN;
	req.emr_out_buf = payload;
	req.emr_out_length = MC_CMD_DRIVER_EVENT_OUT_LEN;

	MCDI_IN_SET_DWORD(req, DRIVER_EVENT_IN_EVQ, evq);

	MCDI_IN_SET_DWORD(req, DRIVER_EVENT_IN_DATA_LO,
	    EFX_QWORD_FIELD(data, EFX_DWORD_0));
	MCDI_IN_SET_DWORD(req, DRIVER_EVENT_IN_DATA_HI,
	    EFX_QWORD_FIELD(data, EFX_DWORD_1));

	efx_mcdi_exelwte(enp, &req);

	if (req.emr_rc != 0) {
		rc = req.emr_rc;
		goto fail1;
	}

	return (0);

fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

			void
ef10_ev_qpost(
	__in	efx_evq_t *eep,
	__in	uint16_t data)
{
	efx_nic_t *enp = eep->ee_enp;
	efx_qword_t event;

	EFX_POPULATE_QWORD_3(event,
	    ESF_DZ_DRV_CODE, ESE_DZ_EV_CODE_DRV_GEN_EV,
	    ESF_DZ_DRV_SUB_CODE, 0,
	    ESF_DZ_DRV_SUB_DATA_DW0, (uint32_t)data);

	(void) efx_mcdi_driver_event(enp, eep->ee_index, event);
}

	__checkReturn	efx_rc_t
ef10_ev_qmoderate(
	__in		efx_evq_t *eep,
	__in		unsigned int us)
{
	efx_nic_t *enp = eep->ee_enp;
	efx_nic_cfg_t *encp = &(enp->en_nic_cfg);
	efx_dword_t dword;
	uint32_t mode;
	efx_rc_t rc;

	/* Check that hardware and MCDI use the same timer MODE values */
	EFX_STATIC_ASSERT(FFE_CZ_TIMER_MODE_DIS ==
	    MC_CMD_SET_EVQ_TMR_IN_TIMER_MODE_DIS);
	EFX_STATIC_ASSERT(FFE_CZ_TIMER_MODE_IMMED_START ==
	    MC_CMD_SET_EVQ_TMR_IN_TIMER_MODE_IMMED_START);
	EFX_STATIC_ASSERT(FFE_CZ_TIMER_MODE_TRIG_START ==
	    MC_CMD_SET_EVQ_TMR_IN_TIMER_MODE_TRIG_START);
	EFX_STATIC_ASSERT(FFE_CZ_TIMER_MODE_INT_HLDOFF ==
	    MC_CMD_SET_EVQ_TMR_IN_TIMER_MODE_INT_HLDOFF);

	if (us > encp->enc_evq_timer_max_us) {
		rc = EILWAL;
		goto fail1;
	}

	/* If the value is zero then disable the timer */
	if (us == 0) {
		mode = FFE_CZ_TIMER_MODE_DIS;
	} else {
		mode = FFE_CZ_TIMER_MODE_INT_HLDOFF;
	}

	if (encp->enc_bug61265_workaround) {
		uint32_t ns = us * 1000;

		rc = efx_mcdi_set_evq_tmr(enp, eep->ee_index, mode, ns);
		if (rc != 0)
			goto fail2;
	} else {
		unsigned int ticks;

		if ((rc = efx_ev_usecs_to_ticks(enp, us, &ticks)) != 0)
			goto fail3;

		if (encp->enc_bug35388_workaround) {
			EFX_POPULATE_DWORD_3(dword,
			    ERF_DD_EVQ_IND_TIMER_FLAGS,
			    EFE_DD_EVQ_IND_TIMER_FLAGS,
			    ERF_DD_EVQ_IND_TIMER_MODE, mode,
			    ERF_DD_EVQ_IND_TIMER_VAL, ticks);
			EFX_BAR_VI_WRITED(enp, ER_DD_EVQ_INDIRECT,
			    eep->ee_index, &dword, 0);
		} else {
			/*
			 * NOTE: The TMR_REL field introduced in Medford2 is
			 * ignored on earlier EF10 controllers. See bug66418
			 * comment 9 for details.
			 */
			EFX_POPULATE_DWORD_3(dword,
			    ERF_DZ_TC_TIMER_MODE, mode,
			    ERF_DZ_TC_TIMER_VAL, ticks,
			    ERF_FZ_TC_TMR_REL_VAL, ticks);
			EFX_BAR_VI_WRITED(enp, ER_DZ_EVQ_TMR_REG,
			    eep->ee_index, &dword, 0);
		}
	}

	return (0);

fail3:
	EFSYS_PROBE(fail3);
fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}


#if EFSYS_OPT_QSTATS
			void
ef10_ev_qstats_update(
	__in				efx_evq_t *eep,
	__inout_ecount(EV_NQSTATS)	efsys_stat_t *stat)
{
	unsigned int id;

	for (id = 0; id < EV_NQSTATS; id++) {
		efsys_stat_t *essp = &stat[id];

		EFSYS_STAT_INCR(essp, eep->ee_stat[id]);
		eep->ee_stat[id] = 0;
	}
}
#endif /* EFSYS_OPT_QSTATS */

#if EFSYS_OPT_RX_PACKED_STREAM || EFSYS_OPT_RX_ES_SUPER_BUFFER

static	__checkReturn	boolean_t
ef10_ev_rx_packed_stream(
	__in		efx_evq_t *eep,
	__in		efx_qword_t *eqp,
	__in		const efx_ev_callbacks_t *eecp,
	__in_opt	void *arg)
{
	uint32_t label;
	uint32_t pkt_count_lbits;
	uint16_t flags;
	boolean_t should_abort;
	efx_evq_rxq_state_t *eersp;
	unsigned int pkt_count;
	unsigned int lwrrent_id;
	boolean_t new_buffer;

	pkt_count_lbits = EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_DSC_PTR_LBITS);
	label = EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_QLABEL);
	new_buffer = EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_EV_ROTATE);

	flags = 0;

	eersp = &eep->ee_rxq_state[label];

	/*
	 * RX_DSC_PTR_LBITS has least significant bits of the global
	 * (not per-buffer) packet counter. It is guaranteed that
	 * maximum number of completed packets fits in lbits-mask.
	 * So, modulo lbits-mask arithmetic should be used to callwlate
	 * packet counter increment.
	 */
	pkt_count = (pkt_count_lbits - eersp->eers_rx_stream_npackets) &
	    EFX_MASK32(ESF_DZ_RX_DSC_PTR_LBITS);
	eersp->eers_rx_stream_npackets += pkt_count;

	if (new_buffer) {
		flags |= EFX_PKT_PACKED_STREAM_NEW_BUFFER;
#if EFSYS_OPT_RX_PACKED_STREAM
		/*
		 * If both packed stream and equal stride super-buffer
		 * modes are compiled in, in theory credits should be
		 * be maintained for packed stream only, but right now
		 * these modes are not distinguished in the event queue
		 * Rx queue state and it is OK to increment the counter
		 * regardless (it might be event cheaper than branching
		 * since neighbour structure member are updated as well).
		 */
		eersp->eers_rx_packed_stream_credits++;
#endif
		eersp->eers_rx_read_ptr++;
	}
	lwrrent_id = eersp->eers_rx_read_ptr & eersp->eers_rx_mask;

	/* Check for errors that ilwalidate checksum and L3/L4 fields */
	if (EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_TRUNC_ERR) != 0) {
		/* RX frame truncated */
		EFX_EV_QSTAT_INCR(eep, EV_RX_FRM_TRUNC);
		flags |= EFX_DISCARD;
		goto deliver;
	}
	if (EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_ECRC_ERR) != 0) {
		/* Bad Ethernet frame CRC */
		EFX_EV_QSTAT_INCR(eep, EV_RX_ETH_CRC_ERR);
		flags |= EFX_DISCARD;
		goto deliver;
	}

	if (EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_PARSE_INCOMPLETE)) {
		EFX_EV_QSTAT_INCR(eep, EV_RX_PARSE_INCOMPLETE);
		flags |= EFX_PKT_PACKED_STREAM_PARSE_INCOMPLETE;
		goto deliver;
	}

	if (EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_IPCKSUM_ERR))
		EFX_EV_QSTAT_INCR(eep, EV_RX_IPV4_HDR_CHKSUM_ERR);

	if (EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_TCPUDP_CKSUM_ERR))
		EFX_EV_QSTAT_INCR(eep, EV_RX_TCP_UDP_CHKSUM_ERR);

deliver:
	/* If we're not discarding the packet then it is ok */
	if (~flags & EFX_DISCARD)
		EFX_EV_QSTAT_INCR(eep, EV_RX_OK);

	EFSYS_ASSERT(eecp->eec_rx_ps != NULL);
	should_abort = eecp->eec_rx_ps(arg, label, lwrrent_id, pkt_count,
	    flags);

	return (should_abort);
}

#endif /* EFSYS_OPT_RX_PACKED_STREAM || EFSYS_OPT_RX_ES_SUPER_BUFFER */

static	__checkReturn	boolean_t
ef10_ev_rx(
	__in		efx_evq_t *eep,
	__in		efx_qword_t *eqp,
	__in		const efx_ev_callbacks_t *eecp,
	__in_opt	void *arg)
{
	efx_nic_t *enp = eep->ee_enp;
	uint32_t size;
	uint32_t label;
	uint32_t mac_class;
	uint32_t eth_tag_class;
	uint32_t l3_class;
	uint32_t l4_class;
	uint32_t next_read_lbits;
	uint16_t flags;
	boolean_t cont;
	boolean_t should_abort;
	efx_evq_rxq_state_t *eersp;
	unsigned int desc_count;
	unsigned int last_used_id;

	EFX_EV_QSTAT_INCR(eep, EV_RX);

	/* Discard events after RXQ/TXQ errors, or hardware not available */
	if (enp->en_reset_flags &
	    (EFX_RESET_RXQ_ERR | EFX_RESET_TXQ_ERR | EFX_RESET_HW_UNAVAIL))
		return (B_FALSE);

	/* Basic packet information */
	label = EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_QLABEL);
	eersp = &eep->ee_rxq_state[label];

#if EFSYS_OPT_RX_PACKED_STREAM || EFSYS_OPT_RX_ES_SUPER_BUFFER
	/*
	 * Packed stream events are very different,
	 * so handle them separately
	 */
	if (eersp->eers_rx_packed_stream)
		return (ef10_ev_rx_packed_stream(eep, eqp, eecp, arg));
#endif

	size = EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_BYTES);
	cont = EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_CONT);
	next_read_lbits = EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_DSC_PTR_LBITS);
	eth_tag_class = EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_ETH_TAG_CLASS);
	mac_class = EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_MAC_CLASS);
	l3_class = EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_L3_CLASS);

	/*
	 * RX_L4_CLASS is 3 bits wide on Huntington and Medford, but is only
	 * 2 bits wide on Medford2. Check it is safe to use the Medford2 field
	 * and values for all EF10 controllers.
	 */
	EFX_STATIC_ASSERT(ESF_FZ_RX_L4_CLASS_LBN == ESF_DE_RX_L4_CLASS_LBN);
	EFX_STATIC_ASSERT(ESE_FZ_L4_CLASS_TCP == ESE_DE_L4_CLASS_TCP);
	EFX_STATIC_ASSERT(ESE_FZ_L4_CLASS_UDP == ESE_DE_L4_CLASS_UDP);
	EFX_STATIC_ASSERT(ESE_FZ_L4_CLASS_UNKNOWN == ESE_DE_L4_CLASS_UNKNOWN);

	l4_class = EFX_QWORD_FIELD(*eqp, ESF_FZ_RX_L4_CLASS);

	if (EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_DROP_EVENT) != 0) {
		/* Drop this event */
		return (B_FALSE);
	}
	flags = 0;

	if (cont != 0) {
		/*
		 * This may be part of a scattered frame, or it may be a
		 * truncated frame if scatter is disabled on this RXQ.
		 * Overlength frames can be received if e.g. a VF is configured
		 * for 1500 MTU but connected to a port set to 9000 MTU
		 * (see bug56567).
		 * FIXME: There is not yet any driver that supports scatter on
		 * Huntington.  Scatter support is required for OSX.
		 */
		flags |= EFX_PKT_CONT;
	}

	if (mac_class == ESE_DZ_MAC_CLASS_UCAST)
		flags |= EFX_PKT_UNICAST;

	/*
	 * Increment the count of descriptors read.
	 *
	 * In NO_CONT_EV mode, RX_DSC_PTR_LBITS is actually a packet count, but
	 * when scatter is disabled, there is only one descriptor per packet and
	 * so it can be treated the same.
	 *
	 * TODO: Support scatter in NO_CONT_EV mode.
	 */
	desc_count = (next_read_lbits - eersp->eers_rx_read_ptr) &
	    EFX_MASK32(ESF_DZ_RX_DSC_PTR_LBITS);
	eersp->eers_rx_read_ptr += desc_count;

	/* Callwlate the index of the last descriptor consumed */
	last_used_id = (eersp->eers_rx_read_ptr - 1) & eersp->eers_rx_mask;

	if (eep->ee_flags & EFX_EVQ_FLAGS_NO_CONT_EV) {
		if (desc_count > 1)
			EFX_EV_QSTAT_INCR(eep, EV_RX_BATCH);

		/* Always read the length from the prefix in NO_CONT_EV mode. */
		flags |= EFX_PKT_PREFIX_LEN;

		/*
		 * Check for an aborted scatter, signalled by the ABORT bit in
		 * NO_CONT_EV mode. The ABORT bit was not used before NO_CONT_EV
		 * mode was added as it was broken in Huntington silicon.
		 */
		if (EFX_QWORD_FIELD(*eqp, ESF_EZ_RX_ABORT) != 0) {
			flags |= EFX_DISCARD;
			goto deliver;
		}
	} else if (desc_count > 1) {
		/*
		 * FIXME: add error checking to make sure this a batched event.
		 * This could also be an aborted scatter, see Bug36629.
		 */
		EFX_EV_QSTAT_INCR(eep, EV_RX_BATCH);
		flags |= EFX_PKT_PREFIX_LEN;
	}

	/* Check for errors that ilwalidate checksum and L3/L4 fields */
	if (EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_TRUNC_ERR) != 0) {
		/* RX frame truncated */
		EFX_EV_QSTAT_INCR(eep, EV_RX_FRM_TRUNC);
		flags |= EFX_DISCARD;
		goto deliver;
	}
	if (EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_ECRC_ERR) != 0) {
		/* Bad Ethernet frame CRC */
		EFX_EV_QSTAT_INCR(eep, EV_RX_ETH_CRC_ERR);
		flags |= EFX_DISCARD;
		goto deliver;
	}
	if (EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_PARSE_INCOMPLETE)) {
		/*
		 * Hardware parse failed, due to malformed headers
		 * or headers that are too long for the parser.
		 * Headers and checksums must be validated by the host.
		 */
		EFX_EV_QSTAT_INCR(eep, EV_RX_PARSE_INCOMPLETE);
		goto deliver;
	}

	if ((eth_tag_class == ESE_DZ_ETH_TAG_CLASS_VLAN1) ||
	    (eth_tag_class == ESE_DZ_ETH_TAG_CLASS_VLAN2)) {
		flags |= EFX_PKT_VLAN_TAGGED;
	}

	switch (l3_class) {
	case ESE_DZ_L3_CLASS_IP4:
	case ESE_DZ_L3_CLASS_IP4_FRAG:
		flags |= EFX_PKT_IPV4;
		if (EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_IPCKSUM_ERR)) {
			EFX_EV_QSTAT_INCR(eep, EV_RX_IPV4_HDR_CHKSUM_ERR);
		} else {
			flags |= EFX_CKSUM_IPV4;
		}

		/*
		 * RX_L4_CLASS is 3 bits wide on Huntington and Medford, but is
		 * only 2 bits wide on Medford2. Check it is safe to use the
		 * Medford2 field and values for all EF10 controllers.
		 */
		EFX_STATIC_ASSERT(ESF_FZ_RX_L4_CLASS_LBN ==
		    ESF_DE_RX_L4_CLASS_LBN);
		EFX_STATIC_ASSERT(ESE_FZ_L4_CLASS_TCP == ESE_DE_L4_CLASS_TCP);
		EFX_STATIC_ASSERT(ESE_FZ_L4_CLASS_UDP == ESE_DE_L4_CLASS_UDP);
		EFX_STATIC_ASSERT(ESE_FZ_L4_CLASS_UNKNOWN ==
		    ESE_DE_L4_CLASS_UNKNOWN);

		if (l4_class == ESE_FZ_L4_CLASS_TCP) {
			EFX_EV_QSTAT_INCR(eep, EV_RX_TCP_IPV4);
			flags |= EFX_PKT_TCP;
		} else if (l4_class == ESE_FZ_L4_CLASS_UDP) {
			EFX_EV_QSTAT_INCR(eep, EV_RX_UDP_IPV4);
			flags |= EFX_PKT_UDP;
		} else {
			EFX_EV_QSTAT_INCR(eep, EV_RX_OTHER_IPV4);
		}
		break;

	case ESE_DZ_L3_CLASS_IP6:
	case ESE_DZ_L3_CLASS_IP6_FRAG:
		flags |= EFX_PKT_IPV6;

		/*
		 * RX_L4_CLASS is 3 bits wide on Huntington and Medford, but is
		 * only 2 bits wide on Medford2. Check it is safe to use the
		 * Medford2 field and values for all EF10 controllers.
		 */
		EFX_STATIC_ASSERT(ESF_FZ_RX_L4_CLASS_LBN ==
		    ESF_DE_RX_L4_CLASS_LBN);
		EFX_STATIC_ASSERT(ESE_FZ_L4_CLASS_TCP == ESE_DE_L4_CLASS_TCP);
		EFX_STATIC_ASSERT(ESE_FZ_L4_CLASS_UDP == ESE_DE_L4_CLASS_UDP);
		EFX_STATIC_ASSERT(ESE_FZ_L4_CLASS_UNKNOWN ==
		    ESE_DE_L4_CLASS_UNKNOWN);

		if (l4_class == ESE_FZ_L4_CLASS_TCP) {
			EFX_EV_QSTAT_INCR(eep, EV_RX_TCP_IPV6);
			flags |= EFX_PKT_TCP;
		} else if (l4_class == ESE_FZ_L4_CLASS_UDP) {
			EFX_EV_QSTAT_INCR(eep, EV_RX_UDP_IPV6);
			flags |= EFX_PKT_UDP;
		} else {
			EFX_EV_QSTAT_INCR(eep, EV_RX_OTHER_IPV6);
		}
		break;

	default:
		EFX_EV_QSTAT_INCR(eep, EV_RX_NON_IP);
		break;
	}

	if (flags & (EFX_PKT_TCP | EFX_PKT_UDP)) {
		if (EFX_QWORD_FIELD(*eqp, ESF_DZ_RX_TCPUDP_CKSUM_ERR)) {
			EFX_EV_QSTAT_INCR(eep, EV_RX_TCP_UDP_CHKSUM_ERR);
		} else {
			flags |= EFX_CKSUM_TCPUDP;
		}
	}

deliver:
	/* If we're not discarding the packet then it is ok */
	if (~flags & EFX_DISCARD)
		EFX_EV_QSTAT_INCR(eep, EV_RX_OK);

	EFSYS_ASSERT(eecp->eec_rx != NULL);
	should_abort = eecp->eec_rx(arg, label, last_used_id, size, flags);

	return (should_abort);
}

static	__checkReturn	boolean_t
ef10_ev_tx(
	__in		efx_evq_t *eep,
	__in		efx_qword_t *eqp,
	__in		const efx_ev_callbacks_t *eecp,
	__in_opt	void *arg)
{
	efx_nic_t *enp = eep->ee_enp;
	uint32_t id;
	uint32_t label;
	boolean_t should_abort;

	EFX_EV_QSTAT_INCR(eep, EV_TX);

	/* Discard events after RXQ/TXQ errors, or hardware not available */
	if (enp->en_reset_flags &
	    (EFX_RESET_RXQ_ERR | EFX_RESET_TXQ_ERR | EFX_RESET_HW_UNAVAIL))
		return (B_FALSE);

	if (EFX_QWORD_FIELD(*eqp, ESF_DZ_TX_DROP_EVENT) != 0) {
		/* Drop this event */
		return (B_FALSE);
	}

	/* Per-packet TX completion (was per-descriptor for Falcon/Siena) */
	id = EFX_QWORD_FIELD(*eqp, ESF_DZ_TX_DESCR_INDX);
	label = EFX_QWORD_FIELD(*eqp, ESF_DZ_TX_QLABEL);

	EFSYS_PROBE2(tx_complete, uint32_t, label, uint32_t, id);

	EFSYS_ASSERT(eecp->eec_tx != NULL);
	should_abort = eecp->eec_tx(arg, label, id);

	return (should_abort);
}

static	__checkReturn	boolean_t
ef10_ev_driver(
	__in		efx_evq_t *eep,
	__in		efx_qword_t *eqp,
	__in		const efx_ev_callbacks_t *eecp,
	__in_opt	void *arg)
{
	unsigned int code;
	boolean_t should_abort;

	EFX_EV_QSTAT_INCR(eep, EV_DRIVER);
	should_abort = B_FALSE;

	code = EFX_QWORD_FIELD(*eqp, ESF_DZ_DRV_SUB_CODE);
	switch (code) {
	case ESE_DZ_DRV_TIMER_EV: {
		uint32_t id;

		id = EFX_QWORD_FIELD(*eqp, ESF_DZ_DRV_TMR_ID);

		EFSYS_ASSERT(eecp->eec_timer != NULL);
		should_abort = eecp->eec_timer(arg, id);
		break;
	}

	case ESE_DZ_DRV_WAKE_UP_EV: {
		uint32_t id;

		id = EFX_QWORD_FIELD(*eqp, ESF_DZ_DRV_EVQ_ID);

		EFSYS_ASSERT(eecp->eec_wake_up != NULL);
		should_abort = eecp->eec_wake_up(arg, id);
		break;
	}

	case ESE_DZ_DRV_START_UP_EV:
		EFSYS_ASSERT(eecp->eec_initialized != NULL);
		should_abort = eecp->eec_initialized(arg);
		break;

	default:
		EFSYS_PROBE3(bad_event, unsigned int, eep->ee_index,
		    uint32_t, EFX_QWORD_FIELD(*eqp, EFX_DWORD_1),
		    uint32_t, EFX_QWORD_FIELD(*eqp, EFX_DWORD_0));
		break;
	}

	return (should_abort);
}

static	__checkReturn	boolean_t
ef10_ev_drv_gen(
	__in		efx_evq_t *eep,
	__in		efx_qword_t *eqp,
	__in		const efx_ev_callbacks_t *eecp,
	__in_opt	void *arg)
{
	uint32_t data;
	boolean_t should_abort;

	EFX_EV_QSTAT_INCR(eep, EV_DRV_GEN);
	should_abort = B_FALSE;

	data = EFX_QWORD_FIELD(*eqp, ESF_DZ_DRV_SUB_DATA_DW0);
	if (data >= ((uint32_t)1 << 16)) {
		EFSYS_PROBE3(bad_event, unsigned int, eep->ee_index,
		    uint32_t, EFX_QWORD_FIELD(*eqp, EFX_DWORD_1),
		    uint32_t, EFX_QWORD_FIELD(*eqp, EFX_DWORD_0));

		return (B_TRUE);
	}

	EFSYS_ASSERT(eecp->eec_software != NULL);
	should_abort = eecp->eec_software(arg, (uint16_t)data);

	return (should_abort);
}

#endif	/* EFX_OPTS_EF10() */

#if EFSYS_OPT_RIVERHEAD || EFX_OPTS_EF10()

	__checkReturn	boolean_t
ef10_ev_mcdi(
	__in		efx_evq_t *eep,
	__in		efx_qword_t *eqp,
	__in		const efx_ev_callbacks_t *eecp,
	__in_opt	void *arg)
{
	efx_nic_t *enp = eep->ee_enp;
	unsigned int code;
	boolean_t should_abort = B_FALSE;

	EFX_EV_QSTAT_INCR(eep, EV_MCDI_RESPONSE);

	code = EFX_QWORD_FIELD(*eqp, MCDI_EVENT_CODE);
	switch (code) {
	case MCDI_EVENT_CODE_BADSSERT:
		efx_mcdi_ev_death(enp, EINTR);
		break;

	case MCDI_EVENT_CODE_CMDDONE:
		efx_mcdi_ev_cpl(enp,
		    MCDI_EV_FIELD(eqp, CMDDONE_SEQ),
		    MCDI_EV_FIELD(eqp, CMDDONE_DATALEN),
		    MCDI_EV_FIELD(eqp, CMDDONE_ERRNO));
		break;

#if EFSYS_OPT_MCDI_PROXY_AUTH
	case MCDI_EVENT_CODE_PROXY_RESPONSE:
		/*
		 * This event notifies a function that an authorization request
		 * has been processed. If the request was authorized then the
		 * function can now re-send the original MCDI request.
		 * See SF-113652-SW "SR-IOV Proxied Network Access Control".
		 */
		efx_mcdi_ev_proxy_response(enp,
		    MCDI_EV_FIELD(eqp, PROXY_RESPONSE_HANDLE),
		    MCDI_EV_FIELD(eqp, PROXY_RESPONSE_RC));
		break;
#endif /* EFSYS_OPT_MCDI_PROXY_AUTH */

#if EFSYS_OPT_MCDI_PROXY_AUTH_SERVER
	case MCDI_EVENT_CODE_PROXY_REQUEST:
		efx_mcdi_ev_proxy_request(enp,
			MCDI_EV_FIELD(eqp, PROXY_REQUEST_BUFF_INDEX));
		break;
#endif /* EFSYS_OPT_MCDI_PROXY_AUTH_SERVER */

	case MCDI_EVENT_CODE_LINKCHANGE: {
		efx_link_mode_t link_mode;

		ef10_phy_link_ev(enp, eqp, &link_mode);
		should_abort = eecp->eec_link_change(arg, link_mode);
		break;
	}

	case MCDI_EVENT_CODE_SENSOREVT: {
#if EFSYS_OPT_MON_STATS
		efx_mon_stat_t id;
		efx_mon_stat_value_t value;
		efx_rc_t rc;

		/* Decode monitor stat for MCDI sensor (if supported) */
		if ((rc = mcdi_mon_ev(enp, eqp, &id, &value)) == 0) {
			/* Report monitor stat change */
			should_abort = eecp->eec_monitor(arg, id, value);
		} else if (rc == ENOTSUP) {
			should_abort = eecp->eec_exception(arg,
				EFX_EXCEPTION_UNKNOWN_SENSOREVT,
				MCDI_EV_FIELD(eqp, DATA));
		} else {
			EFSYS_ASSERT(rc == ENODEV);	/* Wrong port */
		}
#endif
		break;
	}

	case MCDI_EVENT_CODE_SCHEDERR:
		/* Informational only */
		break;

	case MCDI_EVENT_CODE_REBOOT:
		/* Falcon/Siena only (should not been seen with Huntington). */
		efx_mcdi_ev_death(enp, EIO);
		break;

	case MCDI_EVENT_CODE_MC_REBOOT:
		/* MC_REBOOT event is used for Huntington (EF10) and later. */
		efx_mcdi_ev_death(enp, EIO);
		break;

	case MCDI_EVENT_CODE_MAC_STATS_DMA:
#if EFSYS_OPT_MAC_STATS
		if (eecp->eec_mac_stats != NULL) {
			eecp->eec_mac_stats(arg,
			    MCDI_EV_FIELD(eqp, MAC_STATS_DMA_GENERATION));
		}
#endif
		break;

	case MCDI_EVENT_CODE_FWALERT: {
		uint32_t reason = MCDI_EV_FIELD(eqp, FWALERT_REASON);

		if (reason == MCDI_EVENT_FWALERT_REASON_SRAM_ACCESS)
			should_abort = eecp->eec_exception(arg,
				EFX_EXCEPTION_FWALERT_SRAM,
				MCDI_EV_FIELD(eqp, FWALERT_DATA));
		else
			should_abort = eecp->eec_exception(arg,
				EFX_EXCEPTION_UNKNOWN_FWALERT,
				MCDI_EV_FIELD(eqp, DATA));
		break;
	}

	case MCDI_EVENT_CODE_TX_ERR: {
		/*
		 * After a TXQ error is detected, firmware sends a TX_ERR event.
		 * This may be followed by TX completions (which we discard),
		 * and then finally by a TX_FLUSH event. Firmware destroys the
		 * TXQ automatically after sending the TX_FLUSH event.
		 */
		enp->en_reset_flags |= EFX_RESET_TXQ_ERR;

		EFSYS_PROBE2(tx_descq_err,
			    uint32_t, EFX_QWORD_FIELD(*eqp, EFX_DWORD_1),
			    uint32_t, EFX_QWORD_FIELD(*eqp, EFX_DWORD_0));

		/* Inform the driver that a reset is required. */
		eecp->eec_exception(arg, EFX_EXCEPTION_TX_ERROR,
		    MCDI_EV_FIELD(eqp, TX_ERR_DATA));
		break;
	}

	case MCDI_EVENT_CODE_TX_FLUSH: {
		uint32_t txq_index = MCDI_EV_FIELD(eqp, TX_FLUSH_TXQ);

		/*
		 * EF10 firmware sends two TX_FLUSH events: one to the txq's
		 * event queue, and one to evq 0 (with TX_FLUSH_TO_DRIVER set).
		 * We want to wait for all completions, so ignore the events
		 * with TX_FLUSH_TO_DRIVER.
		 */
		if (MCDI_EV_FIELD(eqp, TX_FLUSH_TO_DRIVER) != 0) {
			should_abort = B_FALSE;
			break;
		}

		EFX_EV_QSTAT_INCR(eep, EV_DRIVER_TX_DESCQ_FLS_DONE);

		EFSYS_PROBE1(tx_descq_fls_done, uint32_t, txq_index);

		EFSYS_ASSERT(eecp->eec_txq_flush_done != NULL);
		should_abort = eecp->eec_txq_flush_done(arg, txq_index);
		break;
	}

	case MCDI_EVENT_CODE_RX_ERR: {
		/*
		 * After an RXQ error is detected, firmware sends an RX_ERR
		 * event. This may be followed by RX events (which we discard),
		 * and then finally by an RX_FLUSH event. Firmware destroys the
		 * RXQ automatically after sending the RX_FLUSH event.
		 */
		enp->en_reset_flags |= EFX_RESET_RXQ_ERR;

		EFSYS_PROBE2(rx_descq_err,
			    uint32_t, EFX_QWORD_FIELD(*eqp, EFX_DWORD_1),
			    uint32_t, EFX_QWORD_FIELD(*eqp, EFX_DWORD_0));

		/* Inform the driver that a reset is required. */
		eecp->eec_exception(arg, EFX_EXCEPTION_RX_ERROR,
		    MCDI_EV_FIELD(eqp, RX_ERR_DATA));
		break;
	}

	case MCDI_EVENT_CODE_RX_FLUSH: {
		uint32_t rxq_index = MCDI_EV_FIELD(eqp, RX_FLUSH_RXQ);

		/*
		 * EF10 firmware sends two RX_FLUSH events: one to the rxq's
		 * event queue, and one to evq 0 (with RX_FLUSH_TO_DRIVER set).
		 * We want to wait for all completions, so ignore the events
		 * with RX_FLUSH_TO_DRIVER.
		 */
		if (MCDI_EV_FIELD(eqp, RX_FLUSH_TO_DRIVER) != 0) {
			should_abort = B_FALSE;
			break;
		}

		EFX_EV_QSTAT_INCR(eep, EV_DRIVER_RX_DESCQ_FLS_DONE);

		EFSYS_PROBE1(rx_descq_fls_done, uint32_t, rxq_index);

		EFSYS_ASSERT(eecp->eec_rxq_flush_done != NULL);
		should_abort = eecp->eec_rxq_flush_done(arg, rxq_index);
		break;
	}

	default:
		EFSYS_PROBE3(bad_event, unsigned int, eep->ee_index,
		    uint32_t, EFX_QWORD_FIELD(*eqp, EFX_DWORD_1),
		    uint32_t, EFX_QWORD_FIELD(*eqp, EFX_DWORD_0));
		break;
	}

	return (should_abort);
}

#endif	/* EFSYS_OPT_RIVERHEAD || EFX_OPTS_EF10() */

#if EFX_OPTS_EF10()

		void
ef10_ev_rxlabel_init(
	__in		efx_evq_t *eep,
	__in		efx_rxq_t *erp,
	__in		unsigned int label,
	__in		efx_rxq_type_t type)
{
	efx_evq_rxq_state_t *eersp;
#if EFSYS_OPT_RX_PACKED_STREAM || EFSYS_OPT_RX_ES_SUPER_BUFFER
	boolean_t packed_stream = (type == EFX_RXQ_TYPE_PACKED_STREAM);
	boolean_t es_super_buffer = (type == EFX_RXQ_TYPE_ES_SUPER_BUFFER);
#endif

	_NOTE(ARGUNUSED(type))
	EFSYS_ASSERT3U(label, <, EFX_ARRAY_SIZE(eep->ee_rxq_state));
	eersp = &eep->ee_rxq_state[label];

	EFSYS_ASSERT3U(eersp->eers_rx_mask, ==, 0);

#if EFSYS_OPT_RX_PACKED_STREAM
	/*
	 * For packed stream modes, the very first event will
	 * have a new buffer flag set, so it will be incremented,
	 * yielding the correct pointer. That results in a simpler
	 * code than trying to detect start-of-the-world condition
	 * in the event handler.
	 */
	eersp->eers_rx_read_ptr = packed_stream ? ~0 : 0;
#else
	eersp->eers_rx_read_ptr = 0;
#endif
	eersp->eers_rx_mask = erp->er_mask;
#if EFSYS_OPT_RX_PACKED_STREAM || EFSYS_OPT_RX_ES_SUPER_BUFFER
	eersp->eers_rx_stream_npackets = 0;
	eersp->eers_rx_packed_stream = packed_stream || es_super_buffer;
#endif
#if EFSYS_OPT_RX_PACKED_STREAM
	if (packed_stream) {
		eersp->eers_rx_packed_stream_credits = (eep->ee_mask + 1) /
		    EFX_DIV_ROUND_UP(EFX_RX_PACKED_STREAM_MEM_PER_CREDIT,
		    EFX_RX_PACKED_STREAM_MIN_PACKET_SPACE);
		EFSYS_ASSERT3U(eersp->eers_rx_packed_stream_credits, !=, 0);
		/*
		 * A single credit is allocated to the queue when it is started.
		 * It is immediately spent by the first packet which has NEW
		 * BUFFER flag set, though, but still we shall take into
		 * account, as to not wrap around the maximum number of credits
		 * accidentally
		 */
		eersp->eers_rx_packed_stream_credits--;
		EFSYS_ASSERT3U(eersp->eers_rx_packed_stream_credits, <=,
		    EFX_RX_PACKED_STREAM_MAX_CREDITS);
	}
#endif
}

		void
ef10_ev_rxlabel_fini(
	__in		efx_evq_t *eep,
	__in		unsigned int label)
{
	efx_evq_rxq_state_t *eersp;

	EFSYS_ASSERT3U(label, <, EFX_ARRAY_SIZE(eep->ee_rxq_state));
	eersp = &eep->ee_rxq_state[label];

	EFSYS_ASSERT3U(eersp->eers_rx_mask, !=, 0);

	eersp->eers_rx_read_ptr = 0;
	eersp->eers_rx_mask = 0;
#if EFSYS_OPT_RX_PACKED_STREAM || EFSYS_OPT_RX_ES_SUPER_BUFFER
	eersp->eers_rx_stream_npackets = 0;
	eersp->eers_rx_packed_stream = B_FALSE;
#endif
#if EFSYS_OPT_RX_PACKED_STREAM
	eersp->eers_rx_packed_stream_credits = 0;
#endif
}

#endif	/* EFX_OPTS_EF10() */
