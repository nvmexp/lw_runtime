/* SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright(c) 2019-2020 Xilinx, Inc.
 * Copyright(c) 2015-2019 Solarflare Communications Inc.
 */

#ifndef	_SYS_EF10_IMPL_H
#define	_SYS_EF10_IMPL_H

#ifdef	__cplusplus
extern "C" {
#endif

#define	EF10_EVQ_MAXNEVS	32768
#define	EF10_EVQ_MINNEVS	512

#define	EF10_RXQ_MAXNDESCS	4096
#define	EF10_RXQ_MINNDESCS	512

#define	EF10_TXQ_MINNDESCS	512

#define	EF10_EVQ_DESC_SIZE	(sizeof (efx_qword_t))
#define	EF10_RXQ_DESC_SIZE	(sizeof (efx_qword_t))
#define	EF10_TXQ_DESC_SIZE	(sizeof (efx_qword_t))

/* Number of hardware EVQ buffers (for compile-time resource dimensions) */
#define	EF10_EVQ_MAXNBUFS	(64)

/* Maximum independent of EFX_BUG35388_WORKAROUND. */
#define	EF10_TXQ_MAXNBUFS	8

#if EFSYS_OPT_HUNTINGTON
# if (EF10_EVQ_MAXNBUFS < HUNT_EVQ_MAXNBUFS)
#  error "EF10_EVQ_MAXNBUFS too small"
# endif
#endif /* EFSYS_OPT_HUNTINGTON */
#if EFSYS_OPT_MEDFORD
# if (EF10_EVQ_MAXNBUFS < MEDFORD_EVQ_MAXNBUFS)
#  error "EF10_EVQ_MAXNBUFS too small"
# endif
#endif /* EFSYS_OPT_MEDFORD */
#if EFSYS_OPT_MEDFORD2
# if (EF10_EVQ_MAXNBUFS < MEDFORD2_EVQ_MAXNBUFS)
#  error "EF10_EVQ_MAXNBUFS too small"
# endif
#endif /* EFSYS_OPT_MEDFORD2 */

/* Number of hardware PIO buffers (for compile-time resource dimensions) */
#define	EF10_MAX_PIOBUF_NBUFS	(16)

#if EFSYS_OPT_HUNTINGTON
# if (EF10_MAX_PIOBUF_NBUFS < HUNT_PIOBUF_NBUFS)
#  error "EF10_MAX_PIOBUF_NBUFS too small"
# endif
#endif /* EFSYS_OPT_HUNTINGTON */
#if EFSYS_OPT_MEDFORD
# if (EF10_MAX_PIOBUF_NBUFS < MEDFORD_PIOBUF_NBUFS)
#  error "EF10_MAX_PIOBUF_NBUFS too small"
# endif
#endif /* EFSYS_OPT_MEDFORD */
#if EFSYS_OPT_MEDFORD2
# if (EF10_MAX_PIOBUF_NBUFS < MEDFORD2_PIOBUF_NBUFS)
#  error "EF10_MAX_PIOBUF_NBUFS too small"
# endif
#endif /* EFSYS_OPT_MEDFORD2 */



/*
 * FIXME: This is just a power of 2 which fits in an MCDI v1 message, and could
 * possibly be increased, or the write size reported by newer firmware used
 * instead.
 */
#define	EF10_LWRAM_CHUNK 0x80

/*
 * Alignment requirement for value written to RX WPTR: the WPTR must be aligned
 * to an 8 descriptor boundary.
 */
#define	EF10_RX_WPTR_ALIGN 8

/*
 * Max byte offset into the packet the TCP header must start for the hardware
 * to be able to parse the packet correctly.
 */
#define	EF10_TCP_HEADER_OFFSET_LIMIT	208

/* Invalid RSS context handle */
#define	EF10_RSS_CONTEXT_ILWALID	(0xffffffff)


/* EV */

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_ev_init(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern			void
ef10_ev_fini(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_ev_qcreate(
	__in		efx_nic_t *enp,
	__in		unsigned int index,
	__in		efsys_mem_t *esmp,
	__in		size_t ndescs,
	__in		uint32_t id,
	__in		uint32_t us,
	__in		uint32_t flags,
	__in		efx_evq_t *eep);

LIBEFX_INTERNAL
extern			void
ef10_ev_qdestroy(
	__in		efx_evq_t *eep);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_ev_qprime(
	__in		efx_evq_t *eep,
	__in		unsigned int count);

LIBEFX_INTERNAL
extern			void
ef10_ev_qpost(
	__in	efx_evq_t *eep,
	__in	uint16_t data);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_ev_qmoderate(
	__in		efx_evq_t *eep,
	__in		unsigned int us);

#if EFSYS_OPT_QSTATS
LIBEFX_INTERNAL
extern			void
ef10_ev_qstats_update(
	__in				efx_evq_t *eep,
	__inout_ecount(EV_NQSTATS)	efsys_stat_t *stat);
#endif /* EFSYS_OPT_QSTATS */

LIBEFX_INTERNAL
extern			void
ef10_ev_rxlabel_init(
	__in		efx_evq_t *eep,
	__in		efx_rxq_t *erp,
	__in		unsigned int label,
	__in		efx_rxq_type_t type);

LIBEFX_INTERNAL
extern			void
ef10_ev_rxlabel_fini(
	__in		efx_evq_t *eep,
	__in		unsigned int label);

LIBEFX_INTERNAL
extern	__checkReturn	boolean_t
ef10_ev_mcdi(
	__in		efx_evq_t *eep,
	__in		efx_qword_t *eqp,
	__in		const efx_ev_callbacks_t *eecp,
	__in_opt	void *arg);

/* INTR */

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_intr_init(
	__in		efx_nic_t *enp,
	__in		efx_intr_type_t type,
	__in		efsys_mem_t *esmp);

LIBEFX_INTERNAL
extern			void
ef10_intr_enable(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern			void
ef10_intr_disable(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern			void
ef10_intr_disable_unlocked(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_intr_trigger(
	__in		efx_nic_t *enp,
	__in		unsigned int level);

LIBEFX_INTERNAL
extern			void
ef10_intr_status_line(
	__in		efx_nic_t *enp,
	__out		boolean_t *fatalp,
	__out		uint32_t *qmaskp);

LIBEFX_INTERNAL
extern			void
ef10_intr_status_message(
	__in		efx_nic_t *enp,
	__in		unsigned int message,
	__out		boolean_t *fatalp);

LIBEFX_INTERNAL
extern			void
ef10_intr_fatal(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern			void
ef10_intr_fini(
	__in		efx_nic_t *enp);

/* NIC */

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
efx_mcdi_vadaptor_alloc(
	__in		efx_nic_t *enp,
	__in		uint32_t port_id);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
efx_mcdi_vadaptor_free(
	__in		efx_nic_t *enp,
	__in		uint32_t port_id);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_upstream_port_vadaptor_alloc(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_nic_probe(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_nic_set_drv_limits(
	__inout		efx_nic_t *enp,
	__in		efx_drv_limits_t *edlp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_nic_get_vi_pool(
	__in		efx_nic_t *enp,
	__out		uint32_t *vi_countp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_nic_get_bar_region(
	__in		efx_nic_t *enp,
	__in		efx_nic_region_t region,
	__out		uint32_t *offsetp,
	__out		size_t *sizep);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_nic_reset(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_nic_init(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	boolean_t
ef10_nic_hw_unavailable(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern			void
ef10_nic_set_hw_unavailable(
	__in		efx_nic_t *enp);

#if EFSYS_OPT_DIAG

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_nic_register_test(
	__in		efx_nic_t *enp);

#endif	/* EFSYS_OPT_DIAG */

LIBEFX_INTERNAL
extern			void
ef10_nic_fini(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern			void
ef10_nic_unprobe(
	__in		efx_nic_t *enp);


/* MAC */

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_mac_poll(
	__in		efx_nic_t *enp,
	__out		efx_link_mode_t *link_modep);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_mac_up(
	__in		efx_nic_t *enp,
	__out		boolean_t *mac_upp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_mac_addr_set(
	__in	efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_mac_pdu_set(
	__in	efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_mac_pdu_get(
	__in	efx_nic_t *enp,
	__out	size_t *pdu);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_mac_reconfigure(
	__in	efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_mac_multicast_list_set(
	__in				efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_mac_filter_default_rxq_set(
	__in		efx_nic_t *enp,
	__in		efx_rxq_t *erp,
	__in		boolean_t using_rss);

LIBEFX_INTERNAL
extern			void
ef10_mac_filter_default_rxq_clear(
	__in		efx_nic_t *enp);

#if EFSYS_OPT_LOOPBACK

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_mac_loopback_set(
	__in		efx_nic_t *enp,
	__in		efx_link_mode_t link_mode,
	__in		efx_loopback_type_t loopback_type);

#endif	/* EFSYS_OPT_LOOPBACK */

#if EFSYS_OPT_MAC_STATS

LIBEFX_INTERNAL
extern	__checkReturn			efx_rc_t
ef10_mac_stats_get_mask(
	__in				efx_nic_t *enp,
	__inout_bcount(mask_size)	uint32_t *maskp,
	__in				size_t mask_size);

LIBEFX_INTERNAL
extern	__checkReturn			efx_rc_t
ef10_mac_stats_update(
	__in				efx_nic_t *enp,
	__in				efsys_mem_t *esmp,
	__inout_ecount(EFX_MAC_NSTATS)	efsys_stat_t *stat,
	__inout_opt			uint32_t *generationp);

#endif	/* EFSYS_OPT_MAC_STATS */


/* MCDI */

#if EFSYS_OPT_MCDI

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_mcdi_init(
	__in		efx_nic_t *enp,
	__in		const efx_mcdi_transport_t *mtp);

LIBEFX_INTERNAL
extern			void
ef10_mcdi_fini(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern			void
ef10_mcdi_send_request(
	__in			efx_nic_t *enp,
	__in_bcount(hdr_len)	void *hdrp,
	__in			size_t hdr_len,
	__in_bcount(sdu_len)	void *sdup,
	__in			size_t sdu_len);

LIBEFX_INTERNAL
extern	__checkReturn	boolean_t
ef10_mcdi_poll_response(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern			void
ef10_mcdi_read_response(
	__in			efx_nic_t *enp,
	__out_bcount(length)	void *bufferp,
	__in			size_t offset,
	__in			size_t length);

LIBEFX_INTERNAL
extern			efx_rc_t
ef10_mcdi_poll_reboot(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_mcdi_feature_supported(
	__in		efx_nic_t *enp,
	__in		efx_mcdi_feature_id_t id,
	__out		boolean_t *supportedp);

LIBEFX_INTERNAL
extern			void
ef10_mcdi_get_timeout(
	__in		efx_nic_t *enp,
	__in		efx_mcdi_req_t *emrp,
	__out		uint32_t *timeoutp);

#endif /* EFSYS_OPT_MCDI */

/* LWRAM */

#if EFSYS_OPT_LWRAM || EFSYS_OPT_VPD

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_buf_read_tlv(
	__in				efx_nic_t *enp,
	__in_bcount(max_seg_size)	caddr_t seg_data,
	__in				size_t max_seg_size,
	__in				uint32_t tag,
	__deref_out_bcount_opt(*sizep)	caddr_t *datap,
	__out				size_t *sizep);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_buf_write_tlv(
	__inout_bcount(partn_size)	caddr_t partn_data,
	__in				size_t partn_size,
	__in				uint32_t tag,
	__in_bcount(tag_size)		caddr_t tag_data,
	__in				size_t tag_size,
	__out				size_t *total_lengthp);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_partn_read_tlv(
	__in				efx_nic_t *enp,
	__in				uint32_t partn,
	__in				uint32_t tag,
	__deref_out_bcount_opt(*sizep)	caddr_t *datap,
	__out				size_t *sizep);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_partn_write_tlv(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__in			uint32_t tag,
	__in_bcount(size)	caddr_t data,
	__in			size_t size);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_partn_write_segment_tlv(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__in			uint32_t tag,
	__in_bcount(size)	caddr_t data,
	__in			size_t size,
	__in			boolean_t all_segments);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_partn_lock(
	__in			efx_nic_t *enp,
	__in			uint32_t partn);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_partn_unlock(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__out_opt		uint32_t *resultp);

#endif /* EFSYS_OPT_LWRAM || EFSYS_OPT_VPD */

#if EFSYS_OPT_LWRAM

#if EFSYS_OPT_DIAG

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_test(
	__in			efx_nic_t *enp);

#endif	/* EFSYS_OPT_DIAG */

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_type_to_partn(
	__in			efx_nic_t *enp,
	__in			efx_lwram_type_t type,
	__out			uint32_t *partnp);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_partn_size(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__out			size_t *sizep);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_partn_info(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__out			efx_lwram_info_t * enip);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_partn_rw_start(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__out			size_t *chunk_sizep);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_partn_read_mode(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__in			unsigned int offset,
	__out_bcount(size)	caddr_t data,
	__in			size_t size,
	__in			uint32_t mode);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_partn_read(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__in			unsigned int offset,
	__out_bcount(size)	caddr_t data,
	__in			size_t size);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_partn_read_backup(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__in			unsigned int offset,
	__out_bcount(size)	caddr_t data,
	__in			size_t size);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_partn_erase(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__in			unsigned int offset,
	__in			size_t size);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_partn_write(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__in			unsigned int offset,
	__in_bcount(size)	caddr_t data,
	__in			size_t size);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_partn_rw_finish(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__out_opt		uint32_t *verify_resultp);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_partn_get_version(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__out			uint32_t *subtypep,
	__out_ecount(4)		uint16_t version[4]);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_partn_set_version(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__in_ecount(4)		uint16_t version[4]);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_buffer_validate(
	__in			uint32_t partn,
	__in_bcount(buffer_size)
				caddr_t bufferp,
	__in			size_t buffer_size);

LIBEFX_INTERNAL
extern			void
ef10_lwram_buffer_init(
	__out_bcount(buffer_size)
				caddr_t bufferp,
	__in			size_t buffer_size);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_buffer_create(
	__in			uint32_t partn_type,
	__out_bcount(buffer_size)
				caddr_t bufferp,
	__in			size_t buffer_size);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_buffer_find_item_start(
	__in_bcount(buffer_size)
				caddr_t bufferp,
	__in			size_t buffer_size,
	__out			uint32_t *startp);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_buffer_find_end(
	__in_bcount(buffer_size)
				caddr_t bufferp,
	__in			size_t buffer_size,
	__in			uint32_t offset,
	__out			uint32_t *endp);

LIBEFX_INTERNAL
extern	__checkReturn	__success(return != B_FALSE)	boolean_t
ef10_lwram_buffer_find_item(
	__in_bcount(buffer_size)
				caddr_t bufferp,
	__in			size_t buffer_size,
	__in			uint32_t offset,
	__out			uint32_t *startp,
	__out			uint32_t *lengthp);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_buffer_peek_item(
	__in_bcount(buffer_size)
				caddr_t bufferp,
	__in			size_t buffer_size,
	__in			uint32_t offset,
	__out			uint32_t *tagp,
	__out			uint32_t *lengthp,
	__out			uint32_t *value_offsetp);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_buffer_get_item(
	__in_bcount(buffer_size)
				caddr_t bufferp,
	__in			size_t buffer_size,
	__in			uint32_t offset,
	__in			uint32_t length,
	__out			uint32_t *tagp,
	__out_bcount_part(value_max_size, *lengthp)
				caddr_t valuep,
	__in			size_t value_max_size,
	__out			uint32_t *lengthp);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_buffer_insert_item(
	__in_bcount(buffer_size)
				caddr_t bufferp,
	__in			size_t buffer_size,
	__in			uint32_t offset,
	__in			uint32_t tag,
	__in_bcount(length)	caddr_t valuep,
	__in			uint32_t length,
	__out			uint32_t *lengthp);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_buffer_modify_item(
	__in_bcount(buffer_size)
				caddr_t bufferp,
	__in			size_t buffer_size,
	__in			uint32_t offset,
	__in			uint32_t tag,
	__in_bcount(length)	caddr_t valuep,
	__in			uint32_t length,
	__out			uint32_t *lengthp);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_buffer_delete_item(
	__in_bcount(buffer_size)
				caddr_t bufferp,
	__in			size_t buffer_size,
	__in			uint32_t offset,
	__in			uint32_t length,
	__in			uint32_t end);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_lwram_buffer_finish(
	__in_bcount(buffer_size)
				caddr_t bufferp,
	__in			size_t buffer_size);

#endif	/* EFSYS_OPT_LWRAM */


/* PHY */

typedef struct ef10_link_state_s {
	efx_phy_link_state_t	epls;
#if EFSYS_OPT_LOOPBACK
	efx_loopback_type_t	els_loopback;
#endif
	boolean_t		els_mac_up;
} ef10_link_state_t;

LIBEFX_INTERNAL
extern			void
ef10_phy_link_ev(
	__in		efx_nic_t *enp,
	__in		efx_qword_t *eqp,
	__out		efx_link_mode_t *link_modep);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_phy_get_link(
	__in		efx_nic_t *enp,
	__out		ef10_link_state_t *elsp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_phy_power(
	__in		efx_nic_t *enp,
	__in		boolean_t on);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_phy_reconfigure(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_phy_verify(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_phy_oui_get(
	__in		efx_nic_t *enp,
	__out		uint32_t *ouip);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_phy_link_state_get(
	__in		efx_nic_t *enp,
	__out		efx_phy_link_state_t *eplsp);

#if EFSYS_OPT_PHY_STATS

LIBEFX_INTERNAL
extern	__checkReturn			efx_rc_t
ef10_phy_stats_update(
	__in				efx_nic_t *enp,
	__in				efsys_mem_t *esmp,
	__inout_ecount(EFX_PHY_NSTATS)	uint32_t *stat);

#endif	/* EFSYS_OPT_PHY_STATS */

#if EFSYS_OPT_BIST

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_bist_enable_offline(
	__in			efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_bist_start(
	__in			efx_nic_t *enp,
	__in			efx_bist_type_t type);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_bist_poll(
	__in			efx_nic_t *enp,
	__in			efx_bist_type_t type,
	__out			efx_bist_result_t *resultp,
	__out_opt __drv_when(count > 0, __notnull)
	uint32_t	*value_maskp,
	__out_ecount_opt(count)	__drv_when(count > 0, __notnull)
	unsigned long	*valuesp,
	__in			size_t count);

LIBEFX_INTERNAL
extern				void
ef10_bist_stop(
	__in			efx_nic_t *enp,
	__in			efx_bist_type_t type);

#endif	/* EFSYS_OPT_BIST */

/* TX */

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_tx_init(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern			void
ef10_tx_fini(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_tx_qcreate(
	__in		efx_nic_t *enp,
	__in		unsigned int index,
	__in		unsigned int label,
	__in		efsys_mem_t *esmp,
	__in		size_t ndescs,
	__in		uint32_t id,
	__in		uint16_t flags,
	__in		efx_evq_t *eep,
	__in		efx_txq_t *etp,
	__out		unsigned int *addedp);

LIBEFX_INTERNAL
extern		void
ef10_tx_qdestroy(
	__in		efx_txq_t *etp);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_tx_qpost(
	__in			efx_txq_t *etp,
	__in_ecount(ndescs)	efx_buffer_t *ebp,
	__in			unsigned int ndescs,
	__in			unsigned int completed,
	__inout			unsigned int *addedp);

LIBEFX_INTERNAL
extern			void
ef10_tx_qpush(
	__in		efx_txq_t *etp,
	__in		unsigned int added,
	__in		unsigned int pushed);

#if EFSYS_OPT_RX_PACKED_STREAM
LIBEFX_INTERNAL
extern			void
ef10_rx_qpush_ps_credits(
	__in		efx_rxq_t *erp);

LIBEFX_INTERNAL
extern	__checkReturn	uint8_t *
ef10_rx_qps_packet_info(
	__in		efx_rxq_t *erp,
	__in		uint8_t *buffer,
	__in		uint32_t buffer_length,
	__in		uint32_t lwrrent_offset,
	__out		uint16_t *lengthp,
	__out		uint32_t *next_offsetp,
	__out		uint32_t *timestamp);
#endif

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_tx_qpace(
	__in		efx_txq_t *etp,
	__in		unsigned int ns);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_tx_qflush(
	__in		efx_txq_t *etp);

LIBEFX_INTERNAL
extern			void
ef10_tx_qenable(
	__in		efx_txq_t *etp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_tx_qpio_enable(
	__in		efx_txq_t *etp);

LIBEFX_INTERNAL
extern			void
ef10_tx_qpio_disable(
	__in		efx_txq_t *etp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_tx_qpio_write(
	__in			efx_txq_t *etp,
	__in_ecount(buf_length)	uint8_t *buffer,
	__in			size_t buf_length,
	__in			size_t pio_buf_offset);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_tx_qpio_post(
	__in			efx_txq_t *etp,
	__in			size_t pkt_length,
	__in			unsigned int completed,
	__inout			unsigned int *addedp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_tx_qdesc_post(
	__in		efx_txq_t *etp,
	__in_ecount(n)	efx_desc_t *ed,
	__in		unsigned int n,
	__in		unsigned int completed,
	__inout		unsigned int *addedp);

LIBEFX_INTERNAL
extern	void
ef10_tx_qdesc_dma_create(
	__in	efx_txq_t *etp,
	__in	efsys_dma_addr_t addr,
	__in	size_t size,
	__in	boolean_t eop,
	__out	efx_desc_t *edp);

LIBEFX_INTERNAL
extern	void
ef10_tx_qdesc_tso_create(
	__in	efx_txq_t *etp,
	__in	uint16_t ipv4_id,
	__in	uint32_t tcp_seq,
	__in	uint8_t	 tcp_flags,
	__out	efx_desc_t *edp);

LIBEFX_INTERNAL
extern	void
ef10_tx_qdesc_tso2_create(
	__in			efx_txq_t *etp,
	__in			uint16_t ipv4_id,
	__in			uint16_t outer_ipv4_id,
	__in			uint32_t tcp_seq,
	__in			uint16_t tcp_mss,
	__out_ecount(count)	efx_desc_t *edp,
	__in			int count);

LIBEFX_INTERNAL
extern	void
ef10_tx_qdesc_vlantci_create(
	__in	efx_txq_t *etp,
	__in	uint16_t vlan_tci,
	__out	efx_desc_t *edp);

LIBEFX_INTERNAL
extern	void
ef10_tx_qdesc_checksum_create(
	__in	efx_txq_t *etp,
	__in	uint16_t flags,
	__out	efx_desc_t *edp);

#if EFSYS_OPT_QSTATS

LIBEFX_INTERNAL
extern			void
ef10_tx_qstats_update(
	__in				efx_txq_t *etp,
	__inout_ecount(TX_NQSTATS)	efsys_stat_t *stat);

#endif /* EFSYS_OPT_QSTATS */

typedef uint32_t	efx_piobuf_handle_t;

#define	EFX_PIOBUF_HANDLE_ILWALID	((efx_piobuf_handle_t)-1)

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_nic_pio_alloc(
	__inout		efx_nic_t *enp,
	__out		uint32_t *bufnump,
	__out		efx_piobuf_handle_t *handlep,
	__out		uint32_t *blknump,
	__out		uint32_t *offsetp,
	__out		size_t *sizep);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_nic_pio_free(
	__inout		efx_nic_t *enp,
	__in		uint32_t bufnum,
	__in		uint32_t blknum);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_nic_pio_link(
	__inout		efx_nic_t *enp,
	__in		uint32_t vi_index,
	__in		efx_piobuf_handle_t handle);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_nic_pio_unlink(
	__inout		efx_nic_t *enp,
	__in		uint32_t vi_index);


/* VPD */

#if EFSYS_OPT_VPD

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_vpd_init(
	__in			efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_vpd_size(
	__in			efx_nic_t *enp,
	__out			size_t *sizep);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_vpd_read(
	__in			efx_nic_t *enp,
	__out_bcount(size)	caddr_t data,
	__in			size_t size);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_vpd_verify(
	__in			efx_nic_t *enp,
	__in_bcount(size)	caddr_t data,
	__in			size_t size);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_vpd_reinit(
	__in			efx_nic_t *enp,
	__in_bcount(size)	caddr_t data,
	__in			size_t size);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_vpd_get(
	__in			efx_nic_t *enp,
	__in_bcount(size)	caddr_t data,
	__in			size_t size,
	__inout			efx_vpd_value_t *evvp);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_vpd_set(
	__in			efx_nic_t *enp,
	__in_bcount(size)	caddr_t data,
	__in			size_t size,
	__in			efx_vpd_value_t *evvp);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_vpd_next(
	__in			efx_nic_t *enp,
	__in_bcount(size)	caddr_t data,
	__in			size_t size,
	__out			efx_vpd_value_t *evvp,
	__inout			unsigned int *contp);

LIBEFX_INTERNAL
extern __checkReturn		efx_rc_t
ef10_vpd_write(
	__in			efx_nic_t *enp,
	__in_bcount(size)	caddr_t data,
	__in			size_t size);

LIBEFX_INTERNAL
extern				void
ef10_vpd_fini(
	__in			efx_nic_t *enp);

#endif	/* EFSYS_OPT_VPD */


/* RX */

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_rx_init(
	__in		efx_nic_t *enp);

#if EFSYS_OPT_RX_SCATTER
LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_rx_scatter_enable(
	__in		efx_nic_t *enp,
	__in		unsigned int buf_size);
#endif	/* EFSYS_OPT_RX_SCATTER */


#if EFSYS_OPT_RX_SCALE

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_rx_scale_context_alloc(
	__in		efx_nic_t *enp,
	__in		efx_rx_scale_context_type_t type,
	__in		uint32_t num_queues,
	__out		uint32_t *rss_contextp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_rx_scale_context_free(
	__in		efx_nic_t *enp,
	__in		uint32_t rss_context);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_rx_scale_mode_set(
	__in		efx_nic_t *enp,
	__in		uint32_t rss_context,
	__in		efx_rx_hash_alg_t alg,
	__in		efx_rx_hash_type_t type,
	__in		boolean_t insert);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_rx_scale_key_set(
	__in		efx_nic_t *enp,
	__in		uint32_t rss_context,
	__in_ecount(n)	uint8_t *key,
	__in		size_t n);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_rx_scale_tbl_set(
	__in		efx_nic_t *enp,
	__in		uint32_t rss_context,
	__in_ecount(n)	unsigned int *table,
	__in		size_t n);

LIBEFX_INTERNAL
extern	__checkReturn	uint32_t
ef10_rx_prefix_hash(
	__in		efx_nic_t *enp,
	__in		efx_rx_hash_alg_t func,
	__in		uint8_t *buffer);

#endif /* EFSYS_OPT_RX_SCALE */

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_rx_prefix_pktlen(
	__in		efx_nic_t *enp,
	__in		uint8_t *buffer,
	__out		uint16_t *lengthp);

LIBEFX_INTERNAL
extern				void
ef10_rx_qpost(
	__in			efx_rxq_t *erp,
	__in_ecount(ndescs)	efsys_dma_addr_t *addrp,
	__in			size_t size,
	__in			unsigned int ndescs,
	__in			unsigned int completed,
	__in			unsigned int added);

LIBEFX_INTERNAL
extern			void
ef10_rx_qpush(
	__in		efx_rxq_t *erp,
	__in		unsigned int added,
	__inout		unsigned int *pushedp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_rx_qflush(
	__in		efx_rxq_t *erp);

LIBEFX_INTERNAL
extern		void
ef10_rx_qenable(
	__in		efx_rxq_t *erp);

union efx_rxq_type_data_u;

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_rx_qcreate(
	__in		efx_nic_t *enp,
	__in		unsigned int index,
	__in		unsigned int label,
	__in		efx_rxq_type_t type,
	__in_opt	const union efx_rxq_type_data_u *type_data,
	__in		efsys_mem_t *esmp,
	__in		size_t ndescs,
	__in		uint32_t id,
	__in		unsigned int flags,
	__in		efx_evq_t *eep,
	__in		efx_rxq_t *erp);

LIBEFX_INTERNAL
extern			void
ef10_rx_qdestroy(
	__in		efx_rxq_t *erp);

LIBEFX_INTERNAL
extern			void
ef10_rx_fini(
	__in		efx_nic_t *enp);

#if EFSYS_OPT_FILTER

enum efx_filter_replacement_policy_e;

typedef struct ef10_filter_handle_s {
	uint32_t	efh_lo;
	uint32_t	efh_hi;
} ef10_filter_handle_t;

typedef struct ef10_filter_entry_s {
	uintptr_t efe_spec; /* pointer to filter spec plus busy bit */
	ef10_filter_handle_t efe_handle;
} ef10_filter_entry_t;

/*
 * BUSY flag indicates that an update is in progress.
 * AUTO_OLD flag is used to mark and sweep MAC packet filters.
 */
#define	EFX_EF10_FILTER_FLAG_BUSY	1U
#define	EFX_EF10_FILTER_FLAG_AUTO_OLD	2U
#define	EFX_EF10_FILTER_FLAGS		3U

/*
 * Size of the hash table used by the driver. Doesn't need to be the
 * same size as the hardware's table.
 */
#define	EFX_EF10_FILTER_TBL_ROWS 8192

/* Only need to allow for one directed and one unknown unicast filter */
#define	EFX_EF10_FILTER_UNICAST_FILTERS_MAX	2

/* Allow for the broadcast address to be added to the multicast list */
#define	EFX_EF10_FILTER_MULTICAST_FILTERS_MAX	(EFX_MAC_MULTICAST_LIST_MAX + 1)

/*
 * For encapsulated packets, there is one filter each for each combination of
 * IPv4 or IPv6 outer frame, VXLAN, GENEVE or LWGRE packet type, and unicast or
 * multicast inner frames.
 */
#define	EFX_EF10_FILTER_ENCAP_FILTERS_MAX	12

typedef struct ef10_filter_table_s {
	ef10_filter_entry_t	eft_entry[EFX_EF10_FILTER_TBL_ROWS];
	efx_rxq_t		*eft_default_rxq;
	boolean_t		eft_using_rss;
	uint32_t		eft_unicst_filter_indexes[
	    EFX_EF10_FILTER_UNICAST_FILTERS_MAX];
	uint32_t		eft_unicst_filter_count;
	uint32_t		eft_mulcst_filter_indexes[
	    EFX_EF10_FILTER_MULTICAST_FILTERS_MAX];
	uint32_t		eft_mulcst_filter_count;
	boolean_t		eft_using_all_mulcst;
	uint32_t		eft_encap_filter_indexes[
	    EFX_EF10_FILTER_ENCAP_FILTERS_MAX];
	uint32_t		eft_encap_filter_count;
} ef10_filter_table_t;

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_filter_init(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern			void
ef10_filter_fini(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_filter_restore(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_filter_add(
	__in		efx_nic_t *enp,
	__inout		efx_filter_spec_t *spec,
	__in		enum efx_filter_replacement_policy_e policy);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_filter_delete(
	__in		efx_nic_t *enp,
	__inout		efx_filter_spec_t *spec);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_filter_supported_filters(
	__in				efx_nic_t *enp,
	__out_ecount(buffer_length)	uint32_t *buffer,
	__in				size_t buffer_length,
	__out				size_t *list_lengthp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_filter_reconfigure(
	__in				efx_nic_t *enp,
	__in_ecount(6)			uint8_t const *mac_addr,
	__in				boolean_t all_unicst,
	__in				boolean_t mulcst,
	__in				boolean_t all_mulcst,
	__in				boolean_t brdcst,
	__in_ecount(6*count)		uint8_t const *addrs,
	__in				uint32_t count);

LIBEFX_INTERNAL
extern		void
ef10_filter_get_default_rxq(
	__in		efx_nic_t *enp,
	__out		efx_rxq_t **erpp,
	__out		boolean_t *using_rss);

LIBEFX_INTERNAL
extern		void
ef10_filter_default_rxq_set(
	__in		efx_nic_t *enp,
	__in		efx_rxq_t *erp,
	__in		boolean_t using_rss);

LIBEFX_INTERNAL
extern		void
ef10_filter_default_rxq_clear(
	__in		efx_nic_t *enp);


#endif /* EFSYS_OPT_FILTER */

LIBEFX_INTERNAL
extern	__checkReturn			efx_rc_t
efx_mcdi_get_function_info(
	__in				efx_nic_t *enp,
	__out				uint32_t *pfp,
	__out_opt			uint32_t *vfp);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
efx_mcdi_privilege_mask(
	__in			efx_nic_t *enp,
	__in			uint32_t pf,
	__in			uint32_t vf,
	__out			uint32_t *maskp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
efx_mcdi_get_port_assignment(
	__in		efx_nic_t *enp,
	__out		uint32_t *portp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
efx_mcdi_get_port_modes(
	__in		efx_nic_t *enp,
	__out		uint32_t *modesp,
	__out_opt	uint32_t *lwrrent_modep,
	__out_opt	uint32_t *default_modep);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_nic_get_port_mode_bandwidth(
	__in		efx_nic_t *enp,
	__out		uint32_t *bandwidth_mbpsp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
efx_mcdi_get_mac_address_pf(
	__in			efx_nic_t *enp,
	__out_ecount_opt(6)	uint8_t mac_addrp[6]);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
efx_mcdi_get_mac_address_vf(
	__in			efx_nic_t *enp,
	__out_ecount_opt(6)	uint8_t mac_addrp[6]);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
efx_mcdi_get_clock(
	__in		efx_nic_t *enp,
	__out		uint32_t *sys_freqp,
	__out		uint32_t *dpcpu_freqp);


LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
efx_mcdi_get_rxdp_config(
	__in		efx_nic_t *enp,
	__out		uint32_t *end_paddingp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
efx_mcdi_get_vector_cfg(
	__in		efx_nic_t *enp,
	__out_opt	uint32_t *vec_basep,
	__out_opt	uint32_t *pf_lwecp,
	__out_opt	uint32_t *vf_lwecp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
efx_mcdi_alloc_vis(
	__in		efx_nic_t *enp,
	__in		uint32_t min_vi_count,
	__in		uint32_t max_vi_count,
	__out		uint32_t *vi_basep,
	__out		uint32_t *vi_countp,
	__out		uint32_t *vi_shiftp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
efx_mcdi_free_vis(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_get_privilege_mask(
	__in			efx_nic_t *enp,
	__out			uint32_t *maskp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
efx_mcdi_nic_board_cfg(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
efx_mcdi_entity_reset(
	__in		efx_nic_t *enp);

#if EFSYS_OPT_FW_SUBVARIANT_AWARE

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
efx_mcdi_get_nic_global(
	__in		efx_nic_t *enp,
	__in		uint32_t key,
	__out		uint32_t *valuep);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
efx_mcdi_set_nic_global(
	__in		efx_nic_t *enp,
	__in		uint32_t key,
	__in		uint32_t value);

#endif	/* EFSYS_OPT_FW_SUBVARIANT_AWARE */

#if EFSYS_OPT_EVB
LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_evb_init(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern			void
ef10_evb_fini(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_evb_vswitch_alloc(
	__in		efx_nic_t *enp,
	__out		efx_vswitch_id_t *vswitch_idp);


LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_evb_vswitch_free(
	__in		efx_nic_t *enp,
	__in		efx_vswitch_id_t vswitch_id);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_evb_vport_alloc(
	__in		efx_nic_t *enp,
	__in		efx_vswitch_id_t vswitch_id,
	__in		efx_vport_type_t vport_type,
	__in		uint16_t vid,
	__in		boolean_t vlan_restrict,
	__out		efx_vport_id_t *vport_idp);


LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_evb_vport_free(
	__in		efx_nic_t *enp,
	__in		efx_vswitch_id_t vswitch_id,
	__in		efx_vport_id_t vport_id);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_evb_vport_mac_addr_add(
	__in		efx_nic_t *enp,
	__in		efx_vswitch_id_t vswitch_id,
	__in		efx_vport_id_t vport_id,
	__in_ecount(6)	uint8_t *addrp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_evb_vport_mac_addr_del(
	__in		efx_nic_t *enp,
	__in		efx_vswitch_id_t vswitch_id,
	__in		efx_vport_id_t vport_id,
	__in_ecount(6)	uint8_t *addrp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_evb_vadaptor_alloc(
	__in		efx_nic_t *enp,
	__in		efx_vswitch_id_t vswitch_id,
	__in		efx_vport_id_t vport_id);


LIBEFX_INTERNAL
extern __checkReturn	efx_rc_t
ef10_evb_vadaptor_free(
	__in		efx_nic_t *enp,
	__in		efx_vswitch_id_t vswitch_id,
	__in		efx_vport_id_t vport_id);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_evb_vport_assign(
	__in		efx_nic_t *enp,
	__in		efx_vswitch_id_t vswitch_id,
	__in		efx_vport_id_t vport_id,
	__in		uint32_t vf_index);

LIBEFX_INTERNAL
extern	__checkReturn				efx_rc_t
ef10_evb_vport_reconfigure(
	__in					efx_nic_t *enp,
	__in					efx_vswitch_id_t vswitch_id,
	__in					efx_vport_id_t vport_id,
	__in_opt				uint16_t *vidp,
	__in_bcount_opt(EFX_MAC_ADDR_LEN)	uint8_t *addrp,
	__out_opt				boolean_t *fn_resetp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_evb_vport_stats(
	__in		efx_nic_t *enp,
	__in		efx_vswitch_id_t vswitch_id,
	__in		efx_vport_id_t vport_id,
	__out		efsys_mem_t *esmp);

#endif  /* EFSYS_OPT_EVB */

#if EFSYS_OPT_MCDI_PROXY_AUTH_SERVER
LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_proxy_auth_init(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern			void
ef10_proxy_auth_fini(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn		efx_rc_t
ef10_proxy_auth_mc_config(
	__in			efx_nic_t *enp,
	__in			efsys_mem_t *request_bufferp,
	__in			efsys_mem_t *response_bufferp,
	__in			efsys_mem_t *status_bufferp,
	__in			uint32_t block_cnt,
	__in_ecount(op_count)	uint32_t *op_listp,
	__in			size_t op_count);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_proxy_auth_disable(
	__in		efx_nic_t *enp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_proxy_auth_privilege_modify(
	__in		efx_nic_t *enp,
	__in		uint32_t fn_group,
	__in		uint32_t pf_index,
	__in		uint32_t vf_index,
	__in		uint32_t add_privileges_mask,
	__in		uint32_t remove_privileges_mask);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_proxy_auth_set_privilege_mask(
	__in		efx_nic_t *enp,
	__in		uint32_t vf_index,
	__in		uint32_t mask,
	__in		uint32_t value);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_proxy_auth_complete_request(
	__in		efx_nic_t *enp,
	__in		uint32_t fn_index,
	__in		uint32_t proxy_result,
	__in		uint32_t handle);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_proxy_auth_exec_cmd(
	__in		efx_nic_t *enp,
	__inout		efx_proxy_cmd_params_t *paramsp);

LIBEFX_INTERNAL
extern	__checkReturn	efx_rc_t
ef10_proxy_auth_get_privilege_mask(
	__in		efx_nic_t *enp,
	__in		uint32_t pf_index,
	__in		uint32_t vf_index,
	__out		uint32_t *maskp);

#endif  /* EFSYS_OPT_MCDI_PROXY_AUTH_SERVER */

#if EFSYS_OPT_RX_PACKED_STREAM

/* Data space per credit in packed stream mode */
#define	EFX_RX_PACKED_STREAM_MEM_PER_CREDIT (1 << 16)

/*
 * Received packets are always aligned at this boundary. Also there always
 * exists a gap of this size between packets.
 * (see SF-112241-TC, 4.5)
 */
#define	EFX_RX_PACKED_STREAM_ALIGNMENT 64

/*
 * Size of a pseudo-header prepended to received packets
 * in packed stream mode
 */
#define	EFX_RX_PACKED_STREAM_RX_PREFIX_SIZE 8

/* Minimum space for packet in packed stream mode */
#define	EFX_RX_PACKED_STREAM_MIN_PACKET_SPACE		\
	EFX_P2ROUNDUP(size_t,				\
	    EFX_RX_PACKED_STREAM_RX_PREFIX_SIZE +	\
	    EFX_MAC_PDU_MIN +				\
	    EFX_RX_PACKED_STREAM_ALIGNMENT,		\
	    EFX_RX_PACKED_STREAM_ALIGNMENT)

/* Maximum number of credits */
#define	EFX_RX_PACKED_STREAM_MAX_CREDITS 127

#endif /* EFSYS_OPT_RX_PACKED_STREAM */

#if EFSYS_OPT_RX_ES_SUPER_BUFFER

/*
 * Maximum DMA length and buffer stride alignment.
 * (see SF-119419-TC, 3.2)
 */
#define	EFX_RX_ES_SUPER_BUFFER_BUF_ALIGNMENT	64

#endif

#ifdef	__cplusplus
}
#endif

#endif	/* _SYS_EF10_IMPL_H */
