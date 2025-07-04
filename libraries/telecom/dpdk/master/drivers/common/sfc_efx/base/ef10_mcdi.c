/* SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright(c) 2019-2020 Xilinx, Inc.
 * Copyright(c) 2012-2019 Solarflare Communications Inc.
 */

#include "efx.h"
#include "efx_impl.h"


#if EFSYS_OPT_RIVERHEAD || EFX_OPTS_EF10()

#if EFSYS_OPT_MCDI

#ifndef WITH_MCDI_V2
#error "WITH_MCDI_V2 required for EF10 MCDIv2 commands."
#endif


	__checkReturn	efx_rc_t
ef10_mcdi_init(
	__in		efx_nic_t *enp,
	__in		const efx_mcdi_transport_t *emtp)
{
	efx_mcdi_iface_t *emip = &(enp->en_mcdi.em_emip);
	efsys_mem_t *esmp = emtp->emt_dma_mem;
	efx_dword_t dword;
	efx_rc_t rc;

	EFSYS_ASSERT(EFX_FAMILY_IS_EF100(enp) || EFX_FAMILY_IS_EF10(enp));
	EFSYS_ASSERT(enp->en_features & EFX_FEATURE_MCDI_DMA);

	/*
	 * All EF10 firmware supports MCDIv2 and MCDIv1.
	 * Medford BootROM supports MCDIv2 and MCDIv1.
	 * Huntington BootROM supports MCDIv1 only.
	 */
	emip->emi_max_version = 2;

	/* A host DMA buffer is required for EF10 MCDI */
	if (esmp == NULL) {
		rc = EILWAL;
		goto fail1;
	}

	/*
	 * Ensure that the MC doorbell is in a known state before issuing MCDI
	 * commands. The recovery algorithm requires that the MC command buffer
	 * must be 256 byte aligned. See bug24769.
	 */
	if ((EFSYS_MEM_ADDR(esmp) & 0xFF) != 0) {
		rc = EILWAL;
		goto fail2;
	}
	EFX_POPULATE_DWORD_1(dword, EFX_DWORD_0, 1);
	switch (enp->en_family) {
#if EFSYS_OPT_RIVERHEAD
	case EFX_FAMILY_RIVERHEAD:
		EFX_BAR_FCW_WRITED(enp, ER_GZ_MC_DB_HWRD_REG, &dword);
		break;
#endif	/* EFSYS_OPT_RIVERHEAD */
	default:
		EFX_BAR_WRITED(enp, ER_DZ_MC_DB_HWRD_REG, &dword, B_FALSE);
		break;
	}

	/* Save initial MC reboot status */
	(void) ef10_mcdi_poll_reboot(enp);

	/* Start a new epoch (allow fresh MCDI requests to succeed) */
	efx_mcdi_new_epoch(enp);

	return (0);

fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

			void
ef10_mcdi_fini(
	__in		efx_nic_t *enp)
{
	efx_mcdi_iface_t *emip = &(enp->en_mcdi.em_emip);

	emip->emi_new_epoch = B_FALSE;
}

/*
 * In older firmware all commands are processed in a single thread, so a long
 * running command for one PCIe function can block processing for another
 * function (see bug 61269).
 *
 * In newer firmware that supports multithreaded MCDI processing, we can extend
 * the timeout for long-running requests which we know firmware may choose to
 * process in a background thread.
 */
#define	EF10_MCDI_CMD_TIMEOUT_US	(10 * 1000 * 1000)
#define	EF10_MCDI_CMD_LONG_TIMEOUT_US	(60 * 1000 * 1000)

			void
ef10_mcdi_get_timeout(
	__in		efx_nic_t *enp,
	__in		efx_mcdi_req_t *emrp,
	__out		uint32_t *timeoutp)
{
	efx_nic_cfg_t *encp = &(enp->en_nic_cfg);

	switch (emrp->emr_cmd) {
	case MC_CMD_POLL_BIST:
	case MC_CMD_LWRAM_ERASE:
	case MC_CMD_LICENSING_V3:
	case MC_CMD_LWRAM_UPDATE_FINISH:
		if (encp->enc_lwram_update_verify_result_supported != B_FALSE) {
			/*
			 * Potentially longer running commands, which firmware
			 * may choose to process in a background thread.
			 */
			*timeoutp = EF10_MCDI_CMD_LONG_TIMEOUT_US;
			break;
		}
		/* FALLTHRU */
	default:
		*timeoutp = EF10_MCDI_CMD_TIMEOUT_US;
		break;
	}
}

			void
ef10_mcdi_send_request(
	__in			efx_nic_t *enp,
	__in_bcount(hdr_len)	void *hdrp,
	__in			size_t hdr_len,
	__in_bcount(sdu_len)	void *sdup,
	__in			size_t sdu_len)
{
	const efx_mcdi_transport_t *emtp = enp->en_mcdi.em_emtp;
	efsys_mem_t *esmp = emtp->emt_dma_mem;
	efx_dword_t dword;
	unsigned int pos;

	EFSYS_ASSERT(EFX_FAMILY_IS_EF100(enp) || EFX_FAMILY_IS_EF10(enp));

	/* Write the header */
	for (pos = 0; pos < hdr_len; pos += sizeof (efx_dword_t)) {
		dword = *(efx_dword_t *)((uint8_t *)hdrp + pos);
		EFSYS_MEM_WRITED(esmp, pos, &dword);
	}

	/* Write the payload */
	for (pos = 0; pos < sdu_len; pos += sizeof (efx_dword_t)) {
		dword = *(efx_dword_t *)((uint8_t *)sdup + pos);
		EFSYS_MEM_WRITED(esmp, hdr_len + pos, &dword);
	}

	/* Guarantee ordering of memory (MCDI request) and PIO (MC doorbell) */
	EFSYS_DMA_SYNC_FOR_DEVICE(esmp, 0, hdr_len + sdu_len);
	EFSYS_PIO_WRITE_BARRIER();

	/* Ring the doorbell to post the command DMA address to the MC */
	EFX_POPULATE_DWORD_1(dword, EFX_DWORD_0,
	    EFSYS_MEM_ADDR(esmp) >> 32);
	switch (enp->en_family) {
#if EFSYS_OPT_RIVERHEAD
	case EFX_FAMILY_RIVERHEAD:
		EFX_BAR_FCW_WRITED(enp, ER_GZ_MC_DB_LWRD_REG, &dword);
		break;
#endif	/* EFSYS_OPT_RIVERHEAD */
	default:
		EFX_BAR_WRITED(enp, ER_DZ_MC_DB_LWRD_REG, &dword, B_FALSE);
		break;
	}

	EFX_POPULATE_DWORD_1(dword, EFX_DWORD_0,
	    EFSYS_MEM_ADDR(esmp) & 0xffffffff);
	switch (enp->en_family) {
#if EFSYS_OPT_RIVERHEAD
	case EFX_FAMILY_RIVERHEAD:
		EFX_BAR_FCW_WRITED(enp, ER_GZ_MC_DB_HWRD_REG, &dword);
		break;
#endif	/* EFSYS_OPT_RIVERHEAD */
	default:
		EFX_BAR_WRITED(enp, ER_DZ_MC_DB_HWRD_REG, &dword, B_FALSE);
		break;
	}
}

	__checkReturn	boolean_t
ef10_mcdi_poll_response(
	__in		efx_nic_t *enp)
{
	const efx_mcdi_transport_t *emtp = enp->en_mcdi.em_emtp;
	efsys_mem_t *esmp = emtp->emt_dma_mem;
	efx_dword_t hdr;

	EFSYS_MEM_READD(esmp, 0, &hdr);
	EFSYS_MEM_READ_BARRIER();

	return (EFX_DWORD_FIELD(hdr, MCDI_HEADER_RESPONSE) ? B_TRUE : B_FALSE);
}

			void
ef10_mcdi_read_response(
	__in			efx_nic_t *enp,
	__out_bcount(length)	void *bufferp,
	__in			size_t offset,
	__in			size_t length)
{
	const efx_mcdi_transport_t *emtp = enp->en_mcdi.em_emtp;
	efsys_mem_t *esmp = emtp->emt_dma_mem;
	unsigned int pos = 0;
	efx_dword_t data;
	size_t remaining = length;

	while (remaining > 0) {
		size_t chunk = MIN(remaining, sizeof (data));

		EFSYS_MEM_READD(esmp, offset + pos, &data);
		memcpy((uint8_t *)bufferp + pos, &data, chunk);
		pos += chunk;
		remaining -= chunk;
	}
}

			efx_rc_t
ef10_mcdi_poll_reboot(
	__in		efx_nic_t *enp)
{
	efx_mcdi_iface_t *emip = &(enp->en_mcdi.em_emip);
	efx_dword_t dword;
	uint32_t old_status;
	uint32_t new_status;
	efx_rc_t rc;

	old_status = emip->emi_mc_reboot_status;

	/* Update MC reboot status word */
	switch (enp->en_family) {
#if EFSYS_OPT_RIVERHEAD
	case EFX_FAMILY_RIVERHEAD:
		EFX_BAR_FCW_READD(enp, ER_GZ_MC_SFT_STATUS, &dword);
		break;
#endif	/* EFSYS_OPT_RIVERHEAD */
	default:
		EFX_BAR_READD(enp, ER_DZ_BIU_MC_SFT_STATUS_REG,
			      &dword, B_FALSE);
		break;
	}
	new_status = dword.ed_u32[0];

	/* MC has rebooted if the value has changed */
	if (new_status != old_status) {
		emip->emi_mc_reboot_status = new_status;

		/*
		 * FIXME: Ignore detected MC REBOOT for now.
		 *
		 * The Siena support for checking for MC reboot from status
		 * flags is broken - see comments in siena_mcdi_poll_reboot().
		 * As the generic MCDI code is shared the EF10 reboot
		 * detection suffers similar problems.
		 *
		 * Do not report an error when the boot status changes until
		 * this can be handled by common code drivers (and reworked to
		 * support Siena too).
		 */
		_NOTE(CONSTANTCONDITION)
		if (B_FALSE) {
			rc = EIO;
			goto fail1;
		}
	}

	return (0);

fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

	__checkReturn	efx_rc_t
ef10_mcdi_feature_supported(
	__in		efx_nic_t *enp,
	__in		efx_mcdi_feature_id_t id,
	__out		boolean_t *supportedp)
{
	efx_nic_cfg_t *encp = &(enp->en_nic_cfg);
	uint32_t privilege_mask = encp->enc_privilege_mask;
	efx_rc_t rc;

	EFSYS_ASSERT(EFX_FAMILY_IS_EF100(enp) || EFX_FAMILY_IS_EF10(enp));

	/*
	 * Use privilege mask state at MCDI attach.
	 */

	switch (id) {
	case EFX_MCDI_FEATURE_FW_UPDATE:
		/*
		 * Admin privilege must be used prior to introduction of
		 * specific flag.
		 */
		*supportedp =
		    EFX_MCDI_HAVE_PRIVILEGE(privilege_mask, ADMIN);
		break;
	case EFX_MCDI_FEATURE_LINK_CONTROL:
		/*
		 * Admin privilege used prior to introduction of
		 * specific flag.
		 */
		*supportedp =
		    EFX_MCDI_HAVE_PRIVILEGE(privilege_mask, LINK) ||
		    EFX_MCDI_HAVE_PRIVILEGE(privilege_mask, ADMIN);
		break;
	case EFX_MCDI_FEATURE_MACADDR_CHANGE:
		/*
		 * Admin privilege must be used prior to introduction of
		 * mac spoofing privilege (at v4.6), which is used up to
		 * introduction of change mac spoofing privilege (at v4.7)
		 */
		*supportedp =
		    EFX_MCDI_HAVE_PRIVILEGE(privilege_mask, CHANGE_MAC) ||
		    EFX_MCDI_HAVE_PRIVILEGE(privilege_mask, MAC_SPOOFING) ||
		    EFX_MCDI_HAVE_PRIVILEGE(privilege_mask, ADMIN);
		break;
	case EFX_MCDI_FEATURE_MAC_SPOOFING:
		/*
		 * Admin privilege must be used prior to introduction of
		 * mac spoofing privilege (at v4.6), which is used up to
		 * introduction of mac spoofing TX privilege (at v4.7)
		 */
		*supportedp =
		    EFX_MCDI_HAVE_PRIVILEGE(privilege_mask, MAC_SPOOFING_TX) ||
		    EFX_MCDI_HAVE_PRIVILEGE(privilege_mask, MAC_SPOOFING) ||
		    EFX_MCDI_HAVE_PRIVILEGE(privilege_mask, ADMIN);
		break;
	default:
		rc = ENOTSUP;
		goto fail1;
	}

	return (0);

fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

#endif	/* EFSYS_OPT_MCDI */

#endif	/* EFSYS_OPT_RIVERHEAD || EFX_OPTS_EF10() */
