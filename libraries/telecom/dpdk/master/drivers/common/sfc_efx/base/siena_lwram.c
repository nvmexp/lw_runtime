/* SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright(c) 2019-2020 Xilinx, Inc.
 * Copyright(c) 2009-2019 Solarflare Communications Inc.
 */

#include "efx.h"
#include "efx_impl.h"

#if EFSYS_OPT_SIENA

#if EFSYS_OPT_VPD || EFSYS_OPT_LWRAM

	__checkReturn		efx_rc_t
siena_lwram_partn_size(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__out			size_t *sizep)
{
	efx_rc_t rc;
	efx_lwram_info_t eni = { 0 };

	if ((1 << partn) & ~enp->en_u.siena.enu_partn_mask) {
		rc = ENOTSUP;
		goto fail1;
	}

	if ((rc = efx_mcdi_lwram_info(enp, partn, &eni)) != 0)
		goto fail2;

	*sizep = eni.eni_partn_size;

	return (0);

fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

	__checkReturn		efx_rc_t
siena_lwram_partn_info(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__out			efx_lwram_info_t * enip)
{
	efx_rc_t rc;

	if ((rc = efx_mcdi_lwram_info(enp, partn, enip)) != 0)
		goto fail1;

	if (enip->eni_write_size == 0)
		enip->eni_write_size = SIENA_LWRAM_CHUNK;

	return (0);

fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}


	__checkReturn		efx_rc_t
siena_lwram_partn_lock(
	__in			efx_nic_t *enp,
	__in			uint32_t partn)
{
	efx_rc_t rc;

	if ((rc = efx_mcdi_lwram_update_start(enp, partn)) != 0) {
		goto fail1;
	}

	return (0);

fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

	__checkReturn		efx_rc_t
siena_lwram_partn_read(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__in			unsigned int offset,
	__out_bcount(size)	caddr_t data,
	__in			size_t size)
{
	size_t chunk;
	efx_rc_t rc;

	while (size > 0) {
		chunk = MIN(size, SIENA_LWRAM_CHUNK);

		if ((rc = efx_mcdi_lwram_read(enp, partn, offset, data, chunk,
			    MC_CMD_LWRAM_READ_IN_V2_DEFAULT)) != 0) {
			goto fail1;
		}

		size -= chunk;
		data += chunk;
		offset += chunk;
	}

	return (0);

fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

	__checkReturn		efx_rc_t
siena_lwram_partn_erase(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__in			unsigned int offset,
	__in			size_t size)
{
	efx_rc_t rc;

	if ((rc = efx_mcdi_lwram_erase(enp, partn, offset, size)) != 0) {
		goto fail1;
	}

	return (0);

fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

	__checkReturn		efx_rc_t
siena_lwram_partn_write(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__in			unsigned int offset,
	__out_bcount(size)	caddr_t data,
	__in			size_t size)
{
	size_t chunk;
	efx_rc_t rc;

	while (size > 0) {
		chunk = MIN(size, SIENA_LWRAM_CHUNK);

		if ((rc = efx_mcdi_lwram_write(enp, partn, offset,
			    data, chunk)) != 0) {
			goto fail1;
		}

		size -= chunk;
		data += chunk;
		offset += chunk;
	}

	return (0);

fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

	__checkReturn		efx_rc_t
siena_lwram_partn_unlock(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__out_opt		uint32_t *verify_resultp)
{
	boolean_t reboot;
	uint32_t flags = 0;
	efx_rc_t rc;

	/*
	 * Reboot into the new image only for PHYs. The driver has to
	 * explicitly cope with an MC reboot after a firmware update.
	 */
	reboot = (partn == MC_CMD_LWRAM_TYPE_PHY_PORT0 ||
		    partn == MC_CMD_LWRAM_TYPE_PHY_PORT1 ||
		    partn == MC_CMD_LWRAM_TYPE_DISABLED_CALLISTO);

	rc = efx_mcdi_lwram_update_finish(enp, partn, reboot, flags,
		    verify_resultp);
	if (rc != 0)
		goto fail1;

	return (0);

fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

#endif	/* EFSYS_OPT_VPD || EFSYS_OPT_LWRAM */

#if EFSYS_OPT_LWRAM

typedef struct siena_parttbl_entry_s {
	unsigned int		partn;
	unsigned int		port;
	efx_lwram_type_t	lwtype;
} siena_parttbl_entry_t;

static siena_parttbl_entry_t siena_parttbl[] = {
	{MC_CMD_LWRAM_TYPE_DISABLED_CALLISTO,	1, EFX_LWRAM_NULLPHY},
	{MC_CMD_LWRAM_TYPE_DISABLED_CALLISTO,	2, EFX_LWRAM_NULLPHY},
	{MC_CMD_LWRAM_TYPE_MC_FW,		1, EFX_LWRAM_MC_FIRMWARE},
	{MC_CMD_LWRAM_TYPE_MC_FW,		2, EFX_LWRAM_MC_FIRMWARE},
	{MC_CMD_LWRAM_TYPE_MC_FW_BACKUP,	1, EFX_LWRAM_MC_GOLDEN},
	{MC_CMD_LWRAM_TYPE_MC_FW_BACKUP,	2, EFX_LWRAM_MC_GOLDEN},
	{MC_CMD_LWRAM_TYPE_EXP_ROM,		1, EFX_LWRAM_BOOTROM},
	{MC_CMD_LWRAM_TYPE_EXP_ROM,		2, EFX_LWRAM_BOOTROM},
	{MC_CMD_LWRAM_TYPE_EXP_ROM_CFG_PORT0,	1, EFX_LWRAM_BOOTROM_CFG},
	{MC_CMD_LWRAM_TYPE_EXP_ROM_CFG_PORT1,	2, EFX_LWRAM_BOOTROM_CFG},
	{MC_CMD_LWRAM_TYPE_PHY_PORT0,		1, EFX_LWRAM_PHY},
	{MC_CMD_LWRAM_TYPE_PHY_PORT1,		2, EFX_LWRAM_PHY},
	{MC_CMD_LWRAM_TYPE_FPGA,		1, EFX_LWRAM_FPGA},
	{MC_CMD_LWRAM_TYPE_FPGA,		2, EFX_LWRAM_FPGA},
	{MC_CMD_LWRAM_TYPE_FPGA_BACKUP,		1, EFX_LWRAM_FPGA_BACKUP},
	{MC_CMD_LWRAM_TYPE_FPGA_BACKUP,		2, EFX_LWRAM_FPGA_BACKUP},
	{MC_CMD_LWRAM_TYPE_FC_FW,		1, EFX_LWRAM_FCFW},
	{MC_CMD_LWRAM_TYPE_FC_FW,		2, EFX_LWRAM_FCFW},
	{MC_CMD_LWRAM_TYPE_CPLD,		1, EFX_LWRAM_CPLD},
	{MC_CMD_LWRAM_TYPE_CPLD,		2, EFX_LWRAM_CPLD},
	{MC_CMD_LWRAM_TYPE_LICENSE,		1, EFX_LWRAM_LICENSE},
	{MC_CMD_LWRAM_TYPE_LICENSE,		2, EFX_LWRAM_LICENSE}
};

	__checkReturn		efx_rc_t
siena_lwram_type_to_partn(
	__in			efx_nic_t *enp,
	__in			efx_lwram_type_t type,
	__out			uint32_t *partnp)
{
	efx_mcdi_iface_t *emip = &(enp->en_mcdi.em_emip);
	unsigned int i;

	EFSYS_ASSERT3U(type, !=, EFX_LWRAM_ILWALID);
	EFSYS_ASSERT3U(type, <, EFX_LWRAM_NTYPES);
	EFSYS_ASSERT(partnp != NULL);

	for (i = 0; i < EFX_ARRAY_SIZE(siena_parttbl); i++) {
		siena_parttbl_entry_t *entry = &siena_parttbl[i];

		if (entry->port == emip->emi_port && entry->lwtype == type) {
			*partnp = entry->partn;
			return (0);
		}
	}

	return (ENOTSUP);
}


#if EFSYS_OPT_DIAG

	__checkReturn		efx_rc_t
siena_lwram_test(
	__in			efx_nic_t *enp)
{
	efx_mcdi_iface_t *emip = &(enp->en_mcdi.em_emip);
	siena_parttbl_entry_t *entry;
	unsigned int i;
	efx_rc_t rc;

	/*
	 * Iterate over the list of supported partition types
	 * applicable to *this* port
	 */
	for (i = 0; i < EFX_ARRAY_SIZE(siena_parttbl); i++) {
		entry = &siena_parttbl[i];

		if (entry->port != emip->emi_port ||
		    !(enp->en_u.siena.enu_partn_mask & (1 << entry->partn)))
			continue;

		if ((rc = efx_mcdi_lwram_test(enp, entry->partn)) != 0) {
			goto fail1;
		}
	}

	return (0);

fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

#endif	/* EFSYS_OPT_DIAG */


#define	SIENA_DYNAMIC_CFG_SIZE(_nitems)					\
	(sizeof (siena_mc_dynamic_config_hdr_t) + ((_nitems) *		\
	sizeof (((siena_mc_dynamic_config_hdr_t *)NULL)->fw_version[0])))

	__checkReturn		efx_rc_t
siena_lwram_get_dynamic_cfg(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__in			boolean_t vpd,
	__out			siena_mc_dynamic_config_hdr_t **dcfgp,
	__out			size_t *sizep)
{
	siena_mc_dynamic_config_hdr_t *dcfg = NULL;
	size_t size;
	uint8_t cksum;
	unsigned int vpd_offset;
	unsigned int vpd_length;
	unsigned int hdr_length;
	unsigned int lwersions;
	unsigned int pos;
	unsigned int region;
	efx_rc_t rc;

	EFSYS_ASSERT(partn == MC_CMD_LWRAM_TYPE_DYNAMIC_CFG_PORT0 ||
		    partn == MC_CMD_LWRAM_TYPE_DYNAMIC_CFG_PORT1);

	/*
	 * Allocate sufficient memory for the entire dynamiccfg area, even
	 * if we're not actually going to read in the VPD.
	 */
	if ((rc = siena_lwram_partn_size(enp, partn, &size)) != 0)
		goto fail1;

	if (size < SIENA_LWRAM_CHUNK) {
		rc = EILWAL;
		goto fail2;
	}

	EFSYS_KMEM_ALLOC(enp->en_esip, size, dcfg);
	if (dcfg == NULL) {
		rc = ENOMEM;
		goto fail3;
	}

	if ((rc = siena_lwram_partn_read(enp, partn, 0,
	    (caddr_t)dcfg, SIENA_LWRAM_CHUNK)) != 0)
		goto fail4;

	/* Verify the magic */
	if (EFX_DWORD_FIELD(dcfg->magic, EFX_DWORD_0)
	    != SIENA_MC_DYNAMIC_CONFIG_MAGIC)
		goto ilwalid1;

	/* All future versions of the structure must be backwards compatible */
	EFX_STATIC_ASSERT(SIENA_MC_DYNAMIC_CONFIG_VERSION == 0);

	hdr_length = EFX_WORD_FIELD(dcfg->length, EFX_WORD_0);
	lwersions = EFX_DWORD_FIELD(dcfg->num_fw_version_items, EFX_DWORD_0);
	vpd_offset = EFX_DWORD_FIELD(dcfg->dynamic_vpd_offset, EFX_DWORD_0);
	vpd_length = EFX_DWORD_FIELD(dcfg->dynamic_vpd_length, EFX_DWORD_0);

	/* Verify the hdr doesn't overflow the partn size */
	if (hdr_length > size || vpd_offset > size || vpd_length > size ||
	    vpd_length + vpd_offset > size)
		goto ilwalid2;

	/* Verify the header has room for all it's versions */
	if (hdr_length < SIENA_DYNAMIC_CFG_SIZE(0) ||
	    hdr_length < SIENA_DYNAMIC_CFG_SIZE(lwersions))
		goto ilwalid3;

	/*
	 * Read the remaining portion of the dcfg, either including
	 * the whole of VPD (there is no vpd length in this structure,
	 * so we have to parse each tag), or just the dcfg header itself
	 */
	region = vpd ? vpd_offset + vpd_length : hdr_length;
	if (region > SIENA_LWRAM_CHUNK) {
		if ((rc = siena_lwram_partn_read(enp, partn, SIENA_LWRAM_CHUNK,
		    (caddr_t)dcfg + SIENA_LWRAM_CHUNK,
		    region - SIENA_LWRAM_CHUNK)) != 0)
			goto fail5;
	}

	/* Verify checksum */
	cksum = 0;
	for (pos = 0; pos < hdr_length; pos++)
		cksum += ((uint8_t *)dcfg)[pos];
	if (cksum != 0)
		goto ilwalid4;

	goto done;

ilwalid4:
	EFSYS_PROBE(ilwalid4);
ilwalid3:
	EFSYS_PROBE(ilwalid3);
ilwalid2:
	EFSYS_PROBE(ilwalid2);
ilwalid1:
	EFSYS_PROBE(ilwalid1);

	/*
	 * Construct a new "null" dcfg, with an empty version vector,
	 * and an empty VPD chunk trailing. This has the neat side effect
	 * of testing the exception paths in the write path.
	 */
	EFX_POPULATE_DWORD_1(dcfg->magic,
			    EFX_DWORD_0, SIENA_MC_DYNAMIC_CONFIG_MAGIC);
	EFX_POPULATE_WORD_1(dcfg->length, EFX_WORD_0, sizeof (*dcfg));
	EFX_POPULATE_BYTE_1(dcfg->version, EFX_BYTE_0,
			    SIENA_MC_DYNAMIC_CONFIG_VERSION);
	EFX_POPULATE_DWORD_1(dcfg->dynamic_vpd_offset,
			    EFX_DWORD_0, sizeof (*dcfg));
	EFX_POPULATE_DWORD_1(dcfg->dynamic_vpd_length, EFX_DWORD_0, 0);
	EFX_POPULATE_DWORD_1(dcfg->num_fw_version_items, EFX_DWORD_0, 0);

done:
	*dcfgp = dcfg;
	*sizep = size;

	return (0);

fail5:
	EFSYS_PROBE(fail5);
fail4:
	EFSYS_PROBE(fail4);

	EFSYS_KMEM_FREE(enp->en_esip, size, dcfg);

fail3:
	EFSYS_PROBE(fail3);
fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

	__checkReturn		efx_rc_t
siena_lwram_get_subtype(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__out			uint32_t *subtypep)
{
	efx_mcdi_req_t req;
	EFX_MCDI_DECLARE_BUF(payload, MC_CMD_GET_BOARD_CFG_IN_LEN,
		MC_CMD_GET_BOARD_CFG_OUT_LENMAX);
	efx_word_t *fw_list;
	efx_rc_t rc;

	req.emr_cmd = MC_CMD_GET_BOARD_CFG;
	req.emr_in_buf = payload;
	req.emr_in_length = MC_CMD_GET_BOARD_CFG_IN_LEN;
	req.emr_out_buf = payload;
	req.emr_out_length = MC_CMD_GET_BOARD_CFG_OUT_LENMAX;

	efx_mcdi_exelwte(enp, &req);

	if (req.emr_rc != 0) {
		rc = req.emr_rc;
		goto fail1;
	}

	if (req.emr_out_length_used < MC_CMD_GET_BOARD_CFG_OUT_LENMIN) {
		rc = EMSGSIZE;
		goto fail2;
	}

	if (req.emr_out_length_used <
	    MC_CMD_GET_BOARD_CFG_OUT_FW_SUBTYPE_LIST_OFST +
	    (partn + 1) * sizeof (efx_word_t)) {
		rc = ENOENT;
		goto fail3;
	}

	fw_list = MCDI_OUT2(req, efx_word_t,
			    GET_BOARD_CFG_OUT_FW_SUBTYPE_LIST);
	*subtypep = EFX_WORD_FIELD(fw_list[partn], EFX_WORD_0);

	return (0);

fail3:
	EFSYS_PROBE(fail3);
fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

	__checkReturn		efx_rc_t
siena_lwram_partn_get_version(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__out			uint32_t *subtypep,
	__out_ecount(4)		uint16_t version[4])
{
	siena_mc_dynamic_config_hdr_t *dcfg;
	siena_parttbl_entry_t *entry;
	uint32_t dcfg_partn;
	unsigned int i;
	efx_rc_t rc;

	if ((1 << partn) & ~enp->en_u.siena.enu_partn_mask) {
		rc = ENOTSUP;
		goto fail1;
	}

	if ((rc = siena_lwram_get_subtype(enp, partn, subtypep)) != 0)
		goto fail2;

	/*
	 * Some partitions are accessible from both ports (for instance BOOTROM)
	 * Find the highest version reported by all dcfg structures on ports
	 * that have access to this partition.
	 */
	version[0] = version[1] = version[2] = version[3] = 0;
	for (i = 0; i < EFX_ARRAY_SIZE(siena_parttbl); i++) {
		siena_mc_fw_version_t *verp;
		unsigned int nitems;
		uint16_t temp[4];
		size_t length;

		entry = &siena_parttbl[i];
		if (entry->partn != partn)
			continue;

		dcfg_partn = (entry->port == 1)
			? MC_CMD_LWRAM_TYPE_DYNAMIC_CFG_PORT0
			: MC_CMD_LWRAM_TYPE_DYNAMIC_CFG_PORT1;
		/*
		 * Ingore missing partitions on port 2, assuming they're due
		 * to to running on a single port part.
		 */
		if ((1 << dcfg_partn) &  ~enp->en_u.siena.enu_partn_mask) {
			if (entry->port == 2)
				continue;
		}

		if ((rc = siena_lwram_get_dynamic_cfg(enp, dcfg_partn,
		    B_FALSE, &dcfg, &length)) != 0)
			goto fail3;

		nitems = EFX_DWORD_FIELD(dcfg->num_fw_version_items,
			    EFX_DWORD_0);
		if (nitems < entry->partn)
			goto done;

		verp = &dcfg->fw_version[partn];
		temp[0] = EFX_WORD_FIELD(verp->version_w, EFX_WORD_0);
		temp[1] = EFX_WORD_FIELD(verp->version_x, EFX_WORD_0);
		temp[2] = EFX_WORD_FIELD(verp->version_y, EFX_WORD_0);
		temp[3] = EFX_WORD_FIELD(verp->version_z, EFX_WORD_0);
		if (memcmp(version, temp, sizeof (temp)) < 0)
			memcpy(version, temp, sizeof (temp));

done:
		EFSYS_KMEM_FREE(enp->en_esip, length, dcfg);
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

	__checkReturn		efx_rc_t
siena_lwram_partn_rw_start(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__out			size_t *chunk_sizep)
{
	efx_rc_t rc;

	if ((rc = siena_lwram_partn_lock(enp, partn)) != 0)
		goto fail1;

	if (chunk_sizep != NULL)
		*chunk_sizep = SIENA_LWRAM_CHUNK;

	return (0);

fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

	__checkReturn		efx_rc_t
siena_lwram_partn_rw_finish(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__out_opt		uint32_t *verify_resultp)
{
	efx_rc_t rc;

	if ((rc = siena_lwram_partn_unlock(enp, partn, verify_resultp)) != 0)
		goto fail1;

	return (0);

fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

	__checkReturn		efx_rc_t
siena_lwram_partn_set_version(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__in_ecount(4)		uint16_t version[4])
{
	efx_mcdi_iface_t *emip = &(enp->en_mcdi.em_emip);
	siena_mc_dynamic_config_hdr_t *dcfg = NULL;
	siena_mc_fw_version_t *fwverp;
	uint32_t dcfg_partn;
	size_t dcfg_size;
	unsigned int hdr_length;
	unsigned int vpd_length;
	unsigned int vpd_offset;
	unsigned int nitems;
	unsigned int required_hdr_length;
	unsigned int pos;
	uint8_t cksum;
	uint32_t subtype;
	size_t length;
	efx_rc_t rc;

	dcfg_partn = (emip->emi_port == 1)
		? MC_CMD_LWRAM_TYPE_DYNAMIC_CFG_PORT0
		: MC_CMD_LWRAM_TYPE_DYNAMIC_CFG_PORT1;

	if ((rc = siena_lwram_partn_size(enp, dcfg_partn, &dcfg_size)) != 0)
		goto fail1;

	if ((rc = siena_lwram_partn_lock(enp, dcfg_partn)) != 0)
		goto fail2;

	if ((rc = siena_lwram_get_dynamic_cfg(enp, dcfg_partn,
	    B_TRUE, &dcfg, &length)) != 0)
		goto fail3;

	hdr_length = EFX_WORD_FIELD(dcfg->length, EFX_WORD_0);
	nitems = EFX_DWORD_FIELD(dcfg->num_fw_version_items, EFX_DWORD_0);
	vpd_length = EFX_DWORD_FIELD(dcfg->dynamic_vpd_length, EFX_DWORD_0);
	vpd_offset = EFX_DWORD_FIELD(dcfg->dynamic_vpd_offset, EFX_DWORD_0);

	/*
	 * NOTE: This function will blatt any fields trailing the version
	 * vector, or the VPD chunk.
	 */
	required_hdr_length = SIENA_DYNAMIC_CFG_SIZE(partn + 1);
	if (required_hdr_length + vpd_length > length) {
		rc = ENOSPC;
		goto fail4;
	}

	if (vpd_offset < required_hdr_length) {
		(void) memmove((caddr_t)dcfg + required_hdr_length,
			(caddr_t)dcfg + vpd_offset, vpd_length);
		vpd_offset = required_hdr_length;
		EFX_POPULATE_DWORD_1(dcfg->dynamic_vpd_offset,
				    EFX_DWORD_0, vpd_offset);
	}

	if (hdr_length < required_hdr_length) {
		(void) memset((caddr_t)dcfg + hdr_length, 0,
			required_hdr_length - hdr_length);
		hdr_length = required_hdr_length;
		EFX_POPULATE_WORD_1(dcfg->length,
				    EFX_WORD_0, hdr_length);
	}

	/* Get the subtype to insert into the fw_subtype array */
	if ((rc = siena_lwram_get_subtype(enp, partn, &subtype)) != 0)
		goto fail5;

	/* Fill out the new version */
	fwverp = &dcfg->fw_version[partn];
	EFX_POPULATE_DWORD_1(fwverp->fw_subtype, EFX_DWORD_0, subtype);
	EFX_POPULATE_WORD_1(fwverp->version_w, EFX_WORD_0, version[0]);
	EFX_POPULATE_WORD_1(fwverp->version_x, EFX_WORD_0, version[1]);
	EFX_POPULATE_WORD_1(fwverp->version_y, EFX_WORD_0, version[2]);
	EFX_POPULATE_WORD_1(fwverp->version_z, EFX_WORD_0, version[3]);

	/* Update the version count */
	if (nitems < partn + 1) {
		nitems = partn + 1;
		EFX_POPULATE_DWORD_1(dcfg->num_fw_version_items,
				    EFX_DWORD_0, nitems);
	}

	/* Update the checksum */
	cksum = 0;
	for (pos = 0; pos < hdr_length; pos++)
		cksum += ((uint8_t *)dcfg)[pos];
	dcfg->csum.eb_u8[0] -= cksum;

	/* Erase and write the new partition */
	if ((rc = siena_lwram_partn_erase(enp, dcfg_partn, 0, dcfg_size)) != 0)
		goto fail6;

	/* Write out the new structure to lwram */
	if ((rc = siena_lwram_partn_write(enp, dcfg_partn, 0,
	    (caddr_t)dcfg, vpd_offset + vpd_length)) != 0)
		goto fail7;

	EFSYS_KMEM_FREE(enp->en_esip, length, dcfg);

	siena_lwram_partn_unlock(enp, dcfg_partn, NULL);

	return (0);

fail7:
	EFSYS_PROBE(fail7);
fail6:
	EFSYS_PROBE(fail6);
fail5:
	EFSYS_PROBE(fail5);
fail4:
	EFSYS_PROBE(fail4);

	EFSYS_KMEM_FREE(enp->en_esip, length, dcfg);
fail3:
	EFSYS_PROBE(fail3);
fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

#endif	/* EFSYS_OPT_LWRAM */

#endif	/* EFSYS_OPT_SIENA */
