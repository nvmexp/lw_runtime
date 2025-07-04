/* SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright(c) 2019-2020 Xilinx, Inc.
 * Copyright(c) 2009-2019 Solarflare Communications Inc.
 */

#include "efx.h"
#include "efx_impl.h"

#if EFSYS_OPT_LWRAM

#if EFSYS_OPT_SIENA

static const efx_lwram_ops_t	__efx_lwram_siena_ops = {
#if EFSYS_OPT_DIAG
	siena_lwram_test,		/* elwo_test */
#endif	/* EFSYS_OPT_DIAG */
	siena_lwram_type_to_partn,	/* elwo_type_to_partn */
	siena_lwram_partn_info,		/* elwo_partn_info */
	siena_lwram_partn_rw_start,	/* elwo_partn_rw_start */
	siena_lwram_partn_read,		/* elwo_partn_read */
	siena_lwram_partn_read,		/* elwo_partn_read_backup */
	siena_lwram_partn_erase,	/* elwo_partn_erase */
	siena_lwram_partn_write,	/* elwo_partn_write */
	siena_lwram_partn_rw_finish,	/* elwo_partn_rw_finish */
	siena_lwram_partn_get_version,	/* elwo_partn_get_version */
	siena_lwram_partn_set_version,	/* elwo_partn_set_version */
	NULL,				/* elwo_partn_validate */
};

#endif	/* EFSYS_OPT_SIENA */

#if EFX_OPTS_EF10()

static const efx_lwram_ops_t	__efx_lwram_ef10_ops = {
#if EFSYS_OPT_DIAG
	ef10_lwram_test,		/* elwo_test */
#endif	/* EFSYS_OPT_DIAG */
	ef10_lwram_type_to_partn,	/* elwo_type_to_partn */
	ef10_lwram_partn_info,		/* elwo_partn_info */
	ef10_lwram_partn_rw_start,	/* elwo_partn_rw_start */
	ef10_lwram_partn_read,		/* elwo_partn_read */
	ef10_lwram_partn_read_backup,	/* elwo_partn_read_backup */
	ef10_lwram_partn_erase,		/* elwo_partn_erase */
	ef10_lwram_partn_write,		/* elwo_partn_write */
	ef10_lwram_partn_rw_finish,	/* elwo_partn_rw_finish */
	ef10_lwram_partn_get_version,	/* elwo_partn_get_version */
	ef10_lwram_partn_set_version,	/* elwo_partn_set_version */
	ef10_lwram_buffer_validate,	/* elwo_buffer_validate */
};

#endif	/* EFX_OPTS_EF10() */

	__checkReturn	efx_rc_t
efx_lwram_init(
	__in		efx_nic_t *enp)
{
	const efx_lwram_ops_t *elwop;
	efx_rc_t rc;

	EFSYS_ASSERT3U(enp->en_magic, ==, EFX_NIC_MAGIC);
	EFSYS_ASSERT3U(enp->en_mod_flags, &, EFX_MOD_PROBE);
	EFSYS_ASSERT(!(enp->en_mod_flags & EFX_MOD_LWRAM));

	switch (enp->en_family) {
#if EFSYS_OPT_SIENA
	case EFX_FAMILY_SIENA:
		elwop = &__efx_lwram_siena_ops;
		break;
#endif	/* EFSYS_OPT_SIENA */

#if EFSYS_OPT_HUNTINGTON
	case EFX_FAMILY_HUNTINGTON:
		elwop = &__efx_lwram_ef10_ops;
		break;
#endif	/* EFSYS_OPT_HUNTINGTON */

#if EFSYS_OPT_MEDFORD
	case EFX_FAMILY_MEDFORD:
		elwop = &__efx_lwram_ef10_ops;
		break;
#endif	/* EFSYS_OPT_MEDFORD */

#if EFSYS_OPT_MEDFORD2
	case EFX_FAMILY_MEDFORD2:
		elwop = &__efx_lwram_ef10_ops;
		break;
#endif	/* EFSYS_OPT_MEDFORD2 */

	default:
		EFSYS_ASSERT(0);
		rc = ENOTSUP;
		goto fail1;
	}

	enp->en_elwop = elwop;
	enp->en_mod_flags |= EFX_MOD_LWRAM;

	enp->en_lwram_partn_locked = EFX_LWRAM_PARTN_ILWALID;

	return (0);

fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

#if EFSYS_OPT_DIAG

	__checkReturn		efx_rc_t
efx_lwram_test(
	__in			efx_nic_t *enp)
{
	const efx_lwram_ops_t *elwop = enp->en_elwop;
	efx_rc_t rc;

	EFSYS_ASSERT3U(enp->en_magic, ==, EFX_NIC_MAGIC);
	EFSYS_ASSERT3U(enp->en_mod_flags, &, EFX_MOD_LWRAM);

	if ((rc = elwop->elwo_test(enp)) != 0)
		goto fail1;

	return (0);

fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

#endif	/* EFSYS_OPT_DIAG */

	__checkReturn		efx_rc_t
efx_lwram_size(
	__in			efx_nic_t *enp,
	__in			efx_lwram_type_t type,
	__out			size_t *sizep)
{
	const efx_lwram_ops_t *elwop = enp->en_elwop;
	efx_lwram_info_t eni = { 0 };
	uint32_t partn;
	efx_rc_t rc;

	EFSYS_ASSERT3U(enp->en_magic, ==, EFX_NIC_MAGIC);
	EFSYS_ASSERT3U(enp->en_mod_flags, &, EFX_MOD_LWRAM);

	if ((rc = elwop->elwo_type_to_partn(enp, type, &partn)) != 0)
		goto fail1;

	if ((rc = elwop->elwo_partn_info(enp, partn, &eni)) != 0)
		goto fail2;

	*sizep = eni.eni_partn_size;

	return (0);

fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);
	*sizep = 0;

	return (rc);
}

extern	__checkReturn		efx_rc_t
efx_lwram_info(
	__in			efx_nic_t *enp,
	__in			efx_lwram_type_t type,
	__out			efx_lwram_info_t *enip)
{
	const efx_lwram_ops_t *elwop = enp->en_elwop;
	uint32_t partn;
	efx_rc_t rc;

	EFSYS_ASSERT3U(enp->en_magic, ==, EFX_NIC_MAGIC);
	EFSYS_ASSERT3U(enp->en_mod_flags, &, EFX_MOD_LWRAM);

	if ((rc = elwop->elwo_type_to_partn(enp, type, &partn)) != 0)
		goto fail1;

	if ((rc = elwop->elwo_partn_info(enp, partn, enip)) != 0)
		goto fail2;

	return (0);

fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}


	__checkReturn		efx_rc_t
efx_lwram_get_version(
	__in			efx_nic_t *enp,
	__in			efx_lwram_type_t type,
	__out			uint32_t *subtypep,
	__out_ecount(4)		uint16_t version[4])
{
	const efx_lwram_ops_t *elwop = enp->en_elwop;
	uint32_t partn;
	efx_rc_t rc;

	EFSYS_ASSERT3U(enp->en_magic, ==, EFX_NIC_MAGIC);
	EFSYS_ASSERT3U(enp->en_mod_flags, &, EFX_MOD_PROBE);
	EFSYS_ASSERT3U(enp->en_mod_flags, &, EFX_MOD_LWRAM);

	if ((rc = elwop->elwo_type_to_partn(enp, type, &partn)) != 0)
		goto fail1;

	if ((rc = elwop->elwo_partn_get_version(enp, partn,
		    subtypep, version)) != 0)
		goto fail2;

	return (0);

fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

	__checkReturn		efx_rc_t
efx_lwram_rw_start(
	__in			efx_nic_t *enp,
	__in			efx_lwram_type_t type,
	__out_opt		size_t *chunk_sizep)
{
	const efx_lwram_ops_t *elwop = enp->en_elwop;
	uint32_t partn;
	efx_rc_t rc;

	EFSYS_ASSERT3U(enp->en_magic, ==, EFX_NIC_MAGIC);
	EFSYS_ASSERT3U(enp->en_mod_flags, &, EFX_MOD_LWRAM);

	if ((rc = elwop->elwo_type_to_partn(enp, type, &partn)) != 0)
		goto fail1;

	EFSYS_ASSERT3U(enp->en_lwram_partn_locked, ==, EFX_LWRAM_PARTN_ILWALID);

	if ((rc = elwop->elwo_partn_rw_start(enp, partn, chunk_sizep)) != 0)
		goto fail2;

	enp->en_lwram_partn_locked = partn;

	return (0);

fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

	__checkReturn		efx_rc_t
efx_lwram_read_chunk(
	__in			efx_nic_t *enp,
	__in			efx_lwram_type_t type,
	__in			unsigned int offset,
	__out_bcount(size)	caddr_t data,
	__in			size_t size)
{
	const efx_lwram_ops_t *elwop = enp->en_elwop;
	uint32_t partn;
	efx_rc_t rc;

	EFSYS_ASSERT3U(enp->en_magic, ==, EFX_NIC_MAGIC);
	EFSYS_ASSERT3U(enp->en_mod_flags, &, EFX_MOD_LWRAM);

	if ((rc = elwop->elwo_type_to_partn(enp, type, &partn)) != 0)
		goto fail1;

	EFSYS_ASSERT3U(enp->en_lwram_partn_locked, ==, partn);

	if ((rc = elwop->elwo_partn_read(enp, partn, offset, data, size)) != 0)
		goto fail2;

	return (0);

fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

/*
 * Read from the backup (writeable) store of an A/B partition.
 * For non A/B partitions, there is only a single store, and so this
 * function has the same behaviour as efx_lwram_read_chunk().
 */
	__checkReturn		efx_rc_t
efx_lwram_read_backup(
	__in			efx_nic_t *enp,
	__in			efx_lwram_type_t type,
	__in			unsigned int offset,
	__out_bcount(size)	caddr_t data,
	__in			size_t size)
{
	const efx_lwram_ops_t *elwop = enp->en_elwop;
	uint32_t partn;
	efx_rc_t rc;

	EFSYS_ASSERT3U(enp->en_magic, ==, EFX_NIC_MAGIC);
	EFSYS_ASSERT3U(enp->en_mod_flags, &, EFX_MOD_LWRAM);

	if ((rc = elwop->elwo_type_to_partn(enp, type, &partn)) != 0)
		goto fail1;

	EFSYS_ASSERT3U(enp->en_lwram_partn_locked, ==, partn);

	if ((rc = elwop->elwo_partn_read_backup(enp, partn, offset,
		    data, size)) != 0)
		goto fail2;

	return (0);

fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

	__checkReturn		efx_rc_t
efx_lwram_erase(
	__in			efx_nic_t *enp,
	__in			efx_lwram_type_t type)
{
	const efx_lwram_ops_t *elwop = enp->en_elwop;
	unsigned int offset = 0;
	efx_lwram_info_t eni = { 0 };
	uint32_t partn;
	efx_rc_t rc;

	EFSYS_ASSERT3U(enp->en_magic, ==, EFX_NIC_MAGIC);
	EFSYS_ASSERT3U(enp->en_mod_flags, &, EFX_MOD_LWRAM);

	if ((rc = elwop->elwo_type_to_partn(enp, type, &partn)) != 0)
		goto fail1;

	EFSYS_ASSERT3U(enp->en_lwram_partn_locked, ==, partn);

	if ((rc = elwop->elwo_partn_info(enp, partn, &eni)) != 0)
		goto fail2;

	if ((rc = elwop->elwo_partn_erase(enp, partn, offset,
		    eni.eni_partn_size)) != 0)
		goto fail3;

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
efx_lwram_write_chunk(
	__in			efx_nic_t *enp,
	__in			efx_lwram_type_t type,
	__in			unsigned int offset,
	__in_bcount(size)	caddr_t data,
	__in			size_t size)
{
	const efx_lwram_ops_t *elwop = enp->en_elwop;
	uint32_t partn;
	efx_rc_t rc;

	EFSYS_ASSERT3U(enp->en_magic, ==, EFX_NIC_MAGIC);
	EFSYS_ASSERT3U(enp->en_mod_flags, &, EFX_MOD_LWRAM);

	if ((rc = elwop->elwo_type_to_partn(enp, type, &partn)) != 0)
		goto fail1;

	EFSYS_ASSERT3U(enp->en_lwram_partn_locked, ==, partn);

	if ((rc = elwop->elwo_partn_write(enp, partn, offset, data, size)) != 0)
		goto fail2;

	return (0);

fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

	__checkReturn		efx_rc_t
efx_lwram_rw_finish(
	__in			efx_nic_t *enp,
	__in			efx_lwram_type_t type,
	__out_opt		uint32_t *verify_resultp)
{
	const efx_lwram_ops_t *elwop = enp->en_elwop;
	uint32_t partn;
	uint32_t verify_result = 0;
	efx_rc_t rc;

	EFSYS_ASSERT3U(enp->en_magic, ==, EFX_NIC_MAGIC);
	EFSYS_ASSERT3U(enp->en_mod_flags, &, EFX_MOD_LWRAM);

	if ((rc = elwop->elwo_type_to_partn(enp, type, &partn)) != 0)
		goto fail1;

	EFSYS_ASSERT3U(enp->en_lwram_partn_locked, ==, partn);

	if ((rc = elwop->elwo_partn_rw_finish(enp, partn, &verify_result)) != 0)
		goto fail2;

	enp->en_lwram_partn_locked = EFX_LWRAM_PARTN_ILWALID;

	if (verify_resultp != NULL)
		*verify_resultp = verify_result;

	return (0);

fail2:
	EFSYS_PROBE(fail2);
	enp->en_lwram_partn_locked = EFX_LWRAM_PARTN_ILWALID;

fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	/* Always report verification result */
	if (verify_resultp != NULL)
		*verify_resultp = verify_result;

	return (rc);
}

	__checkReturn		efx_rc_t
efx_lwram_set_version(
	__in			efx_nic_t *enp,
	__in			efx_lwram_type_t type,
	__in_ecount(4)		uint16_t version[4])
{
	const efx_lwram_ops_t *elwop = enp->en_elwop;
	uint32_t partn;
	efx_rc_t rc;

	EFSYS_ASSERT3U(enp->en_magic, ==, EFX_NIC_MAGIC);
	EFSYS_ASSERT3U(enp->en_mod_flags, &, EFX_MOD_PROBE);
	EFSYS_ASSERT3U(enp->en_mod_flags, &, EFX_MOD_LWRAM);

	if ((rc = elwop->elwo_type_to_partn(enp, type, &partn)) != 0)
		goto fail1;

	/*
	 * The Siena implementation of elwo_set_version() will attempt to
	 * acquire the LWRAM_UPDATE lock for the DYNAMIC_CONFIG partition.
	 * Therefore, you can't have already acquired the LWRAM_UPDATE lock.
	 */
	EFSYS_ASSERT3U(enp->en_lwram_partn_locked, ==, EFX_LWRAM_PARTN_ILWALID);

	if ((rc = elwop->elwo_partn_set_version(enp, partn, version)) != 0)
		goto fail2;

	return (0);

fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

/* Validate buffer contents (before writing to flash) */
	__checkReturn		efx_rc_t
efx_lwram_validate(
	__in			efx_nic_t *enp,
	__in			efx_lwram_type_t type,
	__in_bcount(partn_size)	caddr_t partn_data,
	__in			size_t partn_size)
{
	const efx_lwram_ops_t *elwop = enp->en_elwop;
	uint32_t partn;
	efx_rc_t rc;

	EFSYS_ASSERT3U(enp->en_magic, ==, EFX_NIC_MAGIC);
	EFSYS_ASSERT3U(enp->en_mod_flags, &, EFX_MOD_PROBE);
	EFSYS_ASSERT3U(enp->en_mod_flags, &, EFX_MOD_LWRAM);

	if ((rc = elwop->elwo_type_to_partn(enp, type, &partn)) != 0)
		goto fail1;

	if (elwop->elwo_buffer_validate != NULL) {
		if ((rc = elwop->elwo_buffer_validate(partn,
			    partn_data, partn_size)) != 0)
			goto fail2;
	}

	return (0);

fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}


void
efx_lwram_fini(
	__in		efx_nic_t *enp)
{
	EFSYS_ASSERT3U(enp->en_magic, ==, EFX_NIC_MAGIC);
	EFSYS_ASSERT3U(enp->en_mod_flags, &, EFX_MOD_PROBE);
	EFSYS_ASSERT3U(enp->en_mod_flags, &, EFX_MOD_LWRAM);

	EFSYS_ASSERT3U(enp->en_lwram_partn_locked, ==, EFX_LWRAM_PARTN_ILWALID);

	enp->en_elwop = NULL;
	enp->en_mod_flags &= ~EFX_MOD_LWRAM;
}

#endif	/* EFSYS_OPT_LWRAM */

#if EFSYS_OPT_LWRAM || EFSYS_OPT_VPD

/*
 * Internal MCDI request handling
 */

	__checkReturn		efx_rc_t
efx_mcdi_lwram_partitions(
	__in			efx_nic_t *enp,
	__out_bcount(size)	caddr_t data,
	__in			size_t size,
	__out			unsigned int *npartnp)
{
	efx_mcdi_req_t req;
	EFX_MCDI_DECLARE_BUF(payload, MC_CMD_LWRAM_PARTITIONS_IN_LEN,
		MC_CMD_LWRAM_PARTITIONS_OUT_LENMAX);
	unsigned int npartn;
	efx_rc_t rc;

	req.emr_cmd = MC_CMD_LWRAM_PARTITIONS;
	req.emr_in_buf = payload;
	req.emr_in_length = MC_CMD_LWRAM_PARTITIONS_IN_LEN;
	req.emr_out_buf = payload;
	req.emr_out_length = MC_CMD_LWRAM_PARTITIONS_OUT_LENMAX;

	efx_mcdi_exelwte(enp, &req);

	if (req.emr_rc != 0) {
		rc = req.emr_rc;
		goto fail1;
	}

	if (req.emr_out_length_used < MC_CMD_LWRAM_PARTITIONS_OUT_LENMIN) {
		rc = EMSGSIZE;
		goto fail2;
	}
	npartn = MCDI_OUT_DWORD(req, LWRAM_PARTITIONS_OUT_NUM_PARTITIONS);

	if (req.emr_out_length_used < MC_CMD_LWRAM_PARTITIONS_OUT_LEN(npartn)) {
		rc = ENOENT;
		goto fail3;
	}

	if (size < npartn * sizeof (uint32_t)) {
		rc = ENOSPC;
		goto fail3;
	}

	*npartnp = npartn;

	memcpy(data,
	    MCDI_OUT2(req, uint32_t, LWRAM_PARTITIONS_OUT_TYPE_ID),
	    (npartn * sizeof (uint32_t)));

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
efx_mcdi_lwram_metadata(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__out			uint32_t *subtypep,
	__out_ecount(4)		uint16_t version[4],
	__out_bcount_opt(size)	char *descp,
	__in			size_t size)
{
	efx_mcdi_req_t req;
	EFX_MCDI_DECLARE_BUF(payload, MC_CMD_LWRAM_METADATA_IN_LEN,
		MC_CMD_LWRAM_METADATA_OUT_LENMAX);
	efx_rc_t rc;

	req.emr_cmd = MC_CMD_LWRAM_METADATA;
	req.emr_in_buf = payload;
	req.emr_in_length = MC_CMD_LWRAM_METADATA_IN_LEN;
	req.emr_out_buf = payload;
	req.emr_out_length = MC_CMD_LWRAM_METADATA_OUT_LENMAX;

	MCDI_IN_SET_DWORD(req, LWRAM_METADATA_IN_TYPE, partn);

	efx_mcdi_exelwte_quiet(enp, &req);

	if (req.emr_rc != 0) {
		rc = req.emr_rc;
		goto fail1;
	}

	if (req.emr_out_length_used < MC_CMD_LWRAM_METADATA_OUT_LENMIN) {
		rc = EMSGSIZE;
		goto fail2;
	}

	if (MCDI_OUT_DWORD_FIELD(req, LWRAM_METADATA_OUT_FLAGS,
		LWRAM_METADATA_OUT_SUBTYPE_VALID)) {
		*subtypep = MCDI_OUT_DWORD(req, LWRAM_METADATA_OUT_SUBTYPE);
	} else {
		*subtypep = 0;
	}

	if (MCDI_OUT_DWORD_FIELD(req, LWRAM_METADATA_OUT_FLAGS,
		LWRAM_METADATA_OUT_VERSION_VALID)) {
		version[0] = MCDI_OUT_WORD(req, LWRAM_METADATA_OUT_VERSION_W);
		version[1] = MCDI_OUT_WORD(req, LWRAM_METADATA_OUT_VERSION_X);
		version[2] = MCDI_OUT_WORD(req, LWRAM_METADATA_OUT_VERSION_Y);
		version[3] = MCDI_OUT_WORD(req, LWRAM_METADATA_OUT_VERSION_Z);
	} else {
		version[0] = version[1] = version[2] = version[3] = 0;
	}

	if (MCDI_OUT_DWORD_FIELD(req, LWRAM_METADATA_OUT_FLAGS,
		LWRAM_METADATA_OUT_DESCRIPTION_VALID)) {
		/* Return optional descrition string */
		if ((descp != NULL) && (size > 0)) {
			size_t desclen;

			descp[0] = '\0';
			desclen = (req.emr_out_length_used
			    - MC_CMD_LWRAM_METADATA_OUT_LEN(0));

			EFSYS_ASSERT3U(desclen, <=,
			    MC_CMD_LWRAM_METADATA_OUT_DESCRIPTION_MAXNUM);

			if (size < desclen) {
				rc = ENOSPC;
				goto fail3;
			}

			memcpy(descp, MCDI_OUT2(req, char,
				LWRAM_METADATA_OUT_DESCRIPTION),
			    desclen);

			/* Ensure string is NUL terminated */
			descp[desclen] = '\0';
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

	__checkReturn		efx_rc_t
efx_mcdi_lwram_info(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__out			efx_lwram_info_t *enip)
{
	EFX_MCDI_DECLARE_BUF(payload, MC_CMD_LWRAM_INFO_IN_LEN,
		MC_CMD_LWRAM_INFO_V2_OUT_LEN);
	efx_mcdi_req_t req;
	efx_rc_t rc;

	req.emr_cmd = MC_CMD_LWRAM_INFO;
	req.emr_in_buf = payload;
	req.emr_in_length = MC_CMD_LWRAM_INFO_IN_LEN;
	req.emr_out_buf = payload;
	req.emr_out_length = MC_CMD_LWRAM_INFO_V2_OUT_LEN;

	MCDI_IN_SET_DWORD(req, LWRAM_INFO_IN_TYPE, partn);

	efx_mcdi_exelwte_quiet(enp, &req);

	if (req.emr_rc != 0) {
		rc = req.emr_rc;
		goto fail1;
	}

	if (req.emr_out_length_used < MC_CMD_LWRAM_INFO_OUT_LEN) {
		rc = EMSGSIZE;
		goto fail2;
	}

	enip->eni_partn_size = MCDI_OUT_DWORD(req, LWRAM_INFO_OUT_SIZE);

	enip->eni_address = MCDI_OUT_DWORD(req, LWRAM_INFO_OUT_PHYSADDR);

	enip->eni_erase_size = MCDI_OUT_DWORD(req, LWRAM_INFO_OUT_ERASESIZE);

	enip->eni_write_size =
			(req.emr_out_length_used <
			    MC_CMD_LWRAM_INFO_V2_OUT_LEN) ?
			0 : MCDI_OUT_DWORD(req, LWRAM_INFO_V2_OUT_WRITESIZE);

	enip->eni_flags = 0;

	if (MCDI_OUT_DWORD_FIELD(req, LWRAM_INFO_OUT_FLAGS,
		LWRAM_INFO_OUT_PROTECTED))
		enip->eni_flags |= EFX_LWRAM_FLAG_READ_ONLY;

	if (MCDI_OUT_DWORD_FIELD(req, LWRAM_INFO_OUT_FLAGS,
		LWRAM_INFO_OUT_READ_ONLY))
		enip->eni_flags |= EFX_LWRAM_FLAG_READ_ONLY;

	return (0);

fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

/*
 * MC_CMD_LWRAM_UPDATE_START_V2 must be used to support firmware-verified
 * LWRAM updates. Older firmware will ignore the flags field in the request.
 */
	__checkReturn		efx_rc_t
efx_mcdi_lwram_update_start(
	__in			efx_nic_t *enp,
	__in			uint32_t partn)
{
	EFX_MCDI_DECLARE_BUF(payload, MC_CMD_LWRAM_UPDATE_START_V2_IN_LEN,
		MC_CMD_LWRAM_UPDATE_START_OUT_LEN);
	efx_mcdi_req_t req;
	efx_rc_t rc;

	req.emr_cmd = MC_CMD_LWRAM_UPDATE_START;
	req.emr_in_buf = payload;
	req.emr_in_length = MC_CMD_LWRAM_UPDATE_START_V2_IN_LEN;
	req.emr_out_buf = payload;
	req.emr_out_length = MC_CMD_LWRAM_UPDATE_START_OUT_LEN;

	MCDI_IN_SET_DWORD(req, LWRAM_UPDATE_START_V2_IN_TYPE, partn);

	MCDI_IN_POPULATE_DWORD_1(req, LWRAM_UPDATE_START_V2_IN_FLAGS,
	    LWRAM_UPDATE_START_V2_IN_FLAG_REPORT_VERIFY_RESULT, 1);

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

	__checkReturn		efx_rc_t
efx_mcdi_lwram_read(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__in			uint32_t offset,
	__out_bcount(size)	caddr_t data,
	__in			size_t size,
	__in			uint32_t mode)
{
	efx_mcdi_req_t req;
	EFX_MCDI_DECLARE_BUF(payload, MC_CMD_LWRAM_READ_IN_V2_LEN,
		MC_CMD_LWRAM_READ_OUT_LENMAX);
	efx_rc_t rc;

	if (size > MC_CMD_LWRAM_READ_OUT_LENMAX) {
		rc = EILWAL;
		goto fail1;
	}

	req.emr_cmd = MC_CMD_LWRAM_READ;
	req.emr_in_buf = payload;
	req.emr_in_length = MC_CMD_LWRAM_READ_IN_V2_LEN;
	req.emr_out_buf = payload;
	req.emr_out_length = MC_CMD_LWRAM_READ_OUT_LENMAX;

	MCDI_IN_SET_DWORD(req, LWRAM_READ_IN_V2_TYPE, partn);
	MCDI_IN_SET_DWORD(req, LWRAM_READ_IN_V2_OFFSET, offset);
	MCDI_IN_SET_DWORD(req, LWRAM_READ_IN_V2_LENGTH, size);
	MCDI_IN_SET_DWORD(req, LWRAM_READ_IN_V2_MODE, mode);

	efx_mcdi_exelwte(enp, &req);

	if (req.emr_rc != 0) {
		rc = req.emr_rc;
		goto fail1;
	}

	if (req.emr_out_length_used < MC_CMD_LWRAM_READ_OUT_LEN(size)) {
		rc = EMSGSIZE;
		goto fail2;
	}

	memcpy(data,
	    MCDI_OUT2(req, uint8_t, LWRAM_READ_OUT_READ_BUFFER),
	    size);

	return (0);

fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}

	__checkReturn		efx_rc_t
efx_mcdi_lwram_erase(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__in			uint32_t offset,
	__in			size_t size)
{
	efx_mcdi_req_t req;
	EFX_MCDI_DECLARE_BUF(payload, MC_CMD_LWRAM_ERASE_IN_LEN,
		MC_CMD_LWRAM_ERASE_OUT_LEN);
	efx_rc_t rc;

	req.emr_cmd = MC_CMD_LWRAM_ERASE;
	req.emr_in_buf = payload;
	req.emr_in_length = MC_CMD_LWRAM_ERASE_IN_LEN;
	req.emr_out_buf = payload;
	req.emr_out_length = MC_CMD_LWRAM_ERASE_OUT_LEN;

	MCDI_IN_SET_DWORD(req, LWRAM_ERASE_IN_TYPE, partn);
	MCDI_IN_SET_DWORD(req, LWRAM_ERASE_IN_OFFSET, offset);
	MCDI_IN_SET_DWORD(req, LWRAM_ERASE_IN_LENGTH, size);

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

/*
 * The LWRAM_WRITE MCDI command is a V1 command and so is supported by both
 * Sienna and EF10 based boards.  However EF10 based boards support the use
 * of this command with payloads up to the maximum MCDI V2 payload length.
 */
	__checkReturn		efx_rc_t
efx_mcdi_lwram_write(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__in			uint32_t offset,
	__in_bcount(size)	caddr_t data,
	__in			size_t size)
{
	efx_mcdi_req_t req;
	uint8_t *payload;
	efx_rc_t rc;
	size_t max_data_size;
	size_t payload_len = enp->en_nic_cfg.enc_mcdi_max_payload_length;

	max_data_size = payload_len - MC_CMD_LWRAM_WRITE_IN_LEN(0);
	EFSYS_ASSERT3U(payload_len, >, 0);
	EFSYS_ASSERT3U(max_data_size, <, payload_len);

	if (size > max_data_size) {
		rc = EILWAL;
		goto fail1;
	}

	EFSYS_KMEM_ALLOC(enp->en_esip, payload_len, payload);
	if (payload == NULL) {
		rc = ENOMEM;
		goto fail2;
	}

	(void) memset(payload, 0, payload_len);
	req.emr_cmd = MC_CMD_LWRAM_WRITE;
	req.emr_in_buf = payload;
	req.emr_in_length = MC_CMD_LWRAM_WRITE_IN_LEN(size);
	req.emr_out_buf = payload;
	req.emr_out_length = MC_CMD_LWRAM_WRITE_OUT_LEN;

	MCDI_IN_SET_DWORD(req, LWRAM_WRITE_IN_TYPE, partn);
	MCDI_IN_SET_DWORD(req, LWRAM_WRITE_IN_OFFSET, offset);
	MCDI_IN_SET_DWORD(req, LWRAM_WRITE_IN_LENGTH, size);

	memcpy(MCDI_IN2(req, uint8_t, LWRAM_WRITE_IN_WRITE_BUFFER),
	    data, size);

	efx_mcdi_exelwte(enp, &req);

	if (req.emr_rc != 0) {
		rc = req.emr_rc;
		goto fail3;
	}

	EFSYS_KMEM_FREE(enp->en_esip, payload_len, payload);

	return (0);

fail3:
	EFSYS_PROBE(fail3);
	EFSYS_KMEM_FREE(enp->en_esip, payload_len, payload);
fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	return (rc);
}


/*
 * MC_CMD_LWRAM_UPDATE_FINISH_V2 must be used to support firmware-verified
 * LWRAM updates. Older firmware will ignore the flags field in the request.
 */
	__checkReturn		efx_rc_t
efx_mcdi_lwram_update_finish(
	__in			efx_nic_t *enp,
	__in			uint32_t partn,
	__in			boolean_t reboot,
	__in			uint32_t flags,
	__out_opt		uint32_t *verify_resultp)
{
	const efx_nic_cfg_t *encp = &enp->en_nic_cfg;
	efx_mcdi_req_t req;
	EFX_MCDI_DECLARE_BUF(payload, MC_CMD_LWRAM_UPDATE_FINISH_V2_IN_LEN,
		MC_CMD_LWRAM_UPDATE_FINISH_V2_OUT_LEN);
	uint32_t verify_result = MC_CMD_LWRAM_VERIFY_RC_UNKNOWN;
	efx_rc_t rc = 0;

	req.emr_cmd = MC_CMD_LWRAM_UPDATE_FINISH;
	req.emr_in_buf = payload;
	req.emr_in_length = MC_CMD_LWRAM_UPDATE_FINISH_V2_IN_LEN;
	req.emr_out_buf = payload;
	req.emr_out_length = MC_CMD_LWRAM_UPDATE_FINISH_V2_OUT_LEN;

	MCDI_IN_SET_DWORD(req, LWRAM_UPDATE_FINISH_V2_IN_TYPE, partn);
	MCDI_IN_SET_DWORD(req, LWRAM_UPDATE_FINISH_V2_IN_REBOOT, reboot);

	if (!encp->enc_lwram_update_poll_verify_result_supported) {
		flags &= ~EFX_LWRAM_UPDATE_FLAGS_BACKGROUND;
		flags &= ~EFX_LWRAM_UPDATE_FLAGS_POLL;
	}

	MCDI_IN_POPULATE_DWORD_3(req, LWRAM_UPDATE_FINISH_V2_IN_FLAGS,
	    LWRAM_UPDATE_FINISH_V2_IN_FLAG_REPORT_VERIFY_RESULT,
	    1,
	    LWRAM_UPDATE_FINISH_V2_IN_FLAG_RUN_IN_BACKGROUND,
	    (flags & EFX_LWRAM_UPDATE_FLAGS_BACKGROUND) ? 1 : 0,
	    LWRAM_UPDATE_FINISH_V2_IN_FLAG_POLL_VERIFY_RESULT,
	    (flags & EFX_LWRAM_UPDATE_FLAGS_POLL) ? 1 : 0
	    );

	efx_mcdi_exelwte(enp, &req);

	if (req.emr_rc != 0) {
		rc = req.emr_rc;
		goto fail1;
	}

	if (req.emr_out_length_used < MC_CMD_LWRAM_UPDATE_FINISH_V2_OUT_LEN) {
		verify_result = MC_CMD_LWRAM_VERIFY_RC_UNKNOWN;
		if (encp->enc_lwram_update_verify_result_supported) {
			/* Result of update verification is missing */
			rc = EMSGSIZE;
			goto fail2;
		}
	} else {
		verify_result =
		    MCDI_OUT_DWORD(req, LWRAM_UPDATE_FINISH_V2_OUT_RESULT_CODE);
	}

	if (encp->enc_lwram_update_verify_result_supported) {
		if ((verify_result != MC_CMD_LWRAM_VERIFY_RC_SUCCESS) &&
		    (verify_result != MC_CMD_LWRAM_VERIFY_RC_PENDING)) {
			/* Update verification failed */
			rc = EILWAL;
			goto fail3;
		}
	}

	if (verify_resultp != NULL)
		*verify_resultp = verify_result;

	return (0);

fail3:
	EFSYS_PROBE(fail3);
fail2:
	EFSYS_PROBE(fail2);
fail1:
	EFSYS_PROBE1(fail1, efx_rc_t, rc);

	/* Always report verification result */
	if (verify_resultp != NULL)
		*verify_resultp = verify_result;

	return (rc);
}

#if EFSYS_OPT_DIAG

	__checkReturn		efx_rc_t
efx_mcdi_lwram_test(
	__in			efx_nic_t *enp,
	__in			uint32_t partn)
{
	efx_mcdi_req_t req;
	EFX_MCDI_DECLARE_BUF(payload, MC_CMD_LWRAM_TEST_IN_LEN,
		MC_CMD_LWRAM_TEST_OUT_LEN);
	int result;
	efx_rc_t rc;

	req.emr_cmd = MC_CMD_LWRAM_TEST;
	req.emr_in_buf = payload;
	req.emr_in_length = MC_CMD_LWRAM_TEST_IN_LEN;
	req.emr_out_buf = payload;
	req.emr_out_length = MC_CMD_LWRAM_TEST_OUT_LEN;

	MCDI_IN_SET_DWORD(req, LWRAM_TEST_IN_TYPE, partn);

	efx_mcdi_exelwte(enp, &req);

	if (req.emr_rc != 0) {
		rc = req.emr_rc;
		goto fail1;
	}

	if (req.emr_out_length_used < MC_CMD_LWRAM_TEST_OUT_LEN) {
		rc = EMSGSIZE;
		goto fail2;
	}

	result = MCDI_OUT_DWORD(req, LWRAM_TEST_OUT_RESULT);
	if (result == MC_CMD_LWRAM_TEST_FAIL) {

		EFSYS_PROBE1(lwram_test_failure, int, partn);

		rc = (EILWAL);
		goto fail3;
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

#endif	/* EFSYS_OPT_DIAG */


#endif /* EFSYS_OPT_LWRAM || EFSYS_OPT_VPD */
