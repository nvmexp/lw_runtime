/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2018 Intel Corporation
 */

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <unistd.h>
#include <inttypes.h>

#include <rte_common.h>
#include <rte_log.h>
#include <rte_debug.h>
#include <rte_memory.h>
#include <rte_eal.h>
#include <rte_byteorder.h>
#include <rte_errno.h>

#include "bpf_impl.h"

static struct rte_bpf *
bpf_load(const struct rte_bpf_prm *prm)
{
	uint8_t *buf;
	struct rte_bpf *bpf;
	size_t sz, bsz, insz, xsz;

	xsz =  prm->nb_xsym * sizeof(prm->xsym[0]);
	insz = prm->nb_ins * sizeof(prm->ins[0]);
	bsz = sizeof(bpf[0]);
	sz = insz + xsz + bsz;

	buf = mmap(NULL, sz, PROT_READ | PROT_WRITE,
		MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (buf == MAP_FAILED)
		return NULL;

	bpf = (void *)buf;
	bpf->sz = sz;

	memcpy(&bpf->prm, prm, sizeof(bpf->prm));

	memcpy(buf + bsz, prm->xsym, xsz);
	memcpy(buf + bsz + xsz, prm->ins, insz);

	bpf->prm.xsym = (void *)(buf + bsz);
	bpf->prm.ins = (void *)(buf + bsz + xsz);

	return bpf;
}

/*
 * Check that user provided external symbol.
 */
static int
bpf_check_xsym(const struct rte_bpf_xsym *xsym)
{
	uint32_t i;

	if (xsym->name == NULL)
		return -EILWAL;

	if (xsym->type == RTE_BPF_XTYPE_VAR) {
		if (xsym->var.desc.type == RTE_BPF_ARG_UNDEF)
			return -EILWAL;
	} else if (xsym->type == RTE_BPF_XTYPE_FUNC) {

		if (xsym->func.nb_args > EBPF_FUNC_MAX_ARGS)
			return -EILWAL;

		/* check function arguments */
		for (i = 0; i != xsym->func.nb_args; i++) {
			if (xsym->func.args[i].type == RTE_BPF_ARG_UNDEF)
				return -EILWAL;
		}

		/* check return value info */
		if (xsym->func.ret.type != RTE_BPF_ARG_UNDEF &&
				xsym->func.ret.size == 0)
			return -EILWAL;
	} else
		return -EILWAL;

	return 0;
}

struct rte_bpf *
rte_bpf_load(const struct rte_bpf_prm *prm)
{
	struct rte_bpf *bpf;
	int32_t rc;
	uint32_t i;

	if (prm == NULL || prm->ins == NULL ||
			(prm->nb_xsym != 0 && prm->xsym == NULL)) {
		rte_errno = EILWAL;
		return NULL;
	}

	rc = 0;
	for (i = 0; i != prm->nb_xsym && rc == 0; i++)
		rc = bpf_check_xsym(prm->xsym + i);

	if (rc != 0) {
		rte_errno = -rc;
		RTE_BPF_LOG(ERR, "%s: %d-th xsym is invalid\n", __func__, i);
		return NULL;
	}

	bpf = bpf_load(prm);
	if (bpf == NULL) {
		rte_errno = ENOMEM;
		return NULL;
	}

	rc = bpf_validate(bpf);
	if (rc == 0) {
		bpf_jit(bpf);
		if (mprotect(bpf, bpf->sz, PROT_READ) != 0)
			rc = -ENOMEM;
	}

	if (rc != 0) {
		rte_bpf_destroy(bpf);
		rte_errno = -rc;
		return NULL;
	}

	return bpf;
}

#ifndef RTE_LIBRTE_BPF_ELF
struct rte_bpf *
rte_bpf_elf_load(const struct rte_bpf_prm *prm, const char *fname,
	const char *sname)
{
	if (prm == NULL || fname == NULL || sname == NULL) {
		rte_errno = EILWAL;
		return NULL;
	}

	RTE_BPF_LOG(ERR, "%s() is not supported with current config\n"
		"rebuild with libelf installed\n",
		__func__);
	rte_errno = ENOTSUP;
	return NULL;
}
#endif
