/*
** C type colwersions.
** Copyright (C) 2005-2017 Mike Pall. See Copyright Notice in luajit.h
*/

#ifndef _LJ_CCOLW_H
#define _LJ_CCOLW_H

#include "lj_obj.h"
#include "lj_ctype.h"

#if LJ_HASFFI

/* Compressed C type index. ORDER CCX. */
enum {
  CCX_B,	/* Bool. */
  CCX_I,	/* Integer. */
  CCX_F,	/* Floating-point number. */
  CCX_C,	/* Complex. */
  CCX_V,	/* Vector. */
  CCX_P,	/* Pointer. */
  CCX_A,	/* Refarray. */
  CCX_S		/* Struct/union. */
};

/* Colwert C type info to compressed C type index. ORDER CT. ORDER CCX. */
static LJ_AINLINE uint32_t ccolw_idx(CTInfo info)
{
  uint32_t idx = ((info >> 26) & 15u);  /* Dispatch bits. */
  lua_assert(ctype_type(info) <= CT_MAYCOLWERT);
#if LJ_64
  idx = ((uint32_t)(U64x(f436fff5,fff7f021) >> 4*idx) & 15u);
#else
  idx = (((idx < 8 ? 0xfff7f021u : 0xf436fff5) >> 4*(idx & 7u)) & 15u);
#endif
  lua_assert(idx < 8);
  return idx;
}

#define ccolw_idx2(dinfo, sinfo) \
  ((ccolw_idx((dinfo)) << 3) + ccolw_idx((sinfo)))

#define CCX(dst, src)		((CCX_##dst << 3) + CCX_##src)

/* Colwersion flags. */
#define CCF_CAST	0x00000001u
#define CCF_FROMTV	0x00000002u
#define CCF_SAME	0x00000004u
#define CCF_IGNQUAL	0x00000008u

#define CCF_ARG_SHIFT	8
#define CCF_ARG(n)	((n) << CCF_ARG_SHIFT)
#define CCF_GETARG(f)	((f) >> CCF_ARG_SHIFT)

LJ_FUNC int lj_ccolw_compatptr(CTState *cts, CType *d, CType *s, CTInfo flags);
LJ_FUNC void lj_ccolw_ct_ct(CTState *cts, CType *d, CType *s,
			    uint8_t *dp, uint8_t *sp, CTInfo flags);
LJ_FUNC int lj_ccolw_tv_ct(CTState *cts, CType *s, CTypeID sid,
			   TValue *o, uint8_t *sp);
LJ_FUNC int lj_ccolw_tv_bf(CTState *cts, CType *s, TValue *o, uint8_t *sp);
LJ_FUNC void lj_ccolw_ct_tv(CTState *cts, CType *d,
			    uint8_t *dp, TValue *o, CTInfo flags);
LJ_FUNC void lj_ccolw_bf_tv(CTState *cts, CType *d, uint8_t *dp, TValue *o);
LJ_FUNC int lj_ccolw_multi_init(CTState *cts, CType *d, TValue *o);
LJ_FUNC void lj_ccolw_ct_init(CTState *cts, CType *d, CTSize sz,
			      uint8_t *dp, TValue *o, MSize len);

#endif

#endif
