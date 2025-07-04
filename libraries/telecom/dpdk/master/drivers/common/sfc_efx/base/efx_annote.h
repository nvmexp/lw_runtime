/* SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright(c) 2019-2020 Xilinx, Inc.
 * Copyright(c) 2018-2019 Solarflare Communications Inc.
 */

#ifndef	_SYS_EFX_ANNOTE_H
#define	_SYS_EFX_ANNOTE_H

#if defined(_WIN32) || defined(_WIN64)
#define	EFX_HAVE_WINDOWS_ANNOTATIONS 1
#else
#define	EFX_HAVE_WINDOWS_ANNOTATIONS 0
#endif	/* defined(_WIN32) || defined(_WIN64) */

#if defined(__sun)
#define	EFX_HAVE_SOLARIS_ANNOTATIONS 1
#else
#define	EFX_HAVE_SOLARIS_ANNOTATIONS 0
#endif	/* defined(__sun) */

#if !EFX_HAVE_WINDOWS_ANNOTATIONS

/* Ignore Windows SAL annotations on other platforms */
#define	__in
#define	__in_opt
#define	__in_ecount(_n)
#define	__in_ecount_opt(_n)
#define	__in_bcount(_n)
#define	__in_bcount_opt(_n)

#define	__out
#define	__out_opt
#define	__out_ecount(_n)
#define	__out_ecount_opt(_n)
#define	__out_ecount_part(_n, _l)
#define	__out_bcount(_n)
#define	__out_bcount_opt(_n)
#define	__out_bcount_part(_n, _l)
#define	__out_bcount_part_opt(_n, _l)

#define	__deref_out
#define	__deref_inout

#define	__inout
#define	__inout_opt
#define	__inout_ecount(_n)
#define	__inout_ecount_opt(_n)
#define	__inout_bcount(_n)
#define	__inout_bcount_opt(_n)
#define	__inout_bcount_full_opt(_n)

#define	__deref_out_bcount_opt(n)

#define	__checkReturn
#define	__success(_x)

#define	__drv_when(_p, _c)

#endif	/* !EFX_HAVE_WINDOWS_ANNOTATIONS */

#if !EFX_HAVE_SOLARIS_ANNOTATIONS

#if EFX_HAVE_WINDOWS_ANNOTATIONS

/*
 * Support some SunOS/Solaris style _NOTE() annotations
 *
 * At present with the facilities provided in the WDL and the SAL we can only
 * easily act upon _NOTE(ARGUNUSED(arglist)) annotations.
 *
 * Intermediate macros to expand individual _NOTE annotation types into
 * something the WDK or SAL can understand.  They shouldn't be used directly,
 * for example EFX_NOTE_ARGUNUSED() is only used as an intermediate step on the
 * transformation of _NOTE(ARGUNSED(arg1, arg2)) into
 * UNREFERENCED_PARAMETER((arg1, arg2));
 */
#define	EFX_NOTE_ALIGNMENT(_fname, _n)
#define	EFX_NOTE_ARGUNUSED(...)		UNREFERENCED_PARAMETER((__VA_ARGS__));
#define	EFX_NOTE_CONSTANTCONDITION
#define	EFX_NOTE_CONSTCOND
#define	EFX_NOTE_EMPTY
#define	EFX_NOTE_FALLTHROUGH
#define	EFX_NOTE_FALLTHRU
#define	EFX_NOTE_LINTED(_msg)
#define	EFX_NOTE_NOTREACHED
#define	EFX_NOTE_PRINTFLIKE(_n)
#define	EFX_NOTE_SCANFLIKE(_n)
#define	EFX_NOTE_VARARGS(_n)

#define	_NOTE(_annotation)		EFX_NOTE_ ## _annotation

#else

/* Ignore Solaris annotations on other platforms */

#define	_NOTE(_annotation)

#endif	/* EFX_HAVE_WINDOWS_ANNOTATIONS */

#endif	/* !EFX_HAVE_SOLARIS_ANNOTATIONS */

#endif	/* _SYS_EFX_ANNOTE_H */
