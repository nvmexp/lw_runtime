/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited
 */

#ifndef INCLUDED_LWOS_STATIC_ANALYSIS_H
#define INCLUDED_LWOS_STATIC_ANALYSIS_H

/**
 * @file
 *
 * Macros/functions/etc for static analysis of code.
 */

#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

/**
 * These macros are used for whitelisting coverity violations. The macros are
 * only enabled when a coverity scan is being run.
 */
#ifdef LW_IS_COVERITY
/**
 * LWOS_MISRA - Define a MISRA rule for LWOS_COV_WHITELIST.
 *
 * @param type - This should be Rule or Directive depending on if you're dealing
 *               with a MISRA rule or directive.
 * @param num  - This is the MISRA rule/directive number. Replace hyphens and
 *               periods in the rule/directive number with underscores. Example:
 *               14.2 should be 14_2.
 *
 * This is a colwenience macro for defining a MISRA rule for the
 * LWOS_COV_WHITELIST macro.
 *
 * Example 1: For defining MISRA rule 14.2, use LWOS_MISRA(Rule, 14_2).
 * Example 2: For defining MISRA directive 4.7, use LWOS_MISRA(Directive, 4_7).
 */
#define LWOS_MISRA(type, num) MISRA_C_2012_##type##_##num

/**
 * LWOS_CERT - Define a CERT C rule for LWOS_COV_WHITELIST.
 *
 * @param num - This is the CERT C rule number. Replace hyphens and periods in
 *              the rule number with underscores. Example: INT30-C should be
 *              INT30_C.
 *
 * This is a colwenience macro for defining a CERT C rule for the
 * LWOS_COV_WHITELIST macro.
 *
 * Example: For defining CERT C rule INT30-C, use LWOS_CERT(INT30_C).
 */
#define LWOS_CERT(num) CERT_##num

/**
 * Helper macro for stringifying the _Pragma() string
 */
#define LWOS_COV_STRING(x) #x

/**
 * LWOS_COV_WHITELIST - Whitelist a coverity violation on the next line.
 *
 * @param type        - This is the whitelisting category. Valid values are
 *                      deviate or false_positive.
 *                      deviate is for an approved rule deviation.
 *                      false_positive is normally used for a bug in coverity
 *                      which causes a false violation to appear in the scan.
 * @param checker     - This is the MISRA or CERT C rule causing the violation.
 *                      Use the LWOS_MISRA() or LWOS_CERT() macro to define
 *                      this field.
 * @param comment_str - This is the comment that you want associated with this
 *                      whitelisting. This should normally be a bug number
 *                      (ex: coverity bug) or JIRA task ID (ex: RFD). Unlike the
 *                      other arguments, this argument must be a quoted string.
 *
 * Use this macro to whitelist a coverity violation in the next line of code.
 */
#define LWOS_COV_WHITELIST(type, checker, comment_str) \
        _Pragma(LWOS_COV_STRING(coverity compliance type checker comment_str))
#else
/**
 * no-op macros for normal compilation - whitelisting is disabled when a
 * coverity scan is NOT being run
 */
#define LWOS_MISRA(type, num)
#define LWOS_CERT(num)
#define LWOS_COV_WHITELIST(type, checker, comment_str)
#endif

/**
 * Exit codes for function that call _exit().
 * Only LSBits are visible to parent process
 */
#define ERR_INT30_C_ADDU32  0x377
#define ERR_INT30_C_ADDU64  0x376
#define ERR_INT30_C_SUBU32  0x375
#define ERR_INT30_C_SUBU64  0x374
#define ERR_INT30_C_MULTU32 0x373
#define ERR_INT30_C_MULTU64 0x372
#define ERR_INT32_C_ADDS32  0x371
#define ERR_INT32_C_ADDS64  0x370
#define ERR_INT32_C_SUBS32  0x369
#define ERR_INT32_C_SUBS64  0x368
#define ERR_INT31_C_CAST_U32TOS32  0x360
#define ERR_INT31_C_CAST_U32TOU16  0x359
#define ERR_INT31_C_CAST_U32TOU8   0x358
#define ERR_INT31_C_CAST_U64TOU32  0x357
#define ERR_INT31_C_CAST_U64TOS32  0x356
#define ERR_INT31_C_CAST_S32TOU32  0x355
#define ERR_INT31_C_CAST_S64TOU32  0x354
#define ERR_INT31_C_CAST_S32TOU64  0x353
#define ERR_INT31_C_CAST_S32TOU8   0x352
#define ERR_INT31_C_CAST_S64TOU64  0x351

/**
 * Define macro for virtual address bit mask
 */
#define VA_ADDR_MASK      0xFFFFFFFFFFFFUL
#define OFFSET_MASK       0xFFFFUL

typedef uint32_t LwBoolVar;
/**
 * Define macro to replace boolean True
 */
#ifdef LwBoolTrue
#undef LwBoolTrue
#endif
#define LwBoolTrue 0xBABAFACEU

/**
 * Define macro to replace boolean False
 */
#ifdef LwBoolFalse
#undef LwBoolFalse
#endif
#define LwBoolFalse ~LwBoolTrue
/*
 * Utility function to check for CERT-C violation.
 */

/** CERT INT30-C:
 *  Precondition test to ensure that unsigned integer add operation do not wrap.
 */
static inline bool AddU32(uint32_t op1, uint32_t op2, uint32_t *result)
{
    bool e = false;
    if (((UINT32_MAX - op1) < op2) == false) {
        *result = op1 + op2;
        e = true;
    }

    return e;
}

/** CERT INT30-C:
 *  Precondition test to ensure that unsigned integer add operation do not wrap.
 */
static inline bool AddU64(uint64_t op1, uint64_t op2, uint64_t *result)
{
    bool e = false;

    if (UINT64_MAX > op1){
        if (((UINT64_MAX - op1) < op2) == false) {
            *result = op1 + op2;
            e = true;
        }
    }
    return e;
}

/** CERT INT30-C:
 *  Precondition test to ensure that unsigned integer multiplication operation
 *  do not wrap.
 */
static inline bool MultU32(uint32_t op1, uint32_t op2, uint32_t *result)
{
    bool e = true;

    if ((op1 == 0U) || (op2 == 0U)) {
       *result = 0U;
    } else if ((op1 > (UINT32_MAX / op2)) == false)  {
        *result = op1 * op2;
    } else {
        e = false;
    }

    return e;
}

/** CERT INT30-C:
 *  Precondition test to ensure that unsigned integer multiplication operation
 *  do not wrap.
 */
static inline bool MultU64(uint64_t op1, uint64_t op2, uint64_t *result)
{
    bool e = true;

    if ((op1 == 0U) || (op2 == 0U)) {
       *result = 0U;
    } else if ((op1 > (UINT64_MAX / op2)) == false)  {
        *result = op1 * op2;
    } else {
        e = false;
    }

    return e;
}

/** CERT INT30-C:
 *  Precondition test to ensure that unsigned integer subtraction operation
 *  do not wrap.
 */
static inline bool SubU32(uint32_t op1, uint32_t op2, uint32_t *result)
{
    bool e = false;

    if ((op1 < op2) == false) {
        *result = op1 - op2;
        e = true;
    }

    return e;
}

/** CERT INT30-C:
 *  Precondition test to ensure that unsigned integer subtraction operation
 *  do not wrap.
 */
static inline bool SubU64(uint64_t op1, uint64_t op2, uint64_t *result)
{
    bool e = false;

    if ((op1 < op2) == false) {
        *result = op1 - op2;
        e = true;
    }

    return e;
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that unsigned integer add operation do not wrap.
 * Function calls _exit() on overflow.
 * Only use when overflow cannot happen by design.
 */
static inline
void AddU32WithExit(uint32_t op1, uint32_t op2, uint32_t *result)
{
    if (((UINT32_MAX - op1) < op2) == false) {
        *result = op1 + op2;
    } else {
        _exit(ERR_INT30_C_ADDU32);
    }
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that unsigned integer add operation do not wrap.
 * Function calls _exit() on overflow.
 * Only use when overflow cannot happen by design.
 */
static inline
void AddU64WithExit(uint64_t op1, uint64_t op2, uint64_t *result)
{
    if (((UINT64_MAX - op1) < op2) == false) {
        *result = op1 + op2;
    } else {
        _exit(ERR_INT30_C_ADDU64);
    }
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that unsigned integer multiplication operation
 * do not wrap.
 * Function calls _exit() on overflow.
 * Only use when overflow cannot happen by design.
 */
static inline
void MultU32WithExit(uint32_t op1, uint32_t op2, uint32_t *result)
{
    if ((op1 == 0U) || (op2 == 0U)) {
       *result = 0U;
    } else if ((op1 > (UINT32_MAX / op2)) == true)  {
        _exit(ERR_INT30_C_MULTU32);
    } else {
        *result = op1 * op2;
    }
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that unsigned integer multiplication operation
 * do not wrap.
 * Function calls _exit() on overflow.
 * Only use when overflow cannot happen by design.
 */
static inline
void MultU64WithExit(uint64_t op1, uint64_t op2, uint64_t *result)
{
    if ((op1 == 0UL) || (op2 == 0UL)) {
        *result = 0UL;
    } else if ((op1 > (UINT64_MAX / op2)) == true) {
        _exit(ERR_INT30_C_MULTU64);
    } else {
        *result = op1 * op2;
    }
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that unsigned integer subtraction operation
 * do not wrap.
 * Function calls _exit() on overflow.
 * Only use when underflow cannot happen by design.
 */
static inline
void SubU32WithExit(uint32_t op1, uint32_t op2, uint32_t *result)
{
    if ((op1 < op2) == false) {
        *result = op1 - op2;
    } else {
        _exit(ERR_INT30_C_SUBU32);
    }
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that unsigned integer subtraction operation
 * do not wrap.
 * Function calls _exit() on overflow.
 * Only use when underflow cannot happen by design.
 */
static inline
void SubU64WithExit(uint64_t op1, uint64_t op2, uint64_t *result)
{
    if ((op1 < op2) == false) {
        *result = op1 - op2;
    } else {
        _exit(ERR_INT30_C_SUBU64);
    }
}


/**
 * CERT INT32-C:
 * Precondition test to ensure that signed integer add operation do not overflow.
 * Function calls _exit() on overflow.
 * Only use when overflow cannot happen by design.
 */
static inline
void AddS32WithExit(int32_t op1, int32_t op2, int32_t *result)
{
    if (((op2 > 0) && (op1 > (INT32_MAX - op2))) ||
    ((op2 < 0) && (op1 < (INT32_MIN - op2)))) {
        _exit(ERR_INT32_C_ADDS32);
    } else {
        *result = op1 + op2;
    }
}

static inline
void AddS64WithExit(int64_t op1, int64_t op2, int64_t *result)
{
    if (((op2 > 0) && (op1 > (INT64_MAX - op2))) ||
    ((op2 < 0) && (op1 < (INT64_MIN - op2)))) {
        _exit(ERR_INT32_C_ADDS64);
    } else {
        *result = op1 + op2;
    }
}


/** Casting functions to fix CERT INT31-C:
 *  Ensure integer colwersions do not result in lost or misinterpreted data.
 */
static inline
uint32_t CastS32toU32WithExit(int32_t op)
{
    if (op < 0) {
        _exit(ERR_INT31_C_CAST_S32TOU32);
    } else {
        return (uint32_t)op;
    }
}

static inline
int32_t CastU32toS32WithExit(uint32_t op)
{
    if (op > (uint32_t)INT32_MAX) {
        _exit(ERR_INT31_C_CAST_U32TOS32);
    } else {
        return (int32_t)op;
    }
}

static inline
uint16_t CastU32toU16WithExit(uint32_t op)
{
    if (op > (uint32_t)UINT16_MAX) {
        _exit(ERR_INT31_C_CAST_U32TOU16);
    } else {
        return (uint16_t)op;
    }
}

static inline
uint8_t CastU32toU8WithExit(uint32_t op)
{
    if (op > (uint32_t)UINT8_MAX) {
        _exit(ERR_INT31_C_CAST_U32TOU8);
    } else {
        return (uint8_t)op;
    }
}

static inline
uint32_t CastU64toU32WithExit(uint64_t op)
{
    if (op > UINT32_MAX) {
        _exit(ERR_INT31_C_CAST_U64TOU32);
    } else {
        return (uint32_t)op;
    }
}

static inline
int32_t CastU64toS32WithExit(uint64_t op)
{
    if (op > INT32_MAX) {
        _exit(ERR_INT31_C_CAST_U64TOS32);
    } else {
        return (int32_t)op;
    }
}

static inline
uint32_t CastS64toU32WithExit(int64_t op)
{
    if ((op > (int64_t)UINT32_MAX) || (op < 0)) {
        _exit(ERR_INT31_C_CAST_S64TOU32);
    } else {
        return (uint32_t)op;
    }
}

static inline
uint64_t CastS32toU64WithExit(int32_t op)
{
    if (op < 0) {
        _exit(ERR_INT31_C_CAST_S32TOU64);
    } else {
        return (uint64_t)op;
    }
}

static inline
uint8_t CastS32toU8WithExit(int32_t op)
{
    if ((op > (int32_t)UINT8_MAX) || (op < 0)) {
        _exit(ERR_INT31_C_CAST_S32TOU8);
    } else {
        return (uint8_t)op;
    }
}

static inline
uint64_t CastS64toU64WithExit(int64_t op)
{
    if (op < 0) {
        _exit(ERR_INT31_C_CAST_S64TOU64);
    } else {
        return (uint64_t)op;
    }
}

/** CERT INT31-C:
 * Ensure that integer colwersions do not result in lost or misinterpreted data.
 */
static inline
bool LwColwertUint64toUint32(uint64_t op, uint32_t *result)
{
    bool e = false;

    if ((op > UINT32_MAX) == false) {
            *result = (uint32_t)op;
            e = true;
    }

    return e;
}


/** CERT INT31-C:
 * Ensure that integer colwersions do not result in lost or misinterpreted data.
 */
static inline
bool LwColwertUint32toUint8(uint32_t op, uint8_t *result)
{
    bool e = false;

    if ((op > (uint32_t )UINT8_MAX) == false) {
            *result = (uint8_t)op;
            e = true;
    }

    return e;
}

#endif // INCLUDED_LWOS_STATIC_ANALYSIS_H
