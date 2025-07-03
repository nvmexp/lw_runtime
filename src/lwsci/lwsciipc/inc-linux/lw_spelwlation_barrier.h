/*
 * LWPU GPZ vulnerability mitigation definitions.
 *
 * Copyright (c) 2018-2019, LWPU CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of LWPU CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * There are two copies (listed at confluence.lwpu.com/x/DtprBg)
 * of this file for legacy reasons.
 *
 * Both files need to be kept in sync if any changes are required.
 */

#ifndef INCLUDED_LW_SPELWLATION_BARRIER_H
#define INCLUDED_LW_SPELWLATION_BARRIER_H

/**
 * @file
 * @brief <b>LWPU Spelwlation Barrier for Spectre Variant 1 Mitigation </b>
 *
 * @b Description: This file contains helper functions for mitigating
 *    the Spectre variant-1 vulnerability.
 */


/**
 * @defgroup lw_spelwlation_barrier Spelwlation Barrier for Spectre-V1 Mitigation
 * @{
 */

#define LW_SPELWLATION_BARRIER_VERSION 2

/*
 * GNU-C/MSC/clang - x86/x86_64 : __x86_64, __i386, __i386__
 * GNU-C - THUMB mode           : __GNUC__, __thumb__
 * GNU-C - ARM modes            : __GNUC__, __arm__, __aarch64__
 * armclang - THUMB mode        : __ARMCC_VERSION, __thumb__
 * armclang - ARM modes         : __ARMCC_VERSION, __arm__, __aarch64__
 * GHS - THUMB mode             : __ghs__, __THUMB__
 * GHS - ARM modes              : __ghs__, __ARM__, __ARM64__
 */

#if defined(_M_IX86) || defined(__i386__) || defined(__i386) \
    || defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
 /* All x86 */
 #define LW_SPELWLATION_BARRIER_x86

#elif defined(macintosh) || defined(__APPLE__) \
    || defined(__powerpc) || defined(__powerpc__) || defined(__powerpc64__) \
    || defined(__POWERPC__) || defined(__ppc) || defined(__ppc__) \
    || defined(__ppc64__) || defined(__PPC__) \
    || defined(__PPC64__) || defined(_ARCH_PPC) || defined(_ARCH_PPC64)
 /* All PowerPC */
 #define LW_SPELWLATION_BARRIER_PPC

#elif (defined(__GNUC__) && defined(__thumb__)) \
    || (defined(__ARMCC_VERSION) && defined(__thumb__)) \
    || (defined(__ghs__) && defined(__THUMB__))
 /* ARM-thumb mode(<=ARMv7)/T32 (ARMv8) */
 #define LW_SPELWLATION_BARRIER_ARM_COMMON
 #define LW_SPEC_BARRIER_CSDB ".inst.w 0xf3af8014\n"

#elif (defined(__GNUC__) && defined(__arm__)) \
    || (defined(__ARMCC_VERSION) && defined(__arm__)) \
    || (defined(__ghs__) && defined(__ARM__))
 /* aarch32(ARMv8) / arm(<=ARMv7) mode */
 #define LW_SPELWLATION_BARRIER_ARM_COMMON
 #define LW_SPEC_BARRIER_CSDB ".inst 0xe320f014\n"

#elif (defined(__GNUC__) && defined(__aarch64__)) \
    || (defined(__ARMCC_VERSION) && defined(__aarch64__)) \
    || (defined(__ghs__) && defined(__ARM64__))
 /* aarch64(ARMv8) mode */
 #define LW_SPELWLATION_BARRIER_ARM_COMMON
 #define LW_SPEC_BARRIER_CSDB "HINT #20\n"

#else
 #error "Unknown compiler/chip family"
#endif

/**
 * Provides a general-purpose spelwlation barrier, but not the preferred method.
 *
 * This approach provides full protection against variant-1 vulnerability.
 * For information on the recommended method, see lw_array_index_no_spelwlate(),
 * below.
 *
 * ### Semantics
 * After lw_spelwlation_barrier() is called, this API prevents subsequent memory
 * reads, when the memory is:
 * - Within the same scope as lw_spelwlation_barrier(), or
 * - Directly within a nested scope.
 *
 * This prevention continues until all
 * conditions for entering that scope have been architecturally resolved.
 *
 * For example:
 *
 * \code
 * if (untrusted_index_from_user < bound)  {
 *  ...
 *  lw_spelwlation_barrier();
 *  ...
 *  x = array1[untrusted_index_from_user];
 *  bit = x & 1;
 *  y = array2[0x100 * bit];
 * }
 * \endcode
 */

#if defined(LW_SPELWLATION_BARRIER_x86)
// Delete after all references are changed to lw_spelwlation_barrier
/** This macro changes all spelwlation_barrier() calls to lw_spelwlation_barrier().*/
#define spelwlation_barrier() lw_spelwlation_barrier()

static inline void lw_spelwlation_barrier(void)
{

#if defined(_MSC_VER) && !defined(__clang__)
    _mm_lfence();
#endif

#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__ ("lfence" : : : "memory");
#endif

}

#elif defined(LW_SPELWLATION_BARRIER_PPC)

static inline void lw_spelwlation_barrier(void)
{
    asm volatile("ori 31,31,0");
}

#elif defined(LW_SPELWLATION_BARRIER_ARM_COMMON)

/* Note: Cortex-A9 GNU-assembler seems to complain about DSB SY */
static inline void lw_spelwlation_barrier(void)
{
    asm volatile
    (
        "DSB sy\n"
        "ISB\n"
        : : : "memory"
    );
}

#endif

/**
 * Provides the preferred method for spelwlation barrier mitigation.
 *
 * This approach "de-spelwlates" an array index that
 * has already been bounds-checked.
 *
 * This approach is preferred over lw_spelwlation_barrier() because it:
 * - Is just as effective.
 * - Clearly identifies what array index is being de-spelwlated and is thus
 *   self-commenting. In contrast, the general-purpose spelwlation barrier requires
 *   an explanation of what array index is being de-spelwlated.
 * - Performs substantially better on ARM Cortex-A cores.
 *   The improvement is expected to be tens of cycles per invocation. Within
 *   tight loops, this improvement can be significant.
 *
 * ### Semantics
 * The return value will not be spelwlative, provided the following
 * conditions are satisfied:
 * - \a count is non-zero.
 * - Caller has already validated or otherwise established that \a index < \a count.
 *
 * For example, the following snippet satisfies the conditions described above.
 * As a result, the \c array1[\c untrusted_index_from_user] read is
 * always done with \c untrusted_index_from_user < count.
 *
 * \code
 * if (untrusted_index_from_user < bound)  {
 *  untrusted_index_from_user = lw_array_index_no_spelwlate(
 *                  untrusted_index_from_user, bound);
 *  ...
 *  x = array1[untrusted_index_from_user];
 *  ...
 * }
 * \endcode
 *
 * The use of lw_array_index_no_spelwlate() in the above example ensures that
 * subsequent uses of \c untrusted_index_from_user will not execute spelwlatively.
 * Those uses will wait for the bounds check to complete.
 *
 * @param[in] index non-zero Index as given by the user that is to be
 *            de-spelwlated
 * @param[in] count Number of elements in the array
 *
 * @return index if 0<=index<count. Otherwise, 0.
 *
 * @note
 *  - It is assumed that the array in question has valid indices belonging to
 *    the set [0, count-1]. If the starting valid index is > 0,
 *    the caller must reduce the \a index or \a count arguments, such that
 *    \a index starts from 0. The caller must correspondingly adjust the return
 *    value.
 */

#if defined(LW_SPELWLATION_BARRIER_x86)

// TODO Add implementation for _MSC_VER and PPC
#if defined(__GNUC__) || defined(__clang__)

static inline unsigned long lw_array_index_no_spelwlate(unsigned long index,
                        unsigned long count)
{
    unsigned long mask;

    __asm__ __volatile__
    (
        "CMP     %2, %1          \n"
        "SBB     %0, %0          \n"
        : "=r"(mask) : "r"(index), "r"(count) : "cc"
    );

    return (index & mask);
}

static inline unsigned int lw_array_index_no_spelwlate_u32(unsigned int index,
                        unsigned int count)
{
    unsigned int mask;

    __asm__ __volatile__
    (
        "CMP     %2, %1          \n"
        "SBB     %0, %0          \n"
        : "=r"(mask) : "r"(index), "r"(count) : "cc"
    );

    return (index & mask);
}

#endif


#elif defined(LW_SPELWLATION_BARRIER_ARM_COMMON)
static inline unsigned long lw_array_index_no_spelwlate(unsigned long index,
                        unsigned long count)
{
    unsigned long mask;

    asm volatile
    (
        "CMP %[ind], %[cnt] \n"
        "SBC %[res], %[cnt], %[cnt] \n"
        LW_SPEC_BARRIER_CSDB
        : [res] "=r" (mask) : [ind] "r" (index), [cnt] "r" (count): "cc"
    );

    return (index & mask);
}

static inline unsigned int lw_array_index_no_spelwlate_u32(unsigned int index,
                        unsigned int count)
{
    unsigned int mask;

    asm volatile
    (
        "CMP %w[ind], %w[cnt] \n"
        "SBC %w[res], %w[cnt], %w[cnt] \n"
        LW_SPEC_BARRIER_CSDB
        : [res] "=r" (mask) : [ind] "r" (index), [cnt] "r" (count): "cc"
    );

    return (index & mask);
}

#endif

/** @} */

#endif // INCLUDED_LW_SPELWLATION_BARRIER_H
