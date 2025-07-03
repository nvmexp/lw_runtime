/******************************************************************************
 * Copyright (c) 2011-2019, LWPU CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are not permitted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL LWPU CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

// {$lw-internal-release file}

/**
 * \file
 * \brief Jetfire directives
 */

/******************************************************************************
 * MACROS
 ******************************************************************************/

/// LWOPT recipe enumerants
#if JETFIRE_ENABLED
    #if defined(__LWDACC_RTC__)
        #define JETFIRE_MAC_LOOP_PRAGMA _Pragma("lwopt 0")
    #else
        #define JETFIRE_MAC_LOOP_PRAGMA #pragma lwopt 0
    #endif
    #define JETFIRE_MAC_LOOP_HEADER asm volatile (".pragma \"lwopt 0\";\n");
#else
    #define JETFIRE_MAC_LOOP_PRAGMA
    #define JETFIRE_MAC_LOOP_HEADER
#endif


namespace jetfire
{

/******************************************************************************
 * WARP SCHEDULING
 ******************************************************************************/

/**
 * Advise the warp scheduler to context switch to a different warp before
 * exelwting the next instruction
 */
__device__ __inline__ void warp_switch()
{
#if JETFIRE_ENABLED
    asm volatile (".pragma \"next knob WarpOpexPrev=1\";\n" : : : "memory");
#endif
}


/******************************************************************************
 * IFENCE
 ******************************************************************************/

/**
 * Interference fence (IFENCE).
 *
 * The storage-aliasing decisions (e.g., register allocation) made by the
 * compiler will strongly influence a program's instruction-level parallelism,
 * memory-level parallelism, effective latency-hiding, and SM oclwpancy. When
 * two variables are allocated different storage, their live-ranges can overlap
 * in the program's final instruction schedule, which allows the instructions
 * that produce them to operate conlwrrently. Interference fences provide a
 * means for the program(mer) to indicate whether sets of program variables may
 * or may not alias the same storage, i.e., to "dial up" or "dial down" the
 * register pressure at a given program location.
 *
 * In particular, two independent variables CANNOT be assigned the same
 * physical register when their textual live ranges both span the same
 * interference fence.  This explicitly precludes register anti-dependences
 * between these variables, which enables instruction-level parallelism
 * at the location of the interference fence.
 *
 * Colwersely, an interference fence that textually isolates the live-ranges of
 * two different variables ensures they MAY be allocated the same physical
 * register.  This can reduce register pressure at the location of the
 * interference fence (at the expense of increased instruction serialization
 * and latency).
 *
 * Intuitively, an interference fence has the same effect on register
 * allocation as a code-motion barrier, but with much less restriction on the
 * compiler's ability to optimize the program's instruction schedule.  Although
 * code motion barriers are necessary for inter-thread signaling and
 * synchronization, they can be excessively heavy-handed for the purpose of
 * controlling register allocation. A code-motion barrier affects
 * storage-aliasing indirectly: it establishes an explicit instruction ordering
 * that determines which variable live-ranges MUST overlap, viz. those
 * live-ranges that textually span the same code-motion barrier. Consequently
 * those variables CANNOT be placed in the same registers. However, a code
 * motion barrier has no effect on the compiler's decision whether to alias two
 * variables unrelated by the barrier, i.e. two variables whose live-ranges do
 * not both span the same barrier. Yet the code-motion barrier still has the
 * effect of precluding the compiler from moving the corresponding def|use
 * instructions across the barrier to improve the program's instruction mix.
 *
 * In contrast, the interference fence constrains storage-aliasing directly by
 * augmenting the program's "interference graph".  Every imperative program
 * implies a logical interference graph: each vertex in the graph represents a
 * unique variable in the program, and edges connect pairs of vertices which
 * cannot alias the same storage (i.e., they can be live at the same time).
 * When two variables span the same interference fence, an artificial
 * "interference" edge is placed between those two nodes in the graph, and the
 * compiler must allocate separate storage for them.  Colwersely, when two
 * variables are isolated by an interference fence, a "preference" edge is
 * inserted between the two nodes in the interference graph, which prohibits
 * the compiler from making program transformations that would imply an
 * interference edge between the two nodes.  Instead of being explicitly
 * (over)prescribed, the final instruction ordering is only constrained via
 * register anti-dependences.  Consequently, interference fences afford the
 * optimizing compiler much greater freedom to reorganize the instruction mix
 * for increased instruction-level parallelism between functional units and
 * shorter overall issue schedules.
 *
 * Consider the following two functionally-equivalent program expressions, and
 * consider how the interference fence(s) serve to preserve programmer intent
 * regarding register pressure and ILP:
 *
 *     // Intent: maximum memory-level parallelism (maximum register pressure)
 *     __device__ Foo(([X],[Y],&[Z])
 *     {
 *         a = Load();
 *         b = Load();
 *         c = Load();
 *
 *         #pragma ifence   // a,b,c,[X],[Y],[Z] must all be live here
 *
 *         Store(a);
 *         Store(b);
 *         Store(c);
 *
 *         [Z] = MacTile([X],[Y],[Z]);    // A sea of unrelated FFMAs
 *     }
 *
 *     // Intent: No memory level parallelism (minimal register pressure)
 *     __device__ Foo(([X],[Y],&[Z])
 *     {
 *         a = Load();
 *         Store(a);
 *
 *         #pragma ifence      // only [X],[Y],[Z] must be live here
 *
 *         b = Load();
 *         Store(b);
 *
 *         #pragma ifence      // only [X],[Y],[Z] must be live here
 *
 *         c = Load();
 *         Store(c);
 *
 *         [Z] = MacTile([X],[Y],[Z]);  // A sea of unrelated FFMAs
 *     }
 *
 */
__device__ __inline__ void ifence(bool enabled = true)
{
#if JETFIRE_ENABLED
    // Usually the ifence can improve performance, while when register pressure is high, it makes 
    // compiler use local memory operations which are inefficient, so be careful with it
    if( enabled ) {
        asm volatile (".pragma \"next knob FenceInterference\";\n" : : : "memory");
    }
#endif
}




} // namespace jetfire

