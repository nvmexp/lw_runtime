/***************************************************************************************************
 * Copyright (c) 2011-2019, LWPU CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR 
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LWPU CORPORATION BE LIABLE 
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <xmma/xmma.h>
#include <xmma/utils.h>

// CP ASYNC FEATURES ///////////////////////////////////////////////////////////////////////////////

#ifndef XMMA_INTERNAL_LWVM_ENABLED
#define XMMA_INTERNAL_LWVM_ENABLED 0
#endif

#if ! defined(LWDA_CP_ASYNC_SUPPORTED)
#define LWDA_CP_ASYNC_SUPPORTED (__LWDACC_VER_MAJOR__ >= 11) || ((__LWDACC_VER_MAJOR__ == 10) &&\
    (__LWDACC_VER_MINOR__ >= 2) && defined(XMMA_INTERNAL_LWVM_ENABLED))
#endif

#if ! defined(LWDA_CP_ASYNC_ENABLED)
#define LWDA_CP_ASYNC_ENABLED (LWDA_CP_ASYNC_SUPPORTED)
#endif

#if LWDA_CP_ASYNC_ENABLED && defined(__LWDA_ARCH__) && (__LWDA_ARCH__ >= 800)
#define LWDA_CP_ASYNC_ACTIVATED 1
#endif

#if ! defined(LWDA_CP_ASYNC_MBARRIER_ARRIVE_SUPPORTED)
#define LWDA_CP_ASYNC_MBARRIER_ARRIVE_SUPPORTED (LWDA_CP_ASYNC_SUPPORTED) && (__LWDACC_VER_MAJOR__ >= 11)
#endif

#if ! defined(LWDA_CP_ASYNC_MBARRIER_ARRIVE_ENABLED)
#define LWDA_CP_ASYNC_MBARRIER_ARRIVE_ENABLED (LWDA_CP_ASYNC_MBARRIER_ARRIVE_SUPPORTED)
#endif

#if LWDA_CP_ASYNC_MBARRIER_ARRIVE_ENABLED && defined(__LWDA_ARCH__) && (__LWDA_ARCH__ >= 800)
#define LWDA_CP_ASYNC_MBARRIER_ARRIVE_ACTIVATED 1
#endif

#ifdef JETFIRE_ENABLED
#define DisableWar_SW254906_MACRO  asm volatile(" .pragma \"global knob DisableWar_SW2549067\";\n");
#else
#define DisableWar_SW254906_MACRO
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////

//it is exelwted per thread, i.e., each thread can call and init a barrier.
//need a bar.sync after using it.
inline __device__ void bar_create(void * bar_ptr, int init_count) {

    unsigned smem_ptr = get_smem_pointer(bar_ptr);

    asm volatile("{\n\t"
#if LWDA_CP_ASYNC_MBARRIER_ARRIVE_ACTIVATED
            "mbarrier.init.shared.b64 [%1], %0; \n\t"
#else
            ".reg .s32                negCnt, count, expectedCount;\n\t"
            ".reg .s64                comboCnt; \n\t"
            "neg.s32                  negCnt, %0;\n\t "
            "and.b32                  count, negCnt, 0x7fffffff; \n\t"
            "and.b32                  expectedCount, negCnt, 0x3fffffff; \n\t"
            "mov.b64                  comboCnt, {expectedCount, count}; \n\t"
            "st.shared.s64            [%1], comboCnt; \n\t"
#endif
            "}"
            : :"r"(init_count),"r"(smem_ptr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Arrive_wait {
    public:
        inline __device__ Arrive_wait() {
            bar_base_ = NULL;
        }

        inline __device__ Arrive_wait(uint64_t * bar_base, int id =0) {
            bar_base_ = bar_base;
            id_ = id;
        }

        inline __device__ bool bar_peek(int id,unsigned int bar_phase) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 900
            uint64_t * bar_ptr = reinterpret_cast<uint64_t *>(bar_base_ + id);
            unsigned smem_ptr = get_smem_pointer(bar_ptr);
	    uint32_t result32;
            asm volatile("{\n\t"
                         ".reg .pred       P1; \n\t"
			 "mbarrier.try_wait.parity.nosleep.shared.b64 P1, [%1], %2; \n\t"
			 "selp.b32 %0, 1, 0, P1; \n\t"
			 "}"
			 :"=r" (result32):"r"(smem_ptr),"r"(bar_phase));
            return result32 != 0;
#else
            uint64_t * bar_ptr = reinterpret_cast<uint64_t *>(bar_base_ + id);
            unsigned int output_phase = (bar_ptr[0] >> 63) & 1;

            return output_phase != bar_phase;
#endif
        }

        inline __device__ void bar_wait(int id, unsigned int bar_phase) {
            
#ifdef JETFIRE_ENABLED
            asm volatile(".pragma \"set knob DontInsertYield\";\n" : : : "memory");
#endif

#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 900
            uint64_t * bar_ptr = reinterpret_cast<uint64_t *>(bar_base_ + id);
            unsigned smem_ptr = get_smem_pointer(bar_ptr);
            asm volatile("{\n\t"
                         ".reg .pred                P1; \n\t"
                         "LAB_WAIT: \n\t"
			 //"mbarrier.try_wait.parity.b64 P1, [%0], %1; \n\t"
			 "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1; \n\t"
                         "@P1                       bra.uni DONE; \n\t"
                         "bra.uni                   LAB_WAIT; \n\t"
                         "DONE: \n\t"
                         "}"
                         : : "r"(smem_ptr),"r"(bar_phase));
#else
            uint64_t * bar_ptr = reinterpret_cast<uint64_t *>(bar_base_ + id);
            unsigned smem_ptr = get_smem_pointer(bar_ptr);

            asm volatile("{\n\t"
                         ".reg .pred                P1; \n\t"
                         ".reg .s32                 high, low; \n\t"
                         ".reg .u32                 lwrrentPhase; \n\t"
                         "ld.volatile.shared.v2.s32 { low, high }, [%0]; \n\t"
                         "shr.u32                   lwrrentPhase, high, 31; \n\t"
                         "setp.ne.u32               P1, lwrrentPhase, %1; \n\t"
                         "@P1                       bra.uni DONE; \n\t"
                         "LAB_WAIT: \n\t"
                         "ld.volatile.shared.v2.s32 { low, high }, [%0]; \n\t"
                         "shr.u32                   lwrrentPhase, high, 31; \n\t"
                         "setp.ne.u32               P1, lwrrentPhase, %1; \n\t"
                         "@P1                       bra.uni DONE; \n\t"
                         "bra.uni                   LAB_WAIT; \n\t"
                         "DONE: \n\t"
                         "}"
                         : : "r"(smem_ptr),"r"(bar_phase));
#endif

#ifdef JETFIRE_ENABLED
            asm volatile(".pragma \"reset knob DontInsertYield\";\n" : : : "memory");
#endif

        }
	
	//Set the expected_transaction_count and add 1 arrive count (1 transction = 1 Byte)
	//This PTX maps to SYNCS.ARRIVES.TRANS64.A1TR.
	inline __device__ void bar_arrive_set_transactioncnt(int id, int expected_copy_bytes)
	{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 900
            uint64_t * bar_ptr = reinterpret_cast<uint64_t *>(bar_base_ + id);
            unsigned smem_ptr = get_smem_pointer(bar_ptr);
            asm volatile("{\n\t"
			 "mbarrier.arrive.expect_copy.shared.b64 _, [%0], %1; \n\t"
                         "}"
                         : : "r"(smem_ptr),"r"(expected_copy_bytes));
#endif
        }

        inline __device__ void bar_arrive_normal(int id, bool flag = true) {

#if LWDA_CP_ASYNC_ACTIVATED && ! (LWDA_CP_ASYNC_MBARRIER_ARRIVE_ACTIVATED)
            asm("membar.cta;");
#endif
            uint64_t * bar_ptr = reinterpret_cast<uint64_t *>(bar_base_ + id);
            unsigned smem_ptr = get_smem_pointer(bar_ptr);

            //to make distance for the dependence between atoms.arrive and shfl
            if ( flag == true ) {
#if LWDA_CP_ASYNC_MBARRIER_ARRIVE_ACTIVATED
                asm volatile("{\n\t"
			    ".reg .b64 state; \n\t"
	                    "mbarrier.arrive.shared.b64   state, [%0];\n\t"
			    "}"
                            : : "r"(smem_ptr));
#elif LWDA_CP_ASYNC_ACTIVATED
                asm volatile("{\n\t"
			    ".reg .b64  state; \n\t"
                            "atom.shared.arrive.b64       state, [%0];"
			    "}"
                            : :"r"(smem_ptr));
#else
                assert(0);
#endif
            }


        }

        inline __device__  void bar_arrive_ldgsts(int id) {

            uint64_t * bar_ptr = reinterpret_cast<uint64_t *> (bar_base_ + id);
            unsigned smem_ptr = get_smem_pointer(bar_ptr);

#if LWDA_CP_ASYNC_MBARRIER_ARRIVE_ACTIVATED
            asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" : : "r"(smem_ptr));
#elif LWDA_CP_ASYNC_ACTIVATED
            asm volatile("cp.async.arrive.shared.b64 [%0];" : : "r"(smem_ptr));
#else
            assert(0);
#endif

        }

        inline __device__ uint64_t * bar_base() {
            return bar_base_;
        }

    private:
        // smem barrier base pointer
        uint64_t * bar_base_;
	// barrier id
        int id_;
};

// Set the expected_transaction_count and add 1 arrive count (1 transction = 1 Byte)
// This PTX maps to SYNCS.ARRIVES.TRANS64.A1TR.
inline __device__ void bar_arrive_set_transactioncnt( unsigned smem_ptr,
                                                      unsigned expected_copy_bytes ) {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 900
    asm volatile( "{\n\t"
                  "mbarrier.arrive.expect_copy.shared.b64 _, [%0], %1; \n\t"
                  "}"
                  :
                  : "r"( smem_ptr ), "r"( expected_copy_bytes ) );
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}//namespace xmma
