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
#include <xmma/utils.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
// CTA Register Reconfig FEATURES /
// To program this feature, there are two requirements:
// 1.  specify the new target register count for reg_alloc and reg_dealloc
//     There are two limitation by Arch to be valid register count: 
//        a) Register count needs to be mutliple of 8.
//        b) Total register being allocate (acquired) is no larger than total register being 
//           deallocated (released).
// 2.  call them in the proper location to form the scope that  the new targe count applies.
//       The scope starts from the reg_alloc /reg_dealloc calling point and ends with EXIT inst.
//
// Developer can try different target regsiter counts until finding the one which leads to no 
// register spill. The total registers being deallocated is the subtraction between default regsiter count 
// and new lower target register count and then by  total threads calling this reg_dealloc.  
// The default register count            =
//      65536 / Threads Per CTA ;  //The result needs to be nearest value that is multiple of 8.
// The total register being de_allocated  =
//      total threads in calling reg_dealloc x (default register count - new lower target count)
// The total register being allocated    =
//      total threads in calling reg_alloc x (new higher target count - default register count)
//
// So changes on either total threads in calling reg_alloc/reg_dealloc or default register count or 
// new target count can change the total registers being deallocated or allocated, thus needs to 
// make sure the above requirements in 1.b) is valid.
//
// The register reconfig target register count  calucator link (for temporary usage):
// https://docs.google.com/spreadsheets/d/1OhuUeHUib2RB9UcLmyn6JA3KeOT_K4P_A1lg7pd_Lkg/edit#gid=1008465518
//
// More descriptions in: https://confluence.lwpu.com/pages/viewpage.action?pageId=656672843
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {


////////////////////////////////////////////////////////////////////////////////////////////////////
// resize by increasing the registers per thread to be target number (immediate integer);
////////////////////////////////////////////////////////////////////////////////////////////////////
// The new target regsier count can be adjusted to make sure no regsiter spill happens.
// Adjusting new target count for reg_alloc needs to ajust the new target count for reg_dealloc 
// as the registers newly allocated can't exceed the register being deallocated.
inline __device__ void reg_alloc()
{
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 900
    const int TARGET_REG_COUNT = 232;  //Example values (use with reg_delloc's VALUE together): 224, 232, 208, 216
    asm volatile("{\n\t"
		"setmaxreg.alloc.sync.u32   %0;\n\t"
		"}"
		::"n"(TARGET_REG_COUNT)); 
#endif
}
//////////////////////////////////////////////////////////////////////////////////////////////////////
// resize by decreasing the registers per thread to be target number (immediate integer);
////////////////////////////////////////////////////////////////////////////////////////////////////
// The new target regsier count can be adjusted to make sure no regsiter spill happens.
inline __device__ void reg_dealloc()
{
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 900
    const int TARGET_REG_COUNT = 40;// Exmaple values: 56,  40, 88, 72
    asm volatile("{\n\t"
		 "setmaxreg.dealloc.sync.u32  %0; \n\t"
	         "}"
	         ::"n"(TARGET_REG_COUNT));
#endif
}

}//namespace xmma_new
