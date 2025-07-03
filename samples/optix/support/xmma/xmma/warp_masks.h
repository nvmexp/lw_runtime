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

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int WARPS_M, int WARPS_N, int WARPS_K >
struct Warp_masks {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Warp_masks<8, 1, 1> { enum { M = 0xe0, N = 0x00, K = 0x00 }; };
template<>
struct Warp_masks<4, 2, 1> { enum { M = 0x60, N = 0x80, K = 0x00 }; };
template<>
struct Warp_masks<4, 1, 2> { enum { M = 0x60, N = 0x00, K = 0x80 }; };
template<>
struct Warp_masks<4, 1, 1> { enum { M = 0x60, N = 0x00, K = 0x00 }; };
template<>
struct Warp_masks<2, 4, 1> { enum { M = 0x20, N = 0xc0, K = 0x00 }; };
template<>
struct Warp_masks<2, 2, 2> { enum { M = 0x20, N = 0x40, K = 0x80 }; };
template<>
struct Warp_masks<2, 2, 1> { enum { M = 0x20, N = 0x40, K = 0x00 }; };
template<>
struct Warp_masks<2, 1, 2> { enum { M = 0x20, N = 0x00, K = 0x40 }; };
template<>
struct Warp_masks<2, 1, 1> { enum { M = 0x20, N = 0x00, K = 0x00 }; };
template<>
struct Warp_masks<1, 8, 1> { enum { M = 0x00, N = 0xe0, K = 0x00 }; };
template<>
struct Warp_masks<1, 4, 2> { enum { M = 0x00, N = 0x60, K = 0x80 }; };
template<>
struct Warp_masks<1, 4, 1> { enum { M = 0x00, N = 0x60, K = 0x00 }; };
template<>
struct Warp_masks<1, 2, 2> { enum { M = 0x00, N = 0x20, K = 0x40 }; };
template<>
struct Warp_masks<1, 2, 1> { enum { M = 0x00, N = 0x20, K = 0x00 }; };
template<>
struct Warp_masks<1, 1, 4> { enum { M = 0x00, N = 0x00, K = 0x60 }; };
template<>
struct Warp_masks<1, 1, 2> { enum { M = 0x00, N = 0x00, K = 0x20 }; };
template<>
struct Warp_masks<1, 1, 1> { enum { M = 0x00, N = 0x00, K = 0x00 }; };

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma

