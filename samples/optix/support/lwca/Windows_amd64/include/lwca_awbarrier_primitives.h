/*
 * Copyright 1993-2019 LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to LWPU intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to LWPU and is being provided under the terms and
 * conditions of a form of LWPU software license agreement by and
 * between LWPU and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of LWPU is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * LWPU DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL LWPU BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef _LWDA_AWBARRIER_PRIMITIVES_H_
# define _LWDA_AWBARRIER_PRIMITIVES_H_

# include "lwda_awbarrier_helpers.h"

# if !defined(_LWDA_AWBARRIER_ARCH_700_OR_LATER)
#  error This file requires compute capability 7.0 or greater.
# endif

typedef uint64_t __mbarrier_t;
typedef uint64_t __mbarrier_token_t;

_LWDA_AWBARRIER_QUALIFIER __host__
uint32_t __mbarrier_maximum_count()
{
    return _LWDA_AWBARRIER_MAX_COUNT;
}

_LWDA_AWBARRIER_QUALIFIER
void __mbarrier_init(__mbarrier_t* barrier, uint32_t expected_count)
{
    _LWDA_AWBARRIER_ASSERT(__isShared(barrier));
    _LWDA_AWBARRIER_ASSERT(expected_count > 0 && expected_count <= _LWDA_AWBARRIER_MAX_COUNT);

    _LWDA_AWBARRIER_INTERNAL_NAMESPACE::awbarrier_init(barrier, expected_count);
}

_LWDA_AWBARRIER_QUALIFIER
void __mbarrier_ilwal(__mbarrier_t* barrier)
{
    _LWDA_AWBARRIER_ASSERT(__isShared(barrier));

    _LWDA_AWBARRIER_INTERNAL_NAMESPACE::awbarrier_ilwal(barrier);
}

_LWDA_AWBARRIER_QUALIFIER
__mbarrier_token_t __mbarrier_arrive(__mbarrier_t* barrier)
{
    _LWDA_AWBARRIER_ASSERT(__isShared(barrier));

    return _LWDA_AWBARRIER_INTERNAL_NAMESPACE::awbarrier_arrive_drop<false>(barrier);
}

_LWDA_AWBARRIER_QUALIFIER
__mbarrier_token_t __mbarrier_arrive_and_drop(__mbarrier_t* barrier)
{
    _LWDA_AWBARRIER_ASSERT(__isShared(barrier));

    return _LWDA_AWBARRIER_INTERNAL_NAMESPACE::awbarrier_arrive_drop<true>(barrier);
}

_LWDA_AWBARRIER_QUALIFIER
bool __mbarrier_test_wait(__mbarrier_t* barrier, __mbarrier_token_t token)
{
    _LWDA_AWBARRIER_ASSERT(__isShared(barrier));

    return _LWDA_AWBARRIER_INTERNAL_NAMESPACE::awbarrier_test_wait(barrier, token);
}

_LWDA_AWBARRIER_QUALIFIER
uint32_t __mbarrier_token_pending_count(__mbarrier_token_t token)
{
    return _LWDA_AWBARRIER_INTERNAL_NAMESPACE::awbarrier_token_pending_count(token);
}

#endif /* !_LWDA_AWBARRIER_PRIMITIVES_H_ */
