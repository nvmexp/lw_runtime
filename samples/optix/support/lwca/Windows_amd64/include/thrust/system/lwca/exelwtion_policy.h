/******************************************************************************
 * Copyright (c) 2016, LWPU CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the LWPU CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL LWPU CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

// histogram
// sort (radix-sort, merge-sort)

#include <thrust/detail/config.h>
#include <thrust/system/lwca/detail/exelwtion_policy.h>
#include <thrust/system/lwca/detail/par.h>

// pass
// ----------------
#include <thrust/system/lwca/detail/adjacent_difference.h>
#include <thrust/system/lwca/detail/copy.h>
#include <thrust/system/lwca/detail/copy_if.h>
#include <thrust/system/lwca/detail/count.h>
#include <thrust/system/lwca/detail/equal.h>
#include <thrust/system/lwca/detail/extrema.h>
#include <thrust/system/lwca/detail/fill.h>
#include <thrust/system/lwca/detail/find.h>
#include <thrust/system/lwca/detail/for_each.h>
#include <thrust/system/lwca/detail/gather.h>
#include <thrust/system/lwca/detail/generate.h>
#include <thrust/system/lwca/detail/inner_product.h>
#include <thrust/system/lwca/detail/mismatch.h>
#include <thrust/system/lwca/detail/partition.h>
#include <thrust/system/lwca/detail/reduce_by_key.h>
#include <thrust/system/lwca/detail/remove.h>
#include <thrust/system/lwca/detail/replace.h>
#include <thrust/system/lwca/detail/reverse.h>
#include <thrust/system/lwca/detail/scatter.h>
#include <thrust/system/lwca/detail/swap_ranges.h>
#include <thrust/system/lwca/detail/tabulate.h>
#include <thrust/system/lwca/detail/transform.h>
#include <thrust/system/lwca/detail/transform_reduce.h>
#include <thrust/system/lwca/detail/transform_scan.h>
#include <thrust/system/lwca/detail/uninitialized_copy.h>
#include <thrust/system/lwca/detail/uninitialized_fill.h>
#include <thrust/system/lwca/detail/unique.h>
#include <thrust/system/lwca/detail/unique_by_key.h>

// fail
// ----------------
// fails with mixed types
#include <thrust/system/lwca/detail/reduce.h>

// mixed types are not compiling, commented in testing/scan.lw
#include <thrust/system/lwca/detail/scan.h>

// stubs passed
// ----------------
#include <thrust/system/lwca/detail/binary_search.h>
#include <thrust/system/lwca/detail/merge.h>
#include <thrust/system/lwca/detail/scan_by_key.h>
#include <thrust/system/lwca/detail/set_operations.h>
#include <thrust/system/lwca/detail/sort.h>

// work in progress

