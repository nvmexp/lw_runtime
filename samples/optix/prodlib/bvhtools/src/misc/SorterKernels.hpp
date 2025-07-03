// Copyright LWPU Corporation 2016
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma once

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------------

size_t radixSortTempSize(size_t key_bytes, size_t value_bytes, unsigned int num_elements);

void radixSort(void* d_temp_storage, size_t temp_storage_bytes,
  unsigned int * d_keys_in, unsigned int * d_keys_out,
  unsigned int * d_values_in, unsigned int * d_values_out,
  unsigned int num_elements, void* stream);

void radixSort(void* d_temp_storage, size_t temp_storage_bytes,
  unsigned long long * d_keys_in, unsigned long long * d_keys_out,
  unsigned int * d_values_in, unsigned int * d_values_out,
  unsigned int num_elements, void* stream);

//------------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
