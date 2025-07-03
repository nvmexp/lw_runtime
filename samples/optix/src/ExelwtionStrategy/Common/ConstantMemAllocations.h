// Copyright (c) 2017, LWPU CORPORATION.
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

#include <lwca.h>
#include <vector>

namespace optix {
class PersistentStream;

class ConstantMemAllocations
{

  public:
    ConstantMemAllocations();
    // Things we can put into constant memory.  Each one of these member variables
    // identifies the specific entry in the entries list.

    // 1. static initializers - these come from the input code
    size_t staticInitializers = 0;
    // 2. struct Global
    size_t structGlobalSize = 0;
    // 3. traversable table
    size_t traversableTableSize = 0;
    // 4. object record
    size_t objectRecordSize = 0;
    // 5. buffer table
    size_t bufferTableSize = 0;
    // 6. texture table
    size_t textureTableSize = 0;
    // 7. program table
    size_t programTableSize = 0;
    // 8. cpid -> state id
    size_t programToEntryStateIDSize = 0;
    // 9. jumpTable
    size_t jumpTableSize = 0;
    // 10. buffer contents
    //
    // remaining
    size_t remainingSize = 0;

    bool operator==( const ConstantMemAllocations& other ) const;
    bool operator!=( const ConstantMemAllocations& other ) const;
    bool operator<( const ConstantMemAllocations& other ) const;
};

}  // end namespace optix
