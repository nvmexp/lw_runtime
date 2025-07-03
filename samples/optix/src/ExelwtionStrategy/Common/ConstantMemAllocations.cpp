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

#include <Util/PersistentStream.h>
#include <src/LWCA/Memory.h>
#include <src/ExelwtionStrategy/Common/ConstantMemAllocations.h>

using namespace optix;

ConstantMemAllocations::ConstantMemAllocations()
{
}

bool ConstantMemAllocations::operator==( const ConstantMemAllocations& other ) const
{
    return staticInitializers == other.staticInitializers && structGlobalSize == other.structGlobalSize
           && traversableTableSize == other.traversableTableSize && objectRecordSize == other.objectRecordSize
           && bufferTableSize == other.bufferTableSize && textureTableSize == other.textureTableSize
           && programTableSize == other.programTableSize && programToEntryStateIDSize == other.programToEntryStateIDSize
           && jumpTableSize == other.jumpTableSize;
}

bool ConstantMemAllocations::operator!=( const ConstantMemAllocations& other ) const
{
    return !( *this == other );
}

bool ConstantMemAllocations::operator<( const ConstantMemAllocations& other ) const
{
    if( staticInitializers != other.staticInitializers )
        return staticInitializers < other.staticInitializers;
    if( structGlobalSize != other.structGlobalSize )
        return structGlobalSize < other.structGlobalSize;
    if( traversableTableSize != other.traversableTableSize )
        return traversableTableSize < other.traversableTableSize;
    if( objectRecordSize != other.objectRecordSize )
        return objectRecordSize < other.objectRecordSize;
    if( bufferTableSize != other.bufferTableSize )
        return bufferTableSize < other.bufferTableSize;
    if( textureTableSize != other.textureTableSize )
        return textureTableSize < other.textureTableSize;
    if( programTableSize != other.programTableSize )
        return programTableSize < other.programTableSize;
    if( programToEntryStateIDSize != other.programToEntryStateIDSize )
        return programToEntryStateIDSize < other.programToEntryStateIDSize;
    if( jumpTableSize != other.jumpTableSize )
        return jumpTableSize < other.jumpTableSize;
    return remainingSize < other.remainingSize;
}
