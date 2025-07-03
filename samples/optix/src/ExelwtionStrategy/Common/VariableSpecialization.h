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

#include <Objects/VariableReferenceBinding.h>
#include <Util/IDMap.h>


namespace optix {
class PersistentStream;

struct VariableSpecialization
{
    //
    // WARNING: This is a persistent class. If you change anything you
    // should also update the readOrWrite function and bump the the
    // version number.
    //

    // These identify how we lookup the variable's value.
    enum LookupKind
    {
        SingleId,
        SingleScope,
        Unused,
        GenericLookup
    };

    // These identify how we load the data after we lookup the variables value.
    // Really only applicable to things like buffers and textures.
    enum AccessKind
    {
        HWTextureOnly,
        SWTextureOnly,
        TexHeap,
        TexHeapSingleOffset,
        PitchedLinear,
        PitchedLinearPreferLDG,
        GenericAccess
    };

    LookupKind               lookupKind = Unused;
    AccessKind               accessKind = GenericAccess;
    VariableReferenceBinding singleBinding;
    int                      singleId             = -1;
    unsigned int             singleOffset         = 0;
    unsigned short           dynamicVariableToken = IDMap<std::string, unsigned short>::ILWALID_INDEX;
    int                      texheapUnit          = -3;
    bool                     isReadOnly           = false;

    void setGeneric( unsigned short token );
    void setSingleBinding( const VariableReferenceBinding& b );
    void setSingleId( int singleId );
    void setUnused();

    bool operator==( const VariableSpecialization& b ) const;
    bool operator!=( const VariableSpecialization& b ) const;
    bool operator<( const VariableSpecialization& b ) const;
    bool preferLDG() const;
};

// Persistence support
void readOrWrite( PersistentStream* stream, VariableSpecialization* vs, const char* label );
}
