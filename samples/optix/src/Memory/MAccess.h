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

#include <corelib/system/LwdaDriver.h>

#include <prodlib/exceptions/Assert.h>

#include <LWCA/Array.h>
#include <LWCA/TexObject.h>

#include <stdint.h>
#include <vector>


namespace optix {

struct LinearAccess
{
    char* ptr = nullptr;
};

struct PitchedLinearAccess
{
    char*    ptr   = nullptr;
    uint64_t pitch = 0;  // Note: 64-bits is probably overkill.
};

struct TexObjectAccess
{  // SM_30+
    lwca::TexObject texObject;
    unsigned int    indexOffset = 0;
};

struct DemandTexObjectAccess : TexObjectAccess
{
    unsigned int startPage;
    unsigned int numPages;
    unsigned int minMipLevel;
    unsigned int maxMipLevel;
};

struct TexReferenceAccess
{  // SM_20 and texheap only
    int          texUnit     = -1;
    unsigned int indexOffset = 0;
};

struct DemandLoadAccess
{
    unsigned int virtualPageBegin = 0;
};

struct DemandLoadArrayAccess
{
    unsigned int virtualPageBegin = 0;
    unsigned int numPages         = 0;
    unsigned int minMipLevel      = 0;
};

struct DemandLoadTileArrayAccess
{
    // Intentionally empty struct to possibly be used later if some
    // information is needed for the tile array MBuffers.
};

struct LwdaSparseAccess
{
    unsigned int virtualPageBegin = 0;
};

struct LwdaSparseBackingAccess
{
    LWmemGenericAllocationHandle handle = 0;
};

//============================================================================
// Holds the pieces of information necessary to get data out of memory.
class MAccess
{
  public:
    static const int OPTIX_MAX_MIP_LEVELS = 16;

    enum Kind
    {
        LINEAR,
        MULTI_PITCHED_LINEAR,
        TEX_OBJECT,
        TEX_REFERENCE,
        LWDA_SPARSE,
        LWDA_SPARSE_BACKING,
        DEMAND_LOAD,
        DEMAND_LOAD_ARRAY,
        DEMAND_LOAD_TILE_ARRAY,
        DEMAND_TEX_OBJECT,
        NONE
    };

    MAccess();

    static MAccess makeLinear( char* ptr );
    static MAccess makeLwdaSparse( unsigned int pageBegin );
    static MAccess makeLwdaSparseBacking( LWmemGenericAllocationHandle handle );
    static MAccess makeDemandLoad( unsigned int pageBegin );
    static MAccess makeDemandLoadArray( unsigned int pageBegin, unsigned int numPages, unsigned int minMipLevel );
    static MAccess makeDemandLoadTileArray();
    static MAccess makeMultiPitchedLinear( const PitchedLinearAccess* pitchedLinear, int count );
    static MAccess makeTexObject( const lwca::TexObject& texObject, unsigned int indexOffset );
    static MAccess makeDemandTexObject( const lwca::TexObject& texObject,
                                        unsigned int           indexOffset,
                                        unsigned int           pageBegin,
                                        unsigned int           numPages,
                                        unsigned int           minMipLevel,
                                        unsigned int           maxMipLevel );
    static MAccess makeTexReference( unsigned int texUnit, unsigned int indexOffset );
    static MAccess makeNone();

    bool operator==( const MAccess& other ) const;
    bool operator!=( const MAccess& other ) const;

    Kind getKind() const { return m_kind; }

    LinearAccess        getLinear() const;
    PitchedLinearAccess getPitchedLinear( int idx ) const;
    TexObjectAccess           getTexObject() const;
    DemandTexObjectAccess     getDemandTexObject() const;
    TexReferenceAccess        getTexReference() const;
    DemandLoadAccess          getDemandLoad() const;
    DemandLoadArrayAccess     getDemandLoadArray() const;
    DemandLoadTileArrayAccess getDemandLoadTileArray() const;
    LwdaSparseBackingAccess   getLwdaSparseBacking() const;
    LwdaSparseAccess          getLwdaSparse() const;

    char* getLinearPtr() const;

  private:
    // Note: keep member data small, since the "getter" functions return by value
    Kind m_kind;
    union
    {
        LinearAccess              m_linear;
        TexObjectAccess           m_texObject;
        TexReferenceAccess        m_texReference;
        DemandLoadAccess          m_demandLoad;
        DemandLoadArrayAccess     m_demandLoadArray;
        DemandLoadTileArrayAccess m_demandLoadTileArray;
        DemandTexObjectAccess     m_demandTexObject;
        LwdaSparseBackingAccess   m_lwdaSparseBacking;
        LwdaSparseAccess          m_lwdaSparse;
    };
    std::vector<PitchedLinearAccess> m_pitchedLinear;
};

}  // namespace optix
