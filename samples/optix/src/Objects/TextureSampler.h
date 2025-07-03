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

#include <Memory/GfxInteropResource.h>
#include <Memory/MBufferListener.h>
#include <Memory/MTextureSampler.h>
#include <Objects/Buffer.h>  // Definition needed for LinkedPtr
#include <Objects/ManagedObject.h>
#include <Objects/VariableType.h>
#include <Util/LinkedPtr.h>
#include <Util/TextureDescriptor.h>
#include <o6/optix.h>

#include <vector>

namespace optix {

class TextureSampler : public ManagedObject, public MTextureSamplerListener, public MBufferListener
{
  public:
    // A normal texturesampler creation
    TextureSampler( Context* context );

    // Create a texturesampler from a graphics resource on the given device
    TextureSampler( Context* context, const GfxInteropResource& resource, Device* gfxInteropDevice );

    ~TextureSampler() override;

    // Texture interpolation properties
    void setFilterModes( RTfiltermode minFilter, RTfiltermode magFilter, RTfiltermode mipFilter );
    void getFilterModes( RTfiltermode& minFilter, RTfiltermode& magFilter, RTfiltermode& mipFilter ) const;
    void setMaxAnisotropy( float maxAniso );
    float getMaxAnisotropy() const;
    void setMaxMipLevelClamp( float maxLevel );
    float getMaxMipLevelClamp() const;
    void setMinMipLevelClamp( float minLevel );
    float getMinMipLevelClamp() const;
    void setMipLevelBias( float bias );
    float getMipLevelBias() const;
    void setWrapMode( unsigned int dim, RTwrapmode wrapMode );
    RTwrapmode getWrapMode( unsigned int dim ) const;
    void getWrapModes( RTwrapmode& dim0, RTwrapmode& dim1, RTwrapmode& dim2 ) const;
    void setIndexMode( RTtextureindexmode mode );
    RTtextureindexmode getIndexMode() const;
    void setReadMode( RTtexturereadmode mode );
    RTtexturereadmode getReadMode() const;

    const TextureDescriptor& getDescriptor() const { return m_textureDescriptor; }

    // Size and shape
    RTformat     getBufferFormat() const;
    unsigned int getDimensionality() const;

    // Backing store for non-interop textures. Note: Mip functionality
    // has been moved into buffer.
    void    setBuffer( Buffer* );
    Buffer* getBuffer() const;

    // Textures always have an ID, but that ID may not have been released to the API (aka bindless).
    int  getId() const;
    int  getAPIId();
    bool isBindless() const;

    // Returns true if this is an interop texture and therefore has no
    // Buffer object as backing.
    bool          isInteropTexture() const;
    MBufferHandle getBackingMBuffer() const;

    // Interop
    void                      registerGfxInteropResource();
    void                      unregisterGfxInteropResource();
    const GfxInteropResource& getGfxInteropResource() const;

    const MTextureSamplerHandle& getMTextureSampler() const;
    void                         updateDescriptor();
    void                         writeHeader() const;

    // Called by dependent buffer when it is reallocated
    void mBufferWasChanged();

    // The backing buffer was changed to a new element type
    void bufferFormatDidChange();

    // Called by update manager when the bindings change.  Since we do
    // not track the details of the binding, we will notify the buffer
    // that it should reallocate if necessary.
    void textureBindingsDidChange();

    // if any of the variables referencing this texture sampler are attached, then it is attached
    virtual bool isAttached() const;

    // Used by dependent buffer to determine the type of allocation needed
    bool usesTexFetch() const;
    bool usesNonTexFetch() const;

    // throws exceptions in case of invalid
    void validate() const;

    // LinkedPtr relationship management
    void detachFromParents();
    void detachLinkedChild( const LinkedPtr_Link* link );

    //------------------------------------------------------------------------
    // Graph attachment property
    //------------------------------------------------------------------------
    void receivePropertyDidChange_Attachment( bool added );

    //------------------------------------------------------------------------
    // Index for validation list
    //------------------------------------------------------------------------
    struct validationIndex_fn
    {
        int& operator()( const TextureSampler* sampler ) { return sampler->m_validationIndex; }
    };

    // Demand load texture information
    unsigned int getGutterWidth() const;
    unsigned int getTileWidth() const;
    unsigned int getTileHeight() const;

  private:
    LinkedPtr<TextureSampler, Buffer> m_buffer;  // Attached buffer for non-interop buffers

    // Interpolation modes
    TextureDescriptor m_textureDescriptor;

    // Identifier that identifies the object for any number of
    // purposes (but at least identifying the object on the device).
    ReusableID m_id;

    // Marks whether the program is bindless - i.e., program's ID has been released through the API
    void markAsBindless();
    bool m_isBindless;

    // Manage validation
    void subscribeForValidation();
    void unsubscribeForValidation();

    // Memory allocation management
    MBufferHandle         m_backing;            // Low-level storage - shared with Buffer or Standalone (interop)
    MTextureSamplerHandle m_msampler;           // Texture sampler resource
    bool                  m_isInterop = false;  // True if allocated as interop
    void eventMTextureSamplerMAccessDidChange( const Device* device, const MAccess& oldMTA, const MAccess& newMTA ) override;
    void eventMBufferMAccessDidChange( const MBuffer* buffer, const Device* device, const MAccess& oldMBA, const MAccess& newMBA ) override;

    // For notifying variables of changes
    void notifyVariables_FormatDidChange();

    // Attachment property
    GraphPropertySingle<> m_attachment;

    // Used by IndexedVector in Validationmanager. Mutable so that scope
    // can remain const in validation manager.
    mutable int m_validationIndex = -1;

    void reallocateTextureSampler();

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_TEXTURE_SAMPLER};
};

inline bool TextureSampler::isA( ManagedObjectType type ) const
{
    return type == m_objectType || ManagedObject::isA( type );
}

}  // namespace optix
