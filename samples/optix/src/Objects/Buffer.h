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

#include <Objects/ManagedObject.h>

#include <Device/DeviceSet.h>
#include <FrontEnd/Canonical/CanonicalProgramID.h>
#include <Memory/GfxInteropResource.h>
#include <Memory/MBuffer.h>
#include <Memory/MBufferListener.h>
#include <Memory/MapMode.h>
#include <Objects/GraphProperty.h>
#include <Objects/Variable.h>

#include <map>
#include <vector>

namespace optix {

class CallSiteIdentifier;
class Device;
class MappedSet;
class PostprocessingStage;
class VariableReference;

class Buffer : public ManagedObject, public MBufferListener
{
    typedef std::pair<size_t, size_t> MapRange;
    typedef std::vector<MapRange> MapRangeVector;

  public:
    // A normal buffer creation
    Buffer( Context* context, unsigned int type );

    // Create a Buffer from a graphics resource on the given device
    Buffer( Context* context, unsigned int type, const GfxInteropResource& resource, Device* gfxInteropDevice );
    // Create a demand loaded buffer from a callback
    Buffer( Context* context, unsigned int type, RTbuffercallback callback, void* callbackData );

    ~Buffer() override;

    // React to device changes
    void postSetActiveDevices( const DeviceSet& removedDevices );

    // public API
    RTformat     getFormat() const;
    RTbuffertype getType() const;
    unsigned int getDimensionality() const;
    size_t       getElementSize() const;  // in bytes
    size_t       getWidth() const;
    size_t       getHeight() const;
    size_t       getDepth() const;
    unsigned int getMipLevelCount() const;
    unsigned int getMipTailFirstLevel() const;
    void getSize( size_t dims[3] ) const;
    size_t getTotalSizeInBytes() const;
    size_t getLevelSizeInBytes( unsigned int level ) const;
    size_t getLevelWidth( unsigned int level ) const;
    size_t getLevelHeight( unsigned int level ) const;
    size_t getLevelDepth( unsigned int level ) const;
    unsigned int getPageWidth() const { return m_pageWidth; }
    unsigned int getPageHeight() const { return m_pageHeight; }
    unsigned int getPageDepth() const { return m_pageDepth; }

    bool empty() const;
    bool isElementSizeInitialized() const;

    MapMode getMapModeForAPI( unsigned int apiMapFlags ) const;

    void setFormat( RTformat fmt );
    void setElementSize( size_t elementSize );
    void setMipLevelCount( unsigned int levels );
    void checkElementSizeInitialized() const;

    void setSize( size_t dimensionality, size_t* dims );
    void setSize1D( size_t width )
    {
        size_t dims[] = {width};
        setSize( 1, dims );
    }
    void setSize2D( size_t width, size_t height )
    {
        size_t dims[] = {width, height};
        setSize( 2, dims );
    }
    void setSize3D( size_t width, size_t height, size_t depth )
    {
        size_t dims[] = {width, height, depth};
        setSize( 3, dims );
    }

    void* map( MapMode mapMode, unsigned int level = 0 );
    void unmap( unsigned int level = 0 );
    bool isMappedHost() const;
    bool isMappedHost( unsigned int level ) const;
    void* getMappedHostPtr( unsigned int level = 0 ) const;
    bool isInterop() const;

    // Graphics interop (OGL, D3D9, D3D10, D3D11 )
    const GfxInteropResource& getGfxInteropResource() const;
    bool                      isGfxInterop() const;
    void                      registerGfxInteropResource();
    void                      unregisterGfxInteropResource();

    // LWCA interop. Warning: getting the interop pointer may have
    // side-effects, since it will freeze the pointer until the buffer
    // is destroyed or another pointer is set.
    void setInteropPointer( void* ptr, Device* device );
    void* getInteropPointer( Device* device );
    bool isLwdaInterop() const;
    void markDirty();

    BufferDimensions getDimensions() const;
    MBufferHandle    getMBuffer() const;
    void             writeHeader();

    // Demand loading support.
    RTbuffercallback getCallback() const { return m_callback; }
    void*            getCallbackData() const { return m_callbackData; }
    unsigned int     getPageSize() const { return BUFFER_PAGE_SIZE_IN_BYTES; }
    bool             isDemandLoad() const { return m_callback != nullptr; }
    bool             hasTextureAttached() const;
    TextureSampler*  getAttachedTextureSampler() const;

    // helper function
    static void checkBufferType( unsigned int type, bool isInteropBuffer );

    // Buffers always have an ID, but that ID may not have been released to the API (aka bindless).
    int  getId() const;
    int  getAPIId();
    bool isBindless() const;
    void markAsBindlessForInternalUse();

    // Special case for non mipmapped demand load buffers that are small enough to fit into
    // a single level of a LWCA HW sparse texture miptail.
    void switchLwdaSparseArrayToLwdaArray( DeviceSet devices ) const;

    void bufferBindingsDidChange( VariableReferenceID refid, bool added );
    void attachedTextureDidChange();
    void hasRawAccessDidChange();

    // if any of the variables or texture samplers referencing this buffer are attached, then it is attached
    bool isAttached() const;

    // throws exceptions in case of invalid
    void validate() const;

    // Linked pointer relationship management
    void detachFromParents();
    void detachLinkedChild( const LinkedPtr_Link* link );

    // Called when this buffer is set/removed from a variable attached to a PostprocessingStage.
    void addOrRemovePostprocessingStage( bool added );

    //------------------------------------------------------------------------
    // Raw access property
    //------------------------------------------------------------------------
  private:
    void addOrRemoveProperty_rawAccess( bool added );

    //------------------------------------------------------------------------
    // Graph attachment property
    //------------------------------------------------------------------------
  public:
    void receivePropertyDidChange_Attachment( bool added );

    //------------------------------------------------------------------------
    // Index for validation list
    //------------------------------------------------------------------------
    struct validationIndex_fn
    {
        int& operator()( const Buffer* buffer ) { return buffer->m_validationIndex; }
    };

  private:
    const unsigned int BUFFER_PAGE_SIZE_IN_BYTES = 4096U;

    // Buffer properties
    RTbuffertype     m_iotype;
    unsigned int     m_flags = 0;
    BufferDimensions m_bufferSize;

    // The primary resource, owned by the memory manager
    MBufferHandle m_mbuffer;

    // Devices for which pointers have been retrieved with getInteropPointer()
    // or set with setInteropPointer().
    DeviceSet m_clientVisiblePointers;

    // Devices for which pointers have been set with setInteropPointer().
    DeviceSet m_setPointers;

    // Managing external mappings to host
    std::vector<void*> m_mappedHostPtrs;  // separate pointer per MIP level

    // Identifier that identifies the object for any number of purposes (but at least
    // identifying the object on the device).
    ReusableID m_id;

    // Holds whether this buffer has been mapped at least once.
    bool m_mappedOnce = false;

    // Marks whether the program is bindless - i.e., program's ID has been released through the API
    void markAsBindless();
    bool m_isBindless = false;

    // Manage validation
    void subscribeForValidation();

    bool canHaveMipLevels() const;

    // Memory allocation management
    void          updateMBuffer();
    MBufferPolicy determineBufferPolicy( bool gfxInterop, MBufferPolicy oldPolicy ) const;
    bool hasVariableBindings() const;
    bool anyTextureUsesTexFetch() const;
    bool allTexturesUseTexFetch() const;
    void eventMBufferMAccessDidChange( const MBuffer* buffer, const Device* device, const MAccess& oldMBA, const MAccess& newMBA ) override;

    // For notifying variables and attached texture samplers of changes
    void notifyParent_SizeOrFormatDidChange();

    void programIdBufferContentChanged( void* bufPtr, bool added );
    void programIdBufferVarBindingChanged( const VariableReference* varref, bool added );
    void updateProgramIdBufferCallsite( int* bufPtr, CallSiteIdentifier* csId, bool added );

    // Count of the number of references that require raw buffer accesses
    GraphPropertySingle<> m_rawAccess;

    // Attachment property
    GraphPropertySingle<> m_attachment;

    // Used by IndexedVector in Validationmanager. Mutable so that scope
    // can remain const in validation manager.
    mutable int m_validationIndex = -1;

    // Helper for common code in constructor
    Buffer( unsigned int type, Context* context, RTbuffercallback callback = nullptr, void* callbackData = nullptr );

    // Non empty if there are post-processing stages that have variables bound to this buffer.
    GraphPropertySingle<int> m_usedInPostProcessingStage;

    // Links to the bindless callable programs contained in the buffer for call site handling.
    // Only used if format is RT_FORMAT_PROGRAM_ID
    std::vector<LinkedPtr<Buffer, Program>> m_bindlessCallables;
    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_BUFFER};

  private:
    // Callback for demand loaded buffers
    RTbuffercallback m_callback     = nullptr;
    void*            m_callbackData = nullptr;

    // Page dimensions (in elements) for demand loaded buffers and textures.
    unsigned int m_pageWidth  = 0;
    unsigned int m_pageHeight = 0;
    unsigned int m_pageDepth  = 0;
};

inline bool Buffer::isA( ManagedObjectType type ) const
{
    return type == m_objectType || ManagedObject::isA( type );
}

}  // namespace optix
