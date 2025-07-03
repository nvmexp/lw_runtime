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

#include <Objects/Buffer.h>

#include <Context/BindingManager.h>
#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Context/ProgramManager.h>
#include <Context/TableManager.h>
#include <Context/UpdateManager.h>
#include <Context/ValidationManager.h>
#include <Device/CPUDevice.h>
#include <Device/Device.h>
#include <Device/DeviceManager.h>
#include <Exceptions/AlreadyMapped.h>
#include <Exceptions/TypeMismatch.h>
#include <Memory/DemandLoad/PagingService.h>
#include <Memory/MemoryManager.h>
#include <Objects/GeometryTriangles.h>
#include <Objects/PostprocessingStage.h>
#include <Objects/TextureSampler.h>
#include <Util/LinkedPtrHelpers.h>
#include <Util/Misc.h>

#include <corelib/misc/String.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidContext.h>
#include <prodlib/exceptions/IlwalidOperation.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/ValidationError.h>
#include <prodlib/math/Bits.h>
#include <prodlib/misc/GLFunctions.h>
#include <prodlib/misc/RTFormatUtil.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

#include <cmath>
#include <private/optix_declarations_private.h>

using namespace prodlib;
using namespace corelib;

namespace optix {

// Helper class used for buffer data access during call site updating
// without going through map/unmap. Needed when buffer variable bindings change
// and when programs are destroyed.
struct ScopedRawMapper
{
  public:
    ScopedRawMapper( Context* context, MBufferHandle mBuffer )
        : m_mbuffer( mBuffer )
        , m_context( context )
    {
        MemoryManager* mm = m_context->getMemoryManager();
        // We are directly going through the MBuffer here because using map/unmap
        // would cause an update of the call site with the content of the buffer.
        if( !mm->isMappedToHost( m_mbuffer ) )
        {
            m_data       = mm->mapToHost( m_mbuffer, MAP_READ_WRITE );
            m_needsUnmap = true;
        }
        else
        {
            m_data = mm->getMappedToHostPtr( m_mbuffer );
        }
    }
    ~ScopedRawMapper()
    {
        if( m_needsUnmap )
        {
            m_context->getMemoryManager()->unmapFromHost( m_mbuffer );
        }
    }
    void* get() { return m_data; }

  private:
    MBufferHandle m_mbuffer;
    Context*      m_context    = nullptr;
    void*         m_data       = nullptr;
    bool          m_needsUnmap = false;
};

Buffer::Buffer( Context* context, unsigned int type )
    : Buffer( type, context )
{
    try
    {
        // Create the buffer
        MemoryManager* mm     = m_context->getMemoryManager();
        MBufferPolicy  policy = determineBufferPolicy( false, MBufferPolicy::unused );
        m_mbuffer             = mm->allocateMBuffer( m_bufferSize, policy, this );
    }
    catch( ... )
    {
        // If an exception was thrown (say in allocateMBuffer), we need to clean up after
        // ourselves.  Note there might be other instances of objects that would need to be
        // cleaned up.
        if( m_mbuffer )
            m_mbuffer->removeListener( this );
        m_context->getValidationManager()->unsubscribeForValidation( this );
        throw;
    }
}

Buffer::Buffer( Context* context, unsigned int type, const GfxInteropResource& resource, Device* interopDevice )
    : Buffer( type, context )
{
    try
    {
        RT_ASSERT( resource.kind != GfxInteropResource::NONE );

        // Create the mbuffer
        MemoryManager* mm     = m_context->getMemoryManager();
        MBufferPolicy  policy = determineBufferPolicy( true, MBufferPolicy::unused );
        m_mbuffer             = mm->allocateMBuffer( resource, interopDevice, policy, this );

        // Update the current size
        m_bufferSize = m_mbuffer->getDimensions();
    }
    catch( ... )
    {
        // If an exception was thrown (say in allocateMBuffer when the Gfx resource's format
        // isn't supported), we need to clean up after ourselves.  Note there might be other
        // instances of objects that would need to be cleaned up.
        if( m_mbuffer )
            m_mbuffer->removeListener( this );
        m_context->getValidationManager()->unsubscribeForValidation( this );
        throw;
    }
}

Buffer::Buffer( Context* context, unsigned int type, RTbuffercallback callback, void* callbackData )
    : Buffer( type, context, callback, callbackData )
{
    try
    {
        // TODO: Do real memory manager junk, not the stuff for a plain buffer object.
        MemoryManager* mm     = m_context->getMemoryManager();
        MBufferPolicy  policy = determineBufferPolicy( false, MBufferPolicy::unused );
        m_mbuffer             = mm->allocateMBuffer( m_bufferSize, policy, this );

        if( callback )
            m_context->getPagingManager()->enable();
    }
    catch( ... )
    {
        if( m_mbuffer )
            m_mbuffer->removeListener( this );
        m_context->getValidationManager()->unsubscribeForValidation( this );
        throw;
    }
}

// Private constructor for the code shared between both cases. Note
// that argument order is switched to provide an overload
Buffer::Buffer( unsigned int type, Context* context, RTbuffercallback callback, void* callbackData )
    : ManagedObject( context, RT_OBJECT_BUFFER )
    , m_iotype( static_cast<RTbuffertype>( onlyFlagBits( type, RT_BUFFER_INPUT_OUTPUT ) ) )
    , m_flags( maskOutFlags( type, RT_BUFFER_INPUT_OUTPUT ) )
    , m_mappedHostPtrs( 1, nullptr )
    , m_id( context->getObjectManager()->registerObject( this ) )
    , m_callback( callback )
    , m_callbackData( callbackData )
{
    // Validate
    subscribeForValidation();
}

Buffer::~Buffer()
{
    // This isn't ideal.  The TableManager keeps ahold of all the buffer headers.  Right now
    // there isn't a mechanism to notify the TableManager directly that the buffer header is
    // going away.  When we create the Buffer we notify the ObjectManager, but no notification
    // makes it back to the ObjectManager when an object is being deleted.  The m_id is freed,
    // and the object is removed from the ObjectManger's list behind the OM's back.
    m_context->getTableManager()->clearBufferHeader( *m_id );
    if( m_mbuffer )
        m_mbuffer->removeListener( this );
    m_mbuffer.reset();
    m_context->getValidationManager()->unsubscribeForValidation( this );
    RT_ASSERT_MSG( m_validationIndex == -1, "Failed to remove object from validation list" );
}

void Buffer::postSetActiveDevices( const DeviceSet& removedDevices )
{
    // Update LWCA interop pointer sets. Note that this is just the high-level
    // tracking of these sets -- the resource-level updates are taken care of by
    // the MemoryManager. We need to react here as well because updating the sets
    // may require a policy change.

    m_clientVisiblePointers -= removedDevices;
    m_setPointers -= removedDevices;

    updateMBuffer();
}

bool Buffer::isAttached() const
{
    return !m_attachment.empty();
}

void Buffer::addOrRemoveProperty_rawAccess( bool added )
{
    m_rawAccess.addOrRemoveProperty( added );
}

void Buffer::validate() const
{
    // Attachment isn't always required to perform validation.  rtBufferValidate doesn't require
    // attachment, but ValidationManager does.  Do the attachment check in ValidationManager.

    if( getDimensionality() == 0 )
        throw ValidationError( RT_EXCEPTION_INFO, "Buffer dimensionality is not set" );
    if( getFormat() == RT_FORMAT_UNKNOWN )
        throw ValidationError( RT_EXCEPTION_INFO, "Buffer format is not set" );
}

void Buffer::detachFromParents()
{
    auto iter = m_linkedPointers.begin();
    while( iter != m_linkedPointers.end() )
    {
        LinkedPtr_Link* parentLink = *iter;

        // Parents can be variables or texture samplers
        if( Variable* variable = getLinkToBufferFrom<Variable>( parentLink ) )
            variable->detachLinkedChild( parentLink );
        else if( TextureSampler* sampler = getLinkToBufferFrom<TextureSampler>( parentLink ) )
            sampler->detachLinkedChild( parentLink );
        else if( PostprocessingStage* stage = getLinkToBufferFrom<PostprocessingStage>( parentLink ) )
            stage->detachLinkedChild( parentLink );
        else if( GeometryTriangles* gt = getLinkToBufferFrom<GeometryTriangles>( parentLink ) )
            gt->detachLinkedChild( parentLink );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to Buffer" );

        iter = m_linkedPointers.begin();
    }
}

void Buffer::detachLinkedChild( const LinkedPtr_Link* link )
{
    unsigned int index;
    if( getElementIndex( m_bindlessCallables, link, index ) )
    {
        Program*        program = m_bindlessCallables[index].get();
        ScopedRawMapper buffer( m_context, m_mbuffer );
        int*            data = (int*)buffer.get();
        // set all oclwrences of the program to 0
        for( size_t i = 0, e = getWidth() * getHeight() * getDepth(); i < e; ++i )
        {
            if( data[i] == program->getId() )
                data[i] = 0;
        }
        // update the callsites (will also take care of the linked pointers)
        programIdBufferContentChanged( data, true );
    }
}

RTformat Buffer::getFormat() const
{
    return m_bufferSize.format();
}

RTbuffertype Buffer::getType() const
{
    return m_iotype;
}

unsigned int Buffer::getDimensionality() const
{
    return m_bufferSize.dimensionality();
}

size_t Buffer::getElementSize() const
{
    return m_bufferSize.elementSize();
}

size_t Buffer::getWidth() const
{
    return m_bufferSize.width();
}

size_t Buffer::getHeight() const
{
    return m_bufferSize.height();
}

size_t Buffer::getDepth() const
{
    return m_bufferSize.depth();
}

unsigned int Buffer::getMipLevelCount() const
{
    return m_bufferSize.mipLevelCount();
}

unsigned int Buffer::getMipTailFirstLevel() const
{
    return m_context->getPagingManager()->getSoftwareMipTailFirstLevel( m_mbuffer->getDimensions() );
}

size_t Buffer::getTotalSizeInBytes() const
{
    return m_bufferSize.getTotalSizeInBytes();
}

size_t Buffer::getLevelSizeInBytes( unsigned int level ) const
{
    return m_bufferSize.getLevelSizeInBytes( level );
}

size_t Buffer::getLevelWidth( unsigned int level ) const
{
    return m_bufferSize.levelWidth( level );
}

size_t Buffer::getLevelHeight( unsigned int level ) const
{
    return m_bufferSize.levelHeight( level );
}

size_t Buffer::getLevelDepth( unsigned int level ) const
{
    return m_bufferSize.levelDepth( level );
}

void Buffer::notifyParent_SizeOrFormatDidChange()
{
    for( auto parentLink : m_linkedPointers )
    {
        // Parents can be variables, texture samplers, or post-processing stages
        if( Variable* variable = getLinkToBufferFrom<Variable>( parentLink ) )
            variable->bufferFormatDidChange();
        else if( TextureSampler* ts = getLinkToBufferFrom<TextureSampler>( parentLink ) )
            ts->bufferFormatDidChange();
        else if( PostprocessingStage* ps = getLinkToBufferFrom<PostprocessingStage>( parentLink ) )
            ps->bufferFormatDidChange( parentLink );
        else if( GeometryTriangles* gt = getLinkToBufferFrom<GeometryTriangles>( parentLink ) )
            gt->bufferFormatDidChange();
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to Buffer" );
    }
}

void Buffer::setFormat( RTformat fmt )
{
    if( isMappedHost() )
        throw AlreadyMapped( RT_EXCEPTION_INFO, "Cannot resize buffer while mapped" );

    if( fmt == m_bufferSize.format() && fmt != RT_FORMAT_UNKNOWN )
        return;

    unsigned int elementSize = ::getElementSize( fmt );

    if( fmt == RT_FORMAT_USER && m_bufferSize.format() == RT_FORMAT_USER )
        // keep the size if the format is already RT_FORMAT_USER
        elementSize = m_bufferSize.elementSize();

    if( fmt == RT_FORMAT_BUFFER_ID )
    {
        // If our buffer is a buffer of buffer, there are some restrictions.
        if( flagsOn( m_iotype, RT_BUFFER_OUTPUT ) )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Cannot use format RT_FORMAT_BUFFER_ID with output buffer." );
        // TODO: review whether this restriction is really necessary
        if( isInterop() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Cannot use format RT_FORMAT_BUFFER_ID with interop buffer." );

        if( getMipLevelCount() > 1 )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Cannot use format RT_FORMAT_BUFFER_ID with multiple mip levels" );
    }

    if( fmt == RT_FORMAT_PROGRAM_ID )
    {
        if( getMipLevelCount() > 1 )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Cannot use format RT_FORMAT_PROGRAM_ID with multiple mip levels" );
        if( flagsOn( m_iotype, RT_BUFFER_OUTPUT ) )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Cannot use format RT_FORMAT_PROGRAM_ID with output buffer." );
        if( isInterop() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Cannot use format RT_FORMAT_PROGRAM_ID with interop buffer." );
        if( m_flags & RT_BUFFER_DISCARD_HOST_MEMORY )
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "Cannot use format RT_FORMAT_PROGRAM_ID with RT_BUFFER_DISCARD_HOST_MEMORY buffer." );
    }

    if( fmt == RT_FORMAT_UNKNOWN || ( elementSize == 0 && fmt != RT_FORMAT_USER ) )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Illegal buffer format: ", fmt );

    // Reject any format where elements straddle a page boundary in a demand buffer.
    // For the user format, we'll do the check again when they set the element size.
    if( isDemandLoad() )
    {
        if( elementSize > 16 )
        {
            // The gutter fetching logic assumes the element size is <= sizeof(float4).
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "Element size larger than 16 bytes not allowed for demand buffers.  ", fmt );
        }
        if( fmt != RT_FORMAT_USER && ( PagingService::PAGE_SIZE_IN_BYTES % elementSize != 0 ) )
        {
            throw IlwalidValue( RT_EXCEPTION_INFO, "Elements must evenly pack into page size for demand buffers.  ", fmt );
        }
    }

    m_bufferSize.setFormat( fmt, elementSize );

    // Notify any variables attached to this buffer that the size changed
    notifyParent_SizeOrFormatDidChange();

    // Change policy if necessary
    updateMBuffer();
}

void Buffer::getSize( size_t dims[3] ) const
{
    dims[0] = m_bufferSize.width();
    dims[1] = m_bufferSize.height();
    dims[2] = m_bufferSize.depth();
}

void Buffer::setSize( size_t dimensionality, size_t* dims )
{
    if( isMappedHost() )
        throw AlreadyMapped( RT_EXCEPTION_INFO, "Cannot resize buffer while mapped" );

    if( dimensionality < 1 || dimensionality > 3 )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Illegal buffer dimensionality: ", dimensionality );

    if( isDemandLoad() && dimensionality == 3 )
    {
        // 3D demand load buffers are not supported
        throw IlwalidValue( RT_EXCEPTION_INFO, "Illegal demand load buffer dimensionality: ", dimensionality );
    }

    if( flagsOn( m_flags, RT_BUFFER_LWBEMAP ) )
    {
        if( dims[0] != dims[1] )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Buffer height must be equal to width for lwbe buffer: ", dims[0], dims[1] );

        if( flagsOn( m_flags, RT_BUFFER_LAYERED ) )
        {
            if( dims[2] % 6 != 0 )
                throw IlwalidValue( RT_EXCEPTION_INFO,
                                    "Buffer depth must be a multiple of six for layered lwbe buffer: ", dims[2] );
        }
        else if( dims[2] != 6 )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Buffer depth must be six for lwbe buffer: ", dims[2] );
    }

    size_t w = dims[0];
    size_t h = dimensionality >= 2 ? dims[1] : 1;
    size_t d = dimensionality >= 3 ? dims[2] : 1;
    if( w == 0 )
        h = d = 0;

    BufferDimensions newBufferSize( m_bufferSize.format(), m_bufferSize.elementSize(), dimensionality, w, h, d,
                                    m_bufferSize.mipLevelCount(), flagsOn( m_flags, RT_BUFFER_LWBEMAP ),
                                    flagsOn( m_flags, RT_BUFFER_LAYERED ) );

    // If the new size of the buffer has changed then update the size and policy.
    if( newBufferSize != m_bufferSize || newBufferSize.zeroSized() )
    {
        if( m_mappedOnce && getFormat() == RT_FORMAT_PROGRAM_ID )
        {
            // The current content will be lost during reallocation.
            // Update the callsite mappings of the current content
            ScopedRawMapper buffer( m_context, m_mbuffer );
            int*            data = (int*)buffer.get();
            programIdBufferContentChanged( data, false );
            // mark as never have been mapped before, so it gets memset at next map.
            m_mappedOnce = false;
        }

        m_bufferSize = newBufferSize;

        // Notify any variables attached to this buffer that the size changed
        notifyParent_SizeOrFormatDidChange();

        // Validate the buffer
        subscribeForValidation();


        // Gotten pointers become invalid. Triggers a policy revert. Note that set pointers
        // remain valid -- we assume the user knows what he's doing and the pointer he set
        // either points to a large enough allocation, or he'll re-set it after the resize.
        if( m_clientVisiblePointers.count() > 0 && m_setPointers.empty() )
            m_clientVisiblePointers.clear();

        updateMBuffer();
        writeHeader();
    }
}

bool Buffer::empty() const
{
    return m_bufferSize.zeroSized();
}

bool Buffer::isElementSizeInitialized() const
{
    return m_bufferSize.elementSize() != 0;
}

MapMode Buffer::getMapModeForAPI( unsigned int apiMapFlags ) const
{
    // Check for nonsensical buffer type / map mode combinations.
    if( ( m_iotype == RT_BUFFER_OUTPUT && apiMapFlags == RT_BUFFER_MAP_WRITE ) ||  // clang-format fail
        ( m_iotype == RT_BUFFER_OUTPUT && apiMapFlags == RT_BUFFER_MAP_WRITE_DISCARD ) )
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid map mode for output buffer" );
    }

    // A read-write map request might come from an rtBufferMap (old-style without
    // map mode). This and the fact that it was the required mode for
    // rtBufferMapEx in Optix 3.9 means it needs to work on any buffer type,
    // including output-only.  We just map it to something a little more
    // reasonable here to make things clearer for the MemoryManager (though it
    // wouldn't trigger unnecessary copies anyway thanks to sync policies).
    if( m_iotype == RT_BUFFER_OUTPUT && apiMapFlags == RT_BUFFER_MAP_READ_WRITE )
    {
        return MAP_READ;
    }

    // Translate API value to internal map mode.  We can use a switch over a
    // "flags" value, because lwrrently they're all mutually exclusive.
    switch( apiMapFlags )
    {
        // clang-format off
        case RT_BUFFER_MAP_READ:              return MAP_READ;
        case RT_BUFFER_MAP_WRITE:             return MAP_READ_WRITE;
        case RT_BUFFER_MAP_READ_WRITE:        return MAP_READ_WRITE;
        case RT_BUFFER_MAP_WRITE_DISCARD:     return MAP_WRITE_DISCARD;
        // clang-format on

        default:
            break;
    }
    throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid combination of map flags" );
}

MBufferPolicy Buffer::determineBufferPolicy( bool gfxInterop, MBufferPolicy oldPolicy ) const
{
    // A tex heap policy overrides all other preferences
    if( m_flags & RT_BUFFER_INTERNAL_PREFER_TEX_HEAP )
    {
        // See the comment in markAsBindless.
        RT_ASSERT( isBindless() == false );
        return MBufferPolicy::internal_preferTexheap;  // Allocated in texture heap
    }

    // A demand load buffer has a special policy independent of other
    // factors.
    if( isDemandLoad() )
    {
        return hasTextureAttached() ? MBufferPolicy::texture_readonly_demandload : MBufferPolicy::readonly_demandload;
    }

    // Note: A graphics MBuffer for a texture ends up with the right
    // policy only after attaching to a texture sampler.


    // Determine the group required by the texture backing
    MBufferPolicy policy = MBufferPolicy::unused;
    if( hasTextureAttached() )
    {
        if( hasVariableBindings() )
        {
            // Used as both texture and buffer.  Disallow until we implement it (presumably with surfaces)
            throw IlwalidValue( RT_EXCEPTION_INFO, "Buffers used as both buffer and texture are not supported" );
        }

        RT_ASSERT( m_iotype == RT_BUFFER_INPUT );  // Should have been verified by API

        // "Fetch" functions like tex1Dfetch() require linear memory, at least lwrrently.
        // If fetch isn't used anywhere, we can use an array. Note that for bindless,
        // anyTextureUsesTexFetch will return false and we'll always use an array. In case
        // a texture is accessed through both fetch and non-fetch, use linear to allow SW
        // fallback.
        //
        // TODO: Kostya found that in a lot of cases, fetch actually does work with arrays.
        // Figure out if that's official and if it works on all SM versions we care about.

        if( !anyTextureUsesTexFetch() )
            policy = m_flags & RT_BUFFER_DISCARD_HOST_MEMORY ? MBufferPolicy::texture_array_discard_hostmem :
                                                               MBufferPolicy::texture_array;
        else
            policy = m_flags & RT_BUFFER_DISCARD_HOST_MEMORY ? MBufferPolicy::texture_linear_discard_hostmem :
                                                               MBufferPolicy::texture_linear;
    }
    else
    {
        // No texture uses
        if( gfxInterop )
        {
            // Graphics interop policy
            policy = MBufferPolicy::readonly_gfxInterop;
            if( m_iotype == RT_BUFFER_INPUT_OUTPUT )
                policy = MBufferPolicy::readwrite_gfxInterop;
            else if( m_iotype == RT_BUFFER_OUTPUT )
                policy = MBufferPolicy::writeonly_gfxInterop;
        }
        else if( hasVariableBindings() || isBindless() || m_clientVisiblePointers.count() > 0
                 || ( !m_usedInPostProcessingStage.empty() ) )
        {
            // Basic buffer type
            policy = m_flags & RT_BUFFER_DISCARD_HOST_MEMORY ? MBufferPolicy::readonly_discard_hostmem : MBufferPolicy::readonly;

            if( m_iotype == RT_BUFFER_INPUT_OUTPUT )
                policy = MBufferPolicy::readwrite;
            else if( m_iotype == RT_BUFFER_OUTPUT )
                policy = MBufferPolicy::writeonly;

            // Buffer constraints for special use-cases
            if( m_flags & RT_BUFFER_GPU_LOCAL )
                policy = MBufferPolicy::gpuLocal;  // Read-write on the GPU, persistent between launches

            ProgramManager* pm = m_context->getProgramManager();

            const bool copyOnDirty = flagsOn( m_flags, RT_BUFFER_COPY_ON_DIRTY );
            const bool lwdaInterop = m_clientVisiblePointers.count() > 0;

            bool bindlessRaw            = isBindless() && pm->hasRawBindlessBufferAccesses();
            bool hasRawVariableBindings = !m_rawAccess.empty();
            bool needsRawAccess         = bindlessRaw || hasRawVariableBindings;

            policy = translatePolicy( policy, needsRawAccess, lwdaInterop, copyOnDirty );
        }
        else
        {
            if( m_flags & RT_BUFFER_DISCARD_HOST_MEMORY )
            {
                if( getFormat() == RT_FORMAT_PROGRAM_ID )
                    throw IlwalidValue( RT_EXCEPTION_INFO,
                                        "RT_BUFFER_DISCARD_HOST_MEMORY is not supported for buffer format "
                                        "RT_FORMAT_PROGRAM_ID" );

                if( oldPolicy == MBufferPolicy::unused )
                {
                    // Buffers with the discard flag set are readonly_discard by default.
                    policy = MBufferPolicy::readonly_discard_hostmem;
                }
                else
                {
                    // No policy could be determined for a discard buffer. Keep the old policy
                    // to avoid unnecessary synchronizations.
                    policy = oldPolicy;
                }
            }
            else
            {
                // Buffer not used but we still need an allocation
                policy = MBufferPolicy::unused;  // Still allocated on host if mapped
            }
        }
    }

    return policy;
}

void Buffer::updateMBuffer()
{
    // During shutdown, we may no longer have an MBuffer. Ignore it.
    if( !m_mbuffer )
        return;

    MemoryManager* mm      = m_context->getMemoryManager();
    bool           changed = false;

    // Resize the buffer if necessary
    if( m_bufferSize != m_mbuffer->getDimensions() )
    {
        mm->changeSize( m_mbuffer, m_bufferSize );
        changed = true;
    }

    // Change the policy if necessary
    MBufferPolicy policy = determineBufferPolicy( isGfxInterop(), m_mbuffer->getPolicy() );
    if( policy != m_mbuffer->getPolicy() )
    {
        mm->changePolicy( m_mbuffer, policy );
        changed = true;
    }

    // Early exit for no changes
    if( !changed )
        return;

    unsigned int levelCount = getMipLevelCount();
    for( unsigned int level = 0; level < levelCount; ++level )
    {
        if( isMappedHost( level ) )
        {
            // Paranoia - verify that the host pointer has been preserved by
            // the memory manager.
            const CPUDevice* cpuDevice   = m_context->getDeviceManager()->cpuDevice();
            const MAccess&   memAccess   = m_mbuffer->getAccess( cpuDevice );
            void*            new_hostPtr = nullptr;
            if( memAccess.getKind() == MAccess::MULTI_PITCHED_LINEAR )
                new_hostPtr = memAccess.getPitchedLinear( level ).ptr;
            else
            {
                RT_ASSERT( level == 0 );
                new_hostPtr = memAccess.getLinearPtr();
            }
            RT_ASSERT( m_mappedHostPtrs[level] == new_hostPtr );
        }
    }

    // Update the buffer header
    writeHeader();

    // Notify attached textures of the update
    for( auto link : m_linkedPointers )
    {
        if( TextureSampler* sampler = getLinkToBufferFrom<TextureSampler>( link ) )
            sampler->mBufferWasChanged();
    }
}

// Attachment property
void Buffer::receivePropertyDidChange_Attachment( bool added )
{
    bool changed = m_attachment.addOrRemoveProperty( added );
    if( changed )
    {
        // Re-validate the texture if it is newly attached
        if( added )
            subscribeForValidation();
    }
}


// Callback from memory manager
void Buffer::eventMBufferMAccessDidChange( const MBuffer* buffer, const Device* device, const MAccess& oldMA, const MAccess& newMA )
{
    // We need to update the sampler's tile width after its sparse backing array has been created.
    if( hasTextureAttached() && newMA.getKind() == MAccess::LWDA_SPARSE )
        getAttachedTextureSampler()->writeHeader();

    m_context->getUpdateManager()->eventBufferMAccessDidChange( this, device, oldMA, newMA );
    if( isDemandLoad() )
        m_context->getPagingManager()->bufferMAccessDidChange( this, device, oldMA, newMA );
}

void Buffer::checkElementSizeInitialized() const
{
    if( m_bufferSize.elementSize() == 0 )
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "Buffer element size cannot be zero." );
    }
}

void Buffer::setElementSize( size_t elementSize )
{
    if( m_bufferSize.format() != RT_FORMAT_USER )
        throw TypeMismatch( RT_EXCEPTION_INFO, "Cannot set element size of non user-typed buffer" );

    if( elementSize == 0 )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Illegal zero element size for buffer" );

    if( isMappedHost() )
        throw AlreadyMapped( RT_EXCEPTION_INFO, "Cannot set buffer element size while mapped" );

    if( elementSize == m_bufferSize.elementSize() )
        return;

    // Reject any format where elements straddle a page boundary in a demand buffer
    if( isDemandLoad() )
    {
        if( elementSize > 16 )
        {
            // The gutter fetching logic assumes the element size is <= sizeof(float4).
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "Element size larger than 16 bytes not allowed for demand buffers." );
        }
        if( PagingService::PAGE_SIZE_IN_BYTES % elementSize != 0 )
        {
            throw IlwalidValue( RT_EXCEPTION_INFO, "Elements must evenly pack into page size for demand buffers." );
        }
    }

    m_bufferSize.setFormat( RT_FORMAT_USER, elementSize );
    // Notify any variables attached to this buffer that the size changed
    notifyParent_SizeOrFormatDidChange();

    // Update the allocation and policy
    updateMBuffer();
}

void Buffer::setMipLevelCount( unsigned int levels )
{
    if( levels == 0 )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Illegal zero MIP level count for buffer" );

    if( isMappedHost() )
        throw AlreadyMapped( RT_EXCEPTION_INFO, "Cannot change buffer mip level count while mapped" );

    if( levels == m_bufferSize.mipLevelCount() )
        return;

    if( !canHaveMipLevels() )
        throw IlwalidValue( RT_EXCEPTION_INFO,
                            "Cannot change buffer mip level count for format RT_FORMAT_BUFFER_ID and "
                            "RT_FORMAT_PROGRAM_ID" );

    m_mappedHostPtrs.resize( levels, nullptr );
    m_bufferSize.setMipLevelCount( levels );
    // Notify any variables attached to this buffer that the size changed
    notifyParent_SizeOrFormatDidChange();

    // Update the allocation and policy
    updateMBuffer();
}

bool Buffer::canHaveMipLevels() const
{
    RTformat format = getFormat();
    return format != RT_FORMAT_PROGRAM_ID && format != RT_FORMAT_BUFFER_ID;
}

bool Buffer::hasTextureAttached() const
{
    // Note: consider tracking this via a graph property counter should
    // performance becomes an issue
    for( auto link : m_linkedPointers )
    {
        if( getLinkToBufferFrom<TextureSampler>( link ) )
            return true;
    }
    return false;
}

TextureSampler* Buffer::getAttachedTextureSampler() const
{
    if( !isDemandLoad() )
    {
        throw IlwalidOperation( RT_EXCEPTION_INFO, "Can only get attached texture sampler for demand load buffers" );
    }

    for( auto link : m_linkedPointers )
    {
        if( TextureSampler* textureSampler = getLinkToBufferFrom<TextureSampler>( link ) )
        {
            return textureSampler;
        }
    }
    return nullptr;
}

bool Buffer::hasVariableBindings() const
{
    BindingManager*                                 bm       = m_context->getBindingManager();
    const BindingManager::IlwerseTextureBindingSet& bindings = bm->getIlwerseBindingsForBufferId( getId() );
    return !bindings.empty();
}

void Buffer::bufferBindingsDidChange( VariableReferenceID refid, bool added )
{
    // Raw Variable Reference: propagates from reference
    const VariableReference* varref = m_context->getProgramManager()->getVariableReferenceById( refid );
    if( varref->canPointerEscape() || varref->hasIllFormedAccess() )
        this->addOrRemoveProperty_rawAccess( added );

    if( m_mappedOnce && getFormat() == RT_FORMAT_PROGRAM_ID )
    {
        RT_ASSERT( getMipLevelCount() == 1 );
        programIdBufferVarBindingChanged( varref, added );
    }

    // Bindings changed, potentially reallocate buffer with new policy
    updateMBuffer();
}

void Buffer::programIdBufferContentChanged( void* bufPtr, bool added )
{
    RT_ASSERT( m_mappedOnce && getFormat() == RT_FORMAT_PROGRAM_ID );

    const ProgramManager* pm = m_context->getProgramManager();

    const BindingManager::IlwerseBufferBindingSet& bindings =
        m_context->getBindingManager()->getIlwerseBindingsForBufferId( getId() );
    for( VariableReferenceID refid : bindings )
    {
        const VariableReference* varref = m_context->getProgramManager()->getVariableReferenceById( refid );
        CallSiteIdentifier* csId = pm->getCallSiteByUniqueName( CallSiteIdentifier::generateCallSiteUniqueName( varref ) );
        if( csId )
        {
            updateProgramIdBufferCallsite( static_cast<int*>( bufPtr ), csId, added );
        }
    }
}

void Buffer::programIdBufferVarBindingChanged( const VariableReference* varref, bool added )
{
    RT_ASSERT( m_mappedOnce && getFormat() == RT_FORMAT_PROGRAM_ID );

    const ProgramManager* pm = m_context->getProgramManager();
    CallSiteIdentifier* csId = pm->getCallSiteByUniqueName( CallSiteIdentifier::generateCallSiteUniqueName( varref ) );

    if( csId )
    {
        ScopedRawMapper buffer( m_context, m_mbuffer );
        updateProgramIdBufferCallsite( (int*)buffer.get(), csId, added );
    }
}

void Buffer::updateProgramIdBufferCallsite( int* bufPtr, CallSiteIdentifier* csId, bool added )
{
    m_bindlessCallables.clear();

    const ObjectManager* om = m_context->getObjectManager();
    for( size_t i = 0, e = getWidth() * getHeight() * getDepth(); i < e; ++i )
    {
        int progId = bufPtr[i];
        // Using the no throw variant to get the program in case the buffer
        // contains invalid data, e.g. when it is only partially filled.
        Program* callee = om->getProgramByIdNoThrow( progId );
        if( callee != nullptr && callee->isBindless() )
        {
            if( added )
            {
                LinkedPtr<Buffer, Program> link;
                link.set( this, callee );
                m_bindlessCallables.emplace_back( std::move( link ) );
            }
            std::vector<CanonicalProgramID> callees = callee->getCanonicalProgramIDs();
            csId->addOrRemovePotentialCallees( callees, added );
        }
        else if( progId > 0 )
        {
            ureport2( m_context->getUsageReport(), "CALLABLE TRACKING" )
                << "Warning: Found invalid program ID in buffer: " << progId << ". Setting it to 0.\n";
            bufPtr[i] = 0;
        }
    }
}

void Buffer::attachedTextureDidChange()
{
    // Attached texture changed, potentially reallocate buffer with new policy
    updateMBuffer();
}

void Buffer::hasRawAccessDidChange()
{
    // The buffer switched between having zero and non-zero raw pointer accesses
    updateMBuffer();
}

bool Buffer::allTexturesUseTexFetch() const
{
    // Note: consider tracking this via a graph property counter should
    // performance becomes an issue
    for( auto link : m_linkedPointers )
    {
        if( TextureSampler* sampler = getLinkToBufferFrom<TextureSampler>( link ) )
            if( sampler->usesNonTexFetch() )
                return false;
    }
    return true;
}

bool Buffer::anyTextureUsesTexFetch() const
{
    // Determine if all of the attached textures use only texfetch
    // functions.  It would be possible to track these properties
    // incrementally but we expect to have a small number of texture
    // references per buffer

    // Note: consider tracking this via a graph property counter should
    // performance becomes an issue
    for( auto link : m_linkedPointers )
    {
        if( TextureSampler* sampler = getLinkToBufferFrom<TextureSampler>( link ) )
            if( sampler->usesTexFetch() )
                return true;
    }
    return false;
}

void* Buffer::map( MapMode mode, unsigned int level )
{
    if( m_mappedHostPtrs[level] )
        throw AlreadyMapped( RT_EXCEPTION_INFO, "Buffer is already mapped" );
    if( m_clientVisiblePointers.count() > 1 )
        throw IlwalidOperation( RT_EXCEPTION_INFO,
                                "Map not allowed on buffer with get/set pointers on multiple devices" );
    if( m_bufferSize.format() == RT_FORMAT_UNKNOWN )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Map not allowed on buffer with unset format." );

    if( m_mbuffer )
    {
        MemoryManager* mm = m_context->getMemoryManager();

        // assume continuous memory for all MIP levels
        // host interop with user owned memory per level may break this assumption
        if( mm->isMappedToHost( m_mbuffer ) )
        {
            m_mappedHostPtrs[level] =
                mm->getMappedToHostPtr( m_mbuffer ) + m_mbuffer->getDimensions().getLevelOffsetInBytes( level );
        }
        else
        {
            // Extend the mode for MIP buffers, in order to support the user mapping
            // different levels with different modes (e.g. during mipmap creation some
            // levels may want read while others may want write).
            if( getMipLevelCount() > 1 )
                mode                = MAP_READ_WRITE;
            m_mappedHostPtrs[level] = mm->mapToHost( m_mbuffer, mode ) + m_mbuffer->getDimensions().getLevelOffsetInBytes( level );
        }
    }
    else
    {
        m_mappedHostPtrs[level] = nullptr;
    }

    llog( 40 ) << "Buffer::map()\n"
               << "\t m_mapHostPtr[" << level << "] = " << m_mappedHostPtrs[level] << "\n";

    // Notify any parent post-processing stages about the map
    if( ( mode == MAP_READ_WRITE || mode == MAP_WRITE_DISCARD ) && ( !m_usedInPostProcessingStage.empty() ) )
    {
        for( auto parentLink : m_linkedPointers )
        {
            if( PostprocessingStage* stage = getLinkToBufferFrom<PostprocessingStage>( parentLink ) )
                stage->bufferWasMapped( parentLink );
        }
    }

    if( m_mappedHostPtrs[level] && getFormat() == RT_FORMAT_PROGRAM_ID )
    {
        RT_ASSERT( level == 0 );
        if( !m_mappedOnce )
        {
            // set program ID buffers to 0 on first map to avoid having potentially valid
            // IDs (maybe only in the future) in uninitialized data which would cause
            // problems regarding updates of the call sites
            memset( m_mappedHostPtrs[level], 0, ( getWidth() * getHeight() * getDepth() ) * sizeof( int ) );
        }
        else
        {
            programIdBufferContentChanged( m_mappedHostPtrs[level], false );
        }
    }

    m_mappedOnce = true;

    return m_mappedHostPtrs[level];
}

void Buffer::unmap( unsigned int level )
{
    if( !isMappedHost( level ) )
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "Buffer is not mapped." );
    }
    if( getFormat() == RT_FORMAT_PROGRAM_ID )
    {
        RT_ASSERT( level == 0 );
        programIdBufferContentChanged( m_mappedHostPtrs[level], true );
    }
    m_mappedHostPtrs[level] = nullptr;
    MemoryManager* mm       = m_context->getMemoryManager();
    if( !isMappedHost() )
        mm->unmapFromHost( m_mbuffer );
}

bool Buffer::isMappedHost() const
{
    for( auto ptr : m_mappedHostPtrs )
    {
        if( ptr )
        {
            return true;
        }
    }
    return false;
}

bool Buffer::isMappedHost( unsigned int level ) const
{
    return m_mappedHostPtrs[level];
}

void* Buffer::getMappedHostPtr( unsigned int level ) const
{
    return m_mappedHostPtrs[level];
}

bool Buffer::isInterop() const
{
    return isGfxInterop() || isLwdaInterop();
}

const GfxInteropResource& Buffer::getGfxInteropResource() const
{
    return m_mbuffer->getGfxInteropResource();
}

bool Buffer::isGfxInterop() const
{
    return m_mbuffer->getGfxInteropResource().kind != GfxInteropResource::NONE;
}

void Buffer::registerGfxInteropResource()
{
    RT_ASSERT( m_mbuffer );
    m_context->getMemoryManager()->registerGfxInteropResource( m_mbuffer );
}

void Buffer::unregisterGfxInteropResource()
{
    RT_ASSERT( m_mbuffer );
    m_context->getMemoryManager()->unregisterGfxInteropResource( m_mbuffer );
}

void Buffer::setInteropPointer( void* ptr, Device* device )
{
    if( ( m_iotype == RT_BUFFER_OUTPUT || m_iotype == RT_BUFFER_INPUT_OUTPUT ) && !( m_flags & RT_BUFFER_GPU_LOCAL ) )
        throw IlwalidOperation( RT_EXCEPTION_INFO,
                                "rtBufferSetPointer() not allowed for output or input/output buffers" );
    if( m_setPointers.count() != m_clientVisiblePointers.count() )  // some pointers have been gotten
        throw IlwalidOperation( RT_EXCEPTION_INFO,
                                "Cannot call rtBufferSetPointer() if rtBufferGetPointer() was first called on another "
                                "device." );

    // Mark this as an externally allocated pointer
    m_clientVisiblePointers |= DeviceSet( device );
    m_setPointers |= DeviceSet( device );

    // Allocate/update the buffer if necessary. Setting a pointer may cause a policy change.
    updateMBuffer();

    MemoryManager* mm = m_context->getMemoryManager();
    mm->setLwdaInteropPointer( m_mbuffer, ptr, device );
}

void* Buffer::getInteropPointer( Device* device )
{
    if( m_setPointers.count() != 0 && !m_setPointers.isSet( device ) )
        throw IlwalidOperation( RT_EXCEPTION_INFO,
                                "Calling rtBufferGetPointer() after rtBufferSetPointer() is only valid on devices "
                                "where "
                                "the pointer has previously been set." );

    m_clientVisiblePointers |= DeviceSet( device );

    // Allocate/update the buffer if necessary. Retrieving a pointer may cause a policy change.
    updateMBuffer();

    MemoryManager* mm = m_context->getMemoryManager();
    return mm->getLwdaInteropPointer( m_mbuffer, device );
}

bool Buffer::isLwdaInterop() const
{
    return m_clientVisiblePointers.count() > 0;
}

void Buffer::markDirty()
{
    if( m_clientVisiblePointers.count() == 0 )
        throw IlwalidOperation( RT_EXCEPTION_INFO,
                                "Must set or get buffer device pointer before calling rtBufferMarkDirty()." );
    if( m_clientVisiblePointers.count() > 1 )
        throw IlwalidOperation( RT_EXCEPTION_INFO,
                                "Mark dirty not allowed on buffers with get/set pointers on multiple devices" );
    if( !flagsOn( m_flags, RT_BUFFER_COPY_ON_DIRTY ) )
        throw IlwalidOperation( RT_EXCEPTION_INFO,
                                "Mark dirty only allowed on buffers created with RT_BUFFER_COPY_ON_DIRTY" );

    MemoryManager* mm = m_context->getMemoryManager();
    return mm->markDirtyLwdaInterop( m_mbuffer );
}

void Buffer::checkBufferType( const unsigned int type, bool isInteropBuffer )
{
    // Check basis type (input,output,input_output)
    if( onlyFlagBits( type, RT_BUFFER_INPUT_OUTPUT ) == 0 )
        throw IlwalidValue( RT_EXCEPTION_INFO, "The specified buffer type is not valid: ", type );

    // Is this a valid type including all flags that we support?
    // clang-format off
  unsigned int known_flags = (RT_BUFFER_INPUT_OUTPUT |
                              RT_BUFFER_GPU_LOCAL |
                              RT_BUFFER_COPY_ON_DIRTY |
                              RT_BUFFER_DISCARD_HOST_MEMORY |
                              RT_BUFFER_PARTITIONED_INTERNAL |
                              RT_BUFFER_DEVICE_ONLY_INTERNAL |
                              RT_BUFFER_PINNED_INTERNAL |
                              RT_BUFFER_WRITECOMBINED_INTERNAL |
                              RT_BUFFER_LAYERED |
                              RT_BUFFER_LWBEMAP |
                              RT_BUFFER_HINT_STATIC);
    // clang-format on

    if( maskOutFlags( type, known_flags ) != 0 )
        throw IlwalidValue( RT_EXCEPTION_INFO, "The specified buffer type is not valid: ", type );

    // Check if GPU_LOCAL is used correctly
    if( flagsOn( type, RT_BUFFER_GPU_LOCAL ) )
    {
        // Allowed buffer types are:
        // input_output -> scratchpad memory for the user with initialization

        // Interop buffers don't support RT_BUFFER_GPU_LOCAL
        if( isInteropBuffer )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Interop buffers do not support the RT_BUFFER_GPU_LOCAL flag: ", type );
        // Only input -> will not read back anyway -> flag is not needed
        if( ( type & RT_BUFFER_INPUT_OUTPUT ) == RT_BUFFER_INPUT )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Input buffers do not support the RT_BUFFER_GPU_LOCAL flag: ", type );
        // Only output -> device is in theory not allowed to read values and what is written will not be read back -> flag does not make sense
        if( ( type & RT_BUFFER_INPUT_OUTPUT ) == RT_BUFFER_OUTPUT )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Output buffers do not support the RT_BUFFER_GPU_LOCAL flag: ", type );
    }

    // Check if HINT_STATIC is used correctly
    if( flagsOn( type, RT_BUFFER_HINT_STATIC ) )
    {
        // Must be input type
        if( !flagsOn( type, RT_BUFFER_INPUT ) )
            throw IlwalidValue( RT_EXCEPTION_INFO, "RT_BUFFER_HINT_STATIC is only valid for input buffers" );

        // Disallow input/output.  Not because it wouldn't work, but it seems more likely
        // for people to misuse it than to benefit from it.
        if( flagsOn( type, RT_BUFFER_OUTPUT ) )
            throw IlwalidValue( RT_EXCEPTION_INFO, "RT_BUFFER_HINT_STATIC is only valid for input buffers" );
    }

    if( flagsOn( type, RT_BUFFER_DISCARD_HOST_MEMORY ) )
    {
        // Must be input type
        if( !flagsOn( type, RT_BUFFER_INPUT ) )
            throw IlwalidValue( RT_EXCEPTION_INFO, "RT_BUFFER_DISCARD_HOST_MEMORY is only valid for input buffers" );

        // Disallow input/output.
        if( flagsOn( type, RT_BUFFER_OUTPUT ) )
            throw IlwalidValue( RT_EXCEPTION_INFO, "RT_BUFFER_DISCARD_HOST_MEMORY is only valid for input buffers" );

        if( isInteropBuffer )
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "Interop buffers do not support the RT_BUFFER_DISCARD_HOST_MEMORY flag: ", type );

        if( flagsOn( type, RT_BUFFER_GPU_LOCAL ) )
            throw IlwalidValue( RT_EXCEPTION_INFO, "RT_BUFFER_DISCARD_HOST_MEMORY is invalid for GPU local buffers" );
    }
}

BufferDimensions Buffer::getDimensions() const
{
    return m_bufferSize;
}

MBufferHandle Buffer::getMBuffer() const
{
    return m_mbuffer;
}

void Buffer::writeHeader()
{
    if( !m_id )
        return;  // This can happen during construction of the object

    // If it's a demand-loaded buffer, ensure that the page size is up to date, since it depends on
    // the dimensionality and element size.
    uint3 pageDims = m_context->getPagingManager()->getPageDimensions( getDimensions() );
    m_pageWidth    = pageDims.x;
    m_pageHeight   = pageDims.y;
    m_pageDepth    = pageDims.z;

    m_context->getTableManager()->writeBufferHeader( *m_id, getWidth(), getHeight(), getDepth(), getPageWidth(),
                                                     getPageHeight(), getPageDepth() );
}

int Buffer::getId() const
{
    RT_ASSERT( m_id != nullptr );
    return *m_id;
}

int Buffer::getAPIId()
{
    // The side effect of getting the ID through the API is that it is marked bindless
    markAsBindless();
    return getId();
}

// The reason we have this function instead of just using getAPIId() is to be able to
// distinguish between when the API requires the buffer to be bindless and when we need to
// mark it as bindless for internal use.
void Buffer::markAsBindlessForInternalUse()
{
    markAsBindless();
}

void Buffer::switchLwdaSparseArrayToLwdaArray( DeviceSet devices ) const
{
    m_mbuffer->switchLwdaSparseArrayToLwdaArray( devices );
}

void Buffer::markAsBindless()
{
    // At the moment texheap is used only for the BVH.
    // The runtime function that access buffers require a stack allocation of 16 Bytes.
    // This is used only when the buffer lives on the texheap.
    // We want to avoid this allocation in as many cases as possible, since bindless buffers don't use
    // the texheap for now we make this impossible, thus removing the need for allocation when
    // accessing bindless buffers.
    RT_ASSERT_MSG( m_mbuffer->getPolicy() != MBufferPolicy::internal_preferTexheap,
                   "Bindless buffers cannot live in the texheap" );

    if( m_isBindless )
        return;

    m_isBindless = true;

    updateMBuffer();
    writeHeader();
}

bool Buffer::isBindless() const
{
    return m_isBindless;
}

void Buffer::subscribeForValidation()
{
    m_context->getValidationManager()->subscribeForValidation( this );
}

void Buffer::addOrRemovePostprocessingStage( bool added )
{
    bool changed = m_usedInPostProcessingStage.addOrRemoveProperty( added );

    // The policy might need to change since a stage is now using this buffer.
    if( changed )
        updateMBuffer();
}

}  // namespace optix
