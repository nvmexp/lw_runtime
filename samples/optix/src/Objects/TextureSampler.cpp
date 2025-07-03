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

#include <Objects/TextureSampler.h>

#include <Context/BindingManager.h>
#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Context/ProgramManager.h>
#include <Context/TableManager.h>
#include <Context/UpdateManager.h>
#include <Context/ValidationManager.h>
#include <Device/DeviceManager.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <Memory/DemandLoad/PagingService.h>
#include <Memory/MResources.h>
#include <Memory/MemoryManager.h>
#include <Util/LinkedPtrHelpers.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidOperation.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/ValidationError.h>
#include <prodlib/misc/RTFormatUtil.h>
#include <prodlib/system/Knobs.h>

using namespace prodlib;
using namespace corelib;

namespace optix {

TextureSampler::TextureSampler( Context* context )
    : ManagedObject( context, RT_OBJECT_TEXTURE_SAMPLER )
    , m_isBindless( false )
{
    m_id = m_context->getObjectManager()->registerObject( this );
    subscribeForValidation();
}

TextureSampler::TextureSampler( Context* context, const GfxInteropResource& resource, Device* interopDevice )
    : ManagedObject( context, RT_OBJECT_TEXTURE_SAMPLER )
    , m_isBindless( false )
    , m_isInterop( true )
{
    m_id = m_context->getObjectManager()->registerObject( this );
    subscribeForValidation();

    try
    {
        // This could be the first API call that requires a device.
        m_context->getDeviceManager()->enableActiveDevices();  // Do this avoid assertion on LWDADevice::makeLwrrent()  SGP: I would rather push this into the API for every create function

        // Allocate the low-level backing ourselves
        MemoryManager* mm     = m_context->getMemoryManager();
        MBufferPolicy  policy = MBufferPolicy::texture_gfxInterop;
        m_backing             = mm->allocateMBuffer( resource, interopDevice, policy );

        // And create the texture sampler object
        reallocateTextureSampler();
    }
    catch( ... )
    {
        // If an exception was thrown (say in allocateMBuffer when the Gfx resource's format
        // isn't supported), we need to clean up after ourselves.  Note there might be other
        // instances of objects that would need to be cleaned up.
        if( m_msampler )
            m_msampler->removeListener( this );
        if( m_buffer && m_buffer->isDemandLoad() )
            m_buffer->getMBuffer()->removeListener( this );
        m_context->getValidationManager()->unsubscribeForValidation( this );
        throw;
    }
}

TextureSampler::~TextureSampler()
{
    if( m_msampler )
    {
        m_msampler->removeListener( this );
        m_msampler->releaseVirtualPages( m_context->getPagingManager() );
    }
    if( m_buffer && m_buffer->isDemandLoad() )
        m_buffer->getMBuffer()->removeListener( this );
    m_context->getValidationManager()->unsubscribeForValidation( this );
    RT_ASSERT_MSG( m_validationIndex == -1, "Failed to remove object from validation list" );

    m_context->getTableManager()->clearTextureHeader( *m_id );
}

void TextureSampler::detachFromParents()
{
    auto iter = m_linkedPointers.begin();
    while( iter != m_linkedPointers.end() )
    {
        LinkedPtr_Link* parentLink = *iter;

        if( Variable* variable = getLinkToTextureSamplerFrom<Variable>( parentLink ) )
            variable->detachLinkedChild( parentLink );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to TextureSampler" );

        iter = m_linkedPointers.begin();
    }
}

bool TextureSampler::isAttached() const
{
    return !m_attachment.empty();
}

void TextureSampler::validate() const
{
    // Attachment isn't always required to perform validation.  rtBufferValidate doesn't require
    // attachment, but ValidationManager does.  Do the attachment check in ValidationManager.

    if( m_backing == nullptr )
        throw ValidationError( RT_EXCEPTION_INFO, "No buffers assigned to texture" );

    const BufferDimensions& dim = m_backing->getDimensions();
    if( !isSupportedTextureFormat( dim.format() ) )
        throw ValidationError( RT_EXCEPTION_INFO, "The specified texture format is not supported." );

    if( !isInteropTexture() )
    {
        Buffer* buffer = getBuffer();
        if( buffer == nullptr )
            throw ValidationError( RT_EXCEPTION_INFO, "No buffers assigned to texture" );

        // validate buffer properties
        if( buffer->getType() & RT_BUFFER_OUTPUT )
            throw ValidationError( RT_EXCEPTION_INFO, "Cannot attach output buffers to TextureSamplers" );
    }

    m_textureDescriptor.validate();
}

void TextureSampler::detachLinkedChild( const LinkedPtr_Link* link )
{
    if( link == &m_buffer )
        setBuffer( nullptr );

    else
        RT_ASSERT_FAIL_MSG( "Invalid child link in detachLinkedChild" );
}

void TextureSampler::setBuffer( Buffer* newBuffer )
{
    RT_ASSERT_MSG( !isInteropTexture(), "Cannot set newBuffer for interop texture" );

    // Demand loading does not support attaching multiple samplers to the same demand loaded buffer.
    // We need to get the sampler from the buffer (via Buffer::getAttachedTextureSampler()) in
    // DevicePaging::copyTileToDevice in order to choose a tile pool that matches the filter and
    // wrap modes, etc.
    if( newBuffer && newBuffer->isDemandLoad() && newBuffer->hasTextureAttached() )
    {
        // TODO: the following exception breaks the Arnold bathroom scene.
        // throw IlwalidOperation( RT_EXCEPTION_INFO, "Demand buffer already attached to a texture sampler");
    }

    Buffer* oldBuffer = m_buffer.get();
    if( oldBuffer == newBuffer )
        return;

    if( oldBuffer )
    {
        if( oldBuffer->isDemandLoad() )
        {
            oldBuffer->getMBuffer()->removeListener( this );
            if( m_msampler )
                m_msampler->releaseVirtualPages( m_context->getPagingManager() );
        }
        if( isAttached() )
            oldBuffer->receivePropertyDidChange_Attachment( false );
    }

    // Attach the newBuffer
    m_buffer.set( this, newBuffer );

    if( newBuffer )
    {
        if( isAttached() )
            newBuffer->receivePropertyDidChange_Attachment( true );
        if( newBuffer->isDemandLoad() )
        {
            newBuffer->getMBuffer()->addListener( this );
            if( m_msampler )
                m_msampler->releaseVirtualPages( m_context->getPagingManager() );
        }
    }

    // Notify the variables attached to this samppler that they might
    // have a new format.
    if( !oldBuffer || !newBuffer || oldBuffer->getDimensions() != newBuffer->getDimensions() )
        notifyVariables_FormatDidChange();

    // Notify old and new newBuffer to reallocate if necessary
    if( oldBuffer )
        oldBuffer->attachedTextureDidChange();
    if( newBuffer )
        newBuffer->attachedTextureDidChange();

    reallocateTextureSampler();
    writeHeader();
}

Buffer* TextureSampler::getBuffer() const
{
    if( isInteropTexture() )
        throw IlwalidValue( RT_EXCEPTION_INFO,
                            "A buffer cannot be queried from an RTtexturesampler when it has been created from a "
                            "interop "
                            "texture." );

    return m_buffer.get();
}

// Attachment property
void TextureSampler::receivePropertyDidChange_Attachment( bool added )
{
    bool changed = m_attachment.addOrRemoveProperty( added );
    if( changed )
    {
        // Re-validate the texture if it is newly attached
        if( added )
            subscribeForValidation();

        if( m_buffer )
            m_buffer->receivePropertyDidChange_Attachment( added );
    }
}

unsigned int TextureSampler::getGutterWidth() const
{
    return std::max( 1U, static_cast<unsigned int>( std::ceil( getMaxAnisotropy() / 2 ) ) );
}

unsigned int TextureSampler::getTileWidth() const
{
    return m_context->getPagingManager()->getTileWidth();
}

unsigned int TextureSampler::getTileHeight() const
{
    return m_context->getPagingManager()->getTileHeight();
}

// Callback from memory manager
void TextureSampler::eventMTextureSamplerMAccessDidChange( const Device* device, const MAccess& oldMA, const MAccess& newMA )
{
    m_context->getUpdateManager()->eventTextureSamplerMAccessDidChange( this, device, oldMA, newMA );
}

void TextureSampler::eventMBufferMAccessDidChange( const MBuffer* buffer, const Device* device, const MAccess& oldMBA, const MAccess& newMBA )
{
    // Record the virtual page information for the buffer associated with the mip tail.
    if( oldMBA.getKind() != MAccess::DEMAND_LOAD_ARRAY && newMBA.getKind() == MAccess::DEMAND_LOAD_ARRAY )
    {
        const DemandLoadArrayAccess& access = newMBA.getDemandLoadArray();
        m_context->getTableManager()->writeDemandTextureDeviceHeader( getId(), access.virtualPageBegin, access.numPages,
                                                                      access.minMipLevel );
    }

    if( oldMBA.getKind() != MAccess::LWDA_SPARSE && newMBA.getKind() == MAccess::LWDA_SPARSE )
    {
        writeHeader();
        const LwdaSparseAccess& access = newMBA.getLwdaSparse();
        m_context->getTableManager()->writeDemandTextureDeviceHeader( getId(), access.virtualPageBegin, 0, 0 );
    }
}

void TextureSampler::notifyVariables_FormatDidChange()
{
    for( auto parentLink : m_linkedPointers )
    {
        if( Variable* variable = getLinkToTextureSamplerFrom<Variable>( parentLink ) )
            variable->textureSamplerFormatDidChange();
        else
            RT_ASSERT_FAIL_MSG( std::string( "Invalid parent link to TextureSampler: " ) + typeid( *parentLink ).name() );
    }
}

void TextureSampler::bufferFormatDidChange()
{
    // Size of the attached buffer changed.
    subscribeForValidation();
    notifyVariables_FormatDidChange();
    // Do not reallocate here - wait for the callback that the mbuffer changed
}

RTformat TextureSampler::getBufferFormat() const
{
    return m_backing ? m_backing->getDimensions().format() : RT_FORMAT_UNKNOWN;
}

unsigned int TextureSampler::getDimensionality() const
{
    return m_backing ? m_backing->getDimensions().dimensionality() : 0;
}

int TextureSampler::getId() const
{
    RT_ASSERT( m_id != nullptr );
    return *m_id;
}

int TextureSampler::getAPIId()
{
    // The side effect of getting the ID through the API is that it is marked bindless
    markAsBindless();
    return getId();
}

void TextureSampler::markAsBindless()
{
    if( m_isBindless )
        return;

    m_isBindless = true;
    writeHeader();
}

bool TextureSampler::isBindless() const
{
    return m_isBindless;
}

void TextureSampler::subscribeForValidation()
{
    m_context->getValidationManager()->subscribeForValidation( this );
}

// TODO: consider using RT_FORMAT directly instead of this mapping.  Alternatively, move this to Util/RTFormatUtil.h
static int getInternalTexFormat( RTformat format )
{
    switch( format )
    {
        // TODO implement SW fallback for half textures and remove this HALF->SHORT hack
        case RT_FORMAT_HALF:
            return (int)cort::TEX_FORMAT_SHORT1;
        case RT_FORMAT_HALF2:
            return (int)cort::TEX_FORMAT_SHORT2;
        case RT_FORMAT_HALF4:
            return (int)cort::TEX_FORMAT_SHORT4;

        case RT_FORMAT_FLOAT:
            return (int)cort::TEX_FORMAT_FLOAT1;
        case RT_FORMAT_FLOAT2:
            return (int)cort::TEX_FORMAT_FLOAT2;
        case RT_FORMAT_FLOAT4:
            return (int)cort::TEX_FORMAT_FLOAT1;

        case RT_FORMAT_BYTE:
            return (int)cort::TEX_FORMAT_BYTE1;
        case RT_FORMAT_BYTE2:
            return (int)cort::TEX_FORMAT_BYTE2;
        case RT_FORMAT_BYTE4:
            return (int)cort::TEX_FORMAT_BYTE4;

        case RT_FORMAT_UNSIGNED_BYTE:
            return (int)cort::TEX_FORMAT_UNSIGNED_BYTE1;
        case RT_FORMAT_UNSIGNED_BYTE2:
            return (int)cort::TEX_FORMAT_UNSIGNED_BYTE2;
        case RT_FORMAT_UNSIGNED_BYTE4:
            return (int)cort::TEX_FORMAT_UNSIGNED_BYTE4;

        case RT_FORMAT_SHORT:
            return (int)cort::TEX_FORMAT_SHORT1;
        case RT_FORMAT_SHORT2:
            return (int)cort::TEX_FORMAT_SHORT2;
        case RT_FORMAT_SHORT4:
            return (int)cort::TEX_FORMAT_SHORT4;

        case RT_FORMAT_UNSIGNED_SHORT:
            return (int)cort::TEX_FORMAT_UNSIGNED_SHORT1;
        case RT_FORMAT_UNSIGNED_SHORT2:
            return (int)cort::TEX_FORMAT_UNSIGNED_SHORT2;
        case RT_FORMAT_UNSIGNED_SHORT4:
            return (int)cort::TEX_FORMAT_UNSIGNED_SHORT4;

        case RT_FORMAT_INT:
            return (int)cort::TEX_FORMAT_INT1;
        case RT_FORMAT_INT2:
            return (int)cort::TEX_FORMAT_INT2;
        case RT_FORMAT_INT4:
            return (int)cort::TEX_FORMAT_INT4;

        case RT_FORMAT_UNSIGNED_INT:
            return (int)cort::TEX_FORMAT_UNSIGNED_INT1;
        case RT_FORMAT_UNSIGNED_INT2:
            return (int)cort::TEX_FORMAT_UNSIGNED_INT2;
        case RT_FORMAT_UNSIGNED_INT4:
            return (int)cort::TEX_FORMAT_UNSIGNED_INT4;

        // Note: Compressed formats are not supported by the cort software pipeline.
        // They would also require a new encoding (see cort::InternalTexFormat).
        case RT_FORMAT_UNSIGNED_BC1:
        case RT_FORMAT_UNSIGNED_BC4:
        case RT_FORMAT_BC4:
            return cort::TEX_FORMAT_UNSIGNED_INT2;

        case RT_FORMAT_UNSIGNED_BC2:
        case RT_FORMAT_UNSIGNED_BC3:
        case RT_FORMAT_UNSIGNED_BC5:
        case RT_FORMAT_BC5:
        case RT_FORMAT_UNSIGNED_BC6H:
        case RT_FORMAT_BC6H:
        case RT_FORMAT_UNSIGNED_BC7:
            return cort::TEX_FORMAT_UNSIGNED_INT4;

        default:
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Unsupported texture format: ", format );
    }
}

const MTextureSamplerHandle& TextureSampler::getMTextureSampler() const
{
    return m_msampler;
}

void TextureSampler::updateDescriptor()
{
    subscribeForValidation();
    writeHeader();
    if( m_msampler )
        m_context->getMemoryManager()->updateTextureDescriptor( m_msampler, m_textureDescriptor );
}

static bool isPowerOfTwo( unsigned int x )
{
    return ( x & ( x - 1 ) ) == 0;
}

void TextureSampler::writeHeader() const
{
    if( !m_id )
        return;  // This can happen during construction of the object

    if( !m_backing )
    {
        m_context->getTableManager()->clearTextureHeader( *m_id );
    }
    else
    {
        BufferDimensions dims       = m_backing->getDimensions();
        RTformat         buffer_fmt = dims.format();

        if( buffer_fmt == RT_FORMAT_USER )
        {
            // In pre-Goldenrod, we just changed the format to FLOAT4 here in order to make texheap work.
            // That hack is no longer needed, so we can properly disallow user-format.
            throw IlwalidValue( RT_EXCEPTION_INFO, "Texture samplers with RT_FORMAT_USER are not supported" );
        }

        bool         isDemandLoad  = m_buffer ? m_buffer->isDemandLoad() : false;
        unsigned int mipLevelCount = m_buffer ? m_buffer->getMipLevelCount() : 0;

        m_context->getTableManager()->writeTextureHeader(
            *m_id, (int)dims.width(), (int)dims.height(), (int)dims.depth(), mipLevelCount, getInternalTexFormat( buffer_fmt ),
            (int)m_textureDescriptor.wrapMode[0], (int)m_textureDescriptor.wrapMode[1], (int)m_textureDescriptor.wrapMode[2],
            (int)( m_textureDescriptor.indexMode == RT_TEXTURE_INDEX_NORMALIZED_COORDINATES ),
            (int)( m_textureDescriptor.magFilterMode == RT_FILTER_LINEAR ),
            (int)( m_textureDescriptor.readMode == RT_TEXTURE_READ_NORMALIZED_FLOAT ), isDemandLoad );
        if( isDemandLoad )
        {
            float        ilwAnisotropy     = 1.f / m_textureDescriptor.maxAnisotropy;
            unsigned int tileWidth         = 0;
            unsigned int tileHeight        = 0;
            unsigned int tileGutterWidth   = 0;
            unsigned int mipTailFirstLevel = 0;
            if( m_context->getPagingManager()->getLwrrentPagingMode() == PagingMode::SOFTWARE_SPARSE )
            {
                tileWidth         = getTileWidth();
                tileHeight        = getTileHeight();
                mipTailFirstLevel = m_buffer->getMipTailFirstLevel();
                tileGutterWidth   = std::max( 1U, static_cast<unsigned int>( m_textureDescriptor.maxAnisotropy / 2 ) );
            }
            else if( m_context->getPagingManager()->getLwrrentPagingMode() == PagingMode::LWDA_SPARSE_HYBRID
                     || m_context->getPagingManager()->getLwrrentPagingMode() == PagingMode::LWDA_SPARSE_HARDWARE )
            {
                LWDA_ARRAY_SPARSE_PROPERTIES props = m_backing->getSparseTextureProperties();
                tileWidth                          = props.tileExtent.width;
                tileHeight                         = props.tileExtent.height;
                mipTailFirstLevel                  = props.miptailFirstLevel;
            }
            unsigned int isInitialized = 0;
            unsigned int isSquarePowerOfTwo =
                dims.width() == dims.height() && isPowerOfTwo( dims.width() ) && isPowerOfTwo( dims.height() );
            unsigned int mipmapFilterMode = ( m_textureDescriptor.mipFilterMode == RT_FILTER_LINEAR ) ? 1 : 0;
            m_context->getTableManager()->writeDemandTextureHeader( *m_id, mipTailFirstLevel, ilwAnisotropy, tileWidth,
                                                                    tileHeight, tileGutterWidth, isInitialized,
                                                                    isSquarePowerOfTwo, mipmapFilterMode );
        }
    }
}

void TextureSampler::mBufferWasChanged()
{
    notifyVariables_FormatDidChange();
    reallocateTextureSampler();
}

void TextureSampler::reallocateTextureSampler()
{
    // Share the backing store with the attached buffer (if any)
    if( m_buffer )
        m_backing = m_buffer->getMBuffer();
    else if( !m_isInterop )
        m_backing.reset();

    // Revalidate since we changed the backing
    subscribeForValidation();

    // Reallocate the texture sampler from the attached buffer and
    // trigger events
    MTextureSamplerHandle new_msampler;
    if( m_backing )
        new_msampler = getContext()->getMemoryManager()->attachMTextureSampler( m_backing, m_textureDescriptor, this );
    m_msampler       = new_msampler;

    // Rewrite the header
    writeHeader();
}

bool TextureSampler::usesTexFetch() const
{
    // This can be made more efficient by tracking, but some
    // optimization of the lookupkind representation could help
    // considerably.
    BindingManager* bm = m_context->getBindingManager();
    ProgramManager* pm = m_context->getProgramManager();
    for( const auto& refid : bm->getIlwerseBindingsForTextureId( getId() ) )
    {
        const VariableReference* vref = pm->getVariableReferenceById( refid );
        using namespace TextureLookup;
        for( LookupKind kind = beginLookupKind(); kind != endLookupKind(); kind = nextEnum( kind ) )
        {
            if( vref->usesTextureLookupKind( kind ) && isLookupTexfetch( kind ) )
                return true;
        }
    }
    return false;
}

bool TextureSampler::usesNonTexFetch() const
{
    // This can be made more efficient by tracking, but some
    // optimization of the lookupkind representation could help
    // considerably.
    BindingManager* bm = m_context->getBindingManager();
    ProgramManager* pm = m_context->getProgramManager();
    for( const auto& refid : bm->getIlwerseBindingsForTextureId( getId() ) )
    {
        const VariableReference* vref = pm->getVariableReferenceById( refid );
        using namespace TextureLookup;
        for( LookupKind kind = beginLookupKind(); kind != endLookupKind(); kind = nextEnum( kind ) )
        {
            if( vref->usesTextureLookupKind( kind ) && !isLookupTexfetch( kind ) )
                return true;
        }
    }
    return false;
}

void TextureSampler::textureBindingsDidChange()
{
    if( m_buffer )
        m_buffer->attachedTextureDidChange();
}

void TextureSampler::setReadMode( RTtexturereadmode readmode )
{
    if( readmode != RT_TEXTURE_READ_ELEMENT_TYPE && readmode != RT_TEXTURE_READ_NORMALIZED_FLOAT
        && readmode != RT_TEXTURE_READ_ELEMENT_TYPE_SRGB && readmode != RT_TEXTURE_READ_NORMALIZED_FLOAT_SRGB )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid read mode for TextureSampler" );

    if( m_textureDescriptor.readMode == readmode )
        return;
    m_textureDescriptor.readMode = readmode;
    updateDescriptor();
}

RTtexturereadmode TextureSampler::getReadMode() const
{
    return m_textureDescriptor.readMode;
}

void TextureSampler::setIndexMode( RTtextureindexmode indexmode )
{
    if( indexmode != RT_TEXTURE_INDEX_NORMALIZED_COORDINATES && indexmode != RT_TEXTURE_INDEX_ARRAY_INDEX )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid indexing mode for TextureSampler" );

    if( m_textureDescriptor.indexMode == indexmode )
        return;
    m_textureDescriptor.indexMode = indexmode;
    updateDescriptor();
}

RTtextureindexmode TextureSampler::getIndexMode() const
{
    return m_textureDescriptor.indexMode;
}


void TextureSampler::getFilterModes( RTfiltermode& minFilter, RTfiltermode& magFilter, RTfiltermode& mipFilter ) const
{
    minFilter = m_textureDescriptor.minFilterMode;
    magFilter = m_textureDescriptor.magFilterMode;
    mipFilter = m_textureDescriptor.mipFilterMode;
}

void TextureSampler::setFilterModes( RTfiltermode minFilter, RTfiltermode magFilter, RTfiltermode mipFilter )
{
    if( minFilter != RT_FILTER_NEAREST && minFilter != RT_FILTER_LINEAR )
        throw IlwalidValue( RT_EXCEPTION_INFO, "TextureSampler minification filter is invalid" );
    if( magFilter != RT_FILTER_NEAREST && magFilter != RT_FILTER_LINEAR )
        throw IlwalidValue( RT_EXCEPTION_INFO, "TextureSampler magnification filter is invalid" );
    if( mipFilter != RT_FILTER_NEAREST && mipFilter != RT_FILTER_LINEAR && mipFilter != RT_FILTER_NONE )
        throw IlwalidValue( RT_EXCEPTION_INFO, "TextureSampler mipmapping filter is invalid" );

    // clang-format off
    if( m_textureDescriptor.minFilterMode == minFilter &&
        m_textureDescriptor.magFilterMode == magFilter &&
        m_textureDescriptor.mipFilterMode == mipFilter )
        return;
    // clang-format on

    m_textureDescriptor.minFilterMode = minFilter;
    m_textureDescriptor.magFilterMode = magFilter;
    m_textureDescriptor.mipFilterMode = mipFilter;
    updateDescriptor();
}

RTwrapmode TextureSampler::getWrapMode( unsigned int dim ) const
{
    RT_ASSERT( dim < 3 );
    return m_textureDescriptor.wrapMode[dim];
}

void TextureSampler::getWrapModes( RTwrapmode& dim0, RTwrapmode& dim1, RTwrapmode& dim2 ) const
{
    dim0 = m_textureDescriptor.wrapMode[0];
    dim1 = m_textureDescriptor.wrapMode[1];
    dim2 = m_textureDescriptor.wrapMode[2];
}

void TextureSampler::setWrapMode( unsigned int dim, RTwrapmode mode )
{
    if( mode != RT_WRAP_REPEAT && mode != RT_WRAP_CLAMP_TO_EDGE && mode != RT_WRAP_MIRROR && mode != RT_WRAP_CLAMP_TO_BORDER )
        throw IlwalidValue( RT_EXCEPTION_INFO, "TextureSampler wrap mode is invalid" );

    if( dim >= 3 )
        throw IlwalidValue( RT_EXCEPTION_INFO, "TextureSampler wrap mode dimensionality is invalid" );

    if( m_textureDescriptor.wrapMode[dim] == mode )
        return;
    m_textureDescriptor.wrapMode[dim] = mode;
    updateDescriptor();
    if( m_msampler )
        m_msampler->releaseVirtualPages( m_context->getPagingManager() );
}

float TextureSampler::getMaxAnisotropy() const
{
    return m_textureDescriptor.maxAnisotropy;
}

void TextureSampler::setMaxAnisotropy( float maxAniso )
{
    if( m_textureDescriptor.maxAnisotropy == maxAniso )
        return;
    m_textureDescriptor.maxAnisotropy = maxAniso;
    updateDescriptor();
}


float TextureSampler::getMaxMipLevelClamp() const
{
    return m_textureDescriptor.maxMipLevelClamp;
}

void TextureSampler::setMaxMipLevelClamp( float maxLevel )
{
    if( m_textureDescriptor.maxMipLevelClamp == maxLevel )
        return;
    m_textureDescriptor.maxMipLevelClamp = maxLevel;
    updateDescriptor();
}

float optix::TextureSampler::getMinMipLevelClamp() const
{
    return m_textureDescriptor.minMipLevelClamp;
}

void TextureSampler::setMinMipLevelClamp( float minLevel )
{
    if( m_textureDescriptor.minMipLevelClamp == minLevel )
        return;
    m_textureDescriptor.minMipLevelClamp = minLevel;
    updateDescriptor();
}

float optix::TextureSampler::getMipLevelBias() const
{
    return m_textureDescriptor.mipLevelBias;
}

void TextureSampler::setMipLevelBias( float bias )
{
    if( m_textureDescriptor.mipLevelBias == bias )
        return;
    m_textureDescriptor.mipLevelBias = bias;
    updateDescriptor();
}

bool TextureSampler::isInteropTexture() const
{
    // We will not have an attached buffer in the case of interop
    return m_isInterop;
}

MBufferHandle TextureSampler::getBackingMBuffer() const
{
    return m_backing;
}

void TextureSampler::registerGfxInteropResource()
{
    if( !isInteropTexture() )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Not an interop texture" );

    RT_ASSERT( m_backing );
    m_context->getMemoryManager()->registerGfxInteropResource( m_backing );
    reallocateTextureSampler();
}

void TextureSampler::unregisterGfxInteropResource()
{
    if( !isInteropTexture() )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Not an interop texture" );

    RT_ASSERT( m_backing );
    m_context->getMemoryManager()->unregisterGfxInteropResource( m_backing );
}

const GfxInteropResource& TextureSampler::getGfxInteropResource() const
{
    return m_backing->getGfxInteropResource();
}

}  // namespace optix
