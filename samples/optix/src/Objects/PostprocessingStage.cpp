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

#include <Objects/PostprocessingStage.h>

#include <LWCA/ErrorCheck.h>
#include <LWCA/Function.h>
#include <LWCA/Module.h>
#include <LWCA/Stream.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/misc/TimeViz.h>

#include <corelib/system/LwdaDriver.h>

#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>

#include <Objects/CommandList.h>
#include <Objects/Variable.h>

#include <ExelwtionStrategy/CORTTypes.h>

#include <Util/LinkedPtrHelpers.h>

#include <corelib/system/System.h>

#include <lwda_runtime.h>


namespace optix {

PostprocessingStage::PostprocessingStage( Context* context, const std::string& builtin_name )
    : LexicalScope( context, RT_OBJECT_POSTPROCESSINGSTAGE )
{
    m_name    = std::string( "POST " ) + builtin_name;
    m_stageId = m_context->getObjectManager()->registerObject( this );
}

PostprocessingStage::~PostprocessingStage()
{
}

void PostprocessingStage::launch( RTsize width, RTsize height )
{
    TIMEVIZ_SCOPE( m_name.c_str() );
    doLaunch( width, height );
}

void PostprocessingStage::validate() const
{
    LexicalScope::validate();
}

void PostprocessingStage::detachFromParents()
{
    auto iter = m_linkedPointers.begin();
    while( iter != m_linkedPointers.end() )
    {
        LinkedPtr_Link* parentLink = *iter;

        // Parents can be command lists only
        if( CommandList* list = getLinkToPostprocessingStageFrom<CommandList>( parentLink ) )
            list->detachLinkedChild( parentLink );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to PostprocessingStage" );

        iter = m_linkedPointers.begin();
    }
}

void PostprocessingStage::detachLinkedChild( const LinkedPtr_Link* link )
{
}

void PostprocessingStage::bufferWasMapped( LinkedPtr_Link* link )
{
}

void PostprocessingStage::bufferFormatDidChange( LinkedPtr_Link* link )
{
}

void PostprocessingStage::bufferVariableValueDidChange( Variable* var, Buffer* oldBuffer, Buffer* newBuffer )
{
    if( oldBuffer )
        oldBuffer->addOrRemovePostprocessingStage( false );
    if( newBuffer )
        newBuffer->addOrRemovePostprocessingStage( true );
}

void* PostprocessingStage::copyToGpuIfNeeded( MBufferHandle& buffer, void* devicePtr, LWDADevice* device )
{
    if( devicePtr == nullptr )
        return nullptr;

    if( !buffer->isZeroCopy( device->allDeviceListIndex() ) )
        return devicePtr;

    void*       gpuOutDevPtr = devicePtr;
    lwdaError_t error        = lwdaMalloc( (void**)&gpuOutDevPtr, buffer->getDimensions().getTotalSizeInBytes() );
    if( error != lwdaSuccess )
        return devicePtr;

    error = lwdaMemcpy( gpuOutDevPtr, devicePtr, buffer->getDimensions().getTotalSizeInBytes(), lwdaMemcpyDeviceToDevice );
    if( error != lwdaSuccess )
    {
        lwdaFree( gpuOutDevPtr );
        return devicePtr;
    }

    return gpuOutDevPtr;
}

size_t PostprocessingStage::getRecordBaseSize() const
{
    return 0;
}

void PostprocessingStage::reallocateRecord()
{
}

void PostprocessingStage::notifyParents_offsetDidChange() const
{
}

void PostprocessingStage::sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool addedToUnresolvedSet ) const
{
}

void PostprocessingStage::sendPropertyDidChange_Attachment( bool added ) const
{
}

void PostprocessingStage::sendPropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const
{
}
}
