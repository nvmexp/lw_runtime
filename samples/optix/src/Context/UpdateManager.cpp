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

#include <Context/UpdateManager.h>

#include <Context/Context.h>

#include <prodlib/exceptions/Assert.h>

#include <algorithm>

using namespace optix;

UpdateManager::UpdateManager( Context* context )
    : m_context( context )
{
}

UpdateManager::~UpdateManager()
{
}

void UpdateManager::registerUpdateListener( UpdateEventListener* listener )
{
    m_listeners.push_back( listener );
}

void UpdateManager::unregisterUpdateListener( UpdateEventListener* listener )
{
    m_listeners.erase( std::remove( m_listeners.begin(), m_listeners.end(), listener ), m_listeners.end() );
}

// Context
void UpdateManager::eventActiveDevicesWillChange( const DeviceArray& oldDevices, const DeviceArray& newDevices )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventActiveDevicesWillChange( oldDevices, newDevices );
}

void UpdateManager::eventContextSetStackSize( const size_t oldSize, const size_t newSize )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventContextSetStackSize( oldSize, newSize );
}

void UpdateManager::eventContextSetAttributeStackSize( const size_t oldContinuationStackSize,
                                                       const size_t oldDirectCallableStackSizeFromTraversal,
                                                       const size_t oldDirectCallableStackSizeFromState,
                                                       const size_t newContinuationStackSize,
                                                       const size_t newDirectCallableStackSizeFromTraversal,
                                                       const size_t newDirectCallableStackSizeFromState )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventContextSetAttributeStackSize( oldContinuationStackSize, oldDirectCallableStackSizeFromTraversal,
                                                     oldDirectCallableStackSizeFromState, newContinuationStackSize,
                                                     newDirectCallableStackSizeFromTraversal, newDirectCallableStackSizeFromState );
}

void UpdateManager::eventContextSetMaxCallableProgramDepth( const unsigned int oldMaxDepth, const unsigned int newMaxDepth )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventContextSetMaxCallableProgramDepth( oldMaxDepth, newMaxDepth );
}

void UpdateManager::eventContextSetMaxTraceDepth( const unsigned int oldMaxDepth, const unsigned int newMaxDepth )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventContextSetMaxTraceDepth( oldMaxDepth, newMaxDepth );
}

void UpdateManager::eventContextSetEntryPointCount( unsigned int oldCount, unsigned int newCount )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventContextSetEntryPointCount( oldCount, newCount );
}

void UpdateManager::eventContextSetRayTypeCount( unsigned int oldCount, unsigned int newCount )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventContextSetRayTypeCount( oldCount, newCount );
}

void UpdateManager::eventContextSetExceptionFlags( const uint64_t oldFlags, const uint64_t newFlags )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventContextSetExceptionFlags( oldFlags, newFlags );
}

void UpdateManager::eventContextSetPrinting( bool        oldEnabled,
                                             size_t      oldBufferSize,
                                             const int3& oldLaunchIndex,
                                             bool        newEnabled,
                                             size_t      newBufferSize,
                                             const int3& newLaunchIndex )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventContextSetPrinting( oldEnabled, oldBufferSize, oldLaunchIndex, newEnabled, newBufferSize, newLaunchIndex );
}

void UpdateManager::eventContextSetPreferFastRecompiles( bool oldValue, bool newValue )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventContextSetPreferFastRecompiles( oldValue, newValue );
}

void UpdateManager::eventContextMaxTransformDepthChanged( int oldDepth, int newDepth )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventContextMaxTransformDepthChanged( oldDepth, newDepth );
}

void UpdateManager::eventContextMaxAccelerationHeightChanged( int oldValue, int newValue )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventContextMaxAccelerationHeightChanged( oldValue, newValue );
}

void UpdateManager::eventContextHasMotionTransformsChanged( bool newValue )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventContextHasMotionTransformsChanged( newValue );
}

void UpdateManager::eventContextNeedsUniversalTraversalChanged( bool newValue )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventContextNeedsUniversalTraversalChanged( newValue );
}

void UpdateManager::eventContextHasMotionBlurChanged( bool newValue )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventContextHasMotionBlurChanged( newValue );
}

// Bindings

void UpdateManager::eventVariableBindingsDidChange( VariableReferenceID refid, const VariableReferenceBinding& binding, bool added )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventVariableBindingsDidChange( refid, binding, added );
}

void UpdateManager::eventBufferBindingsDidChange( VariableReferenceID refid, int binding, bool added )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventBufferBindingsDidChange( refid, binding, added );
}

void UpdateManager::eventTextureBindingsDidChange( VariableReferenceID refid, int binding, bool added )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventTextureBindingsDidChange( refid, binding, added );
}

// Global Scope
void UpdateManager::eventGlobalScopeRayGenerationProgramDidChange( unsigned int index, Program* oldProgram, Program* newProgram )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventGlobalScopeRayGenerationProgramDidChange( index, oldProgram, newProgram );
}

void UpdateManager::eventGlobalScopeExceptionProgramDidChange( unsigned int index, Program* oldProgram, Program* newProgram )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventGlobalScopeExceptionProgramDidChange( index, oldProgram, newProgram );
}

// CanonicalProgram
void UpdateManager::eventCanonicalProgramCreate( const CanonicalProgram* cp )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventCanonicalProgramCreate( cp );
}

void UpdateManager::eventCanonicalProgramSemanticTypeDidChange( const CanonicalProgram* cp, SemanticType stype, bool added )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventCanonicalProgramSemanticTypeDidChange( cp, stype, added );
}

void UpdateManager::eventCanonicalProgramInheritedSemanticTypeDidChange( const CanonicalProgram* cp,
                                                                         SemanticType            stype,
                                                                         SemanticType            inheritedStype,
                                                                         bool                    added )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventCanonicalProgramInheritedSemanticTypeDidChange( cp, stype, inheritedStype, added );
}

void UpdateManager::eventCanonicalProgramUsedByRayTypeDidChange( const CanonicalProgram* cp, unsigned int rayType, bool added )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventCanonicalProgramUsedByRayTypeDidChange( cp, rayType, added );
}

void UpdateManager::eventCanonicalProgramDirectCallerDidChange( const CanonicalProgram* cp, CanonicalProgramID cpid, bool added )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventCanonicalProgramDirectCallerDidChange( cp, cpid, added );
}

void UpdateManager::eventCanonicalProgramPotentialCalleesDidChange( const CanonicalProgram* cp, const CallSiteIdentifier* cs, bool added )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventCanonicalProgramPotentialCalleesDidChange( cp, cs, added );
}

// Buffer

void UpdateManager::eventBufferMAccessDidChange( const Buffer* buffer, const Device* device, const MAccess& oldMA, const MAccess& newMA )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventBufferMAccessDidChange( buffer, device, oldMA, newMA );
}

// TextureSampler

void UpdateManager::eventTextureSamplerMAccessDidChange( const TextureSampler* sampler,
                                                         const Device*         device,
                                                         const MAccess&        oldMA,
                                                         const MAccess&        newMA )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventTextureSamplerMAccessDidChange( sampler, device, oldMA, newMA );
}

// TableManager
void UpdateManager::eventTableManagerObjectRecordResized( size_t oldSize, size_t newSize )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventTableManagerObjectRecordResized( oldSize, newSize );
}

void UpdateManager::eventTableManagerBufferHeaderTableResized( size_t oldSize, size_t newSize )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventTableManagerBufferHeaderTableResized( oldSize, newSize );
}

void UpdateManager::eventTableManagerProgramHeaderTableResized( size_t oldSize, size_t newSize )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventTableManagerProgramHeaderTableResized( oldSize, newSize );
}

void UpdateManager::eventTableManagerTextureHeaderTableResized( size_t oldSize, size_t newSize )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventTableManagerTextureHeaderTableResized( oldSize, newSize );
}

void UpdateManager::eventTableManagerTraversableHeaderTableResized( size_t oldSize, size_t newSize )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventTableManagerTraversableHeaderTableResized( oldSize, newSize );
}

void UpdateManager::eventActiveDevicesSupportLwdaSparseTexturesDidChange( bool newValue )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventActiveDevicesSupportLwdaSparseTexturesDidChange( newValue );
}

void UpdateManager::eventPagingModeDidChange( PagingMode newValue )
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventPagingModeDidChange( newValue );
}

// Nuclear event - should always ilwalidate plans
void UpdateManager::eventNuclear()
{
    for( UpdateEventListener* listener : m_listeners )
        listener->eventNuclear();
}

/*
 * No-op implementation of events
 */
void UpdateEventListenerNop::eventActiveDevicesWillChange( const DeviceArray& oldDevices, const DeviceArray& newDevices )
{
}

void UpdateEventListenerNop::eventContextSetStackSize( size_t oldSize, size_t newSize )
{
}

void UpdateEventListenerNop::eventContextSetAttributeStackSize( size_t oldContinuationStackSize,
                                                                size_t oldDirectCallableStackSizeFromTraversal,
                                                                size_t oldDirectCallableStackSizeFromState,
                                                                size_t newContinuationStackSize,
                                                                size_t newDirectCallableStackSizeFromTraversal,
                                                                size_t newDirectCallableStackSizeFromState )
{
}

void UpdateEventListenerNop::eventContextSetMaxCallableProgramDepth( unsigned int oldMaxDepth, unsigned int newMaxDepth )
{
}

void UpdateEventListenerNop::eventContextSetMaxTraceDepth( unsigned int oldMaxDepth, unsigned int newMaxDepth )
{
}

void UpdateEventListenerNop::eventContextSetEntryPointCount( unsigned int oldCount, unsigned int newCount )
{
}

void UpdateEventListenerNop::eventContextSetRayTypeCount( unsigned int oldCount, unsigned int newCount )
{
}

void UpdateEventListenerNop::eventContextSetExceptionFlags( const uint64_t oldFlags, uint64_t newFlags )
{
}

void UpdateEventListenerNop::eventContextSetPrinting( bool        oldEnabled,
                                                      size_t      oldBufferSize,
                                                      const int3& oldLaunchIndex,
                                                      bool        newEnabled,
                                                      size_t      newBufferSize,
                                                      const int3& newLaunchIndex )
{
}

void UpdateEventListenerNop::eventContextSetPreferFastRecompiles( bool oldValue, bool newValue )
{
}

void UpdateEventListenerNop::eventContextMaxTransformDepthChanged( int oldDepth, int newDepth )
{
}

void UpdateEventListenerNop::eventContextMaxAccelerationHeightChanged( int oldValue, int newValue )
{
}

void UpdateEventListenerNop::eventContextHasMotionTransformsChanged( bool newValue )
{
}

void UpdateEventListenerNop::eventContextNeedsUniversalTraversalChanged( bool newValue )
{
}

void UpdateEventListenerNop::eventContextHasMotionBlurChanged( bool newValue )
{
}

// Bindings
void UpdateEventListenerNop::eventVariableBindingsDidChange( VariableReferenceID refid, const VariableReferenceBinding& binding, bool added )
{
}

void UpdateEventListenerNop::eventBufferBindingsDidChange( VariableReferenceID refid, int bufid, bool added )
{
}

void UpdateEventListenerNop::eventTextureBindingsDidChange( VariableReferenceID refid, int texid, bool added )
{
}

// Global Scope
void UpdateEventListenerNop::eventGlobalScopeRayGenerationProgramDidChange( unsigned int index, Program* oldProgram, Program* newProgram )
{
}

void UpdateEventListenerNop::eventGlobalScopeExceptionProgramDidChange( unsigned int index, Program* oldProgram, Program* newProgram )
{
}

// CanonicalProgram
void UpdateEventListenerNop::eventCanonicalProgramCreate( const CanonicalProgram* cp )
{
}

void UpdateEventListenerNop::eventCanonicalProgramSemanticTypeDidChange( const CanonicalProgram* cp, SemanticType stype, bool added )
{
}

void UpdateEventListenerNop::eventCanonicalProgramInheritedSemanticTypeDidChange( const CanonicalProgram* cp,
                                                                                  SemanticType            stype,
                                                                                  SemanticType inheritedStype,
                                                                                  bool         added )
{
}

void UpdateEventListenerNop::eventCanonicalProgramUsedByRayTypeDidChange( const CanonicalProgram* cp, unsigned int rayType, bool added )
{
}

void UpdateEventListenerNop::eventCanonicalProgramDirectCallerDidChange( const CanonicalProgram* cp, CanonicalProgramID cpid, bool added )
{
}

void UpdateEventListenerNop::eventCanonicalProgramPotentialCalleesDidChange( const CanonicalProgram*   cp,
                                                                             const CallSiteIdentifier* cs,
                                                                             bool                      added )
{
}

// Buffer
void UpdateEventListenerNop::eventBufferMAccessDidChange( const Buffer* buffer, const Device* device, const MAccess& oldMA, const MAccess& newMA )
{
}

// TextureSampler
void UpdateEventListenerNop::eventTextureSamplerMAccessDidChange( const TextureSampler* sampler,
                                                                  const Device*         device,
                                                                  const MAccess&        oldMA,
                                                                  const MAccess&        newMA )
{
}

// Table manager (including object records)
void UpdateEventListenerNop::eventTableManagerObjectRecordResized( size_t oldSize, size_t newSize )
{
}

void UpdateEventListenerNop::eventTableManagerBufferHeaderTableResized( size_t oldSize, size_t newSize )
{
}

void UpdateEventListenerNop::eventTableManagerProgramHeaderTableResized( size_t oldSize, size_t newSize )
{
}

void UpdateEventListenerNop::eventTableManagerTextureHeaderTableResized( size_t oldSize, size_t newSize )
{
}

void UpdateEventListenerNop::eventTableManagerTraversableHeaderTableResized( size_t oldSize, size_t newSize )
{
}

// Demand loading
void UpdateEventListenerNop::eventActiveDevicesSupportLwdaSparseTexturesDidChange( bool newValue ) 
{
}

void UpdateEventListenerNop::eventPagingModeDidChange( PagingMode newValue ) 
{
}

// Nuclear event - should always ilwalidate plans
void UpdateEventListenerNop::eventNuclear()
{
}
