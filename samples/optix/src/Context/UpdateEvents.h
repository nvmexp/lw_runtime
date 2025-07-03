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


// This file gets included by UpdateManager.h to implement the methods
// in UpdateManger, UpdateEventInterface and UpdateEventNOP.  Enter
// the event here and it will be declared in all three classes.

// Context
virtual void eventActiveDevicesWillChange( const DeviceArray& oldDevices, const DeviceArray& newDevices ) EVT_PURE;
virtual void eventContextSetStackSize( size_t oldSize, size_t newSize ) EVT_PURE;
virtual void eventContextSetAttributeStackSize( size_t oldContinuationStackSize,
                                                size_t oldDirectCallableStackSizeFromTraversal,
                                                size_t oldDirectCallableStackSizeFromState,
                                                size_t newContinuationStackSize,
                                                size_t newDirectCallableStackSizeFromTraversal,
                                                size_t newDirectStackStizeFromState ) EVT_PURE;
virtual void eventContextSetMaxCallableProgramDepth( unsigned int oldMaxDepth, unsigned int newMaxDepth ) EVT_PURE;
virtual void eventContextSetMaxTraceDepth( unsigned int oldMaxDepth, unsigned int newMaxDepth ) EVT_PURE;
virtual void eventContextSetEntryPointCount( unsigned int oldCount, unsigned int newCount ) EVT_PURE;
virtual void eventContextSetRayTypeCount( unsigned int oldCount, unsigned int newCount ) EVT_PURE;
virtual void eventContextSetExceptionFlags( uint64_t oldFlags, uint64_t newFlags ) EVT_PURE;
virtual void eventContextSetPrinting( bool        oldEnabled,
                                      size_t      oldBufferSize,
                                      const int3& oldLaunchIndex,
                                      bool        newEnabled,
                                      size_t      newBufferSize,
                                      const int3& newLaunchIndex ) EVT_PURE;
virtual void eventContextSetPreferFastRecompiles( bool oldValue, bool newValue ) EVT_PURE;
virtual void eventContextMaxTransformDepthChanged( int oldDepth, int newDepth ) EVT_PURE;
virtual void eventContextMaxAccelerationHeightChanged( int oldValue, int newValue ) EVT_PURE;
virtual void eventContextHasMotionTransformsChanged( bool newValue ) EVT_PURE;
virtual void eventContextNeedsUniversalTraversalChanged( bool newValue ) EVT_PURE;
virtual void eventContextHasMotionBlurChanged( bool newValue ) EVT_PURE;

// Bindings (BindingManager)
virtual void eventVariableBindingsDidChange( VariableReferenceID refid, const VariableReferenceBinding& binding, bool added ) EVT_PURE;
virtual void eventBufferBindingsDidChange( VariableReferenceID refid, int bufferid, bool added ) EVT_PURE;
virtual void eventTextureBindingsDidChange( VariableReferenceID refid, int texid, bool added ) EVT_PURE;

// Global Scope
virtual void eventGlobalScopeRayGenerationProgramDidChange( unsigned int index, Program* oldProgram, Program* newProgram ) EVT_PURE;
virtual void eventGlobalScopeExceptionProgramDidChange( unsigned int index, Program* oldProgram, Program* newProgram ) EVT_PURE;

// CanonicalProgram
virtual void eventCanonicalProgramCreate( const CanonicalProgram* cp ) EVT_PURE;
virtual void eventCanonicalProgramSemanticTypeDidChange( const CanonicalProgram* cp, SemanticType stype, bool added ) EVT_PURE;
virtual void eventCanonicalProgramInheritedSemanticTypeDidChange( const CanonicalProgram* cp,
                                                                  SemanticType            stype,
                                                                  SemanticType            inheritedStype,
                                                                  bool                    added ) EVT_PURE;
virtual void eventCanonicalProgramUsedByRayTypeDidChange( const CanonicalProgram* cp, unsigned int stype, bool added ) EVT_PURE;
virtual void eventCanonicalProgramDirectCallerDidChange( const CanonicalProgram* cp, CanonicalProgramID caller_cpid, bool added ) EVT_PURE;
virtual void eventCanonicalProgramPotentialCalleesDidChange( const CanonicalProgram* cp, const CallSiteIdentifier* cs, bool added ) EVT_PURE;

// Buffer
virtual void eventBufferMAccessDidChange( const Buffer* buffer, const Device* device, const MAccess& oldMA, const MAccess& newMA ) EVT_PURE;

// TextureSampler
virtual void eventTextureSamplerMAccessDidChange( const TextureSampler* sampler,
                                                  const Device*         device,
                                                  const MAccess&        oldMA,
                                                  const MAccess&        newMA ) EVT_PURE;

// TableManager
virtual void eventTableManagerObjectRecordResized( size_t oldSize, size_t newSize ) EVT_PURE;
virtual void eventTableManagerBufferHeaderTableResized( size_t oldSize, size_t newSize ) EVT_PURE;
virtual void eventTableManagerProgramHeaderTableResized( size_t oldSize, size_t newSize ) EVT_PURE;
virtual void eventTableManagerTextureHeaderTableResized( size_t oldSize, size_t newSize ) EVT_PURE;
virtual void eventTableManagerTraversableHeaderTableResized( size_t oldSize, size_t newSize ) EVT_PURE;

// Demand loading
virtual void eventActiveDevicesSupportLwdaSparseTexturesDidChange( bool newValue ) EVT_PURE;
virtual void eventPagingModeDidChange( PagingMode newValue ) EVT_PURE;

// Nuclear event - should always ilwalidate plans
virtual void eventNuclear() EVT_PURE;
