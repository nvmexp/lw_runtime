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

#include <ExelwtionStrategy/Common/ConstantMemoryPlan.h>
#include <ExelwtionStrategy/Common/ProgramPlan.h>
#include <ExelwtionStrategy/Common/SpecializationPlan.h>
#include <ExelwtionStrategy/Plan.h>
#include <ExelwtionStrategy/RTX/CompiledProgramCache.h>
#include <ExelwtionStrategy/RTX/RTXCompile.h>
#include <FrontEnd/Canonical/CanonicalProgramID.h>
#include <ThreadPool/Job.h>
#include <prodlib/compiler/ModuleCache.h>

#include <mutex>
#include <vector>

namespace optix {

class CanonicalProgram;
class LWDADevice;
class RTXES;

class RTXPlan : public Plan
{
  public:
    RTXPlan( Context* context, CompiledProgramCache* compiledProgramCache, const DeviceSet& devices, int numLaunchDevices );

    //------------------------------------------------------------------------
    // Plan interface
    //------------------------------------------------------------------------

    std::string summaryString() const override;
    bool supportsLaunchConfiguration( unsigned int entry, int dimensionality, const DeviceSet& devices, int numLaunchDevices ) const override;
    bool isCompatibleWith( const Plan* otherPlan ) const override;
    void compile() const override;
    void createPlan( unsigned int entry, int dimensionality );

    // Returns the RTX exception flags (taking knobs into account).
    static unsigned int getRtxExceptionFlags( Context* context );
    // Returns the rtcore exception flags (taking knobs into account).
    static unsigned int getRtcoreExceptionFlags( unsigned int rtxExceptionFlags );

    // Returns true if the combination of stype and inheritedStype describe
    // a bound callable program that is not called as a continuation call.
    static bool isDirectCalledBoundCallable( SemanticType stype, SemanticType inheritedStype );

  private:
    //------------------------------------------------------------------------
    // Potentially ilwalidating events
    //------------------------------------------------------------------------
    // Note: others are handled in specialization plan and program plan
    void eventContextSetAttributeStackSize( size_t oldContinuationStackSize,
                                            size_t oldDirectCallableStackSizeFromTraversal,
                                            size_t oldDirectCallableStackSizeFromState,
                                            size_t newContinuationStackSize,
                                            size_t newDirectStackSizeFromTraversal,
                                            size_t newDirectStackSizeFromState ) override;
    void eventContextSetMaxCallableProgramDepth( unsigned int oldMaxDepth, unsigned int newMaxDepth ) override;
    void eventContextSetMaxTraceDepth( unsigned int oldMaxDepth, unsigned int newMaxDepth ) override;
    void eventContextMaxTransformDepthChanged( int oldDepth, int newDepth ) override;
    void eventContextMaxAccelerationHeightChanged( int oldValue, int newValue ) override;
    void eventContextNeedsUniversalTraversalChanged( bool needsUniversalTraversal ) override;
    void eventContextHasMotionBlurChanged( bool needsMotionBlur ) override;
    void eventPagingModeDidChange( PagingMode newMode ) override;

    //------------------------------------------------------------------------
    // Helper functions
    //------------------------------------------------------------------------
    struct PerUniqueDevice;
    void computeAttributeDataSizes( const std::set<CanonicalProgramID>& isectPrograms,
                                    int                                 maxAttributeRegisterCount,
                                    int&                                registerCount,
                                    int&                                memoryCount ) const;

    int getMaxNumCallableProgramRegisters( const CanonicalProgramID cpID, bool isBound ) const;

    void determineRtxCompileOptions( const LWDADevice*          device,
                                     const PerUniqueDevice&     pud,
                                     RtcCompileOptions*         rtcoreOptions,
                                     RTXCompile::CompileParams* rtxParams ) const;

  public:
    typedef std::vector<const CanonicalProgram*> AttributeDecoderList;

  private:
    /// \param mutex    The mutex is locked when the method is called and is expected to be locked
    ///                 when the method returns. The method may unlock it temporarily for
    ///                 sections that are thread-safe.
    ModuleEntryRefPair getOrCompileProgram( const LWDADevice*                          device,
                                            const RTXCompile::CompileParams&           rtxParams,
                                            const RtcCompileOptions&                   options,
                                            const AttributeDecoderList&                attributeDecoders,
                                            const std::set<const CallSiteIdentifier*>& heavyweightCallSites,
                                            int                                        numConlwrrentLaunchDevices,
                                            PagingMode                                 pagingMode,
                                            SemanticType                               stype,
                                            SemanticType                               inheritedStype,
                                            const CanonicalProgram*                    cp,
                                            std::mutex&                                mutex ) const;

    /// \param mutex    The mutex is locked when the method is called and is expected to be locked
    ///                 when the method returns. The method may unlock it temporarily for
    ///                 sections that are thread-safe.
    ModuleEntryRefPair compileProgramToRTX( const RTXCompile::Options& options, std::mutex& mutex ) const;

    bool loadModuleFromDiskCache( PersistentStream* stream, ModuleEntryRefPair& cachedModule, const CompiledProgramCacheKey& cacheKey ) const;

    void saveModuleToDiskCache( PersistentStream* stream, ModuleEntryRefPair* compiledModule, const CompiledProgramCacheKey& cacheKey ) const;

    //------------------------------------------------------------------------
    // Device independent plan
    //------------------------------------------------------------------------

    unsigned int                          m_entry                                = ~0U;
    int                                   m_dimensionality                       = 0;
    bool                                  m_useContextAttributesForStackSize     = false;
    size_t                                m_continuationStackSize                = 0;
    size_t                                m_directCallableStackSizeFromTraversal = 0;
    size_t                                m_directCallableStackSizeFromState     = 0;
    unsigned int                          m_maxCallableProgramDepth              = 0;
    unsigned int                          m_maxTraceDepth                        = 0;
    unsigned int                          m_maxTransformHeight                   = 0;
    unsigned int                          m_maxAccelerationHeight                = 0;
    std::unique_ptr<SpecializationPlan>   m_specializationPlan;
    std::unique_ptr<prodlib::ModuleCache> m_moduleCache;
    int                                   m_numLaunchDevices        = 0;
    bool                                  m_needsUniversalTraversal = false;
    bool                                  m_hasMotionBlur           = false;
    PagingMode                            m_pagingMode              = PagingMode::UNKNOWN;

    //------------------------------------------------------------------------
    // Device dependent plan
    //------------------------------------------------------------------------
    struct PerUniqueDevice
    {
        PerUniqueDevice() {}
        PerUniqueDevice( PerUniqueDevice&& other );

        LWDADevice*                  m_archetype        = nullptr;
        int                          m_numBoundTextures = -1;
        std::unique_ptr<ProgramPlan> m_programPlan;
        bool                         m_canPromotePayload         = true;
        int                          m_maxPayloadRegisterCount   = 0;
        int                          m_maxAttributeRegisterCount = 0;
        int                          m_maxAttributeMemoryCount   = 0;
        AttributeDecoderList         m_attributeDecoders;
        int                          m_maxCallableProgramParamRegisterCount = 0;

        bool isCompatibleWith( const PerUniqueDevice& other ) const;
    };

    std::vector<PerUniqueDevice>        m_perUniqueDevice;  // indexed by unique device idx
    std::unique_ptr<ConstantMemoryPlan> m_constantMemoryPlan;
#if RTCORE_API_VERSION >= 25
    ModuleEntryRefPair deduplicateRtcModule( const LWDADevice* device, RtcCompiledModule newModule, Rtlw32 entryIndex ) const;
#endif

    //------------------------------------------------------------------------
    // Compiled program cache (passed from RTXES)
    //------------------------------------------------------------------------
    CompiledProgramCache* m_compiledProgramCache = nullptr;

    //------------------------------------------------------------------------
    // Job to parallelize the invocation of getOrCompileProgram()
    //------------------------------------------------------------------------
    class GetOrCompileProgramJob : public FragmentedJob
    {
      public:
        struct InputData
        {
            CanonicalProgramID  cpId;
            optix::SemanticType stype;
            optix::SemanticType inheritedStype;
            int                 numConlwrrentLaunchDevices;
            PagingMode          pagingMode;
        };

        GetOrCompileProgramJob( bool                                       parallelize,
                                const std::vector<InputData>&              input,
                                const RTXPlan&                             rtxPlan,
                                const Context*                             context,
                                LWDADevice*                                lwdaDevice,
                                const RTXCompile::CompileParams&           rtxParams,
                                const RtcCompileOptions&                   rtcCompileOptions,
                                const AttributeDecoderList&                attributeDecoders,
                                int                                        m_maxAttributeRegisterCount,
                                const std::set<const CallSiteIdentifier*>& heavyweightCallSites );

        void exelwteFragment( size_t index, size_t count ) noexcept override;

        size_t getThreadLimit() const override { return m_threadLimit; }

        // Returns the output of the i-th fragment or rethrows the corresponding exception.
        const ModuleEntryRefPair& getOutput( size_t index );

      private:
        const std::vector<InputData>&   m_input;
        std::vector<ModuleEntryRefPair> m_output;
        std::vector<std::exception_ptr> m_exception;

        size_t                                     m_threadLimit;
        const RTXPlan&                             m_rtxPlan;
        const Context*                             m_context;
        LWDADevice*                                m_lwdaDevice;
        const RTXCompile::CompileParams&           m_rtxParams;
        const RtcCompileOptions&                   m_rtcCompileOptions;
        const AttributeDecoderList&                m_attributeDecoders;
        int                                        m_maxAttributeRegisterCount;
        const std::set<const CallSiteIdentifier*>& m_heavyweightCallSites;

        // Used to serialize some portions of the code run by exelwteFragment() that are not yet thread-safe.
        std::mutex m_mutex;
    };
};
}  // namespace optix
