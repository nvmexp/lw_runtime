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

#include <LWCA/ComputeCapability.h>
#include <FrontEnd/Canonical/CallSiteIdentifier.h>
#include <FrontEnd/Canonical/CanonicalProgramID.h>
#include <Objects/GraphProperty.h>
#include <Objects/SemanticType.h>
#include <corelib/misc/Concepts.h>

#include <vector>

namespace llvm {
class Function;
}

namespace optix {
class Context;
class Device;
class PersistentStream;
class ProgramManager;
class UpdateManager;
class VariableReference;
class CanonicalProgram;

void readOrWrite( PersistentStream* stream, CanonicalProgram* cp, const char* label );

// Canonical representation of any input programs, including any
// derived data to make compiling the programs easier.
class CanonicalProgram : private corelib::NonCopyable
{
  public:
    struct IDCompare
    {
        bool operator()( const CanonicalProgram* c1, const CanonicalProgram* c2 ) const { return c1->getID() < c2->getID(); }
    };

    ~CanonicalProgram() NOEXCEPT_FALSE;

    // Metadata
    const std::string& getInputFunctionName() const;
    const std::string& getUniversallyUniqueName() const;
    unsigned           getFunctionSignature() const;
    size_t             getPTXHash() const;
    int                get32bitAttributeKind() const;

    // Target architecture
    lwca::ComputeCapability getTargetMin() const;
    lwca::ComputeCapability getTargetMax() const;
    bool isValidForDevice( const Device* device ) const;

    // Module information
    const llvm::Function* llvmFunction() const;
    const llvm::Function* llvmIntersectionFunction() const;
    const llvm::Function* llvmAttributeDecoder() const;
    CanonicalProgramID    getID() const;
    Context*              getContext() const;

    // Usage information. Note that these violate the const-ness of
    // CanonicalProgram. However, it is helpful to retain the policy
    // that this class is const everywhere so that it is clear that
    // other (non-graph) properties can never change.
    void receivePropertyDidChange_UsedAsSemanticType( SemanticType stype, bool added ) const;
    void receivePropertyDidChange_InheritedSemanticType( SemanticType type, bool added ) const;
    void receivePropertyDidChange_UsedByRayType( unsigned int rayType, bool added ) const;
    void receivePropertyDidChange_UsedOnDevice( const Device* device, bool added ) const;
    void receivePropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const;

    void receivePropertyDidChange_calledFromCallsite( CallSiteIdentifier* csId, bool added ) const;

    // Colwenience functions to query the semantic types lwrrently
    // attached to the canonical program.
    bool isUsedAsSemanticType( SemanticType stype ) const;
    bool isUsedAsSemanticTypes( const std::vector<SemanticType>& stypes ) const;
    bool isUsedAsInheritedSemanticType( SemanticType stype ) const;
    bool isUsedAsInheritedSemanticTypes( const std::vector<SemanticType>& stypes ) const;
    bool         isUsedAsSingleSemanticType() const;
    SemanticType getSingleSemanticType() const;
    void getAllUsedSemanticTypes( std::vector<SemanticType>& stypes ) const;

    // Query the ray types that will lwrrently reach the canonical
    // program.
    bool isUsedByRayTypes( const GraphProperty<unsigned int, false>& ) const;
    const GraphProperty<unsigned int, false>& producesRayTypes() const;
    bool tracesUnknownRayType() const;

    // Query whether this canonical program is lwrrently reachable by
    // one of the callers
    bool hasDirectCaller( const std::set<const CanonicalProgram*, IDCompare>& callers ) const;

    // Callwlate the set of semantic types a bound callable program
    // is used from. A bound callable program is compiled once
    // per inherited semantic type.
    std::vector<SemanticType> getInheritedSemanticTypes() const;

    // Query whether the canonical program is used on the specified
    // device.
    bool isUsedOnDevice( const Device* device ) const;

    // Representation of variable and attribute references
    typedef std::vector<const VariableReference*> VariableReferenceListType;
    const VariableReferenceListType&              getAttributeReferences() const;
    const VariableReferenceListType&              getVariableReferences() const;
    int getMaxAttributeData32bitValues() const { return m_maxAttributeData32bitValues; }

    // SLOW search for variable of a given name - used only for testing
    const VariableReference* findAttributeReference( const std::string& name ) const;
    const VariableReference* findVariableReference( const std::string& name ) const;

    // Search by id.  Not as slow, but still slow.
    const VariableReference* findVariableReference( VariableReferenceID id ) const;

    // Query properties of this canonical program.
    bool callsBoundCallableProgram() const { return m_callsBoundCallableProgram; }
    bool callsBindlessCallableProgram() const { return m_callsBindlessCallableProgram; }
    bool callsReportIntersection() const { return m_callsReportIntersection; }
    bool canBindlessBufferPointerEscape() const { return m_bindlessBufferPointerMayEscape; }
    bool canPayloadPointerEscape() const { return m_payloadPointerMayEscape; }
    bool canGlobalPointerEscape() const { return m_globalPointerMayEscape; }
    bool canGlobalConstPointerEscape() const { return m_globalConstPointerMayEscape; }

    bool hasPayloadStores() const { return m_hasPayloadStores; }
    bool hasDynamicAccessesToPayload() const { return m_hasDynamicPayloadAccesses; }
    bool hasPayloadAccesses() const { return m_hasPayloadAccesses; }
    bool hasBufferStores() const { return m_hasBufferStores; }
    bool hasAttributeStores() const { return m_hasAttributeStores; }
    bool hasAttributeLoads() const { return m_hasAttributeLoads; }
    bool isBuiltInIntersection() const { return m_isBuiltInIntersection; }
    bool hasLwrrentRayAccess() const { return m_hasLwrrentRayAccess; }
    // Nothing can lwrrently set this property. Deferred
    // attributes could technically produce different results if there
    // is a global write in the intersection program.
    bool hasGlobalStores() const { return m_hasGlobalStores; }

    bool callsTrace() const { return m_callsTrace; }
    bool hasMotionIndexArg() const { return m_hasMotionIndexArg; }
    int  getMaxPayloadSize() const { return m_maxPayloadSize; }
    int  getMaxPayloadRegisterCount() const { return m_maxPayloadRegisterCount; }

    bool hasLaunchIndexAccesses() const { return m_accessesLaunchIndex; }

    void validateSemanticType( SemanticType stype ) const;

    // Used to restore from a cache.
    CanonicalProgram( Context* context, size_t ptxHash );

    const std::vector<const CallSiteIdentifier*>& getCallSites() const;

    const GraphProperty<CallSiteIdentifier*> getCallingCallSites() const { return m_calledFromCallsites; }

    const GraphProperty<CanonicalProgramID>& getDirectCallers() const { return m_directCaller; }

  private:
    //
    // WARNING: This is a persistent class. If you change anything you
    // should also update the readOrWrite function.
    //
    Context*                m_context = nullptr;
    std::string             m_inputFunctionName;
    std::string             m_universallyUniqueName;
    ReusableID              m_id;
    unsigned                m_signatureId = 0u;
    lwca::ComputeCapability m_targetMin;
    lwca::ComputeCapability m_targetMax;
    size_t                  m_ptxHash                 = 0;
    size_t                  m_universallyUniqueNumber = 0;

    // ONLY these properties are mutable. See explanation above.
    mutable GraphProperty<SemanticType>       m_usedAsSemanticType;
    mutable GraphProperty<SemanticType>       m_inheritedSemanticType;
    mutable GraphProperty<unsigned int>       m_usedByRayType;
    mutable GraphProperty<unsigned int>       m_usedOnDevice;  // indexed by allDeviceListIndex
    mutable GraphProperty<CanonicalProgramID> m_directCaller;  // which CanonicalProgram objects call this

    // callSites that call this CP
    // Needed because (as opposed to m_directCaller) we need to be able to
    // tell the individual callsite in the direct caller that this program has
    // changed its heavyweight status
    mutable GraphProperty<CallSiteIdentifier*> m_calledFromCallsites;

    // named callSites within this CP. These are owned by this CP. No need to
    // have this as a (uncounted) GraphProperty, this will not change after c14n.
    // Needed to propagate heavyweight changes of this CP to its callees.
    std::vector<const CallSiteIdentifier*> m_ownedCallSites;

    /*
   * The remainder of the properties are created at construction time
   * and must never change.
   */
    GraphProperty<unsigned int, false> m_producesRayTypes;
    bool m_producesUnknownRayType = false;

    // Program properties. Warning: it is easy to forget to add these
    // to the readOrWriteFunction. Please update the numbered comments
    // when you do so.
    bool m_callsTrace              = false;  // 1
    bool m_callsGetPrimitiveIndex  = false;  // 2
    bool m_callsGetInstanceFlags   = false;  // 3
    bool m_callsGetRayFlags        = false;  // 4
    bool m_accessesHitKind         = false;  // 5
    bool m_traceHasTime            = false;  // 6
    bool m_callsThrow              = false;  // 7
    bool m_callsTerminateRay       = false;  // 8
    bool m_callsIgnoreIntersection = false;  // 9
    bool m_callsIntersectChild     = false;  // 10

    bool m_callsPotentialIntersection   = false;  // 11
    bool m_callsReportIntersection      = false;  // 12
    bool m_isBuiltInIntersection        = false;  // 13
    bool m_callsTransform               = false;  // 14
    bool m_callsExceptionCode           = false;  // 15
    bool m_callsBoundCallableProgram    = false;  // 16
    bool m_callsBindlessCallableProgram = false;  // 17
    bool m_hasLwrrentRayAccess          = false;  // 18
    bool m_hasLwrrentTimeAccess         = false;  // 19
    bool m_accessesIntersectionDistance = false;  // 20

    int m_maxPayloadSize              = 0;  // 21
    int m_maxPayloadRegisterCount     = 0;  // 22
    int m_maxAttributeData32bitValues = 0;  // 23

    bool m_bindlessBufferPointerMayEscape = false;  // 24
    bool m_payloadPointerMayEscape        = false;  // 25
    bool m_globalPointerMayEscape         = false;  // 26
    bool m_globalConstPointerMayEscape    = false;  // 27

    bool m_hasPayloadStores          = false;  // 28
    bool m_hasPayloadAccesses        = false;  // 29
    bool m_hasDynamicPayloadAccesses = false;  // 30
    bool m_hasBufferStores           = false;  // 31
    bool m_hasAttributeStores        = false;  // 32
    bool m_hasAttributeLoads         = false;  // 33
    bool m_hasGlobalStores           = false;  // 34

    // AABB program property
    bool m_hasMotionIndexArg = false;  // 35

    bool m_callsGetLowestGroupChildIndex = false;  // 36
    bool m_callsGetRayMask               = false;  // 37

    bool m_accessesLaunchIndex = false;  // 38

    // Representation of variable and attribute references.
    VariableReferenceListType m_attributeReferences;
    VariableReferenceListType m_variableReferences;

    // LLVM function or bitcode (only one will exist at a time)
    mutable llvm::Function*   m_function = nullptr;
    mutable std::vector<char> m_lazyLoadBitcode;

    // LLVM function or bitcode (only one will exist at a time) for intersection
    mutable llvm::Function*   m_intersectionFunction = nullptr;
    mutable std::vector<char> m_lazyLoadIntersectionBitcode;

    // LLVM function or bitcode (only one will exist at a time) for attributes
    mutable llvm::Function*   m_attributeDecoder = nullptr;
    mutable std::vector<char> m_lazyLoadAttributeBitcode;

    // Set properties of the canonical program. These are created at
    // canonicalization time and are immutable over the lifetime of
    // the canonical program object once it has been finalized.
    // clang-format off
    void markCallsTransform()             { m_callsTransform = true; }
    void markCallsTrace()                 { m_callsTrace = true; }
    void markTracesUnknownRayType();
    void markCallsGetPrimitiveIndex()     { m_callsGetPrimitiveIndex = true; }
    void markCallsGetInstanceFlags()      { m_callsGetInstanceFlags = true; }
    void markCallsGetRayFlags()           { m_callsGetRayFlags = true; }
    void markCallsGetRayMask()            { m_callsGetRayMask = true; }
    void markAccessesHitKind()            { m_accessesHitKind = true; }
    void markTraceHasTime()               { m_traceHasTime = true; }
    void markCallsThrow()                 { m_callsThrow = true; }
    void markCallsTerminateRay()          { m_callsTerminateRay = true; }
    void markCallsIgnoreIntersection()    { m_callsIgnoreIntersection = true; }
    void markCallsIntersectChild()        { m_callsIntersectChild = true; }
    void markCallsPotentialIntersection() { m_callsPotentialIntersection = true; }
    void markCallsReportIntersection()    { m_callsReportIntersection = true; }
    void markIsBuiltInIntersection() { m_isBuiltInIntersection = true; }
    void markCallsExceptionCode()         { m_callsExceptionCode = true; }
    void markCallsBoundCallableProgram()    { m_callsBoundCallableProgram = true; }
    void markCallsBindlessCallableProgram() { m_callsBindlessCallableProgram = true; }
    void markAccessesLwrrentRay()           { m_hasLwrrentRayAccess = true; }
    void markAccessesLwrrentTime()          { m_hasLwrrentTimeAccess = true; }
    void markAccessesIntersectionDistance() { m_accessesIntersectionDistance = true; }
    void setMaxPayloadSize( int maxPayloadSize ) { m_maxPayloadSize = maxPayloadSize; }
    void setMaxPayloadRegisterCount( int maxPayloadRegisterCount ) { m_maxPayloadRegisterCount = maxPayloadRegisterCount; }
    void setMaxAttributeData32bitValues( int count ) { m_maxAttributeData32bitValues = count; }
    // clang-format on

    void markBindlessBufferPointerMayEscape() { m_bindlessBufferPointerMayEscape = true; }
    void markPayloadPointerMayEscape() { m_payloadPointerMayEscape = true; }
    void markGlobalPointerMayEscape() { m_globalPointerMayEscape = true; }
    void markGlobalConstPointerMayEscape() { m_globalConstPointerMayEscape = true; }

    void markHasPayloadStores() { m_hasPayloadStores = true; }
    void markHasDynamicPayloadAccesses() { m_hasDynamicPayloadAccesses = true; }
    void markHasPayloadAccesses() { m_hasPayloadAccesses = true; }
    void markHasBufferStores() { m_hasBufferStores = true; }
    void markHasAttributeStores() { m_hasAttributeStores = true; }
    void markHasAttributeLoads() { m_hasAttributeLoads = true; }
    void markHasGlobalStores() { m_hasGlobalStores = true; }

    void markHasMotionIndexArg() { m_hasMotionIndexArg = true; }
    void markCallsGetLowestGroupChildIndex() { m_callsGetLowestGroupChildIndex = true; }

    void markHasLaunchIndexAccesses() { m_accessesLaunchIndex = true; }

    /*
   * Internal helper functions
   */

    // Constructors/destructors
    // - Note, CanonicalPrograms can only be created by
    // - canonicalization or by the program manager (which reads them
    // - from a cache)
    friend class C14n;
    friend class ProgramManager;
    friend class NodegraphPrinter;
    CanonicalProgram( const std::string&      originalFunctionName,
                      lwca::ComputeCapability targetMin,
                      lwca::ComputeCapability targetMax,
                      size_t                  ptxHash,
                      Context*                context );
    void finalize( llvm::Function* function );

    // Persistence support
    friend void optix::readOrWrite( PersistentStream* stream, CanonicalProgram* cp, const char* label );
};

}  // namespace optix
