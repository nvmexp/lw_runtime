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

#include <Context/UpdateManager.h>
#include <ExelwtionStrategy/Common/Specializations.h>

namespace optix {
class LWDADevice;
class Plan;
class ProgramPlan;

class SpecializationPlan : public UpdateEventListenerNop
{
  public:
    using PerDeviceProgramPlan = std::vector<std::pair<LWDADevice*, ProgramPlan*>>;
    // Will ilwalidate the associated Plan
    SpecializationPlan( Plan* parent, Context* context, const PerDeviceProgramPlan& perDeviceProgramPlan );
    ~SpecializationPlan() override;

    // Determine if the specializations are compatible with another
    bool isCompatibleWith( const SpecializationPlan& otherPlan ) const;

    // Return the specialization sumamry
    std::string summaryString() const;

    // Retrieve the specializations
    const Specializations& getSpecializations() const;

    // Return a new set of specializations that are narrowed as much as
    // possible for the specified canonical program.
    Specializations narrowFor( const CanonicalProgram* cp ) const;

  private:
    //------------------------------------------------------------------------
    // Potentially ilwalidating events
    //------------------------------------------------------------------------

    void eventContextSetExceptionFlags( uint64_t oldFlags, uint64_t newFlags ) override;
    void eventContextSetPrinting( bool        oldEnabled,
                                  size_t      oldBufferSize,
                                  const int3& oldLaunchIndex,
                                  bool        newEnabled,
                                  size_t      newBufferSize,
                                  const int3& newLaunchIndex ) override;
    void eventContextSetPreferFastRecompiles( bool oldValue, bool newValue ) override;
    void eventContextHasMotionTransformsChanged( bool newValue ) override;

    void eventVariableBindingsDidChange( VariableReferenceID refid, const VariableReferenceBinding& binding, bool added ) override;

    void eventBufferBindingsDidChange( VariableReferenceID refid, int texid, bool added ) override;
    void eventBufferMAccessDidChange( const Buffer* buf, const Device* device, const MAccess& oldMS, const MAccess& newMS ) override;

    void eventTextureBindingsDidChange( VariableReferenceID refid, int texid, bool added ) override;
    void eventTextureSamplerMAccessDidChange( const TextureSampler* sampler,
                                              const Device*         device,
                                              const MAccess&        oldMS,
                                              const MAccess&        newMS ) override;

    //------------------------------------------------------------------------
    // Helper functions
    //------------------------------------------------------------------------

    void ilwalidateSpecialization( VariableReferenceID refid );
    void determineSpecializationForVariable( const VariableReference* varref, VariableSpecialization& vs, bool preferFastRecompiles );
    void issueUnspecializedWarning( VariableReferenceID refid ) const;
    void createPlan( const PerDeviceProgramPlan& perDeviceProgramPlan );


    //------------------------------------------------------------------------
    // Member data
    //------------------------------------------------------------------------

    // The specializations
    Specializations m_specializations;

    // Whether scene has any motion transforms
    bool m_hasMotionTransforms = true;

    // The plan to which we are attached
    Plan* m_plan = nullptr;

    Context* m_context = nullptr;
};
}  // namespace optix
