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

#include <LWCA/Module.h>
#include <Objects/LexicalScope.h>
#include <Objects/PostprocessingStage.h>
#include <PostProcessor/SSIMPredictor/i_ssim.h>

#include <corelib/system/ExelwtableModule.h>

#include <string>

namespace optix {

class SSIMPredictorLogger;

/// Built-in ssim predictor post processing stage
class PostprocessingStageSSIMPredictor : public PostprocessingStage
{
  public:
    PostprocessingStageSSIMPredictor( Context* context, void* ssim_predictor );

    ~PostprocessingStageSSIMPredictor() override;

    /// Initialize the ssim predictor. This will load the DLL on demand and
    /// then get the necessary functions.
    void initialize( RTsize width, RTsize height ) override;

    void validate() const override;

    // Called when a linked buffer is destroyed.
    void detachLinkedChild( const LinkedPtr_Link* link ) override;

    // Called when the linked buffer is mapped for write.
    void bufferWasMapped( LinkedPtr_Link* link ) override;

    // Called when the linked buffer format changes in any way (size, element format, etc)
    void bufferFormatDidChange( LinkedPtr_Link* link ) override;

  protected:
    void doLaunch( RTsize width, RTsize height ) override;

    // Override to be able to monitor buffers for changes (lwrrently only "training_data_buffer" variable)
    void bufferVariableValueDidChange( Variable* var, Buffer* oldBuffer, Buffer* newBuffer ) override;

  private:
    /// The ExelwtableModule holds the ssim_predictor DLL
    /// TODO: Maybe we should move to some manager?
    static std::unique_ptr<corelib::ExelwtableModule> m_ssimPredictorLibrary;

    typedef void*                 SSIMPredictorFactoryFunction();
    SSIMPredictorFactoryFunction* m_factory = nullptr;

    typedef const unsigned char*      SSIMPredictorDefaultDataFunction( bool );
    SSIMPredictorDefaultDataFunction* m_get_default_data = nullptr;

    typedef unsigned int                  SSIMPredictorDefaultDataSizeFunction( bool );
    SSIMPredictorDefaultDataSizeFunction* m_get_default_data_size = nullptr;

    /// The ssim_predictor belong to this stage
    LW::SSIM::ISsim* m_ssimPredictor = nullptr;

    /// The ssim_predictor logger.
    std::unique_ptr<SSIMPredictorLogger> m_ssimPredictorLogger;

    /// The linked pointer to the child "training_data_buffer" buffer
    LinkedPtr<PostprocessingStage, Buffer> m_trainingDataBufferLink;

    // True if the training data is marked as dirty in which case it needs to be passed to the ssim_predictor.
    bool m_trainingDataDirty = true;

    // True if the "hdr" stage variable has been set to true, in which case the hdr default training
    // data will be selected.
    bool m_useHdrTrainingData = false;

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_POSTPROC_STAGE_SSIM};
};

inline bool PostprocessingStageSSIMPredictor::isA( ManagedObjectType type ) const
{
    return type == m_objectType || PostprocessingStage::isA( type );
}

}  // namespace optix
