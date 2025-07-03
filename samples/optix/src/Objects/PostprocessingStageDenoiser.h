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

#include <corelib/system/ExelwtableModule.h>
#include <optix_types.h>
#include <string>

namespace optix {

class DenoiserLogBuffer
{
  public:
    static void callback( unsigned int level, const char* tag, const char* message, void* cbdata )
    {
        DenoiserLogBuffer* self = static_cast<DenoiserLogBuffer*>( cbdata );
        self->callback( level, tag, message );
    }

    void callback( unsigned int level, const char* tag, const char* message )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        m_messages.push_back( message ? message : "(no message)" );
    }

    std::string getMessage( const char* const str )
    {
        std::string res(str);
        res += std::string(". ");
        std::lock_guard<std::mutex> lock( m_mutex );
        for ( const auto& m : m_messages )
            res += m + std::string( "\n" );
        m_messages.clear();
        return res;
    }

  private:
    // Mutex that protects m_messages.
    std::mutex m_mutex;

    // Needs m_mutex.
    std::vector<std::string> m_messages;
};

/// Built in denoiser post processing stage
class PostprocessingStageDenoiser : public PostprocessingStage
{
  public:
    PostprocessingStageDenoiser( Context* context, void* denoiser );

    ~PostprocessingStageDenoiser() override;

    /// Initialize the denoiser. This will load the DLL on demand and
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
    /// The denoiser belong to this stage
    OptixDenoiser        m_denoiser;
    OptixDenoiserOptions m_denoiserOptions;
    OptixDeviceContext   m_optixDevice;
    DenoiserLogBuffer    m_logBuffer;

    void*        m_stateMem;
    size_t       m_stateSizeInBytes;
    void*        m_scratchMem;
    size_t       m_scratchSizeInBytes;

    unsigned int m_width;
    unsigned int m_height;

    /// The linked pointer to the child "training_data_buffer" buffer
    LinkedPtr<PostprocessingStage, Buffer> m_trainingDataBufferLink;

    // True if the training data is marked as dirty in which case it needs to be passed to the denoiser.
    bool m_trainingDataDirty = true;

    // True if the "hdr" stage variable has been set to true, in which case the hdr default training
    // data will be selected.
    bool m_useHdrTrainingData = false;

    void destroy();

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_POSTPROC_STAGE_DENOISER};
};

inline bool PostprocessingStageDenoiser::isA( ManagedObjectType type ) const
{
    return type == m_objectType || PostprocessingStage::isA( type );
}

}  // namespace optix
