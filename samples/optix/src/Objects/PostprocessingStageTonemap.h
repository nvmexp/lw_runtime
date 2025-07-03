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
#include <string>


namespace optix {


/// Built in tonemap post processing stage
class PostprocessingStageTonemap : public PostprocessingStage
{
  public:
    PostprocessingStageTonemap( Context* context );

    ~PostprocessingStageTonemap() override;

    void initialize( RTsize width, RTsize height ) override;

    void validate() const override;

  protected:
    void doLaunch( RTsize width, RTsize height ) override;

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_POSTPROC_STAGE_TONEMAP};
};

inline bool PostprocessingStageTonemap::isA( ManagedObjectType type ) const
{
    return type == m_objectType || PostprocessingStage::isA( type );
}

}  // namespace optix
