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

#include <Objects/Buffer.h>
#include <Objects/LexicalScope.h>
#include <Objects/TextureSampler.h>

namespace optix {

class Buffer;
class Context;
class LexicalScope;
class TextureSampler;
class ValidationManager
{
  public:
    ValidationManager( Context* context );
    ValidationManager( const ValidationManager& ) = delete;
    ValidationManager& operator=( const ValidationManager& ) = delete;
    virtual ~ValidationManager();

    void run();

    void subscribeForValidation( const LexicalScope* object );
    void unsubscribeForValidation( const LexicalScope* object );
    void subscribeForValidation( const Buffer* buffer );
    void unsubscribeForValidation( const Buffer* buffer );
    void subscribeForValidation( const TextureSampler* sampler );
    void unsubscribeForValidation( const TextureSampler* sampler );

  private:
    Context* m_context;

    typedef IndexedVector<const LexicalScope*, LexicalScope::validationIndex_fn> ScopeListType;
    ScopeListType m_objectsForValidation;

    typedef IndexedVector<const Buffer*, Buffer::validationIndex_fn> BufferListType;
    BufferListType m_buffersForValidation;

    typedef IndexedVector<const TextureSampler*, TextureSampler::validationIndex_fn> TextureSamplerListType;
    TextureSamplerListType m_texturesForValidation;
};
}
