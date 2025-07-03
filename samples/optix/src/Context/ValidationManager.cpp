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

#include <Context/ValidationManager.h>

#include <Context/Context.h>

using namespace optix;

ValidationManager::ValidationManager( Context* context )
    : m_context( context )
{
}

ValidationManager::~ValidationManager()
{
}

void ValidationManager::subscribeForValidation( const LexicalScope* object )
{
    m_objectsForValidation.addItem( object );
}

void ValidationManager::subscribeForValidation( const Buffer* buffer )
{
    m_buffersForValidation.addItem( buffer );
}

void ValidationManager::subscribeForValidation( const TextureSampler* ts )
{
    m_texturesForValidation.addItem( ts );
}

void ValidationManager::unsubscribeForValidation( const LexicalScope* object )
{
    m_objectsForValidation.removeItem( object );
}

void ValidationManager::unsubscribeForValidation( const Buffer* buffer )
{
    m_buffersForValidation.removeItem( buffer );
}

void ValidationManager::unsubscribeForValidation( const TextureSampler* ts )
{
    m_texturesForValidation.removeItem( ts );
}

void ValidationManager::run()
{
    m_context->saveNodeGraph( "validation" );

    // validate textures
    for( const auto& sampler : m_texturesForValidation )
        if( sampler->isAttached() )
            sampler->validate();
    m_texturesForValidation.clear();

    // validate buffers
    for( const auto& buffer : m_buffersForValidation )
        if( buffer->isAttached() )
            buffer->validate();
    m_buffersForValidation.clear();

    // validate objects
    for( const auto& scope : m_objectsForValidation )
        if( scope->isAttached() )
            scope->validate();
    m_objectsForValidation.clear();
}
