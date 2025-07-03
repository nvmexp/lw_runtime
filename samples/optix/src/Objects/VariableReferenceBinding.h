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

#include <FrontEnd/Canonical/VariableReference.h>
#include <Objects/ObjectClass.h>
#include <stddef.h>
#include <string>

namespace optix {
class PersistentStream;

struct VariableReferenceBinding
{
    //
    // WARNING: This is a persistent class. If you change anything you
    // should also update the readOrWrite function and bump the the
    // version number.
    //

    VariableReferenceBinding( ObjectClass scopeClass, size_t offset );
    VariableReferenceBinding() = default;

    static VariableReferenceBinding makeDefaultValueBinding();

    ObjectClass scopeClass() const { return m_scopeClass; }
    size_t      offset() const { return m_offset; }
    bool        isDefaultValue() const { return m_isDefaultValue; }

    bool operator<( const VariableReferenceBinding& rhs ) const;
    bool operator==( const VariableReferenceBinding& rhs ) const;
    bool operator!=( const VariableReferenceBinding& rhs ) const;

    std::string toString() const;

  private:
    ObjectClass m_scopeClass     = RT_OBJECT_UNKNOWN;
    size_t      m_offset         = 0;
    bool        m_isDefaultValue = false;

    friend void readOrWrite( PersistentStream* stream, VariableReferenceBinding* vrb, const char* label );
};

// Persistence support
void readOrWrite( PersistentStream* stream, VariableReferenceBinding* vrb, const char* label );
}
