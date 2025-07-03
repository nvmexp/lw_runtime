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

#include <string>

/*
 * Canonical mangling
 */
namespace optix {

// Canonical Mangling

std::string canonicalMangleVariableName( const std::string& name );
std::string canonicalDemangleVariableName( const std::string& name );

// Prepends 'prepend' before the last namespace.  Assumes demangled input.
// name =      abc   prepend = name::   returns =       name::abc
// name = xyz::abc   prepend = name::   returns = xyz:::name::abc
std::string canonicalPrependNamespace( const std::string& name, const std::string& prepend );

// PTX Mangling

std::string ptxMangleVariableName( const std::string& name );
std::string ptxDemangleVariableName( const std::string& name );
}
