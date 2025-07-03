// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO MODULE SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma once

#include <lwca.h>
#include <string>

namespace optix {
namespace lwca {

class Function;
class SurfRef;
class TexRef;

class Module
{
  public:
    Module();

    // Get the low-level module
    LWmodule       get();
    const LWmodule get() const;

    // Returns a function handle.
    Function getFunction( const std::string& name, LWresult* returnResult = nullptr ) const;

    // Returns a global pointer from a module.  If bytes is non-null, return the size of the global.
    LWdeviceptr getGlobal( size_t* bytes, const std::string& name, LWresult* returnResult = nullptr ) const;

    // Returns a handle to a surface reference.
    SurfRef getSurfRef( const std::string& name, LWresult* returnResult = nullptr ) const;

    // Returns a handle to a texture reference.
    TexRef getTexRef( const std::string& name, LWresult* returnResult = nullptr ) const;

    // Loads a module from either a lwbin, ptx, or fatbin file.
    static Module load( const std::string& fname, LWresult* returnResult = nullptr );

    // Load a module from an image in memory (lwbin, ptx, or fatbin).
    static Module loadData( const void* image, LWresult* returnResult = nullptr );

    // Load a module from an image in memory (lwbin, ptx, or fatbin) with options and their corresponding values.
    static Module loadDataEx( const void* image, unsigned int numOptions, LWjit_option* options, void** optiolwalues, LWresult* returnResult = nullptr );

    // Load a module from an image in memory (lwbin, ptx, or fatbin) with options and their corresponding values,
    // and hide functions from tools if the obscure bit is set.
    static Module loadDataExHidden( const void* image, unsigned int numOptions, LWjit_option* options, void** optiolwalues, LWresult* returnResult = nullptr );

    // Load a module from fatbin in memory.
    static Module loadFatBinary( const void* fatLwbin, LWresult* returnResult = nullptr );

    // Unloads a module.
    void unload( LWresult* returnResult = nullptr );

  private:
    explicit Module( LWmodule module );

    LWmodule m_module;
};

}  // namespace lwca
}  // namespace optix
