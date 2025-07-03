// Copyright LWPU Corporation 2014
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

#include <map>
#include <string>

namespace llvm {
class LLVMContext;
class Module;
}

namespace prodlib {

/// A cache for LLVMs modules
class ModuleCache
{
  public:
    ModuleCache();
    ~ModuleCache();

    /// Adds module to the ModuleCache
    ///
    /// If another module with the same name already exists in the cache, it is
    /// replaced with the new one.
    ///
    /// \param name        Module name
    /// \param module      LLVM module
    void addModule( const std::string& name, llvm::Module* module );

    /// Gets a module from the ModuleCache
    ///
    /// \param name        Module name
    /// \return            NULL if the module is not found in the cache
    llvm::Module* getModule( const std::string& name );

    /// Gets or creates a module from the ModuleCache.  If the filename
    /// is not empty, the module will be read from the file.  Otherwise,
    /// it will be constructed from the in-memory copy of the bitcode.
    ///
    /// \param name        The (optional) cache from which to load the module
    /// \param name        Module name
    /// \param bitcode     The bitcode from which to create the module
    /// \param bitcodeSize The size of the bitcode
    /// \param filename    If non-empty, the filename from which to read the module
    /// \return            the module
    static llvm::Module* getOrCreateModule( ModuleCache*       moduleCache,
                                            llvm::LLVMContext& context,
                                            const std::string& name,
                                            const char*        bitcode,
                                            size_t             bitcodeSize,
                                            const std::string& filename = "" );

  private:
  private:
    ModuleCache( const ModuleCache& );             // forbidden
    ModuleCache& operator=( const ModuleCache& );  // forbidden

    typedef std::map<std::string, llvm::Module*> ModuleMap;
    ModuleMap m_modules;
};
}
