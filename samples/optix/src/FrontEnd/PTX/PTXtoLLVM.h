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

#include <mutex>
#include <prodlib/misc/String.h>
#include <string>
#include <vector>

namespace llvm {
class DataLayout;
class LLVMContext;
class Module;
}

namespace prodlib {
class ModuleCache;
}

// Typedef for decryption callback decryptString(...) defined in EncryptionManager.h
typedef bool ( *DecryptCall )( void*, const char*, size_t, char*, size_t*, size_t, size_t* );

namespace optix {

/// Translates a PTX string into an LLVM module.
///
/// PTX instructions are either replaced by LLVM equivalents or
/// special intrinsics (prefixed with "optix.ptx") that will be
/// substituted by \ref LLVMtoPTX.  Most of the work is handled by the
/// lower-level PTX front-end class (PTXFE) to avoid namespace
/// pollution from the PTX data structures.
class PTXtoLLVM
{
  public:
    PTXtoLLVM( llvm::LLVMContext& llvmContext, const llvm::DataLayout* dataLayout );
    ~PTXtoLLVM();

    /// Translates a PTX string or strings into an LLVM module.
    ///
    /// Throws an exception on error
    ///
    /// \param name        A name for the module
    /// \param declString  A string containing declarations to be parsed before the PTX strings (if non-empty)
    /// \param ptxStrings  One or more strings containing the PTX code.
    /// \param dumpName    Identifier to be included LLVM IR dump filename.
    /// \param decrypter   Instance of the EncryptionManager.
    /// \param decryptCall Decryption callback.
    /// \return            An LLVM module
    llvm::Module* translate( const std::string&                      name,
                             const std::string&                      declString,
                             const std::vector<prodlib::StringView>& ptxStrings,
                             bool                                    parseLineNumbers,
                             const std::string&                      dumpName         = "",
                             void*                                   decrypter        = nullptr,
                             DecryptCall                             decryptCall      = nullptr );

    /// Translates a given llvm::Module into a Module fit for OptiX. It replaces the
    /// inline assembly that calls OptiX functions with the functions that are declared
    /// in the given declString.
    ///
    /// Throws an exception on error
    ///
    /// \param name              A name for the module
    /// \param declString        A string containing declarations to be parsed (if non-empty)
    /// \param module            The input LLVM module
    /// \param skipOptimization  Whether optimization should be skipped for the module
    /// \return                  The processed LLVM module
    llvm::Module* translate( const std::string& name, const std::string& declString, llvm::Module* module, bool skipOptimization );

  private:
    PTXtoLLVM( const PTXtoLLVM& ) = delete;             // forbidden
    PTXtoLLVM& operator=( const PTXtoLLVM& ) = delete;  // forbidden

    llvm::LLVMContext&      m_llvmContext;
    const llvm::DataLayout* m_dataLayout;
    unsigned int            m_targetArch;
};

}  // namespace optix
