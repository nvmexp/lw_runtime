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
#include <vector>


namespace llvm {
class CallInst;
class Function;
class FunctionType;
class LLVMContext;
class Module;
class Type;
class Value;
}

namespace optix {
class LLVMManager;

namespace TextureLookup {
enum LookupKind
{
    // Txq size functions
    Texture_txq_width,   // ()
    Texture_txq_height,  // ()
    Texture_txq_depth,   // ()
    Texture_size,        // ()  // No exact equivalent in lwca

    // Tex functions
    Texture_tex_1d,     // (float)
    Texture_tex_2d,     // (float, float)
    Texture_tex_3d,     // (float, float, float)
    Texture_tex_a1d,    // (i32, float)
    Texture_tex_a2d,    // (i32, float, float)
    Texture_tex_lwbe,   // (float, float, float)
    Texture_tex_alwbe,  // (i32, float, float, float)

    // TLD/"fetch" functions (linear memory only)
    Texture_texfetch_1d,     // (i32)
    Texture_texfetch_2d,     // (i32, i32) // Not exposed in lwca
    Texture_texfetch_3d,     // (i32, i32, i32) // Not exposed in lwca
    Texture_texfetch_a1d,    // (i32, i32) // Not exposed in lwca
    Texture_texfetch_a2d,    // (i32, i32, i32) // Not exposed in lwca
    Texture_texfetch_2dms,   // (i32, i32, i32) // Not exposed in lwca
    Texture_texfetch_a2dms,  // (i32, i32, i32, i32) // Not exposed in lwca

    // Mip level
    Texture_texlevel_1d,     // (float, float)
    Texture_texlevel_2d,     // (float, float, float)
    Texture_texlevel_3d,     // (float, float, float, float)
    Texture_texlevel_a1d,    // (i32, float, float)
    Texture_texlevel_a2d,    // (i32, float, float, float)
    Texture_texlevel_lwbe,   // (float, float, float, float)
    Texture_texlevel_alwbe,  // (i32, float, float, float, float)

    // Mip grad
    Texture_texgrad_1d,     // (float, float)
    Texture_texgrad_2d,     // (float, float, float, float, float, float)
    Texture_texgrad_3d,     // (float, float, float, float, float, float, float, float, float)
    Texture_texgrad_a1d,    // (i32, float, float)
    Texture_texgrad_a2d,    // (i32, float, float, float, float, float, float)
    Texture_texgrad_lwbe,   // (float, float, float, float, float, float, float)
    Texture_texgrad_alwbe,  // (i32, float, float, float, float, float, float, float)

    // TLD4
    Texture_tld4r_2d,     // (float, float)
    Texture_tld4g_2d,     // (float, float)
    Texture_tld4b_2d,     // (float, float)
    Texture_tld4a_2d,     // (float, float)
    Texture_tld4r_a2d,    // (i32, float, float) // Not exposed in PTX
    Texture_tld4g_a2d,    // (i32, float, float) // Not exposed in PTX
    Texture_tld4b_a2d,    // (i32, float, float) // Not exposed in PTX
    Texture_tld4a_a2d,    // (i32, float, float) // Not exposed in PTX
    Texture_tld4r_lwbe,   // (float, float, float) // Not exposed in PTX
    Texture_tld4g_lwbe,   // (float, float, float) // Not exposed in PTX
    Texture_tld4b_lwbe,   // (float, float, float) // Not exposed in PTX
    Texture_tld4a_lwbe,   // (float, float, float) // Not exposed in PTX
    Texture_tld4r_alwbe,  // (i32, float, float, float) // Not exposed in PTX
    Texture_tld4g_alwbe,  // (i32, float, float, float) // Not exposed in PTX
    Texture_tld4b_alwbe,  // (i32, float, float, float) // Not exposed in PTX
    Texture_tld4a_alwbe,  // (i32, float, float, float) // Not exposed in PTX

    Texture_END
};
// Iterator functionality.  If Lookup is extended, these need to be updated.
inline LookupKind beginLookupKind()
{
    return Texture_tex_1d;
}
inline LookupKind endLookupKind()
{
    return static_cast<LookupKind>( Texture_tld4a_lwbe + 1 );
}
inline LookupKind nextEnum( LookupKind kind )
{
    return static_cast<LookupKind>( kind + 1 );
}

// Lookup properties
int getLookupDimensionality( LookupKind kind );
int getLookupNumCoords( LookupKind kind );
bool isLookupTexfetch( LookupKind kind );
bool isLookupLayered( LookupKind kind );
std::string toString( LookupKind kind );
LookupKind fromString( const std::string& str );
std::string getArgumentSuffix( TextureLookup::LookupKind kind );

// Compiler utilities
void packParamsFromOptiXFunction( LookupKind kind, llvm::CallInst* CI, std::vector<llvm::Value*>& args );
LookupKind getLookupKindFromOptiXFunction( llvm::CallInst* CI );
LookupKind getLookupKindFromLWVMFunction( llvm::Function* fn );
llvm::FunctionType* getPlaceholderFunctionType( LookupKind kind, bool bindless, LLVMManager* llvmManager );
// If fnTy is non-null, then a declaration of the function is added if not found.
llvm::Function* getLookupFunction( LookupKind kind, const std::string& argKind, llvm::Module* module, llvm::FunctionType* fnTy = nullptr );
}
}
