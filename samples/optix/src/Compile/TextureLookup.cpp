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

#include <Compile/TextureLookup.h>
#include <Context/LLVMManager.h>

#include <corelib/compiler/LLVMUtil.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidValue.h>

#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

using namespace optix;
using namespace corelib;
using namespace prodlib;


static void checkLookupKindRange( optix::TextureLookup::LookupKind kind )
{
    if( (unsigned int)kind >= (unsigned int)optix::TextureLookup::Texture_END )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Unknown texture lookup kind: ", kind );
}

int optix::TextureLookup::getLookupDimensionality( LookupKind kind )
{
    checkLookupKindRange( kind );

    static const int kindToDim[] = {
        // Txq size functions
        0,  // Texture_txq_width,
        0,  // Texture_txq_height,
        0,  // Texture_txq_depth,
        0,  // Texture_size,

        // Tex functions
        1,  // Texture_tex_1d,
        2,  // Texture_tex_2d,
        3,  // Texture_tex_3d,
        1,  // Texture_tex_a1d,
        2,  // Texture_tex_a2d,
        2,  // Texture_tex_lwbe,
        2,  // Texture_tex_alwbe

        // TLD/"fetch" functions (linear memory only)
        1,  // Texture_texfetch_1d,
        2,  // Texture_texfetch_2d,
        3,  // Texture_texfetch_3d,
        1,  // Texture_texfetch_a1d,
        2,  // Texture_texfetch_a2d,
        2,  // Texture_texfetch_2dms,
        2,  // Texture_texfetch_a2dms,

        // Mip level
        1,  // Texture_texlevel_1d,
        2,  // Texture_texlevel_2d,
        3,  // Texture_texlevel_3d,
        1,  // Texture_texlevel_a1d,
        2,  // Texture_texlevel_a2d,
        2,  // Texture_texlevel_lwbe,
        2,  // Texture_texlevel_alwbe,

        // Mip grad
        1,  // Texture_texgrad_1d,
        2,  // Texture_texgrad_2d,
        3,  // Texture_texgrad_3d,
        1,  // Texture_texgrad_a1d,
        2,  // Texture_texgrad_a2d,
        2,  // Texture_texgrad_lwbe,
        2,  // Texture_texgrad_alwbe,

        // TLD4
        2,  // Texture_tld4r_2d,
        2,  // Texture_tld4g_2d,
        2,  // Texture_tld4b_2d,
        2,  // Texture_tld4a_2d,
        2,  // Texture_tld4r_a2d,
        2,  // Texture_tld4g_a2d,
        2,  // Texture_tld4b_a2d,
        2,  // Texture_tld4a_a2d,
        2,  // Texture_tld4r_lwbe,
        2,  // Texture_tld4g_lwbe,
        2,  // Texture_tld4b_lwbe,
        2,  // Texture_tld4a_lwbe,
        2,  // Texture_tld4r_alwbe,
        2,  // Texture_tld4g_alwbe,
        2,  // Texture_tld4b_alwbe,
        2,  // Texture_tld4a_alwbe
    };

    return kindToDim[kind];
}

int optix::TextureLookup::getLookupNumCoords( LookupKind kind )
{
    checkLookupKindRange( kind );

    static const int kindToNumCoords[] = {
        // Txq size functions
        0,  // Texture_txq_width,
        0,  // Texture_txq_height,
        0,  // Texture_txq_depth,
        0,  // Texture_size,

        // Tex functions
        1,  // Texture_tex_1d,
        2,  // Texture_tex_2d,
        3,  // Texture_tex_3d,
        1,  // Texture_tex_a1d,
        2,  // Texture_tex_a2d,
        3,  // Texture_tex_lwbe,
        3,  // Texture_tex_alwbe

        // TLD/"fetch" functions (linear memory only)
        1,  // Texture_texfetch_1d,
        2,  // Texture_texfetch_2d,
        3,  // Texture_texfetch_3d,
        1,  // Texture_texfetch_a1d,
        2,  // Texture_texfetch_a2d,
        2,  // Texture_texfetch_2dms,
        2,  // Texture_texfetch_a2dms,

        // Mip level
        1,  // Texture_texlevel_1d,
        2,  // Texture_texlevel_2d,
        3,  // Texture_texlevel_3d,
        1,  // Texture_texlevel_a1d,
        2,  // Texture_texlevel_a2d,
        3,  // Texture_texlevel_lwbe,
        3,  // Texture_texlevel_alwbe,

        // Mip grad
        1,  // Texture_texgrad_1d,
        2,  // Texture_texgrad_2d,
        3,  // Texture_texgrad_3d,
        1,  // Texture_texgrad_a1d,
        2,  // Texture_texgrad_a2d,
        3,  // Texture_texgrad_lwbe,
        3,  // Texture_texgrad_alwbe,

        // TLD4
        2,  // Texture_tld4r_2d,
        2,  // Texture_tld4g_2d,
        2,  // Texture_tld4b_2d,
        2,  // Texture_tld4a_2d,
        2,  // Texture_tld4r_a2d,
        2,  // Texture_tld4g_a2d,
        2,  // Texture_tld4b_a2d,
        2,  // Texture_tld4a_a2d,
        3,  // Texture_tld4r_lwbe,
        3,  // Texture_tld4g_lwbe,
        3,  // Texture_tld4b_lwbe,
        3,  // Texture_tld4a_lwbe,
        3,  // Texture_tld4r_alwbe,
        3,  // Texture_tld4g_alwbe,
        3,  // Texture_tld4b_alwbe,
        3,  // Texture_tld4a_alwbe
    };

    return kindToNumCoords[kind];
}


bool optix::TextureLookup::isLookupTexfetch( LookupKind kind )
{
    checkLookupKindRange( kind );

    switch( kind )
    {
        // TLD/"fetch" functions (linear memory only)
        case Texture_texfetch_1d:
        case Texture_texfetch_2d:
        case Texture_texfetch_3d:
        case Texture_texfetch_a1d:
        case Texture_texfetch_a2d:
        case Texture_texfetch_2dms:
        case Texture_texfetch_a2dms:
            return true;
        default:
            return false;
    }
}

bool optix::TextureLookup::isLookupLayered( LookupKind kind )
{
    checkLookupKindRange( kind );

    switch( kind )
    {
        case Texture_tex_a1d:
        case Texture_tex_a2d:
        case Texture_tex_alwbe:
        case Texture_texfetch_a1d:
        case Texture_texfetch_a2d:
        case Texture_texfetch_a2dms:
        case Texture_texlevel_a1d:
        case Texture_texlevel_a2d:
        case Texture_texlevel_alwbe:
        case Texture_texgrad_a1d:
        case Texture_texgrad_a2d:
        case Texture_texgrad_alwbe:
        case Texture_tld4r_a2d:
        case Texture_tld4g_a2d:
        case Texture_tld4b_a2d:
        case Texture_tld4a_a2d:
        case Texture_tld4r_alwbe:
        case Texture_tld4g_alwbe:
        case Texture_tld4b_alwbe:
        case Texture_tld4a_alwbe:
            return true;
        default:
            return false;
    }
}

void optix::TextureLookup::packParamsFromOptiXFunction( LookupKind kind, llvm::CallInst* CI, std::vector<llvm::Value*>& args )
{
    checkLookupKindRange( kind );

    args.push_back( CI->getArgOperand( 0 ) );
    if( isLookupLayered( kind ) )
    {
        args.push_back( CI->getArgOperand( 5 ) );
    }
    int numCoords = getLookupNumCoords( kind );
    for( int coord = 0; coord < numCoords; coord++ )
    {
        args.push_back( CI->getArgOperand( 2 + coord ) );
    }

    switch( kind )
    {
        // Mip level
        case Texture_texlevel_1d:
        case Texture_texlevel_2d:
        case Texture_texlevel_3d:
        case Texture_texlevel_a1d:
        case Texture_texlevel_a2d:
        case Texture_texlevel_lwbe:
        case Texture_texlevel_alwbe:
        {
            args.push_back( CI->getArgOperand( 6 ) );
            return;
        }

        // Mip grad
        case Texture_texgrad_1d:
        case Texture_texgrad_a1d:
        {
            args.insert( args.end(), CI->op_begin() + 6, CI->op_begin() + 7 );
            args.insert( args.end(), CI->op_begin() + 9, CI->op_begin() + 10 );
            return;
        }
        case Texture_texgrad_2d:
        case Texture_texgrad_a2d:
        {
            args.insert( args.end(), CI->op_begin() + 6, CI->op_begin() + 8 );
            args.insert( args.end(), CI->op_begin() + 9, CI->op_begin() + 11 );
            return;
        }
        case Texture_texgrad_3d:
        case Texture_texgrad_lwbe:
        case Texture_texgrad_alwbe:
        {
            args.insert( args.end(), CI->op_begin() + 6, CI->op_begin() + 9 );
            args.insert( args.end(), CI->op_begin() + 9, CI->op_begin() + 12 );
            return;
        }
        default:
            return;
    }
}

TextureLookup::LookupKind optix::TextureLookup::getLookupKindFromOptiXFunction( llvm::CallInst* CI )
{
    llvm::StringRef name = CI->getCalledFunction()->getName();

    if( name == "_rt_texture_get_size_id" )
    {
        return Texture_size;
    }

    unsigned int dim    = getConstantValueOrAssert( CI->getArgOperand( 1 ) );
    bool         fetch  = name == "_rt_texture_get_fetch_id";
    bool         gather = name == "_rt_texture_get_gather_id";
    unsigned int comp   = 0;
    if( name == "_rt_texture_get_gather_id" )
    {
        comp = getConstantValueOrAssert( CI->getArgOperand( 4 ) );
    }
    bool mipLevel = name == "_rt_texture_get_level_id";
    bool mipGrad  = name == "_rt_texture_get_grad_id";
    bool mipBase  = !mipLevel && !mipGrad && !fetch && !gather;

    // TLD/"fetch" functions (linear memory only)
    if( fetch )
    {
        if( dim == 1 )
            return Texture_texfetch_1d;
        if( dim == 2 )
            return Texture_texfetch_2d;
        if( dim == 3 )
            return Texture_texfetch_3d;
        if( dim == 4 )
            return Texture_texfetch_a1d;
        if( dim == 5 )
            return Texture_texfetch_a2d;
        if( dim == 8 )
            return Texture_texfetch_2dms;
        if( dim == 9 )
            return Texture_texfetch_a2dms;
    }

    // TLD4
    if( gather )
    {
        if( dim == 2 && comp == 0 )
            return Texture_tld4r_2d;
        if( dim == 2 && comp == 1 )
            return Texture_tld4g_2d;
        if( dim == 2 && comp == 2 )
            return Texture_tld4b_2d;
        if( dim == 2 && comp == 3 )
            return Texture_tld4a_2d;
        if( dim == 5 && comp == 0 )
            return Texture_tld4r_a2d;
        if( dim == 5 && comp == 1 )
            return Texture_tld4g_a2d;
        if( dim == 5 && comp == 2 )
            return Texture_tld4b_a2d;
        if( dim == 5 && comp == 3 )
            return Texture_tld4a_a2d;
        if( dim == 6 && comp == 0 )
            return Texture_tld4r_lwbe;
        if( dim == 6 && comp == 1 )
            return Texture_tld4g_lwbe;
        if( dim == 6 && comp == 2 )
            return Texture_tld4b_lwbe;
        if( dim == 6 && comp == 3 )
            return Texture_tld4a_lwbe;
        if( dim == 7 && comp == 0 )
            return Texture_tld4r_alwbe;
        if( dim == 7 && comp == 1 )
            return Texture_tld4g_alwbe;
        if( dim == 7 && comp == 2 )
            return Texture_tld4b_alwbe;
        if( dim == 7 && comp == 3 )
            return Texture_tld4a_alwbe;
    }

    // Tex functions
    if( mipBase )
    {
        if( dim == 1 )
            return Texture_tex_1d;
        if( dim == 2 )
            return Texture_tex_2d;
        if( dim == 3 )
            return Texture_tex_3d;
        if( dim == 4 )
            return Texture_tex_a1d;
        if( dim == 5 )
            return Texture_tex_a2d;
        if( dim == 6 )
            return Texture_tex_lwbe;
        if( dim == 7 )
            return Texture_tex_alwbe;
    }

    // Mip level
    if( mipLevel )
    {
        if( dim == 1 )
            return Texture_texlevel_1d;
        if( dim == 2 )
            return Texture_texlevel_2d;
        if( dim == 3 )
            return Texture_texlevel_3d;
        if( dim == 4 )
            return Texture_texlevel_a1d;
        if( dim == 5 )
            return Texture_texlevel_a2d;
        if( dim == 6 )
            return Texture_texlevel_lwbe;
        if( dim == 7 )
            return Texture_texlevel_alwbe;
    }

    // Mip grad
    if( mipGrad )
    {
        if( dim == 1 )
            return Texture_texgrad_1d;
        if( dim == 2 )
            return Texture_texgrad_2d;
        if( dim == 3 )
            return Texture_texgrad_3d;
        if( dim == 4 )
            return Texture_texgrad_a1d;
        if( dim == 5 )
            return Texture_texgrad_a2d;
        if( dim == 6 )
            return Texture_texgrad_lwbe;
        if( dim == 7 )
            return Texture_texgrad_alwbe;
    }

    throw IlwalidValue( RT_EXCEPTION_INFO, "Unknown texture function: ", name.str() );
}

TextureLookup::LookupKind optix::TextureLookup::getLookupKindFromLWVMFunction( llvm::Function* fn )
{
    // Note: the functions called "unsupported" below are not lwrrently
    // parsed from PTX frontent.  To implement them, just change the
    // word "unsupported" to the name of the correct lookup function.
    llvm::StringRef name = fn->getName();

    // Tex functions
    if( name == "llvm.lwvm.tex.unified.1d.v4f32.f32" || name == "llvm.lwvm.tex.unified.1d.v4s32.f32" )
        return Texture_tex_1d;
    if( name == "llvm.lwvm.tex.unified.2d.v4f32.f32" || name == "llvm.lwvm.tex.unified.2d.v4s32.f32" )
        return Texture_tex_2d;
    if( name == "llvm.lwvm.tex.unified.3d.v4f32.f32" || name == "llvm.lwvm.tex.unified.3d.v4s32.f32" )
        return Texture_tex_3d;
    if( name == "unsupported" )
        return Texture_tex_a1d;
    if( name == "unsupported" )
        return Texture_tex_a2d;
    if( name == "unsupported" )
        return Texture_tex_lwbe;
    if( name == "unsupported" )
        return Texture_tex_alwbe;

    // TLD/"fetch" functions (linear memory only)
    if( name == "llvm.lwvm.tex.unified.1d.v4f32.s32" )
        return Texture_texfetch_1d;
    if( name == "unsupported" )
        return Texture_texfetch_2d;
    if( name == "unsupported" )
        return Texture_texfetch_3d;
    if( name == "unsupported" )
        return Texture_texfetch_a1d;
    if( name == "unsupported" )
        return Texture_texfetch_a2d;
    if( name == "unsupported" )
        return Texture_texfetch_2dms;
    if( name == "unsupported" )
        return Texture_texfetch_a2dms;

    // Mip level
    if( name == "unsupported" )
        return Texture_texlevel_1d;
    if( name == "unsupported" )
        return Texture_texlevel_2d;
    if( name == "unsupported" )
        return Texture_texlevel_3d;
    if( name == "unsupported" )
        return Texture_texlevel_a1d;
    if( name == "unsupported" )
        return Texture_texlevel_a2d;
    if( name == "unsupported" )
        return Texture_texlevel_lwbe;
    if( name == "unsupported" )
        return Texture_texlevel_alwbe;

    // Mip grad
    if( name == "unsupported" )
        return Texture_texgrad_1d;
    if( name == "unsupported" )
        return Texture_texgrad_2d;
    if( name == "unsupported" )
        return Texture_texgrad_3d;
    if( name == "unsupported" )
        return Texture_texgrad_a1d;
    if( name == "unsupported" )
        return Texture_texgrad_a2d;
    if( name == "unsupported" )
        return Texture_texgrad_lwbe;
    if( name == "unsupported" )
        return Texture_texgrad_alwbe;

    // TLD4
    if( name == "unsupported" )
        return Texture_tld4r_2d;
    if( name == "unsupported" )
        return Texture_tld4g_2d;
    if( name == "unsupported" )
        return Texture_tld4b_2d;
    if( name == "unsupported" )
        return Texture_tld4a_2d;
    if( name == "unsupported" )
        return Texture_tld4r_a2d;
    if( name == "unsupported" )
        return Texture_tld4g_a2d;
    if( name == "unsupported" )
        return Texture_tld4b_a2d;
    if( name == "unsupported" )
        return Texture_tld4a_a2d;
    if( name == "unsupported" )
        return Texture_tld4r_lwbe;
    if( name == "unsupported" )
        return Texture_tld4g_lwbe;
    if( name == "unsupported" )
        return Texture_tld4b_lwbe;
    if( name == "unsupported" )
        return Texture_tld4a_lwbe;
    if( name == "unsupported" )
        return Texture_tld4r_alwbe;
    if( name == "unsupported" )
        return Texture_tld4g_alwbe;
    if( name == "unsupported" )
        return Texture_tld4b_alwbe;
    if( name == "unsupported" )
        return Texture_tld4a_alwbe;

    throw IlwalidValue( RT_EXCEPTION_INFO, "Unknown texture function: ", name.str() );
}

llvm::FunctionType* optix::TextureLookup::getPlaceholderFunctionType( LookupKind kind, bool bindless, LLVMManager* llvmManager )
{
    using namespace llvm;
    Type* statePtrTy = llvmManager->getStatePtrType();
    Type* floatTy    = llvmManager->getFloatType();
    Type* i32Ty      = llvmManager->getI32Type();
    ;
    Type* float4Ty = llvmManager->getFloat4Type();
    Type* uint3Ty  = llvmManager->getUint3Type();

    std::string suf = optix::TextureLookup::getArgumentSuffix( kind );
    SmallVector<llvm::Type*, 16> args;
    args.push_back( statePtrTy );
    if( bindless )
        args.push_back( i32Ty );

    for( char c : suf )
    {
        if( c == 'i' || c == 'j' )
        {
            args.push_back( i32Ty );
        }
        else if( c == 'f' )
        {
            args.push_back( floatTy );
        }
    }
    Type* returnType = nullptr;
    switch( kind )
    {
        case Texture_txq_width:
        case Texture_txq_height:
        case Texture_txq_depth:
        {
            returnType = i32Ty;
            break;
        }
        case Texture_size:
        {
            returnType = uint3Ty;
            break;
        }
        default:
        {
            returnType = float4Ty;
            break;
        }
    }
    return FunctionType::get( returnType, args, false );
}

std::string optix::TextureLookup::toString( LookupKind kind )
{
    checkLookupKindRange( kind );

    static const char* kindToStr[] = {
        // Txq size functions
        "txq_width",   // Texture_txq_width,
        "txq_height",  // Texture_txq_height,
        "txq_depth",   // Texture_txq_depth,
        "size",        // Texture_size,

        // Tex functions
        "tex_1d",     // Texture_tex_1d,
        "tex_2d",     // Texture_tex_2d,
        "tex_3d",     // Texture_tex_3d,
        "tex_a1d",    // Texture_tex_a1d,
        "tex_a2d",    // Texture_tex_a2d,
        "tex_lwbe",   // Texture_tex_lwbe,
        "tex_alwbe",  // Texture_tex_alwbe

        // TLD/"fetch" functions (linear memory only)
        "texfetch_1d",     // Texture_texfetch_1d,
        "texfetch_2d",     // Texture_texfetch_2d,
        "texfetch_3d",     // Texture_texfetch_3d,
        "texfetch_a1d",    // Texture_texfetch_a1d,
        "texfetch_a2d",    // Texture_texfetch_a2d,
        "texfetch_lwbe",   // Texture_texfetch_2dms,
        "texfetch_alwbe",  // Texture_texfetch_a2dms

        // Mip level
        "texlevel_1d",     // Texture_texlevel_1d,
        "texlevel_2d",     // Texture_texlevel_2d,
        "texlevel_3d",     // Texture_texlevel_3d,
        "texlevel_a1d",    // Texture_texlevel_a1d,
        "texlevel_a2d",    // Texture_texlevel_a2d,
        "texlevel_lwbe",   // Texture_texlevel_lwbe,
        "texlevel_alwbe",  // Texture_texlevel_alwbe

        // Mip grad
        "texgrad_1d",     // Texture_texgrad_1d,
        "texgrad_2d",     // Texture_texgrad_2d,
        "texgrad_3d",     // Texture_texgrad_3d,
        "texgrad_a1d",    // Texture_texgrad_a1d,
        "texgrad_a2d",    // Texture_texgrad_a2d,
        "texgrad_lwbe",   // Texture_texgrad_lwbe,
        "texgrad_alwbe",  // Texture_texgrad_alwbe

        // TLD4
        "tld4r_2d",     // Texture_tld4r_2d,
        "tld4g_2d",     // Texture_tld4g_2d,
        "tld4b_2d",     // Texture_tld4b_2d,
        "tld4a_2d",     // Texture_tld4a_2d,
        "tld4r_a2d",    // Texture_tld4r_a2d,
        "tld4g_a2d",    // Texture_tld4g_a2d,
        "tld4b_a2d",    // Texture_tld4b_a2d,
        "tld4a_a2d",    // Texture_tld4a_a2d,
        "tld4r_lwbe",   // Texture_tld4r_lwbe,
        "tld4g_lwbe",   // Texture_tld4g_lwbe,
        "tld4b_lwbe",   // Texture_tld4b_lwbe,
        "tld4a_lwbe",   // Texture_tld4a_lwbe,
        "tld4r_alwbe",  // Texture_tld4r_alwbe,
        "tld4g_alwbe",  // Texture_tld4g_alwbe,
        "tld4b_alwbe",  // Texture_tld4b_alwbe,
        "tld4a_alwbe",  // Texture_tld4a_alwbe
    };

    return kindToStr[kind];
}

TextureLookup::LookupKind optix::TextureLookup::fromString( const std::string& str )
{
    // Txq size functions
    if( str == "txq_width" )
        return Texture_txq_width;
    if( str == "txq_height" )
        return Texture_txq_height;
    if( str == "txq_depth" )
        return Texture_txq_depth;
    if( str == "size" )
        return Texture_size;

    // Tex functions
    if( str == "tex_1d" )
        return Texture_tex_1d;
    if( str == "tex_2d" )
        return Texture_tex_2d;
    if( str == "tex_3d" )
        return Texture_tex_3d;
    if( str == "tex_a1d" )
        return Texture_tex_a1d;
    if( str == "tex_a2d" )
        return Texture_tex_a2d;
    if( str == "tex_lwbe" )
        return Texture_tex_lwbe;
    if( str == "tex_alwbe" )
        return Texture_tex_alwbe;

    // TLD/"fetch" functions (linear memory only)
    if( str == "texfetch_1d" )
        return Texture_texfetch_1d;
    if( str == "texfetch_2d" )
        return Texture_texfetch_2d;
    if( str == "texfetch_3d" )
        return Texture_texfetch_3d;
    if( str == "texfetch_a1d" )
        return Texture_texfetch_a1d;
    if( str == "texfetch_a2d" )
        return Texture_texfetch_a2d;
    if( str == "texfetch_2dms" )
        return Texture_texfetch_2dms;
    if( str == "texfetch_a2dms" )
        return Texture_texfetch_a2dms;

    // Mip level
    if( str == "texlevel_1d" )
        return Texture_texlevel_1d;
    if( str == "texlevel_2d" )
        return Texture_texlevel_2d;
    if( str == "texlevel_3d" )
        return Texture_texlevel_3d;
    if( str == "texlevel_a1d" )
        return Texture_texlevel_a1d;
    if( str == "texlevel_a2d" )
        return Texture_texlevel_a2d;
    if( str == "texlevel_lwbe" )
        return Texture_texlevel_lwbe;
    if( str == "texlevel_alwbe" )
        return Texture_texlevel_alwbe;

    // Mip grad
    if( str == "texgrad_1d" )
        return Texture_texgrad_1d;
    if( str == "texgrad_2d" )
        return Texture_texgrad_2d;
    if( str == "texgrad_3d" )
        return Texture_texgrad_3d;
    if( str == "texgrad_a1d" )
        return Texture_texgrad_a1d;
    if( str == "texgrad_a2d" )
        return Texture_texgrad_a2d;
    if( str == "texgrad_lwbe" )
        return Texture_texgrad_lwbe;
    if( str == "texgrad_alwbe" )
        return Texture_texgrad_alwbe;

    // TLD4
    if( str == "tld4r_2d" )
        return Texture_tld4r_2d;
    if( str == "tld4g_2d" )
        return Texture_tld4g_2d;
    if( str == "tld4b_2d" )
        return Texture_tld4b_2d;
    if( str == "tld4a_2d" )
        return Texture_tld4a_2d;
    if( str == "tld4r_a2d" )
        return Texture_tld4r_a2d;
    if( str == "tld4g_a2d" )
        return Texture_tld4g_a2d;
    if( str == "tld4b_a2d" )
        return Texture_tld4b_a2d;
    if( str == "tld4a_a2d" )
        return Texture_tld4a_a2d;
    if( str == "tld4r_lwbe" )
        return Texture_tld4r_lwbe;
    if( str == "tld4g_lwbe" )
        return Texture_tld4g_lwbe;
    if( str == "tld4b_lwbe" )
        return Texture_tld4b_lwbe;
    if( str == "tld4a_lwbe" )
        return Texture_tld4a_lwbe;
    if( str == "tld4r_alwbe" )
        return Texture_tld4r_alwbe;
    if( str == "tld4g_alwbe" )
        return Texture_tld4g_alwbe;
    if( str == "tld4b_alwbe" )
        return Texture_tld4b_alwbe;
    if( str == "tld4a_alwbe" )
        return Texture_tld4a_alwbe;

    throw IlwalidValue( RT_EXCEPTION_INFO, "Unknown texture lookup kind: ", str );
}

std::string optix::TextureLookup::getArgumentSuffix( TextureLookup::LookupKind kind )
{
    checkLookupKindRange( kind );

    static const char* kindToSuf[] = {
        // Txq size functions
        "",  // Texture_txq_width,
        "",  // Texture_txq_height,
        "",  // Texture_txq_depth,
        "",  // Texture_size,

        // Tex functions
        "f",     // Texture_tex_1d,
        "ff",    // Texture_tex_2d,
        "fff",   // Texture_tex_3d,
        "jf",    // Texture_tex_a1d,
        "jff",   // Texture_tex_a2d,
        "fff",   // Texture_tex_lwbe,
        "jfff",  // Texture_tex_alwbe

        // TLD/"fetch" functions (linear memory only)
        "i",     // Texture_texfetch_1d,
        "ii",    // Texture_texfetch_2d,
        "iii",   // Texture_texfetch_3d,
        "ji",    // Texture_texfetch_a1d,
        "jii",   // Texture_texfetch_a2d,
        "jii",   // Texture_texfetch_2dms,
        "jjii",  // Texture_texfetch_a2dms,

        // Mip level
        "ff",     // Texture_texlevel_1d,
        "fff",    // Texture_texlevel_2d,
        "ffff",   // Texture_texlevel_3d,
        "jff",    // Texture_texlevel_a1d,
        "jfff",   // Texture_texlevel_a2d,
        "ffff",   // Texture_texlevel_lwbe,
        "jffff",  // Texture_texlevel_alwbe,

        // Mip grad
        "fff",        // Texture_texgrad_1d,
        "ffffff",     // Texture_texgrad_2d,
        "fffffffff",  // Texture_texgrad_3d,
        "jfff",       // Texture_texgrad_a1d,
        "jffffff",    // Texture_texgrad_a2d,
        "fffffff",    // Texture_texgrad_lwbe,
        "jfffffff",   // Texture_texgrad_alwbe,

        // TLD4
        "ff",    // Texture_tld4r_2d,
        "ff",    // Texture_tld4g_2d,
        "ff",    // Texture_tld4b_2d,
        "ff",    // Texture_tld4a_2d,
        "jff",   // Texture_tld4r_a2d,
        "jff",   // Texture_tld4g_a2d,
        "jff",   // Texture_tld4b_a2d,
        "jff",   // Texture_tld4a_a2d,
        "fff",   // Texture_tld4r_lwbe,
        "fff",   // Texture_tld4g_lwbe,
        "fff",   // Texture_tld4b_lwbe,
        "fff",   // Texture_tld4a_lwbe,
        "jfff",  // Texture_tld4r_alwbe,
        "jfff",  // Texture_tld4g_alwbe,
        "jfff",  // Texture_tld4b_alwbe,
        "jfff",  // Texture_tld4a_alwbe
    };

    return kindToSuf[kind];
}

llvm::Function* optix::TextureLookup::getLookupFunction( LookupKind          kind,
                                                         const std::string&  lookuptype,
                                                         llvm::Module*       module,
                                                         llvm::FunctionType* fnTy )
{
    std::string argPrefix = "tbb";  // unsigned short
    if( lookuptype == "id" )
        argPrefix         = "jbb";  // unsigned int
    std::string argSuffix = getArgumentSuffix( kind );
    std::string kindstr   = toString( kind );
    size_t      length    = lookuptype.length() + kindstr.length() + 20;
    std::string fname     = "_ZN4cort" + std::to_string( length ) + "Texture_getElement_" + lookuptype + "_" + kindstr
                        + "EPNS_14CanonicalStateE" + argPrefix + argSuffix;
    llvm::Function* fn = module->getFunction( fname );
    if( fnTy == nullptr )
    {
        // When we don't have a fnTy we take what was or wasn't found.
        RT_ASSERT_MSG( fn != nullptr, "Texture lookup function not found: " + fname );
        return fn;
    }
    if( fn )
    {
        RT_ASSERT_MSG( fn->getFunctionType() == fnTy, "Texture lookup funciton type doesn't match expected type" );
    }
    else
    {
        fn = llvm::Function::Create( fnTy, llvm::GlobalValue::ExternalLinkage, fname, module );
    }

    return fn;
}
