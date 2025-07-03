//
//  Copyright (c) 2019 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#include <ExelwtionStrategy/RTX/RTXCompile.h>
// #include <Objects/SemanticType.h>

#include <gtest/gtest.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

using namespace optix;
using namespace llvm;

class RTXCompileTest : public testing::Test
{
};

TEST_F( RTXCompileTest, defineVideoCalls )
{
    RTXCompile::AttributeDecoderList attributeDecoders;
    const std::set<std::string>      heavyWeightCallSites;
    RTXCompile::CompileParams        params;
    RTXCompile::Options              rtxOptions{};
    RTXCompile                       rtxCompile( rtxOptions, attributeDecoders, nullptr, 0 );
    LLVMContext                      context;
    Module                           module( "module", context );

    const char*        videoMax   = "optix.ptx.video.vmax.s32.s32.s32.min.selsec.noSel.noSel.noSel.noSel";
    Type*              i32Ty      = Type::getInt32Ty( context );
    std::vector<Type*> paramTypes = {i32Ty, i32Ty, i32Ty};
    FunctionType*      funcTy     = FunctionType::get( i32Ty, paramTypes, false /*isVarArg*/ );
    Function*          decl = Function::Create( funcTy, GlobalValue::LinkageTypes::ExternalLinkage, videoMax, &module );

    bool fellbackToLWPTX = false;
    rtxCompile.linkPTXFrontEndIntrinsics( &module, false, false, fellbackToLWPTX );
}
