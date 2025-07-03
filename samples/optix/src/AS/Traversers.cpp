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

#include <AS/Traversers.h>
#include <Context/LLVMManager.h>
#include <Util/Misc.h>  // countof()
#include <corelib/compiler/LLVMUtil.h>
#include <corelib/compiler/LocalMemorySpaceOpt.h>
#include <corelib/misc/String.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/system/Knobs.h>

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>

using namespace optix;
using namespace prodlib;
using namespace corelib;
using namespace llvm;

namespace {
// clang-format off
  Knob<int> k_smemStackBytes( RT_DSTRING( "acceleration.traverser.smemStackBytes" ), 16*4, RT_DSTRING( "Number of bytes for the shared memory stack (for sm50+)" ) );
// clang-format on
}

//------------------------------------------------------------------------------
bool TraverserParams::operator<( const TraverserParams& other ) const
{
    if( traverserName != other.traverserName )
        return traverserName < other.traverserName;
    else if( isGeom != other.isGeom )
        return isGeom < other.isGeom;
    else if( hasMotion != other.hasMotion )
        return hasMotion < other.hasMotion;
    else
        return bakedChildPtrs < other.bakedChildPtrs;
}

//------------------------------------------------------------------------------
// Inline dst func into calls to src func from func
static bool inlineFunc( Function* func, const std::string& dstFuncName, const std::string& srcFuncName )
{
    Module*   module  = func->getParent();
    Function* srcFunc = module->getFunction( srcFuncName );
    Function* dstFunc = module->getFunction( dstFuncName );
    RT_ASSERT( srcFunc != nullptr );
    RT_ASSERT( dstFunc != nullptr );

    std::vector<CallInst*> calls;
    for( Function::user_iterator UI = dstFunc->user_begin(), UE = dstFunc->user_end(); UI != UE; ++UI )
    {
        if( CallInst* call = dyn_cast<CallInst>( *UI ) )
        {
            Function* parentFunc = call->getParent()->getParent();
            if( parentFunc == func )
                calls.push_back( call );
        }
    }

    bool success = true;
    for( CallInst* call : calls )
    {
        call->setCalledFunction( srcFunc );

        InlineFunctionInfo IFI;
        if( !InlineFunction( call, IFI ) )
            success = false;
    }

    return success;
}

//------------------------------------------------------------------------------
// Inline "<dstFuncName>_<suffix>" into the calls to dstFunc from func.
static void inlineFuncVariant( Function* func, const std::string& dstFuncName, const std::string& suffix )
{
    inlineFunc( func, dstFuncName, dstFuncName + "_" + suffix );
}

//------------------------------------------------------------------------------
// Finds the highest version of smFunc in the module not greater than smVersion
// and inlines it into ilwocations of smFunc from func. Each version of smFunc has
// the suffix  "_default" or "_smXx" where "X" and "x" are the major and minor
// SM versions respectively.
static void inlineSmFunc( Function* func, const std::string& smFuncName, unsigned smVersion )
{
    Module*   module  = func->getParent();
    Function* srcFunc = module->getFunction( smFuncName + "_default" );

    // search for SM versions
    unsigned           bestVersion = 0;
    const std::string& baseName    = smFuncName + "_sm";
    for( Module::iterator F = module->begin(), FE = module->end(); F != FE; ++F )
    {
        StringRef name = F->getName();
        if( !name.startswith( baseName ) )
            continue;
        unsigned version = atoi( name.substr( baseName.length() ).str().c_str() );
        if( bestVersion < version && version <= smVersion )
        {
            bestVersion = version;
            srcFunc     = &*F;
        }
    }

    if( !srcFunc )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Missing implementation for function ", smFuncName );

    inlineFunc( func, smFuncName, srcFunc->getName().str() );
}

//------------------------------------------------------------------------------
llvm::Function* optix::createTraverserFunc( const TraverserParams& p, LLVMManager* llvmManager )
{
    Module* runtimeToClone = llvmManager->getTraverserRuntime();
    Module* runtime        = CloneModule( *runtimeToClone ).release();

    std::vector<std::string> parts = tokenize( p.traverserName, "_" );

    Function* func = nullptr;
    if( parts.size() <= 1 )
    {
        // do nothing - error out below
    }
    else if( parts[0] == "bounds" )
    {
        std::string funcName = p.traverserName;
        if( parts[1] == "noaccel" )
            funcName = funcName + "_" + ( p.isGeom ? "geometry" : "group" );
        else if( parts[1] == "bvh" || parts[1] == "rtcbvh" )
            funcName = funcName + "_" + ( p.hasMotion ? "motion" : "nomotion" );
        func         = runtime->getFunction( funcName );
        RT_ASSERT_MSG( func != nullptr, "Bounds function not found: " + funcName );
        stripAllBut( func, false /*resetCallingColw*/ );
    }
    else if( parts[1] == "noaccel" )
    {
        std::string funcName = p.traverserName + "_" + ( p.isGeom ? "geometry" : "group" );
        func                 = runtime->getFunction( funcName );
        stripAllBut( func, false /*resetCallingColw*/ );
    }
    else if( ( parts[1] == "bvh8" || parts[1] == "bvh" ) && parts.size() >= 2 )
    {
        // Pick the base function based on loop type.
        const std::string&       loopType = parts[2];
        std::string              funcName;
        std::vector<const char*> smFuncNames;
        if( parts[1] == "bvh" )
        {
            funcName = "traverse_bvh_" + loopType;
            if( p.hasMotion )
                funcName += "_with_motion";  // motion bvh
            smFuncNames = {"raySpanMin", "raySpanMax"};
        }
        else
        {
            funcName = "traverse_bvh8_" + loopType;
        }

        func = runtime->getFunction( funcName );
        if( !func )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Bad loop type: ", loopType );

        // Specialize based on leaf node type
        std::string leafType = p.isGeom ? "geometry" : "group";
        inlineFuncVariant( func, "intersectLeaf", leafType );
        funcName += "_" + leafType;

        // Specialize GeometryInstance offset type
        std::string giOffsetType = p.bakedChildPtrs ? "baked" : "default";
        inlineFuncVariant( func, "getGiOffset", giOffsetType );
        funcName += +"_gi" + giOffsetType;

        // Specialize based on leaf range type
        const char* rangeType = "contiguous";
        inlineFuncVariant( func, "getLeafRange", rangeType );

        // Specialize based on SM version
        for( const char* name : smFuncNames )
            inlineSmFunc( func, name, p.smVersion );
        funcName += "_sm" + std::to_string( p.smVersion );

        int smemStackBytes = ( p.isGeom && p.smVersion >= 50 ) ? k_smemStackBytes.get() : 0;
        setInt32GlobalInitializer( "SMEM_STACK_BYTES", smemStackBytes, func->getParent() );

        // Set the name
        func->setName( funcName );

        stripAllBut( func, false /*resetCallingColw*/ );

        // Colwert the stack to local
        LocalMemorySpaceOpt opt;
        opt.addValueName( "localStack" );
        opt.runOnFunction( *func );
    }

    if( !func )
        throw IlwalidValue( RT_EXCEPTION_INFO, "No traverser function found." );

    return func;
}
