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

#include <FrontEnd/PTX/Canonical/C14n.h>

#include <Compile/AttributeUtil.h>
#include <Compile/FindAttributeSegments.h>
#include <Compile/SaveSetOptimizer.h>
#include <Compile/TextureLookup.h>
#include <Compile/Utils.h>
#include <Context/Context.h>
#include <Context/LLVMManager.h>
#include <Context/ObjectManager.h>
#include <Context/ProgramManager.h>
#include <Context/UpdateManager.h>
#include <ExelwtionStrategy/Compile.h>  // Regex parsing of getBufferElement() etc.
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <FrontEnd/Canonical/FrontEndHelpers.h>
#include <FrontEnd/Canonical/GetSetOptimization.h>
#include <FrontEnd/Canonical/IntrinsicsManager.h>
#include <FrontEnd/Canonical/Mangle.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <FrontEnd/Canonical/VariableSemantic.h>
#include <FrontEnd/PTX/Canonical/UberPointer.h>
#include <Objects/VariableType.h>
#include <Util/ContainerAlgorithm.h>

#include <internal/optix_declarations.h>
#include <internal/optix_defines.h>


#include <exp/context/ErrorHandling.h>
#include <optix_types.h>

#include <corelib/compiler/CoreIRBuilder.h>
#include <corelib/compiler/LLVMUtil.h>
#include <corelib/compiler/LiveValues.h>
#include <corelib/compiler/LWVMAddressSpaces.h>
#include <corelib/math/MathUtil.h>
#include <corelib/misc/String.h>

#include <corelib/misc/String.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/CompileError.h>
#include <prodlib/system/Knobs.h>

#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/Analysis/Passes.h>  // createTypeBasedAliasAnalysisPass()
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/CallSite.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>  // MemSetInst
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <sstream>

// Print a value to error stream
#define P( val )                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        llvm::errs() << #val << ": " << ( val ) << '\n';                                                               \
    } while( 0 )

/*
 * Canonicalization colwerts the raw-colwerted LLVM to a standard form
 * for subsequent processing.
 *
 * The canonical form of a program is defined by:
 *
 * User variables: fetched with getVariableValue.*.  Default value is
 * represented as llvm GlobalVariables with an optional initializer.
 *
 * Annotations on user variables: represented as LLVM metadata.  Since
 * global values cannot have metadata, we use named metadata with two
 * elements in the tuple 1. The reference to the user variable, and
 * 2. The string annotation.
 *
 * NOTE: Please keep this list up to date as Canonicalization evolves.
 *
 */

using namespace optix;
using namespace llvm;
using namespace corelib;
using namespace prodlib;


namespace {
// clang-format off
Knob<bool>        k_saveLLVM                   ( RT_DSTRING("c14n.saveLLVM"),                                false, RT_DSTRING( "Save LLVM in a file cpN.ll where N is the canonical program ID" ) );
Knob<int>         k_optLevel                   ( RT_DSTRING("c14n.optlevel"),                                2,     RT_DSTRING( "C14n post-canonicalization optimization level" ) );
Knob<bool>        k_enableC14nLICM             ( RT_DSTRING("c14n.enableLICM"),                              false, RT_DSTRING( "Perform Loop Ilwariant Code Motion (LICM) optimization" ) );
Knob<bool>        k_enableGetSetOpt            ( RT_DSTRING("c14n.enableGetSetOpt"),                         true,  RT_DSTRING( "Enable optimization of buffer/variable/attribute/payload get and set calls." ) );
Knob<bool>        k_onlyLocalGetSetOpt         ( RT_DSTRING("c14n.onlyLocalGetSetOpt"),                      false, RT_DSTRING( "Disable control-flow dependent get/set optimization, i.e., only optimize calls within the same basic block." ) );
Knob<bool>        k_disableTerminateRay        ( RT_DSTRING("compile.disableTerminateRay"),                  false, RT_DSTRING( "Ignore all calls to rtTerminateRay." ) );
Knob<bool>        k_disableIgnoreIntersection  ( RT_DSTRING("compile.disableIgnoreIntersection"),            false, RT_DSTRING( "Ignore all calls to rtIgnoreIntersection." ) );
Knob<std::string> k_attributeOptimizer         ( RT_DSTRING("canonical.attributeOptimizer"),                 "valueWeb", RT_DSTRING( "Optimize deferred attribute continuation" ) );
Knob<bool>        k_deferAttributes            ( RT_DSTRING("compile.deferAttributes"),                      false, RT_DSTRING( "Generate code to defer attributes" ) );
// clang-format on
}  // namespace

// Support functions.
// -----------------------------------------------------------------------------
static unsigned int deducePointeeSize( CallInst* call );
static Value* getUseSkipCasts( Value* val );
static Value* getDefSkipCasts( Value* val );
static void nameBasicBlocks( Function* function );
static bool hasDynamicPayloadAccesses( Function* function );

static void dumpAsm( llvm::Module*      module,
                     CanonicalProgramID id,
                     const std::string& funcName = "",
                     const char*        suffix   = "" )
{
#if defined( DEBUG ) || defined( DEVELOP )
    if( k_saveLLVM.get() )
    {
        std::ostringstream filename;
        filename << "cp-" << std::setfill( '0' ) << std::setw( 4 ) << id << "-" << funcName << suffix << ".ll";
        lprint << "Writing LLVM ASM file to: " << filename.str() << "\n";
        saveModuleToAsmFile( module, filename.str() );
    }
#endif
}


static void dumpAsm( llvm::Function* func, CanonicalProgramID id, const char* suffix = "" )
{
    dumpAsm( func->getParent(), id, func->getName().str(), suffix );
}

//------------------------------------------------------------------------------
static GlobalVariable* getGlobalVariable( Value* val, const DataLayout& dataLayout )
{
    if( isa<Constant>( val ) )
    {
        Constant*    C  = cast<Constant>( val );
        GlobalValue* GV = nullptr;
        APInt        offset;
        bool         isConstOffset = corelib::isConstantOffsetFromGlobal( C, GV, offset, dataLayout );

        if( isConstOffset && offset == 0 )
            return dyn_cast<GlobalVariable>( GV );
    }

    return nullptr;
}

//------------------------------------------------------------------------------
static GlobalVariable* getGlobalVariableForRTState( CallInst* CI, int argPos, const DataLayout& dataLayout, const std::string& stateType )
{
    GlobalVariable* gv = getGlobalVariable( CI->getArgOperand( argPos ), dataLayout );
    if( !gv )
    {
        Function*         caller         = CI->getParent()->getParent();
        const std::string callerFuncName = caller->getName();
        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ),
                            "OptiX state access (" + stateType + ") failed in function (" + callerFuncName + ")." );
    }

    return gv;
}


//------------------------------------------------------------------------------
static corelib::AddressSpace getAddressSpace( Value* val )
{
    SmallVector<Value*, 4> Worklist;
    SmallSet<Value*, 16>   Visited;

    //errs() << "****************\n"<<"val = "<<*val<<"\n";
    Worklist.push_back( val );
    corelib::AddressSpace address_space = corelib::ADDRESS_SPACE_UNKNOWN;
    while( !Worklist.empty() )
    {
        Value* V  = Worklist.pop_back_val();
        Type*  ty = V->getType();
        //errs() << "V = "<<*V<<"\n  type = "<<*ty<<"\n";
        if( ty->isPointerTy() )
        {
            corelib::AddressSpace lwrrent_addrspace = static_cast<corelib::AddressSpace>( ty->getPointerAddressSpace() );
            //errs() << "*** Found address_space "<<lwrrent_addrspace<<" ***\n";
            if( address_space == corelib::ADDRESS_SPACE_UNKNOWN )
            {
                address_space = lwrrent_addrspace;
            }
            else if( address_space != lwrrent_addrspace )
            {
                std::ostringstream o;
                o << "Address space mismatch: other (" << address_space << ") != current (" << lwrrent_addrspace << ")";
                throw CompileError( RT_EXCEPTION_INFO, corelib::LLVMErrorInfo( V ), o.str() );
            }
        }
        else if( dyn_cast<LoadInst>( V ) )
        {
            // Don't look at the uses of this instruction since we can't look through memory to determine pointer space
        }
        else if( dyn_cast<Argument>( V ) )
        {
            corelib::AddressSpace lwrrent_addrspace = corelib::ADDRESS_SPACE_GENERIC;
            if( address_space == corelib::ADDRESS_SPACE_UNKNOWN )
            {
                address_space = lwrrent_addrspace;
            }
            else if( address_space != lwrrent_addrspace )
            {
                std::ostringstream o;
                o << "Address space mismatch: other (" << address_space << ") != current (" << lwrrent_addrspace << ")";
                throw CompileError( RT_EXCEPTION_INFO, corelib::LLVMErrorInfo( V ), o.str() );
            }
        }
        else if( User* U = dyn_cast<User>( V ) )
        {
            //errs() << "\t"<<"U = "<<*U<<"\n";
            for( unsigned i = 0, IS = U->getNumOperands(); i < IS; ++i )
            {
                Value* operand = U->getOperand( i );
                //errs() << "\t\t" << "operand["<<i<<"] = "<<*operand<<"\n";
                if( std::get<1>( Visited.insert( operand ) ) )
                    Worklist.push_back( operand );
            }
        }
        else
        {
            throw CompileError( RT_EXCEPTION_INFO, corelib::LLVMErrorInfo( V ),
                                "Unknown Value when trying to figure out pointer space for ray payload argument to "
                                "rt_trace" );
        }
    }
    return address_space;
}

// -----------------------------------------------------------------------------
C14n::C14n( llvm::Function*         function,
            CanonicalizationType    type,
            lwca::ComputeCapability targetMin,
            lwca::ComputeCapability targetMax,
            size_t                  ptxHash,
            Context*                context,
            LLVMManager*            llvmManager,
            ProgramManager*         programManager,
            ObjectManager*          objectManager )
    : m_function( function )
    , m_type( type )
    , m_context( context )
    , m_llvmManager( llvmManager )
    , m_programManager( programManager )
    , m_objectManager( objectManager )
{
    m_cp.reset( new CanonicalProgram( function->getName().str(), targetMin, targetMax, ptxHash, context ) );
    m_up.reset( new UberPointerSet( m_llvmManager ) );
}

C14n::~C14n()
{
}


void C14n::checkVariableSizes( llvm::GlobalVariable* G, const VariableType& vtype, unsigned int ptxSize )
{
    llvm::DataLayout DL( G->getParent() );
    llvm::Type*      type     = G->getType()->getPointerElementType();  // GlobalVariable is always a pointer type
    size_t           llvmSize = DL.getTypeStoreSize( type );
    size_t           apiSize  = vtype.computeSize();

    if( llvmSize != apiSize )
        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ), "Unexpected size for type" );
    if( llvmSize != ptxSize )
        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ), "Inconsistent sizes in PTX for type" );
}

VariableReference* C14n::getOrAddVariable( llvm::GlobalVariable* G, const VariableType& vtype, bool isAttribute )
{
    std::string name = G->getName().str();
    std::map<std::string, VariableReference*>::iterator iter = m_variableReferences.find( name );
    if( iter == m_variableReferences.end() )
    {
        // Add variable and token
        unsigned short token         = registerVariableName( name );
        bool           isInitialized = false;
        if( G->hasInitializer() && vtype.isTypeWithValidDefaultValue() )
            isInitialized      = true;
        std::string annotation = "";
        // FIXME: valgrind says that we have a memory leak here for the swimmingShark binary.
        // AM: it is not clear to me who is the owner of this pointer (i.e. who should call delete) since we store it
        // both in the private member variable m_variableReferences and in m_cp->m_variableReferences.
        // Smart pointers should be able to get rid of this confusion.
        VariableReference* varref = new VariableReference( m_cp.get(), name, token, vtype, isInitialized, annotation );
        m_variableReferences.insert( std::make_pair( name, varref ) );

        // Register the reference, which will assign an ID
        m_programManager->registerVariableReference( varref );

        // Add to the appropriate list in the canonical program
        if( isAttribute )
            m_cp->m_attributeReferences.push_back( varref );
        else
            m_cp->m_variableReferences.push_back( varref );

        // Remove from unprocessed list if applicable, add to toRemove list
        if( !isInitialized || isAttribute )
            m_globalsToRemove.push_back( G );

        // Reset iterator
        iter = m_variableReferences.find( name );
    }

    // Verify the match
    if( vtype != iter->second->m_vtype )
        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ), "Variable types are not consistent" );

    return iter->second;
}

void C14n::canonicalizeInternalRegister( llvm::GlobalVariable* G )
{
    // Originally these variables were intended to be a magic back
    // door to read and write ptx registers during internal
    // developent.  Unfortunately, some of them were exposed to the
    // outside via the exception mechanism, so we need to support
    // reading them for backward compatibility.
    //
    // Supported registers are:
    //   rayIndex_x, rayIndex_y, rayIndex_z
    //   exception_detail0-8
    //   exception_64_detail0-6
    // Note that other internal registers will not be used any
    // longer - they will be relegated to built-in functions.
    //
    // Make sure to keep this backward compability before replacing the
    // mechanism in optix_device.h.

    llvm::StringRef variableName = G->getName();
    Module*         module       = G->getParent();
    m_globalsToRemove.push_back( G );
    Type*        statePtrType = m_llvmManager->getStatePtrType();
    DataLayout   dataLayout( module );
    LLVMContext& llvmContext = module->getContext();

    if( variableName.startswith( "_ZN21rti_internal_register14reg_rayIndex" ) )
    {
        size_t initialOffset = 0;
        if( variableName == "_ZN21rti_internal_register14reg_rayIndex_xE" )
            initialOffset = 0;
        else if( variableName == "_ZN21rti_internal_register14reg_rayIndex_yE" )
            initialOffset = 4;
        else if( variableName == "_ZN21rti_internal_register14reg_rayIndex_zE" )
            initialOffset = 8;
        else
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ),
                                "Unknown rti_internal_register::reg_rayIndex variable" );

        llvm::Type*   uint3Type  = m_llvmManager->getUint3Type();
        Type*         argTypes[] = {statePtrType};
        FunctionType* fntype     = FunctionType::get( uint3Type, argTypes, false );
        size_t        eltSize    = dataLayout.getTypeStoreSize( fntype->getReturnType() );
        Constant*     getter     = module->getOrInsertFunction( "optixi_getLaunchIndex", fntype );
        Constant*     setter     = nullptr;
        int           upkind = getOrAddUberPointerKind( "launchIndex", getter, setter, nullptr, nullptr, UberPointer::PointeeType::LaunchIndex );

        UberPointerTransform upTransform( module, m_llvmManager, variableName, UberPointer::PointeeType::LaunchIndex );
        upTransform.translate( G, upkind, eltSize, initialOffset );
        if( upTransform.hasStores() )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ), "Writes found to read-only variable" );
    }
    else if( variableName.startswith( "_ZN21rti_internal_register21reg_exception_detail" ) )
    {
        llvm::StringRef ntext = variableName.drop_front( 48 ).drop_back( 1 );
        unsigned int    n     = 0;
        if( ntext.getAsInteger( 10, n ) || n > 8 )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ), "Invalid access to optix exception register" );

        Type*         argTypes[] = {statePtrType, Type::getInt32Ty( llvmContext )};
        FunctionType* fntype     = FunctionType::get( llvm::Type::getInt32Ty( llvmContext ), argTypes, false );
        Constant*     getter     = module->getOrInsertFunction( "optixi_getExceptionDetail", fntype );
        Constant*     setter     = nullptr;
        int           upkind     = getOrAddUberPointerKind( "exception_detail", getter, setter, nullptr, nullptr,
                                              UberPointer::PointeeType::ExceptionDetail );
        size_t               eltSize = dataLayout.getTypeStoreSize( fntype->getReturnType() );
        UberPointerTransform upTransform( module, m_llvmManager, variableName, UberPointer::PointeeType::ExceptionDetail );
        upTransform.translate( G, upkind, makeInt32( n, llvmContext ), eltSize, 0 );
        if( upTransform.hasStores() )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ), "Writes found to read-only variable" );
    }
    else if( variableName.startswith( "_ZN21rti_internal_register24reg_exception_64_detail" ) )
    {
        llvm::StringRef ntext = variableName.drop_front( 51 ).drop_back( 1 );
        unsigned int    n     = 0;
        if( ntext.getAsInteger( 10, n ) || n > 6 )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ),
                                "Invalid access to optix exception register (64-bit)" );

        Type*         argTypes[] = {statePtrType, Type::getInt32Ty( llvmContext )};
        FunctionType* fntype     = FunctionType::get( Type::getInt64Ty( llvmContext ), argTypes, false );
        Constant*     getter     = module->getOrInsertFunction( "optixi_getExceptionDetail64", fntype );
        Constant*     setter     = nullptr;
        int           upkind     = getOrAddUberPointerKind( "exception_detail64", getter, setter, nullptr, nullptr,
                                              UberPointer::PointeeType::ExceptionDetail );
        size_t eltSize = dataLayout.getTypeStoreSize( fntype->getReturnType() );

        UberPointerTransform upTransform( module, m_llvmManager, variableName, UberPointer::PointeeType::ExceptionDetail );
        upTransform.translate( G, upkind, makeInt32( n, llvmContext ), eltSize, 0 );
        if( upTransform.hasStores() )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ), "Writes found to read-only variable" );
    }
    else
    {
        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ), "Invalid access to optix internal register" );
    }
}

// Replace loads with call to canonical getter
//
void C14n::canonicalizeInstanceVariable( llvm::GlobalVariable* G,
                                         const std::string&    semantic,
                                         unsigned int          ptxSize,
                                         VariableType          vtype,
                                         unsigned int          typeenum,
                                         const std::string&    annotation )
{
    // Check for callable program variables.  Lwrrently typeenum can be
    // either RT_FORMAT_PROGRAM_ID or RT_FORMAT_UNKNOWN.
    if( typeenum == rti_internal_typeinfo::_OPTIX_TYPE_ENUM_PROGRAM_ID )
    {
        if( vtype.baseType() != VariableType::UserData )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ), "Inconsistent type specifiers for program id" );
        vtype = VariableType( VariableType::ProgramId, 1 );
    }
    else if( typeenum == rti_internal_typeinfo::_OPTIX_TYPE_ENUM_PROGRAM_AS_ID )
    {
        vtype = VariableType::createForProgramVariable();
    }

    // Check that the symbol size matches what we expect
    checkVariableSizes( G, vtype, ptxSize );

    Module*     module = G->getParent();
    std::string name   = G->getName().str();
    if( m_variableReferences.find( name ) != m_variableReferences.end() )
        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ), "Variable name is not unique" );

    if( G->use_empty() )
        return;

    // Override the type with the one defined in OptiX because lwcc
    // generates [12 x i8] instead of [3 x float].
    Type* valueType = G->getType()->getPointerElementType();
    valueType       = getCleanType( vtype, valueType, m_llvmManager->getOptixRayType() );
    DataLayout dataLayout( G->getParent() );
    size_t     elementSize = dataLayout.getTypeStoreSize( valueType );

    // Add the Variable
    VariableReference* vref = getOrAddVariable( G, vtype );
    vref->m_annotation      = annotation;

    // Construct the getter
    std::string   getterName   = "optixi_getVariableValue." + vref->getUniversallyUniqueName();
    Type*         statePtrType = m_llvmManager->getStatePtrType();
    Type*         paramTypes[] = {statePtrType, valueType->getPointerTo()};
    FunctionType* fntype       = FunctionType::get( valueType, paramTypes, false );
    Constant*     getter       = module->getOrInsertFunction( getterName, fntype );
    Constant*     setter       = nullptr;

    // Translate the pointer
    llvm::Constant*      genericVal   = llvm::ConstantExpr::getAddrSpaceCast( G, valueType->getPointerTo() );
    llvm::Value*         defaultValue = llvm::ConstantExpr::getBitCast( genericVal, valueType->getPointerTo() );
    int                  upkind = getOrAddUberPointerKind( name, getter, setter, nullptr, nullptr, UberPointer::PointeeType::Variable, defaultValue );
    UberPointerTransform upTransform( module, m_llvmManager, name, UberPointer::PointeeType::Variable );
    upTransform.translate( G, upkind, elementSize, 0 );

    if( upTransform.hasStores() )
        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ), "Writes found to read-only variable" );

    // Mark the loaded values as users of this variable
    for( Value* load : upTransform.getLoads() )
    {
        // Additional check necessary for bound callable programs.
        if( vtype.baseType() == VariableType::Program )
        {
            // Only one use of the value
            if( load->getNumUses() != 1 )
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( load ),
                                    "More than one use of a bound callable program variable (" + name + ") found." );
            // Single use must be _rt_callable_program_from_id_64
            Value*    use  = getUseSkipCasts( load->user_back() );
            CallInst* call = dyn_cast<CallInst>( use );
            if( !call || call->getCalledFunction() == nullptr
                || ( call->getCalledFunction()->getName() != "_rt_callable_program_from_id_64"
                     && call->getCalledFunction()->getName() != "_rt_callable_program_from_id_v2_64" ) )
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( use ),
                                    "Use of bound callable program " + name
                                        + " that wasn't a call to _rt_callable_program_from_id_64" );
        }
        m_variableValues.insert( std::make_pair( load, vref ) );
    }
}

void C14n::canonicalizeSemanticVariable( llvm::GlobalVariable* G,
                                         const std::string&    semantic,
                                         unsigned int          ptxSize,
                                         const VariableType&   vtype,
                                         const std::string&    annotation )
{
    VariableSemantic stype = getVariableSemanticFromString( semantic, G->getName() );

    // Do some type checking and build the getter/setter functions
    Module*      module       = G->getParent();
    Type*        statePtrType = m_llvmManager->getStatePtrType();
    Type*        argTypes[]   = {statePtrType};
    LLVMContext& llvmContext  = module->getContext();
    std::string  name         = G->getName();

    DataLayout               dataLayout( G->getParent() );
    llvm::Constant*          getter      = nullptr;
    llvm::Constant*          setter      = nullptr;
    llvm::Constant*          getAddress  = nullptr;
    llvm::Type*              valueType   = nullptr;
    UberPointer::PointeeType pointeeType = UberPointer::PointeeType::Unknown;
    switch( stype )
    {
        case VS_PAYLOAD:
        {
            pointeeType = UberPointer::PointeeType::Payload;
            valueType   = G->getType()->getPointerElementType();
            valueType   = getCleanType( vtype, valueType, m_llvmManager->getOptixRayType() );
            llvm::DataLayout DL( G->getParent() );
            unsigned int     eltSize = DL.getTypeStoreSize( valueType );

            FunctionType*      fntype = FunctionType::get( valueType, argTypes, false );
            const std::string& uuname = m_cp->getUniversallyUniqueName();
            std::string        getterName =
                "optixi_getPayloadValue.prd" + std::to_string( eltSize ) + "b." + uuname + "." + G->getName().str();
            getter = module->getOrInsertFunction( getterName, fntype );

            Type*         argTypes2[] = {statePtrType, valueType};
            FunctionType* fntype2     = FunctionType::get( Type::getVoidTy( llvmContext ), argTypes2, false );
            std::string   setterName =
                "optixi_setPayloadValue.prd" + std::to_string( eltSize ) + "b." + uuname + "." + G->getName().str();
            setter = module->getOrInsertFunction( setterName, fntype2 );

            getAddress = GetPayloadAddressCall::createFunction( m_llvmManager, module );

            m_cp->markHasPayloadAccesses();
        }
        break;
        case VS_ATTRIBUTE:
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ), "Invalid semantic type" );
        case VS_LAUNCHINDEX:
        {
            pointeeType          = UberPointer::PointeeType::LaunchIndex;
            valueType            = m_llvmManager->getUint3Type();
            FunctionType* fntype = FunctionType::get( valueType, argTypes, false );
            name                 = "launchIndex";
            getter               = module->getOrInsertFunction( "optixi_getLaunchIndex", fntype );
            if( vtype.baseType() != VariableType::Int && vtype.baseType() != VariableType::Uint )
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ), "Invalid variable type with semantic " + semantic );

            m_cp->markHasLaunchIndexAccesses();
        }
        break;
        case VS_LAUNCHDIM:
        {
            pointeeType          = UberPointer::PointeeType::LaunchDim;
            valueType            = m_llvmManager->getUint3Type();
            FunctionType* fntype = FunctionType::get( valueType, argTypes, false );
            name                 = "launchDim";
            getter               = module->getOrInsertFunction( "optixi_getLaunchDim", fntype );
            if( vtype.baseType() != VariableType::Int && vtype.baseType() != VariableType::Uint )
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ), "Invalid variable type with semantic " + semantic );
        }
        break;
        case VS_LWRRENTRAY:
        {
            pointeeType          = UberPointer::PointeeType::LwrrentRay;
            valueType            = m_llvmManager->getOptixRayType();
            FunctionType* fntype = FunctionType::get( valueType, argTypes, false );
            getter               = module->getOrInsertFunction( "optixi_getLwrrentRay", fntype );
            name                 = "lwrrentRay";
            m_cp->markAccessesLwrrentRay();
            if( vtype != VariableType( VariableType::Ray, 1 ) )
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ),
                                    "Invalid variable type with semantic rtLwrrentRay" );
        }
        break;
        case VS_LWRRENTTIME:
        {
            pointeeType          = UberPointer::PointeeType::LwrrentTime;
            valueType            = llvm::Type::getFloatTy( llvmContext );
            FunctionType* fntype = FunctionType::get( valueType, argTypes, false );
            getter               = module->getOrInsertFunction( "optixi_getLwrrentTime", fntype );
            name                 = "lwrrentTime";
            m_cp->markAccessesLwrrentTime();
            if( vtype != VariableType( VariableType::Float, 1 ) )
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ),
                                    "Expected variable with semantic rtLwrrentTime to be of type float" );
        }
        break;
        case VS_INTERSECTIONDISTANCE:
        {
            pointeeType          = UberPointer::PointeeType::TMax;
            valueType            = llvm::Type::getFloatTy( llvmContext );
            FunctionType* fntype = FunctionType::get( valueType, argTypes, false );
            getter               = module->getOrInsertFunction( "optixi_getLwrrentTmax", fntype );
            name                 = "tmax";
            m_cp->markAccessesIntersectionDistance();
            if( vtype != VariableType( VariableType::Float, 1 ) )
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ),
                                    "Invalid variable type with semantic rtIntersectionDistance" );
        }
        break;
        case VS_SUBFRAMEINDEX:
        {
            pointeeType          = UberPointer::PointeeType::SubframeIndex;
            valueType            = m_llvmManager->getI32Type();
            FunctionType* fntype = FunctionType::get( valueType, argTypes, false );
            name                 = "subframeIndex";
            getter               = module->getOrInsertFunction( "optixi_getSubframeIndex", fntype );
            if( vtype.baseType() != VariableType::Int && vtype.baseType() != VariableType::Uint )
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ), "Invalid variable type with semantic " + semantic );
        }
        break;
    }

    // Check that the symbol size matches what we expect
    checkVariableSizes( G, vtype, ptxSize );

    // Remove the global when we are done
    m_globalsToRemove.push_back( G );

    // Translate to an uberpointer reference
    RT_ASSERT( getter != nullptr );
    RT_ASSERT( pointeeType != UberPointer::PointeeType::Unknown );
    size_t eltSize = dataLayout.getTypeStoreSize( valueType );

    int                  upkind = getOrAddUberPointerKind( name, getter, setter, nullptr, getAddress, pointeeType );
    UberPointerTransform upTransform( module, m_llvmManager, G->getName(), pointeeType );
    upTransform.translate( G, upkind, eltSize, 0 );

    if( stype == VS_PAYLOAD )
    {
        // Handle properties for payload
        if( upTransform.pointerEscapes() )
            m_cp->markPayloadPointerMayEscape();
        if( upTransform.hasStores() )
            m_cp->markHasPayloadStores();
    }
    else
    {
        // Only VS_PAYLOAD can have writes
        if( upTransform.hasStores() )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( G ), "Writes found to read-only variable" );
    }
}

void C14n::canonicalizeAttribute( llvm::GlobalVariable* G,
                                  const std::string&    semantic,
                                  unsigned int          ptxSize,
                                  const VariableType&   vtype,
                                  const std::string&    annotation )
{
    // Add the variable to the list
    Module*    module = G->getParent();
    DataLayout dataLayout( module );
    // Replace "attribute " prefix and replace it with "_attribute_".
    std::string name = "_attribute_" + semantic.substr( 10, semantic.size() );
    // Rename the attribute with the semantic name of the attribute.
    // This is done to make sure that the variables that refer to the same attribute in different
    // programs have the same token.
    G->setName( name );
    VariableReference* vref = nullptr;
    // Check that the symbol size matches what we expect
    checkVariableSizes( G, vtype, ptxSize );
    auto varIter = m_variableReferences.find( name );
    if( varIter != m_variableReferences.end() )
    {
        // In case we already have an attribute with the given name use the associated variable reference.
        // This functionality is used by the kajiya sample.
        vref = varIter->second;
    }
    else
    {
        // Assign token and reference id, even if the attribute is unused
        vref                  = getOrAddVariable( G, vtype, /*isAttribute*/ true );
        vref->m_isInitialized = false;
    }

    Type* valueType           = G->getType()->getPointerElementType();
    valueType                 = getCleanType( vtype, valueType, m_llvmManager->getOptixRayType() );
    Type*        statePtrType = m_llvmManager->getStatePtrType();
    LLVMContext& llvmContext  = module->getContext();

    // Construct getter/setter
    Type*           argTypes1[] = {statePtrType};
    FunctionType*   fntype1     = FunctionType::get( valueType, argTypes1, false );
    std::string     getterName  = "optixi_getAttributeValue." + vref->getUniversallyUniqueName();
    llvm::Constant* getter      = module->getOrInsertFunction( getterName, fntype1 );

    Type*           argTypes2[] = {statePtrType, valueType};
    FunctionType*   fntype2     = FunctionType::get( Type::getVoidTy( llvmContext ), argTypes2, false );
    std::string     setterName  = "optixi_setAttributeValue." + vref->getUniversallyUniqueName();
    llvm::Constant* setter      = module->getOrInsertFunction( setterName, fntype2 );

    // Translate into an UberPointer reference
    int upkind = getOrAddUberPointerKind( name, getter, setter, nullptr, nullptr, UberPointer::PointeeType::Attribute );
    UberPointerTransform upTransform( module, m_llvmManager, name, UberPointer::PointeeType::Attribute );
    size_t               eltSize = dataLayout.getTypeStoreSize( valueType );
    upTransform.translate( G, upkind, eltSize, 0 );
    if( upTransform.hasStores() )
        m_cp->markHasAttributeStores();
    if( upTransform.hasLoads() )
        m_cp->markHasAttributeLoads();
}

const llvm::Constant* C14n::getInitializer( const llvm::Module* module, const std::string& name ) const
{
    std::string                 variableName = canonicalMangleVariableName( name );
    const llvm::GlobalVariable* GV           = module->getGlobalVariable( variableName, true );
    if( GV && GV->hasInitializer() )
        return GV->getInitializer();
    else
        return nullptr;
}

bool C14n::getString( const llvm::Module* module, const std::string& name, std::string& returnString ) const
{
    const llvm::Constant* C = getInitializer( module, name );
    if( !C )
        return false;

    // Empty string can be any type
    if( C->isZeroValue() )
    {
        returnString = "";
        return true;
    }

    if( const llvm::ConstantDataSequential* CDS = llvm::dyn_cast<const llvm::ConstantDataSequential>( C ) )
    {
        if( CDS->isCString() )
        {
            returnString = CDS->getAsCString();
            return true;
        }
    }

    throw CompileError( RT_EXCEPTION_INFO, "Unhandled variable initializer" );
}

bool C14n::getTypeInfo( const llvm::Module* module, const std::string& name, unsigned int& returnKind, unsigned int& returnSize ) const
{
    const llvm::Constant* C = getInitializer( module, name );
    if( !C )
        return false;

    if( const llvm::ConstantDataArray* CDA = llvm::dyn_cast<const llvm::ConstantDataArray>( C ) )
    {
        size_t totalSize = CDA->getNumElements() * CDA->getElementByteSize();
        if( totalSize != 8 )
            throw CompileError( RT_EXCEPTION_INFO, "Malformed type info" );

        llvm::StringRef data = CDA->getRawDataValues();
        RT_ASSERT( data.size() == 8 );
        memcpy( &returnKind, data.data() + 0, 4 );
        memcpy( &returnSize, data.data() + 4, 4 );
        return true;
    }

    throw CompileError( RT_EXCEPTION_INFO, "Unhandled variable initializer" );
}

bool C14n::getInt( const llvm::Module* module, const std::string& name, unsigned int& returnKind ) const
{
    const llvm::Constant* C = getInitializer( module, name );
    if( !C )
        return false;

    if( const llvm::ConstantDataArray* CDA = llvm::dyn_cast<const llvm::ConstantDataArray>( C ) )
    {
        size_t totalSize = CDA->getNumElements() * CDA->getElementByteSize();
        if( totalSize != 4 )
            throw CompileError( RT_EXCEPTION_INFO, "Malformed type info" );

        llvm::StringRef data = CDA->getRawDataValues();
        RT_ASSERT( data.size() == 4 );
        memcpy( &returnKind, data.data() + 0, 4 );
        return true;
    }
    else if( const llvm::ConstantInt* CI = llvm::dyn_cast<const llvm::ConstantInt>( C ) )
    {
        if( CI->getBitWidth() <= 32 )
        {
            returnKind = CI->getZExtValue();
            return true;
        }
    }
    throw CompileError( RT_EXCEPTION_INFO, "Unhandled variable initializer" );
}

void C14n::canonicalizeVariables( llvm::Module* module, const llvm::Module* originalModule )
{
    /*
   * The optix header files will annotate optix variables with
   * aditional information.  These are represented at variables in
   * other "magic" namespaces.
   *
   * Data available per type:
   *  Variable:         typeinfo, typename, typeenum, semantic, annotation
   *  Callable program: typeinfo, typename,           semantic, annotation  // typename is return retype, semantic is "rt_call", annotation is parameters
   *  Buffer:           optional annotation
   *  TextureSampler:   optional annotation
   *
   */
    for( llvm::Module::global_iterator G = module->global_begin(), GE = module->global_end(); G != GE; ++G )
    {
        llvm::StringRef variableName = G->getName();
        if( variableName.startswith( "_ZN21rti_internal_register" ) )
        {
            canonicalizeInternalRegister( &*G );
        }
        else
        {
            // This is a regular global.  See what information we have about
            // it to determine if it is an optix variable
            std::string demangledName = canonicalDemangleVariableName( variableName );
            std::string semantic;
            bool        have_semantic =
                getString( originalModule, canonicalPrependNamespace( demangledName, "rti_internal_semantic::" ), semantic );
            if( have_semantic )
            {
                std::string  typename_string, annotation;
                unsigned int vkind, ptxSize, typeenum;
                bool         have_typeinfo =
                    getTypeInfo( originalModule, canonicalPrependNamespace( demangledName, "rti_internal_typeinfo::" ),
                                 vkind, ptxSize );
                bool have_typename =
                    getString( originalModule, canonicalPrependNamespace( demangledName, "rti_internal_typename::" ), typename_string );
                bool have_typeenum =
                    getInt( originalModule, canonicalPrependNamespace( demangledName, "rti_internal_typeenum::" ), typeenum );
                getString( originalModule, canonicalPrependNamespace( demangledName, "rti_internal_annotation::" ), annotation );

                // Prior to OptiX 3.6 programs did not have a typeenum.
                if( !have_typeenum )
                    typeenum = RT_FORMAT_UNKNOWN;

                if( !have_typeinfo || !have_typename )
                    throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( &*G ),
                                        "Malformed type annotations for variable: " + demangledName );

                // Check that kind is valid
                if( vkind != rti_internal_typeinfo::_OPTIX_VARIABLE )
                    throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( &*G ), "Invalid type kind" );

                // Parse typename_string
                VariableType vtype = parseTypename( typename_string );
                if( vtype.baseType() == VariableType::UserData )
                {
                    // User data matches the size of the LLVM variable
                    llvm::DataLayout DL( G->getParent() );
                    vtype = VariableType( VariableType::UserData, DL.getTypeStoreSize( G->getType()->getPointerElementType() ) );
                }
                // We have a declaration of a "rtVariable".  If it is empty
                // then it will be a regular optix variable, otherwise it will
                // be one of a handful of special optix variables.

                if( semantic.empty() )
                {
                    canonicalizeInstanceVariable( &*G, semantic, ptxSize, vtype, typeenum, annotation );
                }
                else if( semantic.compare( 0, 9, "attribute" ) == 0 )
                {
                    canonicalizeAttribute( &*G, semantic, ptxSize, vtype, annotation );
                }
                else
                {
                    canonicalizeSemanticVariable( &*G, semantic, ptxSize, vtype, annotation );
                }
            }
        }
    }
}

C14n::TraceVariant C14n::getTraceVariant( const llvm::StringRef& name )
{
    if( name.equals( "_rt_trace_64" ) )
        return TraceVariant::PLAIN;
    if( name.equals( "_rt_trace_with_time_64" ) )
        return TraceVariant::TIME;
    if( name.equals( "_rt_trace_mask_flags_64" ) )
        return TraceVariant::MASK_FLAGS;
    if( name.equals( "_rt_trace_time_mask_flags_64" ) )
        return TraceVariant::TIME_MASK_FLAGS;

    RT_ASSERT( !"invalid trace function name" );
    return TraceVariant::PLAIN;
}

size_t C14n::getParameterCount( const TraceVariant variant )
{
    switch( variant )
    {
        case TraceVariant::PLAIN:
            return 12;
        case TraceVariant::TIME:
            return 13;
        case TraceVariant::MASK_FLAGS:
            return 14;
        case TraceVariant::TIME_MASK_FLAGS:
            return 15;
    }

    RT_ASSERT( !"invalid trace function variant" );
    return 0;
}

bool C14n::hasTime( const TraceVariant variant )
{
    return TraceVariant::TIME_MASK_FLAGS == variant || TraceVariant::TIME == variant;
}

bool C14n::hasMaskAndFlags( const TraceVariant variant )
{
    return TraceVariant::MASK_FLAGS == variant || TraceVariant::TIME_MASK_FLAGS == variant;
}

/*
 * This function will canonicalize any complex function that changes
 * control flow or requires additional information about the IR.  Most
 * functions can be canonicalized by simply linking a stub into
 * C14nRuntime.ll.
 */
void C14n::canonicalizeComplexFunctions( llvm::Module* module )
{
    LLVMContext&        llvmContext = module->getContext();
    std::vector<Value*> toDelete;

    // Get common types
    Type* i8Ty       = m_llvmManager->getI8Type();
    Type* i32Ty      = m_llvmManager->getI32Type();
    Type* u64Ty      = m_llvmManager->getI64Type();
    Type* f32Ty      = m_llvmManager->getFloatType();
    Type* statePtrTy = m_llvmManager->getStatePtrType();
    Type* rmTy       = TraceGlobalPayloadBuilder::getRayMaskType( llvmContext );
    Type* rfTy       = TraceGlobalPayloadBuilder::getRayFlagsType( llvmContext );

    // _rt_trace_64 and variants
    Function* traces[] = {module->getFunction( "_rt_trace_64" ), module->getFunction( "_rt_trace_with_time_64" ),
                          module->getFunction( "_rt_trace_mask_flags_64" ),
                          module->getFunction( "_rt_trace_time_mask_flags_64" )};
    for( Function* fn : traces )
    {
        if( !fn )
            continue;

        const TraceVariant variant           = getTraceVariant( fn->getName() );
        const bool         fnHasTime         = hasTime( variant );
        const bool         fnHasMaskAndFlags = hasMaskAndFlags( variant );
        const size_t       parCount          = getParameterCount( variant );
        // PRD and size are the last two parameters in all _rt_trace_xzy variants
        const int prdArgIndex = parCount - 2;

        if( ( fn->arg_size() != parCount ) || fn->isVarArg() )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( fn ), "Malformed call to " + fn->getName().str() );

        m_cp->markCallsTrace();

        m_globalsToRemove.push_back( fn );

        unsigned int maxPayloadSize          = m_cp->getMaxPayloadSize();
        unsigned int maxPayloadRegisterCount = m_cp->getMaxPayloadRegisterCount();

        for( CallInst* CI : getCallsToFunction( fn ) )
        {
            corelib::CoreIRBuilder irb{CI};

            // Get the size of the payload and update the max size.
            unsigned int eltSizeInBytes = getConstantValueOrAssert( CI->getArgOperand( prdArgIndex + 1 ) );
            maxPayloadSize              = std::max( maxPayloadSize, eltSizeInBytes );
            unsigned int numRegs        = ( eltSizeInBytes + 3 ) / 4;
            maxPayloadRegisterCount     = std::max( maxPayloadRegisterCount, numRegs );

            // Get the PRD pointer.  It comes as an integer but we do not know which address space it contains.  Figure it out.
            Value* prdArg = CI->getArgOperand( prdArgIndex );
            RT_ASSERT( prdArg->getType() == u64Ty );

            AddressSpace addressSpace = getAddressSpace( prdArg );
            if( addressSpace == ADDRESS_SPACE_UNKNOWN )
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( prdArg ),
                                    "Unable to determine address space for ray payload argument to rt_trace" );

            if( addressSpace == ADDRESS_SPACE_LOCAL )
            {
                // Old PTX as in the path_tracer_test0 trace does not colwert the
                // payload pointer to generic address space. Leaving it as local results
                // in SASS that masks out everything but the lowest 24bits of the
                // address, which leads to IllegalAddress errors when accessing the
                // pointer later when RTX can't promote the payload to registers (see OP-1429).
                // We work around this by adding the colwersion to the generic
                // address space ourselves.
                Type*  localPrdType   = PointerType::get( i8Ty, ADDRESS_SPACE_LOCAL );
                Value* prdPtr         = irb.CreateIntToPtr( prdArg, localPrdType );
                Type*  genericPrdType = PointerType::get( i8Ty, ADDRESS_SPACE_GENERIC );
                Value* prdGenericPtr  = irb.CreateAddrSpaceCast( prdPtr, genericPrdType );
                prdArg                = irb.CreatePtrToInt( prdGenericPtr, u64Ty );
                addressSpace          = ADDRESS_SPACE_GENERIC;
            }

            // Build the value type and the function prototype.
            // We do not use getCleanType() for the payload type to not confuse the optimizer
            // or get into trouble with alignment (e.g. vector types are assumed to have
            // alignment 16 if not explicitly specified).
            llvm::Type*        prdElementType = ArrayType::get( Type::getInt8Ty( llvmContext ), eltSizeInBytes );
            llvm::Type*        prdType        = prdElementType->getPointerTo( addressSpace );
            const std::string& uuname         = m_cp->getUniversallyUniqueName();
            std::string        funcName   = "optixi_trace." + uuname + ".prd" + std::to_string( eltSizeInBytes ) + "b";
            Type*              argTypes[] = {statePtrTy, i32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty,
                                i32Ty,      f32Ty, f32Ty, f32Ty, i32Ty, rmTy,  rfTy,  prdType};
            FunctionType* funType = FunctionType::get( Type::getVoidTy( llvmContext ), argTypes, false );
            Function*     cfun    = dyn_cast<Function>( module->getOrInsertFunction( funcName, funType ) );

            cfun->addFnAttr( Attribute::AlwaysInline );

            // Replace the call
            Value* statePtr = CI->getParent()->getParent()->arg_begin();
            int    operand  = 0;
            Value* node     = CI->getArgOperand( operand++ );
            Value* ox       = CI->getArgOperand( operand++ );
            Value* oy       = CI->getArgOperand( operand++ );
            Value* oz       = CI->getArgOperand( operand++ );
            Value* dx       = CI->getArgOperand( operand++ );
            Value* dy       = CI->getArgOperand( operand++ );
            Value* dz       = CI->getArgOperand( operand++ );
            Value* rayType  = CI->getArgOperand( operand++ );
            Value* tmin     = CI->getArgOperand( operand++ );
            Value* tmax     = CI->getArgOperand( operand++ );
            Value* time     = fnHasTime ? CI->getArgOperand( operand++ ) : ConstantFP::getNullValue( f32Ty );
            Value* hasTime  = irb.getInt32( fnHasTime );
            Value* rayMask  = fnHasMaskAndFlags ? irb.CreateTrunc( CI->getArgOperand( operand++ ), rmTy ) :
                                                 TraceGlobalPayloadBuilder::getDefaultRayMask( llvmContext );
            Value* rayFlags = fnHasMaskAndFlags ? irb.CreateTrunc( CI->getArgOperand( operand++ ), rfTy ) :
                                                  TraceGlobalPayloadBuilder::getDefaultRayFlags( llvmContext );
            Value* prdptr = irb.CreateIntToPtr( prdArg, prdType, "payloadPtr" );
            Value* args[] = {statePtr, node, ox,   oy,   oz,      dx,      dy,       dz,
                             rayType,  tmin, tmax, time, hasTime, rayMask, rayFlags, prdptr};
            irb.CreateCall( cfun, args );

            RT_ASSERT( CI->use_empty() );
            CI->eraseFromParent();

            if( ConstantInt* rayTypeConstant = dyn_cast<ConstantInt>( rayType ) )
            {
                int rayTypeVal = static_cast<int>( rayTypeConstant->getSExtValue() );
                if( !m_cp->m_producesRayTypes.contains( rayTypeVal ) )
                    m_cp->m_producesRayTypes.addOrRemoveProperty( rayTypeVal, true );
            }
            else
            {
                if( m_context )
                {
                    ureport1( m_context->getUsageReport(), "USAGE HINT" )
                        << "For improved performance and memory overhead, rtTrace function parameter 'rayType' should "
                           "be specified as int literal or const local variable when possible."
                        << std::endl;
                }

                m_cp->markTracesUnknownRayType();
            }

            if( fnHasTime )
            {
                m_cp->markTraceHasTime();
            }
        }

        m_cp->setMaxPayloadSize( maxPayloadSize );
        m_cp->setMaxPayloadRegisterCount( maxPayloadRegisterCount );
    }

    if( module->getFunction( "_rt_throw" ) )
    {
        m_cp->markCallsThrow();
    }

    if( module->getFunction( "_rt_intersect_child" ) )
    {
        m_cp->markCallsIntersectChild();
    }

    if( module->getFunction( "_rt_potential_intersection" ) )
    {
        m_cp->markCallsPotentialIntersection();
    }

    if( module->getFunction( "_rt_report_intersection" ) )
    {
        m_cp->markCallsReportIntersection();
    }

    if( module->getFunction( "_rti_report_full_intersection_ff" ) )
    {
        m_cp->markIsBuiltInIntersection();
    }

    if( module->getFunction( "_rt_get_exception_code" ) )
    {
        m_cp->markCallsExceptionCode();
    }

    if( Function* fn = module->getFunction( "_rt_trace_64_global_payload" ) )
    {
        canonicalizeTraceGlobal( fn, toDelete, TraceVariant::PLAIN );
    }

    if( Function* fn = module->getFunction( "_rt_trace_with_time_64_global_payload" ) )
    {
        canonicalizeTraceGlobal( fn, toDelete, TraceVariant::TIME );
    }

    if( Function* fn = module->getFunction( "_rt_trace_mask_flags_64_global_payload" ) )
    {
        canonicalizeTraceGlobal( fn, toDelete, TraceVariant::MASK_FLAGS );
    }

    if( Function* fn = module->getFunction( "_rt_trace_time_mask_flags_64_global_payload" ) )
    {
        canonicalizeTraceGlobal( fn, toDelete, TraceVariant::TIME_MASK_FLAGS );
    }

    // _rt_buffer_get_size_64
    if( Function* fn = module->getFunction( "_rt_buffer_get_size_64" ) )
    {
        canonicalizeGetBufferSize( fn );
    }

    // _rt_buffer_get_64
    if( Function* fn = module->getFunction( "_rt_buffer_get_64" ) )
    {
        canonicalizeAccessBuffer( fn, toDelete );
    }

    if( Function* fn = module->getFunction( "_rt_load_or_request_64" ) )
    {
        canonicalizeLoadOrRequest( fn, toDelete );
    }

    // _rt_buffer_get_id_size_64
    if( Function* fn = module->getFunction( "_rt_buffer_get_id_size_64" ) )
    {
        canonicalizeGetBufferSizeFromId( fn );
    }
    // _rt_buffer_get_id_64
    if( Function* fn = module->getFunction( "_rt_buffer_get_id_64" ) )
    {
        canonicalizeAccessBufferFromId( fn, toDelete );
    }

    // _rt_terminate_ray
    if( Function* fn = module->getFunction( "_rt_terminate_ray" ) )
    {
        m_globalsToRemove.push_back( fn );
        canonicalizeTerminateRay( fn, statePtrTy, toDelete );
        if( !k_disableTerminateRay.get() )
        {
            m_cp->markCallsTerminateRay();
        }
    }

    // _rt_ignore_intersection
    if( Function* fn = module->getFunction( "_rt_ignore_intersection" ) )
    {
        m_globalsToRemove.push_back( fn );
        canonicalizeIgnoreIntersection( fn, toDelete );
        if( !k_disableIgnoreIntersection.get() )
        {
            m_cp->markCallsIgnoreIntersection();
        }
    }

    // llvm.lwvm.texsurf.handle.p1i64
    // llvm.lwvm.tex.*
    if( Function* fn = module->getFunction( "llvm.lwvm.texsurf.handle.p1i64" ) )
    {
        canonicalizeTextures( fn, toDelete );
    }

    // _rt_texture_get_*
    canonicalizeBindlessTextures( module, toDelete );

    // _rt_load_or_request_texture_get_*_id
    canonicalizeDemandLoadBindlessTextures( module, toDelete );

    // _rt_print_active
    if( Function* fn = module->getFunction( "_rt_print_active" ) )
    {
        canonicalizeFunctionPrintActive( module, fn, toDelete );
    }

    // _rt_print_start_64
    if( Function* fn = module->getFunction( "_rt_print_start_64" ) )
    {
        canonicalizeFunctionPrintStart( module, fn, toDelete );
    }

    // _rt_print_start
    if( Function* fn = module->getFunction( "_rt_print_start" ) )
    {
        canonicalizeFunctionPrintStart( module, fn, toDelete );
    }

    // _rt_print_write32
    if( Function* fn = module->getFunction( "_rt_print_write32" ) )
    {
        // these should have all been taken care of by the print_start
        // code so just delete it.
        if( fn->arg_size() != 2 || fn->isVarArg() )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( fn ), "Malformed call to " + fn->getName().str() );
        m_globalsToRemove.push_back( fn );
    }

    // _rti_comment
    for( Module::iterator F = module->begin(), FE = module->end(); F != FE; ++F )
    {
        if( !F->getName().startswith( "_rti_comment" ) )
            continue;

        m_globalsToRemove.push_back( &*F );
        RT_ASSERT( F->isDeclaration() );

        for( Value::user_iterator U = F->user_begin(), UE = F->user_end(); U != UE; ++U )
        {
            CallInst* CI = dyn_cast<CallInst>( *U );
            if( !CI )
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ), "Invalid use of function: " + F->getName().str() );
            RT_ASSERT( CI->getCalledFunction() == &*F );
            toDelete.push_back( CI );
        }
    }

    // _rt_get_primitive_index
    if( module->getFunction( "_rt_get_primitive_index" ) )
    {
        m_cp->markCallsGetPrimitiveIndex();
    }

    // _rt_is_triangle_hit, _rt_is_triangle_hit_back_face, _rt_is_triangle_hit_front_face
    if( module->getFunction( "_rt_is_triangle_hit" ) || module->getFunction( "_rt_is_triangle_hit_back_face" )
        || module->getFunction( "_rt_is_triangle_hit_front_face" ) )
    {
        m_cp->markAccessesHitKind();
    }

    // _rt_get_instance_flags
    if( module->getFunction( "_rti_get_instance_flags" ) )
    {
        m_cp->markCallsGetInstanceFlags();
    }

    // _rt_get_lowest_group_child_index
    if( module->getFunction( "_rt_get_lowest_group_child_index" ) )
    {
        m_cp->markCallsGetLowestGroupChildIndex();
    }

    // _rt_get_ray_flags
    if( module->getFunction( "_rt_get_ray_flags" ) )
    {
        m_cp->markCallsGetRayFlags();
    }

    // _rt_get_ray_mask
    if( module->getFunction( "_rt_get_ray_mask" ) )
    {
        m_cp->markCallsGetRayMask();
    }

    // _rt_get_transform
    if( Function* fn = module->getFunction( "_rt_get_transform" ) )
    {
        validateFunctionGetTransform( fn );
    }

    // _rt_transform_tuple
    if( Function* fn = module->getFunction( "_rt_transform_tuple" ) )
    {
        validateFunctionTransformTuple( fn );
    }

    // _rt_throw
    if( Function* fn = module->getFunction( "_rt_throw" ) )
    {
        validateFunctionThrow( fn );
    }

    removeValues( toDelete );  // It is safe to assume that locate() will have replaced all the meaningful end-uses
}

void C14n::canonicalizeTraceGlobal( llvm::Function* fn, std::vector<Value*>& toDelete, const TraceVariant variant )
{
    Module*      module = fn->getParent();
    Type*        f32Ty  = m_llvmManager->getFloatType();
    Type*        i32Ty  = m_llvmManager->getI32Type();
    LLVMContext& ctx    = m_llvmManager->llvmContext();
    Type*        rmTy   = TraceGlobalPayloadBuilder::getRayMaskType( ctx );
    Type*        rfTy   = TraceGlobalPayloadBuilder::getRayFlagsType( ctx );

    const bool   fnHasTime         = hasTime( variant );
    const bool   fnHasMaskAndFlags = hasMaskAndFlags( variant );
    const size_t parCount          = getParameterCount( variant );
    // PRD and size are the last two parameters in all _rt_trace_xzy variants
    const int prdArgIndex = parCount - 2;

    if( ( fn->arg_size() != parCount ) || fn->isVarArg() )
        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( fn ), "Malformed call to " + fn->getName().str() );

    m_cp->markCallsTrace();

    m_globalsToRemove.push_back( fn );

    for( CallInst* CI : getCallsToFunction( fn ) )
    {
        corelib::CoreIRBuilder irb{CI};

        Value*       elementSizeValue = CI->getArgOperand( prdArgIndex + 1 );
        unsigned int elementSize      = getConstantValueOrAssert( elementSizeValue );

        TraceGlobalPayloadBuilder tgpBuilder( module );

        tgpBuilder.setCanonicalState( CI->getParent()->getParent()->arg_begin() );
        int operand = 0;
        tgpBuilder.setNode( CI->getArgOperand( operand++ ) );
        tgpBuilder.setOx( CI->getArgOperand( operand++ ) );
        tgpBuilder.setOy( CI->getArgOperand( operand++ ) );
        tgpBuilder.setOz( CI->getArgOperand( operand++ ) );
        tgpBuilder.setDx( CI->getArgOperand( operand++ ) );
        tgpBuilder.setDy( CI->getArgOperand( operand++ ) );
        tgpBuilder.setDz( CI->getArgOperand( operand++ ) );
        tgpBuilder.setRayType( CI->getArgOperand( operand++ ) );
        tgpBuilder.setTMin( CI->getArgOperand( operand++ ) );
        tgpBuilder.setTMax( CI->getArgOperand( operand++ ) );
        tgpBuilder.setTime( fnHasTime ? CI->getArgOperand( operand++ ) : ConstantFP::getNullValue( f32Ty ) );
        tgpBuilder.setHasTime( ConstantInt::get( i32Ty, fnHasTime ) );
        tgpBuilder.setRayMask( fnHasMaskAndFlags ? irb.CreateTrunc( CI->getArgOperand( operand++ ), rmTy ) :
                                                   TraceGlobalPayloadBuilder::getDefaultRayMask( ctx ) );
        tgpBuilder.setRayFlags( fnHasMaskAndFlags ? irb.CreateTrunc( CI->getArgOperand( operand++ ), rfTy ) :
                                                    TraceGlobalPayloadBuilder::getDefaultRayFlags( ctx ) );
        tgpBuilder.setElementSize( elementSizeValue );

        tgpBuilder.create( m_cp->getUniversallyUniqueName(), elementSize, m_llvmManager, CI );

        toDelete.push_back( CI );
    }
}

void C14n::canonicalizeGetBufferSize( llvm::Function* fn )
{
    if( fn->arg_size() != 3 || fn->isVarArg() )
        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( fn ), "Malformed call to " + fn->getName().str() );
    m_globalsToRemove.push_back( fn );

    llvm::Module* module      = fn->getParent();
    LLVMContext&  llvmContext = module->getContext();
    DataLayout    dataLayout( module->getDataLayout() );
    Type*         statePtrTy      = m_llvmManager->getStatePtrType();
    Type*         u64Ty           = Type::getInt64Ty( llvmContext );
    Type*         size3Ty         = m_llvmManager->getSize3Type();
    Type*         size4elements[] = {u64Ty, u64Ty, u64Ty, u64Ty};
    Type*         ptx_size4type   = llvm::StructType::get( llvmContext, size4elements, false );

    for( CallInst* CI : getCallsToFunction( fn ) )
    {
        corelib::CoreIRBuilder irb{CI};

        // Locate constant values
        GlobalVariable* gvar           = getGlobalVariableForRTState( CI, 0, dataLayout, "rtBuffer" );
        unsigned int    dimensionality = getConstantValueOrAssert( CI->getArgOperand( 1 ) );
        if( dimensionality < 1 || 3 < dimensionality )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ), "Invalid buffer dimensionality" );

        unsigned int eltSize = getConstantValueOrAssert( CI->getArgOperand( 2 ) );

        // Create the variable if necessary and build the name of the getter function
        VariableType             vtype( VariableType::Buffer, eltSize, dimensionality );
        const VariableReference* vref     = getOrAddVariable( gvar, vtype );
        std::string              funcName = "optixi_getBufferSize." + vref->getUniversallyUniqueName();

        // Replace the call
        Type*         argTypes = statePtrTy;
        FunctionType* funType  = FunctionType::get( size3Ty, argTypes, false );
        Function*     cfun     = dyn_cast<Function>( module->getOrInsertFunction( funcName, funType ) );
        cfun->setOnlyReadsMemory();
        cfun->setDoesNotThrow();

        Value* statePtr = CI->getParent()->getParent()->arg_begin();
        Value* args     = statePtr;
        Value* size     = irb.CreateCall( cfun, args, "buffer.size" );

        // buffer_getSize returns a size3 struct, but the ptx wants size4.  Colwert.
        Value* result = UndefValue::get( ptx_size4type );
        for( unsigned int d = 0; d < dimensionality; ++d )
        {
            llvm::Value* el = irb.CreateExtractValue( size, d );
            result          = irb.CreateInsertValue( result, el, d );
        }

        // Replace the original instruction
        Instruction* lastInstruction = dyn_cast<Instruction>( result );
        lastInstruction->removeFromParent();
        llvm::ReplaceInstWithInst( CI, lastInstruction );
    }
}

// Transform uses of rt_buffer_get_64 into uses of UberPointers.
void C14n::canonicalizeAccessBuffer( llvm::Function* fn, std::vector<llvm::Value*>& toDelete )
{
    if( fn->arg_size() != 7 || fn->isVarArg() )
        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( fn ), "Malformed call to " + fn->getName().str() );
    m_globalsToRemove.push_back( fn );

    llvm::Module* module      = fn->getParent();
    LLVMContext&  llvmContext = module->getContext();
    DataLayout    dataLayout( module->getDataLayout() );
    Type*         statePtrTy = m_llvmManager->getStatePtrType();
    Type*         u64Ty      = Type::getInt64Ty( llvmContext );

    for( CallInst* callInst : getCallsToFunction( fn ) )
    {
        // Locate constant values
        GlobalVariable* G              = getGlobalVariableForRTState( callInst, 0, dataLayout, "rtBuffer" );
        unsigned int    dimensionality = getConstantValueOrAssert( callInst->getArgOperand( 1 ) );
        if( dimensionality < 1 || 3 < dimensionality )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( callInst ), "Invalid buffer dimensionality" );

        Value*       elementSizeValue = callInst->getArgOperand( 2 );
        unsigned int eltSize          = getConstantValueOrAssert( elementSizeValue );
        // Unlike C14n::canonicalizeFunctionBufferGetId we don't check that the argument 'elementSize' matches the size of
        // the use of the pointer. This is because mismatches between the argument 'elementSize' and the type of the
        // variable associated with the buffer trigger a validation error.

        // Identify which rtVariable we are accessing with this memory operation.
        // If this is the first access to a variable create a new Variable reference.
        VariableType       vtype( VariableType::Buffer, eltSize, dimensionality );
        VariableReference* vref = getOrAddVariable( G, vtype );

        // We do not have additional type info, so use an array of bytes
        // but getCleanType will colwert it to a word array if possible.
        Type* valueType = ArrayType::get( Type::getInt8Ty( llvmContext ), eltSize );
        valueType       = getCleanType( valueType );

        // Create four new functions in the module, a getter, a setter, an atomic setter and a get_address.
        // These three functions do not take in input the buffer but their name is mangled so to reference the rtVariable we
        // are accessing. This is how the name is made-up:
        // optixi_[set | get]BufferElement.variableUniversallyUniqueName
        // The atomic function is: optixi_atomicSetBufferElement.variableUniversallyUniqueName
        // The getter function takes in input the canonical state and the coordinate of the element to access.
        // The setter function takes in input the canonical state, the coordinate of the element to set and the value to
        //   store.
        // The atomic setter function takes in input the canonical state, the coordinate of the element to set, its offset
        //   within the buffer element, an enumeration identifying the specific atomic instruction (ADD, SUB, EXCH, ...) and
        //   the value to store.
        // The getter function takes in input the canonical state and the coordinate of the element to access and returns a
        //   pointer to the requested element.

        std::vector<llvm::Type*> argumentTypes = {statePtrTy};
        argumentTypes.insert( std::end( argumentTypes ), dimensionality, u64Ty );

        FunctionType*   getterFunctionType = FunctionType::get( valueType, argumentTypes, false );
        std::string     getterName         = GetBufferElement::createUniqueName( vref );
        llvm::Constant* getter             = module->getOrInsertFunction( getterName, getterFunctionType );

        argumentTypes.push_back( valueType );
        FunctionType*   setterFunctionType = FunctionType::get( Type::getVoidTy( llvmContext ), argumentTypes, false );
        std::string     setterName         = SetBufferElement::createUniqueName( vref );
        llvm::Constant* setter             = module->getOrInsertFunction( setterName, setterFunctionType );

        // Always use i64 as op type for the atomic operations to cover both 32 and 64 bit atomics. Casting to the
        // proper type will be done later.
        Type* atomicOpType = Type::getInt64Ty( llvmContext );

        AtomicSetBufferElementBuilder asb( module );
        Constant* atomicSetter = asb.createFunction( atomicOpType, vref, dimensionality, m_llvmManager );

        GetBufferElementAddressBuilder getElementAddressBuilder( module );
        Constant* getAddress = getElementAddressBuilder.createFunction( vref, dimensionality, m_llvmManager );

        // Translate to UberPointer
        const std::string uberPointerName = vref->m_variableName + ".elt";
        int               upkind = getOrAddUberPointerKind( uberPointerName, getter, setter, atomicSetter, getAddress,
                                              UberPointer::PointeeType::Buffer );
        UberPointerTransform upTransform( module, m_llvmManager, G->getName(), UberPointer::PointeeType::Buffer );

        std::vector<llvm::Value*> bufferIndices;
        bufferIndices.reserve( dimensionality );
        for( unsigned int d = 0; d < dimensionality; ++d )
            bufferIndices.push_back( callInst->getArgOperand( 3 + d ) );
        // Replace all the uses of the current pointer to use a UberPointer.
        upTransform.translate( callInst, upkind, bufferIndices, eltSize, 0 );

        // Mark the loaded values as users of this variable
        for( Value* load : upTransform.getLoads() )
        {
            m_variableValues.insert( std::make_pair( load, vref ) );
        }

        // Set properties on vref and canonical program
        if( upTransform.pointerEscapes() )
            vref->markPointerMayEscape();
        if( upTransform.hasStores() )
        {
            vref->markHasBufferStores();
            m_cp->markHasBufferStores();
        }

        // Cleanup
        toDelete.push_back( callInst );
    }
}

// Transform
//      bool _rt_load_or_request_64( Buffer* buffer,
//              unsigned int dimensionality, unsigned int elementSize,
//              unsigned long x, unsigned long y, unsigned long z, unsigned long w,
//              void* ptr )
// into
//      i32 optixi_loadOrRequestBuffer.variableUniversallyUniqueName(
//          statePtrTy canonicalState,
//          i32 elementSize,
//          i64 ptr,
//          i64 x, i64 y )
// for dimensionality = 2.
//
// The output is modeled by LoadOrRequestBufferElement, but the input is unmodeled.
//
void C14n::canonicalizeLoadOrRequest( llvm::Function* fn, std::vector<llvm::Value*>& toDelete )
{
    const int inputArgDimensionality = 1;
    const int inputArgElementSize    = 2;
    const int inputArgX              = 3;
    const int inputArgY              = 4;
    const int inputArgZ              = 5;
    const int inputArgPtr            = 7;
    const int inputArgCount          = 8;
    if( fn->arg_size() != inputArgCount )
        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( fn ),
                            "Malformed call to " + fn->getName().str() + "; expected " + std::to_string( inputArgCount )
                                + " arguments, got " + std::to_string( fn->arg_size() ) );
    if( fn->isVarArg() )
        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( fn ),
                            "Malformed call to " + fn->getName().str() + " as varargs" );

    m_globalsToRemove.push_back( fn );

    llvm::Module* module      = fn->getParent();
    LLVMContext&  llvmContext = module->getContext();
    DataLayout    dataLayout( module->getDataLayout() );
    Type*         statePtrTy = m_llvmManager->getStatePtrType();
    Type*         u64Ty      = Type::getInt64Ty( llvmContext );
    Type*         u32Ty      = Type::getInt32Ty( llvmContext );

    for( CallInst* callInst : getCallsToFunction( fn ) )
    {
        // callInst represents a call to _rt_load_or_request_64
        GlobalVariable*    G              = getGlobalVariableForRTState( callInst, 0, dataLayout, "rtBuffer" );
        const unsigned int dimensionality = getConstantValueOrAssert( callInst->getArgOperand( inputArgDimensionality ) );
        if( dimensionality < 1 || dimensionality > 3 )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( callInst ), "Invalid buffer dimensionality" );

        // Identify which rtVariable we are accessing with this memory operation.
        // If this is the first access to a variable create a new Variable reference.
        unsigned int       eltSize = getConstantValueOrAssert( callInst->getArgOperand( inputArgElementSize ) );
        VariableType       vtype( VariableType::DemandBuffer, eltSize, dimensionality );
        VariableReference* vref = getOrAddVariable( G, vtype );

        // Create a new function in the module that performs a load or request for the buffer.
        // This function does not take in input the buffer but the name is mangled so to reference the rtVariable we
        // are accessing. This is how the name is made-up:
        // optixi_loadOrRequestBuffer.variableUniversallyUniqueName
        std::vector<llvm::Type*> argumentTypes{statePtrTy, u32Ty, u64Ty};  // canonicalState, elementSize, ptr
        argumentTypes.insert( std::end( argumentTypes ), dimensionality, u64Ty );

        FunctionType*   loadOrRequestFunctionType = FunctionType::get( u32Ty, argumentTypes, false );
        std::string     loadOrRequestName         = LoadOrRequestBufferElement::createUniqueName( vref );
        llvm::Constant* loadOrRequest = module->getOrInsertFunction( loadOrRequestName, loadOrRequestFunctionType );

        Value*                    statePtr = callInst->getParent()->getParent()->arg_begin();
        std::vector<llvm::Value*> arguments;
        arguments.push_back( statePtr );
        arguments.push_back( callInst->getArgOperand( inputArgElementSize ) );
        arguments.push_back( callInst->getArgOperand( inputArgPtr ) );
        arguments.push_back( callInst->getArgOperand( inputArgX ) );
        if( dimensionality > 1 )
            arguments.push_back( callInst->getArgOperand( inputArgY ) );
        if( dimensionality > 2 )
            arguments.push_back( callInst->getArgOperand( inputArgZ ) );
        CoreIRBuilder irb{callInst};
        // TODO: Get mapOrRequest as a Function* that doesn't abort like llvm::cast does
        Value* newCallInst = irb.CreateCall( llvm::cast<Function>( loadOrRequest ), arguments );
        callInst->replaceAllUsesWith( newCallInst );
        newCallInst->takeName( callInst );

        toDelete.push_back( callInst );
    }
}

void C14n::canonicalizeGetBufferSizeFromId( llvm::Function* function )
{
    if( function->arg_size() != 3 || function->isVarArg() )
        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( function ), "Malformed call to " + function->getName().str() );
    m_globalsToRemove.push_back( function );

    llvm::Module* module          = function->getParent();
    LLVMContext&  llvmContext     = module->getContext();
    Type*         statePtrTy      = m_llvmManager->getStatePtrType();
    Type*         i32Ty           = Type::getInt32Ty( llvmContext );
    Type*         i64Ty           = Type::getInt64Ty( llvmContext );
    Type*         size3Ty         = m_llvmManager->getSize3Type();
    Type*         size4elements[] = {i64Ty, i64Ty, i64Ty, i64Ty};
    Type*         ptx_size4type   = llvm::StructType::get( llvmContext, size4elements, false );

    for( CallInst* CI : getCallsToFunction( function ) )
    {
        corelib::CoreIRBuilder irb{CI};

        // Locate constant values
        Value*       bufferId       = CI->getArgOperand( 0 );
        unsigned int dimensionality = getConstantValueOrAssert( CI->getArgOperand( 1 ) );
        if( dimensionality < 1 || 3 < dimensionality )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ), "Invalid buffer dimensionality" );

        unsigned int eltSize = getConstantValueOrAssert( CI->getArgOperand( 2 ) );

        // Create the variable if necessary and build the name of the getter function
        VariableType vtype( VariableType::Buffer, eltSize, dimensionality );

        // Replace the call
        Type*         argTypes[] = {statePtrTy, i32Ty};
        FunctionType* funType    = FunctionType::get( size3Ty, argTypes, false );
        Function*     cfun       = dyn_cast<Function>( module->getOrInsertFunction( GET_BUFFER_SIZE_ID, funType ) );

        Value* statePtr = CI->getParent()->getParent()->arg_begin();
        Value* args[]   = {statePtr, bufferId};
        Value* size     = irb.CreateCall( cfun, args, "buffer.size" );

        // buffer_getSize returns a size3 struct, but the ptx wants size4.  Colwert.
        Value* result = UndefValue::get( ptx_size4type );
        for( unsigned int d = 0; d < dimensionality; ++d )
        {
            llvm::Value* el = irb.CreateExtractValue( size, d );
            result          = irb.CreateInsertValue( result, el, d );
        }

        // Replace the original instruction
        Instruction* lastInstruction = dyn_cast<Instruction>( result );
        lastInstruction->removeFromParent();
        llvm::ReplaceInstWithInst( CI, lastInstruction );
    }
}

void C14n::canonicalizeAccessBufferFromId( llvm::Function* function, std::vector<llvm::Value*>& toDelete )
{
    if( function->arg_size() != 7 || function->isVarArg() )
        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( function ), "Malformed call to " + function->getName().str() );
    m_globalsToRemove.push_back( function );

    Module*      module      = function->getParent();
    LLVMContext& llvmContext = module->getContext();
    Type*        statePtrTy  = m_llvmManager->getStatePtrType();
    Type*        i32Ty       = m_llvmManager->getI32Type();
    Type*        i64Ty       = m_llvmManager->getI64Type();
    DataLayout   dataLayout( module->getDataLayout() );

    for( CallInst* CI : getCallsToFunction( function ) )
    {
        Value*       bufferId       = CI->getArgOperand( 0 );
        unsigned int dimensionality = getConstantValueOrAssert( CI->getArgOperand( 1 ) );
        if( dimensionality < 1 || 3 < dimensionality )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ), "Invalid buffer dimensionality" );

        Value*   elementSizeValue = CI->getArgOperand( 2 );
        unsigned eltSize          = getConstantValueOrAssert( elementSizeValue );

        unsigned int deducedSize = deducePointeeSize( CI );
        // If the size of the pointer does not match the argument size then create a call to
        // optixi_get_buffer_id_element_address.
        if( deducedSize != eltSize )
        {
            Function* caller   = CI->getParent()->getParent();
            Value*    statePtr = caller->arg_begin();

            GetBufferElementAddressFromIdBuilder builder( module );
            builder.setCanonicalState( statePtr ).setBufferId( bufferId ).setElementSize( elementSizeValue );
            const int FIRST_DIMENSION_INDEX = 3;
            // The switch intentionally omits break instructions.
            switch( dimensionality )
            {
                case 3:
                {
                    builder.setZ( CI->getArgOperand( FIRST_DIMENSION_INDEX + 2 ) );
                }
                case 2:
                {
                    builder.setY( CI->getArgOperand( FIRST_DIMENSION_INDEX + 1 ) );
                }
                case 1:
                {
                    builder.setX( CI->getArgOperand( FIRST_DIMENSION_INDEX ) );
                    break;
                }
                default:
                {
                    RT_ASSERT_FAIL();
                    break;
                }
            }

            Function* canonicalFunction = builder.createFunction( dimensionality, m_llvmManager );
            CallInst* call              = builder.create( canonicalFunction, CI );

            CI->replaceAllUsesWith( call );
            toDelete.push_back( CI );

            /// Mark the buffer's variable reference as having an ill-formed access (which
            // ensures that the buffer gets a "raw access" property).
            if( VariableReference* varref = m_variableValues.lookup( bufferId ) )
                varref->markHasIllFormedAccess();
            continue;
        }

        // We do not have additional type info, so use an array of bytes
        // but getCleanType will colwert it to a word array if possible.
        Type* valueType = ArrayType::get( Type::getInt8Ty( llvmContext ), eltSize );
        valueType       = getCleanType( valueType );

        // Construct getter/setter
        std::vector<Type*> argumentTypes = {statePtrTy, i32Ty};
        argumentTypes.insert( std::end( argumentTypes ), dimensionality, i64Ty );

        FunctionType* getterFunctionType = FunctionType::get( valueType, argumentTypes, false );
        // The name of setter and getter is mangled to avoid clashes for functions with different data type sizes and
        // different dimensionalities.
        // Encoding just the size is ok, since "getCleanType" only returns integer types and not floating point types.

        Constant* getter = module->getOrInsertFunction( GetBufferElementFromId::createUniqueName( dimensionality, valueType ),
                                                        getterFunctionType );

        argumentTypes.push_back( valueType );
        FunctionType* setterFunctionType = FunctionType::get( Type::getVoidTy( llvmContext ), argumentTypes, false );
        Constant* setter = module->getOrInsertFunction( SetBufferElementFromId::createUniqueName( dimensionality, valueType ),
                                                        setterFunctionType );

        // Always use i64 as op type for the atomic operations to cover both 32 and 64 bit atomics. Casting to the
        // proper type will be done later.
        Type* atomicOpType = Type::getInt64Ty( llvmContext );

        AtomicSetBufferElementFromIdBuilder asbId( module );
        Constant* atomicSetter = asbId.createFunction( atomicOpType, eltSize, dimensionality, m_llvmManager );

        GetBufferElementAddressFromIdBuilder getIdElementAddressBuilder( module );
        Constant* getAddress = getIdElementAddressBuilder.createFunction( dimensionality, m_llvmManager );

        // Translate to UberPointer
        std::vector<llvm::Value*> args;
        args.reserve( dimensionality );
        for( unsigned int d = 0; d < dimensionality; ++d )
            args.push_back( CI->getArgOperand( 3 + d ) );
        const std::string    UP_NAME = "bufferId";
        int                  kindId = getOrAddUberPointerKind( UP_NAME, getter, setter, atomicSetter, getAddress, UberPointer::PointeeType::BufferID );
        UberPointerTransform upTransform( module, m_llvmManager, UP_NAME, UberPointer::PointeeType::BufferID );
        upTransform.translate( CI, kindId, bufferId, args, eltSize, 0 );

        // Set properties on canonical program
        if( upTransform.pointerEscapes() )
            m_cp->markBindlessBufferPointerMayEscape();
        if( upTransform.hasStores() )
            m_cp->markHasBufferStores();

        // Cleanup
        toDelete.push_back( CI );
    }
}

static FunctionType* getFunctionTypeFromCallInst( CallInst* CI, Type* prepend1, Type* prepend2 )
{
    SmallVector<Type*, 4> args;
    args.push_back( prepend1 );
    args.push_back( prepend2 );
    unsigned int N = CI->getNumArgOperands();
    for( unsigned int i = 0; i < N; ++i )
    {
        Value* A = CI->getArgOperand( i );
        args.push_back( A->getType() );
    }
    Type* returnTy = CI->getType();
    return FunctionType::get( returnTy, args, false );
}

void C14n::canonicalizeCallableProgramFromId( llvm::Module* module )
{
    // link calls to variables that they came from
    LLVMContext& llvmContext = module->getContext();

    llvm::SmallVector<Function*, 2> id_fns;
    Function* id_fn1 = module->getFunction( "_rt_callable_program_from_id_64" );
    if( id_fn1 != nullptr )
    {
        if( id_fn1->arg_size() != 1 || id_fn1->isVarArg()
            || id_fn1->getFunctionType()->getParamType( 0 ) != Type::getInt32Ty( llvmContext ) )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( id_fn1 ),
                                "Malformed signature of " + id_fn1->getName().str() );
        if( id_fn1->user_begin() != id_fn1->user_end() )
            id_fns.push_back( id_fn1 );
    }
    Function* id_fn2 = module->getFunction( "_rt_callable_program_from_id_v2_64" );
    if( id_fn2 != nullptr )
    {
        if( id_fn2->arg_size() != 2 || id_fn2->isVarArg()
            || id_fn2->getFunctionType()->getParamType( 0 ) != Type::getInt32Ty( llvmContext ) )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( id_fn2 ),
                                "Malformed signature of " + id_fn2->getName().str() );
        if( id_fn2->user_begin() != id_fn2->user_end() )
            id_fns.push_back( id_fn2 );
    }

    for( Function* id_fn : id_fns )
    {
        m_globalsToRemove.push_back( id_fn );
        std::vector<Instruction*> toDelete;

        // Process all users of the id function
        for( Value::user_iterator UI = id_fn->user_begin(), UE = id_fn->user_end(); UI != UE; )
        {
            CallInst* rtCallableCI = dyn_cast<CallInst>( *UI++ );
            if( !rtCallableCI )
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( rtCallableCI ),
                                    "Invalid use of function: " + id_fn->getName().str() );
            RT_ASSERT( rtCallableCI->getCalledFunction() == id_fn );

            // Remove the call to the ID function
            toDelete.push_back( rtCallableCI );

            // Determine whether the ID can be connected to an rtVariable
            Value*             ID     = rtCallableCI->getArgOperand( 0 );
            VariableReference* varref = m_variableValues.lookup( ID );

            std::string callSiteName;
            if( rtCallableCI->getNumArgOperands() == 2 )
            {
                RT_ASSERT( id_fn == id_fn2 );
                // extract the call site name if present
                if( Value* csId = rtCallableCI->getArgOperand( 1 ) )
                {
                    ConstantExpr* ce = dyn_cast<ConstantExpr>( csId );
                    if( ce )
                    {
                        ConstantExpr* ce2 = dyn_cast<ConstantExpr>( ce->getOperand( 0 ) );
                        if( ce2 )
                        {
                            Value*          val = ce2->getOperand( 0 );
                            GlobalVariable* gv  = dyn_cast<GlobalVariable>( val );
                            if( gv )
                            {
                                ConstantDataArray* cda = dyn_cast<ConstantDataArray>( gv->getInitializer() );
                                if( cda )
                                {
                                    callSiteName = cda->getAsCString().str();
                                }
                            }
                        }
                    }
                }
            }

            std::string function_name = "optixi_callBindless";
            if( varref && varref->getType().baseType() == VariableType::Program )
            {
                function_name = "optixi_callBound." + varref->getUniversallyUniqueName();
                if( !callSiteName.empty() )
                {
                    throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( id_fn ),
                                        "Callsites of bound callable programs cannot be named: " + id_fn->getName().str() );
                }
                m_cp->markCallsBoundCallableProgram();
            }
            else
            {
                std::string csUniqueName;

                if( varref && ( varref->getType().isProgramId() || varref->getType().isBuffer() ) )
                {
                    // generate a callsite identifier
                    if( callSiteName.empty() )
                    {
                        // Adding the CP's ptx hash to the name of the call site to
                        // avoid collisions with user defined call sites.
                        callSiteName = varref->getInputName() + corelib::ptr_to_string( m_cp->getPTXHash(), 32 );
                        csUniqueName = CallSiteIdentifier::generateCallSiteUniqueName( varref );
                    }
                    else
                    {
                        csUniqueName = m_cp->getUniversallyUniqueName() + "." + callSiteName;
                    }
                }
                else
                {
                    if( !callSiteName.empty() )
                        csUniqueName = m_cp->getUniversallyUniqueName() + "." + callSiteName;
                }

                if( csUniqueName.empty() )
                    function_name = "optixi_callBindless";
                else
                {
                    function_name = "optixi_callBindless." + csUniqueName;

                    // Do not recreate existing csId
                    auto it = algorithm::find_if( m_cp->m_ownedCallSites, [&csUniqueName]( const CallSiteIdentifier* mapping ) {
                        return csUniqueName == mapping->getUniversallyUniqueName();
                    } );

                    if( it == m_cp->m_ownedCallSites.end() )
                    {
                        CallSiteIdentifier* csId = new CallSiteIdentifier( callSiteName, m_cp.get() );
                        m_programManager->registerCallSite( csId );
                        m_cp->m_ownedCallSites.push_back( csId );
                    }
                }
                m_cp->markCallsBindlessCallableProgram();
            }

            // Find calls based on use of output of _rt_callable_program_from_id_64, and replace
            // call with call to functionName.

            // ; before
            //  %1 = tail call i64 @_rt_callable_program_from_id_64(i32 %call013)
            //  %2 = inttoptr i64 %1 to i32 ()*
            //  %3 = tail call i32 %2()

            // ; after
            //  %3 = tail call optixi_call_<function_name>(%"struct.cort::CanonicalState"* %0, i32 ID, <args>)


            // Although we expect only a single use of the id returned from
            // _rt_callable_program_from_id_64 and only a single use of the
            // cast, we process them with a worklist to ensure that it is
            // robust.
            SmallVector<CallInst*, 4>    calls;
            SmallVector<Instruction*, 4> worklist;
            SmallSet<Instruction*, 4>    visited;

            worklist.push_back( rtCallableCI );

            while( !worklist.empty() )
            {
                Instruction* inst = worklist.back();
                worklist.pop_back();

                if( !std::get<1>( visited.insert( inst ) ) )
                    continue;

                for( Instruction::user_iterator I = inst->user_begin(), E = inst->user_end(); I != E; ++I )
                {
                    if( isa<CastInst>( *I ) )
                    {
                        Instruction* use = cast<Instruction>( *I );
                        toDelete.push_back( use );
                        worklist.push_back( use );
                    }
                    else if( CallInst* CI = dyn_cast<CallInst>( *I ) )
                    {
                        if( getDefSkipCasts( CI->getCalledValue() ) != rtCallableCI )
                            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ),
                                                "Use of _rt_callable_program_from_id_64 not used as the function "
                                                "pointer "
                                                "in an "
                                                "indirect call" );
                        calls.push_back( CI );
                    }
                    else
                    {
                        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( *I ),
                                            "Use of _rt_callable_program_from_id_64 not used as a call or cast" );
                    }
                }
            }

            if( calls.empty() )
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( rtCallableCI ),
                                    "No use of a callable program from _rt_callable_program_from_id_64 found." );

            for( CallInst* CI : calls )
            {
                Instruction* insertBefore = CI;
                Value*       statePtr     = insertBefore->getParent()->getParent()->arg_begin();

                // Build argument list
                Type*         i32Ty      = m_llvmManager->getI32Type();
                llvm::Type*   statePtrTy = m_llvmManager->getStatePtrType();
                FunctionType* oldFType   = getFunctionTypeFromCallInst( CI, statePtrTy, i32Ty );
                FunctionType* newFType   = getCleanFunctionType( oldFType );

                int                 num_args = CI->getNumArgOperands() + 2;
                std::vector<Value*> args( num_args );
                args[0] = statePtr;
                args[1] = ID;

                for( int i = 2; i < num_args; i++ )
                {
                    Value* arg   = CI->getArgOperand( i - 2 );
                    Type*  oldTy = arg->getType();
                    Type*  newTy = newFType->getParamType( i );
                    if( oldTy != newTy )
                    {
                        // Cast arg through an alloca
                        arg = castThroughAlloca( arg, newTy, CI );
                    }
                    args[i] = arg;
                }

                // Use the public function type (without canonical state and i32 id) to register the function signature in order to print at exception.
                llvm::SmallVector<Type*, 10> publicParamTypes;
                publicParamTypes.append( newFType->param_begin() + 2, newFType->param_end() );
                FunctionType* publicFType =
                    FunctionType::get( newFType->getReturnType(), publicParamTypes, newFType->isVarArg() );
                unsigned sig = registerCallableFunctionSignature( publicFType );
                if( varref )
                {
                    if( varref->getType().baseType() == VariableType::Program )
                    {
                        varref->m_vtype = VariableType::createForCallableProgramVariable( sig );
                    }
                }

                // Insert a call to the canonical function
                corelib::CoreIRBuilder irb{insertBefore};
                Value* new_function = module->getOrInsertFunction( function_name + ".sig" + std::to_string( sig ), newFType );
                Value* result       = irb.CreateCall( new_function, args );

                // Update the return value
                Type* oldReturnTy = CI->getType();
                Type* newReturnTy = result->getType();
                irb.SetInsertPoint( &*( ++BasicBlock::iterator( insertBefore ) ) );
                if( oldReturnTy != newReturnTy )
                {
                    AllocaInst* alloca =
                        corelib::CoreIRBuilder{corelib::getFirstNonAlloca( rtCallableCI->getParent()->getParent() )}.CreateAlloca( newReturnTy );
                    irb.CreateStore( result, alloca );
                    Value* castPtr = irb.CreateBitCast( alloca, oldReturnTy->getPointerTo() );
                    result         = irb.CreateLoad( castPtr );
                }
                CI->replaceAllUsesWith( result );
                result->takeName( CI );

                toDelete.push_back( CI );
            }
        }

        // Delete old calls.  We need to iterate in reverse insertion order, since we used
        // forward data flow to find the sequence of instructions, we go backward to delete
        // the bottom uses first.
        for( auto I = toDelete.rbegin(), IE = toDelete.rend(); I != IE; ++I )
            ( *I )->eraseFromParent();
    }
}

void C14n::canonicalizeTextures( llvm::Function* function, std::vector<llvm::Value*>& toDelete )
{
    Module*    module = function->getParent();
    DataLayout dataLayout( module );

    // llvm.lwvm.texsurf.handle.p1i64
    // llvm.lwvm.tex.*
    if( Function* fn = module->getFunction( "llvm.lwvm.texsurf.handle.p1i64" ) )
    {
        if( fn->arg_size() != 2 || fn->isVarArg() )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( fn ), "Malformed call to " + fn->getName().str() );
        toDelete.push_back( fn );

        for( Value::user_iterator U = fn->user_begin(), UE = fn->user_end(); U != UE; ++U )
        {
            CallInst* CI = dyn_cast<CallInst>( *U );
            if( !CI )
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ), "Invalid use of function: " + fn->getName().str() );
            RT_ASSERT( CI->getCalledFunction() == fn );

            // Locate constant values
            GlobalVariable* gvar = getGlobalVariableForRTState( CI, 1, dataLayout, "rtTexture" );

            // Get the dimensionality from the attached lookup functions.
            // Note that PTX allows mixing texture lookup dimensionality for
            // the same texture reference but OptiX does not (because it is
            // part of the type).  We could relax that constraint if necessary.
            unsigned int dimensionality = ~0u;
            for( Value::user_iterator U2 = CI->user_begin(), UE2 = CI->user_end(); U2 != UE2; ++U2 )
            {
                CallInst* fetch = dyn_cast<CallInst>( *U2 );
                if( !fetch )
                    throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ),
                                        "Invalid use of function: " + fn->getName().str() );

                TextureLookup::LookupKind kind = TextureLookup::getLookupKindFromLWVMFunction( fetch->getCalledFunction() );
                unsigned int              dim = TextureLookup::getLookupDimensionality( kind );
                if( dim > 0 )
                {  // dim == 0 for txq
                    if( dim > 3 || ( dimensionality != ~0u && dimensionality != dim ) )
                        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ), "Invalid texture dimensionality" );
                }
                dimensionality = dim;
            }

            // Create the variable and get the token value
            VariableType       vtype( VariableType::TextureSampler, 1, dimensionality );
            VariableReference* vref = getOrAddVariable( gvar, vtype );

            // Replace each texture fetch with a call to the get function
            for( Value::user_iterator U2 = CI->user_begin(), UE2 = CI->user_end(); U2 != UE2; ++U2 )
            {
                CallInst*              fetch         = dyn_cast<CallInst>( *U2 );
                Function*              usingFunction = fetch->getParent()->getParent();
                corelib::CoreIRBuilder irb{fetch};

                // Build the function prototype. For simplicity, CORT texture
                // functions always return a float4 even when the data is
                // encoded as an integer.  PTX instructions are similarly
                // isomorphic in return value.
                TextureLookup::LookupKind kind = TextureLookup::getLookupKindFromLWVMFunction( fetch->getCalledFunction() );
                FunctionType* funType = TextureLookup::getPlaceholderFunctionType( kind, false, m_llvmManager );
                vref->addTextureLookupKind( kind );

                std::string getterName = "optixi_getTexture_" + toString( kind ) + "Value." + vref->getUniversallyUniqueName();
                llvm::Function* getter = dyn_cast<llvm::Function>( module->getOrInsertFunction( getterName, funType ) );

                // Build the parameter list.  Note that the LLVM function may
                // have additional unused parameters.
                unsigned int numTexArgs = funType->getNumParams() - 1;
                RT_ASSERT( numTexArgs <= fetch->getCalledFunction()->getFunctionType()->getNumParams() - 1 );
                std::vector<llvm::Value*> args( numTexArgs + 1 );
                args[0] = usingFunction->arg_begin();  // StatePtr
                for( unsigned int d = 0; d < numTexArgs; ++d )
                    args[d + 1]     = fetch->getArgOperand( 1 + d );

                // Call the canonical getValue function
                Value* texres = irb.CreateCall( getter, args, "tex.res" );

                // Repack into the result vector, casting elements from float if necessary
                Value* result = UndefValue::get( fetch->getType() );
                for( unsigned int i = 0; i < result->getType()->getStructNumElements(); ++i )
                {
                    llvm::Value* el                = irb.CreateExtractValue( texres, i );
                    Type*        resultElementType = result->getType()->getStructElementType( i );
                    llvm::Value* castel            = irb.CreateBitCast( el, resultElementType );
                    result                         = irb.CreateInsertValue( result, castel, i );
                }

                // Replace the original instruction
                fetch->replaceAllUsesWith( result );
                toDelete.push_back( fetch );
            }
            toDelete.push_back( CI );
        }
    }
}

void C14n::canonicalizeBindlessTextures( llvm::Module* module, std::vector<llvm::Value*>& toDelete )
{
    Function* texBindlessFns[] = {
        module->getFunction( "_rt_texture_get_f_id" ),      module->getFunction( "_rt_texture_get_i_id" ),
        module->getFunction( "_rt_texture_get_u_id" ),      module->getFunction( "_rt_texture_get_size_id" ),
        module->getFunction( "_rt_texture_get_gather_id" ), module->getFunction( "_rt_texture_get_fetch_id" ),
        module->getFunction( "_rt_texture_get_base_id" ),   module->getFunction( "_rt_texture_get_level_id" ),
        module->getFunction( "_rt_texture_get_grad_id" )};
    for( Function* fn : texBindlessFns )
    {
        if( !fn )
            continue;

        if( fn->isVarArg() )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( fn ), "Malformed call to " + fn->getName().str() );
        toDelete.push_back( fn );

        for( Value::user_iterator U = fn->user_begin(), UE = fn->user_end(); U != UE; ++U )
        {
            CallInst* CI = dyn_cast<CallInst>( *U );
            if( !CI )
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ), "Invalid use of function: " + fn->getName().str() );
            RT_ASSERT( CI->getCalledFunction() == fn );

            TextureLookup::LookupKind kind = TextureLookup::getLookupKindFromOptiXFunction( CI );

            Function*              usingFunction = CI->getParent()->getParent();
            corelib::CoreIRBuilder irb{CI};

            // Build the function prototype. For simplicity, CORT texture
            // functions always return a float4 even when the data is
            // encoded as an integer.  PTX instructions are similarly
            // isomorphic in return value.
            FunctionType* funType = TextureLookup::getPlaceholderFunctionType( kind, true, m_llvmManager );

            std::string     getterName = "optixi_getTexture_" + toString( kind ) + "ValueFromId";
            llvm::Function* getter     = dyn_cast<llvm::Function>( module->getOrInsertFunction( getterName, funType ) );

            // Build the parameter list.  Note that the LLVM function may
            // have additional unused parameters.
            std::vector<llvm::Value*> args;
            args.push_back( usingFunction->arg_begin() );  // StatePtr

            TextureLookup::packParamsFromOptiXFunction( kind, CI, args );
            // Call the canonical getValue function
            Value* texres = irb.CreateCall( getter, args, "tex.res" );
            // Repack into the result vector, casting elements from float if necessary
            Value* result = UndefValue::get( CI->getType() );
            for( unsigned int i = 0; i < result->getType()->getStructNumElements(); ++i )
            {
                llvm::Value* el                = irb.CreateExtractValue( texres, i );
                Type*        resultElementType = result->getType()->getStructElementType( i );
                llvm::Value* castel            = irb.CreateBitCast( el, resultElementType );
                result                         = irb.CreateInsertValue( result, castel, i );
            }

            // Replace the original instruction
            CI->replaceAllUsesWith( result );
            toDelete.push_back( CI );
        }
    }
}

// Transform
//
//     uint4 _rt_texture_load_or_request_f_id(
//          int texId,
//          int dimensionality,
//          float x, float y, float z, float w,
//          bool* isResidentPtr)
//
// for dimensionality 2 into
//
//     float4 optixi_textureLoadOrRequst2(
//          statePtrTy canonicalState,
//          i32 texId,
//          float x, float y,
//          i64 isResidentPtr )
//
// follows by bitcasts and insertelements to construct uint4.
//
void C14n::canonicalizeDemandLoadBindlessTextures( llvm::Module* module, std::vector<llvm::Value*>& toDelete )
{
    Function* texBindlessFns[] = {module->getFunction( "_rt_texture_load_or_request_f_id" ),
                                  module->getFunction( "_rt_texture_load_or_request_u_id" ),
                                  module->getFunction( "_rt_texture_load_or_request_i_id" ),
                                  module->getFunction( "_rt_texture_lod_load_or_request_f_id" ),
                                  module->getFunction( "_rt_texture_lod_load_or_request_u_id" ),
                                  module->getFunction( "_rt_texture_lod_load_or_request_i_id" ),
                                  module->getFunction( "_rt_texture_grad_load_or_request_f_id" ),
                                  module->getFunction( "_rt_texture_grad_load_or_request_u_id" ),
                                  module->getFunction( "_rt_texture_grad_load_or_request_i_id" )};
    for( Function* fn : texBindlessFns )
    {
        if( !fn )
            continue;

        if( fn->isVarArg() )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( fn ), "Malformed call to " + fn->getName().str() );
        toDelete.push_back( fn );

        for( Value::user_iterator useIter = fn->user_begin(), useEnd = fn->user_end(); useIter != useEnd; ++useIter )
        {
            CallInst* callInst = dyn_cast<CallInst>( *useIter );
            if( !callInst )
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( callInst ),
                                    "Invalid use of function: " + fn->getName().str() );
            RT_ASSERT( callInst->getCalledFunction() == fn );

            Function*              usingFunction = callInst->getParent()->getParent();
            corelib::CoreIRBuilder irb{callInst};

            std::string funcName( fn->getName() );
            bool        isLod  = funcName.find( "_lod_" ) != std::string::npos;
            bool        isGrad = funcName.find( "_grad_" ) != std::string::npos;

            const int textureIdArg      = 0;
            const int dimensionalityArg = 1;
            const int firstCoordArg     = 2;
            const int levelArg          = 6;
            const int firstGradArg      = 6;

            // Construct callee name, e.g. optix_texture[Grad]LoadOrRequest2.
            std::string getterName( "optixi_texture" );
            getterName += ( isLod ? "Lod" : "" );
            getterName += ( isGrad ? "Grad" : "" );
            getterName += "LoadOrRequest";
            unsigned int dimensionality = getConstantValueOrAssert( callInst->getArgOperand( dimensionalityArg ) );
            getterName += std::to_string( dimensionality );

            // Build the function prototype. For simplicity, CORT texture functions always return a
            // float4 even when the data is encoded as an integer.  PTX instructions are similarly
            // isomorphic in return value.
            Type* i32Ty    = m_llvmManager->getI32Type();
            Type* i64Ty    = m_llvmManager->getI64Type();
            Type* floatTy  = m_llvmManager->getFloatType();
            Type* float4Ty = m_llvmManager->getFloat4Type();

            // Gather argument types, starting with the state pointer and textureId.
            std::vector<llvm::Type*> argTypes{
                m_llvmManager->getStatePtrType(),
                i32Ty  // textureId
            };

            // Add a float parameter for each dimension.
            argTypes.insert( argTypes.end(), dimensionality, floatTy );

            // Add level or gradient parameters as needed.
            if( isLod )
            {
                argTypes.push_back( floatTy );  // level
            }
            else if( isGrad )
            {
                argTypes.insert( argTypes.end(), 2 * dimensionality, floatTy );
            }
            argTypes.push_back( i64Ty );  // isResidentPtr

            // Construct the function type and create the function if necessary.
            llvm::FunctionType* funType = FunctionType::get( float4Ty, argTypes, false );
            llvm::Function*     getter = dyn_cast<llvm::Function>( module->getOrInsertFunction( getterName, funType ) );

            // Build the arguments for the call instruction.
            std::vector<llvm::Value*> args{
                usingFunction->arg_begin(),              // StatePtr
                callInst->getArgOperand( textureIdArg )  // id
            };
            for( int coord = 0; coord < dimensionality; coord++ )
            {
                args.push_back( callInst->getArgOperand( firstCoordArg + coord ) );  // x, y, z
            }
            if( isLod )
            {
                // TODO: using const ints instead of magic numbers here.
                args.push_back( callInst->getArgOperand( levelArg ) );  // level
            }
            else if( isGrad )
            {
                const int NUM_GRADIENTS      = 2;
                const int MAX_DIMENSIONALITY = 3;
                for( int i = 0; i < NUM_GRADIENTS; ++i )
                {
                    for( int j = 0; j < dimensionality; j++ )
                    {
                        args.push_back( callInst->getArgOperand( firstGradArg + i * MAX_DIMENSIONALITY + j ) );  // dPdx_x, ...
                    }
                }
            }
            args.push_back( callInst->getArgOperand( callInst->getNumArgOperands() - 1 ) );  // isResidentPtr

            // Create the call instruction.
            Value* texres = irb.CreateCall( getter, args, "tex.res" );

            // Repack into the result vector, casting elements from float if necessary
            Value* result = UndefValue::get( callInst->getType() );
            for( unsigned int i = 0; i < result->getType()->getStructNumElements(); ++i )
            {
                llvm::Value* el                = irb.CreateExtractValue( texres, i );
                Type*        resultElementType = result->getType()->getStructElementType( i );
                llvm::Value* castel            = irb.CreateBitCast( el, resultElementType );
                result                         = irb.CreateInsertValue( result, castel, i );
            }

            // Replace the original instruction
            callInst->replaceAllUsesWith( result );
            toDelete.push_back( callInst );
        }
    }
}


void C14n::validateFunctionGetTransform( llvm::Function* function )
{
    for( Value::user_iterator U = function->user_begin(), UE = function->user_end(); U != UE; ++U )
    {
        CallInst* CI = dyn_cast<CallInst>( *U );
        if( !CI )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ),
                                "Invalid use of function: " + function->getName().str() );
        RT_ASSERT( CI->getCalledFunction() == function );
        m_cp->markCallsTransform();

        unsigned int kindArgument = getConstantValueOrAssert( CI->getArgOperand( 1 ) );

        if( kindArgument != RT_WORLD_TO_OBJECT && kindArgument != RT_OBJECT_TO_WORLD )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ),
                                "The argument of _rt_get_transform must be either RT_WORLD_TO_OBJECT or "
                                "RT_OBJECT_TO_WORLD" );
    }
}

void C14n::validateFunctionTransformTuple( llvm::Function* function )
{
    for( Value::user_iterator U = function->user_begin(), UE = function->user_end(); U != UE; ++U )
    {
        CallInst* CI = dyn_cast<CallInst>( *U );
        if( !CI )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ),
                                "Invalid use of function: " + function->getName().str() );
        RT_ASSERT( CI->getCalledFunction() == function );
        m_cp->markCallsTransform();

        unsigned int kindArgument = getConstantValueOrAssert( CI->getArgOperand( 1 ) );

        if( kindArgument != RT_WORLD_TO_OBJECT && kindArgument != RT_OBJECT_TO_WORLD
            && kindArgument != ( RT_WORLD_TO_OBJECT | RT_INTERNAL_ILWERSE_TRANSPOSE )
            && kindArgument != ( RT_OBJECT_TO_WORLD | RT_INTERNAL_ILWERSE_TRANSPOSE ) )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ),
                                "The argument of _rt_transform_tuple must be either RT_WORLD_TO_OBJECT or "
                                "RT_OBJECT_TO_WORLD" );
    }
}

void C14n::validateFunctionThrow( llvm::Function* function )
{
    for( Value::user_iterator U = function->user_begin(), UE = function->user_end(); U != UE; ++U )
    {
        CallInst* CI = dyn_cast<CallInst>( *U );
        if( !CI )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ),
                                "Invalid use of function: " + function->getName().str() );
        RT_ASSERT( CI->getCalledFunction() == function );

        RT_ASSERT( CI->getNumArgOperands() == 2 );

        Value* code = CI->getArgOperand( 1 );
        RT_ASSERT( code->getType()->isIntegerTy( 32 ) );

        // If the code is a constant we do a compile time check.
        if( isa<Constant>( code ) )
        {
            const unsigned int val = getConstantValueOrAssert( code );
            if( val < RT_EXCEPTION_USER )
                throw prodlib::CompileError( RT_EXCEPTION_INFO,
                                             "User exception must have exception code >= RT_EXCEPTION_USER" );
            if( val > RT_EXCEPTION_USER_MAX )
                throw prodlib::CompileError( RT_EXCEPTION_INFO,
                                             "User exception must have exception code <= RT_EXCEPTION_USER_MAX" );
        }
    }
}

void C14n::validateRemainingFunctions( llvm::Function* function )
{
    const bool force_inline = ( m_context ) ? m_context->getForceInlineUserFunctions() : true;

    Module* module = function->getParent();
    for( Module::iterator F = module->begin(), FE = module->end(); F != FE; ++F )
    {
        // If internal optix state is accessed in non-inlined user function, throw error.
        if( F->getName().startswith( "optixi_" ) && F->isDeclaration() && !force_inline )
        {
            for( CallInst* CI : getCallsToFunction( &*F ) )
            {
                Function* caller = CI->getParent()->getParent();
                if( caller != function && !caller->getName().startswith( "_rti_" )
                    && !caller->getName().startswith( "_rt_" ) )
                {
                    const std::string fnName = F->getName();
                    throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ),
                                        "OptiX state access (" + fnName + ") found in non-inlined function ("
                                            + caller->getName().str() + ")" );
                }
            }
        }

        // If the function begins with _rt, throw an error
        if( F->getName().startswith( "_rt" ) && F->isDeclaration() )
        {
            if( F->getName().startswith( "_rt_pickle_pointer" ) )
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( &*F ),
                                    "Pickling/unpickling pointers is deprecated: " + F->getName().str() );
            else if( F->getName().startswith( "_rt_unpickle_pointer" ) )
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( &*F ),
                                    "Pickling/unpickling pointers is deprecated: " + F->getName().str() );
            else
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( &*F ),
                                    "Unprocessed rt function found: " + F->getName().str() );
        }
    }
}

void C14n::validateRemainingVariables( llvm::Module* module )
{
    for( Module::global_iterator G = module->global_begin(), GE = module->global_end(); G != GE; ++G )
    {
        // It is illegal to write to other global symbols.  Allow special symbols for the testing infrastructure.
        if( !G->getName().startswith( "__testing_allowed_global" ) && globalVariableIsWritten( &*G ) )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( &*G ),
                                "Illegal writes to global variable: " + G->getName().str() );

        if( !G->hasInitializer() )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( &*G ), "Unitialized global found: " + G->getName().str() );
    }
}

// Walk the dependence tree of the given instruction and colwert all operations
// to use pointers in the generic address space. Do not relwrse beyond
// comparisons, loads, stores, and explicit address space casts.
void C14n::changeAddressSpaceTo( Instruction* inst, unsigned newAddrSpace ) const
{
    Type* oldType = inst->getType();

    // If this instruction returns a pointer to generic address space, we can stop.
    if( oldType->isPointerTy() && oldType->getPointerAddressSpace() == newAddrSpace )
    {
        return;
    }

    // There's a bunch of instructions for which we don't need to relwrse into
    // the uses, and some that we should never see here (or, if we do, may have
    // to think about again).
    switch( inst->getOpcode() )
    {
        // Nothing to do for cmp/load/store since only their operands need to change.
        case Instruction::ICmp:
        case Instruction::FCmp:
        case Instruction::Load:
        case Instruction::Store:
        // If we hit an explicit address space cast, it should be there for a reason
        // and we don't touch it or any of its dependencies.
        case Instruction::AddrSpaceCast:
        {
            return;
        }

        // Calls require their pointer arguments to be in generic address space.
        // Thus, we should never hit one: If we are changing to generic, this call
        // should not be in the dependency tree. If we are changing to something
        // else, we have introduced wrong code since this call now has an operand
        // that is not in generic address space.
        case Instruction::Call:
        {
            RT_ASSERT_FAIL_MSG( "call must only receive generic address space pointers" );
        }

        case Instruction::Alloca:
        case Instruction::AtomicCmpXchg:
        case Instruction::AtomicRMW:
        case Instruction::Fence:
        case Instruction::VAArg:
        case Instruction::ExtractElement:
        case Instruction::InsertElement:
        case Instruction::ExtractValue:
        case Instruction::InsertValue:
        case Instruction::ShuffleVector:
        case Instruction::LandingPad:
        {
            RT_ASSERT_FAIL_MSG( "unexpected operation during change to generic address space" );
        }

        default:
            break;
    }

    // Before relwrsing into uses, remove all uses that are cvta that colwert to
    // global address space and add the uses of the cvta instruction to the use
    // list of the current instruction, unless our target is global address space.
    // First, collect these calls.
    Module*   module     = inst->getParent()->getParent()->getParent();
    Function* cvtaGlobal = module->getFunction( "optix.ptx.cvta.to.u64.global" );
    if( cvtaGlobal && newAddrSpace != ADDRESS_SPACE_GLOBAL )
    {
        std::vector<CallInst*> cvtaCalls;
        for( Value::user_iterator U = inst->user_begin(), UE = inst->user_end(); U != UE; ++U )
        {
            CallInst* call = dyn_cast<CallInst>( *U );
            if( !call || call->getCalledFunction() != cvtaGlobal )
                continue;

            RT_ASSERT( call->getType() == call->getArgOperand( 0 )->getType() );
            cvtaCalls.push_back( call );
        }

        // Now, replace the uses and remove the calls. We can directly remove the
        // calls since they can't be in one of the use iterators that we used for
        // relwrsion.
        for( CallInst* call : cvtaCalls )
        {
            call->replaceAllUsesWith( call->getArgOperand( 0 ) );
            call->eraseFromParent();
        }
    }

    // Now, relwrse into the (updated) uses.
    for( Value::user_iterator U = inst->user_begin(), UE = inst->user_end(); U != UE; ++U )
    {
        RT_ASSERT( isa<Instruction>( *U ) );
        Instruction* useInst = cast<Instruction>( *U );
        changeAddressSpaceTo( useInst, newAddrSpace );
    }

    // If the return type of this instruction is a pointer, change its address
    // space.
    if( !oldType->isPointerTy() )
        return;

    Type* newType = PointerType::get( oldType->getPointerElementType(), newAddrSpace );
    inst->mutateType( newType );
}


// Create or retrieve the 'kind' of the uber pointer that will use the given setter and getter.
int C14n::getOrAddUberPointerKind( const std::string&       name,
                                   llvm::Constant*          getter,
                                   llvm::Constant*          setter,
                                   llvm::Constant*          atomicSetter,
                                   llvm::Constant*          getAddress,
                                   UberPointer::PointeeType type,
                                   llvm::Value*             defaultValue )
{
    RT_ASSERT( getter != nullptr );
    Function* getterFunction = dyn_cast_or_null<Function>( getter );
    RT_ASSERT( getterFunction != nullptr );
    getterFunction->setOnlyReadsMemory();
    getterFunction->setDoesNotThrow();

    Function* setterFunction = dyn_cast_or_null<Function>( setter );
    RT_ASSERT( setter == nullptr || setterFunction != nullptr );
    if( setterFunction )
        setterFunction->setDoesNotThrow();

    Function* atomicSetterFunction = dyn_cast_or_null<Function>( atomicSetter );
    RT_ASSERT( atomicSetter == nullptr || atomicSetterFunction != nullptr );
    if( atomicSetterFunction )
        atomicSetterFunction->setDoesNotThrow();

    Function* getAddressFunction = dyn_cast_or_null<Function>( getAddress );
    RT_ASSERT( getAddress == nullptr || getAddressFunction != nullptr );

    return m_up->getOrAddUberPointerKind( name, getterFunction, setterFunction, atomicSetterFunction,
                                          getAddressFunction, type, defaultValue );
}

unsigned short C14n::registerVariableName( const std::string& name )
{
    return m_objectManager->registerVariableName( name );
}

unsigned C14n::registerCallableFunctionSignature( FunctionType* ftype )
{
    return m_programManager->registerFunctionType( ftype );
}

static unsigned int deducePointeeSize( CallInst* call )
{
    StringRef functionName = call->getCalledFunction()->getName();
    RT_ASSERT_MSG( functionName == "_rt_buffer_get_id_64" || functionName == "_rt_buffer_get_64",
                   "Trying to deduce type of the output of the wrong function." );

    std::vector<Instruction*> worklist;
    std::set<Instruction*>    visited;
    Type*                     pointeeType = nullptr;

    worklist.push_back( call );

    while( !worklist.empty() )
    {
        Instruction* inst = worklist.back();
        worklist.pop_back();

        if( !visited.insert( inst ).second )
            continue;

        for( Instruction::user_iterator I = inst->user_begin(), E = inst->user_end(); I != E; ++I )
        {
            Instruction* user = dyn_cast<Instruction>( *I );
            RT_ASSERT( user != nullptr );

            if( IntToPtrInst* cast = dyn_cast<IntToPtrInst>( user ) )
            {
                PointerType* ptrType = dyn_cast<PointerType>( cast->getDestTy() );
                RT_ASSERT( ptrType != nullptr );
                // Ensure consistency with other casts.
                if( !pointeeType )
                    pointeeType = ptrType->getPointerElementType();
                else if( pointeeType != ptrType->getPointerElementType() )
                    return 0;
            }
            else if( isa<CastInst>( *I ) || isa<BinaryOperator>( *I ) || isa<PHINode>( *I ) )
            {
                worklist.push_back( user );
            }
        }
    }

    if( pointeeType )
    {
        DataLayout dataLayout( call->getParent()->getParent()->getParent()->getDataLayout() );
        return static_cast<int>( dataLayout.getTypeStoreSize( pointeeType ) );
    }
    else
    {
        // No cast found, unable to deduce type.
        return 0;
    }
}

static Value* getUseSkipCasts( Value* val )
{
    while( isa<CastInst>( val ) && val->getNumUses() == 1 )
        val = val->user_back();
    return val;
}

static Value* getDefSkipCasts( Value* val )
{
    while( CastInst* CI = dyn_cast<CastInst>( val ) )
        val = CI->getOperand( 0 );
    return val;
}

static void nameBasicBlocks( Function* function )
{
    for( Function::iterator BB = function->begin(), BBE = function->end(); BB != BBE; ++BB )
    {
        std::string              n;
        llvm::raw_string_ostream newName( n );
        newName << function->getName() << "_" << BB->getName();
        BB->setName( newName.str() );
    }
}

static bool hasDynamicPayloadAccesses( Function* function )
{
    Module* module = function->getParent();
    for( Function& modFunction : *module )
    {
        if( isPayloadSet( &modFunction ) || isPayloadGet( &modFunction ) )
        {
            for( CallInst* call : getCallsToFunction( &modFunction ) )
            {
                Value* offset = call->getArgOperand( 1 );
                if( !dyn_cast<ConstantInt>( offset ) )
                {
                    return true;
                }
            }
        }
    }
    return false;
}

// static function.
// Any knobs or Context attributes which affect canonicalization should be
// listed here so they are taken into account for disk cache lookups.
std::string C14n::getCanonicalizationOptions( const Context* context )
{
    std::ostringstream options;

    // Handle the knobs
    k_disableTerminateRay.print( options, false, false, false );
    options << ", ";
    k_disableIgnoreIntersection.print( options, false, false, false );

    // Handle the Context attributes
    if( context )
    {
        options << ", ";
        options << ( context->getForceInlineUserFunctions() ? "forceinline" : "noforceinline" );
    }

    return options.str();
}

CanonicalProgram* C14n::run()
{
    CanonicalProgram* ret = nullptr;
    switch( m_type )
    {
        case CanonicalizationType::CT_PTX:
            ret = canonicalizePtx();
            break;
        case CanonicalizationType::CT_TRAVERSER:
            ret = canonicalizeTraverser();
            break;
    }

    if( ret )
    {
        llog( 20 ) << "Created new canonical program with id " << ret->getID() << ": " << ret->getUniversallyUniqueName()
                   << " (SM min/max " << ret->getTargetMin() << "/" << ret->getTargetMax() << ")\n";
    }

    return ret;
}

CanonicalProgram* C14n::canonicalizePtx()
{
    //----------------------------------------------------------------------------
    // Create a new module with only the specified function and it's dependents.
    int id = m_cp->getID();
    dumpAsm( m_function, id, "-0-initial" );

    // Remove function pointer initializers of global variables since addStateParameter() can not handle them.
    // Needs to happen before makePartialClone() because that method leaves behind dangling uses resulting from
    // initializers of eliminated variables.
    removeFunctionPointerInitializersOfGlobalVariables( m_function->getParent() );

    Function* newFunction = makePartialClone( m_function, m_llvmManager );
    Module*   newModule   = newFunction->getParent();
    RT_ASSERT( sizeof( void* ) != 4 || newModule->getTargetTriple() == "lwptx-lwpu-lwca" );
    RT_ASSERT( sizeof( void* ) != 8 || newModule->getTargetTriple() == "lwptx64-lwpu-lwca" );
    // The existing module identifier should contain the filename where we got the original
    // source from.  Let's just preserve it in the new module name by appending it.
    newModule->setModuleIdentifier( std::string( "Canonical_" ) + m_function->getName().str() + " from "
                                    + newModule->getModuleIdentifier() );

    //----------------------------------------------------------------------------
    // assign unique name
    const std::string fname = m_cp->getUniversallyUniqueName();
    newFunction->setName( fname );
    dumpAsm( newModule, id, fname, "-1-cloned" );

    //----------------------------------------------------------------------------
    makeUserFunctionsNoInline( newModule );

    //----------------------------------------------------------------------------
    earlyCheckIntersectionProgram( newFunction );
    alignUserAllocas( newFunction );

    newFunction = addStateParameter( newFunction, m_llvmManager );
    replaceExitWithReturn( newFunction );
    deleteLwvmMetadata( newModule );
    canonicalizeVariables( newModule, m_function->getParent() );
    dumpAsm( newModule, id, fname, "-2-can-vars" );

    //----------------------------------------------------------------------------
    canonicalizeComplexFunctions( newModule );
    dumpAsm( newModule, id, fname, "-3-can-funcs" );

    //----------------------------------------------------------------------------
    m_up->finalizeUberPointerGetsAndSets( newModule );
    dumpAsm( newModule, id, fname, "-4-uberpointer" );

    //----------------------------------------------------------------------------
    canonicalizeCallableProgramFromId( newModule );
    dumpAsm( newModule, id, fname, "-5-callableProgram" );

    removeValues( m_globalsToRemove );

    //----------------------------------------------------------------------------
    ensureFirstOrderForm( newModule );
    dumpAsm( newModule, id, fname, "-6-fof" );


    //----------------------------------------------------------------------------
    // Cleanup and sanity check before optimizing
    // TODO(Kincaid): Run DCE and re-enable these.
    validateRemainingFunctions( newFunction );
    validateRemainingVariables( newModule );
    if( !checkIndirectCalls( newFunction ) )
        throw CompileError( RT_EXCEPTION_INFO, "Calls to illegal indirect functions found" );

    // If the function is a kernel, we assume that it is a normal
    // RT_PROGRAM and handle the parameters used by intersect and aabb
    // programs. Otherwise, assume it is a callable program.
    if( m_function->getCallingColw() == llvm::CallingColw::PTX_Kernel )
    {
        newFunction = handleKernelParameters( newFunction, m_llvmManager->getStatePtrType() );
        if( hasMotionIndexArg( newFunction ) )
            m_cp->markHasMotionIndexArg();
    }
    else
    {
        RT_ASSERT( newFunction->use_empty() );  // We're not expecting anyone to use an RT_CALLABLE_PROGRAM.
        RT_ASSERT( !newFunction->isDeclaration() );
        optix_exp::ErrorDetails errDetails;
        OptixResult             res = handleCallableProgramParameters( newFunction, errDetails );
        if( res != OPTIX_SUCCESS )
            throw CompileError( RT_EXCEPTION_INFO, corelib::stringf( "%s. %s", optix_exp::APIError::getErrorString( res ),
                                                                     errDetails.m_compilerFeedback.str().c_str() ) );

        // Use the public function type (without canonical state) to register the function signature in order to print at exception.
        llvm::SmallVector<Type*, 10> publicParamTypes;
        FunctionType* newFType = newFunction->getFunctionType();
        publicParamTypes.append( newFType->param_begin() + 1, newFType->param_end() );
        FunctionType* publicFType = FunctionType::get( newFType->getReturnType(), publicParamTypes, newFType->isVarArg() );
        m_cp->m_signatureId       = registerCallableFunctionSignature( publicFType );
    }
    dumpAsm( newModule, id, fname, "-7-validated" );

    //----------------------------------------------------------------------------
    optimizeModuleForC14n( newModule, k_optLevel.get(), k_enableC14nLICM.get() );
    dumpAsm( newModule, id, fname, "-8-optimized" );

    //----------------------------------------------------------------------------
    // Perform dead store elimination for UberPointer "set" function calls and
    // redundant load elimination for UberPointer "get" function calls.
    // GetSet optimization might introduce slowdowns due to values
    // for which we remove redundancies not being spilled or rematerialized
    // later if their live ranges are extended too far.
    if( k_enableGetSetOpt.get() )
    {
        const bool onlyLocal = k_onlyLocalGetSetOpt.get();
        optimizeUberPointerGetsAndSets( newFunction, onlyLocal );
        dumpAsm( newModule, id, fname, "-9-get-set-opt" );

        // Optimize again to exploit newly uncovered optimization potential. Only
        // use a minimal set of passes to limit the runtime.
        optimizeModuleAfterGetSetOpt( newModule );
    }

    //----------------------------------------------------------------------------
    removeGlobalAddressSpaceModifiers( newFunction );

    if( hasDynamicPayloadAccesses( newFunction ) )
        m_cp->markHasDynamicPayloadAccesses();

    // Here we used to run removeRedundantAddrSpaceCasts at this point.
    // Since a new version of SROA is able of removing allocations even if a pointer to it is addrspacecasted we don't need this anymore.
    addTypeLinkageFunction( newModule, m_llvmManager );
    nameBasicBlocks( newFunction );
    dumpAsm( newModule, id, fname, "-A-name-basic-blocks" );

    //----------------------------------------------------------------------------
    // Final error checks for intersection and extract deferred
    // attribute functions
    if( m_cp->m_callsPotentialIntersection || m_cp->m_callsReportIntersection )
    {
        checkIntersectionProgram( newFunction );
        makeIntersectionAndAttributeFunctions( newFunction );
        computeAttributeData32bitValues( m_cp->m_intersectionFunction );
        dumpAsm( m_cp->m_intersectionFunction->getParent(), id, fname, "-B-intersection" );
        dumpAsm( m_cp->m_attributeDecoder->getParent(), id, fname, "-C-attributeDecoder" );
    }
    else
    {
        computeAttributeData32bitValues( newFunction );
    }

    //----------------------------------------------------------------------------
    // Finalize the canonical program
    m_cp->finalize( newFunction );

    return m_cp.release();
}


const VariableReference* C14n::addVariable( const std::string& name, const VariableType& vtype )
{
    // Add variable and token
    unsigned short token         = m_objectManager->registerVariableName( name );
    bool           isInitialized = false;
    std::string    annotation    = "";

    // Do not recreate existing variable
    auto it = algorithm::find_if( m_cp->m_variableReferences, [&name]( const VariableReference* varref ) {
        return varref->getUniversallyUniqueName() == name;
    } );

    if( it != m_cp->m_variableReferences.end() )
    {
        return *it;
    }
    VariableReference* varref = new VariableReference( m_cp.get(), name, token, vtype, isInitialized, annotation );

    // Register the reference, which will assign an ID
    m_programManager->registerVariableReference( varref );

    // Add to the appropriate list in the canonical program
    m_cp->m_variableReferences.push_back( varref );

    return varref;
}


static bool getVarInfo( const Function& func, StringRef& varName, VariableType& varType, StringRef& prefix, StringRef& suffix )
{
    // Split function into prefix_varName
    auto res = func.getName().split( '_' );
    if( res.second.empty() )
        return false;
    prefix  = res.first;
    varName = res.second;

    // Is this a valid prefix?
    if( prefix != "getBufferElement" &&  // clang-format fail
        prefix != "getBufferSize" &&     //
        prefix != "getVariableValue" )
    {
        return false;
    }

    // Compute variable info.
    // TODO: This table can go away if we can figure out how to compute this stuff from just
    // the function signature.
    struct VarInfo
    {
        VariableType varType;
        const char*  suffix;
    };
    // clang-format off
  static const std::map<std::string, VarInfo> varInfoMap =
  {
    { "bvh",                { VariableType(VariableType::Buffer, 16, 1),  ".4.f.32" } },
    { "children",           { VariableType(VariableType::Buffer, 4, 1),   ".i32"    } },
    { "group_entities",     { VariableType(VariableType::Buffer, 4, 1),   ".i32"    } },
    { "primitive_entities", { VariableType(VariableType::Buffer, 8, 1),   ".i64"    } },
    { "prim_counts",        { VariableType(VariableType::Buffer, 4, 1),   ".i32"    } },
    { "motion_steps",       { VariableType(VariableType::Int,    1, 0),   ".i32"    } },
    { "motion_stride",      { VariableType(VariableType::Uint,   1, 0),   ".u32"    } },
    { "motion_time_range",  { VariableType(VariableType::Float,  2, 0),   ".f32"    } },
  };
    // clang-format on

    // Is this a supported varName?
    auto itv = varInfoMap.find( varName );
    if( itv == varInfoMap.end() )
        return false;

    const VarInfo& varInfo = itv->second;
    varType                = varInfo.varType;
    suffix                 = ( prefix != "getBufferSize" ) ? varInfo.suffix : "";
    return true;
}


CanonicalProgram* C14n::canonicalizeTraverser()
{
    Module* module = m_function->getParent();

    // assign unique name
    m_function->setName( m_cp->getUniversallyUniqueName() );

    // Handle variable references
    for( Function& func : *module )
    {
        StringRef    varName, prefix, suffix;
        VariableType varType;
        if( !getVarInfo( func, varName, varType, prefix, suffix ) )
            continue;

        // Create a variable reference
        const VariableReference* varRef = addVariable( varName, varType );

        // Rename to canonical form
        func.setName( "optixi_" + prefix + "." + varRef->getUniversallyUniqueName() + suffix );
    }

    optimizeModuleForC14n( module, k_optLevel.get(), k_enableC14nLICM.get() );

    m_cp->finalize( m_function );
    dumpAsm( module, m_cp->getID(), m_function->getName().str().c_str(), "" );

    return m_cp.release();
}

static bool containsAllocaInst( const Function* F )
{
    const BasicBlock& entry = F->getEntryBlock();
    for( const Instruction& I : entry )
        if( isa<AllocaInst>( I ) )
            return true;
    return false;
}

static void generateAttributeSetters( corelib::CoreIRBuilder&                            irb,
                                      Value*                                             attributes,
                                      Function*                                          inFunction,
                                      const CanonicalProgram::VariableReferenceListType& attributeReferences )
{
    LLVMContext& llvmContext = attributes->getContext();
    Module*      module      = inFunction->getParent();
    Value*       statePtr    = inFunction->arg_begin();
    Type*        i64Ty       = Type::getInt32Ty( llvmContext );

    unsigned int idx = 0;
    for( const VariableReference* varref : attributeReferences )
    {
        std::string name = varref->getUniversallyUniqueName();

        // Extract the value from the attribute struct
        Value* att           = irb.CreateExtractValue( attributes, {idx++} );
        Type*  attributeType = att->getType();

        // Form the setter function and call it
        Type*         argTypes[] = {statePtr->getType(), i64Ty, attributeType};
        FunctionType* fntype     = FunctionType::get( Type::getVoidTy( llvmContext ), argTypes, false );
        std::string   setterName =
            "optixi_setAttributeValue." + varref->getUniversallyUniqueName() + "." + varref->getType().toString();
        llvm::Constant* setter = module->getOrInsertFunction( setterName, fntype );
        Value*          args[] = {statePtr, ConstantInt::get( i64Ty, 0 ), att};
        irb.CreateCall( setter, args );
    }
}

static void moveInstructionAndDeps( Instruction* I, Instruction* insertBefore )
{
    BasicBlock* moveTo = insertBefore->getParent();
    for( Instruction::op_iterator Op = I->op_begin(), OpE = I->op_end(); Op != OpE; ++Op )
    {
        if( Instruction* D = dyn_cast<Instruction>( *Op ) )
        {
            if( D->getParent() != moveTo )
                D->moveBefore( insertBefore );
        }
    }
    I->moveBefore( insertBefore );
}

void C14n::makeIntersectionAndAttributeFunctions( llvm::Function* isectFunction )
{
    LLVMContext& llvmContext = isectFunction->getContext();
    Type*        i1Ty        = Type::getInt1Ty( llvmContext );
    Type*        i8Ty        = Type::getInt8Ty( llvmContext );
    Type*        i32Ty       = Type::getInt32Ty( llvmContext );
    Type*        floatTy     = Type::getFloatTy( llvmContext );

    // Clone module for the alternate intersection program
    m_cp->m_intersectionFunction = isectFunction = makePartialClone( isectFunction, m_llvmManager );
    Value*  statePtr                             = isectFunction->arg_begin();
    Type*   statePtrTy                           = statePtr->getType();
    Module* module                               = isectFunction->getParent();

    // Patch the attributes to the local allocas
    std::map<std::string, llvm::AllocaInst*> allocas;
    bool useUniqueName = true;
    patchAttributesToLocalAllocas( m_programManager, isectFunction, useUniqueName, &allocas );

    // Build an aggregate to contain all of the attributes.  Keep this order based on the
    // order of the CanonicalProgram's attribute references.
    std::vector<Type*> types;
    for( const VariableReference* varref : m_cp->getAttributeReferences() )
    {
        std::string       attrName = varref->getUniversallyUniqueName();
        llvm::AllocaInst* alloca   = allocas.at( attrName );
        types.push_back( alloca->getType()->getPointerElementType() );
    }
    Type* attributesType = StructType::get( llvmContext, types );

    // Find all of the "attribute segments", which are defined as the
    // section of code that results after rtPotentialIntersection
    // returns true and the resulting call to rtReportIntersection.
    Function*              rtPI              = getFunctionOrAssert( module, "optixi_isPotentialIntersection" );
    Function*              rtRI              = getFunctionOrAssert( module, "optixi_reportIntersection" );
    AttributeSegmentVector attributeSegments = findAttributeSegments( isectFunction, rtPI, rtRI );


    // Create the new form of reportIntersection
    Type*         argTypes[] = {statePtrTy, floatTy, i32Ty, i8Ty, attributesType};
    FunctionType* reportType = FunctionType::get( i1Ty, argTypes, false );
    std::string   name       = "optixi_reportFullIntersection." + m_cp->getUniversallyUniqueName() + ".nondeferred";
    Constant*     c          = module->getOrInsertFunction( name, reportType );
    Function*     fullReport = dyn_cast<Function>( c );
    RT_ASSERT_MSG( fullReport != nullptr, "Function already found but it is the wrong type" );


    // Colwert reportIntersection to reportFullIntersection with the
    // attributes and t-value passed in directly.
    for( size_t idx = 0; idx < attributeSegments.size(); ++idx )
    {
        AttributeSegment& segment    = attributeSegments[idx];
        Value*            attributes = UndefValue::get( attributesType );

        corelib::CoreIRBuilder irb( segment.rtRI );
        unsigned int           attIdx = 0;
        for( const VariableReference* varref : m_cp->getAttributeReferences() )
        {
            std::string       attrName = varref->getUniversallyUniqueName();
            llvm::AllocaInst* alloca   = allocas.at( attrName );
            Value*            att      = irb.CreateLoad( alloca );
            attributes                 = irb.CreateInsertValue( attributes, att, {attIdx} );
            attIdx++;
        }

        Value*    t         = segment.rtPI->getArgOperand( 1 );
        Value*    matlIndex = segment.rtRI->getArgOperand( 1 );
        Value*    hitKind   = ConstantInt::get( i8Ty, idx );
        Value*    args[]    = {statePtr, t, matlIndex, hitKind, attributes};
        CallInst* newReport = irb.CreateCall( fullReport, args );
        segment.rtRI->replaceAllUsesWith( newReport );
        segment.rtRI->eraseFromParent();
        segment.rtRI = newReport;
    }

    // Remove the original reportIntersection function
    rtRI->eraseFromParent();

    // And optimize to remove allocas
    optimizeModuleForC14n( module, k_optLevel.get(), k_enableC14nLICM.get() );

    // Decide whether to defer attributes
    // It may not be safe to use deferred attributes if:
    // 1. There are stores to any buffer
    // 2. There are writes to a global pointer
    // 3. There are stores to a ray payload
    // 4. There are calls to a callable program
    // 5. The intersect program contains an alloca
    bool defer = k_deferAttributes.get()
                 && !( m_cp->hasBufferStores() || m_cp->hasGlobalStores() || m_cp->hasPayloadStores()
                       || m_cp->callsBindlessCallableProgram() || containsAllocaInst( isectFunction ) );

    bool attributesWereDeferred = false;
    if( defer )
    {
        attributesWereDeferred = generateDeferredAttributes( isectFunction, attributeSegments );
    }

    // Deferred attributes may have declined to proceed if it was not
    // profitable. Generate a trivial decoder if we do not have a real
    // one.
    if( !attributesWereDeferred )
    {
        generateDefaultAttributeDecoder( attributesType );
    }

    // Optimize one last time
    optimizeModuleForC14n( module, k_optLevel.get(), k_enableC14nLICM.get() );


    std::string fileName( m_cp->m_intersectionFunction->getName().str() );
    dumpAsm( m_cp->m_intersectionFunction->getParent(), m_cp->getID(), fileName.c_str(), "-B-intersect-function" );
    dumpAsm( m_cp->m_attributeDecoder->getParent(), m_cp->getID(), fileName.c_str(), "-B-attribute-decoder" );
}

static void rematerializeValue( Instruction* I, corelib::CoreIRBuilder& irb, SmallSet<Value*, 16>& finished )
{
    if( !std::get<1>( finished.insert( I ) ) )
    {
        return;
    }

    // Assume that the value was rematerialized.  Restore operands
    // before moving the instruction.
    for( unsigned int i = 0; i < I->getNumOperands(); ++i )
    {
        Instruction* op = dyn_cast<Instruction>( I->getOperand( i ) );
        if( !op )
            continue;
        rematerializeValue( op, irb, finished );
    }
    I->moveBefore( &*irb.GetInsertPoint() );
}

bool C14n::generateDeferredAttributes( Function* isectFunction, std::vector<AttributeSegment>& attributeSegments )
{
    LLVMContext&           llvmContext = isectFunction->getContext();
    Value*                 statePtr    = isectFunction->arg_begin();
    Type*                  statePtrTy  = statePtr->getType();
    Type*                  i1Ty        = Type::getInt1Ty( llvmContext );
    IntegerType*           i8Ty        = Type::getInt8Ty( llvmContext );
    Type*                  i32Ty       = Type::getInt32Ty( llvmContext );
    Type*                  floatTy     = Type::getFloatTy( llvmContext );
    Type*                  voidTy      = Type::getVoidTy( llvmContext );
    Module*                module      = isectFunction->getParent();
    DataLayout             DL( module );
    corelib::CoreIRBuilder irb( &*isectFunction->getEntryBlock().getFirstInsertionPt() );

    std::vector<Function*> decoders( attributeSegments.size() );

    for( size_t idx = 0; idx < attributeSegments.size(); ++idx )
    {
        AttributeSegment& segment = attributeSegments[idx];

        // Clone function and remap the potential/report functions for
        // the attribute computation.
        ValueToValueMapTy originalToAttributeMap;
        Function* attributeFunction = CloneFunction( isectFunction, originalToAttributeMap, /*CodeInfo=*/nullptr );
        attributeFunction->setName( "__decode_attributes." + std::to_string( idx ) + "." + m_cp->getUniversallyUniqueName() );
        decoders[idx] = attributeFunction;
        module->getFunctionList().push_back( attributeFunction );
        CallInst* rtPI = cast<CallInst>( &*originalToAttributeMap.lookup( segment.rtPI ) );
        CallInst* rtRI = cast<CallInst>( &*originalToAttributeMap.lookup( segment.rtRI ) );

        // In the new attribute function, replace rtReportFullIntersection
        // with attribute setter.
        irb.SetInsertPoint( rtRI );
        Value* atts = rtRI->getArgOperand( 4 );
        generateAttributeSetters( irb, atts, attributeFunction, m_cp->m_attributeReferences );

        // Assume that rtPotentialIntersection returns true for this
        // segment.
        rtPI->replaceAllUsesWith( ConstantInt::getTrue( llvmContext ) );

        // Split the block before rtReportIntersection and insert a
        // return.
        BasicBlock* block = rtRI->getParent();
        block->splitBasicBlock( rtRI );
        block->getTerminator()->eraseFromParent();
        irb.SetInsertPoint( block );
        irb.CreateRetVoid();

        // Compute the set of live values at rtPotentialIntersection
        // in the attribute function.
        LiveValues lv( rtPI );
        lv.runOnFunction( attributeFunction );
        const InstSetVector& restoreSet = lv.getLiveValues( 0 );

        // Optimize the resulting saveset
        std::unique_ptr<SaveSetOptimizer> copt     = SaveSetOptimizer::create( k_attributeOptimizer.get(), DL );
        std::string                       idString = stringf( "atts-%02zu", idx );
        copt->run( restoreSet, idString );

#if defined( OPTIX_ENABLE_LOGGING )
        // Debugging
        if( log::active( 20 ) )
        {
            printSet( module, restoreSet, "attribute restores" );
            printSet( module, copt->getSaveSet(), "attribute saves" );
            printSet( module, copt->getRematSet(), "attribute remat" );
        }
#endif

        // create a type from the saveset
        const InstSetVector&  saveSet = copt->getSaveSet();
        std::vector<Type*>    saveTypes;
        for( Instruction* inst : saveSet )
            saveTypes.push_back( inst->getType() );
        Type* saveTy = StructType::get( llvmContext, saveTypes );

        // Ilwert the map
        ValueToValueMapTy attributeToOriginalMap;
        for( ValueToValueMapTy::const_iterator V = originalToAttributeMap.begin(), VE = originalToAttributeMap.end(); V != VE; ++V )
            attributeToOriginalMap.insert( std::make_pair( &*V->second, const_cast<Value*>( &*V->first ) ) );

        // If rtPotentialIntersection returned true, branch directly
        // to the reportIntersection (skipping the attribute
        // computation).  Otherwise, continue in the original block.
        BasicBlock*  lwrBlock        = segment.rtPI->getParent();
        Instruction* nextInstruction = &*( ++BasicBlock::iterator( segment.rtPI ) );
        BasicBlock*  nextBlock       = lwrBlock->splitBasicBlock( nextInstruction );
        BasicBlock*  reportBlock     = segment.rtRI->getParent()->splitBasicBlock( segment.rtRI );
        lwrBlock->getTerminator()->eraseFromParent();
        irb.SetInsertPoint( lwrBlock );
        irb.CreateCondBr( segment.rtPI, reportBlock, nextBlock );

        // Store saved values in the original function.
        irb.SetInsertPoint( segment.rtRI );
        Value*       savedValues = UndefValue::get( saveTy );
        unsigned int saveIndex   = 0;
        for( Instruction* inst : saveSet )
        {
            // Map the instruction from the attribute function back to the original
            RT_ASSERT( attributeToOriginalMap.count( inst ) != 0 );
            Instruction* O = cast<Instruction>( &*attributeToOriginalMap.lookup( inst ) );

            savedValues = irb.CreateInsertValue( savedValues, O, {saveIndex} );
            saveIndex++;
        }

        // Create a new call to reportIntersection.
        Type*         argTypes[] = {statePtrTy, floatTy, i32Ty, i8Ty, saveTy};
        FunctionType* reportType = FunctionType::get( i1Ty, argTypes, false );
        std::string   name =
            "optixi_reportFullIntersection." + m_cp->getUniversallyUniqueName() + ".deferred." + std::to_string( idx );
        Function* fn = Function::Create( reportType, GlobalValue::ExternalLinkage, name, module );

        // And call it
        Value* t         = segment.rtRI->getArgOperand( 1 );
        Value* matlIndex = segment.rtRI->getArgOperand( 2 );
        Value* args[]    = {statePtr, t, matlIndex, ConstantInt::get( i8Ty, idx ), savedValues};
        Value* newReport = irb.CreateCall( fn, args );
        segment.rtRI->replaceAllUsesWith( newReport );
        newReport->takeName( segment.rtRI );
        segment.rtRI->eraseFromParent();

        // Back in the attribute Function, split the entry block and
        // ensure that all alloca instructions are before the break.
        BasicBlock*  entry       = &attributeFunction->getEntryBlock();
        Instruction* entryInsert = &*( entry->getFirstInsertionPt() );
        BasicBlock*  newBlock    = entry->splitBasicBlock( entryInsert );
        for( BasicBlock::iterator iter = newBlock->begin(); iter != newBlock->end(); )
        {
            if( AllocaInst* I = dyn_cast<AllocaInst>( &*iter++ ) )
            {
                moveInstructionAndDeps( I, entryInsert );
            }
        }

        // Get the attribute data
        FunctionType* getType = FunctionType::get( saveTy, {statePtrTy}, false );
        std::string attributeGetName = "optixi_getAttributeData." + m_cp->getUniversallyUniqueName() + "." + std::to_string( idx );
        Function* attributeGet = Function::Create( getType, GlobalValue::ExternalLinkage, attributeGetName, module );
        irb.SetInsertPoint( entry->getTerminator() );
        Value* statePtr      = attributeFunction->arg_begin();
        Value* attributeData = irb.CreateCall( attributeGet, {statePtr} );

        // Restore the saved values
        SmallSet<Value*, 16> finished;
        unsigned int attidx = 0;
        for( Instruction* inst : saveSet )
        {
            Value* att = irb.CreateExtractValue( attributeData, {attidx} );
            attidx++;
            inst->replaceAllUsesWith( att );
            att->takeName( inst );
            finished.insert( att );
        }

        // Rematerialize other values that were not in the restore set.
        for( Instruction* inst : restoreSet )
        {
            if( saveSet.count( inst ) == 0 )
                rematerializeValue( inst, irb, finished );
        }

        // Branch to the rtPotentialIntersection call
        BasicBlock*  newAttributeBlock = rtPI->getParent()->splitBasicBlock( rtPI );
        Instruction* ipt               = entry->getTerminator();
        irb.SetInsertPoint( ipt );
        ipt->replaceAllUsesWith( irb.CreateBr( newAttributeBlock ) );
        ipt->eraseFromParent();
        rtPI->eraseFromParent();

        // Cleanup control flow
        legacy::FunctionPassManager FPM( attributeFunction->getParent() );
        FPM.add( createCFGSimplificationPass() );
        FPM.doInitialization();
        FPM.run( *attributeFunction );
        FPM.run( *isectFunction );
        FPM.doFinalization();
        RT_ASSERT( !verifyModule( *attributeFunction->getParent() ) );
    }

    // Now create a single function with a switch statement to each of
    // the individual attribute functions.
    FunctionType* decoderFunctionType = FunctionType::get( voidTy, {statePtrTy}, false );
    std::string   name                = "__decode_attributes." + m_cp->getUniversallyUniqueName();
    Function*     decoderFunction = Function::Create( decoderFunctionType, GlobalValue::ExternalLinkage, name, module );

    // Create a few basic blocks
    BasicBlock* entry    = BasicBlock::Create( llvmContext, "start", decoderFunction );
    BasicBlock* finished = BasicBlock::Create( llvmContext, "finished", decoderFunction );
    irb.SetInsertPoint( finished );
    irb.CreateRetVoid();
    BasicBlock* trapBlock = BasicBlock::Create( llvmContext, "ilwalidKind", decoderFunction );
    irb.SetInsertPoint( trapBlock );
    irb.CreateUnreachable();

    // Get the attributeKind
    statePtr                        = decoderFunction->arg_begin();
    FunctionType* attributeKindType = FunctionType::get( i8Ty, {statePtrTy}, false );
    Constant*     c = module->getOrInsertFunction( "optixi_getDeferredAttributeKind", attributeKindType );
    Function*     getAttributeKind = dyn_cast<Function>( c );
    RT_ASSERT( getAttributeKind != nullptr );
    irb.SetInsertPoint( entry );
    Value* kind = irb.CreateCall( getAttributeKind, {statePtr} );

    // Create a switch with a case for each kind
    SwitchInst*            sw = irb.CreateSwitch( kind, trapBlock, attributeSegments.size() );
    std::vector<CallInst*> calls( attributeSegments.size() );
    for( unsigned int idx = 0; idx < attributeSegments.size(); ++idx )
    {
        BasicBlock* newBlock = BasicBlock::Create( llvmContext, stringf( "kind%u", idx ), decoderFunction, finished );
        sw->addCase( ConstantInt::get( i8Ty, idx ), newBlock );
        irb.SetInsertPoint( newBlock );
        calls[idx] = irb.CreateCall( decoders[idx], {statePtr} );
        irb.CreateBr( finished );
    }

    // Inline the calls and erase the independent functions
    for( unsigned int idx = 0; idx < attributeSegments.size(); ++idx )
    {
        InlineFunctionInfo IFI;
        bool               success = InlineFunction( calls[idx], IFI );
        if( !success )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( calls[idx] ), "Cannot inline attribute decoder" );
        decoders[idx]->eraseFromParent();
    }

    // Put the decoder in it's own module
    Module* newModule = CloneModule( *decoderFunction->getParent() ).release();
    newModule->getFunction( isectFunction->getName() )->eraseFromParent();
    m_cp->m_attributeDecoder = newModule->getFunction( decoderFunction->getName() );

    // Remove the decoder from the original
    decoderFunction->eraseFromParent();

    return true;
}

void C14n::generateDefaultAttributeDecoder( Type* attributesType )
{
    LLVMContext& llvmContext = attributesType->getContext();
    Type*        statePtrTy  = m_llvmManager->getStatePtrType();
    Type*        voidTy      = Type::getVoidTy( llvmContext );


    Module*       module = new Module( "decoder", llvmContext );
    FunctionType* funTy  = FunctionType::get( voidTy, {statePtrTy}, false );
    Function*     fn     = Function::Create( funTy, GlobalValue::ExternalLinkage,
                                     "__decode_attributes." + m_cp->getUniversallyUniqueName(), module );
    BasicBlock* entry = BasicBlock::Create( llvmContext, "decode", fn );


    // Load attributes
    FunctionType* attributeGetTy = FunctionType::get( attributesType, statePtrTy, false );
    Constant* getFnC = module->getOrInsertFunction( "optixi_getAttributeData." + m_cp->getUniversallyUniqueName(), attributeGetTy );
    Function* getFn = dyn_cast<Function>( getFnC );
    RT_ASSERT( getFn != nullptr );
    Value* statePtr = fn->arg_begin();


    corelib::CoreIRBuilder irb( entry );
    Value*                 atts = irb.CreateCall( getFn, statePtr );
    generateAttributeSetters( irb, atts, fn, m_cp->m_attributeReferences );
    irb.CreateRetVoid();
    m_cp->m_attributeDecoder = fn;
}

void C14n::computeAttributeData32bitValues( llvm::Function* fn )
{
    int maxAttributeData32bitValues = 0;
    for( Function& F : *fn->getParent() )
    {
        if( !F.isDeclaration() )
            continue;

        if( ReportFullIntersection::isIntrinsic( &F ) )
        {
            const std::vector<CallInst*>& calls = getCallsToFunction( &F, fn );
            for( CallInst* call : calls )
            {
                ReportFullIntersection* rfi = dyn_cast<ReportFullIntersection>( call );
                RT_ASSERT( rfi );
                int numValues               = corelib::getNumRequiredRegisters( rfi->getAttributeData()->getType() );
                maxAttributeData32bitValues = std::max( numValues, maxAttributeData32bitValues );
            }
        }
    }
    m_cp->setMaxAttributeData32bitValues( maxAttributeData32bitValues );
}
