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

#ifndef _LWISA
#define _LWISA
#endif
#include <Compile/Utils.h>
#include <Context/ObjectManager.h>
#include <Context/ProgramManager.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <ExelwtionStrategy/Compile.h>
#include <ExelwtionStrategy/Plan.h>
#include <ExelwtionStrategy/RTX/RTXIntrinsics.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <FrontEnd/Canonical/IntrinsicsManager.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <Objects/VariableType.h>
#include <Util/ContainerAlgorithm.h>
#include <corelib/compiler/CoreIRBuilder.h>
#include <corelib/compiler/LLVMUtil.h>
#include <corelib/misc/String.h>
#include <corelib/system/System.h>
#include <prodlib/compiler/ModuleCache.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/math/Bits.h>
#include <prodlib/misc/TimeViz.h>
#include <prodlib/system/Knobs.h>

#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>

using namespace optix;
using namespace prodlib;
using namespace llvm;
using namespace corelib;

namespace {
// clang-format off
  Knob<std::string>  k_dbgPrintVals( RT_DSTRING("compile.dbgPrintVals"), "", RT_DSTRING("A list of values to print. Format [func1] val1 val2 ... [func2] val1 val2 ...  Prefix with * to print operand tree"));
// clang-format on

using Values = Value* [];
}

optix::AtomicSubElementType optix::valueToSubElementType( llvm::Value* val )
{
    ConstantInt* subElementType = dyn_cast<ConstantInt>( val );
    RT_ASSERT( subElementType != nullptr );
    return static_cast<AtomicSubElementType>( subElementType->getZExtValue() );
}

llvm::Value* optix::castToInt64( llvm::Instruction* insertBefore, Value* val )
{
    corelib::CoreIRBuilder irb( insertBefore );

    Value* bitCastResult = nullptr;
    if( val->getType() == irb.getFloatTy() )
    {
        Value* bc     = irb.CreateBitCast( val, irb.getInt32Ty() );
        bitCastResult = irb.CreateZExt( bc, irb.getInt64Ty() );
    }
    else
    {
        bitCastResult = irb.CreateZExtOrBitCast( val, irb.getInt64Ty() );
    }
    return bitCastResult;
}

template <typename T>
llvm::Type* optix::getAtomicOperands( T* call, llvm::Value** oOperand, llvm::Value** oCompareOperand )
{
    corelib::CoreIRBuilder irb( call );

    *oOperand                    = call->getOperand();
    llvm::Type* valueType        = ( *oOperand )->getType();
    *oCompareOperand             = call->getCompareOperand();
    AtomicSubElementType subType = valueToSubElementType( call->getSubElementType() );

    switch( subType )
    {
        case AtomicSubElementType::INT32:
        {
            *oOperand = irb.CreateTrunc( *oOperand, irb.getInt32Ty(), ( *oOperand )->getName() + ".caster" );

            *oCompareOperand =
                irb.CreateTrunc( *oCompareOperand, irb.getInt32Ty(), ( *oCompareOperand )->getName() + ".caster" );

            valueType = irb.getInt32Ty();
        }
        break;
        case AtomicSubElementType::FLOAT32:
        {
            *oOperand = irb.CreateTrunc( *oOperand, irb.getInt32Ty() );
            *oOperand = irb.CreateBitCast( *oOperand, irb.getFloatTy(), ( *oOperand )->getName() + ".caster" );

            *oCompareOperand = irb.CreateTrunc( *oCompareOperand, irb.getInt32Ty() );
            *oCompareOperand =
                irb.CreateBitCast( *oCompareOperand, irb.getFloatTy(), ( *oCompareOperand )->getName() + ".caster" );

            valueType = irb.getFloatTy();
        }
        break;
        case AtomicSubElementType::INT64:
            // No colwersion needed.
            break;
        default:
            RT_ASSERT_FAIL_MSG( "Unsupported AtomicSubElementType." );
    }

    return valueType;
}

template llvm::Type* optix::getAtomicOperands<RtxiAtomicSetBufferElement>( RtxiAtomicSetBufferElement* call,
                                                                           llvm::Value**               oOperand,
                                                                           llvm::Value** oCompareOperand );

// Style rule: helper functions used only in single .cpp file are
//             static.  Classes are placed in an anonymous namespace.

static void replaceGetBuffer( Function* F, Function* bufferElementFuncs[], std::vector<Value*>& toDelete, const ProgramManager* programManager );
static void replaceSetBuffer( Function* F, Function* bufferElementFuncs[], std::vector<Value*>& toDelete, const ProgramManager* programManager );
static void replaceGetBufferElementAddress( Function*             function,
                                            Function*             bufferElementFuncs[],
                                            const ProgramManager* programManager,
                                            std::vector<Value*>&  toDelete );
static void replaceGetTexValueFromId( Function* function, Constant* swtexonly, Constant* hwtexonly, std::vector<Value*>& toDelete );
static void replaceGetBufferSize( Function*            function,
                                  VariableReferenceID  refid,
                                  unsigned short       token,
                                  Type*                tokenType,
                                  Twine                varname,
                                  Function*            runtimeFunction,
                                  std::vector<Value*>& toDelete );
static void replaceAtomicIntrinsic( Function*             F,
                                    Type*                 tokenType,
                                    Function*             bufferElementFuncs[],
                                    const ProgramManager* programManager,
                                    std::vector<Value*>&  toDelete );
static void replaceGetPayloadAddress( Function* function, Function* runtimeFunction, std::vector<Value*>& toDelete );

static void replaceGetSetAttribute( Function*                 function,
                                    bool                      isSet,
                                    Function*                 attributeFunction,
                                    const AttributeOffsetMap& offsets,
                                    const ProgramManager*     programManager );


//------------------------------------------------------------------------------
unsigned int optix::computeAttributeOffsets( AttributeOffsetMap&                 attributeOffsets,
                                             const std::set<CanonicalProgramID>& CPs,
                                             ProgramManager*                     programManager )
{
    typedef std::map<unsigned short, std::pair<unsigned int, unsigned int>> NameToSizeAndAlignmentMap;
    NameToSizeAndAlignmentMap allAttributes;
    for( int CP : CPs )
    {
        const CanonicalProgram*                            cp         = programManager->getCanonicalProgramById( CP );
        const CanonicalProgram::VariableReferenceListType& attributes = cp->getAttributeReferences();

        for( const VariableReference* varref : attributes )
        {
            const VariableType& vtype     = varref->getType();
            unsigned int        size      = vtype.computeSize();
            unsigned int        alignment = vtype.computeAlignment();
            unsigned short      token     = varref->getVariableToken();
            std::pair<NameToSizeAndAlignmentMap::iterator, bool> inserted =
                allAttributes.insert( std::make_pair( token, std::make_pair( size, alignment ) ) );
            bool already_inserted = !inserted.second;
            if( already_inserted )
            {
                NameToSizeAndAlignmentMap::iterator pos = inserted.first;
                // Existing attribute - switch the type if the new one is larger
                if( size > pos->second.first )
                    pos->second.first = size;
                if( alignment > pos->second.second )
                    pos->second.second = alignment;
            }
        }
    }

    // Compute attribute offsets
    unsigned int lwrOffset = 0;
    attributeOffsets.clear();
    for( NameToSizeAndAlignmentMap::const_iterator iter = allAttributes.begin(); iter != allAttributes.end(); ++iter )
    {
        unsigned int size = iter->second.first;
        // We have to align the attributes, otherwise we get misaligned accesses (OP-422).
        unsigned int alignment = iter->second.second;
        lwrOffset              = align( lwrOffset, alignment );
        attributeOffsets.insert( std::make_pair( iter->first, lwrOffset ) );
        lwrOffset += size;
    }

    // TODO: Do we need to force alignment of every attribute to a
    //       16-byte boundary (OP-422)?
    //lwrOffset = align( lwrOffset, 16 );

    return lwrOffset;
}


//------------------------------------------------------------------------------
void optix::createJumpTable( Module* module, std::vector<Function*>& functions )
{
    if( GlobalVariable* gv = module->getNamedGlobal( "jumptable" ) )
    {

        // Complete the initializer with nulls for empty slots
        ArrayType*   tableType = cast<ArrayType>( gv->getType()->getElementType() );
        PointerType* pfnType   = cast<PointerType>( tableType->getElementType() );

        std::vector<Constant*> constants;
        constants.reserve( functions.size() );
        for( const auto& iter : functions )
        {
            Constant* c = iter;
            if( c == nullptr )
                constants.push_back( ConstantPointerNull::get( pfnType ) );
            else
                constants.push_back( ConstantExpr::getPointerCast( c, pfnType ) );
        }

        // Create a new global variable
        ArrayType*      newTableType = ArrayType::get( pfnType, constants.size() );
        Constant*       initializer  = ConstantArray::get( newTableType, constants );
        std::string     name         = "cort_jumptable";
        GlobalVariable* jumptable    = new GlobalVariable( *module,    // Global inserted at end of module globals list
                                                        newTableType,  // type of the variable
                                                        true,          // is this variable constant
                                                        GlobalValue::InternalLinkage,    // symbol linkage
                                                        initializer,                     // Static initializer
                                                        name,                            // Name
                                                        nullptr,                         // InsertBefore
                                                        GlobalVariable::NotThreadLocal,  // Thread local
                                                        0 );                             // The variable's address space

        // Replace the old variable and remove it from the module
        Value* cast = ConstantExpr::getBitCast( jumptable, gv->getType() );
        gv->replaceAllUsesWith( cast );
        gv->eraseFromParent();
    }
}

//------------------------------------------------------------------------------
bool optix::parsePlaceholderName( const StringRef& fname, StringRef& kind, bool& isSet, llvm::StringRef& variableReferenceUniqueName )
{
    SmallVector<StringRef, 5> matches;
    if( !optixi_PlaceholderRegex.match( fname, &matches ) )
        return false;

    // matches[0] = optixi_(get|set).(kind)Value.uniqueName.variableType
    // matches[1] = get|set
    // matches[2] = kind
    // matches[3] = unqiueName
    // matches[4] = .variableType

    isSet = ( matches[1] == "set" ? true : false );
    kind  = matches[2];

    if( kind == "Payload" )  // Payload will be handled later in parsePayloadName
        return false;

    variableReferenceUniqueName = matches[3];
    return true;
}

//------------------------------------------------------------------------------
bool optix::parseBufferSizeName( const StringRef& fname, llvm::StringRef& variableReferenceUniqueName )
{
    SmallVector<StringRef, 2> matches;

    if( !optixi_BufferSizeRegex.match( fname, &matches ) )
        return false;

    if( matches.size() != 2 )
        return false;

    // matches[0] = optixi_getBufferSize.uniqueName
    // matches[1] = uniqueName

    variableReferenceUniqueName = matches[1];
    return true;
}

//------------------------------------------------------------------------------
bool optix::parsePayloadName( const StringRef& fname, bool& isSet, llvm::StringRef& payloadName )
{
    SmallVector<StringRef, 4> matches;
    if( !optixi_PayloadRegex.match( fname, &matches ) )
        return false;
    if( matches.size() != 4 )
        return false;

    // matches[0] = optixi_(get|set)PayloadValue.prdXXb.uniqueName.type
    // matches[1] = get|set
    // matches[2] = XX (PayloadSize)
    // matches[3] = uniqueName.type

    isSet       = ( matches[1] == "set" ? true : false );
    payloadName = matches[3];

    return true;
}

bool optix::parseCallableProgramName( const StringRef& fname, bool& isBound, llvm::StringRef& variableReferenceUniqueName, unsigned& sig )
{
    SmallVector<StringRef, 5> matches;

    if( !optixi_CallableRegex.match( fname, &matches ) )
        return false;

    // matches[0] = optixi_callBound.uniqueName.sig3
    // matches[1] = Bound
    // matches[2] = .uniqueName
    // matches[3] = uniqueName
    // matches[4] = 3

    // matches[0] = optixi_callBindless.sig1
    // matches[1] = Bindless
    // matches[2] =
    // matches[3] =
    // matches[4] = 1

    // matches[0] = optixi_callBindless.callSiteUniqueName.sig1
    // matches[1] = Bindless
    // matches[2] = .callSiteUniqueName
    // matches[3] = callSiteUniqueName
    // matches[4] = 1
    if( matches.size() != 5 )
        return false;

    isBound = ( matches[1] == "Bound" );

    bool hasVariableReference = isBound || !matches[2].empty();
    if( hasVariableReference )
    {
        variableReferenceUniqueName = matches[3];
    }
    return !matches[4].getAsInteger( 0, sig );
}

//------------------------------------------------------------------------------
bool optix::isIntrinsic( StringRef functionName, const std::string& prefix, const std::string& varRefUniqueName )
{
    if( !functionName.startswith( prefix ) )
        return false;

    return functionName == prefix + "." + varRefUniqueName
           || functionName.startswith( prefix + "." + varRefUniqueName + "." );
}

//------------------------------------------------------------------------------
Function* optix::getAtomicFunction( Module* module, AtomicOpType atomicOperator, Type* type, llvm::FunctionType* fnTy )
{
    LLVMContext& context = type->getContext();
    const Type*  i32Ty   = Type::getInt32Ty( context );
    const Type*  i64Ty   = Type::getInt64Ty( context );
    const Type*  floatTy = Type::getFloatTy( context );

    const std::string ATOMIC_PREFIX = "atomic";
    std::string       functionName;

    switch( atomicOperator )
    {
        case AtomicOpType::ADD:
            functionName = "Add";
            break;
        case AtomicOpType::SUB:
            functionName = "Sub";
            break;
        case AtomicOpType::EXCH:
            functionName = "Exch";
            break;
        case AtomicOpType::MIN:
            functionName = "Min";
            break;
        case AtomicOpType::MAX:
            functionName = "Max";
            break;
        case AtomicOpType::INC:
            functionName = "Inc";
            break;
        case AtomicOpType::DEC:
            functionName = "Dec";
            break;
        case AtomicOpType::CAS:
            functionName = "CAS";
            break;
        case AtomicOpType::AND:
            functionName = "And";
            break;
        case AtomicOpType::OR:
            functionName = "Or";
            break;
        case AtomicOpType::XOR:
            functionName = "Xor";
            break;
        case AtomicOpType::INVALID:
            RT_ASSERT_FAIL_MSG( "Invalid atomic operation found" );
            break;
        default:
            RT_ASSERT_FAIL_MSG( "Unrecognized atomic operation" );
            break;
    }

    functionName         = ATOMIC_PREFIX + functionName;
    std::string nameSize = std::to_string( functionName.size() );

    std::string argumentTypes;
    if( type == i32Ty )
        argumentTypes = "jj";
    else if( type == i64Ty )
        argumentTypes = "yy";
    else if( type == floatTy )
        argumentTypes = "ff";

    if( functionName == ATOMIC_PREFIX + "CAS" )
    {
        if( type == i32Ty )
            argumentTypes += "j";
        else if( type == i64Ty )
            argumentTypes += "y";
    }

    functionName = "_ZN4cort" + nameSize + functionName + "EP" + argumentTypes;

    llvm::Function* fn = module->getFunction( functionName );
    if( fnTy == nullptr )
    {
        // When we don't have a fnTy we take what was or wasn't found.
        RT_ASSERT_MSG( fn != nullptr, "Atomic function not found: " + functionName );
        return fn;
    }
    if( fn )
    {
        RT_ASSERT_MSG( fn->getFunctionType() == fnTy, "Atomic funciton type doesn't match expected type" );
    }
    else
    {
        fn = llvm::Function::Create( fnTy, llvm::GlobalValue::ExternalLinkage, functionName, module );
    }

    return fn;
}


llvm::CallInst* optix::createAtomicCall( llvm::Type*        returnType,
                                         AtomicOpType       op,
                                         llvm::Value*       addressPtr,
                                         llvm::Value*       compareOperand,
                                         llvm::Value*       atomicOperand,
                                         llvm::Instruction* insertBefore )
{
    Module* module = insertBefore->getParent()->getParent()->getParent();

    SmallVector<Value*, 3> args;
    args.push_back( addressPtr );
    if( op == AtomicOpType::CAS )
        args.push_back( compareOperand );
    args.push_back( atomicOperand );

    SmallVector<Type*, 3> paramTypes( args.size() );
    algorithm::transform( args, paramTypes.begin(), []( Value* arg ) { return arg->getType(); } );
    FunctionType* fnTy           = FunctionType::get( returnType, paramTypes, false );
    Function*     atomicFunction = getAtomicFunction( module, op, returnType, fnTy );

    CallInst* atomicCall = corelib::CoreIRBuilder{insertBefore}.CreateCall( atomicFunction, args );
    return atomicCall;
}

//------------------------------------------------------------------------------
void optix::replacePlaceholderAccessors( Module*                   module,
                                         const AttributeOffsetMap& attributeOffsets,
                                         const ProgramManager*     programManager,
                                         bool                      hwtexonly,
                                         bool                      swtexonly,
                                         SemanticType*             stype,
                                         SemanticType*             inheritedStype )
{
    LLVMContext& context   = module->getContext();
    Type*        i8PtrTy   = Type::getInt8PtrTy( context );
    Type*        i16Ty     = Type::getInt16Ty( context );
    Type*        i32Ty     = Type::getInt32Ty( context );
    Type*        tokenType = i16Ty;

    Function* bufferSizeFunc = getFunctionOrAssert( module, "_ZN4cort14Buffer_getSizeEPNS_14CanonicalStateEt" );
    Function* bufferSizeFromIdFunc =
        getFunctionOrAssert( module, "_ZN4cort20Buffer_getSizeFromIdEPNS_14CanonicalStateEj" );
    DataLayout DL( module );
    Function*  bufferElementFunc[] = {
        getFunctionOrAssert( module, "_ZN4cort26Buffer_getElementAddress1dEPNS_14CanonicalStateEtjPcy" ),
        getFunctionOrAssert( module, "_ZN4cort26Buffer_getElementAddress2dEPNS_14CanonicalStateEtjPcyy" ),
        getFunctionOrAssert( module, "_ZN4cort26Buffer_getElementAddress3dEPNS_14CanonicalStateEtjPcyyy" ),
    };
    Function* bufferElementFromIdFunc[] = {
        getFunctionOrAssert( module, "_ZN4cort32Buffer_getElementAddress1dFromIdEPNS_14CanonicalStateEjjPcy" ),
        getFunctionOrAssert( module, "_ZN4cort32Buffer_getElementAddress2dFromIdEPNS_14CanonicalStateEjjPcyy" ),
        getFunctionOrAssert( module, "_ZN4cort32Buffer_getElementAddress3dFromIdEPNS_14CanonicalStateEjjPcyyy" ),
    };

    Constant* swtexonlyConstant = swtexonly ? ConstantInt::getTrue( context ) : ConstantInt::getFalse( context );
    Constant* hwtexonlyConstant = hwtexonly ? ConstantInt::getTrue( context ) : ConstantInt::getFalse( context );

    // TODO: toReplace is only needed as a dummy right now. we have to change signature of replaceSetBufferFromId before we can remove this,
    // which affects RTX code path
    std::vector<InstPair> toReplace;
    std::vector<Value*>   toDelete;

    auto functions = getFunctions( module );
    for( const auto& F : functions )
    {
        if( !F->isDeclaration() || !isOptixIntrinsic( F ) )
            continue;
        StringRef kind;
        bool      isSet = false;  // GCC -Werror=maybe-uninitialized
        StringRef uniqueName;

        if( AtomicSetBufferElement::isIntrinsic( F ) )
        {
            replaceAtomicIntrinsic( F, tokenType, bufferElementFunc, programManager, toDelete );
        }
        else if( AtomicSetBufferElementFromId::isIntrinsic( F ) )
        {
            replaceAtomicIntrinsicFromId<AtomicSetBufferElementFromId>( F, bufferElementFromIdFunc, true, toDelete );
        }
        else if( GetBufferElementAddress::isIntrinsic( F ) )
        {
            replaceGetBufferElementAddress( F, bufferElementFunc, programManager, toDelete );
        }
        else if( GetBufferElementAddressFromId::isIntrinsic( F ) )
        {
            replaceGetBufferElementAddressFromId<GetBufferElementAddressFromId>( F, bufferElementFromIdFunc, true, toDelete );
        }
        else if( GetPayloadAddressCall::isIntrinsic( F ) )
        {
            Function* getPayloadAddressFunc =
                getFunctionOrAssert( module, "_ZN4cort28TraceFrame_getPayloadAddressEPNS_14CanonicalStateE" );
            replaceGetPayloadAddress( F, getPayloadAddressFunc, toDelete );
        }
        else if( GetBufferElement::isIntrinsic( F ) )
        {
            replaceGetBuffer( F, bufferElementFunc, toDelete, programManager );
        }
        else if( SetBufferElement::isIntrinsic( F ) )
        {
            replaceSetBuffer( F, bufferElementFunc, toDelete, programManager );
        }
        else if( GetBufferElementFromId::isIntrinsic( F ) )
        {
            replaceGetBufferFromId<GetBufferElementFromId>( F, bufferElementFromIdFunc, true, toDelete, toReplace );
        }
        else if( SetBufferElementFromId::isIntrinsic( F ) )
        {
            replaceSetBufferFromId<SetBufferElementFromId>( F, bufferElementFromIdFunc, true, toDelete, toReplace );
        }
        else if( SetAttributeValue::isIntrinsic( F ) )
        {
            Function* attributeFunc =
                getFunctionOrAssert( module,
                                     "_ZN4cort37TraceFrame_getLwrrentAttributeAddressEPNS_14CanonicalStateEtj" );
            replaceGetSetAttribute( F, true, attributeFunc, attributeOffsets, programManager );
        }
        else if( GetAttributeValue::isIntrinsic( F ) )
        {
            Function* attributeFunc =
                getFunctionOrAssert( module,
                                     "_ZN4cort37TraceFrame_getLwrrentAttributeAddressEPNS_14CanonicalStateEtj" );
            replaceGetSetAttribute( F, false, attributeFunc, attributeOffsets, programManager );
        }
        else if( parsePlaceholderName( F->getName(), kind, isSet, uniqueName ) )
        {
            // Delete the declaration later
            toDelete.push_back( F );

            const VariableReference* varRef = programManager->getVariableReferenceByUniversallyUniqueName( uniqueName );
            const std::string&       varname = varRef->getInputName();

            // The value type is the same as the return value (for get) or the last parameter (for set)
            FunctionType* fntype     = F->getFunctionType();
            unsigned int  nargs      = fntype->getNumParams();
            Type*         valueType  = isSet ? fntype->getParamType( nargs - 1 ) : fntype->getReturnType();
            Value*        tokelwalue = ConstantInt::get( tokenType, varRef->getVariableToken() );
            for( CallInst* call : getCallsToFunction( F ) )
            {
                Value* pointer = nullptr;
                // Build the parameters which can vary slightly per kind
                SmallVector<Value*, 5> args;
                Function*              getValueFunc = nullptr;
                corelib::CoreIRBuilder irb( call );
                Value*                 state = call->getArgOperand( 0 );
                args.push_back( state );
                args.push_back( tokelwalue );

                if( kind == "Variable" )
                {

                    Value* defaultValue =
                        irb.CreateBitCast( call->getArgOperand( 2 ), i8PtrTy, varname + ".defaultVal" );
                    args.push_back( defaultValue );
                    if( stype )
                    {
                        RT_ASSERT( inheritedStype );
                        args.push_back( ConstantInt::get( i32Ty, *stype ) );
                        args.push_back( ConstantInt::get( i32Ty, *inheritedStype ) );
                    }

                    // clang-format off
          Function* variableFunc = stype ?
            getFunctionOrAssert( module, "_ZN4cort29Runtime_lookupVariableAddressEPNS_14CanonicalStateEtPcN5optix12SemanticTypeES4_" ) :
            getFunctionOrAssert( module, "_ZN4cort29Runtime_lookupVariableAddressEPNS_14CanonicalStateEtPc" );
                    // clang-format on

                    pointer       = irb.CreateCall( variableFunc, args, varname + ".ptr" );
                    Value* offset = call->getArgOperand( 1 );
                    pointer       = irb.CreateGEP( pointer, offset );
                    pointer       = irb.CreateBitCast( pointer, valueType->getPointerTo(), varname + ".typedPtr" );
                }
                else if( kind == "Attribute" )
                {
                    RT_ASSERT_FAIL_MSG( "Replacement of Attribute placeholders should not be performed here." );
                }
                else if( kind.startswith( "Texture_" ) )
                {
                    unsigned int N = nargs - 1;
                    RT_ASSERT( N >= 1 && N <= 3 );

                    TextureLookup::LookupKind lkind = TextureLookup::fromString( kind.drop_front( 8 ) );
                    getValueFunc                    = TextureLookup::getLookupFunction( lkind, "token", module );
                    args.push_back( hwtexonlyConstant );
                    args.push_back( swtexonlyConstant );
                    for( unsigned int i = 0; i < N; i++ )
                        args.push_back( call->getArgOperand( 1 + i ) );
                }
                else if( kind == "Buffer" )
                {
                    RT_ASSERT_FAIL_MSG( "Replacement of Buffer placeholders should not be performed here." );
                }
                else
                {
                    RT_ASSERT_FAIL_MSG( LLVMErrorInfo( F ) + " Invalid placeholder kind: "
                                        + std::string( isSet ? "set" : "get" ) + kind.str() );
                }
                if( getValueFunc )
                {
                    // Texture returns a value, not a pointer.
                    RT_ASSERT( !isSet );
                    Instruction* tex = irb.CreateCall( getValueFunc, args );
                    call->replaceAllUsesWith( tex );
                    call->eraseFromParent();
                }
                if( pointer != nullptr )
                {
                    unsigned int size  = DL.getTypeStoreSize( valueType );
                    unsigned int align = MinAlign( size, 16 );
                    if( isSet )
                    {
                        Value*       value = call->getArgOperand( nargs - 1 );
                        Instruction* store = irb.CreateAlignedStore( value, pointer, align );
                        RT_ASSERT_MSG( call->getType()->isVoidTy(),
                                       "Changed a placeholder function to now use the interstate?" );
                        call->replaceAllUsesWith( store );
                        call->eraseFromParent();
                    }
                    else
                    {
                        // Get
                        Instruction* load = irb.CreateAlignedLoad( pointer, align );
                        call->replaceAllUsesWith( load );
                        call->eraseFromParent();
                    }
                }
            }
        }
        else if( parseBufferSizeName( F->getName(), uniqueName ) )
        {
            const VariableReference* varRef = programManager->getVariableReferenceByUniversallyUniqueName( uniqueName );

            const std::string& varname = varRef->getInputName();
            replaceGetBufferSize( F, varRef->getReferenceID(), varRef->getVariableToken(), tokenType, varname,
                                  bufferSizeFunc, toDelete );
        }
        else if( F->getName() == GET_BUFFER_SIZE_ID )
        {
            replaceGetBufferSizeFromId( F, bufferSizeFromIdFunc, toDelete );
        }
        else if( parsePayloadName( F->getName(), isSet, uniqueName ) )
        {
            // The value type is the same as the return value (for get) or the last parameter (for set)
            // Since we use the user-allocated payload pointer directly we have to be conservative
            // with the alignment we assume.
            FunctionType* fntype    = F->getFunctionType();
            unsigned int  nargs     = fntype->getNumParams();
            Type*         valueType = isSet ? fntype->getParamType( nargs - 1 ) : fntype->getReturnType();
            unsigned int  prdAlign  = 0;

            Function* getPayloadAddressFunc =
                getFunctionOrAssert( module, "_ZN4cort28TraceFrame_getPayloadAddressEPNS_14CanonicalStateE" );

            auto calls = getCallsToFunction( F );
            for( const auto& call : calls )
            {
                // Build the parameters
                Value*                 state  = call->getArgOperand( 0 );
                Value*                 args[] = {state};
                corelib::CoreIRBuilder irb( call );
                // Call the function and load or store the value
                Value* i8ptr  = irb.CreateCall( getPayloadAddressFunc, args, uniqueName + ".i8ptr" );
                Value* offset = nullptr;
                if( isSet )
                    offset = call->getArgOperand( nargs - 2 );
                else
                    offset = call->getArgOperand( nargs - 1 );

                RT_ASSERT( offset->getType()->isIntegerTy() );

                Value* gepArgs[]     = {offset};
                i8ptr                = irb.CreateGEP( i8ptr, gepArgs );
                Value*       ptr     = irb.CreateBitCast( i8ptr, valueType->getPointerTo(), uniqueName + ".typedPtr" );
                Instruction* newInst = nullptr;
                if( isSet )
                {
                    Value* value = call->getArgOperand( nargs - 1 );
                    newInst      = irb.CreateAlignedStore( value, ptr, prdAlign );
                }
                else
                {
                    // Get
                    newInst = irb.CreateAlignedLoad( ptr, prdAlign );
                }
                call->replaceAllUsesWith( newInst );
                call->eraseFromParent();
            }
        }
        else if( F->getName().startswith( "optixi_getTexture" ) && F->getName().endswith( "ValueFromId" ) )
        {
            replaceGetTexValueFromId( F, swtexonlyConstant, hwtexonlyConstant, toDelete );
        }
    }

    RT_ASSERT_MSG( toReplace.empty(), "toReplace should be empty!" );  // TODO remove (see above)

    for( auto inst : toDelete )
    {
        if( GlobalValue* gv = dyn_cast<GlobalValue>( inst ) )
            gv->eraseFromParent();
        else if( Instruction* instruction = dyn_cast<Instruction>( inst ) )
            instruction->eraseFromParent();
        else
            RT_ASSERT_FAIL_MSG( LLVMErrorInfo( inst ) + " toDelete value is not GlobalValue or Instruction" );
    }
}

//------------------------------------------------------------------------------
static void replaceGetTexValueFromId( Function* function, Constant* swtexonly, Constant* hwtexonly, std::vector<Value*>& toDelete )
{
    Module*      module   = function->getParent();
    LLVMContext& context  = module->getContext();
    StringRef    funcName = function->getName();
    StringRef    kind     = funcName.substr( 18, funcName.find( "ValueFromId" ) - 18 );

    FunctionType* fntype = function->getFunctionType();
    unsigned int  nargs  = fntype->getNumParams();
    unsigned int  N      = nargs - 2;
    auto          calls  = getCallsToFunction( function );
    for( const auto& call : calls )
    {
        // Build the parameters which can vary slightly per kind
        SmallVector<Value*, 8> args;
        Function*              getValueFunc = nullptr;
        Value*                 state        = call->getArgOperand( 0 );
        corelib::CoreIRBuilder irb( call );
        args.push_back( state );

        // TODO: MIP SW texture
        Constant* swtexonlyConstant = ConstantInt::getFalse( context );
        Constant* hwtexonlyConstant = ConstantInt::getTrue( context );

        TextureLookup::LookupKind lkind = TextureLookup::fromString( kind );
        args.push_back( call->getArgOperand( 1 ) );
        getValueFunc = TextureLookup::getLookupFunction( lkind, "id", module );
        args.push_back( hwtexonlyConstant );
        args.push_back( swtexonlyConstant );
        for( unsigned int i = 0; i < N; ++i )
            args.push_back( call->getArgOperand( 2 + i ) );

        Instruction* texrResult = irb.CreateCall( getValueFunc, args );
        call->replaceAllUsesWith( texrResult );
        call->eraseFromParent();
    }
}

//------------------------------------------------------------------------------
static void replaceGetBuffer( Function* F, Function* bufferElementFuncs[], std::vector<Value*>& toDelete, const ProgramManager* programManager )
{
    Module*      module = F->getParent();
    DataLayout   DL( module );
    LLVMContext& context = module->getContext();
    Type*        i8Ty    = Type::getInt8Ty( context );
    Type*        i16Ty   = Type::getInt16Ty( context );
    Type*        i32Ty   = Type::getInt32Ty( context );
    toDelete.push_back( F );

    for( CallInst* I : getCallsToFunction( F ) )
    {
        GetBufferElement* call = dyn_cast<GetBufferElement>( I );
        if( !call )
            continue;

        unsigned int             dimensions = call->getDimensionality();
        std::string              uniqueName = GetBufferElement::parseUniqueName( F->getName() );
        const VariableReference* varref     = programManager->getVariableReferenceByUniversallyUniqueName( uniqueName );
        int                      token      = varref->getVariableToken();
        Value*                   tokelwalue = ConstantInt::get( i16Ty, token, false );
        Value*                   offset     = call->getOffset();
        Type*                    valueType  = call->getCalledFunction()->getReturnType();
        int                      size       = DL.getTypeStoreSize( valueType );
        corelib::CoreIRBuilder   irb( call );
        corelib::CoreIRBuilder   irb_entry( corelib::getFirstNonAlloca( call->getParent()->getParent() ) );

        AllocaInst* stackTmp = irb_entry.CreateAlloca( i8Ty, ConstantInt::get( i32Ty, 16 ), "stackTmp" );
        stackTmp->setAlignment( 16 );
        Function* getAddressFunc = bufferElementFuncs[dimensions - 1];

        std::vector<Value*> args = {call->getStatePtr(), tokelwalue, call->getElementSize(), stackTmp};
        for( unsigned int i = 0; i < dimensions; ++i )
            args.push_back( call->getIndex( i ) );

        Value*       basePointer = irb.CreateCall( getAddressFunc, args );
        Value*       i8ptr       = irb.CreateGEP( basePointer, offset );
        Value*       ptr         = irb.CreateBitCast( i8ptr, valueType->getPointerTo(), "typedPtr" );
        unsigned int align       = MinAlign( size, 16 );
        Instruction* load        = irb.CreateAlignedLoad( ptr, align );
        call->replaceAllUsesWith( load );
        call->eraseFromParent();
    }
}

//------------------------------------------------------------------------------
static void replaceSetBuffer( Function* F, Function* bufferElementFuncs[], std::vector<Value*>& toDelete, const ProgramManager* programManager )
{
    Module*      module = F->getParent();
    DataLayout   DL( module );
    LLVMContext& context = module->getContext();
    Type*        i8Ty    = Type::getInt8Ty( context );
    Type*        i16Ty   = Type::getInt16Ty( context );
    Type*        i32Ty   = Type::getInt32Ty( context );
    toDelete.push_back( F );

    for( CallInst* I : getCallsToFunction( F ) )
    {
        SetBufferElement* call = dyn_cast<SetBufferElement>( I );
        if( !call )
            continue;

        unsigned int             dimensions = call->getDimensionality();
        std::string              uniqueName = SetBufferElement::parseUniqueName( F->getName() );
        const VariableReference* varref     = programManager->getVariableReferenceByUniversallyUniqueName( uniqueName );

        int                    token      = varref->getVariableToken();
        Value*                 tokelwalue = ConstantInt::get( i16Ty, token, false );
        Value*                 offset     = call->getOffset();
        Value*                 toStore    = call->getValueToSet();
        Type*                  valueType  = toStore->getType();
        int                    size       = DL.getTypeStoreSize( valueType );
        corelib::CoreIRBuilder irb( call );
        corelib::CoreIRBuilder irb_entry( corelib::getFirstNonAlloca( call->getParent()->getParent() ) );

        AllocaInst* stackTmp = irb_entry.CreateAlloca( i8Ty, ConstantInt::get( i32Ty, 16 ), "stackTmp" );
        stackTmp->setAlignment( 16 );
        Function* getAddressFunc = bufferElementFuncs[dimensions - 1];

        std::vector<Value*> args = {call->getStatePtr(), tokelwalue, call->getElementSize(), stackTmp};
        for( unsigned int i = 0; i < dimensions; ++i )
            args.push_back( call->getIndex( i ) );

        Value* basePointer = irb.CreateCall( getAddressFunc, args );

        Value*       i8ptr = irb.CreateGEP( basePointer, offset );
        Value*       ptr   = irb.CreateBitCast( i8ptr, valueType->getPointerTo(), "typedPtr" );
        unsigned int align = MinAlign( size, 16 );
        Instruction* store = irb.CreateAlignedStore( toStore, ptr, align );
        call->replaceAllUsesWith( store );
        call->eraseFromParent();
    }
}

//------------------------------------------------------------------------------
static void replaceGetBufferSize( Function*            function,
                                  VariableReferenceID  refid,
                                  unsigned short       token,
                                  Type*                tokenType,
                                  const Twine          varname,
                                  Function* const      runtimeFunction,
                                  std::vector<Value*>& toDelete )
{
    // Delete the declaration later
    toDelete.push_back( function );

    // The value type is the same as the return value, which should
    // be the same size as the uint3.
    Value* tokelwalue = ConstantInt::get( tokenType, token );
    for( CallInst* call : getCallsToFunction( function ) )
    {
        corelib::CoreIRBuilder irb( call );
        Value*                 state = call->getArgOperand( 0 );

        // Call the function and return the value
        Value*       args[] = {state, tokelwalue};
        Instruction* size3  = irb.CreateCall( runtimeFunction, args, varname + ".size" );
        call->replaceAllUsesWith( size3 );
        call->eraseFromParent();
    }
}

//------------------------------------------------------------------------------
void optix::replaceGetBufferSizeFromId( Function* function, Function* const runtimeFunction, std::vector<Value*>& toDelete )
{
    // Delete the declaration later
    toDelete.push_back( function );

    // The value type is the same as the return value, which should
    // be the same size as the uint3.
    for( CallInst* call : getCallsToFunction( function ) )
    {
        corelib::CoreIRBuilder irb( call );
        Value*                 state    = call->getArgOperand( 0 );
        Value*                 bufferId = call->getArgOperand( 1 );

        // Call the function and return the value
        Value*       args[] = {state, bufferId};
        Instruction* size3  = irb.CreateCall( runtimeFunction, args, ".size" );
        call->replaceAllUsesWith( size3 );
        call->eraseFromParent();
    }
}

//------------------------------------------------------------------------------
template <class Intrinsic>
void optix::replaceGetBufferFromId( Function*              F,
                                    Function*              bufferElementFuncs[],
                                    bool                   generateAlloca,
                                    std::vector<Value*>&   toDelete,
                                    std::vector<InstPair>& toReplace )
{
    Module*      module = F->getParent();
    DataLayout   DL( module );
    LLVMContext& context = F->getContext();
    Type*        i8Ty    = Type::getInt8Ty( context );
    Type*        i32Ty   = Type::getInt32Ty( context );
    toDelete.push_back( F );

    for( CallInst* CI : getCallsToFunction( F ) )
    {
        Intrinsic* call = dyn_cast<Intrinsic>( CI );
        RT_ASSERT( call );

        Value*                 state       = call->getStatePtr();
        Value*                 bufferId    = call->getBufferId();
        Value*                 elementSize = call->getElementSize();
        Value*                 offset      = call->getOffset();
        corelib::CoreIRBuilder irb( call );

        std::vector<Value*> args = {state, bufferId, elementSize};
        if( generateAlloca )
        {
            corelib::CoreIRBuilder irb_entry( corelib::getFirstNonAlloca( call->getParent()->getParent() ) );
            AllocaInst*            stackTmp = irb_entry.CreateAlloca( i8Ty, ConstantInt::get( i32Ty, 16 ), "stackTmp" );
            stackTmp->setAlignment( 16 );
            args.push_back( stackTmp );
        }

        Type* valueType = call->getCalledFunction()->getReturnType();
        int   size      = DL.getTypeStoreSize( valueType );

        unsigned dimensions = call->getDimensionality();

        for( unsigned int i = 0; i < dimensions; ++i )
            args.push_back( call->getIndex( i ) );

        Function* getAddressFunc = bufferElementFuncs[dimensions - 1];

        Value*       basePointer = irb.CreateCall( getAddressFunc, args );
        Value*       i8ptr       = irb.CreateGEP( basePointer, offset );
        Value*       ptr         = irb.CreateBitCast( i8ptr, valueType->getPointerTo(), "typedPtr" );
        unsigned int align       = MinAlign( size, 16 );
        Instruction* load        = irb.CreateAlignedLoad( ptr, align );
        call->replaceAllUsesWith( load );
        call->eraseFromParent();
    }
}

template void optix::replaceGetBufferFromId<GetBufferElementFromId>( llvm::Function*       F,
                                                                     llvm::Function*       bufferElementFuncs[],
                                                                     bool                  generateAlloca,
                                                                     corelib::ValueVector& toDelete,
                                                                     std::vector<corelib::InstPair>& toReplace );

template void optix::replaceGetBufferFromId<RtxiGetBufferElementFromId>( llvm::Function*       F,
                                                                         llvm::Function*       bufferElementFuncs[],
                                                                         bool                  generateAlloca,
                                                                         corelib::ValueVector& toDelete,
                                                                         std::vector<corelib::InstPair>& toReplace );


//------------------------------------------------------------------------------
template <class Intrinsic>
void optix::replaceSetBufferFromId( Function*              F,
                                    Function*              bufferElementFuncs[],
                                    bool                   generateAlloca,
                                    std::vector<Value*>&   toDelete,
                                    std::vector<InstPair>& toReplace )
{
    Module*      module = F->getParent();
    DataLayout   DL( module );
    LLVMContext& context = F->getContext();
    Type*        i8Ty    = Type::getInt8Ty( context );
    Type*        i32Ty   = Type::getInt32Ty( context );
    toDelete.push_back( F );

    for( CallInst* CI : getCallsToFunction( F ) )
    {
        Intrinsic* call = dyn_cast<Intrinsic>( CI );
        RT_ASSERT( call );

        Value*                 state       = call->getStatePtr();
        Value*                 bufferId    = call->getBufferId();
        Value*                 elementSize = call->getElementSize();
        Value*                 offset      = call->getOffset();
        Type*                  valueType   = call->getValueToSet()->getType();
        Value*                 toStore     = call->getValueToSet();
        int                    size        = DL.getTypeStoreSize( valueType );
        corelib::CoreIRBuilder irb( call );

        std::vector<Value*> args = {state, bufferId, elementSize};
        if( generateAlloca )
        {
            corelib::CoreIRBuilder irb_entry( corelib::getFirstNonAlloca( call->getParent()->getParent() ) );
            AllocaInst*            stackTmp = irb_entry.CreateAlloca( i8Ty, ConstantInt::get( i32Ty, 16 ), "stackTmp" );
            stackTmp->setAlignment( 16 );
            args.push_back( stackTmp );
        }

        unsigned dimensions = call->getDimensionality();

        for( unsigned int i = 0; i < dimensions; ++i )
            args.push_back( call->getIndex( i ) );

        Function* getAddressFunc = bufferElementFuncs[dimensions - 1];

        Value*       basePointer = irb.CreateCall( getAddressFunc, args );
        Value*       i8ptr       = irb.CreateGEP( basePointer, offset );
        Value*       ptr         = irb.CreateBitCast( i8ptr, valueType->getPointerTo(), "typedPtr" );
        unsigned int align       = MinAlign( size, 16 );
        Instruction* store       = irb.CreateAlignedStore( toStore, ptr, align );
        call->replaceAllUsesWith( store );
        call->eraseFromParent();
    }
}

template void optix::replaceSetBufferFromId<SetBufferElementFromId>( Function*              F,
                                                                     Function*              bufferElementFuncs[],
                                                                     bool                   generateAlloca,
                                                                     std::vector<Value*>&   toDelete,
                                                                     std::vector<InstPair>& toReplace );

template void optix::replaceSetBufferFromId<RtxiSetBufferElementFromId>( Function*              F,
                                                                         Function*              bufferElementFuncs[],
                                                                         bool                   generateAlloca,
                                                                         std::vector<Value*>&   toDelete,
                                                                         std::vector<InstPair>& toReplace );

//------------------------------------------------------------------------------
void replaceGetSetAttribute( Function* function, bool isSet, Function* attributeFunction, const AttributeOffsetMap& offsets, const ProgramManager* programManager )
{
    const bool     hasInterstate = function->getFunctionType()->getNumParams() == 4;
    unsigned short token         = 0;
    StringRef      uniqueName;
    LLVMContext&   context = function->getContext();
    Type*          i16Ty   = Type::getInt16Ty( context );
    Type*          i32Ty   = Type::getInt32Ty( context );
    DataLayout     dataLayout( function->getParent() );

    bool success = isSet ? SetAttributeValue::parseUniqueName( function, uniqueName ) :
                           GetAttributeValue::parseUniqueName( function, uniqueName );
    RT_ASSERT( success );
    (void)success;

    const VariableReference* varref = programManager->getVariableReferenceByUniversallyUniqueName( uniqueName );
    token                           = varref->getVariableToken();

    for( CallInst* call : getCallsToFunction( function ) )
    {
        Function*              calledFunction = call->getCalledFunction();
        corelib::CoreIRBuilder irb( call );
        Type* accessType = isSet ? calledFunction->getFunctionType()->getParamType( call->getNumArgOperands() - 1 ) :
                                   calledFunction->getReturnType();

        std::vector<Value*> args;
        Value*              pointerOffset = nullptr;
        if( hasInterstate )
        {
            Value* interstate = call->getArgOperand( 0 );
            Value* state      = call->getArgOperand( 1 );
            pointerOffset     = call->getArgOperand( 2 );
            args.push_back( interstate );
            args.push_back( state );
        }
        else
        {
            Value* state  = call->getArgOperand( 0 );
            pointerOffset = call->getArgOperand( 1 );
            args.push_back( state );
        }
        args.push_back( ConstantInt::get( i16Ty, token, false ) );

        auto offset = offsets.find( token );
        RT_ASSERT_MSG( offset != offsets.end(), "Attribute offset not found" );
        Value* offsetValue = ConstantInt::get( i32Ty, offset->second );
        args.push_back( offsetValue );

        // Call the function and load or store the value
        Value* ptr           = irb.CreateCall( attributeFunction, args, uniqueName + ".ptr" );
        ptr                  = irb.CreateGEP( ptr, pointerOffset );
        ptr                  = irb.CreateBitCast( ptr, accessType->getPointerTo(), uniqueName + ".typedPtr" );
        unsigned int size    = dataLayout.getTypeStoreSize( accessType );
        unsigned int align   = MinAlign( size, 16 );
        Instruction* newInst = nullptr;
        if( isSet )
        {
            Value* value = call->getArgOperand( call->getNumArgOperands() - 1 );
            newInst      = irb.CreateAlignedStore( value, ptr, align );
            RT_ASSERT_MSG( call->getType()->isVoidTy(), "Changed a placeholder function to now use the interstate?" );
        }
        else
        {
            newInst = irb.CreateAlignedLoad( ptr, align );
        }
        call->replaceAllUsesWith( newInst );
        call->eraseFromParent();
    }
}

//------------------------------------------------------------------------------
template <typename CallType>
static void replaceAtomicIntrinsicCall( CallType* call, std::vector<Value*>& bufferAccessArgs, Value* offset, Function* bufferElementFuncs[] )
{
    Module*      module  = call->getParent()->getParent()->getParent();
    LLVMContext& context = module->getContext();
    Type*        i64Ty   = Type::getInt64Ty( context );

    // Extract the atomic operand and compare operand colwerted to the sub element type from the call.
    llvm::Value* atomicOperand  = nullptr;
    llvm::Value* compareOperand = nullptr;
    llvm::Type*  valueType      = getAtomicOperands( call, &atomicOperand, &compareOperand );

    unsigned dimensions = call->getDimensionality();
    for( unsigned int i = 0; i < dimensions; ++i )
        bufferAccessArgs.push_back( call->getIndex( i ) );

    Value*       atomicOperation = call->getOperation();
    ConstantInt* opType          = dyn_cast<ConstantInt>( atomicOperation );
    RT_ASSERT_MSG( opType != nullptr, "Atomic operation type is not a constant" );
    AtomicOpType op             = static_cast<AtomicOpType>( opType->getZExtValue() );
    Function*    bufferFunction = bufferElementFuncs[dimensions - 1];
    RT_ASSERT( bufferFunction != nullptr );

    corelib::CoreIRBuilder irb( call );
    CallInst*              basePtr    = irb.CreateCall( bufferFunction, bufferAccessArgs );
    Value*                 base       = irb.CreatePtrToInt( basePtr, i64Ty );
    Value*                 address    = irb.CreateBinOp( Instruction::Add, base, offset );
    Value*                 addressPtr = irb.CreateIntToPtr( address, PointerType::getUnqual( valueType ) );

    CallInst* atomicCall = createAtomicCall( valueType, op, addressPtr, compareOperand, atomicOperand, call );

    // Cast the atomic call back to i64
    Value* bitCastResult = castToInt64( call, atomicCall );

    call->replaceAllUsesWith( bitCastResult );
    call->eraseFromParent();
}

void replaceAtomicIntrinsic( Function*             F,
                             Type*                 tokenType,
                             Function*             bufferElementFuncs[],
                             const ProgramManager* programManager,
                             std::vector<Value*>&  toDelete )
{
    Module*      module  = F->getParent();
    LLVMContext& context = module->getContext();
    Type*        i8Ty    = Type::getInt8Ty( context );
    Type*        i32Ty   = Type::getInt32Ty( context );
    toDelete.push_back( F );
    Value* tokelwalue = ConstantInt::get( tokenType, getOptixAtomicToken( F, programManager ) );

    for( CallInst* I : getCallsToFunction( F ) )
    {
        AtomicSetBufferElement* call = dyn_cast<AtomicSetBufferElement>( I );
        if( !call )
            continue;

        Value* state     = call->getStatePtr();
        Value* sizeValue = call->getElementSize();
        Value* offset    = call->getOffset();

        corelib::CoreIRBuilder irb_entry( corelib::getFirstNonAlloca( call->getParent()->getParent() ) );

        AllocaInst* stackTmp = irb_entry.CreateAlloca( i8Ty, ConstantInt::get( i32Ty, 16 ), "stackTmp" );
        stackTmp->setAlignment( 16 );

        std::vector<Value*> bufferAccessArgs = {state, tokelwalue, sizeValue, stackTmp};

        replaceAtomicIntrinsicCall( call, bufferAccessArgs, offset, bufferElementFuncs );
    }
}

//------------------------------------------------------------------------------
template <class Intrinsic>
void optix::replaceAtomicIntrinsicFromId( Function* F, Function* bufferElementFuncs[], bool generateAlloca, std::vector<Value*>& toDelete )
{
    Module*      module  = F->getParent();
    LLVMContext& context = module->getContext();
    Type*        i8Ty    = Type::getInt8Ty( context );
    Type*        i32Ty   = Type::getInt32Ty( context );
    toDelete.push_back( F );

    for( CallInst* I : getCallsToFunction( F ) )
    {
        Intrinsic* call = dyn_cast<Intrinsic>( I );
        RT_ASSERT( call );

        Value* state     = call->getStatePtr();
        Value* bufferId  = call->getBufferId();
        Value* sizeValue = call->getElementSize();
        Value* offset    = call->getOffset();

        std::vector<Value*> bufferAccessArgs = {state, bufferId, sizeValue};
        if( generateAlloca )
        {
            corelib::CoreIRBuilder irb_entry( corelib::getFirstNonAlloca( call->getParent()->getParent() ) );
            AllocaInst*            stackTmp = irb_entry.CreateAlloca( i8Ty, ConstantInt::get( i32Ty, 16 ), "stackTmp" );
            stackTmp->setAlignment( 16 );
            bufferAccessArgs.push_back( stackTmp );
        }

        replaceAtomicIntrinsicCall( call, bufferAccessArgs, offset, bufferElementFuncs );
    }
}

template void optix::replaceAtomicIntrinsicFromId<AtomicSetBufferElementFromId>( Function* F,
                                                                                 Function* bufferElementFuncs[],
                                                                                 bool      generateAlloca,
                                                                                 std::vector<Value*>& toDelete );

template void optix::replaceAtomicIntrinsicFromId<RtxiAtomicSetBufferElementFromId>( Function* F,
                                                                                     Function* bufferElementFuncs[],
                                                                                     bool      generateAlloca,
                                                                                     std::vector<Value*>& toDelete );

//------------------------------------------------------------------------------
static void replaceGetBufferElementAddress( Function*             F,
                                            Function*             bufferElementFuncs[],
                                            const ProgramManager* programManager,
                                            std::vector<Value*>&  toDelete )
{
    Module*      module  = F->getParent();
    LLVMContext& context = module->getContext();
    Type*        i8Ty    = Type::getInt8Ty( context );
    Type*        i16Ty   = Type::getInt16Ty( context );
    Type*        i32Ty   = Type::getInt32Ty( context );
    Type*        i64Ty   = Type::getInt64Ty( context );
    toDelete.push_back( F );

    auto calls = getCallsToFunction( F );
    for( const auto& I : calls )
    {
        GetBufferElementAddress* call = dyn_cast<GetBufferElementAddress>( I );
        if( !call )
            continue;

        StringRef                uniqueName = call->parseUniqueName();
        const VariableReference* varref     = programManager->getVariableReferenceByUniversallyUniqueName( uniqueName );
        unsigned short           token      = varref->getVariableToken();
        corelib::CoreIRBuilder   irb( call );
        corelib::CoreIRBuilder   irb_entry( corelib::getFirstNonAlloca( call->getParent()->getParent() ) );

        Value* tokelwalue = ConstantInt::get( i16Ty, token, false );

        AllocaInst* stackTmp = irb_entry.CreateAlloca( i8Ty, ConstantInt::get( i32Ty, 16 ), "stackTmp" );
        stackTmp->setAlignment( 16 );
        std::vector<Value*> args = {call->getStatePtr(), tokelwalue, call->getElementSize(), stackTmp};

        unsigned int dimensions = call->getDimensionality();
        for( unsigned int i = 0; i < dimensions; ++i )
            args.push_back( call->getIndex( i ) );

        Function* runtimeFunction = bufferElementFuncs[dimensions - 1];
        CallInst* getPointerCall  = irb.CreateCall( runtimeFunction, args );
        Value*    rawPointer      = irb.CreatePtrToInt( getPointerCall, i64Ty );

        call->replaceAllUsesWith( rawPointer );
        call->eraseFromParent();
    }
}

//------------------------------------------------------------------------------
template <class Intrinsic>
void optix::replaceGetBufferElementAddressFromId( Function*            F,
                                                  Function*            bufferElementFromIdFuncs[],
                                                  bool                 generateAlloca,
                                                  std::vector<Value*>& toDelete )
{
    Module*      module  = F->getParent();
    LLVMContext& context = module->getContext();
    Type*        i8Ty    = Type::getInt8Ty( context );
    Type*        i32Ty   = Type::getInt32Ty( context );
    Type*        i64Ty   = Type::getInt64Ty( context );
    toDelete.push_back( F );

    auto calls = getCallsToFunction( F );
    for( const auto& I : calls )
    {
        Intrinsic* call = dyn_cast<Intrinsic>( I );
        RT_ASSERT( call );

        Value*                 idValue = call->getBufferId();
        corelib::CoreIRBuilder irb( call );

        std::vector<Value*> args = {call->getStatePtr(), idValue, call->getElementSize()};
        if( generateAlloca )
        {
            corelib::CoreIRBuilder irb_entry( corelib::getFirstNonAlloca( call->getParent()->getParent() ) );
            AllocaInst*            stackTmp = irb_entry.CreateAlloca( i8Ty, ConstantInt::get( i32Ty, 16 ), "stackTmp" );
            stackTmp->setAlignment( 16 );
            args.push_back( stackTmp );
        }

        unsigned int dimensions = call->getDimensionality();
        for( unsigned int i = 0; i < dimensions; ++i )
            args.push_back( call->getIndex( i ) );

        Function* runtimeFunction = bufferElementFromIdFuncs[dimensions - 1];
        CallInst* getPointerCall  = irb.CreateCall( runtimeFunction, args );
        Value*    rawPointer      = irb.CreatePtrToInt( getPointerCall, i64Ty );

        call->replaceAllUsesWith( rawPointer );
        call->eraseFromParent();
    }
}

template void optix::replaceGetBufferElementAddressFromId<GetBufferElementAddressFromId>( Function* F,
                                                                                          Function* bufferElementFromIdFuncs[],
                                                                                          bool generateAlloca,
                                                                                          std::vector<Value*>& toDelete );
template void optix::replaceGetBufferElementAddressFromId<RtxiGetBufferElementAddressFromId>( Function* F,
                                                                                              Function* bufferElementFromIdFuncs[],
                                                                                              bool generateAlloca,
                                                                                              std::vector<Value*>& toDelete );

//------------------------------------------------------------------------------
static void replaceGetPayloadAddress( Function* F, Function* runtimeFunction, std::vector<Value*>& toDelete )
{
    Module*      module  = F->getParent();
    LLVMContext& context = module->getContext();
    Type*        i64Ty   = Type::getInt64Ty( context );
    toDelete.push_back( F );

    auto calls = getCallsToFunction( F );
    for( const auto& I : calls )
    {
        GetPayloadAddressCall* call = dyn_cast<GetPayloadAddressCall>( I );
        if( !call )
            continue;

        corelib::CoreIRBuilder irb( call );

        Value*    args[]         = {call->getStatePtr()};
        CallInst* getPointerCall = irb.CreateCall( runtimeFunction, args );
        Value*    rawPointer     = irb.CreatePtrToInt( getPointerCall, i64Ty );

        call->replaceAllUsesWith( rawPointer );
        call->eraseFromParent();
    }
}

//------------------------------------------------------------------------------
bool optix::parseTraceName( const StringRef& fname )
{
    SmallVector<StringRef, 3> matches;
    if( !optixi_TraceRegex.match( fname, &matches ) )
        return false;
    return matches.size() == 3;
}

//------------------------------------------------------------------------------
unsigned int optix::replaceTracePlaceholders( Module* module )
{
    TIMEVIZ_FUNC;

    DataLayout   DL( module );
    LLVMContext& context = module->getContext();
    Type*        i8PtrTy = Type::getInt8PtrTy( context );
    Function* traceFunc  = getFunctionOrAssert( module, "_ZN4cort13Runtime_traceEPNS_14CanonicalStateEjffffffjfffjPc" );
    Function* traceGlobalPayload =
        getFunctionOrAssert( module, "_ZN4cort28Runtime_trace_global_payloadEPNS_14CanonicalStateEjffffffjfffj" );

    unsigned int maxPayloadSize = 0;
    auto         functions      = getFunctions( module );
    for( const auto& F : functions )
    {
        if( !F->isDeclaration() || !F->getName().startswith( "optixi_" ) )
            continue;
        if( parseTraceName( F->getName() ) )
        {
            // The value type is the same as the last argument
            FunctionType* fntype = F->getFunctionType();
            RT_ASSERT( fntype->getNumParams() == 16 );

            for( const auto& call : getCallsToFunction( F ) )
            {  // Replace optixi_trace by Runtime_trace, passing all arguments
                // (payload is cast to void*) plus the payload size.
                RT_ASSERT( call->getType()->isVoidTy() );
                RT_ASSERT( call->getNumArgOperands() == 16 );

                // Build the parameters
                Value* state   = call->getArgOperand( 0 );
                Value* node    = call->getArgOperand( 1 );
                Value* ox      = call->getArgOperand( 2 );
                Value* oy      = call->getArgOperand( 3 );
                Value* oz      = call->getArgOperand( 4 );
                Value* dx      = call->getArgOperand( 5 );
                Value* dy      = call->getArgOperand( 6 );
                Value* dz      = call->getArgOperand( 7 );
                Value* rayType = call->getArgOperand( 8 );
                Value* tmin    = call->getArgOperand( 9 );
                Value* tmax    = call->getArgOperand( 10 );
                Value* time    = call->getArgOperand( 11 );
                Value* hasTime = call->getArgOperand( 12 );
                // ignore mask and flags
                Value* payload = call->getArgOperand( 15 );

                corelib::CoreIRBuilder irb( call );

                // TODO: The maximum payload size is not really required anymore since
                //       we store the pointer directly.
                RT_ASSERT( payload->getType()->isPointerTy() );
                Type*        valueType = payload->getType()->getPointerElementType();
                unsigned int eltSize   = DL.getTypeStoreSize( valueType );
                maxPayloadSize         = std::max( eltSize, maxPayloadSize );

                // Cast the payload back to a generic i8* without address space.
                if( payload->getType()->getPointerAddressSpace() != 0 )
                    payload = irb.CreateAddrSpaceCast( payload, i8PtrTy );
                else
                    payload = irb.CreateBitCast( payload, i8PtrTy );

                Value* args[] = {state, node, ox, oy, oz, dx, dy, dz, rayType, tmin, tmax, time, hasTime, payload};
                irb.CreateCall( traceFunc, args );

                call->eraseFromParent();
            }
        }

        if( TraceGlobalPayloadCall::isIntrinsic( F ) )
        {
            auto calls = getCallsToFunction( F );
            for( const auto& U : calls )
            {
                TraceGlobalPayloadCall* call = dyn_cast<TraceGlobalPayloadCall>( U );
                if( !call )
                    continue;

                Value* state   = call->getStatePtr();
                Value* node    = call->getNode();
                Value* ox      = call->getOx();
                Value* oy      = call->getOy();
                Value* oz      = call->getOz();
                Value* dx      = call->getDx();
                Value* dy      = call->getDy();
                Value* dz      = call->getDz();
                Value* rayType = call->getRayType();
                Value* tmin    = call->getTMin();
                Value* tmax    = call->getTMax();
                Value* time    = call->getTime();
                Value* hasTime = call->getHasTime();
                // ignore mask and flags
                Value* args[] = {state, node, ox, oy, oz, dx, dy, dz, rayType, tmin, tmax, time, hasTime};

                corelib::CoreIRBuilder irb( call );
                irb.CreateCall( traceGlobalPayload, args );
                call->eraseFromParent();
            }
        }
    }
    return maxPayloadSize;
}

//------------------------------------------------------------------------------
void optix::generateFunctionFromPrototype( Function* dst, Function* src, int numCopyParameters, const std::string& newName )
{
    // Clone the source function and splice it into the destination function
    ValueToValueMapTy VMap;
    Function*         clone = CloneFunction( src, VMap );
    dst->copyAttributesFrom( clone );
    dst->getBasicBlockList().splice( dst->begin(), clone->getBasicBlockList() );

    // First N parameters are directly copied
    Function::arg_iterator DI = dst->arg_begin();
    Function::arg_iterator CI = clone->arg_begin();
    for( int i = 0; i < numCopyParameters; ++i )
    {
        RT_ASSERT( DI != dst->arg_end() );
        RT_ASSERT( CI != clone->arg_end() );
        CI->replaceAllUsesWith( DI );
        DI->takeName( CI );
        ++CI, ++DI;
    }

    // The only remaining parameter is a vararg placeholder. We map
    // remaining arguments from the destination function to the
    // call instruction. Note there must be only ONE call.
    Value* vararg = CI++;
    RT_ASSERT( CI == clone->arg_end() );
    RT_ASSERT( vararg->hasNUses( 1 ) );

    CallInst* ci = dyn_cast<CallInst>( *vararg->user_begin() );
    RT_ASSERT_MSG( ci != nullptr, "function generation requires all uses to be used directly in a call" );

    corelib::CoreIRBuilder irb( ci );

    // Build a new call
    SmallVector<Value*, 8> newArgs;
    unsigned int nargs = ci->getNumArgOperands();
    for( unsigned int i = 0; i < nargs; ++i )
    {
        Value* arg = ci->getArgOperand( i );
        if( arg == vararg )
            break;  // Stop at the placeholder argument
        newArgs.push_back( ci->getArgOperand( i ) );
    }

    // Add remaining function parameters
    for( Function::arg_iterator AI = DI, AIE = dst->arg_end(); AI != AIE; ++AI )
        newArgs.push_back( AI );

    // Replace the call instruction with the new one
    SmallVector<Type*, 8> newTypes;
    for( Value* newArg : newArgs )
        newTypes.push_back( newArg->getType() );
    Type*         newReturnType = dst->getReturnType();
    FunctionType* newType       = FunctionType::get( newReturnType, newTypes, false );

    Value* callee;
    if( Function* oldFunc = ci->getCalledFunction() )
    {
        // This is a named function. Create a new name by appending "extName" to the function
        Constant* c       = oldFunc->getParent()->getOrInsertFunction( newName, newType );
        Function* newFunc = dyn_cast<Function>( c );
        RT_ASSERT_MSG( newFunc != nullptr, "Function already found but it is the wrong type" );
        callee = newFunc;
    }
    else
    {
        // Indirect function call
        callee = irb.CreateBitCast( ci->getCalledValue(), newType->getPointerTo() );
    }
    CallInst* newCall = irb.CreateCall( callee, newArgs );
    // Don't take the name if the new return type is void
    if( !newCall->getType()->isVoidTy() )
        newCall->takeName( ci );

    // Follow the uses of the call instruction. Only a limited number of possibilties are handled:
    // 1. A phi instruction where the other edges are an undef. Replace with an undef of the new type
    // 2. A return instruction. Replace with a new return.
    // This code does not handle more than one use at present either.
    // Because of these constraints, this function should only used for internal functions.
    Instruction* newPrev = newCall;
    if( newReturnType->isVoidTy() )
        newPrev          = nullptr;
    Instruction* oldPrev = ci;
    SmallVector<Instruction*, 3> toDelete;
    for( ;; )
    {
        // Follow the single use
        RT_ASSERT( oldPrev->hasNUses( 1 ) );
        Instruction* oldNext = dyn_cast<Instruction>( *oldPrev->user_begin() );
        RT_ASSERT( oldNext != nullptr );

        if( PHINode* oldPhi = dyn_cast<PHINode>( oldNext ) )
        {
            // Replicate the PHI node assuming that all incoming edges are
            // either the return value or an undef
            if( newPrev )
            {
                unsigned int numEdges = oldPhi->getNumIncomingValues();
                PHINode*     newPhi   = corelib::CoreIRBuilder{oldPhi}.CreatePHI( newReturnType, numEdges );
                for( unsigned int i = 0; i < numEdges; ++i )
                {
                    Value*      oldValue = oldPhi->getIncomingValue( i );
                    BasicBlock* block    = oldPhi->getIncomingBlock( i );
                    if( oldValue == oldPrev )
                    {
                        // Add an edge with the value
                        newPhi->addIncoming( newPrev, block );
                    }
                    else if( isa<UndefValue>( oldValue ) )
                    {
                        UndefValue* undef = UndefValue::get( newReturnType );
                        newPhi->addIncoming( undef, block );
                    }
                    else
                    {
                        RT_ASSERT_FAIL_MSG( LLVMErrorInfo( oldPhi ) + " Complex PHINode not handled" );
                    }
                }

                newPhi->takeName( oldPhi );
                newPrev = newPhi;
            }
            toDelete.push_back( oldPrev );
            oldPrev = oldNext;
        }
        else if( ReturnInst* oldRet = dyn_cast<ReturnInst>( oldNext ) )
        {
            ReturnInst* newRet = corelib::CoreIRBuilder{oldRet}.CreateRet( newPrev );
            toDelete.push_back( oldPrev );
            toDelete.push_back( oldNext );
            oldPrev = nullptr;
            if( newPrev )
                newPrev = newRet;
            break;  // Stop at returns
        }
        else
        {
            RT_ASSERT_FAIL_MSG( LLVMErrorInfo( oldNext ) + " Invalid instruction type found" );
        }
    }

    // Objects are inserted into the list in def to use order, so delete them in reverse.
    for( auto I = toDelete.rbegin(), IE = toDelete.rend(); I != IE; ++I )
        ( *I )->eraseFromParent();

    // Now we are done with the clone because we stripped the body out
    delete clone;
}

//------------------------------------------------------------------------------
void optix::generateCallableProgramDispatch( Module* module )
{
    // There are three forms of callable program ilwocations in the
    // input. These get colwerted to calls to an appropriate
    // Program_ilwoke.

    // ret optixi_callBound.r0.t0.name.sig0( state, id, args... )
    // ret optixi_callBindless.r0.t0.name.sig0( state, id, args... )
    // ret optixi_callBindless.sig0( state, id, args... )

    Function* boundPrototype =
        module->getFunction( "_ZN4cort40Runtime_ilwokeBoundCallableProgram_protoEPNS_14CanonicalStateEii" );
    Function* bindlessPrototype =
        module->getFunction( "_ZN4cort43Runtime_ilwokeBindlessCallableProgram_protoEPNS_14CanonicalStateEii" );
    RT_ASSERT( boundPrototype != nullptr );
    RT_ASSERT( bindlessPrototype != nullptr );

    for( Module::iterator F = module->begin(), FE = module->end(); F != FE; ++F )
    {
        if( !F->isDeclaration() || !F->getName().startswith( "optixi_callB" ) )
            continue;
        bool      isBound;
        StringRef varRefUniqueName;
        unsigned  sig;
        bool      result = parseCallableProgramName( F->getName(), isBound, varRefUniqueName, sig );
        RT_ASSERT_MSG( result, LLVMErrorInfo( &( *F ) ) + " Invalid callable program placeholder: " + F->getName().str() );

        Function*   prototypeFunction = isBound ? boundPrototype : bindlessPrototype;
        std::string newName           = "optixi_callIndirect_sig" + std::to_string( sig );
        generateFunctionFromPrototype( &( *F ), prototypeFunction, 2, newName );
    }
}

//------------------------------------------------------------------------------
llvm::Function* optix::findOrCreateRuntimeFunction( llvm::Module*                   module,
                                                    const std::string&              name,
                                                    llvm::Type*                     returnType,
                                                    const std::vector<llvm::Type*>& argTypes )
{
    FunctionType* fnTy = FunctionType::get( returnType, argTypes, false );
    Function*     fn   = module->getFunction( name );
    if( fn )
    {
        RT_ASSERT_MSG( fn->getFunctionType() == fnTy, "Unexpected type for runtime function" );
    }
    else
    {
        fn = Function::Create( fnTy, GlobalValue::ExternalLinkage, name, module );
    }
    return fn;
}

//------------------------------------------------------------------------------
void optix::generateTextureDeclarations( Module* module, const std::string& basename, int count )
{
    if( count == 0 )
        return;

    LLVMContext& context   = module->getContext();
    MDString*    str       = MDString::get( context, "texture" );
    Value*       av        = ConstantInt::get( Type::getInt32Ty( context ), 1 );
    Type*        Int8PtrTy = Type::getInt8PtrTy( context );
    Type*        i64Ty     = Type::getInt64Ty( context );
    Type*        i64Ptr1Ty = i64Ty->getPointerTo( 1 );
    Type*        mdTy      = Type::getMetadataTy( context );

    std::vector<GlobalVariable*> texs( count );
    std::vector<Value*>          mds( count );
    std::vector<Constant*>       inits( count );
    for( int unit = 0; unit < count; ++unit )
    {
        std::string  texname      = "tex" + std::to_string( unit );
        NamedMDNode* lwvmannotate = module->getOrInsertNamedMetadata( "lwvm.annotations" );

        GlobalVariable* texrefVar = new GlobalVariable( *module,
                                                        Type::getInt64Ty( context ),   // type of the variable
                                                        false,                         // is this variable constant
                                                        GlobalValue::InternalLinkage,  // symbol linkage
                                                        ConstantInt::get( Type::getInt64Ty( context ), 0 ),  // Static initializer
                                                        texname,                                             // Name
                                                        nullptr,  // InsertBefore -- we want it to be appended to module's global list
                                                        GlobalVariable::NotThreadLocal,  // Thread local
                                                        1 );                             // The variable's address space
        texrefVar->setAlignment( 8 );


        std::vector<Metadata*> mds = {CastValueToMd( texrefVar ), str, CastValueToMd( av )};
        MDNode* annot    = MDNode::get( context, mds );
        lwvmannotate->addOperand( annot );
        mds[unit]      = annot;
        Constant* init = ConstantExpr::getAddrSpaceCast( texrefVar, Int8PtrTy );
        inits[unit]    = ConstantExpr::getPointerCast( init, Int8PtrTy );
        texs[unit]     = texrefVar;
    }
    // Emit llvm.used to retain texture
    ArrayType*      ATy = ArrayType::get( Int8PtrTy, count );
    GlobalVariable* GV  = new GlobalVariable( *module, ATy, false, GlobalValue::AppendingLinkage,
                                             ConstantArray::get( ATy, inits ), "llvm.used" );

    GV->setSection( "llvm.metadata" );

    // Get the handle function declaration
    Function* getHandleFunction = module->getFunction( "optix.cort.texref2handle" );
    if( !getHandleFunction )
        return;  // Function not needed
    RT_ASSERT( getHandleFunction->isDeclaration() );
    Value* index = getHandleFunction->arg_begin();

    Type*         argTypes2[] = {mdTy, i64Ptr1Ty};
    FunctionType* htype       = FunctionType::get( i64Ty, argTypes2, false );
    Function* handleFunc = dyn_cast<Function>( module->getOrInsertFunction( "llvm.lwvm.texsurf.handle.p1i64", htype ) );

    // Create the switch statement
    // TODO: experiment with making this a table
    BasicBlock* entry    = BasicBlock::Create( context, "", getHandleFunction );
    BasicBlock* badblock = BasicBlock::Create( context, "bad", getHandleFunction );
    corelib::CoreIRBuilder{badblock}.CreateUnreachable();
    SwitchInst* switchInst = corelib::CoreIRBuilder{entry}.CreateSwitch( index, badblock, count );
    for( int unit = 0; unit < count; ++unit )
    {
        BasicBlock* block = BasicBlock::Create( context, "tex_" + Twine( unit ), getHandleFunction, badblock );
        corelib::CoreIRBuilder irb{block};
        Value*                 args[] = {mds[unit], texs[unit]};
        Value*                 call   = irb.CreateCall( handleFunc, args );
        irb.CreateRet( call );

        ConstantInt* funcIndex = makeInt32( unit, context );
        switchInst->addCase( funcIndex, block );
    }
}

//------------------------------------------------------------------------------
// Assigns the name "v" to every unnamed, non-void instruction. The names will
// be uniqued by adding an integer suffix
void optix::dbgNameUnnamedVals( Module* module )
{
    TIMEVIZ_FUNC;
    Type* voidTy    = Type::getVoidTy( module->getContext() );
    auto  functions = getFunctions( module );
    for( const auto& F : functions )
    {
        Function* func = dyn_cast<Function>( F );
        if( !func || func->isDeclaration() )
            continue;
        for( inst_iterator I = inst_begin( func ), IE = inst_end( func ); I != IE; ++I )
        {
            if( !I->hasName() && I->getType() != voidTy )
                I->setName( "v" );
        }
    }
}

//------------------------------------------------------------------------------
static void dbgPrintValTree( Instruction* inst, SmallPtrSet<Instruction*, 16>& visited )
{
    SmallVector<Instruction*, 16> workList;
    visited.insert( inst );
    workList.push_back( inst );
    while( !workList.empty() )
    {
        Instruction* lwr = workList.pop_back_val();
        dbgPrint( lwr, valueToString( lwr ) );
        //Function* func = lwr->getParent()->getParent();
        //dbgPrintVal( lwr, (instructionString(lwr) + " (" + func->getName() + ")").str() ); // include func name
        for( Instruction::op_iterator O = lwr->op_begin(), OE = lwr->op_end(); O != OE; ++O )
        {
            Instruction* opInst = dyn_cast<Instruction>( *O );
            if( opInst && std::get<1>( visited.insert( opInst ) ) )
                workList.push_back( opInst );
        }
    }
};


//-----------------------------------------------------------------------------
// Lwrrently only the prefix (up to the first '.' or end of string) of each value
// name is compared
void optix::dbgInsertPrintsVals( Module* module )
{
    TIMEVIZ_FUNC;

    if( k_dbgPrintVals.get().empty() )
        return;

    std::vector<std::string> funcs = corelib::tokenize( k_dbgPrintVals.get(), "[" );
    for( const std::string& i : funcs )
    {
        std::vector<std::string> vals     = corelib::tokenize( i, "], " );
        std::string              funcName = vals[0];
        vals.erase( vals.begin(), vals.begin() + 1 );

        std::vector<int> printTree( vals.size(), 0 );
        for( size_t v = 0; v < vals.size(); ++v )
        {
            if( vals[v][0] == '*' )
            {
                printTree[v] = 1;
                vals[v]      = vals[v].substr( 1 );
            }
        }

        auto functions = getFunctions( module );
        for( const auto& F : functions )
        {
            Function* func = dyn_cast<Function>( F );
            if( !func || func->isDeclaration() || func->getName().find( funcName ) == StringRef::npos )
                continue;

            SmallPtrSet<Instruction*, 16> visited;  // Avoid printing the same value more than once
            for( inst_iterator I = inst_begin( func ), IE = inst_end( func ); I != IE; ++I )
            {
                if( !I->hasName() )
                    continue;

                // Not the most efficient implementation
                StringRef name = I->getName();
                for( size_t v = 0; v < vals.size(); ++v )
                {
                    if( name.startswith( vals[v] ) && ( name.size() <= vals[v].size() || name[vals[v].size()] == '.' ) )
                    {
                        if( printTree[v] )
                            dbgPrintValTree( &*I, visited );
                        else if( std::get<1>( visited.insert( &*I ) ) )
                            dbgPrint( &*I );
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
bool optix::includeForUniquifyingConstMemoryInitializers( const GlobalVariable* G )
{
    return ( G->hasInitializer() && G->getType()->getPointerAddressSpace() == ADDRESS_SPACE_CONST && !G->use_empty()
             && G->getLinkage() == GlobalValue::InternalLinkage );
}

//------------------------------------------------------------------------------
void optix::uniquifyConstMemoryInitializers( Module* module )
{
    // Match by type and initializer value
    std::map<Constant*, GlobalVariable*> static_initializers;
    std::vector<GlobalVariable*> toDelete;
    for( Module::global_iterator G = module->global_begin(), GE = module->global_end(); G != GE; ++G )
    {
        // Look for globals with initializers, declared in constant memory, and has at least one use
        if( includeForUniquifyingConstMemoryInitializers( &( *G ) ) )
        {
            auto inserted_pair = static_initializers.insert( std::make_pair( G->getInitializer(), &( *G ) ) );
            if( inserted_pair.second == false )
            {
                // it wasn't inserted, so replace this value with the old one.
                G->replaceAllUsesWith( inserted_pair.first->second );
                toDelete.push_back( &( *G ) );
            }
        }
    }
    for( GlobalVariable* I : toDelete )
        I->eraseFromParent();
}
