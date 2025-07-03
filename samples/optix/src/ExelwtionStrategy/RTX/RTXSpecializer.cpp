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

#include <ExelwtionStrategy/RTX/RTXSpecializer.h>

#include <Context/ProgramManager.h>
#include <ExelwtionStrategy/Compile.h>
#include <ExelwtionStrategy/RTX/LDGOptimization.h>
#include <ExelwtionStrategy/RTX/RTXExceptionInstrumenter.h>
#include <ExelwtionStrategy/RTX/RTXIntrinsics.h>
#include <FrontEnd/Canonical/FrontEndHelpers.h>
#include <FrontEnd/Canonical/IntrinsicsManager.h>
#include <FrontEnd/Canonical/LineInfo.h>
#include <Util/ContainerAlgorithm.h>
#include <corelib/compiler/CoreIRBuilder.h>
#include <corelib/compiler/LLVMUtil.h>
#include <corelib/misc/String.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/system/Knobs.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include <cctype>

using namespace optix;
using namespace prodlib;
using namespace corelib;
using namespace llvm;

namespace {
// clang-format off
  Knob<std::string> k_saveLLVM( RT_DSTRING("rtx.saveLLVM"), "", RT_DSTRING( "Save LLVM stages during compilation" ) );
// clang-format on

using Values = Value* [];
}

// Support functions.
// -----------------------------------------------------------------------------
static void eraseFromParent( const std::vector<llvm::Value*>& toRemove );
static Instruction* createLoadOrLDG( Value* typedPtr, bool createLDG, const std::string& varName, unsigned int align, Instruction* insertBefore );

// -----------------------------------------------------------------------------
std::string RTXSpecializer::computeDumpName( SemanticType stype, const std::string& functionName )
{
    // Add semantic type to dumped file names for consistency with rtcore and to disambiguate different
    // uses of the null program.
    std::string semanticType = optix::semanticTypeToString( stype );
    algorithm::transform( semanticType, semanticType.begin(), ::tolower );
    std::string dumpFunctionName = "_" + semanticType + "__" + functionName;
    return dumpFunctionName;
}


// -----------------------------------------------------------------------------
RTXVariableSpecializer::RTXVariableSpecializer( const Specializations& specializations,
                                                llvm::Function*        entryFunction,
                                                SemanticType           stype,
                                                SemanticType           inheritedStype,
                                                bool                   deviceSupportsLDG,
                                                bool                   useConstMemory,
                                                const ProgramManager*  programManager,
                                                int                    launchCounterForDebugging )
    : m_specializations( specializations )
    , m_stype( stype )
    , m_inheritedStype( inheritedStype )
    , m_deviceSupportsLDG( deviceSupportsLDG )
    , m_useConstMemory( useConstMemory )
    , m_programManager( programManager )
    , m_launchCounterForDebugging( launchCounterForDebugging )
{
    initializeRuntimeFunctions( entryFunction );
}

void RTXVariableSpecializer::runOnModule( Module* module, const std::string& dumpName )
{
    int dumpId = 0;

    dump( module, dumpName, dumpId++, "init" );

    // Handle textures separately, because we have to look for all the functions that match
    // Texture_* rather than looking for the function by name directly (we can't construct
    // the function name from the available data, we have to pull it out of the function
    // name itself).
    applyTextureSpecializations( module );

    for( const auto& iter : m_specializations.m_varspec )
    {
        const VariableReferenceID     refid = iter.first;
        const VariableSpecialization& vs    = iter.second;

        const VariableReference* varref = m_programManager->getVariableReferenceById( refid );

        if( varref->getType().isBuffer() )
        {
            specializeBuffer( module, refid, vs );
            specializeRtxiGetBufferId( module, refid, vs );
            specializeRtxiGetBufferSize( module, refid, vs );
        }
        else
        {
            specializeVariable( module, refid, vs );
        }
    }

    // Specialize bindless buffers
    // This specialization is safe only if no bindless buffers live in texture memory.
    // This cannot happen right now.
    // If in the future we add this functionality this specialization has to be enabled selectively.
    specializeBindlessBuffers( module );

    dump( module, dumpName, dumpId++, "specialized" );
}

//------------------------------------------------------------------------------
// Initialize function pointers - creating them if necessary
void RTXVariableSpecializer::initializeRuntimeFunctions( Function* entryFunction )
{
    Module*      module      = entryFunction->getParent();
    LLVMContext& llvmContext = module->getContext();
    m_statePtrTy             = entryFunction->getFunctionType()->getParamType( 0 );
    m_i32Ty                  = Type::getInt32Ty( llvmContext );
    m_i8PtrTy                = Type::getInt8PtrTy( llvmContext );
    m_constMemi8PtrTy        = Type::getInt8PtrTy( llvmContext, ADDRESS_SPACE_CONST );

    Type* i16Ty         = Type::getInt16Ty( llvmContext );
    Type* i64Ty         = Type::getInt64Ty( llvmContext );
    m_genericLookupFunc = findOrCreateRuntimeFunction(
        module, "_ZN4cort29Runtime_lookupVariableAddressEPNS_14CanonicalStateEtPcN5optix12SemanticTypeES4_", m_i8PtrTy,
        {m_statePtrTy, i16Ty, m_i8PtrTy, m_i32Ty, m_i32Ty} );

    m_bufferElementFromIdFuncs[0] = findOrCreateRuntimeFunction(
        module, "_ZN10Megakernel39Buffer_getElementAddress1dFromId_linearEPN4cort14CanonicalStateEjjy", m_i8PtrTy,
        {m_statePtrTy, m_i32Ty, m_i32Ty, i64Ty} );
    m_bufferElementFromIdFuncs[1] = findOrCreateRuntimeFunction(
        module, "_ZN10Megakernel39Buffer_getElementAddress2dFromId_linearEPN4cort14CanonicalStateEjjyy", m_i8PtrTy,
        {m_statePtrTy, m_i32Ty, m_i32Ty, i64Ty, i64Ty} );
    m_bufferElementFromIdFuncs[2] = findOrCreateRuntimeFunction(
        module, "_ZN10Megakernel39Buffer_getElementAddress3dFromId_linearEPN4cort14CanonicalStateEjjyyy", m_i8PtrTy,
        {m_statePtrTy, m_i32Ty, m_i32Ty, i64Ty, i64Ty, i64Ty} );

    // Find the first getsize intrinsic to determine the current size3 type
    for( Function* F : getFunctions( module ) )
    {
        if( RtxiGetBufferSizeFromId::isIntrinsic( F ) || RtxiGetBufferSize::isIntrinsic( F ) )
        {
            Type* size3Ty = F->getReturnType();
            m_bufferSizeFromIdFunc =
                findOrCreateRuntimeFunction( module, "_ZN4cort20Buffer_getSizeFromIdEPNS_14CanonicalStateEj", size3Ty,
                                             {m_statePtrTy, m_i32Ty} );
            break;
        }
    }
}

// -----------------------------------------------------------------------------
static bool isConstantZero( Value* value )
{
    ConstantInt* c = dyn_cast<ConstantInt>( value );
    if( !c )
        return false;

    return c->getZExtValue() == 0;
}

// -----------------------------------------------------------------------------
Value* RTXVariableSpecializer::loadSpecializedVariable( Module*                       module,
                                                        const VariableSpecialization& vs,
                                                        Type*                         returnType,
                                                        Value*                        stateptr,
                                                        Instruction*                  insertBefore,
                                                        const std::string&            varname,
                                                        unsigned short                token,
                                                        Value*                        defaultValue,
                                                        Value*                        offset )
{
    // if (specialization says it is constant)
    //   return vs.constantValue;
    // else
    //   get a pointer to the data based on specialization
    //   load data
    //   return data

    if( vs.lookupKind == VariableSpecialization::SingleId )
    {
        // Make the specialization hold the Constant and then you can
        // specialize for any type.  Otherwise you would have to store each type in a
        // different member variable.
        Value* idValue = ConstantInt::get( returnType, vs.singleId );
        return idValue;
    }

    Value*       ptr          = nullptr;
    bool         constBasePtr = false;
    LLVMContext& context      = module->getContext();
    DataLayout   DL( module );

    corelib::CoreIRBuilder irb{insertBefore};

    if( vs.lookupKind == VariableSpecialization::SingleScope )
    {
        if( vs.singleBinding.isDefaultValue() )
        {
            ptr = defaultValue;
        }
        else
        {
            // Get the getter function based on the scope and whether the object record is in
            // constant memory.
            std::string scopeclass = getNameForClass( vs.singleBinding.scopeClass() );
            std::string gettername = std::string( "Megakernel_getObjectBase_" ) + scopeclass;
            Type*       ptrTy      = m_i8PtrTy;
            if( m_useConstMemory )
            {
                gettername += "_FromConst";
                constBasePtr = true;
                ptrTy        = m_constMemi8PtrTy;
            }
            std::vector<Type*> argTypes = {m_statePtrTy, m_i32Ty};
            if( scopeclass == "Program" )
                argTypes.push_back( m_i32Ty );
            Function* getter = findOrCreateRuntimeFunction( module, gettername, ptrTy, argTypes );

            // Create the arguments to the getter.
            // For Program lookup, we need to pass the semantic type.
            Type*  i32Ty               = Type::getInt32Ty( context );
            Value* singleBindingOffset = ConstantInt::get( i32Ty, vs.singleBinding.offset() );
            SmallVector<Value*, 3> args;
            args.push_back( stateptr );
            args.push_back( singleBindingOffset );
            if( scopeclass == "Program" )
                args.push_back( ConstantInt::get( i32Ty, m_stype ) );

            // Call the getter (this returns a pointer)
            ptr = irb.CreateCall( getter, args, varname + ".ptr" );
        }
    }
    else if( vs.lookupKind == VariableSpecialization::GenericLookup )
    {
        // Use generic lookup based on the semantic type.
        if( defaultValue == nullptr )
            defaultValue = Constant::getNullValue( Type::getInt8PtrTy( context ) );
        else
            defaultValue = irb.CreateBitCast( defaultValue, Type::getInt8PtrTy( context ), varname + ".i8ptr" );

        RT_ASSERT( token == vs.dynamicVariableToken );
        Value* tokelw         = ConstantInt::get( Type::getInt16Ty( context ), vs.dynamicVariableToken );
        Value* stype          = ConstantInt::get( Type::getInt32Ty( context ), m_stype );
        Value* inheritedStype = ConstantInt::get( Type::getInt32Ty( context ), m_inheritedStype );
        Value* args[]         = {stateptr, tokelw, defaultValue, stype, inheritedStype};
        ptr                   = irb.CreateCall( m_genericLookupFunc, args, varname + ".ptr" );
    }
    else if( vs.lookupKind == VariableSpecialization::Unused )
    {
        return UndefValue::get( returnType );
    }
    else
    {
        RT_ASSERT_FAIL_MSG( "Unsupported variable specialization " + varname );
    }

    if( offset != nullptr && !isConstantZero( offset ) )
    {
        Type* type = ptr->getType()->getPointerElementType();
        if( type->isVectorTy() || type->isArrayTy() )
        {
            Type*  subType      = type->isVectorTy() ? type->getVectorElementType() : type->getArrayElementType();
            size_t subTypeSize  = DL.getTypeStoreSize( subType );
            size_t destTypeSize = DL.getTypeStoreSize( returnType );
            Type*  i64Ty        = Type::getInt64Ty( context );
            Type*  i8Ty         = Type::getInt8Ty( context );
            if( destTypeSize < subTypeSize )
            {
                // User structs and some array types can be represented as LLVM
                // arrays of an integral type (commonly i32). If the destination
                // of the load is narrower than the base type of the array, then
                // the offset is in units of bytes, so we need to cast the
                // pointer to a byte array to make the units match. Otherwise,
                // the offset for the load will be a computed as
                // offset*sizeof(struct), rather than the correct byte offset.

                size_t      arraySizeInBytes = DL.getTypeStoreSize( type );
                llvm::Type* arrayType        = ArrayType::get( i8Ty, arraySizeInBytes );
                ptr = irb.CreateBitCast( ptr, PointerType::get( arrayType, ptr->getType()->getPointerAddressSpace() ) );
            }
            else
            {
                // If the offset is constant the division is optimized away.
                offset = irb.CreateUDiv( offset, ConstantInt::get( i64Ty, subTypeSize ), "newoffset" );
            }
            Value* args[2] = {ConstantInt::get( i64Ty, 0 ), offset};
            ptr            = irb.CreateInBoundsGEP( ptr, args );
        }
        else
        {
            ptr = irb.CreateInBoundsGEP( ptr, offset );
        }
    }

    // Cast the pointer to the appropriate type
    Type*  VT       = returnType->getPointerTo( constBasePtr ? ADDRESS_SPACE_CONST : ADDRESS_SPACE_GENERIC );
    Value* typedptr = nullptr;
    if( VT->getPointerAddressSpace() != ptr->getType()->getPointerAddressSpace() )
    {
        typedptr = irb.CreateAddrSpaceCast( ptr, VT, varname + ".ptr" );
    }
    else
    {
        typedptr = irb.CreateBitCast( ptr, VT, varname + ".ptr" );
    }
    // Load the value out of the object record
    const unsigned int size  = DL.getTypeStoreSize( returnType );
    const unsigned int align = MinAlign( MinAlign( size, vs.singleBinding.offset() ), 16 );
    // Don't use LDG for default values.  This allows llvm to
    // optimize away the load and insert the default value directly.
    const bool createLDG = vs.preferLDG() && m_deviceSupportsLDG && !constBasePtr && !vs.singleBinding.isDefaultValue();
    return createLoadOrLDG( typedptr, createLDG, varname, align, insertBefore );
}

static inline bool isVariableValueFunc( const Function* F, const VariableReference* varref )
{
    static const char variableValue[] = "optixi_getVariableValue";
    return isIntrinsic( F->getName(), variableValue, varref->getUniversallyUniqueName() );
}

// Load the variable using specializeVariable()
// Replace all uses of function with specialized value
void RTXVariableSpecializer::specializeVariable( Module* module, VariableReferenceID refID, const VariableSpecialization& vs )
{
    ValueVector              toRemove;
    const VariableReference* varref  = m_programManager->getVariableReferenceById( refID );
    const std::string&       varname = varref->getInputName();
    for( Function* F : getFunctions( module ) )
    {
        if( isVariableValueFunc( F, varref ) )
        {
            Type* returnType = F->getFunctionType()->getReturnType();
            for( CallInst* CI : getCallsToFunction( F ) )
            {
                Value* V = loadSpecializedVariable( module, vs, returnType, CI->getArgOperand( 0 ), CI, varname,
                                                    varref->getVariableToken(), CI->getArgOperand( 2 ), CI->getArgOperand( 1 ) );
                CI->replaceAllUsesWith( V );
                V->takeName( CI );
                toRemove.push_back( CI );
            }
        }
    }
    eraseFromParent( toRemove );
}

void RTXVariableSpecializer::specializeRtxiGetBufferId( llvm::Module* module, VariableReferenceID refID, const VariableSpecialization& vs )
{
    ValueVector              toRemove;
    const VariableReference* varref  = m_programManager->getVariableReferenceById( refID );
    const std::string&       varname = varref->getInputName();
    RT_ASSERT( varref->getType().isBuffer() );
    std::string placeholdername_id = RtxiGetBufferId::getFunctionName( varref->getUniversallyUniqueName() );
    Function*   pId                = module->getFunction( placeholdername_id );
    if( !pId )
        return;

    LLVMContext& context = module->getContext();
    Type*        i32Ty   = Type::getInt32Ty( context );

    for( CallInst* CI : getCallsToFunction( pId ) )
    {
        RT_ASSERT( CI->getNumArgOperands() == 1 );
        Value* state = CI->getArgOperand( 0 );
        Value* id = loadSpecializedVariable( module, vs, i32Ty, state, CI, varname, varref->getVariableToken(), nullptr, nullptr );
        CI->replaceAllUsesWith( id );
        id->takeName( CI );
        toRemove.push_back( CI );
    }
    eraseFromParent( toRemove );
}

template <class BufferAccess>
Value* RTXVariableSpecializer::createGetElementAddressCall( VariableReferenceID           refID,
                                                            Type*                         resultType,
                                                            BufferAccess*                 origCall,
                                                            const VariableSpecialization& vs )
{
    corelib::CoreIRBuilder builder{origCall};

    const VariableReference* varref = m_programManager->getVariableReferenceById( refID );
    RT_ASSERT( varref->getType().isBuffer() );
    unsigned int dim = varref->getType().bufferDimensionality();
    RT_ASSERT( 1 <= dim && dim <= 3 );
    RT_ASSERT( origCall->getDimensionality() == dim );

    Value* state       = origCall->getStatePtr();
    Value* bufferId    = origCall->getBufferId();
    Value* elementSize = origCall->getElementSize();
    Value* offset      = origCall->getOffset();
    Value* x           = origCall->getX();

    std::vector<Value*> args = {state, bufferId, elementSize, x};
    if( dim > 1 )
        args.push_back( origCall->getY() );
    if( dim > 2 )
        args.push_back( origCall->getZ() );

    const std::string& varname = varref->getInputName();

    Value* basePointer = builder.CreateCall( m_bufferElementFromIdFuncs[dim - 1], args, varname + ".ptr" );
    Value* i8ptr       = builder.CreateGEP( basePointer, offset );
    if( resultType->isPointerTy() )
        return builder.CreateBitCast( i8ptr, resultType, varname + ".typedPtr" );
    else
    {
        RT_ASSERT( resultType->isIntegerTy() );
        return builder.CreatePtrToInt( i8ptr, resultType, varname + ".ptr" );
    }
}

void RTXVariableSpecializer::specializeRtxiGetBufferSize( Module* module, VariableReferenceID refID, const VariableSpecialization& vs )
{
    ValueVector              toRemove;
    const VariableReference* varref = m_programManager->getVariableReferenceById( refID );
    RT_ASSERT( varref->getType().isBuffer() );
    std::string placeholdername_size = RtxiGetBufferSize::getFunctionName( varref->getUniversallyUniqueName() );
    Function*   pSize                = module->getFunction( placeholdername_size );
    if( !pSize )
        return;


    // Our intrinsic to be replaced is *not* the FromId variant, but it has exactly the same semantic.
    RT_ASSERT( m_bufferSizeFromIdFunc != nullptr );
    replaceGetBufferSizeFromId( pSize, m_bufferSizeFromIdFunc, toRemove );

    eraseFromParent( toRemove );
}

void RTXVariableSpecializer::specializeBuffer( Module* module, VariableReferenceID refID, const VariableSpecialization& vs )
{
    switch( vs.accessKind )
    {
        case VariableSpecialization::PitchedLinear:
        case VariableSpecialization::PitchedLinearPreferLDG:
        {
            specializeBuffer_pitchedLinear( module, refID, vs );
        }
        break;
        case VariableSpecialization::TexHeap:
        case VariableSpecialization::TexHeapSingleOffset:
        {
            RT_ASSERT_FAIL_MSG( "Error: texheap not supported in megakernel" );
        }
        break;
        default:
        {
            // Leave for replacePlaceholderAccessors
        }
        break;
    }
}

void RTXVariableSpecializer::specializeBuffer_pitchedLinear( Module* module, VariableReferenceID refID, const VariableSpecialization& vs )
{
    ValueVector              toRemove;
    const VariableReference* varref         = m_programManager->getVariableReferenceById( refID );
    const std::string&       varDesc        = varref->getUniversallyUniqueName();
    const unsigned int       dimensionality = varref->getType().bufferDimensionality();
    LLVMContext&             context        = module->getContext();
    DataLayout               DL( module );

    for( Function* F : getFunctions( module ) )
    {
        if( RtxiGetBufferElement::isIntrinsic( F, varDesc ) )
        {
            for( CallInst* CI : getCallsToFunction( F ) )
            {
                RT_ASSERT( isa<RtxiGetBufferElement>( CI ) );
                RtxiGetBufferElement* call = cast<RtxiGetBufferElement>( CI );
                RT_ASSERT( call->getDimensionality() == dimensionality );

                Type* valueTy = F->getReturnType();
                Value* typedPtr = createGetElementAddressCall<RtxiGetBufferElement>( refID, valueTy->getPointerTo(), call, vs );

                unsigned int size    = DL.getTypeStoreSize( valueTy );
                unsigned int align   = MinAlign( size, 16 );
                const bool createLDG = vs.preferLDG() && m_deviceSupportsLDG && isInGlobalOrGenericAddrSpace( typedPtr );
                Instruction* V       = createLoadOrLDG( typedPtr, createLDG, varref->getInputName(), align, CI );

                CI->replaceAllUsesWith( V );
                V->takeName( CI );
                toRemove.push_back( CI );
            }
            toRemove.push_back( F );
        }
        else if( RtxiSetBufferElement::isIntrinsic( F, varDesc ) )
        {
            for( CallInst* CI : getCallsToFunction( F ) )
            {
                RT_ASSERT( isa<RtxiSetBufferElement>( CI ) );
                RtxiSetBufferElement* call = cast<RtxiSetBufferElement>( CI );
                RT_ASSERT( call->getDimensionality() == dimensionality );

                Value* valueToSet   = call->getValueToSet();
                Type*  valueToSetTy = valueToSet->getType();
                Value* typedPtr =
                    createGetElementAddressCall<RtxiSetBufferElement>( refID, valueToSetTy->getPointerTo(), call, vs );

                unsigned int           size  = DL.getTypeStoreSize( valueToSetTy );
                unsigned int           align = MinAlign( size, 16 );
                corelib::CoreIRBuilder builder{CI};
                builder.CreateAlignedStore( valueToSet, typedPtr, align );

                toRemove.push_back( CI );
            }
            toRemove.push_back( F );
        }
        else if( RtxiGetBufferElementAddress::isIntrinsic( F, varDesc ) )
        {
            for( CallInst* CI : getCallsToFunction( F ) )
            {
                RT_ASSERT( isa<RtxiGetBufferElementAddress>( CI ) );
                RtxiGetBufferElementAddress* call = cast<RtxiGetBufferElementAddress>( CI );
                RT_ASSERT( call->getDimensionality() == dimensionality );

                Type*  i64Ty    = Type::getInt64Ty( context );
                Value* typedPtr = createGetElementAddressCall<RtxiGetBufferElementAddress>( refID, i64Ty, call, vs );

                call->replaceAllUsesWith( typedPtr );
                typedPtr->takeName( CI );

                toRemove.push_back( CI );
            }
            toRemove.push_back( F );
        }
        else if( RtxiAtomicSetBufferElement::isIntrinsic( F, varDesc ) )
        {
            for( CallInst* CI : getCallsToFunction( F ) )
            {
                RT_ASSERT( isa<RtxiAtomicSetBufferElement>( CI ) );
                RtxiAtomicSetBufferElement* call = cast<RtxiAtomicSetBufferElement>( CI );
                RT_ASSERT( call->getDimensionality() == dimensionality );

                // Extract the atomic operand and compare operand colwerted to the sub element type from the call.
                llvm::Value* atomicOperand  = nullptr;
                llvm::Value* compareOperand = nullptr;
                llvm::Type*  valueType      = getAtomicOperands( call, &atomicOperand, &compareOperand );

                Value* addressPtr =
                    createGetElementAddressCall<RtxiAtomicSetBufferElement>( refID, valueType->getPointerTo(), call, vs );

                ConstantInt* opType = dyn_cast<ConstantInt>( call->getOperation() );
                RT_ASSERT_MSG( opType, "Atomic operation type is not a constant" );
                AtomicOpType op = static_cast<AtomicOpType>( opType->getZExtValue() );

                CallInst* atomicCall = createAtomicCall( valueType, op, addressPtr, compareOperand, atomicOperand, call );

                Value* bitCastResult = castToInt64( call, atomicCall );

                call->replaceAllUsesWith( bitCastResult );
                atomicCall->takeName( call );
                toRemove.push_back( call );
            }
        }
    }

    eraseFromParent( toRemove );
}

void RTXVariableSpecializer::specializeBindlessBuffers( Module* module )
{
    std::vector<InstPair> toReplace;
    std::vector<Value*>   toDelete;

    for( Function* F : getFunctions( module ) )
    {
        if( RtxiGetBufferSizeFromId::isIntrinsic( F ) )
        {
            RT_ASSERT( m_bufferSizeFromIdFunc != nullptr );
            replaceGetBufferSizeFromId( F, m_bufferSizeFromIdFunc, toDelete );
        }
        else if( RtxiGetBufferElementFromId::isIntrinsic( F ) )
        {
            replaceGetBufferFromId<RtxiGetBufferElementFromId>( F, m_bufferElementFromIdFuncs, false, toDelete, toReplace );
        }
        else if( RtxiSetBufferElementFromId::isIntrinsic( F ) )
        {
            replaceSetBufferFromId<RtxiSetBufferElementFromId>( F, m_bufferElementFromIdFuncs, false, toDelete, toReplace );
        }
        else if( RtxiGetBufferElementAddressFromId::isIntrinsic( F ) )
        {
            replaceGetBufferElementAddressFromId<RtxiGetBufferElementAddressFromId>( F, m_bufferElementFromIdFuncs, false, toDelete );
        }
        else if( RtxiAtomicSetBufferElementFromId::isIntrinsic( F ) )
        {
            replaceAtomicIntrinsicFromId<RtxiAtomicSetBufferElementFromId>( F, m_bufferElementFromIdFuncs, false, toDelete );
        }
    }

    for( InstPair& instPair : toReplace )
        ReplaceInstWithInst( instPair.first, instPair.second );

    for( Value* value : toDelete )
    {
        if( GlobalValue* gv = dyn_cast<GlobalValue>( value ) )
            gv->eraseFromParent();
        else if( Instruction* instruction = dyn_cast<Instruction>( value ) )
            instruction->eraseFromParent();
        else
            RT_ASSERT_FAIL_MSG( LLVMErrorInfo( value ) + " toDelete value is not GlobalValue or Instruction" );
    }
}

// This code replaces calls to optixi_getTextureXXX
void RTXVariableSpecializer::applyTextureSpecializations( Module* module )
{
    ValueVector  toRemove;
    LLVMContext& context = module->getContext();
    Type*        i32Ty   = Type::getInt32Ty( context );

    for( Function& F : *module )
    {
        if( !F.isDeclaration() || !F.getName().startswith( "optixi_" ) )
            continue;
        StringRef kind;
        bool      isSet = false;
        StringRef uniqueName;
        if( !parsePlaceholderName( F.getName(), kind, isSet, uniqueName ) )
            continue;
        if( !kind.startswith( "Texture_" ) )
            continue;

        // The value type is the same as the return value (for get) or the last parameter (for set)
        FunctionType* fntype = F.getFunctionType();
        unsigned int  nargs  = fntype->getNumParams();
        unsigned int  N      = nargs - 1;
        RT_ASSERT( N >= 1 && N <= 3 );
        const VariableReference* varref  = m_programManager->getVariableReferenceByUniversallyUniqueName( uniqueName );
        const std::string&       varname = varref->getInputName();
        RT_ASSERT( varref->getType().isTextureSampler() );
        for( CallInst* call : getCallsToFunction( &F ) )
        {
            corelib::CoreIRBuilder irb{call};

            // Build the parameters which can vary slightly per kind
            SmallVector<Value*, 8> args;
            Function* getValueFunc = nullptr;
            Value*    state        = call->getArgOperand( 0 );
            args.push_back( state );

            // Determine specialization
            const auto&                   iter      = m_specializations.m_varspec.find( varref->getReferenceID() );
            const VariableSpecialization& vs        = iter->second;
            bool                          swtexonly = vs.accessKind == VariableSpecialization::SWTextureOnly;
            Constant* swtexonlyConstant = swtexonly ? ConstantInt::getTrue( context ) : ConstantInt::getFalse( context );
            bool      hwtexonly         = vs.accessKind == VariableSpecialization::HWTextureOnly;
            Constant* hwtexonlyConstant = hwtexonly ? ConstantInt::getTrue( context ) : ConstantInt::getFalse( context );

            TextureLookup::LookupKind lkind   = TextureLookup::fromString( kind.drop_front( 8 ) );
            Value*                    idValue = loadSpecializedVariable( module, vs, i32Ty, state, call, varname,
                                                      varref->getVariableToken(), nullptr, nullptr );
            args.push_back( idValue );
            args.push_back( hwtexonlyConstant );
            args.push_back( swtexonlyConstant );
            for( unsigned int i = 0; i < N; ++i )
                args.push_back( call->getArgOperand( 1 + i ) );
            SmallVector<Type*, 8> paramTypes( args.size() );
            algorithm::transform( args, paramTypes.begin(), []( Value* arg ) { return arg->getType(); } );
            FunctionType* fnTy = FunctionType::get( call->getType(), paramTypes, false );
            getValueFunc       = TextureLookup::getLookupFunction( lkind, "id", module, fnTy );
            Instruction* tex   = irb.CreateCall( getValueFunc, args );
            call->replaceAllUsesWith( tex );
            tex->takeName( call );
            toRemove.push_back( call );
        }
        toRemove.push_back( &F );
    }

    eraseFromParent( toRemove );
}

//// WARNING: this function changes the value of specializations.
//// This should be refactored, see the comment in MegakernelCompile::applySpecializations.
//void RTXVariableSpecializer::performLDGReadOnlyAnalysis( Module* module, const std::vector<VariableSpecialization>& specializations )
//{
//  for( VariableReferenceID refID = 0; refID < specializations.size(); ++refID )
//  {
//    // The fact that this is a potentially unsafe operation is reflected by the
//    // need to use a const_cast here. The other option would be to use a
//    // separate data structure, but this way it is much more explicit what is
//    // happening.
//    const VariableSpecialization& vs = specializations[refID];
//
//    // If this varref is already set to use LDG, or requires a
//    // different type of access, skip the analysis.
//    if( vs.accessKind != VariableSpecialization::PitchedLinear )
//      continue;
//
//    // Check if there is any "set" call for this varref.
//    const VariableReference* varref = m_programManager->getVariableReferenceById( refID );
//    if( isWritten( varref, module ) )
//      continue;
//
//    lwarn << "Using LDG for buffer that is never written: " << varref->getName() << " (id " << varref->getReferenceID() << ")"
//          << " - this may be unsafe (megakernel.useUnsafeLDGAnalysis enabled)\n";
//
//    VariableSpecialization& nonConstVS = const_cast<Specialization&>( vs );
//    nonConstVS.accessKind              = VariableSpecialization::PitchedLinearPreferLDG;
//  }
//}


//------------------------------------------------------------------------------
void eraseFromParent( const ValueVector& toRemove )
{
    for( Value* value : toRemove )
    {
        RT_ASSERT( isa<Function>( value ) || isa<Instruction>( value ) );
        if( Function* fn = dyn_cast<Function>( value ) )
            fn->eraseFromParent();
        else
            cast<Instruction>( value )->eraseFromParent();
    }
}

//------------------------------------------------------------------------------
static Instruction* createLoadOrLDG( Value* typedPtr, const bool createLDG, const std::string& varName, const unsigned int align, Instruction* insertBefore )
{
    corelib::CoreIRBuilder irb{insertBefore};
    Instruction*           V = irb.CreateAlignedLoad( typedPtr, align, varName );

    // If this "get" is marked as one that should use LDG and it is in global
    // or generic address space, create the appropriate intrinsic that LWVM
    // understands. Otherwise, do not touch the load.
    if( createLDG )
    {
        Instruction* ldg = genLDGIntrinsic( cast<LoadInst>( V ) );
        V->eraseFromParent();
        V = ldg;
    }

    return V;
}

// Helper functions.
// -----------------------------------------------------------------------------
void RTXVariableSpecializer::dump( llvm::Module* module, const std::string& functionName, int dumpId, const std::string& suffix )
{
    addMissingLineInfoAndDump( module, k_saveLLVM.get(), suffix, dumpId, m_launchCounterForDebugging,
                               functionName + "-RTXVariableSpecializer" );
}

RTXGlobalSpecializer::RTXGlobalSpecializer( int          dimensionality,
                                            unsigned int minTransformDepth,
                                            unsigned int maxTransformDepth,
                                            bool         printEnabled,
                                            int          launchCounterForDebugging )
    : m_dimensionality( dimensionality )
    , m_minTransformDepth( minTransformDepth )
    , m_maxTransformDepth( maxTransformDepth )
    , m_printEnabled( printEnabled )
    , m_launchCounterForDebugging( launchCounterForDebugging )
{
}

void RTXGlobalSpecializer::runOnModule( llvm::Module* module, const std::string& dumpName )
{
    int dumpId = 0;

    dump( module, dumpName, dumpId++, "init" );

    // If printing is disabled, remove all code that originated from rtPrintf calls.
    if( !m_printEnabled )
        specializePrintActive( module );

    // If the max transform depth is 0, remove all code related to transforms.
    // This is achieved by replacing all uses of getLwrrent/CommittedTransformDepth by 0
    // and removing all uses of setLwrrent/CommittedTransformDepth. All functions that
    // use these will resort to the default behavior if no transform was present and all
    // other code will be removed by llvm (such as loops with begin == end).
    specializeTransformDepth( module, m_minTransformDepth, m_maxTransformDepth );

    specializeGetLaunchIndex( module );

    dump( module, dumpName, dumpId++, "specialized" );
}

//------------------------------------------------------------------------------
// Specialize optixi_isPrintingEnabled calls. This is achieved by simply
// replacing the call by a constant zero, which will allow later optimizations
// to remove all code on the print side of the branch.
// TODO: We could also specialize if printing is enabled for all
//       indices, but that is not relevant for performance so we skip it for
//       now.
void RTXGlobalSpecializer::specializePrintActive( Module* module )
{
    Function* isPrintEnabledFn = module->getFunction( "optixi_isPrintingEnabled" );
    if( !isPrintEnabledFn )
        return;

    for( CallInst* call : getCallsToFunction( isPrintEnabledFn ) )
    {
        RT_ASSERT( call->getType()->isIntegerTy() );

        Constant* zero = ConstantInt::getNullValue( call->getType() );
        call->replaceAllUsesWith( zero );
        call->eraseFromParent();
    }
}

void RTXGlobalSpecializer::specializeGetLaunchIndex( llvm::Module* module )
{
    const char* oldFunc       = "_ZN4cort21Raygen_getLaunchIndexEPNS_14CanonicalStateE";
    const char* newFunc1dOr2d = "_ZN4cort27Raygen_getLaunchIndex1dOr2dEPNS_14CanonicalStateE";
    const char* newFunc3d     = "_ZN4cort23Raygen_getLaunchIndex3dEPNS_14CanonicalStateE";

    replaceFunctionWithFunction( module, oldFunc, m_dimensionality == 3 ? newFunc3d : newFunc1dOr2d );
}

//------------------------------------------------------------------------------
// This is used when we decide whether to move the max transform depth into the
// interstate. We do not move it if specializeTransformDepth will replace the
// get/set functions by a constant. This simplifies the code below since it
// does not have to care about the interstate.
bool RTXGlobalSpecializer::canSpecializeTransformDepthToConstant( unsigned int minTransformDepth, unsigned int maxTransformDepth )
{
    return minTransformDepth == maxTransformDepth;
}

//------------------------------------------------------------------------------
// Specialize for transform depth ==0, ==1, or <=1.
// VariableSpecialization for ==0 and ==1 is achieved by simply removing "set" calls
// and replacing "get" calls by constant zero or one, which will allow later
// optimizations to remove code.
// VariableSpecialization for <= 1 is achieved by replacing functions such as
// Runtime_getTransform by variants that use an if-statement rather than a loop.
void RTXGlobalSpecializer::specializeTransformDepth( Module* module, unsigned int minTransformDepth, unsigned int maxTransformDepth )
{
    RT_ASSERT( minTransformDepth <= maxTransformDepth );

    Function* getLwrrentFn =
        module->getFunction( "_ZN4cort35TraceFrame_getLwrrentTransformDepthEPNS_14CanonicalStateE" );
    if( !getLwrrentFn )
        getLwrrentFn = module->getFunction(
            "_ZN10Megakernel44TraceFrame_getLwrrentTransformDepth_registerEPN4cort14CanonicalStateE" );
    RT_ASSERT( getLwrrentFn );

    Function* getCommittedFn =
        module->getFunction( "_ZN4cort37TraceFrame_getCommittedTransformDepthEPNS_14CanonicalStateE" );
    if( !getCommittedFn )
        getCommittedFn = module->getFunction(
            "_ZN10Megakernel46TraceFrame_getCommittedTransformDepth_registerEPN4cort14CanonicalStateE" );
    RT_ASSERT( getCommittedFn );

    Function* setLwrrentFn =
        module->getFunction( "_ZN4cort35TraceFrame_setLwrrentTransformDepthEPNS_14CanonicalStateEh" );
    if( !setLwrrentFn )
        setLwrrentFn = module->getFunction(
            "_ZN10Megakernel44TraceFrame_setLwrrentTransformDepth_registerEPN4cort14CanonicalStateEh" );
    RT_ASSERT( setLwrrentFn );

    Function* setCommittedFn =
        module->getFunction( "_ZN4cort37TraceFrame_setCommittedTransformDepthEPNS_14CanonicalStateEh" );
    if( !setCommittedFn )
        setCommittedFn = module->getFunction(
            "_ZN10Megakernel46TraceFrame_setCommittedTransformDepth_registerEPN4cort14CanonicalStateEh" );
    RT_ASSERT( setCommittedFn );

    if( canSpecializeTransformDepthToConstant( minTransformDepth, maxTransformDepth ) )
    {
        RT_ASSERT( getLwrrentFn->getReturnType()->isIntegerTy() );
        RT_ASSERT( getCommittedFn->getReturnType() == getLwrrentFn->getReturnType() );
        RT_ASSERT( setLwrrentFn->getReturnType()->isVoidTy() );
        RT_ASSERT( setCommittedFn->getReturnType()->isVoidTy() );

        Constant* constDepth = ConstantInt::get( getLwrrentFn->getReturnType(), minTransformDepth, false /* isSigned */ );

        // Replace calls to getLwrrentTransformDepth by the transform depth.
        for( CallInst* call : getCallsToFunction( getLwrrentFn ) )
        {
            call->replaceAllUsesWith( constDepth );
            call->eraseFromParent();
        }

        // Replace calls to getCommittedTransformDepth by the transform depth.
        for( CallInst* call : getCallsToFunction( getCommittedFn ) )
        {
            call->replaceAllUsesWith( constDepth );
            call->eraseFromParent();
        }

        // Remove calls to setLwrrentTransformDepth.
        for( CallInst* call : getCallsToFunction( setLwrrentFn ) )
            call->eraseFromParent();

        // Remove calls to setCommittedTransformDepth.
        for( CallInst* call : getCallsToFunction( setCommittedFn ) )
            call->eraseFromParent();

        // Remove the functions.
        getLwrrentFn->eraseFromParent();
        getCommittedFn->eraseFromParent();
        setLwrrentFn->eraseFromParent();
        setCommittedFn->eraseFromParent();
    }
    else if( maxTransformDepth == 1 )
    {
        // Replace the default versions of the functions with those specialized for current depth <= 1.
        RT_ASSERT( minTransformDepth == 0 );
        replaceFunctionWithFunction( module, "_ZN4cort30Runtime_applyLwrrentTransformsEPNS_14CanonicalStateEjffff",
                                     "_ZN4cort40Runtime_applyLwrrentTransforms_atMostOneEPNS_14CanonicalStateEjffff" );
        replaceFunctionWithFunction( module, "_ZN4cort20Runtime_getTransformEPNS_14CanonicalStateEj",
                                     "_ZN4cort30Runtime_getTransform_atMostOneEPNS_14CanonicalStateEj" );
    }
    else
    {
        // Erase all the specialization functions since they are not used.
        auto erase = [&module]( const char* const name ) {
            auto fn = getFunctionOrAssert( module, name );
            RT_ASSERT( fn->use_empty() );
            fn->eraseFromParent();
        };
        erase( "_ZN4cort40Runtime_applyLwrrentTransforms_atMostOneEPNS_14CanonicalStateEjffff" );
        erase( "_ZN4cort30Runtime_getTransform_atMostOneEPNS_14CanonicalStateEj" );
    }
}

// -----------------------------------------------------------------------------
void RTXGlobalSpecializer::dump( llvm::Module* module, const std::string& functionName, int dumpId, const std::string& suffix )
{
    addMissingLineInfoAndDump( module, k_saveLLVM.get(), suffix, dumpId, m_launchCounterForDebugging,
                               functionName + "-RTXGlobalSpecializer" );
}
