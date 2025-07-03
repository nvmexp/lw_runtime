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

// Note: this file should be self-contained - i.e., not contain any optixisms.
#include <FrontEnd/Canonical/FrontEndHelpers.h>
#include <FrontEnd/Canonical/LineInfo.h>
#include <FrontEnd/PTX/PTXFrontEnd.h>
#include <FrontEnd/PTX/printPTX.h>
#include <FrontEnd/PTX/PTXHeader.h>
#include <FrontEnd/PTX/PTXInstructions_bin.h>

#include <corelib/compiler/LLVMUtil.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/CompileError.h>
#include <prodlib/system/Knobs.h>

#include "ptxConstructors.h"
#include "ptxInstructions.h"

#include <llvm/IR/Attributes.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Regex.h>

// Temporary include until we've fully moved over to lwvm70
#include <lwvm/Support/APIUpgradeUtilities.h>

#include <cstring>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

#define PTXFE_ASSERT( x )

using namespace prodlib;
using namespace optix;

// TODO:
// Is readnone dangerous especially for clock?
// Destroy module if translation or parsing failed
// How to set debug flag?
// Fill in PTXFE_ASSERT with better error handling
//
// Finish functionality:
// Implicit register for .CC bit
// Result merging for video instructions
// Negate operands for video instructions
//
// Finish texture support:
// Texref perameters
// Other tex related instructions - txq, tld4, ...
// Finish texture testing - mipmaps, txq, tld, samplers, tex arrays - b32 colwersion
//
// Finish functions:
// kernel pointer arguments (test_functions1.ptx)
// fix params of byte array (test_functions4).  It looks like a bug in lwptx
// file a bug with OCG over test_functions5 which miscompiles
// byte returns and parameters are using .b8 load/stores.  Use CreateAlignedLoad?
// modify lwvm to handle byte arrays as arguments/return values
// Address of kernels and other functions
// Address of labels


/*************************************************************
*
* PTX parsing utility functions.  Note that the PTXErrorHandler
* defined here is separate from the one in Util, which should be
* retired when PTXStitch is retired.
*
*************************************************************/

namespace {
struct PTXErrorHandler
{
    PTXErrorHandler( llvm::raw_ostream& errors );
    ~PTXErrorHandler();

    llvm::raw_ostream& m_errors;

  private:
    PTXErrorHandler* m_previousHandler;
    void*            m_oldLogger;
};

static PTXErrorHandler* s_lwrrentHandler;

static void ptxLogger( String message )
{
    if( s_lwrrentHandler )
        s_lwrrentHandler->m_errors << message << '\n';
    else
        std::cerr << "UNHANDLED PTX ERROR: " << message << '\n';  // Should not happen...
}

static void myabort()
{
    ptxLogger( String( "ptxparse aborted" ) );
}

static void myexit( int code )
{
    if( code )
        ptxLogger( String( "ptxparse exited abnormally" ) );
    else
        ptxLogger( String( "ptxparse exited normally" ) );
}

PTXErrorHandler::PTXErrorHandler( llvm::raw_ostream& errors )
    : m_errors( errors )
{
    m_previousHandler = s_lwrrentHandler;
    s_lwrrentHandler  = this;
    m_oldLogger       = (void*)stdSetLogLine( &ptxLogger );
    stdSetTerminators( &myabort, &myexit );
}

PTXErrorHandler::~PTXErrorHandler()
{
    s_lwrrentHandler = m_previousHandler;
    stdSetLogLine( (stdLogLineFunc)m_oldLogger );
    if( !s_lwrrentHandler )
        stdSetTerminators( &abort, &exit );
}
}

/*************************************************************
*
* Setup and high-level control
*
*************************************************************/

PTXFrontEnd::PTXFrontEnd( llvm::LLVMContext& llvmContext, const llvm::DataLayout* dataLayout, Debug_info_mode debugMode )
    : m_context( llvmContext )
    , m_dataLayout( dataLayout )
    , m_atomTable( nullptr )
    , m_ptxState( nullptr )
    , m_state( Initial )
    , m_errorString()
    , m_errorStream( m_errorString )
    , m_builder( llvmContext )
    , m_md_builder( llvmContext )
    , m_module( nullptr )
    , m_lwrr_func( nullptr )
    , m_lwrr_ptx_func( nullptr )
    , m_debug_info_mode( debugMode )
    , m_debug_info( nullptr )
    , m_usedGlobals()
    , m_lwrrentScope( nullptr )
{
}
PTXFrontEnd::PTXFrontEnd( llvm::Module* module, const llvm::DataLayout* dataLayout, Debug_info_mode debugMode, bool skipOptimization )
    : m_context( module->getContext() )
    , m_dataLayout( dataLayout )
    , m_atomTable( nullptr )
    , m_ptxState( nullptr )
    , m_state( Initial )
    , m_errorString()
    , m_errorStream( m_errorString )
    , m_builder( module->getContext() )
    , m_md_builder( module->getContext() )
    , m_module( module )
    , m_skipOptimizations( skipOptimization )
    , m_lwrr_func( nullptr )
    , m_lwrr_ptx_func( nullptr )
    , m_debug_info_mode( debugMode )
    , m_debug_info( nullptr )
    , m_usedGlobals()
    , m_lwrrentScope( nullptr )
{
}


PTXFrontEnd::~PTXFrontEnd()
{
    if( m_atomTable )
        FreeIAtomTable( m_atomTable );
    if( m_ptxState )
        ptxDeleteObject( m_ptxState );
}

// Adding dummyInfoPtr to resemble changed ptxCreateEmptyState() interface.
static void RTAddExtraPreProcessorMacroFlags( ptxParsingState ptxState, void* dummyInfoPtr ) {}

bool PTXFrontEnd::parsePTX( const std::string& name, const std::string& declString, const prodlib::StringView& ptxString, void* decrypter, DecryptCall decryptCall )
{
    advanceState( Initial, Parsing );

    // Create the module with the appropriate data layout and target triple
    m_module = new llvm::Module( name, m_context );
    m_module->setDataLayout( m_dataLayout->getStringRepresentation() );
    if( m_dataLayout->getPointerSizeInBits( 0 ) == 64 )
    {
        m_module->setTargetTriple( "lwptx64-lwpu-lwca" );
    }
    else
    {
        m_module->setTargetTriple( "lwptx-lwpu-lwca" );
    }

    parsePTX( declString, ptxString, decrypter, decryptCall );

    advanceState( Parsing, Parsed );
    return m_state != Error;
}

static void add_file( DebugIndexedFile* file, FrontEndDebugInfo* di )
{
    di->add_file( *file );
}

llvm::Module* PTXFrontEnd::translateModule()
{
    if( m_state == Error )
        return nullptr;
    advanceState( Parsed, Translating );

    if( m_debug_info_mode != DEBUG_INFO_OFF )
    {
        std::string fname        = k_addMissingLineInfo.get() ? std::string( "PTXFrontEnd/translateModule" ) :
                                                                std::string{ getGeneratedCodeDirectory() } + "internal";
        bool        onlyLineInfo = m_debug_info_mode == DEBUG_INFO_LINE;
        // Do not try to generate "full" debug info if the PTX does not include any.
        // Otherwise validation will fail if there are "inlinable functions" in the module
        // since the function will have a debug attachment, but the inlinable call will not.
        if( m_ptxState->foundDebugInfo || m_ptxState->enableLineInfoGeneration )
        {
            m_debug_info = new FrontEndDebugInfo( fname, m_module, onlyLineInfo, m_ptxState );
            // let FrontEndDebugInfo reuse error() and warn()
            m_debug_info->setErrorCall( std::bind( &PTXFrontEnd::error, this ) );
            m_debug_info->setWarnCall( std::bind( &PTXFrontEnd::warn, this ) );

            mapRangeTraverse( m_ptxState->dwarfFiles, (stdEltFun)add_file, m_debug_info );
            // handle data from DWARF section .debug_str
            m_debug_info->storeDwarfDebugStrSection();
        }
        else
        {
            // Always create a DICompileUnit if debug info is requested, otherwise the
            // D2IR backend will produce an error if the top-level debug metadata is undefined.
            FrontEndDebugInfo::createCompileUnitForModule( fname, onlyLineInfo, m_module );
        }
    }

    // Obey the order: first generate forward declarations, then global variables (initializers for function pointers
    // might use the forward declarations), then functions (which might use global variables).
    //
    // Note the visible functions are placed into global symbols, so must be created last
    push_scope( nullptr );
    for( stdList_t p = m_ptxState->globalSymbols->FunctionSeq; p != nullptr; p = p->tail )
        declareFunction( static_cast<ptxSymbolTableEntry>( p->head ) );

    for( stdList_t p = m_ptxState->globalSymbols->VariableSeq; p != nullptr; p = p->tail )
        processGlobalVariable( static_cast<ptxSymbolTableEntry>( p->head ) );

    // Beware that 'set' in set Traverse() refers to the data structure.
    // traverse the set parsedObjects and apply the callback processParsedObjects_cb to each element,
    // passing this as a parameter
    setTraverse( m_ptxState->parsedObjects, processParsedObjects_cb, this );

    // Process global functions imported from the LWVM bitcode
    colwertAsmCallsToCallInstructions();

    // Process global functions imported from PTX input
    for( stdList_t p = m_ptxState->globalSymbols->FunctionSeq; p != nullptr; p = p->tail )
        processGlobalFunction( static_cast<ptxSymbolTableEntry>( p->head ) );

    pop_scope();

    addLifetimesToParamAllocas( m_paramAllocas );

    if( m_lwrrentScope != nullptr )
    {
        error() << "PTXFrontEnd internal error, scopes remain\n";
        return nullptr;
    }

    emitLLVMUsed();

    if( m_debug_info != nullptr )
    {
        const std::set<std::string>& missingInlinedFunctions = m_debug_info->getMissingInlinedFunctions();
        if( missingInlinedFunctions.size() )
        {
            auto        it                     = missingInlinedFunctions.cbegin();
            std::string missingFunctionsString = *it;
            for( auto end = missingInlinedFunctions.cend(); it != end; ++it )
                missingFunctionsString += ", " + *it;
            warn() << "Inlined function" << ( missingInlinedFunctions.size() > 1 ? "s: " : ": " )
                   << missingFunctionsString << ( missingInlinedFunctions.size() > 1 ? " are" : " is" )
                   << " not accessible. Debug info might be incomplete.\n";
        }
        delete m_debug_info;
        m_debug_info = nullptr;
    }

    // There is a new function called __lw_ptx_builtin_suspend which is related
    // to coroutines. The forward declaration of that one has internal linkage.
    // LLVM does not like that.
    // We just remove those functions and produce an error if their use is not empty.
    llvm::SmallVector<llvm::Function*, 2> functionsToRemove;
    for( llvm::Function& f : *m_module )
    {
        auto linkage = f.getLinkage();
        if( f.isDeclaration() && linkage == llvm::GlobalValue::InternalLinkage )
        {
            if( !f.use_empty() )
                error() << "Undefined function used: " << f.getName().str() << "\n";
            else
                functionsToRemove.push_back( &f );
        }
    }
    for( llvm::Function* f : functionsToRemove )
        f->eraseFromParent();

    llvm::Value* av = llvm::ConstantInt::get( llvm::Type::getInt32Ty( m_context ), getTargetArch() );
    addLWVMAnnotation( av, "targetArch" );
    av = llvm::ConstantInt::get( llvm::Type::getInt1Ty( m_context ), m_ptxState->enablePtxDebug );
    addLWVMAnnotation( av, "ptxDebug" );

    advanceState( Translating, Translated );

    if( m_state == Error )
    {
        return nullptr;
    }
#if !( DEBUG )// || DEVELOP ) // TODO figure out a way to enable extra checks for DEVELOP builds
    return m_module;
#endif
    std::string errs;
    std::unique_ptr<llvm::raw_string_ostream> outputStream( new llvm::raw_string_ostream( errs ) );
    if( llvm::verifyModule( *m_module, outputStream.get() ) )
    {
        error() << errs;
        error() << "module verification failed\n";
        return nullptr;
    }

    return m_module;
}

unsigned int PTXFrontEnd::getTargetArch()
{
    if( m_state == Initial || m_state == Parsing || m_state == Error )
    {
        error() << "Target architecture invalid";
        return 0;
    }

    unsigned int target;
    sscanf( m_ptxState->target_arch, "%*[^0-9]%u", &target );
    return target;
}

std::string PTXFrontEnd::getErrorString()
{
    return m_errorStream.str();
}

void PTXFrontEnd::advanceState( State lwrState, State nextState )
{
    if( m_state == Error )
        return;

    if( m_state != lwrState )
    {
        m_errorStream << "Invalid state in PTXFrontEnd";
        m_state = Error;
    }
    else
    {
        m_state = nextState;
    }
}

llvm::raw_ostream& PTXFrontEnd::error()
{
    m_state = Error;
    return m_errorStream;
}

llvm::raw_ostream& PTXFrontEnd::warn()
{
    m_errorStream << "WARN ";
    return m_errorStream;
}

/*************************************************************
*
* Forward declarations of utility functions
*
*************************************************************/
static llvm::GlobalValue::LinkageTypes getLLVMLinkage( ptxSymbolTableEntry entry );
static bool blockIsUnreachable( llvm::BasicBlock* bb );
static ptxCodeLocation get_first_code_location( ptxSymbolTable block, ptxParsingState parsingState );
static bool hasResult( ptxInstructionTemplate tmplate );
static bool hasResultp( ptxInstruction tmplate );
static bool hasCC( ptxInstructionTemplate tmplate );
static bool hasBAR( ptxInstructionTemplate tmplate );
static bool hasSREG( ptxInstructionTemplate tmplate );
static bool hasMemArg( ptxInstructionTemplate tmplate );
static bool isReadNone( ptxInstructionTemplate tmplate );
static bool isVideoInstr( ptxInstructionTemplate tmplate );
static bool isSignedType( ptxType type );
static bool isSignedType( ptxInstruction instr, uInt arg );
static std::string getIntrinsicName( ptxInstruction instr, ptxParsingState parseState );

const unsigned int PTXFrontEnd::s_translateStorage[ptxMAXStorage] = {
    ADDRESS_SPACE_UNKNOWN,  // ptxUNSPECIFIEDStorage
    ADDRESS_SPACE_CODE,     // ptxCodeStorage
    ADDRESS_SPACE_REGS,     // ptxRegStorage
    ADDRESS_SPACE_REGS,     // ptxSregStorage
    ADDRESS_SPACE_CONST,    // ptxConstStorage
    ADDRESS_SPACE_GLOBAL,   // ptxGlobalStorage
    ADDRESS_SPACE_LOCAL,    // ptxLocalStorage
    ADDRESS_SPACE_GENERIC,  // ptxParamStorage is in alloca space
    ADDRESS_SPACE_SHARED,   // ptxSharedStorage
    ADDRESS_SPACE_GENERIC,  // ptxSurfStorage
    ADDRESS_SPACE_GENERIC,  // ptxTexStorage
    ADDRESS_SPACE_GENERIC,  // ptxTexSamplerStorage
    ADDRESS_SPACE_GENERIC   // ptxGenericStorage
};

/*************************************************************
*
* PTX Parsing
*
*************************************************************/

void PTXFrontEnd::parsePTX( const std::string& declString, const prodlib::StringView& ptxString, void* decrypter, DecryptCall decryptCall )
{
    PTXErrorHandler eh( m_errorStream );

    // Set up parser
    gpuFeaturesProfile gpu = gpuGetFeaturesProfile( ( String ) "sm_30" );
    if( !gpu )
    {
        error() << "Internal error finding sm_30\n";
        return;
    }

    initializeAtomTable( m_memoryPool, "PTXParse memory pool", &m_atomTable );

    // use dummy PtxInfoPtr to resemble changed ptxCreateEmptyState() interface. It gets only used in call to AddExtraPreProcessorMacroFlags
    void* ptxInfo{};
    stdMap_t deobfuscatedStringMap = mapNEW(String, 128);
    m_ptxState = ptxCreateEmptyState( ptxInfo, gpu, m_atomTable, nullptr, RTAddExtraPreProcessorMacroFlags, 0, nullptr, nullptr, &deobfuscatedStringMap );
    if( !m_ptxState )
    {
        error() << "Internal error while creating PTX parser state\n";
        return;
    }

    if( !declString.empty() )
    {
        Bool res0 = ptxParseInputString( const_cast<String>( "setupParser" ), const_cast<String>( declString.c_str() ),
                                         0, m_ptxState, false, false, false, static_cast<uInt32>( declString.size() ), nullptr, nullptr );
        if( !res0 )
        {
            error() << "Internal error while parsing builtin declarations\n";
            return;
        }
    }

    if( ptxString.size() > 0 )
    {
        Bool res1 = ptxParseInputString( const_cast<String>( m_module->getModuleIdentifier().c_str() ),
                                         const_cast<String>( ptxString.data() ),
                                         /*obfuscationKey=*/0,
                                         /*ptxState=*/m_ptxState,
                                         /*debugInfo=*/false,
                                         /*debugOneLineBB=*/false,
                                         /*lineInfo=*/false,
                                         /*ptxStringLength=*/static_cast<uInt32>( ptxString.size() ),
                                         /*decrypter=*/decrypter,
                                         /*decryptCall=*/(GenericCallback)decryptCall );

        if( !res1 )
        {
            error() << "Cannot parse input PTX string\n";
            return;
        }
    }
}

/*************************************************************
*
* Module-level methods
*
*************************************************************/

// Callback: transfers control to processParsedObjects
void PTXFrontEnd::processParsedObjects_cb( void* tablePtr, void* selfPtr )
{
    ptxSymbolTable table = static_cast<ptxSymbolTable>( tablePtr );
    PTXFrontEnd*   self  = static_cast<PTXFrontEnd*>( selfPtr );
    self->processParsedObject( table );
}

// passes global symbols found in the parsed objects to processGlobalSymbols.
void PTXFrontEnd::processParsedObject( ptxSymbolTable tablePtr )
{
    // Skip if we had an error in a previous object
    if( m_state == Error )
        return;

    // Forward declare functions
    for( stdList_t p = tablePtr->FunctionSeq; p != nullptr; p = p->tail )
        declareFunction( static_cast<ptxSymbolTableEntry>( p->head ) );

    // Process variables
    for( stdList_t p = tablePtr->VariableSeq; p != nullptr; p = p->tail )
        processGlobalVariable( static_cast<ptxSymbolTableEntry>( p->head ) );

    // Process functions
    for( stdList_t p = tablePtr->FunctionSeq; p != nullptr; p = p->tail )
        processGlobalFunction( static_cast<ptxSymbolTableEntry>( p->head ) );
}

void PTXFrontEnd::processGlobalVariable( ptxSymbolTableEntry entry )
{
    RT_ASSERT( entry->kind == ptxVariableSymbol );
    // ignore RegStorage for the magic A7 register (whatever A7 means)
    if( entry->storage.kind == ptxSregStorage || entry->storage.kind == ptxRegStorage )
        return;
    const int addressSpace = get_address_space( entry->storage.kind );

    //
    // Create the global variable in llvm IR
    //
    llvm::Type*                     globalType = getLLVMType( entry->symbol->type, false );
    llvm::GlobalValue::LinkageTypes linkage    = getLLVMLinkage( entry );

    llvm::Constant* initValue = nullptr;
    if( entry->initialValue )
    {
        initValue = processInitializer( entry, entry->initialValue, globalType );
    }
    else if( entry->scope != ptxExternalScope )
    {
        // Initialize to zero or NULL value if no ptx initializer present
        if( llvm::PointerType::classof( globalType ) )
        {
            if( entry->symbol->type->kind != ptxOpaqueType || addressSpace == ADDRESS_SPACE_SHARED )
            {
                initValue = llvm::ConstantPointerNull::get( static_cast<llvm::PointerType*>( globalType ) );
            }
        }
        else if( globalType->isIntegerTy() )
        {
            initValue = llvm::ConstantInt::get( globalType, 0 );
        }
        else if( globalType->isFloatingPointTy() )
        {
            initValue = llvm::ConstantFP::get( globalType, 0.0 );
        }
        else if( globalType->isAggregateType() )
        {
            initValue = llvm::ConstantAggregateZero::get( globalType );
        }
        else
        {
            std::string              typeName;
            llvm::raw_string_ostream stream( typeName );
            globalType->print( stream );
            RT_ASSERT_FAIL_MSG( "Missing initializer for type: " + stream.str() );
        }
    }

    llvm::GlobalVariable* globalVar = new llvm::GlobalVariable( *m_module,  // Global inserted at end of module globals list
                                                                globalType,                    // type of the variable
                                                                false,      // is this variable constant. TODO: isConst attribute was removed from ptxType. See CL 26313802
                                                                linkage,                       // symbol linkage
                                                                initValue,                     // Static initializer
                                                                entry->symbol->name,           // Name
                                                                nullptr,  // InsertBefore -- we want it to be appended to module's global list
                                                                llvm::GlobalVariable::NotThreadLocal,  // Thread local
                                                                addressSpace );  // The variable's address space

    globalVar->setAlignment( 1 << entry->symbol->logAlignment );

    set_symbol_address( entry->symbol, globalVar );
    add_var_debug_info( entry, globalVar );

    // handle entities of opaque types
    if( entry->symbol->type->kind == ptxOpaqueType )
    {
        // alignment should be always 8 bytes for opaque entities
        globalVar->setAlignment( 8 );

        ptxType type = entry->symbol->type;
        if( isTEXREF( type ) )
            addLWVMAnnotation( globalVar, "texture" );
        else if( isSAMPLERREF( type ) )
            addLWVMAnnotation( globalVar, "sampler" );
        else if( isSURFREF( type ) )
            addLWVMAnnotation( globalVar, "surface" );
    }

    // mark all globals as used for now, OptiX might access globals even if they are
    // not used in the code
    markUsed( globalVar );
}

// Translates an initializer.
llvm::Constant* PTXFrontEnd::processInitializer( ptxSymbolTableEntry entry, ptxInitializer init, llvm::Type* var_type )
{
    switch( init->kind )
    {
        case ptxExpressionInitializer:
        {
            ptxExpression expr = init->cases.Expression.expr;
            if( !expr->isConstant )
            {
                error() << "Non-constant expression initializer";
                return nullptr;
            }
            return processExpressionInitializer( entry, expr, var_type );
        }

        case ptxNamedFieldInitializer:
        {
            error() << "Unimplemented initializer: ptxNamedFieldInitializer\n";
        }

        case ptxStructuredInitializer:
        {
            // Array literal initializer
            return processStructuredInitializer( entry, init->cases.Structured.list, var_type );
        }
    }

    error() << "Invalid initializer kind: " << init->kind << '\n';
    return nullptr;
}

llvm::Constant* PTXFrontEnd::processExpressionInitializer( ptxSymbolTableEntry entry, ptxExpression expr, llvm::Type* var_type )
{
    switch( expr->kind )
    {
        case ptxIntConstantExpression:
        {
            ptxType type = expr->type;
            Int64   val  = expr->cases.IntConstant.i;

            // In PTX IR, the variable initializer's size does not always match
            // that of the variable itself.  (eg, initializer will be an 8 byte
            // type when variable is a .u32.  llvm does not allow this, so we
            // will force the same size.
            const int width = var_type ? var_type->getIntegerBitWidth() : ptxGetTypeSizeInBits( type );

            return llvm::ConstantInt::get( m_context, llvm::APInt( width, val, isSignedType( type ) ) );
        }
        case ptxFloatConstantExpression:
        {
            llvm::Type* type = getLLVMType( expr, false, nullptr );
            return llvm::ConstantFP::get( type, expr->cases.FloatConstant.flt );
        }

        case ptxSymbolExpression:
        {
            String          symname   = expr->cases.Symbol.symbol->symbol->name;
            llvm::Constant* globalSym = m_module->getNamedValue( symname );
            if( globalSym )
            {
                // GlobalValues are always pointers.
                if( !var_type->isPointerTy() )
                {
                    // If the variable type is not a pointer type, it needs to be
                    // a 64 bit int and we need to cast the pointer to it.
                    llvm::Type* i64Ty = llvm::Type::getInt64Ty( m_context );
                    if( var_type != i64Ty )
                    {
                        error() << "Invalid type for symbol " << entry->symbol->unMangledName
                                << ". Expected pointer type.\n";
                        return nullptr;
                    }
                    globalSym = llvm::ConstantExpr::getPtrToInt( globalSym, i64Ty );
                }
            }
            else
            {
                RT_ASSERT_FAIL_MSG( corelib::stringf( "Trying to assign unknown symbol %s to variable %s.\n", symname,
                                                      entry->symbol->unMangledName ) );
            }
            return globalSym;
        }

        default:
        {
            error() << "Unimplemented expression initializer for symbol \"" << entry->symbol->unMangledName
                    << "\": " << expr->kind << "\n";
            return nullptr;
        }
    }

    error() << "Invalid expression initializer for symbol \"" << entry->symbol->unMangledName << "\": " << expr->kind << "\n";
    return nullptr;
}

// Translates an PTX struct initializer.
llvm::Constant* PTXFrontEnd::processStructuredInitializer( ptxSymbolTableEntry entry, stdList_t list, llvm::Type* var_type )
{
    if( var_type->isArrayTy() )
    {
        llvm::Type* elem_type = var_type->getArrayElementType();
        size_t      N         = var_type->getArrayNumElements();
        if( elem_type->isFloatTy() )
            return processArrayInitializer<float>( entry, list, N, elem_type );
        else if( elem_type->isDoubleTy() )
            return processArrayInitializer<double>( entry, list, N, elem_type );
        else if( elem_type->isIntegerTy( 8 ) )
            return processArrayInitializer<uint8_t>( entry, list, N, elem_type );
        else if( elem_type->isIntegerTy( 16 ) )
            return processArrayInitializer<uint16_t>( entry, list, N, elem_type );
        else if( elem_type->isIntegerTy( 32 ) )
            return processArrayInitializer<uint32_t>( entry, list, N, elem_type );
        else if( elem_type->isIntegerTy( 64 ) )
            return processArrayInitializer<uint64_t>( entry, list, N, elem_type );
    }
    error() << "Invalid array initializer elem type: " << *var_type << '\n';
    return nullptr;
}

// Translates a PTX array initializer.
template <typename T>
llvm::Constant* PTXFrontEnd::processArrayInitializer( ptxSymbolTableEntry entry, stdList_t list, size_t N, llvm::Type* elem_type )
{
    std::vector<llvm::Constant*> constants;
    for( stdList_t p = list; p != nullptr; p = p->tail )
    {
        ptxInitializer init = static_cast<ptxInitializer>( p->head );
        if( init->kind != ptxExpressionInitializer )
        {
            error() << "Invalid array initializer element for symbol \"" << entry->symbol->unMangledName
                    << "\": " << init->kind << '\n';
            return nullptr;
        }

        ptxExpression expr = init->cases.Expression.expr;
        switch( expr->kind )
        {
            case ptxIntConstantExpression:
            {
                T val = T( expr->cases.IntConstant.i );
                constants.push_back( llvm::ConstantInt::get( elem_type, val ) );
                break;
            }
            case ptxFloatConstantExpression:
            {
                T val = T( expr->cases.DoubleConstant.dbl );
                constants.push_back( llvm::ConstantFP::get( elem_type, val ) );
                break;
            }
            case ptxAddressOfExpression:
            {
                llvm::IntegerType* int_t = llvm::dyn_cast<llvm::IntegerType>( elem_type );
                if( !int_t || ( int_t->getBitWidth() != 64 ) )
                {
                    error() << "Invalid pointer cast in array/struct initializer for symbol \""
                            << entry->symbol->unMangledName << "\"\n";
                    return nullptr;
                }

                // Retrieve the GlobalVariable of the pointed-to Symbol and cast
                // its pointer value to the passed-in element type
                ptxExpression         lhs  = expr->cases.AddressOf.lhs;
                std::string           name = lhs->cases.Symbol.symbol->symbol->name;
                llvm::GlobalVariable* gvar = m_module->getNamedGlobal( name );

                if( gvar == nullptr )
                {
                    error() << "Unable to find GlobalVariable " << lhs->cases.Symbol.symbol->symbol->unMangledName
                            << " for array/struct initializer for symbol \"" << entry->symbol->unMangledName << "\"\n";
                    return nullptr;
                }

                llvm::Constant* init  = llvm::ConstantExpr::getAddrSpaceCast( gvar, elem_type->getPointerTo() );
                llvm::Constant* init2 = llvm::ConstantExpr::getPointerCast( init, elem_type );

                constants.push_back( init2 );
                break;
            }
            case ptxSymbolExpression:
            {
                llvm::Regex vtableRegex( "_ZTV(N)?[0-9]+(.+)$" );
                llvm::SmallVector<llvm::StringRef, 3> matches;
                if( vtableRegex.match( entry->symbol->name, &matches ) )
                {
                    // If the mangled name of the array starts with _ZTV, it is a virtual function table.
                    error() << "Virtual function calls are not supported in class \"" << matches[2]
                            << "\" (vtable symbol name: \"" << entry->symbol->name << "\").\n";
                }
                else
                {
                    error() << "PTX symbols of certain types (e.g. pointers to functions) cannot be used to initialize "
                               "array \""
                            << entry->symbol->unMangledName << "\".\n";
                }
                return nullptr;
            }
            default:
            {
                error() << "Unsupported array initializer element type " << expr->kind << " for symbol \""
                        << entry->symbol->unMangledName << "\"\n";
                return nullptr;
            }
        }
    }

    // zero-pad the end of the array
    while( constants.size() < N )
    {
        if( elem_type->isFloatingPointTy() )
            constants.push_back( llvm::ConstantFP::get( elem_type, 0 ) );
        else
            constants.push_back( llvm::ConstantInt::get( elem_type, 0 ) );
    }

    return llvm::ConstantArray::get( llvm::ArrayType::get( elem_type, N ), constants );
}


// Mark a global as used.
void PTXFrontEnd::markUsed( llvm::GlobalValue* global )
{
    llvm::Type*     Int8PtrTy = llvm::Type::getInt8PtrTy( m_context, 0 );
    llvm::Constant* init      = llvm::ConstantExpr::getAddrSpaceCast( global, Int8PtrTy );
    llvm::Constant* init2     = llvm::ConstantExpr::getPointerCast( init, Int8PtrTy );
    m_usedGlobals.push_back( init2 );
}

// Generated llvm.used array.
void PTXFrontEnd::emitLLVMUsed()
{
    // Don't create llvm.used if there is no need.
    if( m_usedGlobals.empty() )
        return;

    llvm::Type*      Int8PtrTy = llvm::Type::getInt8PtrTy( m_context );
    llvm::ArrayType* ATy       = llvm::ArrayType::get( Int8PtrTy, m_usedGlobals.size() );

    llvm::GlobalVariable* GV = new llvm::GlobalVariable( *m_module, ATy, false, llvm::GlobalValue::AppendingLinkage,
                                                         llvm::ConstantArray::get( ATy, m_usedGlobals ), "llvm.used" );

    GV->setSection( "llvm.metadata" );
    m_usedGlobals.clear();
}

void PTXFrontEnd::addLifetimesToParamAllocas( const std::vector<llvm::AllocaInst*> paramAllocas )
{
    // Param storage is usually in a local scope around a function call and can often be
    // translated back into values using SROA. Unfortunately in the presence of loops
    // without lifetime markers SROA often can't do it without generating really bad code
    // with PHI nodes.
    for( llvm::AllocaInst* AI : paramAllocas )
    {
        // Find the first and last use of an alloca in the same basic block. If all uses
        // aren't in the same basic block then don't put in lifetime markers for now.
        bool               safeToProcess = true;
        llvm::BasicBlock*  definingBlock = nullptr;
        llvm::Instruction* firstUse      = nullptr;
        llvm::Instruction* lastUse       = nullptr;

        llvm::SmallPtrSet<llvm::Instruction*, 16> visited;
        llvm::SmallVector<llvm::Instruction*, 16> work;
        work.push_back( AI );
        while( !work.empty() && safeToProcess )
        {
            llvm::Instruction* I = work.back();
            work.pop_back();
            if( !visited.insert( I ).second )
                continue;

            for( auto UI = I->user_begin(), E = I->user_end(); UI != E; )
            {
                llvm::Instruction* User = llvm::dyn_cast<llvm::Instruction>( *UI++ );

                // Only consider uses that are instructions.
                if( !User )
                    continue;

                if( !definingBlock )
                {
                    definingBlock = User->getParent();
                }
                else if( definingBlock != User->getParent() )
                {
                    safeToProcess = false;
                    break;
                }

                if( llvm::dyn_cast<llvm::StoreInst>( User ) || llvm::dyn_cast<llvm::LoadInst>( User ) )
                {
                    // Intentially left blank. This is the terminating criteria
                }
                else if( llvm::GetElementPtrInst* GEP = llvm::dyn_cast<llvm::GetElementPtrInst>( User ) )
                {
                    RT_ASSERT_MSG( GEP->getPointerOperand() == I, "Use isn't the pointer to GEP" );
                    work.push_back( GEP );
                }
                else if( llvm::BitCastInst* BC = llvm::dyn_cast<llvm::BitCastInst>( User ) )
                {
                    work.push_back( BC );
                }
                else
                {
                    safeToProcess = false;
                    break;
                }

                if( !firstUse || corelib::isEarlierInst( User, firstUse ) )
                    firstUse = User;
                if( !lastUse || corelib::isEarlierInst( lastUse, User ) )
                    lastUse = User;
            }
        }
        if( !( safeToProcess && firstUse && lastUse ) )
            continue;

        m_builder.SetInsertPoint( firstUse );
        m_builder.CreateLifetimeStart( AI, m_builder.getInt64( m_dataLayout->getTypeAllocSize( AI->getAllocatedType() ) ) );
        m_builder.SetInsertPoint( lastUse->getNextNode() );
        m_builder.CreateLifetimeEnd( AI, m_builder.getInt64( m_dataLayout->getTypeAllocSize( AI->getAllocatedType() ) ) );
    }
}

/*************************************************************
*
* Function-level methods
*
*************************************************************/

/// Creates declarations for PTX functions
void PTXFrontEnd::declareFunction( ptxSymbolTableEntry entry )
{
    RT_ASSERT( entry->kind == ptxFunctionSymbol || entry->kind == ptxLabelSymbol /* for protodeclarations */ );

    llvm::FunctionType* functionType = getLLVMFunctionType( entry->aux );

    llvm::GlobalValue::LinkageTypes linkage = getLLVMLinkage( entry );
    RT_ASSERT( m_module->getFunction( entry->symbol->name ) == nullptr );
    llvm::Function* func = llvm::Function::Create( functionType, linkage, entry->symbol->name, m_module );
    set_symbol_address( entry->symbol, func );
}

/// Create the body for a previously declared function
void PTXFrontEnd::processGlobalFunction( ptxSymbolTableEntry entry )
{
    ptxSymbolTableEntryAux aux = entry->aux;

    // Only process functions with a body
    if( aux->body == nullptr )
        return;

    // Save the current ptx function for generateFunctionReturn
    m_lwrr_ptx_func = entry;

    // The function should have been created previously be
    // declareFunction.  Look for it in the module.
    m_lwrr_func = m_module->getFunction( entry->symbol->name );
    RT_ASSERT( m_lwrr_func != nullptr );

    if( aux->isEntry )
    {
        // this is a kernel, mark this by adding an annotation to the reserved lwvm.annotations
        // named meta data
        addLWVMAnnotation( m_lwrr_func, "kernel" );

        // and by setting the calling convention (which seems to be the "modern" way to do it)
        m_lwrr_func->setCallingColw( llvm::CallingColw::PTX_Kernel );
    }

    // Add LWVM-specific annotations to the kernel
    if( aux->maxnreg )
    {
        // not yet supported in 3.2 LWPTX backend
    }
    if( aux->maxntid[0] || aux->maxntid[1] || aux->maxntid[2] )
    {
        addLWVMAnnotation( m_lwrr_func, "maxntidx", aux->maxntid[0] );
        addLWVMAnnotation( m_lwrr_func, "maxntidy", aux->maxntid[1] );
        addLWVMAnnotation( m_lwrr_func, "maxntidz", aux->maxntid[2] );
    }
    if( aux->reqntid[0] || aux->reqntid[1] || aux->reqntid[2] )
    {
        addLWVMAnnotation( m_lwrr_func, "reqntidx", aux->reqntid[0] );
        addLWVMAnnotation( m_lwrr_func, "reqntidy", aux->reqntid[1] );
        addLWVMAnnotation( m_lwrr_func, "reqntidz", aux->reqntid[2] );
    }
    if( aux->minnctapersm )
    {
        addLWVMAnnotation( m_lwrr_func, "minctasm", aux->minnctapersm );
    }

    ptxSymbolTable body = aux->body;

    // generate debug info
    if( m_debug_info != nullptr )
        m_debug_info->start_function( entry, m_lwrr_func );

    // Add a start point block. Since LLVM does not allow branching to
    // the first basic block (but PTX does), add an empty block and
    // later add a branch to the real entry.  Alloca statements and
    // parameter loads will go in this block.
    llvm::BasicBlock* start_bb = llvm::BasicBlock::Create( m_context, "Start", m_lwrr_func );
    m_builder.SetInsertPoint( start_bb );
    // SetInsertPoint(BasicBlock *TheBB) does not set the current debug location
    // so m_builder will still have the previously set debug location which is in the
    // previously processed function.
    // Clear it manually.
    m_builder.SetLwrrentDebugLocation( llvm::DebugLoc() );

    // Process all instructions in the function.  This will also store
    // the function parameters into their alloca'd storage after the
    // symbols have been declared and generate an implicit return.
    processFunctionBlockStmt( body, aux->funcProtoAttrInfo->fparams, aux->funcProtoAttrInfo->rparams );

    // Final basic block can get orphaned.  Delete it.
    llvm::BasicBlock* lwr_bb = m_builder.GetInsertBlock();
    if( blockIsUnreachable( lwr_bb ) )
        lwr_bb->eraseFromParent();

    // Void return will not get handled by the code is processFunctionBlockStmt
    else if( !aux->funcProtoAttrInfo->rparams )
        m_builder.CreateRetVoid();

    // finish generated debug info
    if (m_debug_info != nullptr)
        m_debug_info->finished_function();
    m_builder.ClearInsertionPoint();
}

// For error message output.
template <typename T>
static std::string llvmToString( T* llvm )
{
    if( !llvm )
        return "";
    std::string              str;
    llvm::raw_string_ostream rso( str );
    llvm->print( rso );
    rso.flush();
    return str;
}

// Return a string that specifies the PTX register that is needed for the given LLVM type.
// Note: This does not handle aggregate types (array, struct, vector), those need to be handled
//       on the caller side.
static std::string getPtxTypeString( llvm::Type* type, const llvm::DataLayout& dl )
{
    switch( type->getTypeID() )
    {
        case llvm::Type::PointerTyID:
        case llvm::Type::IntegerTyID:
            switch( dl.getTypeSizeInBits( type ) )
            {
                case 1:
                    return ".b1";
                case 2:
                    return ".b2";
                case 4:
                    return ".b4";
                case 8:
                    return ".b8";
                case 16:
                    return ".b16";
                case 32:
                    return ".b32";
                case 64:
                    return ".b64";
                case 128:
                    return ".o128";
            }
        case llvm::Type::HalfTyID:
            return ".f16";
        case llvm::Type::FloatTyID:
            return ".f32";
        case llvm::Type::DoubleTyID:
            return ".f64";
        case llvm::Type::VoidTyID:
        case llvm::Type::X86_FP80TyID:
        case llvm::Type::FP128TyID:
        case llvm::Type::PPC_FP128TyID:
        case llvm::Type::LabelTyID:
        case llvm::Type::MetadataTyID:
        case llvm::Type::X86_MMXTyID:
        case llvm::Type::TokenTyID:
        case llvm::Type::FunctionTyID:
        case llvm::Type::StructTyID:
        case llvm::Type::ArrayTyID:
        case llvm::Type::VectorTyID:
            break;
    }
    RT_ASSERT_FAIL_MSG( ( "Unhandled LLVM->PTX type: " + llvmToString( type ) ).c_str() );
    return "";
}

// Translate the given LLVM type to a string that represents the type for PTX code generation.
// Use the given prefix string to name the PTX parameter. argc specifies the count within the
// PTX parameters and is added to the given prefix string to generate unique names.
// The result is put into the given stringstream.
static void llvmTypeToPtxParams( llvm::Type* type, llvm::Module* module, const std::string& prefix, int& argc, std::stringstream& str )
{
    const llvm::DataLayout& dl( module->getDataLayout() );
    if( type->isArrayTy() )
    {
        std::string eleStr = getPtxTypeString( type->getArrayElementType(), dl );
        for( uint64_t i = 0; i < type->getArrayNumElements(); ++i )
        {
            if( argc > 0 )
                str << ",";
            str << ".reg " << eleStr << " " << prefix << argc++ + "\n";
        }
    }
    else if( type->isStructTy() )
    {
        for( uint64_t i = 0; i < type->getStructNumElements(); ++i )
        {
            std::string eleStr = getPtxTypeString( type->getStructElementType( i ), dl );
            if( argc > 0 )
                str << ",";
            str << ".reg " << eleStr << " " << prefix << argc++ << "\n";
        }
    }
    else if( type->isVectorTy() )
    {
        const llvm::VectorType* vt     = llvm::cast<llvm::VectorType>( type );
        std::string             eleStr = getPtxTypeString( vt->getElementType(), dl );
        for( uint64_t i = 0; i < vt->getNumElements(); ++i )
        {
            if( argc > 0 )
                str << ",";
            str << ".reg " << eleStr << " " << prefix << argc++ << "\n";
        }
    }
    else
    {
        if( argc > 0 )
            str << ",";
        str << ".reg " << getPtxTypeString( type, dl ) + " " << prefix << argc++ << "\n";
    }
}

// Translate the given ASM call into a string that defines a PTX function which takes compatible
// argument types as the ASM call and has a compatible return type.
static void generatePtxFunctionFromInlineAsm( llvm::CallInst* call, const std::string& functionName, std::stringstream& str )
{
    llvm::Module* module   = call->getParent()->getParent()->getParent();
    llvm::Type*   callType = call->getType();

    str << ".visible .func ";
    int numReturlwals = 0;
    if( !callType->isVoidTy() )
    {
        str << "(";
        llvmTypeToPtxParams( callType, module, "func_retval", numReturlwals, str );
        str << ") ";
    }
    str << functionName + "(";

    int argc = 0;
    for( int i = 0; i < call->getNumArgOperands(); ++i )
    {
        llvm::Value* argVal  = call->getArgOperand( i );
        llvm::Type*  argType = argVal->getType();
        llvmTypeToPtxParams( argType, module, "func_arg", argc, str );
    }
    str << ")\n{\n";

    llvm::InlineAsm* inlineAsm    = llvm::dyn_cast<llvm::InlineAsm>( call->getCalledValue() );
    RT_ASSERT( inlineAsm );
    std::string      inlineAsmStr = inlineAsm->getAsmString();
    for( int i = 0; i < numReturlwals; ++i )
    {
        std::string indexStr = std::to_string( i );
        // Generate a regex pattern which matches '$' followed by exactly the number we are looking
        // for. E.g.. it should match $3 but not $30, so we need to use negative look ahead.
        std::string pattern  = "\\$" + indexStr + "(?!\\d)";
        // Now replace the $-parameter in the inline ASM with the matching return value.
        inlineAsmStr         = std::regex_replace( inlineAsmStr, std::regex( pattern ), "func_retval" + indexStr );
    }
    for( int i = 0; i < argc; ++i )
    {
        // Use the same replacing as above, but the inline ASM just counts up the $-placeholder
        // and does not separate input and outputs.
        std::string pattern = "\\$" + std::to_string( numReturlwals + i ) + "(?!\\d)";
        inlineAsmStr = std::regex_replace( inlineAsmStr, std::regex( pattern ), "func_arg" + std::to_string( i ) );
    }
    str << inlineAsmStr;

    str << "\n}\n\n";
}

// Returns true if F calls a function which is not present in the given module.
static bool callsFunctionNotPresentInModule( llvm::Function* F, llvm::Module* module )
{
    for( auto bb_it = F->begin(), bb_end = F->end(); bb_it != bb_end; ++bb_it )
    {
        for( auto inst_it = bb_it->begin(), inst_end = bb_it->end(); inst_it != inst_end; inst_it++ )
        {
            if( llvm::CallInst* callInst = llvm::dyn_cast<llvm::CallInst>( inst_it ) )
            {
                llvm::Function* callee = callInst->getCalledFunction();
                RT_ASSERT_MSG( callee,
                               "Internal error: Inline ASM parsing succeeded, but temporary module miralwlously "
                               "lost the called functions." );

                llvm::Function* calledFunction = module->getFunction( callee->getName() );
                if( !calledFunction )
                {
                    return true;
                }
            }
        }
    }
    return false;
}

void PTXFrontEnd::colwertAsmCallsToCallInstructions()
{
    // Extract all inline ASM calls from m_module, put them into a temporary PTX module
    // in individual functions and translate that module. Afterwards we clone the functions
    // into m_module and replace the ASM calls with calls to those functions which are
    // then inlined.
    // Collecting all ASMs of the module at once allows to only run the parser once for handling
    // all inline ASM of a module compared to doing it per function or per call.

    std::vector<llvm::CallInst*> toDelete;
    // Number of generated functions. Used to generate unique names.
    int funcCount = 0;
    // Only create a single function per unique ASM string to avoid generating duplicate functions.
    std::map<std::string, std::string> inlineAsmToFuncName;
    // Map ASM call to the generated function name.
    std::map<llvm::CallInst*, std::string> asmCallToFuncName;

    // The final (temporary) module.
    std::stringstream ptxModule;
    ptxModule << ".version " << ptxGetLatestMajorVersion() << "." << ptxGetLatestMinorVersion() << "\n.target sm_"
              << m_targetArch << "\n.address_size " << m_dataLayout->getPointerSizeInBits() << "\n\n";

    // Find ASM statements.
    for( llvm::Function& function : *m_module )
    {
        if( function.isDeclaration() )
            continue;
        for( llvm::BasicBlock& bb : function.getBasicBlockList() )
        {
            for( llvm::Instruction& inst : bb.getInstList() )
            {
                llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>( &inst );
                if( !call )
                    continue;

                llvm::InlineAsm* inlineAsm = llvm::dyn_cast<llvm::InlineAsm>( call->getCalledValue() );
                if( !inlineAsm )
                    continue;

                if( !inlineAsm->hasSideEffects() && call->use_empty() && !m_skipOptimizations )
                {
                    // Unused inline asm without sideeffects can just be removed.
                    // Leave them for -O0 mode, though.
                    toDelete.push_back( call );
                    continue;
                }

                const std::string& inlineAsmStr = inlineAsm->getAsmString();
                auto               iter         = inlineAsmToFuncName.find( inlineAsmStr );
                std::string        functionName;
                if( iter != inlineAsmToFuncName.end() )
                {
                    functionName = iter->second;
                }
                else
                {
                    do
                    {
                        // Create a unique function name and ensure that it is not colliding with existing functions in m_module.
                        functionName = "__OPTIX__tmp__inlineAsmFunction__" + std::to_string( funcCount++ ) + "__";
                    } while( m_module->getFunction( functionName ) );
                    generatePtxFunctionFromInlineAsm( call, functionName, ptxModule );
                    inlineAsmToFuncName.insert( std::make_pair( inlineAsmStr, functionName ) );
                }
                asmCallToFuncName.insert( std::make_pair( call, functionName ) );
            }
        }
    }

    if( funcCount > 0 )
    {
        // If there is any inline ASM, translate the temporary module.
        std::string headers = optix::retrieveOptixPTXDeclarations();
        PTXFrontEnd frontEnd( m_context, m_dataLayout, PTXFrontEnd::DEBUG_INFO_OFF );
        std::string ptxStr  = ptxModule.str();
        bool        success = frontEnd.parsePTX( "tmp", headers, { ptxStr.c_str(), ptxStr.size() }, nullptr, nullptr );
        if( !success )
        {
            error() << "Inline ASM handling failed. ASM parsing failed" << frontEnd.getErrorString() << "\n";
            return;
        }
        llvm::Module* tempModule = frontEnd.translateModule();
        if( !tempModule )
        {
            error() << "Inline ASM handling failed. ASM translation failed" << frontEnd.getErrorString() << "\n";
            return;
        }

        // Run SROA on the generated module to clean up parameter allocas.
        // We need to do this up front to not get in the way of later compile stages which may
        // need constant values for certain call parameters (e.g. _optix_trace for the payload).
        // Running SROA on m_module later would get in the way of -O0 mode. At this point we only
        // have the functions that we just created.
        llvm::legacy::PassManager PM;
        PM.add( llvm::createSROAPass() );
        PM.add( llvm::createInstructionCombiningPass() );  // clean up after SFT
        PM.run( *tempModule );

        // Collect the calls to the temporary functions for inlining. We know which calls
        // to inline, so this should be better than marking the functions as always inline and
        // running an alwaysInlinerPass which would need additional analysis.
        std::vector<llvm::CallInst*> callsToInline;
        // Replace each ASM call with a call to the matching function.
        for( const auto& it : asmCallToFuncName )
        {
            // We might already have the function in the module due to the caching in inlineAsmToFuncName.
            llvm::Function* newFunction = m_module->getFunction( it.second );
            if( !newFunction )
            {
                llvm::Function* F = tempModule->getFunction( it.second );
                if( !F )
                {
                    error() << "Inline ASM handling failed. Internal error: Tmp function \"" << it.second << "\" not found.\n";
                    continue;
                }

                // m_module already contains the PTXHeaders, so if F calls a function which is not present
                // in m_module it must be one of the PTX wrapper functions.
                if( !m_needsPtxInstructionsModuleLinked )
                    m_needsPtxInstructionsModuleLinked = callsFunctionNotPresentInModule( F, m_module );
                newFunction = llvm::Function::Create( F->getFunctionType(), F->getLinkage(), F->getName(), m_module );
            }
            std::vector<llvm::Value*> args;
            llvm::CallInst*           call = it.first;
            corelib::CoreIRBuilder    irb{call};
            for( int i = 0; i < call->getNumArgOperands(); ++i )
            {
                llvm::Value* callArg      = call->getArgOperand( i );
                llvm::Type*  newFuncArgTy = newFunction->getFunctionType()->getParamType( i );
                if(newFuncArgTy != callArg->getType())
                {
                    if( newFuncArgTy->isIntegerTy( m_dataLayout->getPointerSizeInBits() ) && callArg->getType()->isPointerTy() )
                    {
                        callArg = irb.CreatePtrToInt( callArg, newFuncArgTy );
                    }
                    else
                    {
                        error() << "Inline ASM handling arg type mismatch: " << llvmToString( newFuncArgTy ) << " vs. " << llvmToString( callArg->getType() ) << "\n";
                        continue;
                    }
                }
                args.push_back( callArg );
            }

            llvm::CallInst*        newCall = irb.CreateCall( newFunction, args );
            callsToInline.push_back( newCall );

            llvm::Value* value = newCall;
            if( value->getType() != call->getType() )
            {
                if( value->getType()->isIntegerTy( m_dataLayout->getPointerSizeInBits() ) && call->getType()->isPointerTy() )
                {
                    value = irb.CreateIntToPtr( value, call->getType() );
                }
                else
                {
                    error() << "Inline ASM handling return type mismatch: " << llvmToString( value->getType() ) << " vs. "
                            << llvmToString( call->getType() ) << "\n";
                    continue;
                }
            }
            call->replaceAllUsesWith( value );
            toDelete.push_back( call );
        }

        // Link and destroy the temporary Module
        llvm::Linker linker( *m_module );
        std::string errs;
        std::unique_ptr<llvm::Module> lwrrLinkModule( tempModule );
        if( linker.linkInModule( std::move( lwrrLinkModule ), llvm::Linker::StrictMatch ) )
        {
            error() << "Inline ASM handling failed to link temporary module. " << errs << "\n";
            return;
        }

        // Inline all generated calls and remove the temporary functions that were just linked into the module.
        // Note that multiple calls may have the same callee, so we use a set.
        std::set<llvm::Function*> functionsToRemove;
        for( llvm::CallInst* ci : callsToInline )
            functionsToRemove.insert( ci->getCalledFunction() );
        corelib::inlineCalls( callsToInline );
        for( llvm::Function* f : functionsToRemove )
            f->eraseFromParent();
    }

    for( llvm::CallInst* ci : toDelete )
        ci->eraseFromParent();
}

// Process a statement block of the current function.
void PTXFrontEnd::processFunctionBlockStmt( ptxSymbolTable block, stdList_t fparams, stdList_t rparams )
{
    Block_scope scope( *this, get_first_code_location( block, m_ptxState ) );

    // Declare all variables including alloca-based storage for ptx
    // registers and function parameters
    processBlockVariables( block->VariableSeq );

    // handle function parameters if applicable
    initializeFunctionParameters( fparams );

    // create basic blocks for each label in this function
    for( stdList_t p = block->LabelSeq; p != nullptr; p = p->tail )
        processFunctionLabel( static_cast<ptxSymbolTableEntry>( p->head ) );

    // iterate over all statements (instructions)
    unsigned int instrIndex = 0;
    for( stdList_t p = block->statements; p != nullptr; p = p->tail, instrIndex++ )
        processFunctionStatement( static_cast<ptxStatement>( p->head ), instrIndex );

    // The last instruction might be a label
    Scope::Idx2Block_map::const_iterator it = m_lwrrentScope->idx2block.find( instrIndex );
    if( it != m_lwrrentScope->idx2block.end() )
    {
        llvm::BasicBlock* bb = it->second;

        // If the prior instruction was a branch we may have already been
        // put here.  Only enter if we are switching blocks
        if( bb != m_builder.GetInsertBlock() )
        {
            enter_block( bb );
        }
    }

    // Generate an implicit return if applicable.  This needs to be done
    // before the scope is popped or the symbols will disappear.  Don't
    // bother if the block is unreachable.
    llvm::BasicBlock* lwr_bb = m_builder.GetInsertBlock();
    if( rparams && lwr_bb->getTerminator() == nullptr && !blockIsUnreachable( lwr_bb ) )
        generateFunctionReturn();
}

// Handle local variable declarations inside a block.
void PTXFrontEnd::processBlockVariables( stdList_t ptxSymbolTableEntries )
{
    if( !ptxSymbolTableEntries )
        return;

    // Use two builders
    // * m_builder to insert non-Alloca instructions
    // * builderAlloca to insert alloca instructions
    //
    // The logic below searches for the first non-alloca instruction
    // and puts a dummy instruction afterwards. The alloca instructions
    // will be inserted before the dummy instruction and the non-alloca
    // instruction after this dummy instruction. This way it is not
    // necessary to save/restore the IP for each variable.

    // Alloca needs to be placed in the entry block if we are not
    // already placing it there
    llvm::IRBuilderBase::InsertPoint save = m_builder.saveIP();
    llvm::BasicBlock*                bb   = &*m_lwrr_func->begin();
    /// The LLVM instruction builder.
    corelib::CoreIRBuilder builderAlloca( m_context );
    if( bb->empty() )
    {
        m_builder.SetInsertPoint( bb );
    }
    else
    {
        // Not using corelib::getFirstNonAlloca here, because it requires the IR to be
        // valid. At this point the entry block may consist of AllocaInst instructions
        // only which will cause the the iterator to point to no valid instruction.
        llvm::BasicBlock::iterator it = bb->getFirstInsertionPt();
        while( llvm::isa<llvm::AllocaInst>( it ) )
            ++it;
        m_builder.SetInsertPoint( &*it );
    }

    // create a dummy instruction as insert point for the alloca instructions
    llvm::UnreachableInst* itAlloca = m_builder.CreateUnreachable();
    builderAlloca.SetInsertPoint( itAlloca );

    for( stdList_t p = ptxSymbolTableEntries; p != nullptr; p = p->tail )
    {
        ptxSymbolTableEntry entry = static_cast<ptxSymbolTableEntry>( p->head );

        // Entry also contains allocations for "register" memory that will later get promoted to
        // registers with mem2reg
        if( entry->kind != ptxVariableSymbol )
        {
            error() << "Illegal block variable kind: " << entry->kind << '\n';
            return;
        }

        switch( entry->storage.kind )
        {
            case ptxParamStorage:
            case ptxRegStorage:
            case ptxLocalStorage:
            {
                llvm::Type* var_type = getLLVMType( entry->symbol->type, false );

                if( entry->range == 0 )
                {
                    llvm::AllocaInst* var_adr =
                        builderAlloca.CreateAlloca( var_type, /*ArraySize=*/nullptr, entry->symbol->name );
                    add_var_debug_info( entry, var_adr );
                    var_adr->setAlignment( 1u << entry->symbol->logAlignment );

                    if( entry->storage.kind == ptxLocalStorage )
                    {
                        // LWVM returns a generic pointer.  Colwert it to a local
                        // memory pointer before storing.  TODO: consider moving this
                        // to loadExpression
                        llvm::Type*  local_type = var_type->getPointerTo( ADDRESS_SPACE_LOCAL );
                        llvm::Value* local_adr  = m_builder.CreatePointerCast( var_adr, local_type );
                        set_symbol_address( entry->symbol, local_adr );
                    }
                    else
                    {
                        // Store pointer directly
                        set_symbol_address( entry->symbol, var_adr );
                        if( entry->storage.kind == ptxParamStorage )
                        {
                            m_paramAllocas.push_back( var_adr );
                        }
                    }
                }
                else
                {
                    // it seems that we can safely ignore ranges, because all instantiated variables
                    // are also in the variable list
                }
                continue;
            }
            default:
            {
#define __ptxKindNameMacro( x, y, z ) y,
                const char* names[] = {"unknown", ptxStorageKindIterate( __ptxKindNameMacro ) "unknown"};
#undef __ptxKindNameMacro
                if( entry->symbol->unMangledName )
                    error() << "Unsupported storage kind \"" << names[entry->storage.kind] << "\" for symbol "
                            << entry->symbol->unMangledName << '\n';
                else
                    error() << "Unsupported storage kind: " << names[entry->storage.kind] << '\n';

                continue;
            }
        }
        error() << "Illegal block storage kind: " << entry->storage.kind << '\n';
    }

    itAlloca->eraseFromParent();
    m_builder.restoreIP( save );
}

// Process a function label
void PTXFrontEnd::processFunctionLabel( ptxSymbolTableEntry label )
{
    // Update of the ptx parser (CL 26053661) introduced three new label kinds apart from
    // ptxLabelSymbol. At least ptxCallPrototypeSymbol was formerly mapped to ptxLabelSymbol,
    // so assume all these label kinds can be treated the same as ptxLabelSymbol here.
    if( !( label->kind == ptxLabelSymbol || label->kind == ptxBranchTargetSymbol || label->kind == ptxCallTargetSymbol
           || label->kind == ptxCallPrototypeSymbol ) )
    {
        error() << "Illegal function label kind: " << label->kind << '\n';
        return;
    }

    // Skip ptx marker label
    if( 0 == std::strncmp( label->symbol->name, "__$endLabel$__", 14 ) )
        return;

    m_lwrrentScope->label2Idx[label->symbol] = label->aux->listIndex;

    if( m_lwrrentScope->idx2block.find( label->aux->listIndex ) == m_lwrrentScope->idx2block.end() )
    {
        // create a new basic block for this index
        llvm::BasicBlock* labeled_bb = llvm::BasicBlock::Create( m_context, label->symbol->name, m_lwrr_func );

        // and enter it into the lookup table
        m_lwrrentScope->idx2block[label->aux->listIndex] = labeled_bb;
    }

    // handle ptxBranchTargetSymbol
    if( label->kind == ptxBranchTargetSymbol )
    {
        if( label->initialValue->kind == ptxStructuredInitializer )
        {
            std::vector<ptxSymbol> branchTargets;
            stdList_t              list = label->initialValue->cases.Structured.list;
            for( stdList_t p = list; p != nullptr; p = p->tail )
            {
                ptxInitializer init = static_cast<ptxInitializer>( p->head );
                if( init->kind != ptxExpressionInitializer )
                {
                    //error()
                    return;
                }

                ptxExpression expr = init->cases.Expression.expr;
                if( expr->kind != ptxSymbolExpression )
                {
                    //error()
                    return;
                }
                ptxSymbolTableEntry labelEntry = expr->cases.Symbol.symbol;
                branchTargets.push_back( labelEntry->symbol );
            }
            m_lwrrentScope->label2Branchtargets[label->symbol] = branchTargets;
        }
    }
}

/// Initialize the local copies of the function parameters
void PTXFrontEnd::initializeFunctionParameters( stdList_t list )
{
    llvm::Function::arg_iterator arg = m_lwrr_func->arg_begin();
    for( stdList_t p = list; p != nullptr; p = p->tail, ++arg )
    {
        if( arg == m_lwrr_func->arg_end() )
        {
            error() << "Internal error: function parameter mismatch\n";
            return;
        }
        ptxVariableInfo var  = static_cast<ptxVariableInfo>( p->head );
        llvm::Value*    addr = get_symbol_address( var->symbol );
        m_builder.CreateAlignedStore( arg, addr, 1 << var->symbol->logAlignment );
        mark_var_assigned( var->symbol, arg );
    }
}

// Generate necessary instructions to handle a function return.
void PTXFrontEnd::generateFunctionReturn()
{
    ptxSymbolTableEntryAux aux = m_lwrr_ptx_func->aux;

    if( aux->funcProtoAttrInfo->rparams == nullptr )
    {
        // no return
        m_builder.CreateRetVoid();
    }
    else if( aux->funcProtoAttrInfo->rparams->tail == nullptr )
    {
        // Single return value
        ptxVariableInfo var   = static_cast<ptxVariableInfo>( aux->funcProtoAttrInfo->rparams->head );
        llvm::Value*    adr   = get_symbol_address( var->symbol );
        llvm::Value*    value = m_builder.CreateAlignedLoad( adr, 1 << var->symbol->logAlignment );
        if( llvm::isa<llvm::ArrayType>( value->getType() ) )
        {
            // LWVM can't handle returning an array.  An array of bytes is a common pattern, so we wrap that in a struct.
            m_builder.CreateAggregateRet( &value, 1 );
        }
        else
        {
            // Normal return
            m_builder.CreateRet( value );
        }
    }
    else
    {
        // Multiple return values
        std::vector<llvm::Value*> values;
        for( stdList_t p = aux->funcProtoAttrInfo->rparams; p != nullptr; p = p->tail )
        {
            ptxVariableInfo var   = static_cast<ptxVariableInfo>( p->head );
            llvm::Value*    adr   = get_symbol_address( var->symbol );
            llvm::Value*    value = m_builder.CreateAlignedLoad( adr, 1 << var->symbol->logAlignment );
            values.push_back( value );
        }
        m_builder.CreateAggregateRet( values.data(), (unsigned)values.size() );
    }
}


/*************************************************************
*
* Instruction-level methods
*
*************************************************************/

void PTXFrontEnd::processFunctionStatement( ptxStatement statement, unsigned int instrIndex )
{
    // check if we should enter a new block. Look only in the current scope for the instruction.
    Scope::Idx2Block_map::const_iterator it = m_lwrrentScope->idx2block.find( instrIndex );
    if( it != m_lwrrentScope->idx2block.end() )
    {
        llvm::BasicBlock* bb = it->second;

        // If the prior instruction was a branch we may have already been
        // put here.  Only enter if we are switching blocks
        if( bb != m_builder.GetInsertBlock() )
        {
            enter_block( bb );
        }
    }

    switch( statement->kind )
    {
        case ptxInstructionStatement:
        {
            translateInstruction( statement->cases.Instruction.instruction, instrIndex );
            return;
        }
        case ptxPragmaStatement:
        {
            // Ignore pragmas.
            return;
        }
    }
    error() << "Illegal function statement kind: " << statement->kind << '\n';
}

// Transition to a new basic block
void PTXFrontEnd::enter_block( llvm::BasicBlock* bb )
{
    llvm::BasicBlock* lwr_bb = m_builder.GetInsertBlock();
    if( lwr_bb->getTerminator() == nullptr )
    {
        // current block has no terminator, add a fall through
        m_builder.CreateBr( bb );
    }
    m_builder.SetInsertPoint( bb );
    m_lwrrentBlockValues.reset();
}

bool isSupportedPtxInstruction( ptxInstructionCode code )
{
    switch( code )
    {
        case ptx_wmma_load_a_Instr:
        case ptx_wmma_load_b_Instr:
        case ptx_wmma_load_c_Instr:
        case ptx_wmma_mma_Instr:
        case ptx_mbarrier_arrive_Instr:
        case ptx_mbarrier_ilwal_Instr:
        case ptx_mbarrier_test_wait_Instr:
        case ptx_wmma_store_d_Instr:
        case ptx_mma_Instr:
        case ptx_mbarrier_arrive_drop_Instr:
        case ptx_mbarrier_init_Instr:
        case ptx_mbarrier_pending_count_Instr:
        case ptx_mbarrier_test_wait_parity_Instr:
#if LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
        case ptx_mbarrier_try_wait_Instr:
        case ptx_mbarrier_try_wait_parity_Instr:
#endif
            return false;
        default:
            return true;
    }
}

llvm::Function* PTXFrontEnd::getPtxInstructionFunction( ptxInstruction instr )
{
    std::string     func_name = getIntrinsicName( instr, m_ptxState );
    llvm::Function* func      = m_module->getFunction( func_name );
    if( func )
        return func;

    ptxInstructionTemplate tmplate = instr->tmplate;

    if( !isSupportedPtxInstruction( static_cast<ptxInstructionCode>( tmplate->code ) ) )
    {
        error() << "Unsupported instruction: " << func_name;
        return nullptr;
    }

    // Build return type
    llvm::Type* dst_type = nullptr;
    if( hasResult( tmplate ) )
    {
        dst_type = getLLVMArgType( instr, 0 );
        if( !dst_type )
            return nullptr;

        if( hasResultp( instr ) )
        {
            // Build a struct type to contain the original type and a predicate
            llvm::Type* types[] = {dst_type, llvm::Type::getInt1Ty( m_context )};
            dst_type            = llvm::StructType::get( m_context, llvm::ArrayRef<llvm::Type*>( types ) );
        }
    }
    else
    {
        if( hasResultp( instr ) )
        {
            error() << "Unexpected resultp";
        }
        dst_type = llvm::Type::getVoidTy( m_context );
    }

    // Build argument types
    unsigned int             start = hasResult( tmplate ) ? 1 : 0;
    std::vector<llvm::Type*> src_types;
    for( unsigned int i = start; i < tmplate->nrofArguments; ++i )
    {
        llvm::Type* type = getLLVMArgType( instr, i );
        if( !type )
            return nullptr;
        src_types.push_back( type );
    }

    // Create function and cast it to the appropriate type
    llvm::FunctionType* func_type = llvm::FunctionType::get( dst_type, src_types, false );
    func                          = llvm::cast<llvm::Function>( m_module->getOrInsertFunction( func_name, func_type ) );
    func->addFnAttr( llvm::Attribute::NoUnwind );
    if( tmplate->code == ptx_trap_Instr || tmplate->code == ptx_exit_Instr )
    {
        // Add noreturn attribute to trap and exit to help the optimizer
        func->addFnAttr( llvm::Attribute::NoReturn );
    }
    else if( isReadNone( tmplate ) )
    {
        // Add readnone attribute to most instructions to help the optimizer
        func->addFnAttr( llvm::Attribute::ReadNone );
    }

    return func;
}

// Translate a function instruction.
void PTXFrontEnd::translateInstruction( ptxInstruction instr, unsigned int instrIndex )
{
    if( m_debug_info )
    {
        ptxCodeLocation loc =
            (ptxCodeLocation)rangemapApply( m_ptxState->ptxToSourceLine.instructionMap, (rangemapRange_t)instr->loc->lwrLoc.lineNo );
        m_debug_info->set_lwrr_loc( loc, m_builder );
    }

    ptxInstructionTemplate tmplate = instr->tmplate;

    switch( static_cast<ptxInstructionCode>( tmplate->code ) )
    {
        case ptx_bra_Instr:
            return translate_bra( instr, instrIndex );
        case ptx_brx_idx_Instr:
            return translate_brx_idx( instr );
        case ptx_call_Instr:
            return translate_call( instr );
        case ptx_ret_Instr:
            return translate_ret( instr, instrIndex );
        case ptx_exit_Instr:
        case ptx_trap_Instr:
            return translate_exit_or_trap( instr, instrIndex );
        case ptx_ld_Instr:
            return translate_ld( instr );
        case ptx_st_Instr:
            return translate_st( instr );
        case ptx_tex_Instr:
            return translate_tex( instr );
        default:
            return translate_default( instr );
    }
}

void PTXFrontEnd::translate_default( ptxInstruction instr )
{
    llvm::BasicBlock* resume_bb = translateGuard( instr );

    llvm::Function* func = getPtxInstructionFunction( instr );
    if( !func )
    {
        // There was error, skip this instruction
        return;
    }

    std::vector<llvm::Value*> args;
    loadSrcOperands( instr, args );
    if( m_state == Error )
        return;

    llvm::Value* ret = m_builder.CreateCall( func, args );

    if( hasResult( instr->tmplate ) )
        storeDstOperands( instr, ret );

    resumeAfterGuard( resume_bb );
}

// Translate a bra instruction.
void PTXFrontEnd::translate_bra( ptxInstruction instr, unsigned int instrIndex )
{
    // Determine target of branch
    ptxSymbol targetLabel = nullptr;
    if( instr->tmplate->nrofArguments != 1 || instr->tmplate->argType[0] != ptxTargetAType )
    {
        error() << "Illegal branch target\n";
        return;
    }

    if( instr->arguments[0]->kind == ptxSymbolExpression )
    {
        targetLabel = instr->arguments[0]->cases.Symbol.symbol->symbol;
    }
    else
    {
        error() << "Unimplmented branch type\n";
        return;
    }

    // Look for the label relwrsively in current and parent scopes
    if( m_lwrrentScope == nullptr )
    {
        error() << "Invalid scope in translate branch\n";
        return;
    }

    llvm::BasicBlock* tgt_bb = m_lwrrentScope->findLabel( targetLabel );
    if( !tgt_bb )
    {
        error() << "Label '" << targetLabel->name << "' not found in scope\n";
        return;
    }

    // Find the successor block
    llvm::BasicBlock*                    fallthrough_bb = nullptr;
    unsigned int                         next           = instrIndex + 1;
    Scope::Idx2Block_map::const_iterator bit            = m_lwrrentScope->idx2block.find( next );
    if( bit != m_lwrrentScope->idx2block.end() )
    {
        fallthrough_bb = bit->second;
    }
    else
    {
        // found a block without a label
        fallthrough_bb = llvm::BasicBlock::Create( m_context, "fallthrough", m_lwrr_func );
    }

    if( ptxExpression guard = instr->guard )
    {
        // Conditional branch, examine predicate
        llvm::Value* condition = loadExpression( guard, false, false, ARG_COLWERSION_NORMAL, instr );
        m_builder.CreateCondBr( condition, tgt_bb, fallthrough_bb );
    }
    else
    {
        // unconditional branch
        m_builder.CreateBr( tgt_bb );
    }

    enter_block( fallthrough_bb );
}

void PTXFrontEnd::translate_brx_idx( ptxInstruction instr )
{
    llvm::Value* index{};
    ptxSymbol    label{};

    if( instr->tmplate->nrofArguments != 2 )
    {
        error() << "Unexpected number of arguments in branch target idx instruction\n";
        return;
    }

    if( instr->tmplate->argType[0] == ptxU32AType )
    {
        ptxExpression indexExpr = instr->arguments[0];
        if( indexExpr->kind != ptxSymbolExpression )
        {
            error() << "Invalid branch target idx instruction\n";
            return;
        }
        index = loadExpression( indexExpr, false, false, ARG_COLWERSION_NORMAL, instr );
    }

    if( instr->tmplate->argType[1] == ptxSymbolAType )
    {
        ptxExpression labelExpr = instr->arguments[1];
        if( labelExpr->kind != ptxSymbolExpression )
        {
            error() << "Invalid branch targets list\n";
            return;
        }
        label = labelExpr->cases.Symbol.symbol->symbol;
    }

    const std::vector<ptxSymbol>* targetLabels = m_lwrrentScope->lookupBranchtargets( label );
    if( !targetLabels )
    {
        error() << "No branch targets list found for label " << label->name << "\n";
        return;
    }

    loweringBrxIdx2Switch( index, targetLabels );
}

// Lowering the brx.idx to a switch instruction.
// Another version could had been based on lowering brx.idx to an indirectbr. The block addresses of the labels
// are kept in a GV. While the generated code looked good it caused issues in a later rtcore pass, where the
// function had been renamed but not its used resources - hence the GV leaked and caused assertions. Hence not
// followed through.
void optix::PTXFrontEnd::loweringBrxIdx2Switch( llvm::Value* index, const std::vector<ptxSymbol>* targetLabels )
{
    llvm::BasicBlock* default_bb = llvm::BasicBlock::Create( m_context, "UnreachableBlock", m_lwrr_func );
    corelib::CoreIRBuilder{ default_bb }.CreateUnreachable();

    llvm::SwitchInst*             stmt         = m_builder.CreateSwitch( index, default_bb, targetLabels->size() );
    for( unsigned int i = 0; i < targetLabels->size(); ++i )
        stmt->addCase( llvm::ConstantInt::get( llvm::Type::getInt32Ty( m_context ), i ),
                       m_lwrrentScope->findLabel( ( *targetLabels )[i] ) );
}

void PTXFrontEnd::translate_call( ptxInstruction instr )
{
    llvm::BasicBlock* resume_bb = translateGuard( instr );
    ;

    // Parse the call.  It may or may not have arguments and may or may
    // not have a return value.  Look for the function identifier and
    // then look for returns on the left, arguments on the right.
    uInt fnidx = 999;
    for( unsigned int i = 0; i < instr->tmplate->nrofArguments; ++i )
    {
        if( instr->tmplate->argType[i] == ptxTargetAType )
        {
            fnidx = i;
            break;
        }
    }
    if( fnidx == 999 )
    {
        error() << "Unhandled call format\n";
        return;
    }
    ptxExpression fnExpr = instr->arguments[fnidx];
    if( fnExpr->kind != ptxSymbolExpression )
    {
        error() << "Unhandled call target\n";
        return;
    }
    std::string  fnName = fnExpr->cases.Symbol.symbol->symbol->name;
    llvm::Value* fn     = m_module->getFunction( fnName );
    if( fn == nullptr )
    {
        fn = m_lwrrentScope->lookupAdr( fnExpr->cases.Symbol.symbol->symbol );
        if( fn == nullptr )
        {
            error() << "Call target not found: " << fnName << "\n";
            return;
        }
        // An indirect call. Load the address
        fn = m_builder.CreateAlignedLoad( fn, 1 << fnExpr->cases.Symbol.symbol->symbol->logAlignment );
    }

    // Build arguments
    stdList_t fnargs = nullptr;
    if( fnidx + 1 < instr->tmplate->nrofArguments )
    {
        ptxExpression argExpr = instr->arguments[fnidx + 1];
        if( argExpr->kind != ptxParamListExpression )
        {
            error() << "Unhandled argument list type: " << argExpr->kind << '\n';
            return;
        }
        fnargs = argExpr->cases.ParamList.elements;
        if( fnidx + 2 < instr->tmplate->nrofArguments )
        {
            // label specifying prototype
            ptxSymbolTableEntry label       = instr->arguments[fnidx + 2]->cases.Symbol.symbol;
            llvm::Type*         funcTypePtr = nullptr;
            if( label->initialValue != nullptr )
            {
                // .calltarget - Label initializer is a list of functions. Take type of the first one
                ptxInitializer init   = (ptxInitializer)label->initialValue->cases.Structured.list->head;
                ptxSymbol      symbol = init->cases.Expression.expr->cases.Symbol.symbol->symbol;
                funcTypePtr           = m_module->getFunction( symbol->name )->getType();
            }
            else
            {
                // .callprototype - Prototype in label->aux
                llvm::FunctionType* funcType = getLLVMFunctionType( label->aux );
                funcTypePtr                  = llvm::PointerType::get( funcType, 0 );
            }
            fn = m_builder.CreateIntToPtr( fn, funcTypePtr );
        }
    }

    std::vector<llvm::Value*> args;
    for( stdList_t p = fnargs; p != nullptr; p = p->tail )
    {
        ptxExpression argExpr  = static_cast<ptxExpression>( p->head );
        llvm::Value*  argValue = loadExpression( argExpr, false, false, ARG_COLWERSION_NORMAL, instr );
        args.push_back( argValue );
    }

    // Generate the call
    llvm::Value* ret = m_builder.CreateCall( fn, args );

    // Unpack the results
    stdList_t fnrets = nullptr;
    if( fnidx > 0 )
    {
        ptxExpression retExpr = instr->arguments[fnidx - 1];
        if( retExpr->kind != ptxParamListExpression )
        {
            error() << "Unhandled return list type: " << retExpr->kind << '\n';
            return;
        }
        fnrets = retExpr->cases.ParamList.elements;
    }
    if( fnrets == nullptr )
    {
        // Nothing
    }
    else if( fnrets->tail == nullptr )
    {
        // A single value.  But if it is an array type we need to unpack it first.
        RT_ASSERT( fnrets->head != nullptr );
        ptxExpression retExpr = static_cast<ptxExpression>( fnrets->head );
        if( retExpr->kind != ptxSinkExpression && retExpr->type->kind == ptxArrayType )
        {
            ret = m_builder.CreateExtractValue( ret, 0 );
        }
        storeExpression( retExpr, ret, ARG_COLWERSION_NORMAL, instr );
    }
    else
    {
        // Multiple values
        int idx = 0;
        for( stdList_t p = fnrets; p != nullptr; p = p->tail, ++idx )
        {
            ptxExpression retExpr  = static_cast<ptxExpression>( p->head );
            llvm::Value*  retValue = m_builder.CreateExtractValue( ret, idx );
            storeExpression( retExpr, retValue, ARG_COLWERSION_NORMAL, instr );
        }
    }

    resumeAfterGuard( resume_bb );
}

// Translate a ret instruction.
void PTXFrontEnd::translate_ret( ptxInstruction instr, unsigned int instrIndex )
{
    // Find the successor block
    llvm::BasicBlock*                    fallthrough_bb = nullptr;
    unsigned int                         next           = instrIndex + 1;
    Scope::Idx2Block_map::const_iterator bit            = m_lwrrentScope->idx2block.find( next );
    if( bit != m_lwrrentScope->idx2block.end() )
    {
        fallthrough_bb = bit->second;
    }
    else
    {
        // found a block without a label
        fallthrough_bb = llvm::BasicBlock::Create( m_context, "", m_lwrr_func );
    }

    if( ptxExpression guard = instr->guard )
    {
        // predicated ret instructions must be expressed by control flow
        llvm::BasicBlock* ret_bb = llvm::BasicBlock::Create( m_context, "", m_lwrr_func );

        // examine predicate
        llvm::Value* condition = loadExpression( guard, false, false, ARG_COLWERSION_NORMAL, instr );
        m_builder.CreateCondBr( condition, ret_bb, fallthrough_bb );

        // Place the return in the return block
        enter_block( ret_bb );
    }

    generateFunctionReturn();

    enter_block( fallthrough_bb );
}

void PTXFrontEnd::translate_exit_or_trap( ptxInstruction instr, unsigned int instrIndex )
{
    llvm::BasicBlock* resume_bb = translateGuard( instr );

    if( instr->tmplate->nrofArguments != 0 )
    {
        error() << "Invalid exit/trap instruction\n";
        return;
    }
    llvm::Function* func = getPtxInstructionFunction( instr );

    m_builder.CreateCall( func );
    m_builder.CreateUnreachable();

    // Unreachable is a terminator instruction so we may need to begin a
    // new basic block
    llvm::BasicBlock*                    fallthrough_bb = nullptr;
    unsigned int                         next           = instrIndex + 1;
    Scope::Idx2Block_map::const_iterator bit            = m_lwrrentScope->idx2block.find( next );
    if( bit != m_lwrrentScope->idx2block.end() )
    {
        fallthrough_bb = bit->second;
    }
    else
    {
        // found a block without a label
        fallthrough_bb = llvm::BasicBlock::Create( m_context, "unreachable", m_lwrr_func );
    }
    enter_block( fallthrough_bb );

    resumeAfterGuard( resume_bb );
}

void PTXFrontEnd::translate_ld( ptxInstruction instr )
{
    if( instr->storage->kind == ptxParamStorage )
    {
        llvm::BasicBlock* resume_bb = translateGuard( instr );
        ;

        // Load the parameter directly from the alloca'd copy.
        llvm::Value* adr = loadExpression( instr->arguments[1], false, false, ARG_COLWERSION_NORMAL, instr );

        // Determine the type of the pointer including the address space
        // and cast the pointer if necessary
        llvm::Type* ptrType = getLLVMArgType( instr, 1 );
        if( adr )
            adr = supercast( adr, ptrType, ARG_COLWERSION_NORMAL );

        // Accomodate translation errors
        if( !adr || !ptrType )
        {
            error() << "Failed to translate load\n";
            return;
        }

        // Load the value
        llvm::Value* val = m_builder.CreateAlignedLoad( adr, 1 << ptxGetTypeLogAlignment( instr->type[0] ) );

        // Store into the destination register
        ArgColwersionKind colwersion = largeArgColwersion( instr, 0 );
        storeExpression( instr->arguments[0], val, colwersion, instr );

        resumeAfterGuard( resume_bb );
    }
    else
    {
        translate_default( instr );
    }
}

void PTXFrontEnd::translate_st( ptxInstruction instr )
{
    if( instr->storage->kind == ptxParamStorage )
    {
        llvm::BasicBlock* resume_bb = translateGuard( instr );
        ;
        // Store the parameter directly to the alloca'd copy.

        // Get the address value and cast it to the appropriate type if
        // necessary.
        llvm::Value* adr     = loadExpression( instr->arguments[0], false, false, ARG_COLWERSION_NORMAL, instr );
        llvm::Type*  ptrType = getLLVMArgType( instr, 0 );
        if( adr )
            adr = supercast( adr, ptrType, ARG_COLWERSION_NORMAL );

        // Get the value to be stored and cast it to the appropriate type if
        // necessary.
        ArgColwersionKind colwersion = largeArgColwersion( instr, 1 );
        llvm::Value*      val        = loadExpression( instr->arguments[1], false, false, colwersion, instr );
        if( val )
            val = supercast( val, ptrType->getPointerElementType(), colwersion );

        // Accomodate translation errors
        if( !adr || !val || !ptrType )
        {
            error() << "Failed to translate store\n";
            return;
        }

        // Issue the store
        m_builder.CreateAlignedStore( val, adr, 1 << ptxGetTypeLogAlignment( instr->type[0] ) );

        resumeAfterGuard( resume_bb );
    }
    else
    {
        translate_default( instr );
    }
}

void PTXFrontEnd::translate_tex( ptxInstruction instr )
{
    ptxExpression texExpr = instr->arguments[1];
    if( !isTEXREF( texExpr->type ) )
    {
        // Assume this is a bindless texture reference and use the default path.
        return translate_default( instr );
    }

    // Handle guard predicate if present
    llvm::BasicBlock* resume_bb = translateGuard( instr );

    if( !hasResult( instr->tmplate ) || instr->tmplate->nrofArguments != 3 )
    {
        error() << "Unhandled texture instruction kind\n";
        return;
    }

    // Get the texture name
    if( texExpr->kind != ptxSymbolExpression )
    {
        error() << "Unhandled texture parameter kind: " << texExpr->kind;
        return;
    }
    String                    texname = texExpr->cases.Symbol.symbol->symbol->name;
    std::vector<llvm::Value*> texargs;

    llvm::GlobalVariable* texrefVar = m_module->getNamedGlobal( texname );
    // texture handle instrinsic gets texture handle from texture GlobalVariable and its annotation
    // delcare i64 %llvm.lwvm.texsurf.handle.p1i64(metadata, i64 addrspace(1)*)
    llvm::Value* texHandleFn =
        llvm::Intrinsic::getDeclaration( m_module, llvm::Intrinsic::lwvm_texsurf_handle, texrefVar->getType() );

    // texture ref annotation
    std::map<llvm::Value*, llvm::MDNode*>::const_iterator annotIt = m_lwvmAnnotations.find( texrefVar );
    PTXFE_ASSERT( annotIt != m_lwvmAnnotations.end() );
    llvm::MDNode* annot = annotIt->second;

    llvm::Value* texHandleFnArgs[] = {UseMdAsValue( m_module->getContext(), annot ), texrefVar};
    llvm::Value* texHandle         = m_builder.CreateCall( texHandleFn, texHandleFnArgs );

    texargs.push_back( texHandle );

    unsigned int dim        = ptxGetTextureDim( instr->modifiers );
    unsigned int vectorSize = ptxGetVectorSize( instr->modifiers );

    ptxExpression argExpr = instr->arguments[2];

    ptxInstructionType argPTXType = instr->tmplate->instrType[1];
    ptxInstructionType retPTXType = instr->tmplate->instrType[0];

    // Get the texture arguments
    if( argExpr->kind == ptxVectorExpression )
    {
        uInt n = 0;
        for( stdList_t p = argExpr->cases.Vector.elements; p != nullptr; p = p->tail )
        {
            n++;
            ptxExpression eltExpr  = static_cast<ptxExpression>( p->head );
            llvm::Value*  eltValue = loadExpression( eltExpr, false, false, ARG_COLWERSION_NORMAL, instr );
            texargs.push_back( eltValue );
            // Stop if there are too many elements in the vector
            if( n >= dim )
                break;
        }
        if( n != dim )
        {
            error() << "Argument list for texture has " << n << " elements but requires " << dim << "\n";
            return;
        }
    }
    else if( argExpr->kind == ptxSymbolExpression )
    {
        llvm::Value* eltValue = loadExpression( argExpr, false, false, ARG_COLWERSION_NORMAL, instr );
        texargs.push_back( eltValue );
    }
    else
    {
        error() << "Unhandled texture argument kind: " << argExpr->kind << '\n';
        return;
    }

    if( 4 != vectorSize )
    {
        // Texture is always result4
        error() << "Invalid texture return vector: " << vectorSize << '\n';
        return;
    }
    int nres = 4;

    llvm::Intrinsic::ID id;

    PTXFE_ASSERT( retPTXType == ptxFloatIType || retPTXType == ptxIntIType );
    PTXFE_ASSERT( argPTXType == ptxFloatIType || argPTXType == ptxIntIType );

    PTXFE_ASSERT( ptxFloatIType < 256 && ptxFloatIType < 256 );
#define PACK( dim, ret, arg ) ( ( dim ) + ( ( ret ) << 8 ) + ( ( arg ) << 16 ) )
    switch( PACK( dim, retPTXType, argPTXType ) )
    {
        // clang-format off
    case PACK( 1, ptxFloatIType, ptxFloatIType ): id = llvm::Intrinsic::lwvm_tex_unified_1d_v4f32_f32; break;
    case PACK( 1, ptxFloatIType, ptxIntIType   ): id = llvm::Intrinsic::lwvm_tex_unified_1d_v4f32_s32; break;
    case PACK( 1, ptxIntIType,   ptxFloatIType ): id = llvm::Intrinsic::lwvm_tex_unified_1d_v4s32_f32; break;
    case PACK( 1, ptxIntIType,   ptxIntIType   ): id = llvm::Intrinsic::lwvm_tex_unified_1d_v4s32_s32; break;

    case PACK( 2, ptxFloatIType, ptxFloatIType ): id = llvm::Intrinsic::lwvm_tex_unified_2d_v4f32_f32; break;
    case PACK( 2, ptxFloatIType, ptxIntIType   ): id = llvm::Intrinsic::lwvm_tex_unified_2d_v4f32_s32; break;
    case PACK( 2, ptxIntIType,   ptxFloatIType ): id = llvm::Intrinsic::lwvm_tex_unified_2d_v4s32_f32; break;
    case PACK( 2, ptxIntIType,   ptxIntIType   ): id = llvm::Intrinsic::lwvm_tex_unified_2d_v4s32_s32; break;

    case PACK( 3, ptxFloatIType, ptxFloatIType ): id = llvm::Intrinsic::lwvm_tex_unified_3d_v4f32_f32; break;
    case PACK( 3, ptxFloatIType, ptxIntIType   ): id = llvm::Intrinsic::lwvm_tex_unified_3d_v4f32_s32; break;
    case PACK( 3, ptxIntIType,   ptxFloatIType ): id = llvm::Intrinsic::lwvm_tex_unified_3d_v4s32_f32; break;
    case PACK( 3, ptxIntIType,   ptxIntIType   ): id = llvm::Intrinsic::lwvm_tex_unified_3d_v4s32_s32; break;

    default: error() << "Unsupported tex instruction"; return;
            // clang-format on
    }
#undef PACK

    llvm::Value* texFetchFn = llvm::Intrinsic::getDeclaration( m_module, id );

    llvm::VectorType* resultVecType = llvm::VectorType::get(
        retPTXType == ptxFloatIType ? llvm::Type::getFloatTy( m_context ) : llvm::Type::getInt32Ty( m_context ), nres );

    llvm::Value* ret = m_builder.CreateCall( texFetchFn, texargs );
    // Colwert the return struct to a vector
    llvm::Value* retVec = llvm::UndefValue::get( resultVecType );
    for( int i = 0; i < nres; i++ )
    {
        llvm::Value* el       = m_builder.CreateExtractValue( ret, i );
        llvm::Value* idxValue = m_builder.getInt32( i );
        retVec                = m_builder.CreateInsertElement( retVec, el, idxValue );
    }

    // Finally, store the value
    // Intentionally not using storeDstOperands here. The tex instruction has resultP modifier set,
    // but these versions of it don't really have it. storeDstOperands would try to handle
    // the output predicate.
    ArgColwersionKind colwersion = largeArgColwersion( instr, 0 );
    storeExpression( instr->arguments[0], retVec, colwersion, instr );

    // Pick up in the resume block after a guarded instruction
    resumeAfterGuard( resume_bb );
}

llvm::BasicBlock* PTXFrontEnd::translateGuard( ptxInstruction instr )
{
    if( !instr->guard )
        return nullptr;

    llvm::Value* pred = loadExpression( instr->guard, false, false, ARG_COLWERSION_NORMAL, instr );

    llvm::BasicBlock* trueBlock  = llvm::BasicBlock::Create( m_context, "guard", m_lwrr_func );
    llvm::BasicBlock* falseBlock = llvm::BasicBlock::Create( m_context, "guard", m_lwrr_func );
    m_builder.CreateCondBr( pred, trueBlock, falseBlock );
    enter_block( trueBlock );
    return falseBlock;  // Resume here
}

void PTXFrontEnd::resumeAfterGuard( llvm::BasicBlock* resume_bb )
{
    // Pick up in the resume block after a guarded instruction
    if( resume_bb )
        enter_block( resume_bb );
}

void PTXFrontEnd::loadSrcOperands( ptxInstruction instr, std::vector<llvm::Value*>& args, bool includeReturlwalue )
{
    unsigned int start = ( includeReturlwalue == false ) ? ( hasResult( instr->tmplate ) ? 1 : 0 ) : 0;
    for( unsigned int i = start; i < instr->tmplate->nrofArguments; ++i )
    {
        // Evaluate the expression
        ArgColwersionKind colwersion = largeArgColwersion( instr, i );
        llvm::Value*      value      = loadExpression( instr->arguments[i], false, false, colwersion, instr );

        if( !value )
        {
            std::string result;
            printPtxOpCode( result, instr, m_ptxState );
            error() << "line " << instr->loc->lwrLoc.lineNo << ", error evaluating argument " << i
                    << " in instruction: " << result << '\n';
            return;
        }

        // Determine the destination type and cast to it
        //  - Sreg parameters can be implicitly narrowed
        //  - Integer constants (and vectors thereof) can be implicitly narrowed
        //  - Symbol addresses can be implicitly narrowed to 32-bit or widened to 64-bit
        //  - FollowAType parameters to instructions marked as LARG can be implicitly narrowed
        //  - Pointer types are automatically colwerted
        //  - fp values automatically colwerted to inter types for .b params
        llvm::Type* intoType = getLLVMArgType( instr, i );
        value                = supercast( value, intoType, colwersion );

        args.push_back( value );
    }
}

void PTXFrontEnd::storeDstOperands( ptxInstruction instr, llvm::Value* ret )
{
    if( hasResultp( instr ) )
    {
        // Extract the predicate and store it (only if it exists)
        if( instr->predicateOutput )
        {
            llvm::Value* pred_return = m_builder.CreateExtractValue( ret, 1 );
            storeExpression( instr->predicateOutput, pred_return, ARG_COLWERSION_NORMAL, instr );
        }
        // Extract the main return value and continue below
        ret = m_builder.CreateExtractValue( ret, 0 );
    }

    ArgColwersionKind colwersion = largeArgColwersion( instr, 0 );
    storeExpression( instr->arguments[0], ret, colwersion, instr );
}

/*************************************************************
*
* Expression methods
*
*************************************************************/

PTXFrontEnd::ArgColwersionKind PTXFrontEnd::largeArgColwersion( ptxInstruction instr, uInt arg ) const
{
    if( ptxHasLARG_Feature( instr->tmplate->features ) && instr->tmplate->argType[arg] == ptxFollowAType )
    {
        // If the LARG flag is set on this instruction, then the value can
        // be implicitly widened/narrowed to fit destination register

        uInt follow = instr->tmplate->followMap[arg];
        PTXFE_ASSERT( follow < instr->tmplate->nrofInstrTypes );
        ptxInstructionType iType       = instr->tmplate->instrType[follow];
        ptxTypeKind        argTypeKind = instr->arguments[arg]->type->kind;

        bool isBitType   = isBitTypeKind( argTypeKind );
        bool isFloatType = isFloatKind( argTypeKind );
        // From ptxInstructionTemplates.c:
        // . (inst-type is .bX) OR (arg-type is .bX) OR (inst-type not .fX AND arg-type not .fX)
        if( iType == ptxBitIType || isBitType || ( iType != ptxFloatIType && !isFloatType ) )
        {
            // Allow widening to the output register width.  The colwersion
            // should be sign-extended if the instruction type is signed.
            return isSignedType( instr, arg ) ? ARG_COLWERSION_LARGEARG_SIGNED : ARG_COLWERSION_LARGEARG_UNSIGNED;
        }
    }

    return ARG_COLWERSION_NORMAL;
}


llvm::Value* PTXFrontEnd::supercast( llvm::Value* value, llvm::Type* toType, ArgColwersionKind colwersion )
{
    llvm::Type* fromType = value->getType();
    if( fromType == toType )
        return value;

    // Cast pointer to pointer
    if( fromType->isPointerTy() && toType->isPointerTy() )
    {
        // The removal of ptxPointerType in the parser caused some information to get lost.
        // For pointers getLLVMType now determines the address space based on the instruction.
        // Unfortunately some instructions ('mov'!) do not have the storage->kind set.
        if( toType->getPointerAddressSpace() == ADDRESS_SPACE_UNKNOWN )
            toType = llvm::PointerType::get( toType->getPointerElementType(), fromType->getPointerAddressSpace() );
        if( fromType->getPointerAddressSpace() == toType->getPointerAddressSpace() )
            return m_builder.CreateBitCast( value, toType );
        else
            return m_builder.CreateAddrSpaceCast( value, toType );
    }

    // Cast pointer to integer, even if it is lossy
    if( fromType->isPointerTy() && toType->isIntegerTy() )
        return m_builder.CreatePtrToInt( value, toType );

    // Cast integer to pointer
    if( fromType->isIntegerTy() && toType->isPointerTy() )
        return m_builder.CreateIntToPtr( value, toType );

    // Cast integer to integer
    if( fromType->isIntegerTy() && toType->isIntegerTy() )
    {
        if( colwersion == ARG_COLWERSION_LARGEARG_SIGNED )
            return m_builder.CreateSExtOrTrunc( value, toType );
        else
            return m_builder.CreateZExtOrTrunc( value, toType );
    }

    // Cast (double,float,half) to integer
    if( fromType->isFloatingPointTy() && toType->isIntegerTy() )
    {
        llvm::IntegerType* intTy = llvm::IntegerType::get( m_context, fromType->getScalarSizeInBits() );
        value                    = m_builder.CreateBitCast( value, intTy );
        return m_builder.CreateZExtOrTrunc( value, toType );
    }

    // Cast integer to (double,float,half)
    if( fromType->isIntegerTy() && toType->isFloatingPointTy() )
    {
        llvm::IntegerType* intTy = llvm::IntegerType::get( m_context, toType->getScalarSizeInBits() );
        value                    = m_builder.CreateZExtOrTrunc( value, intTy );
        return m_builder.CreateBitCast( value, toType );
    }

    // Cast double to float
    if( fromType->isDoubleTy() && toType->isFloatTy() )
        return m_builder.CreateFPTrunc( value, toType );

    // Cast float to double
    // Note - PTX zero-extends the value instead of using a floating point extension.
    if( fromType->isFloatTy() && toType->isDoubleTy() )
    {
        llvm::IntegerType* intTy1 = llvm::IntegerType::get( m_context, 32 );
        llvm::IntegerType* intTy2 = llvm::IntegerType::get( m_context, toType->getScalarSizeInBits() );
        value                     = m_builder.CreateBitCast( value, intTy1 );
        value                     = m_builder.CreateZExt( value, intTy2 );
        value                     = m_builder.CreateBitCast( value, toType );
        return value;
    }

    // Pack vector into integer
    if( fromType->isVectorTy() && toType->isIntegerTy() )
    {
        unsigned int vecWidth = fromType->getVectorNumElements();
        unsigned int elWidth;
        if( colwersion == ARG_COLWERSION_NORMAL )
            // Regular pack
            elWidth = toType->getScalarSizeInBits() / vecWidth;
        else
            // Implicit narrowing
            elWidth              = fromType->getScalarSizeInBits();
        llvm::IntegerType* elTy  = llvm::IntegerType::get( m_context, elWidth );
        llvm::VectorType*  vecTy = llvm::VectorType::get( elTy, vecWidth );
        value                    = supercast( value, vecTy, colwersion );
        if( vecTy->getPrimitiveSizeInBits() > toType->getScalarSizeInBits() )
        {
            llvm::Type* wideIntType = llvm::IntegerType::get( m_context, vecTy->getPrimitiveSizeInBits() );
            value                   = m_builder.CreateBitCast( value, wideIntType );
            value                   = m_builder.CreateTrunc( value, toType );
        }
        else
        {
            value = m_builder.CreateBitCast( value, toType );
        }
        return value;
    }

    // Unpack vector from integer
    if( fromType->isIntegerTy() && toType->isVectorTy() )
    {
        unsigned int vecWidth = toType->getVectorNumElements();
        unsigned int elWidth;
        if( colwersion == ARG_COLWERSION_NORMAL )
            // Regular unpack
            elWidth = fromType->getScalarSizeInBits() / vecWidth;
        else
            // Implicit widening
            elWidth              = toType->getScalarSizeInBits();
        llvm::IntegerType* elTy  = llvm::IntegerType::get( m_context, elWidth );
        llvm::VectorType*  vecTy = llvm::VectorType::get( elTy, vecWidth );
        if( vecTy->getPrimitiveSizeInBits() > fromType->getScalarSizeInBits() )
        {
            llvm::Type* wideIntType = llvm::IntegerType::get( m_context, vecTy->getPrimitiveSizeInBits() );
            value                   = m_builder.CreateZExt( value, wideIntType );
        }
        value = m_builder.CreateBitCast( value, vecTy );
        return supercast( value, toType, colwersion );
    }

    // Cast vector to vector (repeating last element if from is smaller than to).
    //   We settled on the convention of using a <4 x float> for the texture coordinate
    //   the optix.ptx.tex.* intrinsics, so we need to handle the up cast. There is
    //   some danger that this cast will trigger when it shouldn't, i.e. for something
    //   other than texture coordinates.
    if( fromType->isVectorTy() && toType->isVectorTy() && fromType->getVectorNumElements() <= toType->getVectorNumElements() )
    {
        llvm::Value* ret       = llvm::UndefValue::get( toType );
        unsigned int fromWidth = fromType->getVectorNumElements();
        llvm::Value* e         = nullptr;
        for( unsigned int i = 0, ie = toType->getVectorNumElements(); i < ie; ++i )
        {
            llvm::Value* idxValue = llvm::ConstantInt::get( llvm::Type::getInt32Ty( m_context ), i );
            if( i < fromWidth )
            {
                e = m_builder.CreateExtractElement( value, idxValue );
                e = supercast( e, toType->getVectorElementType(), colwersion );
            }
            ret = m_builder.CreateInsertElement( ret, e, idxValue );
        }
        return ret;
    }

    // Cast vector to struct. Vector and struct need to have the same element count and
    // the elements need to have the same bit size. E.g. fromType: <4 x int32>, toType: {i32, f32, f32, f32}.
    // This is needed for the a1d/a2d/alwbe texture instructions.
    if( fromType->isVectorTy() && toType->isStructTy() && fromType->getVectorNumElements() == toType->getStructNumElements() )
    {
        llvm::Type*       vecElementType = fromType->getVectorElementType();
        llvm::Value*      ret            = llvm::UndefValue::get( toType );
        llvm::StructType* type           = llvm::cast<llvm::StructType>( toType );
        for( unsigned int i = 0, e = fromType->getVectorNumElements(); i < e; ++i )
        {
            // cast value from vector to struct element type and add it to the struct
            llvm::Type* structElementType = type->getElementType( i );
            if( structElementType->getPrimitiveSizeInBits() != vecElementType->getPrimitiveSizeInBits() )
            {
                error() << "Invalid cast from " << *vecElementType << " to: " << structElementType
                        << ": Element size mismatch\n";
                return nullptr;
            }
            llvm::Value* idxValue = llvm::ConstantInt::get( llvm::Type::getInt32Ty( m_context ), i );
            llvm::Value* element  = m_builder.CreateExtractElement( value, idxValue );

            if( structElementType != vecElementType )
                element = m_builder.CreateBitCast( element, structElementType );
            ret         = m_builder.CreateInsertValue( ret, element, i );
        }
        return ret;
    }

    // Instructions like tex.1d either take a scalar or a vector with one element. We always want a scalar
    // for those to make it easier for the definition-parsing-generator scripts. getLLVMArgType does the same.
    if( fromType->isVectorTy() && fromType->getVectorNumElements() == 1 && toType == fromType->getVectorElementType() )
    {
        llvm::Value* idxValue = llvm::ConstantInt::get( llvm::Type::getInt32Ty( m_context ), 0 );
        llvm::Value* element  = m_builder.CreateExtractElement( value, idxValue );
        return element;
    }

    error() << "Unexpected cast from: " << *fromType << " to: " << *toType << '\n';
    return nullptr;
}

llvm::Value* PTXFrontEnd::loadExpression( ptxExpression expr, bool addressof, bool addressref, ArgColwersionKind colwersion, ptxInstruction instr )
{
    // Load the expression from the alloca-d registers or llvm symbols. When addressof
    // gets set (relwrsively), we switch to collecting the address of the symbol.  Colwert
    // to the type of the expression.

    llvm::Type* exprType = getLLVMType( expr, false, instr );
    if( !exprType )
        return nullptr;

    switch( expr->kind )
    {
        case ptxBinaryExpression:
        {
            // In theory only place that this can appear is in a base-offset
            // callwlation, but it looks like ptx allows it almost anywhere.
            // (specifically any 64-bit constant).  Cast both operands to an
            // integer, add and cast back to the expression type.
            PTXFE_ASSERT( expr->cases.Binary->op == ptxADDOp );
            PTXFE_ASSERT( expr->cases.Binary->left->kind == ptxIntConstantExpression
                          || expr->cases.Binary->right == ptxIntConstantExpression );
            PTXFE_ASSERT( exprType->isIntegerTy() || exprType()->isPointerTy() );

            {
                // This is an optimization that produces cleaner IR for direct
                // references to global scope variables.  If it is a pointer
                // offset computation where the offset is evenly divisible by
                // the element size, use the GEP instruction instead of casting
                // to int for computing the offsets.  Otherwise, use the default
                // path.
                ptxExpression baseExpr = expr->cases.Binary->left;
                ptxExpression idxExpr  = expr->cases.Binary->right;
                if( expr->cases.Binary->left->kind == ptxIntConstantExpression )
                    std::swap( baseExpr, idxExpr );  // Base and index are swapped

                llvm::Type* baseType = getLLVMType( baseExpr, false, instr );
                if( idxExpr->kind == ptxIntConstantExpression && baseType->isPointerTy() )
                {
                    // These need to be signed, because if the idxExpr is negative, you need to have the
                    // idxValue / elementSize be signed extended property ( (-12*8) / 32 should be -3 not 576 bazillion)
                    int64_t elementSize = baseType->getPointerElementType()->getPrimitiveSizeInBits();
                    int64_t idxValue    = idxExpr->cases.IntConstant.i * 8;

                    if( idxValue % elementSize == 0 )
                    {
                        llvm::Value* base    = loadExpression( baseExpr, addressof, addressref, colwersion, instr );
                        llvm::Type*  idxType = getLLVMType( idxExpr, false, instr );
                        llvm::Value* idx     = llvm::ConstantInt::get( idxType, idxValue / elementSize );
                        llvm::Value* ptr     = m_builder.CreateGEP( base, idx );
                        if( addressref || !exprType->isPointerTy() )
                            // HACK: removal of the ptxPointerType has left us with ugly side effects.
                            // exprType will most likely not be a pointer type here, but an int type.
                            // getLLVMType will return the correct type in some cases because it uses
                            // ptxGetAddressArgBaseType, but it will not work in all cases.
                            // However, the result of the pointer offset computation should always be a pointer type.
                            // So, for those cases we do not colwert the ptr here, but it as it is,
                            // hoping for the best that the caller will know which type it needs.
                            return ptr;
                        else
                            return supercast( ptr, exprType, colwersion );
                    }
                }
            }

            // Default path
            llvm::Type* intTy = exprType;
            if( exprType->isPointerTy() )
                intTy = m_dataLayout->getIntPtrType( exprType );

            llvm::Value* l = loadExpression( expr->cases.Binary->left, addressof, false, colwersion, instr );
            l              = supercast( l, intTy, colwersion );

            llvm::Value* r = loadExpression( expr->cases.Binary->right, addressof, false, colwersion, instr );
            r              = supercast( r, intTy, colwersion );

            llvm::Value* sum = m_builder.CreateAdd( l, r );
            return supercast( sum, exprType, colwersion );
        }

        case ptxIntConstantExpression:
        {
            llvm::Value* ret = llvm::ConstantInt::getSigned( exprType, expr->cases.IntConstant.i );
            return ret;
        }

        case ptxFloatConstantExpression:
        {
            // using the colwersion through a ConstantInt is solely for the issue when dealing with denorm values
            // When floating point handling is configured to eg flush these to zero, even constant value settings
            // different from 0 would be changed to be 0. This workaround avoids "touching" the float value and
            // passed it through.
            if( ptxGetTypeSizeInBytes( expr->type ) == 4 )
            {
                static_assert( sizeof( int32_t ) == sizeof( float ), "int and float type differ in size" );
                llvm::ConstantInt* intConst =
                    llvm::ConstantInt::get( llvm::Type::getInt32Ty( m_context ),
                                            *reinterpret_cast<int32_t*>( &expr->cases.FloatConstant.flt ) );
                llvm::Value* value = llvm::ConstantExpr::getBitCast( intConst, llvm::Type::getFloatTy( m_context ) );
                return supercast( value, exprType, colwersion );
            }
            else
            {
                static_assert( sizeof( int64_t ) == sizeof( double ), "int and float type differ in size" );
                llvm::ConstantInt* intConst =
                    llvm::ConstantInt::get( llvm::Type::getInt64Ty( m_context ),
                                            *reinterpret_cast<int64_t*>( &expr->cases.DoubleConstant.dbl ) );
                llvm::Value* value = llvm::ConstantExpr::getBitCast( intConst, llvm::Type::getDoubleTy( m_context ) );
                return supercast( value, exprType, colwersion );
            }
        }

        case ptxSymbolExpression:
        {
            ptxSymbol     symbol     = expr->cases.Symbol.symbol->symbol;
            ptxSymbolKind symbolKind = expr->cases.Symbol.symbol->kind;
            if( addressof || symbolKind == ptxFunctionSymbol )
            {
                PTXFE_ASSERT( intoType != 0 );
                llvm::Value* pointer = get_symbol_address( symbol );
                return pointer;  // No colwersion - will get colwerted by the addressOfExpression
            }
            else
            {
                ptxStorageKind storageKind = expr->cases.Symbol.symbol->storage.kind;
                if( storageKind == ptxSregStorage )
                {
                    llvm::Value* value = callSregIntrinsic( symbol->name );
                    if( !value )
                        return nullptr;  // Error reported in callSreg
                    return supercast( value, exprType, colwersion );
                }
                else
                {
                    llvm::Value* value = m_lwrrentBlockValues.find( symbol );
                    if( !value )
                    {
                        llvm::Value* adr   = get_symbol_address( symbol );
                        value = m_builder.CreateAlignedLoad( adr, 1 << symbol->logAlignment );
                    }
                    return supercast( value, exprType, colwersion );
                }
            }
        }

        case ptxArrayIndexExpression:
        {
            // PTX does not actually dereference the array - it just returns
            // the address.  Base will always be an array symbol.
            llvm::Value* base      = loadExpression( expr->cases.ArrayIndex->arg, true, false, colwersion, instr );
            llvm::Value* index     = loadExpression( expr->cases.ArrayIndex->index, false, false, colwersion, instr );
            llvm::Value* zero      = llvm::ConstantInt::get( index->getType(), 0 );
            llvm::Value* indices[] = {zero, index};
            return m_builder.CreateGEP( base, indices );
        }

        case ptxVectorSelectExpression:
        {
            PTXFE_ASSERT( !addressof );
            if( expr->cases.VectorSelect->arg->kind == ptxSymbolExpression
                && expr->cases.VectorSelect->arg->cases.Symbol.symbol->storage.kind == ptxSregStorage )
            {
                PTXFE_ASSERT( expr->cases.VectorSelect->dimension == 1 );
                ptxSymbol          sym                = expr->cases.VectorSelect->arg->cases.Symbol.symbol->symbol;
                static const char* selectorToString[] = {".x", ".y", ".z", ".w"};
                std::string        name =
                    std::string( sym->name ) + std::string( selectorToString[expr->cases.VectorSelect->selector[0]] );
                llvm::Value* sreg = callSregIntrinsic( name );
                if( !sreg )
                    return nullptr;  // Error reported in callSreg

                return supercast( sreg, exprType, colwersion );
            }
            else
            {
                // PTX does not support vector swizzle but it would probably be easy to support if
                // this changes.
                PTXFE_ASSERT( expr->cases.VectorSelect->dimension == 1 );

                llvm::Value* vecValue = loadExpression( expr->cases.VectorSelect->arg, false, false, colwersion, instr );
                uInt         idx      = expr->cases.VectorSelect->selector[0];
                llvm::Value* idxValue = llvm::ConstantInt::get( llvm::Type::getInt32Ty( m_context ), idx );
                llvm::Value* elt      = m_builder.CreateExtractElement( vecValue, idxValue );
                return supercast( elt, exprType, colwersion );
            }
        }

        case ptxPredicateExpression:
        {
            PTXFE_ASSERT( !addressof );
            llvm::Value* p = loadExpression( expr->cases.Predicate.arg, false, false, colwersion, instr );
            if( expr->neg )
            {
                p = m_builder.CreateNot( p );
            }
            return p;
        }

        case ptxAddressOfExpression:
        {
            PTXFE_ASSERT( !addressof );
            llvm::Value* ret = loadExpression( expr->cases.AddressOf.lhs, true, false, colwersion, instr );
            return supercast( ret, exprType, colwersion );
        }

        case ptxAddressRefExpression:
        {
            PTXFE_ASSERT( !addressof );
            llvm::Value* ret = loadExpression( expr->cases.AddressRef.arg, false, true, colwersion, instr );
            return ret; // No colwersion - let the instruction take care of it based on getLLVMArgType.
        }

        case ptxVectorExpression:
        {
            if( !exprType->isVectorTy() )
            {
                error() << "Unexpected type in vector expression: " << *exprType << '\n';
                return nullptr;
            }
            // Insert values into the vector. Casts are handled at the
            // individual element level
            llvm::Value* ret    = llvm::UndefValue::get( exprType );
            int          idx    = 0;
            llvm::Type*  elType = exprType->getVectorElementType();
            for( stdList_t p = expr->cases.Vector.elements; p != nullptr; p = p->tail, ++idx )
            {
                ptxExpression el       = static_cast<ptxExpression>( p->head );
                llvm::Value*  elValue  = loadExpression( el, false, false, colwersion, instr );
                llvm::Value*  idxValue = llvm::ConstantInt::get( llvm::Type::getInt32Ty( m_context ), idx );
                if( elValue->getType() != elType )
                {
                    if( elValue->getType()->getPrimitiveSizeInBits() != elType->getPrimitiveSizeInBits() )
                    {
                        error() << "Invalid type in vector expression: " << *elValue->getType()
                                << " does not have the expected bit size of " << elType->getPrimitiveSizeInBits() << '\n';
                        return nullptr;
                    }
                    elValue = m_builder.CreateBitCast( elValue, elType );
                }
                ret = m_builder.CreateInsertElement( ret, elValue, idxValue );
            }
            return ret;
        }

        case ptxVideoSelectExpression:
        {
            PTXFE_ASSERT( !addressof );
            return loadExpression( expr->cases.VideoSelect->arg, false, false, colwersion, instr );
        }

        case ptxSinkExpression:
        case ptxUnaryExpression:
        case ptxStructExpression:
        case ptxLabelReferenceExpression:
        case ptxParamListExpression:
        case ptxByteSelectExpression:
        {
            error() << "Unimplemented expression kind in loadExpression: " << expr->kind << "\n";
            return nullptr;
        }
    }

    error() << "Illegal expression kind in loadExpression: " << expr->kind << '\n';
    return nullptr;
}

void PTXFrontEnd::storeExpression( ptxExpression expr, llvm::Value* value, ArgColwersionKind colwersion, ptxInstruction instr )
{
    // Colwert to the type specified in the expression, but sink
    // expressions do not have a valid type.
    if( expr->kind != ptxSinkExpression )
    {
        llvm::Type* exprType = getLLVMType( expr, false, instr );
        value                = supercast( value, exprType, colwersion );
        // Avoid segfault on an error
        if( !value )
            return;
    }

    switch( expr->kind )
    {
        case ptxSymbolExpression:
        {
            ptxSymbol symbol = expr->cases.Symbol.symbol->symbol;
            llvm::Value* addr = get_symbol_address( symbol );

            // Store the value
            m_builder.CreateAlignedStore( value, addr, 1 << expr->cases.Symbol.symbol->symbol->logAlignment );
            // Keep track of the value we just stored
            m_lwrrentBlockValues.add( symbol, value );
            mark_var_assigned( symbol, addr );
            return;
        }

        case ptxVectorSelectExpression:
        {
            llvm::Value* vecValue = loadExpression( expr->cases.VectorSelect->arg, false, false, colwersion, instr );

            // We are only storing one element of the vector.  We could either:
            // 1. Load old vector, replace element, store vector
            // 2. Pull the symbol evaluation up here, assuming that symbols
            //    are the only possible base of the vector and use
            //    GetElementPtr to address the individual element.
            // Choose #1 since LLVM disrecommends use of GEP on vectors.  This
            // construct is rare in PTX anyway and the optimizer will sort out
            // the inefficiencies.

            uInt         idx      = expr->cases.VectorSelect->selector[0];
            llvm::Value* idxValue = llvm::ConstantInt::get( llvm::Type::getInt32Ty( m_context ), idx );
            vecValue              = m_builder.CreateInsertElement( vecValue, value, idxValue );
            storeExpression( expr->cases.VectorSelect->arg, vecValue, colwersion, instr );
            return;
        }

        case ptxVectorExpression:
        {
            llvm::Type* valueType = value->getType();
            if( !valueType->isVectorTy() )
            {
                error() << "Unexpected vector expression type: " << *valueType << '\n';
                return;
            }
            // Extract the elements of the vector and relwrse.
            int idx = 0;
            for( stdList_t p = expr->cases.Vector.elements; p != nullptr; p = p->tail, ++idx )
            {
                ptxExpression el       = static_cast<ptxExpression>( p->head );
                llvm::Value*  idxValue = llvm::ConstantInt::get( llvm::Type::getInt32Ty( m_context ), idx );
                llvm::Value*  elValue  = m_builder.CreateExtractElement( value, idxValue );
                storeExpression( el, elValue, colwersion, instr );
            }
            return;
        }

        case ptxSinkExpression:
        {
            // No need to store, NOP
            return;
        }

        case ptxVideoSelectExpression:
        {
            storeExpression( &expr->cases.VideoSelect->arg[0], value, colwersion, instr );
            return;
        }

        case ptxBinaryExpression:
        case ptxUnaryExpression:
        case ptxIntConstantExpression:
        case ptxFloatConstantExpression:
        case ptxArrayIndexExpression:
        case ptxPredicateExpression:
        case ptxStructExpression:
        case ptxAddressOfExpression:
        case ptxAddressRefExpression:
        case ptxLabelReferenceExpression:
        case ptxParamListExpression:
        case ptxByteSelectExpression:
        {
            // These should all be illegal for stores
            error() << "Unimplemented expression kind in storeExpression: " << expr->kind << "\n";
            return;
        }
    }

    error() << "Illegal expression kind in storeExpression: " << expr->kind << "\n";
}

// Lookup the LLVM intrinsic for a special register.
llvm::Value* PTXFrontEnd::callSregIntrinsic( const std::string& sreg )
{
    if( sreg.empty() || sreg[0] != '%' )
    {
        error() << "Unhandled special register: " << sreg << '\n';
        return nullptr;
    }
    std::string name = "optix.lwvm.read.ptx.sreg." + sreg.substr( 1 );
    if( sreg == "%tid" || sreg == "%ntid" || sreg == "%ctaid" || sreg == "%nctaid" )
    {
        // A vector register with 4 32-bit elements
        llvm::Type*     vecTy = llvm::VectorType::get( llvm::Type::getInt32Ty( m_context ), 4 );
        llvm::Constant* fn    = m_module->getOrInsertFunction( name, vecTy );
        return m_builder.CreateCall( fn );
    }
    else
    {
        // A scalar register, including the ".x" variants
        unsigned int bits = 32;
        if( sreg == "%clock64" || sreg == "%gridid" || sreg == "%globaltimer" )
            bits           = 64;
        llvm::Constant* fn = m_module->getOrInsertFunction( name, llvm::IntegerType::get( m_context, bits ) );
        return m_builder.CreateCall( fn );
    }
}


/*************************************************************
*
* Type manipulation methods
*
*************************************************************/

unsigned int PTXFrontEnd::get_address_space( ptxPointerAttr attr )
{
    switch( attr )
    {
        case ptxPtrNone:
            return ADDRESS_SPACE_GENERIC;
        case ptxPtrGeneric:
            return ADDRESS_SPACE_GENERIC;
        case ptxPtrConst:
            return ADDRESS_SPACE_CONST;
        case ptxPtrGlobal:
            return ADDRESS_SPACE_GLOBAL;
        case ptxPtrLocal:
            return ADDRESS_SPACE_LOCAL;
        case ptxPtrShared:
            return ADDRESS_SPACE_SHARED;
        case ptxPtrTexref:
            return ADDRESS_SPACE_GENERIC;
        case ptxPtrSamplerref:
            return ADDRESS_SPACE_GENERIC;
        case ptxPtrSurfref:
            return ADDRESS_SPACE_GENERIC;
    }
    error() << "Illegal ptxPointerAttr: " << attr << '\n';
    return ADDRESS_SPACE_GENERIC;
}

llvm::Type* PTXFrontEnd::getLLVMType( ptxExpression expr, bool doubleres, ptxInstruction instr )
{
    ptxType t = ptxGetAddressArgBaseType( expr );
    llvm::Type* type = getLLVMType( t, doubleres );
    if( !type )
        return type;
    if( t != expr->type || ((expr->kind == ptxAddressOfExpression || expr->kind == ptxAddressRefExpression) && !type->isPointerTy()) )
    {
        RT_ASSERT_MSG( instr, "Missing instruction for pointer expression" );
        RT_ASSERT_MSG( instr->storage, "Missing storage info in instr for pointer expression: " + std::string( instr->tmplate->name ) );

        if( type->isArrayTy() )
            type = type->getArrayElementType();
        const int addressSpace = get_address_space( instr->storage->kind );
        return type->getPointerTo( addressSpace );
    }
    return type;
}

llvm::Type* PTXFrontEnd::getLLVMType( ptxType type, bool doubleres )
{
    PTXFE_ASSERT( type->kind == ptxIntType || !doubleres );
    uInt64 typeSize = ptxGetTypeSizeInBytes( type );
    switch( type->kind )
    {
        case ptxTypeB1:
        case ptxTypeB2:
        case ptxTypeB4:
        case ptxTypeB8:
        case ptxTypeB16:
        case ptxTypeB32:
        case ptxTypeB64:
        case ptxTypeB128:
        {
            if( typeSize != 1 && typeSize != 2 && typeSize != 4 && typeSize != 8 )
            {
                error() << "Illegal bit type size: " << typeSize * 8 << '\n';
                return nullptr;
            }
            return llvm::IntegerType::get( m_context, typeSize * 8 );
        }

        case ptxTypeF16:
        case ptxTypeBF16:
        case ptxTypeTF32:
        case ptxTypeF32:
        case ptxTypeF64:
        {
            if( typeSize != 2 && typeSize != 4 && typeSize != 8 )
            {
                error() << "Illegal bit type size: " << typeSize * 8 << '\n';
                return nullptr;
            }
            if( typeSize == 2 )
            {
                return llvm::Type::getHalfTy( m_context );
            }
            else if( typeSize == 4 )
            {
                return llvm::Type::getFloatTy( m_context );
            }
            else
            {
                return llvm::Type::getDoubleTy( m_context );
            }
        }

        case ptxTypeBF16x2:
        case ptxTypeF16x2:
        {
            if( typeSize != 4 )
            {
                error() << "Illegal bit type size: " << typeSize * 8 << '\n';
                return nullptr;
            }
            return llvm::IntegerType::get( m_context, typeSize * 8 );
        }

        case ptxTypeU2:
        case ptxTypeU4:
        case ptxTypeU8:
        case ptxTypeU16:
        case ptxTypeU32:
        case ptxTypeU64:
        case ptxTypeS2:
        case ptxTypeS4:
        case ptxTypeS8:
        case ptxTypeS16:
        case ptxTypeS32:
        case ptxTypeS64:
        {
            if( typeSize != 1 && typeSize != 2 && typeSize != 4 && typeSize != 8 )
            {
                error() << "Illegal integer type size: " << typeSize * 8 << '\n';
                return nullptr;
            }
            unsigned int wide = doubleres ? 2 : 1;
            return llvm::IntegerType::get( m_context, typeSize * 8 * wide );
        }

        case ptxTypePred:
        {
            return llvm::Type::getInt1Ty( m_context );
        }

        case ptxOpaqueType:
        {
            // must be represented as 64bit in LWVM
            return llvm::Type::getInt64Ty( m_context );
        }

        case ptxIncompleteArrayType:
        {
            // must be represented as an array of size 0 in LWVM
            llvm::Type* elementType = getLLVMType( type->cases.IncompleteArray.base, doubleres );
            return llvm::ArrayType::get( elementType, 0 );
        }

        case ptxVectorType:
        {
            llvm::Type* elementType = getLLVMType( type->cases.Vector.base, doubleres );
            return llvm::VectorType::get( elementType, type->cases.Vector.N );
        }

        case ptxArrayType:
        {
            llvm::Type* elementType = getLLVMType( type->cases.Array.base, doubleres );
            return llvm::ArrayType::get( elementType, type->cases.Array.N );
        }

        case ptxLabelType:
        {
            return llvm::Type::getVoidTy( m_context );
        }

        case ptxNOTYPE:
        case ptxMacroType:
        case ptxConditionCodeType:
        case ptxParamListType:
        case ptxTypeE4M3:
        case ptxTypeE5M2:
        case ptxTypeE4M3x2:
        case ptxTypeE5M2x2:
        {
            error() << "Unimplemented ptxInstructionType: " << type->kind << '\n';
            return nullptr;
        }
    }

    error() << "Illegal ptxInstructionType: " << type->kind << '\n';
    return nullptr;
}

bool PTXFrontEnd::followATypeDoesVectorize( ptxInstruction instr, uInt argIndex )
{
    ptxInstructionTemplate tmplate = instr->tmplate;
    RT_ASSERT( tmplate->argType[argIndex] == ptxFollowAType );
    // The only known instruction that includes an argument
    // that does follow the instruction type but does not want
    // to be vectorized is the "lod" argument for tex.level (argIndex == 3)
    return instr->tmplate->code != ptx_tex_level_Instr || argIndex != 3;
}

llvm::Type* PTXFrontEnd::getLLVMArgType( ptxInstruction instr, uInt arg )
{
    ptxInstructionTemplate tmplate = instr->tmplate;
    switch( tmplate->argType[arg] )
    {
        case ptxFollowAType:
        {
            uInt follow = tmplate->followMap[arg];
            if( follow >= tmplate->nrofInstrTypes )
            {
                error() << "Invalid instruction follow map entry: " << follow << '\n';
                return nullptr;
            }

            bool        doubleres = ptxHasDOUBLERES_Feature( tmplate->features ) && ( arg == 0 || arg == 3 );
            llvm::Type* type      = getLLVMType( instr->type[follow], doubleres );
            if( instr->modifiers.VECTOR && followATypeDoesVectorize( instr, arg ) )
            {
                uInt vecMult = ( 1 << instr->modifiers.VECTOR );
                if( arg != 0 && instr->modifiers.TEXTURE && tmplate->code != ptx_sust_b_Instr && tmplate->code != ptx_sust_p_Instr )
                {
                    // Handle texture instructions.
                    // Return value (arg == 0) follows the vector size, the coordinate vector follows the geometry.
                    // The only other VECTOR instructions that have the TEXTURE modifier are sust.b and sust.p. Those
                    // do follow the vector size regardless of the TEXTURE modifier, so they are explicitly excluded above.
                    vecMult = 1;
                    switch( instr->modifiers.TEXTURE )
                    {
                        case ptxNOTEXTURE_MOD:
                            return type;
                        case ptx1D_BUFFER_MOD:
                        case ptx1D_MOD:
                            return type;
                        case ptx2D_MOD:
                            vecMult = 2;
                            break;
                        case ptx3D_MOD:
                            vecMult = 4;
                            break;
                        case ptxA1D_MOD:
                        {
                            llvm::Type*              i32Ty = llvm::Type::getInt32Ty( m_context );
                            llvm::Type*              ctype = type;
                            std::vector<llvm::Type*> types = {i32Ty, type};
                            type                           = llvm::StructType::get( m_context, types );
                            return type;
                        }
                        case ptxA2D_MOD:
                        {
                            llvm::Type* i32Ty = llvm::Type::getInt32Ty( m_context );
                            llvm::Type* ctype = type;
                            if( tmplate->code == ptx_tld4_Instr )
                                ctype                      = llvm::Type::getFloatTy( m_context );
                            std::vector<llvm::Type*> types = {i32Ty, ctype, ctype, ctype};
                            type                           = llvm::StructType::get( m_context, types );
                            return type;
                        }
                        case ptxLWBE_MOD:
                            type    = llvm::Type::getFloatTy( m_context );
                            vecMult = 4;
                            break;
                        case ptxALWBE_MOD:
                        {
                            llvm::Type*              i32Ty   = llvm::Type::getInt32Ty( m_context );
                            llvm::Type*              floatTy = llvm::Type::getFloatTy( m_context );
                            std::vector<llvm::Type*> types   = {i32Ty, floatTy, floatTy, floatTy};
                            type                             = llvm::StructType::get( m_context, types );
                            return type;
                        }
                        case ptx2DMS_MOD:
                        case ptxA2DMS_MOD:
                            type    = llvm::Type::getInt32Ty( m_context );  // b32
                            vecMult = 4;
                            break;
                    }
                }
                return llvm::VectorType::get( type, vecMult );
            }
            else
                return type;
        }
        case ptxU32AType:
            return llvm::Type::getInt32Ty( m_context );

        case ptxU64AType:
            return llvm::Type::getInt64Ty( m_context );

        case ptxS32AType:
        {
            // The 's' operand type means "operand is type .s32 or vector of .s32" and the
            // vector size depends on the geometry. Only used in txq.level, sured.b/p, sust.b/p, suld.b.
            // The tld4 version with optional arguments 'e' and 'f' are not supported (which would hit this case, too).
            // Note that txq.level will always have modifiers->TEXTURE == ptxNOTEXTURE_MOD
            RT_ASSERT( ptx_txq_level_Instr == tmplate->code || ptx_suq_Instr == tmplate->code
                       || ptx_sured_b_Instr == tmplate->code || ptx_sured_p_Instr == tmplate->code
                       || ptx_sust_b_Instr == tmplate->code || ptx_sust_p_Instr == tmplate->code
                       || ptx_suld_b_Instr == tmplate->code );

            llvm::Type* type = llvm::Type::getInt32Ty( m_context );
            switch( instr->modifiers.TEXTURE )
            {
                case ptx1D_MOD:
                    return type;
                case ptx2D_MOD:
                    return llvm::VectorType::get( type, 2 );
                case ptx3D_MOD:
                    return llvm::VectorType::get( type, 4 );
                case ptxA1D_MOD:
                    if( tmplate->code != ptx_sust_b_Instr && tmplate->code != ptx_suld_b_Instr )
                    {
                        error() << "Texture modifier a1d is invalid for instruction: " << tmplate->name << "\n";
                        return nullptr;
                    }
                    return llvm::VectorType::get( type, 2 );
                case ptxA2D_MOD:
                    if( tmplate->code != ptx_sust_b_Instr && tmplate->code != ptx_suld_b_Instr )
                    {
                        error() << "Texture modifier a2d is invalid for instruction: " << tmplate->name << "\n";
                        return nullptr;
                    }
                    return llvm::VectorType::get( type, 4 );
                case ptxNOTEXTURE_MOD:
                case ptx1D_BUFFER_MOD:
                case ptxLWBE_MOD:
                case ptxALWBE_MOD:
                case ptx2DMS_MOD:
                case ptxA2DMS_MOD:
                    break;
            }
            return type;
        }

        case ptxF32AType:
        {
            // The 'f' operand type means "operand is type .f32 or vector of .f32"
            // and the vector size depends on the geometry. Only used in tex.grad.
            RT_ASSERT( ptx_tex_grad_Instr == tmplate->code );
            llvm::Type* type = llvm::Type::getFloatTy( m_context );
            switch( instr->modifiers.TEXTURE )
            {
                case ptx1D_MOD:
                case ptxA1D_MOD:
                    return llvm::VectorType::get( type, 1 );
                case ptx2D_MOD:
                case ptxA2D_MOD:
                    return llvm::VectorType::get( type, 2 );
                case ptx3D_MOD:
                case ptxLWBE_MOD:
                case ptxALWBE_MOD:
                    return llvm::VectorType::get( type, 4 );
                case ptxNOTEXTURE_MOD:
                case ptx1D_BUFFER_MOD:
                case ptx2DMS_MOD:
                case ptxA2DMS_MOD:
                    error() << "Invalid texture modifier for instruction: " << tmplate->name << "\n";
                    return nullptr;
            }
            return type;
        }

        case ptxF16x2AType:
            // TODO (OP-1114): We think we should use 32 bit int to store the 32 bit value and
            // just pass it along, but throw for now until we get tests to back up this decision.
            throw CompileError( RT_EXCEPTION_INFO, "Unhandled ptx packed half float type (f16x2) found" );
            return llvm::Type::getInt32Ty( m_context );

        case ptxScalarF32AType:
            return llvm::Type::getFloatTy( m_context );

        case ptxImageAType:
            // These are texture, surface, sampler arguments
            return llvm::Type::getInt64Ty( m_context );

        case ptxConstantIntAType:
            return llvm::Type::getInt32Ty( m_context );

        case ptxConstantFloatAType:
            throw CompileError( RT_EXCEPTION_INFO, "Unhandled constant float type found" );
            // TODO (OP-1115) figure out if ptxConstantFloatAType should return double or float
            return llvm::Type::getFloatTy( m_context );

        case ptxPredicateAType:
            return llvm::Type::getInt1Ty( m_context );

        case ptxMemoryAType:
        {
            // A pointer type.  The base type will be inherited from the other
            // arguments and the address space is in the instruction.
            std::string instname = tmplate->name;
            if( instname == "ld" || instname == "ld.global" || instname == "ldu" || instname == "st"
                || instname == "atom" || instname == "red" )
            {
                RT_ASSERT( tmplate->nrofInstrTypes == 1 );
                llvm::Type* type = getLLVMType( instr->type[0], false );
                if( instr->modifiers.VECTOR )
                    type = llvm::VectorType::get( type, ( 1 << instr->modifiers.VECTOR ) );

                const int addressSpace = get_address_space( instr->storage->kind );
                return type->getPointerTo( addressSpace );
            }
            else if( instname == "prefetch" || instname == "cctl" )
            {
                return llvm::Type::getInt8PtrTy( m_context );
            }
            else
            {
                error() << "Unimplemented memory argument for instruction: " << instname << '\n';
                return nullptr;
            }
        }

        case ptxSymbolAType:
        case ptxTargetAType:
        case ptxParamListAType:
        case ptxVoidAType:
        case ptxU16AType:
        case ptxB32AType:
        case ptxB64AType:
        case ptxPredicateVectorAType:
        case ptxLabelAType:
        {
            error() << "Unimplemented instruction argument type: " << tmplate->argType[arg] << '\n';
            return nullptr;
        }
    }

    error() << "Invalid instruction argument type: " << tmplate->argType[arg] << '\n';
    return nullptr;
}

/// Create a LLVM function type from a PTX declaration
llvm::FunctionType* PTXFrontEnd::getLLVMFunctionType( ptxSymbolTableEntryAux entry )
{
    // Create the function signature exactly as it oclwrs in PTX.  Note
    // that PTX creates all unsupported argument and return types (like
    // structs) as array of bytes. However, LWPTX does not support that,
    // so we must colwert arrays of bytes into vector or struct for the
    // return type.  Also colwert multiple return values into a struct.
    std::vector<llvm::Type*> fparams;
    typesFromVariableInfoList( fparams, entry->funcProtoAttrInfo->fparams );

    llvm::Type* ret_type;
    {
        std::vector<llvm::Type*> rparams;
        typesFromVariableInfoList( rparams, entry->funcProtoAttrInfo->rparams );
        if( rparams.empty() )
        {
            ret_type = llvm::Type::getVoidTy( m_context );
        }
        else if( rparams.size() == 1 )
        {
            ret_type = rparams[0];
            if( llvm::isa<llvm::ArrayType>( ret_type ) )
            {
                // If it is an array, wrap it in a struct
                llvm::ArrayRef<llvm::Type*> types( ret_type );
                ret_type = llvm::StructType::get( m_context, types );
            }
        }
        else
        {
            ret_type = llvm::StructType::get( m_context, rparams );
        }
    }

    // entry->isVariadic is no longer present so assume this can never be the case in RTX
    return llvm::FunctionType::get( ret_type, fparams, false );
}

void PTXFrontEnd::typesFromVariableInfoList( std::vector<llvm::Type*>& types, stdList_t list )
{
    for( stdList_t p = list; p != nullptr; p = p->tail )
    {
        ptxVariableInfo var  = static_cast<ptxVariableInfo>( p->head );
        llvm::Type*     type = getLLVMType( var->symbol->type, false );
        types.push_back( type );
    }
}

/*************************************************************
*
* PTX and LLVM utility functions
*
*************************************************************/

// Add a LWVM annotation to the given value.
void PTXFrontEnd::addLWVMAnnotation( llvm::Value* val, char const* kind, unsigned annotation )
{
    llvm::NamedMDNode* lwvmannotate = m_module->getOrInsertNamedMetadata( "lwvm.annotations" );
    llvm::MDString*    str          = llvm::MDString::get( m_context, kind );
    llvm::Value*       av           = llvm::ConstantInt::get( llvm::Type::getInt32Ty( m_context ), annotation );

    MetadataValueTy*             mVal   = val ? UseValueAsMd( val ) : nullptr;
    std::vector<llvm::Metadata*> values = { mVal, str, UseValueAsMd( av ) };
    llvm::MDNode*                mdNode = llvm::MDNode::get( m_context, values );

    lwvmannotate->addOperand( mdNode );
    m_lwvmAnnotations.insert( std::make_pair( val, mdNode ) );
}

// Translate a ptx declaration scope into an LLVM linkage type.
static llvm::GlobalValue::LinkageTypes getLLVMLinkage( ptxSymbolTableEntry entry )
{
    if( entry->aux && entry->aux->isEntry )
        return llvm::GlobalValue::ExternalLinkage;

    switch( entry->scope )
    {
        case ptxLocalScope:
            return llvm::GlobalValue::InternalLinkage;
        case ptxStaticScope:
            return llvm::GlobalValue::InternalLinkage;
        case ptxGlobalScope:
            return llvm::GlobalValue::ExternalLinkage;
        case ptxExternalScope:
            return llvm::GlobalValue::ExternalLinkage;
        case ptxWeakScope:
            return llvm::GlobalValue::WeakAnyLinkage;
        case ptxCommonScope:
            return llvm::GlobalValue::CommonLinkage;
    }
    PTXFE_ASSERT( !"Unsupported ptx declaration scope" );
    return llvm::GlobalValue::ExternalLinkage;
}

static bool blockIsUnreachable( llvm::BasicBlock* bb )
{
    return llvm::pred_begin( bb ) == llvm::pred_end( bb ) && bb != &bb->getParent()->getEntryBlock();
}

/// Get the first code location of a block statement.
///
/// \param block  the block
/// \param ptxState the ptxparser state gives us access to mapping information
static ptxCodeLocation get_first_code_location( ptxSymbolTable block, ptxParsingState ptxState )
{
    if( block->numStmts <= 0 )
        return nullptr;

    stdList_t list = block->statements;
    while( list != nullptr )
    {
        stdList_t tail = list->tail;

        ptxStatement ptx_stmt = static_cast<ptxStatement>( list->head );
        switch( ptx_stmt->kind )
        {
            case ptxInstructionStatement: {
                // this gives us the location inside the PTX
                ptxCodeLocation loc = ptx_stmt->cases.Instruction.instruction->loc;
                // from loc->lwrLoc.lineNo we can retrieve the location inside the original source
                if( loc )
                    loc = static_cast<ptxCodeLocation>(
                        rangemapApply( ptxState->ptxToSourceLine.instructionMap, (rangemapRange_t)loc->lwrLoc.lineNo ) );
                // take potential inlinedAt of the first instruction into account
                return FrontEndDebugInfo::getLocalCodeLocation( loc );
            }
            case ptxPragmaStatement:
                // no location
                break;
            default:
                PTXFE_ASSERT( !"Unsupported statement kind" );  // TODO: turn this into an if statement
                break;
        }
        list = tail;
    }
    return nullptr;
}

static bool hasResult( ptxInstructionTemplate tmplate )
{
    // Video instructions are special.  Their destination register is also read.
    return ptxHasRESULT_Feature( tmplate->features ) || isVideoInstr( tmplate );
}

static bool hasResultp( ptxInstruction instr )
{
    // If a tex instruction does not use the result predicate, treat it as if
    // its template does not have the RESULTP feature.
    if( ptxIsTexInstr( instr->tmplate->code ) && !isTEXREF( instr->arguments[1]->type ) && !instr->predicateOutput )
    {
        return false;
    }

    ptxInstructionTemplate tmplate = instr->tmplate;
    if( tmplate->code == ptx_match_Instr && instr->modifiers.SYNC )
    {
        // The match instruction is special: Only the "all" variant has a predicate output
        // while the "any" variant (instr->modifiers.VOTE == 2) does not.
        if( instr->modifiers.VOTE == 2 )
            return false;
    }
    return ptxHasRESULTP_Feature( tmplate->features );
}

static bool hasCC( ptxInstructionTemplate tmplate )
{
    ptxInstructionCode c = ( (ptxInstructionCode)tmplate->code );
    return ptxHasCC_Feature( tmplate->features ) || c == ptx_addc_Instr || c == ptx_subc_Instr || c == ptx_madc_lo_Instr
           || c == ptx_madc_hi_Instr;
}

static bool hasBAR( ptxInstructionTemplate tmplate )
{
    return ptxHasBAR_Feature( tmplate->features );
}

static bool hasSREG( ptxInstructionTemplate tmplate )
{
    return ptxHasSREGARG_Feature( tmplate->features );
}

static bool hasMemArg( ptxInstructionTemplate tmplate )
{
    bool hasMemArg = false;
    for( unsigned int i = 0; i < tmplate->nrofArguments; ++i )
    {
        if( tmplate->argType[i] == ptxMemoryAType || tmplate->argType[i] == ptxImageAType )
        {
            hasMemArg = true;
            break;
        }
    }
    return hasMemArg;
}

static bool isReadNone( ptxInstructionTemplate tmplate )
{
    if( hasMemArg( tmplate ) || hasCC( tmplate ) || hasBAR( tmplate ) || hasSREG( tmplate ) )
        return false;

    // White list
    switch( (ptxInstructionCode)tmplate->code )
    {
        case ptx_pmevent_Instr:
        case ptx_pmevent_mask_Instr:
        case ptx_vote_Instr:
        case ptx_match_Instr:
        case ptx_activemask_Instr:
            return false;
        default:
            return true;
    }

    return true;
}

static bool isVideoInstr( ptxInstructionTemplate tmplate )
{
    switch( (ptxInstructionCode)tmplate->code )
    {
        case ptx_vadd_Instr:
        case ptx_vsub_Instr:
        case ptx_vabsdiff_Instr:
        case ptx_vmin_Instr:
        case ptx_vmax_Instr:
        case ptx_vshl_Instr:
        case ptx_vshr_Instr:
        case ptx_vmad_Instr:
        case ptx_vset_Instr:
        case ptx_vadd2_Instr:
        case ptx_vsub2_Instr:
        case ptx_vavrg2_Instr:
        case ptx_vabsdiff2_Instr:
        case ptx_vmin2_Instr:
        case ptx_vmax2_Instr:
        case ptx_vset2_Instr:
        case ptx_vadd4_Instr:
        case ptx_vsub4_Instr:
        case ptx_vavrg4_Instr:
        case ptx_vabsdiff4_Instr:
        case ptx_vmin4_Instr:
        case ptx_vmax4_Instr:
        case ptx_vset4_Instr:
            return true;
        default:
            return false;
    }
}

static bool isSignedType( ptxType type )
{
    switch( type->kind )
    {
        case ptxVectorType:
            return isSignedType( type->cases.Vector.base );

        case ptxTypeS2:
        case ptxTypeS4:
        case ptxTypeS8:
        case ptxTypeS16:
        case ptxTypeS32:
        case ptxTypeS64:
            return true;

        case ptxTypeB1:
        case ptxTypeB2:
        case ptxTypeB4:
        case ptxTypeB8:
        case ptxTypeB16:
        case ptxTypeB32:
        case ptxTypeB64:
        case ptxTypeB128:

        case ptxTypeU2:
        case ptxTypeU4:
        case ptxTypeU8:
        case ptxTypeU16:
        case ptxTypeU32:
        case ptxTypeU64:

        case ptxTypeF16:
        case ptxTypeBF16:
        case ptxTypeBF16x2:
        case ptxTypeTF32:
        case ptxTypeF16x2:
        case ptxTypeF32:
        case ptxTypeF64:

        case ptxLabelType:
        case ptxMacroType:
        case ptxTypePred:
        case ptxConditionCodeType:
        case ptxOpaqueType:
        case ptxIncompleteArrayType:

        case ptxParamListType:
        case ptxArrayType:

        case ptxNOTYPE:
        case ptxTypeE4M3:
        case ptxTypeE5M2:
        case ptxTypeE4M3x2:
        case ptxTypeE5M2x2:
            return false;
    }
    return false;
}

static bool isSignedType( ptxInstruction instr, uInt arg )
{
    ptxInstructionTemplate tmplate = instr->tmplate;
    switch( tmplate->argType[arg] )
    {
        case ptxFollowAType:
        {
            uInt follow = tmplate->followMap[arg];
            PTXFE_ASSERT( follow < tmplate->nrofInstrTypes );
            return isSignedType( instr->type[follow] );
        }
        break;
        case ptxS32AType:
            return true;
        default:
            return false;
    }
}

static std::string getIntrinsicName( ptxInstruction instr, ptxParsingState parseState )
{
    std::string result;
    result.reserve( 128 );

    if( !isVideoInstr( instr->tmplate ) )
    {
        result.append( "optix.ptx." );

        // If a tex instruction does not use the result predicate, use the
        // "nonsparse." version of the OptiX LLVM wrapper function.
        if( ptxIsTexInstr( instr->tmplate->code ) && !isTEXREF( instr->arguments[1]->type ) && !instr->predicateOutput )
        {
            result.append( "nonsparse." );
        }

        printPtxOpCode( result, instr, parseState );
    }
    else
    {
        // Generate a placeholder string for a video instruction which encodes masks, specifiers and MAD
        result.append( "optix.ptx.video." );
        printPtxOpCode( result, instr, parseState );
        result.append( ".selsec" );
        printVideoSelectSpecifier( result, instr );
    }
    return result;
}


/*************************************************************
*
* Scopes and management thereof
*
*************************************************************/

// Creates a new variable scope and push it.
void PTXFrontEnd::push_scope( ptxCodeLocation loc )
{
    m_lwrrentScope = new Scope( m_lwrrentScope );
    if( m_debug_info != nullptr )
    {
        if( loc && ( loc->lwrLoc.lineNo > 0 ) && ( loc->lwrLoc.linePos == 0 ) )
        {
            loc->lwrLoc.linePos = 1;
        }
        m_debug_info->push_block_scope( loc, m_builder );
    }
}

// Pop a variable scope, running destructors if necessary.
void PTXFrontEnd::pop_scope()
{
    Scope* lose    = m_lwrrentScope;
    m_lwrrentScope = const_cast<Scope*>( m_lwrrentScope->parent );
    delete lose;
    if( m_debug_info != nullptr )
        m_debug_info->pop_block_scope();
}

// Add debug info for a PTX variable.
void PTXFrontEnd::add_var_debug_info( ptxSymbolTableEntry var, llvm::Value* adr )
{
    if( m_debug_info != nullptr )
        m_debug_info->add_variable( var, adr, m_builder.GetInsertBlock() );
}

// Mark a change of the value of the given variable.
void PTXFrontEnd::mark_var_assigned( ptxSymbol var, llvm::Value* new_val )
{
    if( m_debug_info != nullptr )
        m_debug_info->mark_local_value_change( var, new_val, m_builder.GetInsertBlock() );
}

// Get the address of a symbol.
llvm::Value* PTXFrontEnd::get_symbol_address( ptxSymbol sym )
{
    llvm::Value* adr = m_lwrrentScope->lookupAdr( sym );
    if( adr != nullptr )
        return adr;

    error() << "get symbol failed: " << sym->unMangledName << '\n';

    // create an undef
    llvm::Type* tp = getLLVMType( sym->type, false );
    return llvm::UndefValue::get( tp->getPointerTo( ADDRESS_SPACE_UNKNOWN ) );
}

// Store the address of a symbol in the current scope.
void PTXFrontEnd::set_symbol_address( ptxSymbol sym, llvm::Value* adr )
{
    if( m_lwrrentScope == nullptr )
    {
        error() << "No scope in set_symbol_address";
        return;
    }

    if( m_lwrrentScope->symbol_adr_map.find( sym ) != m_lwrrentScope->symbol_adr_map.end() )
        error() << "Duplicate symbol in set_symbol_address: " << sym->unMangledName << '\n';

    m_lwrrentScope->symbol_adr_map[sym] = adr;
}

llvm::BasicBlock* PTXFrontEnd::Scope::findLabel( ptxSymbol targetLabel ) const
{
    Scope::Label2Idx_map::const_iterator it = label2Idx.find( targetLabel );
    if( it != label2Idx.end() )
    {
        Idx2Block_map::const_iterator bit = idx2block.find( it->second );
        if( bit == idx2block.end() )
            return nullptr;
        else
            return bit->second;
    }

    if( parent )
        return parent->findLabel( targetLabel );
    else
        return nullptr;
}

llvm::Value* PTXFrontEnd::Scope::lookupAdr( ptxSymbol sym ) const
{
    Scope const* scope = this;
    do
    {
        Scope::Symbol_map::const_iterator it = scope->symbol_adr_map.find( sym );
        if( it != scope->symbol_adr_map.end() )
        {
            return it->second;
        }
        scope = scope->parent;
    } while( scope );

    return nullptr;
}

const std::vector<ptxSymbol>* PTXFrontEnd::Scope::lookupBranchtargets(ptxSymbol sym) const
{
    Scope::Label2Branchtargets_map::const_iterator it = label2Branchtargets.find(sym);
    if (it != label2Branchtargets.end())
        return &it->second;
    return nullptr;
}

static const llvm::MDNode* getNamedLwvmMetdataNode( llvm::Module* module, const std::string& name )
{
    llvm::NamedMDNode* lwvmMd = module->getNamedMetadata( "lwvm.annotations" );
    if( !lwvmMd )
        return nullptr;
    for( unsigned int i = 0, e = lwvmMd->getNumOperands(); i != e; ++i )
    {
        const llvm::MDNode* elem = lwvmMd->getOperand( i );
        if( !elem || elem->getNumOperands() < 2 )
            continue;
        const llvm::MDString* stypeMD = llvm::dyn_cast<llvm::MDString>( elem->getOperand( 1 ) );
        if( !stypeMD )
            continue;
        const llvm::StringRef mdName = stypeMD->getString();
        if( mdName == name )
            return elem;
    }
    return nullptr;
}

static void markAddAndMulFast( llvm::Module* module )
{
    for( llvm::Function& F : *module )
    {

        for( llvm::inst_iterator I = inst_begin( &F ), IE = inst_end( &F ); I != IE; ++I )
        {
            llvm::BinaryOperator* binOp = llvm::dyn_cast<llvm::BinaryOperator>( &*I );
            if( !binOp )
                continue;
            if( binOp->getOpcode() == llvm::Instruction::FAdd || binOp->getOpcode() == llvm::Instruction::FMul )
            {
                binOp->setFast( true );
            }
        }
    }
}

bool PTXFrontEnd::processInputLWVM( const std::string& declString )
{
    llvm::NamedMDNode* lwvmannotate = m_module->getNamedMetadata( "lwvm.annotations" );
    if( lwvmannotate )
    {
        // Check if the incoming module has targetArch metadata set.
        // If not we use sm_50 as the fallback.
        const llvm::MDNode* archNode = getNamedLwvmMetdataNode( m_module, "targetArch" );
        if( archNode )
        {
            m_targetArch = corelib::getValueFromMetadataSigned( archNode, 0 );
        }

        // If the module had the fmad option set, we need to mark muls and adds
        // as fast so they can be fused to fma's.
        const llvm::MDNode* fmadNode = getNamedLwvmMetdataNode( m_module, "fmad" );
        if( fmadNode )
        {
            if( corelib::getValueFromMetadataSigned( fmadNode, 0 ) != 0 )
                markAddAndMulFast( m_module );
        }
    }
    // remove llvm.globals to avoid stale references to the shaders inside the module
    llvm::GlobalVariable* llvmUsed = m_module->getGlobalVariable( "llvm.used" );
    if( llvmUsed )
    {
        llvmUsed->eraseFromParent();
    }

    // Initialize the parser to parse the declString
    advanceState( Initial, Parsing );

    // We have to add function declarations for the optix functions. Those are stored in the PTX declaration string.
    parsePTX( declString, {"", 0} );

    advanceState( Parsing, Parsed );

    llvm::Value* av = llvm::ConstantInt::get( llvm::Type::getInt32Ty( m_context ), m_targetArch );
    addLWVMAnnotation( av, "targetArch" );

    return m_state != Error;
}
