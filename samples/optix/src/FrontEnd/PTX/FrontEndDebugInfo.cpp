// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL
// LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
// CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
// LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
// INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
// INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGES

#include <FrontEnd/Canonical/LineInfo.h>
#include <FrontEnd/PTX/FrontEndDebugInfo.h>
#include <FrontEnd/PTX/PTXStitch/ptxparse/ptxConstructors.h>
#include <FrontEnd/PTX/PTXStitch/ptxparse/ptxMacroUtils.h>
#include <prodlib/exceptions/Assert.h>

#include <llvm/IR/Instruction.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>

// Temporary include until we've fully moved over to lwvm70
#include <lwvm/Support/APIUpgradeUtilities.h>

#include <algorithm>
#include <stdexcept>
#include <string>


// Split a name into the file and directory parts
static void parse_dirname_and_basename( const std::string& filename, std::string& dirname, std::string& basename )
{
    size_t pos = filename.rfind( '/' );
    if( pos != std::string::npos )
    {
        dirname  = filename.substr( 0, pos );
        basename = filename.substr( pos + 1 );
    }
    else
    {
        pos = filename.rfind( '\\' );
        if( pos != std::string::npos )
        {
            dirname  = filename.substr( 0, pos );
            basename = filename.substr( pos + 1 );
        }
        else
        {
            basename = filename;
            dirname.clear();
        }
    }
}

static void createCompileUnit( const std::string& fname, bool onlyLineinfo, llvm::DIBuilder& diBuilder )
{
    std::string filename( fname.empty() ? "unknown" : fname );
    std::string basename, dirname;
    parse_dirname_and_basename( filename, dirname, basename );

    llvm::DICompileUnit::DebugEmissionKind emissionKind =
        onlyLineinfo ? llvm::DICompileUnit::DebugDirectivesOnly : llvm::DICompileUnit::FullDebug;
    llvm::DIFile* di_file = diBuilder.createFile( basename, dirname );
    diBuilder.createCompileUnit(
        /*Lang=*/llvm::dwarf::DW_LANG_C99,
        /*File=*/di_file,
        /*Producer=*/"LWPU OptiX compiler",
        /*isOptimized=*/onlyLineinfo,
        /*Flags=*/"",  // command line args
        /*RV=*/0,      // run time version
        /*SplitName=*/"",
        /*Kind=*/emissionKind );
}

#if ( DEBUG || DEVELOP )
static void assertOnlyOneCompileUnitPresent( llvm::Module* module )
{
    llvm::DebugInfoFinder dif;
    dif.processModule( *module );
    // There can only be one compile unit. Make sure that the module does not yet contain one.
    RT_ASSERT( dif.compile_unit_count() == 0 );
}
#endif

void FrontEndDebugInfo::createCompileUnitForModule( const std::string& fname, bool onlyLineinfo, llvm::Module* module )
{
#if ( DEBUG || DEVELOP )
    assertOnlyOneCompileUnitPresent( module );
#endif
    llvm::DIBuilder diBuilder( *module );
    createCompileUnit( fname, onlyLineinfo, diBuilder );
}

// Constructor.
FrontEndDebugInfo::FrontEndDebugInfo( const std::string& fname, llvm::Module* module, bool only_lineinfo, ptxParsingState ptxState )
    : m_di_builder( *module )
    , m_only_lineinfo( only_lineinfo )
    , m_ptxState( ptxState )
    , m_module( module )
    , m_loc_function( nullptr )
{
#if ( DEBUG || DEVELOP )
    assertOnlyOneCompileUnitPresent( module );
#endif
    createCompileUnit(fname, m_only_lineinfo, m_di_builder);
    m_dwarfInfo = lwInitDwarf( 1 );
}

// Destructor.
FrontEndDebugInfo::~FrontEndDebugInfo()
{
    m_di_builder.finalize();
    stdFREE( m_dwarfInfo );
}

// Retrieve the function name corresponding to the given label from the DWARF section .debug_str.
std::string FrontEndDebugInfo::getFunctionNameFromDwarfSection( const char* label )
{
    if( !label )
    {
        warn() << "ERROR Cannot retrieve function name from DWARF data when no label was given, debug info might be "
                  "incomplete\n";
        RT_ASSERT_FAIL_MSG( "label name given" );
        return std::string{};
    }
    if( m_debug_str_section_data.empty() )
    {
        warn() << "ERROR Cannot retrieve function name from DWARF data when no data was stored, debug info might be "
                  "incomplete\n";
        RT_ASSERT_FAIL_MSG( "no DWARF section .debug_str available" );
        return std::string{};
    }

    // following ptxparser's types and functionality here to retrieve string pointing to actual location of wanted name
    Pointer dom = (Pointer)label;
    Pointer ran = mapApply( m_ptxState->internalDwarfLabel, dom );
    // this string has the layout of something like '<section>+<offset>', eg '.debug_str+0'
    String res = (String)ran;
    if( !res )
    {
        warn() << "ERROR Cannot retrieve function name from DWARF data section .debug_str, debug info might be incomplete\n";
        return std::string{};
    }
    // as we should be dealing with nothing else
    if( std::string( res, strlen(".debug_str") ) != ".debug_str" )
    {
        warn() << "ERROR Facing section name different than .debug_str, debug info might be incomplete\n";
        return std::string{};
    }
    // let res point right at the offset value, ie behind ".debug_str+"
    std::advance( res, strlen(".debug_str+") );

    std::string retValue;
    try
    {
        int offset = std::stoi( std::string( res ) );
        // as all stored string values are terminated by a '\0' we can generate a string right out of it
        retValue = std::string( &m_debug_str_section_data[offset] );
    }
    catch( std::logic_error& ex )
    {
        warn() << "ERROR Cannot retrieve function name from DWARF data, debug info might be incomplete\n";
    }
    return retValue;
}

// Retrieve data from DWARF section .debug_str in conselwtive memory - strings are separated by '\0'
static std::vector<char> retrieveDebugStringSectionData( ptxParsingState                      state,
                                                         ptxDwarfSection                      debugMiscDwarfSec,
                                                         lwDwarfInfo                          dwarfInfo,
                                                         std::function<llvm::raw_ostream&()>& warn )
{
    // while the original DWARF code in lwdwarf.c is allocating 1000 chars for stringVect, we behave smarter here
    Int32             size       = debugMiscDwarfSec->size;
    stdVector_t       stringVect = vectorCreate( size );
    char*             buffer     = ptxDwarfCreateByteStream( debugMiscDwarfSec, stringVect, state->internalDwarfLabel );
    std::vector<char> buf( buffer, buffer + size );

    // cleaning up resources
    vectorDelete( stringVect );
    stdFREE( buffer );

    return buf;
}

// Add a source filename
void FrontEndDebugInfo::add_file( DebugIndexedFile filerec )
{
    std::string dirname, basename;
    parse_dirname_and_basename( filerec.name, dirname, basename );
    llvm::DIFile* di_file = m_di_builder.createFile( basename, dirname );
    m_di_files.insert( std::make_pair( filerec.index, di_file ) );
}

void FrontEndDebugInfo::start_function( ptxSymbolTableEntry ptx_function, llvm::Function* llvm_func )
{
    m_llvm_func    = llvm_func;
    m_ptx_function = ptx_function;
}

void FrontEndDebugInfo::initializeFunctionDebugInfo()
{
    if( m_di_function ) // already initialized
        return;

    RT_ASSERT( m_llvm_func );
    RT_ASSERT( m_ptx_function );
    RT_ASSERT( !m_lwrrent_di_file );
    unsigned start_line = 1;
    if( ptxCodeLocation loc = m_ptx_function->location )
    {
        m_lwrrent_di_file = retrieveFileByIndex( loc->lwrLoc.fileIndex );
        start_line        = loc->lwrLoc.lineNo;
    }

    if( !m_lwrrent_di_file )
    {
        // try to retrieve LWCA reference first, otherwise fall back onto generated PTX
        unsigned int lastLineNo{};
        if( m_ptx_function->aux && m_ptx_function->aux->mbodyPos )
            // value of the last line inside the PTX source
            lastLineNo = m_ptx_function->aux->mbodyPos->lineNo;
        // retrieve source code location from range map
        if( ptxCodeLocation loc =
                static_cast<ptxCodeLocation>( rangemapApply( m_ptxState->ptxToSourceLine.functionPtxLineRangeMap, lastLineNo ) ) )
        {
            m_loc_function = loc;
            start_line = loc->lwrLoc.lineNo;
            auto iter  = m_di_files.find( loc->lwrLoc.fileIndex );
            if( iter != m_di_files.end() )
                m_lwrrent_di_file = iter->second;
        }
    }
    // final fallback
    if( !m_lwrrent_di_file )
        m_lwrrent_di_file = getDIFile( m_ptx_function->location );

    llvm::DISubroutineType* di_func_type = get_debug_info_type( m_lwrrent_di_file, m_ptx_function );
    // TODO(Kincaid): Check that this works properly
    llvm::DISubprogram* subprogram = m_di_function = m_di_builder.createFunction(
        /*Scope=*/m_lwrrent_di_file,
        /*Name=*/m_ptx_function->symbol->name,
        /*LinkageName=*/m_ptx_function->symbol->name,
        /*File=*/m_lwrrent_di_file,
        /*LineNo=*/start_line,
        /*Ty=*/di_func_type,
        /*isLocalToUnit=*/true,
        /*isDefinition=*/true,
        /*ScopeLine (assume local)=*/0U,
        /*Flags=*/llvm::DINode::FlagPrototyped );
    m_llvm_func->setSubprogram( subprogram );
}

void FrontEndDebugInfo::finished_function()
{
    m_lwrrent_di_file = nullptr;
    m_di_function     = nullptr;
    m_loc_function    = nullptr;
    m_llvm_func       = nullptr;
    m_ptx_function    = nullptr;
}

// Add debug info for a PTX variable.
void FrontEndDebugInfo::add_variable( ptxSymbolTableEntry ptx_var, llvm::Value* var_adr, llvm::BasicBlock* lwrr_block )
{
    if( m_only_lineinfo )
        return;
    ptxCodeLocation loc = ptx_var->location;
    if( !loc )
        return;

    initializeFunctionDebugInfo();

    bool isGlobal = false;
    switch( ptx_var->storage.kind )
    {
        case ptxRegStorage:
        case ptxParamStorage:
        case ptxLocalStorage:
            // local LLVM entity
            break;

        case ptxConstStorage:
        case ptxGlobalStorage:
        case ptxSharedStorage:
        case ptxSurfStorage:
        case ptxTexStorage:
        case ptxTexSamplerStorage:
            // LLVM globaVariable
            isGlobal = true;
            break;

        case ptxCodeStorage:
        case ptxSregStorage:
        case ptxIParamStorage:
        case ptxOParamStorage:
        case ptxFrameStorage:
        default:
            // not supported yet
            RT_ASSERT( !"storage unsupported" );
            return;
    }

    unsigned start_line = loc->lwrLoc.lineNo;
    llvm::DIFile* di_file = getDIFile( loc );

    if( isGlobal )
    {
        llvm::DIType*                     diType  = get_debug_info_type( di_file, ptx_var->symbol->type );
        llvm::DIGlobalVariableExpression* varExpr = m_di_builder.createGlobalVariableExpression(
            /*Context=*/di_file,
            /*Name=*/ptx_var->symbol->name,
            // TODO(Kincaid): What should we put for linkage?
            /*LinkageName=*/"unknown",
            /*File=*/di_file,
            /*LineNo=*/start_line,
            /*DIType=*/diType,
            /*isLocalToUnit=*/ptx_var->scope != ptxExternalScope );
        llvm::DIGlobalVariable* var = varExpr->getVariable();
        (void)var;
    }
    else
    {
        llvm::DIScope* scope  = get_debug_info_scope();
        llvm::DIType*  diType = get_debug_info_type( scope, ptx_var->symbol->type );

        llvm::DILocalVariable* var = m_di_builder.createAutoVariable( scope, ptx_var->symbol->name, di_file, start_line, diType,
                                                                      true,  // preserve even in optimized builds
                                                                      /*Flags=*/llvm::DINode::FlagZero );

        // remember it for later
        RT_ASSERT( m_dbg_var_map.find( ptx_var->symbol ) == m_dbg_var_map.end() );
        m_dbg_var_map[ptx_var->symbol] = var;

        // TODO(Kincaid): Are these the correct arguments?
        m_di_builder.insertDeclare( var_adr, var, nullptr, nullptr, lwrr_block );
    }
}

// Mark a local variable value change.
void FrontEndDebugInfo::mark_local_value_change( ptxSymbol var, llvm::Value* new_val, llvm::BasicBlock* lwrr_block )
{
    if( m_only_lineinfo )
        return;
    initializeFunctionDebugInfo();

    Dbg_var_map::const_iterator it = m_dbg_var_map.find( var );
    if( it != m_dbg_var_map.end() )
    {
        llvm::DILocalVariable* di_var = it->second;
        m_di_builder.insertDbgValueIntrinsic( new_val, di_var, nullptr, nullptr, lwrr_block );
    }
}

// Creates a new variable scope and push it.
void FrontEndDebugInfo::push_block_scope( ptxCodeLocation loc, corelib::CoreIRBuilder& builder )
{
    llvm::DIScope* parentScope = nullptr;

    if( !m_dilb_stack.empty() )
        parentScope = m_dilb_stack.top();
    else
        parentScope = m_di_function;

    unsigned start_line   = 1;
    unsigned start_column = 1;

    llvm::DIFile* di_file = getDIFile( loc );
    if( loc )
    {
        // ensure that set_lwrr_loc()'s inlineAt handling is triggered only by PTXFrontEnd::translateInstruction()
        RT_ASSERT( loc->inlineAtLoc == nullptr );
        set_lwrr_loc( loc, builder );

        start_line   = loc->lwrLoc.lineNo;
        start_column = loc->lwrLoc.linePos;
    }

    if( parentScope )
    {
        llvm::DILexicalBlock* lexicalBlock = m_di_builder.createLexicalBlock( parentScope, di_file, start_line, start_column );
        m_dilb_stack.push( lexicalBlock );
    }
}

// Pop a variable scope, running destructors if necessary.
void FrontEndDebugInfo::pop_block_scope()
{
    // it's possible for the scope stack to be empty here for global scope (see push_block_scope)
    if( !m_dilb_stack.empty() )
        m_dilb_stack.pop();
}

// Get the current LLVM debug info scope.
llvm::DIScope* FrontEndDebugInfo::get_debug_info_scope()
{
    return m_dilb_stack.top();
}

/// Returns a DebugLoc for a new DILocation which is a clone of \p OrigDL
/// inlined at \p InlinedAt. \p IANodes is an inlined-at cache.
static llvm::DebugLoc inlineDebugLoc( llvm::DebugLoc                                      OrigDL,
                                      llvm::DILocation*                                   InlinedAt,
                                      llvm::LLVMContext&                                  Ctx,
                                      llvm::DenseMap<const llvm::MDNode*, llvm::MDNode*>& IANodes )
{
    auto IA = llvm::DebugLoc::appendInlinedAt( OrigDL, InlinedAt, Ctx, IANodes );
    return llvm::DebugLoc::get( OrigDL.getLine(), OrigDL.getCol(), OrigDL.getScope(), IA );
}

// Set the current code location to an IRBuilder.
// Regarding inline handling we assume the following: inlinedAt data originates from instructions only.
// Ie, whenever loc->inlineAtLoc, set_lwrr_loc() is being called from PTXFrontEnd::translateInstruction().
// This is getting tested by an assertion in FrontEndDebugInfo::push_block_scope(), where set_lwrr_loc()
// gets called as well.
void FrontEndDebugInfo::set_lwrr_loc( ptxCodeLocation loc, corelib::CoreIRBuilder& builder )
{
    if( loc == nullptr )
        return;
    initializeFunctionDebugInfo();
    
    // loc has the ptx line number. Look up the source code line number.
    unsigned linePos = loc->lwrLoc.linePos;
    llvm::DIFile* di_file = getDIFile( loc );

    // LWVM handles any debug location with line position 0 as invalid
    // (since line numbers always start at 1 in any software), so avoid that:
    if( linePos == 0 )
        linePos = 1;

    llvm::DIScope* parentScope = nullptr;

    if( !m_dilb_stack.empty() )
        parentScope = m_dilb_stack.top();
    else
        parentScope = m_di_function;

    // for inlined instructions we have to take the scope of the callee
    if( loc->inlineAtLoc )
    {
        String calleeName = loc->functionName;
        std::string name = getFunctionNameFromDwarfSection( calleeName );
        llvm::Function* func = m_module->getFunction( name );
        if( func && func->getSubprogram() )
        {
            parentScope = func->getSubprogram();
        }
        else
        {
            m_missingInlinedFunctions.insert( name );
        }
    }
    llvm::DILexicalBlockFile* lbf  = m_di_builder.createLexicalBlockFile( parentScope, di_file );
    llvm::DebugLoc            dloc = llvm::DebugLoc::get( loc->lwrLoc.lineNo, linePos, lbf );
    // adding existing inlineAtLoc's relwrsively to it
    while( ( loc = loc->inlineAtLoc ) != nullptr )
    {
        ptxCodeLocation inlinedDLoc = loc;
        llvm::DIScope*  localScope{};
        String          calleeName = inlinedDLoc->functionName;
        if( calleeName )
        {
            std::string     name = getFunctionNameFromDwarfSection( calleeName );
            llvm::Function* func = m_module->getFunction( name );
            if( func && func->getSubprogram() )
            {
                localScope = func->getSubprogram();
            }
            else
            {
                m_missingInlinedFunctions.insert( name );
                // fall back to something - here the local function scope
                // Set the localScope to be the current function.
                localScope = m_di_function;
                // Don't continue up the "inlinedAt-stack". This would produce debug locations where
                // the scope and the inlinedAt-field are in the same program. Inlining a function into
                // itself would most likely confuse the debugger.
                while( loc->inlineAtLoc )
                    loc = loc->inlineAtLoc;
            }
        }
        else
            // the very last location has no function name and is the actual function
            localScope = m_di_function;
        llvm::DILocation* inlinedAtNode = llvm::DILocation::getDistinct( m_module->getContext(), inlinedDLoc->lwrLoc.lineNo,
                                                                            inlinedDLoc->lwrLoc.linePos, localScope, nullptr );
        // Returns a DebugLoc for a new DILocation which is a clone of OrigDL inlined at InlinedAt
        llvm::DenseMap<const llvm::MDNode*, llvm::MDNode*> iANodes;
        dloc = inlineDebugLoc( dloc, inlinedAtNode, m_module->getContext(), iANodes );
    }
    builder.SetLwrrentDebugLocation( dloc );
}

// Get the debug info type for a PTX function.
llvm::DISubroutineType* FrontEndDebugInfo::get_debug_info_type( llvm::DIScope* scope, ptxSymbolTableEntry function )
{
    llvm::DIFile* di_file = getDIFile( function->location );
    // FIXME: add signature

    std::vector<llvm::Metadata*> signature_types;
    llvm::DITypeRefArray signature_types_array = m_di_builder.getOrCreateTypeArray( signature_types );
    return m_di_builder.createSubroutineType( signature_types_array );
}

// Get the debug info type for a PTX type.
llvm::DIType* FrontEndDebugInfo::get_debug_info_type( llvm::DIScope* scope, ptxType type )
{
    switch( type->kind )
    {
        case ptxTypeB1:
        case ptxTypeB2:
        case ptxTypeB4:
            return m_di_builder.createBasicType( "bit",
                                                 /*SizeInBits=*/ptxGetTypeSizeInBits( type ),
                                                 llvm::dwarf::DW_ATE_unsigned );
        case ptxTypeB8:
        case ptxTypeB16:
        case ptxTypeB32:
        case ptxTypeB64:
        case ptxTypeB128:
            return m_di_builder.createBasicType( "bit",
                                                 /*SizeInBits=*/ptxGetTypeSizeInBits( type ),
                                                 llvm::dwarf::DW_ATE_unsigned );

        case ptxTypeF16:
        case ptxTypeBF16:
        case ptxTypeTF32:
        case ptxTypeF32:
        case ptxTypeF64:
            return m_di_builder.createBasicType( "float",
                                                 /*SizeInBits=*/ptxGetTypeSizeInBits( type ),
                                                 llvm::dwarf::DW_ATE_float );

        case ptxTypeBF16x2:
        case ptxTypeF16x2:
            RT_ASSERT( !"I don't know how to get debug info type for packed half floats" );
            return m_di_builder.createBasicType( "packed_half_float",
                                                 /*SizeInBits=*/ptxGetTypeSizeInBits( type ),
                                                 /////// Bigler: I have no idea what DW_ATE_* to use or even if I can add one.
                                                 llvm::dwarf::DW_ATE_lo_user );

        case ptxTypeU2:
        case ptxTypeU4:
            return m_di_builder.createBasicType( "uint",
                                                 /*SizeInBits=*/ptxGetTypeSizeInBits( type ),
                                                 llvm::dwarf::DW_ATE_unsigned );
        case ptxTypeU8:
        case ptxTypeU16:
        case ptxTypeU32:
        case ptxTypeU64:
            return m_di_builder.createBasicType( "uint",
                                                 /*SizeInBits=*/ptxGetTypeSizeInBits( type ),
                                                 llvm::dwarf::DW_ATE_unsigned );
        case ptxTypeS2:
        case ptxTypeS4:
            return m_di_builder.createBasicType( "int",
                                                 /*SizeInBits=*/ptxGetTypeSizeInBits( type ),
                                                 llvm::dwarf::DW_ATE_signed );
        case ptxTypeS8:
        case ptxTypeS16:
        case ptxTypeS32:
        case ptxTypeS64:
            return m_di_builder.createBasicType( "int",
                                                 /*SizeInBits=*/ptxGetTypeSizeInBits( type ),
                                                 llvm::dwarf::DW_ATE_signed );

        case ptxTypePred:
            return m_di_builder.createBasicType( "bool",
                                                 /*SizeInBits=*/ptxGetTypeSizeInBits( type ),
                                                 llvm::dwarf::DW_ATE_unsigned );

        case ptxIncompleteArrayType: {
            llvm::DIType* elemType = get_debug_info_type( scope, type->cases.IncompleteArray.base );
            return m_di_builder.createPointerType( elemType,
                                                   /*SizeInBits=*/ptxGetTypeSizeInBits( type ),
                                                   /*AlignInBits=*/8u << ptxGetTypeLogAlignment( type ) );
        }
        case ptxOpaqueType:
            return m_di_builder.createBasicType( "tex",
                                                 /*SizeInBits=*/ptxGetTypeSizeInBits( type ),
                                                 llvm::dwarf::DW_ATE_unsigned );

        // TODO: revisit. ptxPointerType was remove in CL 26202143. Parser uses int now instead.
        //case ptxPointerType:
        //{
        //    llvm::DIType elemType = get_debug_info_type( scope, type->cases.Pointer.base );

        //    return m_di_builder.createPointerType( elemType,
        //                                           /*SizeInBits=*/ptxGetTypeSizeInBits( type ),
        //                                           /*AlignInBits=*/8u << ptxGetTypeLogAlignment( type ) );
        //}
        case ptxVectorType: {
            llvm::DIType* elemType = get_debug_info_type( scope, type->cases.Vector.base );
            llvm::DISubrange* sub = m_di_builder.getOrCreateSubrange( 0, type->cases.Vector.N - 1 );
            llvm::DINodeArray subArray = m_di_builder.getOrCreateArray( sub );

            return m_di_builder.createVectorType(
                /*SizeInBits=*/ptxGetTypeSizeInBits( type ),
                /*AlignInBits=*/8u << ptxGetTypeLogAlignment( type ), elemType, subArray );
        }
        case ptxParamListType:
            // Unsupported
            break;
        case ptxArrayType: {
            llvm::DIType* elemType = get_debug_info_type( scope, type->cases.Array.base );
            llvm::DISubrange* sub = m_di_builder.getOrCreateSubrange( 0, type->cases.Array.N - 1 );
            llvm::DINodeArray subArray = m_di_builder.getOrCreateArray( sub );

            return m_di_builder.createArrayType(
                /*SizeInBits=*/ptxGetTypeSizeInBits( type ),
                /*AlignInBits=*/8u << ptxGetTypeLogAlignment( type ), elemType, subArray );
        }
        case ptxNOTYPE:
        case ptxTypeE4M3:
        case ptxTypeE5M2:
        case ptxTypeE4M3x2:
        case ptxTypeE5M2x2:
        case ptxLabelType:
        case ptxMacroType:
        case ptxConditionCodeType:
            warn() << "ERROR Cannot retrieve DebugInfo type from unsupported PTX type \""
                   << getTypeEnumAsString( nullptr, type->kind )
                   << "\". Debug info might be "
                      "incomplete\n";
            return nullptr;
    }
    RT_ASSERT( !"Unexpected type kind" );
    return nullptr;
}

llvm::DIFile*
FrontEndDebugInfo::retrieveFileByIndex( uInt fileIndex ) const
{
    llvm::DIFile* di_file = nullptr;
    auto iter = m_di_files.find( fileIndex );
    if( iter != m_di_files.end() )
        di_file = iter->second;
    return di_file;
}

llvm::DIFile*
FrontEndDebugInfo::getDIFile( ptxCodeLocation loc )
{
    llvm::DIFile* di_file = loc ? retrieveFileByIndex( loc->lwrLoc.fileIndex ) : nullptr;
    if( !di_file )
    {
        di_file = m_di_builder.createFile( "generated", optix::getGeneratedCodeDirectory() );
    }
    return di_file;
}

void FrontEndDebugInfo::storeDwarfDebugStrSection()
{
    ptxDwarfSection debugStrSection = ptxDwarfGetSectionPointer( m_ptxState, DEBUG_STR_SECTION );
    if( debugStrSection )
        m_debug_str_section_data = retrieveDebugStringSectionData( m_ptxState, debugStrSection, m_dwarfInfo, m_errorCall );
}

// Traverse the given location until the last inlinedAt was found or loc else.
ptxCodeLocation FrontEndDebugInfo::getLocalCodeLocation( ptxCodeLocation loc )
{
    if( !loc )
        return loc;
    ptxCodeLocation res = loc;
    while( ( loc = loc->inlineAtLoc ) != nullptr )
    {
        res = loc;
    }
    return res;
}


llvm::raw_ostream& FrontEndDebugInfo::warn()
{
    return m_warnCall();
}
