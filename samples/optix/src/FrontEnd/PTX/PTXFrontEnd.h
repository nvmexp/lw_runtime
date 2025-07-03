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

#include <FrontEnd/PTX/PTXtoLLVM.h>             // just for the DecryptCall typedef
#include <FrontEnd/PTX/FrontEndDebugInfo.h>
#include <FrontEnd/PTX/PTXStitch/ptxIR_fwd.h>
#include <FrontEnd/PTX/PTXStitch/ptxparse/AtomTable.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/Support/raw_ostream.h>

#include <prodlib/misc/String.h>

#include <map>
#include <string>
#include <vector>

namespace llvm {
class AllocaInst;
class LLVMContext;
class Module;
}

namespace optix {

/// Low-level PTX front-end interface.  Used by PTXtoLLVM to parse PTX
/// and translate to LLVM.  This is a single-use class (parsePTX an be
/// called only once, followed by single invocation of
/// translateModule).
///
class PTXFrontEnd
{
  public:
    /// Supported modes for debug info generation.
    enum Debug_info_mode
    {
        DEBUG_INFO_OFF  = 0,  ///< Do not generate debug info at all.
        DEBUG_INFO_LINE = 1,  ///< generate only line info (enough for --show-src)
        DEBUG_INFO_FULL = 2,  ///< Full debug info (for dwarf generation)
    };

    PTXFrontEnd( llvm::LLVMContext& llvmContext, const llvm::DataLayout* dataLayout, Debug_info_mode debugMode );
    PTXFrontEnd( llvm::Module* module, const llvm::DataLayout* dataLayout, Debug_info_mode debugMode, bool skipOptimization );
    ~PTXFrontEnd();

    /// Parses the PTX string including any prefix headers.  If an error
    /// oclwrs, this function returns false and the error can be
    /// retrieved with getErrorString().
    ///
    /// \param name        A name for the module
    /// \param declString  A string containing declarations to be parsed before the PTX string (if non-empty)
    /// \param ptxString   A string containing the PTX code
    /// \param decrypter   Instance of EncryptionManager
    /// \param decryptCall Decryption callback
    bool parsePTX( const std::string& name, const std::string& declString, const prodlib::StringView& ptxString,
       void* decrypter = nullptr, DecryptCall decryptCall = nullptr );


    bool processInputLWVM( const std::string& declString );

    /// Once the PTX has been parsed, the module can be translated.
    /// Returns a valid module on success.
    ///
    /// \return            An LLVM module
    llvm::Module* translateModule();

    /// Return the SM version from a parsed module.  SM version 3.5 will
    /// return 35.
    ///
    /// \return            The SM target declared in the PTX
    unsigned int getTargetArch();

    /// Once the PTX has been parsed, the module can be translated.
    /// Returns a valid module on success.
    ///
    /// \return            A string describing errors in PTX parsing or LLVM translation.
    std::string getErrorString();

    bool needsPtxInstructionsModuleLinked() const { return m_needsPtxInstructionsModuleLinked; }
  private:
    PTXFrontEnd( const PTXFrontEnd& );             // forbidden
    PTXFrontEnd& operator=( const PTXFrontEnd& );  // forbidden

    /*************************************************************
   * Member variables
   *************************************************************/
    /// The current LLVM context
    llvm::LLVMContext& m_context;

    /// The target datalayout config
    const llvm::DataLayout* m_dataLayout;

    /// The state of the PTX parser and associated memory pools
    IAtomTable*     m_atomTable;
    IMemPool        m_memoryPool;
    ptxParsingState m_ptxState;

    /// Track the current state of the front.  We use a simple state
    /// machine to manage the single-use sequence.
    enum State
    {
        Initial,
        Parsing,
        Parsed,
        Translating,
        Translated,
        Error
    } m_state;

    enum
    {
        /// LWVM address space encodings.
        ADDRESS_SPACE_REGS    = 0,
        ADDRESS_SPACE_CODE    = 0,
        ADDRESS_SPACE_GENERIC = 0,
        ADDRESS_SPACE_GLOBAL  = 1,
        /*reserved*/
        ADDRESS_SPACE_SHARED  = 3,
        ADDRESS_SPACE_CONST   = 4,
        ADDRESS_SPACE_LOCAL   = 5,
        ADDRESS_SPACE_UNKNOWN = 8,
    };

    /// Type of argument colwersion to support "large arg" instructions
    /// where registers can be implicitly narrowed or widened (sign
    /// extended or zero-extended).
    enum ArgColwersionKind
    {
        ARG_COLWERSION_NORMAL,
        ARG_COLWERSION_LARGEARG_SIGNED,
        ARG_COLWERSION_LARGEARG_UNSIGNED,
    };

    /// Current log of errors
    std::string m_errorString;

    // The LLVM error stream bound to m_errorString
    llvm::raw_string_ostream m_errorStream;

    /// The LLVM instruction builder.
    corelib::CoreIRBuilder m_builder;

    /// Used to create MetaData nodes.
    llvm::MDBuilder m_md_builder;

    /// The produced LLVM module
    llvm::Module* m_module;

    /// Whether the module needs the PTX wrapper functions to be linked in (for LWVM input)
    bool m_needsPtxInstructionsModuleLinked = false;
    /// Whether optimizations should be done during inline ASM handling (for LWVM input)
    bool m_skipOptimizations = false;
    /// The target architecture of the incoming LWVM module.
    int m_targetArch = 50;

    /// The current LLVM function during translation.
    llvm::Function* m_lwrr_func;

    /// The current PTX function during translation.
    ptxSymbolTableEntry m_lwrr_ptx_func;

    /// Which debug info should be generated.
    Debug_info_mode m_debug_info_mode;

    /// The debug info helper object if any.
    FrontEndDebugInfo* m_debug_info;

    /// The vector of all used globals.
    std::vector<llvm::Constant*> m_usedGlobals;

    /// List of allocas that were generated from ptxParamStorage. These often have limited
    /// scope, so we want to manually add lifetime markers around their use.
    std::vector<llvm::AllocaInst*> m_paramAllocas;

    /*************************************************************
   * Setup and high level control
   *************************************************************/

    /// Advance the simple error checking state from one state to
    /// another.  If the from state does not match, issue an error.
    void advanceState( State lwrState, State nextState );

    /// A colwenient mechanism for reporting errors and placing the
    /// object in an error state.  Use thusly:
    /// if(somethingWentWrong()){
    ///   errors() << "Description of error\n";
    ///   return;
    /// }
    ///
    /// \return          The llvm stream in which to place errors.
    llvm::raw_ostream& error();
    llvm::raw_ostream& warn();

    /*************************************************************
   * PTX Parsing
   *************************************************************/
    /// Private function to do the main work of parsing PTX.  Declare
    /// the state object, parse the declaration string and then parse
    /// the main ptx body.  Any messages left in m_errors will flag failure.
    ///
    /// \param declString  A string containing declarations to be parsed before the PTX string (if non-empty)
    /// \param ptxString   A string containing the PTX code
    /// \param decrypter   Instance of EncryptionManager
    /// \param decryptCall Decryption callback
    void parsePTX( const std::string& declString, const prodlib::StringView& ptxString, void* decrypter = nullptr, DecryptCall decryptCall = nullptr );

    /*************************************************************
   * Module-level methods
   *************************************************************/
    // Callback: transfers control to processParsedObjects
    static void processParsedObjects_cb( void* tablePtr, void* selfPtr );

    /// Processes the parsed file and collects global symbols processGlobalSymbols.
    ///
    /// \param tablePtr  the symboltable of the object
    void processParsedObject( ptxSymbolTable tablePtr );

    /// Creates the functions and the global variables, and put them into the
    /// collections.
    ///
    /// \param entry  the entry to process
    void processGlobalVariable( ptxSymbolTableEntry entry );

    /// Creates declarations for PTX functions
    ///
    /// \param entry  the entry to process
    void declareFunction( ptxSymbolTableEntry entry );

    /// Create the body for a previously declared function
    ///
    /// \param entry  the entry to process
    void processGlobalFunction( ptxSymbolTableEntry entry );

    /// Colwert inline assembly calls to LWVM calls
    void colwertAsmCallsToCallInstructions();

    /// Process a PTX initializer.
    ///
    /// \param init      the initializer
    /// \param var_type  the type of the initialized object
    llvm::Constant* processInitializer( ptxSymbolTableEntry entry, ptxInitializer init, llvm::Type* var_type );

    /// Process a PTX expression initializer.
    llvm::Constant* processExpressionInitializer( ptxSymbolTableEntry entry, ptxExpression expr, llvm::Type* var_type );

    /// Process an PTX struct initializer.
    llvm::Constant* processStructuredInitializer( ptxSymbolTableEntry entry, stdList_t list, llvm::Type* elem_type );

    /// Process a PTX array initializer.
    ///
    /// \param list   list of initializer expressions
    /// \param N      the size of the array
    template <typename T>
    llvm::Constant* processArrayInitializer( ptxSymbolTableEntry entry, stdList_t list, size_t N, llvm::Type* elem_type );


    /// Mark a global as used.
    ///
    /// \param global  the global to mark
    void markUsed( llvm::GlobalValue* global );

    /// Generated llvm.used array.
    void emitLLVMUsed();

    /// Generate lifetime markers for param allocas
    void addLifetimesToParamAllocas( const std::vector<llvm::AllocaInst*> paramAllocas );


    /*************************************************************
   * Function-level methods
   *************************************************************/

    /// Process a function.
    ///
    /// \param function  the PTX representation of as function
    void processFunction( ptxSymbolTableEntry function, bool declarationsOnly );

    /// Process a statement block of the current function.
    ///
    /// \param block  the block statement
    void processFunctionBlockStmt( ptxSymbolTable block, stdList_t fparams, stdList_t rparams );

    /// Handle variable declarations inside a block.
    ///
    /// \param entry  a variable declaration in the block
    void processBlockVariables( stdList_t ptxSymbolTableEntries );

    /// Process a function label
    ///
    /// \param label  the label
    void processFunctionLabel( ptxSymbolTableEntry label );

    /// Initialize the local copies of the function parameters
    ///
    /// \param list  the list of ptxVariableInfo that describe the formal parameters
    void initializeFunctionParameters( stdList_t list );

    /// Generate necessary instructions to handle a function return.
    void generateFunctionReturn();

    /*************************************************************
   * Instruction-level methods
   *************************************************************/

    /// Process a statement of the current function.
    ///
    /// \param statement  the statement (can be a block, an instruction or pragma)
    /// \param instrIndex the index of the instruction (only increments for instructions not blocks)
    void processFunctionStatement( ptxStatement statement, unsigned int instrIndex );

    /// Enter a new basic block the first time.
    ///
    /// \param bb  the basic block that is entered
    void enter_block( llvm::BasicBlock* bb );

    /// Retrieve a function object from the C intrinsic library which matches
    /// this ptx instruction.
    llvm::Function* getPtxInstructionFunction( ptxInstruction instr );

    /// Translate a function instruction.
    ///
    /// \param instruction  the instruction to process
    void translateInstruction( ptxInstruction instruction, unsigned int instrIndex );

    /// Default instruction translation
    void translate_default( ptxInstruction instr );

    /// Translate a bra instruction.
    void translate_bra( ptxInstruction instr, unsigned int instrIndex );

    /// Translate a brx.idx instruction.
    void translate_brx_idx( ptxInstruction instr );
    void loweringBrxIdx2Switch( llvm::Value* index, const std::vector<ptxSymbol>* targetLabels );

    /// Translate a call instruction.
    void translate_call( ptxInstruction instr );

    /// Translate a ret instruction.
    void translate_ret( ptxInstruction instr, unsigned int instrIndex );

    /// Translate an exit or trap instruction.
    void translate_exit_or_trap( ptxInstruction instr, unsigned int instrIndex );

    /// Translate a load instruction.
    void translate_ld( ptxInstruction instr );

    /// Translate a store instruction.
    void translate_st( ptxInstruction instr );

    /// Translate a tex instruction.
    void translate_tex( ptxInstruction instr );

    /// Translate a video instruction
    void translate_video( ptxInstruction instr );

    /// Translate the guard predicate by branching around the
    /// instruction.  Returns a block that should be passed to
    /// resumeAfterGuard after the instruction is translated.
    llvm::BasicBlock* translateGuard( ptxInstruction instr );

    /// Resume in the new basic block after the guarded instruction.
    void resumeAfterGuard( llvm::BasicBlock* );

    // Load the source registers for the specified instruction.
    void loadSrcOperands( ptxInstruction instr, std::vector<llvm::Value*>& args, bool includeReturlwalue = false );

    // Store the destiniation registers for the specified instruction.
    void storeDstOperands( ptxInstruction instr, llvm::Value* ret );

    /*************************************************************
   * Expression-level methods
   *************************************************************/

    /// Determine argument colwersion algorithm to be used for the
    /// specified instruction and argument.  See section 8.4.1 of the
    /// PTX ISA (version 3.2).

    /// \param instr           The PTX instruction
    /// \param argIndex        The argument index
    /// \return                The colwersion type to be used for loadExpression, storeExpression and supercast
    ArgColwersionKind largeArgColwersion( ptxInstruction instr, uInt argIndex ) const;

    ///
    /// Load the llvm value for the specified expression.
    ///
    /// \param expr            The ptx expression to be loaded.
    /// \param addressof       The address of the expression will be loaded instead of the value
    /// \param addressref      The expression is part of an addressRefExpression, so its address will be loaded
    /// \param colwersion      The type of colwersion to use for width colwersion
    /// \param instr           The instruction this expression is used in
    /// \return                The loaded value
    llvm::Value* loadExpression( ptxExpression expr, bool addressof, bool addressref, ArgColwersionKind colwersion, ptxInstruction instr );

    /// Store the llvm value for the specified expression.
    ///
    /// \param expr            The ptx expression to be stored
    /// \param value           The value to be stored
    /// \param colwersion      The type of colwersion to use for width colwersion
    /// \param instr           The instruction this expression is used in
    void storeExpression( ptxExpression expr, llvm::Value* value, ArgColwersionKind colwersion, ptxInstruction instr );

    /// Cast the specified value to type, allowing a plethora of different colwersions
    ///
    /// \param value           The value to be colwerted
    /// \param intoType        The destination type
    /// \param colwersion      The type of colwersion to use for width colwersion
    /// \return                The colwerted value
    llvm::Value* supercast( llvm::Value* value, llvm::Type* intoType, ArgColwersionKind colwersion );

    /// Generate a call to the LLVM intrinsic to get the value of a
    /// special register.
    ///
    /// \param name  the name of special register
    /// \return      The loaded value
    llvm::Value* callSregIntrinsic( const std::string& name );

    /*************************************************************
   * Type manipulation methods
   *************************************************************/

    /// map from PTX storage to LWVM address space.
    static const unsigned int s_translateStorage[ptxMAXStorage];

    /// Get the LWVM address space for a given PTX pointer attribute
    unsigned int get_address_space( ptxPointerAttr attr );

    /// Get the LWVM address space for a given PTX storage kind.
    static unsigned int get_address_space( ptxStorageKind kind ) { return s_translateStorage[kind]; }

    /// Get the LWVM address space for a given PTX storage kind.
    ///
    /// \param types    A vector in which to place the llvm type objects
    /// \param list     A stdList of type VariableInfo
    void typesFromVariableInfoList( std::vector<llvm::Type*>& types, stdList_t list );

    /// Get the LLVM type from a ptxType
    ///
    /// \param type           The PTX type object
    /// \param doubleres      Specifies that the integer width should be doubled.
    /// \return               The LLVM type
    llvm::Type* getLLVMType( ptxType type, bool doubleres );
    llvm::Type* getLLVMType( ptxExpression expr, bool doubleres, ptxInstruction instr );

    /// Get the LLVM type from an instruction argument type, possibly by
    /// forwarding to getLLVMType.
    ///
    /// \param type      The PTX type object
    /// \param argIndex  The instruction argument index
    /// \return          The LLVM type
    llvm::Type* getLLVMArgType( ptxInstruction instr, uInt argIndex );

    /// Generates a LLVM FunctionType given a function symbol or a label symbol (in case of a protoform function)
    ///
    /// \param entry  the entry to process
    llvm::FunctionType* getLLVMFunctionType( ptxSymbolTableEntryAux entry );

    /// Check whether the given argument of type ptxFollowAType also
    /// follows the VECTOR modifier.
    ///
    /// \param instr        The PTX instruction
    /// \param argIndex     The instruction argument index
    /// \return             true if the VECTOR modifier should be applied to the argument, otherwise false.
    bool followATypeDoesVectorize( ptxInstruction instr, uInt argIndex );

    /*************************************************************
   * PTX and LLVM utility functions
   *************************************************************/

    /// Add a LWVM annotation to the given value.
    ///
    /// \param value      llvm::Value to annotate
    /// \param kind       Annotation kind
    /// \param annotation Annotation value
    void addLWVMAnnotation( llvm::Value* value, char const* kind, unsigned annotation = 1 );

    std::map<llvm::Value*, llvm::MDNode*> m_lwvmAnnotations;

    /*************************************************************
   * Scopes and management thereof
   *************************************************************/

    /// Creates a new block scope and push it.
    ///
    /// \param loc  the code location of the block if any
    void push_scope( ptxCodeLocation loc );

    /// Pop a variable scope, creating running if necessary.
    void pop_scope();

    /// Get the address of a symbol.
    ///
    /// \param sym   a PTX symbol
    llvm::Value* get_symbol_address( ptxSymbol sym );

    /// Store the address of a symbol.
    ///
    /// \param sym   a PTX symbol
    /// \param adr   the address of the symbol
    void set_symbol_address( ptxSymbol sym, llvm::Value* adr );

    /// Add debug info for a PTX variable.
    ///
    /// \param var  the PTX variable's symbol table entry
    /// \param adr  the LLVM address of the variable
    void add_var_debug_info( ptxSymbolTableEntry var, llvm::Value* adr );

    /// Mark a change of the value of the given variable.
    ///
    /// \param var      the PTX variable (symbol) that is changed
    /// \param new_val   the new value of this variable
    void mark_var_assigned( ptxSymbol var, llvm::Value* new_val );
    std::string gelwideoInstrPlaceholder( ptxInstruction instr );

    /// A scope for symbol lookup.
    struct Scope
    {
        Scope( Scope const* parent )
            : parent( parent )
        {
        }

        typedef std::map<ptxSymbol, size_t>                 Label2Idx_map;
        typedef std::map<size_t, llvm::BasicBlock*>         Idx2Block_map;
        typedef std::map<ptxSymbol, llvm::Value*>           Symbol_map;
        typedef std::map<ptxSymbol, std::vector<ptxSymbol>> Label2Branchtargets_map;

        /// The parent scope.  We may need to look here for symbols
        Scope const* parent;

        /// Map a label symbol to an instruction index.
        Label2Idx_map label2Idx;

        /// Map instruction indexes of the current function to the associated basic block.
        Idx2Block_map idx2block;

        /// Map PTX symbols to addresses.
        Symbol_map symbol_adr_map;

        /// Map label symbol to a list of potential branch targets.
        Label2Branchtargets_map label2Branchtargets;

        /// Find the associated LLVM basic block of the given PTX label.
        llvm::BasicBlock* findLabel( ptxSymbol targetLabel ) const;

        /// Lookup the address of an entity.
        ///
        /// \param sym  the PTX symbol of the entity
        llvm::Value* lookupAdr( ptxSymbol sym ) const;

        /// Lookup the branchtargets of the given label.
        const std::vector<ptxSymbol>* lookupBranchtargets(ptxSymbol sym) const;
    };
    Scope* m_lwrrentScope;

    class LwrrentBlockValues
    {
        llvm::DenseMap<ptxSymbol, llvm::Value*> m_entries;

      public:
        void reset() { m_entries.clear(); }
        void add( ptxSymbol symbol, llvm::Value* value )
        {
            auto inserted = m_entries.insert( std::make_pair( symbol, value ) );
            // If the value was already present, overwrite it with the new one
            if( !inserted.second )
                inserted.first->second = value;
        }
        llvm::Value* find( ptxSymbol symbol )
        {
            auto iter = m_entries.find( symbol );
            if( iter != m_entries.end() )
                return iter->second;
            return nullptr;
        }
    };
    LwrrentBlockValues m_lwrrentBlockValues;

    /// RAII-like block scope handler.
    class Block_scope
    {
      public:
        /// Constructor.
        Block_scope( PTXFrontEnd& gen, ptxCodeLocation loc )
            : m_gen( gen )
        {
            gen.push_scope( loc );
        }

        /// Destructor.
        ~Block_scope() { m_gen.pop_scope(); }

      private:
        /// The code generator.
        PTXFrontEnd& m_gen;
    };
};

}  // namespace optix
