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

#pragma once

#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/IRBuilder.h>

// This should be included before LLVM stuff, but there's an issue with "True" being defined.
#include <FrontEnd/PTX/PTXStitch/ptxparse/ptxIR.h>
#include <FrontEnd/PTX/PTXStitch/std/stdTypes.h>
#include <FrontEnd/PTX/PTXStitch/dwarf/lwdwarf.h>

#include <corelib/compiler/CoreIRBuilder.h>

// Need definition of DebugIndexedFile
#include <FrontEnd/PTX/PTXStitch/ptxparse/DebugInfo.h>

#include <map>
#include <stack>

namespace llvm {
class BasicBlock;
class Value;
class raw_ostream;
}

/// A Helper class handling LLVM debug info generation.
class FrontEndDebugInfo
{
  public:
    ///
    /// \param filename       the file name of the compiled PTX module
    /// \param module         the LLVM module
    /// \param only_lineinfo  if true, only line info will be produced
    /// \param ptxState       PTX parser state for data retrieval
    FrontEndDebugInfo( std::string const& filename, llvm::Module* module, bool only_lineinfo, ptxParsingState ptxState );

    /// Destructor.
    ~FrontEndDebugInfo();

    // Allow FrontEndDebugInfo to use a passed in error reporting mechanism.
	void setErrorCall( std::function<llvm::raw_ostream&()> fCall ) { m_errorCall = fCall; }
    // Allow FrontEndDebugInfo to use a passed in warn reporting mechanism.
    void setWarnCall( std::function<llvm::raw_ostream&()> fCall ) { m_warnCall = fCall; }

    /// Add a source filename
    /// \param filerec  the parse file information
    void add_file( DebugIndexedFile filerec );


    /// Mark start of processing the debug info for a PTX function.
    /// The debug info is initialized lazily when debug locations are added.
    ///
    /// \param ptx_func   the PTX function
    /// \param llvm_func  the corresponding LLVM function object
    void start_function( ptxSymbolTableEntry ptx_func, llvm::Function* llvm_func );

    /// Finish adding debug info for a PTX function.
    void finished_function();

    /// Add debug info for a PTX variable.
    ///
    /// \param ptx_var     the PTX variable
    /// \param var_adr     the LLVM variable address
    /// \param ir_builder  the current LLVM IR builder
    void add_variable( ptxSymbolTableEntry ptx_var, llvm::Value* var_adr, llvm::BasicBlock* lwrr_block );

    /// Mark a local variable value change.
    ///
    /// \param var         the PTX variable symbol
    /// \param new_val     the new LLVM value of this variable
    /// \param ir_builder  the current LLVM IR builder
    void mark_local_value_change( ptxSymbol var, llvm::Value* new_val, llvm::BasicBlock* lwrr_block );

    /// Creates a new block scope and push it.
    ///
    /// \param loc      the code location of the block if any
    /// \param builder  the current LLVM IRBuilder
    void push_block_scope( ptxCodeLocation loc, corelib::CoreIRBuilder& builder );

    /// Pop a variable scope, creating running if necessary.
    void pop_block_scope();

    /// Get the current LLVM debug info scope.
    llvm::DIScope* get_debug_info_scope();

    /// Set the current code location to an IRBuilder.
    ///
    /// \param loc      the PTX code location
    /// \param builder  an LLVM IRBuilder
    void set_lwrr_loc( ptxCodeLocation loc, corelib::CoreIRBuilder& builder );

    /// Get the debug info type for a PTX function.
    ///
    /// \param scope       the scope for this type
    /// \param function    the PTX function
    llvm::DISubroutineType* get_debug_info_type( llvm::DIScope* scope, ptxSymbolTableEntry function );

    /// Get the debug info type for a PTX type.
    ///
    /// \param scope       the scope for this type
    /// \param type        the PTX type
    llvm::DIType* get_debug_info_type( llvm::DIScope* scope, ptxType type );

    /// Retrieve the data from DWARF section .debug_str and store it internally.
    void storeDwarfDebugStrSection();

    // set_lwrr_loc collects the functions which are present in the inlineAtLoc
    // ptxCodeLocation fields but not in the LLVM module. We cannot generate inlinedAt
    // fields for the LLVM DILocations for those functions, so this information
    // can be used to generate information messages about potentially missing debug information.
    const std::set<std::string>& getMissingInlinedFunctions() const { return m_missingInlinedFunctions; }

    /// Return the outermost "local" location, ie traversing the chain of inlined functions.
    static ptxCodeLocation getLocalCodeLocation( ptxCodeLocation loc );

    /// Create a DICompileUnit in the module. This should only be used
    /// if the incoming PTX does not include debug info, but debug mode
    /// is enabled.
    /// \param fname         the file name of the compiled PTX module
    /// \param onlyLineInfo  if true, only line info will be produced
    /// \param module        the LLVM module
    static void createCompileUnitForModule( const std::string& fname, bool onlyLineinfo, llvm::Module* module );

  private:

    /// A colwenient mechanism for reporting errors.
    /// \return          The llvm stream in which to place errors.
    llvm::raw_ostream& error();
    /// A colwenient mechanism for reporting warnings.
    /// \return          The llvm stream in which to place errors.
    llvm::raw_ostream& warn();

    void initializeFunctionDebugInfo();

    // Retrieve file-related debug info. Fall back onto PTX input if original source can't be found.
    llvm::DIFile* getDIFile( ptxCodeLocation loc );
    // Retrieve DIFile through index only, ie no fallback provided.
    llvm::DIFile* retrieveFileByIndex( uInt fileIndex ) const;
    // Retrieve the function name corresponding to the given label from the DWARF section .debug_str.
    std::string getFunctionNameFromDwarfSection( const char* label );

    /// The debug info builder if any.
    llvm::DIBuilder m_di_builder;

    /// DIFile objects corresponding to the .file directives in ptx
    std::map<unsigned int, llvm::DIFile*> m_di_files;
    /// DIFile object corresponding to lwrrently active function
    llvm::DIFile*                         m_lwrrent_di_file = nullptr;
    /// The lexical block for the current function.
    llvm::DISubprogram*                   m_di_function = nullptr;

    typedef std::stack<llvm::DILexicalBlock*>           DILB_stack;
    typedef std::map<ptxSymbol, llvm::DILocalVariable*> Dbg_var_map;

    ptxSymbolTableEntry m_ptx_function = nullptr;
    llvm::Function*     m_llvm_func    = nullptr;

    /// The stack for debug info lexical blocks.
    DILB_stack m_dilb_stack;

    /// Maps PTX variables to DIVariables.
    Dbg_var_map m_dbg_var_map;

    /// If True, only line info will be produced.
    bool m_only_lineinfo;

    /// Giving access to all stored data, mappings, etc
    ptxParsingState m_ptxState;
    lwDwarfInfo     m_dwarfInfo;

    /// Giving access to LLVM data
    llvm::Module* m_module;

    /// Store data retrieved from dwarf section .debug_str
    std::vector<std::string> m_debug_str_section;
    std::vector<char>        m_debug_str_section_data;

    /// The lineinfo of the current function.
    ptxCodeLocation m_loc_function;

    std::function<llvm::raw_ostream&()> m_errorCall;
    std::function<llvm::raw_ostream&()> m_warnCall;

    std::set<std::string> m_missingInlinedFunctions;
};
