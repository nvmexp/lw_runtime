// Copyright (c) 2018, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <Util/ContainerAlgorithm.h>

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/InstVisitor.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Pass.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace llvm;

class Options
{
  public:
    bool                     m_diffMode      = false;  //  Report difference in statistics between two files.
    bool                     m_useFileLegend = false;  // Add a legend for readability with longer filenames.\n"
    int                      m_maxNameWidth  = 40;     // Truncate function names (etc.) for readability.
    std::vector<const char*> m_filenames;

    static void PrintUsage( int argc, const char* const* argv )
    {
        fprintf( stderr, "Usage: %s [options] file1.bc [file2.bc ...]\n", argv[0] );
        fprintf( stderr,
                 "Files can be bitcode (.bc) or LLVM IR (.ll)\n"
                 "Options:\n"
                 "  --diff        Report difference in statistics between two files.\n"
                 "  --legend      Add a legend for readability with longer filenames.\n"
                 "  --width N     Truncate function names (etc.) for readability.\n" );
    }

    // Parse command-line arguments.  Returns zero for success.
    int Parse( int argc, const char* const* argv )
    {
        const char* const* end = argv + argc;
        ++argv;  // Skip argv[0], which is the program name.
        while( argv < end )
        {
            if( 0 == strcmp( *argv, "--diff" ) )
            {
                m_diffMode = true;
                ++argv;
            }
            else if( 0 == strcmp( *argv, "--legend" ) )
            {
                m_useFileLegend = true;
                ++argv;
            }
            else if( 0 == strcmp( *argv, "--width" ) )
            {
                if( ++argv < end )
                {
                    m_maxNameWidth = atoi( *argv++ );
                }
                else
                {
                    fprintf( stderr, "Missing value for --width argument\n" );
                    return -1;
                }
            }
            else if( **argv == '-' )
            {
                fprintf( stderr, "Unrecognized command-line argument: '%s'\n", *argv );
                return -1;
            }
            else
            {
                m_filenames.push_back( *argv++ );
            }
        }
        if( m_diffMode && m_filenames.size() != 2 )
        {
            fprintf( stderr, "Diff mode requires exactly two files.\n" );
            return -1;
        }
        if( m_filenames.empty() )
            return -1;  // caller prints usage on error.
        return 0;
    }
};

// Most integer stats are stored in a <string, int> map called a CounterMap
class CounterMap
{
  private:
    using MapType = std::map<std::string, int>;

  public:
    CounterMap( int maxNameWidth = -1 )
        : m_maxNameWidth( maxNameWidth )
    {
    }

    using const_iterator = MapType::const_iterator;

    const_iterator begin() const { return m_map.begin(); }
    const_iterator end() const { return m_map.end(); }

    // Get a reference to the value with the specified key, creating a zero-valued entry
    // if necessary.
    int& operator[]( const std::string& key ) { return m_map[truncate( key )]; }

    // A const method to get the value with the specified key.
    int at( const std::string& key ) const { return m_map.at( truncate( key ) ); }

    // Find the entry for the specified key
    const_iterator find( const std::string& key ) const { return m_map.find( truncate( key ) ); }

    // Type of map entry.
    using Pair = std::pair<std::string, int>;

    // Sort the stats by decreasing value into the given vector of <string,int> pairs.
    void Sort( std::vector<Pair>* o_values ) const
    {
        o_values->assign( m_map.begin(), m_map.end() );
        optix::algorithm::sort( *o_values, []( const Pair& a, const Pair& b ) { return a.second > b.second; } );
    }

    // Reconcile this map to include all the keys in the other map, adding zero-valued entries as needed.
    void Reconcile( const CounterMap& other )
    {
        for( const MapType::value_type& pair : other.m_map )
        {
            if( m_map.find( pair.first ) == m_map.end() )
                m_map.insert( MapType::value_type( pair.first, 0 ) );
        }
    }

    // Construct a CounterMap that contains the difference between the given maps (which are assumed
    // to have been reconciled).
    static CounterMap Diff( const CounterMap& a, const CounterMap& b )
    {
        CounterMap result;
        for( auto it = a.begin(); it != a.end(); ++it )
        {
            const std::string& key    = it->first;
            int                aValue = it->second;
            int                bValue = b.at( key );
            result[key]               = aValue - bValue;
        }
        return result;
    }

  private:
    MapType m_map;
    int     m_maxNameWidth;

    std::string truncate( const std::string& str ) const
    {
        return m_maxNameWidth <= 0 ? str : str.substr( 0, m_maxNameWidth );
    }
};

// Stats for a single bitcode module.
class Stats
{
  public:
    Stats( const char* name, const Options& options )
        : m_options( options )
        , m_name( name )
        , m_counts( kNumMapKinds, CounterMap( options.m_maxNameWidth ) )
    {
    }

    // The map kind is used as an index into an array of maps containing different kinds of stats.
    enum MapKind
    {
        kModule = 0,    // Module stats: number of blocks, instructions, etc.
        kInstSummary,   // Summary of instruction counts: memory, arith, etc.
        kInstDetailed,  // Detailed instruction counts
        kCalls,         // Count for each callee
        kAsm,           // Counts for inline assembly instructions
        kNumMapKinds    // Number of map kinds.
    };

    // Get a reference to the map with the specified kind of stats.
    const CounterMap& operator[]( MapKind kind ) const { return m_counts[kind]; }
    CounterMap& operator[]( MapKind kind ) { return m_counts[kind]; }

    // Get the name of these stats (e.g. the bitcode filename).
    const std::string& GetName() const { return m_name; }

    // Reconcile these stats to include all the keys in the other stats, adding zero-valued entries
    // as needed
    void Reconcile( const Stats& other )
    {
        for( int kind = 0; kind < kNumMapKinds; ++kind )
        {
            m_counts[kind].Reconcile( other.m_counts[kind] );
        }
    }

    // Construct a new Stats with the difference between the given Stats (which are assumed to have
    // been reconciled).
    static Stats Diff( const Stats& a, const Stats& b, const Options& options )
    {
        Stats result( "diff", options );
        for( int kind = 0; kind < kNumMapKinds; ++kind )
        {
            result.m_counts[kind] = CounterMap::Diff( a.m_counts[kind], b.m_counts[kind] );
        }
        return result;
    }

  private:
    const Options&          m_options;
    std::string             m_name;
    std::vector<CounterMap> m_counts;  // Maps containing different kinds of stats.
};

class Reporter
{
  public:
    // Reconcile stats, so that they all include the same keys, adding zero-valued entries as needed.
    static void Reconcile( std::vector<Stats>& o_statsVec )
    {
        // Reconcile the first stats object with all the others.
        for( size_t i = 1; i < o_statsVec.size(); ++i )
            o_statsVec[0].Reconcile( o_statsVec[i] );

        // Reconcile each of the other stats with the first one.
        for( size_t i = 1; i < o_statsVec.size(); ++i )
            o_statsVec[i].Reconcile( o_statsVec[0] );
    }

    // Generate a report
    static void Report( const std::vector<Stats>& statsVec, const Options& options )
    {
        // Use the width of the longest key to size the first column.
        int keyWidth = getMaxKeyWidth( statsVec );
        reportHeader( statsVec, keyWidth, options );
        reportSection( statsVec, keyWidth, "MODULE", Stats::kModule );
        reportSection( statsVec, keyWidth, "SUMMARY", Stats::kInstSummary );
        reportSection( statsVec, keyWidth, "INSTRUCTIONS", Stats::kInstDetailed );
        reportSection( statsVec, keyWidth, "CALLS", Stats::kCalls );
        reportSection( statsVec, keyWidth, "INLINE ASM", Stats::kAsm );
    }

  private:
    // Get the maximum width of any key in any of the given stats.
    static int getMaxKeyWidth( const std::vector<Stats>& statsVec )
    {
        int width = 0;
        for( const Stats& stats : statsVec )
        {
            for( int kind = 0; kind < Stats::kNumMapKinds; ++kind )
            {
                for( const auto& pair : stats[Stats::MapKind( kind )] )
                {
                    width = std::max( width, static_cast<int>( pair.first.size() ) );
                }
            }
        }
        return width;
    }

    // Print a header row that specifies the stats filenames.
    static void reportHeader( const std::vector<Stats>& statsVec, int keyWidth, const Options& options )
    {
        if( options.m_useFileLegend )
        {
            printf( "COLUMNS\n" );
            for( size_t i = 0; i < statsVec.size(); ++i )
            {
                printf( "%zu  %s\n", i + 1, statsVec[i].GetName().c_str() );
            }
            printf( "%-*s", keyWidth, " " );
            for( size_t i = 0; i < statsVec.size(); ++i )
            {
                printf( "\t%8zu", i + 1 );
            }
            printf( "\n" );
        }
        else
        {
            printf( "%-*s", keyWidth, " " );
            for( const Stats& stats : statsVec )
            {
                printf( "\t%8s", stats.GetName().c_str() );
            }
            printf( "\n" );
        }
    }

    static void reportSection( const std::vector<Stats>& statsVec, int keyWidth, const char* heading, Stats::MapKind kind )
    {
        printf( "%s\n", heading );

        // The order of the report is determined by the order of the stats for the first module.
        // We assume that the stats have been reconciled, so they all have the same keys.
        std::vector<CounterMap::Pair> column0;
        statsVec[0][kind].Sort( &column0 );

        // The rows are individual stats (e.g. the number of load instructions).
        // The columns are the stats for each bitcode file.
        for( size_t row = 0; row < column0.size(); ++row )
        {
            // Print the key, followed by the value from the first column.
            std::string key    = column0[row].first;
            int         value0 = column0[row].second;
            printf( "%-*s\t%8i", keyWidth, key.c_str(), value0 );

            // Print the values for the remaining columns.
            for( size_t column = 1; column < statsVec.size(); ++column )
            {
                const CounterMap& map = statsVec[column][kind];
                auto              it  = map.find( key );
                printf( "\t%8i", it == map.end() ? 0 : it->second );
            }
            printf( "\n" );
        }
        printf( "\n" );
    }
};

// This Pass is an instruction visitor that records the count of each kind of
// instruction, plus summary stats.  Note that InstVisitor uses a "lwriously
// relwrring template pattern".
class BitcodeStatsPass : public FunctionPass, public InstVisitor<BitcodeStatsPass>
{
  public:
    static char ID;  // Pass identifier

    BitcodeStatsPass( Stats& stats, const Options& options )
        : FunctionPass( ID )
        , InstVisitor<BitcodeStatsPass>()
        , m_stats( stats )
        , m_options( options )
    {
    }

    void getAnalysisUsage( AnalysisUsage& AU ) const override { AU.setPreservesAll(); }

    virtual bool runOnFunction( Function& F )
    {
        visit( F );
        return false;
    }


  private:
    Stats&         m_stats;
    const Options& m_options;

    friend class InstVisitor<BitcodeStatsPass>;

    void visitFunction( Function& ) { ++m_stats[Stats::kModule]["functions"]; }
    void visitBasicBlock( BasicBlock& ) { ++m_stats[Stats::kModule]["blocks"]; }

    void visitInst( const char* opcode )
    {
        ++m_stats[Stats::kInstDetailed][opcode];
        ++m_stats[Stats::kModule]["instructions"];
    }

// The following macros and #include define a method for each kind of LLVM instruction.

// clang-format off
#define HANDLE_BINARY_INST( N, OPCODE, CLASS )  \
    void visit##OPCODE( CLASS& )                \
    {                                           \
        ++m_stats[Stats::kInstSummary]["Arith/Logical"];           \
        visitInst( #OPCODE );                   \
    }
#define HANDLE_MEMORY_INST( N, OPCODE, CLASS )  \
    void visit##OPCODE( CLASS& )                \
    {                                           \
        ++m_stats[Stats::kInstSummary]["Memory"];                  \
        visitInst( #OPCODE );                   \
    }
#define HANDLE_CAST_INST( N, OPCODE, CLASS )    \
    void visit##OPCODE( CLASS& )                \
    {                                           \
        ++m_stats[Stats::kInstSummary]["Cast"];                    \
        visitInst( #OPCODE );                   \
    }
#include "llvm/IR/Instruction.def"
    // clang-format on


    // clang-format off
    void visitICmp( ICmpInst& inst )                     { ++m_stats[Stats::kInstSummary]["Arith/Logical"]  ; visitInst( "ICmp" ); }
    void visitFCmp( FCmpInst& inst )                     { ++m_stats[Stats::kInstSummary]["Arith/Logical"]  ; visitInst( "FCmp" ); }
    void visitPHI( PHINode& inst )                       { ++m_stats[Stats::kInstSummary]["Phi"]            ; visitInst( "PHI" ); }
    void visitExtractElement( ExtractElementInst& inst ) { ++m_stats[Stats::kInstSummary]["Insert/Extract"] ; visitInst( "ExtractElement" ); }
    void visitInsertElement( InsertElementInst& inst )   { ++m_stats[Stats::kInstSummary]["Insert/Extract"] ; visitInst( "InsertElement" ); }
    void visitExtractValue( ExtractValueInst& inst )     { ++m_stats[Stats::kInstSummary]["Insert/Extract"] ; visitInst( "ExtractValue" ); }
    void visitInsertValue( InsertValueInst& inst )       { ++m_stats[Stats::kInstSummary]["Insert/Extract"] ; visitInst( "InsertValue" ); }
    void visitShuffleVector( ShuffleVectorInst& inst )   { ++m_stats[Stats::kInstSummary]["Other"]          ; visitInst( "ShuffleVector" ); }
    void visitLandingPad( LandingPadInst& inst )         { ++m_stats[Stats::kInstSummary]["Other"]          ; visitInst( "LandingPad" ); }
    void visitSelect( SelectInst& inst )                 { ++m_stats[Stats::kInstSummary]["Other"]          ; visitInst( "Select" ); }
    void visitUserOp1( Instruction& inst )               { ++m_stats[Stats::kInstSummary]["Other"]          ; visitInst( "UserOp1" ); }
    void visitUserOp2( Instruction& inst )               { ++m_stats[Stats::kInstSummary]["Other"]          ; visitInst( "UserOp2" ); }
    void visitVAArg( VAArgInst& inst )                   { ++m_stats[Stats::kInstSummary]["Other"]          ; visitInst( "VAArg" ); }
    // clang-format on


    void visitCall( CallInst& inst )
    {
        Function* callee = inst.getCalledFunction();
        if( callee )
        {
            ++m_stats[Stats::kInstSummary]["Call"];
            ++m_stats[Stats::kCalls][callee->getName()];
            visitInst( "Call" );
        }
        else if( InlineAsm* asmExp = findAsmExp( inst ) )
        {
            ++m_stats[Stats::kInstSummary]["InlineAsm"];
            ++m_stats[Stats::kAsm][asmExp->getAsmString()];
            visitInst( "InlineAsm" );
        }
    }

    static InlineAsm* findAsmExp( CallInst& inst )
    {
        for( Use* use = inst.op_begin(); use != inst.op_end(); ++use )
        {
            if( InlineAsm* asmExp = dyn_cast<InlineAsm>( use->get() ) )
                return asmExp;
        }
        return nullptr;
    }
};

char BitcodeStatsPass::ID = 0;

// Read bitcode (.bc) or parse IR (.ll), yielding a module.
std::unique_ptr<Module> parseModule( const char* filename, LLVMContext& context )
{
    // Read file into memory buffer.
    ErrorOr<std::unique_ptr<MemoryBuffer>> res = MemoryBuffer::getFileOrSTDIN( filename );
    std::error_code ec;
    if( !res )
    {
        ec = res.getError();
        std::cerr << "Error opening bitcode file: " << ec.message() << ": " << filename << "\n";
        return std::unique_ptr<Module>();
    }
    std::unique_ptr<MemoryBuffer>& mb = res.get();
    std::unique_ptr<Module>        module;
    bool                           isBitcode = llvm::sys::path::extension( filename ) == ".bc";
    if( isBitcode )
    {
        std::string                       msg;
        Expected<std::unique_ptr<Module>> exMod = parseBitcodeFile( mb->getMemBufferRef(), context );
        if( !exMod )
        {
            exMod.takeError();
            std::cerr << "Error reading bitcode: " << filename << "\n";
        }
        else
            module = std::move( *exMod );
    }
    else
    {
        SMDiagnostic err;
        module = parseIR( mb->getMemBufferRef(), err, context );
        if( !module )
        {
            err.print( filename, llvm::errs() );
        }
    }
    return module;
}

int main( int argc, const char* const* argv )
{
    Options options;
    if( options.Parse( argc, argv ) )
    {
        Options::PrintUsage( argc, argv );
        return -1;
    }
    LLVMContext context;

    // Retain the stats for all the bitcode files.
    std::vector<Stats> statsVec;
    statsVec.reserve( options.m_filenames.size() );

    // Collect stats for each file.
    for( int i = 0; i < options.m_filenames.size(); ++i )
    {
        fprintf( stderr, "[Scanning %s...]\n", options.m_filenames[i] );

        std::unique_ptr<Module> module( parseModule( options.m_filenames[i], context ) );
        if( !module )
            return -1;  // error already reported

        // Gather and report instruction counts.
        legacy::PassManager MPM;
        statsVec.push_back( Stats( options.m_filenames[i], options ) );
        MPM.add( new BitcodeStatsPass( statsVec.back(), options ) );
        MPM.run( *module );
    }

    Reporter::Reconcile( statsVec );
    if( options.m_diffMode )
    {
        statsVec.push_back( Stats::Diff( statsVec[0], statsVec[1], options ) );
    }
    Reporter::Report( statsVec, options );

    return 0;
}
