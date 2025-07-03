// Copyright LWPU Corporation 2013
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <prodlib/exceptions/Backtrace.h>

#include <enableBacktrace.h>

#ifdef OPTIX_ENABLE_STACK_TRACE
#include <iostream>
#include <map>
#include <sstream>
#include <string.h>

#if defined( WIN32 )
// clang-format off
#include <windows.h>

// VS 2015 (1900) and Windows SDK 8.1 (0x0603) don't get along.  Specifically, Dbghelp.h generates the following warning:
// 1>C:\Program Files (x86)\Windows Kits\8.1\Include\um\Dbghelp.h(1544): warning C4091: 'typedef ': ignored on left of '' when no variable is declared
#include <ntverp.h>
#if _MSC_VER == 1900 && VER_PRODUCTVERSION_W == 0x0603
#  pragma warning ( push )
#  pragma warning ( disable : 4091 )
#  define FIXED_WARNING_4091
#endif

#include <Dbghelp.h>

#ifdef FIXED_WARNING_4091
#  pragma warning ( pop )
#  undef FIX_WARNING_4091
#endif
// clang-format on
#else
#include <execinfo.h>
#include <unistd.h>
#if defined( __linux__ )
#include <cxxabi.h>
#endif
#endif


#define MAX_SYMBOL_LENGTH 8192

namespace prodlib {
/* \brief This class provides functionality to create a backtraces for debugging purposes.
    **/
class Backtrace
{
  public:
    /* \brief Default Constructor. Initializes debug symbols */
    Backtrace();
    ~Backtrace();

    /* \brief Create a backtrace of the calling thread. Each string in the returned vector contains information about one line
       * \param firstFrame Skip the given amount of frames in the backtrace.
       * \param maxFrames Walk up the stack up to of maxFrames after skipping skipFrames.
       * \param exclusionList list of names to exclude in the backtrace
       * \return A std::vector containing strings where each string contains a textual representation for one stackframe.
       **/
    std::vector<std::string> backtrace( unsigned int skipFrames, unsigned int maxFrames, const std::vector<std::string>& exclusionList );

  private:
#if defined( WIN32 )
    CRITICAL_SECTION m_lock;      // DbgHelp is not debugging safe. Global lock.
    HANDLE           m_hProcess;  // Handle of the running process.
    PSYMBOL_INFO     m_symbolInfo;

    typedef PVOID address_type;

    std::map<PVOID, std::string> m_resolvedAddresses;
#else
    typedef void* address_type;
#endif

#if defined( WIN32 )
    bool m_debugInfoAvailable;
#endif

    std::vector<std::string> resolveAddresses( const std::vector<address_type>& addresses, const std::vector<std::string>& exclusionList );

#if defined( __linux__ )
    /// Demangles a mangled name using abi::__cxa_demangle().
    ///
    /// Returns the demangled name in case of success, and the mangled name in case of failure.
    static std::string demangle( const std::string& mangledName );
#endif
};

#if defined( WIN32 )
Backtrace::Backtrace()
    : m_debugInfoAvailable( false )
{
    // Allocate memory for symbols
    m_symbolInfo               = (PSYMBOL_INFO)malloc( sizeof( SYMBOL_INFO ) + MAX_SYMBOL_LENGTH );
    m_symbolInfo->SizeOfStruct = sizeof( SYMBOL_INFO );
    m_symbolInfo->MaxNameLen   = MAX_SYMBOL_LENGTH;

    m_hProcess = GetLwrrentProcess();
    InitializeCriticalSection( &m_lock );

    SymSetOptions( SYMOPT_DEBUG | SYMOPT_LOAD_LINES );
    m_debugInfoAvailable = ( TRUE == SymInitialize( m_hProcess, 0, FALSE ) );
    SymSetOptions( SYMOPT_DEBUG | SYMOPT_LOAD_LINES );
}
#else
Backtrace::Backtrace()
{
}
#endif

Backtrace::~Backtrace()
{
#if defined( WIN32 )
    // free up memory allocated for symbol infos
    free( m_symbolInfo );
#endif
}

bool exclude( const std::string& name, const std::vector<std::string>& exclusionList )
{
    for( size_t i = 0; i < exclusionList.size(); i++ )
    {
        if(::strncmp( name.c_str(), exclusionList[i].c_str(), exclusionList[i].size() ) == 0 )
        {
            return ( true );
        }
    }
    return ( false );
}

#if WIN32 && !_WIN64
#pragma warning( disable : 4748 )
#endif
std::vector<std::string> Backtrace::backtrace( unsigned int firstFrame, unsigned int maxFrames, const std::vector<std::string>& exclusionList )
{
    std::vector<std::string> stackWalk;

#if defined( WIN32 )
    CONTEXT lwrrentContext;
#ifdef _M_IX86
    // on x86 assembly is required to initialize CONTEXT
    memset( &lwrrentContext, 0, sizeof( CONTEXT ) );
    lwrrentContext.ContextFlags = CONTEXT_CONTROL;

    // fetch registers required for StackWalk64
    __asm {
      Label:
        mov [lwrrentContext.Ebp], ebp;
        mov [lwrrentContext.Esp], esp;
        mov eax, [Label];
        mov [lwrrentContext.Eip], eax;
    }
#else
    // on all other platforms there's an helper function
    RtlCaptureContext( &lwrrentContext );
#endif

    //
    // Set up stack frame.
    //
    DWORD        machineType;
    STACKFRAME64 stackFrame;
    memset( &stackFrame, 0, sizeof( STACKFRAME64 ) );

#ifdef _M_IX86
    machineType                 = IMAGE_FILE_MACHINE_I386;
    stackFrame.AddrPC.Offset    = lwrrentContext.Eip;
    stackFrame.AddrPC.Mode      = AddrModeFlat;
    stackFrame.AddrFrame.Offset = lwrrentContext.Ebp;
    stackFrame.AddrFrame.Mode   = AddrModeFlat;
    stackFrame.AddrStack.Offset = lwrrentContext.Esp;
    stackFrame.AddrStack.Mode   = AddrModeFlat;
#elif _M_X64
    machineType                 = IMAGE_FILE_MACHINE_AMD64;
    stackFrame.AddrPC.Offset    = lwrrentContext.Rip;
    stackFrame.AddrPC.Mode      = AddrModeFlat;
    stackFrame.AddrFrame.Offset = lwrrentContext.Rsp;
    stackFrame.AddrFrame.Mode   = AddrModeFlat;
    stackFrame.AddrStack.Offset = lwrrentContext.Rsp;
    stackFrame.AddrStack.Mode   = AddrModeFlat;
#elif _M_IA64
    machineType                  = IMAGE_FILE_MACHINE_IA64;
    stackFrame.AddrPC.Offset     = lwrrentContext.StIIP;
    stackFrame.AddrPC.Mode       = AddrModeFlat;
    stackFrame.AddrFrame.Offset  = lwrrentContext.IntSp;
    stackFrame.AddrFrame.Mode    = AddrModeFlat;
    stackFrame.AddrBStore.Offset = lwrrentContext.RsBSP;
    stackFrame.AddrBStore.Mode   = AddrModeFlat;
    stackFrame.AddrStack.Offset  = lwrrentContext.IntSp;
    stackFrame.AddrStack.Mode    = AddrModeFlat;
#else
#error "Unsupported platform"
#endif

    std::vector<address_type> addresses;

    for( unsigned int frameNumber = firstFrame; frameNumber < firstFrame + maxFrames; frameNumber++ )
    {
        PVOID  backTrace;
        USHORT capturedFrames = RtlCaptureStackBackTrace( frameNumber, 1, &backTrace, NULL );
        if( !capturedFrames || !backTrace )
        {
            break;
        }

        addresses.push_back( backTrace );
    }

    // Now resolve addresses
    // TODO: check to see if Dbghelp.dll is present
    stackWalk = resolveAddresses( addresses, exclusionList );

// defined(WIN32)
#elif defined( __linux__ )
    std::vector<void*> trace( firstFrame + maxFrames );
    int                actualFrames = ::backtrace( (void**)&trace[0], firstFrame + maxFrames );

    // To obtain meaningful symbol names -fvisibility=hidden must not be used.
    char** symbols = backtrace_symbols( (void**)&trace[0], actualFrames );

    // Demangle symbol names. No file name or line numbers (this requires using the debug
    // information).
    if( symbols )
    {
        for( int i = firstFrame; i < actualFrames; i++ )
        {
            // Find parentheses and plus sign surrounding the mangled name.
            char* left  = index( symbols[i], '(' );
            char* plus  = left ? index( left, '+' ) : nullptr;
            char* right = plus ? index( plus, ')' ) : nullptr;
            if( left && plus && right )
            {
                std::string demangledName = demangle( std::string( left + 1, plus - left - 1 ) );
                demangledName += " ";
                demangledName += std::string( plus, right - plus );
                stackWalk.push_back( demangledName );
            }
            else
            {
                stackWalk.push_back( symbols[i] );
            }
        }
        free( symbols );
    }
    else
    {
        stackWalk.push_back( "error retrieving backtrace" );
    }
#endif

    return stackWalk;
}
#if WIN32 && !_WIN64
#pragma warning( default : 4748 )
#endif

std::vector<std::string> Backtrace::resolveAddresses( const std::vector<address_type>& addresses,
                                                      const std::vector<std::string>&  exclusionList )
{
    std::vector<std::string> stackWalk;

#if defined( WIN32 )
    // Structure for line information
    IMAGEHLP_LINE64 line;
    line.SizeOfStruct = sizeof( IMAGEHLP_LINE64 );


    // Structure for module information
    IMAGEHLP_MODULE64 module;
    module.SizeOfStruct = sizeof( IMAGEHLP_MODULE64 );

    //
    // Dbghelp is is singlethreaded, so acquire a lock.
    //
    // Note that the code assumes that
    // SymInitialize( GetLwrrentProcess(), NULL, TRUE ) has
    // already been called.
    //
    EnterCriticalSection( &m_lock );

    for( std::size_t i = 0; i < addresses.size(); ++i )
    {
        address_type address        = addresses[i];
        bool         exclude_symbol = false;
        std::string  symbol_name;

        if( m_debugInfoAvailable )
        {
            // fetch function name
            DWORD64 displacementSymbol = 0;
            if( !SymFromAddr( m_hProcess, reinterpret_cast<DWORD64>( address ), &displacementSymbol, m_symbolInfo ) )
            {
                // SymFromAddr failed
                DWORD error = GetLastError();
                if( error == ERROR_MOD_NOT_FOUND )
                {
                    if( !SymRefreshModuleList( m_hProcess )
                        || !SymFromAddr( m_hProcess, reinterpret_cast<DWORD64>( address ), &displacementSymbol, m_symbolInfo ) )
                    {
                        error       = GetLastError();
                        symbol_name = "<symbol not found>";
                    }
                    else
                    {
                        symbol_name = m_symbolInfo->Name;
                    }
                }
                else
                {
                    symbol_name = "<symbol not founds>";
                }
            }
            else
            {
                symbol_name = m_symbolInfo->Name;
            }

            exclude_symbol = exclude( symbol_name, exclusionList );

            //m_resolvedAddresses[address] = symbol_name;
        }

        if( !exclude_symbol )
        {
            if( m_debugInfoAvailable )
            {
                std::ostringstream symbol_info;

                symbol_info << "["
                            << "IP=" << address << "] " << symbol_name;

                // fetch line if we can
                DWORD displacement2 = 0;
                if( SymGetLineFromAddr64( m_hProcess, reinterpret_cast<DWORD64>( address ), &displacement2, &line ) == TRUE )
                {
                    std::string clean_filename = line.FileName;
                    std::size_t offset         = clean_filename.find_last_of( '\\' ) + 1;
                    if( offset < clean_filename.length() )
                        clean_filename = clean_filename.substr( offset );

                    symbol_info << ", " << clean_filename << ":" << line.LineNumber;
                }

                // fetch module if we can
                SymRefreshModuleList( m_hProcess );
                if( TRUE == SymGetModuleInfo64( m_hProcess, reinterpret_cast<DWORD64>( address ), &module ) )
                {
                    std::string clean_imagename = module.ImageName;
                    std::size_t offset          = clean_imagename.find_last_of( '\\' ) + 1;
                    if( offset < clean_imagename.length() )
                        clean_imagename = clean_imagename.substr( offset );

                    symbol_info << "\t(" << clean_imagename << ")";

                    if( SymPdb == module.SymType )
                        stackWalk.push_back( symbol_info.str() );
                    else
                    {
                        std::stringstream ss;
                        ss << address;
                        stackWalk.push_back( ss.str() );
                    }
                }
            }
            else  // no debug info available
            {
                // Just push back the address as a string
                std::stringstream ss;
                ss << "0x" << address;
                stackWalk.push_back( ss.str() );
            }
        }
    }

    LeaveCriticalSection( &m_lock );
#endif

    return stackWalk;
}

#if defined( __linux__ )
std::string Backtrace::demangle( const std::string& mangledName )
{
    int   status;
    char* result = abi::__cxa_demangle( mangledName.c_str(), 0, 0, &status );
    if( result )
    {
        std::string demangledName = result;
        free( result );
        return demangledName;
    }
    else
    {
        return mangledName + "()";
    }
}
#endif

std::vector<std::string> backtrace( int skipFrames, unsigned int maxFrames, const std::vector<std::string>& exclusionList )
{
    static Backtrace trace;
    return trace.backtrace( skipFrames, maxFrames, exclusionList );
}

}  // end namespace prodlib

#else  // #ifdef OPTIX_ENABLE_STACK_TRACE  (no backtrace enabled)

namespace prodlib {
std::vector<std::string> backtrace( int skipFrames, unsigned int maxFrames, const std::vector<std::string>& exclusionList )
{
    return std::vector<std::string>();
}
}  // end namespace prodlib

#endif  // #ifdef OPTIX_ENABLE_STACK_TRACE
