// Copyright LWPU Corporation 2008
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <prodlib/system/Logger.h>

#include <corelib/misc/String.h>
#include <corelib/system/System.h>
#include <corelib/system/Timer.h>
#include <prodlib/system/Knobs.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>

#if defined( WIN32 )
#include <windows.h>  // for HANDLE
#else
#include <unistd.h>
#endif

using namespace corelib;

namespace {
// clang-format off
PublicKnob<std::string> k_logFile     ( RT_PUBLIC_DSTRING("log.file"),     "stderr", RT_PUBLIC_DSTRING("Location of log output (stderr, stdout, or filename)") );
Knob<int>         k_logLevel          ( RT_DSTRING("log.level"),           5,  RT_DSTRING("Log level (0 is off; 100 is maximum verbosity)") );
Knob<bool>        k_logPrintTimestamp ( RT_DSTRING("log.printTimestamp"),  false,  RT_DSTRING("Add timestamp to each log output") );
Knob<bool>        k_logPrintOrigin    ( RT_DSTRING("log.printOrigin"),     false,  RT_DSTRING("Add source origin info to each log output") );
Knob<bool>        k_logPrintLevel     ( RT_DSTRING("log.printLevel"),      false,  RT_DSTRING("Add message level info to each log output") );
Knob<std::string> k_restrictToFiles   ( RT_DSTRING("log.restrictToFiles"), "",  RT_DSTRING("Restrict logging of higher levels to messages from source files with the specified names. Separate multiple files by semicolon.") );
PublicKnob<bool>  k_colored           ( RT_PUBLIC_DSTRING("log.colored"),  true, RT_PUBLIC_DSTRING("Use colored output for console log messages") );
// clang-format on
}

static bool checkATTY( FILE* file )
{
// Don't do isatty() on Win because it won't report 'true' on non-default
// terminals.
#ifdef WIN32
    return true;
#else
    return isatty( fileno( file ) );
#endif
}

//------------------------------------------------------------------------------
//
//  Base logger
//
//------------------------------------------------------------------------------

namespace {

class Logger
{
  public:
    Logger( std::ostream& out );

    void setLogLevel( int level );
    int getLogLevel() const;

    void setFileRestriction( const std::vector<std::string>& files );

    void setPrintOrigin( bool onoff );
    bool getPrintOrigin() const { return m_print_origin; }

    void setPrintLevel( bool onoff );
    bool getPrintLevel() const { return m_print_level; }

    void setPrintTime( bool onoff );
    bool getPrintTime() const { return m_print_time; }

    void setUseColor( bool onoff );
    bool getUseColor() const { return m_use_color; }

    bool active( int level );

    std::ostream& stream( int level, const char* file, int line );

  private:
    // Returns a formatted string with the delta time from construction and now.
    std::string lwrrentTime();

    // Set text color for logging to console.
    enum TextColor
    {
        COL_RESET,
        COL_BRIGHT,
        COL_RED,
        COL_YELLOW,
        COL_GREEN,
        COL_CYAN
    };
    void setColor( TextColor col );

    int              m_level = 0;             // active log level
    std::set<size_t> m_filehashes;            // hash values for file names if logging is restricted to certain files
    bool             m_print_origin = false;  // Print the originating file,lineno
    bool             m_print_level  = false;  // Print the log level of each log entry
    bool             m_print_time   = false;  // Print the time stamp of each log entry
    bool             m_use_color    = false;  // Colorized printing

    timerTick     m_startTime = 0;  // start timestamp
    std::ostream& m_out;            // output stream that takes all log msgs
};


Logger::Logger( std::ostream& out )
    : m_out( out )
{
    // start timing
    m_startTime = getTimerTick();
}


void Logger::setLogLevel( int level )
{
    m_level = level;
    if( m_level < 0 )
        m_level = 0;
    if( m_level > 100 )
        m_level = 100;
}


int Logger::getLogLevel() const
{
    return m_level;
}


void Logger::setFileRestriction( const std::vector<std::string>& files )
{
    // process list of allowed files if logging is restricted to some files
    const std::vector<std::string> fnames = tokenize( k_restrictToFiles.get(), ";" );
    for( size_t i = 0; i < fnames.size(); ++i )
        m_filehashes.insert( hashString( strip( fnames[i] ) ) );
}


void Logger::setPrintOrigin( bool onoff )
{
    m_print_origin = onoff;
}


void Logger::setPrintLevel( bool onoff )
{
    m_print_level = onoff;
}


void Logger::setPrintTime( bool onoff )
{
    m_print_time = onoff;
}


void Logger::setUseColor( bool onoff )
{
    m_use_color = onoff;
}


std::ostream& Logger::stream( int level, const char* file, int line )
{

    bool file_active = true;

    // restrict to file only for detail log levels, we don't want to hide errors etc
    if( level > prodlib::log::LEV_PRINT && !m_filehashes.empty() )
    {
        // TODO: restrict to certain subpaths not just files, allow regex
        std::string  ff( file );
        const size_t pos = ff.find_last_of( "/\\" );
        if( pos != std::string::npos )
            ff      = std::string( ff.begin() + pos + 1, ff.end() );
        file_active = m_filehashes.find( hashString( ff ) ) != m_filehashes.end();
    }

    if( file_active && active( level ) )
    {
        m_out.flush();  // flush so we don't lose much on a crash
        // Make sure failbit and badbit is off.  You need to also turn off badbit, because if
        // you try to write to a stream that has failbit turned off, badbit gets set.
        m_out.clear( m_out.rdstate() & ~std::ostream::failbit & ~std::ostream::badbit );

        setColor( COL_BRIGHT );
        {
            // add timestamp
            if( m_print_time )
                m_out << lwrrentTime() << " ";

            // add level
            if( m_print_level )
                m_out << std::setw( 1 ) << "[" << std::right << std::setw( 2 ) << level << "] ";

            // add source origin
            if( m_print_origin )
                m_out << "[" << file << ":" << line << "] ";
        }

        // add level-specific text
        if( level == prodlib::log::LEV_FATAL )
        {
            setColor( COL_RED );
            m_out << "[FATAL] ";
        }
        if( level == prodlib::log::LEV_ERROR )
        {
            setColor( COL_RED );
            m_out << "[ERROR] ";
        }
        if( level == prodlib::log::LEV_WARNING )
        {
            setColor( COL_YELLOW );
            m_out << "[WARNING] ";
        }

        setColor( COL_RESET );
    }
    else
    {
        // Set the fail bit on the stream.
        m_out.clear( m_out.rdstate() | std::ostream::failbit );
    }

    return m_out;
}


bool Logger::active( int level )
{
    // TODO: make sure this alway returns false for optix dev log in public build
    return level <= m_level;
}


std::string Logger::lwrrentTime()
{
    return formatTime( getDeltaSeconds( m_startTime, getTimerTick() ) );
}


void Logger::setColor( TextColor col )
{
    if( !m_use_color )
        return;

    bool use_ansi_colors = true;

#ifdef _WIN32
    HANDLE hdl = ILWALID_HANDLE_VALUE;
    if( m_out.rdbuf() == std::cout.rdbuf() )
        hdl = GetStdHandle( STD_OUTPUT_HANDLE );
    else if( m_out.rdbuf() == std::cerr.rdbuf() )
        hdl = GetStdHandle( STD_ERROR_HANDLE );
    if( hdl != ILWALID_HANDLE_VALUE )
    {
        int c = 0;
        switch( col )
        {
            case COL_RESET:
                c = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE;
                break;
            case COL_BRIGHT:
                c = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY;
                break;
            case COL_RED:
                c = FOREGROUND_RED | FOREGROUND_INTENSITY;
                break;
            case COL_YELLOW:
                c = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY;
                break;
            case COL_GREEN:
                c = FOREGROUND_GREEN | FOREGROUND_INTENSITY;
                break;
            case COL_CYAN:
                c = FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY;
                break;
            default:
                break;
        }
        // If this call fails, it could mean we have a non-default terminal.
        // In this case, try the ANSI escape code variant.
        use_ansi_colors = SetConsoleTextAttribute( hdl, (WORD)c ) != TRUE;
    }
#endif

    // Path for Mac/Linux and l33t terminals on Win.
    if( use_ansi_colors )
    {
        switch( col )
        {
            case COL_RESET:
                m_out << "\x1b[0m";
                break;
            case COL_BRIGHT:
                m_out << "\x1b[1m";
                break;
            case COL_RED:
                m_out << "\x1b[31m\x1b[1m";
                break;
            case COL_YELLOW:
                m_out << "\x1b[33m\x1b[1m";
                break;
            case COL_GREEN:
                m_out << "\x1b[32m\x1b[1m";
                break;
            case COL_CYAN:
                m_out << "\x1b[36m\x1b[1m";
                break;
            default:
                break;
        }
    }
}

}  // end anonymous namespace


//------------------------------------------------------------------------------
//
//  OptiX Developer Log
//
//------------------------------------------------------------------------------

namespace {

// Configures a Logger for use as an OptiX internal dev logger
class OptiXLogger
{
  public:
    OptiXLogger();

    Logger&       getLogger();
    std::ostream& getStream() { return m_out; }

  private:
    std::unique_ptr<Logger> m_logger;
    std::ofstream           m_outfile;  // logfile in case logging goes to a file
    std::ostream            m_out;      // output stream that takes all log msgs

    void createLogStream( bool& useColor, const std::string& logFile );
    void setFileRestrictions( const std::string& files );
    void setUseColor( bool useColor );
};


OptiXLogger::OptiXLogger()
    : m_out( 0 )
{
#if defined( OPTIX_ENABLE_LOGGING )
    // init file/level with knobs
    std::string log_file  = k_logFile.get();
    int         log_level = k_logLevel.get();

    // special elw vars can override knobs
    const char* const elwvar_file   = "OPTIX_LOG_FILE";
    const char* const elwvar_level  = "OPTIX_LOG_LEVEL";
    const char*       elw_log_file  = ::getelw( elwvar_file );
    const char*       elw_log_level = ::getelw( elwvar_level );
    if( elw_log_file )
        log_file = elw_log_file;
    if( elw_log_level )
        log_level = atoi( elw_log_level );

    bool use_color = k_colored.get();
    createLogStream( use_color, log_file );

    //
    // Output info on where the log settings came from
    //
    if( log_level >= 10 )
    {
        if( elw_log_level )
            m_out << "Elw variable \"" << elwvar_level << "\"";
        else
            m_out << "Knob \"" << k_logLevel.getName() << "\"";
        m_out << " setting log level to " << log_level << ".\n";

        if( elw_log_file )
            m_out << "Elw variable \"" << elwvar_file << "\"";
        else
            m_out << "Knob \"" << k_logFile.getName() << "\"";
        m_out << " setting log file to \"" << log_file << "\".\n";
    }
    m_out.flush();

    //
    // Create and configure logger
    //
    m_logger.reset( new Logger( m_out ) );
    m_logger->setLogLevel( log_level );
    m_logger->setPrintOrigin( k_logPrintOrigin.get() );
    m_logger->setPrintLevel( k_logPrintLevel.get() );
    m_logger->setPrintTime( k_logPrintTimestamp.get() );
    m_logger->setUseColor( use_color );

    if( !k_restrictToFiles.get().empty() )
        setFileRestrictions( k_restrictToFiles.get() );
#endif

    // register callbacks for changes to the log level (ie the value of k_logLevel) etc
    // It is safe to do this even for RELASE_PUBLIC, as the non-official knobs would be inactive anyways,
    // due to the definition of RT_DSTRING vs RT_PUBLIC_DSTRING.
    k_logLevel.registerUpdateCB( [this]( int v ) { this->getLogger().setLogLevel( v ); } );
    k_logPrintOrigin.registerUpdateCB( [this]( bool v ) { this->getLogger().setPrintOrigin( v ); } );
    k_logPrintLevel.registerUpdateCB( [this]( bool v ) { this->getLogger().setPrintLevel( v ); } );
    k_logPrintTimestamp.registerUpdateCB( [this]( bool v ) { this->getLogger().setPrintTime( v ); } );
    k_logFile.registerUpdateCB( [this]( const std::string& f ) {
        bool useColor = this->getLogger().getUseColor();
        this->createLogStream( useColor, f );
        this->getLogger().setUseColor( useColor );
    } );
    k_restrictToFiles.registerUpdateCB( [this]( const std::string& fs ) { this->setFileRestrictions( fs ); } );
    k_colored.registerUpdateCB( [this]( bool v ) { this->setUseColor( v ); } );
}


Logger& OptiXLogger::getLogger()
{
    return *m_logger;
}

void OptiXLogger::createLogStream( bool& useColor, const std::string& logFile )
{
    if( logFile == "stdout" )
    {
        // Only use color for TTYs
        useColor = useColor && checkATTY( stdout );
        m_out.rdbuf( std::cout.rdbuf() );
    }
    else if( logFile == "stderr" )
    {
        // Only use color for TTYs
        useColor = useColor && checkATTY( stderr );
        m_out.rdbuf( std::cerr.rdbuf() );
    }
    else
    {
        m_outfile.open( logFile.c_str(), std::ios_base::app );
        if( !m_outfile )
        {
            std::cerr << "OptiX: Error opening log file (" << logFile << ") for writing. Logging disabled.\n";
        }
        else
        {
            // Only use color for TTYs
            useColor = false;
            m_outfile << "\n\n-------------------------------------\n"  // clang-format fail
                      << "Starting OptiX log @ " << getTimestamp()      //
                      << "\n-------------------------------------\n";   //
        }
        m_out.rdbuf( m_outfile.rdbuf() );
    }
}

void OptiXLogger::setFileRestrictions( const std::string& files )
{
    const std::vector<std::string> fnames = tokenize( files, ";" );
    m_logger->setFileRestriction( fnames );
}

void OptiXLogger::setUseColor( bool useColor )
{
    std::string logFile = k_logFile.get();
    if( logFile == "stdout" )
    {
        // Only use color for TTYs
        useColor &= checkATTY( stdout );
    }
    else if( logFile == "stderr" )
    {
        // Only use color for TTYs
        useColor &= checkATTY( stderr );
    }
    else
        useColor = false;
    m_logger->setUseColor( useColor );
}

// Singleton accessor
Logger& logger()
{
    static OptiXLogger g_loggerSingleton;
    return g_loggerSingleton.getLogger();
}

}  //  end anonymous namespace


//------------------------------------------------------------------------------
//
//  External API
//
//------------------------------------------------------------------------------

bool prodlib::log::active( int level )
{
#if defined( OPTIX_ENABLE_LOGGING )
    return logger().active( level );
#else
    return false;
#endif
}


int prodlib::log::level()
{
#if defined( OPTIX_ENABLE_LOGGING )
    return logger().getLogLevel();
#else
    return 0;
#endif
}


std::ostream& prodlib::log::stream( int level, const char* file, int line )
{
#if defined( OPTIX_ENABLE_LOGGING )
    return logger().stream( level, file, line );
#else
    static OptiXLogger g_nullLogger;
    return g_nullLogger.getStream();
#endif
}
