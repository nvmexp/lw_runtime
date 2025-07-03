/*
 * Copyright (c) 2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <prodlib/system/System.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <direct.h>  // for _chdir and _getcwd
#include <io.h>      // for _findfirst() and _findnext()
#include <windows.h>
#define MAXPATHLEN _MAX_PATH
#else                              // POSIX
#include <dirent.h>
#include <errno.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <unistd.h>  // for getcwd() and chdir()
#endif

#if defined( __APPLE__ )
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/mach_init.h>
#include <mach/mach_types.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#include <corelib/system/System.h>

#include <algorithm>
#include <fstream>

std::string prodlib::getHostName()
{
#ifdef _WIN32
    std::string name;
    if( corelib::getelw( "COMPUTERNAME", name ) )
        return name;
    else
        return "";
#else  // POSIX
    char buffer[512];
    int  err = gethostname( buffer, 512 );
    if( err != 0 )
        return "";
    return buffer;
#endif
}

// Return number of CPU cores.
unsigned int prodlib::getNumberOfCPUCores()
{
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo( &sysinfo );
    return (unsigned int)sysinfo.dwNumberOfProcessors;
#else  // POSIX
    return (unsigned int)sysconf( _SC_NPROCESSORS_ONLN );
#endif
}

unsigned int prodlib::getCPUClockRateInKhz()
{
#ifdef _WIN32
    // TODO: implement this function for windows
    return 0;
#elif defined __linux__
    // TODO: implement this function for linux
    return 0;
#else
    int          mib[2];
    unsigned int freq;
    size_t       len;

    mib[0] = CTL_HW;
    mib[1] = HW_CPU_FREQ;
    len    = sizeof( freq );
    if( sysctl( mib, 2, &freq, &len, NULL, 0 ) != 0 )
        return -1;

    return freq / 1000;
#endif
}

size_t prodlib::getAvailableSystemMemoryInBytes()
{
// See interesting information about this problem here:
// http://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process
#if defined( __APPLE__ )

    vm_size_t              page_size;
    mach_port_t            mach_port;
    mach_msg_type_number_t count;
    vm_statistics_data_t   vm_stats;

    mach_port               = mach_host_self();
    count                   = sizeof( vm_stats ) / sizeof( natural_t );
    int64_t total_available = 0;

    if( KERN_SUCCESS == host_page_size( mach_port, &page_size )
        && KERN_SUCCESS == host_statistics( mach_port, HOST_VM_INFO, (host_info_t)&vm_stats, &count ) )
    {
        int64_t free_bytes = (int64_t)vm_stats.free_count * (int64_t)page_size;
        total_available += free_bytes;

        int64_t inactive_bytes = (int64_t)vm_stats.inactive_count * (int64_t)page_size;
        total_available += inactive_bytes;

        // hard to use
        // (int64_t)vm_stats.active_count * (int64_t)page_size;
        // (int64_t)vm_stats.wire_count   * (int64_t)page_size;
    }
    return static_cast<size_t>( total_available );
#elif defined( __linux__ )
    long pages     = sysconf( _SC_AVPHYS_PAGES );
    long page_size = sysconf( _SC_PAGE_SIZE );
    return static_cast<size_t>( pages * page_size );
#elif defined( _WIN32 )
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof( MEMORYSTATUSEX );
    GlobalMemoryStatusEx( &memInfo );
    return static_cast<size_t>( memInfo.ullAvailPhys );
#else
#error "Unknown system"
#endif
}

size_t prodlib::getTotalSystemMemoryInBytes()
{
    // See note in getAvailableSystemMemory()
#if defined( __APPLE__ )
    // TODO: Add support for macOS
    return 0;
#elif defined( __linux__ )
    long pages     = sysconf( _SC_PHYS_PAGES );
    long page_size = sysconf( _SC_PAGE_SIZE );
    return static_cast<size_t>( pages * page_size );
#elif defined( _WIN32 )
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof( MEMORYSTATUSEX );
    GlobalMemoryStatusEx( &memInfo );
    return static_cast<size_t>( memInfo.ullTotalPhys );
#else
#error "Unknown system"
#endif
}

std::string prodlib::getCPUName()
{
#if defined( __APPLE__ )
    char   name[100];
    size_t len = sizeof( name );

    if( sysctlbyname( "machdep.cpu.brand_string", name, &len, NULL, 0 ) != 0 )
        return "Unknown";
    return name;
#elif defined( __linux__ )
    // TODO: query the actual CPU name (Linux)
    return "Unknown";
#elif defined( _WIN32 )
    std::string name;
    if( corelib::getelw( "PROCESSOR_IDENTIFIER", name ) )
        return name;
    else
        return "Unknown";
#else
#error "Unknown system"
#endif
}

std::string prodlib::getPlatform()
{
#if defined( __APPLE__ )
    return "Mac";
#elif defined( __linux__ )
    return "Linux";
#elif defined( _WIN32 )
    return "Windows";
#else
#error "Unknown system"
#endif
}

// Check if a given file exists.
bool prodlib::fileExists( const char* file )
{
#ifdef _WIN32
    std::wstring wideFile = corelib::colwertUtf8ToUtf16( std::string( file ) );
    DWORD attrib = GetFileAttributesW( wideFile.c_str() );
    return ( attrib != ILWALID_FILE_ATTRIBUTES ) && !( attrib & FILE_ATTRIBUTE_DIRECTORY );
#else
    using namespace std;
    ifstream fin( file, ios::in | ios::binary );
    return !fin.fail();
#endif
}

// Check if a given file is writable.
bool prodlib::fileIsWritable( const char* file )
{
#ifdef _WIN32
    std::wstring wideFile = corelib::colwertUtf8ToUtf16( std::string( file ) );
    DWORD attrib = GetFileAttributesW( wideFile.c_str() );
    return ( attrib != ILWALID_FILE_ATTRIBUTES ) && !( attrib & FILE_ATTRIBUTE_READONLY );
#else
    FILE* fp = fopen( file, "a" );
    if( !fp )
        return false;
    if( fclose( fp ) != 0 )
        return false;
    return true;
#endif
}

// Check if a given directory exists.
bool prodlib::dirExists( const char* path )
{
#ifdef _WIN32
    std::wstring widePath = corelib::colwertUtf8ToUtf16( std::string(path) );
    DWORD attrib = GetFileAttributesW( widePath.c_str() );
    return ( attrib != ILWALID_FILE_ATTRIBUTES ) && ( attrib & FILE_ATTRIBUTE_DIRECTORY );
#else
    DIR* dir = opendir( path );
    if( dir == NULL )
        return 0;
    else
    {
        closedir( dir );
        return 1;
    }
#endif
}

// Create the given directory, return false on error. Return true if the
// directory is created or already exists.
bool prodlib::createDir( const char* path )
{
#ifdef _WIN32
    std::wstring wpath = corelib::colwertUtf8ToUtf16( std::string(path) );
    bool res = CreateDirectoryW( wpath.c_str(), 0 ) != 0;
    return ( res || GetLastError() == ERROR_ALREADY_EXISTS );
#else
    errno = 0;
    int rc = mkdir( path, S_IRWXU | S_IRWXG | S_IROTH );
    return ( rc == 0 || ( rc < 0 && errno == EEXIST ) );
#endif
}

// Create the given directory and any intermediate directories, return false on error.
bool prodlib::createDirectories( const char* path )
{
    // 'Sanitize' the path by colwerting backslashes to forward slashes, which
    // should work properly on both Windows and Linux.
    std::string sanitized( path );
    std::replace( sanitized.begin(), sanitized.end(), '\\', '/' );

    // Walk the path from the root to the target, and try to create any missing
    // intermediate directories.
    for( size_t pos = sanitized.find( '/' ); pos != std::string::npos; pos = sanitized.find( '/', pos + 1 ) )
    {
        const std::string lwr = sanitized.substr( 0, pos + 1);
        if( !prodlib::dirExists( lwr.c_str() ) && !createDir( lwr.c_str() ) )
            return false;
    }

    // Make sure the target directory was created, which might not be the case
    // if the supplied path had a trailing slash.
    if( !prodlib::dirExists( path ) && !prodlib::createDir( path ) )
        return false;
    else
        return true;
}
