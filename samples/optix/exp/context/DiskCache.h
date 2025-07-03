// Copyright LWPU Corporation 2017
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

#define __OPTIX_INCLUDE_INTERNAL_HEADERS__
#include <optix_7_types.h>
#undef __OPTIX_INCLUDE_INTERNAL_HEADERS__

#include <exp/context/DiskCacheDatabase.h>

#include <memory>
#include <string>

namespace optix {
class PersistentStream;
}

namespace optix_exp {
class DeviceContextLogger;
class ErrorDetails;

class DiskCache
{
  public:
    DiskCache()                   = default;
    DiskCache( const DiskCache& ) = delete;
    DiskCache& operator=( const DiskCache& ) = delete;
    ~DiskCache()                             = default;

    OptixResult init( DeviceContextLogger& clog,
                      ErrorDetails&        errDetails,
                      const std::string&   fileName,
                      const std::string&   path = "<USE DEFAULT>" );
    OptixResult init( DeviceContextLogger& clog,
                      ErrorDetails&        errDetails,
                      const std::string&   fileName,
                      const std::string&   path,
                      size_t               lowWaterMark,
                      size_t               highWaterMark );
    OptixResult destroy( DeviceContextLogger& clog );

    OptixResult setPath( const std::string& path, const std::string& fileName, DeviceContextLogger& clog, ErrorDetails& errDetails );

    // Returns true if the disk cache is enabled
    bool isActive() const;
    void setIsActive( bool active );

    // Returns true if the disk cache has been disabled by the environment variable
    bool isDisabledByElwironment() const;

    // Returns true if the disk cache memory limits have been set by the environment variable
    bool isSizeSetByElwironment() const;

    // Looks for the entry in the cache and returns a persistent
    // stream object set in the the reading direction if the object
    // was found. Returns a null pointer if nothing was found.
    OptixResult find( const std::string&                        key,
                      std::unique_ptr<optix::PersistentStream>& stream,
                      DeviceContextLogger&                      clog,
                      ErrorDetails&                             errDetails ) const;

    // Reserves an entry in the cache and returns a persistent stream
    // object set in the writing direction if the object was
    // found. Returns a null pointer if the cache file could not be
    // created.
    OptixResult insert( const std::string& key, std::unique_ptr<optix::PersistentStream>& stream, ErrorDetails& errDetails );

    // Creates a stream that computes the md5 hash of the
    // persistent object. Does not modify the object.
    std::unique_ptr<optix::PersistentStream> createHasher() const;

    // Sets the memory limits for the garbage collection of the underlying
    // DiskCacheDatabase.
    //
    // Returns true if the memory limits have been succesfully changed and
    // false in case of invalid parameter values.
    OptixResult setMemoryLimits( size_t lowWaterMark, size_t highWaterMark, DeviceContextLogger& clog, ErrorDetails& errDetails );

    // Returns the memory limits for the garbage collection (0 means disabled).
    void getMemoryLimits( size_t& lowWaterMark, size_t& highWaterMark ) const;

    // Returns the path to the database file, minus the filename.
    std::string getPath( const std::string& fileName ) const;

  private:
    // Note: Access to these members requires the owner of the disk cache
    // to provide thread-safety mechanisms (see mutex lock in DeviceContext)
    std::string               m_lwrrentDatabasePath;
    mutable DiskCacheDatabase m_database;
    bool                      m_active              = false;
    bool                      m_disabledByElw       = false;
    bool                      m_sizeSetByElw        = false;
    bool                      m_exportReadableFiles = false;
};
}  // namespace optix_exp
