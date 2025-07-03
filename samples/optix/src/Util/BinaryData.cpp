//
//  Copyright (c) 2019 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#include <Util/BinaryData.h>

#include <Util/MakeUnique.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/UnknownError.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <atomic>
#include <mutex>
#include <string>

// Since this is a "C" function the exact signature doesn't matter if we only need the name.
extern "C" int __declspec( dllexport ) optixQueryFunctionTable();

namespace optix {
namespace data {

namespace {

class Module
{
  public:
    Module()
    {
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                   (LPCSTR)&optixQueryFunctionTable, &m_module);
    }
    ~Module()
    {
    }

    operator HMODULE() const { return m_module; }

  private:
    // nullptr for m_module is OK because that's what we want in static library optix
    // used for white box unit testing where the resources are linked into the test
    // exelwtable.  The same is also true for tools linked against the static library.
    HMODULE m_module = nullptr;
};

}  // namespace

/// Use getModule() to obtain the module, instead of accessing this global directly.
static std::mutex              g_moduleMutex;
static std::unique_ptr<Module> g_module;

static const Module& getModule()
{
    {
        std::lock_guard<std::mutex> lock( g_moduleMutex );
        if( !g_module )
        {
// Need to keep the HMODULE for resources around for the lifetime of the program, but
// we need to delay obtaining it until we are asked for a piece of binary data.
            g_module = makeUnique<Module>();
        }
    }
    return *g_module;
}

/// g_loaderCounter is a reference count that tells us when we can release g_module.
static std::atomic_int g_loaderCounter{0};

/// Called by optix_exp::DeviceContext c'tor
void acquireLoader()
{
    ++g_loaderCounter;
}

/// Called by optix_exp::DeviceContext d'tor.  When there are no outstanding device
/// contexts, we can release the library.
void releaseLoader()
{
    const int newValue = --g_loaderCounter;
    RT_ASSERT_MSG( newValue >= 0, "Loader counter underflow" );
    if( newValue == 0 )
    {
        std::lock_guard<std::mutex> lock( g_moduleMutex );
        g_module.reset();
    }
}

static HRSRC findBlob( const char* name )
{
    const HRSRC resource = FindResourceA( getModule(), name, "BLOB" );
    if( resource == nullptr )
    {
        throw prodlib::UnknownError( RT_EXCEPTION_INFO, "Couldn't find resource " + std::string{name} + ": "
                                                            + std::to_string(::GetLastError() ) );
    }

    return resource;
}

const char* getBinaryDataPtr( const char* name )
{
    const HGLOBAL data = LoadResource( getModule(), findBlob( name ) );
    if( data == nullptr )
    {
        throw prodlib::UnknownError( RT_EXCEPTION_INFO, "Couldn't load resource " + std::string{name} + ": "
                                                            + std::to_string(::GetLastError() ) );
    }

    const void* const ptr = LockResource( data );
    if( ptr == nullptr )
    {
        throw prodlib::UnknownError( RT_EXCEPTION_INFO, "Couldn't lock resource " + std::string{name} + ": "
                                                            + std::to_string(::GetLastError() ) );
    }

    return static_cast<const char*>( ptr );
}

std::size_t getBinaryDataSize( const char* name )
{
    const DWORD size = SizeofResource( getModule(), findBlob( name ) );
    if( size == 0 )
    {
        throw prodlib::UnknownError( RT_EXCEPTION_INFO, "Couldn't find size of resource " + std::string{name} + ": "
                                                            + std::to_string(::GetLastError() ) );
    }
    return size;
}

}  // namespace data
}  // namespace optix
