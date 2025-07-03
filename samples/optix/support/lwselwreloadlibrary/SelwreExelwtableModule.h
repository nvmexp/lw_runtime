// Copyright LWPU Corporation 2018
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

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <common/inc/lwSelwreLoadLibrary.h>
#endif

#include <corelib/system/ExelwtableModule.h>

#include <mutex>

#if defined( DEBUG ) || defined( DEVELOP )
#if !defined( OPTIX_ENABLE_SIDELOAD )
#define OPTIX_ENABLE_SIDELOAD
#endif
#endif

// Class to load a shared library in a secure way from the driver path. This is lwrrently only
// implemented on Windows. Otherwise it will load from the configured library path.
// TODO: Figure out what we should do on Linux and Mac
class SelwreExelwtableModule : public corelib::ExelwtableModule
{
    using corelib::ExelwtableModule::ExelwtableModule;

#ifdef _WIN32
    // Only on Windows we load from the driver path for now, on the other platforms we just
    // use ExcelwtableModule.
  public:
    bool init() override
    {
        if( m_object )
            return true;
#if defined( OPTIX_ENABLE_SIDELOAD )
        // In the debug and develop build try loading from the binary directory, first.
        if( ExelwtableModule::init() )
            return true;
#endif
        {
            // This has to be locked across all possible callers of lwLoadSystemLibrary,
            // not just multiple callers into this SelwreExelwtableModule.
            static std::mutex s_mutex;
            std::lock_guard<std::mutex> lock( s_mutex );
            m_object = lwLoadSystemLibraryExA( m_name.c_str(), 0 );
            lwReleaseSelwreLoadLibraryResources();
        }
        if( m_object )
        {
            m_initialized = true;
            return true;
        }
        return false;
    }
#endif
};
