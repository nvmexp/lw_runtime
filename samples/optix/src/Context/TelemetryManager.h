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

#include <Context/Context.h>
#include <corelib/misc/Concepts.h>
#include <map>
#include <string>

namespace optix {

class Context;

class TelemetryManager : private corelib::NonCopyable
{

  public:
    TelemetryManager( Context* context );
    ~TelemetryManager();

    // Add data to the collection.
    void setContextCreateData( const char* name, const char* value );
    void setContextCreateData( const char* name, int value );

    void setContextTeardownData( const char* name, const char* value );
    void setContextTeardownData( const char* name, int value );

    // Send gathered data to the GFE servers.
    bool uploadContextCreateData();
    bool uploadContextTeardownData();

  private:
    // Turn telemetry on/off
    const bool m_telemetry_on = false;

    Context* m_context = nullptr;
    bool     m_is_initialized;

    std::string m_contextUUID;
    std::string m_clientUUID;

    // Collect telemetry data
    typedef std::map<std::string, int>         TelemetryNumData;
    typedef std::map<std::string, std::string> TelemetryStrData;

    TelemetryNumData m_ContextCreateNumData;
    TelemetryStrData m_ContextCreateStrData;

    TelemetryNumData m_ContextTeardownNumData;
    TelemetryStrData m_ContextTeardownStrData;
};
}
