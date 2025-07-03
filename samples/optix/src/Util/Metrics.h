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

#include <stdint.h>

namespace optix {

// Utility class for logging metrics. This class is intended for low-volume logging
// in code that is not performance critical. All functions but init() are thread-safe.
class Metrics
{
  public:
    // All other functions will no-op if this function returns false. Metrics
    // are enabled with a knob.
    static bool isEnabled();

    // This function is a no-op if the Metrics enable knob has not been set.
    static void setEnabled( bool val );

    // Call before logging metrics.
    static void init();

    // Log named metric values
    static void logInt( const char* name, uint64_t value );
    static void logFloat( const char* name, double value );
    static void logString( const char* name, const char* value );

    // Scopes group logged name/value pairs together.
    enum ScopeType
    {
        OBJECT,
        ARRAY
    };
    static void pushScope( const char* name = nullptr, ScopeType type = OBJECT );
    static void popScope();

    // Manually flush buffered log records. Useful to control buffer size and
    // avoid unanticipated stalls when writing to disk.
    static void flush();
};

// An RAII wrapper for Metrics::pushScope()/popScope() to avoid unpopped scopes in the
// presence of exceptions.
class MetricsScope
{
  public:
    MetricsScope( const char* name = nullptr, Metrics::ScopeType type = Metrics::OBJECT )
    {
        Metrics::pushScope( name, type );
    }

    ~MetricsScope() { Metrics::popScope(); }

    MetricsScope( const MetricsScope& ) = delete;
    MetricsScope& operator=( const MetricsScope& ) = delete;
};

}  // namespace optix