#include <string>
#include <unordered_map>

#include "ProfilingSettings.h"
#include "ProfilingTelemetry.h"
#include "Log.h"

#if (ANSEL_PROFILE_TYPE == ANSEL_PROFILE_TELEMETRY)

#if _M_AMD64
#   pragma comment(lib, "rad_tm_win64.lib")
#else
#   pragma comment(lib, "rad_tm_win32.lib")
#endif

namespace shadermod
{
    tm_uint64 g_timeSpanId = 0;
    std::unordered_map<std::string, tm_uint64> g_profileZoneSpanIdsMap;

    void ProfileInit(const char * name)
    {
        tmLoadLibrary(TM_RELEASE);

        tm_error err;

        err = tmInitialize(0, 0);

        if (err != TM_OK)
        {
            if (err == TMERR_ALREADY_INITIALIZED)
            {
                LOG_WARN("Profiler already initialized");
            }
            else
            {
                LOG_WARN("Error initializing profiler, %d", (int)err);
                return;
            }
        }

        err = tmOpen(
            0,                      // unused
            "Ansel",                // program name
            __DATE__ " " __TIME__,  // identifier, could be date time, or a build number
            "localhost",            // telemetry server address
            TMCT_TCP,               // network capture
            TELEMETRY_DEFAULT_PORT, // telemetry server port
            TMOF_INIT_NETWORKING,   // flags
            100                     // timeout in milliseconds ... pass -1 for infinite
            );

        if (err == TM_OK)
        {
            printf("Connected to the Telemetry server!\n");
        }
        else if (err == TMERR_DISABLED)
        {
            printf("Telemetry is disabled via #define NTELEMETRY\n");
        }
        else if (err == TMERR_UNINITIALIZED)
        {
            printf("tmInitialize failed or was not called\n");
        }
        else if (err == TMERR_NETWORK_NOT_INITIALIZED)
        {
            printf("WSAStartup was not called before tmOpen! Call WSAStartup or pass TMOF_INIT_NETWORKING.\n");
        }
        else if (err == TMERR_NULL_API)
        {
            printf("There is no Telemetry API (the DLL isn't in the EXE's path)!\n");
        }
        else if (err == TMERR_COULD_NOT_CONNECT)
        {
            printf("There is no Telemetry server running\n");
        }

        tmThreadName(
            0,                      // Capture mask (0 means capture everything)
            0,                      // Thread id (0 means use the current thread)
            "Ansel Main Thread"     // Name of the thread
            );
    }

    void ProfileDeinit()
    {
        tmClose(0);

        // Call tmShutdown before your game exits
        tmShutdown();

        // Free the memory you passed to tmInitialize [not needed, we initialized with tmInitialize(0, 0)]
        //free(telemetry_memory);
    }

    tm_uint64 ProfileGetSpanId(const char * name)
    {
        auto it = g_profileZoneSpanIdsMap.find(name);
        if (it == g_profileZoneSpanIdsMap.end())
        {
            tm_uint64 profileZoneSpanId = g_timeSpanId++;
            g_profileZoneSpanIdsMap.insert(std::make_pair(std::string(name), profileZoneSpanId));
            return profileZoneSpanId;
        }
        else
        {
            return it->second;
        }
    }
}

#endif
