#pragma once

#include <stdio.h>
#include <assert.h>

#include "ProfilingSettings.h"

#if (ANSEL_PROFILE_TYPE == ANSEL_PROFILE_TELEMETRY)

#include "rad_tm.h"

#define TELEMETRY_BOUNDED_EVENTS_AS_SPANS   0

#define ANSEL_PROFILE_INIT(name)    shadermod::ProfileInit(name);
#define ANSEL_PROFILE_DEINIT()      shadermod::ProfileDeinit();

#define ANSEL_PROFILE_ENDFRAME()    tmTick(0);

#if (TELEMETRY_BOUNDED_EVENTS_AS_SPANS == 1)
#define ANSEL_PROFILE_START(zone_name, zone_descr)  { tmBeginTimeSpan(0, shadermod::ProfileGetSpanId(#zone_name), 0, #zone_name); }
#define ANSEL_PROFILE_STOP(zone_name)               { tmEndTimeSpan(0, shadermod::ProfileGetSpanId(#zone_name)); }
#else
#define ANSEL_PROFILE_START(zone_name, zone_descr)  { tmEnter(0, 0, #zone_name); }
#define ANSEL_PROFILE_STOP(zone_name)               { tmLeave(0); }
#endif

#define ANSEL_PROFILE_ZONE(zone_name, zone_descr)   tmZone(0, 0, #zone_name);

#define ANSEL_PROFILE_VALUE_FLOAT(value_name, value_descr, value)   tmPlot(0, TM_PLOT_UNITS_REAL, TM_PLOT_DRAW_LINE, value, #value_name);
#define ANSEL_PROFILE_VALUE_INT(value_name, value_descr, value)     tmPlot(0, TM_PLOT_UNITS_INTEGER, TM_PLOT_DRAW_LINE, value, #value_name);
#define ANSEL_PROFILE_VALUE_UINT(value_name, value_descr, value)    tmPlot(0, TM_PLOT_UNITS_INTEGER, TM_PLOT_DRAW_LINE, value, #value_name);

namespace shadermod
{

    void ProfileInit(const char * name);
    void ProfileDeinit();

    void ProfileLogValueFloat(const char * name, const char * descr, float value);
    void ProfileLogValueInt(const char * name, const char * descr, int value);
    void ProfileLogValueUInt(const char * name, const char * descr, unsigned int value);

    tm_uint64 ProfileGetSpanId(const char * name);
}

#endif
