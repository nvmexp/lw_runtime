#pragma once

#define ANSEL_PROFILE_DISABLED      0
#define ANSEL_PROFILE_BASIC         1
#define ANSEL_PROFILE_TELEMETRY     2

#define ANSEL_PROFILE_TYPE          ANSEL_PROFILE_DISABLED

#if (ANSEL_PROFILE_TYPE == ANSEL_PROFILE_DISABLED)

#define ANSEL_PROFILE_INIT(name)
#define ANSEL_PROFILE_DEINIT()

#define ANSEL_PROFILE_ENDFRAME()

#define ANSEL_PROFILE_START(zone_name, zone_descr)
#define ANSEL_PROFILE_STOP(zone_name)

#define ANSEL_PROFILE_ZONE(zone_name, zone_descr)

#define ANSEL_PROFILE_VALUE_FLOAT(value_name, value_descr, value)
#define ANSEL_PROFILE_VALUE_INT(value_name, value_descr, value)
#define ANSEL_PROFILE_VALUE_UINT(value_name, value_descr, value)

#endif
