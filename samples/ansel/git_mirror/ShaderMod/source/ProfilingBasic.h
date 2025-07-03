#pragma once

#define ANSEL_PROFILE_ENABLE

#include <string>
#include <string.h>
#include <Windows.h>
#include <unordered_map>
#include <assert.h>

#include "ir/SpecializedPool.h"
#include "Timer.h"
#include "ProfilingSettings.h"

#if (ANSEL_PROFILE_TYPE == ANSEL_PROFILE_BASIC)

#define ANSEL_PROFILE_INIT(name)    shadermod::ProfileInit(name);
#define ANSEL_PROFILE_DEINIT()      shadermod::ProfileDeinit();

#define ANSEL_PROFILE_ENDFRAME()

#define ANSEL_PROFILE_START(zone_name, zone_descr)  { shadermod::ProfileZone * pz_##zone_name = shadermod::ProfileGetZone( #zone_name ); assert(pz_##zone_name != nullptr); pz_##zone_name->start(); }
#define ANSEL_PROFILE_STOP(zone_name)               { shadermod::ProfileZone * pz_##zone_name = shadermod::ProfileGetZone( #zone_name ); assert(pz_##zone_name != nullptr); pz_##zone_name->stop(); }

#define ANSEL_PROFILE_ZONE(zone_name, zone_descr)   shadermod::ProfileZone apz_##zone_name( #zone_name, true );

#define ANSEL_PROFILE_VALUE_FLOAT(value_name, value_descr, value)   shadermod::ProfileLogValueFloat(#value_name, value_descr, value);
#define ANSEL_PROFILE_VALUE_INT(value_name, value_descr, value)     shadermod::ProfileLogValueInt(#value_name, value_descr, value);
#define ANSEL_PROFILE_VALUE_UINT(value_name, value_descr, value)    shadermod::ProfileLogValueFloat(#value_name, value_descr, value);

namespace shadermod
{

    void ProfileInit(const char * name);
    void ProfileDeinit();

    void ProfileLogValueFloat(const char * name, const char * descr, float value);
    void ProfileLogValueInt(const char * name, const char * descr, int value);
    void ProfileLogValueUInt(const char * name, const char * descr, unsigned int value);

    class ProfileZone
    {
    protected:

        static const size_t profileNameMaxLen = 64;
        char profileName[profileNameMaxLen];

        bool m_isAuto = false;
        Timer m_timer;

    public:

        ProfileZone(const char * name)
        {
            sprintf_s(profileName, profileNameMaxLen, "%s", name);
        }

        ProfileZone(const char * name, bool isAuto):
            m_isAuto(isAuto)
        {
            sprintf_s(profileName, profileNameMaxLen, "%s", name);
            if (m_isAuto)
            {
                start();
            }
        }

        void start()
        {
            m_timer.Start();
        }
        void stop()
        {
            double zoneElapsedTime = m_timer.Time();
            ProfileLogValueFloat(profileName, nullptr, (float)zoneElapsedTime);
        }

        ~ProfileZone()
        {
            if (m_isAuto)
            {
                stop();
            }
        }
    };

    ProfileZone * ProfileGetZone(const char * name);
}

#endif
