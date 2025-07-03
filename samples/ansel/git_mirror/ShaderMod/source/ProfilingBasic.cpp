#include "ProfilingSettings.h"
#include "ProfilingBasic.h"

#if (ANSEL_PROFILE_TYPE == ANSEL_PROFILE_BASIC)

#include "Timer.h"

namespace shadermod
{

    wchar_t g_profileRootPath[] = L"";
    std::unordered_map<std::string, FILE *> g_profileOpenedFilesMap;

    ir::Pool<ProfileZone> g_profileZonesPool;
    std::unordered_map<std::string, ProfileZone *> g_profileZonesMap;

    FILE * ProfileGetFile(const char * name, bool * wasFound = nullptr)
    {
        if (wasFound)
        {
            *wasFound = false;
        }

        wchar_t filename[MAX_PATH];
        swprintf_s(filename, MAX_PATH, L"%sanselProfile_%hs", g_profileRootPath, name);

        FILE * fp = nullptr;

        auto it = g_profileOpenedFilesMap.find(name);
        if (it == g_profileOpenedFilesMap.end())
        {
            fp = _wfsopen(filename, L"wt", _SH_DENYWR);
            if (fp)
            {
                g_profileOpenedFilesMap.insert(std::make_pair(std::string(name), fp));
            }
        }
        else
        {
            if (wasFound)
            {
                *wasFound = true;
            }

            fp = it->second;
        }

        return fp;
    }

    void ProfileLogValueFloat(const char * name, const char * descr, float value)
    {
        bool wasFileAlreadyOpened;
        FILE * fp = ProfileGetFile(name, &wasFileAlreadyOpened);

        if (!fp)
            return;

        if (descr && !wasFileAlreadyOpened)
        {
            fprintf(fp, "%s\n", descr);
        }

        fprintf(fp, "%f\n", value);
    }

    void ProfileLogValueInt(const char * name, const char * descr, int value)
    {
        bool wasFileAlreadyOpened;
        FILE * fp = ProfileGetFile(name, &wasFileAlreadyOpened);

        if (!fp)
            return;

        if (descr && !wasFileAlreadyOpened)
        {
            fprintf(fp, "%s\n", descr);
        }

        fprintf(fp, "%d\n", value);
    }

    void ProfileLogValueUInt(const char * name, const char * descr, unsigned int value)
    {
        bool wasFileAlreadyOpened;
        FILE * fp = ProfileGetFile(name, &wasFileAlreadyOpened);

        if (!fp)
            return;

        if (descr && !wasFileAlreadyOpened)
        {
            fprintf(fp, "%s\n", descr);
        }

        fprintf(fp, "%u\n", value);
    }

    ProfileZone * ProfileGetZone(const char * name)
    {
        auto it = g_profileZonesMap.find(name);
        if (it == g_profileZonesMap.end())
        {
            ProfileZone * profileZone = g_profileZonesPool.getElement();
            if (profileZone)
            {
                new (profileZone) ProfileZone(name, false);
                g_profileZonesMap.insert(std::make_pair(std::string(name), profileZone));
                return profileZone;
            }
            else
            {
                // Error
                return nullptr;
            }
        }
        else
        {
            return it->second;
        }
    }

    void ProfileInit(const char * name)
    {
    }

    void ProfileDeinit()
    {
        g_profileOpenedFilesMap.clear();

        for (auto it = g_profileZonesMap.begin(); it != g_profileZonesMap.end(); ++it)
        {
            g_profileZonesPool.putElement(it->second);
        }
        g_profileZonesMap.clear();

        g_profileZonesPool.destroy();
    }
}

#endif