#include "DcgmVersion.hpp"
#include "version.hpp"

#include <cstddef>
#include <cstdio>


namespace
{
template <std::size_t N, std::size_t Z>
void SafeCopyTo(char (&dst)[N], char const (&src)[Z])
{
    snprintf(dst, N, "%s", src);
}
} // namespace

dcgmReturn_t DECLDIR GetVersionInfo(dcgmVersionInfo_t* pVersionInfo)
{
    if (NULL == pVersionInfo)
    {
        return DCGM_ST_BADPARAM;
    }

    if (pVersionInfo->version != dcgmVersionInfo_version)
    {
        return DCGM_ST_VER_MISMATCH;
    }

    pVersionInfo->version = dcgmVersionInfo_version;
    SafeCopyTo(pVersionInfo->changelist, DCGMLIB_CHANGELIST);
    SafeCopyTo(pVersionInfo->platform, DCGMLIB_BUILD_PLATFORM);
    SafeCopyTo(pVersionInfo->branch, DCGMLIB_BUILD_BRANCH);
    SafeCopyTo(pVersionInfo->driverVersion, DCGMLIB_DRIVER_VERSION);
    SafeCopyTo(pVersionInfo->buildDate, DCGMLIB_BUILD_DATE);

    return DCGM_ST_OK;
}
