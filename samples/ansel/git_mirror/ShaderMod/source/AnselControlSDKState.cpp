#define NOMINMAX
#include <Windows.h>
#include <shlobj.h>
#include <Shlwapi.h>
#include <tchar.h>
#include <array>
#include <sstream>
#include <algorithm>

#include "anselcontrol/Version.h"
#include "anselutils/Utils.h"
#include "CommonStructs.h"
#include "Utils.h"
#include "AnselControlSDKState.h"
#include "UI.h"
#include "Config.h"
#include "Log.h"
#include "drs/LwDrsWrapper.h"
#include "darkroom/StringColwersion.h"
#include "i18n/LocalizedStringHelper.h"
#include "i18n/text.en-US.h"

namespace
{
    BOOL IsWow64()
    {
        typedef BOOL(WINAPI *LPFN_ISWOW64PROCESS) (HANDLE, PBOOL);
        LPFN_ISWOW64PROCESS fnIsWow64Process;
        BOOL bIsWow64 = FALSE;

        //IsWow64Process is not available on all supported versions of Windows.
        //Use GetModuleHandle to get a handle to the DLL that contains the function
        //and GetProcAddress to get a pointer to the function if available.

        fnIsWow64Process = (LPFN_ISWOW64PROCESS)GetProcAddress(
            GetModuleHandle(TEXT("kernel32")), "IsWow64Process");

        if (NULL != fnIsWow64Process)
            if (!fnIsWow64Process(GetLwrrentProcess(), &bIsWow64))
                bIsWow64 = TRUE;

        return bIsWow64;
    }

    // This function is copied from ShadowPlayDdiShimWrapper.cpp (in //sw/dev/gpu_drv/bugfix_main/drivers/common/src)
    void justifyProfileName(wchar_t* pName)
    {
        // replace illegal characters by spaces for creating file folder with this profile name
        wchar_t* s = pName;
        while (*s != NULL)
        {
            if (*s == '\\' || *s == '/' || *s == ':' || *s == '*' || *s == '?' || *s == '\"' || *s == '<' || *s == '>' || *s == '|')
            {
                *s = L' ';
            }
            ++s;
        }
    }

    template <typename T> int sgn(T val) { return (T(0) < val) - (val < T(0)); }

    bool checkAnselControlSdkTimestampBasedVersion(const uint64_t sdkVersion, uint32_t& major, uint32_t& minor)
    {
        // < Sep 12, 2016:                  10 2016 06 12 232722 - timestamp based, correct
        // > Sep 12, 2016, < Oct 12, 2016:  1601600 09 30 045454 - timestamp based with bug (year has two unnecessary trailing zeros
        //                                       ^^
        // and is clashing with minor - 0.14 + 2016 = 0.16)
        // > Oct 12, 2016:                  major.minor.hash (16 bit, 16 bit, 32 bit)

        const std::array<uint64_t, 19> pow10 = { 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000,
            10000000000ull, 100000000000ull, 1000000000000ull, 10000000000000ull, 100000000000000ull, 1000000000000000ull, 10000000000000000ull,
            100000000000000000ull, 1000000000000000000ull };

        const auto getDigits = [pow10](const uint64_t n, size_t offset, size_t digits) -> uint64_t
        {
            if (offset + digits >= pow10.size()) return 0ull;

            const uint64_t rightDigits = n % pow10[offset + digits];
            return rightDigits / pow10[offset];
        };

        const auto seconds = getDigits(sdkVersion, 0, 2);
        const auto minutes = getDigits(sdkVersion, 2, 2);
        const auto hours = getDigits(sdkVersion, 4, 2);
        const auto day = getDigits(sdkVersion, 6, 2);
        const auto month = getDigits(sdkVersion, 8, 2);
        const auto year = getDigits(sdkVersion, 10, 4);

        // in case year is 2016 (< Sep 12, 2016) or 1600 (> Sep 12, 2016, < Oct 12, 2016), month, day, hour, minutes and seconds are within limits - 
        // we're using the old version

        if (!(
            (seconds >= 0 && seconds < 60) &&
            (minutes >= 0 && minutes < 60) &&
            (hours >= 0 && hours <= 23) &&
            (year == 2016 || year == 1600) &&
            // working around another bug that was in the previous versioning scheme - when a day number is below 10 it is padded with spaces, not zeros
            // this leads to incorrect math (still it's revertible and identifiable) :/
            // For all day numbers between 1 and 9, [day] portion of the version is from '36' (1st day) to '44' (9th day) and [month] portion is from '46' (May, when versioning was introduced) 
            // to '51' (October, when this scheme was no longer in use)
            (((month >= 46 && month <= 51) && (day >= 36 && day <= 44)) ||
            // or it is a normal date
            ((month >= 1 && month <= 12) && (day >= 1 && day <= 31)))))
            return false;
        

        major = 0;
        if (year == 2016)
            minor = static_cast<uint32_t>(getDigits(sdkVersion, 14, 2));
        else if (year == 1600)
            minor = static_cast<uint32_t>(getDigits(sdkVersion, 15, 2)) - 2; // '2' in '2016' clashes with minor

        return true;
    }
}

AnselControlSDKState::AnselControlSDKState() :
    m_configuration(nullptr),
    m_getVersion(nullptr),
    m_getControlConfigurationSize(nullptr),
    m_getControlConfiguration(nullptr),
    m_getControlConfigurationChanged(nullptr),
    m_initializeConfiguration(nullptr),

    m_getCaptureShotScenarioState(nullptr),
    m_ilwalidateCaptureShotScenarioState(nullptr),
    m_ilwalidateAllState(nullptr),

    m_reportServerControlVersion(nullptr),
    m_setLastCaptureUtf8Path(nullptr)
{
    m_DLLfound = false;
    m_anselControlSdkVersion = 0ull;

    m_shotDescToTake.shotType = ShotType::kNone;
}

/* A group of getters */
bool AnselControlSDKState::isConfigured() const { return m_configuration != nullptr; }
bool AnselControlSDKState::isDetected() const { return m_DLLfound; }

void AnselControlSDKState::checkReportReadiness()
{
    if (!isDetected())
        return;

    if (m_readinessReported)
        return;

    if (m_configuration && m_configuration->readyCallback)
    {
        m_configuration->readyCallback(m_configuration->userPointer);
        m_readinessReported = true;
    }
}
void AnselControlSDKState::checkGetControlConfiguration()
{
    if (!isDetected())
        return;

    if (!m_getControlConfigurationChanged())
    {
        return;
    }

    // Choose maximum size for Configuration and allocate that much
    const auto configurationSizeAnselControlSDKVersion = m_getControlConfigurationSize();
    const auto configurationSizeLwCameraVersion = static_cast<uint32_t>(sizeof(ansel::Configuration));
    m_configurationStorage = std::make_unique<char[]>(std::max(configurationSizeAnselControlSDKVersion, configurationSizeLwCameraVersion));
    // allocate and default initialize Configuration object using its storage
    // we select between two ways of initializing Configuration object - plain Configuration default ctor or
    // initializeConfiguration function provided by the Ansel Control SDK, which can initialize the Configuration object better
    // in case Ansel Control SDK's version thinks Configuration structure is larger than LwCamera think it is.
    // This should mean that LwCamera is older than Ansel Control SDK.
    if (configurationSizeAnselControlSDKVersion <= configurationSizeLwCameraVersion || !m_initializeConfiguration)
        m_configuration = new (&m_configurationStorage[0]) anselcontrol::Configuration();
    else
        m_initializeConfiguration(*m_configuration);


    // Ansel Control SDK fills in our configuration object with what it has configured
    m_getControlConfiguration(*m_configuration);
}

HMODULE AnselControlSDKState::findAnselControlSDKModule()
{
    const std::array<std::wstring, 6> anselSDKnames = {
#if _M_AMD64
        _T("AnselControlSDK64.dll"),
        _T("AnselControlSDK64d.dll"),
#else
        _T("AnselControlSDK32.dll"),
        _T("AnselControlSDK32d.dll"),
#endif
    };

    HMODULE hAnselSDK = NULL;

    for (auto& moduleName : anselSDKnames)
        if (hAnselSDK = GetModuleHandle(moduleName.c_str()))
            break;

    return hAnselSDK;
}

AnselControlSDKState::DetectionStatus AnselControlSDKState::detectAndInitializeAnselControlSDK()
{
    DetectionStatus status = DetectionStatus::kSUCCESS;

    // If DLL has already been detected and initialized we don't repeat that process here
    if (isDetected())
        return status;

    HMODULE hAnselControlSDK = findAnselControlSDKModule();
    if (hAnselControlSDK)
    {
        LOG_INFO("AnselSDK DLL found in process");
        m_DLLfound = true;

        if (!(m_getControlConfiguration = (PFNGETCONFIGURATIONFUNC)GetProcAddress(hAnselControlSDK, "getControlConfiguration")))
            m_DLLfound = false;
        if (!(m_getControlConfigurationChanged = (PFNBOOLFUNC)GetProcAddress(hAnselControlSDK, "getControlConfigurationChanged")))
            m_DLLfound = false;
        if (!(m_getControlConfigurationSize = (PFNGETCONFIGURATIONSIZE)GetProcAddress(hAnselControlSDK, "getControlConfigurationSize")))
            m_DLLfound = false;
        if (!(m_getCaptureShotScenarioState = (PFNGETCAPTURESHOTSTATE)GetProcAddress(hAnselControlSDK, "getCaptureShotScenarioState")))
            m_DLLfound = false;
        if (!(m_ilwalidateCaptureShotScenarioState = (PFNBOOLFUNC)GetProcAddress(hAnselControlSDK, "ilwalidateCaptureShotScenarioState")))
            m_DLLfound = false;
        if (!(m_ilwalidateAllState = (PFNBOOLFUNC)GetProcAddress(hAnselControlSDK, "ilwalidateAllState")))
            m_DLLfound = false;

        if (!(m_reportServerControlVersion = (PFNREPORTSERVERCONTROLVERSIONFUNC)GetProcAddress(hAnselControlSDK, "reportServerControlVersion")))
            m_DLLfound = false;
        if (!(m_setLastCaptureUtf8Path = (PFNSETLASTCAPTUREUTF8PATHFUNC)GetProcAddress(hAnselControlSDK, "setLastCaptureUtf8Path")))
            m_DLLfound = false;

        if (!m_DLLfound)
        {
            LOG_INFO("Some of the required functions cannot be found in AnselSDK");
        }

        if (!(m_getVersion = (PFNGETVERSIONFUNC)GetProcAddress(hAnselControlSDK, "getVersion")))
        {
            m_anselControlSdkVersion = 0ull;
            LOG_INFO("AnselControlSDK getVersion function is unavailable");
        }
        else
        {
            m_anselControlSdkVersion = m_getVersion();
            const auto major = getAnselControlSdkVersionMajor();
            const auto minor = getAnselControlSdkVersionMinor();
            const auto commit = getAnselControlSdkVersionCommit();
            LOG_INFO("AnselControlSDK version is %d.%d.%08x", major, minor, commit);

            // perform the version cross check
            // Do not try to work with Ansel SDK that is newer than what this LwCamera support
            // This check was introduced when Ansel SDK is at 0.19
            // major api required, minor api change backwards

            if ( major < ANSEL_CONTROL_SDK_PRODUCT_VERSION_MAJOR ||
                (major == ANSEL_CONTROL_SDK_PRODUCT_VERSION_MAJOR &&
                minor <= ANSEL_CONTROL_SDK_PRODUCT_VERSION_MINOR))
            {
                // it's fine
            }
            else
            {
                m_DLLfound = false;
                status = DetectionStatus::kDRIVER_API_MISMATCH;
                LOG_INFO("AnselSDK version is newer than what the driver supports (%d.%d vs. %d.%d)", 
                    major, minor, ANSEL_CONTROL_SDK_PRODUCT_VERSION_MAJOR, ANSEL_CONTROL_SDK_PRODUCT_VERSION_MINOR);
            }
        }

        // these two functions allow us to initialize Configuration and SessionConfiguration on the AnselSDK side.
        // this is needed to fully initialize the object in case AnselSDK's version of either of those is larger
        // that what LwCamera thinks they are
        m_initializeConfiguration = (PFNINITIALIZECONFIGURATIONFUNC)GetProcAddress(hAnselControlSDK, "initializeControlConfiguration");
    }
    else
    {
        status = DetectionStatus::kDLL_NOT_FOUND;
        static bool wasWarningReported = false;
        if (!wasWarningReported)
        {
            LOG_VERBOSE("AnselControlSDK DLL *not* found in process");
            wasWarningReported = true;
        }
    }

    if (status == DetectionStatus::kSUCCESS)
    {
        m_reportServerControlVersion(ANSEL_CONTROL_SDK_VERSION);
        checkGetControlConfiguration();
        checkReportReadiness();
    }

    return status;
}

uint32_t AnselControlSDKState::getAnselControlSdkVersionMajor() const 
{ 
    // detect old versioning scheme that was in use before 0.15:
    // Version format is XXX.YYY.TIMESTAMP, where TIMESTAMP is:
    // YYYY.MM.DD.HH.MM.SS

    // check if it's the old timestamp based versioning scheme
    uint32_t major = 0, minor = 0;
    if (checkAnselControlSdkTimestampBasedVersion(m_anselControlSdkVersion, major, minor))
        return major;

    return (m_anselControlSdkVersion & 0xFFFF000000000000ull) >> 48; 
}

uint32_t AnselControlSDKState::getAnselControlSdkVersionMinor() const 
{ 
    // detect old versioning scheme that was in use before 0.15:
    // Version format is XXX.YYY.TIMESTAMP, where TIMESTAMP is:
    // YYYY.MM.DD.HH.MM.SS
    uint32_t major = 0, minor = 0;
    if (checkAnselControlSdkTimestampBasedVersion(m_anselControlSdkVersion, major, minor))
        return minor;

    return (m_anselControlSdkVersion & 0x0000FFFF00000000ull) >> 32;
}

uint32_t AnselControlSDKState::getAnselControlSdkVersionCommit() const 
{ 
    // detect old versioning scheme that was in use before 0.15 and return 0 as commit hash (we used timestamp before):
    // Version format is XXX.YYY.TIMESTAMP, where TIMESTAMP is:
    // YYYY.MM.DD.HH.MM.SS
    uint32_t major = 0, minor = 0;
    if (checkAnselControlSdkTimestampBasedVersion(m_anselControlSdkVersion, major, minor))
        return 0;

    return m_anselControlSdkVersion & 0x00000000FFFFFFFFull;
}

bool AnselControlSDKState::isSDKDetectedAndSessionActive()
{
    if (!isDetected()) return false;

    return true;
}

void AnselControlSDKState::onInputEvent(const input::InputEvent& ev, const input::MomentaryKeyboardState& kbdSt,
    const input::MomentaryMouseState& mouseSt,
    const input::MomentaryGamepadState& gpadSt,
    const input::FolwsChecker& folwsChecker,
    const input::MouseTrapper& mouseTrapper) 
{
    if (!isDetected())
        return;
}

void AnselControlSDKState::processInput(
        input::InputState * inputCapture,
        bool isCameraInteractive
        )
{
    if (!isDetected())
        return;

    // TODO: additional check here
    const float gamepadStickBacklashGap = 0.2f;
}

void AnselControlSDKState::initCaptureShotScenario()
{
    if (!isDetected())
        return;

    captureShotScenarioState.m_controlState = ControlState::kPrestart;
}

anselcontrol::ShotType colwertShotTypesToControlShotType(const ShotType & shotType)
{
    switch (shotType)
    {
        case ShotType::k360:
            return anselcontrol::ShotType::kShotType360Mono;
        case ShotType::k360Stereo:
            return anselcontrol::ShotType::kShotType360Stereo;
        case ShotType::kHighRes:
            return anselcontrol::ShotType::kShotTypeSuperResolution;
        case ShotType::kStereo:
            return anselcontrol::ShotType::kShotTypeStereo;
        case ShotType::kRegular:
            return anselcontrol::ShotType::kShotTypeRegular;
        default:
        {
            // TODO: throw an error
            return anselcontrol::ShotType::kShotTypeRegular;
        }
    }
}

ShotType colwertControlShotTypesToShotType(const anselcontrol::ShotType & controlShotType)
{
    switch (controlShotType)
    {
        case anselcontrol::ShotType::kShotType360Mono:
            return ShotType::k360;
        case anselcontrol::ShotType::kShotType360Stereo:
            return ShotType::k360Stereo;
        case anselcontrol::ShotType::kShotTypeSuperResolution:
            return ShotType::kHighRes;
        case anselcontrol::ShotType::kShotTypeStereo:
            return ShotType::kStereo;
        case anselcontrol::ShotType::kShotTypeRegular:
            return ShotType::kRegular;
        default:
        {
            // TODO: throw an error
            return ShotType::kNone;
        }
    }
}

void AnselControlSDKState::updateCaptureShotScenario()
{
    if (!isDetected())
        return;

    switch (captureShotScenarioState.m_controlState)
    {
    case ControlState::kPrestart:
        {
            m_anselPrestartRequested = true;
            break;
        }
    case ControlState::kStart:
        {
            m_anselStartRequested = true;
            break;
        }
    case ControlState::kCapture:
        {
            if (captureShotScenarioState.m_shotState.shotDesc.shotType == anselcontrol::ShotType::kShotTypeRegular || m_captureCameraInitialized)
            {
                if (!m_captureStarted)
                {
                    m_shotDescToTake.shotType = colwertControlShotTypesToShotType(captureShotScenarioState.m_shotState.shotDesc.shotType);
                    m_shotDescToTake.resolution360 = (uint64_t)captureShotScenarioState.m_shotState.shotDesc.sphericalResolution;
                    m_shotDescToTake.highResMult = (uint32_t)captureShotScenarioState.m_shotState.shotDesc.superresMult;

                    m_captureInProgress = true;
                    m_captureStarted = true;
                }
                if (m_captureStarted && !m_captureInProgress)
                {
                    captureShotScenarioState.m_controlState = ControlState::kStop;
                }
            }
            break;
        }
    case ControlState::kStop:
        {
            m_captureStarted = false;
            m_anselStopRequested = true;
            break;
        }
    case ControlState::kPoststop:
        {
            m_anselPoststopRequested = true;
            break;
        }
    case ControlState::kNone:
        {
            m_lwrrentScenario = ScenarioType::kNone;
            m_needsControl = false;
            break;
        }
    }
}

AnselControlSDKUpdateParameters AnselControlSDKState::updateControlState()
{
    AnselControlSDKUpdateParameters outputParameters = { 0 };

    if (!isDetected())
        return outputParameters;

    checkReportReadiness();
    checkGetControlConfiguration();

    bool actionRequested = false;

    anselcontrol::CaptureShotState captureShotState;
    
    if (m_lwrrentScenario == ScenarioType::kNone)
    {
        m_getCaptureShotScenarioState(captureShotState);
        if (captureShotState.isValid)
        {
            m_lwrrentScenario = ScenarioType::kCaptureShot;
            actionRequested = true;
            initCaptureShotScenario();
            captureShotScenarioState.m_shotState = captureShotState;
            m_ilwalidateCaptureShotScenarioState();
        }
    }
    else
    {
        // We're not accepting any requests while we're performing scenario
        m_ilwalidateCaptureShotScenarioState();
    }

    if (actionRequested)
    {
        m_needsControl = true;
    }

    if (m_needsControl)
    {
        switch (m_lwrrentScenario)
        {
            case ScenarioType::kCaptureShot:
            {
                updateCaptureShotScenario();
                break;
            }
            default:
                break;
        }
    }

    return outputParameters;
}
