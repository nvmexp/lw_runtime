#define NOMINMAX
#include <Windows.h>
#include <shlobj.h>
#include <Shlwapi.h>
#include <tchar.h>
#include <array>
#include <sstream>
#include <algorithm>

#include "ansel/Version.h"
#include "anselutils/CameraControllerFreeHybrid.h"
#include "anselutils/CameraControllerFreeGlobalRotation.h"
#include "anselutils/Utils.h"
#include "CommonStructs.h"
#include "Utils.h"
#include "AnselSDKState.h"
#include "UI.h"
#include "Config.h"
#include "Log.h"
#include "drs/LwDrsWrapper.h"
#include "darkroom/StringColwersion.h"
#include "i18n/LocalizedStringHelper.h"
#include "i18n/text.en-US.h"

namespace
{
#define SAFE_DELETE(pObject) { if(pObject) { delete pObject; (pObject) = nullptr; } }
    // The whole Session object is now a monostate of AnselSDKState
    std::unique_ptr<anselutils::Session> s_anselSession;

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

    void updateCameraForSDKLessThan15(ansel::Camera& cam)
    {
        auto copyCameraWithoutNearFarPlanes = [](const ansel::Camera& in, ansel::Camera * outCamera)
        {
            if (outCamera == nullptr)
                return;
            // These are all the field that ansel::Camera had before Ansel SDK 1.5
            outCamera->position = in.position;
            outCamera->rotation = in.rotation;
            outCamera->fov = in.fov;
            outCamera->projectionOffsetX = in.projectionOffsetX;
            outCamera->projectionOffsetY = in.projectionOffsetY;
        };

        if (s_anselSession)
        {
            // Ansel SDK 1.5 adds three new parameters, near and far plane values and aspect ratio
            // to the central ansel::Camera object
            // In order to be backwards compatible and because the game Camera object is passed by reference
            // all the way down the stack to the camera controller, we upgrade Camera object here, so that all deeper
            // levels could access the two new parameters legally, without risking to overwrite the game stack space
            // This new 'camera' object here should be of the proper size (larger by three floats) with near, far plane
            // values and aspect ratio reset to zero.
            // We pay with an additional copy
            ansel::Camera camera;
            copyCameraWithoutNearFarPlanes(cam, &camera);
            camera.nearPlane = 0.0f;
            camera.farPlane = 0.0f;
            camera.aspectRatio = 0.0f;
            s_anselSession->updateCamera(camera);
            copyCameraWithoutNearFarPlanes(camera, &cam);
        }
    }

    void updateCamera(ansel::Camera& cam)
    {
        if (s_anselSession)
            s_anselSession->updateCamera(cam);
    }

    bool restoreGameCamera(ansel::Camera& cam)
    {
        wchar_t appPath[APP_PATH_MAXLEN];
        GetModuleFileName(NULL, appPath, APP_PATH_MAXLEN);
        const auto* filepart = PathFindFileName(appPath);
        if (*filepart != L'\0')
        {
            HKEY key = 0;
            constexpr size_t bufferSize = 256;
            char buf[bufferSize];
            unsigned long bufSize = bufferSize;
            const std::string keyName = std::string("SavedCamera_") + darkroom::getUtf8FromWstr(filepart);
            if (RegGetValueA(HKEY_LWRRENT_USER, "SOFTWARE\\LWPU Corporation\\Ansel", keyName.c_str(), RRF_RT_ANY, nullptr, buf, &bufSize) == ERROR_SUCCESS)
            {
                OutputDebugStringA(buf);
                std::istringstream iss(buf);
                iss >> cam.position.x >> cam.position.y >> cam.position.z >> cam.rotation.x >> cam.rotation.y >> cam.rotation.z >> cam.rotation.w >> cam.fov;
                return true;
            }
        }
        return false;
    }

    void saveGameCamera(const ansel::Camera& cam)
    {
        wchar_t appPath[APP_PATH_MAXLEN];
        GetModuleFileName(NULL, appPath, APP_PATH_MAXLEN);
        const auto* filepart = PathFindFileName(appPath);
        if (*filepart != L'\0')
        {
            HKEY key = 0;
            const std::string keyName = std::string("SavedCamera_") + darkroom::getUtf8FromWstr(filepart);

            if (RegCreateKeyEx(HKEY_LWRRENT_USER, _T("SOFTWARE\\LWPU Corporation\\Ansel"), 0, _T(""), 0, KEY_WRITE, NULL, &key, NULL) == ERROR_SUCCESS)
            {
                std::stringstream oss;
                oss << cam.position.x << " " << cam.position.y << " " << cam.position.z << " " <<
                    cam.rotation.x << " " << cam.rotation.y << " " << cam.rotation.z << " " << cam.rotation.w << " " << cam.fov;
                const auto camString = oss.str();
                RegSetValueExA(key, keyName.c_str(), 0, REG_SZ, reinterpret_cast<const BYTE*>(camString.c_str()), DWORD(camString.size() + 1));
            }
        }
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

    bool checkAnselSdkTimestampBasedVersion(const uint64_t sdkVersion, uint32_t& major, uint32_t& minor)
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

AnselSDKState::AnselSDKState() :
    m_anselCameraControllerFree(nullptr),
    m_anselSavedController(nullptr),
    m_anselCameraControllerRelative(nullptr),
    m_configuration(nullptr),
    m_sessionConfiguration(nullptr),
    m_getVersion(nullptr),
    m_getConfigurationSize(nullptr),
    m_getSessionConfigurationSize(nullptr),
    m_setSessionFunctions(nullptr),
    m_setUpdateCameraFunc(nullptr),
    m_getConfigurationPfn(nullptr),
    m_clearBufferBindHint(nullptr),
    m_clearBufferFinishedHint(nullptr),
    m_getBufferBindHintActive(nullptr),
    m_getBufferFinishedHintActive(nullptr),
    m_clearBufferBindHint13(nullptr),
    m_clearBufferFinishedHint13(nullptr),
    m_getBufferBindHintActive13(nullptr),
    m_getBufferFinishedHintActive13(nullptr),
    m_getBufferBindHintActive017(nullptr),
    m_getBufferFinishedHintActive017(nullptr),
    m_initializeConfiguration(nullptr),
    m_initializeSessionConfiguration(nullptr)
{
    m_camFOV = 0.0f;
    m_hdrHintsSupported = false;
    m_needsFOVupdate = false;
    m_needsDarkroomProcInit = true;
    m_captureState = CAPTURE_NOT_STARTED;
    m_captureOffsetWait = 0;
    m_DLLfound = false;
    m_progressTotal = 0;
    m_progressLwrrent = 0;
    m_allowColwertToJPG = true;
    m_anselSdkVersion = 0ull;
}

/* A group of getters */
ansel::Camera AnselSDKState::getDisplayCamera() const
{
    ansel::Camera camera = {0};
    if (s_anselSession)
        camera = s_anselSession->getDisplayCamera();
    return camera;
}

ansel::Camera AnselSDKState::getOriginalCamera() const
{
    ansel::Camera camera = { 0 };
    if (s_anselSession)
        camera = s_anselSession->getOriginalCamera();
    return camera;
}

bool AnselSDKState::isCameraChanged() const
{
    return m_isCameraMoving;
}

bool AnselSDKState::isConfigured() const { return m_configuration && m_configuration->startSessionCallback != nullptr; }
bool AnselSDKState::isDetected() const { return m_DLLfound; }
bool AnselSDKState::isSessionActive() const { return s_anselSession != nullptr; }
bool AnselSDKState::isHdrHintsSupported() const { return m_hdrHintsSupported; }
bool AnselSDKState::isCameraInitialized() const { return isSessionActive() && s_anselSession->isDisplayCameraInitialized(); }
const ansel::Configuration& AnselSDKState::getConfiguration() const { return *m_configuration; }
const ansel::SessionConfiguration& AnselSDKState::getSessionConfiguration() const { return *m_sessionConfiguration; }

int AnselSDKState::getCaptureLatency() const { return m_cfgShotCaptureLatency; }
int AnselSDKState::getSettleLatency() const { return m_cfgShotSettleLatency; }
int AnselSDKState::getProgressLwrrent() const { return m_progressLwrrent; }
int AnselSDKState::getProgressTotal() const { return m_progressTotal; }
int AnselSDKState::getCaptureState() const { return m_captureState; }
void AnselSDKState::setCaptureState(int captureState) { m_captureState = captureState; }
void AnselSDKState::setCaptureLatency(int captureLatency) { m_cfgShotCaptureLatency = captureLatency; }
void AnselSDKState::setSettleLatency(int settleLatency) { m_cfgShotSettleLatency = settleLatency; }
void AnselSDKState::setProgressLwrrent(int progressLwrrent) { m_progressLwrrent = progressLwrrent; }
void AnselSDKState::setProgressTotal(int progressTotal) {m_progressTotal = progressTotal;  }

void AnselSDKState::abortCapture() { m_captureState = CAPTURE_ABORT; }
void AnselSDKState::resetCaptureState() { m_captureState = CAPTURE_NOT_STARTED; }

anselutils::CameraControllerFree* AnselSDKState::getCameraControllerFree() { return m_anselCameraControllerFree; }

bool AnselSDKState::isUserControlUpdated() const
{
    if (m_isUserControlDescListDirty)
        return m_isUserControlDescListDirty();

    // in case the callback is nullptr, we report no changes, hence UI shouldn't bother about any user controls
    return false;
}

void AnselSDKState::clearUserControlUpdated() const
{
    if (m_clearUserControlDescListDirtyFlag)
        m_clearUserControlDescListDirtyFlag();
}

std::vector<AnselUserControlDesc> AnselSDKState::getUserControlDescriptions() const
{
    std::vector<AnselUserControlDesc> descs;

    // Remove high quality game setting toggle if changeQualityCallback is undefined.
    // We have to do it this way because we want to make sure the high quality game setting is the first setting
    // added, and settings may be added before Ansel configuration, and thus the changeQualityCallback is defined.
    if (!m_configuration->changeQualityCallback && m_removeUserControl)
    {
        m_removeUserControl(HIGH_QUALITY_CONTROL_ID);
    }

    // always return empty descriptions list, unless we have all the callbacks set
    if (m_lockUserControlDescriptions &&
        m_unlockUserControlDescriptions &&
        m_getUserControlDescription &&
        m_getUserControlDescriptionsSize)
    {
        const uint32_t descsCount = m_getUserControlDescriptionsSize();
        m_lockUserControlDescriptions();
        for (uint32_t i = 0u; i < descsCount; ++i)
        {
            ansel::UserControlType userControlType;
            void* userControlValue;
            const char** labels = nullptr;
            uint32_t userControlId;
            uint32_t labelsCount = 0u;
            m_getUserControlDescription(i, userControlType, labels, labelsCount, userControlValue, userControlId);
            AnselUserControlDesc desc;
            // 'labels' is an array of strings that looks like this: {'en-US', 'English label', 'es-ES', 'Etiqueta inglesa', ... }
            // in case we receive odd amount of labels, we only go through full pairs
            if (labels)
                for (uint32_t k = 0u; k < labelsCount >> 1; ++k)
                    desc.labels.push_back({ labels[k << 1], labels[(k << 1) + 1] });
            desc.info.userControlId = userControlId;
            desc.info.userControlType = userControlType;
            desc.info.value = userControlValue;
            descs.push_back(desc);
        }
        m_unlockUserControlDescriptions();
    }

    return descs;
}

void AnselSDKState::userControlChanged(uint32_t id, const void* value) const
{
    if (m_userControlValueChanged)
        m_userControlValueChanged(id, value);
}

void AnselSDKState::getTelemetryData(AnselSDKStateTelemetry& telemetry) const
{
    telemetry.usedGamepadForCameraDuringTheSession = m_usedGamepadForCameraDuringTheSession;
}

void AnselSDKState::initTitleAndDrsNames()
{
    if (m_titleForFileNaming.empty())
    {
        // First try to get a profile name:
        wchar_t appPath[APP_PATH_MAXLEN];
        appPath[0] = '\0'; // null terminating buffer at start
        DWORD length = GetModuleFileNameW(NULL, appPath, APP_PATH_MAXLEN);

        // Buffer should always be large enough but we might as well check so that the code
        // is robust
        if (length < APP_PATH_MAXLEN)
        {
            if (drs::getProfileName(m_drsProfileName))
            {
                m_drsAppName = PathFindFileName(appPath);
                m_titleForFileNaming = m_drsProfileName;
                justifyProfileName((wchar_t*)m_titleForFileNaming.c_str());
                m_titleForTaggingUtf8 = darkroom::getUtf8FromWstr(m_drsProfileName);
            }
            // Fall back to titleNameUtf8 specified by integration
            else if (m_configuration && m_configuration->titleNameUtf8)
            {
                m_titleForTaggingUtf8 = m_configuration->titleNameUtf8;
                m_titleForFileNaming = darkroom::getWstrFromUtf8(m_titleForTaggingUtf8);
                justifyProfileName((wchar_t*)m_titleForFileNaming.c_str());
            }
            // Finally fall back to exe name
            else
            {
                const wchar_t* fileNameInPath = PathFindFileNameW(appPath);
                const wchar_t* fileNameExtension = PathFindExtensionW(appPath);
                std::wstring exeName;
                if (fileNameExtension && fileNameInPath)
                    exeName = std::wstring(fileNameInPath, fileNameExtension);

                m_titleForTaggingUtf8 = darkroom::getUtf8FromWstr(exeName);
                m_titleForFileNaming = exeName;
            }
        }
    }
}

HMODULE AnselSDKState::findAnselSDKModule()
{
    const std::array<std::wstring, 6> anselSDKnames = {
#if _M_AMD64
        _T("AnselSDK64.dll"),
        _T("AnselSDK64d.dll"),
        // fall back to older version (for temporary backwards compatibility)
        _T("LwCameraSDK64.dll"),
        _T("LwCameraSDK64d.dll"),
        _T("LwCameraSDK.dll"),
        _T("LwCameraSDKd.dll"),
#else
        _T("AnselSDK32.dll"),
        _T("AnselSDK32d.dll"),
        // fall back to older version (for temporary backwards compatibility)
        _T("LwCameraSDK32.dll"),
        _T("LwCameraSDK32d.dll"),
        _T("LwCameraSDK.dll"),
        _T("LwCameraSDKd.dll"),
#endif
    };

    HMODULE hAnselSDK = NULL;

    for (auto& moduleName : anselSDKnames)
        if (hAnselSDK = GetModuleHandle(moduleName.c_str()))
            break;

    return hAnselSDK;
}

void AnselSDKState::setSessionData(SessionFunc sessionStart, SessionFunc sessionStop, void * sessionUserData)
{
    if (m_DLLfound && m_setSessionFunctions)
        m_setSessionFunctions(sessionStart, sessionStop, sessionUserData);
}

AnselSDKState::DetectionStatus AnselSDKState::detectAndInitializeAnselSDK(const std::wstring& installationPath,
    const std::wstring& intermediatePath, const std::wstring& snapshotPath,
    SessionFunc sessionStart, SessionFunc sessionStop, BufferFinishedCallback bufferFinished, void* sessionUserData)
{
    DetectionStatus status = DetectionStatus::kSUCCESS;

    // If DLL has already been detected and initialized we don't repeat that process here
    if (isDetected())
        return status;

    HMODULE hAnselSDK = findAnselSDKModule();
    if (hAnselSDK)
    {
        LOG_INFO("AnselSDK DLL found in process");
        m_DLLfound = true;
        m_hdrHintsSupported = true;

        if (!(m_setUpdateCameraFunc = (PFNCWSETUPDATECAMERAFUNC)GetProcAddress(hAnselSDK, "setUpdateCameraFunc")))
            m_DLLfound = false;
        if (!(m_getConfigurationPfn = (PFNCWGETCONFIGURATIONFUNC)GetProcAddress(hAnselSDK, "getConfiguration")))
            m_DLLfound = false;
        if (!(m_getConfigurationSize = (PFNCWGETCONFIGURATIONSIZE)GetProcAddress(hAnselSDK, "getConfigurationSize")))
            m_DLLfound = false;
        if (!(m_getSessionConfigurationSize = (PFNCWGETSESSIONCONFIGURATIONSIZE)GetProcAddress(hAnselSDK, "getSessionConfigurationSize")))
            m_DLLfound = false;

        // These SDK Internal functions can be NULL:
        m_addUserControl = (PFNADDUSERCONTROLFUNC)GetProcAddress(hAnselSDK, "addUserControl_Internal");
        m_setUserControlLabelLocalization = (PFNSETUSERCONTROLLABELLOCALIZATIONFUNC)GetProcAddress(hAnselSDK, "setUserControlLabelLocalization_Internal");
        m_removeUserControl = (PFNREMOVEUSERCONTROLFUNC)GetProcAddress(hAnselSDK, "removeUserControl_Internal");
        m_getUserControlValue = (PFNGETUSERCONTROLVALUEFUNC)GetProcAddress(hAnselSDK, "getUserControlValue_Internal");
        m_setUserControlValue = (PFNSETUSERCONTROLVALUEFUNC)GetProcAddress(hAnselSDK, "setUserControlValue_Internal");

        if (!m_DLLfound)
        {
            LOG_INFO("Some of the required functions cannot be found in AnselSDK");
        }

        // User control APIs (introduced in Ansel SDK 1.2)
        m_getUserControlDescriptionsSize = (PFNGETUSERCONTROLDESCRIPTIONSIZEFUNC)GetProcAddress(hAnselSDK, "getUserControlDescriptionsSize");
        m_lockUserControlDescriptions = (PFLWOIDFUNC)GetProcAddress(hAnselSDK, "lockUserControlDescriptions");
        m_unlockUserControlDescriptions = (PFLWOIDFUNC)GetProcAddress(hAnselSDK, "unlockUserControlDescriptions");
        m_clearUserControlDescListDirtyFlag = (PFLWOIDFUNC)GetProcAddress(hAnselSDK, "clearUserControlDescListDirtyFlag");
        m_isUserControlDescListDirty = (PFNBOOLFUNC)GetProcAddress(hAnselSDK, "isUserControlDescListDirty");
        m_userControlValueChanged = (PFNUSERCONTROLVALUECHANGEDFUNC)GetProcAddress(hAnselSDK, "userControlValueChanged");
        m_getUserControlDescription = (PFNGETUSERCONTROLDESCRIPTIONFUNC)GetProcAddress(hAnselSDK, "getUserControlDescription");

        if (!(m_getVersion = (PFNGETVERSIONFUNC)GetProcAddress(hAnselSDK, "getVersion")))
        {
            m_anselSdkVersion = 0ull;
            LOG_INFO("AnselSDK getVersion function is unavailable");
        }
        else
        {
            m_anselSdkVersion = m_getVersion();
            const auto major = getAnselSdkVersionMajor();
            const auto minor = getAnselSdkVersionMinor();
            const auto commit = getAnselSdkVersionCommit();
            LOG_INFO("AnselSDK version is %d.%d.%08x", major, minor, commit);

            // perform the version cross check
            // Do not try to work with Ansel SDK that is newer than what this LwCamera support
            // This check was introduced when Ansel SDK is at 0.19
            // major api required, minor api change backwards

            if ( major < ANSEL_SDK_PRODUCT_VERSION_MAJOR ||
                (major == ANSEL_SDK_PRODUCT_VERSION_MAJOR &&
                minor <= ANSEL_SDK_PRODUCT_VERSION_MINOR))
            {
                // it's fine
            }
            else
            {
                m_DLLfound = false;
                status = DetectionStatus::kDRIVER_API_MISMATCH;
                LOG_INFO("AnselSDK version is newer than what the driver supports (%d.%d vs. %d.%d)",
                    major, minor, ANSEL_SDK_PRODUCT_VERSION_MAJOR, ANSEL_SDK_PRODUCT_VERSION_MINOR);
            }

            // HDR hints (introduced in Ansel SDK 0.13)
            FARPROC getBufferBindHintActive = GetProcAddress(hAnselSDK, "getHdrBufferBindHintActive");
            if (!getBufferBindHintActive)
                getBufferBindHintActive = GetProcAddress(hAnselSDK, "getBufferBindHintActive");

            FARPROC getBufferFinishedHintActive = GetProcAddress(hAnselSDK, "getHdrBufferFinishedHintActive");
            if (!getBufferFinishedHintActive)
                getBufferFinishedHintActive = GetProcAddress(hAnselSDK, "getBufferFinishedHintActive");

            FARPROC clearBufferBindHint = GetProcAddress(hAnselSDK, "clearHdrBufferBindHint");
            if (!clearBufferBindHint)
                clearBufferBindHint = GetProcAddress(hAnselSDK, "clearBufferBindHint");

            FARPROC clearBufferFinishedHint = GetProcAddress(hAnselSDK, "clearHdrBufferFinishedHint");
            if (!clearBufferFinishedHint)
                clearBufferFinishedHint = GetProcAddress(hAnselSDK, "clearBufferFinishedHint");

            if (!getBufferBindHintActive || !getBufferFinishedHintActive ||
                !clearBufferBindHint || !clearBufferFinishedHint)
                m_hdrHintsSupported = false;

            if (m_hdrHintsSupported)
            {
                // Hints API was changed in 0.18 (0.18 adds support for threading and copy modes)
                if (major == 0 && minor < 18)
                {
                    m_getBufferBindHintActive017 = (PFNBOOLFUNC)getBufferBindHintActive;
                    m_getBufferFinishedHintActive017 = (PFNBOOLFUNC)getBufferFinishedHintActive;
                    m_clearBufferBindHint13 = (PFLWOIDFUNC)clearBufferBindHint;
                    m_clearBufferFinishedHint13 = (PFLWOIDFUNC)clearBufferFinishedHint;
                }
                // then again in Ansel SDK 1.4 - supporting multiple buffer types (HDR, Depth, HUDless, etc)
                else if (major < 1 || (major == 1 && minor <= 3))
                {
                    m_getBufferBindHintActive13 = (PFNBUFFERBINDHINTFUNC13)getBufferBindHintActive;
                    m_getBufferFinishedHintActive13 = (PFNBUFFERFINISHEDHINTFUNC13)getBufferFinishedHintActive;
                    m_clearBufferBindHint13 = (PFLWOIDFUNC)clearBufferBindHint;
                    m_clearBufferFinishedHint13 = (PFLWOIDFUNC)clearBufferFinishedHint;
                }
                else
                {
                    m_getBufferBindHintActive = (PFNBUFFERBINDHINTFUNC)getBufferBindHintActive;
                    m_getBufferFinishedHintActive = (PFNBUFFERFINISHEDHINTFUNC)getBufferFinishedHintActive;
                    m_clearBufferBindHint = (PFNCLEARHINTFUNC)clearBufferBindHint;
                    m_clearBufferFinishedHint = (PFNCLEARHINTFUNC)clearBufferFinishedHint;
                }
            }
        }

        // these two functions allow us to initialize Configuration and SessionConfiguration on the AnselSDK side.
        // this is needed to fully initialize the object in case AnselSDK's version of either of those is larger
        // that what LwCamera thinks they are
        m_initializeConfiguration = (PFNCWINITIALIZECONFIGURATIONFUNC)GetProcAddress(hAnselSDK, "initializeConfiguration");
        m_initializeSessionConfiguration = (PFNCWINITIALIZESESSIONCONFIGURATIONFUNC)GetProcAddress(hAnselSDK, "initializeSessionConfiguration");

        // SDK can start/stop session (introduced in Ansel SDK 0.15) - this will be nullptr for older games
        m_setSessionFunctions = (PFNCWSETSESSIONFUNCTIONS)GetProcAddress(hAnselSDK, "setSessionFunctions");
        setSessionData(sessionStart, sessionStop, sessionUserData);

        m_setBufferFinishedCallback = (PFNCWSETBUFFERFINISHEDCALLBACK)GetProcAddress(hAnselSDK, "setBufferFinishedCallback");
        if (m_setBufferFinishedCallback)
        {
            m_setBufferFinishedCallback(bufferFinished);
        }

        if (m_DLLfound && m_needsDarkroomProcInit)
        {
            const std::wstring toolsPath = installationPath + L"\\";
            std::wstring highres = toolsPath + L"HighresBlender64.exe";
            std::wstring equirect = toolsPath + L"SphericalEquirect64.exe";
            std::wstring mogrify = toolsPath + L"LwImageColwert64.exe";
            std::wstring thumbnailTool = toolsPath + L"ThumbnailTool64.exe";

#if defined(_M_IX86)
            // if we run LwCamera32.dll, but actually the system supports 64bit, use 64bit tools
            if (!IsWow64())
            {
                highres = toolsPath + L"HighresBlender32.exe";
                equirect = toolsPath + L"SphericalEquirect32.exe";
                mogrify = toolsPath + L"LwImageColwert32.exe";
                thumbnailTool = toolsPath + L"ThumbnailTool32.exe";
        }
#endif
            const std::wstring tempDir = intermediatePath;
            int retcode = darkroom::initializeJobProcessing(highres, equirect, mogrify, thumbnailTool, tempDir);

            m_needsDarkroomProcInit = false;
        }
    }
    else
    {
        status = DetectionStatus::kDLL_NOT_FOUND;
        LOG_INFO("AnselSDK DLL *not* found in process");
    }

    return status;
}

uint32_t AnselSDKState::getAnselSdkVersionMajor() const
{
    // detect old versioning scheme that was in use before 0.15:
    // Version format is XXX.YYY.TIMESTAMP, where TIMESTAMP is:
    // YYYY.MM.DD.HH.MM.SS

    // check if it's the old timestamp based versioning scheme
    uint32_t major = 0, minor = 0;
    if (checkAnselSdkTimestampBasedVersion(m_anselSdkVersion, major, minor))
        return major;

    return (m_anselSdkVersion & 0xFFFF000000000000ull) >> 48;
}

uint32_t AnselSDKState::getAnselSdkVersionMinor() const
{
    // detect old versioning scheme that was in use before 0.15:
    // Version format is XXX.YYY.TIMESTAMP, where TIMESTAMP is:
    // YYYY.MM.DD.HH.MM.SS
    uint32_t major = 0, minor = 0;
    if (checkAnselSdkTimestampBasedVersion(m_anselSdkVersion, major, minor))
        return minor;

    return (m_anselSdkVersion & 0x0000FFFF00000000ull) >> 32;
}

uint32_t AnselSDKState::getAnselSdkVersionCommit() const
{
    // detect old versioning scheme that was in use before 0.15 and return 0 as commit hash (we used timestamp before):
    // Version format is XXX.YYY.TIMESTAMP, where TIMESTAMP is:
    // YYYY.MM.DD.HH.MM.SS
    uint32_t major = 0, minor = 0;
    if (checkAnselSdkTimestampBasedVersion(m_anselSdkVersion, major, minor))
        return 0;

    return m_anselSdkVersion & 0x00000000FFFFFFFFull;
}

bool AnselSDKState::isBufferBindHintActive(ansel::BufferType bufferType, uint64_t& threadId, ansel::HintType& hintType) const
{
    if (isHdrHintsSupported())
    {
        if (m_getBufferBindHintActive017)
        {
            threadId = 0xFFFFFFFFFFFFFFFFull;
            hintType = ansel::kHintTypePreBind;
            return m_getBufferBindHintActive017();
        }
        else if (m_getBufferBindHintActive13 && bufferType == ansel::kBufferTypeHDR)
        {
            return m_getBufferBindHintActive13(threadId, hintType);
        }
        else if (m_getBufferBindHintActive)
        {
            return m_getBufferBindHintActive(bufferType, threadId, hintType);
        }
    }

    return false;
}

bool AnselSDKState::isBufferFinishedHintActive(ansel::BufferType bufferType, uint64_t& threadId) const
{
    if (isHdrHintsSupported())
    {
        if (m_getBufferFinishedHintActive017)
        {
            threadId = 0xFFFFFFFFFFFFFFFFull;
            return m_getBufferFinishedHintActive017();
        }
        else if (m_getBufferFinishedHintActive13 && bufferType == ansel::kBufferTypeHDR)
        {
            return m_getBufferFinishedHintActive13(threadId);
        }
        else if (m_getBufferFinishedHintActive)
            return m_getBufferFinishedHintActive(bufferType, threadId);
    }
    return false;
}

void AnselSDKState::clearBufferBindHint(ansel::BufferType bufferType)
{
    if (isHdrHintsSupported())
    {
        if (m_clearBufferBindHint13 && bufferType == ansel::kBufferTypeHDR)
            return m_clearBufferBindHint13();
        else if (m_clearBufferBindHint)
            return m_clearBufferBindHint(bufferType);
    }
}

void AnselSDKState::clearBufferFinishedHint(ansel::BufferType bufferType)
{
    if (isHdrHintsSupported())
    {
        if (m_clearBufferFinishedHint13 && bufferType == ansel::kBufferTypeHDR)
            return m_clearBufferFinishedHint13();
        else if (m_clearBufferFinishedHint)
            return m_clearBufferFinishedHint(bufferType);
    }
}

bool AnselSDKState::isSDKDetectedAndSessionActive()
{
    if (!isDetected()) return false;
    if (!isSessionActive()) return false;

    return true;
}

bool AnselSDKState::startSession(uint32_t width, uint32_t height, bool useHybridController)
{
    bool enhancedSessionAllowed = false;

    if (!isDetected()) return enhancedSessionAllowed;
    if (isSessionActive()) return enhancedSessionAllowed;

    // Given:
    // * Configuration/SessionConfiguration can only grow with time. We can't remove fields or change their meaning
    // Possible cases:
    // * AnselSDK is older than LwAnsel - AnselSDK!Configuration size < LwAnsel!Configuration size.
    //   works fine, LwAnsel allocates larger struct, AnselSDK fills in only fields it knows about.
    // * AnselSDK is newer than LwAnsel - AnselSDK!Configuration size > LwAnsel!Configuration size.
    //   LwAnsel should ask AnselSDK how large Configuration object is, allocate that much and create smaller object at this location
    //   AnselSDK than will fill all the fields of a larger object, writing beyond Configuration object LwAnsel created, but it wont crash
    //   because LwAnsel allocated as much storage as AnselSDK could possibly access with its larger object

    // Choose maximum size for Configuration and allocate that much
    const auto configurationSizeAnselSDKVersion = m_getConfigurationSize();
    const auto configurationSizeLwCameraVersion = static_cast<uint32_t>(sizeof(ansel::Configuration));
    m_configurationStorage = std::make_unique<char[]>(std::max(configurationSizeAnselSDKVersion, configurationSizeLwCameraVersion));
    // allocate and default initialize Configuration object using its storage
    // we select between two ways of initializing Configuration object - plain Configuration default ctor or
    // initializeConfiguration function provided by the Ansel SDK, which can initialize the Configuration object better
    // in case Ansel SDK's version thinks Configuration structure is larger than LwCamera think it is.
    // This should mean that LwCamera is older than Ansel SDK.
    if (configurationSizeAnselSDKVersion <= configurationSizeLwCameraVersion || !m_initializeConfiguration)
        m_configuration = new (&m_configurationStorage[0]) ansel::Configuration();
    else
        m_initializeConfiguration(*m_configuration);


    // AnselSDK fills in our configuration object with what it has configured
    m_getConfigurationPfn(*m_configuration);

    // if it wasn't configured yet, return false
    if (!isConfigured())
    {
        LOG_WARN(LogChannel::kAnselSdk, "SDK has not been configured by application yet! Cannot start session.");
        return enhancedSessionAllowed;
    }

    const char* trueStr = "true";
    const char* falseStr = "false";

    LOG_INFO(LogChannel::kAnselSdk, "Configuration:");
    LOG_INFO(LogChannel::kAnselSdk, ">\ttitleNameUtf8: %s", m_configuration->titleNameUtf8 ? m_configuration->titleNameUtf8 : "(null)");
    LOG_INFO(LogChannel::kAnselSdk, ">\tcaptureLatency: %d", m_configuration->captureLatency);
    LOG_INFO(LogChannel::kAnselSdk, ">\tsettleLatency: %d", m_configuration->captureSettleLatency);
    LOG_INFO(LogChannel::kAnselSdk, ">\tfovType: %s", m_configuration->fovType == ansel::kHorizontalFov ? "horizontal" : "vertical" );
    LOG_INFO(LogChannel::kAnselSdk, ">\tmetersInWorldUnit: %f", m_configuration->metersInWorldUnit);
    LOG_INFO(LogChannel::kAnselSdk, ">\ttranslationalSpeedInWordlUnitsPerSecond: %f", m_configuration->translationalSpeedInWorldUnitsPerSecond);
    LOG_INFO(LogChannel::kAnselSdk, ">\trotationalSpeedInDegreesPerSecond: %f", m_configuration->rotationalSpeedInDegreesPerSecond);
    LOG_INFO(LogChannel::kAnselSdk, ">\tisCameraOffcenteredProjectionSupported: %s", m_configuration->isCameraOffcenteredProjectionSupported ? trueStr : falseStr);
    LOG_INFO(LogChannel::kAnselSdk, ">\tisCameraTranslationSupported: %s", m_configuration->isCameraTranslationSupported ? trueStr : falseStr);
    LOG_INFO(LogChannel::kAnselSdk, ">\tisCameraRotationSupported: %s", m_configuration->isCameraRotationSupported ? trueStr : falseStr);
    LOG_INFO(LogChannel::kAnselSdk, ">\tisCameraFovSupported: %s", m_configuration->isCameraFovSupported ? trueStr : falseStr);
    LOG_INFO(LogChannel::kAnselSdk, ">\tright:   (%g, %g, %g)", m_configuration->right.x, m_configuration->right.y, m_configuration->right.z);
    LOG_INFO(LogChannel::kAnselSdk, ">\tup:      (%g, %g, %g)", m_configuration->up.x, m_configuration->up.y, m_configuration->up.z);
    LOG_INFO(LogChannel::kAnselSdk, ">\tforward: (%g, %g, %g)", m_configuration->forward.x, m_configuration->forward.y, m_configuration->forward.z);

    m_cfgShotCaptureLatency = m_configuration->captureLatency;
    m_cfgShotSettleLatency = m_configuration->captureSettleLatency;
    m_cfgMetersInUnit = m_configuration->metersInWorldUnit;

    if (m_configuration->startSessionCallback)
    {
        const auto sessionConfigurationSizeAnselSDKVersion = m_getSessionConfigurationSize();
        const auto sessionConfigurationSizeLwCameraVersion = static_cast<uint32_t>(sizeof(ansel::SessionConfiguration));
        // Choose maximum size for SessionConfiguration and allocate that much
        m_sessionConfigurationStorage = std::make_unique<char[]>(std::max(sessionConfigurationSizeAnselSDKVersion, sessionConfigurationSizeLwCameraVersion));
        // allocate and default initialize SessionConfiguration object using its storage
        if (sessionConfigurationSizeAnselSDKVersion <= sessionConfigurationSizeLwCameraVersion || !m_initializeSessionConfiguration)
            m_sessionConfiguration = new (&m_sessionConfigurationStorage[0]) ansel::SessionConfiguration();
        else
            m_initializeSessionConfiguration(*m_sessionConfiguration);
        // startSessionCallback returned boolean previously, but now it returns more granular status - StartSessionStatus
        // bool != int, so existing integrations return 1 byte for true/false
        // interpret just the lower byte to remain compatible with existing Ansel integrations
        m_sessionConfiguration->translationalSpeedInWorldUnitsPerSecond = m_configuration->translationalSpeedInWorldUnitsPerSecond;
        m_sessionConfiguration->rotationalSpeedInDegreesPerSecond = m_configuration->rotationalSpeedInDegreesPerSecond;
        ansel::StartSessionStatus status = static_cast<ansel::StartSessionStatus>(m_configuration->startSessionCallback(*m_sessionConfiguration, m_configuration->userPointer) & 0x000000FF);
        // TODO: analyze SessionConfiguration fully
        if (status == ansel::kAllowed)
        {
            enhancedSessionAllowed = true;
            s_anselSession = std::make_unique<anselutils::Session>(*m_configuration, width, height);
            LOG_INFO(LogChannel::kAnselSdk, "Session settings:");
            const char* allowedStr = "allowed";
            const char* disallowedStr = "disallowed";
            LOG_INFO(LogChannel::kAnselSdk, ">\t360 is %s", m_sessionConfiguration->is360MonoAllowed ? allowedStr : disallowedStr);
            LOG_INFO(LogChannel::kAnselSdk, ">\t360 Stereo is %s", m_sessionConfiguration->is360StereoAllowed ? allowedStr : disallowedStr);
            LOG_INFO(LogChannel::kAnselSdk, ">\tRaw HDR is %s", m_sessionConfiguration->isRawAllowed ? allowedStr : disallowedStr);
            LOG_INFO(LogChannel::kAnselSdk, ">\tFoV change is %s", m_sessionConfiguration->isFovChangeAllowed ? allowedStr : disallowedStr);
            LOG_INFO(LogChannel::kAnselSdk, ">\tFoV upper limit is %f", m_sessionConfiguration->maximumFovInDegrees);
            LOG_INFO(LogChannel::kAnselSdk, ">\tHighres is %s", m_sessionConfiguration->isHighresAllowed ? allowedStr : disallowedStr);
            LOG_INFO(LogChannel::kAnselSdk, ">\tPause is %s", m_sessionConfiguration->isPauseAllowed ? allowedStr : disallowedStr);
            LOG_INFO(LogChannel::kAnselSdk, ">\tRotation is %s", m_sessionConfiguration->isRotationAllowed ? allowedStr : disallowedStr);
            LOG_INFO(LogChannel::kAnselSdk, ">\tTranslation is %s", m_sessionConfiguration->isTranslationAllowed ? allowedStr : disallowedStr);
            LOG_INFO(LogChannel::kAnselSdk, ">\tRotational speed is %f degrees/second", m_sessionConfiguration->rotationalSpeedInDegreesPerSecond);
            LOG_INFO(LogChannel::kAnselSdk, ">\tTranslational speed is %f world-units/second", m_sessionConfiguration->translationalSpeedInWorldUnitsPerSecond);
        }
        else
        {
            LOG_INFO(LogChannel::kAnselSdk, "Application does not allow session to be started at this time");
        }
    }

    if (s_anselSession)
    {
        const auto& config = *m_configuration;
        const auto& sessionConfig = *m_sessionConfiguration;
        const auto sdkMajor = getAnselSdkVersionMajor();
        const auto sdkMinor = getAnselSdkVersionMinor();
        if (sdkMajor > 1 || (sdkMajor == 1 && sdkMinor >= 5))
            m_setUpdateCameraFunc(updateCamera);
        else
            m_setUpdateCameraFunc(updateCameraForSDKLessThan15);
        // initialize camera controllers
        m_anselCameraControllerRelative = new anselutils::CameraControllerRelative(config.right, config.up, config.forward);
        if (useHybridController)
        {
            m_anselCameraControllerFree = new anselutils::CameraControllerFreeHybrid(config.right, config.up, config.forward);
            static_cast<anselutils::CameraControllerFreeHybrid *>(m_anselCameraControllerFree)->resetRollOnceOnLookaround(true);
        }
        else
            m_anselCameraControllerFree = new anselutils::CameraControllerFreeGlobalRotation(config.right, config.up, config.forward);

        m_anselCameraControllerFree->setTranslationalSpeed(sessionConfig.translationalSpeedInWorldUnitsPerSecond);
        m_anselCameraControllerFree->setRotationalSpeed(sessionConfig.rotationalSpeedInDegreesPerSecond);
        m_anselCameraControllerFree->setCameraFOV(90.0f);
        // set free camera as our camera controller
        s_anselSession->setCameraController(m_anselCameraControllerFree);
        // create director
        m_director = std::make_unique<darkroom::CameraDirector>();
        m_needsFOVupdate = true;
        m_usedGamepadForCameraDuringTheSession = false;
    }

    return enhancedSessionAllowed;
}

void AnselSDKState::stopClientSession()
{
    m_setUpdateCameraFunc(nullptr);

    if (m_configuration->stopSessionCallback)
        m_configuration->stopSessionCallback(m_configuration->userPointer);

    callQualityCallback(false);
}

void AnselSDKState::stopSession()
{
    if (!isSDKDetectedAndSessionActive())
        return;

    stopClientSession();

    m_director.reset(nullptr);
    s_anselSession.reset(nullptr);

    SAFE_DELETE(m_anselCameraControllerFree);
    SAFE_DELETE(m_anselCameraControllerRelative);
    m_anselSavedController = nullptr;
    m_usedGamepadForCameraDuringTheSession = false;
}

void AnselSDKState::handleDarkroomJobFinish(ErrorManager & errorManager, AnselSDKUpdateParameters & outputParameters, const AnselSDKUIParameters & uiParameters, bool keepIntermediateShots)
{
    for (size_t i = 0u; i < m_directorJobs.size(); ++i)
    {
        HANDLE hndl = m_directorJobs[i].hndl;
        DWORD wfso = WaitForSingleObject(hndl, 1);

        if (wfso == WAIT_ABANDONED || wfso == WAIT_OBJECT_0)
        {
            const wchar_t* fileNameInPath = PathFindFileName(m_directorJobs[i].pathFilename.c_str());

            // New messages appear on top, so the "N Files" message should be filed earlier
            if (m_directorJobs.size() > 1)
            {
                const size_t bufSize = 32;
                wchar_t buf[bufSize];
                swprintf_s(buf, bufSize, L"%d", (int)m_directorJobs.size() - 1);
                m_messageParams.resize(0);
                m_messageParams.push_back(buf);
                uiParameters.uiInterface->displayMessage(AnselUIBase::MessageType::kNFilesRemainingToProcess, m_messageParams);
            }

            m_messageParams.resize(0);
            m_messageParams.push_back(fileNameInPath);
            uiParameters.uiInterface->displayMessage(AnselUIBase::MessageType::kProcessingCompleted, m_messageParams);

            outputParameters.processingCompleted = true;
            outputParameters.captureStatus = AnselUIBase::MessageType::kShotSaved;
            outputParameters.processedAbsPath = m_directorJobs[i].pathFilename;

            CloseHandle(hndl);
            m_directorJobs.erase(m_directorJobs.begin() + i);
            --i;
        }
    }
}

darkroom::ShotDescription AnselSDKState::prepareShotDescription(const AnselSDKCaptureParameters& captureParameters, const AnselSDKUIParameters& uiParameters, bool& allowColwertToJPG)
{
    using darkroom::ShotDescription;
    using lwanselutils::appendTimeW;

    ShotDescription desc;
    std::wstring path;

    allowColwertToJPG = true;

    desc.targetPath = captureParameters.snapshotPath;

    if (captureParameters.shotToTake == ShotType::k360)
    {
        desc.type = ShotDescription::EShotType::SPHERICAL_MONO_PANORAMA;
        desc.overlap = darkroom::DefaultRecommendedSphericalOverlap;
        desc.bmpWidth = captureParameters.width;
        desc.bmpHeight = captureParameters.height;
        desc.panoWidth = (darkroom::CameraDirector::estimateSphericalPanoramaWidth(uiParameters.sphericalQualityFov, desc.bmpWidth) / 2)  & ~1u;
        desc.horizontalFov = uiParameters.sphericalQualityFov;

        const std::wstring tempFolderShotType = captureParameters.intermediateFolderPath + L"spherical\\";
        path = appendTimeW(tempFolderShotType.c_str(), L"\\");

        desc.path = path;
    }
    else if (captureParameters.shotToTake == ShotType::kHighRes)
    {
        desc.type = ShotDescription::EShotType::HIGHRES;
        // TODO: replace with getMultiplier to avoid mismatch
        desc.highresMultiplier = uiParameters.highresMultiplier;
        desc.bmpWidth = captureParameters.width;
        desc.bmpHeight = captureParameters.height;
        desc.produceRegularImage = captureParameters.isEnhancedHighresEnabled;
        desc.panoWidth = 0;

        const auto panoWidth = desc.highresMultiplier * captureParameters.width;
        const auto panoHeight = desc.highresMultiplier * captureParameters.height;
        const int maxDim = std::max(panoWidth, panoHeight);
        allowColwertToJPG = (maxDim < 65500); // libjpeg-turbo limitation

        // from HighresBlender.cpp
        ansel::Camera camInfo = s_anselSession->getDisplayCamera();

        desc.horizontalFov = lwanselutils::colwertToHorizontalFov(camInfo, *m_configuration, desc.bmpWidth, desc.bmpHeight);

        const std::wstring tempFolderShotType = captureParameters.intermediateFolderPath + L"highres\\";
        path = appendTimeW(tempFolderShotType.c_str(), L"\\");

        desc.path = path;
    }
#if (ENABLE_STEREO_SHOTS == 1)
    else if (captureParameters.shotToTake == ShotType::kStereo)
    {
        desc.type = ShotDescription::EShotType::STEREO_REGULAR;
        desc.overlap = darkroom::DefaultRecommendedSphericalOverlap;
        desc.bmpWidth = captureParameters.width;
        desc.bmpHeight = captureParameters.height;
        desc.panoWidth = 0;
        const std::wstring tempFolderShotType = captureParameters.intermediateFolderPath + L"stereo\\";
        path = appendTimeW(tempFolderShotType.c_str(), L"\\");

        desc.path = path;
        desc.eyeSeparation = (uiParameters.eyeSeparation * 0.01f) / m_cfgMetersInUnit;
    }
    else if (captureParameters.shotToTake == ShotType::k360Stereo)
    {
        desc.type = ShotDescription::EShotType::SPHERICAL_STEREO_PANORAMA;
        desc.overlap = darkroom::DefaultRecommendedSphericalOverlap;
        desc.bmpWidth = captureParameters.width;
        desc.bmpHeight = captureParameters.height;
        desc.panoWidth = (darkroom::CameraDirector::estimateSphericalPanoramaWidth(uiParameters.sphericalQualityFov, desc.bmpWidth) / 2)  & ~1u;
        desc.horizontalFov = uiParameters.sphericalQualityFov;
        const std::wstring tempFolderShotType = captureParameters.intermediateFolderPath + L"stereo_spherical\\";
        path = appendTimeW(tempFolderShotType.c_str(), L"\\");

        desc.path = path;
        desc.eyeSeparation = (uiParameters.eyeSeparation * 0.01f) / m_cfgMetersInUnit;
    }
#endif
    return desc;
}

void AnselSDKState::onInputEvent(const input::InputEvent& ev, const input::MomentaryKeyboardState& kbdSt,
    const input::MomentaryMouseState& mouseSt,
    const input::MomentaryGamepadState& gpadSt,
    const input::FolwsChecker& folwsChecker,
    const input::MouseTrapper& mouseTrapper)
{
    // process debug camera save/restore
    if (s_anselSession)
    {
        if (kbdSt.isKeyDown(VK_CONTROL) && kbdSt.isKeyDown(VK_MENU) && kbdSt.isKeyStateChangedToUp('C'))
            saveGameCamera(s_anselSession->getDisplayCamera());
        else if (kbdSt.isKeyDown(VK_CONTROL) && kbdSt.isKeyDown(VK_MENU) && kbdSt.isKeyStateChangedToUp('R'))
        {
            ansel::Camera cam;
            if (restoreGameCamera(cam))
            {
                s_anselSession->setDisplayCamera(cam);
                if (s_anselSession->getCameraController())
                    s_anselSession->getCameraController()->reset();
                m_needsFOVupdate = true;
            }
        }
    }
}

void AnselSDKState::processInput(
        input::InputState * inputCapture,
        bool isCameraInteractive,
        const AnselSDKUIParameters & uiParameters,
        const AnselSDKMiscParameters & miscParameters,
        ErrorManager & errorManager
        )
{
    // TODO: additional check here
    const float gamepadStickBacklashGap = 0.2f;

    // Process keys
    const bool rightStickPressDown = inputCapture->getGamepadState().isButtonDown(input::EGamepadButton::kRightStickPress);

    if (inputCapture->getKeyboardState().isKeyDown(VK_SHIFT) || rightStickPressDown)
    {
        // switch between automatic and manual modes here
        const bool EnableAutomaticMode = true;
        m_anselCameraControllerFree->setAccelerationMode(EnableAutomaticMode);
        if (!EnableAutomaticMode && inputCapture->getKeyboardState().isKeyDown(VK_SHIFT))//TODO: was key down
        {
            float newSpeedMult = m_anselCameraControllerFree->getTranslationalSpeedMultiplier();
            const auto wheel = inputCapture->getMouseState().getAclwmulatedCoordWheel();

            if (wheel != 0)
            {
                newSpeedMult += float(wheel) / 40.0f;
            }

            // speed manipulations shouldn't change directions or make the camera stop

            if (newSpeedMult < uiParameters.cameraSpeedMultiplier)
                newSpeedMult = uiParameters.cameraSpeedMultiplier;

            m_anselCameraControllerFree->setTranslationalSpeedMultiplier(newSpeedMult);
        }
        else
            m_anselCameraControllerFree->setTranslationalSpeedMultiplier(uiParameters.cameraSpeedMultiplier);
    }
    else if (inputCapture->getKeyboardState().isKeyDown(VK_CONTROL))
        m_anselCameraControllerFree->setTranslationalSpeedMultiplier(1.0f / uiParameters.cameraSpeedMultiplier);
    else
        m_anselCameraControllerFree->setTranslationalSpeedMultiplier(1.0f);


    // Rotation [mouse]
    ////////////////////////////////////////////////////////////////
    if (isCameraInteractive && uiParameters.isCameraDragActive)
    {
        m_isCameraMoving = true;
        m_anselCameraControllerFree->adjustCameraYaw(inputCapture->getMouseState().getAclwmulatedCoordX() * input::mouseSensititvityCamera);
        m_anselCameraControllerFree->adjustCameraPitch(-inputCapture->getMouseState().getAclwmulatedCoordY() * input::mouseSensititvityCamera);
    }

    // Translation [keyboard]
    ////////////////////////////////////////////////////////////////
    if (isCameraInteractive && inputCapture->getKeyboardState().isKeyDown('W'))
    {
        m_isCameraMoving = true;
        m_anselCameraControllerFree->moveCameraForward(1.0f);
    }
    if (isCameraInteractive && inputCapture->getKeyboardState().isKeyDown('S'))
    {
        m_isCameraMoving = true;
        m_anselCameraControllerFree->moveCameraForward(-1.0f);
    }
    if (isCameraInteractive && inputCapture->getKeyboardState().isKeyDown('A'))
    {
        m_isCameraMoving = true;
        m_anselCameraControllerFree->moveCameraRight(-1.0f);
    }
    if (isCameraInteractive && inputCapture->getKeyboardState().isKeyDown('D'))
    {
        m_isCameraMoving = true;
        m_anselCameraControllerFree->moveCameraRight(1.0f);
    }
    // Either Z or Y changes altitude, since there are two common layouts: QWERTY and QWERTZ
    if (isCameraInteractive && (inputCapture->getKeyboardState().isKeyDown('Z') || inputCapture->getKeyboardState().isKeyDown('Y')))
    {
        m_isCameraMoving = true;
        m_anselCameraControllerFree->moveCameraUp(1.0f);
    }
    if (isCameraInteractive && inputCapture->getKeyboardState().isKeyDown('X'))
    {
        m_isCameraMoving = true;
        m_anselCameraControllerFree->moveCameraUp(-1.0f);
    }

    static double waitingTimeSeconds = -1.0;
    const double secondsHoldQE = 1.0;
    if (isCameraInteractive && (inputCapture->getKeyboardState().isKeyDown('Q') && inputCapture->getKeyboardState().isKeyDown('E')))
    {
        if (miscParameters.useHybridController)
        {
            if (waitingTimeSeconds < 0.0)
                waitingTimeSeconds = 0.0;

            waitingTimeSeconds += miscParameters.dtSeconds;

            if (waitingTimeSeconds > secondsHoldQE)
            {
                m_isCameraMoving = true;
                anselutils::CameraControllerFreeHybrid * camController = static_cast<anselutils::CameraControllerFreeHybrid *>(m_anselCameraControllerFree);
                camController->restoreRoll();
            }
        }
    }
    else
    {
        waitingTimeSeconds = -1.0;
    }

    float rollSpeedMult = m_anselCameraControllerFree->getTranslationalSpeedMultiplier();
    if (isCameraInteractive && inputCapture->getKeyboardState().isKeyDown('Q'))
    {
        m_isCameraMoving = true;
        m_anselCameraControllerFree->adjustCameraRoll(-rollSpeedMult);
    }
    if (isCameraInteractive && inputCapture->getKeyboardState().isKeyDown('E'))
    {
        m_isCameraMoving = true;
        m_anselCameraControllerFree->adjustCameraRoll(rollSpeedMult);
    }

    // Rotation [gamepad]
    ////////////////////////////////////////////////////////////////
    float gamepadRotationX = input::GamepadState::removeBacklash(input::GamepadState::axisToFloat(inputCapture->getGamepadState().getAxisRX()), gamepadStickBacklashGap);
    float gamepadRotationY = input::GamepadState::removeBacklash(input::GamepadState::axisToFloat(inputCapture->getGamepadState().getAxisRY()), gamepadStickBacklashGap);
    float gamepadRotationZ = float(inputCapture->getGamepadState().isButtonDown(input::EGamepadButton::kRightShoulder)) - float(inputCapture->getGamepadState().isButtonDown(input::EGamepadButton::kLeftShoulder));

    if (inputCapture->getGamepadState().isButtonDown(input::EGamepadButton::kLeftStickPress))
    {
        m_isCameraMoving = true;
        m_anselCameraControllerFree->setCameraRoll(0.0f);
        gamepadRotationZ = 0.0f;
    }

    const float gamepadTranslationX = input::GamepadState::removeBacklash(input::GamepadState::axisToFloat(inputCapture->getGamepadState().getAxisLX()), gamepadStickBacklashGap);
    const float gamepadTranslationY = input::GamepadState::removeBacklash(input::GamepadState::axisToFloat(inputCapture->getGamepadState().getAxisLY()), gamepadStickBacklashGap);

    float gamepadTranslationHeight = input::GamepadState::removeBacklash(input::GamepadState::axisToFloat(inputCapture->getGamepadState().getAxisZ()), gamepadStickBacklashGap);

    if (fabs(gamepadRotationX) > FLT_EPSILON || fabs(gamepadRotationY) > FLT_EPSILON ||
        fabs(gamepadRotationZ) > FLT_EPSILON || fabs(gamepadTranslationX) > FLT_EPSILON ||
        fabs(gamepadTranslationY) > FLT_EPSILON || fabs(gamepadTranslationHeight) > FLT_EPSILON ||
        rightStickPressDown)
    {
        m_isCameraMoving = true;
        m_usedGamepadForCameraDuringTheSession = true;
    }

    m_anselCameraControllerFree->adjustCameraYaw(gamepadRotationX);
    m_anselCameraControllerFree->adjustCameraPitch(-gamepadRotationY);
    m_anselCameraControllerFree->adjustCameraRoll(gamepadRotationZ);

    m_anselCameraControllerFree->moveCameraRight(gamepadTranslationX);
    m_anselCameraControllerFree->moveCameraForward(-gamepadTranslationY);
    m_anselCameraControllerFree->moveCameraUp(gamepadTranslationHeight);
}

void AnselSDKState::handleCaptureTaskStartup(AnselSDKUpdateParameters& outputParameters, ErrorManager & errorManager, const AnselSDKCaptureParameters & captureParameters, const AnselSDKUIParameters& uiParameters)
{
    if (outputParameters.shotToTake != ShotType::kNone)
    {
        if (m_director)
        {
            darkroom::ShotDescription desc = prepareShotDescription(captureParameters, uiParameters, m_allowColwertToJPG);
            outputParameters.makeScreenshot = true;
            outputParameters.shotToTake = ShotType::kNone;

            // TODO: deal with this windows-specific function
            // Windows-specific
            if (!desc.path.empty() && lwanselutils::CreateDirectoryRelwrsively(desc.path.c_str()) && lwanselutils::CreateDirectoryRelwrsively(desc.targetPath.c_str()))
            {
                using darkroom::Error;

                desc.generateThumbnail = captureParameters.generateThumbnail;

                Error retcode = m_director->startCaptureTask(desc);

                if (retcode == Error::kIlwalidArgument)
                {
                    // ShotDescription is invalid
                    outputParameters.captureStatus = AnselUIBase::MessageType::kFailedToStartCaptureIlwalidArgument;
                    uiParameters.uiInterface->displayMessage(outputParameters.captureStatus);
                }
                else if (retcode == Error::kNotEnoughFreeSpace)
                {
                    // not enough free space to save all temporary tiles and output at the same time
                    outputParameters.captureStatus = AnselUIBase::MessageType::kFailedToStartCaptureNoSpaceLeft;
                    uiParameters.uiInterface->displayMessage(outputParameters.captureStatus);
                }
                else if (retcode == Error::kTargetPathNotWriteable || retcode == Error::kCouldntCreateFile)
                {
                    // no permissions to write to the target path (or path is incorrect)
                    // std::wstring(L"Failed to start capture: path is incorrect or no permissions")
                    outputParameters.captureStatus = AnselUIBase::MessageType::kFailedToStartCapturePathIncorrectOrPermissions;
                    uiParameters.uiInterface->displayMessage(outputParameters.captureStatus);
                }

                if (retcode == Error::kSuccess)
                {
                    darkroom::CaptureTaskEstimates estimates;
                    estimates = darkroom::CameraDirector::estimateCaptureTask(desc);
                    // TODO probably unify this for each shot type, lwrrently for multipart shots it is done here,
                    //  and for regulars - in the ExelwtePostProcessing
                    m_progressLwrrent = 0;
                    m_progressTotal = (int)estimates.inputDatasetFrameCount;

                    // All non-regular shots require this
                    m_captureOffsetWait = m_cfgShotCaptureLatency;
                    m_cameraSettleLatency = m_cfgShotSettleLatency;
                    m_fileSaveSettleLatency = 0;

                    switch (desc.type)
                    {
                    case darkroom::ShotDescription::EShotType::HIGHRES:
                        m_captureState = CAPTURE_HIGHRES;
                        break;
                    case darkroom::ShotDescription::EShotType::STEREO_REGULAR:
                        m_captureState = CAPTURE_REGULARSTEREO;
                        break;
                    case darkroom::ShotDescription::EShotType::SPHERICAL_MONO_PANORAMA:
                        m_captureState = CAPTURE_360;
                        break;
                    case darkroom::ShotDescription::EShotType::SPHERICAL_STEREO_PANORAMA:
                        m_captureState = CAPTURE_360STEREO;
                        break;
                    default:
                        m_captureState = CAPTURE_REGULAR;
                        break;
                    }
                    outputParameters.needToDisableSnapButton = true;

                    if (m_configuration->startCaptureCallback)
                    {
                        if (getAnselSdkVersionMajor() == 0 && getAnselSdkVersionMinor() <= 18)
                        {
                            // this is a copy of StartCaptureCallback definiton from Ansel SDK 0.18
                            typedef void(*StartCaptureCallback018)(void* userPointer);
                            StartCaptureCallback018 startCaptureCallback018 = reinterpret_cast<StartCaptureCallback018>(m_configuration->startCaptureCallback);
                            startCaptureCallback018(m_configuration->userPointer);
                        }
                        else
                        {
                            // Ansel SDK 0.19 changes startCaptureCallback signature to pass CaptureConfiguration into the game
                            ansel::CaptureConfiguration captureCfg;
                            switch (desc.type)
                            {
                                case darkroom::ShotDescription::EShotType::HIGHRES:
                                    captureCfg.captureType = ansel::kCaptureTypeSuperResolution;
                                    break;
                                case darkroom::ShotDescription::EShotType::STEREO_REGULAR:
                                    captureCfg.captureType = ansel::kCaptureTypeStereo;
                                    break;
                                case darkroom::ShotDescription::EShotType::SPHERICAL_MONO_PANORAMA:
                                    captureCfg.captureType = ansel::kCaptureType360Mono;
                                    break;
                                case darkroom::ShotDescription::EShotType::SPHERICAL_STEREO_PANORAMA:
                                    captureCfg.captureType = ansel::kCaptureType360Stereo;
                                    break;
                                default:
                                    LOG_ERROR("Multipart shot is not passed to the Ansel SDK");
                                    break;
                            }
                            m_configuration->startCaptureCallback(captureCfg, m_configuration->userPointer);
                        }
                    }
                }
                else
                {
                    outputParameters.makeScreenshot = false;
                }
            }
        }
        // "Regular"/"Regular-UI" shots were processed earlier to decrease latency
    }
}


std::wstring GetFolderPathFromFullpath(const std::wstring& path)
{
    std::size_t found = path.find_last_of(L"/\\");
    std::wstring folder = path.substr(0, found);
    return folder;
}


void AnselSDKState::startProcessingDarkroomJob(const std::wstring& path, std::wstring& outputFilename, ErrorManager& errorManager, const AnselSDKCaptureParameters & captureParameters, const AnselSDKUIParameters & uiParameters)
{
    if (isDetected() && isSessionActive())
    {
        std::wstring folderPath = GetFolderPathFromFullpath(path);
        // Launch processing
        HANDLE hndl = 0;
        using darkroom::Error;
        Error retcode = darkroom::processJob(folderPath,
            hndl, m_titleForFileNaming,
            captureParameters.tagDescription,
            captureParameters.tagAppCMSID,
            captureParameters.tagAppShortName,
            captureParameters.snapshotPath,
            outputFilename,
            captureParameters.tagModel,
            captureParameters.tagSoftware,
            captureParameters.tagDrsName,
            captureParameters.tagDrsProfileName,
            captureParameters.tagActiveFilters,
            captureParameters.isShotRawHDR,
            captureParameters.keepIntermediateShots,
            captureParameters.forceLosslessSuperRes,
            captureParameters.forceLossless360,
            captureParameters.enhancedHighresCoefficient,
            captureParameters.isEnhancedHighresEnabled,
            captureParameters.generateThumbnail);

        if (retcode == Error::kUnknownJobType)
        {
            m_messageParams.resize(0);
            m_messageParams.push_back(L"[kUnknownJobType]");
            uiParameters.uiInterface->displayMessage(AnselUIBase::MessageType::kFailedToFinishCapture, m_messageParams);
        }
        else if (retcode == Error::kIlwalidArgumentCount)
        {
            m_messageParams.resize(0);
            m_messageParams.push_back(L"[kIlwalidArgumentCount]");
            uiParameters.uiInterface->displayMessage(AnselUIBase::MessageType::kFailedToFinishCapture, m_messageParams);
        }
        else if (retcode == Error::kCouldntStartupTheProcess)
        {
            m_messageParams.resize(0);
            m_messageParams.push_back(L"[kCouldntStartupTheProcess]");
            uiParameters.uiInterface->displayMessage(AnselUIBase::MessageType::kFailedToFinishCapture, m_messageParams);
        }
        else if (retcode == Error::kCouldntCreateFile)
        {
            m_messageParams.resize(0);
            m_messageParams.push_back(L"[kCouldntCreateFile]");
            uiParameters.uiInterface->displayMessage(AnselUIBase::MessageType::kFailedToFinishCapture, m_messageParams);
        }

        m_directorJobs.push_back({ hndl, outputFilename });
    }
}

void AnselSDKState::stopCapture()
{
    resetCaptureState();

    if (s_anselSession && s_anselSession->getCameraController() == m_anselCameraControllerRelative)
    {
        s_anselSession->setCameraController(m_anselSavedController);
        s_anselSession->setDisplayCamera(m_savedCamera);
        if (m_configuration->stopCaptureCallback)
            m_configuration->stopCaptureCallback(m_configuration->userPointer);
    }
}

void AnselSDKState::setCameraControllerRelativeIfNeeded()
{
    if (s_anselSession && s_anselSession->getCameraController() != m_anselCameraControllerRelative)
    {
        m_anselSavedController = s_anselSession->getCameraController();
        m_savedCamera = s_anselSession->getDisplayCamera();

        if (m_captureState == CAPTURE_360 || m_captureState == CAPTURE_360STEREO)
            m_anselCameraControllerRelative->setCameraBaseAndLevelWithHorizon(m_savedCamera);
        else
            m_anselCameraControllerRelative->setCameraBase(m_savedCamera);

        if (m_configuration && m_configuration->fovType == ansel::kVerticalFov)
            m_anselCameraControllerRelative->setCameraFOV(static_cast<float>(anselutils::colwertVerticalToHorizontalFov(m_savedCamera.fov, s_anselSession->getViewportWidth(), s_anselSession->getViewportHeight())));
        else
            m_anselCameraControllerRelative->setCameraFOV(m_savedCamera.fov);
        s_anselSession->setCameraController(m_anselCameraControllerRelative);
    }
}

void AnselSDKState::setNextCameraFromSequence()
{
    float px, py, pz, rx, ry, rz, sox, soy, fov;
    // pop next camera in sequence and setup relative camera controller
    if (m_director->nextCamera(&px, &py, &pz, &rx, &ry, &rz, &sox, &soy, &fov))
    {
        m_anselCameraControllerRelative->setCameraPositionRelativeToBase(px, py, pz);
        m_anselCameraControllerRelative->setCameraRotationRelativeToBase(rx, ry, rz);
        m_anselCameraControllerRelative->setProjection(sox, soy);
        if (fov > 0.0001f)
            m_anselCameraControllerRelative->setCameraFOV(fov);
    }
}

// Function must be called strictly prior to the AnselSDK::update, otherwise counters will be off
bool AnselSDKState::isLwrrentShotCapture()
{
    return (isDetected() && m_director && !m_director->isCameraNamesSequenceEmpty()) && (m_fileSaveSettleLatency == m_cfgShotSettleLatency);
}

bool AnselSDKState::processDirectorSequence()
{
    if (m_director && !m_director->isCamerasSequenceEmpty())
    {
        // detect first frame and switch to cwCameraControllerDarkroom
        setCameraControllerRelativeIfNeeded();

        if (m_cameraSettleLatency == m_cfgShotSettleLatency)
        {
            setNextCameraFromSequence();
            m_cameraSettleLatency = 0;
        }
        else
            ++m_cameraSettleLatency;

        return true;
    }
    else
        return false;
}

bool AnselSDKState::processDirectorLatencies(bool& needToMakeLwrrentShot, bool& needToFinalizeDirector, bool& needsToolsProcess, std::wstring& filePath, AnselSDKUpdateParameters& outputParameters)
{
    bool processed = false;
    if (isDetected() && m_captureOffsetWait != 0)
    {
        processed = true;
        --m_captureOffsetWait;
    }
    else if (isDetected() && m_director && !m_director->isCameraNamesSequenceEmpty())
    {
        processed = true;
        needToMakeLwrrentShot = m_fileSaveSettleLatency == m_cfgShotSettleLatency;
        if (needToMakeLwrrentShot)
        {
            filePath = m_director->getSequencePath() + darkroom::getWstrFromUtf8(m_director->nextShotName());

            m_fileSaveSettleLatency = 0;
        }
        else
            ++m_fileSaveSettleLatency;

        // We're making the last shot
        if (needToMakeLwrrentShot && m_director->isCameraNamesSequenceEmpty())
        {
            outputParameters.makeScreenshot = false;
            needsToolsProcess = true;
            needToFinalizeDirector = true;
        }
        return true;
    }
    return processed;
}

void AnselSDKState::updateTileUV(float * tlU, float * tlV, float * brU, float * brV)
{
    if (isDetected() && m_director && !m_director->isCameraNamesSequenceEmpty())
    {
        // We should set tileUV only when we do not wait for initial captureOffset
        //  and we just saved picture on the prev frame
        if (m_fileSaveSettleLatency == 0 && m_captureOffsetWait == 0)
        {
            m_director->nextShotTileUV(tlU, tlV, brU, brV);
        }
    }
    else
    {
        // Set tile UVs to defaults
        *tlU = 0.0f;
        *tlV = 0.0f;
        *brU = 1.0f;
        *brV = 1.0f;
    }
}

AnselSDKUpdateParameters AnselSDKState::update(
        input::InputState * inputCapture,
        bool isCameraInteractive,
        ErrorManager & errorManager,
        const AnselSDKCaptureParameters & captureParameters,
        const AnselSDKUIParameters & uiParameters,
        const AnselSDKMiscParameters & miscParameters,
        ShotSaver* ss
        )
{
    using lwanselutils::appendTimeW;

    AnselSDKUpdateParameters outputParameters = { 0 };
    outputParameters.makeScreenshot = captureParameters.makeScreenshot;
    outputParameters.shotToTake = captureParameters.shotToTake;
    outputParameters.needToFinalizeDirector = false;
    outputParameters.screenshotTaken = false;
    outputParameters.processingCompleted = false;
    outputParameters.isResetRollAvailable = false;
    outputParameters.captureStatus = AnselUIBase::MessageType::kNone;

    bool needToMakeLwrrentShot = false;
    bool needsToolsProcess = false;

    std::wstring shotName, additionalShotName;

    if (isDetected() && isSessionActive() && s_anselSession->isDisplayCameraInitialized() && uiParameters.isHighQualityEnabled)
    {
        if (isCameraChanged())
        {
            if (m_highQualitySetting)
            {
                callQualityCallback(false);
                m_highQualitySetting = false;
            }
        }
        else if (!m_highQualitySetting)
        {
            callQualityCallback(true);
            m_highQualitySetting = true;
        }
    }
    else if (m_highQualitySetting)
    {
        callQualityCallback(false);
        m_highQualitySetting = false;
    }

    m_isCameraMoving = false;

    if (outputParameters.makeScreenshot)
    {
        const wchar_t * const bmpExtension = L".bmp";
        const wchar_t * const pngExtension = L".png";
        const wchar_t * const exrExtension = L".exr";
        const wchar_t * const jxrExtension = L".jxr";

        std::wstring filePath;
        if (processDirectorLatencies(needToMakeLwrrentShot, outputParameters.needToFinalizeDirector, needsToolsProcess, filePath, outputParameters))
        {
            if (filePath.find(L"thumbnail") != std::wstring::npos)
            {
                filePath += bmpExtension;
            }
            else
            {
                if (captureParameters.isShotHDR)
                {
                    filePath += exrExtension;
                }
                else if (captureParameters.isShotHDRJXR)
                {
                    filePath += jxrExtension;
                }
                else
                {
                    filePath += bmpExtension;
                }
            }

            shotName = filePath.c_str();
        }
        else
        {
            const wchar_t * fileExtension = pngExtension;

            std::wstring filenameName;

            const bool isHdr = captureParameters.isShotRawHDR || captureParameters.isShotDisplayHDR;

            // check free space before capturing a regular screenshot
            // we use very simple heuristic here - a screenshot can't take more than it's uncompressed
            // representation. Furthermore we require darkroom::MinimumFreeSpaceAfterCapture of space to remain
            // free after the capture.
            ULARGE_INTEGER freeBytes, totalBytes, totalFreeBytes;

            if (!captureParameters.snapshotPath.empty())
            {
                if (!lwanselutils::CreateDirectoryRelwrsively(captureParameters.snapshotPath.c_str()))
                {
                    LOG_ERROR("Save shot failed: failed to create directories");
                    uiParameters.uiInterface->displayMessage(AnselUIBase::MessageType::kFailedToSaveShotFailedCreateDiretory);
                    outputParameters.makeScreenshot = false;
                    needToMakeLwrrentShot = false;
                }
                else
                {
                    GetDiskFreeSpaceEx(captureParameters.snapshotPath.c_str(), &freeBytes, &totalBytes, &totalFreeBytes);
                    const auto bytesPerPixel = isHdr ? 16 : 3;
                    if (captureParameters.width * captureParameters.height * bytesPerPixel + darkroom::MinimumFreeSpaceAfterCapture > freeBytes.QuadPart)
                    {
                        uiParameters.uiInterface->displayMessage(AnselUIBase::MessageType::kFailedToSaveShotNoSpaceLeft);
                        outputParameters.makeScreenshot = false;
                        needToMakeLwrrentShot = false;
                    }
                    else
                    {
                        if (captureParameters.isShotHDR)
                        {
                            fileExtension = exrExtension;
                        }
                        else if (captureParameters.isShotHDRJXR)
                        {
                            fileExtension = jxrExtension;
                        }

                        const auto now = darkroom::generateTimestamp();
                        filenameName = darkroom::generateFileName(darkroom::JobType::REGULAR, now, m_titleForFileNaming, fileExtension, captureParameters.isShotRawHDR ? L" Raw" : L"");

                        shotName = captureParameters.snapshotPath + std::wstring(filenameName);

                        if (captureParameters.pPresentResourceDataAdditional)
                        {
                            additionalShotName = captureParameters.snapshotPath + darkroom::generateFileName(darkroom::JobType::REGULAR, now, m_titleForFileNaming, L".png", captureParameters.isShotHDR ? darkroom::gThumbnailSuffix : L"");
                        }

                        outputParameters.makeScreenshot = false;
                        needToMakeLwrrentShot = true;
                    }
                }
            }
        }
    }

    // Darkroom
    handleDarkroomJobFinish(errorManager, outputParameters, uiParameters, captureParameters.keepIntermediateShots);

    if (isDetected() && isSessionActive() && s_anselSession->isDisplayCameraInitialized())
    {
        if (m_captureState == CAPTURE_ABORT)
        {
            outputParameters.needToFinalizeDirector = true;
            resetCaptureState();

            m_progressTotal = 0;
            m_progressLwrrent = 0;
            needToMakeLwrrentShot = false;
            outputParameters.makeScreenshot = false;
            outputParameters.shotToTake = ShotType::kNone;

            if (m_director)
            {
                darkroom::deleteDirectory(std::wstring(m_director->getSequencePath()));
                m_director->abortCaptureTask();
            }
        }

        if (outputParameters.shotToTake != ShotType::kRegular)
            handleCaptureTaskStartup(outputParameters, errorManager, captureParameters, uiParameters);
        else
            outputParameters.shotToTake = ShotType::kNone;

        // we either work with CameraControllerRelative or CameraControllerFree
        if (!processDirectorSequence())
        {
            // When Ansel wants to update the FOV in the UI, m_needsFOVUpdate is true.
            // When the FOV has changed in the UI, and Ansel wants to update the FOV in the UI, the changed FOV from the UI takes precendence.
            // This is to avoid overwriting user changes to the FOV by Ansel changes, which are caused by state restoration at session start or via hotkeys during a session.
            outputParameters.needsFOVupdate = uiParameters.isFOVChanged || m_needsFOVupdate;
            if (uiParameters.isFOVChanged)
            {
                m_anselCameraControllerFree->setCameraFOV(uiParameters.fovSliderValue);
                outputParameters.lwrrentHorizontalFov = uiParameters.fovSliderValue;
                m_isCameraMoving = true;
                m_needsFOVupdate = false;
            }
            else if (m_needsFOVupdate)
            {
                // This parameter transfer is needed to correctly set the UI FOV to match the actual FOV when it is changed outside of the UI.
                outputParameters.lwrrentHorizontalFov = lwanselutils::colwertToHorizontalFov(s_anselSession->getDisplayCamera(), *m_configuration, captureParameters.width, captureParameters.height);
                m_needsFOVupdate = false;
            }


            if (uiParameters.isRollChanged)
            {
                m_anselCameraControllerFree->setCameraRoll(uiParameters.rollDegrees);
                m_isCameraMoving = true;
            }

            if (miscParameters.useHybridController)
            {
                anselutils::CameraControllerFreeHybrid * camController = static_cast<anselutils::CameraControllerFreeHybrid *>(m_anselCameraControllerFree);
                outputParameters.isResetRollAvailable = !camController->isControllerGlobal() && !camController->isRollBeingRemoved();
                if (captureParameters.restoreRoll)
                {
                    m_isCameraMoving = true;
                    camController->restoreRoll();
                }
            }

            if (inputCapture)
                processInput(inputCapture, isCameraInteractive, uiParameters, miscParameters, errorManager);
        }

        // let's get it back so that slider can aclwrately reflect current value
        outputParameters.roll = m_anselCameraControllerFree->getCameraRoll();

        // TODO: probably we need to finalize as soon as camera queue gets emptied
        if (outputParameters.needToFinalizeDirector)
            stopCapture();
    }

    if (needToMakeLwrrentShot)
    {
        ++m_progressLwrrent;
        HRESULT shotStatus = S_FALSE;
        bool forceSDRCapture = (!captureParameters.isShotHDR && !captureParameters.isShotHDRJXR);
        if (shotName.find(L"thumbnail") != std::wstring::npos)
        {
            shotStatus = ss->saveShot(captureParameters, forceSDRCapture, shotName,
                captureParameters.pPresentResourceDataAdditional != nullptr);

            outputParameters.screenshotTaken = true;
        }
        else
        {
            shotStatus = ss->saveShot(captureParameters, forceSDRCapture, shotName);
            if (shotStatus == S_OK)
            {
                if (!additionalShotName.empty())
                {
                    shotStatus = ss->saveShot(captureParameters, true, additionalShotName, true);
                }
                LOG_INFO("Shot saved (%d)", m_progressLwrrent);

                outputParameters.screenshotTaken = true;
            }
            else
            {
                LOG_INFO("Shot wasn't saved (%d)", m_progressLwrrent);
            }
        }
        std::wstring outFilename;
        if (needsToolsProcess)
        {
            startProcessingDarkroomJob(shotName, outFilename, errorManager, captureParameters, uiParameters);
        }

        if (shotStatus == S_OK && !outputParameters.makeScreenshot)
        {
            if (outFilename.empty() || outFilename.length() <= 4)
            {
                outputParameters.processingCompleted = true;
                outputParameters.captureStatus = AnselUIBase::MessageType::kShotSaved;
                if (captureParameters.isShotHDRJXR && additionalShotName != L"")
                {
                    outputParameters.processedAbsPath = additionalShotName;
                }
                else
                {
                    outputParameters.processedAbsPath = shotName;
                }
                m_messageParams.resize(0);
                m_messageParams.push_back(outputParameters.processedAbsPath);
                uiParameters.uiInterface->displayMessage(AnselUIBase::MessageType::kShotSaved, m_messageParams);
            }
            else
            {
                m_messageParams.resize(0);
                m_messageParams.push_back(outFilename);
                uiParameters.uiInterface->displayMessage(AnselUIBase::MessageType::kProcessingFile, m_messageParams);
            }
        }
        else if (shotStatus != S_OK && !outputParameters.makeScreenshot)
        {
            m_messageParams.resize(0);
            m_messageParams.push_back(shotName);
            outputParameters.captureStatus = AnselUIBase::MessageType::kFailedToSaveShot;
            uiParameters.uiInterface->displayMessage(AnselUIBase::MessageType::kFailedToSaveShot, m_messageParams);
        }
    }

    return outputParameters;
}

void AnselSDKState::callQualityCallback(bool isHighQuality)
{
    if (m_configuration->changeQualityCallback)
    {
        m_configuration->changeQualityCallback(isHighQuality, m_configuration->userPointer);
    }
}

