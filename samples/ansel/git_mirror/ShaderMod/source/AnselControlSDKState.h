#pragma once

#include "Config.h"
#include "CommonTools.h"
#include "anselcontrol/Configuration.h"
#include "anselcontrol/Interface.h"
#include "darkroom/StringColwersion.h"
#include "UIBase.h"

#include <string>
#include <vector>
#include <memory>

struct AnselControlSDKUpdateParameters
{
    bool isSomething;
};

class AnselControlSDKState: public input::InputEventsConsumerInterface, public AnselUIBase
{
    // Client/UI part

public:

    bool m_anselPrestartRequested = false;
    bool m_anselStartRequested = false;
    bool m_anselStopRequested = false;
    bool m_anselPoststopRequested = false;
    bool m_anselSessionRequested = true;

    bool m_captureStarted = false;
    bool m_captureInProgress = false;
    bool m_captureCameraInitialized = false;
    ShotDesc m_shotDescToTake;

    input::InputHandlerForIPC m_inputstate;

    virtual LANGID getLangId() const override { return MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US); }

    // Use `enum struct ShotType` defined in "CommonStructs.h" to navigate this array
    virtual void setShotTypePermissions(bool shotHDREnabled, const bool * shotTypeEnabled, int arraySize) override { }

    virtual void setAnselSDKDetected(bool detected) override { }
    virtual void onSessionStarted(bool isCameraWorksSessionActive) override { }
    virtual void onSessionStopped() override { }

    virtual void setScreenSize(uint32_t width, uint32_t height) override { }

    virtual void setFadeState(bool fadeState) override { }

    virtual bool isEffectListRequested() override { return false; }
    virtual void repopulateEffectsList(const std::vector<std::wstring> & filterIds, const std::vector<std::wstring> & filterNames) override { }

    virtual bool isStackFiltersListRequested() override { return false; }
    virtual void stackFiltersListDone(const std::vector<std::wstring> & filterIds) override { }

#ifdef ENABLE_STYLETRANSFER
    virtual void mainSwitchStyleTransfer(bool status) override { }

    virtual bool isStylesListRequested() override { return false; }
    virtual void repopulateStylesList(const std::vector<std::wstring> & stylesIds, const std::vector<std::wstring> & stylesNames, const std::vector<std::wstring> & stylesPaths) override { }

    virtual bool isStyleNetworksListRequested() override { return false; }
    virtual void repopulateStyleNetworksList(const std::vector<std::wstring> & netIds, const std::vector<std::wstring> & netNames) override { }

    const std::wstring strEmpty = L"";
    virtual const std::wstring & getLwrrentStyle() const override { return strEmpty; }
    virtual const std::wstring & getLwrrentStylePath() const override { return strEmpty; }

    virtual const std::wstring & getLwrrentStyleNetwork() const override { return strEmpty; }
    virtual void setLwrrentStyleNetwork(const std::wstring & netid) override { }

    virtual void setStyleTransferStatus(bool isStyleTransferEnabled) override { }

    virtual void showRestyleDownloadConfirmation() override { }
    virtual RestyleDownloadStatus getRestyleDownloadConfirmationStatus() override { return RestyleDownloadStatus::kNone; }
    virtual void clearRestyleDownloadConfirmationStatus() override { }

    virtual void toggleRestyleProgressBar(bool status) override { }
    virtual void setRestyleProgressBarValue(float value) override { }
    virtual void setRestyleProgressState(RestyleProgressState progressState) override { }

    virtual void toggleRestyleProgressIndicator(bool status) override { }

    virtual bool isUIInteractionActive() override { return false; }
    virtual bool isStyleTransferEnabled() const override { return false; }
#endif

    virtual bool isHighResolutionRecalcRequested() override { return false; }
    virtual void highResolutionRecalcDone(const std::vector<HighResolutionEntry> & highResEntries) override { }

    virtual size_t getLwrrentFilterNum() const override { return 0; }
    // These two functions better have event queue trigger alternative to avoid
    //  polling all the stack each frame
    virtual const std::wstring getLwrrentFilter(size_t effectStackIdx) const override { return shadermod::Tools::wstrNone; }
    virtual bool getLwrrentFilterInfoQuery(size_t effectStackIdx) override { return false; }
    // Response is in the `updateEffectControlsInfo`
    virtual bool getLwrrentFilterResetValues(size_t effectStackIdx) override { return false; }
    virtual void lwrrentFilterResetValuesDone(size_t effectStackIdx) override { }

    virtual int getLwrrentFilterOldStackIdx(size_t effectStackIdx) const override { return 0; }
    virtual void updateLwrrentFilterStackIndices() override { }


    // In order to report an error, just call `updateEffectControls(ANY, nullptr);`
    // TODO: probably we'll need some more detailed error reporting? But lwrrently all that UI needs to do - just drop filter selection to None and report that error was registered.
    virtual void updateEffectControls(size_t effectStackIdx, EffectPropertiesDescription * effectDescription) override { }
    virtual void updateEffectControlsInfo(size_t effectStackIdx, EffectPropertiesDescription * effectDescription) override { }

    virtual bool isUpdateGameSpecificControlsRequired() override { return false; }
    virtual void updateGameSpecificControls(EffectPropertiesDescription * effectDescription) override { }
    virtual void updateGameSpecificControlsInfo(EffectPropertiesDescription * effectDescription) override { }

    virtual std::vector<EffectChange>& getEffectChanges() override { return m_effectChanges; }
    virtual void getEffectChangesDone() override { }

    // These are values that are retrieved from GFE. LwCamera does not have a way to get these on its own.
    virtual std::wstring getAppCMSID() const override { return L""; }
    virtual std::wstring getAppShortName() const override { return L""; }

    //  We need shotCaptureRequested return desc structure (with resolutions/multipliers etc, to potentially avoid unnecessary get/setters)
    virtual ShotDesc shotCaptureRequested() override
    {
        return m_shotDescToTake;
    }
    virtual void shotCaptureDone(Status status) override
    {
        m_shotDescToTake.shotType = ShotType::kNone;
    }

    struct CaptureProgressState
    {
        int numShotsTotal;
        int shotIdx;
        std::wstring absPath;
    };
    CaptureProgressState m_captureProgressState;

    virtual void onCaptureStarted(int numShotsTotal) override
    {
        m_captureProgressState.absPath = L"";
        m_captureProgressState.numShotsTotal = numShotsTotal;
        if (m_configuration && m_configuration->captureProgressCallback)
        {
            m_configuration->captureProgressCallback(anselcontrol::kCaptureStarted, numShotsTotal, m_configuration->userPointer);
        }
    }
    virtual void onCaptureTaken(int shotIdx) override
    {
        m_captureProgressState.shotIdx = shotIdx;
        if (m_configuration && m_configuration->captureProgressCallback)
        {
            m_configuration->captureProgressCallback(anselcontrol::kCaptureShotTaken, shotIdx, m_configuration->userPointer);
        }
    }
    virtual void onCaptureStopped(AnselUIBase::MessageType status) override
    {
        if (m_configuration && m_configuration->captureProgressCallback)
        {
            m_configuration->captureProgressCallback(anselcontrol::kCaptureStopped, m_captureProgressState.shotIdx, m_configuration->userPointer);
        }
        m_captureProgressState.numShotsTotal = 0;
        m_captureProgressState.shotIdx = 0;
    }
    virtual void onCaptureProcessingDone(int status, const std::wstring & absPath) override
    {
        m_captureProgressState.absPath = absPath;

        std::string absPathUTF8 = darkroom::getUtf8FromWstr(absPath);
        m_setLastCaptureUtf8Path(absPathUTF8.c_str());

        if (m_configuration && m_configuration->captureProgressCallback)
        {
            m_configuration->captureProgressCallback(anselcontrol::kCaptureProcessed, 0, m_configuration->userPointer);
        }
        m_captureInProgress = false;
    }

    virtual bool isGridOfThirdsEnabled() const override { return false; }

    virtual bool isShotEXR() const override { return false; }
    virtual bool isShotJXR() const override { return false; }
    virtual bool isShotPreviewRequired() const override { return false; }
    virtual double getFOVDegrees() const override { return 0.0; }
    virtual double getRollDegrees() const override { return 0.0; }
    virtual bool processFOVChange() override { return false; }
    virtual bool processRollChange() override { return false; }
    virtual bool getCameraDragActive() override { return false; }
    virtual bool isCameraInteractive() override { return false; }
    virtual bool isHighresEnhance() const override { return false; }
    virtual bool isHighQualityEnabled() const override { return true; }

    virtual void setFOVDegrees(double fov) override { }
    virtual void setRollDegrees(double roll) override { }
    virtual void setFovControlEnabled(bool enabled) override { }
    virtual void setFOVLimitsDegrees(double lo, double hi) override { }
    virtual void setRollLimitsDegrees(double lo, double hi) override { }
    virtual void set360WidthLimits(uint64_t lo, uint64_t hi) override { }
    virtual void setHighresEnhance(bool enhance) override { }
    virtual void setHighQualityEnabled(bool setting) override { }

    virtual bool isResetRollNeeded() const  override { return false; }
    virtual void resetRollDone() override { }
    virtual void setResetRollStatus(bool isAvailable) override { }

    virtual bool isAnselPrestartRequested() override { return m_anselPrestartRequested; }
    virtual bool isAnselStartRequested() override { return m_anselStartRequested; }
    virtual bool isAnselSDKSessionRequested() override { return m_anselSessionRequested; }
    virtual bool isAnselStopRequested() override { return m_anselStopRequested; }
    virtual bool isAnselPoststopRequested() override { return m_anselPoststopRequested; }
    virtual bool isAnselFeatureSetRequested() override { return false; }

    virtual void anselPrestartDone(Status status, bool isSDKDetected, bool requireSDK) override
    {
        m_anselPrestartRequested = false;

        switch (m_lwrrentScenario)
        {
            case ScenarioType::kCaptureShot:
            {
                captureShotScenarioState.m_controlState = ControlState::kStart;
                break;
            }
            default:
                break;
        }
    }
    virtual void anselStartDone(Status status) override
    {
        m_anselStartRequested = false;

        switch (m_lwrrentScenario)
        {
            case ScenarioType::kCaptureShot:
            {
                captureShotScenarioState.m_controlState = (status == Status::kDeclined) ? ControlState::kNone : ControlState::kCapture;
                break;
            }
            default:
                break;
        }
    }
    virtual void anselStopDone(Status status) override
    {
        m_anselStopRequested = false;

        switch (m_lwrrentScenario)
        {
            case ScenarioType::kCaptureShot:
            {
                captureShotScenarioState.m_controlState = ControlState::kPoststop;
                break;
            }
            default:
                break;
        }
    }
    virtual void anselPoststopDone(Status status) override
    {
        m_anselPoststopRequested = false;

        switch (m_lwrrentScenario)
        {
            case ScenarioType::kCaptureShot:
            {
                captureShotScenarioState.m_controlState = ControlState::kNone;
                break;
            }
            default:
                break;
        }
        m_ilwalidateAllState();
    }
    virtual void anselFeatureSetRequestDone() override {}

    virtual bool isSDKCaptureAbortRequested() override { return false; }
    virtual void sdkCaptureAbortDone(int status) override { }

    virtual void forceEnableUI() override { }
    virtual void forceDisableUI() override { }

    virtual void update(double dt) override { }

    virtual input::InputHandler& getInputHandler() override { return m_inputstate; }
    virtual const input::InputHandler& getInputHandler() const override { return m_inputstate; }

    virtual void getTelemetryData(UISpecificTelemetryData &) const override { }

    virtual void emergencyAbort() override { }

    virtual void setModdingStatus(bool isModdingAllowed) override { }

    virtual bool isModdingAllowed() override { return false; }
    bool queryIsModdingEnabled() override { return true; }
    bool isModdingEnabled() override { return true; }

    virtual void setDefaultEffectPath(const wchar_t * defaultEffectPath) override { }

    virtual void addGameplayOverlayNotification(NotificationType notificationType, ErrorManager::ErrorEntry notification, bool allowSame) override { }

    virtual void displayMessage(MessageType msgType) override { }
    virtual void displayMessage(MessageType msgType, const std::vector<std::wstring> & parameters
#if ANSEL_SIDE_PRESETS
                        , bool removeLastLine = false
#endif
    ) override { }

    virtual void reportFatalError(FatalErrorCode code, const std::string& filename, uint32_t line, const std::string& data) override { }
    virtual void reportNonFatalError(uint32_t code, const std::string& filename, uint32_t line, const std::string& data) override { }

    void updateSettings(const std::map<std::string, std::wstring>& settings) override {}

    virtual bool isRequestingControl() override
    {
        return m_needsControl;
    }
    virtual void rejectControlRequest() override
    {
        m_needsControl = false;
        m_ilwalidateAllState();
    }

public:

    // General logic

    AnselControlSDKState();

    const anselcontrol::Configuration& getConfiguration() const;

    bool isSDKDetectedAndSessionActive();

    enum class DetectionStatus
    {
        kSUCCESS = 0,
        kDLL_NOT_FOUND = 1,
        kDRIVER_API_MISMATCH = 2,

        kNUM_ENTRIES
    };

    HMODULE findAnselControlSDKModule();

    DetectionStatus detectAndInitializeAnselControlSDK();

    bool isConfigured() const;
    bool isDetected() const;

    uint32_t getAnselControlSdkVersionMajor() const;
    uint32_t getAnselControlSdkVersionMinor() const;
    uint32_t getAnselControlSdkVersionCommit() const;

    enum class ControlState
    {
        kNone,
        kPrestart,
        kStart,
        kCapture,
        kStop,
        kPoststop,

        kNUM_ENTRIES
    };

    enum class ScenarioType
    {
        kNone,
        kCaptureShot,

        kNUM_ENTRIES
    };

    ScenarioType m_lwrrentScenario = ScenarioType::kNone;

    struct CaptureShotScenarioState
    {
        ControlState m_controlState = ControlState::kNone;
        anselcontrol::CaptureShotState m_shotState;
    };
    CaptureShotScenarioState captureShotScenarioState;
    void initCaptureShotScenario();
    void updateCaptureShotScenario();

    AnselControlSDKUpdateParameters updateControlState();

    bool m_readinessReported = false;
    void checkReportReadiness();
    void checkGetControlConfiguration();

    bool m_needsControl = false;
    bool isControlNeeded() const { return m_needsControl; }

    bool isExlusive()
    {
        return (m_configuration && m_configuration->exclusiveMode);
    }

private:

    void processInput(input::InputState * inputCapture, bool isCameraInteractive);

    virtual void onInputEvent(const input::InputEvent& ev, const input::MomentaryKeyboardState& kbdSt,
        const input::MomentaryMouseState& mouseSt,
        const input::MomentaryGamepadState& gpadSt,
        const input::FolwsChecker& folwsChecker,
        const input::MouseTrapper& mouseTrapper) override;

    bool m_DLLfound = false;

    std::vector<EffectChange> m_effectChanges;

    uint64_t m_anselControlSdkVersion;

    anselcontrol::Configuration* m_configuration;
    std::unique_ptr<char[]> m_configurationStorage;

    //Telemetry - please don;t use except for stats { 

    bool m_usedGamepadForCameraDuringTheSession;
    //}

    // defined in AnselControlSDK too

    typedef void(__cdecl *PFNINITIALIZECONFIGURATIONFUNC) (anselcontrol::Configuration & cfg);
    typedef void(__cdecl *PFNGETCONFIGURATIONFUNC) (anselcontrol::Configuration & cfg);
    typedef uint32_t(__cdecl *PFNGETCONFIGURATIONSIZE) ();

    typedef void(__cdecl *PFNGETCAPTURESHOTSTATE)(anselcontrol::CaptureShotState & captureShotState);
    typedef uint32_t(__cdecl *PFNGETCONFIGURATIONSIZE) ();

    typedef void(__cdecl *PFNREPORTSERVERCONTROLVERSIONFUNC)(uint64_t);

    typedef bool(__cdecl *PFNSETLASTCAPTUREUTF8PATHFUNC)(const char *);

    typedef uint64_t(__cdecl *PFNGETVERSIONFUNC)();
    typedef void(__cdecl *PFLWOIDFUNC)();
    typedef bool(__cdecl *PFNBOOLFUNC)();

    PFNGETCONFIGURATIONFUNC m_getControlConfiguration;
    PFNBOOLFUNC m_getControlConfigurationChanged;
    PFNGETCONFIGURATIONSIZE m_getControlConfigurationSize;
    PFNGETCAPTURESHOTSTATE m_getCaptureShotScenarioState;
    PFNBOOLFUNC m_ilwalidateCaptureShotScenarioState;
    PFNBOOLFUNC m_ilwalidateAllState;

    PFNGETVERSIONFUNC m_getVersion;
    PFNREPORTSERVERCONTROLVERSIONFUNC m_reportServerControlVersion;

    PFNSETLASTCAPTUREUTF8PATHFUNC m_setLastCaptureUtf8Path;

    PFNINITIALIZECONFIGURATIONFUNC m_initializeConfiguration;
};
