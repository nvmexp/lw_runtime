#pragma once
#include <vector>
#include <unordered_map>
#include "Config.h"
#include "AnselIPC.h"
#include "UIBase.h"
#include "InputHandler.h"
#include "AnselPreset.h"


#if IPC_ENABLED == 1
struct IPCState
{
    struct IPCSDKState
    {
        bool sdkDetected = false;
        bool captureAbortRequested = false;
        bool startAnselSDKSession = true;
        // this specifies if the session is Ansel or not
        bool isAnselSessionActive = false;
        // this might be an Ansel or Modding session
        bool isLwCameraSessionActive = false;
        bool isPrestartRequested = false;
        bool isStartRequested = false;
        bool isStopRequested = false;
        bool isPoststopRequested = false;
        bool isHighresResolutionListRequested = false;
        bool isHighresEnhancementRequested = false;
        bool isGameSpecificControlUpdateRequired = false;

        bool cameraDragActive = false;

        bool allowedHDR = false;
        std::vector<bool> allowedShotTypes;
        uint32_t highresMultiplier = 0;
        uint32_t pano360ResultWidth = 0;
        uint32_t pano360MinWidth = 0;
        uint32_t pano360MaxWidth = 0;
        double fovMin = 0.0;
        double fovMax = 0.0;
        double rollMin = 0.0;
        double rollMax = 0.0;
        float cameraFov = 0.0f;
        float cameraRoll = 0.0f;
        ShotType shotTypeToTake = ShotType::kNone;
        bool isShotPreviewRequired = false;
        bool isShotEXR = false;
        bool isShotJXR = false;
        bool isThumbnailRequired = false;
        bool isCameraRollChanged = false;
        bool isCameraFOVChanged = false;
        bool isHighQualityEnabled = false;

        // TODO: int needs to change to a buffer type enum
        std::vector<AnselUIBase::HighResolutionEntry> highresResolutions;

        bool isFeatureSetRequested = false;
    };
    struct FiltersSDKState
    {
        enum class FilterResponseType
        {
            kNone,
            kSet,
            kInsert,
            kSetFilterAndAttributes
        };
        bool effectListRequested = false;
        // if modding is desired by the UI at the moment (disambiguates Ansel and FreeStyle sessions)
        bool isModdingAllowed = false;
        // if modding is enabled as a feature. It's a kill switch. Helps to decide if we're going to unload LwCamera
        // for a non Ansel SDK integrated game.
        bool isModdingEnabled = true;
        // is modding available in principle from Ansel Core standpoint. Perhaps it can't be due to limitations of a particular graphics
        // API used by the application
        bool isModdingAvailable = false;
        bool isGridOfThirdsEnabled = false;
        bool isStackFilterListRequested = false;
        std::vector<bool> lwrrentFilterInfoQuery;
        std::vector<bool> lwrrentFilterResetValues;
        std::vector<std::wstring> lwrrentFilterIds;
        std::vector<uint32_t> lwrrentFilterOldIndices;
        std::vector<FilterResponseType> filterSetResponseRequired;
        std::unordered_map<UINT, std::vector<std::pair<int, AnselIpc::Status> > > setAttributeResponses; // Maps an index into filterSetResponseRequired to the corresponding vector of set attribute responses.
        // filterId, filterName
        std::vector<std::pair<std::wstring, std::wstring>> filterNames;

        std::vector<AnselUIBase::EffectChange> effectChanges;
#if ANSEL_SIDE_PRESETS
        AnselPreset* activePreset;
        // stackIdx from UI -> stackIdx internal to Ansel
        std::vector<uint32_t> stackIdxTranslation;

        struct AnselPresetError
        {
            AnselIpc::Status status;
            size_t stackIdx;
            std::wstring filterId;
            std::wstring message;
        };
        std::vector<AnselPresetError> presetErrors;
#endif
    };
    struct StyleTransferState
    {
        bool isAvailable = false;
        bool isEnabled = false;
        bool styleListRequested = false;
        // these two flags are set to true to trigger Ansel Server to send
        // the network list initially to the IPC UI backend, but skip
        // sending the message through the MessageBus, because it was actually
        // requested. This network list needs to be cached in the IPC UI
        // to check network selection requests against the list of all possible
        // choices
        bool networkListRequested = true;
        bool dontSendStyleListResponse = true;
        // sideloading part
        AnselIpc::SideloadProgress progressState = AnselIpc::SideloadProgress::kSideloadProgressIdle;
        bool showProgress = false;
        int downloadProgress = 0;
        AnselUIBase::RestyleDownloadStatus downloadConfirmationStatus = AnselUIBase::RestyleDownloadStatus::kNone;
        
        std::wstring lwrrentStyleFullPath;
        std::wstring lwrrentNetworkId;
        std::map<std::wstring, std::wstring> netIds;
    };

    bool ClientMeetsMinimumVersion(unsigned int minMajor, unsigned int minMinor, unsigned int minPatch) const;

    input::InputHandlerForIPC m_inputstate;

    WORD lang = LANG_ENGLISH, subLang = SUBLANG_ENGLISH_US;

    IPCSDKState sdk;
    FiltersSDKState filters;
    StyleTransferState restyle;
    std::map<std::string, std::wstring> settings;
    uint32_t screenWidth = 0, screenHeight = 0;
    bool fadeState = true;

    std::wstring appCMSID = L"";
    std::wstring appShortName = L"";

    unsigned int ipcClientVersionMajor = 0;
    unsigned int ipcClientVersionMinor = 0;
    unsigned int ipcClientVersionPatch = 0;
};

class UIIPC : public input::InputEventsConsumerInterface, public AnselUIBase
{
public:
    UIIPC();
    void checkInitObserver();
    bool needToSendAnselReady() const;
    bool sendAnselReady();
    void disconnectIpc();
    virtual ~UIIPC();
        
    // Use `enum struct ShotType` defined in "CommonStructs.h" to navigate this array
    LANGID getLangId() const override;
    void release();
    void setShotTypePermissions(bool shotHDREnabled, const bool * shotTypeEnabled, int arraySize) override;
    void setFovControlEnabled(bool enabled) override;
    void setFOVLimitsDegrees(double lo, double hi) override;
    void setRollLimitsDegrees(double lo, double hi) override;
    void set360WidthLimits(uint64_t lo, uint64_t hi) override;
    void setAnselSDKDetected(bool detected) override;
    void onSessionStarted(bool isCameraWorksSessionActive) override;
    void onSessionStopped() override;
    void setScreenSize(uint32_t width, uint32_t height) override;
    void setFadeState(bool fadeState) override;
    bool isEffectListRequested() override;
    void repopulateEffectsList(const std::vector<std::wstring> & filterIds, const std::vector<std::wstring> & filterNames) override;
#ifdef ENABLE_STYLETRANSFER
    bool isStylesListRequested() override;
    void repopulateStylesList(const std::vector<std::wstring> & styleIds, const std::vector<std::wstring> & styleNames, const std::vector<std::wstring> & stylePaths) override;
    bool isStyleNetworksListRequested() override;
    void repopulateStyleNetworksList(const std::vector<std::wstring> & netIds, const std::vector<std::wstring> & netNames);

    void showRestyleDownloadConfirmation() override;
    AnselUIBase::RestyleDownloadStatus getRestyleDownloadConfirmationStatus() override;
    void clearRestyleDownloadConfirmationStatus() override;

    void toggleRestyleProgressBar(bool status) override;
    void setRestyleProgressBarValue(float value) override;
    void setRestyleProgressState(RestyleProgressState progressState) override;
    void toggleRestyleProgressIndicator(bool status) override;

    const std::wstring & getLwrrentStyle() const override;
    const std::wstring & getLwrrentStylePath() const override;

    const std::wstring & getLwrrentStyleNetwork() const override;
    void setLwrrentStyleNetwork(const std::wstring & netid) override;

    void setStyleTransferStatus(bool isStyleTransferEnabled) override { }

    bool isUIInteractionActive() override;
    void mainSwitchStyleTransfer(bool status) override;
    bool isStyleTransferEnabled() const override;
#endif

    bool isStackFiltersListRequested() override;
    void stackFiltersListDone(const std::vector<std::wstring> & filterIds);

    bool isHighResolutionRecalcRequested() override;
    void highResolutionRecalcDone(const std::vector<HighResolutionEntry> & highResEntries) override;
    size_t getLwrrentFilterNum() const override;

    const std::wstring getLwrrentFilter(size_t effectStackIdx) const override;
    bool getLwrrentFilterInfoQuery(size_t effectStackIdx) override;

    bool getLwrrentFilterResetValues(size_t effectStackIdx) override;
    void lwrrentFilterResetValuesDone(size_t effectStackIdx) override;

    int getLwrrentFilterOldStackIdx(size_t effectStackIdx) const override;
    void updateLwrrentFilterStackIndices() override;

    // In order to report an error, just call `updateEffectControls(ANY, nullptr);`
    // TODO: probably we'll need some more detailed error reporting? But lwrrently all that UI needs to do - just drop filter selection to None and report that error was registered.
    void updateEffectControls(size_t effectIdx, EffectPropertiesDescription * effectDescription) override;    // Change to the attrib structure
    void updateEffectControlsInfo(size_t effectIdx, EffectPropertiesDescription * effectDescription) override;  // Change to the attrib structure

    bool isUpdateGameSpecificControlsRequired() override;
    void updateGameSpecificControls(EffectPropertiesDescription * effectDesc) override;
    void updateGameSpecificControlsInfo(EffectPropertiesDescription * effectDesc) override;

    virtual std::wstring getAppCMSID() const override;
    virtual std::wstring getAppShortName() const override;

    //  we need shotCaptureRequested return desc structure (with resolutions/multipliers etc, to potentially avoid unnecessary get/setters)
    virtual AnselUIBase::ShotDesc shotCaptureRequested() override;
    void shotCaptureDone(AnselUIBase::Status status) override;

    void onCaptureStarted(int numShotsTotal) override;
    void onCaptureTaken(int shotIdx) override;
    void onCaptureStopped(AnselUIBase::MessageType status) override;
    void onCaptureProcessingDone(int status, const std::wstring & absPath) override;

    bool isGridOfThirdsEnabled() const override;

    bool isShotEXR() const override;
    bool isShotJXR() const override;
    bool isShotPreviewRequired() const override;
    double getFOVDegrees() const override;
    double getRollDegrees() const override;
    bool processFOVChange() override;
    bool processRollChange() override;
    bool getCameraDragActive() override;
    // TODO: implement this
    bool isCameraInteractive() override { return true; }
    bool isHighresEnhance() const override;
    bool isHighQualityEnabled() const override;
    void setFOVDegrees(double fov) override;
    void setRollDegrees(double roll) override;
    
    // TODO: implement these
    bool isResetRollNeeded() const override { return false; }
    void resetRollDone() override {}
    void setResetRollStatus(bool isAvailable) override {}

    void setHighresEnhance(bool enhance) override {}
    void setHighQualityEnabled(bool setting) override;

    bool isAnselPrestartRequested() override;
    bool isAnselStartRequested() override;
    bool isAnselSDKSessionRequested() override;
    bool isAnselStopRequested() override;
    bool isAnselPoststopRequested() override;
    bool isAnselFeatureSetRequested() override;

    void anselPrestartDone(AnselUIBase::Status status, bool isSDKDetected, bool requireSDK) override;
    void anselStartDone(AnselUIBase::Status status) override;
    void anselStopDone(AnselUIBase::Status status) override;
    void anselPoststopDone(AnselUIBase::Status status) override;
    void anselFeatureSetRequestDone() override;

    bool isSDKCaptureAbortRequested() override;
    void sdkCaptureAbortDone(int status) override;

    void forceEnableUI() override;
    void forceDisableUI() override;
    bool isModdingAllowed() override;
    bool isModdingEnabled() override;
    bool queryIsModdingEnabled() override;

    // TODO: implement this
    void setDefaultEffectPath(const wchar_t * defaultEffectPath) override { };

    void updateSettings(const std::map<std::string, std::wstring>& settings) override;

    void update(double dt) override;
    std::vector<AnselUIBase::EffectChange>& getEffectChanges() override;
    void getEffectChangesDone() override;

    //this is the callback triggered for each input event
    void onInputEvent(const input::InputEvent& ev, const input::MomentaryKeyboardState& kbdSt,
        const input::MomentaryMouseState& mouseSt, const input::MomentaryGamepadState& gpadSt,
        const input::FolwsChecker& folwsChecker, const input::MouseTrapper& mouseTrapper) override;

    virtual input::InputHandler& getInputHandler() override;
    virtual const input::InputHandler& getInputHandler() const override;
    
    void exelwteBusMessages();
    
    input::InputHandlerForIPC& getInputHandlerForIpc();
    const input::InputHandlerForIPC& getInputHandlerForIpc() const;

    virtual void getTelemetryData(UISpecificTelemetryData &ret) const override;

    void emergencyAbort() override
    {
        // Nothing needs to be done (yet)
    }

    void setModdingStatus(bool isModdingAllowed) override;

    virtual void addGameplayOverlayNotification(NotificationType notificationType, ErrorManager::ErrorEntry notification, bool allowSame) override
    {
        // No notifications for the IPC UI
    }

    // TODO: implement these
    virtual void displayMessage(MessageType msgType) override;
    virtual void displayMessage(MessageType msgType, const std::vector<std::wstring> & parameters
#if ANSEL_SIDE_PRESETS
                        , bool removeLastLine = false
#endif
    ) override;

    virtual void reportFatalError(FatalErrorCode code, const std::string& filename, uint32_t line, const std::string& data) override;
    virtual void reportNonFatalError(uint32_t code, const std::string& filename, uint32_t line, const std::string& data) override;

    virtual bool isRequestingControl() override
    {
        return m_state.sdk.isPrestartRequested || m_state.sdk.isStartRequested;
    }
    virtual void rejectControlRequest() override
    {
        m_state.sdk.isPrestartRequested = false;
        m_state.sdk.isStartRequested = false;

        // TODO: send rejecting message?
        //m_ipc->sendAnselEnableResponse(kSessionRejected);
    }

private:

    bool m_wasCaptureAborted = false;

    void setAllFiltersToNone();
    void setFilterToNone(size_t effIdx);
    bool            m_needToSendAnselReady = false;
    bool            m_isIPCInitialized = false;
    AnselIPCMessageBusObserver*  m_ipc = nullptr;
    bool            m_ipcModeEnabled = false;
#if ANSEL_SIDE_PRESETS
public:
#endif
    IPCState          m_state;

};

#endif
