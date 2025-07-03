#pragma once

#include <stdint.h>

#include "CommonStructs.h"
#include "InputHandler.h"
#include "ui/common.h"
#include "ir/UserConstant.h"
#include "ErrorReporting.h"
#include "i18n/text.en-US.h"
#include "i18n/LocalizedStringHelper.h"

#define TEMP_SELECTABLE_FILTER_ID  -1
#define TEMP_ADJUSTMENTS_ID         1
#define TEMP_FX_ID                  2
#define TEMP_GAMESPECIFIC_ID        AnselUIBase::GameSpecificStackIdx

namespace shadermod
{
    class MultiPassEffect;
}

class AnselUIBase
{
public:

    enum class NotificationType
    {
        kWelcome = 0,
        kSessDeclined = 1,
        kDrvUpdate = 2,

        kPassthrough,

        kNUM_ENTRIES
    };
    enum class MessageType
    {
        kNone,
        kEffectRequiresDepth,
        kEffectRequiresHDR,
        kErrorParsingFile,
        kCouldntCreateFileNotEnoughSpaceOrPermissions,
        kCouldntSaveFile,
        kShotWithUISaved,
        kUnableToSaveShotWithUI,
        kFailedToStartCaptureIlwalidArgument,
        kFailedToStartCaptureNoSpaceLeft,
        kFailedToStartCapturePathIncorrectOrPermissions,
        kFailedToFinishCapture,
        kFailedToSaveShotFailedCreateDiretory,
        kFailedToSaveShotNoSpaceLeft,
        kProcessingCompleted,
        kNFilesRemainingToProcess,
        kShotSaved,
        kProcessingFile,
        kFailedToSaveShot,
        kModdingNetworkActivity,
        kStyle_CantStartDownload,
        kStyle_Downloading,
        kStyle_Install_Success,
        kStyle_Install_Failed,
        kStyle_ComputeCapTooLow,
        kStyle_CouldntLoadLibrestyle,
        kStyle_NoModelFound,
        kStyle_NotEnoughVRAM,
        kStyle_FailedToLoadStyle,
        kStyle_FailedToInitalizeStyleTransfer,
        kStyle_UserFolder,

        kNUM_ENTRIES
    };

    enum class Status
    {
        kOk = 0,
        kOkAnsel = 1,
        kOkFiltersOnly = 2,
        kDeclined = 3,
        kUnknown
    };

    // TODO: make it UI-impl specific
    //virtual HRESULT init(HMODULE hResourceModule, ID3D11Device* d3dDevice, input::InputState* inputCapture, AnselServer* pAnselServer, const std::wstring& installationFolderPath) = 0;
    
    virtual LANGID getLangId() const = 0;

    AnselUIBase() { populateTextDisplayMessages(); m_langID = LANGIDFROMLCID(GetUserDefaultLCID()); }

    virtual ~AnselUIBase() { releaseDisplayMessageIntermediateBuf(); }

    // Use `enum struct ShotType` defined in "CommonStructs.h" to navigate this array
    virtual void setShotTypePermissions(bool shotHDREnabled, const bool * shotTypeEnabled, int arraySize) = 0;

    virtual void setAnselSDKDetected(bool detected) = 0;
    virtual void onSessionStarted(bool isCameraWorksSessionActive) = 0;
    virtual void onSessionStopped() = 0;

    virtual void setScreenSize(uint32_t width, uint32_t height) = 0;

    virtual void setFadeState(bool fadeState) = 0;

    virtual bool isEffectListRequested() = 0;
    virtual void repopulateEffectsList(const std::vector<std::wstring> & filterIds, const std::vector<std::wstring> & filterNames) = 0;

    virtual bool isStackFiltersListRequested() = 0;
    virtual void stackFiltersListDone(const std::vector<std::wstring> & filterIds) = 0;

    enum class RestyleDownloadStatus
    {
        kNone,
        kConfirmed,
        kRejected,

        kNUM_ENTRIES
    };
    enum class RestyleProgressState
    {
        kDownloadProgress,
        kInstalling,

        kNUM_ENTRIES
    };

#ifdef ENABLE_STYLETRANSFER
    virtual void mainSwitchStyleTransfer(bool status) = 0;

    virtual bool isStylesListRequested() = 0;
    virtual void repopulateStylesList(const std::vector<std::wstring> & stylesIds, const std::vector<std::wstring> & stylesNames, const std::vector<std::wstring> & stylesPaths) = 0;

    virtual bool isStyleNetworksListRequested() = 0;
    virtual void repopulateStyleNetworksList(const std::vector<std::wstring> & netIds, const std::vector<std::wstring> & netNames) = 0;

    virtual const std::wstring & getLwrrentStyle() const = 0;
    virtual const std::wstring & getLwrrentStylePath() const = 0;

    virtual const std::wstring & getLwrrentStyleNetwork() const = 0;
    virtual void setLwrrentStyleNetwork(const std::wstring & netid) = 0;

    virtual void setStyleTransferStatus(bool isStyleTransferEnabled) = 0;

    virtual void showRestyleDownloadConfirmation() = 0;
    virtual RestyleDownloadStatus getRestyleDownloadConfirmationStatus() = 0;
    virtual void clearRestyleDownloadConfirmationStatus() = 0;

    virtual void toggleRestyleProgressBar(bool status) = 0;
    virtual void setRestyleProgressBarValue(float value) = 0;
    virtual void setRestyleProgressState(RestyleProgressState progressState) = 0;

    virtual void toggleRestyleProgressIndicator(bool status) = 0;

    virtual bool isUIInteractionActive() = 0;
    virtual bool isStyleTransferEnabled() const = 0;
#endif

    struct HighResolutionEntry
    {
        // TODO: add multipliers here too
        int64_t width, height;
        int64_t byteSize;
    };

    virtual bool isHighResolutionRecalcRequested() = 0;
    virtual void highResolutionRecalcDone(const std::vector<HighResolutionEntry> & highResEntries) = 0;

    virtual size_t getLwrrentFilterNum() const = 0;
    // These two functions better have event queue trigger alternative to avoid
    //  polling all the stack each frame
    virtual const std::wstring getLwrrentFilter(size_t effectStackIdx) const = 0;
    virtual bool getLwrrentFilterInfoQuery(size_t effectStackIdx) = 0;
    // Response is in the `updateEffectControlsInfo`
    virtual bool getLwrrentFilterResetValues(size_t effectStackIdx) = 0;
    virtual void lwrrentFilterResetValuesDone(size_t effectStackIdx) = 0;
    
    static const int oldStackIndexCreated = -1;
    virtual int getLwrrentFilterOldStackIdx(size_t effectStackIdx) const = 0;
    virtual void updateLwrrentFilterStackIndices() = 0;

    enum class ControlType
    {
        kSlider = 0,
        kCheckbox,
        kColorPicker,
        kFlyout,
        kEditbox,
        kRadioButton,

        kNUM_ENTRIES
    };

    enum class DataType
    {
        kFloat = 0,
        kBool,
        kInt,

        kNUM_ENTRIES
    };

    struct EffectPropertiesDescription
    {
        struct EffectAttributes
        {
            EffectAttributes() {}
            ~EffectAttributes() {}

            const shadermod::ir::UserConstant* userConstant;

            uint32_t controlId;
            std::wstring displayName;
            std::wstring displayNameEnglish;
            std::wstring uiMeasurementUnit;
            ControlType controlType = ControlType::kSlider;
            DataType dataType = DataType::kFloat;

            shadermod::ir::TypelessVariable defaultValue, lwrrentValue;
            shadermod::ir::TypelessVariable milwalue, maxValue;
            shadermod::ir::TypelessVariable uiMilwalue, uiMaxValue;
            shadermod::ir::TypelessVariable stepSizeUI;
            std::wstring valueDisplayName[4];

            float stickyValue, stickyRegion;

            static float getStepSize(float stepSizeUI_in, float milwalue_in, float maxValue_in, float uiMilwalue_in, float uiMaxValue_in)
            {
                return static_cast<float>(fabs(static_cast<long double>(stepSizeUI_in * (maxValue_in - milwalue_in) / (uiMaxValue_in - uiMilwalue_in))));
            }
        };

        EffectPropertiesDescription()
        {

        }

        std::wstring filterId;
        std::wstring filterDisplayName;
        std::wstring filterDisplayNameEnglish;
        std::vector<EffectAttributes> attributes;
    };

    static const uint32_t GameSpecificStackIdx = 0xFFffFFff;
    static const uint32_t ControlIDUnknown = -1;
    struct EffectChange
    {
        std::wstring filterId;
        uint32_t stackIdx = 0;
        uint32_t controlId = 0;
        std::string controlName;
        shadermod::ir::TypelessVariable value;
    };

    // In order to report an error, just call `updateEffectControls(ANY, nullptr);`
    // TODO: probably we'll need some more detailed error reporting? But lwrrently all that UI needs to do - just drop filter selection to None and report that error was registered.
    virtual void updateEffectControls(size_t effectStackIdx, EffectPropertiesDescription * effectDescription) = 0;
    virtual void updateEffectControlsInfo(size_t effectStackIdx, EffectPropertiesDescription * effectDescription) = 0;

    virtual bool isUpdateGameSpecificControlsRequired() = 0;
    virtual void updateGameSpecificControls(EffectPropertiesDescription * effectDescription) = 0;
    virtual void updateGameSpecificControlsInfo(EffectPropertiesDescription * effectDescription) = 0;

    virtual std::vector<EffectChange>& getEffectChanges() = 0;
    virtual void getEffectChangesDone() = 0;

    virtual std::wstring getAppCMSID() const = 0;
    virtual std::wstring getAppShortName() const = 0;

    struct ShotDesc
    {
        uint64_t resolution360 = 0;
        uint32_t highResMult = 0;
        ShotType shotType = ShotType::kNumEntries;
        bool thumbnailRequired = false;
    };

    //  we need shotCaptureRequested return desc structure (with resolutions/multipliers etc, to potentially avoid unnecessary get/setters)
    virtual ShotDesc shotCaptureRequested() = 0;
    virtual void shotCaptureDone(Status status) = 0;

    virtual void onCaptureStarted(int numShotsTotal) = 0;
    virtual void onCaptureTaken(int shotIdx) = 0;
    virtual void onCaptureStopped(AnselUIBase::MessageType status) = 0;
    virtual void onCaptureProcessingDone(int status, const std::wstring & absPath) = 0;

    virtual bool isGridOfThirdsEnabled() const = 0;

    virtual bool isShotEXR() const = 0;
    virtual bool isShotJXR() const = 0;
    virtual bool isShotPreviewRequired() const = 0;
    virtual double getFOVDegrees() const = 0;
    virtual double getRollDegrees() const = 0;
    virtual bool processFOVChange() = 0;
    virtual bool processRollChange() = 0;
    virtual bool getCameraDragActive() = 0;
    virtual bool isCameraInteractive() = 0;
    virtual bool isHighresEnhance() const = 0;
    virtual bool isHighQualityEnabled() const = 0;

    virtual void setFOVDegrees(double fov) = 0;
    virtual void setRollDegrees(double roll) = 0;
    virtual void setFovControlEnabled(bool enabled) = 0;
    virtual void setFOVLimitsDegrees(double lo, double hi) = 0;
    virtual void setRollLimitsDegrees(double lo, double hi) = 0;
    virtual void set360WidthLimits(uint64_t lo, uint64_t hi) = 0;
    virtual void setHighresEnhance(bool enhance) = 0;
    virtual void setHighQualityEnabled(bool setting) = 0;

    virtual bool isResetRollNeeded() const  = 0;
    virtual void resetRollDone() = 0;
    virtual void setResetRollStatus(bool isAvailable) = 0;

    virtual bool isAnselPrestartRequested() = 0;
    virtual bool isAnselStartRequested() = 0;
    virtual bool isAnselSDKSessionRequested() = 0;
    virtual bool isAnselStopRequested() = 0;
    virtual bool isAnselPoststopRequested() = 0;
    virtual bool isAnselFeatureSetRequested() = 0;

    virtual void anselPrestartDone(Status status, bool isSDKDetected, bool requireSDK) = 0;
    virtual void anselStartDone(Status status) = 0;
    virtual void anselStopDone(Status status) = 0;
    virtual void anselPoststopDone(Status status) = 0;
    virtual void anselFeatureSetRequestDone() = 0;

    virtual bool isSDKCaptureAbortRequested() = 0;
    virtual void sdkCaptureAbortDone(int status) = 0;

    virtual void forceEnableUI() = 0;
    virtual void forceDisableUI() = 0;

    virtual void update(double dt) = 0;

    virtual input::InputHandler& getInputHandler() = 0;
    virtual const input::InputHandler& getInputHandler() const = 0;

    virtual void getTelemetryData(UISpecificTelemetryData &) const = 0;
    
    virtual void emergencyAbort() = 0;

    virtual void setModdingStatus(bool isModdingAllowed) = 0;

    virtual bool isModdingAllowed() = 0;

    virtual bool queryIsModdingEnabled() = 0;
    virtual bool isModdingEnabled() = 0;

    virtual void setDefaultEffectPath(const wchar_t * defaultEffectPath) = 0;

    virtual void updateSettings(const std::map<std::string, std::wstring>& settings) = 0;

    virtual void addGameplayOverlayNotification(NotificationType notificationType, ErrorManager::ErrorEntry notification, bool allowSame) = 0;

    virtual void displayMessage(MessageType msgType) = 0;
    virtual void displayMessage(MessageType msgType, const std::vector<std::wstring> & parameters
#if ANSEL_SIDE_PRESETS
                                            , bool removeLastLine = false
#endif
    ) = 0;

    virtual void reportFatalError(FatalErrorCode code, const std::string& filename, uint32_t line, const std::string& data) = 0;
    virtual void reportNonFatalError(uint32_t code, const std::string& filename, uint32_t line, const std::string& data) = 0;

    virtual bool isRequestingControl() = 0;
    virtual void rejectControlRequest() = 0;

protected:
    std::vector<std::wstring> m_textDisplayMessages;
    LANGID m_langID = (LANGID)-1;
    void populateTextDisplayMessages()
    {
        m_textDisplayMessages.resize(static_cast<size_t>(AnselUIBase::MessageType::kNUM_ENTRIES));
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kEffectRequiresDepth] = i18n::getLocalizedString(IDS_EFFECTREQUIRESDEPTH, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kEffectRequiresHDR] = i18n::getLocalizedString(IDS_EFFECTREQUIRESDEPTH, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kErrorParsingFile] = i18n::getLocalizedString(IDS_ERRORPARSINGFILE, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kCouldntCreateFileNotEnoughSpaceOrPermissions] = i18n::getLocalizedString(IDS_COULDNTCREATEFILENOTENOUGHSPACEORPERMISSIONS, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kCouldntSaveFile] = i18n::getLocalizedString(IDS_COULDNTSAVEFILE, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kShotWithUISaved] = i18n::getLocalizedString(IDS_SHOTWITHUISAVED, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kUnableToSaveShotWithUI] = i18n::getLocalizedString(IDS_UNABLETOSAVESHOTWITHUI, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kFailedToStartCaptureIlwalidArgument] = i18n::getLocalizedString(IDS_FAILEDTOSTARTCAPTUREILWALIDARGUMENT, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kFailedToStartCaptureNoSpaceLeft] = i18n::getLocalizedString(IDS_FAILEDTOSTARTCAPTURENOSPACELEFT, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kFailedToStartCapturePathIncorrectOrPermissions] = i18n::getLocalizedString(IDS_FAILEDTOSTARTCAPTUREPATHINCORRECTORPERMISSIONS, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kFailedToFinishCapture] = i18n::getLocalizedString(IDS_FAILEDTOFINISHCAPTURE, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kFailedToSaveShotFailedCreateDiretory] = i18n::getLocalizedString(IDS_FAILEDTOSAVESHOTFAILEDCREATEDIRETORY, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kFailedToSaveShotNoSpaceLeft] = i18n::getLocalizedString(IDS_FAILEDTOSAVESHOTNOSPACELEFT, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kProcessingCompleted] = i18n::getLocalizedString(IDS_PROCESSINGCOMPLETED, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kNFilesRemainingToProcess] = i18n::getLocalizedString(IDS_NFILESREMAININGTOPROCESS, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kShotSaved] = i18n::getLocalizedString(IDS_SHOTSAVED, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kProcessingFile] = i18n::getLocalizedString(IDS_PROCESSINGFILE, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kFailedToSaveShot] = i18n::getLocalizedString(IDS_FAILEDTOSAVESHOT, m_langID);

        // TODO: add translation
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kModdingNetworkActivity] = L"Network activity detected!\n";

#ifdef ENABLE_STYLETRANSFER
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kStyle_CantStartDownload] = i18n::getLocalizedString(IDS_STYLE_CANTSTARTDOWNLOAD, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kStyle_Downloading] = i18n::getLocalizedString(IDS_STYLE_DOWNLOADING, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kStyle_Install_Success] = i18n::getLocalizedString(IDS_STYLE_INSTALL_SUCCESS, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kStyle_Install_Failed] = i18n::getLocalizedString(IDS_STYLE_INSTALL_FAILED, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kStyle_ComputeCapTooLow] = i18n::getLocalizedString(IDS_STYLE_COMPUTECAPTOOLOW, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kStyle_CouldntLoadLibrestyle] = i18n::getLocalizedString(IDS_STYLE_COULDNTLOADLIBRESTYLE, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kStyle_NoModelFound] = i18n::getLocalizedString(IDS_STYLE_NOMODELFOUND, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kStyle_NotEnoughVRAM] = i18n::getLocalizedString(IDS_STYLE_NOTENOUGHVRAM, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kStyle_FailedToLoadStyle] = i18n::getLocalizedString(IDS_STYLE_FAILEDTOLOADSTYLE, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kStyle_FailedToInitalizeStyleTransfer] = i18n::getLocalizedString(IDS_STYLE_FAILEDTOINITALIZESTYLETRANSFER, m_langID);
        m_textDisplayMessages[(size_t)AnselUIBase::MessageType::kStyle_UserFolder] = L"\U0001F4C2 %s";
#endif
    }

    size_t m_displayMessageIntermediateBufCapacity = 0;
    wchar_t * m_displayMessageIntermediateBuf = nullptr;
    void releaseDisplayMessageIntermediateBuf()
    {
        delete m_displayMessageIntermediateBuf;
        m_displayMessageIntermediateBuf = nullptr;
        m_displayMessageIntermediateBufCapacity = 0;
    }
    template<typename... Args>
    wchar_t * buildMessageIntermediateBuf(const wchar_t * format, Args... args)
    {
        // We need to use _snwprintf, which causes CRT security warning, so we need to disable it in order to
        // compile, due to /WX. We cannot use _snwprintf_s, since its behavior is different from _snwprintf -
        // it throws all sorts of exceptions and returns wrong results if buffer is too small or nullptr.
#pragma warning(push)
#pragma warning(disable:4996)
        size_t size = _snwprintf(nullptr, 0, format, args...) + 1; // Extra space for '\0'
#pragma warning(pop)

        if (m_displayMessageIntermediateBufCapacity < size)
        {
            releaseDisplayMessageIntermediateBuf();
        }

        if (m_displayMessageIntermediateBuf == nullptr)
        {
            m_displayMessageIntermediateBuf = new wchar_t[size];
            m_displayMessageIntermediateBufCapacity = size;
        }

#pragma warning(push)
#pragma warning(disable:4996)
        _snwprintf(m_displayMessageIntermediateBuf, size, format, args...);
#pragma warning(pop)

        return m_displayMessageIntermediateBuf;
    }

    const std::wstring buildDisplayMessage(MessageType msgType)
    {
        if ((size_t)msgType < (size_t)MessageType::kNUM_ENTRIES)
        {
            return m_textDisplayMessages[(size_t)msgType];
        }
        return L"Unknown message.";
    }
    const std::wstring buildDisplayMessage(MessageType msgType, const std::vector<std::wstring> & parameters
#if ANSEL_SIDE_PRESETS
                                            , bool removeLastLine = false
#endif
    )
    {
        if ((size_t)msgType < (size_t)MessageType::kNUM_ENTRIES)
        {
            const std::wstring msgFormat = m_textDisplayMessages[(size_t)msgType];

            const size_t argsNum = parameters.size();
            switch (argsNum)
            {
            case 1:
                buildMessageIntermediateBuf(msgFormat.c_str(), parameters[0].c_str());
                break;
            case 2:
                buildMessageIntermediateBuf(msgFormat.c_str(), parameters[0].c_str(), parameters[1].c_str());
                break;
            case 3:
                buildMessageIntermediateBuf(msgFormat.c_str(), parameters[0].c_str(), parameters[1].c_str(), parameters[2].c_str());
                break;
            case 4:
                buildMessageIntermediateBuf(msgFormat.c_str(), parameters[0].c_str(), parameters[1].c_str(), parameters[2].c_str(), parameters[3].c_str());
                break;
            default:
                assert(false && "Too many parameters for the error report");
                break;
            }

            size_t msgSize = wcslen(m_displayMessageIntermediateBuf);
            return std::wstring(m_displayMessageIntermediateBuf, m_displayMessageIntermediateBuf + msgSize);
        }
        return L"Unknown message.";
    }
};