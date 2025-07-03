#include "UIIPC.h"
#include "Shlwapi.h"
#include "darkroom/StringColwersion.h"
#include "Log.h"
#include "ir/FileHelpers.h"
#include "CommonTools.h"

#if IPC_ENABLED == 1

using AnselIpc::Status;

bool IPCState::ClientMeetsMinimumVersion(unsigned int minMajor, unsigned int minMinor, unsigned int minPatch) const
{
    if (ipcClientVersionMajor > minMajor
        || ipcClientVersionMajor == minMajor && ipcClientVersionMinor > minMinor
        || ipcClientVersionMajor == minMajor && ipcClientVersionMinor == minMinor && ipcClientVersionPatch >= minPatch)
    {
        LOG_DEBUG("Passed client minimum IPC version check. Need:\"%d.%d.%d\" Have:\"%d.%d.%d\"", minMajor, minMinor, minPatch, ipcClientVersionMajor, ipcClientVersionMinor, ipcClientVersionPatch);
        return true;
    }
    LOG_WARN("Failed client minimum IPC version check. Need:\"%d.%d.%d\" Have:\"%d.%d.%d\"", minMajor, minMinor, minPatch, ipcClientVersionMajor, ipcClientVersionMinor, ipcClientVersionPatch);
    return false;
}

UIIPC::UIIPC() 
{
}

void UIIPC::checkInitObserver()
{
    if (!m_isIPCInitialized)
    {
        LOG_DEBUG("Creating the message bus observer");
        m_ipc = new AnselIPCMessageBusObserver(m_state);
        setAllFiltersToNone();
        m_state.m_inputstate.init();
        m_state.m_inputstate.addEventConsumer(this);
        m_isIPCInitialized = true;
        m_needToSendAnselReady = true;
#if ANSEL_SIDE_PRESETS
        m_state.filters.stackIdxTranslation.push_back(0);
#endif
    }
}

UIIPC::~UIIPC()
{
    m_state.m_inputstate.removeEventConsumer(this);
    release(); 
}

bool UIIPC::needToSendAnselReady() const
{
    return m_needToSendAnselReady;
}

bool UIIPC::sendAnselReady()
{
    LOG_DEBUG("Sending Ansel ready request (with creation counter)");
    if (!m_ipc->isAlive())
        return false;
    m_ipc->sendAnselReadyRequest();
    m_needToSendAnselReady = false;
    return true;
}

#ifdef ENABLE_STYLETRANSFER
bool UIIPC::isUIInteractionActive()
{
    return false;
}
#endif

void UIIPC::forceEnableUI() { }
void UIIPC::forceDisableUI() { }
void UIIPC::update(double dt)
{
    m_state.m_inputstate.update();
}
void UIIPC::release()
{
    delete m_ipc;
    m_ipc = nullptr;
    m_state.m_inputstate.deinit();
    m_isIPCInitialized = false;
}

void UIIPC::displayMessage(MessageType msgType)
{
    if ((size_t)msgType < (size_t)MessageType::kNUM_ENTRIES)
    {
        bool shouldSend = true;
        AnselIpc::Status status = AnselIpc::Status::kOk;
        
        if (msgType == MessageType::kCouldntSaveFile)
            status = AnselIpc::kCouldntSaveFile;
        else if (msgType == MessageType::kFailedToStartCaptureIlwalidArgument)
            status = AnselIpc::kFailedToStart;
        else if (msgType == MessageType::kFailedToStartCapturePathIncorrectOrPermissions ||
            msgType == MessageType::kCouldntCreateFileNotEnoughSpaceOrPermissions)
            status = AnselIpc::kPermissionDenied;
        else if (msgType == MessageType::kFailedToSaveShotFailedCreateDiretory)
            status = AnselIpc::kFailedToSaveShotFailedCreateDiretory;
        else if (msgType == MessageType::kFailedToStartCaptureNoSpaceLeft ||
            msgType == MessageType::kFailedToSaveShotNoSpaceLeft)
        {
            status = AnselIpc::kFailedToSaveShotNoSpaceLeft;
        }
        else if (msgType == MessageType::kStyle_CantStartDownload)
            status = AnselIpc::kStyleCantStartDownload;
        else if (msgType == MessageType::kStyle_Downloading)
            status = AnselIpc::kStyleDownloading;
        else if (msgType == MessageType::kStyle_Install_Success)
            status = AnselIpc::kStyleInstallSuccess;
        else if (msgType == MessageType::kStyle_Install_Failed)
            status = AnselIpc::kStyleInstallFailed;
        else if (msgType == MessageType::kStyle_ComputeCapTooLow)
            status = AnselIpc::kStyleComputeCapTooLow;
        else if (msgType == MessageType::kStyle_CouldntLoadLibrestyle)
            status = AnselIpc::kStyleCouldntLoadLibrestyle;
        else if (msgType == MessageType::kStyle_NoModelFound)
            status = AnselIpc::kStyleNoModelFound;
        else if (msgType == MessageType::kStyle_NotEnoughVRAM)
            status = AnselIpc::kStyleNotEnoughVRAM;
        else if (msgType == MessageType::kStyle_FailedToLoadStyle)
            status = AnselIpc::kStyleFailedToLoadStyle;
        else if (msgType == MessageType::kStyle_FailedToInitalizeStyleTransfer)
            status = AnselIpc::kStyleFailedToInitalizeStyleTransfer;
        else
            shouldSend = false;

        if (shouldSend)
        {
            std::vector<std::string> data;
            data.push_back(darkroom::getUtf8FromWstr(buildDisplayMessage(msgType)));
            m_ipc->sendAnselStatusReportRequest(status, data);
        }
    }
}

void UIIPC::displayMessage(MessageType msgType, const std::vector<std::wstring> & parameters
#if ANSEL_SIDE_PRESETS
    , bool removeLastLine
#endif
)
{
    if ((size_t)msgType < (size_t)MessageType::kNUM_ENTRIES)
    {
        std::vector<std::string> data(parameters.size());
        std::transform(parameters.cbegin(), parameters.cend(), data.begin(), [](const auto& p) { return darkroom::getUtf8FromWstr(p); });
        std::vector<std::wstring> parametersCopy = parameters;

        bool shouldSend = true;
        AnselIpc::Status status = AnselIpc::Status::kOk;
        if (msgType == MessageType::kEffectRequiresDepth)
        {
            status = AnselIpc::kEffectRequiresDepth;
            if (!parameters.empty())
            {
                parametersCopy[0] = shadermod::ir::filehelpers::GetFileName(parametersCopy[0]);
            }
        }
        else if (msgType == MessageType::kShotWithUISaved)
            status = AnselIpc::kShotWithUISaved;
        else if (msgType == MessageType::kUnableToSaveShotWithUI)
            status = AnselIpc::kUnableToSaveShotWithUI;
        else if (msgType == MessageType::kErrorParsingFile)
        {
            status = AnselIpc::kErrorParsingFile;
            if (!parameters.empty())
            {
                parametersCopy[0] = shadermod::ir::filehelpers::GetFileName(parametersCopy[0]);
            }
        }
        else if (msgType == MessageType::kShotSaved)
            status = AnselIpc::kShotSaved;
        else if (msgType == MessageType::kProcessingFile)
            status = AnselIpc::kProcessingFile;
        else if (msgType == MessageType::kFailedToSaveShot)
            status = AnselIpc::kFailedToSaveShot;
        else if (msgType == MessageType::kFailedToFinishCapture)
            status = AnselIpc::kFailedToFinishCapture;
        else if (msgType == MessageType::kProcessingCompleted)
            status = AnselIpc::kProcessingCompleted;
        else if (msgType == MessageType::kNFilesRemainingToProcess)
            status = AnselIpc::kNFilesRemainingToProcess;

        if (shouldSend)
        {
            data.push_back(darkroom::getUtf8FromWstr(buildDisplayMessage(msgType, parametersCopy)));
#if ANSEL_SIDE_PRESETS
            if (removeLastLine)
                data.pop_back();
#endif
            m_ipc->sendAnselStatusReportRequest(status, data);
        }
    }
}

void UIIPC::setFilterToNone(size_t effIdx)
{
    if (effIdx < m_state.filters.lwrrentFilterIds.size())
    {
        m_state.filters.lwrrentFilterIds[effIdx] = shadermod::Tools::wstrNone;
        m_state.filters.lwrrentFilterInfoQuery[effIdx] = false;
        m_state.filters.lwrrentFilterResetValues[effIdx] = false;
        m_state.filters.lwrrentFilterOldIndices[effIdx] = false;
        m_state.filters.filterSetResponseRequired[effIdx] = IPCState::FiltersSDKState::FilterResponseType::kNone;
    }
}

void UIIPC::setAllFiltersToNone()
{
    if (m_state.filters.lwrrentFilterIds.size() == 0)
    {
        m_state.filters.lwrrentFilterIds.resize(1);
        m_state.filters.lwrrentFilterInfoQuery.resize(1);
        m_state.filters.lwrrentFilterResetValues.resize(1);
        m_state.filters.lwrrentFilterOldIndices.resize(1);
        m_state.filters.filterSetResponseRequired.resize(1);
    }

    for (UINT i = 0; i < m_state.filters.lwrrentFilterIds.size(); i++)
    {
        setFilterToNone(i);
    }
}

bool UIIPC::isStackFiltersListRequested()
{
    const auto ret = m_state.filters.isStackFilterListRequested;
    m_state.filters.isStackFilterListRequested = false;
    return ret;
}

void UIIPC::stackFiltersListDone(const std::vector<std::wstring> & filterIds)
{
    std::vector<std::string> stack(filterIds.size());
    std::transform(filterIds.cbegin(), filterIds.cend(), stack.begin(), [](const auto& x) { return darkroom::getUtf8FromWstr(x); });
    m_ipc->sendGetStackInfoResponse(stack);
}

void UIIPC::reportFatalError(FatalErrorCode code, const std::string& filename, uint32_t line, const std::string& data)
{
    m_ipc->sendReportErrorResponse(AnselIpc::ErrorType::kFatal, uint32_t(code), filename, line, data);
}

void UIIPC::reportNonFatalError(uint32_t code, const std::string& filename, uint32_t line, const std::string& data)
{
    m_ipc->sendReportErrorResponse(AnselIpc::ErrorType::kNonFatal, code, filename, line, data);
}

bool UIIPC::isGridOfThirdsEnabled() const { return m_state.filters.isGridOfThirdsEnabled; }

LANGID UIIPC::getLangId() const { return MAKELANGID(m_state.lang, m_state.subLang); }
bool UIIPC::isSDKCaptureAbortRequested() { return m_state.sdk.captureAbortRequested; }
bool UIIPC::isEffectListRequested() { return m_state.filters.effectListRequested; }
#ifdef ENABLE_STYLETRANSFER
bool UIIPC::isStylesListRequested() { return m_state.restyle.styleListRequested; }
bool UIIPC::isStyleNetworksListRequested() { return m_state.restyle.networkListRequested; }

void UIIPC::showRestyleDownloadConfirmation()
{
    m_ipc->sendStyleTransferSideloadChoiceRequest();
}

AnselUIBase::RestyleDownloadStatus UIIPC::getRestyleDownloadConfirmationStatus()
{
    return m_state.restyle.downloadConfirmationStatus;
}

void UIIPC::clearRestyleDownloadConfirmationStatus()
{
    m_state.restyle.downloadConfirmationStatus = AnselUIBase::RestyleDownloadStatus::kNone;
}

void UIIPC::toggleRestyleProgressBar(bool status)
{
    m_state.restyle.showProgress = status;
    if (!m_state.restyle.showProgress)
        m_state.restyle.progressState = AnselIpc::SideloadProgress::kSideloadProgressIdle;
}

void UIIPC::setRestyleProgressBarValue(float value) 
{
    m_state.restyle.downloadProgress = int(value * 100.0f);
}

void UIIPC::setRestyleProgressState(RestyleProgressState progressState) 
{
    if (progressState == RestyleProgressState::kDownloadProgress)
        m_state.restyle.progressState = AnselIpc::SideloadProgress::kSideloadProgressDownloading;
    else if (progressState == RestyleProgressState::kInstalling)
        m_state.restyle.progressState = AnselIpc::SideloadProgress::kSideloadProgressInstalling;
    else
        m_state.restyle.progressState = AnselIpc::SideloadProgress::kSideloadProgressIdle;
}

void UIIPC::toggleRestyleProgressIndicator(bool status) 
{
    m_state.restyle.showProgress = status;
    if (!m_state.restyle.showProgress)
        m_state.restyle.progressState = AnselIpc::SideloadProgress::kSideloadProgressIdle;
}
#endif

size_t UIIPC::getLwrrentFilterNum() const { return m_state.filters.lwrrentFilterIds.size(); }
const std::wstring UIIPC::getLwrrentFilter(size_t effectStackIdx) const
{
    return (effectStackIdx < m_state.filters.lwrrentFilterIds.size()) ? m_state.filters.lwrrentFilterIds[effectStackIdx] : shadermod::Tools::wstrNone;
}
bool UIIPC::getLwrrentFilterInfoQuery(size_t effectStackIdx)
{
    return (effectStackIdx < m_state.filters.lwrrentFilterInfoQuery.size()) ? m_state.filters.lwrrentFilterInfoQuery[effectStackIdx] : false;
}
bool UIIPC::getLwrrentFilterResetValues(size_t effectStackIdx)
{
    return (effectStackIdx < m_state.filters.lwrrentFilterResetValues.size()) ? m_state.filters.lwrrentFilterResetValues[effectStackIdx] : false;
}
void UIIPC::lwrrentFilterResetValuesDone(size_t effectStackIdx)
{
    if (effectStackIdx < m_state.filters.lwrrentFilterResetValues.size())
    {
        m_state.filters.lwrrentFilterResetValues[effectStackIdx] = false;
        m_ipc->sendResetFilterValuesResponse(AnselIpc::Status::kOk, int(effectStackIdx));
    }
}

int UIIPC::getLwrrentFilterOldStackIdx(size_t effectStackIdx) const
{
    if (effectStackIdx < m_state.filters.lwrrentFilterOldIndices.size())
        return int(m_state.filters.lwrrentFilterOldIndices[effectStackIdx]);
    return int(effectStackIdx);
}

void UIIPC::updateLwrrentFilterStackIndices()
{
    for (size_t index = 0u; index < m_state.filters.lwrrentFilterOldIndices.size(); ++index)
        m_state.filters.lwrrentFilterOldIndices[index] = uint32_t(index);
}

#ifdef ENABLE_STYLETRANSFER
const std::wstring& UIIPC::getLwrrentStyle() const
{
    static const std::wstring DBGstyleNone;
    if (!m_state.restyle.lwrrentStyleFullPath.empty())
    {
        static std::wstring styleName;
        const auto start = PathFindFileName(m_state.restyle.lwrrentStyleFullPath.c_str());
        const auto end = PathFindExtension(start);
        styleName = decltype(styleName)(start, end);
        return styleName;
    }
    return DBGstyleNone;
}

const std::wstring& UIIPC::getLwrrentStylePath() const
{
    static const std::wstring DBGstyleNone;
    if (!m_state.restyle.lwrrentStyleFullPath.empty())
        return m_state.restyle.lwrrentStyleFullPath;
    return DBGstyleNone;
}

const std::wstring & UIIPC::getLwrrentStyleNetwork() const
{
    static const std::wstring DBGstyleNetworkNone;
    if (!m_state.restyle.lwrrentNetworkId.empty())
        return m_state.restyle.lwrrentNetworkId;
    return DBGstyleNetworkNone;
}
void UIIPC::setLwrrentStyleNetwork(const std::wstring & netid)
{
    m_state.restyle.lwrrentNetworkId = netid;
}
#endif

double UIIPC::getFOVDegrees() const { return m_state.sdk.cameraFov; }
double UIIPC::getRollDegrees() const { return m_state.sdk.cameraRoll; }
bool UIIPC::processFOVChange()
{
    if (m_state.sdk.isCameraFOVChanged)
    {
        m_state.sdk.isCameraFOVChanged = false;
        return true;
    }
    return false;
}
bool UIIPC::processRollChange() 
{ 
    if (m_state.sdk.isCameraRollChanged)
    {
        m_state.sdk.isCameraRollChanged = false;
        return true;
    }
    return false;
}
bool UIIPC::isAnselPrestartRequested() { return m_state.sdk.isPrestartRequested; }
bool UIIPC::isAnselStartRequested() { return m_state.sdk.isStartRequested; }
bool UIIPC::isAnselSDKSessionRequested() { return m_state.sdk.startAnselSDKSession; }
bool UIIPC::isAnselStopRequested() { return m_state.sdk.isStopRequested; }
bool UIIPC::isAnselPoststopRequested() { return m_state.sdk.isPoststopRequested; }
bool UIIPC::isAnselFeatureSetRequested() { return m_state.sdk.isFeatureSetRequested; }
void UIIPC::setFOVDegrees(double fov) 
{
    const auto fovFloat = float(fov);
    if (m_state.sdk.cameraFov != fovFloat)
        m_ipc->sendUpdateFovRequest(fovFloat);
    m_state.sdk.cameraFov = fovFloat;
}
void UIIPC::setRollDegrees(double roll) 
{
    const auto rollFloat = float(roll);
    if (m_state.sdk.cameraRoll != rollFloat)
        m_ipc->sendUpdateRollRequest(rollFloat);
    m_state.sdk.cameraRoll = rollFloat;
}
bool UIIPC::getCameraDragActive() { return m_state.sdk.cameraDragActive; }
void UIIPC::setAnselSDKDetected(bool detected) { m_state.sdk.sdkDetected = detected; }
void UIIPC::onCaptureStarted(int numShotsTotal) { m_ipc->sendCaptureShotStartedResponse(AnselIpc::Status::kOk, numShotsTotal); }
void UIIPC::onCaptureTaken(int shotIdx) { m_ipc->sendCaptureProgressResponse(shotIdx); }
void UIIPC::onCaptureProcessingDone(int status, const std::wstring & absPath) { m_ipc->sendCaptureProcessingDoneResponse(AnselIpc::Status::kOk, absPath); }
void UIIPC::onSessionStarted(bool isCameraWorksSessionActive) 
{ 
    m_state.sdk.isLwCameraSessionActive = true;
    m_state.sdk.isAnselSessionActive = isCameraWorksSessionActive; 
}
void UIIPC::onSessionStopped() 
{ 
    m_state.sdk.isLwCameraSessionActive = false;
    m_state.sdk.isAnselSessionActive = false; 
}
bool UIIPC::isHighResolutionRecalcRequested() { return m_state.sdk.isHighresResolutionListRequested; }
bool UIIPC::isShotEXR() const {
	return m_state.sdk.isShotEXR;
}
bool UIIPC::isShotJXR() const {
	return m_state.sdk.isShotJXR;
}
bool UIIPC::isShotPreviewRequired() const { return m_state.sdk.isShotPreviewRequired; }

void UIIPC::setModdingStatus(bool isModdingAvailable)
{
    m_state.filters.isModdingAvailable = isModdingAvailable;
}

bool UIIPC::queryIsModdingEnabled()
{
    if (!m_ipc->isAlive())
        return false;

    m_ipc->sendGetEnabledFeatureSetRequest();
    return true;
}

bool UIIPC::isModdingEnabled()
{
    return m_state.filters.isModdingEnabled;
}

bool UIIPC::isModdingAllowed()
{
    return m_state.filters.isModdingAllowed;
}

void UIIPC::updateSettings(const std::map<std::string, std::wstring>& settings)
{
    m_state.settings = settings;
}

bool UIIPC::isHighresEnhance() const
{
    return m_state.sdk.isHighresEnhancementRequested;
}

bool UIIPC::isHighQualityEnabled() const
{
    return m_state.sdk.isHighQualityEnabled;
}

void UIIPC::setHighQualityEnabled(bool setting)
{
    m_state.sdk.isHighQualityEnabled = setting;
}

#ifdef ENABLE_STYLETRANSFER
void UIIPC::mainSwitchStyleTransfer(bool status) { m_state.restyle.isAvailable = status; }
bool UIIPC::isStyleTransferEnabled() const { return m_state.restyle.isEnabled; }
#endif

void UIIPC::onCaptureStopped(AnselUIBase::MessageType msgType)
{
    // GFE client doesn't want to get CaptureShotFinished response in case shot was aborted
    if (!m_wasCaptureAborted)
    {
        bool shouldSend = true;
        AnselIpc::Status status = AnselIpc::Status::kOk;

        if (msgType == MessageType::kCouldntSaveFile)
            status = AnselIpc::kCouldntSaveFile;
        else if (msgType == MessageType::kFailedToStartCaptureIlwalidArgument)
            status = AnselIpc::kFailedToStart;
        else if (msgType == MessageType::kFailedToStartCapturePathIncorrectOrPermissions ||
            msgType == MessageType::kCouldntCreateFileNotEnoughSpaceOrPermissions)
            status = AnselIpc::kPermissionDenied;
        else if (msgType == MessageType::kFailedToSaveShotFailedCreateDiretory)
            status = AnselIpc::kFailedToSaveShotFailedCreateDiretory;
        else if (msgType == MessageType::kFailedToStartCaptureNoSpaceLeft ||
            msgType == MessageType::kFailedToSaveShotNoSpaceLeft)
            status = AnselIpc::kFailedToSaveShotNoSpaceLeft;

        if (shouldSend)
            m_ipc->sendCaptureShotFinishedResponse(status);
    }
    else
    {
        m_wasCaptureAborted = false;
    }
}

std::vector<AnselUIBase::EffectChange>& UIIPC::getEffectChanges()
{
    return m_state.filters.effectChanges;
}

void UIIPC::getEffectChangesDone()
{
    m_state.filters.effectChanges.resize(0);
}

void UIIPC::sdkCaptureAbortDone(int status)
{
    m_wasCaptureAborted = true;
    m_state.sdk.captureAbortRequested = false;
    m_ipc->sendAbortCaptureResponse(AnselIpc::Status::kOk);
}

void UIIPC::setShotTypePermissions(bool shotHDREnabled, const bool * shotTypeEnabled, int arraySize)
{
    if (shotTypeEnabled != NULL && arraySize != -1)
    {
        m_state.sdk.allowedShotTypes = std::vector<bool>(shotTypeEnabled, shotTypeEnabled + arraySize);
    }

    // m_state.sdk.allowedShotTypes must be initialized first.
    if (!m_state.sdk.allowedShotTypes.empty() && m_state.sdk.allowedHDR != shotHDREnabled)
    {
        LOG_DEBUG(" m_state.sdk.allowedHDR set from %s to %s", m_state.sdk.allowedHDR ? "true" : "false", shotHDREnabled ? "true" : "false");
        m_state.sdk.allowedHDR = shotHDREnabled;
        m_ipc->sendAnselShotPermissionsResponse();
    }
}

void UIIPC::setScreenSize(uint32_t width, uint32_t height)
{
    m_state.screenWidth = width;
    m_state.screenHeight = height;
}
void UIIPC::setFadeState(bool fadeState)
{
    m_state.fadeState = true;
}

bool UIIPC::isUpdateGameSpecificControlsRequired()
{
    return m_state.sdk.isGameSpecificControlUpdateRequired;
}

void UIIPC::updateGameSpecificControls(EffectPropertiesDescription * effectDesc)
{
    m_state.sdk.isGameSpecificControlUpdateRequired = false;
    m_ipc->sendRemoveAllUIRequest();

    if (effectDesc == nullptr)
        return;

    for (unsigned int uccnt = 0, ucend = (unsigned int)effectDesc->attributes.size(); uccnt < ucend; ++uccnt)
    {
        const EffectPropertiesDescription::EffectAttributes & lwrAttrib = effectDesc->attributes[uccnt];
        if (lwrAttrib.controlType == AnselUIBase::ControlType::kSlider)
        {
            float value = 0.5f;
            lwrAttrib.lwrrentValue.get(&value, 1);
            m_ipc->sendAddUIElementRequest(AnselIpc::ControlType::kControlSlider, lwrAttrib.controlId, darkroom::getUtf8FromWstr(lwrAttrib.displayName), "", {}, {}, &value);
        }
        else if (lwrAttrib.controlType == AnselUIBase::ControlType::kCheckbox)
        {
            float floatVal = 0.0f;
            lwrAttrib.lwrrentValue.get(&floatVal, 1);
            bool value = floatVal != 0.0f;
            m_ipc->sendAddUIElementRequest(AnselIpc::ControlType::kControlBoolean, lwrAttrib.controlId, darkroom::getUtf8FromWstr(lwrAttrib.displayName), "", {}, {}, &value);
        }
    }
}

void UIIPC::updateGameSpecificControlsInfo(EffectPropertiesDescription * effectDesc)
{
}

void UIIPC::updateEffectControls(size_t effectIdx, EffectPropertiesDescription * effectDescription)
{
    if (effectDescription == nullptr)
    {
        setFilterToNone(effectIdx);
    }

#if ANSEL_SIDE_PRESETS
    size_t basePresetIdx = (m_state.filters.activePreset ? m_state.filters.activePreset->stackIdx : -1);
    size_t presetFilterCount = (m_state.filters.activePreset ? m_state.filters.activePreset->filters.size() : 0);
    if (basePresetIdx != -1 && effectIdx >= basePresetIdx && effectIdx < (basePresetIdx + presetFilterCount))
    {
        // Ignore updateEffectControls for presets - they have no controls.
        return;
    }
#endif

    if (m_state.filters.filterSetResponseRequired[effectIdx] == IPCState::FiltersSDKState::FilterResponseType::kNone)
        return;


    if (effectDescription == nullptr)
    {
        if (m_state.filters.filterSetResponseRequired[effectIdx] == IPCState::FiltersSDKState::FilterResponseType::kSet)
            m_ipc->sendSetFilterResponse(AnselIpc::Status::kFailed, (int)effectIdx);
        else if (m_state.filters.filterSetResponseRequired[effectIdx] == IPCState::FiltersSDKState::FilterResponseType::kInsert)
            m_ipc->sendInsertFilterResponse(AnselIpc::Status::kFailed, (int)effectIdx);
        else if (m_state.filters.filterSetResponseRequired[effectIdx] == IPCState::FiltersSDKState::FilterResponseType::kSetFilterAndAttributes)
            m_ipc->sendSetFilterAndAttributeResponse(AnselIpc::Status::kFailed, (int)effectIdx);
    }
    else
    {
        if (m_state.filters.filterSetResponseRequired[effectIdx] == IPCState::FiltersSDKState::FilterResponseType::kSet)
            m_ipc->sendSetFilterResponse(AnselIpc::Status::kOk, (int)effectIdx, *effectDescription);
        else if (m_state.filters.filterSetResponseRequired[effectIdx] == IPCState::FiltersSDKState::FilterResponseType::kInsert)
            m_ipc->sendInsertFilterResponse(AnselIpc::Status::kOk, (int)effectIdx, *effectDescription);
        else if (m_state.filters.filterSetResponseRequired[effectIdx] == IPCState::FiltersSDKState::FilterResponseType::kSetFilterAndAttributes)
            m_ipc->sendSetFilterAndAttributeResponse(AnselIpc::Status::kOk, (int)effectIdx, *effectDescription, m_state.filters.setAttributeResponses[(UINT)effectIdx]);
    }

    m_state.filters.filterSetResponseRequired[effectIdx] = IPCState::FiltersSDKState::FilterResponseType::kNone;
    m_state.filters.setAttributeResponses.erase((UINT)effectIdx);
}

void UIIPC::updateEffectControlsInfo(size_t effectIdx, EffectPropertiesDescription * effectDescription)
{
    if (effectIdx < m_state.filters.lwrrentFilterInfoQuery.size())
        m_state.filters.lwrrentFilterInfoQuery[effectIdx] = false;
    if (effectDescription == nullptr)
    {
        m_ipc->sendGetFilterInfoResponse(AnselIpc::Status::kFailed, (int)effectIdx);
    }
    else
        m_ipc->sendGetFilterInfoResponse(AnselIpc::Status::kOk, (int)effectIdx, *effectDescription);
}

void UIIPC::repopulateEffectsList(const std::vector<std::wstring>& filterIds, const std::vector<std::wstring>& filterNames)
{
    if (filterIds.size() != filterNames.size())
    {
        // boom
    }

    auto& filters = m_state.filters.filterNames;
    filters.clear();

    for (auto i = 0u; i < filterIds.size(); ++i)
        filters.push_back(std::make_pair(filterIds[i], filterNames[i]));

    m_state.filters.effectListRequested = false;

    m_ipc->sendFilterListResponse();
}

#ifdef ENABLE_STYLETRANSFER
void UIIPC::repopulateStylesList(const std::vector<std::wstring>& styleIds, const std::vector<std::wstring> &styleNames, const std::vector<std::wstring> &stylePaths)
{
}

void UIIPC::repopulateStyleNetworksList(const std::vector<std::wstring> & netIds, const std::vector<std::wstring> & netNames)
{
    if (netIds.size() == netNames.size())
    {
        m_state.restyle.networkListRequested = false;
        // create a map netId -> netName out of the two vectors
        std::transform(netIds.begin(), netIds.end(), netNames.begin(), 
            std::inserter(m_state.restyle.netIds, m_state.restyle.netIds.end()), [](const std::wstring& a, const std::wstring& b) { return std::make_pair(a, b);    });
        if (m_state.restyle.dontSendStyleListResponse)
            m_state.restyle.dontSendStyleListResponse = false;
        else
        {
            m_ipc->sendGetStyleTransferModelListResponse(netIds, netNames);
        }
    }
}
#endif

void UIIPC::setFovControlEnabled(bool enabled)
{

}

void UIIPC::setFOVLimitsDegrees(double lo, double hi)
{
    m_state.sdk.fovMin = lo;
    m_state.sdk.fovMax = hi;
}
void UIIPC::setRollLimitsDegrees(double lo, double hi)
{
    m_state.sdk.rollMin = lo;
    m_state.sdk.rollMax = hi;
}
void UIIPC::set360WidthLimits(uint64_t lo, uint64_t hi)
{
    // TODO: change type to 64bit int here in the state/message
    m_state.sdk.pano360MinWidth = (uint32_t)lo;
    m_state.sdk.pano360MaxWidth = (uint32_t)hi;
}

void UIIPC::highResolutionRecalcDone(const std::vector<HighResolutionEntry> & highResEntries) 
{
    m_state.sdk.highresResolutions = highResEntries;
    m_ipc->sendGetHighresResolutionListResponse();
    m_state.sdk.isHighresResolutionListRequested = false;
}

std::wstring UIIPC::getAppCMSID() const
{
    return m_state.appCMSID;
}

std::wstring UIIPC::getAppShortName() const
{
    return m_state.appShortName;
}

AnselUIBase::ShotDesc UIIPC::shotCaptureRequested()
{
    AnselUIBase::ShotDesc shotDesc;

    shotDesc.shotType = m_state.sdk.shotTypeToTake;
    shotDesc.resolution360 = m_state.sdk.pano360ResultWidth;
    shotDesc.highResMult = m_state.sdk.highresMultiplier;
    shotDesc.thumbnailRequired = m_state.sdk.isThumbnailRequired;

    return shotDesc;
}

void UIIPC::shotCaptureDone(AnselUIBase::Status status) 
{
    m_state.sdk.isShotEXR = false;
    m_state.sdk.isShotPreviewRequired = false;
    m_state.sdk.shotTypeToTake = ShotType::kNone;
    if (status != AnselUIBase::Status::kOk)
        m_ipc->sendCaptureShotStartedResponse(AnselIpc::Status::kFailed, 0);
}

void UIIPC::anselPrestartDone(AnselUIBase::Status status, bool isSDKDetected, bool requireSDK)
{
    // TODO: do we need to propagate 'requireSDK' tio the GFE client?
    m_state.sdk.sdkDetected = isSDKDetected;
    m_state.sdk.isPrestartRequested = false;
}
void UIIPC::anselStartDone(AnselUIBase::Status status)
{
    if (status == AnselUIBase::Status::kOkAnsel)
    {
        m_ipc->sendAnselEnableResponse(AnselIpc::Status::kOkAnsel);
    }
    else if (status == AnselUIBase::Status::kOkFiltersOnly)
    {
        m_ipc->sendAnselEnableResponse(AnselIpc::Status::kOkModsOnly);
    }
    else
    {
        m_ipc->sendAnselEnableResponse(AnselIpc::Status::kProcessDeclined);
    }
    m_state.sdk.isStartRequested = false;
}
void UIIPC::anselStopDone(AnselUIBase::Status status)
{
    if (status != AnselUIBase::Status::kDeclined && status != AnselUIBase::Status::kUnknown)
        m_ipc->sendAnselEnableResponse(AnselIpc::Status::kOk);
    else
        m_ipc->sendAnselEnableResponse(AnselIpc::Status::kFailed);
    m_state.sdk.isStopRequested = false;
}
void UIIPC::anselPoststopDone(AnselUIBase::Status status)
{
    m_state.sdk.isPoststopRequested = false;
}

void UIIPC::anselFeatureSetRequestDone()
{
    m_state.sdk.isFeatureSetRequested = false;
    m_ipc->sendGetFeatureSetResponse();
}

void UIIPC::onInputEvent(const input::InputEvent& ev, const input::MomentaryKeyboardState& kbdSt,
    const input::MomentaryMouseState& mouseSt, const input::MomentaryGamepadState& gpadSt,
    const input::FolwsChecker& folwsChecker, const input::MouseTrapper& mouseTrapper)
{
    bool mouseDownThisTime = mouseSt.isButtonStateChangedToDown(input::EMouseButton::kLButton);
    bool mouseUpThisTime = mouseSt.isButtonStateChangedToUp(input::EMouseButton::kLButton);

    if (mouseUpThisTime)
    {
        m_state.sdk.cameraDragActive = false;
    }
    else if (mouseDownThisTime)
    {
        m_state.sdk.cameraDragActive = true;
    }
}

input::InputHandler& UIIPC::getInputHandler()
{
    return m_state.m_inputstate;
}

const input::InputHandler& UIIPC::getInputHandler() const
{
    return m_state.m_inputstate;
}

void UIIPC::exelwteBusMessages()
{
    if (m_isIPCInitialized)
        m_ipc->exelwteBusMessages();
}
    
input::InputHandlerForIPC& UIIPC::getInputHandlerForIpc()
{
    return m_state.m_inputstate;
}

const input::InputHandlerForIPC& UIIPC::getInputHandlerForIpc() const
{
    return m_state.m_inputstate;
}

void UIIPC::getTelemetryData(UISpecificTelemetryData &ret) const
{
    ret.roll = (float) getRollDegrees();
    ret.fov = (float) getFOVDegrees();
    ret.kindOfShot = m_state.sdk.shotTypeToTake;
    ret.isShotHDR = m_state.sdk.isShotEXR;

    ret.resolution360 = m_state.sdk.pano360ResultWidth;
    ret.highresMult = m_state.sdk.highresMultiplier;

    ret.usedGamepadForUIDuringTheSession = false; //can't define that on the standalone side when in IPC mode
}

void UIIPC::disconnectIpc()
{
    if (m_ipc)
        m_ipc->disconnectIpc();
}

#endif
