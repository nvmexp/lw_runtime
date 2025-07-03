#pragma once

// IPC

#if IPC_ENABLED == 1
#pragma warning(push)
#pragma warning(disable:4267)
#include "MessageBus.h"
#pragma warning(pop)
#include "UIBase.h"
#include "ipc.pb.h"

#include <mutex>
#include <functional>

struct IPCState;

namespace ipc
{
    void onInstall();
}

class AnselIPCMessageQueue
{
public:
    AnselIPCMessageQueue() = default;
    AnselIPCMessageQueue(const AnselIPCMessageQueue&) = delete;
    AnselIPCMessageQueue& operator=(const AnselIPCMessageQueue&) = delete;

    //to be called on the producer thread
    void parseAndPush(const std::string& data);

    //to be called on the consumer thread: {
    std::vector<AnselIpc::AnselIPCMessage>::const_iterator consumeBegin() const;
    std::vector<AnselIpc::AnselIPCMessage>::const_iterator consumeEnd() const;
    void swap();
    // }to be called on the consumer thread
protected:

    std::vector<AnselIpc::AnselIPCMessage>  m_queueA, m_queueB;
    bool m_isAConsumer;
    std::mutex  m_produceMutex;
};

class AnselIPCMessageBusObserver : public BusObserver
{
public:
    AnselIPCMessageBusObserver(IPCState& state);
    ~AnselIPCMessageBusObserver();
    void onBusMessage(const BusMessage& msg); //this happens on a separate thread! This and only this!
    void disconnectIpc();
    void exelwteBusMessages();
    uint32_t getHighresMultiplier() const;
    uint32_t get360Width() const;
    bool getAnselEnable() const;
    bool getAnselDisable() const;
    bool isAlive();
    void clearFlags();

    template<typename T, void(AnselIpc::AnselIPCResponse::*func)(T* obj)>
    void sendStatusResponse(AnselIpc::Status status);

    void sendGetStackInfoResponse(const std::vector<std::string>& stack);
    void sendRemoveFilterResponse(AnselIpc::Status status);
    void sendMoveFilterResponse(AnselIpc::Status status);
    void sendAnselEnableResponse(AnselIpc::Status status);
    void sendCaptureShotFinishedResponse(AnselIpc::Status status);
    void sendAbortCaptureResponse(AnselIpc::Status status);
    void sendSetStyleTransferEnabledResponse(AnselIpc::Status status);
    void sendSetStyleTransferStyleResponse(AnselIpc::Status status);

    void sendAnselCaptureModeResponse(bool);
    void sendAnselShotPermissionsResponse();
    void sendCaptureShotStartedResponse(AnselIpc::Status status, uint32_t shotCount);
    void sendCaptureProcessingDoneResponse(AnselIpc::Status status, const std::wstring & absFilename);
    void sendLogFilenameResponse(const std::string& filename);
    void sendIsAnselSDKIntegrationAvailableResponse(AnselIpc::Status status);
    void sendResetFilterValuesResponse(AnselIpc::Status status, int stackIdx);
    void sendGetFilterInfoResponse(AnselIpc::Status status, int stackIdx);
    void sendGetFilterInfoResponse(AnselIpc::Status status, int stackIdx, const AnselUIBase::EffectPropertiesDescription&);
    template<typename T, typename Q>
    void sendFilterResponse(Q func, AnselIpc::Status status, int stackIdx);
    template<typename T, typename Q>
    void sendFilterResponse(Q func, AnselIpc::Status status, int stackIdx, const AnselUIBase::EffectPropertiesDescription&);
    void sendSetFilterResponse(AnselIpc::Status status, int stackIdx);
    void sendSetFilterResponse(AnselIpc::Status status, int stackIdx, const AnselUIBase::EffectPropertiesDescription&);
    void sendSetFilterAndAttributeResponse(AnselIpc::Status status, int stackIdx);
    void sendSetFilterAndAttributeResponse(AnselIpc::Status status, int stackIdx, const AnselUIBase::EffectPropertiesDescription&, std::vector<std::pair<int, AnselIpc::Status> > setAttributeResponses);
    void sendInsertFilterResponse(AnselIpc::Status status, int stackIdx);
    void sendInsertFilterResponse(AnselIpc::Status status, int stackIdx, const AnselUIBase::EffectPropertiesDescription&);
    void sendLwrrentFovResponse();
    void sendCaptureProgressResponse(uint32_t shotNo);
    void sendGetScreenResolutionResponse();

    void sendFilterListResponse();
    void sendRemoveAllUIRequest();
    void sendAddUIElementRequest(AnselIpc::ControlType controlType, uint32_t id, 
        const std::string& text, const std::string& data, const std::map<int32_t, 
        std::string>& options, const std::vector<int32_t>& pulldownIds,
        void* value = nullptr);
    void sendGetHighresResolutionListResponse();
    void sendGetStyleTransferModelListResponse(const std::vector<std::wstring> & netIds, const std::vector<std::wstring> & netNames);
    void sendReportErrorResponse(AnselIpc::ErrorType type, uint32_t code, const std::string& filename, uint32_t line, const std::string& data);
    void sendGetSettingsResponse();
    void sendAnselStatusReportRequest(AnselIpc::Status status);
    void sendAnselStatusReportRequest(AnselIpc::Status status, const std::vector<std::string>&);
    void sendStyleTransferSideloadChoiceRequest();
    void sendUpdateRollRequest(float roll);
    void sendUpdateFovRequest(float fov);
    void sendGetEnabledFeatureSetRequest();
    void sendAnselReadyRequest();
    void sendGetFeatureSetResponse();
private:
    bool m_anselIpcEnable = false;
    bool m_anselIpcDisable = false;
    uint32_t m_highresMultiplier = 2u;
    uint32_t m_360Width = 4096u;
    
    void processUiControlChangedRequest(const AnselIpc::AnselIPCMessage&);
    void sendIsAnselAvailableResponse();
    void send360ResolutionRangeResponse();
    void sendFovRangeResponse();
    void sendUiReadyResponse();
    void sendRollRangeResponse();
    void sendIpcVersionResponse();
    void sendSetStyleTransferModelResponse(AnselIpc::Status status);
    void sendUiChangeResponse(AnselIpc::Status status);
    void sendResetEntireStackResponse(AnselIpc::Status status);
    void sendSetFovResponse(AnselIpc::Status status);
    void sendSetRollResponse(AnselIpc::Status status);
    void sendSetFilterAttributeRequest(AnselIpc::Status status);
    void sendSetGridOfThirdsEnabledResponse(AnselIpc::Status status);
    void sendInputEventResponse(AnselIpc::Status status);
    void sendSetLangIdResponse(AnselIpc::Status status);
    void sendGetProcessInfoResponse();
    void processCaptureShotRequest(const AnselIpc::AnselIPCMessage&);
    void processEstimateCaptureRequest(const AnselIpc::AnselIPCMessage&);
    void processSetFovRequest(const AnselIpc::AnselIPCMessage&);
    void processSetRollRequest(const AnselIpc::AnselIPCMessage&);
    void sendIsAnselModdingAvailableResponse();
    void sendStyleTransferSideloadProgressResponse(AnselIpc::SideloadProgress, int progress);
    void sendSetHighQualityResponse(AnselIpc::Status status);
    void sendSetCMSInfoResponse(AnselIpc::Status status);
    void handleInputEventRequest(const AnselIpc::InputEventRequest& inputEvent);

    template<typename T>
    void processAttributeType(
        std::wstring filterIdToSet, const size_t stackIdx,
        int valueCount, int controlidCount,
        std::function<T(int)> getValue,
        std::function<uint32_t(int)> getControlID);

    int getUIStackIdxFromAnselStackIdx(int stackIdx);
    AnselIpc::Status prepareSetFilter(size_t stackIdx, std::wstring filterIdToSet);
    void deleteFilter(size_t stackIdx);
    
#if ANSEL_SIDE_PRESETS
    void handleSetPreset(size_t stackIdx, std::wstring filterIdToSet, bool insert);
    void clearPreset(size_t filtersRemaining);
    void handleSetOverPreset(size_t stackIdx, std::wstring filterIdToSet);
    void handleDeletePreset();
    void sendPresetErrorMessage(AnselIpc::Status status, size_t stackIdx, std::wstring filterIdToset, std::wstring message);
    void updateTranslationLayer(size_t stackIdx);
    size_t colwertSetFilterStackIdx(size_t stackIdx);
#endif

    template<typename T>
    void setAttribute(
        std::wstring filterIdToSet, const size_t stackIdx,
        T* values, UINT numValues, uint32_t controlID, std::string controlName);

    IPCState& m_state;

    std::string m_lwrAnselIPCModule;
    MessageBus m_messageBus;
    AnselIPCMessageQueue  m_queue;
};

#endif