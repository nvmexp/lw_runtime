// IPC
#include <windowsx.h>
#include "Config.h"
#define NOMINMAX
#include "Log.h"

#define ANSEL_DLL_EXPORTS
#include "Ansel.h"

#if IPC_ENABLED == 1

#define POCO_NO_AUTOMATIC_LIBS
#include "ipc/AnselIPC.h"
#include "ipc/UIIPC.h"
#include "darkroom/StringColwersion.h"
#include "darkroom/Director.h"
#include "GenericBusMessage.h"
#include "MessageBusSelwre.h"
#include "google\protobuf\text_format.h"
#include "RegistrySettings.h"
#include "AnselPreset.h"
#include "ir/FileHelpers.h"
#include "CommonTools.h"

#include "Allowlisting.h"
#include <lwapi.h>
#include <LwApiDriverSettings.h>

const char* AnselIPCSystem = "Ansel";
const char* AnselIPCModule = "LwCamera";
const char* GFEIPCSystem = AnselIPCSystem;
const char* GFEIPCModule = "LwCameraControl";

const float s_min360Resolution = 4096.0f;
const float s_max360Resolution = 8192.0f;

using namespace AnselIpc;

extern "C" ANSEL_DLL_API uint64_t __cdecl GetIpcVersion()
{
    IpcVersionResponse resp;
    uint64_t version = uint64_t(resp.major()) << 32;
    version |= uint64_t(resp.minor()) << 16;
    version |= uint64_t(resp.patch());
    return version;
}

void ApplyDRS(LwU32 keyID, LwU32 value, bool reportErrors = false)
{
    LwAPI_Initialize();
    LwAPI_Status ret;
    LwDRSSessionHandle hSession;
    LwDRSProfileHandle hProfile;

    ret = LwAPI_DRS_CreateSession(&hSession);
    ret = LwAPI_DRS_LoadSettings(hSession);
    ret = LwAPI_DRS_GetBaseProfile(hSession, &hProfile);
    if (ret != LWAPI_OK && reportErrors)
    {
        LOG_DEBUG("LwAPI_DRS_GetBaseProfile %d\n", ret);
    }

    {
        //ANSELENABLE
        LWDRS_SETTING lwApiSetting = { 0 };
        lwApiSetting.version = LWDRS_SETTING_VER;
        lwApiSetting.settingId = keyID;
        lwApiSetting.settingType = LWDRS_DWORD_TYPE;
        lwApiSetting.u32LwrrentValue = value;

        ret = LwAPI_DRS_SetSetting(hSession, hProfile, &lwApiSetting);
        if (ret != LWAPI_OK && reportErrors)
        {
            LOG_DEBUG("LwAPI_DRS_SetSetting %d\n", ret);
        }
    }

    ret = LwAPI_DRS_SaveSettings(hSession);
    if (ret != LWAPI_OK && reportErrors)
    {
        LOG_DEBUG("LwAPI_DRS_SaveSettings %d\n", ret);
    }
    ret = LwAPI_DRS_DestroySession(hSession);
    if (ret != LWAPI_OK && reportErrors)
    {
        LOG_DEBUG("LwAPI_DRS_DestroySession %d\n", ret);
    }
}

extern "C" ANSEL_DLL_API void __cdecl SetFreeStyleStatus(bool isFreeStyleEnabled)
{
    // Using same object as AnselServer uses to avoid reg path discrepancies
    RegistrySettings regSettings;
    std::wstring value = isFreeStyleEnabled ? L"True" : L"False";
    regSettings.setValue(regSettings.registryPathAnselWrite(), darkroom::getWstrFromUtf8(Settings::FreestyleEnabled).c_str(), value);
}

void FormatStringForPrinting(std::string& str)
{
    // Double the '%' chars so that formatted print works
    size_t percentCount = std::count(str.begin(), str.end(), '%');
    if (percentCount > 0)
    {
        size_t srcPos = str.size() - 1;
        str.resize(str.size() + percentCount);
        size_t dstPos = str.size() - 1;
        do
        {
            str[dstPos] = str[srcPos];
            // Double the '%' chars
            if ('%' == str[srcPos])
            {
                dstPos--;
                str[dstPos] = '%';
            }
            srcPos--;
            dstPos--;
        } while (srcPos != 0 && dstPos != srcPos);
    }
}

void LogIPCMessage(const AnselIPCMessage& message)
{
    // used to debug filter properties
    if (getLogSeverity() <= LogSeverity::kDebug)
    {
        const bool printAnselIpcMessage = true;
        if (printAnselIpcMessage)
        {
            using Printer = google::protobuf::TextFormat;
            std::string tmpstr;
            Printer::PrintToString(message, &tmpstr);
            FormatStringForPrinting(tmpstr);
            LOG_DEBUG(tmpstr.c_str());
        }
    }
}

namespace ipc
{
    void sendIpcVersionResponse(MessageBus& messageBus)
    {
        AnselIPCMessage message;
        AnselIPCResponse* response = new AnselIPCResponse;
        IpcVersionResponse* resp = new IpcVersionResponse;
        resp->set_major(resp->major());
        resp->set_minor(resp->minor());
        resp->set_patch(resp->patch());
        response->set_allocated_ipcversionresponse(resp);
        message.set_allocated_response(response);
        // TODO: change AnselIPCModule to something that complies with the new module naming convention
        GenericBusMessage msg(AnselIPCSystem, AnselIPCModule, message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
        messageBus.postMessage(msg);
        LogIPCMessage(message);
    }

    void onInstall()
    {
        MessageBus messageBus(getMessageBusInterfaceSelwre(MESSAGE_BUS_INTERFACE_VERSION));
        sendIpcVersionResponse(messageBus);
    }
}

namespace
{
    template<typename T, typename Q, typename S>
    void fillIpcPropFromAttributeValueHelper(T* prop, Q memberFunc, const AnselUIBase::EffectPropertiesDescription::EffectAttributes& value, unsigned int dims, S autoDeduceArg)
    {
        const unsigned int maxDims = MAX_GROUPED_VARIABLE_DIMENSION;
        S default_value[maxDims], current[maxDims], min[maxDims], max[maxDims];
        value.defaultValue.get(default_value, dims);
        value.lwrrentValue.get(current, dims);
        value.milwalue.get(min, dims);
        value.maxValue.get(max, dims);

        S uiMilwalue[maxDims], uiMaxValue[maxDims], stepSizeUI[maxDims];
        value.uiMilwalue.get(uiMilwalue, dims);
        value.uiMaxValue.get(uiMaxValue, dims);
        value.stepSizeUI.get(stepSizeUI, dims);

        for (unsigned int i = 0u; i < dims; ++i)
        {
            auto values = (prop->*memberFunc)();
            values->set_lwrrent(current[i]);
            values->set_default_(default_value[i]);
            values->set_minimum(min[i]);
            values->set_maximum(max[i]);

            // Callwlate the internal step size
            S stepSize = static_cast<S>(value.getStepSize(
                                                    static_cast<float>(stepSizeUI[i]),
                                                    static_cast<float>(min[i]),
                                                    static_cast<float>(max[i]),
                                                    static_cast<float>(uiMilwalue[i]),
                                                    static_cast<float>(uiMaxValue[i])
                                                ));

            values->set_stepsize(stepSize);

            values->set_stepsizeui(stepSizeUI[i]);

            values->set_milwalueui(uiMilwalue[i]);
            values->set_maxvalueui(uiMaxValue[i]);

            // Deprecated. Set for backwards compatability.
            prop->set_milwalueui(static_cast<float>(uiMilwalue[i]));
            prop->set_maxvalueui(static_cast<float>(uiMaxValue[i]));

            values->set_displayname(darkroom::getUtf8FromWstr(value.valueDisplayName[i]));
        }
    }

    template<typename T>
    void fillIpcPropFromAttributeValue(T* prop, const AnselUIBase::EffectPropertiesDescription::EffectAttributes& value)
    {
        const auto dims = value.defaultValue.getDimensionality();
        const auto type = value.defaultValue.getType();
        if (type == shadermod::ir::UserConstDataType::kFloat)
            fillIpcPropFromAttributeValueHelper(prop, &T::add_valuesfloat, value, dims, float());
        else if (type == shadermod::ir::UserConstDataType::kBool)
        {
            bool default_value[4], current[4], min[4], max[4];
            value.lwrrentValue.get(current, dims);
            value.defaultValue.get(default_value, dims);
            value.milwalue.get(min, dims);
            value.maxValue.get(max, dims);
            for (unsigned int i = 0u; i < dims; ++i)
            {
                auto values = prop->add_valuesbool();
                values->set_lwrrent(current[i]);
                values->set_default_(default_value[i]);
                values->set_minimum(min[i]);
                values->set_maximum(max[i]);
                values->set_displayname(darkroom::getUtf8FromWstr(value.valueDisplayName[i]));
            }

            // Deprecated. Set for backwards compatability.
            prop->set_milwalueui(0.0f);
            prop->set_maxvalueui(1.0f);
        }
        else if (type == shadermod::ir::UserConstDataType::kInt)
            fillIpcPropFromAttributeValueHelper(prop, &T::add_valuesint, value, dims, int());
        else if (type == shadermod::ir::UserConstDataType::kUInt)
            fillIpcPropFromAttributeValueHelper(prop, &T::add_valuesuint, value, dims, unsigned int());
        else
            LOG_ERROR("Unexpected filter attribute data type");
    }

    template<typename T>
    std::vector<T> apply_permutation(const std::vector<T>& input, const std::vector<uint32_t>& indices)
    {
        std::vector<T> result(input.size());
        for (size_t i = 0; i < input.size(); i++)
            result[i] = input[indices[i]];
        return result;
    }
}

//to be called on the producer thread
void AnselIPCMessageQueue::parseAndPush(const std::string& data)
{
    std::lock_guard<std::mutex> lock(m_produceMutex);

    if (m_isAConsumer)
    {
        m_queueB.resize(m_queueB.size() + 1);
        AnselIPCMessage& msg = m_queueB.back();
        msg.ParseFromArray(data.data(), int(data.size()));
        LOG_DEBUG("Parsing into queue B");
    }
    else
    {
        m_queueA.resize(m_queueA.size() + 1);
        AnselIPCMessage& msg = m_queueA.back();
        msg.ParseFromArray(data.data(), int(data.size()));
        LOG_DEBUG("Parsing into queue A");
    }
}

std::vector<AnselIPCMessage>::const_iterator AnselIPCMessageQueue::consumeBegin() const
{
    return m_isAConsumer ? m_queueA.begin() : m_queueB.begin();
}

std::vector<AnselIPCMessage>::const_iterator AnselIPCMessageQueue::consumeEnd() const
{
    return m_isAConsumer ? m_queueA.end() : m_queueB.end();
}

void AnselIPCMessageQueue::swap()
{
    if (m_isAConsumer)
        m_queueA.resize(0);
    else
        m_queueB.resize(0);

    std::lock_guard<std::mutex> lock(m_produceMutex);

    m_isAConsumer = !m_isAConsumer;
}

AnselIPCMessageBusObserver::AnselIPCMessageBusObserver(IPCState& state) : m_state(state), m_messageBus(getMessageBusInterfaceSelwre(MESSAGE_BUS_INTERFACE_VERSION))
{
    m_lwrAnselIPCModule = std::string(AnselIPCModule);
    m_messageBus.addObserver(this, AnselIPCSystem, m_lwrAnselIPCModule.c_str());
}

void AnselIPCMessageBusObserver::disconnectIpc()
{
    LOG_DEBUG("Disconnecting from IPC");
    m_messageBus.removeObserver(this);
}

AnselIPCMessageBusObserver::~AnselIPCMessageBusObserver()
{
    disconnectIpc();
}

bool getMouseDeltas(int lParam, bool isCoordsDelta, int *xPos, int *yPos)
{
    static int xPosPrev = 0, yPosPrev = 0;
    if (isCoordsDelta)
    {
        *xPos = GET_X_LPARAM(lParam);
        *yPos = GET_Y_LPARAM(lParam);
    }
    else
    {
        *xPos = GET_X_LPARAM(lParam) - xPosPrev;
        *yPos = GET_Y_LPARAM(lParam) - yPosPrev;
        xPosPrev = GET_X_LPARAM(lParam);
        yPosPrev = GET_Y_LPARAM(lParam);
    }

    return true;
}

void AnselIPCMessageBusObserver::onBusMessage(const BusMessage& msg)
{
    const bool logFullBusMessage = false;
    if (logFullBusMessage)
    {
        using Printer = google::protobuf::TextFormat;
        std::string tmpstr;
        Printer::PrintToString(msg, &tmpstr);
        LOG_DEBUG(tmpstr.c_str());
    }

    if (msg.has_joined())
    {
        const char* joined = msg.joined() ? "JOIN" : "LEAVE";
        LOG_DEBUG("Joined to MessageBus (%s, %s), skipping", msg.source_system().c_str(), msg.source_module().c_str());
    }

    if (!(msg.source_system() == AnselIPCSystem && msg.source_module() == GFEIPCModule))
    {
        LOG_DEBUG("Received something from another source (%s, %s), skipping", msg.source_system().c_str(), msg.source_module().c_str());
        return;
    }

    if (msg.has_generic())
    {
        const BusMessage_Generic& generic = msg.generic();
        const auto& data = generic.data();
        m_queue.parseAndPush(data);
    }
}

void AnselIPCMessageBusObserver::sendRemoveAllUIRequest()
{
    AnselIPCMessage message;
    AnselIPCRequest* request = new AnselIPCRequest;
    UIControlRemoveAllRequest* req = new UIControlRemoveAllRequest;
    request->set_allocated_uicontrolremoveallrequest(req);
    message.set_allocated_request(request);
    GenericBusMessage msg(AnselIPCSystem, AnselIPCModule, message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);
}

void AnselIPCMessageBusObserver::sendAddUIElementRequest(ControlType controlType,
    uint32_t id,
    const std::string& text,
    const std::string& data,
    const std::map<int32_t, std::string>& options,
    const std::vector<int32_t>& pulldownIds,
    void* value)
{
    static uint32_t requestId = 0;
    requestId += 1;
    if (controlType == kControlSlider)
    {
        AnselIPCMessage message;
        AnselIPCRequest* request = new AnselIPCRequest;
        AddUIElementRequest* req = new AddUIElementRequest;
        UIDescSlider* sld = new UIDescSlider;
        sld->set_id(id);
        sld->set_interval(0.0f);
        sld->set_milwalue(0.0f);
        sld->set_maxvalue(1.0f);
        sld->set_text(text);
        if (value == nullptr)
            sld->set_value(0.5f);
        else
            sld->set_value(*static_cast<float*>(value));
        req->set_allocated_uidescslider(sld);
        req->set_controltype(controlType);
        req->set_requestid(requestId);
        req->set_visible(true);
        request->set_allocated_adduielementrequest(req);
        message.set_allocated_request(request);
        GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
        m_messageBus.postMessage(msg);
        LOG_DEBUG("AddUIElementRequest: slider");
    }
    else if (controlType == kControlBoolean)
    {
        AnselIPCMessage message;
        AnselIPCRequest* request = new AnselIPCRequest;
        AddUIElementRequest* req = new AddUIElementRequest;
        UIDescBoolean* ckbx = new UIDescBoolean;
        ckbx->set_id(id);
        ckbx->set_text(text);
        if (value == nullptr)
            ckbx->set_set(false);
        else
            ckbx->set_set(*static_cast<bool*>(value));
        req->set_allocated_uidescboolean(ckbx);
        req->set_controltype(controlType);
        req->set_requestid(requestId);
        req->set_visible(true);
        request->set_allocated_adduielementrequest(req);
        message.set_allocated_request(request);
        GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
        m_messageBus.postMessage(msg);
        LOG_DEBUG("AddUIElementRequest: boolean");
    }
    else if (controlType == kControlLabel)
    {
        AnselIPCMessage message;
        AnselIPCRequest* request = new AnselIPCRequest;
        AddUIElementRequest* req = new AddUIElementRequest;
        UIDescLabel* label = new UIDescLabel;
        label->set_text(text);
        req->set_allocated_uidesclabel(label);
        req->set_controltype(controlType);
        req->set_requestid(requestId);
        req->set_visible(true);
        request->set_allocated_adduielementrequest(req);
        message.set_allocated_request(request);
        GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
        m_messageBus.postMessage(msg);
        LOG_DEBUG("AddUIElementRequest: label");
    }
    else if (controlType == kControlButton)
    {
        AnselIPCMessage message;
        AnselIPCRequest* request = new AnselIPCRequest;
        AddUIElementRequest* req = new AddUIElementRequest;
        UIDescButton* button = new UIDescButton;
        button->set_id(id);
        button->set_text(text);
        req->set_allocated_uidescbutton(button);
        req->set_controltype(controlType);
        req->set_requestid(requestId);
        req->set_visible(true);
        request->set_allocated_adduielementrequest(req);
        message.set_allocated_request(request);
        GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
        m_messageBus.postMessage(msg);
        LOG_DEBUG("AddUIElementRequest: button");
    }
    else if (controlType == kControlEdit)
    {
        AnselIPCMessage message;
        AnselIPCRequest* request = new AnselIPCRequest;
        AddUIElementRequest* req = new AddUIElementRequest;
        UIDescEdit* edit = new UIDescEdit;
        edit->set_id(id);
        edit->set_text(text);
        edit->set_allowedtype(EditAllowedType::kFloat);
        edit->set_data(data.c_str());
        req->set_allocated_uidescedit(edit);
        req->set_controltype(controlType);
        req->set_requestid(requestId);
        req->set_visible(true);
        request->set_allocated_adduielementrequest(req);
        message.set_allocated_request(request);
        GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
        m_messageBus.postMessage(msg);
        LOG_DEBUG("AddUIElementRequest: edit");
    }
    else if (controlType == kControlList || controlType == kControlRadioButton)
    {
        AnselIPCMessage message;
        AnselIPCRequest* request = new AnselIPCRequest;
        AddUIElementRequest* req = new AddUIElementRequest;
        UIDescList* list = new UIDescList;
        list->set_id(id);
        list->set_text(text);
        list->set_selected(0);
        list->set_type(ListSelectionType::kFlyout);
        for (const auto& k : options)
        {
            auto values = list->add_values();
            values->set_key(k.first);
            values->set_value(k.second);
        }
        req->set_allocated_uidesclist(list);
        req->set_controltype(controlType);
        req->set_requestid(requestId);
        req->set_visible(true);
        request->set_allocated_adduielementrequest(req);
        message.set_allocated_request(request);
        GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
        m_messageBus.postMessage(msg);
        LOG_DEBUG("AddUIElementRequest: list");
    }
    else if (controlType == kControlPulldown)
    {
        AnselIPCMessage message;
        AnselIPCRequest* request = new AnselIPCRequest;
        AddUIElementRequest* req = new AddUIElementRequest;
        UIDescPulldown* pulldown = new UIDescPulldown;
        pulldown->set_id(id);
        pulldown->set_text(text.c_str());
        for (auto id : pulldownIds)
            pulldown->add_controlidlist(id);
        req->set_allocated_uidescpulldown(pulldown);
        req->set_controltype(controlType);
        req->set_requestid(requestId);
        req->set_visible(true);
        request->set_allocated_adduielementrequest(req);
        message.set_allocated_request(request);
        GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
        m_messageBus.postMessage(msg);
        LOG_DEBUG("AddUIElementRequest: pulldown");
    }
    else
    {
        LOG_DEBUG("AddUIElementRequest was not sent: unrecognized control type");
    }
}

template<typename T>
void AnselIPCMessageBusObserver::setAttribute(std::wstring filterIdToSet, const size_t stackIdx,
    T* values, UINT numValues, uint32_t controlID, std::string controlName )
{
    AnselUIBase::EffectChange effectChange;
    effectChange.filterId = filterIdToSet;
    effectChange.stackIdx = int(stackIdx);
    effectChange.controlId = controlID;
    effectChange.value = shadermod::ir::TypelessVariable(values, numValues);
    effectChange.controlName = controlName;
    m_state.filters.effectChanges.push_back(effectChange);
    m_state.filters.setAttributeResponses[(UINT)stackIdx].push_back(std::pair<int, AnselIpc::Status>(effectChange.controlId, kOk));

    LOG_DEBUG("    effectChange(controlId: %i, value.%s: %s )", effectChange.controlId, typeid(T).name(), effectChange.value.stringify().c_str());
}


AnselIpc::Status AnselIPCMessageBusObserver::prepareSetFilter(size_t stackIdx, std::wstring filterIdToSet)
{
    // Fill missing pieces with "None"
    if (stackIdx >= m_state.filters.lwrrentFilterIds.size())
    {
        const auto prevFiltersNum = m_state.filters.lwrrentFilterIds.size();
        const auto newSize = stackIdx + 1;
        m_state.filters.lwrrentFilterIds.resize(newSize);
        m_state.filters.lwrrentFilterInfoQuery.resize(newSize);
        m_state.filters.lwrrentFilterResetValues.resize(newSize);
        m_state.filters.lwrrentFilterOldIndices.resize(newSize);
        m_state.filters.filterSetResponseRequired.resize(newSize);

        for (size_t i = prevFiltersNum; i < stackIdx; ++i)
        {
            m_state.filters.lwrrentFilterIds[i] = shadermod::Tools::wstrNone;
            m_state.filters.lwrrentFilterInfoQuery[i] = false;
            m_state.filters.lwrrentFilterResetValues[i] = false;
            m_state.filters.lwrrentFilterOldIndices[i] = uint32_t(i);
            m_state.filters.filterSetResponseRequired[i] = IPCState::FiltersSDKState::FilterResponseType::kSet;
#if ANSEL_SIDE_PRESETS
            m_state.filters.stackIdxTranslation.push_back((unsigned int) i);
#endif
        }
        m_state.filters.lwrrentFilterInfoQuery[stackIdx] = false;
        m_state.filters.lwrrentFilterResetValues[stackIdx] = false;
        m_state.filters.lwrrentFilterOldIndices[stackIdx] = uint32_t(stackIdx);
    }

    if (m_state.filters.lwrrentFilterIds[stackIdx] == filterIdToSet)
    {
        return kAlreadySet;
    }

    return kOk;
}

void AnselIPCMessageBusObserver::deleteFilter(size_t stackIdx)
{
    m_state.filters.lwrrentFilterIds.erase(m_state.filters.lwrrentFilterIds.begin() + stackIdx);
    m_state.filters.lwrrentFilterInfoQuery.erase(m_state.filters.lwrrentFilterInfoQuery.begin() + stackIdx);
    m_state.filters.lwrrentFilterResetValues.erase(m_state.filters.lwrrentFilterResetValues.begin() + stackIdx);
    m_state.filters.lwrrentFilterOldIndices.erase(m_state.filters.lwrrentFilterOldIndices.begin() + stackIdx);
    m_state.filters.filterSetResponseRequired.erase(m_state.filters.filterSetResponseRequired.begin() + stackIdx);
}

#if ANSEL_SIDE_PRESETS

void AnselIPCMessageBusObserver::sendPresetErrorMessage(AnselIpc::Status status, size_t stackIdx, std::wstring filterIdToSet, std::wstring message)
{
    IPCState::FiltersSDKState::AnselPresetError error;
    error.status = status;
    error.stackIdx = stackIdx;
    error.filterId = filterIdToSet;
    error.message = message;

    m_state.filters.presetErrors.push_back(error);
}

void AnselIPCMessageBusObserver::handleSetPreset(size_t stackIdx, std::wstring filterIdToSet, bool insert)
{
    if (m_state.filters.activePreset != nullptr)
    {
        if (m_state.filters.activePreset->presetID.compare(filterIdToSet) == 0)
        {
            // If the preset being set is also the one already set, we treat it as though it is being moved in the stack
            // We do this to prevent the case where a filter before a preset is removed, and so the stack is updated via GFE re-setting all filters
            // In this situation, it will try to set the preset while the current preset still exists.
            // This will probably work, until GFE changes its behavior
            handleSetOverPreset(stackIdx, filterIdToSet);
        }
        else
        {
            sendPresetErrorMessage(kAlreadySet, stackIdx, filterIdToSet, L"Only one preset can be applied at a time.");
            return;
        }
    }

    m_state.filters.activePreset = new AnselPreset();
    m_state.filters.activePreset->stackIdx = stackIdx;

    std::vector<std::wstring> unappliedFilters;
    AnselIpc::Status status = ParseAnselPreset(filterIdToSet, m_state.filters.filterNames, m_state.filters.activePreset, unappliedFilters);
    if (unappliedFilters.size() != 0)
    {
        // Some filters have not been applied successfully
        std::wstring fullError;
        if (m_state.filters.activePreset->filters.size() == 0)
        {
            fullError = L"Warning: none of the filters in this preset are available: ";
        }
        else
        {
            fullError = L"Warning: some filters in this preset are not available: ";
        }

        // Join based on https://www.oreilly.com/library/view/c-cookbook/0596007612/ch04s09.html
        for (std::vector<std::wstring>::const_iterator p = unappliedFilters.begin(); p != unappliedFilters.end(); ++p)
        {
            LOG_WARN("Filter %s unavailable in preset %s", (*p).c_str(), filterIdToSet.c_str());
            fullError += *p;
            if (p != unappliedFilters.end() - 1)
                fullError += L", ";
        }
        sendPresetErrorMessage(AnselIpc::Status::kEffectRequiresDepth, stackIdx, filterIdToSet, fullError);
    }

    if (status != kOk)
    {
        if (unappliedFilters.size() == 0)
        {
            sendPresetErrorMessage(status, stackIdx, filterIdToSet, L"Failure parsing preset.");
        }
        delete m_state.filters.activePreset;
        m_state.filters.activePreset = NULL;
        return;
    }

    for (size_t i = 0; i < m_state.filters.activePreset->filters.size(); i++)
    {
        if (insert || i > 0)
        {
            // We insert after the i == 0 case because we need to avoid overwriting other filters when setting multiple filters
            m_state.filters.lwrrentFilterIds.insert(m_state.filters.lwrrentFilterIds.begin() + stackIdx + i, m_state.filters.activePreset->filters[i].filterID);
            m_state.filters.lwrrentFilterInfoQuery.insert(m_state.filters.lwrrentFilterInfoQuery.begin() + stackIdx + i, false);
            m_state.filters.lwrrentFilterResetValues.insert(m_state.filters.lwrrentFilterResetValues.begin() + stackIdx + i, false);
            m_state.filters.lwrrentFilterOldIndices.insert(m_state.filters.lwrrentFilterOldIndices.begin() + stackIdx + i, (uint32_t) (stackIdx+i));
            m_state.filters.filterSetResponseRequired.insert(m_state.filters.filterSetResponseRequired.begin() + stackIdx + i, IPCState::FiltersSDKState::FilterResponseType::kNone);
        }
        else
        {
            prepareSetFilter(stackIdx+i, m_state.filters.activePreset->filters[i].filterID);
            m_state.filters.lwrrentFilterIds[stackIdx+i] = m_state.filters.activePreset->filters[i].filterID;
        }

        for (auto attribute : m_state.filters.activePreset->filters[i].attributes)
        {
            // Here, we preform a truncating colwersion to std::string. This will not work correctly for filters with special characters requiring std::wstring to represent
            setAttribute(m_state.filters.activePreset->filters[i].filterID, stackIdx + i, attribute.second.data(), (UINT) attribute.second.size(), AnselUIBase::ControlIDUnknown, darkroom::getUtf8FromWstr(attribute.first));
        }
    }

    if (insert)
    {
        m_state.filters.filterSetResponseRequired[stackIdx] = IPCState::FiltersSDKState::FilterResponseType::kInsert;
    }
    else
    {
        m_state.filters.filterSetResponseRequired[stackIdx] = IPCState::FiltersSDKState::FilterResponseType::kSet;
    }

    // update translation layer
    for (size_t i = 0; i < m_state.filters.stackIdxTranslation.size(); i++)
    {
        updateTranslationLayer(i);
    }
}

void AnselIPCMessageBusObserver::clearPreset(size_t filtersRemaining)
{
    assert(m_state.filters.activePreset);

    // Remove all the preset filters, except the first filtersRemaining filters.
    for (size_t i = m_state.filters.activePreset->filters.size(); i > filtersRemaining; i--)
    {
        // We remove from the end first, because the size of the filter stack changes when we do this.
        deleteFilter(m_state.filters.activePreset->stackIdx + i-1);
    }

    // Then, update the translation layer
    // Without a preset, the translation layer should just be a passthrough array
    for (uint32_t i = 0; i < m_state.filters.stackIdxTranslation.size(); i++)
    {
        m_state.filters.stackIdxTranslation[i] = i;
    }

    // Then, set the active preset to null
    delete m_state.filters.activePreset;
    m_state.filters.activePreset = nullptr;
}

void AnselIPCMessageBusObserver::handleDeletePreset()
{
    clearPreset(0);
}

void AnselIPCMessageBusObserver::handleSetOverPreset(size_t stackIdx, std::wstring filterIdToSet)
{
    // We want this to look as though we were just replacing the first filter in the preset with the new thing
    clearPreset(1);
}

void AnselIPCMessageBusObserver::updateTranslationLayer(size_t stackIdx)
{
    if (stackIdx == 0)
    {
        m_state.filters.stackIdxTranslation[stackIdx] = 0;
    }
    else if (m_state.filters.activePreset && stackIdx == m_state.filters.activePreset->stackIdx + 1)
    {
        m_state.filters.stackIdxTranslation[stackIdx] = (uint32_t)(m_state.filters.stackIdxTranslation[stackIdx - 1] + m_state.filters.activePreset->filters.size());
    }
    else
    {
        m_state.filters.stackIdxTranslation[stackIdx] = m_state.filters.stackIdxTranslation[stackIdx - 1] + 1;
    }
}

size_t AnselIPCMessageBusObserver::colwertSetFilterStackIdx(size_t stackIdx)
{
    // colwert through translation layer
    size_t oldMaxStackIdx = m_state.filters.stackIdxTranslation.size();
    if (stackIdx >= oldMaxStackIdx)
    {
        m_state.filters.stackIdxTranslation.resize(stackIdx+1);
    }
    for (size_t i = oldMaxStackIdx; i <= stackIdx; i++)
    {
        updateTranslationLayer(i);
    }

    return m_state.filters.stackIdxTranslation[stackIdx];
}
#endif

template<typename T>
void AnselIPCMessageBusObserver::processAttributeType(
    std::wstring filterIdToSet, const size_t stackIdx,
    int valueCount, int controlIDCount,
    std::function<T(int)> getValue,
    std::function<uint32_t(int)> getControlID)
{
    assert(valueCount == controlIDCount);
    int validCount = std::min(valueCount, controlIDCount);
    /* Group up multidimensional values for the same ID */
    std::unordered_map<UINT, std::pair<UINT, T[MAX_GROUPED_VARIABLE_DIMENSION]> > groupedValuesPerID;
    for (int i = 0; i < validCount; i++)
    {
        uint32_t lwrControlId = getControlID(i);
        T lwrValue = getValue(i);
        if (groupedValuesPerID.find(lwrControlId) == groupedValuesPerID.end()) groupedValuesPerID[lwrControlId].first = 0;
        groupedValuesPerID[lwrControlId].second[groupedValuesPerID[lwrControlId].first] = lwrValue;
        groupedValuesPerID[lwrControlId].first++;
    }

    for (auto controlIDItr = groupedValuesPerID.begin(); controlIDItr != groupedValuesPerID.end(); controlIDItr++)
    {
        setAttribute(filterIdToSet, stackIdx, controlIDItr->second.second, controlIDItr->second.first, controlIDItr->first, "");
    }

    /* Number of values must match number of controlIDs */
    /* Report errors if valueCount != controlIDCount */
    for (int i = validCount; i < valueCount; i++)
    {
        m_state.filters.setAttributeResponses[(UINT)stackIdx].push_back(std::pair<int, AnselIpc::Status>(-1, kIlwalidRequest));
    }

    for (int i = validCount; i < controlIDCount; i++)
    {
        uint32_t controlId = getControlID(i);
        m_state.filters.setAttributeResponses[(UINT)stackIdx].push_back(std::pair<int, AnselIpc::Status>(controlId, kIlwalidRequest));
    }
}

#define ProcessVarType(varType, varDeclaration) processAttributeType<varDeclaration>( \
    filterIdToSet, stackIdx, \
    message.request().setfilterandattributesrequest().varType##values_size(), \
    message.request().setfilterandattributesrequest().varType##controlids_size(), \
    [&message](auto i) -> varDeclaration {return message.request().setfilterandattributesrequest().varType##values().Get(i);}, \
    [&message](auto i) -> uint32_t {return message.request().setfilterandattributesrequest().varType##controlids().Get(i);})

void AnselIPCMessageBusObserver::exelwteBusMessages()
{
    m_queue.swap();

    for (auto it = m_queue.consumeBegin(), end = m_queue.consumeEnd(); it != end; ++it)
    {
        const AnselIPCMessage& message = *it;

        // used to debug incoming messages
        if (getLogSeverity() <= LogSeverity::kDebug)
        {
            const bool printAnselIpcMessage = true;
            if (printAnselIpcMessage)
            {
                using Printer = google::protobuf::TextFormat;
                std::string tmpstr;
                Printer::PrintToString(message, &tmpstr);
                FormatStringForPrinting(tmpstr);
                LOG_DEBUG(tmpstr.c_str());
            }
        }

        switch (message.message_case())
        {
        case AnselIPCMessage::MessageCase::kRequest:
        {
            switch (message.request().request_case())
            {
            case AnselIPCRequest::RequestCase::kSetGridOfThirdsEnabledRequest:
            {
                LOG_DEBUG("kSetGridOfThirdsEnabledRequest received");
                // only allow this when Ansel session is active
                if (m_state.sdk.isAnselSessionActive)
                {
                    const auto& request = message.request().setgridofthirdsenabledrequest();
                    m_state.filters.isGridOfThirdsEnabled = request.enabled();
                    sendSetGridOfThirdsEnabledResponse(Status::kOk);
                }
                else
                {
                    LOG_DEBUG("kSetGridOfThirdsEnabledRequest ignored due to no active Ansel session");
                }
                break;
            }
            case AnselIPCRequest::RequestCase::kStyleTransferSideloadProgressRequest:
            {
                sendStyleTransferSideloadProgressResponse(m_state.restyle.progressState, m_state.restyle.downloadProgress);
                break;
            }
            case AnselIPCRequest::RequestCase::kGetGameSpecificControlsRequest:
                LOG_DEBUG("kGetGameSpecificControlsRequest received");
                // only allow this when Ansel session is active
                if (m_state.sdk.isAnselSessionActive)
                {
                    m_state.sdk.isGameSpecificControlUpdateRequired = true;
                }
                else
                {
                    LOG_DEBUG("kGetGameSpecificConrolsRequest ignored due to no active Ansel session");
                }
                break;
            case AnselIPCRequest::RequestCase::kUiReadyRequest:
                LOG_DEBUG("kUiReadyRequest received");
                sendUiReadyResponse();
                break;
            case AnselIPCRequest::RequestCase::kGetSettingsRequest:
                LOG_DEBUG("kGetSettingsRequest received");
                sendGetSettingsResponse();
                break;
            case AnselIPCRequest::RequestCase::kUiControlChangedRequest :
                LOG_DEBUG("kUiControlChangedRequest received");
                processUiControlChangedRequest(message);
                break;
            case AnselIPCRequest::RequestCase::kIpcVersionRequest:
                LOG_DEBUG("kIpcVersionRequest received");
                sendIpcVersionResponse();
                break;
            case AnselIPCRequest::RequestCase::kGetAnselShotPermissionsRequest:
                LOG_DEBUG("kGetAnselShotPermissionsRequest received");
                // only allow this when Ansel or Freestyle session is active
                if (m_state.sdk.isLwCameraSessionActive)
                {
                    sendAnselShotPermissionsResponse();
                }
                else
                {
                    LOG_DEBUG("kGetAnselShotPermissionsRequest ignored due to no active LwCamera session");
                }
                break;
            case AnselIPCRequest::RequestCase::kGetFilterListRequest:
                LOG_DEBUG("kGetFilterListRequest received");
                // only allow getting a filter list if any session (Ansel or Freestyle)
                // is active
                if (m_state.sdk.isLwCameraSessionActive)
                {
                    m_state.filters.effectListRequested = true;
                }
                else
                {
                    LOG_DEBUG("kGetFilterListRequest ignored due to no active LwCamera session");
                }
                break;
            case AnselIPCRequest::RequestCase::kIsAnselModdingAvailableRequest:
                LOG_DEBUG("kIsAnselModdingAvailableRequest received");
                sendIsAnselModdingAvailableResponse();
                break;
            case AnselIPCRequest::RequestCase::kGetFeatureSetRequest:
                LOG_DEBUG("kGetFeatureSetRequest received");
                LogIPCMessage(message);
                m_state.sdk.isFeatureSetRequested = true;
                if (message.request().getfeaturesetrequest().has_requestorsipcversion())
                {
                    m_state.ipcClientVersionMajor = message.request().getfeaturesetrequest().requestorsipcversion().major();
                    m_state.ipcClientVersionMinor = message.request().getfeaturesetrequest().requestorsipcversion().minor();
                    m_state.ipcClientVersionPatch = message.request().getfeaturesetrequest().requestorsipcversion().patch();
                }
                break;

            case AnselIPCRequest::RequestCase::kSetCMSInfoRequest:
                LOG_DEBUG("kSetCMSInfoRequest received");
                LogIPCMessage(message);
                if (message.request().setcmsinforequest().has_cmsid())
                {
                    m_state.appCMSID = darkroom::getWstrFromUtf8(message.request().setcmsinforequest().cmsid());
                }
                if (message.request().setcmsinforequest().has_shortname())
                {
                    m_state.appShortName = darkroom::getWstrFromUtf8(message.request().setcmsinforequest().shortname());
                }
                sendSetCMSInfoResponse(AnselIpc::Status::kOk);
                break;
            case AnselIPCRequest::RequestCase::kMoveFilterRequest:
            {
                LOG_DEBUG("kMoveFilterRequest received");
                // only allow this when a session (Ansel or Freestyle) is active
                if (m_state.sdk.isLwCameraSessionActive)
                {
                    const auto& moveRequest = message.request().movefilterrequest();
                    const auto& newOrder = moveRequest.desiredstackindices();
                    // check indices are unique
                    auto indices = decltype(m_state.filters.lwrrentFilterOldIndices)(newOrder.begin(), newOrder.end());
                    const auto indicesLength = indices.size();
                    const auto endIt = std::unique(indices.begin(), indices.end());
                    indices.resize(std::distance(indices.begin(), endIt));
                    const auto uniqueLength = indices.size();
                    if (indicesLength == uniqueLength)
                    {
                        const auto lwrrentLen = m_state.filters.lwrrentFilterOldIndices.size();
                        // and each is less than the total stack length
                        if (std::find_if(indices.cbegin(), indices.cend(), [&](uint32_t x) { return x >= lwrrentLen; }) == indices.cend())
                        {
                            if (m_state.filters.lwrrentFilterOldIndices.size() == indices.size())
                            {
                                // move indices into the state
                                m_state.filters.lwrrentFilterOldIndices = indices;
                                m_state.filters.lwrrentFilterIds = apply_permutation(m_state.filters.lwrrentFilterIds, indices);
                                m_state.filters.lwrrentFilterInfoQuery = apply_permutation(m_state.filters.lwrrentFilterInfoQuery, indices);
                                m_state.filters.lwrrentFilterResetValues = apply_permutation(m_state.filters.lwrrentFilterResetValues, indices);
                                m_state.filters.filterSetResponseRequired = apply_permutation(m_state.filters.filterSetResponseRequired, indices);
                                sendMoveFilterResponse(AnselIpc::Status::kOk);
                            }
                            else
                                sendMoveFilterResponse(kIlwalidRequest);
                        }
                        else
                            sendMoveFilterResponse(kIlwalidRequest);
                    }
                    else
                        sendMoveFilterResponse(kIlwalidRequest);
                }
                else
                {
                    LOG_DEBUG("kMoveFilterRequest ignored due to no active LwCamera session");
                }
                break;
            }
            case AnselIPCRequest::RequestCase::kGetStackInfoRequest:
            {
                LOG_DEBUG("kGetStackInfoRequest received");
                // only allow this when a session (Ansel or Freestyle) is active
                if (m_state.sdk.isLwCameraSessionActive)
                {
                    m_state.filters.isStackFilterListRequested = true;
                }
                else
                {
                    LOG_DEBUG("kGetStackInfoRequest ignored due to no active LwCamera session");
                }
                break;
            }
            case AnselIPCRequest::RequestCase::kRemoveFilterRequest:
            {
                LOG_DEBUG("kRemoveFilterRequest received");
                // only allow this when a session (Ansel or Freestyle) is active
                if (m_state.sdk.isLwCameraSessionActive)
                {
                    const auto& request = message.request().removefilterrequest();
                    size_t stackIdx = request.stackidx();
                    const bool stackIdxValid = stackIdx < m_state.filters.lwrrentFilterIds.size();
                    const auto stackLen = m_state.filters.lwrrentFilterIds.size();
                    // stack is empty if there are no slots or there is a single slot
                    bool stackIsEmpty = stackLen == 0 || std::all_of(m_state.filters.lwrrentFilterIds.cbegin(),
                        m_state.filters.lwrrentFilterIds.cend(), [](const auto& filterId)
                    {
                        return filterId == shadermod::Tools::wstrNone;
                    });

                    if (stackIdxValid && !stackIsEmpty)
                    {
#if ANSEL_SIDE_PRESETS
                        if (m_state.filters.activePreset && m_state.filters.activePreset->stackIdx == stackIdx)
                        {
                            handleDeletePreset();
                        }
                        else
                        {
                            stackIdx = m_state.filters.stackIdxTranslation[stackIdx];
#endif
                            deleteFilter(stackIdx);
                            sendRemoveFilterResponse(kOk);
#if ANSEL_SIDE_PRESETS
                        }
#endif
                    }
                    else
                        sendRemoveFilterResponse(kIlwalidRequest);
                }
                else
                {
                    LOG_DEBUG("kRemoveFilterRequest ignored due to no active LwCamera session");
                }
                break;
            }
            case AnselIPCRequest::RequestCase::kInsertFilterRequest:
            {
                LOG_DEBUG("kInsertFilterRequest received");
                // only allow this when a session (Ansel or Freestyle) is active
                if (m_state.sdk.isLwCameraSessionActive)
                {
                    const auto& request = message.request().insertfilterrequest();
                    size_t stackIdx = request.stackidx();
                    std::wstring filterIdToSet = darkroom::getWstrFromUtf8(request.filterid());
                    if (size_t(stackIdx) < m_state.filters.lwrrentFilterIds.size())
                    {
#if ANSEL_SIDE_PRESETS
                        if (m_state.filters.activePreset && stackIdx == m_state.filters.activePreset->stackIdx)
                        {
                            handleDeletePreset();
                        }

                        std::wstring ini_suffix = L".ini";
                        if (std::equal(ini_suffix.rbegin(), ini_suffix.rend(), filterIdToSet.rbegin()))
                        {
                            handleSetPreset(stackIdx, filterIdToSet, true);

                            break;
                        }

                        stackIdx = colwertSetFilterStackIdx(stackIdx);
#endif
                        m_state.filters.lwrrentFilterIds.insert(m_state.filters.lwrrentFilterIds.begin() + stackIdx, filterIdToSet);
                        m_state.filters.lwrrentFilterInfoQuery.insert(m_state.filters.lwrrentFilterInfoQuery.begin() + stackIdx, false);
                        m_state.filters.lwrrentFilterResetValues.insert(m_state.filters.lwrrentFilterResetValues.begin() + stackIdx, false);
                        m_state.filters.lwrrentFilterOldIndices.insert(m_state.filters.lwrrentFilterOldIndices.begin() + stackIdx, (uint32_t) stackIdx);
                        m_state.filters.filterSetResponseRequired.insert(m_state.filters.filterSetResponseRequired.begin() + stackIdx, IPCState::FiltersSDKState::FilterResponseType::kInsert);
                    }
                    else
                        sendInsertFilterResponse(kIlwalidRequest, request.stackidx());
                }
                else
                {
                    LOG_DEBUG("kInsertFilterRequest ignored due to no active LwCamera session");
                }
                break;
            }
            case AnselIPCRequest::RequestCase::kSetAnselEnabledRequest:
            {
                LOG_DEBUG("kSetAnselEnabledRequest received");
                IpcVersionResponse version;
                const auto& enableRequest = message.request().setanselenabledrequest();

                if (version.major() != enableRequest.major())
                {
                    sendAnselEnableResponse(kIncompatibleVersion);
                    break;
                }
                else
                {
                    const bool enable = enableRequest.enabled();

                    // disabling already disabled Ansel or Modding
                    if (!m_state.sdk.isLwCameraSessionActive && !enable)
                    {
                        sendAnselEnableResponse(kAlreadyDisabled);
                        break;
                    }
                    // enabling already enabled Ansel
                    else if (m_state.sdk.isAnselSessionActive && enable)
                    {
                        sendAnselEnableResponse(kAlreadyEnabled);
                        break;
                    }
                    // enabling already enabled Modding
                    else if (m_state.sdk.isLwCameraSessionActive && enable && enableRequest.has_pauseapplication() && !enableRequest.pauseapplication())
                    {
                        sendAnselEnableResponse(kAlreadyEnabled);
                        break;
                    }
                    // upgrading Modding session to Ansel is supported

                    if (enable)
                    {
                        m_state.sdk.isPrestartRequested = true;
                        m_state.sdk.isStartRequested = true;
                        if (enableRequest.has_pauseapplication())
                            m_state.sdk.startAnselSDKSession = enableRequest.pauseapplication();
                        else
                            m_state.sdk.startAnselSDKSession = true;
                    }
                    else
                    {
                        m_state.sdk.isPoststopRequested = true;
                        m_state.sdk.isStopRequested = true;
                        if (enableRequest.has_leavefiltersenabled())
                            m_state.filters.isModdingAllowed = enableRequest.leavefiltersenabled();
                    }
                }
            }
                break;
            case AnselIPCRequest::RequestCase::kSetLangIdRequest:
                LOG_DEBUG("kSetLangIdRequest received");
                m_state.lang = (WORD)message.request().setlangidrequest().lang();
                m_state.subLang = (WORD)message.request().setlangidrequest().sublang();
                sendSetLangIdResponse(kOk);
                break;
            case AnselIPCRequest::RequestCase::kEstimateCaptureRequest:
                LOG_DEBUG("kEstimateCaptureRequest received");
                processEstimateCaptureRequest(message);
                break;
            case AnselIPCRequest::RequestCase::kCaptureShotRequest:
                LOG_INFO("kCaptureShotRequest received");
                // allow for both Ansel and Freestyle (Ansel-lite) sessions
                if (m_state.sdk.isLwCameraSessionActive)
                {
                    processCaptureShotRequest(message);
                }
                else
                {
                    LOG_DEBUG("kCaptureShotRequest ignored due to no active LwCamera session");
                }
                break;
            case AnselIPCRequest::RequestCase::kGetHighresResolutionListRequest:
                LOG_DEBUG("kGetHighresResolutionListRequest received");
                // only allow this when Ansel session is active
                if (m_state.sdk.isAnselSessionActive)
                {
                    m_state.sdk.isHighresResolutionListRequested = true;
                }
                else
                {
                    LOG_DEBUG("kGetHighresResolutionListRequest ignored due to no active Ansel session");
                }
                break;
            case AnselIPCRequest::RequestCase::kGet360ResolutionRangeRequest:
                LOG_DEBUG("kGet360ResolutionRangeRequest received");
                // only allow this when Ansel session is active
                if (m_state.sdk.isAnselSessionActive)
                {
                    send360ResolutionRangeResponse();
                }
                else
                {
                    LOG_DEBUG("kGet360ResolutionRangeRequest ignored due to no active Ansel session");
                }
                break;
            case AnselIPCRequest::RequestCase::kGetFOVRangeRequest:
                LOG_DEBUG("kGetFOVRangeRequest received");
                // only allow this when Ansel session is active
                if (m_state.sdk.isAnselSessionActive)
                {
                    sendFovRangeResponse();
                }
                else
                {
                    LOG_DEBUG("kGetFOVRangeRequest ignored due to no active Ansel session");
                }
                break;
            case AnselIPCRequest::RequestCase::kGetRollRangeRequest:
                LOG_DEBUG("kGetRollRangeRequest received");
                // only allow this when Ansel session is active
                if (m_state.sdk.isAnselSessionActive)
                {
                    sendRollRangeResponse();
                }
                else
                {
                    LOG_DEBUG("kGetRollRangeRequest ignored due to no active Ansel session");
                }
                break;
            case AnselIPCRequest::RequestCase::kSetFOVRequest:
                LOG_DEBUG("kSetFOVRequest received");
                // only allow this when Ansel session is active
                if (m_state.sdk.isAnselSessionActive)
                {
                    processSetFovRequest(message);
                }
                else
                {
                    LOG_DEBUG("kSetFOVRequest ignored due to no active Ansel session");
                }
                break;
            case AnselIPCRequest::RequestCase::kSetRollRequest:
                LOG_DEBUG("kSetRollRequest received");
                // only allow this when Ansel session is active
                if (m_state.sdk.isAnselSessionActive)
                {
                    processSetRollRequest(message);
                }
                else
                {
                    LOG_DEBUG("kSetRollRequest ignored due to no active Ansel session");
                }
                break;
            case AnselIPCRequest::RequestCase::kAbortCaptureRequest:
                LOG_DEBUG("kAbortCaptureRequest received");
                // only allow this when Ansel session is active
                if (m_state.sdk.isAnselSessionActive)
                {
                    m_state.sdk.captureAbortRequested = true;
                }
                else
                {
                    LOG_DEBUG("kAbortCaptureRequest ignored due to no active Ansel session");
                }
                break;
            case AnselIPCRequest::RequestCase::kGetAnselEnabledRequest:
                LOG_DEBUG("kGetAnselEnabledRequest received");
                sendAnselCaptureModeResponse(m_state.sdk.isAnselSessionActive);
                break;
            case AnselIPCRequest::RequestCase::kGetLwrrentFOVRequest:
                LOG_DEBUG("kGetLwrrentFOVRequest received");
                // only allow this when Ansel session is active
                if (m_state.sdk.isAnselSessionActive)
                {
                    sendLwrrentFovResponse();
                }
                else
                {
                    LOG_DEBUG("kGetLwrrentFOVRequest ignored due to no active Ansel session");
                }
                break;
            case AnselIPCRequest::RequestCase::kIsAnselAvailableRequest:
                LOG_DEBUG("kIsAnselAvailableRequest received");
                sendIsAnselAvailableResponse();
                break;
            case AnselIPCRequest::RequestCase::kGetFilterInfoRequest:
            {
                LOG_DEBUG("kGetFilterInfoRequest received");
                // only allow this when a session (Ansel or Freestyle) is active
                if (m_state.sdk.isLwCameraSessionActive)
                {
                    int stackIdx = message.request().getfilterinforequest().stackidx();
#if ANSEL_SIDE_PRESETS
                    stackIdx = m_state.filters.stackIdxTranslation[stackIdx];
#endif

                    if (stackIdx < int(m_state.filters.lwrrentFilterInfoQuery.size()))
                    {
                        m_state.filters.lwrrentFilterInfoQuery[stackIdx] = true;
                    }
                    else
                    {
                        sendGetFilterInfoResponse(kOutOfRange, stackIdx);
                    }
                }
                else
                {
                    LOG_DEBUG("kGetFilterInfoRequest ignored due to no LwCamera session");
                }
            }
                break;
            case AnselIPCRequest::RequestCase::kResetFilterValuesRequest:
            {
                LOG_DEBUG("kResetFilterValuesRequest received");
                // only allow this when a session (Ansel or Freestyle) is active
                if (m_state.sdk.isLwCameraSessionActive)
                {
                    int stackIdx = message.request().resetfiltervaluesrequest().stackidx();
#if ANSEL_SIDE_PRESETS
                    stackIdx = m_state.filters.stackIdxTranslation[stackIdx];
#endif
                    if (stackIdx < int(m_state.filters.lwrrentFilterResetValues.size()))
                    {
                        m_state.filters.lwrrentFilterResetValues[stackIdx] = true;
                    }
                    else
                    {
                        sendResetFilterValuesResponse(kOutOfRange, stackIdx);
                    }
                }
                else
                {
                    LOG_DEBUG("kResetFilterValuesRequest ignored due to no active LwCamera session");
                }
            }
                break;
            case AnselIPCRequest::RequestCase::kResetAllFilterValuesRequest:
            {
                LOG_DEBUG("kResetAllFilterValuesRequest received");
                // only allow this when a session (Ansel or Freestyle) is active
                if (m_state.sdk.isLwCameraSessionActive)
                {
                    for (size_t stackIdx = 0, stackIdxEnd = m_state.filters.lwrrentFilterResetValues.size(); stackIdx < stackIdxEnd; ++stackIdx)
                    {
#if ANSEL_SIDE_PRESETS
                        stackIdx = m_state.filters.stackIdxTranslation[stackIdx];
#endif
                        m_state.filters.lwrrentFilterResetValues[stackIdx] = true;
                    }
                }
                else
                {
                    LOG_DEBUG("kResetAllFilterValuesRequest ignored due to no active LwCamera session");
                }
            }
            break;
            case AnselIPCRequest::RequestCase::kResetEntireStackRequest:
            {
                LOG_DEBUG("kResetEntireStackRequest received");
                // only allow this when a session (Ansel or Freestyle) is active
                if (m_state.sdk.isLwCameraSessionActive)
                {
                    m_state.filters.lwrrentFilterIds = decltype(m_state.filters.lwrrentFilterIds)();
                    m_state.filters.lwrrentFilterInfoQuery = decltype(m_state.filters.lwrrentFilterInfoQuery)();
                    m_state.filters.lwrrentFilterResetValues = decltype(m_state.filters.lwrrentFilterResetValues)();
                    m_state.filters.lwrrentFilterOldIndices = decltype(m_state.filters.lwrrentFilterOldIndices)();
                    m_state.filters.filterSetResponseRequired = decltype(m_state.filters.filterSetResponseRequired)();
#if ANSEL_SIDE_PRESETS
                    m_state.filters.stackIdxTranslation = decltype(m_state.filters.stackIdxTranslation)();
                    delete m_state.filters.activePreset;
                    m_state.filters.activePreset = nullptr;
#endif

                    sendResetEntireStackResponse(kOk);
                }
                else
                {
                    LOG_DEBUG("kResetEntireStackRequest ignored due to no active LwCamera session");
                }
            }
            break;
            case AnselIPCRequest::RequestCase::kSetFilterRequest:
            {
                LOG_DEBUG("kSetFilterRequest received");
                // only allow this when a session (Ansel or Freestyle) is active
                if (m_state.sdk.isLwCameraSessionActive)
                {
                    size_t stackIdx = size_t(message.request().setfilterrequest().stackidx());
                    std::wstring filterIdToSet = darkroom::getWstrFromUtf8(message.request().setfilterrequest().filterid());

#if ANSEL_SIDE_PRESETS
                    if (m_state.filters.activePreset && stackIdx == m_state.filters.activePreset->stackIdx)
                    {
                        handleSetOverPreset(stackIdx, filterIdToSet);
                    }

                    size_t lastDotPos = filterIdToSet.find_last_of('.');
                    std::wstring lwrExtension = (lastDotPos != std::wstring::npos) ? filterIdToSet.substr(lastDotPos + 1) : L"";

                    if (lwrExtension == L"ini")
                    {
                        handleSetPreset(stackIdx, filterIdToSet, false);

                        // Need to still send the SetFilterResponse for the preset.
                        std::wstring directory;
                        std::wstring fileName;
                        bool success = shadermod::ir::filehelpers::SplitPathIntoDirectoryAndFileName(filterIdToSet, directory, fileName);
                        lastDotPos = fileName.find_last_of('.');
                        std::wstring fileNameNoExtension = (lastDotPos != std::wstring::npos) ? fileName.substr(0, lastDotPos) : fileName;
                        AnselUIBase::EffectPropertiesDescription epd;
                        epd.filterDisplayName = fileNameNoExtension;
                        epd.filterDisplayNameEnglish = fileNameNoExtension;
                        epd.filterId = filterIdToSet;
                        sendSetFilterResponse(kOk, int(stackIdx), epd);

                        break;
                    }

                    stackIdx = colwertSetFilterStackIdx(stackIdx);
#endif

                    if (prepareSetFilter(stackIdx, filterIdToSet) == kOk)
                    {
                        m_state.filters.lwrrentFilterIds[stackIdx] = filterIdToSet;
                        m_state.filters.filterSetResponseRequired[stackIdx] = IPCState::FiltersSDKState::FilterResponseType::kSet;
                    }
                    else
                    {
                        sendSetFilterResponse(kAlreadySet, int(stackIdx));
                    }
                }
                else
                {
                    LOG_DEBUG("kSetFilterRequest ignored due to no active LwCamera session");
                }
            }
                break;

            case AnselIPCRequest::RequestCase::kGetProcessInfoRequest:
                sendGetProcessInfoResponse();
                break;
            case AnselIPCRequest::RequestCase::kGetScreenResolutionRequest:
                sendGetScreenResolutionResponse();
                break;
            case AnselIPCRequest::RequestCase::kSetStyleTransferModelRequest:
            {
                const auto netId = darkroom::getWstrFromUtf8(message.request().setstyletransfermodelrequest().modelid());
                // we will accept any netId if we don't know all network ids yet.
                // Otherwise, we will check if the requested netId is known
                if (m_state.restyle.netIds.find(netId) != m_state.restyle.netIds.end())
                {
                    m_state.restyle.lwrrentNetworkId = netId;
                    sendSetStyleTransferModelResponse(kOk);
                }
                else
                {
                    sendSetStyleTransferModelResponse(kStyleNoModelFound);
                }
            }
                break;
            case AnselIPCRequest::RequestCase::kSetStyleTransferStyleRequest:
                m_state.restyle.lwrrentStyleFullPath = darkroom::getWstrFromUtf8(message.request().setstyletransferstylerequest().fullyqualifiedpath());
                sendSetStyleTransferStyleResponse(kOk);
                break;
            case AnselIPCRequest::RequestCase::kSetStyleTransferEnabledRequest:
                if (!m_state.sdk.isAnselSessionActive)
                {
                    sendSetStyleTransferEnabledResponse(kStyleAnselSessionRequired);
                    break;
                }
                if (m_state.restyle.lwrrentNetworkId.empty())
                {
                    sendSetStyleTransferEnabledResponse(kStyleUnspecifiedNetwork);
                    break;
                }
                else if (m_state.restyle.lwrrentStyleFullPath.empty())
                {
                    sendSetStyleTransferEnabledResponse(kStyleUnspecifiedStyle);
                    break;
                }

                m_state.restyle.isEnabled = message.request().setstyletransferenabledrequest().enabled();
                sendSetStyleTransferEnabledResponse(kOk);
                break;
            case AnselIPCRequest::RequestCase::kGetStyleTransferModelListRequest:
                m_state.restyle.networkListRequested = true;
                break;
            case AnselIPCRequest::RequestCase::kSetFilterAttributeRequest:
            {
                LOG_DEBUG("kSetFilterAttributeRequest received");
                AnselUIBase::EffectChange effectChange;
                effectChange.filterId = darkroom::getWstrFromUtf8(message.request().setfilterattributerequest().filterid());
                effectChange.controlId = message.request().setfilterattributerequest().controlid();
                effectChange.stackIdx = message.request().setfilterattributerequest().stackidx();
#if ANSEL_SIDE_PRESETS
                effectChange.stackIdx = m_state.filters.stackIdxTranslation[effectChange.stackIdx];
#endif

                bool success = true;
                if (message.request().setfilterattributerequest().floatvalue_size())
                {
                    const int dims = message.request().setfilterattributerequest().floatvalue().size();
                    effectChange.value.setDimensionality(dims);
                    effectChange.value = decltype(effectChange.value)(message.request().setfilterattributerequest().floatvalue().data(), dims);
                }
                else if (message.request().setfilterattributerequest().boolvalue_size())
                {
                    const int dims = message.request().setfilterattributerequest().boolvalue().size();
                    effectChange.value.setDimensionality(dims);
                    effectChange.value = decltype(effectChange.value)(message.request().setfilterattributerequest().boolvalue().data(), dims);
                }
                else if (message.request().setfilterattributerequest().uintvalue_size())
                {
                    const int dims = message.request().setfilterattributerequest().uintvalue().size();
                    effectChange.value.setDimensionality(dims);
                    effectChange.value = decltype(effectChange.value)(message.request().setfilterattributerequest().uintvalue().data(), dims);
                }
                else if (message.request().setfilterattributerequest().intvalue_size())
                {
                    const int dims = message.request().setfilterattributerequest().intvalue().size();
                    effectChange.value.setDimensionality(dims);
                    effectChange.value = decltype(effectChange.value)(message.request().setfilterattributerequest().intvalue().data(), dims);
                }
                else
                {
                    sendSetFilterAttributeRequest(kIlwalidRequest);
                    success = false;
                }

                if (success)
                {
                    m_state.filters.effectChanges.push_back(effectChange);
                    sendSetFilterAttributeRequest(kOk);
                }
            }
                break;

            case AnselIPCRequest::RequestCase::kIsAnselSDKIntegrationAvailableRequest:
            {
                LOG_DEBUG("kIsAnselSDKIntegrationAvailableRequest received");
                if (m_state.sdk.sdkDetected)
                {
                    sendIsAnselSDKIntegrationAvailableResponse(kOk);
                }
                else
                {
                    sendIsAnselSDKIntegrationAvailableResponse(kFailed);
                }
                break;
            }

            case AnselIPCRequest::RequestCase::kInputEventRequest:
            {
                LOG_DEBUG("kInputEventRequest received");
                handleInputEventRequest(message.request().inputeventrequest());
                break;
            }

            case AnselIPCRequest::RequestCase::kMultipleInputEventRequest:
            {
                LOG_DEBUG("kMultipleInputEventRequest received");
                const auto& multiInputEvents = message.request().multipleinputeventrequest();

                for(int eventIdx = 0; eventIdx < multiInputEvents.inputevents_size(); eventIdx++)
                {
                    handleInputEventRequest(multiInputEvents.inputevents(eventIdx));
                }

                break;
            }


            case AnselIPCRequest::RequestCase::kSetHighQualityRequest:
            {
                bool setting = message.request().sethighqualityrequest().setting();
                LOG_DEBUG("SetHighQualityRequest received with setting %s.", setting ? "TRUE" : "FALSE");
                // The UIIPC is not in charge of setting High Quality. This is handled by injected game
                // engine settings. But IPC still has the option to override and set this.
                m_state.sdk.isHighQualityEnabled = setting;
                sendSetHighQualityResponse(AnselIpc::Status::kOk);
                break;
            }

            case AnselIPCRequest::RequestCase::kSetFilterAndAttributesRequest:
            {
                LOG_DEBUG("kSetFilterAndAttributesRequest received");
                LOG_DEBUG("  filterid: %s", message.request().setfilterandattributesrequest().filterid().c_str());
                LOG_DEBUG("  stackidx: %i", message.request().setfilterandattributesrequest().stackidx());
                LOG_DEBUG("  float (values, controlids) count: (%i, %i)", message.request().setfilterandattributesrequest().floatvalues_size(), message.request().setfilterandattributesrequest().floatcontrolids_size());
                LOG_DEBUG("  bool  (values, controlids) count: (%i, %i)", message.request().setfilterandattributesrequest().boolvalues_size(), message.request().setfilterandattributesrequest().boolcontrolids_size());
                LOG_DEBUG("  int   (values, controlids) count: (%i, %i)", message.request().setfilterandattributesrequest().intvalues_size(), message.request().setfilterandattributesrequest().intcontrolids_size());
                LOG_DEBUG("  uint  (values, controlids) count: (%i, %i)", message.request().setfilterandattributesrequest().uintvalues_size(), message.request().setfilterandattributesrequest().uintcontrolids_size());
                // only allow this when a session (Ansel or Freestyle) is active
                if (m_state.sdk.isLwCameraSessionActive)
                {
                    // Set Filter
                    size_t stackIdx = size_t(message.request().setfilterandattributesrequest().stackidx());
                    std::wstring filterIdToSet = darkroom::getWstrFromUtf8(message.request().setfilterandattributesrequest().filterid());

#if ANSEL_SIDE_PRESETS
                    if (m_state.filters.activePreset && stackIdx == m_state.filters.activePreset->stackIdx)
                    {
                        handleSetOverPreset(stackIdx, filterIdToSet);
                    }
                    std::wstring ini_suffix = L".ini";
                    if (std::equal(ini_suffix.rbegin(), ini_suffix.rend(), filterIdToSet.rbegin()))
                    {
                        handleSetPreset(stackIdx, filterIdToSet, false);
                        break;
                    }
                    stackIdx = colwertSetFilterStackIdx(stackIdx);
#endif
                    prepareSetFilter(stackIdx, filterIdToSet);

                    m_state.filters.lwrrentFilterIds[stackIdx] = filterIdToSet;
                    m_state.filters.filterSetResponseRequired[stackIdx] = IPCState::FiltersSDKState::FilterResponseType::kSetFilterAndAttributes;

                    // Set Attributes
                    // floats
                    ProcessVarType(float, float);
                    // bools
                    ProcessVarType(bool, bool);
                    // ints
                    ProcessVarType(int, int);
                    // uints
                    ProcessVarType(uint, UINT);
                }
                else
                {
                    LOG_DEBUG("kSetFilterAndAttributesRequest ignored due to no active LwCamera session");
                }
                break;
            }
            }
            break;
        }
        case AnselIPCMessage::MessageCase::kResponse:
        {
            switch (message.response().response_case())
            {
            case AnselIPCResponse::ResponseCase::kGetEnabledFeatureSetResponse:
                if (message.response().getenabledfeaturesetresponse().has_modsavailable())
                {
                    m_state.filters.isModdingEnabled = message.response().getenabledfeaturesetresponse().modsavailable();
                    LOG_DEBUG("Received modsAvailable = %d", m_state.filters.isModdingEnabled);
                }
                break;
            case AnselIPCResponse::ResponseCase::kStyleTransferSideloadChoiceResponse:
            {
                const auto resp = message.response().styletransfersideloadchoiceresponse();
                const auto choice = resp.choice();
                if (choice == AnselIpc::SideloadChoice::kYes)
                    m_state.restyle.downloadConfirmationStatus = AnselUIBase::RestyleDownloadStatus::kConfirmed;
                else if (choice == AnselIpc::SideloadChoice::kNo)
                    m_state.restyle.downloadConfirmationStatus = AnselUIBase::RestyleDownloadStatus::kRejected;
                else
                    m_state.restyle.downloadConfirmationStatus = AnselUIBase::RestyleDownloadStatus::kNone;
                break;
            }
            case AnselIPCResponse::ResponseCase::kAddUIElementResponse:
                const auto resp = message.response().adduielementresponse();
                if (resp.has_status())
                    LOG_DEBUG("AddUIElementResponse status = %d", resp.status());
                if (resp.has_requestid())
                    LOG_DEBUG("AddUIElementResponse requestId = %d", resp.requestid());
                if (resp.has_uidescboolean())
                    LOG_DEBUG("AddUIElementResponse Boolean controlId = %d", resp.uidescboolean().id());
                if (resp.has_uidescslider())
                    LOG_DEBUG("AddUIElementResponse Slider controlId = %d", resp.uidescslider().id());
                if (resp.has_uidesclabel())
                    LOG_DEBUG("AddUIElementResponse Label controlId = %d", resp.uidesclabel().id());

                break;
            }
        }
            break;
        default:
            // shouldn't happen
            LOG_WARN("Invalid AnselIPCMessage::MessageCase: %d", message.message_case());
            break;
        }
    }
}

uint32_t AnselIPCMessageBusObserver::getHighresMultiplier() const { return m_highresMultiplier; }
bool AnselIPCMessageBusObserver::getAnselEnable() const { return m_anselIpcEnable; }
bool AnselIPCMessageBusObserver::getAnselDisable() const { return m_anselIpcDisable; }
uint32_t AnselIPCMessageBusObserver::get360Width() const { return m_360Width; }

void AnselIPCMessageBusObserver::sendUpdateRollRequest(float roll)
{
    AnselIPCMessage message;
    AnselIPCRequest* request = new AnselIPCRequest;
    UpdateRollRequest* req = new UpdateRollRequest;
    req->set_roll(roll);
    request->set_allocated_updaterollrequest(req);
    message.set_allocated_request(request);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("UpdateRollRequest sent");
}

void AnselIPCMessageBusObserver::sendUpdateFovRequest(float fov)
{
    AnselIPCMessage message;
    AnselIPCRequest* request = new AnselIPCRequest;
    UpdateFovRequest* req = new UpdateFovRequest;
    req->set_fov(fov);
    request->set_allocated_updatefovrequest(req);
    message.set_allocated_request(request);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("UpdateFovRequest sent: %f", fov);
}

void AnselIPCMessageBusObserver::sendAnselStatusReportRequest(AnselIpc::Status status)
{
    AnselIPCMessage message;
    AnselIPCRequest* request = new AnselIPCRequest;
    AnselStatusReportRequest* req = new AnselStatusReportRequest;
    req->set_status(status);
    request->set_allocated_anselstatusreportrequest(req);
    message.set_allocated_request(request);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("AnselStatusReportRequest sent");
}

void AnselIPCMessageBusObserver::sendAnselStatusReportRequest(AnselIpc::Status status, const std::vector<std::string>& data)
{
    AnselIPCMessage message;
    AnselIPCRequest* request = new AnselIPCRequest;
    AnselStatusReportRequest* req = new AnselStatusReportRequest;
    req->set_status(status);
    for (const auto& item : data)
    {
        auto* data_pb = req->add_data();
        data_pb->set_stringvalue(item);
    }
    request->set_allocated_anselstatusreportrequest(req);
    message.set_allocated_request(request);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("AnselStatusReportRequest sent");
    LogIPCMessage(message);
}

void AnselIPCMessageBusObserver::sendGetSettingsResponse()
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    GetSettingsResponse* resp = new GetSettingsResponse;
    for (const auto& setting : m_state.settings)
    {
        auto* setting_pb = resp->add_settings();
        setting_pb->set_name(setting.first.c_str());
        setting_pb->set_value(darkroom::getUtf8FromWstr(setting.second.c_str()));
    }
    response->set_allocated_getsettingsresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("GetSettingsResponse sent");
    LogIPCMessage(message);
}

void AnselIPCMessageBusObserver::sendReportErrorResponse(AnselIpc::ErrorType type, uint32_t code, const std::string& filename, uint32_t line, const std::string& data)
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    ReportErrorResponse* resp = new ReportErrorResponse;
    resp->set_type(type);
    resp->set_code(code);
    resp->set_filename(filename);
    resp->set_line(line);
    resp->set_reason(data);
    response->set_allocated_reporterrorresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("FatalErrorResponse sent");
}

void AnselIPCMessageBusObserver::sendStyleTransferSideloadProgressResponse(AnselIpc::SideloadProgress status, int progress)
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    StyleTransferSideloadProgressResponse* resp = new StyleTransferSideloadProgressResponse;
    resp->set_status(status);
    resp->set_progress(progress);
    response->set_allocated_styletransfersideloadprogressresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("StyleTransferSideloadProgressResponse sent");
}

void AnselIPCMessageBusObserver::sendStyleTransferSideloadChoiceRequest()
{
    AnselIPCMessage message;
    AnselIPCRequest* request = new AnselIPCRequest;
    StyleTransferSideloadChoiceRequest* req = new StyleTransferSideloadChoiceRequest;
    // TODO: This is hard-coded for now, but should be supplied by the Ansel server later
    req->set_packagesizeinbytes(60 * 1024 * 1024);
    request->set_allocated_styletransfersideloadchoicerequest(req);
    message.set_allocated_request(request);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("StyleTransferSideloadChoiceRequest sent");
}

void AnselIPCMessageBusObserver::sendGetStackInfoResponse(const std::vector<std::string>& stack)
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    GetStackInfoResponse* resp = new GetStackInfoResponse;
    for (const auto& filterId : stack)
        resp->add_filterids(filterId);
    response->set_allocated_getstackinforesponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("GetStackInfoResponse sent");
}

void AnselIPCMessageBusObserver::sendGetFeatureSetResponse()
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    GetFeatureSetResponse* resp = new GetFeatureSetResponse;
    LOG_DEBUG("Modding available: %d", m_state.filters.isModdingAvailable);
    LOG_DEBUG("SDK detected: %d", m_state.sdk.sdkDetected);
    LOG_DEBUG("Restyle available: %d", m_state.restyle.isAvailable);
    resp->set_modsavailable(m_state.filters.isModdingAvailable);
    resp->set_sdkdetected(m_state.sdk.sdkDetected);
    resp->set_restyleavailable(m_state.restyle.isAvailable);
    resp->set_allowoffline(getAllowOffline());
    response->set_allocated_getfeaturesetresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LogIPCMessage(message);
}

void AnselIPCMessageBusObserver::sendGetEnabledFeatureSetRequest()
{
    AnselIPCMessage message;
    AnselIPCRequest* request = new AnselIPCRequest;
    GetEnabledFeatureSetRequest* req = new GetEnabledFeatureSetRequest;
    request->set_allocated_getenabledfeaturesetrequest(req);
    message.set_allocated_request(request);
    GenericBusMessage msg(AnselIPCSystem, AnselIPCModule, message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);
}

void AnselIPCMessageBusObserver::sendIsAnselModdingAvailableResponse()
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    IsAnselModdingAvailableResponse* resp = new IsAnselModdingAvailableResponse;
    if (m_state.filters.isModdingAvailable)
        resp->set_status(kOk);
    else
        resp->set_status(kDisabled);
    response->set_allocated_isanselmoddingavailableresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("IsAnselModdingAvailableResponse sent");
}

void AnselIPCMessageBusObserver::sendGetScreenResolutionResponse()
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    GetScreenResolutionResponse* resp = new GetScreenResolutionResponse;
    resp->set_xresolution(m_state.screenWidth);
    resp->set_yresolution(m_state.screenHeight);
    if (m_state.screenWidth > 0 && m_state.screenHeight > 0)
        resp->set_status(kOk);
    else
        resp->set_status(kFailed);
    response->set_allocated_getscreenresolutionresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("GetScreenResolutionResponse sent");
}

void AnselIPCMessageBusObserver::sendUiReadyResponse()
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    UIReadyResponse* resp = new UIReadyResponse;
    response->set_allocated_uireadyresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("UiReadyResponse sent");
}

void AnselIPCMessageBusObserver::sendGetProcessInfoResponse()
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    GetProcessInfoResponse* resp = new GetProcessInfoResponse;
    resp->set_status(kOk);
    resp->set_processid(GetLwrrentProcessId());
    wchar_t appPath[APP_PATH_MAXLEN];
    GetModuleFileName(NULL, appPath, APP_PATH_MAXLEN);
    const auto appPathUtf8 = darkroom::getUtf8FromWstr(appPath);
    resp->set_processpath(appPathUtf8);
    response->set_allocated_getprocessinforesponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);
    LOG_DEBUG("GetProcessInfoResponse sent");
}

void AnselIPCMessageBusObserver::sendIpcVersionResponse()
{
    ipc::sendIpcVersionResponse(m_messageBus);
    LOG_DEBUG("IpcVersionResponse sent");
}

void AnselIPCMessageBusObserver::sendGetFilterInfoResponse(Status status, int stackIdx, const AnselUIBase::EffectPropertiesDescription& props)
{
#if ANSEL_SIDE_PRESETS
    stackIdx = getUIStackIdxFromAnselStackIdx(stackIdx);
#endif
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    GetFilterInfoResponse* resp = new GetFilterInfoResponse;
    FilterProperties* properties = new FilterProperties;
    resp->set_status(status);
    resp->set_allocated_filterproperties(properties);
    resp->set_stackidx(stackIdx);
    properties->set_filterid(darkroom::getUtf8FromWstr(props.filterId));
    properties->set_filterdisplayname(darkroom::getUtf8FromWstr(props.filterDisplayName));
    properties->set_filterdisplaynameenglish(darkroom::getUtf8FromWstr(props.filterDisplayNameEnglish));
    for (const auto& p : props.attributes)
    {
        auto* prop = properties->add_controls();
        prop->set_controlid(p.controlId);
        prop->set_displayname(darkroom::getUtf8FromWstr(p.displayName));
        prop->set_displaynameenglish(darkroom::getUtf8FromWstr(p.displayNameEnglish));
        if (!p.uiMeasurementUnit.empty())
            prop->set_uimeasurementunit(darkroom::getUtf8FromWstr(p.uiMeasurementUnit));

        fillIpcPropFromAttributeValue(prop, p);

        if (p.controlType == AnselUIBase::ControlType::kSlider)
        {
            prop->set_type(kControlSlider);
            prop->set_uiprecision(0);
        }
        else if (p.controlType == AnselUIBase::ControlType::kCheckbox)
        {
            prop->set_type(kControlBoolean);
            prop->set_uiprecision(2);
        }
        else if (p.controlType == AnselUIBase::ControlType::kColorPicker)
        {
            prop->set_type(kControlColorPicker);
            prop->set_uiprecision(2);
        }
        else if (p.controlType == AnselUIBase::ControlType::kFlyout)
        {
            prop->set_type(kControlPulldown);
            prop->set_uiprecision(0);
        }
        else if (p.controlType == AnselUIBase::ControlType::kRadioButton)
        {
            // Radio Button is only processed in clients that use IPC version 7.6.0 or later.
            if (m_state.ClientMeetsMinimumVersion(7, 6, 0))
            {
                prop->set_type(kControlRadioButton);
            }
            else
            {
                prop->set_type(kControlPulldown);
            }
            prop->set_uiprecision(0);
        }
        else if (p.controlType == AnselUIBase::ControlType::kEditbox)
        {
            prop->set_type(kControlEdit);
            prop->set_uiprecision(0);
        }
        else
        {
            LOG_ERROR("IPC couldn't serialize unknown control type");
        }

        // set hint and list options names
        if (p.userConstant)
        {
            if (!p.userConstant->getUiHint().empty())
                prop->set_tooltip(p.userConstant->getUiHint());

            const auto optsCount = p.userConstant->getNumListOptions();
            for (uint32_t i = 0; i < optsCount; ++i)
                prop->add_labelsui(p.userConstant->getListOptionName(i));
        }
    }
    response->set_allocated_getfilterinforesponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("GetFilterInfoResponse sent");
}

void AnselIPCMessageBusObserver::sendResetFilterValuesResponse(Status status, int stackIdx)
{
#if ANSEL_SIDE_PRESETS
    stackIdx = getUIStackIdxFromAnselStackIdx(stackIdx);
#endif
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    ResetFilterValuesResponse* resp = new ResetFilterValuesResponse;
    resp->set_status(status);
    resp->set_stackidx(stackIdx);
    response->set_allocated_resetfiltervaluesresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("ResetFilterValuesResponse sent");
}

void AnselIPCMessageBusObserver::sendGetFilterInfoResponse(Status status, int stackIdx)
{
#if ANSEL_SIDE_PRESETS
    stackIdx = getUIStackIdxFromAnselStackIdx(stackIdx);
#endif
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    GetFilterInfoResponse* resp = new GetFilterInfoResponse;
    resp->set_status(status);
    resp->set_stackidx(stackIdx);
    response->set_allocated_getfilterinforesponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("GetFilterInfoResponse [simplified] sent");
}

void AnselIPCMessageBusObserver::sendGetStyleTransferModelListResponse(const std::vector<std::wstring> & netIds, const std::vector<std::wstring> & netNames)
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    GetStyleTransferModelListResponse* resp = new GetStyleTransferModelListResponse;

    if (netNames.size() >= netIds.size())
    {
        for (size_t i = 0u; i < netIds.size(); ++i)
        {
            auto* model = resp->add_models();
            model->set_id(darkroom::getUtf8FromWstr(netIds[i]));
            model->set_localizedname(darkroom::getUtf8FromWstr(netNames[i]));
        }
    }
    response->set_allocated_getstyletransfermodellistresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("GetStyleTransferModelListResponse sent");
}

template<typename T>
T* CreateSetFilterResponse(Status status, int stackIdx, const AnselUIBase::EffectPropertiesDescription& props, const IPCState& ipcState)
{
    T* resp = new T;
    FilterProperties* properties = new FilterProperties;
    resp->set_status(status);
    resp->set_allocated_filterproperties(properties);
    resp->set_stackidx(stackIdx);
    properties->set_filterid(darkroom::getUtf8FromWstr(props.filterId));
    properties->set_filterdisplayname(darkroom::getUtf8FromWstr(props.filterDisplayName));
    properties->set_filterdisplaynameenglish(darkroom::getUtf8FromWstr(props.filterDisplayNameEnglish));
    for (const auto& p : props.attributes)
    {
        auto* prop = properties->add_controls();
        prop->set_controlid(p.controlId);
        prop->set_displayname(darkroom::getUtf8FromWstr(p.displayName));
        prop->set_displaynameenglish(darkroom::getUtf8FromWstr(p.displayNameEnglish));
        if (!p.uiMeasurementUnit.empty())
            prop->set_uimeasurementunit(darkroom::getUtf8FromWstr(p.uiMeasurementUnit));

        fillIpcPropFromAttributeValue(prop, p);

        if (p.controlType == AnselUIBase::ControlType::kSlider)
        {
            prop->set_type(kControlSlider);
            prop->set_uiprecision(0);
        }
        else if (p.controlType == AnselUIBase::ControlType::kCheckbox)
        {
            prop->set_type(kControlBoolean);
            prop->set_uiprecision(2);
        }
        else if (p.controlType == AnselUIBase::ControlType::kColorPicker)
        {
            prop->set_type(kControlColorPicker);
            prop->set_uiprecision(2);
        }
        else if (p.controlType == AnselUIBase::ControlType::kFlyout)
        {
            prop->set_type(kControlPulldown);
            prop->set_uiprecision(0);
        }
        else if (p.controlType == AnselUIBase::ControlType::kRadioButton)
        {
            // Radio Button is only processed in clients that use IPC version 7.6.0 or later.
            if (ipcState.ClientMeetsMinimumVersion(7, 6, 0))
            {
                prop->set_type(kControlRadioButton);
            }
            else
            {
                LOG_WARN("Warning: RadioButton not supported by IPC client. Falling back to Pulldown.");
                prop->set_type(kControlPulldown);
            }
            prop->set_uiprecision(0);
        }
        else if (p.controlType == AnselUIBase::ControlType::kEditbox)
        {
            prop->set_type(kControlEdit);
            prop->set_uiprecision(0);
        }
        else
        {
            LOG_ERROR("IPC couldn't serialize unknown control type");
        }

        // set hint
        // set hint and list options names
        if (p.userConstant)
        {
            if (!p.userConstant->getUiHint().empty())
                prop->set_tooltip(p.userConstant->getUiHint());

            const auto optsCount = p.userConstant->getNumListOptions();
            for (uint32_t i = 0; i < optsCount; ++i)
                prop->add_labelsui(p.userConstant->getListOptionName(i));
        }
    }
    return resp;
}

template<typename T, typename Q>
void AnselIPCMessageBusObserver::sendFilterResponse(Q func, Status status, int stackIdx, const AnselUIBase::EffectPropertiesDescription& props)
{
#if ANSEL_SIDE_PRESETS
    stackIdx = getUIStackIdxFromAnselStackIdx(stackIdx);
#endif
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    T* resp = CreateSetFilterResponse<T>(status, stackIdx, props, m_state);
    (response->*func)(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_INFO("SetFilterResponse sent for ID: \"%s\"", darkroom::getUtf8FromWstr(props.filterId).c_str());

    LogIPCMessage(message);
}

template<typename T, typename Q>
void AnselIPCMessageBusObserver::sendFilterResponse(Q func, Status status, int stackIdx)
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    T* resp = new T;

#if ANSEL_SIDE_PRESETS
    stackIdx = getUIStackIdxFromAnselStackIdx(stackIdx);
#endif
    resp->set_status(status);
    resp->set_stackidx(stackIdx);
    (response->*func)(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("SetFilterResponse [simplified] sent");
}

void AnselIPCMessageBusObserver::sendSetFilterResponse(AnselIpc::Status status, int stackIdx)
{
    sendFilterResponse<SetFilterResponse>(&AnselIPCResponse::set_allocated_setfilterresponse, status, stackIdx);
}

void AnselIPCMessageBusObserver::sendSetFilterResponse(AnselIpc::Status status, int stackIdx, const AnselUIBase::EffectPropertiesDescription& desc)
{
    sendFilterResponse<SetFilterResponse>(&AnselIPCResponse::set_allocated_setfilterresponse, status, stackIdx, desc);
}

void AnselIPCMessageBusObserver::sendSetFilterAndAttributeResponse(AnselIpc::Status status, int stackIdx)
{
#if ANSEL_SIDE_PRESETS
    stackIdx = getUIStackIdxFromAnselStackIdx(stackIdx);
#endif
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    SetFilterAndAttributesResponse* resp = new SetFilterAndAttributesResponse;

    SetFilterResponse* setFilterResponse = new SetFilterResponse;
    setFilterResponse->set_status(status);
    setFilterResponse->set_stackidx(stackIdx);
    resp->set_allocated_setfilterresponse(setFilterResponse);

    response->set_allocated_setfilterandattributesresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("SetFilterAndAttributeResponse [simplified] sent");
}

void AnselIPCMessageBusObserver::sendSetFilterAndAttributeResponse(AnselIpc::Status status, int stackIdx, const AnselUIBase::EffectPropertiesDescription& props, std::vector<std::pair<int, AnselIpc::Status> > setAttributeResponses)
{
#if ANSEL_SIDE_PRESETS
    stackIdx = getUIStackIdxFromAnselStackIdx(stackIdx);
#endif
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    SetFilterAndAttributesResponse* resp = new SetFilterAndAttributesResponse;

    // Create the set filter response
    SetFilterResponse* setFilterResponse = CreateSetFilterResponse<SetFilterResponse>(status, stackIdx, props, m_state);
    setFilterResponse->set_status(status);
    setFilterResponse->set_stackidx(stackIdx);
    resp->set_allocated_setfilterresponse(setFilterResponse);

    // Create the set filter attribute responses
    for (size_t i = 0; i < setAttributeResponses.size(); i++)
    {
        resp->add_attributecontrolids(setAttributeResponses[i].first);
        auto* setFilterAttributeResponse = resp->add_setfilterattributeresponses();
        setFilterAttributeResponse->set_status(setAttributeResponses[i].second);
    }

    response->set_allocated_setfilterandattributesresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("SetFilterAndAttributeResponse sent");

    // used to debug filter properties
    if (getLogSeverity() <= LogSeverity::kDebug)
    {
        const bool printAnselIpcMessage = true;
        if (printAnselIpcMessage)
        {
            using Printer = google::protobuf::TextFormat;
            std::string tmpstr;
            Printer::PrintToString(message, &tmpstr);
            FormatStringForPrinting(tmpstr);
            LOG_DEBUG(tmpstr.c_str());
        }
    }
}

void AnselIPCMessageBusObserver::sendInsertFilterResponse(AnselIpc::Status status, int stackIdx)
{
    sendFilterResponse<InsertFilterResponse>(&AnselIPCResponse::set_allocated_insertfilterresponse, status, stackIdx);
}

void AnselIPCMessageBusObserver::sendInsertFilterResponse(AnselIpc::Status status, int stackIdx, const AnselUIBase::EffectPropertiesDescription& desc)
{
    sendFilterResponse<InsertFilterResponse>(&AnselIPCResponse::set_allocated_insertfilterresponse, status, stackIdx, desc);
}


void AnselIPCMessageBusObserver::clearFlags()
{
    m_anselIpcEnable = m_anselIpcDisable = false;
}

bool AnselIPCMessageBusObserver::isAlive()
{
    return m_messageBus.isAlive();
}

void AnselIPCMessageBusObserver::sendAnselReadyRequest()
{
    // this declared as static intentionally:
    // we want this request to return incremented creationCounter
    // everytime it is sent
    static uint32_t globalCounter = 1u;

    AnselIPCMessage message;
    AnselIPCRequest* request = new AnselIPCRequest;
    AnselReadyRequest* req = new AnselReadyRequest;
    req->set_creationcounter(globalCounter++);
    request->set_allocated_anselreadyrequest(req);
    message.set_allocated_request(request);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);

    if (!m_messageBus.isAlive())
        LOG_DEBUG("Message bus is not alive yet");

    m_messageBus.postMessage(msg);

    LOG_DEBUG("sendAnselReadyRequest sent");
}

void AnselIPCMessageBusObserver::sendIsAnselAvailableResponse()
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    IsAnselAvailableResponse* resp = new IsAnselAvailableResponse;
    resp->set_available(true);
    response->set_allocated_isanselavailableresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("IsAnselAvailableResponse sent");
}

void AnselIPCMessageBusObserver::sendLogFilenameResponse(const std::string& filename)
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    LogFilenameResponse* resp = new LogFilenameResponse;
    resp->set_filename(filename);
    response->set_allocated_logfilenameresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("LogFilenameResponse sent");
}

void AnselIPCMessageBusObserver::sendCaptureShotStartedResponse(Status status, uint32_t shotCount)
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    CaptureShotStartedResponse* resp = new CaptureShotStartedResponse;
    resp->set_status(status);
    resp->set_totalshotcount(shotCount);
    response->set_allocated_captureshotstartedresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("CaptureShotStartedResponse sent");
}

void AnselIPCMessageBusObserver::sendCaptureProgressResponse(uint32_t shotNo)
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    CaptureShotProgressResponse* resp = new CaptureShotProgressResponse;
    resp->set_lwrrentshot(shotNo);
    response->set_allocated_captureshotprogressresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("CaptureShotProgressResponse sent");
}

void AnselIPCMessageBusObserver::sendCaptureProcessingDoneResponse(Status status, const std::wstring & absFilename)
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    CaptureShotProcessingFinishedResponse* resp = new CaptureShotProcessingFinishedResponse;
    resp->set_status(status);
    resp->set_absolutefilepath(darkroom::getUtf8FromWstr(absFilename));
    response->set_allocated_captureshotprocessingfinishedresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("CaptureShotProcessingFinishedResponse sent");
}

void AnselIPCMessageBusObserver::sendLwrrentFovResponse()
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    GetLwrrentFOVResponse* resp = new GetLwrrentFOVResponse;
    resp->set_fov(m_state.sdk.cameraFov);
    response->set_allocated_getlwrrentfovresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("GetLwrrentFOVResponse sent");
}

void AnselIPCMessageBusObserver::sendAnselCaptureModeResponse(bool enabled)
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    GetAnselEnabledResponse* resp = new GetAnselEnabledResponse;
    resp->set_enabled(enabled);
    response->set_allocated_getanselenabledresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("GetAnselEnabledResponse sent");
}

void AnselIPCMessageBusObserver::sendAnselShotPermissionsResponse()
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    GetAnselShotPermissionsResponse* resp = new GetAnselShotPermissionsResponse;
    resp->set_isintegrationdetected(m_state.sdk.sdkDetected);
    for (size_t i = 0; i < ShotTypeIPC_ARRAYSIZE; ++i)
    {
        resp->add_isshotallowed(false);
    }
    resp->set_isshotallowed(kRegular, m_state.sdk.allowedShotTypes[(int)ShotType::kRegular]);
    resp->set_isshotallowed(kRegularStereo, m_state.sdk.allowedShotTypes[(int)ShotType::kStereo]);
    resp->set_isshotallowed(kHighres, m_state.sdk.allowedShotTypes[(int)ShotType::kHighRes]);
    resp->set_isshotallowed(kPanorama360Mono, m_state.sdk.allowedShotTypes[(int)ShotType::k360]);
    resp->set_isshotallowed(kPanorama360Stereo, m_state.sdk.allowedShotTypes[(int)ShotType::k360Stereo]);
    resp->set_ishdrallowed(m_state.sdk.allowedHDR);
    response->set_allocated_getanselshotpermissionsresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("GetAnselShotPermissionsResponse sent");
    LOG_DEBUG(" allowedHDR: %s", m_state.sdk.allowedHDR ? "true" : "false");
}

void AnselIPCMessageBusObserver::processEstimateCaptureRequest(const AnselIPCMessage& msg)
{
    auto& req = msg.request().estimatecapturerequest();
    const auto type = req.type();
    darkroom::ShotDescription desc;

    desc.bmpWidth = m_state.screenWidth;
    desc.bmpHeight = m_state.screenHeight;
    desc.overlap = darkroom::DefaultRecommendedSphericalOverlap;
    desc.eyeSeparation = 0.01f; // it doesn't really matter for estimates as long as it is > 0.0f
    desc.panoWidth = 0u;
    if (type == ShotTypeIPC::kRegular)
    {
        desc.type = darkroom::ShotDescription::EShotType::REGULAR;
        desc.generateThumbnail = req.generatethumbnail();
    }
    else if (type == ShotTypeIPC::kHighres)
    {
        desc.type = darkroom::ShotDescription::EShotType::HIGHRES;
        desc.highresMultiplier = req.highresmultiplier();
        desc.horizontalFov = m_state.sdk.cameraFov;
        desc.produceRegularImage = req.highresenhance();
        desc.generateThumbnail = req.generatethumbnail();
    }
    else if (type == ShotTypeIPC::kPanorama360Mono)
    {
        desc.type = darkroom::ShotDescription::EShotType::SPHERICAL_MONO_PANORAMA;
        if (req.horizontal360resolution() > 0.0f)
            desc.horizontalFov = float(darkroom::CameraDirector::estimateTileHorizontalFovSpherical(req.horizontal360resolution(), m_state.screenWidth));
        desc.generateThumbnail = req.generatethumbnail();
    }
    else if (type == ShotTypeIPC::kPanorama360Stereo)
    {
        desc.type = darkroom::ShotDescription::EShotType::SPHERICAL_STEREO_PANORAMA;
        if (req.horizontal360resolution() > 0.0f)
            desc.horizontalFov = float(darkroom::CameraDirector::estimateTileHorizontalFovSpherical(req.horizontal360resolution(), m_state.screenWidth));
        desc.generateThumbnail = req.generatethumbnail();
    }
    else if (type == ShotTypeIPC::kRegularStereo)
    {
        desc.type = darkroom::ShotDescription::EShotType::STEREO_REGULAR;
    }

    const auto estimates = darkroom::CameraDirector::estimateCaptureTask(desc);

    {
        AnselIPCMessage message;
        AnselIPCResponse* response = new AnselIPCResponse;
        EstimateCaptureResponse* resp = new EstimateCaptureResponse;
        resp->set_inputdatasetframecount(estimates.inputDatasetFrameCount);
        resp->set_inputdatasetframesizeinbytes(estimates.inputDatasetFrameSizeInBytes);
        resp->set_inputdatasetsizetotalinbytes(estimates.inputDatasetSizeTotalInBytes);
        resp->set_outputmpixels(estimates.outputMPixels);
        resp->set_outputresolutionx(estimates.outputResolutionX);
        resp->set_outputresolutiony(estimates.outputResolutionY);
        resp->set_outputsizeinbytes(estimates.outputSizeInBytes);
        resp->set_stitchermemoryrequirementsinbytes(estimates.stitcherMemoryRequirementsInBytes);
        response->set_allocated_estimatecaptureresponse(resp);
        message.set_allocated_response(response);
        GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
        m_messageBus.postMessage(msg);

        LOG_DEBUG("EstimateCaptureResponse sent");
    }
}

void AnselIPCMessageBusObserver::processUiControlChangedRequest(const AnselIpc::AnselIPCMessage& msg)
{
    auto& req = msg.request().uicontrolchangedrequest();
    switch (req.controlDescription_case())
    {
    case UIControlChangedRequest::ControlDescriptionCase::kUiDescBoolean:
    {
        AnselUIBase::EffectChange gameSpecificSettingChange;
        gameSpecificSettingChange.filterId = L"GameSpecific";
        gameSpecificSettingChange.stackIdx = AnselUIBase::GameSpecificStackIdx;
        gameSpecificSettingChange.controlId = req.uidescboolean().id();
        gameSpecificSettingChange.value = req.uidescboolean().set();
        m_state.filters.effectChanges.push_back(gameSpecificSettingChange);
        sendUiChangeResponse(Status::kOk);
        break;
    }
    case UIControlChangedRequest::ControlDescriptionCase::kUiDescSlider:
    {
        AnselUIBase::EffectChange gameSpecificSettingChange;
        gameSpecificSettingChange.filterId = L"GameSpecific";
        gameSpecificSettingChange.stackIdx = AnselUIBase::GameSpecificStackIdx;
        gameSpecificSettingChange.controlId = req.uidescslider().id();
        gameSpecificSettingChange.value = req.uidescslider().value();
        m_state.filters.effectChanges.push_back(gameSpecificSettingChange);
        sendUiChangeResponse(Status::kOk);
        break;
    }
    default:
        LOG_DEBUG("Unrecognized control description");
    }
}

void AnselIPCMessageBusObserver::processCaptureShotRequest(const AnselIPCMessage& msg)
{
#if 1
    auto& req = msg.request().captureshotrequest();
    const auto type = req.type();

    // a lot of response messages are sent from the ansel server / ansel sdk state directly
    // (when the appropriate action is actually started)

    if (req.has_isexr())
    {
        m_state.sdk.isShotEXR = req.isexr();
    }
    else
    {
        // Default JXR to true if EXR wasn't specified. This will be overridden below if JXR was specified
        m_state.sdk.isShotJXR = true;
    }

    if (req.has_isjxr())
    {
        m_state.sdk.isShotJXR = req.isjxr();
    }

    // TODO: Disabling JXR since Press Release Testing deemed it not good enough for release
    //m_state.sdk.isShotJXR = false;

    if (type == kRegular)
    {
        m_state.sdk.shotTypeToTake = ShotType::kRegular;
        if (req.has_generatethumbnail())
        {
            m_state.sdk.isShotPreviewRequired = req.generatethumbnail();
            m_state.sdk.isThumbnailRequired = req.generatethumbnail();
        }
        else
        {
            m_state.sdk.isShotPreviewRequired = false;
            m_state.sdk.isThumbnailRequired = false;
        }
    }
    else if (type == kRegularStereo)
    {
        m_state.sdk.shotTypeToTake = ShotType::kStereo;
        if (req.has_generatethumbnail())
        {
            m_state.sdk.isThumbnailRequired = req.generatethumbnail();
            m_state.sdk.isShotPreviewRequired = req.generatethumbnail();
        }
        else
        {
            m_state.sdk.isThumbnailRequired = false;
            m_state.sdk.isShotPreviewRequired = false;
        }
    }
    else if (type == kHighres)
    {
        if (req.has_highresmultiplier())
        {
#if 0
            const auto width = m_ui->getScreenWidth();
            const auto height = m_ui->getScreenHeight();
            const auto minHighresMultiplier = 2u;
            const auto maxHighresMultiplier = m_ui->m_UI->getMaxHighresMultiplier(width, height, m_ui->getMaximumHighresResolution() + 2);

            if (req.highresmultiplier() >= minHighresMultiplier && req.highresmultiplier() <= maxHighresMultiplier)
            {
                m_state.sdk.shotTypeToTake = ShotType::kHighRes;
                m_state.sdk.highresMultiplier = req.highresmultiplier();
            }
            else
                sendCaptureShotResponse(ILWALID_REQUEST, "", 0);
#else
            m_state.sdk.shotTypeToTake = ShotType::kHighRes;
            m_state.sdk.highresMultiplier = req.highresmultiplier();
            if (req.has_generatethumbnail())
            {
                m_state.sdk.isThumbnailRequired = req.generatethumbnail();
                m_state.sdk.isShotPreviewRequired = req.generatethumbnail();
            }
            else
            {
                m_state.sdk.isThumbnailRequired = false;
                m_state.sdk.isShotPreviewRequired = false;
            }
            if (req.has_highresenhance())
                m_state.sdk.isHighresEnhancementRequested = req.highresenhance();
            else
                m_state.sdk.isHighresEnhancementRequested = false;
#endif
        }
    }
    else if (type == kPanorama360Mono)
    {
        if (req.has_horizontal360resolution())
        {
            if (req.horizontal360resolution() >= s_min360Resolution && req.horizontal360resolution() <= s_max360Resolution)
            {
                m_state.sdk.shotTypeToTake = ShotType::k360;
                m_state.sdk.pano360ResultWidth = req.horizontal360resolution();
                if (req.has_generatethumbnail())
                {
                    m_state.sdk.isThumbnailRequired = req.generatethumbnail();
                    m_state.sdk.isShotPreviewRequired = req.generatethumbnail();
                }
                else
                {
                    m_state.sdk.isThumbnailRequired = false;
                    m_state.sdk.isShotPreviewRequired = false;
                }
            }
        }
    }
    else if (type == kPanorama360Stereo)
    {
        if (req.has_horizontal360resolution())
        {
            m_state.sdk.shotTypeToTake = ShotType::k360Stereo;
            m_state.sdk.pano360ResultWidth = req.horizontal360resolution();
            if (req.has_generatethumbnail())
            {
                m_state.sdk.isThumbnailRequired = req.generatethumbnail();
                m_state.sdk.isShotPreviewRequired = req.generatethumbnail();
            }
            else
            {
                m_state.sdk.isThumbnailRequired = false;
                m_state.sdk.isShotPreviewRequired = false;
            }
        }
    }
#endif
}

void AnselIPCMessageBusObserver::sendGetHighresResolutionListResponse()
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    GetHighresResolutionListResponse* resp = new GetHighresResolutionListResponse;
    // fill response with data
    const auto minHighresMultiplier = 2u;
    for (size_t i = 0, iend = m_state.sdk.highresResolutions.size(); i < iend; ++i)
    {
        const AnselUIBase::HighResolutionEntry & highResEntry = m_state.sdk.highresResolutions[i];

        auto* resolutionOption = resp->add_resolutions();
        resolutionOption->set_multiplier(google::protobuf::int32(i + minHighresMultiplier));
        resolutionOption->set_xresolution(int32_t(highResEntry.width));
        resolutionOption->set_yresolution(int32_t(highResEntry.height));
    }
    response->set_allocated_gethighresresolutionlistresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("GetHighresResolutionListResponse sent");
}

void AnselIPCMessageBusObserver::sendFovRangeResponse()
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    GetFOVRangeResponse* resp = new GetFOVRangeResponse;
    resp->set_minfov((float)m_state.sdk.fovMin);
    resp->set_maxfov((float)m_state.sdk.fovMax);
    response->set_allocated_getfovrangeresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("GetFOVRangeResponse sent");
}

void AnselIPCMessageBusObserver::sendRollRangeResponse()
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    GetRollRangeResponse* resp = new GetRollRangeResponse;
    // TODO: do not hard code these values
    resp->set_minroll((float)m_state.sdk.rollMin);
    resp->set_maxroll((float)m_state.sdk.rollMax);
    response->set_allocated_getrollrangeresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("GetRollRangeResponse sent");
}

void AnselIPCMessageBusObserver::sendFilterListResponse()
{
    auto filterList = m_state.filters.filterNames;
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    GetFilterListResponse* getFilterListResponse = new GetFilterListResponse;
    int index = 0;
    for (const auto& filterName : filterList)
    {
        getFilterListResponse->add_filteridlist(darkroom::getUtf8FromWstr(filterName.first).data());
        getFilterListResponse->add_filternamelist(darkroom::getUtf8FromWstr(filterName.second).data());
    }
    response->set_allocated_getfilterlistresponse(getFilterListResponse);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("GetFilterListResponse sent");
}

void AnselIPCMessageBusObserver::send360ResolutionRangeResponse()
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    Get360ResolutionRangeResponse* resp = new Get360ResolutionRangeResponse;
    resp->set_minimumxresolution(m_state.sdk.pano360MinWidth);
    resp->set_maximumxresolution(m_state.sdk.pano360MaxWidth);
    response->set_allocated_get360resolutionrangeresponse(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("Get360ResolutionRangeResponse sent");
}

void AnselIPCMessageBusObserver::processSetFovRequest(const AnselIPCMessage& msg)
{
    const float fov = msg.request().setfovrequest().fov();
    if (fov >= (float)m_state.sdk.fovMin && fov <= (float)m_state.sdk.fovMax)
    {
        m_state.sdk.cameraFov = fov;
        m_state.sdk.isCameraFOVChanged = true;
        sendSetFovResponse(kOk);
    }
    else
        sendSetFovResponse(kOutOfRange);
}

void AnselIPCMessageBusObserver::processSetRollRequest(const AnselIPCMessage& msg)
{
    const float roll = msg.request().setrollrequest().roll();
    if (roll >= (float)m_state.sdk.rollMin && roll <= (float)m_state.sdk.rollMax)
    {
        m_state.sdk.cameraRoll = roll;
        m_state.sdk.isCameraRollChanged = true;
        sendSetRollResponse(kOk);
    }
    else
        sendSetRollResponse(kOutOfRange);
}

// Status responses go below (nothing interesting)

template<typename T, void(AnselIPCResponse::*func)(T* obj)>
void AnselIPCMessageBusObserver::sendStatusResponse(AnselIpc::Status status)
{
    AnselIPCMessage message;
    AnselIPCResponse* response = new AnselIPCResponse;
    T* resp = new T;
    resp->set_status(status);
    (response->*func)(resp);
    message.set_allocated_response(response);
    GenericBusMessage msg(AnselIPCSystem, m_lwrAnselIPCModule.c_str(), message.SerializeAsString(), GFEIPCSystem, GFEIPCModule);
    m_messageBus.postMessage(msg);

    LOG_DEBUG("%s sent", typeid(T).name());
}

void AnselIPCMessageBusObserver::sendUiChangeResponse(AnselIpc::Status status)
{
    sendStatusResponse<UIControlChangedResponse, &AnselIPCResponse::set_allocated_uicontrolchangedresponse>(status);
}

void AnselIPCMessageBusObserver::sendSetStyleTransferEnabledResponse(AnselIpc::Status status)
{
    sendStatusResponse<SetStyleTransferEnabledResponse, &AnselIPCResponse::set_allocated_setstyletransferenabledresponse>(status);
}

void AnselIPCMessageBusObserver::sendSetStyleTransferStyleResponse(AnselIpc::Status status)
{
    sendStatusResponse<SetStyleTransferStyleResponse, &AnselIPCResponse::set_allocated_setstyletransferstyleresponse>(status);
}

void AnselIPCMessageBusObserver::sendSetGridOfThirdsEnabledResponse(AnselIpc::Status status)
{
    sendStatusResponse<SetGridOfThirdsEnabledResponse, &AnselIPCResponse::set_allocated_setgridofthirdsenabledresponse>(status);
}

void AnselIPCMessageBusObserver::sendResetEntireStackResponse(AnselIpc::Status status)
{
    sendStatusResponse<ResetEntireStackResponse, &AnselIPCResponse::set_allocated_resetentirestackresponse>(status);
}

void AnselIPCMessageBusObserver::sendSetFilterAttributeRequest(Status status)
{
    sendStatusResponse<SetFilterAttributeResponse, &AnselIPCResponse::set_allocated_setfilterattributeresponse>(status);
}

void AnselIPCMessageBusObserver::sendIsAnselSDKIntegrationAvailableResponse(Status status)
{
    sendStatusResponse<IsAnselSDKIntegrationAvailableResponse, &AnselIPCResponse::set_allocated_isanselsdkintegrationavailableresponse>(status);
}

void AnselIPCMessageBusObserver::sendAbortCaptureResponse(Status status)
{
    sendStatusResponse<AbortCaptureResponse, &AnselIPCResponse::set_allocated_abortcaptureresponse>(status);
}

void AnselIPCMessageBusObserver::sendCaptureShotFinishedResponse(Status status)
{
    sendStatusResponse<CaptureShotFinishedResponse, &AnselIPCResponse::set_allocated_captureshotfinishedresponse>(status);
}

void AnselIPCMessageBusObserver::sendAnselEnableResponse(Status status)
{
    sendStatusResponse<SetAnselEnabledResponse, &AnselIPCResponse::set_allocated_setanselenabledresponse>(status);
}

void AnselIPCMessageBusObserver::sendSetFovResponse(Status status)
{
    sendStatusResponse<SetFOVResponse, &AnselIPCResponse::set_allocated_setfovresponse>(status);
}

void AnselIPCMessageBusObserver::sendSetRollResponse(Status status)
{
    sendStatusResponse<SetRollResponse, &AnselIPCResponse::set_allocated_setrollresponse>(status);
}

void AnselIPCMessageBusObserver::sendInputEventResponse(Status status)
{
    sendStatusResponse<InputEventResponse, &AnselIPCResponse::set_allocated_inputeventresponse>(status);
}

void AnselIPCMessageBusObserver::sendSetLangIdResponse(Status status)
{
    sendStatusResponse<SetLangIdResponse, &AnselIPCResponse::set_allocated_setlangidresponse>(status);
}

void AnselIPCMessageBusObserver::sendRemoveFilterResponse(AnselIpc::Status status)
{
    sendStatusResponse<RemoveFilterResponse, &AnselIPCResponse::set_allocated_removefilterresponse>(status);
}

void AnselIPCMessageBusObserver::sendMoveFilterResponse(AnselIpc::Status status)
{
    sendStatusResponse<MoveFilterResponse, &AnselIPCResponse::set_allocated_movefilterresponse>(status);
}

void AnselIPCMessageBusObserver::sendSetStyleTransferModelResponse(AnselIpc::Status status)
{
    sendStatusResponse<SetStyleTransferModelResponse, &AnselIPCResponse::set_allocated_setstyletransfermodelresponse>(status);
}

void AnselIPCMessageBusObserver::sendSetHighQualityResponse(AnselIpc::Status status)
{
    sendStatusResponse<SetHighQualityResponse, &AnselIPCResponse::set_allocated_sethighqualityresponse>(status);
}

void AnselIPCMessageBusObserver::sendSetCMSInfoResponse(AnselIpc::Status status)
{
    sendStatusResponse<SetCMSInfoResponse, &AnselIPCResponse::set_allocated_setcmsinforesponse>(status);
}

int AnselIPCMessageBusObserver::getUIStackIdxFromAnselStackIdx(int stackIdx)
{
    return (int) std::distance(m_state.filters.stackIdxTranslation.begin(),
        std::find( m_state.filters.stackIdxTranslation.begin(),
            m_state.filters.stackIdxTranslation.end(),
            stackIdx));
}

void AnselIPCMessageBusObserver::handleInputEventRequest(const AnselIpc::InputEventRequest& inputEvent)
{
    const int32_t msgId = inputEvent.message();
    const int32_t wParam = inputEvent.wparam();
    const int32_t lParam = inputEvent.lparam();

    bool isCoordsDelta = false;
    if (inputEvent.has_isdeltacoords())
        isCoordsDelta = inputEvent.isdeltacoords();

    input::InputEvent ev;
    ev.event.gamepadStateUpdate = { 0 };
    ev.type = input::InputEvent::Type::kGamepadStateUpdate;
    bool wasGamepadStateUpdated = false;
    if (inputEvent.has_leftstickxvalue())
    {
        const auto val = inputEvent.leftstickxvalue();
        ev.event.gamepadStateUpdate.axisLX = static_cast<short>(val > 0.0f ? val * std::numeric_limits<short>::max() : -val * std::numeric_limits<short>::min());
        wasGamepadStateUpdated = true;
    }
    if (inputEvent.has_leftstickyvalue())
    {
        const auto val = inputEvent.leftstickyvalue();
        ev.event.gamepadStateUpdate.axisLY = static_cast<short>(val > 0.0f ? val * std::numeric_limits<short>::max() : -val * std::numeric_limits<short>::min());
        wasGamepadStateUpdated = true;
    }
    if (inputEvent.has_rightstickxvalue())
    {
        const auto val = inputEvent.rightstickxvalue();
        ev.event.gamepadStateUpdate.axisRX = static_cast<short>(val > 0.0f ? val * std::numeric_limits<short>::max() : -val * std::numeric_limits<short>::min());
        wasGamepadStateUpdated = true;
    }
    if (inputEvent.has_rightstickyvalue())
    {
        const auto val = inputEvent.rightstickyvalue();
        ev.event.gamepadStateUpdate.axisRY = static_cast<short>(val > 0.0f ? val * std::numeric_limits<short>::max() : -val * std::numeric_limits<short>::min());
        wasGamepadStateUpdated = true;
    }
    if (inputEvent.has_lefttriggervalue())
    {
        const auto val = inputEvent.lefttriggervalue();
        ev.event.gamepadStateUpdate.axisZ = static_cast<short>(val > 0.0f ? val * std::numeric_limits<short>::max() : -val * std::numeric_limits<short>::min());
        wasGamepadStateUpdated = true;
    }
    if (inputEvent.has_righttriggervalue())
    {
        const auto val = -inputEvent.righttriggervalue();
        ev.event.gamepadStateUpdate.axisZ = static_cast<short>(val > 0.0f ? val * std::numeric_limits<short>::max() : -val * std::numeric_limits<short>::min());
        wasGamepadStateUpdated = true;
    }

    if (wasGamepadStateUpdated)
        m_state.m_inputstate.pushBackIpcInputEvent(ev);

    switch (msgId)
    {
        case WM_KEYDOWN:
            m_state.m_inputstate.pushBackKeyDownEvent(wParam);
            break;
        case WM_KEYUP:
            m_state.m_inputstate.pushBackKeyUpEvent(wParam);
            break;
        case WM_LBUTTONDOWN:
        {
            int xPosDelta, yPosDelta;
            getMouseDeltas(lParam, isCoordsDelta, &xPosDelta, &yPosDelta);

            m_state.m_inputstate.pushBackMouseLButtonDownEvent(xPosDelta, yPosDelta);
            break;
        }
        case WM_LBUTTONUP:
        {
            int xPosDelta, yPosDelta;
            getMouseDeltas(lParam, isCoordsDelta, &xPosDelta, &yPosDelta);

            m_state.m_inputstate.pushBackMouseLButtonUpEvent(xPosDelta, yPosDelta);
            break;
        }
        case WM_RBUTTONDOWN:
        {
            int xPosDelta, yPosDelta;
            getMouseDeltas(lParam, isCoordsDelta, &xPosDelta, &yPosDelta);

            m_state.m_inputstate.pushBackMouseRButtonDownEvent(xPosDelta, yPosDelta);
            break;
        }
        case WM_RBUTTONUP:
        {
            int xPosDelta, yPosDelta;
            getMouseDeltas(lParam, isCoordsDelta, &xPosDelta, &yPosDelta);

            m_state.m_inputstate.pushBackMouseRButtonUpEvent(xPosDelta, yPosDelta);
            break;
        }
        case WM_MBUTTONDOWN:
        {
            int xPosDelta, yPosDelta;
            getMouseDeltas(lParam, isCoordsDelta, &xPosDelta, &yPosDelta);

            m_state.m_inputstate.pushBackMouseMButtonDownEvent(xPosDelta, yPosDelta);
            break;
        }
        case WM_MBUTTONUP:
        {
            int xPosDelta, yPosDelta;
            getMouseDeltas(lParam, isCoordsDelta, &xPosDelta, &yPosDelta);

            m_state.m_inputstate.pushBackMouseMButtonUpEvent(xPosDelta, yPosDelta);
            break;
        }
        case WM_MOUSEMOVE:
        {
            int xPosDelta, yPosDelta;
            getMouseDeltas(lParam, isCoordsDelta, &xPosDelta, &yPosDelta);

            m_state.m_inputstate.pushBackMouseMoveEvent(xPosDelta, yPosDelta);
            break;
        }
        case WM_MOUSEWHEEL:
        {
            int xPosDelta, yPosDelta;
            getMouseDeltas(lParam, isCoordsDelta, &xPosDelta, &yPosDelta);
            int zDelta = GET_WHEEL_DELTA_WPARAM(wParam);
            m_state.m_inputstate.pushBackMouseMoveEvent(xPosDelta, yPosDelta, zDelta);
            break;
        }
        default:
        {
            break;
        }
    };
}

#endif
