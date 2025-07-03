#include <algorithm>
#include <numeric>
#include <iostream>
#include <sstream>
#include <thread>
#include <Windows.h>
#define POCO_NO_AUTOMATIC_LIBS
#pragma warning(push)
#pragma warning(disable:4267 4244)
#include "GenericBusMessage.h"
#include "MessageBus.h"
#include "MessageBusSelwre.h"
#include "google\protobuf\text_format.h"
#include "../../ShaderMod/include/ipc/ipc.pb.h"
#pragma warning(pop)

#pragma comment(lib, "MessageBus.lib")
#pragma comment(lib, "delayimp")

using namespace AnselIpc;

const char* AnselIPCSystem = "Ansel";
const char* AnselIPCModule = "LwCamera";
const char* GFEIPCSystem = AnselIPCSystem;
const char* GFEIPCModule = "LwCameraControl";

int stringToVKey(const std::string & in)
{
    int vKey = 0;
    std::string lowerIn = in;
    std::transform(in.begin(), in.end(), lowerIn.begin(), ::tolower);
    if (lowerIn == "w")
    {
        vKey = 'W';
    }
    else if (lowerIn == "a")
    {
        vKey = 'A';
    }
    else if (lowerIn == "s")
    {
        vKey = 'S';
    }
    else if (lowerIn == "d")
    {
        vKey = 'D';
    }
    else if (lowerIn == "shift")
    {
        vKey = VK_SHIFT;
    }
    else if (lowerIn == "ctrl")
    {
        vKey = VK_CONTROL;
    }

    return vKey;
}

void replaceAll(std::string &s, const std::string &search, const std::string &replace) 
{
    for (size_t pos = 0; ; pos += replace.length()) {
        // Locate the substring to replace
        pos = s.find(search, pos);
        if (pos == std::string::npos) break;
        // Replace by erasing and inserting
        s.erase(pos, search.length());
        s.insert(pos, replace);
    }
}

namespace ipc
{

    const uint16_t broadcastPort = 12345;
    struct AnselMessageBusObserver : public BusObserver
    {
        MessageBus& m_messageBus;

        AnselMessageBusObserver(MessageBus& messageBus) : m_messageBus(messageBus) {}

        void onBusMessage(const BusMessage& msg)
        {
            std::cout << "Received something ..." << std::endl;
            if (msg.has_generic())
            {
                if (!(msg.source_system() == "Ansel" && msg.source_module() == "LwCamera"))
                    return;

                if (msg.has_generic())
                {
                    const BusMessage_Generic& generic = msg.generic();
                    AnselIPCMessage message;
                    const auto& data = generic.data();
                    message.ParseFromArray(data.data(), int(data.size()));

                    if (message.message_case() == AnselIPCMessage::MessageCase::kRequest)
                    {
                        switch (message.request().request_case())
                        {
                        case AnselIPCRequest::RequestCase::kGetEnabledFeatureSetRequest:
                        {
                            AnselIPCMessage message;
                            AnselIPCResponse* response = new AnselIPCResponse;
                            GetEnabledFeatureSetResponse* resp = new GetEnabledFeatureSetResponse;
                            resp->set_modsavailable(true);
                            response->set_allocated_getenabledfeaturesetresponse(resp);
                            message.set_allocated_response(response);
                            GenericBusMessage msg(AnselIPCSystem, GFEIPCModule, message.SerializeAsString());
                            m_messageBus.postMessage(msg);
                            std::cout << "Sent GetEnabledFeatureSetResponse (mods are enabled)" << std::endl;
                        }
                        break;
                        }
                    }


#if 0
                    // DBG code to check if certain responses are getting through
                    AnselIPCMessage::MessageCase msgCase = message.message_case();
                    switch (message.message_case())
                    {
                    case AnselIPCMessage::MessageCase::kResponse:
                    {
                        AnselIPCResponse::ResponseCase responseCase = message.response().response_case();
                        switch (message.response().response_case())
                        {
                        case AnselIPCResponse::ResponseCase::kIsAnselAvailableResponse:
                        {
                            printf("is available responded!\n");
                            break;
                        }
                        }
                        break;
                    }
                    }
#endif
                    using Printer = google::protobuf::TextFormat;
                    std::string tmpstr;
                    Printer::PrintToString(message, &tmpstr);
                    replaceAll(tmpstr, "\\\\", "\\");
                    std::cout << tmpstr << std::endl;
                }
            }
            else if (msg.has_joined())
            {
                const char* joined = msg.joined() ? "JOIN" : "LEAVE";
                std::cout << joined << " from: " << msg.source_system() << ":" << msg.source_module() << std::endl;
            }
            else if (msg.has_status() && msg.bus_peer_size() > 0)
            {
                std::cout << "JOINED bus! Peers on bus as of JOIN: " << std::endl;
                for (int i = 0; i < msg.bus_peer_size(); ++i)
                {
                    const BusMessage::Peer& peer = msg.bus_peer(i);
                    std::cout << "\t" << peer.system() << ":" << peer.module() << std::endl;
                }
            }
            else
            {
                std::cout << "got a busmessage that didn't meet criteria." << std::endl;
            }
        }
    };
}

template<typename T>
T parse_argument(const std::string& line)
{
    std::istringstream ss(line);
    T result = T();
    std::string tmp;
    ss >> tmp >> result;
    return result;
}

template<typename T>
bool parse_arguments_helper(const std::string& line, T& readValue)
{
    std::istringstream ss(line);
    ss >> readValue;

    return true;
}

template<typename T, typename... MoreArgs>
bool parse_arguments_helper(const std::string& line, T& readValue, MoreArgs (&... moreArgs))
{
    std::istringstream ss(line);
    ss >> readValue;

    size_t nextPos = line.find(' ');

    if (nextPos == std::string::npos)
        return false;

    return parse_arguments_helper(line.substr(nextPos + 1), moreArgs...);
}

template<typename... Args>
bool parse_arguments(const std::string& line, Args (&... args))
{
    size_t nextPos = line.find(' ');
    
    if (nextPos == std::string::npos)
        return false;

    return parse_arguments_helper(line.substr(nextPos + 1), args...);
}

template<typename T>
std::vector<T> parse_argument_as_vector(const std::string& input)
{
    std::vector<T> result;
    auto line = input;
    size_t nextPos = line.find(' ');

    while (nextPos != std::string::npos)
    {
        T val;
        line = line.substr(nextPos + 1);
        std::istringstream ss(line);
        ss >> val;
        result.push_back(val);
        nextPos = line.find(' ');
    }
    return result;
}

std::vector<std::string> split(const std::string& s, char separator)
{
    std::vector<std::string> output;
    std::string::size_type prev_pos = 0, pos = 0;

    while ((pos = s.find(separator, pos)) != std::string::npos)
    {
        std::string substring(s.substr(prev_pos, pos - prev_pos));
        output.push_back(substring);
        prev_pos = ++pos;
    }

    output.push_back(s.substr(prev_pos, pos - prev_pos));

    return output;
}

int main(int argc, char** argv)
{
    MessageBus messageBus(getMessageBusInterfaceSelwre(MESSAGE_BUS_INTERFACE_VERSION));
    ipc::AnselMessageBusObserver s_observer(messageBus);

    std::vector<std::string> bufferedCommands;

    messageBus.addObserver(&s_observer, "Ansel", "LwCameraControl");
    std::string line;
    while ((bufferedCommands.size() > 0) || std::getline(std::cin, line))
    {
        if (bufferedCommands.size() > 0)
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            line = bufferedCommands[0];
            bufferedCommands.erase(bufferedCommands.begin());

            std::cout << "parsing buffered: \"" << line << "\"" << std::endl;
        }

        if (line == "quit") { break; }
        else if (line == "seq-enable10")
        {
            // Sequence to test games that enter menu (and hence deny Ansel session) on lost focus
            //	10 commands (and hence 10 seconds) should be enough to restore focus and get back to the gameplay
            for (int i = 0; i < 10; ++i)
            {
                bufferedCommands.push_back("enable");
            }
        }
        else if (line == "seq-rolltest")
        {
            char buf[64];
            for (int i = 0; i < 180; ++i)
            {
                sprintf_s(buf, 64, "setroll %d", i+1);
                bufferedCommands.push_back(buf);
                bufferedCommands.push_back(std::string("setroll 0"));
            }
        }
        else if (line == "seq-filterset")
        {
            bufferedCommands.push_back("set-filter None 0");
            bufferedCommands.push_back("set-filter Adjustments 1");
            bufferedCommands.push_back("set-filter SpecialFX 2");
            bufferedCommands.push_back("set-control 2 2 1.0");		// Setting SpecialFX.Vignette to a maximum (1.0)
        }
        else if (line == "seq-gfeinit")
        {
            bufferedCommands.push_back("fovrange");
            bufferedCommands.push_back("rollrange");
            bufferedCommands.push_back("getfov");
            bufferedCommands.push_back("set-filter None 0");
            bufferedCommands.push_back("get-resolution");
            bufferedCommands.push_back("set-filter Adjustments 1");
            bufferedCommands.push_back("set-filter SpecialFX 2");
            bufferedCommands.push_back("reset-controls 0");
            bufferedCommands.push_back("reset-controls 1");
            bufferedCommands.push_back("reset-controls 2");
        }
        else if (line.find("game-controls") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            GetGameSpecificControlsRequest* req = new GetGameSpecificControlsRequest;
            request->set_allocated_getgamespecificcontrolsrequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("restyle-style") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            SetStyleTransferStyleRequest* req = new SetStyleTransferStyleRequest;
            const auto style = parse_argument<std::string>(line);
            req->set_fullyqualifiedpath(style);
            request->set_allocated_setstyletransferstylerequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("restyle-modellist") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            GetStyleTransferModelListRequest* req = new GetStyleTransferModelListRequest;
            request->set_allocated_getstyletransfermodellistrequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("restyle-model") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            SetStyleTransferModelRequest* req = new SetStyleTransferModelRequest;
            const auto modelId = parse_argument<std::string>(line);
            req->set_modelid(modelId);
            request->set_allocated_setstyletransfermodelrequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line == "restyle-enable")
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            SetStyleTransferEnabledRequest* req = new SetStyleTransferEnabledRequest;
            req->set_enabled(true);
            request->set_allocated_setstyletransferenabledrequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line == "settings")
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            GetSettingsRequest* req = new GetSettingsRequest;
            request->set_allocated_getsettingsrequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line == "restyle-disable")
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            SetStyleTransferEnabledRequest* req = new SetStyleTransferEnabledRequest;
            req->set_enabled(false);
            request->set_allocated_setstyletransferenabledrequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("enable") != std::string::npos || line.find("disable") != std::string::npos)
        {
            const bool enable = line.find("enable") != std::string::npos;

            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            SetAnselEnabledRequest* setAnselEnabledRequest = new SetAnselEnabledRequest;
            const bool arg = parse_argument<bool>(line);
            // enable 0/1 enables Ansel, while optionally not setting the game on pause
            // disable 0/1 disables ansel, while optionally leaving filters enabled
            if (enable && line != "enable")
                setAnselEnabledRequest->set_pauseapplication(arg);				
            else
                setAnselEnabledRequest->set_leavefiltersenabled(arg);				
            setAnselEnabledRequest->set_enabled(enable);
            IpcVersionResponse version;
            setAnselEnabledRequest->set_major(version.major());
            setAnselEnabledRequest->set_minor(version.minor());
            setAnselEnabledRequest->set_patch(version.patch());
            request->set_allocated_setanselenabledrequest(setAnselEnabledRequest);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line == "ready")
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            UIReadyRequest* uiReadyRequest = new UIReadyRequest;
            uiReadyRequest->set_status(kOk);
            request->set_allocated_uireadyrequest(uiReadyRequest);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line == "version")
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            IpcVersionRequest* ipcVersionRequest = new IpcVersionRequest;
            request->set_allocated_ipcversionrequest(ipcVersionRequest);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("gamespecific") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            UIControlChangedRequest* uiChangeRequest = new UIControlChangedRequest;
            std::string type;
            uint32_t id = 0u;
            float value = 0.0f;
            parse_arguments<std::string, uint32_t, float>(line, type, id, value);
            if (type == "boolean")
            {
                UIDescBoolean* control = new UIDescBoolean;
                control->set_id(id);
                control->set_set(value != 0.0f);
                uiChangeRequest->set_allocated_uidescboolean(control);
            }
            else if (type == "slider")
            {
                UIDescSlider* control = new UIDescSlider;
                control->set_id(id);
                control->set_value(value);
                uiChangeRequest->set_allocated_uidescslider(control);
            }
            request->set_allocated_uicontrolchangedrequest(uiChangeRequest);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
            
        }
        else if (line == "filterlist")
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            GetFilterListRequest* getFilterListRequest = new GetFilterListRequest;
            request->set_allocated_getfilterlistrequest(getFilterListRequest);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line == "highreslist")
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            GetHighresResolutionListRequest* req = new GetHighresResolutionListRequest;
            request->set_allocated_gethighresresolutionlistrequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line == "sdk")
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            IsAnselSDKIntegrationAvailableRequest* req = new IsAnselSDKIntegrationAvailableRequest;
            request->set_allocated_isanselsdkintegrationavailablerequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line == "360range")
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            Get360ResolutionRangeRequest* req = new Get360ResolutionRangeRequest;
            request->set_allocated_get360resolutionrangerequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line == "rollrange")
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            GetRollRangeRequest* req = new GetRollRangeRequest;
            request->set_allocated_getrollrangerequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("grid") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            SetGridOfThirdsEnabledRequest* req = new SetGridOfThirdsEnabledRequest;
            req->set_enabled(parse_argument<bool>(line));
            request->set_allocated_setgridofthirdsenabledrequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("estimate") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            EstimateCaptureRequest* req = new EstimateCaptureRequest;
            std::string type;
            uint32_t mult = 0u;
            bool genThumbnail = false, enhanceHighres = false;
            parse_arguments<std::string, uint32_t, bool, bool>(line, type, mult, genThumbnail, enhanceHighres);

            if (type == "regular")
                req->set_type(ShotTypeIPC::kRegular);
            else if (type == "highres")
            {
                req->set_type(ShotTypeIPC::kHighres);
                req->set_highresmultiplier(mult);
                req->set_highresenhance(enhanceHighres);
            }
            else if (type == "360mono")
            {
                req->set_type(ShotTypeIPC::kPanorama360Mono);
                req->set_horizontal360resolution(mult);
            }
            else if (type == "360stereo")
            {
                req->set_type(ShotTypeIPC::kPanorama360Stereo);
                req->set_horizontal360resolution(mult);
            }
            else if (type == "stereo")
                req->set_type(ShotTypeIPC::kRegularStereo);

            req->set_generatethumbnail(genThumbnail);
            
            request->set_allocated_estimatecapturerequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("setroll") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            SetRollRequest* req = new SetRollRequest;
            req->set_roll(parse_argument<float>(line));
            request->set_allocated_setrollrequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line == "fovrange")
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            GetFOVRangeRequest* req = new GetFOVRangeRequest;
            request->set_allocated_getfovrangerequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("setfov") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            SetFOVRequest* req = new SetFOVRequest;
            req->set_fov(parse_argument<float>(line));
            request->set_allocated_setfovrequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("getfov") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            GetLwrrentFOVRequest* req = new GetLwrrentFOVRequest;
            request->set_allocated_getlwrrentfovrequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("shot-regular") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            CaptureShotRequest* req = new CaptureShotRequest;
            request->set_allocated_captureshotrequest(req);
            req->set_type(kRegular);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("shot-stereo") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            CaptureShotRequest* req = new CaptureShotRequest;
            request->set_allocated_captureshotrequest(req);
            const bool exr = parse_argument<bool>(line);
            req->set_type(kRegularStereo);
            req->set_isexr(exr);
            req->set_generatethumbnail(exr);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("shot-exr") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            CaptureShotRequest* req = new CaptureShotRequest;
            request->set_allocated_captureshotrequest(req);
            req->set_type(kRegular);
            req->set_isexr(true);
            req->set_generatethumbnail(true);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("shot-highres") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            CaptureShotRequest* req = new CaptureShotRequest;
            request->set_allocated_captureshotrequest(req);
            uint32_t mult = 0u;
            bool generateThumbnail = false;
            bool enhanceHighres = false;
            bool exr = false;
            parse_arguments<uint32_t, bool, bool, bool>(line, mult, exr, generateThumbnail, enhanceHighres);
            req->set_highresmultiplier(mult);
            std::cout << "Enhance: " << enhanceHighres << std::endl;
            req->set_highresenhance(enhanceHighres);
            req->set_isexr(exr);
            req->set_generatethumbnail(generateThumbnail);
            req->set_type(kHighres);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("shot-360mono") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            CaptureShotRequest* req = new CaptureShotRequest;
            request->set_allocated_captureshotrequest(req);
            uint32_t res = 0;
            bool exr = false;
            parse_arguments<uint32_t, bool>(line, res, exr);
            req->set_horizontal360resolution(res);
            if (exr)
                req->set_generatethumbnail(true);
            req->set_isexr(exr);
            req->set_type(kPanorama360Mono);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("shot-360stereo") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            CaptureShotRequest* req = new CaptureShotRequest;
            request->set_allocated_captureshotrequest(req);
            uint32_t res = 0;
            bool exr = false;
            parse_arguments<uint32_t, bool>(line, res, exr);
            req->set_horizontal360resolution(res);
            req->set_isexr(exr);
            if (exr)
                req->set_generatethumbnail(true);
            req->set_type(kPanorama360Stereo);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("get-resolution") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            GetScreenResolutionRequest* req = new GetScreenResolutionRequest;
            request->set_allocated_getscreenresolutionrequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("abort") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            AbortCaptureRequest* req = new AbortCaptureRequest;
            request->set_allocated_abortcapturerequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("stack-info") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            GetStackInfoRequest* req = new GetStackInfoRequest;
            request->set_allocated_getstackinforequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("reset-stack") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            ResetEntireStackRequest* req = new ResetEntireStackRequest;
            request->set_allocated_resetentirestackrequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("set-filter") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            SetFilterRequest* req = new SetFilterRequest;
            request->set_allocated_setfilterrequest(req);

            const auto args = split(line, ' ');
            const std::string filterId = std::accumulate(std::next(args.cbegin()), std::prev(args.cend()), std::string(), 
                [](const auto& a, const auto& b) { return !a.empty() ? a + " " + b : b;	});
            int stackIdx = std::stoi(*std::prev(args.cend()));

            req->set_stackidx(stackIdx);
            // TODO: use the defaultEffectsFolder that comes via the IPC through the UIBase::setDefaultEffectPath
            req->set_filterid(filterId);

            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("set-filter-path") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            SetFilterRequest* req = new SetFilterRequest;
            request->set_allocated_setfilterrequest(req);

            std::string filterId;
            int stackIdx = 0;
            parse_arguments<std::string, int>(line, filterId, stackIdx);

            req->set_stackidx(stackIdx);
            req->set_filterid(filterId);

            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("insert-filter") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            InsertFilterRequest* req = new InsertFilterRequest;
            request->set_allocated_insertfilterrequest(req);

            const auto args = split(line, ' ');
            const std::string filterId = std::accumulate(std::next(args.cbegin()), std::prev(args.cend()), std::string(),
                [](const auto& a, const auto& b) { return !a.empty() ? a + " " + b : b;	});
            int stackIdx = std::stoi(*std::prev(args.cend()));

            req->set_stackidx(stackIdx);
            req->set_filterid(filterId);

            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("remove-filter") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            RemoveFilterRequest* req = new RemoveFilterRequest;
            request->set_allocated_removefilterrequest(req);

            int stackIdx = parse_argument<int>(line);

            req->set_stackidx(stackIdx);

            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("move-filter") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            MoveFilterRequest* req = new MoveFilterRequest;
            request->set_allocated_movefilterrequest(req);

            std::vector<uint32_t> stack = parse_argument_as_vector<uint32_t>(line);

            for (const auto x : stack)
                req->add_desiredstackindices(x);

            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("get-filterinfo") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            GetFilterInfoRequest* req = new GetFilterInfoRequest;
            request->set_allocated_getfilterinforequest(req);

            int stackIdx = parse_argument<int>(line);

            req->set_stackidx(stackIdx);

            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("reset-controls") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            ResetFilterValuesRequest* req = new ResetFilterValuesRequest;
            request->set_allocated_resetfiltervaluesrequest(req);

            int stackIdx = parse_argument<int>(line);

            req->set_stackidx(stackIdx);

            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("reset-all-controls") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            ResetAllFilterValuesRequest* req = new ResetAllFilterValuesRequest;
            request->set_allocated_resetallfiltervaluesrequest(req);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("set-control") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            SetFilterAttributeRequest* req = new SetFilterAttributeRequest;
            request->set_allocated_setfilterattributerequest(req);

            int stackIdx, controlIdx;
            float value;
            parse_arguments<int, int, float>(line, stackIdx, controlIdx, value);

            req->set_filterid("");	// TODO: lwrently this field is omitted
            req->set_controlid(controlIdx);
            req->set_stackidx(stackIdx);
            req->add_floatvalue(value);

            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("set-control0") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            SetFilterAttributeRequest* req = new SetFilterAttributeRequest;
            request->set_allocated_setfilterattributerequest(req);
            req->set_filterid("custom.yaml");
            req->set_controlid(0);
            req->set_stackidx(0);
            req->add_floatvalue(parse_argument<float>(line));
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("set-control1") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            SetFilterAttributeRequest* req = new SetFilterAttributeRequest;
            request->set_allocated_setfilterattributerequest(req);
            req->set_filterid("custom.yaml");
            req->set_controlid(1);
            req->set_stackidx(0);
            req->add_floatvalue(parse_argument<float>(line));
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line == "featureset")
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            GetFeatureSetRequest* getFeatureSetRequest = new GetFeatureSetRequest;
            request->set_allocated_getfeaturesetrequest(getFeatureSetRequest);
            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line == "is-available")
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            IsAnselAvailableRequest* isAnselAvailableRequest = new IsAnselAvailableRequest;
            request->set_allocated_isanselavailablerequest(isAnselAvailableRequest);

            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("keydown") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            InputEventRequest* req = new InputEventRequest;
            request->set_allocated_inputeventrequest(req);

            std::string keyCode;
            keyCode = parse_argument<std::string>(line);

            int vKey = stringToVKey(keyCode);

            req->set_wparam(vKey);
            req->set_lparam(0);
            req->set_message(WM_KEYDOWN);

            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("keyup") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            InputEventRequest* req = new InputEventRequest;
            request->set_allocated_inputeventrequest(req);

            std::string keyCode;
            keyCode = parse_argument<std::string>(line);

            int vKey = stringToVKey(keyCode);

            req->set_wparam(vKey);
            req->set_lparam(0);
            req->set_message(WM_KEYUP);

            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("gamepad") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;
            InputEventRequest* req = new InputEventRequest;
            request->set_allocated_inputeventrequest(req);

            uint32_t axis = 0u;
            float value = 0.0f;
            parse_arguments<uint32_t, float>(line, axis, value);

            req->set_wparam(0);
            req->set_lparam(0);
            req->set_message(0);

            if (axis == 0u)
                req->set_leftstickxvalue(value);
            else if (axis == 1u)
                req->set_leftstickyvalue(value);
            else if (axis == 2u)
                req->set_rightstickxvalue(value);
            else if (axis == 3u)
                req->set_rightstickyvalue(value);
            else if (axis == 4u)
                req->set_lefttriggervalue(value);
            else if (axis == 5u)
                req->set_righttriggervalue(value);

            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("mouse-move") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;

            int dx = 0, dy = 0;
            unsigned long keystate = 0;
            parse_arguments<int, int>(line, dx, dy);

            InputEventRequest* req = new InputEventRequest;
            request->set_allocated_inputeventrequest(req);

            int lParam = (dx & 0xffff) + ((dy & 0xffff) << 16);

            req->set_isdeltacoords(true);
            req->set_wparam(0);
            req->set_lparam(lParam);
            req->set_message(WM_MOUSEMOVE);

            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("mouse-lmb") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;

            int state = 0;
            state = parse_argument<int>(line);

            InputEventRequest* req = new InputEventRequest;
            request->set_allocated_inputeventrequest(req);

            req->set_isdeltacoords(true);
            req->set_wparam(0);
            req->set_lparam(0);
            req->set_message(state ? WM_LBUTTONDOWN : WM_LBUTTONUP);

            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("set-lang") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;

            int lang = 0, sublang = 0;
            parse_arguments<int, int>(line, lang, sublang);

            SetLangIdRequest* req = new SetLangIdRequest;
            request->set_allocated_setlangidrequest(req);

            req->set_lang(lang);
            req->set_sublang(sublang);

            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else if (line.find("set-highquality") != std::string::npos)
        {
            AnselIPCMessage message;
            AnselIPCRequest* request = new AnselIPCRequest;

            bool setting = parse_argument<bool>(line);

            SetHighQualityRequest* req = new SetHighQualityRequest;
            request->set_allocated_sethighqualityrequest(req);

            req->set_setting(setting);

            message.set_allocated_request(request);
            GenericBusMessage msg("Ansel", "LwCameraControl", message.SerializeAsString());
            messageBus.postMessage(msg);
        }
        else
        {
            GenericBusMessage msg("Ansel", "LwCameraControl", line);
            messageBus.postMessage(msg);
        }
    }

    messageBus.removeObserver(&s_observer);

    return 0;
}