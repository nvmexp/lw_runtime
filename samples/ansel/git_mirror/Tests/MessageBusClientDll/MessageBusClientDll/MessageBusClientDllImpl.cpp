#define MESSAGEBUS_CLIENT_DLL_EXPORTS
#include "MessageBusClientDllInterface.h"

#include <memory>

#define POCO_NO_AUTOMATIC_LIBS
#pragma warning(push)
#pragma warning(disable:4267 4244 4996)
#include "GenericBusMessage.h"
#include "MessageBus.h"
#include "MessageBusSelwre.h"
#pragma warning(pop)

#pragma comment(lib, "MessageBus.lib")
#pragma comment(lib, "delayimp")

namespace
{
	struct AnselMessageBusObserver;

	std::unique_ptr<MessageBus> g_messageBus;
	std::unique_ptr<AnselMessageBusObserver> g_messageBusObserver;
	

	struct AnselMessageBusObserver : public BusObserver
	{
		MessageBus& m_messageBus;
		PFNONMESSAGEBUSCALLBACK onBusMessageCallback = nullptr;

		AnselMessageBusObserver(MessageBus& messageBus, PFNONMESSAGEBUSCALLBACK clbk) : 
			m_messageBus(messageBus), onBusMessageCallback(clbk) {}

		void onBusMessage(const BusMessage& msg)
		{
			const auto message = msg.SerializeAsString();
			if (onBusMessageCallback)
				onBusMessageCallback(message.data(), uint32_t(message.size()));
		}
	};
}

MESSAGEBUS_CLIENT_DLL_API bool __cdecl joinMessageBus(const char* system, const char* module, PFNONMESSAGEBUSCALLBACK clbk)
{
	if (!clbk)
		return false;

	const auto messageBusInterface = getMessageBusInterfaceSelwre(MESSAGE_BUS_INTERFACE_VERSION);

	if (!messageBusInterface)
		return false;

	g_messageBus = std::make_unique<MessageBus>(messageBusInterface);
	if (!g_messageBus)
		return false;

	g_messageBusObserver = std::make_unique<AnselMessageBusObserver>(*g_messageBus, clbk);

	if (!g_messageBusObserver)
		return false;

	g_messageBus->addObserver(g_messageBusObserver.get(), system, module);

	return true;
}

MESSAGEBUS_CLIENT_DLL_API bool __cdecl postMessage(const char* msg, uint32_t size)
{
	if (g_messageBus)
		return g_messageBus->postMessage(msg, size);
	return false;
}

MESSAGEBUS_CLIENT_DLL_API void __cdecl leaveMessageBus()
{
	g_messageBus.reset(nullptr);
}