#include <Winsock2.h>
#include <Ws2ipdef.h>
#include <MinHook.h>
#include "Log.h"
#include "NetworkDetection.h"

#pragma comment( lib, "Ws2_32.lib" )

#define HOOK_EXPORT extern "C"

#if ENABLE_NETWORK_HOOKS == 1
bool isLocalSocket(SOCKET s)
{
    sockaddr_storage saddrIn = { 0 }, saddrOut = { 0 };
    int saddrLen = sizeof(saddrIn);
    // we check addresses of both ends of the socket
    getsockname(s, (sockaddr*)&saddrIn, &saddrLen);
    getpeername(s, (sockaddr*)&saddrOut, &saddrLen);

    // ipv4
    if (saddrIn.ss_family == AF_INET)
    {
        sockaddr_in* saddrInIpv4 = reinterpret_cast<sockaddr_in*>(&saddrIn);
        sockaddr_in* saddrOutpv4 = reinterpret_cast<sockaddr_in*>(&saddrOut);
        return saddrInIpv4->sin_addr.S_un.S_addr == saddrOutpv4->sin_addr.S_un.S_addr;
    }
    // ipv6
    else if (saddrIn.ss_family == AF_INET6)
    {
        SOCKADDR_IN6* saddrInIpv6 = reinterpret_cast<SOCKADDR_IN6*>(&saddrIn);
        SOCKADDR_IN6* saddrOutIpv6 = reinterpret_cast<SOCKADDR_IN6*>(&saddrOut);
        uint32_t* in = reinterpret_cast<uint32_t*>(&saddrInIpv6->sin6_addr.u.Byte[0]);
        uint32_t* out = reinterpret_cast<uint32_t*>(&saddrOutIpv6->sin6_addr.u.Byte[0]);
        // now we compare 16 byte addresses
        return in[0] == out[0] && 
            in[1] == out[1] && 
            in[2] == out[2] && 
            in[3] == out[3];
    }

    // if socket is neither ipv4, not ipv6, we treat it as local
    return true;
}


bool NetworkActivityDetector::m_initialized = false;
bool NetworkActivityDetector::m_deinitialized = false;
std::atomic<uint64_t> NetworkActivityDetector::m_bytesTransferred = 0u;
std::mutex NetworkActivityDetector::m_initializationLock;
std::mutex NetworkActivityDetector::m_deinitializationLock;


// hooks are declared first
// hooks are declared first
int (WSAAPI *WSASendTrampoline)(SOCKET s, LPWSABUF lpBuffers, DWORD dwBufferCount, LPDWORD lpNumberOfBytesSent, DWORD dwFlags, LPWSAOVERLAPPED lpOverlapped, LPWSAOVERLAPPED_COMPLETION_ROUTINE lpCompletionRoutine) = nullptr;
int (WSAAPI *WSASendToTrampoline)(SOCKET s, LPWSABUF lpBuffers, DWORD dwBufferCount, LPDWORD lpNumberOfBytesSent, DWORD dwFlags, const struct sockaddr *lpTo, int iToLen, LPWSAOVERLAPPED lpOverlapped, LPWSAOVERLAPPED_COMPLETION_ROUTINE lpCompletionRoutine) = nullptr;
int (WSAAPI *WSARecvTrampoline)(SOCKET s, LPWSABUF lpBuffers, DWORD dwBufferCount, LPDWORD lpNumberOfBytesRecvd, LPDWORD lpFlags, LPWSAOVERLAPPED lpOverlapped, LPWSAOVERLAPPED_COMPLETION_ROUTINE lpCompletionRoutine) = nullptr;
int (WSAAPI *WSARecvExTrampoline)(SOCKET s, char *buf, int len, int *flags) = nullptr;
int (WSAAPI *WSARecvFromTrampoline)(SOCKET s, LPWSABUF lpBuffers, DWORD dwBufferCount, LPDWORD lpNumberOfBytesRecvd, LPDWORD lpFlags, struct sockaddr *lpFrom, LPINT lpFromlen, LPWSAOVERLAPPED lpOverlapped, LPWSAOVERLAPPED_COMPLETION_ROUTINE lpCompletionRoutine) = nullptr;
int (WINAPI *SendTrampoline)(SOCKET socket, const char* buffer, int length, int flags) = nullptr;
int (WSAAPI *SendToTrampoline)(SOCKET s, const char *buf, int len, int flags, const struct sockaddr *to, int tolen) = nullptr;
int (WSAAPI *RecvTrampoline)(SOCKET s, char *buf, int len, int flags) = nullptr;
int (WSAAPI *RecvFromTrampoline)(SOCKET s, char *buf, int len, int flags, struct sockaddr *from, int *fromlen) = nullptr;

HOOK_EXPORT int WSAAPI HookWSASendNoLocalCheck(SOCKET s, LPWSABUF lpBuffers, DWORD dwBufferCount, LPDWORD lpNumberOfBytesSent, DWORD dwFlags, LPWSAOVERLAPPED lpOverlapped, LPWSAOVERLAPPED_COMPLETION_ROUTINE lpCompletionRoutine)
{
    for (DWORD i = 0; i < dwBufferCount; ++i)
        NetworkActivityDetector::m_bytesTransferred += lpBuffers[i].len;

    return WSASendTrampoline(s, lpBuffers, dwBufferCount, lpNumberOfBytesSent, dwFlags, lpOverlapped, lpCompletionRoutine);
}

HOOK_EXPORT int WSAAPI HookWSASendToNoLocalCheck(SOCKET s, LPWSABUF lpBuffers, DWORD dwBufferCount, LPDWORD lpNumberOfBytesSent, DWORD dwFlags, const struct sockaddr *lpTo, int iToLen, LPWSAOVERLAPPED lpOverlapped, LPWSAOVERLAPPED_COMPLETION_ROUTINE lpCompletionRoutine)
{
    for (DWORD i = 0; i < dwBufferCount; ++i)
        NetworkActivityDetector::m_bytesTransferred += lpBuffers[i].len;

    return WSASendToTrampoline(s, lpBuffers, dwBufferCount, lpNumberOfBytesSent, dwFlags, lpTo, iToLen, lpOverlapped, lpCompletionRoutine);
}

HOOK_EXPORT int WSAAPI HookWSARecvNoLocalCheck(SOCKET s, LPWSABUF lpBuffers, DWORD dwBufferCount, LPDWORD lpNumberOfBytesRecvd, LPDWORD lpFlags, LPWSAOVERLAPPED lpOverlapped, LPWSAOVERLAPPED_COMPLETION_ROUTINE lpCompletionRoutine)
{
    const auto status = WSARecvTrampoline(s, lpBuffers, dwBufferCount, lpNumberOfBytesRecvd, lpFlags, lpOverlapped, lpCompletionRoutine);

    if (status == 0 && lpNumberOfBytesRecvd != nullptr)
        NetworkActivityDetector::m_bytesTransferred += *lpNumberOfBytesRecvd;

    return status;
}

HOOK_EXPORT int WSAAPI HookWSARecvExNoLocalCheck(SOCKET s, char *buf, int len, int *flags)
{
    const int received = WSARecvExTrampoline(s, buf, len, flags);

    if (received > 0)
        NetworkActivityDetector::m_bytesTransferred += received;

    return received;
}

HOOK_EXPORT int WSAAPI HookWSARecvFromNoLocalCheck(SOCKET s, LPWSABUF lpBuffers, DWORD dwBufferCount, LPDWORD lpNumberOfBytesRecvd, LPDWORD lpFlags, struct sockaddr *lpFrom, LPINT lpFromlen, LPWSAOVERLAPPED lpOverlapped, LPWSAOVERLAPPED_COMPLETION_ROUTINE lpCompletionRoutine)
{
    const auto status = WSARecvFromTrampoline(s, lpBuffers, dwBufferCount, lpNumberOfBytesRecvd, lpFlags, lpFrom, lpFromlen, lpOverlapped, lpCompletionRoutine);

    if (status == 0 && lpNumberOfBytesRecvd != nullptr)
        NetworkActivityDetector::m_bytesTransferred += *lpNumberOfBytesRecvd;

    return status;
}

HOOK_EXPORT int WSAAPI HookSendNoLocalCheck(SOCKET s, const char *buf, int len, int flags)
{
    const auto num_bytes_send = SendTrampoline(s, buf, len, flags);

    if (num_bytes_send != SOCKET_ERROR)
        NetworkActivityDetector::m_bytesTransferred += num_bytes_send;

    return num_bytes_send;
}

HOOK_EXPORT int WSAAPI HookSendToNoLocalCheck(SOCKET s, const char *buf, int len, int flags, const struct sockaddr *to, int tolen)
{
    const auto num_bytes_send = SendToTrampoline(s, buf, len, flags, to, tolen);

    if (num_bytes_send != SOCKET_ERROR)
        NetworkActivityDetector::m_bytesTransferred += num_bytes_send;

    return num_bytes_send;
}

HOOK_EXPORT int WSAAPI HookRecvNoLocalCheck(SOCKET s, char *buf, int len, int flags)
{
    const auto num_bytes_received = RecvTrampoline(s, buf, len, flags);

    if (num_bytes_received != SOCKET_ERROR)
        NetworkActivityDetector::m_bytesTransferred += num_bytes_received;

    return num_bytes_received;
}

HOOK_EXPORT int WSAAPI HookRecvFromNoLocalCheck(SOCKET s, char *buf, int len, int flags, struct sockaddr *from, int *fromlen)
{
    const auto num_bytes_received = RecvFromTrampoline(s, buf, len, flags, from, fromlen);

    if (num_bytes_received != SOCKET_ERROR)
        NetworkActivityDetector::m_bytesTransferred += num_bytes_received;

    return num_bytes_received;
}

HOOK_EXPORT int WSAAPI HookWSASend(SOCKET s, LPWSABUF lpBuffers, DWORD dwBufferCount, LPDWORD lpNumberOfBytesSent, DWORD dwFlags, LPWSAOVERLAPPED lpOverlapped, LPWSAOVERLAPPED_COMPLETION_ROUTINE lpCompletionRoutine)
{
    LOG_DEBUG("HookWSASend Called...");
    if (!isLocalSocket(s))
        for (DWORD i = 0; i < dwBufferCount; ++i)
            NetworkActivityDetector::m_bytesTransferred += lpBuffers[i].len;

    return WSASendTrampoline(s, lpBuffers, dwBufferCount, lpNumberOfBytesSent, dwFlags, lpOverlapped, lpCompletionRoutine);
}

HOOK_EXPORT int WSAAPI HookWSASendTo(SOCKET s, LPWSABUF lpBuffers, DWORD dwBufferCount, LPDWORD lpNumberOfBytesSent, DWORD dwFlags, const struct sockaddr *lpTo, int iToLen, LPWSAOVERLAPPED lpOverlapped, LPWSAOVERLAPPED_COMPLETION_ROUTINE lpCompletionRoutine)
{
    LOG_DEBUG("HookWSASendTo Called...");
    if (!isLocalSocket(s))
        for (DWORD i = 0; i < dwBufferCount; ++i)
            NetworkActivityDetector::m_bytesTransferred += lpBuffers[i].len;

    return WSASendToTrampoline(s, lpBuffers, dwBufferCount, lpNumberOfBytesSent, dwFlags, lpTo, iToLen, lpOverlapped, lpCompletionRoutine);
}

HOOK_EXPORT int WSAAPI HookWSARecv(SOCKET s, LPWSABUF lpBuffers, DWORD dwBufferCount, LPDWORD lpNumberOfBytesRecvd, LPDWORD lpFlags, LPWSAOVERLAPPED lpOverlapped, LPWSAOVERLAPPED_COMPLETION_ROUTINE lpCompletionRoutine)
{
    LOG_DEBUG("HookWSARecv Called...");
    const auto status = WSARecvTrampoline(s, lpBuffers, dwBufferCount, lpNumberOfBytesRecvd, lpFlags, lpOverlapped, lpCompletionRoutine);

    if (status == 0 && lpNumberOfBytesRecvd != nullptr)
        if (!isLocalSocket(s))
            NetworkActivityDetector::m_bytesTransferred += *lpNumberOfBytesRecvd;

    return status;
}

HOOK_EXPORT int WSAAPI HookWSARecvEx(SOCKET s, char *buf, int len, int *flags)
{
    LOG_DEBUG("HookWSARecvEx Called...");
    const int received = WSARecvExTrampoline(s, buf, len, flags);

    if (received > 0)
        if (!isLocalSocket(s))
            NetworkActivityDetector::m_bytesTransferred += received;

    return received;
}

HOOK_EXPORT int WSAAPI HookWSARecvFrom(SOCKET s, LPWSABUF lpBuffers, DWORD dwBufferCount, LPDWORD lpNumberOfBytesRecvd, LPDWORD lpFlags, struct sockaddr *lpFrom, LPINT lpFromlen, LPWSAOVERLAPPED lpOverlapped, LPWSAOVERLAPPED_COMPLETION_ROUTINE lpCompletionRoutine)
{
    LOG_DEBUG("HookWSARecvFrom Called...");
    const auto status = WSARecvFromTrampoline(s, lpBuffers, dwBufferCount, lpNumberOfBytesRecvd, lpFlags, lpFrom, lpFromlen, lpOverlapped, lpCompletionRoutine);

    if (status == 0 && lpNumberOfBytesRecvd != nullptr)
        if (!isLocalSocket(s))
            NetworkActivityDetector::m_bytesTransferred += *lpNumberOfBytesRecvd;

    return status;
}

HOOK_EXPORT int WSAAPI HookSend(SOCKET s, const char *buf, int len, int flags)
{
    LOG_DEBUG("HookSend Called...");
    const auto num_bytes_send = SendTrampoline(s, buf, len, flags);

    if (num_bytes_send != SOCKET_ERROR)
        if (!isLocalSocket(s))
            NetworkActivityDetector::m_bytesTransferred += num_bytes_send;

    return num_bytes_send;
}

HOOK_EXPORT int WSAAPI HookSendTo(SOCKET s, const char *buf, int len, int flags, const struct sockaddr *to, int tolen)
{
    LOG_DEBUG("HookSendTo Called...");
    const auto num_bytes_send = SendToTrampoline(s, buf, len, flags, to, tolen);

    if (num_bytes_send != SOCKET_ERROR)
        if (!isLocalSocket(s))
            NetworkActivityDetector::m_bytesTransferred += num_bytes_send;

    return num_bytes_send;
}

HOOK_EXPORT int WSAAPI HookRecv(SOCKET s, char *buf, int len, int flags)
{
    LOG_DEBUG("HookRecv Called...");
    const auto num_bytes_received = RecvTrampoline(s, buf, len, flags);

    if (num_bytes_received != SOCKET_ERROR)
        if (!isLocalSocket(s))
            NetworkActivityDetector::m_bytesTransferred += num_bytes_received;

    return num_bytes_received;
}

HOOK_EXPORT int WSAAPI HookRecvFrom(SOCKET s, char *buf, int len, int flags, struct sockaddr *from, int *fromlen)
{
    LOG_DEBUG("HookRecvFrom Called...");
    const auto num_bytes_received = RecvFromTrampoline(s, buf, len, flags, from, fromlen);

    if (num_bytes_received != SOCKET_ERROR)
        if (!isLocalSocket(s))
            NetworkActivityDetector::m_bytesTransferred += num_bytes_received;

    return num_bytes_received;
}

// hooks end here

// Network detector implementation begins here

bool NetworkActivityDetector::isActivityDetected() const { return m_active; }
uint64_t NetworkActivityDetector::bytesTransferred() const { return m_bytesTransferred; }

void NetworkActivityDetector::setEnabled(bool state)
{
    m_enabled = state;

    if (m_enabled)
        MH_EnableHook(MH_ALL_HOOKS);
    else
        MH_DisableHook(MH_ALL_HOOKS);
}

void NetworkActivityDetector::resetActivity()
{
    m_active = false;
    m_bytesTransferred = 0u;
}

void NetworkActivityDetector::initialize(bool checkIfSocketIsLocal)
{
    if (m_initialized)
        return;

    std::lock_guard<std::mutex> lock(m_initializationLock);
    if (!m_initialized)
    {
        MH_Initialize();

        const auto installHook = [](LPCWSTR module, LPCSTR function, LPVOID detour, LPVOID* original)
        {
            if (MH_CreateHookApi(module, function, detour, original) == MH_OK)
            {
                LOG_INFO("%s hooked successfully", function);
            }
            else
            {
                LOG_DEBUG("%s hook failed", function);
            }
        };

        if (checkIfSocketIsLocal)
        {
            installHook(L"Ws2_32", "send", HookSend, (LPVOID *)&SendTrampoline);
            installHook(L"Ws2_32", "sendto", HookSendTo, (LPVOID *)&SendToTrampoline);
            installHook(L"Ws2_32", "recv", HookRecv, (LPVOID *)&RecvTrampoline);
            installHook(L"Ws2_32", "recvfrom", HookRecvFrom, (LPVOID *)&RecvFromTrampoline);
            installHook(L"Ws2_32", "WSASend", HookWSASend, (LPVOID *)&WSASendTrampoline);
            installHook(L"Ws2_32", "WSASendTo", HookWSASendTo, (LPVOID *)&WSASendToTrampoline);
            installHook(L"Ws2_32", "WSARecv", HookWSARecv, (LPVOID *)&WSARecvTrampoline);
            installHook(L"Mswsock", "WSARecvEx", HookWSARecvEx, (LPVOID *)&WSARecvExTrampoline);
            installHook(L"Ws2_32", "WSARecvFrom", HookWSARecvFrom, (LPVOID *)&WSARecvFromTrampoline);
        }
        else
        {
            installHook(L"Ws2_32", "send", HookSendNoLocalCheck, (LPVOID *)&SendTrampoline);
            installHook(L"Ws2_32", "sendto", HookSendToNoLocalCheck, (LPVOID *)&SendToTrampoline);
            installHook(L"Ws2_32", "recv", HookRecvNoLocalCheck, (LPVOID *)&RecvTrampoline);
            installHook(L"Ws2_32", "recvfrom", HookRecvFromNoLocalCheck, (LPVOID *)&RecvFromTrampoline);
            installHook(L"Ws2_32", "WSASend", HookWSASendNoLocalCheck, (LPVOID *)&WSASendTrampoline);
            installHook(L"Ws2_32", "WSASendTo", HookWSASendToNoLocalCheck, (LPVOID *)&WSASendToTrampoline);
            installHook(L"Ws2_32", "WSARecv", HookWSARecvNoLocalCheck, (LPVOID *)&WSARecvTrampoline);
            installHook(L"Mswsock", "WSARecvEx", HookWSARecvExNoLocalCheck, (LPVOID *)&WSARecvExTrampoline);
            installHook(L"Ws2_32", "WSARecvFrom", HookWSARecvFromNoLocalCheck, (LPVOID *)&WSARecvFromTrampoline);
        }

        m_initialized = true;
        MH_EnableHook(MH_ALL_HOOKS);
        
        // Since we just created more hooks, we have to make sure they ultimately get deinitialized, even if deinitialize has already been called in the past.
        m_deinitialized = false;
    }
}

void NetworkActivityDetector::deinitialize()
{
    std::lock_guard<std::mutex> lock(m_deinitializationLock);
    LOG_DEBUG("Checking to disable all network hooks...");
    if (!m_deinitialized)
    {
        LOG_INFO("Disabling all network hooks...");
        MH_DisableHook(MH_ALL_HOOKS);
        MH_Uninitialize();

        m_deinitialized = true;
    }
}

void NetworkActivityDetector::tick()
{
    const bool ENABLE_FANCY_LOGIC = false;
    if (ENABLE_FANCY_LOGIC)
    {
        if (m_cooldown-- > 0)
        {
            m_traffic += m_bytesTransferred > 0;
            return;
        }
        else
        {
            m_cooldown = NETWORK_DETECTION_WINDOW_FRAMES;
            m_bytesTransferred = 0;

            if (m_traffic > NETWORK_DETECTION_FRAMES_WITH_TRAFFIC_THRESHOLD)
            {
                m_traffic = 0;
                m_active = true;
                return;
            }
            m_traffic = 0;
        }

        // if we're here, this means no (or not enough) network traffic was detected
        m_active = false;
    }
    else
    {
        // this logic is more strict:
        // network activity is detected if at least one byte was transferred (sent or received)
        // activity indicator wont be reset (there is no cooldown period), unless resetActivity is called
        m_active = m_bytesTransferred > 0;
    }
}

NetworkActivityDetector::~NetworkActivityDetector()
{
    deinitialize();
}
#endif
