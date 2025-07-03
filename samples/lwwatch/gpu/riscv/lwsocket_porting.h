/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#ifndef _LWSOCKET_PORTING_H_
#define _LWSOCKET_PORTING_H_
#include "lwwatch.h"
#include "hal.h"

#if LWWATCHCFG_IS_PLATFORM(WINDOWS)
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <errno.h>

typedef SOCKET LwInternalSocket;
#define SOCKET_FN_CLOSE closesocket
#define SOCKET_FN_IOCTL ioctlsocket
#define SOCKET_FN_ERRNO WSAGetLastError()
#define SOCKET_FN_POLL  WSAPoll
#define SOCKET_NO_SIGPIPE 0

#elif LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <poll.h>

typedef int LwInternalSocket;
#define SOCKET_ERROR (-1)
#define ILWALID_SOCKET (-1)

#define SOCKET_FN_CLOSE close
#define SOCKET_FN_IOCTL ioctl
#define SOCKET_FN_POLL  poll

#define SOCKET_FN_ERRNO errno
#define SOCKET_NO_SIGPIPE MSG_NOSIGNAL
#else
#error This platform is not supported by LwSocket.
#endif

#endif
