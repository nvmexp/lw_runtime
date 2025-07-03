/*
 * Copyright (c) 2014-2017, LWPU CORPORATION. All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "socket.h"

#define STREAM_SOCK_PATH    "/tmp/egltest_socket"

const char *socket_path = STREAM_SOCK_PATH;

// ======================== IP server ======================================
IPServer::IPServer()
{
}

bool IPServer::createSocket()
{
    _sockID = socket(AF_INET , SOCK_STREAM , 0);

    if (-1 == _sockID) {
        LOG_ERR("IPServer could not create socket, errno = %d\n", errno);
        return false;
    }

    LOG_INFO("IPServer socket created\n");
    return true;
}

bool IPServer::bind()
{
    _server.sin_family      = AF_INET;
    _server.sin_addr.s_addr = INADDR_ANY;
    _server.sin_port        = htons( 8888 );

    if ( ::bind(_sockID, (struct sockaddr *)&_server, sizeof(_server)) < 0) {
        LOG_INFO("IPServer bind failed, errno = %d\n", errno);
        return false;
    }
    LOG_INFO("IPServer bind done\n");
    return true;
}

bool IPServer::listen()
{
    if (::listen(_sockID , 3) < 0) {
        LOG_ERR("IPServer list failed, errno = %d\n", errno);
        return false;
    }
    return true;
}

SocketID IPServer::accept()
{
    struct sockaddr_in client;
    int c = sizeof(struct sockaddr_in);

    _clientSockID = ::accept(_sockID, (struct sockaddr *)&client, (socklen_t*)&c);
    if (_clientSockID < 0) {
        LOG_ERR("IPServer accept failed, errno = %d\n", errno);
        return -1;
    }
    LOG_INFO("IPServer connection accepted\n");

    return _clientSockID;
}

bool IPServer::receive()
{
    LOG_INFO("IPServer listening...\n");
    char msg[100];

    int readSize = recv(_clientSockID , msg , 100 , 0);
    if (readSize > 0) {
        msg[readSize] = 0;
        LOG_INFO("IPServer received message: %s\n", msg);
    } else {
        LOG_ERR("IPServer didn't receive message, errno = %d\n", errno);
        return false;
    }
    return true;
}

SocketID IPServer::getSockID() const
{
    return _sockID;
}

SocketID IPServer::getClientSockID() const
{
    return _clientSockID;
}
// ======================== IP server ======================================

// ======================== IP client ======================================
IPClient::IPClient(const char * ip)
{
	strncpy(_ip_addr, ip, sizeof(_ip_addr));
}

bool IPClient::createSocket()
{
    _sockID = socket(AF_INET , SOCK_STREAM , 0);

    if (-1 == _sockID) {
        LOG_ERR("IPClient could not create socket, errno =%d\n", errno);
        return false;
    }

    LOG_INFO("IPClient socket created\n");
    return true;
}

bool IPClient::connect()
{
    struct sockaddr_in server;

    server.sin_addr.s_addr = inet_addr(_ip_addr);
    server.sin_family      = AF_INET;
    server.sin_port        = htons( 8888 );

    if (::connect(_sockID , (struct sockaddr *)&server , sizeof(server)) < 0) {
        LOG_INFO("IPClient connect failed, errno = %d\n", errno);
        return false;
    }
    LOG_INFO("IPClient now connected to server\n");
    return true;
}

bool IPClient::send(const char* data)
{
    if ( ::send(_sockID , data , strlen(data) , 0) < 0) {
        LOG_ERR("IPClient send failed, errno = %d\n", errno);
        return false;
    }
    LOG_INFO("IPClient send success!\n");
    return true;
}

SocketID IPClient::getSockID() const
{
    return _sockID;
}
// ======================== IP client ======================================

// ======================== UNIX server ======================================
UnixServer::UnixServer()
{
}

bool UnixServer::createSocket()
{
    _sockID = socket(AF_UNIX , SOCK_STREAM , 0);

    if (-1 == _sockID) {
        LOG_ERR("UnixServer could not create socket, errno = %d\n", errno);
        return false;
    }

    LOG_INFO("UnixServer socket created\n");
    return true;
}

bool UnixServer::bind()
{
    _server.sun_family = AF_UNIX;
    strncpy(_server.sun_path, socket_path, sizeof(_server.sun_path)-1);

    unlink(socket_path);

    if ( ::bind(_sockID, (struct sockaddr *)&_server, sizeof(_server)) < 0) {
        LOG_INFO("UnixServer bind failed, errno = %d\n", errno);
        return false;
    }
    LOG_INFO("UnixServer bind done\n");
    return true;
}

bool UnixServer::listen()
{
    if (::listen(_sockID , 3) < 0) {
        LOG_ERR("UnixServer list failed, errno = %d\n", errno);
        return false;
    }
    return true;
}

SocketID UnixServer::accept()
{
    // Not sure if the accept is fine.
    _clientSockID = ::accept(_sockID, NULL, NULL);
    if (_clientSockID < 0) {
        LOG_ERR("UnixServer accept failed, errno = %d\n", errno);
        return -1;
    }
    LOG_INFO("UnixServer connection accepted\n");

    return _clientSockID;
}

bool UnixServer::receive()
{
    LOG_INFO("UnixServer listening...\n");
    char msg[100];

    int readSize = read(_clientSockID , msg , 100);
    if (readSize > 0) {
        msg[readSize] = 0;
        LOG_INFO("UnixServer received message: %s\n", msg);
    } else {
        LOG_ERR("UnixServer didn't receive message, errno = %d\n", errno);
        return false;
    }
    return true;
}

SocketID UnixServer::getSockID() const
{
    return _sockID;
}

SocketID UnixServer::getClientSockID() const
{
    return _clientSockID;
}
// ======================== UNIX server ======================================

// ======================== UNIX client ======================================
UnixClient::UnixClient()
{
}

bool UnixClient::createSocket()
{
    _sockID = socket(AF_UNIX , SOCK_STREAM , 0);

    if (-1 == _sockID) {
        LOG_ERR("UnixClient could not create socket, errno = %d\n", errno);
        return false;
    }

    LOG_INFO("UnixClient socket created\n");
    return true;
}

bool UnixClient::connect()
{
    struct sockaddr_un server;

    memset(&server, 0, sizeof(server));
    server.sun_family = AF_UNIX;
    strncpy(server.sun_path, socket_path, sizeof(server.sun_path)-1);

    if (::connect(_sockID , (struct sockaddr *)&server , sizeof(server)) < 0) {
        LOG_INFO("UnixClient connect failed, errno = %d\n", errno);
        return false;
    }
    LOG_INFO("UnixClient now connected to server\n");
    return true;
}

bool UnixClient::send(const char* data)
{
    if ( ::write(_sockID , data , strlen(data)+1) < 0) {
        LOG_ERR("UnixClient send failed, errno = %d\n", errno);
        return false;
    }
    LOG_INFO("UnixClient send success!\n");
    return true;
}

SocketID UnixClient::getSockID() const
{
    return _sockID;
}
// ======================== UNIX client ======================================
