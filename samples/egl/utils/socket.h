/*
 * Copyright (c) 2014-2017, LWPU CORPORATION. All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

/**
 * Interface class for creating sockets and exchanging FDs
 * This interface class can then be used to create Unix Sockets
 * or Internet sockets.
 */

#ifndef __SOCKET__
#define __SOCKET__

#include <sys/socket.h>
#include <sys/un.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <errno.h>
#include "defs.h"

typedef int SocketID;

class IServer {
public:
    /** Create a socket **/
    virtual bool  createSocket() = 0;

    /** Binds the socket to a particular address. **/
    virtual bool bind() = 0;

    /** Listen to the socket for connections. **/
    virtual bool listen() = 0;

    /** Accept connections. **/
    virtual SocketID accept() = 0;

    /** Receive messages from the socket. **/
    virtual bool receive() = 0;

    /** Get Socket ID **/
    virtual SocketID getSockID() const = 0;

    /** Get Client Socket ID **/
    virtual SocketID getClientSockID() const = 0;

    virtual ~IServer() {}

protected:
    SocketID  _sockID;
    SocketID  _clientSockID;//Can be a vector!
};

class IClient {
public:
    /** Create a socket **/
    virtual bool createSocket() = 0;

    /** Binds the socket to a particular address. **/
    virtual bool connect() = 0;

    /** Listen to the socket  for connections. **/
    virtual bool send(const char* data) = 0;

    /** Get Socket ID **/
    virtual SocketID getSockID() const = 0;

    virtual ~IClient() {}

protected:
    SocketID  _sockID;
};

/*
 *  IPServer can talk through sockets using the
 *  TCP/IP Protocol. We use this class on cross
 *  partition mode, IVC sockets are exposed as TCP/sockets.
 */
class IPServer : public IServer {
public:
    IPServer();

    virtual bool createSocket();

    virtual bool bind();

    virtual bool listen();

    virtual SocketID accept();

    virtual bool receive();

    virtual SocketID getSockID() const;

    virtual SocketID getClientSockID() const;

    virtual ~IPServer() {}

private:
    struct sockaddr_in _server;
};

class UnixServer : public IServer {
public:
    UnixServer();

    virtual bool createSocket();

    virtual bool bind();

    virtual bool listen();

    virtual SocketID accept();

    virtual bool receive();

    virtual SocketID getSockID() const;

    virtual SocketID getClientSockID() const;

    virtual ~UnixServer() {}
private:
    sockaddr_un _server;
};

/*
 *  IPServer can talk through sockets using the
 *  TCP/IP Protocol. We use this class on cross
 *  partition mode, IVC sockets are exposed as TCP/sockets.
 */
class IPClient : public IClient {
public:
    IPClient(const char *ip);

    virtual bool createSocket();

    virtual bool connect();

    virtual bool send(const char* data);

    SocketID getSockID() const;

    virtual ~IPClient() {}

protected:
    char _ip_addr[20];
};

class UnixClient : public IClient {
public:
    UnixClient();

    virtual bool createSocket();

    virtual bool connect();

    virtual bool send(const char* data);

    SocketID getSockID() const;

    virtual ~UnixClient() {}

protected:
};

#endif // __SOCKET__
