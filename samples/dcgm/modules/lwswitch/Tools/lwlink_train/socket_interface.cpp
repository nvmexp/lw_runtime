
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>


#include "socket_interface.h"

SocketBase::SocketBase()
{
    socketFd = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    
    if (socketFd == -1) {
        std::cout << "SocketBase - socket creation failed \n";
    }
}

SocketBase::~SocketBase()
{
    Close();
}


int
SocketBase::ReadBytes(char * buff, int bufLen)
{
    int bytesRead = 0;
    int readResult;
    while (bytesRead < bufLen)
    {
        readResult = read(socketFd, buff + bytesRead, bufLen - bytesRead);
        if (readResult < 1 ) {
            if(errno == ENOENT)
                std::cerr << "Connection closed\n" ;
            else
                std::cerr << "SocketBase - Error while reading from socket errno=" << errno << " error="<< strerror(errno)  <<"\n";
            return bytesRead;
        }
        bytesRead += readResult;
    }
    //std::cout << "bytes read =" << bytesRead << "bufLen=" << bufLen << "\n";
    return bytesRead;
}

int
SocketBase::SendBytes(char* buff, int bufLen)
{
    int bytesWritten = 0;
    int writeResult;
    
    while (bytesWritten < bufLen)
    {
        writeResult = write(socketFd, buff + bytesWritten, bufLen - bytesWritten);
        if (writeResult < 1 ) {
            std::cout << "SocketBase - Error while writting to socket \n";
            return writeResult;
        }
        bytesWritten += writeResult;
    }
    //std::cout << "bytes written =" << bytesWritten << "\n";

    return bytesWritten;
}

void
SocketBase::Close()
{
    if (socketFd) {
        ::close(socketFd);
        socketFd = -1;
    }
}

SocketClient::SocketClient()
    :SocketBase()
{
    //std::cout << "SocketClient::SocketClient \n";
    // do nothing
}

SocketClient::~SocketClient()
{
    //std::cout << "SocketClient::~SocketClient \n";
    // do nothing    
}

int
SocketClient::ConnectTo(std::string address, int port)
{
    int res;
    struct sockaddr_in sa;    
    
    memset(&sa, 0, sizeof sa);
    sa.sin_family = AF_INET;
    sa.sin_port = htons(port);
    res = inet_pton(AF_INET, address.c_str(), &sa.sin_addr);
    while (connect(socketFd, (struct sockaddr *)&sa, sizeof sa) == -1) {
        std::cout << "connect failed \n";
        sleep(10);
    }    
    return 0;
}

SocketServer::SocketServer()
    :SocketBase()
{
    //std::cout << "SocketServer::SocketServer \n";
    // do nothing    
}

SocketServer::~SocketServer()
{
    //std::cout << "SocketServer::~SocketServer \n";
    // do nothing    
}

int
SocketServer::BindAndListen(int port)
{
    //std::cout << "SocketServer::BindAndListen \n";

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof sa);
    sa.sin_family = AF_INET;
    sa.sin_port = htons(port);
    sa.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(socketFd, (struct sockaddr *)&sa, sizeof sa) == -1) {
        std::cout << "bind failed \n";
    }

    if (listen(socketFd, 10) == -1) {
        std::cout << "listen failed \n";
    }    
    return 0;
}

SocketClient*
SocketServer::Accept( )
{
    int ConnectFd = accept(socketFd, NULL, NULL);

    if (0 > ConnectFd) {
        std::cout << "accept failed \n";
    }
    SocketClient* client = new SocketClient();
    client->socketFd = ConnectFd;
    return client;
}

