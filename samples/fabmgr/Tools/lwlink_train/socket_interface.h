
#pragma once

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

#include "lwlink_lib_ioctl.h"

typedef unsigned long long uint64;
typedef unsigned int       uint32;
typedef unsigned short     uint16;
typedef unsigned char      uint8;

#define CMD_START_READ_DISCOVERY_TOKEN   50
#define CMD_START_OFF_TO_SAFE            51
#define CMD_START_SAFE_TO_HS             52
#define CMD_START_HS_TO_SAFE             53
#define CMD_START_SAFE_TO_OFF            54
#define CMD_START_INTERNODE_CONN_ADD     55
#define CMD_START_READ_SID               56

#define CMD_RESP_READ_DISCOVERY_TOKEN       100
#define CMD_RESP_OFF_TO_SAFE_SUBLINK_DONE   101
#define CMD_RESP_SAFE_TO_HS_SUBLINK_DONE    102
#define CMD_RESP_HS_TO_SAFE_SUBLINK_DONE    103
#define CMD_RESP_SAFE_TO_OFF_SUBLINK_DONE   104
#define CMD_RESP_READ_SID                   105

#define MAX_READ_DISCOVERY_TOKEN 256
#define MAX_DEVICES 512

typedef struct
{
    LwU16 domain;
    LwU8  bus;
    LwU8  device;
    LwU8  function;
    LwU16 nodeId;
    LwU32 linkIndex;
    LwU64 tokelwalue;
    LwU32 phyId;
    LwU32 devType;
} DiscoveryTokenInfo;

typedef struct
{
    uint32 count;
    DiscoveryTokenInfo readDiscToken[MAX_READ_DISCOVERY_TOKEN];
} DiscoveryTokenReadInfo;

typedef struct 
{
    lwlink_endpoint srcEndPoint;
    lwlink_endpoint dstEndPoint;
    LwU32 dstDevType;
    LwU32 nearDevType;
} InterNodeConnectionInfo;

typedef struct
{
    uint32 count;
    InterNodeConnectionInfo connInfo[MAX_READ_DISCOVERY_TOKEN];
} InterNodeConnectionInfoAddMsg;

typedef struct 
{
    uint32 nodeId;
    uint64 gpuOrSwitchId;
    uint64 nearSid;
    uint32 nearLinkIndex;
    uint64 farSid;
    uint32 farLinkIndex;
    LwU16 domain;
    LwU8  bus;
    LwU8  device;
    LwU8  function;
    LwU32 phyId;
    LwU32 devType;
} SidNodeConnectionInfo;

typedef struct
{
    uint32 count;
    SidNodeConnectionInfo connInfo[MAX_DEVICES];
} SidNodeConnectionInfoMsg;

typedef struct 
{
    unsigned int msgCmd;
    unsigned int option;
    DiscoveryTokenReadInfo readDiscTokens;
    InterNodeConnectionInfoAddMsg interNodeConns;
    SidNodeConnectionInfoMsg interNodeSidInfo;
}MsgHdr;

typedef struct
{
    int nearDevType;
    int farDevType;
    lwlink_connection_info connInfo;
} fm_lwlink_conn_info;

class SocketBase {

public:

    SocketBase();    
    virtual ~SocketBase();

    int ReadBytes(char* buff, int bufLen);

    int SendBytes(char* buff, int bufLen);
    void   Close();

    int socketFd;
};

class SocketClient : public SocketBase
{
public:
    
    SocketClient();    
    virtual ~SocketClient();

    int ConnectTo(std::string address, int port);
};

class SocketServer : public SocketBase
{
public:
    
    SocketServer();    
    virtual ~SocketServer();

    int BindAndListen(int port);

    SocketClient* Accept( );
};

