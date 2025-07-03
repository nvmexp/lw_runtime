
#pragma once
/*****************************************************************************/
/*  Fabric Manager socket communication and message handling abstractions    */
/*****************************************************************************/
#include "DcgmFMCommon.h"
#include "fabricmanager.pb.h"

enum FabricManagerCommEventType {
    FM_EVENT_PEER_FM_CONNECT = 1,
    FM_EVENT_PEER_FM_DISCONNECT = 2,
    FM_EVENT_GLOBAL_FM_CONNECT = 3,
    FM_EVENT_GLOBAL_FM_DISCONNECT = 4,
};

/*
* Base class abstraction to dispatch individual protobuf messages and FM socket 
* events (connection/disconnection) to individual message handlers.
*/
class FMMessageHandler
{
public:
    FMMessageHandler() {;}

    virtual ~FMMessageHandler() { ; }

    // deliver protobuf messages for processing
    void virtual handleMessage( lwswitch::fmMessage *pFmMessage ) = 0;

    // deliver socket events for processing
    virtual void handleEvent( FabricManagerCommEventType eventType, uint32 nodeId ) = 0;
};

/*
* Base class abstraction for different communication used by GFM and LFM
*/
class DcgmFMConnBase
{
public:
    DcgmFMConnBase() {;}

    virtual ~DcgmFMConnBase() { ; }

    // this method will be called when we receive a response to a request we made
    // on the client connection.
    virtual int ProcessMessage(lwswitch::fmMessage * pFmMessage, bool &isResponse) = 0;
    // this method will be called when we receive a message on client connection without any active request
    // since DCGM is req-resp semantics, this is an attempt to de-couple that req-resp semantics.
    // eventually we should have only one of this process message route.
    virtual void ProcessUnSolicitedMessage(lwswitch::fmMessage * pFmMessage) = 0;
    // socket event handling
    virtual void ProcessConnect(void) = 0;
    virtual void ProcessDisconnect(void) = 0;

};

/*
* Individual message handlers (like link training, config etc) use these
* interfaces to send the messages to GFM or peer FMs
*/

class FMConnInterface
{
public:
    FMConnInterface() {;}

    virtual ~FMConnInterface() { ; }

    virtual dcgmReturn_t SendMessageToGfm( lwswitch::fmMessage *pFmMessage, bool trackReq ) = 0;

    virtual dcgmReturn_t SendMessageToLfm( uint32 fabricNodeId, lwswitch::fmMessage *pFmMessage, bool trackReq ) = 0;
};
