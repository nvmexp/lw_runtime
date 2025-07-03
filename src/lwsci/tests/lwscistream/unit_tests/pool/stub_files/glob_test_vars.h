//! \file
//! \brief LwSciStream Block declaration.
//!
//! \copyright
//! Copyright (c) 2018-2020 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef GLOB_TEST_VARS_H
#define GLOB_TEST_VARS_H

#include "lwscistream.h"

namespace LwSciStream {

// IpcComm class test glob vars
struct test_IpcComm {
     bool signalDisconnect_fail;
     bool isInitSuccess_fail;
     bool unpackVal_fail;
     bool readFrame_fail;
     bool signalWrite_fail;
     bool waitForConnection_fail;
     bool waitForEvent_flag;
     bool waitForEvent_flag1;
     bool waitForConnection_pass;
     bool waitForConnection_flag;
     bool waitForReadEvent_flag;
     bool IpcDstreadFrame_fail;
     bool IpcDstwaitForReadEvent_flag;
     bool flushWriteSignals_fail;
     bool LwSciIpcGetEvent_Write_Pending;
     bool LwSciIpcGetEvent_Read_Pending;
     bool MsgReceivePulse_flag;
     bool LwSciIpcGetEvent_Disconnect_Request;
     bool LwSciIpcGetEvent_fail;
     bool MsgReceivePulse_r_fail;
     bool MsgSendPulse_r_fail;
     bool LwSciIpcWrite_fail;
     bool LwSciIpcRead_flag;
     bool LwSciIpcRead_fail;
     bool ChannelCreate_r_fail;
     bool ConnectAttach_r_fail;
     bool LwSciIpcSetQnxPulseParam_fail;
     bool LwSciIpcGetEndpointInfo_fail;
     bool LwSciIpcGetEndpointInfo_Ilwalid_size;
     uint8_t unpackValAndBlob;
     uint8_t unpackFenceExport;
     uint8_t unpackBufObjExport;
     uint8_t unpackMsgSyncAttr;
     uint8_t unpackMsgElemAttr;
     uint8_t unpackMsgPacketBuffer;
     uint8_t unpackMsgStatus;
     uint8_t unpackVal;
     uint32_t counter;
};

extern struct test_IpcComm test_comm;

//TrackCount class test glob vars
struct test_TrackCount
{
    bool setDefault_fail;
    bool get_fail;
    bool set_fail_IlwalidState;
    bool set_fail_BadParameter;
    bool pending_event_fail;

};

//TrackArray class test glob vars
struct test_TrackArray
{
    bool performAction_fail;
    bool prepareEvent_fail;

};
struct test_Packet
{
    bool locationUpdate_fail;
    uint8_t counter;
    uint8_t StatusAction_fail_IlwalidState;
    uint8_t StatusAction_fail_BadParameter;
    uint8_t BufferAction_fail_IlwalidState;
    uint8_t BufferAction_fail_BadParameter;
    bool BufferStatusAction_fail_BadParameter;
    bool PayloadSet_fail;
};
//Block class test glob vars
struct test_Block
{
    bool validateWithError_fail;
    bool pktFindByHandle_fail;
    bool validateWithEvent_fail;
    bool pktCreate_fail;
    bool LwSciEventService_CreateLocalEvent_fail;
};

//IpcRecvBuffer class test glob vars
struct test_IpcRecvBuffer
{
    bool unpackVal_fail;
    bool isInitSuccess_fail;
    bool unpackMsgSyncAttr_fail;
    bool unpackBegin_fail;
    bool unpackValAndBlob_fail;
    bool unpackMsgElemAttr_fail;
    bool unpackMsgPacketBuffer_fail;
    bool unpackMsgStatus_fail;
    bool msg;
    uint8_t counter;
    bool unpackFenceExport_fail;
    bool processMsg_unpack_fail;
    bool unpackIlwalidEvent;

};

//IpcSendBuffer class test glob vars
struct test_IpcSendBuffer
{
    bool packValAndBlob_fail;
    bool isInitSuccess_fail;
    bool packVal_fail;
    uint8_t counter;
    bool srcSendPacket_packVal_fail;
    bool processMsg_pack_fail;
    bool dstReusePacket_packVal_fail;
};

extern struct test_TrackCount test_trackcount;
extern struct test_TrackArray test_trackArray;
extern struct test_Block test_block;
extern struct test_IpcRecvBuffer test_ipcrecvbuffer;
extern struct test_IpcSendBuffer test_ipcsendbuffer;
extern struct test_Packet test_packet;


// LwSciBuf Export/Import APIs test glob vars
struct test_LwSciBuf {
    bool LwSciBufAttrListIpcExportUnreconciled_fail;
    bool LwSciBufAttrListIpcImportUnreconciled_fail;
    bool LwSciBufAttrListIpcExportReconciled_fail;
    bool LwSciBufAttrListIpcImportReconciled_fail;
    bool LwSciBufObjIpcExport_fail;
    bool LwSciBufObjIpcImport_fail;
    bool LwSciBufAttrListIpcExportReconciled_blobData_null;
    bool LwSciBufAttrListIpcExportunreconciled_blobData_null;
};

extern struct test_LwSciBuf test_lwscibuf;


// LwSciSync Export/Import APIs test glob vars
struct test_LwSciSync {
    bool LwSciSyncAttrListIpcImportUnreconciled_fail;
    bool LwSciSyncAttrListIpcExportUnreconciled_fail;
    bool LwSciSyncIpcExportAttrListAndObj_fail;
    bool LwSciSyncIpcImportAttrListAndObj_fail;
    bool LwSciSyncIpcExportFence_fail;
    bool LwSciSyncIpcImportFence_fail;
    bool LwSciSyncAttrListClone_fail;
};

extern struct test_LwSciSync test_lwscisync;

// Counter used for checking the number of
// times a function is called.
struct test_function_call {
    uint8_t LwSciIpcGetEvent_counter;
    uint8_t MsgReceivePulse_r_counter;
    uint8_t MsgSendPulse_r_counter;
    uint8_t LwSciIpcWrite_counter;
    uint8_t LwSciIpcRead_counter;
};

extern struct test_function_call test_function_call;

void init_glob_test_vars(void);

} // namespace LwSciStream

#endif /* GLOB_TEST_VARS_H */
