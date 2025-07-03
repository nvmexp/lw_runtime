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

#include "glob_test_vars.h"


namespace LwSciStream {

// IpcComm class test glob var
struct test_IpcComm test_comm;

//TrackCount class test glob var
struct test_TrackCount test_trackcount;

//TrackArray class test glob var
struct test_TrackArray test_trackArray;

//Block class test glob var
struct test_Block test_block;

//IpcRecvBuffer class test glob var
struct test_IpcRecvBuffer test_ipcrecvbuffer;

//IpcSendBuffer class test glob var
struct test_IpcSendBuffer test_ipcsendbuffer;

// LwSciBuf Export/Import APIs test glob var
struct test_LwSciBuf test_lwscibuf;
//function call counter
struct test_function_call test_function_call;

// LwSciSync Export/Import APIs test glob var
struct test_LwSciSync test_lwscisync;

struct test_Packet test_packet;

void init_glob_test_vars(void)
{
    // IpcComm class test glob vars
    test_comm.signalDisconnect_fail = false;
    test_comm.isInitSuccess_fail = false;
    test_comm.readFrame_fail = false;
    test_comm.signalWrite_fail = false;
    test_comm.waitForConnection_fail = false;
    test_comm.waitForEvent_flag = false;
    test_comm.waitForEvent_flag1 = false;
    test_comm.waitForConnection_pass = false;
    test_comm.waitForConnection_flag = false;
    test_comm.waitForReadEvent_flag = false;
    test_comm.IpcDstreadFrame_fail = false;
    test_comm.IpcDstwaitForReadEvent_flag = false;
    test_comm.flushWriteSignals_fail = false;
    test_comm.LwSciIpcGetEndpointInfo_fail = false;
    test_comm.LwSciIpcGetEvent_Write_Pending = false;
    test_comm.LwSciIpcGetEvent_Read_Pending = false;
    test_comm.LwSciIpcGetEvent_Disconnect_Request = false;
    test_comm.MsgReceivePulse_flag = false;
    test_comm.counter = 0;
    test_comm.LwSciIpcGetEvent_fail = false;
    test_comm.MsgReceivePulse_r_fail = false;
    test_comm.MsgSendPulse_r_fail = false;
    test_comm.LwSciIpcWrite_fail = false;
    test_comm.LwSciIpcRead_flag = false;
    test_comm.LwSciIpcRead_fail = false;
    test_comm.unpackVal = 0U;
    test_comm.unpackVal_fail = false;
    test_comm.ChannelCreate_r_fail = false;
    test_comm.ConnectAttach_r_fail = false;
    test_comm.LwSciIpcSetQnxPulseParam_fail = false;
    test_comm.LwSciIpcGetEndpointInfo_Ilwalid_size = false;
    test_comm.unpackValAndBlob = 0U;
    test_comm.unpackFenceExport = 0U;
    test_comm.unpackBufObjExport = 0U;
    test_comm.unpackMsgSyncAttr = 0U;
    test_comm.unpackMsgElemAttr = 0U;
    test_comm.unpackMsgPacketBuffer = 0U;
    test_comm.unpackMsgStatus = 0U;
    test_function_call.LwSciIpcGetEvent_counter = 0U;
    test_function_call.MsgReceivePulse_r_counter = 0U;
    test_function_call.MsgSendPulse_r_counter = 0U;
    test_function_call.LwSciIpcWrite_counter = 0U;
    test_function_call.LwSciIpcRead_counter = 0U;

   // LwSciSync Export/Import APIs test glob vars
    test_lwscisync.LwSciSyncAttrListIpcImportUnreconciled_fail = false;
    test_lwscisync.LwSciSyncAttrListIpcExportUnreconciled_fail = false;
    test_lwscisync.LwSciSyncIpcExportAttrListAndObj_fail = false;
    test_lwscisync.LwSciSyncIpcImportAttrListAndObj_fail = false;
    test_lwscisync.LwSciSyncIpcExportFence_fail = false;
    test_lwscisync.LwSciSyncIpcImportFence_fail = false;
    test_lwscisync.LwSciSyncAttrListClone_fail = false;

    //Trackcount class glob vars
    test_trackcount.setDefault_fail = false;
    test_trackcount.get_fail = false;
    test_trackcount.set_fail_IlwalidState = false;
    test_trackcount.set_fail_BadParameter = false;
    test_trackcount.pending_event_fail = false;

    //Block class glob vars
    test_block.validateWithError_fail = false;
    test_block.pktFindByHandle_fail = false;
    test_block.pktCreate_fail = false;
    test_block.LwSciEventService_CreateLocalEvent_fail = false;

    //IpcRecvBuffer class glob vars
    test_ipcrecvbuffer.unpackVal_fail = false;
    test_ipcrecvbuffer.processMsg_unpack_fail = false;
    test_ipcrecvbuffer.isInitSuccess_fail = false;
    test_ipcrecvbuffer.unpackMsgSyncAttr_fail = false;
    test_ipcrecvbuffer.unpackBegin_fail = false;
    test_ipcrecvbuffer.unpackValAndBlob_fail = false;
    test_ipcrecvbuffer.unpackMsgElemAttr_fail = false;
    test_ipcrecvbuffer.unpackMsgPacketBuffer_fail = false;
    test_ipcrecvbuffer.unpackMsgStatus_fail = false;
    test_ipcrecvbuffer.msg = false;
    test_ipcrecvbuffer.counter = 0;
    test_ipcrecvbuffer.unpackFenceExport_fail = false;
    test_ipcrecvbuffer.unpackIlwalidEvent = false;

    test_ipcsendbuffer.isInitSuccess_fail = false;
    test_ipcsendbuffer.packVal_fail = false;
    test_ipcsendbuffer.packValAndBlob_fail = false;
    test_ipcsendbuffer.counter = 0;
    test_ipcsendbuffer.srcSendPacket_packVal_fail = false;
    test_ipcsendbuffer.processMsg_pack_fail = false;

    test_ipcsendbuffer.dstReusePacket_packVal_fail = false;


    test_lwscibuf.LwSciBufAttrListIpcExportUnreconciled_fail = false;
    test_lwscibuf.LwSciBufAttrListIpcImportUnreconciled_fail = false;
    test_lwscibuf.LwSciBufAttrListIpcExportReconciled_fail = false;
    test_lwscibuf.LwSciBufAttrListIpcImportReconciled_fail = false;
    test_lwscibuf.LwSciBufObjIpcExport_fail = false;
    test_lwscibuf.LwSciBufObjIpcImport_fail = false;
    test_lwscibuf.LwSciBufAttrListIpcExportReconciled_blobData_null = false;


    test_packet.locationUpdate_fail = false;
    test_packet.PayloadSet_fail = false;
    test_packet.counter = 0;

    test_trackArray.performAction_fail = false;
    test_trackArray.prepareEvent_fail = false;

    test_lwscibuf.LwSciBufAttrListIpcExportunreconciled_blobData_null = false;

    test_packet.locationUpdate_fail = false;
    test_packet.counter = 0;

    test_packet.StatusAction_fail_IlwalidState = false;
    test_packet.StatusAction_fail_BadParameter = false;
    test_packet.BufferAction_fail_IlwalidState = false;
    test_packet.BufferAction_fail_BadParameter = false;
    test_packet.BufferStatusAction_fail_BadParameter = false;
}

} // namespace LwSciStream
