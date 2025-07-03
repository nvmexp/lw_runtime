//
// Copyright (c) 2020-2021, LWPU CORPORATION. All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.
//
/// @file

#include "gtest/gtest.h"
#include "sys/neutrino.h"
#include "common_includes.h"
#include "glob_test_vars.h"

using namespace LwSciStream;

constexpr int8_t DELTA1 {1};
constexpr int8_t DELTA2 {3};
constexpr int8_t ipcCode1 {_PULSE_CODE_MINAVAIL + DELTA1};
// QNX pulse code used by internal interrupt.
constexpr int8_t interruptCode1 {ipcCode1 + DELTA1};
constexpr int8_t interruptCode2 {ipcCode1 + DELTA1};
constexpr int32_t valWriteReq1 {static_cast<int32_t>(0xDABBAD00)};
constexpr int32_t valDisconnReq1 { static_cast<int32_t>(0xD15C099) };

int ChannelCreate_r(unsigned __flags)
{
    if (test_comm.ChannelCreate_r_fail == true)
    {
        test_comm.ChannelCreate_r_fail = false;
        return -1;
    }
    return 1;
}

int ConnectAttach_r(_Uint32t __nd, pid_t __pid, int __chid, unsigned __index, int __flags)
{
    if ((test_comm.ConnectAttach_r_fail == true) && (test_comm.counter == 1U))
    {
        test_comm.ConnectAttach_r_fail = false;
        test_comm.counter = 0U;
        return -1;
    }
    if(test_comm.ConnectAttach_r_fail == true)
    {
        test_comm.counter++;
    }
    return 2;
}

int ConnectDetach(int __coid)
{
    return 0;
}

int ChannelDestroy(int __chid)
{
    return 0;
}

int MsgReceivePulse_r(int __chid, void *__pulse, _Sizet __bytes, struct _msg_info *__info)
{
    test_function_call.MsgReceivePulse_r_counter++;
    if (test_comm.MsgReceivePulse_r_fail == true) {
        test_comm.MsgReceivePulse_r_fail = false;
        return 0x104;
    }
    if (test_comm.MsgReceivePulse_flag == true) {

        if(test_comm.counter == 1U)
        {

            test_comm.counter++;
            static_cast<_pulse*>(__pulse)->code = interruptCode1;
            static_cast<_pulse*>(__pulse)->value.sival_int = valWriteReq1;
            return 0;
        }
        test_comm.MsgReceivePulse_flag = false;

        return 0x104;
    }
    if (test_comm.waitForConnection_flag == true) {
         static_cast<_pulse*>(__pulse)->code = ipcCode1;
        static_cast<_pulse*>(__pulse)->value.sival_int = valWriteReq1;
        return 0;
    }

    if (test_comm.LwSciIpcGetEvent_Write_Pending == true) {
        static_cast<_pulse*>(__pulse)->code = interruptCode1;
        static_cast<_pulse*>(__pulse)->value.sival_int = valWriteReq1;
        return 0;
    }

    if (test_comm.LwSciIpcGetEvent_Read_Pending == true) {
        static_cast<_pulse*>(__pulse)->code = ipcCode1;
        static_cast<_pulse*>(__pulse)->value.sival_int = valWriteReq1;
        return 0;
    }

    if (test_comm.LwSciIpcGetEvent_Disconnect_Request == true) {
        static_cast<_pulse*>(__pulse)->code = interruptCode1;
        static_cast<_pulse*>(__pulse)->value.sival_int = valDisconnReq1;
        return 0;
    }
    return 0;
}

int MsgSendPulse_r(int __coid, int __priority, int __code, int  __value)
{
    test_function_call.MsgSendPulse_r_counter++;
    if (test_comm.MsgSendPulse_r_fail == true) {
        test_comm.MsgSendPulse_r_fail = false;
        return 0x104;
    }
    return 0;
}
