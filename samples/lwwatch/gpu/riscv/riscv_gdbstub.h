/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#ifndef _RISCV_GDBSTUB_H_
#define _RISCV_GDBSTUB_H_

#include "lwsocket.h"
#include "riscv_prv.h"

typedef enum
{
    SESSION_ILWALID=0,
    SESSION_CREATED, // session created, neither target nor gdb are initialized
    SESSION_LISTENING, // server socket opened and listening
    SESSION_ATTACHED, // gdb attached
    SESSION_SUSPENDED, // Waiting for breakpoint, cpu running, gdb connected, fgdb returns
    SESSION_CLOSING, // Session is closing and should be removed
} SESSION_STATE;

typedef enum
{
    ALWAYS_CONTINUE=0,    // Resume core on exit. (default)
    ALWAYS_HALT,          // Do not resume core on exit.
    RESTORE_PREVIOUS,     // Restore running status on exit
} EXIT_ACTION;

typedef struct
{
    unsigned char data[NETWORK_BUFFER_SIZE]; // packet payload, 0-terminated
    unsigned len;                            // payload length
} GdbPacket;

typedef struct
{
    // Debugger state
    SESSION_STATE state;
    EXIT_ACTION exitAction;
    LwSocketPair *pConnection;                  // lwsocket.c struct
    GdbPacket *pPacket;                         // last received packet
    // Receive buffer
    unsigned char recvBuf[NETWORK_BUFFER_SIZE];
    unsigned rbc;

    // Debuggee specification
    const RiscVInstance *pInstance;             // instance used to create session
    LwU64 rtosTask;                             // xTCB address

    // Debuggee initial state
    LwBool bTaskWasRunning;

    // Debugger configuration
    LwU64 continuePollMs;                       // How long to poll for halt after continue command is exelwted
    LwU64 stepTicks;                            // How long to wait until we give up on stepping.

    LwU32 cmdQueueAddr;
    LwU32 cmdQueueSize;
    LwU32 msgQueueAddr;
    LwU32 msgQueueSize;
} Session;


#endif
