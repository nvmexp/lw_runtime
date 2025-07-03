/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _RISCV_GDBSTUB_COMMON_H_
#define _RISCV_GDBSTUB_COMMON_H_

typedef enum
{
    REQUEST_OK = 0,
    REQUEST_SUSPEND,
    REQUEST_DETACH,
    REQUEST_ERROR
} REQUEST_RESULT;

enum
{
    GDB_SEND_HEX_ENCODED = 0x1,
    GDB_SEND_ACK         = 0x2,
};

// Hex map for ASCII-HEX colwersion
static const char HEX_MAP[]="0123456789abcdef";

int rvGdbStubSendRawPacket(Session *pSession, const void *pData, size_t len);
int rvGdbStubSendPacket(Session *pSession, const void *pPayload, size_t len, int flags);
int rvGdbStubSendStr(Session *pSession, const char *pString);
int rvGdbStubSendError(Session *pSession, LwS64 code);
int rvGdbStubSendEmpty(Session *pSession);
int rvGdbStubSendOK(Session *pSession);
int rvGdbStubReceivePacket(Session *pSession, GdbPacket *pPacket);
int rvGdbStubHexToVal(const char c);

int rvGdbStubHelperGetByte(const char *pBuffer);
void rvGdbStubHelperEncodeBE(char *pBuffer, LwU64 value, unsigned sizeBytes);
int rvGdbStubHelperDecodeBE(const char *pBuffer, void *pVal, unsigned sizeBytes);
char *rvGdbStubAsciiDecode(const char *pQuery);
TRIGGER_EVENT rvGdbStubMapBreakpointToTrigger(char type);

void rvGdbStubDeleteSession(Session *pSession);
LW_STATUS rvGdbStubInitSession(Session *pSession);

extern LwBool debugPrints;
void stubprintf(const char *format, ...);

#endif //_RISCV_GDBSTUB_COMMON_H_
