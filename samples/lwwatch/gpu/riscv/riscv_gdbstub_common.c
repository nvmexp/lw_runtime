/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <limits.h>

#include <utils/lwassert.h>

#include "lwsocket.h"
#include "riscv_gdbstub.h"

#include "riscv_gdbstub_common.h"

#include "riscv_printing.h"

// Code here is fully shared between the ICD and Task Debugger.
TRIGGER_EVENT rvGdbStubMapBreakpointToTrigger(char type)
{
    switch(type)
    {
        case '0': // sw breakpoint - we don't support them but emulate
        case '1': // hw breakpoint
            return TRIGGER_ON_EXEC;
        case '2': // write watchpoint
            return TRIGGER_ON_STORE;
        case '3': // read watchpoint
            return TRIGGER_ON_LOAD;
        case '4': // access watchpoint - R/W
            return TRIGGER_ON_LOAD | TRIGGER_ON_STORE;
        default:
            lprintf(PL_ERROR, "gdbstub: Unsupported breakpoint type requested.\n");
            return TRIGGER_UNUSED;
    }
}

// Sends "raw" packet to gdb, no copying and thing, no size limit
int rvGdbStubSendRawPacket(Session *pSession, const void *pData, size_t len)
{
    LW_ASSERT_OR_RETURN(pSession, -1);
    LW_ASSERT_OR_RETURN(pSession->pConnection, -1);
    LW_ASSERT_OR_RETURN(pData, -1);
    LW_ASSERT_OR_RETURN(len > 0, -1);
    LW_ASSERT_OR_RETURN(len < NETWORK_BUFFER_SIZE, -1);

    if (config.bPrintGdbCommunication)
    {
        size_t i, printedlen;
        const unsigned char *pb = pData;

        if (len > 32)
            printedlen = 128; // TODO: was 64
        else
            printedlen = len;

        lprintf(PL_DEBUG, "gdbstub: Sending Raw Packet (%zu): '", len);
        for (i = 0; i < printedlen; ++i)
            if (isprint((int)pb[i]))
                lprintf(PL_DEBUG, "%c", (int)pb[i]);
            else
                lprintf(PL_DEBUG, "[%x]", (int)pb[i]);
        if (printedlen < len)
            lprintf(PL_DEBUG, "'...(%zu more bytes)\n", len - printedlen);
        else
            lprintf(PL_DEBUG, "'\n");
    }

    return lwSocketWrite(pSession->pConnection, pData, len);
}

/*
 * Sends packet to gdb.
 *
 * data is buffered / wrapped / copied
 * Not reentrant!!, if bColwertAsciiToHex is set, pData is hex-encoded
 */
int rvGdbStubSendPacket(Session *pSession, const void *pPayload, size_t len,
                           int flags)
{
    static char sendBuf[NETWORK_BUFFER_SIZE];
    const char *pData = pPayload;
    char *pTb = sendBuf;
    unsigned checksum = 0, i;
    unsigned packet_overhead = 4 + (flags & GDB_SEND_ACK ? 1 : 0);

    LW_ASSERT_OR_RETURN(pSession, -1);
    LW_ASSERT_OR_RETURN(pSession->pConnection, -1);
    LW_ASSERT_OR_RETURN(pData, -1);
    LW_ASSERT_OR_RETURN(len < NETWORK_BUFFER_SIZE - packet_overhead, -1);

    if (flags & GDB_SEND_HEX_ENCODED)
        if ((len + packet_overhead) * 2 > NETWORK_BUFFER_SIZE)
            return -1;

    if (flags & GDB_SEND_ACK)
        *pTb++ = '+';
    *pTb++ = '$';

    if (flags & GDB_SEND_HEX_ENCODED)
    {
        for (i=0; i < len; ++i)
        {
            char a, b;
            a = HEX_MAP[(pData[i] >> 4) & 0xF];
            b = HEX_MAP[pData[i] & 0xF];
            *pTb++ = a;
            *pTb++ = b;
            checksum += (unsigned)a + (unsigned)b;
        }
        len = len * 2;
    } else
    {
        for (i=0; i < len; ++i)
        {
            *pTb++ = (pData)[i];
            checksum += (unsigned)pData[i];
        }
    }
    sprintf(pTb, "#%02x", (checksum & 0xFF));

    return rvGdbStubSendRawPacket(pSession, sendBuf, len + packet_overhead);
}

int rvGdbStubSendStr(Session *pSession, const char *pString)
{
    return rvGdbStubSendPacket(pSession, pString, strlen(pString), 0);
}

int rvGdbStubSendError(Session *pSession, LwS64 code)
{
    // Trim the code - we pass LW_STATUS to this function that may be
    // larger than what GDB permits. But for most of the cases we will
    // get proper error code back.
    char codeTrim = code & 0xFF;
    unsigned sum = 'E' + HEX_MAP[codeTrim & 0xF] + HEX_MAP[(codeTrim >> 4) & 0xF];
    char pkt[] = { '$', 'E',
                   HEX_MAP[codeTrim & 0xF],
                   HEX_MAP[(codeTrim >> 4) & 0xF],
                   '#',
                   HEX_MAP[sum & 0xF],
                   HEX_MAP[(sum >> 4) & 0xF],
                 };
    return rvGdbStubSendRawPacket(pSession, pkt, 7);
}

int rvGdbStubSendEmpty(Session *pSession)
{
    return rvGdbStubSendRawPacket(pSession, "$#00", 4);
}

int rvGdbStubSendOK(Session *pSession)
{
    return rvGdbStubSendRawPacket(pSession, "$OK#9A", 6);
}

// Parses packet, returns 0 if packed was parsed succesfully
int rvGdbStubReceivePacket(Session *pSession, GdbPacket *pPacket)
{
    unsigned i;
    int ret, packet_start = -1 /* index of '$' */, packet_end = -1 /* Index of first non-packet character */;

    LW_ASSERT_OR_RETURN(pSession, -1);
    LW_ASSERT_OR_RETURN(pSession->pConnection, -1);
    LW_ASSERT_OR_RETURN(pPacket, -1);

    // Read more data, each packet is at least
    while (1)
    {
        // Look for start of the packet
        if (packet_start < 0)
        {
            for (i=0; i<pSession->rbc; ++i)
                if (pSession->recvBuf[i] == '$')
                {
                    packet_start = i;
                    packet_end = -1;
                    break;
                }
        }

        // Look for end of the packet
        if (packet_start >= 0)
        {
            for (i=(unsigned)packet_start; i < pSession->rbc; ++i)
                if (pSession->recvBuf[i] == '#')
                    if (i + 2 < pSession->rbc) // space for checksum
                    {
                        packet_end = i + 2 + 1;
                        break;
                    }
        }

        // Whole packet was received
        if (packet_start >= 0 && packet_end > packet_start)
        {
            unsigned checksum = 0; // Checksum
            unsigned vChecksum = ~checksum; // Declared checksum

            pPacket->len = 0;

            lprintf(PL_DEBUG, "gdbstub: Received Raw Packet: '");
            for (i=(unsigned)packet_start; i<(unsigned)packet_end; ++i)
                lprintf(PL_DEBUG, "%c", pSession->recvBuf[i]);
            lprintf(PL_DEBUG, "'\n");

            // Copy packet payload (if any) and callwlate checksum
            if (packet_end - packet_start > 4)
            {
                for (i=(unsigned)packet_start + 1; i < (unsigned)packet_end - 3; ++i)
                {
                    pPacket->data[pPacket->len++] = pSession->recvBuf[i];
                    checksum += pSession->recvBuf[i];
                }
                checksum = checksum & 0xFF;

                ret = sscanf((char*)(pSession->recvBuf + packet_end - 2), "%02x", &vChecksum);
                if (ret != 1)
                {
                    vChecksum = ~checksum; // Error parsing checksum
                    lprintf(PL_ERROR, "gdbstub: Failed parsing csum: %c%c\n", pSession->recvBuf[packet_end - 2], pSession->recvBuf[packet_end -1]);
                }
            }

            // Drop packet from receive buffer
            if (pSession->rbc > (unsigned)packet_end)
                memcpy(pSession->recvBuf + packet_start,
                       pSession->recvBuf + packet_end,
                       pSession->rbc - packet_end);
            // truncate receive buffer
            pSession->rbc -= (packet_end - packet_start);
            // No packet payload or invalid packet received - try retransmission
            if (checksum != vChecksum)
            {
                lprintf(PL_ERROR, "gdbstub: Invalid packet received.\n");
                ret = rvGdbStubSendRawPacket(pSession, "-", 1);
                if (ret < 0)  // We can't do much more at this point
                {
                    lprintf(PL_ERROR, "gdbstub: Failed sending retransmission request.\n");
                    return ret;
                }
            }
            // Zero terminate packet
            if (pPacket->len < NETWORK_BUFFER_SIZE)
            {
                pPacket->data[pPacket->len] = 0;
            }
            else
            {
                lprintf(PL_ERROR, "gdbstub: Packet too large.\n");
                return -1;
            }

            if (pPacket->len > 0)
            {
                // Packet was received succesfully, send confirmation and return
                ret = rvGdbStubSendRawPacket(pSession, "+", 1);
                if (ret < 0)
                {
                    lprintf(PL_ERROR, "gdbstub: Failed sending ACK.\n");
                    return ret;
                }

                return 0;
            }

            // Look for new packet
            packet_start = -1;
            packet_end = -1;
        }

        LW_ASSERT_OR_RETURN(NETWORK_BUFFER_SIZE - pSession->rbc > 0, -1);

        ret = lwSocketRead(pSession->pConnection,
                           pSession->recvBuf + pSession->rbc,
                           NETWORK_BUFFER_SIZE - pSession->rbc, LW_TRUE);
        if (ret < 0)
            return ret;
        pSession->rbc += ret;
    }
}

////////////////////////////////////// HELPERS

int rvGdbStubHexToVal(const char c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'z')
        return 10 + c - 'a';
    if (c >= 'A' && c <= 'z')
        return 10 + c - 'A';

    return 0;
}

int rvGdbStubHelperGetByte(const char *pBuffer)
{
    LW_ASSERT_OR_RETURN(pBuffer, -1);

    if (strlen(pBuffer) < 2) //TODO : replace with strnlen
        return -1;

    return rvGdbStubHexToVal(pBuffer[0]) << 4 | rvGdbStubHexToVal(pBuffer[1]);
}

// helper - push big endian value
void rvGdbStubHelperEncodeBE(char *pBuffer, LwU64 value, unsigned sizeBytes)
{
    unsigned i;

    LW_ASSERT_OR_RETURN_VOID(pBuffer);

    for (i=0; i < sizeBytes; ++i)
    {
        *pBuffer++ = HEX_MAP[(value & 0xF0) >> 4];
        *pBuffer++ = HEX_MAP[value & 0xF];
        value >>= 8;
    }
}

// get big endian value
int rvGdbStubHelperDecodeBE(const char *pBuffer, void *pVal, unsigned sizeBytes)
{
    int ret;
    unsigned i;

    if (strlen(pBuffer) < sizeBytes * 2)
        return -1;

    for (i = 0; i < sizeBytes; ++i)
    {
        ret = rvGdbStubHelperGetByte(pBuffer);
        pBuffer += 2;

        if (ret < 0)
            return ret;

        ((unsigned char *)pVal)[i] = ret;
    }
    return 0;
}

// translates query, returns newly allocated pointer
char *rvGdbStubAsciiDecode(const char *pQuery)
{
    unsigned i;
    size_t len;
    char *pOut;

    len = strlen(pQuery);
    if (len % 2)
        return NULL;

    pOut = malloc(len / 2 + 1);
    if (!pOut)
        return NULL;

    memset(pOut, 0, len / 2 + 1);

    for (i=0; i<len / 2; ++i)
        pOut[i] = rvGdbStubHelperGetByte(pQuery + 2*i); // TODO: err check
    pOut[i] = 0;

    return pOut;
}

void rvGdbStubDeleteSession(Session *pSession)
{
    if (pSession->pConnection)
        lwSocketDestroy(pSession->pConnection);
    if (pSession->pPacket)
        free(pSession->pPacket);

    memset(pSession, 0, sizeof(Session)); // zero everything
}

LW_STATUS rvGdbStubInitSession(Session *pSession)
{
    if (!pSession->pConnection)
        pSession->pConnection = lwSocketCreate();

    if (!pSession->pConnection)
        goto err;

    pSession->rbc = 0;
    memset(pSession->recvBuf, 0, NETWORK_BUFFER_SIZE);
    pSession->pPacket = malloc(sizeof(GdbPacket));
    if (!pSession->pPacket)
        goto err;
    memset(pSession->pPacket, 0, sizeof(GdbPacket));
    pSession->state = SESSION_CREATED;
    return LW_OK;

err:
    rvGdbStubDeleteSession(pSession);
    return LW_ERR_GENERIC;
}

void stubSetDebugPrints(LwBool enabled) {
    debugPrints = enabled;
}
