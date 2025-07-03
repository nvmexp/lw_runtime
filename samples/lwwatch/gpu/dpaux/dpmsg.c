/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*********************************************************
//
// dpmsg.cpp - module for displayport message transaction
//
//*********************************************************

#include "dpaux.h"
#include "displayport.h"
#include "dpmsg.h"

/*!
 * @brief callwlateCrc8 - Function to get crc-8 result per input.
 *                        polynomial = X^8 + X^7 + X^6 + X^4 + X^2 + 1
 * 
 *  @param[in]  const LwU8     *data            pointer to input data
 *  @param[in]  LwU8            NumberOfBytes   length of data in bytes
 *  @return     LwU8                            crc-8 result
 */
LwU8 callwlateCrc8(const LwU8 * data, LwU8 NumberOfBytes)
{
    LwU8 BitMask = BIT(7);
    LwU8 BitShift = 7;
    LwU8 ArrayIndex = 0;
    LwU16 NumberOfBits = NumberOfBytes * 8;
    LwU16 Remainder = 0;

    while (NumberOfBits != 0)
    {
        NumberOfBits--;
        Remainder <<= 1;
        Remainder |= (data[ArrayIndex] & BitMask) >> BitShift;
        BitMask >>= 1;
        BitShift--;
        if (BitMask == 0)
        {
            BitMask = BIT(7);
            BitShift = 7;
            ArrayIndex++;
        }
        if ((Remainder & BIT(8)) == BIT(8))
        {
            Remainder ^= 0xd5;
        }
    }

    NumberOfBits = 8;
    while (NumberOfBits != 0)
    {
        NumberOfBits--;
        Remainder <<= 1;
        if ((Remainder & BIT(8)) != 0)
        {
            Remainder ^= 0xd5;
        }
    }
    return Remainder & LW_U8_MAX;
}

/*!
 * @brief callwlateCrc4 - Function to get crc-4 result per input.
 *                        polynomial = X^4 + X + 1
 * 
 *  @param[in]  const LwU8     *data                pointer to input data
 *  @param[in]  LwU8            NumberOfNibbles     length of data in nibbles
 *  @return     LwU8                                crc-4 result
 */
LwU8 callwlateCrc4(const LwU8 * data, int NumberOfNibbles)
{
    LwU8 BitMask = BIT(7);
    LwU8 BitShift = 7;
    LwU8 ArrayIndex = 0;
    int NumberOfBits = NumberOfNibbles * 4;
    LwU8 Remainder = 0;

    while (NumberOfBits != 0)
    {
        NumberOfBits--;
        Remainder <<= 1;
        Remainder |= (data[ArrayIndex] & BitMask) >> BitShift;
        BitMask >>= 1;
        BitShift--;
        if (BitMask == 0)
        {
            BitMask = BIT(7);
            BitShift = 7;
            ArrayIndex++;
        }
        if ((Remainder & BIT(4)) == BIT(4))
        {
            Remainder ^= 0x13;
        }
    }

    NumberOfBits = 4;
    while (NumberOfBits != 0)
    {
        NumberOfBits--;
        Remainder <<= 1;
        if ((Remainder & BIT(4)) != 0)
        {
            Remainder ^= 0x13;
        }
    }
    return Remainder & 0xf;
}

/*!
 * @brief clearMsgReplyPending - Function to clear DOWN_REP pending message per
 *                               request times.
 * 
 *  @param[in]  LwU32           port        auxport for request
 *  @param[in]  REQUEST_TIMES   times       times of request
 */
void clearMsgReplyPending(LwU32 port, REQUEST_TIMES times)
{
    LwU8 dpcd;
    while (times--)
    {
        dpcd = DRF_DEF(_DPCD, _DEVICE_SERVICE_IRQ_VECTOR_ESI0,
                       _DOWN_REP_MSG_RDY, _YES);

        pDpaux[indexGpu].dpauxChWrite(port,
            LW_DPCD_DEVICE_SERVICE_IRQ_VECTOR_ESI0, dpcd);

        dpcd = pDpaux[indexGpu].dpauxChRead(port,
                  LW_DPCD_DEVICE_SERVICE_IRQ_VECTOR_ESI0);

        if (FLD_TEST_DRF(_DPCD, _DEVICE_SERVICE_IRQ_VECTOR_ESI0,
                         _DOWN_REP_MSG_RDY, _NO, dpcd))
            break;
    }
}

BOOL waitMsgReplyAsserting(LwU32 port)
{
    LwU32 totalWait = 0;
    LwU8  dpcd;
    do
    {
        dpcd = pDpaux[indexGpu].dpauxChRead(port,
                  LW_DPCD_DEVICE_SERVICE_IRQ_VECTOR_ESI0);
        if (FLD_TEST_DRF(_DPCD, _DEVICE_SERVICE_IRQ_VECTOR_ESI0,
                         _DOWN_REP_MSG_RDY, _YES, dpcd))
        {
            return TRUE;
        }
        osPerfDelay(MESSAGE_REPLY_RETRY_INTERVAL_MS * 1000);
        totalWait += MESSAGE_REPLY_RETRY_INTERVAL_MS;
    } while (totalWait <= MESSAGE_REPLY_TIMEOUT_MS);
    return FALSE;
}

char * getNakReason(LwU8 reason)
{
    switch (reason)
    {
        case NAK_REASON_WRITE_FAIURE:
            return "WRITE_FAILRE";
        case NAK_REASON_ILWALID_RAD:
            return "ILWALID_RAD";
        case NAK_REASON_CRC_FAILURE:
            return "CRC_FAILURE";
        case NAK_REASON_BAD_PARAM:
            return "BAD_PARAM";
        case NAK_REASON_DEFER:
            return "DEFER";
        case NAK_REASON_LINK_FAILURE:
            return "LINK_FAILURE";
        case NAK_REASON_NO_RESOURCE:
            return "NO_RESOURCE";
        case NAK_REASON_DPCD_FAIL:
            return "DPCD_FAIL";
        case NAK_REASON_I2C_NACK:
            return "I2C_NACK";
        case NAK_REASON_ALLOCATE_FAIL:
            return "ALLOCATE_FAIL";
        default:
            return "unknown";
            break;
    }
}

/*!
 * @brief callwlateHeaderSize - Function to callwate DP_MSG header size since
 *                              that veries by its link_total member.
 * 
 *  @param[in]  LwU8           lnkTotal         specify value of link_total
 *  @return     LwU8                            callwlated header size
 */
LwU8 callwlateHeaderSize(LwU8 lnkTotal)
{
    return lnkTotal ? (LW_UNSIGNED_ROUNDED_DIV(lnkTotal - 1, NIBBLES_PER_BYTE)
                       + MIN_MSG_HEADER_SIZE) : 0;
}

/*!
 * @brief collectDownRepData - Function to collect DOWN_REP data from message
 *                             box since it could be splited into multile
 *                             messages.
 * 
 *  @param[in]  LwU32          port         auxport for request
 *  @param[in]  LwU8          *pBuff        point to buff for replied data
 *  @param[in]  LwU32          buffSize     size of buffer for replied data
 *  @return     LwU32                       retrieved data size
 */
LwU32 collectDownRepData(LwU32 port, LwU8 *pBuff, LwU32 buffSize)
{
    LwU8  msgBox[DP_MESSAGEBOX_SIZE];
    LwU32 dataSize, headerLength, bodyLength, boxIndex, restMsgBytes;
    BOOL  bMsgStart = TRUE;
    BOOL  bMsgEnd   = FALSE;
    BOOL  bFinished = FALSE;

    dataSize = headerLength = bodyLength = boxIndex = restMsgBytes = 0;
    while (bFinished != TRUE)
    {
        LwU32 bytesToRead;
        //
        // for fist time of attempt, read 16 bytes from mbox to get accross
        // header block(<= 10 bytes). Otherwise, read the rest bytes.
        //
        if (bMsgStart || (restMsgBytes > DP_AUX_CHANNEL_MAX_BYTES))
            bytesToRead = DP_AUX_CHANNEL_MAX_BYTES;
        else
            bytesToRead = restMsgBytes;

        if (pDpaux[indexGpu].dpauxChReadMulti(port,
                LW_DPCD_MBOX_DOWN_REP + boxIndex, msgBox + boxIndex,
                bytesToRead) != bytesToRead)
        {
            dprintf("%s: failed to read DOWN_REP registers\n", __FUNCTION__);
            return 0;
        }

        // check if header is valid for parameters.
        if (bMsgStart)
        {
            LwU8 lnkTotal, crcIndex;

            lnkTotal = DRF_VAL(_DPMSG, _HDR, _LINK_TOTAL, msgBox[0]);
            if (lnkTotal == 0)
            {
                dprintf("%s: invalid link_count_total in header\n",
                    __FUNCTION__);
                return 0;
            }

            headerLength = callwlateHeaderSize(lnkTotal);
            crcIndex = (LwU8)headerLength - 1;
            if (callwlateCrc4(msgBox, headerLength * NIBBLES_PER_BYTE - 1) !=
                    DRF_VAL(_DPMSG, _HDR, _CRC, msgBox[crcIndex]))
            {
                dprintf("%s: invalid crc4 set in header\n", __FUNCTION__);
                return 0;
            }
            bodyLength = DRF_VAL(_DPMSG, _HDR, _BODY_LENGTH,
                                 msgBox[crcIndex - 1]);
            restMsgBytes = headerLength + bodyLength;
            if (restMsgBytes > DP_MESSAGEBOX_SIZE)
            {
                dprintf("%s: invalid rest size %d indicated\n",
                        __FUNCTION__, restMsgBytes);
                return 0;
            }
            bMsgEnd = FLD_TEST_DRF(_DPMSG, _HDR, _END_TRANS, _YES,
                                   msgBox[crcIndex]) ? TRUE : FALSE;
            bMsgStart = FALSE;
        }

        // check message body as all bytes read.
        if (restMsgBytes <= bytesToRead)
        {
            // discard ending byte for crc
            LwU32 bytesToCopy = bodyLength - 1;

            // check if body crc valid
            if (callwlateCrc8(msgBox + headerLength, (LwU8)bytesToCopy) !=
                    msgBox[headerLength + bodyLength - 1])
            {
                dprintf("%s: invalid crc8 set in msg body\n", __FUNCTION__);
                return 0;
            }

            // check if retrieved bytes overflow.
            if (dataSize + bytesToCopy > buffSize)
            {
                dprintf("%s: retrieved bytes overflow\n", __FUNCTION__);
                return 0;
            }
            memcpy(&pBuff[dataSize], &msgBox[headerLength], bytesToCopy);
            dataSize += bytesToCopy;

            if (bMsgEnd)
                bFinished = TRUE;
            else
            {
                //
                // if this is not ending message, start over again to get next
                // message data block.
                //
                bMsgStart = TRUE;
                boxIndex = 0;
                clearMsgReplyPending(port, REQ_ONCE);
                if (!waitMsgReplyAsserting(port))
                {
                    dprintf("%s: failed to get next message back\n",
                            __FUNCTION__);
                    return 0;
                }
            }
        }
        else
        {
            restMsgBytes -= bytesToRead;
            boxIndex += bytesToRead;
        }
    }
    return dataSize;
}

/*!
 * @brief headerPacker - Function to pack message header by relative address
 *                       and message body size.
 * 
 *  @param[in]  LwU8       *pMsg            pointer to start of header
 *  @param[in]  LwU8       *pDpRelAddr      point to specified relative address
 *  @param[in]  LwU8        bodySize        size of message body in bytes
 *  @return     LwU8                        size of header
 */
LwU8 headerPacker
(
    LwU8 *pMsg,
    DP_RELATIVE_ADDRESS *pDpRelAddr,
    LwU8  bodySize
)
{
    LwU8 lnkTotal, crc, i, num = 0;

    lnkTotal = pDpRelAddr->hops;
    pMsg[num++] = DRF_NUM(_DPMSG, _HDR, _LINK_TOTAL, lnkTotal) |
                  DRF_NUM(_DPMSG, _HDR, _LINK_REMAIN, lnkTotal - 1);

    for (i = 0; i < lnkTotal - 1; i += NIBBLES_PER_BYTE)
    {
        if (i + 1 < lnkTotal - 1)
        {
            pMsg[num++] = DRF_NUM(_DPMSG, _HDR, _REAR_RAD,
                                  pDpRelAddr->hop[i + 1]) |
                          DRF_NUM(_DPMSG, _HDR, _FRONT_RAD,
                                  pDpRelAddr->hop[i]);
        }
        // pad single port with 0 for byte alignment
        else
        {
            pMsg[num++] = DRF_NUM(_DPMSG, _HDR, _FRONT_RAD,
                                  pDpRelAddr->hop[i]);
        }
    }
    //
    // TODO: to support other kind of messages, broadcast/path must be
    //       indicated properly. Hard-code to fixed value(0/0) for DPCD access
    //       only now.
    //
    pMsg[num++] = DRF_NUM(_DPMSG, _HDR, _BROADCAST_MSG, 0) |
                  DRF_NUM(_DPMSG, _HDR, _PATH_MSG, 0) |
                  DRF_NUM(_DPMSG, _HDR, _BODY_LENGTH, bodySize);
    //
    // TODO: 1. to support splited message, start/end of transaction must be
    //          indicated accordingly. Hard-code both to 1/1 for stand-alone
    //          message now.
    //       2. to support sequence messages (No.0 & No.1), the sequence No.
    //          may be indicated. Hard-code to sequence No.0 for now.
    //
    pMsg[num] = DRF_NUM(_DPMSG, _HDR, _START_TRANS, 1) |
                DRF_NUM(_DPMSG, _HDR, _END_TRANS, 1) |
                DRF_NUM(_DPMSG, _HDR, _SEQ_NO, 0);

    // get crc4 with total nibbles and attach it at the end.
    crc = callwlateCrc4(pMsg, num * NIBBLES_PER_BYTE + 1);
    pMsg[num++] |= DRF_NUM(_DPMSG, _HDR, _CRC, crc);

    // return length of header for reference
    return num;
}
/*!
 * @brief setupRemoteDpcdRead - Function to setup message for remote dpcd read.
 *
 *  @param[in]  LwU32                pMsg       pointer to message beginning.
 *  @param[in]  DP_RELATIVE_ADDRESS *pDpRelAddr pointer to specified relative
 *                                              address
 *  @param[in]  LwU32                address    DPCD address to read
 *  @param[in]  LwU32                size       size of data to read
 *  @return     LwU32                           length of message in bytes
 */
LwU8 setupRemoteDpcdRead
(
    LwU8 *pMsg,
    DP_RELATIVE_ADDRESS *pDpRelAddr,
    LwU32 address,
    LwU8  size
)
{
    LwU8 headeSize, port, *pBody, num = 0;

    headeSize = headerPacker(pMsg, pDpRelAddr, REQ_REMOTE_DPCD_RW_BODY_SIZE);
    pBody = &pMsg[headeSize];
    port = pDpRelAddr->hop[pDpRelAddr->hops - 1];
    pBody[num++] = DPMSG_REMOTE_DPCD_READ;
    pBody[num++] = port << 4 | (LwU8)((address & MAX_DP_AUX_ADDRESS) >> 16);
    pBody[num++] = (address >> 8) & LW_U8_MAX;
    pBody[num++] = address & LW_U8_MAX;
    pBody[num++] = size;
    pBody[num]   = callwlateCrc8(pBody, num);
    // return total message size
    return headeSize + num + 1;
}

/*!
 * @brief setupRemoteDpcdWrite - Function to setup message for remote dpcd
 *                               write.
 *
 *  @param[in]  LwU32                pMsg       pointer to message beginning.
 *  @param[in]  DP_RELATIVE_ADDRESS *pDpRelAddr pointer to specified relative
 *                                              address
 *  @param[in]  LwU32                address    DPCD address to write
 *  @param[in]  LwU32               *pData      pointer to buffer data to write
 *  @param[in]  LwU32                size       size of buffer data to write
 *  @return     LwU32                           length of message in bytes
 */
LwU8 setupRemoteDpcdWrite
(
    LwU8 *pMsg,
    DP_RELATIVE_ADDRESS *pDpRelAddr,
    LwU32 address,
    LwU8 *pData,
    LwU8  size
)
{
    LwU8 headerSize, port, *pBody, num = 0;

    headerSize = headerPacker(pMsg, pDpRelAddr,
                              REQ_REMOTE_DPCD_RW_BODY_SIZE + size);
    pBody = &pMsg[headerSize];
    port = pDpRelAddr->hop[pDpRelAddr->hops - 1];
    pBody[num++] = DPMSG_REMOTE_DPCD_WRITE;
    pBody[num++] = port << 4 | (LwU8)((address & MAX_DP_AUX_ADDRESS) >> 16);
    pBody[num++] = (address >> 8) & LW_U8_MAX;
    pBody[num++] = address & LW_U8_MAX;
    pBody[num++] = size;

    // attach bytes to write
    memcpy(&pBody[num], pData, size);
    num += size;

    pBody[num] = callwlateCrc8(pBody, num);
    // return total message size
    return headerSize + num + 1;
}

/*!
 * @brief dpmsgGetPortAddressFromText - Function to get physical port and
 *                                      relative address per input text.
 *
 * 
 *  @param[in]  LwU32                pPort      pointer to physical port
 *  @param[in]  DP_RELATIVE_ADDRESS *pDpRelAddr pointer to specified relative
 *                                              address
 *  @param[in]  LwU32                source     pointer to input text.
 *  @return     BOOL                            TRUE as input is right format
 */
BOOL dpmsgGetPortAddressFromText
(
    LwU32 *pPort,
    DP_RELATIVE_ADDRESS *pDpRelAddr,
    char  *source
)
{
    BOOL status = TRUE;
    int  i = 0;

    *pPort = source[i++] - '0';
    pDpRelAddr->hops = 0;
    while (source[i] != '\0' && i < MAX_RELATIVE_ADDR_NUM)
    {
        if (source[i++] != '.')
        {
            status = FALSE;
            break;
        }

        if (source[i] >= 'a' && source[i] <= 'f')
        {
            pDpRelAddr->hop[pDpRelAddr->hops] = (LwU8)source[i] - 'a' + 10;
            pDpRelAddr->hops++;
        }
        else if (source[i] >= 'A' && source[i] <= 'F')
        {
            pDpRelAddr->hop[pDpRelAddr->hops] = (LwU8)source[i] - 'A' + 10;
            pDpRelAddr->hops++;
        }
        else if (source[i] >= '0' && source[i] <= '9')
        {
            pDpRelAddr->hop[pDpRelAddr->hops] = (LwU8)source[i] - '0';
            pDpRelAddr->hops++;
        }
        else
        {
            status = FALSE;
            break;
        }
        i++;
    }
    return status;
}

/*!
 * @brief dpmsgDpcdRead - Function to read dpcd with specified port & relative
 *                        relative address if concatenated port.
 * 
 *  @param[in]  LwU32                port       pointer to start of header
 *  @param[in]  DP_RELATIVE_ADDRESS *pDpRelAddr pointer to specified relative
 *                                              address
 *  @param[in]  LwU32                addr       DPCD address to read
 *  @param[in]  LwU8                *pData      pointer to data buffer
 *  @param[in]  LwU32                size       size to read
 *  @return     LwU32                           size of retrieved data 
 */
LwU32 dpmsgDpcdRead
(
    LwU32 port,
    DP_RELATIVE_ADDRESS *pDpRelAddr,
    LwU32 addr,
    LwU8 *pData,
    LwU32 size
)
{
    LwU8  msgBytes, *pMsgBuffer;
    LwU32 buffSize, completedBytes;

    // case of direct read to DP port.
    if (pDpRelAddr == NULL || pDpRelAddr->hops == 0)
    {
        return pDpaux[indexGpu].dpauxChReadMulti(port, addr, pData, size);
    }

    // allocate maximal replied data block to accommodate request message too
    buffSize = MAX_DPCD_READ_SIZE + sizeof(REMOTE_DPCD_READ_ACK_DATA);

    pMsgBuffer = (LwU8 *)malloc(buffSize);
    if (pMsgBuffer == NULL)
    {
        dprintf("%s: failed to allocate memory\n", __FUNCTION__);
        return 0;
    }

    completedBytes = 0;
    while (completedBytes < size)
    {
        REMOTE_DPCD_READ_ACK_DATA *ack;
        LwU32 BytesToRead = size - completedBytes;

        if (BytesToRead > MAX_DPCD_READ_SIZE)
            BytesToRead = MAX_DPCD_READ_SIZE;

        // clear all pending DOWN_REP msg
        clearMsgReplyPending(port, REQ_INFINITE);

        // setup request message
        msgBytes = setupRemoteDpcdRead(pMsgBuffer, pDpRelAddr, addr,
                                       (LwU8)BytesToRead);
        // send request
        if (pDpaux[indexGpu].dpauxChWriteMulti(port, LW_DPCD_MBOX_DOWN_REQ,
                pMsgBuffer, msgBytes) != msgBytes)
        {
            dprintf("%s: failed to send request\n", __FUNCTION__);
            free(pMsgBuffer);
            return 0;
        }

        // wait for reply
        if (!waitMsgReplyAsserting(port))
        {
            dprintf("%s: REMOTE_DPCD_READ timeout to reply\n", __FUNCTION__);
            free(pMsgBuffer);
            return 0;
        }

        msgBytes = (LwU8)collectDownRepData(port, pMsgBuffer, buffSize);
        ack = (REMOTE_DPCD_READ_ACK_DATA*)pMsgBuffer;
        if (FLD_TEST_DRF(_DPMSG, _REPLY_REQ, _TYPE, _ACK, ack->request) &&
            FLD_TEST_DRF(_DPMSG, _REPLY_REQ, _ID, _DPCD_READ, ack->request) &&
            FLD_TEST_DRF_NUM(_DPMSG, _REMOTE_DPCD_PORT, _VAL,
                pDpRelAddr->hop[pDpRelAddr->hops - 1], ack->port) &&
            ack->bytesToRead == BytesToRead &&
            msgBytes == BytesToRead + sizeof(REMOTE_DPCD_READ_ACK_DATA))
        {
            memcpy(pData, (LwU8*)ack + sizeof(REMOTE_DPCD_READ_ACK_DATA),
                   BytesToRead);
            completedBytes += BytesToRead;
            pData += BytesToRead;
            addr += BytesToRead;
        }
        else if (FLD_TEST_DRF(_DPMSG, _REPLY_REQ, _TYPE, _NACK,
                    ack->request) &&
                 msgBytes == sizeof(NACK_DATA))
        {
            NACK_DATA *nack = (NACK_DATA*)pMsgBuffer;
            dprintf("%s: addr: %05x length: %2x NACK/REASON = %s(%x)/%x\n",
                    __FUNCTION__, addr, BytesToRead,
                    getNakReason(nack->reason), nack->reason, nack->data);
            break;
        }
        else
        {
            dprintf("%s: corrupted data\n", __FUNCTION__);
            break;
        }
    }

    free(pMsgBuffer);
    clearMsgReplyPending(port, REQ_ONCE);
    return completedBytes;
}

/*!
 * @brief dpmsgDpcdWrite - Function to read dpcd with specified port & relative
 *                         relative address if concatenated port.
 * 
 *  @param[in]  LwU32                port       pointer to start of header
 *  @param[in]  DP_RELATIVE_ADDRESS *pDpRelAddr pointer to specified relative
 *                                              address
 *  @param[in]  LwU32                addr       DPCD address to write
 *  @param[in]  LwU8                *pData      pointer to data buffer
 *  @param[in]  LwU32                size       size to write
 *  @return     LwU32                           size of sent data
 */
LwU32 dpmsgDpcdWrite
(
    LwU32 port,
    DP_RELATIVE_ADDRESS *pDpRelAddr,
    LwU32 addr,
    LwU8 *pData,
    LwU32 size
)
{
    LwU8  msgBytes, *pMsgBuffer;
    LwU32 buffSize, completedBytes;

    // case of direct read to DP port.
    if (pDpRelAddr == NULL || pDpRelAddr->hops == 0)
    {
        return (pDpaux[indexGpu].dpauxChWriteMulti(port, addr, pData, size));
    }

    // allocate maximal request message to accommodate replied data block too
    buffSize = callwlateHeaderSize(pDpRelAddr->hops) +
               REQ_REMOTE_DPCD_RW_BODY_SIZE + MAX_DPCD_WRITE_SIZE;

    pMsgBuffer = (LwU8 *)malloc(buffSize);
    if (pMsgBuffer == NULL)
    {
        dprintf("%s: failed to allocate memory\n", __FUNCTION__);
        return 0;
    }

    completedBytes = 0;
    while (completedBytes < size)
    {
        REMOTE_DPCD_WRITE_ACK_DATA *ack;
        LwU32 BytesToWrite = size - completedBytes;

        if (BytesToWrite > MAX_DPCD_WRITE_SIZE)
            BytesToWrite = MAX_DPCD_WRITE_SIZE;

        // clear all pending DOWN_REP msg
        clearMsgReplyPending(port, REQ_INFINITE);

        // setup request message
        msgBytes = setupRemoteDpcdWrite(pMsgBuffer, pDpRelAddr, addr, pData,
                                        (LwU8)BytesToWrite);
        // send request
        if (pDpaux[indexGpu].dpauxChWriteMulti(port, LW_DPCD_MBOX_DOWN_REQ,
                pMsgBuffer, msgBytes) != msgBytes)
        {
            dprintf("%s: failed to send request\n", __FUNCTION__);
            free(pMsgBuffer);
            return 0;
        }

        // wait for reply
        if (!waitMsgReplyAsserting(port))
        {
            dprintf("%s: REMOTE_DPCD_WRITE timeout to reply\n", __FUNCTION__);
            free(pMsgBuffer);
            return 0;
        }

        msgBytes = (LwU8)collectDownRepData(port, pMsgBuffer, buffSize);
        ack = (REMOTE_DPCD_WRITE_ACK_DATA*)pMsgBuffer;
        if (FLD_TEST_DRF(_DPMSG, _REPLY_REQ, _TYPE, _ACK, ack->request) &&
            FLD_TEST_DRF(_DPMSG, _REPLY_REQ, _ID, _DPCD_WRITE, ack->request) &&
            FLD_TEST_DRF_NUM(_DPMSG, _REMOTE_DPCD_PORT, _VAL,
                pDpRelAddr->hop[pDpRelAddr->hops - 1], ack->port) &&
            msgBytes == sizeof(REMOTE_DPCD_WRITE_ACK_DATA))
        {
            completedBytes += BytesToWrite;
            pData += BytesToWrite;
            addr += BytesToWrite;
        }
        else if (FLD_TEST_DRF(_DPMSG, _REPLY_REQ, _TYPE, _NACK,
                    ack->request) &&
                 msgBytes == sizeof(NACK_DATA))
        {
            NACK_DATA *nack = (NACK_DATA*)pMsgBuffer;
            dprintf("%s: addr: %05x length: %2x NACK/REASON = %s(%x)/%x\n",
                    __FUNCTION__, addr, BytesToWrite,
                    getNakReason(nack->reason), nack->reason, nack->data);
            break;
        }
        else
        {
            dprintf("%s: corrupted data\n", __FUNCTION__);
            break;
        }
    }

    free(pMsgBuffer);
    clearMsgReplyPending(port, REQ_ONCE);
    return completedBytes;
}
