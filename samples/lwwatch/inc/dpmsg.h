/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef __LWW_DPMSG_H
#define __LWW_DPMSG_H

// 4 sec to timeout
#define MESSAGE_REPLY_TIMEOUT_MS               4000

// 100 ms interval to retry
#define MESSAGE_REPLY_RETRY_INTERVAL_MS         100

#define MAX_DPCD_READ_SIZE                       32
#define MAX_DPCD_WRITE_SIZE                      32
#define NIBBLES_PER_BYTE                          2
#define MAX_RELATIVE_ADDR_NUM                    15

#define MIN_MSG_HEADER_SIZE                       3
#define REQ_REMOTE_DPCD_RW_BODY_SIZE              6

// Define header fields in byte alignment
#define LW_DPMSG_HDR_LINK_REMAIN                3:0
#define LW_DPMSG_HDR_LINK_TOTAL                 7:4

#define LW_DPMSG_HDR_REAR_RAD                   3:0
#define LW_DPMSG_HDR_FRONT_RAD                  7:4

#define LW_DPMSG_HDR_BROADCAST_MSG              7:7
#define LW_DPMSG_HDR_PATH_MSG                   6:6
#define LW_DPMSG_HDR_BODY_LENGTH                5:0

#define LW_DPMSG_HDR_START_TRANS                7:7
#define LW_DPMSG_HDR_START_TRANS_YES              1
#define LW_DPMSG_HDR_START_TRANS_NO               0
#define LW_DPMSG_HDR_END_TRANS                  6:6
#define LW_DPMSG_HDR_END_TRANS_YES                1
#define LW_DPMSG_HDR_END_TRANS_NO                 0
#define LW_DPMSG_HDR_SEQ_NO                     4:4
#define LW_DPMSG_HDR_CRC                        3:0


// Definition used data type/structure
typedef enum
{
    REQ_ONCE        = 1,
    REQ_INFINITE    = 0xFFFFFFFF
} REQUEST_TIMES;

typedef enum
{
    DPMSG_REMOTE_DPCD_READ        = 0x20,
    DPMSG_REMOTE_DPCD_WRITE       = 0x21
}REQ_DP_MSG;

typedef enum
{
    NAK_REASON_WRITE_FAIURE = 0x01,
    NAK_REASON_ILWALID_RAD,
    NAK_REASON_CRC_FAILURE,
    NAK_REASON_BAD_PARAM,
    NAK_REASON_DEFER,
    NAK_REASON_LINK_FAILURE,
    NAK_REASON_NO_RESOURCE,
    NAK_REASON_DPCD_FAIL,
    NAK_REASON_I2C_NACK,
    NAK_REASON_ALLOCATE_FAIL,
}DP_NAK_REASON;

typedef struct
{
    LwU8 hop[MAX_RELATIVE_ADDR_NUM];
    LwU8 hops;
}DP_RELATIVE_ADDRESS;

// Replied data structure
#define LW_DPMSG_REPLY_REQ_ID                    6:0
#define LW_DPMSG_REPLY_REQ_ID_DPCD_READ         0x20
#define LW_DPMSG_REPLY_REQ_ID_DPCD_WRITE        0x21
#define LW_DPMSG_REPLY_REQ_TYPE                  7:7
#define LW_DPMSG_REPLY_REQ_TYPE_ACK                0
#define LW_DPMSG_REPLY_REQ_TYPE_NACK               1

typedef struct
{
    LwU8 request;
    LwU8 guid[16];
    LwU8 reason;
    LwU8 data;
}NACK_DATA;

typedef struct
{
    LwU8 request;
    LwU8 port;
    LwU8 bytesToRead;
}REMOTE_DPCD_READ_ACK_DATA;

typedef struct
{
    LwU8 request;
    LwU8 port;
}REMOTE_DPCD_WRITE_ACK_DATA;

#define LW_DPMSG_REMOTE_DPCD_PORT_VAL            3:0

// Functions for clients
BOOL  dpmsgGetPortAddressFromText(LwU32 *pPort, DP_RELATIVE_ADDRESS *pRad, char *source);
LwU32 dpmsgDpcdRead(LwU32 port, DP_RELATIVE_ADDRESS *pRad, LwU32 addr, LwU8 *data, LwU32 size);
LwU32 dpmsgDpcdWrite(LwU32 port, DP_RELATIVE_ADDRESS *pRad, LwU32 addr, LwU8 *data, LwU32 size);

#endif //__LWW_DPMSG_H

