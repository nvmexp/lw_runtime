/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SMBPBI_SHARED_LWSWITCH_H_
#define _SMBPBI_SHARED_LWSWITCH_H_

#include "inforom/types.h"
#include "inforom/ifrdem.h"

/*!
 *
 * Shared surface between lwswitch and SOE that includes
 * data from the InfoROM needed for OOB queries
 *
 */
typedef struct
{
    struct {
        LwBool  bValid;
        LwU8    boardPartNum[24];
        LwU8    serialNum[16];
        LwU8    marketingName[24];
        LwU32   buildDate;
    } OBD;

    struct {
        LwBool  bValid;
        LwU8    oemInfo[32];
    } OEM;

    struct {
        LwBool  bValid;
        LwU8    inforomVer[16];
    } IMG;

    struct {
        LwBool        bValid;
        LwU64_ALIGN32 uncorrectedTotal;
        LwU64_ALIGN32 correctedTotal;
    } ECC;

    struct _def_inforomdata_dem_object {
        LwBool                              bValid;
        LwBool                              bPresent;   // in the InfoROM image

        union {
            INFOROM_OBJECT_HEADER_V1_00     header;
            INFOROM_DEM_OBJECT_V1_00        v1;
        } object;
    } DEM;
} RM_SOE_SMBPBI_INFOROM_DATA, *PRM_SOE_SMBPBI_INFOROM_DATA;

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define SOE_SMBPBI_OSFP_NUM_XCEIVERS        16

#define SOE_SMBPBI_OSFP_CMD_FIFO_SIZE       32

#define LW_SOE_SMBPBI_OSFP_FIFO_CMD_OPCODE                         7:5
#define LW_SOE_SMBPBI_OSFP_FIFO_CMD_OPCODE_LED_LOCATE_OFF   0x00000000
#define LW_SOE_SMBPBI_OSFP_FIFO_CMD_OPCODE_LED_LOCATE_ON    0x00000001
#define LW_SOE_SMBPBI_OSFP_FIFO_CMD_ARG                            4:0

typedef struct
{
    LwU32   readPtr;    // only the FIFO consumer writes this
    struct
    {
        LwU32   writePtr;
        LwU8    fifoBuffer[SOE_SMBPBI_OSFP_CMD_FIFO_SIZE];
    }       prod;       // only the FIFO producer writes this
                        // in a single DMA transaction. That is
                        // the reason for the nested struct.
} SOE_SMBPBI_OSFP_CMD_FIFO;

typedef struct
{
    LwU8    serialNumber[16];
    LwU8    partNumber[16];
    LwU8    firmwareVersion[4];
    LwU8    hardwareRevision[2];
    LwU8    fruEeprom[256];
} SOE_SMBPBI_OSFP_XCEIVER_INFO;

typedef struct
{
    LwU16   all;
    LwU16   present;
}   SOE_SMBPBI_OSFP_XCEIVER_MASKS;

typedef struct
{
    SOE_SMBPBI_OSFP_XCEIVER_MASKS       xcvrMask;
    LwS16                               temperature[SOE_SMBPBI_OSFP_NUM_XCEIVERS];
}   SOE_SMBPBI_OSFP_XCEIVER_PING_PONG_BUFF;

typedef struct
{
    LwU8                                   ledState[SOE_SMBPBI_OSFP_NUM_XCEIVERS];
    LwU8                                   pingPongBuffIdx; 
    SOE_SMBPBI_OSFP_XCEIVER_PING_PONG_BUFF pingPongBuff[2];
    SOE_SMBPBI_OSFP_XCEIVER_INFO           info[SOE_SMBPBI_OSFP_NUM_XCEIVERS];
    SOE_SMBPBI_OSFP_CMD_FIFO               cmdFifo;
} SOE_SMBPBI_OSFP_DATA, *PSOE_SMBPBI_OSFP_DATA;
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

typedef struct
{
    RM_SOE_SMBPBI_INFOROM_DATA  inforomObjects;
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    SOE_SMBPBI_OSFP_DATA        osfpData;
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
} SOE_SMBPBI_SHARED_SURFACE, *PSOE_SMBPBI_SHARED_SURFACE;

/*!
 * Macros to evaluate offsets into the shared surface
 */

#define SOE_SMBPBI_SHARED_OFFSET_INFOROM(obj, member)   \
                LW_OFFSETOF(SOE_SMBPBI_SHARED_SURFACE, inforomObjects.obj.member)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define SOE_SMBPBI_SHARED_OFFSET_OSFP(member)           \
                LW_OFFSETOF(SOE_SMBPBI_SHARED_SURFACE, osfpData.member)

/*!
 * Macros and functions facilitating work with SOE_SMBPBI_OSFP_CMD_FIFO
 */

#define OSFP_FIFO_SIZE                  SOE_SMBPBI_OSFP_CMD_FIFO_SIZE
#define OSFP_FIFO_PTR(x)                ((x) % OSFP_FIFO_SIZE)
#define OSFP_FIFO_PTR_DIFF(lwr, next)   (((next) >= (lwr)) ? ((next) - (lwr)) :      \
                                            (OSFP_FIFO_SIZE - ((lwr) - (next))))
#define OSFP_FIFO_BYTES_OCLWPIED(pf)    OSFP_FIFO_PTR_DIFF((pf)->readPtr, (pf)->prod.writePtr)

//
// See how much space is available in the FIFO.
// Must leave 1 byte free so the write pointer does not
// catch up with the read pointer. That would be indistinguishable
// from an empty FIFO.
//
#define OSFP_FIFO_BYTES_AVAILABLE(pf)   (OSFP_FIFO_PTR_DIFF((pf)->prod.writePtr, (pf)->readPtr) - \
                                 sizeof((pf)->prod.fifoBuffer[0]))

static LW_INLINE LwU8
osfp_fifo_read_element
(
    SOE_SMBPBI_OSFP_CMD_FIFO    *pFifo
)
{
    LwU8    element;

    element = pFifo->prod.fifoBuffer[pFifo->readPtr];
    pFifo->readPtr = OSFP_FIFO_PTR(pFifo->readPtr + 1);
    return element;
}

static LW_INLINE void
osfp_fifo_write_element
(
    SOE_SMBPBI_OSFP_CMD_FIFO    *pFifo,
    LwU8                        value
)
{

    pFifo->prod.fifoBuffer[pFifo->prod.writePtr] = value;
    pFifo->prod.writePtr = OSFP_FIFO_PTR(pFifo->prod.writePtr + 1);
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#endif // _SMBPBI_SHARED_LWSWITCH_H_
