/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2010-2012,2014,2017 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cle26e_h_
#define _cle26e_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

/* class LWE2_CHANNEL_DMA */
#define  LWE2_CHANNEL_DMA                                       (0x0000E26E)

/*
 * Note that the channel id (chid) is returned in the hErrorContext buffer during alloc
 */

/* flow control data structure from arhost1x_channel.h */
typedef volatile struct _cle26e_tag0 {
 LwU32 FifoStat;                        /* HOST1X_CHANNEL_FIFOSTAT_0        0000-0004*/
 LwU32 IndOff;                          /* HOST1X_CHANNEL_INDOFF_0          0004-0008*/
 LwU32 IndCnt;                          /* HOST1X_CHANNEL_INDCNT_0          0008-000c*/
 LwU32 IndData;                         /* HOST1X_CHANNEL_INDDATA_0         000c-0010*/
 LwU32 Raise;                           /* HOST1X_CHANNEL_RAISE_0           0010-0014*/
 LwU32 DmaStart;                        /* HOST1X_CHANNEL_DMASTART_0        0014-0018*/
 LwU32 Put;                             /* HOST1X_CHANNEL_DMAPUT_0          0018-001c*/
 LwU32 Get;                             /* HOST1X_CHANNEL_DMAGET_0          001c-0020*/
 LwU32 DmaEnd;                          /* HOST1X_CHANNEL_DMAEND_0          0020-0024*/
 LwU32 DmaCtrl;                         /* HOST1X_CHANNEL_DMACTRL_0         0024-0028*/
 LwU32 FbBufBase;                       /* HOST1X_CHANNEL_FBBUFBASE_0       0028-002c*/
 LwU32 CmdSwap;                         /* HOST1X_CHANNEL_CMDSWAP_0         002c-0028*/
 LwV32 Reserved00[22];                  /*                                  0030-0088*/
 LwU32 IndOff2;                         /* HOST1X_CHANNEL_INDOFF2_0         008c-0090*/
 LwU32 TickCountHi;                     /* HOST1X_CHANNEL_TICKCOUNT_HI_0    0090-0094*/
 LwU32 TickCountLow;                    /* HOST1X_CHANNEL_TICKCOUNT_LO_0    0094-0098*/
 LwU32 ChannelCtrl;                     /* HOST1X_CHANNEL_CHANNELCTRL_0     0098-009c*/
 LwV32 Reserved01[0x3d9];               /*                                  009c-1000*/
} LwE26eControl, LwE2ControlDma;

/* fields and values from arhost1x.h */
#define LWE26E_HOST1X_CHANNEL0_BASE                             (0x00000000)
#define LWE26E_HOST1X_CHANNEL_MAP_SIZE_BYTES                    (0x00004000)

/* Context State - fields and values from arhost1x_channel.h */ 
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0                        (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_SELWRE                 (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_WORD_COUNT             (0x00000001)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_RESET_VAL              (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_RESET_MASK             (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_SW_DEFAULT_VAL         (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_SW_DEFAULT_MASK        (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_READ_MASK              (0x9f1f1fff)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_WRITE_MASK             (0x00000000)

/* Command FIFO free count */
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFNUMEMPTY_SHIFT           (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFNUMEMPTY_FIELD           ((0x3ff) << LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFNUMEMPTY_SHIFT)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFNUMEMPTY_RANGE           9:0
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFNUMEMPTY                 9:0
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFNUMEMPTY_WOFFSET         (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFNUMEMPTY_DEFAULT         (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFNUMEMPTY_DEFAULT_MASK    (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFNUMEMPTY_SW_DEFAULT      (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFNUMEMPTY_SW_DEFAULT_MASK (0x00000000)

/* Indicates whether the command FIFO is empty or not */
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFEMPTY_SHIFT              (0x0000000a)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFEMPTY_FIELD              ((0x1) << LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFEMPTY_SHIFT)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFEMPTY_RANGE              10:10
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFEMPTY                    10:10
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFEMPTY_WOFFSET            (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFEMPTY_DEFAULT            (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFEMPTY_DEFAULT_MASK       (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFEMPTY_SW_DEFAULT         (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFEMPTY_SW_DEFAULT_MASK    (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFEMPTY_NOTEMPTY           (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFEMPTY_EMPTY              (0x00000001)

/* 
 * Indicates whether GATHER is active.  If a GATHER command issued via PIO,
 * software must wait for the GATHER to be IDLE before issuing another command. 
 */ 
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER_SHIFT             (0x0000000b)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER_FIELD             ((0x1) << LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER_SHIFT)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER_RANGE             11:11
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER                   11:11
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER_WOFFSET           (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER_DEFAULT           (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER_DEFAULT_MASK      (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER_SW_DEFAULT        (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER_SW_DEFAULT_MASK   (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER_IDLE              (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER_BUSY              (0x00000001)

/* 
 * Indicates whether GATHER3D is active.  If a GATHER3D command issued via PIO,
 * software must wait for the GATHER3D to be IDLE before issuing another command.
 */
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER3D_SHIFT           (0x0000000c)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER3D_FIELD           ((0x1) << LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER3D_SHIFT)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER3D_RANGE           12:12
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER3D                 12:12
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER3D_WOFFSET         (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER3D_DEFAULT         (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER3D_DEFAULT_MASK    (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER3D_SW_DEFAULT      (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER3D_SW_DEFAULT_MASK (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER3D_IDLE            (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_CFGATHER3D_BUSY            (0x00000001)

/* Register write/read FIFO free count */
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_REGFNUMEMPTY_SHIFT             (0x00000010)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_REGFNUMEMPTY_FIELD             ((0x1f) << LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_REGFNUMEMPTY_SHIFT)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_REGFNUMEMPTY_RANGE             20:16
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_REGFNUMEMPTY                   20:16
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_REGFNUMEMPTY_WOFFSET           (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_REGFNUMEMPTY_DEFAULT           (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_REGFNUMEMPTY_DEFAULT_MASK      (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_REGFNUMEMPTY_SW_DEFAULT        (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_REGFNUMEMPTY_SW_DEFAULT_MASK   (0x00000000)

/* Number of entries available for reading in this channel's output FIFO */
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_OUTFENTRIES_SHIFT              (0x00000018)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_OUTFENTRIES_FIELD              ((0x1f) << LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_OUTFENTRIES_SHIFT)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_OUTFENTRIES_RANGE              28:24
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_OUTFENTRIES                    28:24
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_OUTFENTRIES_WOFFSET            (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_OUTFENTRIES_DEFAULT            (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_OUTFENTRIES_DEFAULT_MASK       (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_OUTFENTRIES_SW_DEFAULT         (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_OUTFENTRIES_SW_DEFAULT_MASK    (0x00000000)

/* Indicates that INDCOUNT==0, so it should be OK to issue another read */
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_INDRDY_SHIFT                   (0x0000001f)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_INDRDY_FIELD                   ((0x1) << LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_INDRDY_SHIFT)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_INDRDY_RANGE                   31:31
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_INDRDY                         31:31
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_INDRDY_WOFFSET                 (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_INDRDY_DEFAULT                 (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_INDRDY_DEFAULT_MASK            (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_INDRDY_SW_DEFAULT              (0x00000000)
#define LWE26E_HOST1X_CHANNEL_FIFOSTAT_0_INDRDY_SW_DEFAULT_MASK         (0x00000000)

/*
 * The INDOFF and INDOFF2 registers (along with INDCNT and INDDATA) are used to
 * indirectly read/write modules outside the host. If AUTOINC is set, INDOFFSET
 * increments by 4 on every access of INDDATA.  REGFNUMEMPTY is polled to
 * determine when valid data can be read from INDDATA.
 * 
 * The INDOFF register has limited capability on chips with large memory maps.
 * If the top bit of the memory address is >= 27, all of memory cannot be
 * addressed with INDOFF.  In these cases, use INDOFF2 to set the offset while
 * still using INDOFF to set the other parameters.  Always have INDOFFUPD set
 * to NO_UPDATE in these cases.  For register accesses, using INDOFF (with
 * INDOFFUPD set to UPDATE) is always more efficient, since it only requires
 * one write.
 * 
 * Indirect framebuffer write is STRONGLY DISCOURAGED.  There are better ways
 * to write to memory (direct and through the channel memory map) and there is
 * limited flow control in the host.  It's very easy to get into trouble with
 * indirect framebuffer write. 
 */ 
#define LWE26E_HOST1X_CHANNEL_INDOFF_0                                  (0x00000004)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_SELWRE                           (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_WORD_COUNT                       (0x00000001)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_RESET_VAL                        (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_RESET_MASK                       (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_SW_DEFAULT_VAL                   (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_SW_DEFAULT_MASK                  (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_READ_MASK                        (0xfbfffffd)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_WRITE_MASK                       (0xfbfffffd)

/* auto increment of read/write address */ 
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_AUTOINC_SHIFT                    (0x0000001f)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_AUTOINC_FIELD                    (0x1) << LWE26E_HOST1X_CHANNEL_INDOFF_0_AUTOINC_SHIFT)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_AUTOINC_RANGE                    31:31
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_AUTOINC                          31:31
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_AUTOINC_WOFFSET                  0x0
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_AUTOINC_DEFAULT                  (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_AUTOINC_DEFAULT_MASK             (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_AUTOINC_SW_DEFAULT               (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_AUTOINC_SW_DEFAULT_MASK          (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_AUTOINC_DISABLE                  (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_AUTOINC_ENABLE                   (0x00000001)

/* access type: indirect register or indirect framebuffer */ 
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_ACCTYPE_SHIFT                    (0x0000001e)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_ACCTYPE_FIELD                    ((0x1) << LWE26E_HOST1X_CHANNEL_INDOFF_0_ACCTYPE_SHIFT)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_ACCTYPE_RANGE                    30:30
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_ACCTYPE                          30:30
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_ACCTYPE_WOFFSET                  (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_ACCTYPE_DEFAULT                  (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_ACCTYPE_DEFAULT_MASK             (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_ACCTYPE_SW_DEFAULT               (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_ACCTYPE_SW_DEFAULT_MASK          (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_ACCTYPE_REG                      (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_ACCTYPE_FB                       (0x00000001)

/* 
 * buffer up 32 bits of register data before sending it.
 * Otherwise, register writes will be sent as soon as they are received.
 * Does not support byte writes in 16-bit host. Does not affect framebuffer writes.
 */ 
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_BUF32B_SHIFT                     (0x0000001d)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_BUF32B_FIELD                     ((0x1) << LWE26E_HOST1X_CHANNEL_INDOFF_0_BUF32B_SHIFT)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_BUF32B_RANGE                     29:29
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_BUF32B_WOFFSET                   (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_BUF32B_DEFAULT                   (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_BUF32B_DEFAULT_MASK              (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_BUF32B_SW_DEFAULT                (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_BUF32B_SW_DEFAULT_MASK           (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_BUF32B_NOBUF                     (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_BUF32B_BUF                       (0x00000001)

/* 
 * Indirect framebugger access swap control. 00 = No byte swap
 * 01 = 16-bit byte swap ([31:0] -> {[23:16],[31:24],[7:0],[15:8]})
 * 10 = 32-bit byte swap ([31:0] -> {[7:0],[15:8],[23:16],[31:24]})
 * 11 = 32-bit word swap ([31:0] -> {[15:8],[7:0],[31:24],[23:16]}) 
 */ 
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDSWAP_SHIFT                    (0x0000001b)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDSWAP_FIELD                    ((0x3) << LWE26E_HOST1X_CHANNEL_INDOFF_0_INDSWAP_SHIFT)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDSWAP_RANGE                    28:27
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDSWAP_WOFFSET                  (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDSWAP_DEFAULT                  (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDSWAP_DEFAULT_MASK             (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDSWAP_SW_DEFAULT               (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDSWAP_SW_DEFAULT_MASK          (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDSWAP_NONE                     (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDSWAP_BYTE16                   (0x00000001)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDSWAP_BYTE32                   (0x00000002)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDSWAP_WORD32                   (0x00000003)

/* ACCTYPE=FB: framebuffer address */ 
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFSET_SHIFT                  (0x00000002)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFSET_FIELD                  ((0xffffff) << LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFSET_SHIFT)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFSET_RANGE                  25:2
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFSET                        25:2
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFSET_WOFFSET                (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFSET_DEFAULT                (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFSET_DEFAULT_MASK           (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFSET_SW_DEFAULT             (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFSET_SW_DEFAULT_MASK        (0x00000000)

/* ACCTYPE=REG: register module ID */ 
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_SHIFT                   (0x00000012)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_FIELD                   ((0xff) << LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_SHIFT)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_RANGE                   25:18
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID                         25:18
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_WOFFSET                 (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_DEFAULT                 (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_DEFAULT_MASK            (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_SW_DEFAULT              (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_SW_DEFAULT_MASK         (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_HOST1X                  (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_MPE                     (0x00000001)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_VI                      (0x00000002)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_EPP                     (0x00000003)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_ISP                     (0x00000004)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_GR2D                    (0x00000005)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_GR3D                    (0x00000006)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_DISPLAY                 (0x00000008)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_TVO                     (0x0000000b)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_DISPLAYB                (0x00000009)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_DSI                     (0x0000000c)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDMODID_HDMI                    (0x0000000a)

/* ACCTYPE=REG: register offset ([15:0]) */ 
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDROFFSET_SHIFT                 (0x00000002)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDROFFSET_FIELD                 ((0xffff) << LWE26E_HOST1X_CHANNEL_INDOFF_0_INDROFFSET_SHIFT)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDROFFSET_RANGE                 17:2
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDROFFSET                       17:2
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDROFFSET_WOFFSET               (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDROFFSET_DEFAULT               (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDROFFSET_DEFAULT_MASK          (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDROFFSET_SW_DEFAULT            (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDROFFSET_SW_DEFAULT_MASK       (0x00000000)

/* Optionally disable the update of INDOFFSET when writing this register */ 
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFUPD_SHIFT                  (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFUPD_FIELD                  ((0x1) << LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFUPD_SHIFT)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFUPD_RANGE                  0:0
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFUPD                        0:0
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFUPD_WOFFSET                (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFUPD_DEFAULT                (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFUPD_DEFAULT_MASK           (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFUPD_SW_DEFAULT             (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFUPD_SW_DEFAULT_MASK        (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFUPD_UPDATE                 (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDOFF_0_INDOFFUPD_NO_UPDATE              (0x00000001)

/*
 * Indirect register access count
 * Used to trigger indirect reads.  Holds the number of registers/memory
 * locations that will be read out.  Channels should not request more than
 * there is space available in their output FIFO.  Only the protected channel
 * should make liberal use of this feature for speeding up context switching.
 * 
 * For indirect framebuffer reads, each channel cannot issue more than 
 * LW_HOST1X_MAX_IND_FB_READS at once.  The read data must return and be
 * written into the per-channel output FIFO before any additional reads can
 * be issued. 
 */ 
#define LWE26E_HOST1X_CHANNEL_INDCNT_0                                  (0x00000008)
#define LWE26E_HOST1X_CHANNEL_INDCNT_0_SELWRE                           (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDCNT_0_WORD_COUNT                       (0x00000001)
#define LWE26E_HOST1X_CHANNEL_INDCNT_0_RESET_VAL                        (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDCNT_0_RESET_MASK                       (0x0000ffff)
#define LWE26E_HOST1X_CHANNEL_INDCNT_0_SW_DEFAULT_VAL                   (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDCNT_0_SW_DEFAULT_MASK                  (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDCNT_0_READ_MASK                        (0x0000ffff)
#define LWE26E_HOST1X_CHANNEL_INDCNT_0_WRITE_MASK                       (0x0000ffff)
#define LWE26E_HOST1X_CHANNEL_INDCNT_0_INDCOUNT_SHIFT                   (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDCNT_0_INDCOUNT_FIELD                   ((0xffff) << LWE26E_HOST1X_CHANNEL_INDCNT_0_INDCOUNT_SHIFT)
#define LWE26E_HOST1X_CHANNEL_INDCNT_0_INDCOUNT_RANGE                   15:0
#define LWE26E_HOST1X_CHANNEL_INDCNT_0_INDCOUNT                         15:0
#define LWE26E_HOST1X_CHANNEL_INDCNT_0_INDCOUNT_WOFFSET                 (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDCNT_0_INDCOUNT_DEFAULT                 (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDCNT_0_INDCOUNT_DEFAULT_MASK            (0x0000ffff)
#define LWE26E_HOST1X_CHANNEL_INDCNT_0_INDCOUNT_SW_DEFAULT              (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDCNT_0_INDCOUNT_SW_DEFAULT_MASK         (0x00000000)

/*
 * This register, when written, writes to the data to the INDOFFSET in INDOFF.
 * For reads, a REGFNUMEMPTY number of 32-bit values can be read before needing
 * to poll FIFOSTAT again.
 * The per-channel output FIFO (OUTFENTRIES) is readable via this offset.  A
 * read of INDDATA will pop an entry off of the per-channel output FIFO.
 */
#define LWE26E_HOST1X_CHANNEL_INDDATA_0                         (0x0000000c)
#define LWE26E_HOST1X_CHANNEL_INDDATA_0_SELWRE                  (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDDATA_0_WORD_COUNT              (0x00000001)
#define LWE26E_HOST1X_CHANNEL_INDDATA_0_RESET_VAL               (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDDATA_0_RESET_MASK              (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDDATA_0_SW_DEFAULT_VAL          (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDDATA_0_SW_DEFAULT_MASK         (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDDATA_0_READ_MASK               (0xffffffff)
#define LWE26E_HOST1X_CHANNEL_INDDATA_0_WRITE_MASK              (0xffffffff)

/* read or write data */
#define LWE26E_HOST1X_CHANNEL_INDDATA_0_INDDATA_SHIFT           (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDDATA_0_INDDATA_FIELD           ((0xffffffff) << LWE26E_HOST1X_CHANNEL_INDDATA_0_INDDATA_SHIFT)
#define LWE26E_HOST1X_CHANNEL_INDDATA_0_INDDATA_RANGE           31:0
#define LWE26E_HOST1X_CHANNEL_INDDATA_0_INDDATA_WOFFSET         (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDDATA_0_INDDATA_DEFAULT         (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDDATA_0_INDDATA_DEFAULT_MASK    (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDDATA_0_INDDATA_SW_DEFAULT      (0x00000000)
#define LWE26E_HOST1X_CHANNEL_INDDATA_0_INDDATA_SW_DEFAULT_MASK (0x00000000)

#define LWE26E_HOST1X_CHANNEL_RAISE_0                           (0x00000010)
#define LWE26E_HOST1X_CHANNEL_DMASTART_0                        (0x00000014)
#define LWE26E_HOST1X_CHANNEL_DMAPUT_0                          (0x00000018)
#define LWE26E_HOST1X_CHANNEL_DMAGET_0                          (0x0000001c)
#define LWE26E_HOST1X_CHANNEL_DMAEND_0                          (0x00000020)
#define LWE26E_HOST1X_CHANNEL_DMACTRL_0                         (0x00000024)
#define LWE26E_HOST1X_CHANNEL_FBBUFBASE_0                       (0x00000028)
#define LWE26E_HOST1X_CHANNEL_CMDSWAP_0                         (0x0000002c)
#define LWE26E_HOST1X_CHANNEL_INDOFF2_0                         (0x0000008c)
#define LWE26E_HOST1X_CHANNEL_TICKCOUNT_HI_0                    (0x00000090)
#define LWE26E_HOST1X_CHANNEL_TICKCOUNT_LO_0                    (0x00000094)
#define LWE26E_HOST1X_CHANNEL_CHANNELCTRL_0                     (0x00000098)

/* 
 * Command Sequence 
 * Commands such as ACQUIRE_MLOCK, and WAIT_SYNCPT are for synchronization,
 * while commands such as HCFSETCL, HCFINCR, HCFNONINCR are for writing client
 * block registers.
 * A typical command sequence might be something like:
 *  ACQUIRE_MLOCK
 *  HCFSETCL to display
 *  HCFINCR write several display registers, finishing with SYNCPT_INCR
 *  HCFSETCL to host1x
 *  HCFNONINCR write WAIT_SYNCPT
 *  RELEASE_MLOCK
 * This atomically programs display and waits for a display event to occur.
 */

/*
 * The following is the format of opcodes that can be sent through the
 * command FIFO.

 * Generic command FIFO packet (contains fields common to all opcodes) and
 * is used for initial decode. All command FIFO packets are multiples of
 * 32bits.

 * DATA is based on opcode
 * LWE26E_CMD_METHOD_OPCODE_EXTEND is ACQUIRE_MLOCK, RELEASE_MLOCK
 */
#define LWE26E_CMD_METHOD_DATA                                  27:0
#define LWE26E_CMD_METHOD_OPCODE                                31:28
#define LWE26E_CMD_METHOD_OPCODE_SETCL                          (0x00000000) 
#define LWE26E_CMD_METHOD_OPCODE_INCR                           (0x00000001)
#define LWE26E_CMD_METHOD_OPCODE_NONINCR                        (0x00000002)
#define LWE26E_CMD_METHOD_OPCODE_MASK                           (0x00000003)
#define LWE26E_CMD_METHOD_OPCODE_IMM                            (0x00000004)
#define LWE26E_CMD_METHOD_OPCODE_RESTART                        (0x00000005)
#define LWE26E_CMD_METHOD_OPCODE_GATHER                         (0x00000006)
#define LWE26E_CMD_METHOD_OPCODE_EXTEND                         (0x0000000e)
#define LWE26E_CMD_METHOD_OPCODE_CHDONE                         (0x0000000f)

/* 
 * HCFSETCL
 *  The SetClass opcode is to specify which class is being to be referenced
 *  (may cause rerouting of subsequent methods/data). In addition to switching
 *  classes, the opcode allows some methods to be programmed on the switch
 *  similar to a HCFMASK opcode.
 *  SetClass tells the host which module the command stream should be directed
 *  towards. CLASSID indicates the destination module and class. MASK holds a
 *  6-bit mask of offsets relative to OFFSET that will be written with
 *  subsequent data.  From 0 to 6 data words may follow, according to MASK. 
 */ 
#define LWE26E_CMD_SETCL_MASK                                   5:0                                           
#define LWE26E_CMD_SETCL_CLASSID                                15:6
#define LWE26E_CMD_SETCL_CLASSID_HOST1X                         0x01
#define LWE26E_CMD_SETCL_CLASSID_VIDEO_ENCODE_MPEG              0x20
#define LWE26E_CMD_SETCL_CLASSID_VIDEO_ENCODE_MSENC             0x21
#define LWE26E_CMD_SETCL_CLASSID_VIDEO_ENCODE_LWENC             0x21
#define LWE26E_CMD_SETCL_CLASSID_VIDEO_ENCODE_LWENC1            0x22
#define LWE26E_CMD_SETCL_CLASSID_VIDEO_STREAMING_VI             0x30
#define LWE26E_CMD_SETCL_CLASSID_VIDEO_STREAMING_EPP            0x31
#define LWE26E_CMD_SETCL_CLASSID_VIDEO_STREAMING_ISP            0x32
#define LWE26E_CMD_SETCL_CLASSID_VIDEO_STREAMING_VCI            0x33
#define LWE26E_CMD_SETCL_CLASSID_VIDEO_STREAMING_ISPB           0x34
#define LWE26E_CMD_SETCL_CLASSID_VIDEO_STREAMING_CSI3           0x35
#define LWE26E_CMD_SETCL_CLASSID_VIDEO_STREAMING_VII2C          0x36
#define LWE26E_CMD_SETCL_CLASSID_GRAPHICS_2D_DOWNLOAD           0x50
#define LWE26E_CMD_SETCL_CLASSID_GRAPHICS_2D                    0x51
#define LWE26E_CMD_SETCL_CLASSID_GRAPHICS_2D_SB                 0x52
#define LWE26E_CMD_SETCL_CLASSID_GRAPHICS_2D_DOWNLOAD_CTX1      0x54
#define LWE26E_CMD_SETCL_CLASSID_GRAPHICS_2D_CTX1               0x55
#define LWE26E_CMD_SETCL_CLASSID_GRAPHICS_2D_SB_CTX1            0x56
#define LWE26E_CMD_SETCL_CLASSID_GRAPHICS_2D_DOWNLOAD_CTX2      0x58
#define LWE26E_CMD_SETCL_CLASSID_GRAPHICS_2D_SB_CTX2            0x5A
#define LWE26E_CMD_SETCL_CLASSID_GRAPHICS_VS                    0x5C
#define LWE26E_CMD_SETCL_CLASSID_GRAPHICS_VIC                   0x5D
#define LWE26E_CMD_SETCL_CLASSID_GRAPHICS_3D                    0x60
#define LWE26E_CMD_SETCL_CLASSID_DISPLAY                        0x70
#define LWE26E_CMD_SETCL_CLASSID_DISPLAYB                       0x71
#define LWE26E_CMD_SETCL_CLASSID_HDMI                           0x77
#define LWE26E_CMD_SETCL_CLASSID_DISPLAY_TVO                    0x78
#define LWE26E_CMD_SETCL_CLASSID_DISPLAY_DSI                    0x79
#define LWE26E_CMD_SETCL_CLASSID_DISPLAY_DSIB                   0x7A
#define LWE26E_CMD_SETCL_CLASSID_SOR                            0x7B
#define LWE26E_CMD_SETCL_CLASSID_SOR1                           0x7C
#define LWE26E_CMD_SETCL_CLASSID_DPAUX                          0x7D
#define LWE26E_CMD_SETCL_CLASSID_DPAUX1                         0x7E
#define LWE26E_CMD_SETCL_CLASSID_LWJPG                          0xC0
#define LWE26E_CMD_SETCL_CLASSID_GRAPHICS_VG                    0xD0
#define LWE26E_CMD_SETCL_CLASSID_TSEC                           0xE0
#define LWE26E_CMD_SETCL_CLASSID_TSECB                          0xE1
#define LWE26E_CMD_SETCL_CLASSID_LWDEC                          0xF0
#define LWE26E_CMD_SETCL_CLASSID_PVA0                           0xF1
#define LWE26E_CMD_SETCL_CLASSID_PVA1                           0xF2
#define LWE26E_CMD_SETCL_CLASSID_DLA0                           0xF3
#define LWE26E_CMD_SETCL_CLASSID_DLA1                           0xF4
#define LWE26E_CMD_SETCL_CLASSID_LWDEC1                         0xF5
#define LWE26E_CMD_SETCL_OFFSET                                 27:16
#define LWE26E_CMD_SETCL_OPCODE                                 31:28
#define LWE26E_CMD_SETCL_OPCODE_SETCL                           (0x00000000) 

/* 
 * HCFINCR
 *  The Incrementing opcode indicates the offset should be incremented, for
 *  each data that's part of the packet. The count argument indicates how many
 *  32bit values are following. If channel protect is enabled, it means the
 *  host should prevent a channel switch from oclwring at the end of this
 *  command packet.
 */
#define LWE26E_CMD_INCR_COUNT                                   15:0
#define LWE26E_CMD_INCR_OFFSET                                  27:16
#define LWE26E_CMD_INCR_OPCODE                                  31:28
#define LWE26E_CMD_INCR_OPCODE_INCR                             (0x00000001) 

/* 
 * HCFNONINCR
 *  The Non-Incrementing opcode indicates the same offset should be sent for
 *  each data that's part of the packet. The count argument indicates how many
 *  32bit values are following. If channel protect is enabled, it means the
 *  host should prevent a channel switch from oclwring at the end of this
 *  command packet.
 */                 
#define LWE26E_CMD_NONINCR_COUNT                                15:0
#define LWE26E_CMD_NONINCR_OFFSET                               27:16
#define LWE26E_CMD_NONINCR_OPCODE                               31:28
#define LWE26E_CMD_NONINCR_OPCODE_NONINCR                       (0x00000002)

/* 
 * HCFMASK
 *  The Mask opcode, from the starting offset, generates offsets based on where
 *  the bits are set in the mask. The host expects the amount of data following
 *  to equal the number of bits set. If channel protect is enabled, it means the
 *  host should prevent a channel switch from oclwring at the end of this
 *  command packet.
 */ 
#define LWE26E_CMD_MASK_MASK                                    15:0
#define LWE26E_CMD_MASK_OFFSET                                  27:16
#define LWE26E_CMD_MASK_OPCODE                                  31:28
#define LWE26E_CMD_MASK_OPCODE_MASK                             (0x00000003)

/* 
 * HCFIMM
 *  The Immediate opcode indicates the offset and data are contained in the
 *  same 32bit datum. Only the lowest 16 bits of data are sent to the module
 *  (IMMDATA).  The upper 16 bits are zeroed out.
 */ 
#define LWE26E_CMD_IMM_DATA                                     15:0
#define LWE26E_CMD_IMM_OFFSET                                   27:16
#define LWE26E_CMD_IMM_OPCODE                                   31:28
#define LWE26E_CMD_IMM_OPCODE_IMM                               (0x00000004)

/* 
 * HCFRESTART
 *  The Restart opcode is specific to DMA operation and causes the host
 *  to set DMAGET to (ADDRESS << 4), so the next command fetch will be from
 *  (DMASTART + DMAGET).
 * 
 *  In previous chips bits 27:0 were not decoded and assumed to be 0's
 *  (allowing only simply wrapping of GET back to the top of the command
 *  buffer). Starting with sc17, ADDRESS can be 0, for compatibile
 *  RESTARTs or non-zero acting as a JUMP.
 * 
 *  Note that the jump address granularity is 16 bytes, since the bottom 4 bits
 *  cannot be specified.
 */ 
#define LWE26E_CMD_RESTART_OFFSET                               27:0
#define LWE26E_CMD_RESTART_OPCODE                               31:28
#define LWE26E_CMD_RESTART_OPCODE_RESTART                       (0x00000005)

/* 
 * HCFGATHER
 *  The Gather opcode allows contiguous chunks of memory to be fetched and
 *  placed inline with the command stream, replacing the 2 words of the gather
 *  command.  It optionally can put an incrementing or non-incrementing opcode
 *  in the stream ahead of the gathered data.  This allows for the gathered data
 *  to be a pure data stream and not be required to have host opcodes inside.
 *  The HCFGATHER is two words; first word is described below.  Second word is
 *  base address of chunk.
 */ 
#define LWE26E_CMD_GATHER_COUNT                                 13:0
#define LWE26E_CMD_GATHER_TYPE                                  14:14
#define LWE26E_CMD_GATHER_TYPE_NONINCR                          (0x00000000)           
#define LWE26E_CMD_GATHER_TYPE_INCR                             (0x00000001)
#define LWE26E_CMD_GATHER_INSERT                                15:15
#define LWE26E_CMD_GATHER_OFFSET                                27:16
#define LWE26E_CMD_GATHER_OPCODE                                31:28
#define LWE26E_CMD_GATHER_OPCODE_GATHER                         (0x00000006)

/* 
 * HCFCHDONE
 *  This opcode indicates to the command processor that the current channel is
 *  done processing for now and is willing to give up any of its owned modules
 *  to other channels that need them.
 */ 
#define LWE26E_CMD_CHDONE_UNUSED                                27:0
#define LWE26E_CMD_CHDONE_OPCODE                                31:28
#define LWE26E_CMD_CHDONE_OPCODE_DONE                           (0x0000000f)

/* 
 * HCFEXTEND
 *  The EXTEND opcode encompasses several opcodes,
 *  using a secondary opcode field to complete the decode.
 */ 
#define LWE26E_CMD_EXTEND_VALUE                                 23:0
#define LWE26E_CMD_EXTEND_SUBOP                                 27:24
#define LWE26E_CMD_EXTEND_SUBOP_ACQUIRE_MLOCK                   (0x00000000)
#define LWE26E_CMD_EXTEND_SUBOP_RELEASE_MLOCK                   (0x00000001)
#define LWE26E_CMD_EXTEND_OPCODE                                31:28
#define LWE26E_CMD_EXTEND_OPCODE_EXTEND                         (0x0000000e)

/* 
 * MCFACQUIRE_MLOCK
 *  ACQUIRE_MLOCK and RELEASE_MLOCK operations replace the implicit
 *  locking of modules previously done by SETCL and CHDONE.
 *  SW is responsible for allocating MLOCKs.
 *  MLOCK (Module LOCK) bits are visible to all channels.
 *  ACQUIRE_MLOCK will set MLOCK[indx] to 1, but
 *  if MLOCK[indx] is already 1, then the channel
 *  will stall until the mlock arbiter grants the
 *  lock to that channel.  The mlock arbiter uses
 *  two round-robin rings, high and low priority.
 *  A high priority acquire request is always granted
 *  before a low priority request.  A channel's
 *  priority can be set using the CH_PRIORITY field.
 *  MLOCKs are used by channels to get exclusive ownership of a host client,
 *  however, assignment of MLOCK indices is completely under control of SW.
 */ 
#define LWE26E_CMD_ACQUIRE_MLOCK_INDX                           8:0
#define LWE26E_CMD_ACQUIRE_MLOCK_SUBOP                          27:24
#define LWE26E_CMD_ACQUIRE_MLOCK_SUBOP_ACQUIRE_MLOCK            (0x00000000)
#define LWE26E_CMD_ACQUIRE_MLOCK_OPCODE                         31:28
#define LWE26E_CMD_ACQUIRE_MLOCK_OPCODE_EXTEND                  (0x0000000e)

/* 
 * MCFRELEASE_MLOCK
 *  RELEASE_MLOCK will clear MLOCK[indx].
 *  If one or more channels are waiting, then the
 *  mlock arbiter will pick a channel and
 *  allow that channel's ACQUIRE_MLOCK to complete.
 */ 
#define LWE26E_CMD_RELEASE_MLOCK_INDX                           8:0
#define LWE26E_CMD_RELEASE_MLOCK_SUBOP                          27:24
#define LWE26E_CMD_RELEASE_MLOCK_SUBOP_RELEASE_MLOCK            (0x00000001)
#define LWE26E_CMD_RELEASE_MLOCK_OPCODE                         31:28
#define LWE26E_CMD_RELEASE_MLOCK_OPCODE_EXTEND                  (0x0000000e)



/* fields and values for LWE26E_CMD_SETCL_CLASSID_HOST1X from arhost_uclass.h */

/* 
 * Host class methods are written using command buffer writes.  Before use,
 * HCFSETCL must be used to select host class (1h), allowing subsequent commands 
 * like HCFINCR write these methods. Class method offsets start at 0, and are a 
 * separate address map from CHANNEL and SYNC registers.
 */

/* 
 * LW_CLASS_HOST_INCR_SYNCPT_0
 *  Classes have the INCR_SYNCPT method For host, this method, immediately
 *  increments SYNCPT[indx], irrespective of the cond. Note that INCR_SYNCPT_CNTRL
 *  and INCR_SYNCPT_ERROR are included for consistency with host clients, but
 *  writes to INCR_SYNCPT_CNTRL have no effect on the operation of host1x, and
 *  because there are no condition FIFOs to overflow, INCR_SYNCPT_ERROR will
 *  never be set.
 */ 
#define LWE26E_HOST1X_INCR_SYNCPT                               (0x00000000)
#define LWE26E_HOST1X_INCR_SYNCPT_INDX                          7:0
#define LWE26E_HOST1X_INCR_SYNCPT_COND                          15:8
#define LWE26E_HOST1X_INCR_SYNCPT_COND_IMMEDIATE                (0x00000000)
#define LWE26E_HOST1X_INCR_SYNCPT_COND_OP_DONE                  (0x00000001)
#define LWE26E_HOST1X_INCR_SYNCPT_COND_RD_DONE                  (0x00000002)
#define LWE26E_HOST1X_INCR_SYNCPT_COND_REG_WR_SAFE              (0x00000003)

/* 
 * LW_CLASS_HOST_INCR_SYNCPT_CNTRL_0
 *  If NO_STALL is 1, then when fifos are full,
 *  INCR_SYNCPT methods will be dropped and the
 *  INCR_SYNCPT_ERROR[COND] bit will be set.
 *  If NO_STALL is 0, then when fifos are full,
 *  the client host interface will be stalled.
 * 
 *  If SOFT_RESET is set, then all internal state
 *  of the client syncpt block will be reset.
 *  To do soft reset, first set SOFT_RESET of
 *  all host1x clients affected, then clear all
 *  SOFT_RESETs.
 */ 
#define LWE26E_HOST1X_INCR_SYNCPT_CNTRL                         (0x00000001)
#define LWE26E_HOST1X_INCR_SYNCPT_CNTRL_SOFT_RESET              0:0
#define LWE26E_HOST1X_INCR_SYNCPT_CNTRL_SOFT_RESET_OFF          (0x00000000)
#define LWE26E_HOST1X_INCR_SYNCPT_CNTRL_SOFT_RESET_ON           (0x00000001)
#define LWE26E_HOST1X_INCR_SYNCPT_CNTRL_NO_STALL                8:8
#define LWE26E_HOST1X_INCR_SYNCPT_CNTRL_NO_STALL_OFF            (0x00000000)
#define LWE26E_HOST1X_INCR_SYNCPT_CNTRL_NO_STALL_ON             (0x00000001)

/* 
 * LW_CLASS_HOST_INCR_SYNCPT_ERROR_0
 *  COND_STATUS[COND] is set if the fifo for COND overflows.
 *  This bit is sticky and will remain set until cleared.
 *  Cleared by writing 1.
 */ 
#define LWE26E_HOST1X_INCR_SYNCPT_ERROR                         (0x00000002)
#define LWE26E_HOST1X_INCR_SYNCPT_ERROR_COND_STATUS             31:0

/* 
 *  LW_CLASS_HOST_WAIT_SYNCPT_0
 *  Wait on syncpt method
 *  Command dispatch will stall until
 *  SYNCPT[indx][LW_HOST1X_SYNCPT_THRESH_WIDTH-1:0] >= threshold[LW_HOST1X_SYNCPT_THRESH_WIDTH-1:0]
 *  The comparison takes into account the possibility of wrapping.
 *  Note that more bits are allocated for indx and threshold than may be
 *  used in an implementation
 *  Use LW_HOST1X_SYNCPT_NB_PTS for the number of syncpts, and
 *  LW_HOST1X_SYNCPT_THESH_WIDTH for the number of bits used by the comparison
 */ 
#define LWE26E_HOST1X_WAIT_SYNCPT                               (0x00000008)
#define LWE26E_HOST1X_WAIT_SYNCPT_THRESH                        23:0
#define LWE26E_HOST1X_WAIT_SYNCPT_INDX                          31:24

/* 
 * LW_CLASS_HOST_WAIT_SYNCPT_BASE_0
 *  Wait on syncpt method using base register
 *  Command dispatch will stall until
 *  SYNCPT[indx][LW_HOST1X_SYNCPT_THRESH_WIDTH-1:0] >= (SYNCPT_BASE[base_indx]+offset)
 *  The comparison takes into account the possibility of wrapping.
 *  Note that more bits are allocated for indx and base_indx than may be
 *  used in an implementation.
 *  Use LW_HOST1X_SYNCPT_NB_PTS for the number of syncpts,
 *  Use LW_HOST1X_SYNCPT_NB_BASES for the number of syncpt_bases, and
 *  LW_HOST1X_SYNCPT_THESH_WIDTH for the number of bits used by the comparison
 *  If LW_HOST1X_SYNCPT_THESH_WIDTH is greater than 16, offset is sign-extended
 *  before it is added to SYNCPT_BASE.
 */ 
#define LWE26E_HOST1X_WAIT_SYNCPT_BASE                          (0x00000009)
#define LWE26E_HOST1X_WAIT_SYNCPT_BASE_OFFSET                   15:0
#define LWE26E_HOST1X_WAIT_SYNCPT_BASE_BASE_INDX                23:16
#define LWE26E_HOST1X_WAIT_SYNCPT_BASE_INDX                     31:24

/* 
 * LW_CLASS_HOST_WAIT_SYNCPT_INCR_0
 *  Wait on syncpt increment method
 *  Command dispatch will stall until the next time that SYNCPT[indx] is incremented.
 *  Note that more bits are allocated for indx than may be used in an implementation.
 *  Use LW_HOST1X_SYNCPT_NB_PTS for the number of syncpts.
 */
#define LWE26E_HOST1X_WAIT_SYNCPT_INCR                          (0x0000000a)
#define LWE26E_HOST1X_WAIT_SYNCPT_INCR_INDX                     31:24

/* 
 * LW_CLASS_HOST_LOAD_SYNCPT_BASE_0
 *  Load syncpt base method
 *  SYNCPT_BASE[indx] = value
 */
#define LWE26E_HOST1X_LOAD_SYNCPT_BASE                          (0x0000000b)
#define LWE26E_HOST1X_LOAD_SYNCPT_BASE_VALUE                    23:0
#define LWE26E_HOST1X_LOAD_SYNCPT_BASE_INDX                     31:24

/* 
 * LW_CLASS_HOST_INCR_SYNCPT_BASE_0
 *  Increment syncpt base method
 *  SYNCPT_BASE[indx] += offset
 */ 
#define LWE26E_HOST1X_INCR_SYNCPT_BASE                          (0x0000000c)
#define LWE26E_HOST1X_INCR_SYNCPT_BASE_OFFSET                   23:0
#define LWE26E_HOST1X_INCR_SYNCPT_BASE_INDX                     31:24

/* 
 * LW_CLASS_HOST_CLEAR_0
 *  Clear method.  Any bits set in VECTOR will be cleared in the channel's RAISE
 *  vector.
 */ 
#define LWE26E_HOST1X_CLEAR                                     (0x0000000d)
#define LWE26E_HOST1X_CLEAR_VECTOR_RANGE                        31:0

/* 
 * LW_CLASS_HOST_WAIT_0
 *  Wait method.  Command dispatch will stall until any of the bits set in
 *  VECTOR become set in the channel's RAISE vector.
 */ 
#define LWE26E_HOST1X_WAIT                                      (0x0000000e)
#define LWE26E_HOST1X_WAIT_VECTOR_RANGE                         31:0

/* 
 * LW_CLASS_HOST_WAIT_WITH_INTR_0
 *  Wait w/ interrupt method.  Identical to the WAIT method except an interrupt
 *  will be triggered when the WAIT requirement is satisfied.
 */ 
#define LWE26E_HOST1X_WAIT_WITH_INTR                            (0x0000000f)
#define LWE26E_HOST1X_WAIT_WITH_INTR_VECTOR_RANGE               31:0
 
/* 
 * LW_CLASS_HOST_DELAY_USEC_0
 *  Delay number of microseconds.  Command dispatch will stall until the number
 *  of microseconds indicated in NUSEC has passed.  The timing of microseconds
 *  is controlled by the USEC_CLK register.
 */ 
#define LWE26E_HOST1X_DELAY_USEC                                (0x00000010)
#define LWE26E_HOST1X_DELAY_USEC_NUSEC                          19:0

/* 
 * LW_CLASS_HOST_TICKCOUNT_HI_0
 *  This register value will initialize the high 32 bits of 
 *  tick count value in the host clock counter
 */ 
#define LWE26E_HOST1X_TICKCOUNT_HI                              (0x00000011)
#define LWE26E_HOST1X_TICKCOUNT_HI_TICKS_HI                     31:0

/* 
 * LW_CLASS_HOST_TICKCOUNT_LO_0
 *  This register value will initialize the low 32 bits of 
 *  tick count value in the host clock counter
 */
#define LWE26E_HOST1X_TICKCOUNT_LO                              (0x00000012)
#define LWE26E_HOST1X_TICKCOUNT_LO_TICKS_LO                     31:0

/* 
 * LW_CLASS_HOST_TICKCTRL_0
 *  This register write enables the tick counter on the host clock to start counting
 */ 
#define LWE26E_HOST1X_TICKCTRL                                  (0x00000013)
#define LWE26E_HOST1X_TICKCTRL_TICKCNT                          0:0
#define LWE26E_HOST1X_TICKCTRL_TICKCNT_DISABLE                  (0x00000000)
#define LWE26E_HOST1X_TICKCTRL_TICKCNT_ENABLE                   (0x00000001)

/*
 *  Indirect addressing
 *  These registers (along with INDDATA) are used to indirectly read/write either
 *  register or memory.  Host registers are not accessible using this interface.
 *  If AUTOINC is set, INDOFFSET increments by 4 on every access of INDDATA.
 * 
 *  Either INDCTRL/INDOFF2 or INDOFF can be used, but INDOFF may not be able to
 *  address all memory in chips with large memory maps.  The rundundant bits in
 *  INDCTRL and INDOFF are shared, so writing either offset sets those bits.
 * 
 *  NOTE: due to a HW bug (bug #343175) the following restrictions apply to the
 *  use of indirect memory writes:
 *  (1) at initialization time, do a dummy indirect write (with all byte enables set to zero), and
 *  (2) dedicate an MLOCK for indirect memory writes, then before a channel issues
 *      a set of indirect memory writes it must acquire this MLOCK; after the writes
 *      have been issued, the MLOCK is released -- this will restrict the use of
 *      indirect memory writes to a single channel at a time.
 */

/* 
 * LW_CLASS_HOST_INDCTRL_0
 *  Byte enables.  Will apply to all subsequent data transactions.  Not applicable for reads.
 *  Auto increment of read/write address
 *  Route return data to spool FIFO, only applicable to reads
 *  Access type: indirect register or indirect framebuffer
 *  Read/write
 */ 
#define LWE26E_HOST1X_INDCTRL                                   (0x0000002b)
#define LWE26E_HOST1X_INDCTRL_RWN                               0:0
#define LWE26E_HOST1X_INDCTRL_RWN_WRITE                         (0x00000000)
#define LWE26E_HOST1X_INDCTRL_RWN_READ                          (0x00000001)
#define LWE26E_HOST1X_INDCTRL_ACCTYPE                           1:1
#define LWE26E_HOST1X_INDCTRL_ACCTYPE_REG                       (0x00000000)
#define LWE26E_HOST1X_INDCTRL_ACCTYPE_FB                        (0x00000001)
#define LWE26E_HOST1X_INDCTRL_SPOOL                             26:26
#define LWE26E_HOST1X_INDCTRL_SPOOL_DISABLE                     (0x00000000)
#define LWE26E_HOST1X_INDCTRL_SPOOL_ENABLE                      (0x00000001)
#define LWE26E_HOST1X_INDCTRL_AUTOINC                           27:27
#define LWE26E_HOST1X_INDCTRL_AUTOINC_DISABLE                   (0x00000000)
#define LWE26E_HOST1X_INDCTRL_AUTOINC_ENABLE                    (0x00000001)
#define LWE26E_HOST1X_INDCTRL_INDBE                             31:28

/* 
 * LW_CLASS_HOST_INDOFF2_0 
 */ 
#define LWE26E_HOST1X_INDOFF2                                   (0x0000002c)
#define LWE26E_HOST1X_INDOFF2_INDROFFSET                        17:2
#define LWE26E_HOST1X_INDOFF2_INDOFFSET                         31:2
#define LWE26E_HOST1X_INDOFF2_INDMODID                          25:18
#define LWE26E_HOST1X_INDOFF2_INDMODID_HOST1x                   (0x00000000)
#define LWE26E_HOST1X_INDOFF2_INDMODID_MPE                      (0x00000001)
#define LWE26E_HOST1X_INDOFF2_INDMODID_VI                       (0x00000002)
#define LWE26E_HOST1X_INDOFF2_INDMODID_EPP                      (0x00000003)        
#define LWE26E_HOST1X_INDOFF2_INDMODID_ISP                      (0x00000004)
#define LWE26E_HOST1X_INDOFF2_INDMODID_GR2D                     (0x00000005)
#define LWE26E_HOST1X_INDOFF2_INDMODID_GR3D                     (0x00000006)
#define LWE26E_HOST1X_INDOFF2_INDMODID_DISPLAY                  (0x00000008)
#define LWE26E_HOST1X_INDOFF2_INDMODID_DISPLAYB                 (0x00000009)
#define LWE26E_HOST1X_INDOFF2_INDMODID_HDMI                     (0x0000000a)
#define LWE26E_HOST1X_INDOFF2_INDMODID_TVO                      (0x0000000b)
#define LWE26E_HOST1X_INDOFF2_INDMODID_DSI                      (0x0000000c)

/* 
 * LW_CLASS_HOST_INDOFF_0
 *  Byte enables.  Will apply to all subsequent data transactions.  Not applicable for reads.
 *  Auto increment of read/write address
 *  Route return data to spool FIFO, only applicable to reads
 *  ACCTYPE=FB: framebuffer address
 *  ACCTYPE=REG: register module ID
 *  ACCTYPE=REG: register offset ([15:0])
 *  Access type: indirect register or indirect framebuffer
 *  Read/write
 */ 
#define LWE26E_HOST1X_INDOFF                                    (0x0000002d)
#define LWE26E_HOST1X_INDOFF_RWN                                0:0
#define LWE26E_HOST1X_INDOFF_RWN_WRITE                          (0x00000000)
#define LWE26E_HOST1X_INDOFF_RWN_READ                           (0x00000001)
#define LWE26E_HOST1X_INDOFF_ACCTYPE                            1:1
#define LWE26E_HOST1X_INDOFF_ACCTYPE_REG                        (0x00000000)
#define LWE26E_HOST1X_INDOFF_ACCTYPE_FB                         (0x00000001)
#define LWE26E_HOST1X_INDOFF_INDROFFSET                         17:2
#define LWE26E_HOST1X_INDOFF_INDOFFSET                          25:2
#define LWE26E_HOST1X_INDOFF_INDMODID                           25:18
#define LWE26E_HOST1X_INDOFF_INDMODID_HOST1x                    (0x00000000)
#define LWE26E_HOST1X_INDOFF_INDMODID_MPE                       (0x00000001)
#define LWE26E_HOST1X_INDOFF_INDMODID_VI                        (0x00000002)
#define LWE26E_HOST1X_INDOFF_INDMODID_EPP                       (0x00000003)        
#define LWE26E_HOST1X_INDOFF_INDMODID_ISP                       (0x00000004)
#define LWE26E_HOST1X_INDOFF_INDMODID_GR2D                      (0x00000005)
#define LWE26E_HOST1X_INDOFF_INDMODID_GR3D                      (0x00000006)
#define LWE26E_HOST1X_INDOFF_INDMODID_DISPLAY                   (0x00000008)
#define LWE26E_HOST1X_INDOFF_INDMODID_DISPLAYB                  (0x00000009)
#define LWE26E_HOST1X_INDOFF_INDMODID_HDMI                      (0x0000000a)
#define LWE26E_HOST1X_INDOFF_INDMODID_TVO                       (0x0000000b)
#define LWE26E_HOST1X_INDOFF_INDMODID_DSI                       (0x0000000c)
#define LWE26E_HOST1X_INDOFF_SPOOL                              26:26
#define LWE26E_HOST1X_INDOFF_SPOOL_DISABLE                      (0x00000000)
#define LWE26E_HOST1X_INDOFF_SPOOL_ENABLE                       (0x00000001)
#define LWE26E_HOST1X_INDOFF_AUTOINC                            27:27
#define LWE26E_HOST1X_INDOFF_AUTOINC_DISABLE                    (0x00000000)
#define LWE26E_HOST1X_INDOFF_AUTOINC_ENABLE                     (0x00000001)
#define LWE26E_HOST1X_INDOFF_INDBE                              31:28

/* 
 * LW_CLASS_HOST_INDDATA_0
 *  These registers, when written, either writes to the data to the INDOFFSET in
 *  INDOFF or triggers a read of the offset at INDOFFSET.
 *  read or write data
 */ 
#define LWE26E_HOST1X_INDDATA(i)                                (0x0000002e + (i))
#define LWE26E_HOST1X_INDDATA_SIZE                              31
#define LWE26E_HOST1X_INDDATA_DATA                              31:0        

/* 
 * Channel Opcode Macros
 */
#define LWE26E_CH_OPCODE_ACQUIRE_MUTEX(MutexId) \
    /* mutex ops are extended opcodes: ex-op, op, mutex id \
     * aquire op is 0. \
     */ \
    ((14UL << 28) | (MutexId) )

#define LWE26E_CH_OPCODE_RELEASE_MUTEX(MutexId) \
    /* ex-op, op, mutex id */ \
    ((14UL << 28) | (1 << 24) | (MutexId) )

#define LWE26E_CH_OPCODE_SET_CLASS(ClassId, Offset, Mask) \
    /* op, offset, classid, mask \
     * setclass opcode is 0 \
     */ \
    ((0UL << 28) | ((Offset) << 16) | ((ClassId) << 6) | (Mask))

#define LWE26E_CH_OPCODE_INCR(Addr, Count) \
    /* op, addr, count */ \
    ((1UL << 28) | ((Addr) << 16) | (Count) )

#define LWE26E_CH_OPCODE_NONINCR(Addr, Count) \
    /* op, addr, count */ \
    ((2UL << 28) | ((Addr) << 16) | (Count))

#define LWE26E_CH_OPCODE_IMM(Addr, Value) \
    /* op, addr, count */ \
    ((4UL << 28) | ((Addr) << 16) | (Value))

#define LWE26E_CH_OPCODE_MASK(Addr, Mask) \
    /* op, addr, count */ \
    ((3UL << 28) | ((Addr) << 16) | (Mask))

#define LWE26E_CH_OPCODE_RESTART(Addr) \
    /* op, addr */ \
    ((5UL << 28) | (Addr))

#define LWE26E_CH_OPCODE_GATHER(Offset, Insert, Type, Count) \
    /* op, offset, insert, type, count */ \
    ((6UL << 28) | ((Offset) << 16) | ((Insert) << 15) | ((Type) << 14) | Count)

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cle26e_h_ */
