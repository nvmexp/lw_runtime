/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _FIFO_H_
#define _FIFO_H_

#include "os.h"
#include "gpuanalyze.h"
#include "mmu.h"
#include "deviceinfo.h"

//
// defines
//
#define LWRRENT_CHANNEL              0xffffffff
#define DEVICE_INFO_TYPE_ILWALID     0xffffffff
#define DEVICE_INFO_RUNLIST_ILWALID  0xffffffff
#define RUNLIST_ALL                  0xFFFFFFFF

#define LW_PFIFO_CHANNEL_STATE_BUSY                 BIT(0)
#define LW_PFIFO_CHANNEL_STATE_ENABLE               BIT(1)
#define LW_PFIFO_CHANNEL_STATE_BIND                 BIT(2)
#define LW_PFIFO_CHANNEL_STATE_ACQ_PENDING          BIT(3)
#define LW_PFIFO_CHANNEL_STATE_PENDING              BIT(4)

typedef struct
{
    LwU32  id;                  ///< A channel id that we want to read.
    LwU32  runlistId;           ///< A runlist whose channel id we want to get.
    LwU32  chramPriBase;        ///< A base adddress of runlist's channel ram whose channel id we want to get.
    LwBool bRunListValid;       ///< Whether runlistId argument is valid.
    LwBool bChramPriBaseValid;  ///< Whether chramPriBase and instanceBlockPtr are valid.
} ChannelId;

typedef struct
{
    LwU32 state;                ///< A set of the LW_PFIFO_CHANNEL_xxx
    LwU32 target;               ///< The target register from the channel ram.
    LwU32 engine;
    LwU32 regChannel;           ///< Represents the content of a channel id on the specified location.
    LwU64 instPtr;              ///< Represents decoded pointer to the instance block.
} ChannelInst;

typedef enum
{
    ENGINE_TAG_GR = 0,
    ENGINE_TAG_CE,
    ENGINE_TAG_LWENC,
    ENGINE_TAG_LWDEC,
    ENGINE_TAG_SEC2,
    ENGINE_TAG_LWJPG,
    ENGINE_TAG_IOCTRL,
    ENGINE_TAG_OFA,
    ENGINE_TAG_GSP,
    ENGINE_TAG_FLA,
    ENGINE_TAG_HSHUB,
    ENGINE_TAG_C2C,
    ENGINE_TAG_FSP,
    ENGINE_TAG_UNKNOWN,
    ENGINE_TAG_ILWALID
} ENGINE_TAG;

/**
 * A struct for name-value pairs used during fifo dumping.
 */
typedef struct
{
    const char* strName; ///<  Key in the key-pair.
    LwU32       value;   ///<  Value in the key-pair.
} NameValue;

/**
 * An extended version of NameValue with additional fields used
 * during dumping of engines.
 */
typedef struct
{
    NameValue  nameValue;  ///< NameValue that stores mapping.
    LwU32      instanceId; ///< Instance id of an engine.
    ENGINE_TAG engineTag;  ///< Tag that denotes type of an engine.
} EngineNameValue;

extern NameValue targetMem[4];
extern NameValue trueFalse[2];

#define FIFO_REG_NAME_BUFFER_LEN 100

#define REG_CASE_CHANNEL_STATUS(outStr, d, r, f, s)\
    case LW ## d ## r ## f ## s: \
        outStr = #s; \
        break;

#include "g_fifo_hal.h"                    // (rmconfig) public interface

#endif // _FIFO_H_

