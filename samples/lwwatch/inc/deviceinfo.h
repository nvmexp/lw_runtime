/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2014-2020 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


//*****************************************************
//
// lwwatch deviceinfo Extension
// deviceinfo.h
//
//*****************************************************


#ifndef _DEVICEINFO_H
#define _DEVICEINFO_H

#include "hal.h"

/**
 * @brief Represents indices in engineData that is used for storing data
 * related to appropriate engine.
 */
typedef enum
{
    ENGINE_INFO_TYPE_FIFO_TAG = 0,
    ENGINE_INFO_TYPE_RUNLIST,
    ENGINE_INFO_TYPE_MMU_FAULT_ID,
    ENGINE_INFO_TYPE_RESET,
    ENGINE_INFO_TYPE_INTR,
    ENGINE_INFO_TYPE_ENUM,
    ENGINE_INFO_TYPE_INST_ID,
    ENGINE_INFO_TYPE_ENGINE_TYPE,
    ENGINE_INFO_TYPE_PBDMA_MASK,
    ENGINE_INFO_TYPE_ENGINE_TAG,
    ENGINE_INFO_TYPE_RUNLIST_PRI_BASE,
    ENGINE_INFO_TYPE_RUNLIST_ENGINE_ID,
    ENGINE_INFO_TYPE_CHRAM_PRI_BASE,
    //
    // TYPE_ILWALID can be used as an output parameter to fifoEngineDataXlate to
    // get the index of the searched engine in deviceInfo.pEngines[]
    //
    ENGINE_INFO_TYPE_ILWALID,
    //
    // PBDMA_ID is not directly stored in engineData, but can be passed to
    // fifoEngineDataXlate as an input-only argument
    //
    ENGINE_INFO_TYPE_PBDMA_ID
} ENGINE_INFO_TYPE;

/**
 * @brief Represents information about engine.
 */
typedef struct
{
    LwU32       engineData[ENGINE_INFO_TYPE_ILWALID]; ///< Data specific for the engine.
    const char *engineName;                           ///< Meaningful name for the engine.
    LwU32      *pPbdmaIds;                            ///< Array of pbdmaIds used in device info.
    LwU32      *pPbdmaFaultIds;                       ///< Array of pbdma MMU fault IDs.
    LwU32       numPbdmas;                            ///< Number of pbdmaIds
    LwBool      bHostEng;                             ///< Whether the engine is valid.
} DeviceInfoEngine;

/**
 * @brief Represents one row in device info.
 */
typedef struct
{
    LwBool      bValid;    ///< Whether the row is valid (part of engine info).
    LwBool      bInChain;  ///< Whether the chain bit is set to one.
    LwU32       value;     ///< Full row information.
    LwU32       data;      ///< Part of row that contains useful information.
    const char *type;      ///< Used for storing a type of row (ENGINE TYPE, ENUM, DATA)
} DeviceInfoRow;

/**
 * @brief Represents device info configuration registers.
 */
typedef struct
{
    LwU32  numRows;          ///< Maximal number of rows in device info.
    LwU32  maxRowsPerDevice; ///< Dynamic maximal number of rows per device.
    LwU32  maxDevices;       ///< Dynamic maximal number of devices(engines).
    LwU32  version;          ///< Version of device info that is used for this GPU.
    LwBool bValid;           ///< Whether data in the struct is valid.
} DeviceInfoCfg;

typedef struct
{
    DeviceInfoRow    *pRows;        ///< Rows of device info table.
    DeviceInfoEngine *pEngines;     ///< Array of engines that are decoded from the table.
    LwU32             enginesCount; ///< Number of engines decoded from the table.
    DeviceInfoCfg     cfg;          ///< Content of the configuration register.
    LwBool            bInitialized; ///< Whether data from device info is valid.
    LwU32             maxPbdmas;    ///< Number of pbdmas from the table.
} DeviceInfo;

extern DeviceInfo deviceInfo;

/**
 * @brief Allocates memory and initializes  @ref deviceInfoRows and @ref engineList.
 *
 * @return
 *  LW_OK:
 *    If everything exelwted correctly.
 *  LW_ERR_NO_MEMORY:
 *     If there is insufficient memory for allocation.
 */
LW_STATUS deviceInfoAlloc(void);
/**
 * @brief Parses device info information about the gpu and outputs them in
 * a readable form.
 */
void deviceInfoDump(void);


#endif // _DEVICEINFO_H
