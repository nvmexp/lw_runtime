/**************** Resource Manager Defines and Structures ******************\
*                                                                           *
* Module: TEGRASYS.H                                                        *
*       Defines and structures used for the CheetAh System.                   *
\***************************************************************************/

#ifndef _TEGRASYS_H_
#define _TEGRASYS_H_

#include "os.h"
#include "hal.h"

typedef struct
{
    LwU32 devId;
    char *devName;
}DEVICE_LIST, *PDEVICE_LIST;

typedef struct
{
    /*! Device id (LW_DEVID_*) */
    LwU32 devId;
    /*! Device instance */
    LwU32 devInst;
    /*! Device name */
    char *devName;
    /*! Device major version */
    LwU32 verMaj;
    /*! Device minor version */
    LwU32 verMin;
    /*! Physical start address of device */
    LwU64 start;
    /*! Physical end address of device */
    LwU64 end;
} DEVICE_RELOCATION, *PDEVICE_RELOCATION;

typedef struct
{
    LwU32               numDevices;
    PDEVICE_RELOCATION  pRelocationTable;
}TEGRASYS, *PTEGRASYS;

extern TEGRASYS TegraSysObj[MAX_GPUS];

#include "g_tegrasys_hal.h"

#endif // _TEGRASYS_H_
