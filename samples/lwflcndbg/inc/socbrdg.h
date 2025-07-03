/**************** Resource Manager Defines and Structures ******************\
*                                                                           *
* Module: SOCBRDG.H                                                         *
*       Defines and structures used for the SOC Bridge Object.              *
\***************************************************************************/

#ifndef _SOCBRDG_H_
#define _SOCBRDG_H_

#include "os.h"
#include "hal.h"
#include "tegrasys.h"

/*!
 * Enum used to index entries in the windowInfo table in OBJSOCBRDG
 */
typedef enum
{
    /*! Identifies window always pointing at GPUs BAR1 location */
    SOCBRDG_WINDOW_BAR1 = 0,
    /*! Identifies sliding window used for register access */
    SOCBRDG_WINDOW_REG,
    SOCBRDG_WINDOW_ILWALID,
} SOCBRDG_WINDOW;

typedef struct
{
    /*! CPU physical address of window */
    LwU64 windowLoc;

    /*! Location in device memory window is targeting */
    LwU64 windowTarget;

    /*! Size of the window */
    LwU64 windowSize;
} SOCBRDG_WINDOW_INFO, *PSOCBRDG_WINDOW_INFO;


struct OBJSOCBRDG
{
    /*! CPU physical address of BAR containing register space */
    LwU64 regPhysAddr;

    /*! CPU physical address of BAR containing device windows */
    LwU64 winPhysAddr;

    /*! Table holding information about each device window */
    PSOCBRDG_WINDOW_INFO windowInfo;

    LwU32 winAddrShift;
    LwU32 winAddrMask;

    /*! Parsed device relocation table */
    PDEVICE_RELOCATION pRelocTable;

    /*! Number of entries in pRelocTable */
    LwU32 numDevices;
    
    /*! Base address of device memory in the device's address space */
    LwU64 sysmemBase;

    /*! Size of device memory */
    LwU64 sysmemSizeBytes;
    
    /*! To save the current state of the BAR1 WINDOW */
    LwU64 oldSocBrdgBAR1Window;
    
    /*! To save the current state of the Sliding REG WINDOW */
    LwU64 oldSocBrdgREGWindow;
};

#include "g_socbrdg_hal.h"

#endif // _SOCBRDG_H_
