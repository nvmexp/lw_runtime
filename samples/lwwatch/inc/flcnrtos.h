/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _FLCNRTOS_H_
#define _FLCNRTOS_H_

#include "lw_rtos_extension.h"


/* ------------------------ Types definitions ------------------------------ */
typedef struct _flcnRtosTcb             FLCN_RTOS_TCB, *PFLCN_RTOS_TCB;
typedef struct _flcnRtosXlist           FLCN_RTOS_XLIST, *PFLCN_RTOS_XLIST;
typedef struct _flcnRtosXlistItem       FLCN_RTOS_XLIST_ITEM, *PFALCON_RTOS_XLIST_ITEM;
typedef struct _flcnRtosXminiListItem   FLCN_RTOS_XMINI_LIST_ITEM;
typedef struct _flcnRtosXqueue          FLCN_RTOS_XQUEUE, *PFLCN_RTOS_XQUEUE;


/* ------------------------- Common definies ------------------------------- */

#define configMAX_PRIORITIES              8     // Size of ReadyTaskLists
#define configMAX_TASK_NAME_LEN           8

//
// Used to derefernce pointers in RTOS
//
#define FLCN_RTOS_DEREF_DMEM_PTR(engineBase, ptr, port, pVal) \
    (!thisFlcn? FALSE : \
    (!thisFlcn->pFEIF || !thisFlcn->pFCIF)? FALSE : \
    (thisFlcn->pFCIF->flcnDmemRead(engineBase, ptr, LW_TRUE, 1, port, pVal)) == 1)

#define FLCN_RTOS_DEREF_DMEM_PTR_64(engineBase, ptr, port, pVal) \
    (!thisFlcn? FALSE : \
    (!thisFlcn->pFEIF || !thisFlcn->pFCIF)? FALSE : \
    (thisFlcn->pFCIF->flcnDmemRead(engineBase, ptr, LW_TRUE, 2, port, (LwU32*)pVal)) == 2)

#define FLCN_RTOS_PRINT_SEPARATOR() \
    dprintf("lw: _________________________________________________________________________________________________\n");


/* ----------------- RTOS related function prototypes ----------------------- */
BOOL         flcnRtosTcbGet(LwU32, LwU32, FLCN_RTOS_TCB*);
BOOL         flcnRtosTcbGetLwrrent(FLCN_RTOS_TCB *, LwU32);
void         flcnRtosTcbDump(FLCN_RTOS_TCB*, BOOL, LwU32, LwU8);
void         flcnRtosTcbDumpAll(BOOL);
void         flcnRtosDmemOvlDumpAll();
void         flcnRtosSchedDump(BOOL);
void         flcnRtosEventQueueDumpAll(BOOL);
void         flcnRtosEventQueueDumpByAddr(LwU32);
void         flcnRtosEventQueueDumpByTaskId(LwU32);
void         flcnRtosEventQueueDumpByQueueId(LwU32);
void         flcnRtosEventQueueDumpBySymbol(const char *);
const char * flcnRtosGetTasknameFromTcb(FLCN_RTOS_TCB *);

/*
 * Version defines for the max supported overlays.
 * The latest version should match with what is defined
 * in the RTOS code.
 */
#define FLCN_MAX_ATTACHED_OVLS_IMEM_VER_0 16
#define FLCN_MAX_ATTACHED_OVLS_IMEM_VER_1 32
#define FLCN_MAX_ATTACHED_OVLS_IMEM_VER_2 FLCN_MAX_ATTACHED_OVLS_IMEM_VER_1
#define FLCN_MAX_ATTACHED_OVLS_IMEM_VER_3 FLCN_MAX_ATTACHED_OVLS_IMEM_VER_1
#define FLCN_MAX_ATTACHED_OVLS_IMEM_VER_4 FLCN_MAX_ATTACHED_OVLS_IMEM_VER_1
#define FLCN_MAX_ATTACHED_OVLS_IMEM_VER_5 FLCN_MAX_ATTACHED_OVLS_IMEM_VER_1
#define FLCN_MAX_ATTACHED_OVLS_IMEM_VER_6 FLCN_MAX_ATTACHED_OVLS_IMEM_VER_1

#define FLCN_MAX_ATTACHED_OVLS_DMEM_VER_0 16
#define FLCN_MAX_ATTACHED_OVLS_DMEM_VER_1 32
#define FLCN_MAX_ATTACHED_OVLS_DMEM_VER_2 FLCN_MAX_ATTACHED_OVLS_DMEM_VER_1
#define FLCN_MAX_ATTACHED_OVLS_DMEM_VER_3 FLCN_MAX_ATTACHED_OVLS_DMEM_VER_1
#define FLCN_MAX_ATTACHED_OVLS_DMEM_VER_4 FLCN_MAX_ATTACHED_OVLS_DMEM_VER_1
#define FLCN_MAX_ATTACHED_OVLS_DMEM_VER_5 FLCN_MAX_ATTACHED_OVLS_DMEM_VER_1
#define FLCN_MAX_ATTACHED_OVLS_DMEM_VER_6 FLCN_MAX_ATTACHED_OVLS_DMEM_VER_1

typedef enum
{
    FLCN_TCB_VER_0 = 0,
    FLCN_TCB_VER_1 = 1,
    FLCN_TCB_VER_2 = 2,
    FLCN_TCB_VER_3 = 3,
    FLCN_TCB_VER_4 = 4,
    FLCN_TCB_VER_5 = 5
} FLCN_TCB_VER;

typedef enum
{
    FLCN_TCB_PVT_VER_0      = 0, // version snapped 12/04/2015
    FLCN_TCB_PVT_VER_1      = 1, // version snapped 12/05/2015
    FLCN_TCB_PVT_VER_2      = 2, // version snapped 03/2016
    FLCN_TCB_PVT_VER_3      = 3, // version snapped 07/2016
    FLCN_TCB_PVT_VER_4      = 4, // version snapped 11/2016
    FLCN_TCB_PVT_VER_5      = 5, // version snapped 02/2018
    FLCN_TCB_PVT_VER_6      = 6, // version snapped 10/2018
    FLCN_TCB_PVT_VER_COUNT  = 7
} FLCN_TCB_PVT_VER;

/*
 * This structure contains a union of all the PVT TCB versions
 * that have existed within RM. The latest PVT TCB version must
 * match the definition inside lw_rtos_extension.h.
 */
typedef union
{
    struct
    {
        LwU32   pPrivTcbNext;
        LwU32   pStack;
        LwU32   pData;
        LwU16   stackSize;
        LwU16   usedHeap;
        LwU8    taskID;
        LwBool  bReload;
        LwU8    privilegeLevel;
        LwU8    ovlCnt;
        LwU8    ovlList[0];
    } flcnTcbPvt0;
    struct
    {
        LwU32   pPrivTcbNext;
        LwU32   pStack;
        LwU16   stackSize;
        LwU16   usedHeap;
        LwU8    taskID;
        LwBool  bReload;
        LwU8    privilegeLevel;
        LwU8    ovlCntImem;
        LwU8    ovlCntDmem;
        LwU8    ovlList[0];
    } flcnTcbPvt1;
    struct
    {
        LwU32   pPrivTcbNext;
        LwU32   pStack;
        LwU16   stackSize;
        LwU16   usedHeap;
        LwU8    taskID;
        LwBool  bReload;
        LwBool  bPagingOnDemandImem;
        LwU8    privilegeLevel;
        LwU8    ovlIdxStack;
        LwU8    ovlCntImem;
        LwU8    ovlCntDmem;
        LwU8    ovlList[0];
    } flcnTcbPvt2;
    struct
    {
        LwU32   pPrivTcbNext;
        LwU32   pStack;
        LwU16   stackSize;
        LwU16   usedHeap;
        LwU8    taskID;
        LwBool  bReload;
        LwBool  bPagingOnDemandImem;
        LwU8    privilegeLevel;
        LwU8    ovlIdxStack;
        LwU8    ovlCntImemLS;
        LwU8    ovlCntImemHS;
        LwU8    ovlCntDmem;
        LwU8    ovlList[0];
    } flcnTcbPvt3;
    struct
    {
        LwU32   pPrivTcbNext;
        LwU32   pStack;
        LwU32   spMin;
        LwU16   stackSize;
        LwU16   usedHeap;
        LwU8    taskID;
        LwBool  bReload;
        LwBool  bPagingOnDemandImem;
        LwU8    privilegeLevel;
        LwU8    ovlIdxStack;
        LwU8    ovlCntImemLS;
        LwU8    ovlCntImemHS;
        LwU8    ovlCntDmem;
        LwU8    ovlList[0];
    } flcnTcbPvt4;
    struct
    {
        LwU32   pPrivTcbNext;
        LwU32   pStack;
        LwU32   spMin;
        LwU32   pRunTimeStats;
        LwU16   stackSize;
        LwU16   usedHeap;
        LwU8    taskID;
        LwBool  bReload;
        LwBool  bPagingOnDemandImem;
        LwU8    privilegeLevel;
        LwU8    ovlIdxStack;
        LwU8    ovlCntImemLS;
        LwU8    ovlCntImemHS;
        LwU8    ovlCntDmem;
        LwU8    ovlList[0];
    } flcnTcbPvt5;
    struct
    {
        LwU32   pPrivTcbNext;
        LwU32   pStack;
        LwU32   spMin;
        LwU32   pRunTimeStats;
        LwU32   stackCanary;    // New field in version 6
        LwU16   stackSize;
        LwU16   usedHeap;
        LwU8    taskID;
        LwBool  bReload;
        LwBool  bPagingOnDemandImem;
        LwU8    privilegeLevel;
        LwU8    ovlIdxStack;
        LwU8    ovlCntImemLS;
        LwU8    ovlCntImemHS;
        LwU8    ovlCntDmem;
        LwU8    ovlList[0];
    } flcnTcbPvt6;
} FLCN_TCB_PVT_INT;

typedef struct
{
    LwU32            tcbPvtAddr;
    FLCN_TCB_PVT_VER tcbPvtVer;
    FLCN_TCB_PVT_INT flcnTcbPvt;
} FLCN_TCB_PVT;

/*
 *  The RTOS Xlist Item
 */
struct _flcnRtosXlistItem
{
    LwU32   itemValue;
    LwU32   next;
    LwU32   prev;
    LwU32   owner;
    LwU32   container;
};

/*
 *  The RTOS Xmini List Item
 */
struct _flcnRtosXminiListItem
{
    LwU32   itemValue;
    LwU32   next;
    LwU32   prev;
};

/*
 *  The Xlist
 */
struct _flcnRtosXlist
{
    LwU32                        numItems;
    LwU32                        pIndex;
    FLCN_RTOS_XMINI_LIST_ITEM    listEnd;
};

/*
 *  The RTOS TCB struct.
 */
struct _flcnRtosTcb
{
    FLCN_TCB_VER tcbVer;
    LwU32        tcbAddr;

    union
    {
        // The old tcb structure before we added private tcb field to it
        struct
        {
            LwU32                 pTopOfStack;
            LwU32                 priority;
            LwU32                 pStack;
            LwU32                 tcbNumber;
            char                  taskName[8];
            LwU16                 stackDepth;
            LwU32                 address;
        } flcnTcb0;

        struct
        {
            LwU32                 pxTopOfStack;
            LwU32                 pvTcbPvt;
            FLCN_RTOS_XLIST_ITEM  xGenericListItem;
            FLCN_RTOS_XLIST_ITEM  xEventListItem;
            LwU32                 uxPriority;
            LwU8                  ucTaskID;
        } flcnTcb1;

        // SafeRTOSv5.10.1-lw1.1 Falcon
        // SafeRTOSv5.16.0-lw1.2 Falcon
        struct
        {
            LwU32                 pxTopOfStack;
            LwU32                 pvTcbPvt;
            LwU32                 uxTopOfStackMirror;
            FLCN_RTOS_XLIST_ITEM  xGenericListItem;
            FLCN_RTOS_XLIST_ITEM  xEventListItem;
            LwU32                 uxNotifiedValue;
            LwU32                 xNotifyState;
            LwU8                  ucPriority;
        } flcnTcb2;

        // SafeRTOSv5.16.0-lw1.2 Falcon after adding pcStackBaseAddress (pStack in RM_RTOS_TCB_PVT)
        struct
        {
            LwU32                 pxTopOfStack;
            LwU32                 pvTcbPvt;
            LwU32                 pcStackBaseAddress;
            LwU32                 uxTopOfStackMirror;
            FLCN_RTOS_XLIST_ITEM  xGenericListItem;
            FLCN_RTOS_XLIST_ITEM  xEventListItem;
            LwU32                 uxNotifiedValue;
            LwU32                 xNotifyState;
            LwU8                  ucPriority;
        } flcnTcb3;

        // OpenRTOS Falcon after adding pcStackBaseAddress (pStack in RM_RTOS_TCB_PVT)
        struct
        {
            LwU32                 pxTopOfStack;
            LwU32                 pvTcbPvt;
            LwU32                 pcStackBaseAddress;
            FLCN_RTOS_XLIST_ITEM  xGenericListItem;
            FLCN_RTOS_XLIST_ITEM  xEventListItem;
            LwU32                 uxPriority;
            LwU8                  ucTaskID;
        } flcnTcb4;

        // SafeRTOSv5.16.0-lw1.2 Falcon after removing task notifications.
        // Same as for SafeRTOSv5.16.0-lw1.3
        struct
        {
            LwU32                 pxTopOfStack;
            LwU32                 pvTcbPvt;
            LwU32                 pcStackBaseAddress;
            LwU32                 uxTopOfStackMirror;
            FLCN_RTOS_XLIST_ITEM  xGenericListItem;
            FLCN_RTOS_XLIST_ITEM  xEventListItem;
            LwU8                  ucPriority;
        } flcnTcb5;
    } flcnTcb;
};

/*
 *  The RTOS Xqueue
 */
struct _flcnRtosXqueue
{
    LwU32             head;
    LwU32             tail;
    LwU32             writeTo;
    LwU32             readFrom;
    FLCN_RTOS_XLIST   xTasksWaitingToSend;
    FLCN_RTOS_XLIST   xTasksWaitingToReceive;
    LwU32             messagesWaiting;
    LwU32             length;
    LwU32             itemSize;
    LwS32             rxLock;
    LwS32             txLock;
    LwU32             next;
};

#endif // _FLCNRTOS_H_
