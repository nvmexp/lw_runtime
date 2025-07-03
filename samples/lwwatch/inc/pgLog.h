/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */


#ifndef __PGLOG_H__
#define __PGLOG_H__

/**
 * @brief PMU objects
 *
 * Since there is no support of accessing RM objects, these structures
 * are copied *manually* from the PMU object files. In the future, this file can be
 * removed if RM support becomes available. These structures are used for
 * determining the location of PG log and other PG log information stored
 * in the object PG_LOG.
 *
 * **/

typedef struct
{
    LwU32  dmaBase;
    LwU8   dmaOffset;
    LwU8   dmaIdx;
} RM_PMU_MEM;

typedef struct
{
    LwU8    stopPolicy;         //<! Indicates the stop policy
    LwU8    stopMargin;         //<! Stop margin (lwrrently unused)
    LwU16   eventsPerRecord;    //<! Max events the DMEM staging buffer can hold
    LwU16   eventsPerNSI;       //<! Represents the number of events after which PMU would
                                //<! flush the DMEM and send a non-stall interrupt (NSI) to RM
    LwU16   flushWaterMark;     //<! Represents the number of events after which PMU would
                                //<! first try to attempt a flush to PG log from DMEM
    LwBool  bWakeupEvts;        //<! Option to enable/disable logging wakeup events.
    LwU8    pad[3];             //<! pad
} RM_PMU_PG_LOG_PARAMETERS;

typedef struct
{
    RM_PMU_MEM  dmaSurface;     //!< PG Log surface parameters
    LwU32       dmaSize;        //!< Size of PG log surface.
    LwU16      *pOffset;        //!< DMEM offset of the PG log staging record.
    LwBool      bInitialized;   //!< Is PG log init cmd received by PMU?
    LwU32       recordId;       //!< The current record id.
    LwU32       getOffset;      //!< Get offset of the PG log buffer.
                                //!< (RM reads from this offset)
    LwU32       putOffset;      //!< Put offset of the PG log buffer.
                                //!< (PMU writes at this offset)
    LwBool      bFlushRequired; //!< Set to LW_TRUE if a flush from DMEM
                                //!< to PG log surface is required.
    RM_PMU_PG_LOG_PARAMETERS  params;   //<! operational parameters
} PG_LOG;

typedef struct
{
    LwU32   recordId;   //<! The unique record id identifying each PG record
    LwU32   numEvents;  //<! The number of events in each record
    LwU32   pPut;       //<! Put pointer of the record
    LwU32   rsvd;       //<! Reserved
} RM_PMU_PG_LOG_HEADER;

typedef struct
{
    LwU32   eventType;  //<! Identifier of the PG event
    LwU32   engineId;   //<! Engine on which the PG event oclwrred
    LwU32   timeStamp;  //<! Timestamp in ns
    LwU32   status;     //<! Other status pertaining to the event (lwrrently unused)
} RM_PMU_PG_LOG_ENTRY;

#endif /* __PGLOG_H__ */
