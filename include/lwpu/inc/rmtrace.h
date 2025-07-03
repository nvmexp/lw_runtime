
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2002 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */


#ifndef _RMTRACE_H_
#define _RMTRACE_H_

// the low order byte of an entry type holds the type of the entry
// the higher order byte, in the register op entry case, holds a
// value that designates the type of register operation
typedef LwU16 REGTRACE_ENTRYTYPE;


typedef struct _def_regtrace_entry {
    REGTRACE_ENTRYTYPE Type;
    LwU16              Size; 
} REGTRACE_ENTRY, *PREGTRACE_ENTRY;

typedef struct _def_regtrace_reg_oper_entry {
    REGTRACE_ENTRY      Info;
    LwU32               Reg;
    LwU32               Value;
} REGTRACE_REG_OPER_ENTRY, *PREGTRACE_REG_OPER_ENTRY;

typedef struct _def_regtrace_block_start_entry {
    REGTRACE_ENTRY      Info;
    LwU32               ClassType;
    LwU32               NameBufSize;                // includes space for null char
    // followed immediately by str char data (if any)
} REGTRACE_BLOCKSTART_ENTRY, *PREGTRACE_BLOCKSTART_ENTRY;

typedef struct _def_regtrace_block_end_entry {
    REGTRACE_ENTRY      Info;
} REGTRACE_BLOCKEND_ENTRY, *PREGTRACE_BLOCKEND_ENTRY;

typedef struct _def_tuple_entry {
    REGTRACE_ENTRY      Info;
    LwU32               NameBufSize;                // includes space for null char
    LwU32               Value;    
    // followed immediately by str char data (if any)
} REGTRACE_TUPLE_ENTRY, *PREGTRACE_TUPLE_ENTRY;

typedef struct _def_printf_entry {
    REGTRACE_ENTRY      Info;
    LwU32               StrBufSize;                 // includes space for null char
    // followed immediately by str char data (if any) 
} REGTRACE_PRINTF_ENTRY, *PREGTRACE_PRINTF_ENTRY;


// possible types of an entry into the trace buffer.  The entry type is
// specified in the first byte of an entry

// reserved for RM use only.  Client will never see entry of this type.
#define REGTRACE_ENTRYTYPE_SUBBLOCK_PTR     0

#define REGTRACE_ENTRYTYPE_REG_OPER         1
#define REGTRACE_ENTRYTYPE_BLOCK_START      2
#define REGTRACE_ENTRYTYPE_BLOCK_END        3
#define REGTRACE_ENTRYTYPE_PRINTF           4
#define REGTRACE_ENTRYTYPE_TUPLE            5

// possible types of a register operation entry (REGTRACE_ENTRY_REG_OPER)
#define REGTRACE_REGOP_WRITE                1
#define REGTRACE_REGOP_READ                 2
#define REGTRACE_REGOP_DELAY                4

#define REGTRACE_WRITE_REG08            ( (0<<3) | REGTRACE_REGOP_WRITE)
#define REGTRACE_WRITE_REG16            ( (1<<3) | REGTRACE_REGOP_WRITE)
#define REGTRACE_WRITE_REG32            ( (2<<3) | REGTRACE_REGOP_WRITE)
#define REGTRACE_WRITE_REG08_DIRECT     ( (3<<3) | REGTRACE_REGOP_WRITE)
#define REGTRACE_WRITE_REG16_DIRECT     ( (4<<3) | REGTRACE_REGOP_WRITE)
#define REGTRACE_WRITE_REG32_DIRECT     ( (5<<3) | REGTRACE_REGOP_WRITE)

#define REGTRACE_READ_REG08             ( (10<<3) | REGTRACE_REGOP_READ)
#define REGTRACE_READ_REG16             ( (11<<3) | REGTRACE_REGOP_READ)
#define REGTRACE_READ_REG32             ( (12<<3) | REGTRACE_REGOP_READ)
#define REGTRACE_READ_REG08_DIRECT      ( (13<<3) | REGTRACE_REGOP_READ)
#define REGTRACE_READ_REG16_DIRECT      ( (14<<3) | REGTRACE_REGOP_READ)
#define REGTRACE_READ_REG32_DIRECT      ( (15<<3) | REGTRACE_REGOP_READ)

#define REGTRACE_OS_DELAY               ( (20<<3) | REGTRACE_REGOP_DELAY)
#define REGTRACE_OS_DELAY_US            ( (21<<3) | REGTRACE_REGOP_DELAY)
#define REGTRACE_TIMER_DELAY            ( (22<<3) | REGTRACE_REGOP_DELAY)


#define REGTRACE_GET_ENTRYTYPE(pEntry)  ( ((PREGTRACE_ENTRY)pEntry)->Type & 0xFF)
#define REGTRACE_SIZEOF(pEntry)         ( ((PREGTRACE_ENTRY)pEntry)->Size )

// rounds up to nearest multiple of 4
#define REGTRACE_ROUND_UP(size)         ( ((size+3)>>2)<<2 )

#define REGTRACE_REGOPER_TYPE(pRegOpEntry)    (((pRegOpEntry)->Info.Type) >> 8)
#define REGTRACE_ENTRYTYPE_FROM_OPER(opType)  ((opType << 8) | REGTRACE_ENTRYTYPE_REG_OPER)


// flags to specify which types of reg operations to log
#define REGTRACE_IGNORE_READS           1
#define REGTRACE_IGNORE_WRITES          2
#define REGTRACE_IGNORE_DELAYS          4

#define REGTRACE_IGNORE_NONE            0
#define REGTRACE_IGNORE_ALL             (REGTRACE_IGNORE_READS  | \
                                         REGTRACE_IGNORE_WRITES | \
                                         REGTRACE_IGNORE_DELAYS)

#define REGTRACE_IS_HANDLING_READS(mask)    (!(mask & REGTRACE_IGNORE_READS))
#define REGTRACE_IS_HANDLING_WRITES(mask)   (!(mask & REGTRACE_IGNORE_WRITES))
#define REGTRACE_IS_HANDLING_DELAYS(mask)   (!(mask & REGTRACE_IGNORE_DELAYS))

// returns true if param is of desired register operation type
#define REGTRACE_IS_WRITE(opType)           (opType & REGTRACE_REGOP_WRITE)
#define REGTRACE_IS_READ(opType)            (opType & REGTRACE_REGOP_READ)
#define REGTRACE_IS_DELAY(opType)           (opType & REGTRACE_REGOP_DELAY)


// possible values for the 'cmd' parameter of a data fetch call
#define REGTRACE_CMD_FLUSH_BUFFER           1
#define REGTRACE_CMD_BEGIN_RECORDING        2
#define REGTRACE_CMD_END_RECORDING          3


// #define MAX_NUM_SW_CLASSES                  16    // limits number of sw classes that can be traced at a time


#endif  // _RMTRACE_H_
