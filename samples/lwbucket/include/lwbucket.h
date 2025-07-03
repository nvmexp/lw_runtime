 /****************************************************************************\
|*                                                                            *|
|*      Copyright 2016-2017 LWPU Corporation.  All rights reserved.         *|
|*                                                                            *|
|*  NOTICE TO USER:                                                           *|
|*                                                                            *|
|*  This source code is subject to LWPU ownership rights under U.S. and     *|
|*  international Copyright laws.                                             *|
|*                                                                            *|
|*  This software and the information contained herein is PROPRIETARY and     *|
|*  CONFIDENTIAL to LWPU and is being provided under the terms and          *|
|*  conditions of a Non-Disclosure Agreement. Any reproduction or             *|
|*  disclosure to any third party without the express written consent of      *|
|*  LWPU is prohibited.                                                     *|
|*                                                                            *|
|*  LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE       *|
|*  CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR           *|
|*  IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH       *|
|*  REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF           *|
|*  MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR            *|
|*  PURPOSE. IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL,              *|
|*  INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES            *|
|*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN        *|
|*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING       *|
|*  OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE        *|
|*  CODE.                                                                     *|
|*                                                                            *|
|*  U.S. Government End Users. This source code is a "commercial item"        *|
|*  as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting         *|
|*  of "commercial computer software" and "commercial computer software       *|
|*  documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)     *|
|*  and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through          *|
|*  227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the         *|
|*  source code with only those rights set forth herein.                      *|
|*                                                                            *|
|*  Module: lwbucket.h                                                        *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _LWBUCKET_H
#define _LWBUCKET_H

//******************************************************************************
//
//  Debugger Function Definitions
//
//******************************************************************************
#define DEBUGGER_INITIALIZE             extern "C" HRESULT __stdcall
#define DEBUGGER_NOTIFY                 extern "C" void    __stdcall
#define DEBUGGER_UNINITIALIZE           extern "C" void    __stdcall

#define DEBUGGER_KNOWN_OUTPUT           extern "C" HRESULT __stdcall

#define DEBUGGER_COMMAND                extern "C" HRESULT __stdcall

#define DEBUGGER_ANALYZE                extern "C" HRESULT __stdcall

#define EXPORT_HRESULT                  extern "C" HRESULT __stdcall
#define EXPORT_ULONG                    extern "C" ULONG   __stdcall
#define EXPORT_ULONG64                  extern "C" ULONG64 __stdcall

//******************************************************************************
//
//  Constants
//
//******************************************************************************
#define LWEXT_MODULE_NAME               "lwbucket"  // LwBucket extension module name

#define DEBUG_FILE_EXTENSION            ".dll"      // Debugger file extension

#define MAX_SORT                        8           // Maximum sort type values

#define PAGE_SIZE                       4096        // Page size in bytes

#define MAX_COMMAND_STRING              256         // Maximum debugger command string

#define BITS_PER_BYTE                   8           // Bytes <-> bits colwersion

#define EOS                             0x00        // End of string character (NULL)
#define EOL                             0x0a        // End of line character (Line Feed)
#define BLANK                           ' '         // Blank character
#define TAB                             '\t'        // Tab character
#define DASH                            '-'         // Dash character
#define SLASH                           '/'         // Slash character
#define BACKSPACE                       '\b'        // Backspace character
#define BACKSLASH                       '\\'        // Backslash character
#define SINGLE_QUOTE                    '\''        // Single quote character
#define DOUBLE_QUOTE                    '"'         // Double quote character
#define AMPERSAND                       '&'         // Ampersand character
#define ASTERISK                        '*'         // Asterisk character
#define EQUAL                           '='         // Equal character
#define LESS_THAN                       '<'         // Less than character
#define GREATER_THAN                    '>'         // Greater than character
#define PERCENT                         '%'         // Percent character
#define COLON                           ':'         // Colon character
#define SEMI_COLON                      ';'         // Semi-colon character

// Some values to help with pretty printing
#define BYTE_PRINT_WIDTH                (2 + 2)     // Byte print width formatted as 0xXX
#define WORD_PRINT_WIDTH                (2 + 4)     // Word print width formatted as 0xXXXX
#define DWORD_PRINT_WIDTH               (2 + 8)     // Dword print width formatted as 0xXXXXXXXX
#define QWORD_PRINT_WIDTH               (2 + 16)    // Qword print width formatted as 0xXXXXXXXXXXXXXXXX
#define SINGLE_PRINT_WIDTH              (15)        // Float print width formatted as 15.7e
#define DOUBLE_PRINT_WIDTH              (19)        // Double print width formatted as 19.11e

// Define the not found location value for string find
#define NOT_FOUND                       (static_cast<size_t>(-1))

// Define the verbose flags
#define VERBOSE_CACHE_PHYSICAL_LOAD     0x0000000000000001  // Display Cache physical loads
#define VERBOSE_CACHE_PHYSICAL_FLUSH    0x0000000000000002  // Display Cache physical flushes
#define VERBOSE_CACHE_PHYSICAL_EVICT    0x0000000000000004  // Display Cache physical evictions
#define VERBOSE_CACHE_PHYSICAL_CLEAN    0x0000000000000008  // Display Cache physical cleans
#define VERBOSE_CACHE_VIRTUAL_LOAD      0x0000000000000010  // Display Cache virtual loads
#define VERBOSE_CACHE_VIRTUAL_FLUSH     0x0000000000000020  // Display Cache virtual flushes
#define VERBOSE_CACHE_VIRTUAL_EVICT     0x0000000000000040  // Display Cache virtual evictions
#define VERBOSE_CACHE_VIRTUAL_CLEAN     0x0000000000000080  // Display Cache virtual cleans

#define VERBOSE_CACHE_PHYSICAL_READ     0x0000000000000100  // Display Cache physical read
#define VERBOSE_CACHE_PHYSICAL_WRITE    0x0000000000000200  // Display Cache physical writes
#define VERBOSE_CACHE_VIRTUAL_READ      0x0000000000000400  // Display Cache virtual read
#define VERBOSE_CACHE_VIRTUAL_WRITE     0x0000000000000800  // Display Cache virtual writes

#define VERBOSE_CPU_PHYSICAL_READ       0x0000000000001000  // Display CPU physical reads
#define VERBOSE_CPU_PHYSICAL_WRITE      0x0000000000002000  // Display CPU physical writes
#define VERBOSE_CPU_VIRTUAL_READ        0x0000000000004000  // Display CPU virtual reads
#define VERBOSE_CPU_VIRTUAL_WRITE       0x0000000000008000  // Display CPU virtual writes

#define VERBOSE_DBGENG_CLIENT           0x8000000000000000  // Display DbgEng client interface
#define VERBOSE_DBGENG_CONTROL          0x4000000000000000  // Display DbgEng control interface
#define VERBOSE_DBGENG_DATA_SPACES      0x2000000000000000  // Display DbgEng data spaces interface
#define VERBOSE_DBGENG_REGISTERS        0x1000000000000000  // Display DbgEng registers interface
#define VERBOSE_DBGENG_SYMBOLS          0x0800000000000000  // Display DbgEng symbols interface
#define VERBOSE_DBGENG_SYSTEM_OBJECTS   0x0400000000000000  // Display DbgEng system objects interface
#define VERBOSE_DBGENG_ADVANCED         0x0200000000000000  // Display DbgEng advanced interface
#define VERBOSE_DBGENG_SYMBOL_GROUP     0x0080000000000000  // Display DbgEng symbol group interface
#define VERBOSE_DBGENG_BREAKPOINT       0x0040000000000000  // Display DbgEng breakpoint interface

#define VERBOSE_DBGENG_PHYSICAL_READ    0x0008000000000000  // Display DbgEng physical read interfaces
#define VERBOSE_DBGENG_PHYSICAL_WRITE   0x0004000000000000  // Display DbgEng physical write interfaces
#define VERBOSE_DBGENG_VIRTUAL_READ     0x0002000000000000  // Display DbgEng virtual read interfaces
#define VERBOSE_DBGENG_VIRTUAL_WRITE    0x0001000000000000  // Display DbgEng virtual write interfaces

#define VERBOSE_DBGENG_IO_READ          0x0000800000000000  // Display DbgEng I/O read interfaces
#define VERBOSE_DBGENG_IO_WRITE         0x0000400000000000  // Display DbgEng I/O write nterfaces
#define VERBOSE_DBGENG_INPUT            0x0000200000000000  // Display DbgEng input interfaces
#define VERBOSE_DBGENG_OUTPUT           0x0000100000000000  // Display DbgEng output interfaces

#define ILWALID_CPU_VIRTUAL_ADDRESS     0xffffffffffffffff

#define INITIAL_MASK                    0xffffffffffffffff

// Data formats
enum DataFormat
{
    ByteFormat,                             // BYTE data format
    WordFormat,                             // WORD data format
    DwordFormat,                            // DWORD data format
    QwordFormat,                            // QWORD data format
    SingleFormat,                           // Single precision data format (Float)
    DoubleFormat                            // Double precision data format (Float)

}; // DataFormat

//
//  Define the machine types until we move to a newer WDK
//
#define IMAGE_FILE_MACHINE_UNKNOWN           0
#define IMAGE_FILE_MACHINE_I386              0x014c  // Intel 386.
#define IMAGE_FILE_MACHINE_R3000             0x0162  // MIPS little-endian, 0x160 big-endian
#define IMAGE_FILE_MACHINE_R4000             0x0166  // MIPS little-endian
#define IMAGE_FILE_MACHINE_R10000            0x0168  // MIPS little-endian
#define IMAGE_FILE_MACHINE_WCEMIPSV2         0x0169  // MIPS little-endian WCE v2
#define IMAGE_FILE_MACHINE_ALPHA             0x0184  // Alpha_AXP
#define IMAGE_FILE_MACHINE_SH3               0x01a2  // SH3 little-endian
#define IMAGE_FILE_MACHINE_SH3DSP            0x01a3
#define IMAGE_FILE_MACHINE_SH3E              0x01a4  // SH3E little-endian
#define IMAGE_FILE_MACHINE_SH4               0x01a6  // SH4 little-endian
#define IMAGE_FILE_MACHINE_SH5               0x01a8  // SH5
#define IMAGE_FILE_MACHINE_ARM               0x01c0  // ARM Little-Endian
#define IMAGE_FILE_MACHINE_THUMB             0x01c2  // ARM Thumb/Thumb-2 Little-Endian
#define IMAGE_FILE_MACHINE_ARMNT             0x01c4  // ARM Thumb-2 Little-Endian
#define IMAGE_FILE_MACHINE_AM33              0x01d3
#define IMAGE_FILE_MACHINE_POWERPC           0x01F0  // IBM PowerPC Little-Endian
#define IMAGE_FILE_MACHINE_POWERPCFP         0x01f1
#define IMAGE_FILE_MACHINE_IA64              0x0200  // Intel 64
#define IMAGE_FILE_MACHINE_MIPS16            0x0266  // MIPS
#define IMAGE_FILE_MACHINE_ALPHA64           0x0284  // ALPHA64
#define IMAGE_FILE_MACHINE_MIPSFPU           0x0366  // MIPS
#define IMAGE_FILE_MACHINE_MIPSFPU16         0x0466  // MIPS
#define IMAGE_FILE_MACHINE_AXP64             IMAGE_FILE_MACHINE_ALPHA64
#define IMAGE_FILE_MACHINE_TRICORE           0x0520  // Infineon
#define IMAGE_FILE_MACHINE_CEF               0x0CEF
#define IMAGE_FILE_MACHINE_EBC               0x0EBC  // EFI Byte Code
#define IMAGE_FILE_MACHINE_AMD64             0x8664  // AMD64 (K8)
#define IMAGE_FILE_MACHINE_M32R              0x9041  // M32R little-endian
#define IMAGE_FILE_MACHINE_CEE               0xC0EE

// LW Status codes stolen from lwstatuscodes.h
#define LW_OK                                   (0x00000000)
#define LW_ERR_GENERIC                          (0x0000ffff)

#define LW_ERR_BROKEN_FB                        (0x00000001)
#define LW_ERR_BUFFER_TOO_SMALL                 (0x00000002)
#define LW_ERR_BUSY_RETRY                       (0x00000003)
#define LW_ERR_CALLBACK_NOT_SCHEDULED           (0x00000004)
#define LW_ERR_CARD_NOT_PRESENT                 (0x00000005)
#define LW_ERR_CYCLE_DETECTED                   (0x00000006)
#define LW_ERR_DMA_IN_USE                       (0x00000007)
#define LW_ERR_DMA_MEM_NOT_LOCKED               (0x00000008)
#define LW_ERR_DMA_MEM_NOT_UNLOCKED             (0x00000009)
#define LW_ERR_DUAL_LINK_INUSE                  (0x0000000a)
#define LW_ERR_ECC_ERROR                        (0x0000000b)
#define LW_ERR_FIFO_BAD_ACCESS                  (0x0000000c)
#define LW_ERR_FREQ_NOT_SUPPORTED               (0x0000000d)
#define LW_ERR_GPU_DMA_NOT_INITIALIZED          (0x0000000e)
#define LW_ERR_GPU_IS_LOST                      (0x0000000f)
#define LW_ERR_GPU_IN_FULLCHIP_RESET            (0x00000010)
#define LW_ERR_GPU_NOT_FULL_POWER               (0x00000011)
#define LW_ERR_GPU_UUID_NOT_FOUND               (0x00000012)
#define LW_ERR_HOT_SWITCH                       (0x00000013)
#define LW_ERR_I2C_ERROR                        (0x00000014)
#define LW_ERR_I2C_SPEED_TOO_HIGH               (0x00000015)
#define LW_ERR_ILLEGAL_ACTION                   (0x00000016)
#define LW_ERR_IN_USE                           (0x00000017)
#define LW_ERR_INFLATE_COMPRESSED_DATA_FAILED   (0x00000018)
#define LW_ERR_INSERT_DUPLICATE_NAME            (0x00000019)
#define LW_ERR_INSUFFICIENT_RESOURCES           (0x0000001a)
#define LW_ERR_INSUFFICIENT_PERMISSIONS         (0x0000001b)
#define LW_ERR_INSUFFICIENT_POWER               (0x0000001c)
#define LW_ERR_ILWALID_ACCESS_TYPE              (0x0000001d)
#define LW_ERR_ILWALID_ADDRESS                  (0x0000001e)
#define LW_ERR_ILWALID_ARGUMENT                 (0x0000001f)
#define LW_ERR_ILWALID_BASE                     (0x00000020)
#define LW_ERR_ILWALID_CHANNEL                  (0x00000021)
#define LW_ERR_ILWALID_CLASS                    (0x00000022)
#define LW_ERR_ILWALID_CLIENT                   (0x00000023)
#define LW_ERR_ILWALID_COMMAND                  (0x00000024)
#define LW_ERR_ILWALID_DATA                     (0x00000025)
#define LW_ERR_ILWALID_DEVICE                   (0x00000026)
#define LW_ERR_ILWALID_DMA_SPECIFIER            (0x00000027)
#define LW_ERR_ILWALID_EVENT                    (0x00000028)
#define LW_ERR_ILWALID_FLAGS                    (0x00000029)
#define LW_ERR_ILWALID_FUNCTION                 (0x0000002a)
#define LW_ERR_ILWALID_HEAP                     (0x0000002b)
#define LW_ERR_ILWALID_INDEX                    (0x0000002c)
#define LW_ERR_ILWALID_IRQ_LEVEL                (0x0000002d)
#define LW_ERR_ILWALID_LIMIT                    (0x0000002e)
#define LW_ERR_ILWALID_LOCK_STATE               (0x0000002f)
#define LW_ERR_ILWALID_METHOD                   (0x00000030)
#define LW_ERR_ILWALID_OBJECT                   (0x00000031)
#define LW_ERR_ILWALID_OBJECT_BUFFER            (0x00000032)
#define LW_ERR_ILWALID_OBJECT_HANDLE            (0x00000033)
#define LW_ERR_ILWALID_OBJECT_NEW               (0x00000034)
#define LW_ERR_ILWALID_OBJECT_OLD               (0x00000035)
#define LW_ERR_ILWALID_OBJECT_PARENT            (0x00000036)
#define LW_ERR_ILWALID_OFFSET                   (0x00000037)
#define LW_ERR_ILWALID_OPERATION                (0x00000038)
#define LW_ERR_ILWALID_OWNER                    (0x00000039)
#define LW_ERR_ILWALID_PARAM_STRUCT             (0x0000003a)
#define LW_ERR_ILWALID_PARAMETER                (0x0000003b)
#define LW_ERR_ILWALID_PATH                     (0x0000003c)
#define LW_ERR_ILWALID_POINTER                  (0x0000003d)
#define LW_ERR_ILWALID_REGISTRY_KEY             (0x0000003e)
#define LW_ERR_ILWALID_REQUEST                  (0x0000003f)
#define LW_ERR_ILWALID_STATE                    (0x00000040)
#define LW_ERR_ILWALID_STRING_LENGTH            (0x00000041)
#define LW_ERR_ILWALID_READ                     (0x00000042)
#define LW_ERR_ILWALID_WRITE                    (0x00000043)
#define LW_ERR_ILWALID_XLATE                    (0x00000044)
#define LW_ERR_IRQ_NOT_FIRING                   (0x00000045)
#define LW_ERR_IRQ_EDGE_TRIGGERED               (0x00000046)
#define LW_ERR_MEMORY_TRAINING_FAILED           (0x00000047)
#define LW_ERR_MISMATCHED_SLAVE                 (0x00000048)
#define LW_ERR_MISMATCHED_TARGET                (0x00000049)
#define LW_ERR_MISSING_TABLE_ENTRY              (0x0000004a)
#define LW_ERR_MODULE_LOAD_FAILED               (0x0000004b)
#define LW_ERR_MORE_DATA_AVAILABLE              (0x0000004c)
#define LW_ERR_MORE_PROCESSING_REQUIRED         (0x0000004d)
#define LW_ERR_MULTIPLE_MEMORY_TYPES            (0x0000004e)
#define LW_ERR_NO_FREE_FIFOS                    (0x0000004f)
#define LW_ERR_NO_INTR_PENDING                  (0x00000050)
#define LW_ERR_NO_MEMORY                        (0x00000051)
#define LW_ERR_NO_SUCH_DOMAIN                   (0x00000052)
#define LW_ERR_NO_VALID_PATH                    (0x00000053)
#define LW_ERR_NOT_COMPATIBLE                   (0x00000054)
#define LW_ERR_NOT_READY                        (0x00000055)
#define LW_ERR_NOT_SUPPORTED                    (0x00000056)
#define LW_ERR_OBJECT_NOT_FOUND                 (0x00000057)
#define LW_ERR_OBJECT_TYPE_MISMATCH             (0x00000058)
#define LW_ERR_OPERATING_SYSTEM                 (0x00000059)
#define LW_ERR_OTHER_DEVICE_FOUND               (0x0000005a)
#define LW_ERR_OUT_OF_RANGE                     (0x0000005b)
#define LW_ERR_OVERLAPPING_UVM_COMMIT           (0x0000005c)
#define LW_ERR_PAGE_TABLE_NOT_AVAIL             (0x0000005d)
#define LW_ERR_PID_NOT_FOUND                    (0x0000005e)
#define LW_ERR_PROTECTION_FAULT                 (0x0000005f)
#define LW_ERR_RC_ERROR                         (0x00000060)
#define LW_ERR_REJECTED_VBIOS                   (0x00000061)
#define LW_ERR_RESET_REQUIRED                   (0x00000062)
#define LW_ERR_STATE_IN_USE                     (0x00000063)
#define LW_ERR_SIGNAL_PENDING                   (0x00000064)
#define LW_ERR_TIMEOUT                          (0x00000065)
#define LW_ERR_TIMEOUT_RETRY                    (0x00000066)
#define LW_ERR_TOO_MANY_PRIMARIES               (0x00000067)
#define LW_ERR_UVM_ADDRESS_IN_USE               (0x00000068)
#define LW_ERR_MAX_SESSION_LIMIT_REACHED        (0x00000069)
#define LW_ERR_LIB_RM_VERSION_MISMATCH          (0x0000006a)

// Warnings:
#define LW_WARN_HOT_SWITCH                      (0x00010001)
#define LW_WARN_INCORRECT_PERFMON_DATA          (0x00010002)
#define LW_WARN_MISMATCHED_SLAVE                (0x00010003)
#define LW_WARN_MISMATCHED_TARGET               (0x00010004)
#define LW_WARN_MORE_PROCESSING_REQUIRED        (0x00010005)
#define LW_WARN_NOTHING_TO_DO                   (0x00010006)
#define LW_WARN_NULL_OBJECT                     (0x00010007)
#define LW_WARN_OUT_OF_RANGE                    (0x00010008)

// Some definitions defined by LwWatch debugger extension
#define LW_ERROR                                LW_ERR_GENERIC
#define LW_NOT_SUPPORTED                        LW_ERR_NOT_SUPPORTED
#define LW_RETRY                                LW_ERR_BUSY_RETRY

//******************************************************************************
//
//  Regular Expressions
//
//******************************************************************************
#define EXTEXPR         "Extension DLL chain"

#define IMGEXPR         "^[ \t]*(.+): .*$"
#define IMG_FILENAME    1   

#define PATHEXPR        "^[ \t]*\\[path: (.+)\\].*$"
#define PATH_FILENAME   1

#define FILEEXPR        "^(([\\]{2}([A-Za-z0-9#$_+=-]+)[\\]{1}([A-Za-z0-9#$_+=-]+))|([A-Za-z]{1}:)?)?((([.]{1,2}|[^\\/:*?\"<>|\t]+)?[\\]{1})?(([.]{1,2}|[^\\/:*?\"<>|\t]+){1}[\\]{1})*)(([^\\/:*?\"<>|\t.]+)\\.?([^\\/:*?\"<>|\t.]*))$"
#define FILE_ROOT       1
#define FILE_UNC        2
#define FILE_SERVER     3
#define FILE_SHARE      4
#define FILE_DRIVE      5
#define FILE_PATH       6
#define FILE_FILENAME   11
#define FILE_FILE       12
#define FILE_EXTENSION  13

//******************************************************************************
//
//  Type Definitions
//
//******************************************************************************
typedef ULONG64     QWORD;
typedef ULONG       RMHANDLE;

typedef HRESULT     (*PFN_DISPLAY_KNOWN)(ULONG64 Address, PSTR Buffer, PULONG BufferSize);

//******************************************************************************
//
//  Fowards
//
//******************************************************************************
namespace sym
{
class CEnum;
class CType;
class CField;
class CMember;
}
using sym::CEnum;
using sym::CType;
using sym::CField;
using sym::CMember;

class CString;

//******************************************************************************
//
//  Macros
//
//******************************************************************************
#define DebugClient(pDbgClient)     CDebugClient debugClient(pDbgClient)

#define THROW_ON_FAIL(Expression)   { HRESULT hResult = (Expression); if (FAILED(hResult)) { throw CException(hResult, #Expression); } }

// Generate the correct extension command string (Based on extension name)
#define LWEXT_COMMAND(String)       (LWEXT_MODULE_NAME "." String)

// Generate the correct module name string
#define MODULE(String)              (String "!")

// Colwert non-standard boolean types to standard bool
#define tobool(Value)               (!(!(Value)))

// Colwert integer units to floating point w/given factor value
#define doubleunits(Units, Factor)  (static_cast<double>(Units) / static_cast<double>(Factor))
#define floatunits(Units, Factor)   (static_cast<float>(doubleunits((Units), (Factor))))

// Colwert fraction values to floating point
#define tosingle(Whole, Fraction, Precision)  (static_cast<float>(Whole) + (static_cast<float>(Fraction) / (1L << (Precision))))
#define todouble(Whole, Fraction, Precision)  (static_cast<double>(Whole) + (static_cast<double>(Fraction) / (1LL << (Precision))))

#define fixedsingle(Value, Precision) tosingle(0, Value, Precision)
#define fixeddouble(Value, Precision) todouble(0, Value, Precision)

// Align a value to ceil or floor
#define alignceil(Value, Alignment) ((((Value) + (Alignment) - 1) / (Alignment)) * (Alignment))
#define alignfloor(Value, Alignment)(((Value) / (Alignment)) * (Alignment))

// Check for power of two (Single bit set)
#define poweroftwo(Value)           (!tobool((Value) & ((Value) - 1)))

// Macros to extract subsections of larger data fields
#define lodword(Value)              (static_cast<QWORD>(Value) & 0x00000000ffffffffLL)
#define hidword(Value)              (static_cast<QWORD>(Value) >> 32)

#define loword(Value)               (static_cast<DWORD>(Value) & 0x0000ffff)
#define hiword(Value)               (static_cast<DWORD>(Value) >> 16)

#define lobyte(Value)               (static_cast<WORD>(Value) & 0x00ff)
#define hibyte(Value)               (static_cast<WORD>(Value) >> 8)

// Macros to help format colwersions
#define voidptr(Pointer)            (static_cast<VOID*>(Pointer))

#define charptr(Pointer)            (static_cast<CHAR*>(voidptr(Pointer)))
#define ucharptr(Pointer)           (static_cast<UCHAR*>(voidptr(Pointer)))
#define wcharptr(Pointer)           (static_cast<WCHAR*>(voidptr(Pointer)))
#define uwcharptr(Pointer)          (static_cast<UWCHAR*>(voidptr(Pointer)))
#define intptr(Pointer)             (static_cast<INT*>(voidptr(Pointer)))
#define uintptr(Pointer)            (static_cast<UINT*>(voidptr(Pointer)))
#define shortptr(Pointer)           (static_cast<SHORT*>(voidptr(Pointer)))
#define ushortptr(Pointer)          (static_cast<USHORT*>(voidptr(Pointer)))
#define longptr(Pointer)            (static_cast<LONG*>(voidptr(Pointer)))
#define ulongptr(Pointer)           (static_cast<ULONG*>(voidptr(Pointer)))
#define longlongptr(Pointer)        (static_cast<LONGLONG*>(voidptr(Pointer)))
#define ulonglongptr(Pointer)       (static_cast<ULONGLONG*>(voidptr(Pointer)))
#define floatptr(Pointer)           (static_cast<float*>(voidptr(Pointer)))
#define doubleptr(Pointer)          (static_cast<double*>(voidptr(Pointer)))
#define boolptr(Pointer)            (static_cast<bool*>(voidptr(Pointer)))

#define byteptr(Pointer)            (static_cast<BYTE*>(voidptr(Pointer)))
#define wordptr(Pointer)            (static_cast<WORD*>(voidptr(Pointer)))
#define dwordptr(Pointer)           (static_cast<DWORD*>(voidptr(Pointer)))
#define qwordptr(Pointer)           (static_cast<QWORD*>(voidptr(Pointer)))

#define constvoidptr(Pointer)       (static_cast<const VOID*>(Pointer))

#define constcharptr(Pointer)       (static_cast<const CHAR*>(constvoidptr(Pointer)))
#define constucharptr(Pointer)      (static_cast<const UCHAR*>(constvoidptr(Pointer)))
#define constwcharptr(Pointer)      (static_cast<const WCHAR*>(constvoidptr(Pointer)))
#define constuwcharptr(Pointer)     (static_cast<const UWCHAR*>(constvoidptr(Pointer)))
#define constintptr(Pointer)        (static_cast<const INT*>(constvoidptr(Pointer)))
#define constuintptr(Pointer)       (static_cast<const UINT*>(constvoidptr(Pointer)))
#define constshortptr(Pointer)      (static_cast<const SHORT*>(constvoidptr(Pointer)))
#define constushortptr(Pointer)     (static_cast<const USHORT*>(constvoidptr(Pointer)))
#define constlongptr(Pointer)       (static_cast<const LONG*>(constvoidptr(Pointer)))
#define constulongptr(Pointer)      (static_cast<const ULONG*>(constvoidptr(Pointer)))
#define constlonglongptr(Pointer)   (static_cast<const LONGLONG*>(constvoidptr(Pointer)))
#define constulonglongptr(Pointer)  (static_cast<const ULONGLONG*>(constvoidptr(Pointer)))
#define constfloatptr(Pointer)      (static_cast<const float*>(constvoidptr(Pointer)))
#define constdoubleptr(Pointer)     (static_cast<const double*>(constvoidptr(Pointer)))
#define constboolptr(Pointer)       (static_cast<const bool*>(constvoidptr(Pointer)))

#define constbyteptr(Pointer)       (static_cast<const BYTE*>(constvoidptr(Pointer)))
#define constwordptr(Pointer)       (static_cast<const WORD*>(constvoidptr(Pointer)))
#define constdwordptr(Pointer)      (static_cast<const DWORD*>(constvoidptr(Pointer)))
#define constqwordptr(Pointer)      (static_cast<const QWORD*>(constvoidptr(Pointer)))

#define pvoidptr(Pointer)           (static_cast<PVOID*>(voidptr(Pointer)))

#define funcptr(Function, Pointer)  (static_cast<Function>(voidptr(Pointer)))

#define reftype(Type, Pointer)      (static_cast<Type&>(*static_cast<Type*>(voidptr(Pointer))))

// Define the member function prefix values
#define publicPrefix    
#define protectedPrefix             _
#define privatePrefix               __

// Define the member type method values
#define getBYTEMethod               getByte
#define getWORDMethod               getWord
#define getDWORDMethod              getDword
#define getQWORDMethod              getQword
#define getCHARMethod               getChar
#define getUCHARMethod              getUchar

#define getSHORTMethod              getShort
#define getUSHORTMethod             getUshort
#define getLONGMethod               getLong
#define getULONGMethod              getUlong
#define getLONG64Method             getLong64
#define getULONG64Method            getUlong64
#define getfloatMethod              getFloat
#define getdoubleMethod             getDouble
#define getboolMethod               getBoolean
#define getPVOIDMethod              getStruct

#define getPOINTERMethod            getPointer

#define getCPU_PHYSICALMethod       getUlong64
#define getCPU_VIRTUALMethod        getPointer

#define getTHREADMethod             getPointer
#define getPROCESSMethod            getPointer

// Macros to help generate member function names
#define CONCAT(A, B)        A##B
#define EXPCONCAT(A, B)     CONCAT(A, B)

#define MEMBER_PREFIX(Access, Name) EXPCONCAT(Access##Prefix, Name)
#define MEMBER_METHOD(Type)         EXPCONCAT(get, Type##Method)

// Macros to assist class type, enum, field, and member generation
#define TYPE(Name)                                                              \
private:                                                                        \
static  CMemberType         m_##Name##Type;                                     \
public:                                                                         \
static  const CMemberType&  Name##Type()            { return m_##Name##Type; }

#define ENUM(Name)                                                              \
private:                                                                        \
static  CEnum               m_##Name##Enum;                                     \
public:                                                                         \
static  const CEnum&        Name##Enum()            { return m_##Name##Enum; }

#define FIELD(Name)                                                             \
private:                                                                        \
static  CMemberField        m_##Name##Field;                                    \
public:                                                                         \
static  const CMemberField& Name##Field()           { return m_##Name##Field; }

#define MEMBER(Name, Type, Default, Access)                                     \
private:                                                                        \
        CMember             m_##Name##Member;                                   \
Access:                                                                         \
const   CMember&            MEMBER_PREFIX(Access, Name##Member)() const         \
                                { return m_##Name##Member; }                    \
        Type                MEMBER_PREFIX(Access, Name)(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const \
                                { return (MEMBER_PREFIX(Access, Name##Member)().isPresent() ? MEMBER_PREFIX(Access, Name##Member)().##MEMBER_METHOD(Type)(uIndex1, uIndex2, uIndex3, uIndex4) : Default); }

#define INIT(Name)                                                              \
        m_##Name##Member(&m_##Name##Field)

#define READ(Name, Address)                                                     \
        m_##Name##Member.readData(Address)

#define WRITE(Name, Address)                                                    \
        m_##Name##Member.writeData(Address)

#define SET(Name, Pointer)                                                      \
        m_##Name##Member.setData(Pointer)

#define CLEAR(Name)                                                             \
        m_##Name##Member.clearData()

#define VERBOSE_LEVEL(Level)                        ((commandValue(VerboseOption) & (Level)) != 0)

//******************************************************************************
//
//  Structures
//
//******************************************************************************
typedef struct _KNOWN_STRUCTS
{
    char*               pStructName;        // Pointer to known structure name
    BOOL                bSuppressName;      // TRUE to suppress known structure name
    PFN_DISPLAY_KNOWN   pfnDisplayKnown;    // Function to display known structure

}  KNOWN_STRUCTS, *PKNOWN_STRUCTS;

//******************************************************************************
//
// class CEffectiveProcessor
//
// Class for dealing with the effective processor mode
//
//******************************************************************************
class CEffectiveProcessor
{
private:
        ULONG           m_ulEffectiveProcessor;

public:
                        CEffectiveProcessor(ULONG ulEffectiveProcessor = IMAGE_FILE_MACHINE_UNKNOWN);
                       ~CEffectiveProcessor();

        ULONG           effectiveProcessor() const  { return m_ulEffectiveProcessor; }

}; // class CEffectiveProcessor

//******************************************************************************
//
// class CThreadStorage
//
// Class for dealing with thread local storage
//
//******************************************************************************
class CThreadStorage
{
private:
        CThreadStorage* m_pPrevThreadStorage;       // Pointer to previous thread storage
        CThreadStorage* m_pNextThreadStorage;       // Pointer to next thread storage

        ULONG           m_ulStorageSize;
        ULONG           m_ulStorageOffset;

        void            addThreadStorage(CThreadStorage* pThreadStorage);

public:
                        CThreadStorage(ULONG ulStorageSize);
                       ~CThreadStorage();

        ULONG           storageSize() const         { return m_ulStorageSize; }
        ULONG           storageOffset() const       { return m_ulStorageOffset; }

        void*           threadStorage() const;
        ULONG           totalSize() const;

        CThreadStorage* prevThreadStorage() const   { return m_pPrevThreadStorage; }
        CThreadStorage* nextThreadStorage() const   { return m_pNextThreadStorage; }

}; // class CThreadStorage

//******************************************************************************
//
//  Template for computing array sizes
//
//******************************************************************************
template <typename T, size_t N>
inline size_t
countof(const T (&array)[N])
{
    UNREFERENCED_PARAMETER(array);

    return N;

} // countof

//******************************************************************************
//
//  Functions
//
//******************************************************************************
extern  HRESULT             initializeGlobals(bool bThrow = true);
extern  void                initializeArguments(PCSTR args, char *pCommand, int *argc, char **argv);

extern  ULONG               pointerSize();
extern  ULONG64             pointerMask();
extern  ULONG               pointerWidth();

extern  bool                isInitialized();

extern  bool                isActive();
extern  bool                isAccessible();
extern  bool                isConnected();

extern  ULONG               debugClass();
extern  ULONG               debugQualifer();
extern  ULONG               actualMachine();
extern  ULONG               exelwtingMachine();

extern  bool                isMachine32Bit(ULONG ulMachine);
extern  bool                isMachine64Bit(ULONG ulMachine);
extern  bool                is32Bit();
extern  bool                is64Bit();
extern  bool                is32on64();
extern  bool                isUserMode();
extern  bool                isKernelMode();
extern  bool                isDumpFile();

extern  ULONG64             debuggerVersion();
extern  ULONG               debuggerMajorVersion();
extern  ULONG               debuggerMinorVersion();
extern  ULONG               debuggerReleaseNumber();
extern  ULONG               debuggerBuildNumber();
extern  bool                isDebuggerVersion(ULONG ulMajorVersion, ULONG ulMinorVersion, ULONG ulReleaseNumber = 0, ULONG ulBuildNumber = 0);

extern  bool                isKernelModeAddress(POINTER ptrAddress);

extern  HRESULT             breakCheck(HRESULT hResult);
extern  HRESULT             statusCheck();

extern  bool                userBreak(HRESULT hResult);
extern  bool                ignoreError(HRESULT hResult);

extern  CString             buildDbgCommand(const CString& sString);
extern  CString             buildDbgCommand(const CString& sString, const CString& sOptions);
extern  CString             buildDotCommand(const CString& sString);
extern  CString             buildDotCommand(const CString& sString, const CString& sOptions);
extern  CString             buildExtCommand(const CString& sString);
extern  CString             buildExtCommand(const CString& sString, const CString& sOptions);
extern  CString             buildModCommand(const CString& sString, const CString& sModule);
extern  CString             buildModCommand(const CString& sString, const CString& sModule, const CString& sOptions);
extern  CString             buildModPathCommand(const CString& sString, const CString& path, const CString& sModule);
extern  CString             buildModPathCommand(const CString& sString, const CString& path, const CString& sModule, const CString& sOptions);

extern  CString             buildLwExtCommand(const CString& sString);
extern  CString             buildLwExtCommand(const CString& sString, const CString& sOptions);

extern  const char*         lwExtPrefix();
extern  const char*         lwExtPath();

extern  HANDLE              getModuleHandle();

extern  char*               terminate(char* pBuffer, char* pStart, ULONG ulSize);

extern  HRESULT             getExtensionInfo(const CString& sModule, char* pPrefix, char* pPath);

extern  void                updatePointerSize();

extern  ULONG               dataSize(DataFormat dataFormat);

extern  void* _cdecl        operator new(size_t size, const char* pFunction, const char* pFile, int nLine);
extern  void* _cdecl        operator new[](size_t size, const char* pFunction, const char* pFile, int nLine);
extern  void  _cdecl        operator delete(void* pMemory, const char* pFunction, const char* pFile, int nLine);
extern  void  _cdecl        operator delete[](void* pMemory, const char* pFunction, const char* pFile, int nLine);

// Override new to provide file name and line number for allocation failures
#define new                 new (__FUNCTION__, __FILE__, __LINE__)

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _LWBUCKET_H
