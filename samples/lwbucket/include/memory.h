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
|*  Module: memory.h                                                          *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _MEMORY_H
#define _MEMORY_H

//******************************************************************************
//
//  Constants
//
//******************************************************************************
// Cache types
#define CACHED                          false
#define UNCACHED                        true

//******************************************************************************
//
//  Type Definitions
//
//******************************************************************************
typedef ULONG64     QWORD;

//******************************************************************************
//
//  Functions
//
//******************************************************************************
extern  ULONG           readCpuVirtual(CPU_VIRTUAL vaCpuAddress, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false);
extern  ULONG           readCpuPhysical(CPU_PHYSICAL paCpuAddress, PVOID pBuffer, ULONG ulBufferSize, ULONG ulFlags = DEBUG_PHYSICAL_DEFAULT, bool bUncached = false);
extern  ULONG           writeCpuVirtual(CPU_VIRTUAL vaCpuAddress, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false);
extern  ULONG           writeCpuPhysical(CPU_PHYSICAL paCpuAddress, PVOID pBuffer, ULONG ulBufferSize, ULONG ulFlags = DEBUG_PHYSICAL_DEFAULT, bool bUncached = false);

extern  CHAR            readChar(POINTER ptrAddress, bool bUncached = false);
extern  UCHAR           readUchar(POINTER ptrAddress, bool bUncached = false);
extern  SHORT           readShort(POINTER ptrAddress, bool bUncached = false);
extern  USHORT          readUshort(POINTER ptrAddress, bool bUncached = false);
extern  LONG            readLong(POINTER ptrAddress, bool bUncached = false);
extern  ULONG           readUlong(POINTER ptrAddress, bool bUncached = false);
extern  LONG64          readLong64(POINTER ptrAddress, bool bUncached = false);
extern  ULONG64         readUlong64(POINTER ptrAddress, bool bUncached = false);
extern  float           readFloat(POINTER ptrAddress, bool bUncached = false);
extern  double          readDouble(POINTER ptrAddress, bool bUncached = false);
extern  POINTER         readPointer32(POINTER ptrAddress, bool bUncached = false);
extern  POINTER         readPointer64(POINTER ptrAddress, bool bUncached = false);
extern  POINTER         readPointer(POINTER ptrAddress, bool bUncached = false);
extern  bool            readBoolean(POINTER ptrAddress, bool bUncached = false);
extern  void            readStruct(POINTER ptrAddress, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false);
extern  ULONG64         readBitfield(POINTER ptrAddress, UINT uPosition, UINT uWidth, ULONG ulSize = 0, bool bUncached = false);

inline  void            readBuffer(POINTER ptrAddress, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false)
                            { return readStruct(ptrAddress, pBuffer, ulBufferSize, bUncached); }

inline  BYTE            readByte(POINTER ptrAddress, bool bUncached = false)
                            { return readUchar(ptrAddress, bUncached); }
inline  WORD            readWord(POINTER ptrAddress, bool bUncached = false)
                            { return readUshort(ptrAddress, bUncached); }
inline  DWORD           readDword(POINTER ptrAddress, bool bUncached = false)
                            { return readUlong(ptrAddress, bUncached); }
inline  QWORD           readQword(POINTER ptrAddress, bool bUncached = false)
                            { return readUlong64(ptrAddress, bUncached); }

extern  void            writeChar(POINTER ptrAddress, CHAR charData, bool bUncached = false);
extern  void            writeUchar(POINTER ptrAddress, UCHAR ucharData, bool bUncached = false);
extern  void            writeShort(POINTER ptrAddress, SHORT shortData, bool bUncached = false);
extern  void            writeUshort(POINTER ptrAddress, USHORT ushortData, bool bUncached = false);
extern  void            writeLong(POINTER ptrAddress, LONG longData, bool bUncached = false);
extern  void            writeUlong(POINTER ptrAddress, ULONG ulongData, bool bUncached = false);
extern  void            writeLong64(POINTER ptrAddress, LONG64 long64Data, bool bUncached = false);
extern  void            writeUlong64(POINTER ptrAddress, ULONG64 ulong64Data, bool bUncached = false);
extern  void            writeFloat(POINTER ptrAddress, float floatData, bool bUncached = false);
extern  void            writeDouble(POINTER ptrAddress, double doubleData, bool bUncached = false);
extern  void            writePointer32(POINTER ptrAddress, POINTER pointer32Data, bool bUncached = false);
extern  void            writePointer64(POINTER ptrAddress, POINTER pointer46Data, bool bUncached = false);
extern  void            writePointer(POINTER ptrAddress, POINTER pointerData, bool bUncached = false);
extern  void            writeBoolean(POINTER ptrAddress, bool booleanData, bool bUncached = false);
extern  void            writeStruct(POINTER ptrAddress, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false);
extern  void            writeBitfield(POINTER ptrAddress, ULONG64 ulong64Data, UINT uPosition, UINT uWidth, ULONG ulSize = 0, bool bUncached = false);

inline  void            writeBuffer(POINTER ptrAddress, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false)
                            { return writeStruct(ptrAddress, pBuffer, ulBufferSize, bUncached); }

inline  void            writeByte(POINTER ptrAddress, BYTE byteData, bool bUncached = false)
                            { return writeChar(ptrAddress, byteData, bUncached); }
inline  void            writeWord(POINTER ptrAddress, WORD wordData, bool bUncached = false)
                            { return writeUshort(ptrAddress, wordData, bUncached); }
inline  void            writeDword(POINTER ptrAddress, DWORD dwordData, bool bUncached = false)
                            { return writeUlong(ptrAddress, dwordData, bUncached); }
inline  void            writeQword(POINTER ptrAddress, QWORD qwordData, bool bUncached = false)
                            { return writeUlong64(ptrAddress, qwordData, bUncached); }

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _MEMORY_H
