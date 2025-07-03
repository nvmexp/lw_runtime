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
|*  Module: memory.cpp                                                        *|
|*                                                                            *|
 \****************************************************************************/
#include "precomp.h"

//******************************************************************************
//
// Locals
//
//******************************************************************************




//******************************************************************************

ULONG
readCpuVirtual
(
    CPU_VIRTUAL         vaCpuAddress,
    PVOID               pBuffer,
    ULONG               ulBufferSize,
    bool                bUncached
)
{
    HRESULT             hResult;
    ULONG               ulReadSize = 0;

    assert(pBuffer != NULL);

    // Make sure address doesn't exceed target size
    assert((vaCpuAddress & ~pointerMask()) == 0);

    // Check for cached vs. uncached read
    if (bUncached)
    {
        // Display CPU virtual read if requested
        if (VERBOSE_LEVEL(VERBOSE_CPU_VIRTUAL_READ))
        {
            if (dmlState())
            {
                dPrintf(bold("CPU: "));
            }
            else    // Plain text only
            {
                dPrintf("CPU: ");
            }
            dPrintf("Uncached virtual read from 0x%0*I64x - 0x%0*I64x\n",
                     ADDR(vaCpuAddress), pointerWidth(), (vaCpuAddress.addr() + ulBufferSize - 1));
        }
        // Try to read the requested data (Uncached) [Sign extended]
        hResult = ReadVirtualUncached(vaCpuAddress.address(), pBuffer, ulBufferSize, &ulReadSize);
        if (FAILED(hResult))
        {
            throw CTargetException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error reading CPU virtual memory at address 0x%0*I64x (%d bytes)",
                                   ADDR(vaCpuAddress), ulBufferSize);
        }
    }
    else    // Cached
    {
        // Display CPU virtual read if requested
        if (VERBOSE_LEVEL(VERBOSE_CPU_VIRTUAL_READ))
        {
            if (dmlState())
            {
                dPrintf(bold("CPU: "));
            }
            else    // Plain text only
            {
                dPrintf("CPU: ");
            }
            dPrintf("Cached virtual read from 0x%0*I64x - 0x%0*I64x\n",
                     ADDR(vaCpuAddress), pointerWidth(), (vaCpuAddress.addr() + ulBufferSize - 1));
        }
        // Try to read the requested data (Cached) [Sign extended]
        hResult = ReadVirtual(vaCpuAddress.address(), pBuffer, ulBufferSize, &ulReadSize);
        if (FAILED(hResult))
        {
            throw CTargetException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error reading CPU virtual memory at address 0x%0*I64x (%d bytes)",
                                   ADDR(vaCpuAddress), ulBufferSize);
        }
    }
    return ulReadSize;

} // readCpuVirtual

//******************************************************************************

ULONG
readCpuPhysical
(
    CPU_PHYSICAL        paCpuAddress,
    PVOID               pBuffer,
    ULONG               ulBufferSize,
    ULONG               ulFlags,
    bool                bUncached
)
{
    UNREFERENCED_PARAMETER(bUncached);

    HRESULT             hResult;
    ULONG               ulReadSize = 0;

    assert(pBuffer != NULL);

    // Check for a dump file (Caching attributes only supported for live debugging)
    if (isDumpFile())
    {
        // Override cache flags if a dump file (Only DEBUG_PHYSICAL_DEFAULT supported)
        ulFlags = DEBUG_PHYSICAL_DEFAULT;
    }
    // Check for cached vs. uncached read
    if (bUncached)
    {
        // Display CPU physical read if requested
        if (VERBOSE_LEVEL(VERBOSE_CPU_PHYSICAL_READ))
        {
            if (dmlState())
            {
                dPrintf(bold("CPU: "));
            }
            else    // Plain text only
            {
                dPrintf("CPU: ");
            }
            dPrintf("Uncached physical read from 0x%016I64x - 0x%016I64x\n",
                     paCpuAddress.addr(), (paCpuAddress.addr() + ulBufferSize - 1));
        }
        // Check for a dump file (Caching attributes only supported for live debugging)
        if (isDumpFile())
        {
            // Override cache flags if a dump file (Only DEBUG_PHYSICAL_DEFAULT supported)
            ulFlags = DEBUG_PHYSICAL_DEFAULT;
        }
        // Try to read the requested data
        hResult = ReadPhysical2(paCpuAddress.addr(), ulFlags, pBuffer, ulBufferSize, &ulReadSize);
        if (FAILED(hResult))
        {
            throw CTargetException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error reading CPU physical memory at address 0x%016I64x (%d bytes)",
                                   paCpuAddress.addr(), ulBufferSize);
        }
    }
    else    // Cached
    {
        // Display CPU physical read if requested
        if (VERBOSE_LEVEL(VERBOSE_CPU_PHYSICAL_READ))
        {
            if (dmlState())
            {
                dPrintf(bold("CPU: "));
            }
            else    // Plain text only
            {
                dPrintf("CPU: ");
            }
            dPrintf("Cached physical read from 0x%016I64x - 0x%016I64x\n",
                     paCpuAddress.addr(), (paCpuAddress.addr() + ulBufferSize - 1));
        }
        // Check for a dump file (Caching attributes only supported for live debugging)
        if (isDumpFile())
        {
            // Override cache flags if a dump file (Only DEBUG_PHYSICAL_DEFAULT supported)
            ulFlags = DEBUG_PHYSICAL_DEFAULT;
        }
        // Try to read the requested data
        hResult = ReadPhysical2(paCpuAddress.addr(), ulFlags, pBuffer, ulBufferSize, &ulReadSize);
        if (FAILED(hResult))
        {
            throw CTargetException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error reading CPU physical memory at address 0x%016I64x (%d bytes)",
                                   paCpuAddress.addr(), ulBufferSize);
        }
    }
    return ulReadSize;

} // readCpuPhysical

//******************************************************************************

ULONG
writeCpuVirtual
(
    CPU_VIRTUAL         vaCpuAddress,
    PVOID               pBuffer,
    ULONG               ulBufferSize,
    bool                bUncached
)
{
    HRESULT             hResult;
    ULONG               ulWriteSize = 0;

    assert(pBuffer != NULL);

    // Make sure address doesn't exceed target size
    assert((vaCpuAddress & ~pointerMask()) == 0);

    // Check for cached vs. uncached write
    if (bUncached)
    {
        // Display CPU virtual write if requested
        if (VERBOSE_LEVEL(VERBOSE_CPU_VIRTUAL_WRITE))
        {
            if (dmlState())
            {
                dPrintf(bold("CPU: "));
            }
            else    // Plain text only
            {
                dPrintf("CPU: ");
            }
            dPrintf("Uncached virtual write to 0x%0*I64x - 0x%0*I64x\n",
                     ADDR(vaCpuAddress), pointerWidth(), (vaCpuAddress.addr() + ulBufferSize - 1));
        }
        // Try to write the requested data (Uncached) [Sign extended]
        hResult = WriteVirtualUncached(vaCpuAddress.address(), pBuffer, ulBufferSize, &ulWriteSize);
        if (FAILED(hResult))
        {
            throw CTargetException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error writing CPU virtual memory at address 0x%0*I64x (%d bytes)",
                                   ADDR(vaCpuAddress), ulBufferSize);
        }
    }
    else    // Cached
    {
        // Display CPU virtual write if requested
        if (VERBOSE_LEVEL(VERBOSE_CPU_VIRTUAL_WRITE))
        {
            if (dmlState())
            {
                dPrintf(bold("CPU: "));
            }
            else    // Plain text only
            {
                dPrintf("CPU: ");
            }
            dPrintf("Cached virtual write to 0x%0*I64x - 0x%0*I64x\n",
                     ADDR(vaCpuAddress), pointerWidth(), (vaCpuAddress.addr() + ulBufferSize - 1));
        }
        // Try to write the requested data (Cached) [Sign extended]
        hResult = WriteVirtual(vaCpuAddress.address(), pBuffer, ulBufferSize, &ulWriteSize);
        if (FAILED(hResult))
        {
            throw CTargetException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error writing CPU virtual memory at address 0x%0*I64x (%d bytes)",
                                   ADDR(vaCpuAddress), ulBufferSize);
        }
    }
    return ulWriteSize;

} // writeCpuVirtual

//******************************************************************************

ULONG
writeCpuPhysical
(
    CPU_PHYSICAL        paCpuAddress,
    PVOID               pBuffer,
    ULONG               ulBufferSize,
    ULONG               ulFlags,
    bool                bUncached
)
{
    HRESULT             hResult;
    ULONG               ulWriteSize = 0;

    assert(pBuffer != NULL);

    // Check for a dump file (Caching attributes only supported for live debugging)
    if (isDumpFile())
    {
        // Override cache flags if a dump file (Only DEBUG_PHYSICAL_DEFAULT supported)
        ulFlags = DEBUG_PHYSICAL_DEFAULT;
    }
    // Check for cached vs. uncached write
    if (bUncached)
    {
        // Display CPU physical write if requested
        if (VERBOSE_LEVEL(VERBOSE_CPU_PHYSICAL_WRITE))
        {
            if (dmlState())
            {
                dPrintf(bold("CPU: "));
            }
            else    // Plain text only
            {
                dPrintf("CPU: ");
            }
            dPrintf("Uncached physical write to 0x%016I64x - 0x%016I64x\n",
                     paCpuAddress.addr(), (paCpuAddress.addr() + ulBufferSize - 1));
        }
        // Check for a dump file (Caching attributes only supported for live debugging)
        if (isDumpFile())
        {
            // Override cache flags if a dump file (Only DEBUG_PHYSICAL_DEFAULT supported)
            ulFlags = DEBUG_PHYSICAL_DEFAULT;
        }
        // Try to write the requested data
        hResult = WritePhysical2(paCpuAddress.addr(), ulFlags, pBuffer, ulBufferSize, &ulWriteSize);
        if (FAILED(hResult))
        {
            throw CTargetException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error writing CPU physical memory at address 0x%016I64x (%d bytes)",
                                   paCpuAddress.addr(), ulBufferSize);
        }
    }
    else    // Cached
    {
        // Display CPU physical write if requested
        if (VERBOSE_LEVEL(VERBOSE_CPU_PHYSICAL_WRITE))
        {
            if (dmlState())
            {
                dPrintf(bold("CPU: "));
            }
            else    // Plain text only
            {
                dPrintf("CPU: ");
            }
            dPrintf("Cached physical write to 0x%016I64x - 0x%016I64x\n",
                     paCpuAddress.addr(), (paCpuAddress.addr() + ulBufferSize - 1));
        }
        // Check for a dump file (Caching attributes only supported for live debugging)
        if (isDumpFile())
        {
            // Override cache flags if a dump file (Only DEBUG_PHYSICAL_DEFAULT supported)
            ulFlags = DEBUG_PHYSICAL_DEFAULT;
        }
        // Try to write the requested data
        hResult = WritePhysical2(paCpuAddress.addr(), ulFlags, pBuffer, ulBufferSize, &ulWriteSize);
        if (FAILED(hResult))
        {
            throw CTargetException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error writing CPU physical memory at address 0x%016I64x (%d bytes)",
                                   paCpuAddress.addr(), ulBufferSize);
        }
    }
    return ulWriteSize;

} // writeCpuPhysical

//******************************************************************************

CHAR
readChar
(
    POINTER             ptrAddress,
    bool                bUncached
)
{
    CHAR                charData = 0;

    // Try to read the CHAR at ptrAddress
    readCpuVirtual(ptrAddress, &charData, sizeof(charData), bUncached);

    return charData;

} // readChar

//******************************************************************************

UCHAR
readUchar
(
    POINTER             ptrAddress,
    bool                bUncached
)
{
    UCHAR               ucharData = 0;

    // Try to read the UCHAR at ptrAddress
    readCpuVirtual(ptrAddress, &ucharData, sizeof(ucharData), bUncached);

    return ucharData;

} // readUchar

//******************************************************************************

SHORT
readShort
(
    POINTER             ptrAddress,
    bool                bUncached
)
{
    SHORT               shortData = 0;

    // Try to read the SHORT at ptrAddress
    readCpuVirtual(ptrAddress, &shortData, sizeof(shortData), bUncached);

    return shortData;

} // readShort

//******************************************************************************

USHORT
readUshort
(
    POINTER             ptrAddress,
    bool                bUncached
)
{
    USHORT              ushortData = 0;

    // Try to read the USHORT at ptrAddress
    readCpuVirtual(ptrAddress, &ushortData, sizeof(ushortData), bUncached);

    return ushortData;

} // readUshort

//******************************************************************************

LONG
readLong
(
    POINTER             ptrAddress,
    bool                bUncached
)
{
    LONG                longData = 0;

    // Try to read the LONG at ptrAddress
    readCpuVirtual(ptrAddress, &longData, sizeof(longData), bUncached);

    return longData;

} // readLong

//******************************************************************************

ULONG
readUlong
(
    POINTER             ptrAddress,
    bool                bUncached
)
{
    ULONG               ulongData = 0;

    // Try to read the ULONG at ptrAddress
    readCpuVirtual(ptrAddress, &ulongData, sizeof(ulongData), bUncached);

    return ulongData;

} // readUlong

//******************************************************************************

LONG64
readLong64
(
    POINTER             ptrAddress,
    bool                bUncached
)
{
    LONG64              long64Data = 0;

    // Try to read the ULONG64 at ptrAddress
    readCpuVirtual(ptrAddress, &long64Data, sizeof(long64Data), bUncached);

    return long64Data;

} // readLong64

//******************************************************************************

ULONG64
readUlong64
(
    POINTER             ptrAddress,
    bool                bUncached
)
{
    ULONG64             ulong64Data = 0;

    // Try to read the ULONG64 at ptrAddress
    readCpuVirtual(ptrAddress, &ulong64Data, sizeof(ulong64Data), bUncached);

    return ulong64Data;

} // readUlong64

//******************************************************************************

float
readFloat
(
    POINTER             ptrAddress,
    bool                bUncached
)
{
    float               floatData = 0.0;

    // Try to read the flaot at ptrAddress
    readCpuVirtual(ptrAddress, &floatData, sizeof(floatData), bUncached);

    return floatData;

} // readFloat

//******************************************************************************

double
readDouble
(
    POINTER             ptrAddress,
    bool                bUncached
)
{
    double              doubleData = 0.0;

    // Try to read the double at ptrAddress
    readCpuVirtual(ptrAddress, &doubleData, sizeof(doubleData), bUncached);

    return doubleData;

} // readDouble

//******************************************************************************

POINTER
readPointer32
(
    POINTER             ptrAddress,
    bool                bUncached
)
{
    POINTER             ptr32Data = 0;

    // Try to read the 32-bit pointer at ptrAddress
    readCpuVirtual(ptrAddress, &ptr32Data, sizeof(ULONG), bUncached);

    return ptr32Data;

} // readPointer32

//******************************************************************************

POINTER
readPointer64
(
    POINTER             ptrAddress,
    bool                bUncached
)
{
    POINTER             ptr64Data = 0;

    // Try to read the 64-bit pointer at ptrAddress
    readCpuVirtual(ptrAddress, &ptr64Data, sizeof(ULONG64), bUncached);

    return ptr64Data;

} // readPointer64

//******************************************************************************

POINTER
readPointer
(
    POINTER             ptrAddress,
    bool                bUncached
)
{
    POINTER             ptrData;

    // Check for 64-bit pointers
    if (pointerSize() == 64)
    {
        // Try to read the 64-bit pointer at ptrAddress
        ptrData = readPointer64(ptrAddress, bUncached);
    }
    else    // 32-bit pointers
    {
        // Try to read the 32-bit pointer at ptrAddress
        ptrData = readPointer32(ptrAddress, bUncached);
    }
    return ptrData;

} // readPointer

//******************************************************************************

bool
readBoolean
(
    POINTER             ptrAddress,
    bool                bUncached
)
{
    bool                booleanData = false;

    // Try to read the boolean at ptrAddress
    readCpuVirtual(ptrAddress, &booleanData, sizeof(booleanData), bUncached);

    return booleanData;

} // readBoolean

//******************************************************************************

void
readStruct
(
    POINTER             ptrAddress,
    PVOID               pBuffer,
    ULONG               ulBufferSize,
    bool                bUncached
)
{
    assert(pBuffer != NULL);

    // Try to read ulBufferSize bytes from ptrAddress to pBuffer
    readCpuVirtual(ptrAddress, pBuffer, ulBufferSize, bUncached);

} // readStruct

//******************************************************************************

ULONG64
readBitfield
(
    POINTER             ptrAddress,
    UINT                uPosition,
    UINT                uWidth,
    ULONG               ulSize,
    bool                bUncached
)
{
    union
    {
        UCHAR           ucharData;
        USHORT          ushortData;
        ULONG           ulongData;
        ULONG64         ulong64Data;
    } bitfieldData;
    ULONG64             bitfieldMask;

    // Check for bitfield value size given
    if (ulSize == 0)
    {
        // Compute the value size for the given bitfield
        ulSize = (uPosition + uWidth + 7) / 8;
    }
    // Initialize the bitfield data and compute the bitfield mask value
    bitfieldData.ulong64Data = 0;
    bitfieldMask             = (1LL << uWidth) - 1;

    // Switch on the value size to read
    switch(ulSize)
    {
        case 1:                         // Char size value

            // Try to read the char at ptrAddress
            bitfieldData.ucharData = readUchar(ptrAddress, bUncached);

            break;

        case 2:                         // Short size value

            // Try to read the short at ptrAddress
            bitfieldData.ushortData = readUshort(ptrAddress, bUncached);

            break;

        case 3:                         // Three byte value defaults to Long
        case 4:                         // Long size value

            // Try to read the long at ptrAddress
            bitfieldData.ulongData = readUlong(ptrAddress, bUncached);

            break;

        case 5:                         // Five byte value defaults to Long64
        case 6:                         // Six byte value defaults to Long64
        case 7:                         // Seven byte value defaults to Long64
        case 8:                         // Long64 size value

            // Try to read the long64 at ptrAddress
            bitfieldData.ulong64Data = readUlong64(ptrAddress, bUncached);

            break;

        default:                        // Unknown/invalid value size

            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             ": Invalid bitfield size (%d) at address 0x%0*I64x",
                             ulSize, PTR(ptrAddress));

            break;
    }
    // Mask and shift the bitfield value into position
    bitfieldData.ulong64Data = (bitfieldData.ulong64Data >> uPosition) & bitfieldMask;

    return bitfieldData.ulong64Data;

} // readBitfield

//******************************************************************************

void
writeChar
(
    POINTER             ptrAddress,
    CHAR                charData,
    bool                bUncached
)
{
    // Try to write the CHAR at ptrAddress
    writeCpuVirtual(ptrAddress, &charData, sizeof(charData), bUncached);

} // writeChar

//******************************************************************************

void
writeUchar
(
    POINTER             ptrAddress,
    UCHAR               ucharData,
    bool                bUncached
)
{
    // Try to write the UCHAR at ptrAddress
    writeCpuVirtual(ptrAddress, &ucharData, sizeof(ucharData), bUncached);

} // writeUchar

//******************************************************************************

void
writeShort
(
    POINTER             ptrAddress,
    SHORT               shortData,
    bool                bUncached
)
{
    // Try to write the SHORT at ptrAddress
    writeCpuVirtual(ptrAddress, &shortData, sizeof(shortData), bUncached);

} // writeShort

//******************************************************************************

void
writeUshort
(
    POINTER             ptrAddress,
    USHORT              ushortData,
    bool                bUncached
)
{
    // Try to write the USHORT at ptrAddress
    writeCpuVirtual(ptrAddress, &ushortData, sizeof(ushortData), bUncached);

} // writeUshort

//******************************************************************************

void
writeLong
(
    POINTER             ptrAddress,
    LONG                longData,
    bool                bUncached
)
{
    // Try to write the LONG at ptrAddress
    writeCpuVirtual(ptrAddress, &longData, sizeof(longData), bUncached);

} // writeLong

//******************************************************************************

void
writeUlong
(
    POINTER             ptrAddress,
    ULONG               ulongData,
    bool                bUncached
)
{
    // Try to write the ULONG at ptrAddress
    writeCpuVirtual(ptrAddress, &ulongData, sizeof(ulongData), bUncached);

} // writeUlong

//******************************************************************************

void
writeLong64
(
    POINTER             ptrAddress,
    LONG64              long64Data,
    bool                bUncached
)
{
    // Try to write the LONG64 at ptrAddress
    writeCpuVirtual(ptrAddress, &long64Data, sizeof(long64Data), bUncached);

} // writeLong64

//******************************************************************************

void
writeUlong64
(
    POINTER             ptrAddress,
    ULONG64             ulong64Data,
    bool                bUncached
)
{
    // Try to write the ULONG64 at ptrAddress
    writeCpuVirtual(ptrAddress, &ulong64Data, sizeof(ulong64Data), bUncached);

} // writeUlong64

//******************************************************************************

void
writeFloat
(
    POINTER             ptrAddress,
    float               floatData,
    bool                bUncached
)
{
    // Try to write the float at ptrAddress
    writeCpuVirtual(ptrAddress, &floatData, sizeof(floatData), bUncached);

} // writeFloat

//******************************************************************************

void
writeDouble
(
    POINTER             ptrAddress,
    double              doubleData,
    bool                bUncached
)
{
    // Try to write the double at ptrAddress
    writeCpuVirtual(ptrAddress, &doubleData, sizeof(doubleData), bUncached);

} // writeDouble

//******************************************************************************

void
writePointer32
(
    POINTER             ptrAddress,
    POINTER             ptr32Data,
    bool                bUncached
)
{
    // Try to write the 32-bit pointer at ptrAddress
    writeCpuVirtual(ptrAddress, &ptr32Data, sizeof(ULONG), bUncached);

} // writePointer32

//******************************************************************************

void
writePointer64
(
    POINTER             ptrAddress,
    POINTER             ptr64Data,
    bool                bUncached
)
{
    // Try to write the 64-bit pointer at ptrAddress
    writeCpuVirtual(ptrAddress, &ptr64Data, sizeof(ULONG64), bUncached);

} // writePointer64

//******************************************************************************

void
writePointer
(
    POINTER             ptrAddress,
    POINTER             ptrData,
    bool                bUncached
)
{
    // Check for 64-bit pointers
    if (pointerSize() == 64)
    {
        // Try to write the 64-bit pointer at ptrAddress
        writePointer64(ptrAddress, ptrData, bUncached);
    }
    else    // 32-bit pointers
    {
        // Try to write the 32-bit pointer at ptrAddress
        writePointer32(ptrAddress, ptrData, bUncached);
    }

} // writePointer

//******************************************************************************

void
writeBoolean
(
    POINTER             ptrAddress,
    bool                booleanData,
    bool                bUncached
)
{
    // Try to write the boolean at ptrAddress
    writeCpuVirtual(ptrAddress, &booleanData, sizeof(booleanData), bUncached);

} // writeBoolean

//******************************************************************************

void
writeStruct
(
    POINTER             ptrAddress,
    PVOID               pBuffer,
    ULONG               ulBufferSize,
    bool                bUncached
)
{
    assert(pBuffer != NULL);

    // Try to write ulBufferSize bytes from ptrAddress to pBuffer
    writeCpuVirtual(ptrAddress, pBuffer, ulBufferSize, bUncached);

} // writeStruct

//******************************************************************************

void
writeBitfield
(
    POINTER             ptrAddress,
    ULONG64             bitfieldData,
    UINT                uPosition,
    UINT                uWidth,
    ULONG               ulSize,
    bool                bUncached
)
{
    union
    {
        UCHAR           ucharData;
        USHORT          ushortData;
        ULONG           ulongData;
        ULONG64         ulong64Data;
    } lwrrentData;
    ULONG64             bitfieldMask;

    // Check for bitfield value size given
    if (ulSize == 0)
    {
        // Compute the value size for the given bitfield
        ulSize = (uPosition + uWidth + 7) / 8;
    }
    // Initialize the current data and compute the bitfield mask value
    lwrrentData.ulong64Data = 0;
    bitfieldMask            = (1LL << uWidth) - 1;

    // Switch on the value size to read
    switch(ulSize)
    {
        case 1:                         // Char size value

            // Try to read the char at ptrAddress
            lwrrentData.ucharData = readUchar(ptrAddress, bUncached);

            break;

        case 2:                         // Short size value

            // Try to read the short at ptrAddress
            lwrrentData.ushortData = readUshort(ptrAddress, bUncached);

            break;

        case 3:                         // Three byte value defaults to Long
        case 4:                         // Long size value

            // Try to read the long at ptrAddress
            lwrrentData.ulongData = readUlong(ptrAddress, bUncached);

            break;

        case 5:                         // Five byte value defaults to Long64
        case 6:                         // Six byte value defaults to Long64
        case 7:                         // Seven byte value defaults to Long64
        case 8:                         // Long64 size value

            // Try to read the long64 at ptrAddress
            lwrrentData.ulong64Data = readUlong64(ptrAddress, bUncached);

            break;

        default:                        // Unknown/invalid value size

            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             ": Invalid bitfield size (%d) at address 0x%0*I64x",
                             ulSize, PTR(ptrAddress));

            break;
    }
    // Mask off the bitfield in the current data
    lwrrentData.ulong64Data &= ~(bitfieldMask << uPosition);

    // Mask and shift the bitfield value into position (into current data)
    lwrrentData.ulong64Data |= (bitfieldData & bitfieldMask) << uPosition;

    // Switch on the value size to write
    switch(ulSize)
    {
        case 1:                         // Char size value

            // Try to write the char at ptrAddress
            writeUchar(ptrAddress, lwrrentData.ucharData, bUncached);

            break;

        case 2:                         // Short size value

            // Try to write the short at ptrAddress
            writeUshort(ptrAddress, lwrrentData.ushortData, bUncached);

            break;

        case 3:                         // Three byte value defaults to Long
        case 4:                         // Long size value

            // Try to write the long at ptrAddress
            writeUlong(ptrAddress, lwrrentData.ulongData, bUncached);

            break;

        case 5:                         // Five byte value defaults to Long64
        case 6:                         // Six byte value defaults to Long64
        case 7:                         // Seven byte value defaults to Long64
        case 8:                         // Long64 size value

            // Try to write the long64 at ptrAddress
            writeUlong64(ptrAddress, lwrrentData.ulong64Data, bUncached);

            break;
    }
    return;

} // writeBitfield

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
