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
|*  Module: dbgdataspaces.cpp                                                 *|
|*                                                                            *|
 \****************************************************************************/
#include "dbgprecomp.h"

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
//******************************************************************************
//
// Debugger Data Spaces Interface wrappers
//
//******************************************************************************

HRESULT
ReadVirtual
(
    ULONG64             Offset,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BytesRead
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadVirtual data spaces method
        hResult = pDbgDataSpaces->ReadVirtual(Offset, Buffer, BufferSize, BytesRead);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested (or DbgEng virtual read)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES) || VERBOSE_LEVEL(VERBOSE_DBGENG_VIRTUAL_READ))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadVirtual
            {
                dPrintf("%s ReadVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadVirtual

//******************************************************************************

HRESULT
WriteVirtual
(
    ULONG64             Offset,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BytesWritten
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the WriteVirtual data spaces method
        hResult = pDbgDataSpaces->WriteVirtual(Offset, Buffer, BufferSize, BytesWritten);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested (or DbgEng virtual write)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES) || VERBOSE_LEVEL(VERBOSE_DBGENG_VIRTUAL_WRITE))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s WriteVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed WriteVirtual
            {
                dPrintf("%s WriteVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // WriteVirtual

//******************************************************************************

HRESULT
SearchVirtual
(
    ULONG64             Offset,
    ULONG64             Length,
    PVOID               Pattern,
    ULONG               PatternSize,
    ULONG               PatternGranularity,
    PULONG64            MatchOffset
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Pattern != NULL);
    assert(MatchOffset != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SearchVirtual data spaces method
        hResult = pDbgDataSpaces->SearchVirtual(Offset, Length, Pattern, PatternSize, PatternGranularity, MatchOffset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SearchVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SearchVirtual
            {
                dPrintf("%s SearchVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SearchVirtual

//******************************************************************************

HRESULT
ReadVirtualUncached
(
    ULONG64             Offset,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BytesRead
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadVirtualUncached data spaces method
        hResult = pDbgDataSpaces->ReadVirtualUncached(Offset, Buffer, BufferSize, BytesRead);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested (or DbgEng virtual read)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES) || VERBOSE_LEVEL(VERBOSE_DBGENG_VIRTUAL_READ))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadVirtualUncached %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadVirtualUncached
            {
                dPrintf("%s ReadVirtualUncached %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadVirtualUncached

//******************************************************************************

HRESULT
WriteVirtualUncached
(
    ULONG64             Offset,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BytesWritten
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the WriteVirtualUncached data spaces method
        hResult = pDbgDataSpaces->WriteVirtualUncached(Offset, Buffer, BufferSize, BytesWritten);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested (or DbgEng virtual write)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES) || VERBOSE_LEVEL(VERBOSE_DBGENG_VIRTUAL_WRITE))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s WriteVirtualUncached %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed WriteVirtualUncached
            {
                dPrintf("%s WriteVirtualUncached %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // WriteVirtualUncached

//******************************************************************************

HRESULT
ReadPointersVirtual
(
    ULONG               Count,
    ULONG64             Offset,
    PULONG64            Ptrs
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Ptrs != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadPointersVirtual data spaces method
        hResult = pDbgDataSpaces->ReadPointersVirtual(Count, Offset, Ptrs);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested (or DbgEng virtual read)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES) || VERBOSE_LEVEL(VERBOSE_DBGENG_VIRTUAL_READ))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadPointersVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadPointersVirtual
            {
                dPrintf("%s ReadPointersVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadPointersVirtual

//******************************************************************************

HRESULT
WritePointersVirtual
(
    ULONG               Count,
    ULONG64             Offset,
    PULONG64            Ptrs
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Ptrs != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the WritePointersVirtual data spaces method
        hResult = pDbgDataSpaces->WritePointersVirtual(Count, Offset, Ptrs);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested (or DbgEng virtual write)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES) || VERBOSE_LEVEL(VERBOSE_DBGENG_VIRTUAL_WRITE))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s WritePointersVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed WritePointersVirtual
            {
                dPrintf("%s WritePointersVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // WritePointersVirtual

//******************************************************************************

HRESULT
ReadPhysical
(
    ULONG64             Offset,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BytesRead
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadPhysical data spaces method
        hResult = pDbgDataSpaces->ReadPhysical(Offset, Buffer, BufferSize, BytesRead);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested (or DbgEng physical read)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES) || VERBOSE_LEVEL(VERBOSE_DBGENG_PHYSICAL_READ))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadPhysical %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadPhysical
            {
                dPrintf("%s ReadPhysical %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadPhysical

//******************************************************************************

HRESULT
WritePhysical
(
    ULONG64             Offset,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BytesWritten
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the WritePhysical data spaces method
        hResult = pDbgDataSpaces->WritePhysical(Offset, Buffer, BufferSize, BytesWritten);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested (or DbgEng physical write)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES) || VERBOSE_LEVEL(VERBOSE_DBGENG_PHYSICAL_WRITE))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s WritePhysical %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed WritePhysical
            {
                dPrintf("%s WritePhysical %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // WritePhysical

//******************************************************************************

HRESULT
ReadControl
(
    ULONG               Processor,
    ULONG64             Offset,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BytesRead
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadControl data spaces method
        hResult = pDbgDataSpaces->ReadControl(Processor, Offset, Buffer, BufferSize, BytesRead);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadControl %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadControl
            {
                dPrintf("%s ReadControl %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadControl

//******************************************************************************

HRESULT
WriteControl
(
    ULONG               Processor,
    ULONG64             Offset,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BytesWritten
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the WriteControl data spaces method
        hResult = pDbgDataSpaces->WriteControl(Processor, Offset, Buffer, BufferSize, BytesWritten);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s WriteControl %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed WriteControl
            {
                dPrintf("%s WriteControl %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // WriteControl

//******************************************************************************

HRESULT
ReadIo
(
    ULONG               InterfaceType,
    ULONG               BusNumber,
    ULONG               AddressSpace,
    ULONG64             Offset,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BytesRead
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadIo data spaces method
        hResult = pDbgDataSpaces->ReadIo(InterfaceType, BusNumber, AddressSpace, Offset, Buffer, BufferSize, BytesRead);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng I/O read interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_IO_READ))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadIo %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadIo
            {
                dPrintf("%s ReadIo %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadIo

//******************************************************************************

HRESULT
WriteIo
(
    ULONG               InterfaceType,
    ULONG               BusNumber,
    ULONG               AddressSpace,
    ULONG64             Offset,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BytesWritten
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the WriteIo data spaces method
        hResult = pDbgDataSpaces->WriteIo(InterfaceType, BusNumber, AddressSpace, Offset, Buffer, BufferSize, BytesWritten);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng I/O write interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_IO_WRITE))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s WriteIo %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed WriteIo
            {
                dPrintf("%s WriteIo %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // WriteIo

//******************************************************************************

HRESULT
ReadMsr
(
    ULONG               Msr,
    PULONG64            Value
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Value != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadMsr data spaces method
        hResult = pDbgDataSpaces->ReadMsr(Msr, Value);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadMsr %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadMsr
            {
                dPrintf("%s ReadMsr %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadMsr

//******************************************************************************

HRESULT
WriteMsr
(
    ULONG               Msr,
    ULONG64             Value
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the WriteMsr data spaces method
        hResult = pDbgDataSpaces->WriteMsr(Msr, Value);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s WriteMsr %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed WriteMsr
            {
                dPrintf("%s WriteMsr %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // WriteMsr

//******************************************************************************

HRESULT
ReadBusData
(
    ULONG               BusDataType,
    ULONG               BusNumber,
    ULONG               SlotNumber,
    ULONG               Offset,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BytesRead
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadBusData data spaces method
        hResult = pDbgDataSpaces->ReadBusData(BusDataType, BusNumber, SlotNumber, Offset, Buffer, BufferSize, BytesRead);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadBusData %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadBusData
            {
                dPrintf("%s ReadBusData %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadBusData

//******************************************************************************

HRESULT
WriteBusData
(
    ULONG               BusDataType,
    ULONG               BusNumber,
    ULONG               SlotNumber,
    ULONG               Offset,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BytesWritten
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the WriteBusData data spaces method
        hResult = pDbgDataSpaces->WriteBusData(BusDataType, BusNumber, SlotNumber, Offset, Buffer, BufferSize, BytesWritten);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s WriteBusData %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed WriteBusData
            {
                dPrintf("%s WriteBusData %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // WriteBusData

//******************************************************************************

HRESULT
CheckLowMemory()
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the CheckLowMemory data spaces method
        hResult = pDbgDataSpaces->CheckLowMemory();

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s CheckLowMemory %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed CheckLowMemory
            {
                dPrintf("%s CheckLowMemory %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // CheckLowMemory

//******************************************************************************

HRESULT
ReadDebuggerData
(
    ULONG               Index,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              DataSize
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadDebuggerData data spaces method
        hResult = pDbgDataSpaces->ReadDebuggerData(Index, Buffer, BufferSize, DataSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadDebuggerData %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadDebuggerData
            {
                dPrintf("%s ReadDebuggerData %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadDebuggerData

//******************************************************************************

HRESULT
ReadProcessorSystemData
(
    ULONG               Processor,
    ULONG               Index,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              DataSize
)
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = debugDataSpacesInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadProcessorSystemData data spaces method
        hResult = pDbgDataSpaces->ReadProcessorSystemData(Processor, Index, Buffer, BufferSize, DataSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadProcessorSystemData %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadProcessorSystemData
            {
                dPrintf("%s ReadProcessorSystemData %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadProcessorSystemData

//******************************************************************************

HRESULT
VirtualToPhysical
(
    ULONG64             Virtual,
    PULONG64            Physical
)
{
    PDEBUG_DATA_SPACES2 pDbgDataSpaces2 = debugDataSpaces2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Physical != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the VirtualToPhysical data spaces method
        hResult = pDbgDataSpaces2->VirtualToPhysical(Virtual, Physical);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s VirtualToPhysical %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed VirtualToPhysical
            {
                dPrintf("%s VirtualToPhysical %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // VirtualToPhysical

//******************************************************************************

HRESULT
GetVirtualTranslationPhysicalOffsets
(
    ULONG64             Virtual,
    PULONG64            Offsets,
    ULONG               OffsetsSize,
    PULONG              Levels
)
{
    PDEBUG_DATA_SPACES2 pDbgDataSpaces2 = debugDataSpaces2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid data spaces interface
    if (pDbgDataSpaces2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetVirtualTranslationPhysicalOffsets data spaces method
        hResult = pDbgDataSpaces2->GetVirtualTranslationPhysicalOffsets(Virtual, Offsets, OffsetsSize, Levels);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetVirtualTranslationPhysicalOffsets %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetVirtualTranslationPhysicalOffsets
            {
                dPrintf("%s GetVirtualTranslationPhysicalOffsets %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetVirtualTranslationPhysicalOffsets

//******************************************************************************

HRESULT
ReadHandleData
(
    ULONG64             Handle,
    ULONG               DataType,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              DataSize
)
{
    PDEBUG_DATA_SPACES2 pDbgDataSpaces2 = debugDataSpaces2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid data spaces interface
    if (pDbgDataSpaces2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadHandleData data spaces method
        hResult = pDbgDataSpaces2->ReadHandleData(Handle, DataType, Buffer, BufferSize, DataSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadHandleData %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadHandleData
            {
                dPrintf("%s ReadHandleData %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadHandleData

//******************************************************************************

HRESULT
FillVirtual
(
    ULONG64             Start,
    ULONG               Size,
    PVOID               Pattern,
    ULONG               PatternSize,
    PULONG              Filled
)
{
    PDEBUG_DATA_SPACES2 pDbgDataSpaces2 = debugDataSpaces2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Pattern != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the FillVirtual data spaces method
        hResult = pDbgDataSpaces2->FillVirtual(Start, Size, Pattern, PatternSize, Filled);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s FillVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed FillVirtual
            {
                dPrintf("%s FillVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // FillVirtual

//******************************************************************************

HRESULT
FillPhysical
(
    ULONG64             Start,
    ULONG               Size,
    PVOID               Pattern,
    ULONG               PatternSize,
    PULONG              Filled
)
{
    PDEBUG_DATA_SPACES2 pDbgDataSpaces2 = debugDataSpaces2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Pattern != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the FillPhysical data spaces method
        hResult = pDbgDataSpaces2->FillPhysical(Start, Size, Pattern, PatternSize, Filled);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s FillPhysical %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed FillPhysical
            {
                dPrintf("%s FillPhysical %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // FillPhysical

//******************************************************************************

HRESULT
QueryVirtual
(
    ULONG64             Offset,
    PMEMORY_BASIC_INFORMATION64 Info
)
{
    PDEBUG_DATA_SPACES2 pDbgDataSpaces2 = debugDataSpaces2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Info != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the QueryVirtual data spaces method
        hResult = pDbgDataSpaces2->QueryVirtual(Offset, Info);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s QueryVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed QueryVirtual
            {
                dPrintf("%s QueryVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // QueryVirtual

//******************************************************************************

HRESULT
ReadImageNtHeaders
(
    ULONG64             ImageBase,
    PIMAGE_NT_HEADERS64 Headers
)
{
    PDEBUG_DATA_SPACES3 pDbgDataSpaces3 = debugDataSpaces3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Headers != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadImageNtHeaders data spaces method
        hResult = pDbgDataSpaces3->ReadImageNtHeaders(ImageBase, Headers);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadImageNtHeaders %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadImageNtHeaders
            {
                dPrintf("%s ReadImageNtHeaders %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadImageNtHeaders

//******************************************************************************

HRESULT
ReadTagged
(
    LPGUID              Tag,
    ULONG               Offset,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              TotalSize
)
{
    PDEBUG_DATA_SPACES3 pDbgDataSpaces3 = debugDataSpaces3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid data spaces interface
    if (pDbgDataSpaces3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadTagged data spaces method
        hResult = pDbgDataSpaces3->ReadTagged(Tag, Offset, Buffer, BufferSize, TotalSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadTagged %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadTagged
            {
                dPrintf("%s ReadTagged %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadTagged

//******************************************************************************

HRESULT
StartEnumTagged
(
    PULONG64            Handle
)
{
    PDEBUG_DATA_SPACES3 pDbgDataSpaces3 = debugDataSpaces3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Handle != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the StartEnumTagged data spaces method
        hResult = pDbgDataSpaces3->StartEnumTagged(Handle);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s StartEnumTagged %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed StartEnumTagged
            {
                dPrintf("%s StartEnumTagged %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // StartEnumTagged

//******************************************************************************

HRESULT
GetNextTagged
(
    ULONG64             Handle,
    LPGUID              Tag,
    PULONG              Size
)
{
    PDEBUG_DATA_SPACES3 pDbgDataSpaces3 = debugDataSpaces3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Tag != NULL);
    assert(Size != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNextTagged data spaces method
        hResult = pDbgDataSpaces3->GetNextTagged(Handle, Tag, Size);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNextTagged %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNextTagged
            {
                dPrintf("%s GetNextTagged %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNextTagged

//******************************************************************************

HRESULT
EndEnumTagged
(
    ULONG64             Handle
)
{
    PDEBUG_DATA_SPACES3 pDbgDataSpaces3 = debugDataSpaces3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid data spaces interface
    if (pDbgDataSpaces3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the EndEnumTagged data spaces method
        hResult = pDbgDataSpaces3->EndEnumTagged(Handle);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s EndEnumTagged %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed EndEnumTagged
            {
                dPrintf("%s EndEnumTagged %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // EndEnumTagged

//******************************************************************************

HRESULT
GetOffsetInformation
(
    ULONG               Space,
    ULONG               Which,
    ULONG64             Offset,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              InfoSize
)
{
    PDEBUG_DATA_SPACES4 pDbgDataSpaces4 = debugDataSpaces4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid data spaces interface
    if (pDbgDataSpaces4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetOffsetInformation data spaces method
        hResult = pDbgDataSpaces4->GetOffsetInformation(Space, Which, Offset, Buffer, BufferSize, InfoSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetOffsetInformation %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetOffsetInformation
            {
                dPrintf("%s GetOffsetInformation %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetOffsetInformation

//******************************************************************************

HRESULT
GetNextDifferentlyValidOffsetVirtual
(
    ULONG64             Offset,
    PULONG64            NextOffset
)
{
    PDEBUG_DATA_SPACES4 pDbgDataSpaces4 = debugDataSpaces4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(NextOffset != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNextDifferentlyValidOffsetVirtual data spaces method
        hResult = pDbgDataSpaces4->GetNextDifferentlyValidOffsetVirtual(Offset, NextOffset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNextDifferentlyValidOffsetVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNextDifferentlyValidOffsetVirtual
            {
                dPrintf("%s GetNextDifferentlyValidOffsetVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNextDifferentlyValidOffsetVirtual

//******************************************************************************

HRESULT
GetValidRegiolwirtual
(
    ULONG64             Base,
    ULONG               Size,
    PULONG64            ValidBase,
    PULONG              ValidSize
)
{
    PDEBUG_DATA_SPACES4 pDbgDataSpaces4 = debugDataSpaces4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(ValidBase != NULL);
    assert(ValidSize != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetValidRegiolwirtual data spaces method
        hResult = pDbgDataSpaces4->GetValidRegiolwirtual(Base, Size, ValidBase, ValidSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetValidRegiolwirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetValidRegiolwirtual
            {
                dPrintf("%s GetValidRegiolwirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetValidRegiolwirtual

//******************************************************************************

HRESULT
SearchVirtual2
(
    ULONG64             Offset,
    ULONG64             Length,
    ULONG               Flags,
    PVOID               Pattern,
    ULONG               PatternSize,
    ULONG               PatternGranularity,
    PULONG64            MatchOffset
)
{
    PDEBUG_DATA_SPACES4 pDbgDataSpaces4 = debugDataSpaces4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Pattern != NULL);
    assert(MatchOffset != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SearchVirtual2 data spaces method
        hResult = pDbgDataSpaces4->SearchVirtual2(Offset, Length, Flags, Pattern, PatternSize, PatternGranularity, MatchOffset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SearchVirtual2 %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SearchVirtual2
            {
                dPrintf("%s SearchVirtual2 %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SearchVirtual2

//******************************************************************************

HRESULT
ReadMultiByteStringVirtual
(
    ULONG64             Offset,
    ULONG               MaxBytes,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              StringBytes
)
{
    PDEBUG_DATA_SPACES4 pDbgDataSpaces4 = debugDataSpaces4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid data spaces interface
    if (pDbgDataSpaces4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadMultiByteStringVirtual data spaces method
        hResult = pDbgDataSpaces4->ReadMultiByteStringVirtual(Offset, MaxBytes, Buffer, BufferSize, StringBytes);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadMultiByteStringVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadMultiByteStringVirtual
            {
                dPrintf("%s ReadMultiByteStringVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadMultiByteStringVirtual

//******************************************************************************

HRESULT
ReadMultiByteStringVirtualWide
(
    ULONG64             Offset,
    ULONG               MaxBytes,
    ULONG               CodePage,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              StringBytes
)
{
    PDEBUG_DATA_SPACES4 pDbgDataSpaces4 = debugDataSpaces4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid data spaces interface
    if (pDbgDataSpaces4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadMultiByteStringVirtualWide data spaces method
        hResult = pDbgDataSpaces4->ReadMultiByteStringVirtualWide(Offset, MaxBytes, CodePage, Buffer, BufferSize, StringBytes);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadMultiByteStringVirtualWide %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadMultiByteStringVirtualWide
            {
                dPrintf("%s ReadMultiByteStringVirtualWide %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadMultiByteStringVirtualWide

//******************************************************************************

HRESULT
ReadUnicodeStringVirtual
(
    ULONG64             Offset,
    ULONG               MaxBytes,
    ULONG               CodePage,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              StringBytes
)
{
    PDEBUG_DATA_SPACES4 pDbgDataSpaces4 = debugDataSpaces4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid data spaces interface
    if (pDbgDataSpaces4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadUnicodeStringVirtual data spaces method
        hResult = pDbgDataSpaces4->ReadUnicodeStringVirtual(Offset, MaxBytes, CodePage, Buffer, BufferSize, StringBytes);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadUnicodeStringVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadUnicodeStringVirtual
            {
                dPrintf("%s ReadUnicodeStringVirtual %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadUnicodeStringVirtual

//******************************************************************************

HRESULT
ReadUnicodeStringVirtualWide
(
    ULONG64             Offset,
    ULONG               MaxBytes,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              StringBytes
)
{
    PDEBUG_DATA_SPACES4 pDbgDataSpaces4 = debugDataSpaces4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid data spaces interface
    if (pDbgDataSpaces4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadUnicodeStringVirtualWide data spaces method
        hResult = pDbgDataSpaces4->ReadUnicodeStringVirtualWide(Offset, MaxBytes, Buffer, BufferSize, StringBytes);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadUnicodeStringVirtualWide %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadUnicodeStringVirtualWide
            {
                dPrintf("%s ReadUnicodeStringVirtualWide %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadUnicodeStringVirtualWide

//******************************************************************************

HRESULT
ReadPhysical2
(
    ULONG64             Offset,
    ULONG               Flags,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BytesRead
)
{
    PDEBUG_DATA_SPACES4 pDbgDataSpaces4 = debugDataSpaces4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadPhysical2 data spaces method
        hResult = pDbgDataSpaces4->ReadPhysical2(Offset, Flags, Buffer, BufferSize, BytesRead);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested (or DbgEng physical read)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES) || VERBOSE_LEVEL(VERBOSE_DBGENG_PHYSICAL_READ))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadPhysical2 %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadPhysical2
            {
                dPrintf("%s ReadPhysical2 %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadPhysical2

//******************************************************************************

HRESULT
WritePhysical2
(
    ULONG64             Offset,
    ULONG               Flags,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BytesWritten
)
{
    PDEBUG_DATA_SPACES4 pDbgDataSpaces4 = debugDataSpaces4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid data spaces interface
    if (pDbgDataSpaces4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the WritePhysical2 data spaces method
        hResult = pDbgDataSpaces4->WritePhysical2(Offset, Flags, Buffer, BufferSize, BytesWritten);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng data spaces interface if requested (or DbgEng physical write)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_DATA_SPACES) || VERBOSE_LEVEL(VERBOSE_DBGENG_PHYSICAL_WRITE))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s WritePhysical2 %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed WritePhysical2
            {
                dPrintf("%s WritePhysical2 %s = 0x%08x\n", DML(bold("DbgDataSpaces:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // WritePhysical2

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
