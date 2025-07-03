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
|*  Module: osheader.cpp                                                      *|
|*                                                                            *|
 \****************************************************************************/
#include "osprecomp.h"

//******************************************************************************
//
//  os namespace
//
//******************************************************************************
namespace os
{

//******************************************************************************
//
// Locals
//
//******************************************************************************








//******************************************************************************

PIMAGE_DOS_HEADER
getImageDosHeader
(
    HANDLE              hModule
)
{
    PIMAGE_DOS_HEADER   pImageDosHeader;

    assert(hModule != NULL);

    // Get address of the DOS image header (Should be at module base address [Handle])
    pImageDosHeader = reinterpret_cast<PIMAGE_DOS_HEADER>(hModule);

    // Make sure this looks like a module base address (DOS image header)
    if (pImageDosHeader->e_magic != IMAGE_DOS_SIGNATURE)
    {
        // Invalid image DOS header, clear pointer
        pImageDosHeader = NULL;
    }
    return pImageDosHeader;

} // getImageDosHeader

//******************************************************************************

PIMAGE_FILE_HEADER
getImageFileHeader
(
    HANDLE              hModule
)
{
    DWORD              *pImageSignature;
    PIMAGE_DOS_HEADER   pImageDosHeader;
    PIMAGE_FILE_HEADER  pImageFileHeader = NULL;

    assert(hModule != NULL);

    // Get the image DOS header for this module
    pImageDosHeader = getImageDosHeader(hModule);
    if (pImageDosHeader != NULL)
    {
        // Compute address of the image signature
        pImageSignature = reinterpret_cast<DWORD*>(reinterpret_cast<BYTE*>(pImageDosHeader) + pImageDosHeader->e_lfanew);

        // Make sure this is the correct image signature (NT)
        if (*pImageSignature == IMAGE_NT_SIGNATURE)
        {
            // Compute the address of the image file header
            pImageFileHeader = reinterpret_cast<PIMAGE_FILE_HEADER>(pImageSignature + 1);
        }
    }
    return pImageFileHeader;

} // getImageFileHeader

//******************************************************************************

PIMAGE_EXPORT_DIRECTORY
getImageExportDirectory
(
    HANDLE              hModule
)
{
    BYTE               *pModuleBase;
    PIMAGE_FILE_HEADER  pImageFileHeader;
    union
    {
        PIMAGE_OPTIONAL_HEADER32    arm;
        PIMAGE_OPTIONAL_HEADER32    x86;
        PIMAGE_OPTIONAL_HEADER64    x64;
    } pImageOptionalHeader;
    PIMAGE_EXPORT_DIRECTORY pImageExportDirectory = NULL;

    assert(hModule != NULL);

    // The module handle is really nothing more than the image base address (PE Header)
    pModuleBase = reinterpret_cast<BYTE*>(hModule);

    // Get the image file header for this module
    pImageFileHeader = getImageFileHeader(hModule);
    if (pImageFileHeader)
    {
        // Switch on the machine type
        switch(pImageFileHeader->Machine)
        {
            case IMAGE_FILE_MACHINE_ARMNT:      // ARM machine type

                // Compute the address of the image option header
                pImageOptionalHeader.arm = reinterpret_cast<PIMAGE_OPTIONAL_HEADER32>(pImageFileHeader + 1);

                // Compute the address of the image export directory
                pImageExportDirectory = reinterpret_cast<PIMAGE_EXPORT_DIRECTORY>(pModuleBase + pImageOptionalHeader.x86->DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress);

                break;

            case IMAGE_FILE_MACHINE_I386:       // x86 machine type

                // Compute the address of the image option header
                pImageOptionalHeader.x86 = reinterpret_cast<PIMAGE_OPTIONAL_HEADER32>(pImageFileHeader + 1);

                // Compute the address of the image export directory
                pImageExportDirectory = reinterpret_cast<PIMAGE_EXPORT_DIRECTORY>(pModuleBase + pImageOptionalHeader.x86->DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress);

                break;

            case IMAGE_FILE_MACHINE_AMD64:      // x64 machine type

                // Compute the address of the image option header
                pImageOptionalHeader.x64 = reinterpret_cast<PIMAGE_OPTIONAL_HEADER64>(pImageFileHeader + 1);

                // Compute the address of the image export directory
                pImageExportDirectory = reinterpret_cast<PIMAGE_EXPORT_DIRECTORY>(pModuleBase + pImageOptionalHeader.x64->DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress);

                break;
        }
    }
    return pImageExportDirectory;

} // getImageExportDirectory

} // os namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
