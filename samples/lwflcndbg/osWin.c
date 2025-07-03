//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// osWin.c
// Windows OS dependent routines...
//*****************************************************

//
// includes
//
#include <Windows.h> // For performance counters
#include "lw_ref.h"
#include "os.h"
#include "chip.h"
#include "hal.h"
#include "br04.h"
#include "socbrdg.h"
#include "t12x/t124/dev_fai_bar0.h"
#include "t12x/t124/dev_fai_cfg.h"

#include <ctype.h>   // for islower

#define ILWALID_REGISTER32      0xffffffff
#define SEARCH_BUS_NUM          0x20

//
// globals
//
LwU32 CPUFreq = 0;
ULONG OSMajorVersion;
ULONG OSMinorVersion;

//-----------------------------------------------------
// initLwWatch
//
//-----------------------------------------------------
void initLwWatch()
{
    LARGE_INTEGER liFreq;
    LwU32 lwVirtAddr, fbVirtAddr;

    if (usingMods)
    {
        initLwWatchMods();
        return;
    }

    //
    // If lwBar0 has not been set, search for LW
    // devices.
    //
    if (lwBar0 == 0)
    {
        if (FindLWDevice() == FALSE)
        {
            //
            // Nothing found, user must specify lwBar0
            //
            dprintf("flcndbg: The default !flcndbg.init only searches %d buses for LW GPUs.\n", SEARCH_BUS_NUM);
            dprintf("flcndbg: Call !flcndbg.init <lwBar0> to begin.\n");
            dprintf("flcndbg: Find lwBar0 by using !pci 3 ff and using MEM[0] of the Lwpu VGA Controller.\n");
            
            return;
        }
    }

    //
    // Common OS initialization
    //
    osInit();

    //
    // Get our virt addresses if possible
    //
    if (OSMinorVersion> 5000)
    {
        lwVirtAddr = (LwU32) GetExpression("@@(lwlddmkm!LwDBPtr_Table[0]->DBlwAddr)");
        fbVirtAddr = (LwU32) GetExpression("@@(lwlddmkm!LwDBPtr_Table[0]->DBfbAddr)");
    }
    else
    {
        lwVirtAddr = (LwU32) GetExpression("@@(lw4_mini!LwDBPtr_Table[0]->DBlwAddr)");
        fbVirtAddr = (LwU32) GetExpression("@@(lw4_mini!LwDBPtr_Table[0]->DBfbAddr)");
    }

    if (lwVirtAddr)
        dprintf("flcndbg: lwVirtAddr:    0x%08lx\n", lwVirtAddr);
    if (lwVirtAddr)
        dprintf("flcndbg: fbVirtAddr:    0x%08lx\n", fbVirtAddr);

    //
    // Get the frequency of the processor here.
    //
    if (QueryPerformanceFrequency(&liFreq))
    {
        CPUFreq = liFreq.LowPart;
    }
    else
    {
        dprintf("flcndbg: QueryPerformanceFrequency call failed!\n");
    }

    dprintf("flcndbg: Call !flcndbg.help for a command list.\n");
}

//-----------------------------------------------------
// REG_RD32
// - read an lw register
//-----------------------------------------------------
LwU32 REG_RD32(PhysAddr reg)
{
    LwU32 status;
    PhysAddr address = lwBar0 + reg;    // physical address to read. 
    LwU32 buffer;        // address of an array of bytes to hold the data read. 
    LwU32 bytesRead;     // address of a variable to receive the number of bytes 
                        // actually read. 

    //
    // If in dump mode, then read value from dump contents and return
    //
    if (lwMode == MODE_DUMP)
    {
        return REG_RD32_DUMP(reg);
    }
    //
    // If using lwwatch as lwwatchMods, then read value from Mods and return
    //
    if (usingMods) 
    {
        return REG_RD32Mods(reg);    
    }

    if (lwBar0 == 0)    
    {
        dprintf("flcndbg: lwBar0 has not been set, call !flcndbg.init <physAddr>\n");
        return 0;
    }

    // On SOCBRDG we need to access the Register through the sliding window
    if (IsSocBrdg())
    {
        // Get the GPU Tunnel address to read LW_PFB_PRI_MMU_CTRL
        address = (LwU32)pSocbrdg[indexGpu].socbrdgLwGetTunnelAddress(reg);
    }

    status = readPhysicalMem(address, &buffer, sizeof(buffer), &bytesRead);
    if (status != LW_OK)
    {
        dprintf("flcndbg: readPhysicalMem failed! address: 0x%08lx\n", address);
        return 0;
    }

    return buffer;
}

//-----------------------------------------------------
// REG_WR32
// - write an lw register
//-----------------------------------------------------
void REG_WR32(PhysAddr reg, LwU32 data)
{
    PhysAddr address = lwBar0 + reg;   // physical address to write. 
    LwU32 buffer = data;     // address of an array of bytes to hold the data write. 
    LwU32 size;              // number of bytes to write. 
    LwU32 bytesWritten;      // address of a variable to receive the number of 
                            // bytes actually written. 

    //
    // If using as lwwatchMods, then write data thru. Mods and return
    //
    if (usingMods) 
    {
        REG_WR32Mods(reg, data);    
        return;
    }

    if (lwBar0 == 0)
    {
        dprintf("flcndbg: lwBar0 has not been set, call !flcndbg.init <physAddr>\n");
        return;
    }

    // On SOCBRDG we need to access the Register through the sliding window
    if (IsSocBrdg())
    {
        // Get the GPU Tunnel address to read LW_PFB_PRI_MMU_CTRL
        address = (LwU32)pSocbrdg[indexGpu].socbrdgLwGetTunnelAddress(reg);
    }

    size = sizeof(buffer);
    writePhysicalMem(address, &buffer, size, &bytesWritten);

    if (bytesWritten != size)
    {
        dprintf("lw: writePhysicalMem failed! address: 0x%08lx\n", address);
    }
}

//-----------------------------------------------------
// REG_RD08
// - read an lw register
//-----------------------------------------------------
LwU8 REG_RD08(PhysAddr reg)
{
    LwU32 status;
    PhysAddr address = lwBar0 + reg;    // physical address to read. 
    LwU8 buffer;        // address of an array of bytes to hold the data read. 
    LwU32 bytesRead;     // address of a variable to receive the number of bytes 
                        // actually read. 

    //
    // If using lwwatch as lwwatchMods, then read value from Mods and return
    //
    if (usingMods) 
    {
        return REG_RD08Mods(reg);    
    }

    if (lwBar0 == 0)
    {
        dprintf("flcndbg: lwBar0 has not been set, call !flcndbg.init <physAddr>\n");
        return 0;
    }

    // On SOCBRDG we need to access the Register through the sliding window
    if (IsSocBrdg())
    {
        // Get the GPU Tunnel address to read LW_PFB_PRI_MMU_CTRL
        address = (LwU32)pSocbrdg[indexGpu].socbrdgLwGetTunnelAddress(reg);
    }

    status = readPhysicalMem(address, &buffer, sizeof(buffer), &bytesRead);
    if (status != LW_OK)
    {
        dprintf("lw: readPhysicalMem failed! address: 0x%08lx\n", address);
        return 0;
    }
    
    return buffer;
}

//-----------------------------------------------------
// REG_WR08
// - write an lw register
//-----------------------------------------------------
void REG_WR08(PhysAddr reg, LwU8 data)
{
    PhysAddr address = lwBar0 + reg;   // physical address to write. 
    LwU8 buffer = data;     // address of an array of bytes to hold the data write. 
    LwU32 size;              // number of bytes to write. 
    LwU32 bytesWritten;      // address of a variable to receive the number of 
                            // bytes actually written. 

    //
    // If using lwwatch as lwwatchMods, then write value thru. Mods and return
    //
    if (usingMods) 
    {
        REG_WR08Mods(reg, data);    
        return;
    }

    if (lwBar0 == 0)
    {
        dprintf("flcndbg: lwBar0 has not been set, call !flcndbg.init <physAddr>\n");
        return;
    }

    // On SOCBRDG we need to access the Register through the sliding window
    if (IsSocBrdg())
    {
        // Get the GPU Tunnel address to read LW_PFB_PRI_MMU_CTRL
        address = (LwU32)pSocbrdg[indexGpu].socbrdgLwGetTunnelAddress(reg);
    }

    size = sizeof(buffer);
    writePhysicalMem(address, &buffer, size, &bytesWritten);

    if (bytesWritten != size)
    {
        dprintf("lw: writePhysicalMem failed! address: 0x%08lx\n", address);
    }
}

//-----------------------------------------------------
// FB_RD32
// - read FB memory
//-----------------------------------------------------
LwU32 FB_RD32(LwU32 reg)
{
    if (lwMode == MODE_DUMP)
        return FB_RD32_DUMP(reg);
    else if(usingMods)
        return FB_RD32Mods(reg);    
    else
        return RD_PHYS32(lwBar1 + reg);
}

//-----------------------------------------------------
// FB_RD64
// - read FB memory given 64 bit address
// - 64BIT CAUTION: Not yet implemented.
//-----------------------------------------------------
LwU32 FB_RD32_64(LwU64 reg)
{
    //
    // If using lwwatch as lwwatchHw, then return (not implemented)
    //
    LWWATCHMODS_IMPLEMENTED_ONLY_MESSAGE_AND_RETURN_VAL(ILWALID_REGISTER32);
    
    //
    // Else read value from Mods and return
    //
    return FB_RD32((LwU32)reg);    
}

//-----------------------------------------------------
// FB_WR32
// - write FB memory
//-----------------------------------------------------
void FB_WR32(LwU32 reg, LwU32 data)
{
    if(usingMods)
        FB_WR32Mods(reg, data);
    else
       WR_PHYS32(lwBar1 + reg, data);
}

//-----------------------------------------------------
// virtToPhys
//
//-----------------------------------------------------
LwU64 virtToPhys(LwU64 virtAddr, LwU32 pid)
{
    LwU32 physAddr = 0;
    LwU32 byteIndex;
    LwU32 pageTableIndex;
    LwU32 pageDirIndex;
    LwU32 pteAddr;
    LwU32 pte;
    LwU32 pageFrameNumber;

    //
    // Not implemented for lwwatchMods yet
    // 
    PRINT_LWWATCHMODS_NOT_IMPLEMENTED_MESSAGE_AND_RETURN0();

    //
    // This virtual address is a combination of three fields.
    // Bits 0 to 11 are the byte index.
    // - 12 bits because pages are 2^12 (4096K)
    // Bits 12 to 21 are the page table index.
    // Bits 22 to 31 are the page directory index.
    //
    byteIndex = (LwU32)(virtAddr & 0xfff);                     // [11:0]
    pageTableIndex = (LwU32)((virtAddr & 0x3ff000) >> 12);     // [21:12]
    pageDirIndex = (LwU32)((virtAddr & 0xffc00000) >> 22);     // [31:22]
    dprintf("lw: byteIndex:         0x%08lx\n", byteIndex);   
    dprintf("lw: pageTableIndex:    0x%08lx\n", pageTableIndex);   
    dprintf("lw: pageDirIndex:      0x%08lx\n", pageDirIndex); 

    //
    // The size of each PTE: this is four bytes on non-PAE x86 systems 
    // The size of a page: this is 0x1000 bytes 
    // The PTE_BASE physical address: on a non-PAE system this is 0xc0000000 
    //
    // PTE address = PTE_BASE  
    //               + (page directory index) * PAGE_SIZE
    //               + (page table index) * sizeof(MMPTE)
    //
    pteAddr = 0xc0000000
        + (pageDirIndex * 0x1000)
        + (pageTableIndex * 0x4);

    //
    // Read the PTE at pteAddr
    //
    pte = RD_VIRT32(pteAddr);

    //
    // High 20 bits of the PTE are equal to the page frame number (PFN).
    // The first physical address on the physical page is the PFN multiplied
    // by 0x1000 (shifted left 12 bits). 
    // The byte index is the offset on this page.
    //
    pageFrameNumber = ((pte & 0xfffff000) >> 12);   // [31:12]
    physAddr = (pageFrameNumber << 12) + byteIndex;
    dprintf("lw: pteAddr:           0x%08lx\n", pteAddr); 
    dprintf("lw: pte:               0x%08lx\n", pte); 
    dprintf("lw: pageFrameNumber:   0x%08lx\n", pageFrameNumber); 

    return physAddr;
}

//-----------------------------------------------------
// physToVirt
// - FIXME, I am slow...
//-----------------------------------------------------
LwU64 physToVirt(LwU64 physAddr, LwU32 flags)
{
    LwU32 status;
    LwU32 virtAddr = 0;
    LwU32 byteIndex;
    LwS32 pageTableIndex;
    LwS32 pageDirIndex;
    LwU32 pageFrameNumber;
    LwU32 pageDirBase = 0;
    LwU32 pageTableBase;
    LwU32 pde, pte, pfn;
    LwU32 pageDirBuffer[1024];
    LwU32 pageTableBuffer[1024];
    LwU32 bytesRead;

    //
    // Not implemented for lwwatchMods yet
    // 
    PRINT_LWWATCHMODS_NOT_IMPLEMENTED_MESSAGE_AND_RETURN0();

    //
    // Get the Page Frame Number
    //
    pageFrameNumber = (LwU32)(physAddr & 0xfffff000);  // [31:12]
    dprintf("lw: pageFrameNumber:   0x%08lx\n", pageFrameNumber);

    //
    // Get the Byte Index
    //
    byteIndex = (LwU32)(physAddr - pageFrameNumber);
    dprintf("lw: byteIndex:         0x%08lx\n", byteIndex);

    //
    // CR3 - physical address of the page directory table
    //
    pageDirBase = (LwU32) GetExpression("cr3");
    dprintf("lw: pageDirBase:       0x%08lx\n", pageDirBase);

    // PageDirectory (0-1023 PDEs)-> PageTable (0-1023 PTEs)-> 4K Page (4096 ByteIndices)

    // Read in the page directory to a local buffer - reading phys mem a dw at a time
    // is slow
    status = readPhysicalMem(pageDirBase, pageDirBuffer, sizeof(pageDirBuffer), &bytesRead);
    if (status != LW_OK)
        return 0;

    dprintf("lw: Searching Page Directory - this could take awhile...\n");

    //
    // Search through the Page Directory Entries - start at the end b\c
    // I usually find our pde index ~1000. Just a hack.
    //
    for (pageDirIndex = 1023; pageDirIndex >= 0; pageDirIndex--)
    {
        pde = pageDirBuffer[pageDirIndex];

        // mask off lower 12 flag bits
        pageTableBase = (pde & 0Xfffff000);     // [31:12]

        // Make sure the PDE is valid
        if ( (pde & BIT(0)) && pageTableBase )
        {
            // Read in the page table to a local buffer - reading phys mem a dw at a time
            // is slow
            status = readPhysicalMem(pageTableBase, pageTableBuffer, sizeof(pageTableBuffer), &bytesRead);
            if (status != LW_OK)
                return 0;

            //
            // Search through the Page Table Entries
            //
            for (pageTableIndex = 0; pageTableIndex < 1024; pageTableIndex++)
            {
                pte = pageTableBuffer[pageTableIndex];
               
                // mask off lower 12 flag bits
                pfn = (pte & 0Xfffff000);       // [31:12]

                // Make sure the PTE is valid
                if ( (pte & BIT(0)) && pfn )
                {
                    // Did we find a match?
                    if (pfn == pageFrameNumber)
                    {
                        virtAddr = (pageDirIndex << 22);        // [31:22]
                        virtAddr |= (pageTableIndex << 12);     // [21:12]
                        virtAddr |= byteIndex;                  // [11:0]

                        dprintf("lw: pageDirIndex:      0x%08lx\n", pageDirIndex);
                        dprintf("lw: pageTableIndex:    0x%08lx\n", pageTableIndex);
                        dprintf("lw: virtAddr:          0x%08lx\n", virtAddr);

                        if (flags == 0)
                        {
                            goto done;
                        }
                    }
                }
            }
        }
    }

done:
    if (virtAddr == 0)
    {
        dprintf("lw: virtAddr not found...\n");
    }

    return virtAddr;
}

//-----------------------------------------------------
// readPhysicalMem
// [Read/Write]Physical doesn't work with newer WinDBG
// versions on WinXP.  Must use the Flag versions
// #define PHYS_FLAG_DEFAULT        0
// #define PHYS_FLAG_CACHED         1
// #define PHYS_FLAG_UNCACHED       2
// #define PHYS_FLAG_WRITE_COMBINED 3
//-----------------------------------------------------
LwU32 readPhysicalMem(ULONG64 address, PVOID buf, ULONG size, LwU32 *pSizer)
{
    //
    // Not implemented for lwwatchMods yet
    // 
    PRINT_LWWATCHMODS_NOT_IMPLEMENTED_MESSAGE2();
    if (usingMods) 
    {
        *pSizer = 0;
        return LW_ERR_GENERIC;
    }

    if (OS_IS_WINXP_OR_HIGHER(OSMinorVersion))
        ReadPhysicalWithFlags(address, buf, size, PHYS_FLAG_UNCACHED, pSizer);
    else
        ReadPhysical(address, buf, size, pSizer);

    if (size == *pSizer)
        return LW_OK;
    else
    {
        dprintf("lw: size:  0x%08lx\n", size);           
        dprintf("lw: pSizer: 0x%08lx\n", *pSizer);
        return LW_ERR_GENERIC;
    }
}

//-----------------------------------------------------
// writePhysicalMem
//
//-----------------------------------------------------
LwU32 writePhysicalMem(ULONG64 address, PVOID buf, ULONG size, PULONG pSizew)
{
    //
    // Not implemented for lwwatchMods yet
    // 
    PRINT_LWWATCHMODS_NOT_IMPLEMENTED_MESSAGE2();
    if (usingMods) 
    {
        *pSizew = 0;
        return LW_ERR_GENERIC;
    }

    if (OS_IS_WINXP_OR_HIGHER(OSMinorVersion))
        WritePhysicalWithFlags(address, buf, size, PHYS_FLAG_UNCACHED, pSizew);
    else
        WritePhysical(address, buf, size, pSizew);

    if (size == *pSizew)
        return LW_OK;
    else
    {
        dprintf("lw: size:  0x%08lx\n", size);           
        dprintf("lw: pSizew: 0x%08lx\n", *pSizew);
        return LW_ERR_GENERIC;
    }
}

//-----------------------------------------------------
// readVirtMem
//
//-----------------------------------------------------
LwU32 readVirtMem(ULONG64 address, PVOID buf, ULONG size, PULONG pSizer)
{
    LwU8* pOutBuff = (LwU8*) buf;
    const LwU32 readBlockSize = 0x200;
    LwU32 dataLeft, bytesRead;
    LwU32 i;

    //
    // Not implemented for lwwatchMods yet
    //
    PRINT_LWWATCHMODS_NOT_IMPLEMENTED_MESSAGE2();
    if (usingMods) 
    {
        *pSizer = 0;
        return LW_ERR_GENERIC;
    }

    //Read in blocks of 'readBlockSize'
    *pSizer = 0;
    for (i = 0; i < size/readBlockSize; i++) {
        ReadMemory(address + readBlockSize*i,
                   pOutBuff + readBlockSize * i,
                   readBlockSize, &bytesRead);
        *pSizer += bytesRead;
    }

    //Read what's left over that couldn't fit into a full 'readBlockSize'
    dataLeft = size - readBlockSize*i;
    if (dataLeft) {
        ReadMemory(address + readBlockSize*i,
                   pOutBuff + readBlockSize * i,
                   dataLeft, &bytesRead);
        *pSizer += bytesRead;
    }

    if (size == *pSizer)
        return LW_OK;
    else
    {
        dprintf("lw: size:  0x%08lx\n", size);           
        dprintf("lw: pSizer: 0x%08lx\n", *pSizer);
        return LW_ERR_GENERIC;
    }
}

//-----------------------------------------------------
// writeVirtMem
//
//-----------------------------------------------------
LwU32 writeVirtMem(ULONG64 address, PVOID buf, ULONG size, PULONG pSizew)
{
    //
    // Not implemented for lwwatchMods yet
    // 
    if (usingMods) 
    {
        *pSizew = 0;
        return LW_ERR_GENERIC;
    }

    WriteMemory(address, buf, size, pSizew);

    if (size == *pSizew)
        return LW_OK;
    else
    {
        dprintf("lw: size:  0x%08lx\n", size);           
        dprintf("lw: pSizew: 0x%08lx\n", *pSizew);
        return LW_ERR_GENERIC;
    }
}

void printPciSpaceInfo(LwU32 PCIData[][4], LwU32 devNum, LwU32 totNumDevices)
{
    if ( (totNumDevices == 1) || (devNum == 0))
    {
        dprintf("flcndbg: Num DevID Bus Dev Fnc Bar0       Bar1\n");
    }

    dprintf("flcndbg: %2d  %04x   %x  %02x   %x  0x%08x 0x%08x", devNum,
                 (PCIData[devNum][1]>>16),
                 (PCIData[devNum][0] & 0xff),
                 ((PCIData[devNum][0]>>8) & 0xff),
                 ((PCIData[devNum][0]>>16) & 0xff),
                 PCIData[devNum][2],
                 PCIData[devNum][3]);

    if (PCIData[devNum][0] & BIT(24))
        dprintf(" - VGA Device");
    if (PCIData[devNum][0] & BIT(25))
        dprintf(" - 3D Controller Device"); 

    dprintf("\n");
}

//
// isTegraDevice
//
BOOL isTegraDevice()
{
    LwU32 status;
    LwU32 address = 0x70000804;
    LwU32 buffer;
    LwU32 bytesRead;

    status = readPhysicalMem(address, &buffer, sizeof(buffer), &bytesRead);
    if (status != LW_OK)
    {
        //dprintf("lw: readPhysicalMem failed! address: 0x%08lx\n", address);
        return FALSE;
    }

    isTegraHack = buffer;
    if (IsTegra())
    {
        dprintf("flcndbg: CheetAh Device Detected: address: 0x%08lx buffer: 0x%08lx\n", address, buffer);
        return TRUE;
    }

    return FALSE;
}

static void ProcessIsTegraHack()
{
    LwU32 winShift = 0;
    LwU32 winMask = 0;
    LwU32 winTarget = 0;
    LwU32 addrToRead = 0;

    bIsSocBrdg = TRUE;

    // for rev 2 of peatrans, window address is shifted 8 bits to allow 40b addressing
    if((lwClassCodeRevId & DRF_MASK(LW_FAI_CFG_REV_CC_REVISION_ID)) == LW_FAI_CFG_REV_CC_REVISION_ID_2)
    {
        winShift = 8;
        winMask = (1 << winShift) - 1;
    }

    // Save the Window target to restore it back after the read is complete
    winTarget = RD_PHYS32(lwBar0 + LW_FAI_BAR1_WINDOW1);

    // Set the Window to get the chip-id and then do the actual read
    WR_PHYS32(lwBar0 + LW_FAI_BAR1_WINDOW1, 0x70000804 >> winShift);

    // since we may shift out lower 8 bits in window, add them back in for actual read now
    addrToRead = ((lwBar1 & DRF_SHIFTMASK(LW_FAI_CFG_BAR_1_BASE_ADDRESS)) + 128*1024*1024) + 
        (0x70000804 & winMask); 
    isTegraHack = RD_PHYS32(addrToRead);

    // Restore the Window target back
    WR_PHYS32(lwBar0 + LW_FAI_BAR1_WINDOW1, winTarget);
}

// WAR for the fact that IMAGE_FILE_MACHINE_ARMNT is not defined in ddk lwrrently used
// by lwwatch build process 
#ifndef IMAGE_FILE_MACHINE_ARMNT
// defined in nt8 ddk (winnt.h)
#define IMAGE_FILE_MACHINE_ARMNT             0x01c4  // ARM Thumb-2 Little-Endian
#endif

static BOOL isWOT(PDBGKD_GET_VERSION64 pVersionData)
{
    BOOL isWOA = (pVersionData->MachineType == IMAGE_FILE_MACHINE_ARM      ||
                  pVersionData->MachineType == IMAGE_FILE_MACHINE_THUMB    ||
                  pVersionData->MachineType == IMAGE_FILE_MACHINE_ARMNT);

    return isWOA && isTegraDevice();
}

//-----------------------------------------------------
// FindLWDevice
// - Find the LW device in the system.
// - For now we can just leave init in place so if
//   there are more than two cards we can easily switch
//   lwBar0.
//-----------------------------------------------------
BOOL FindLWDevice()
{
    LwU32 VenDevID = 0;
    LwU32 Bus, Device, Function;
    LwU32 PCIData[MaxLWDevices][4];  // Stores the Bus, Device, Func in one DWORD
                                    // VenDevID in the next
                                    // Then BaseAddress 0 in the next and
                                    // finally BaseAddress 1 in the final.
    LwU32 numDevs = 0, numLwDispDevs = 0, vgaDevNum = 0;
    LwU32 ClassCodeRevID[MaxLWDevices];
    LwU32 i;
    LwU32 devId = 0;
    DBGKD_GET_VERSION64 verData;

    dprintf("flcndbg: Searching for LWPU devices...\n");
    
    memset((void*)ClassCodeRevID, 0, MaxLWDevices * sizeof(LwU32));

    //
    // Not implemented for lwwatchMods
    //
    if (usingMods) 
    {
        dprintf("flcndbg: Using fmodel\n");
        lwBar0 = 0;
        lwBar1 = 0;    
        return TRUE;
    }

    if (Ioctl(IG_GET_KERNEL_VERSION, &verData, sizeof(verData)))
    {
        if (isWOT(&verData))
        {
            // WOT doesn't have a PCIE driver/tree so that pcie tree walk is useless
            return TRUE;
        }
    }

    //
    // Loop over bus, Devices, and Functions
    // Though it looks like we're always listed as function 0
    //
    for (Bus = 0; Bus < SEARCH_BUS_NUM; Bus++)
    {
        for(Device=0; Device<32; Device++)
        {
            for(Function=0; Function<1; Function++)
            {
                GetBusData(PCIConfiguration, Bus, Device, Function, &VenDevID, 0, sizeof(VenDevID));

                if((FLD_TEST_DRF_NUM(_BR04_XVU, _DEV_ID, _VENDOR_ID, LW_BR04_XVU_DEV_ID_VENDOR_ID_LWIDIA, VenDevID))
                  && (numDevs < MaxLWDevices))
                {
                    PCIData[numDevs][0] = Bus | (Device << 8) | (Function << 16);
                    PCIData[numDevs][1] = VenDevID;
                    GetBusData(PCIConfiguration, Bus, Device, Function, &(PCIData[numDevs][2]), 0x10, sizeof(LwU32));
                    GetBusData(PCIConfiguration, Bus, Device, Function, &(PCIData[numDevs][3]), 0x14, sizeof(LwU32));
                    
                    GetBusData(PCIConfiguration, Bus, Device, Function, &(ClassCodeRevID[numDevs]), 0x8, sizeof(LwU32));
                    switch (ClassCodeRevID[numDevs] >> 8)
                    {
                        // VGA Display Controller
                        case 0x30000:
                        {
                            vgaDevNum = numDevs;
                       
                            // set a flag to note that this is a vga device
                            PCIData[numDevs][0] |= 0x1<<24;
                            numLwDispDevs++;
                        }
                        break;

                        // 3D Display Controller
                        case 0x30200:
                        {
                            // set a flag to note that this is a 3d device
                            PCIData[numDevs][0] |= 0x1<<25;
                            numLwDispDevs++;
                        }
                        break;
                    }

                    numDevs++;
                }
            }
        }
    }

    if (numDevs >= MaxLWDevices)
    {
        dprintf("flcndbg: We found more than %d Lwpu devices on this machine!\n", MaxLWDevices);
    }

    // Did we only find one graphics device?
    if( (numLwDispDevs == 1) && ( (PCIData[vgaDevNum][0]>>24) == 1 ) )
    {
        printPciSpaceInfo(PCIData, vgaDevNum, numLwDispDevs);

        // We did, so let's go ahead and enable it.
        lwBar0 = PCIData[vgaDevNum][2];
        lwBar1 = PCIData[vgaDevNum][3];
        lwClassCodeRevId = ClassCodeRevID[vgaDevNum];

        dprintf("\n");

        devId = PCIData[vgaDevNum][1] >> 16;

        if (socbrdgIsBridgeDevid(devId))
        {
            ProcessIsTegraHack();
        }

        return TRUE;
    }
    else if (numDevs == 0)
    {
        // See if we are running a CheetAh Device
        if (isTegraDevice())
        {
            return TRUE;
        }
        else
        {
            dprintf("flcndbg: No Lwpu GPU devices were found!\n");
            return FALSE;
        }
    }
    else
    {
        char InputBuffer[4] = "/0";

        dprintf("flcndbg: Lwpu devices found: %d\n", numDevs);
        for(i = 0; i < numDevs; i++)
        {
            printPciSpaceInfo(PCIData, i, numDevs);
        }

        dprintf("flcndbg: Which one do you want to initialize?  Type a number 0-%d.\n", (numDevs - 1));
        if(GetInputLine("lw: ", InputBuffer, 4))
        {
            if((InputBuffer[0] < 0x30) || (InputBuffer[0] > 0x39))
            {
                dprintf("flcndbg: Try using numbers 0-9 for the first character next time.\n");
                return FALSE;
            }

            i = 0;
            i = (LwU32) (InputBuffer[0]-0x30);

            if(InputBuffer[1] != 0)
            {
                if((InputBuffer[1] < 0x30) || (InputBuffer[1] > 0x39))
                {
                    dprintf("flcndbg: Try using numbers 0-9 for the second character next time.\n");
                    return FALSE;
                }
                
                i = i*10;
                i += (LwU32) (InputBuffer[1]-0x30);
                if(InputBuffer[2] != 0)
                {
                    dprintf("flcndbg: This entered value is bigger than we can handle.\n");
                    dprintf("flcndbg: Try numbers 0 - %d next time.\n", (numDevs - 1));
                    return FALSE;
                }
            }

            if(i >= numDevs)
            {
                dprintf("flcndbg: This value is more than the devices found.\n");
                dprintf("flcndbg: Try numbers 0 - %d next time.\n", (numDevs - 1));
                return FALSE;
            }

            // We found a good number, so let's go ahead and enable it.
            lwBar0 = PCIData[i][2];
            lwBar1 = PCIData[i][3];
            lwClassCodeRevId = ClassCodeRevID[i];

            dprintf("\n");

            devId = PCIData[i][1] >> 16;

            if (socbrdgIsBridgeDevid(devId))
            {
                ProcessIsTegraHack();
            }

            return TRUE;
        }
    }

    return FALSE;
}

//-----------------------------------------------------
// ScanLWTopology
// - Find the LW devices in the system.
// - Add them to a hierarchical data structure.
//-----------------------------------------------------
void ScanLWTopology(LwU32 PcieConfigSpaceBase)
{
    LwU32 Bus;
    
    dprintf("flcndbg: Scanning LWPU topology...\n");

    lwNumBR04s = 0;
    
    //
    // Loop over Bus, Devices, and Functions
    //
    for (Bus = 0; Bus < SEARCH_BUS_NUM; Bus++)
    {
        LwU32 Device;
        for(Device=0; Device<32; Device++)
        {
            LwU32 Function;
            for(Function=0; Function<1; Function++)
            {
                LwU32 VenDevID = 0;
                GetBusData(PCIConfiguration, Bus, Device, Function, &VenDevID, 0, sizeof(VenDevID));

                if(FLD_TEST_DRF_NUM(_BR04_XVU, _DEV_ID, _VENDOR_ID, LW_BR04_XVU_DEV_ID_VENDOR_ID_LWIDIA, VenDevID))
                {
                    LwU32 Bar0 = 0;

                    GetBusData(PCIConfiguration, Bus, Device, Function, &Bar0, 0x10, sizeof(LwU32));

                    // Is this a BR04?
                    if ((DRF_VAL(_BR04_XVD, _DEV_ID, _DEVICE_ID, VenDevID) >> 4) == 0x5B)
                    {
                        // Yes, it is.

                        LwU32 reg, portid;
                        char name[64];
                        PLWWATCHTOPOLOGYNODESTRUCT tmp;
                        BOOL isBR04 = FALSE;

                        // Is BAR0 programmed?
                        if (Bar0 == 0)
                        {
                            // No, it is not.  Point it at the PCI Config Space.

                            //
                            // Address is 1110bbbbbbbbdddddfff000000000000:
                            //   'b' are bus bits,
                            //   'd' are device bits,
                            //   'f' are function bits
                            //

                            Bar0 = PcieConfigSpaceBase |
                              ((Bus << 20) & 0x0ff00000) |
                              ((Device << 15) & 0x000f8000) |
                              ((Function << 12) & 0x7000);
                        }

                        reg = REG_RD32(Bar0 + LW_BR04_XVU_LINK_CAP - lwBar0); // Read register LW_BR04_XVU_LINK_CAP
                        portid = DRF_VAL(_BR04_XVU, _LINK_CAP, _PORT_NUMBER, reg);

                        reg = REG_RD32(Bar0 + LW_BR04_XVU_PCIE_CAP - lwBar0); // Read register LW_BR04_XVU_PCIE_CAP
                        switch (DRF_VAL(_BR04_XVU, _PCIE_CAP, _DEVICE_PORT_TYPE, reg))
                        {
                            case 0:
                                sprintf(name, "BR04 PCI Express Endpoint #%d", portid);
                                break;
                            case 1:
                                sprintf(name, "BR04 Legacy PCI Express Endpoint #%d", portid);
                                break;
                            case 4:
                                sprintf(name, "BR04 Root Port of PCI Express Root Complex #%d", portid);
                                break;
                            case LW_BR04_XVU_PCIE_CAP_DEVICE_PORT_TYPE_SW_UP:
                                sprintf(name, "BR04 Board #%d", lwNumBR04s);
                                isBR04 = TRUE;
                                break;
                            case LW_BR04_XVU_PCIE_CAP_DEVICE_PORT_TYPE_SW_DOWN:
                                portid = Device;
                                sprintf(name, "BR04 Port #%d", portid);
                                break;
                            case 7:
                                sprintf(name, "BR04 PCI Express-to-PCI/PCI-X Bridge #%d", portid);
                                break;
                            case 8:
                                sprintf(name, "BR04 PCI/PCI-X to PCI Express Bridge #%d", portid);
                                break;
                            default:
                                sprintf(name, "BR04 Unknown Device #%d", portid);
                                break;
                        }

                        reg = REG_RD32(Bar0 + LW_BR04_XVU_BUS - lwBar0); // Read register LW_BR04_XVU_BUS

                        tmp = br04AddDeviceToTopology(name, Bar0,
                          DRF_VAL(_BR04_XVU, _BUS, _PRI_NUMBER, reg),
                          DRF_VAL(_BR04_XVU, _BUS, _SEC_NUMBER, reg),
                          DRF_VAL(_BR04_XVU, _BUS, _SUB_NUMBER, reg),
                          portid);
                        if (isBR04)
                            lwBR04s[lwNumBR04s++] = tmp;
                    }
                    else
                    {
                        char name[64] = "";

                        LwU32 ClassCodeRevID = 0;
                        GetBusData(PCIConfiguration, Bus, Device, Function, &ClassCodeRevID, 0x8, sizeof(LwU32));
                        switch (ClassCodeRevID >> 8)
                        {
                            // VGA Display Controller
                            case 0x30000:
                            {
                                DeviceIDToString(VenDevID >> 16, name);
                                br04AddDeviceToTopology(name, Bar0, Bus, Bus, Bus, 0xffffffff);
                            }
                            break;

                            // 3D Display Controller
                            case 0x30200:
                            {
                                DeviceIDToString(VenDevID >> 16, name);
                                br04AddDeviceToTopology(name, Bar0, Bus, Bus, Bus, 0xffffffff);
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
}

//-----------------------------------------------------
// osPerfDelay
// 
//-----------------------------------------------------
void osPerfDelay(LwU32 MicroSeconds)
{
    LARGE_INTEGER liCount0, liCount1;
    LwU32 clockstoadd;
    LwU32 i;

    //
    // Not implemented for lwwatchMods yet
    // 
    PRINT_LWWATCHMODS_NOT_IMPLEMENTED_MESSAGE_AND_RETURN();
    
    if(MicroSeconds < 1000)
        return;

    if(MicroSeconds == 0)
        return;

    if(CPUFreq == 0)
        return;

    if(MicroSeconds > (1000))
    {
        Sleep(MicroSeconds/1000);
        return;
    }

    // Microsecond accurate sleep
    QueryPerformanceCounter(&liCount0);

    //
    // Approximate clocks to add by instead of dividing
    // by 1,000,000, just right shift by 20.  We might
    // return earlier than needed, but we can add that.
    //
    // Also break up the >> 20 into 2 >> 10 so that we
    // don't overflow the LwU32 register.
    //
    clockstoadd = (MicroSeconds * (CPUFreq >> 10)) >> 10;
    if((clockstoadd + liCount0.LowPart) < liCount0.LowPart)
    {
        liCount0.HighPart += 1;
    }
    liCount0.LowPart += clockstoadd;

    if(clockstoadd > 1024*1024)
        dprintf(" LWWATCHERR: Delay is longer that 2^10 Microseconds.  We only handle that now.\n");

    i=0;

    do
    {
        QueryPerformanceCounter(&liCount1);

        if(liCount1.HighPart > liCount0.HighPart)
            break;

        if(liCount1.HighPart == liCount0.HighPart)
        {
            if(liCount1.LowPart >= liCount0.LowPart)
                break;
        }

        i++;
    } while(i<1024*1024);  // only allow 1 million iterations right now.
}

//-----------------------------------------------------
// osGetInputLine
// 
//-----------------------------------------------------
LwU32 osGetInputLine(LwU8 *prompt, LwU8 *buffer, LwU32 size)
{
    return GetInputLine(prompt, buffer, size);
}

//-----------------------------------------------------
// RD_VIRT32
// - read virtual address
//-----------------------------------------------------
LwU32 RD_VIRT32(LwU32 virtAddr)
{
    LwU32 buffer;        // address of an array of bytes to hold the data read. 
    LwU32 size;          // number of bytes to read. 
    LwU32 bytesRead;     // address of a variable to receive the number of bytes actually read. 

    //
    // Not implemented for lwwatchMods yet
    // 
    // To implement this, we need to know if virt address is a REG/FB/instmem
    //
    PRINT_LWWATCHMODS_NOT_IMPLEMENTED_MESSAGE2();
    if (usingMods) 
    {
        return ILWALID_REGISTER32;
    }

    size = sizeof(buffer);
    ReadMemory(virtAddr, &buffer, sizeof(buffer), &bytesRead);
    if (bytesRead == size)
    {
        return buffer;
    }
    else
    {
        dprintf("flcndbg: ReadMemory failed! address: 0x%08lx\n", virtAddr);
        return 0;
    }
}

//-----------------------------------------------------
// RD_PHYS32
// - read physical address
//-----------------------------------------------------
LwU32 RD_PHYS32(PhysAddr physAddr)
{
    LwU32 status;
    LwU32 buffer;        // address of an array of bytes to hold the data read. 
    LwU32 bytesRead;     // address of a variable to receive the number of bytes actually read. 

    //
    // Not implemented for lwwatchMods yet
    // 
    // To implement this, we need to know if virt address is a REG/FB/instmem
    //
    PRINT_LWWATCHMODS_NOT_IMPLEMENTED_MESSAGE2();
    if (usingMods) 
    {
        return ILWALID_REGISTER32;
    }

    status = readPhysicalMem(physAddr, &buffer, sizeof(buffer), &bytesRead);
    if (status != LW_OK)
    {
        dprintf("flcndbg: ReadPhysical failed! address: 0x%08lx\n", physAddr);
        return 0;
    }

    return buffer;
}

//-----------------------------------------------------
// WR_PHYS32
// - write physical address
//-----------------------------------------------------
void WR_PHYS32(PhysAddr physAddr, LwU32 data)
{
    LwU32 status;
    LwU32 bytesRead;     // address of a variable to receive the number of bytes actually read. 

    //
    // Not implemented for lwwatchMods yet
    // To implement this, we need to know if virt address is a REG/FB/instmem
    //
    PRINT_LWWATCHMODS_NOT_IMPLEMENTED_MESSAGE_AND_RETURN();


    status = writePhysicalMem(physAddr, &data, sizeof(data), &bytesRead);
    if (status != LW_OK)
    {
        dprintf("flcndbg: WritePhysical failed! address: 0x%08lx\n", physAddr);
    }
}

//-----------------------------------------------------
// SYSMEM_RD32
//
// ----------------------------------------------------
LwU32 SYSMEM_RD32(LwU64 pa)
{
    LwU32 status;
    LwU32 result;
    LwU32 bytesRead;

    if (usingMods)
    {
        return SYSMEM_RD32Mods(pa);
    }

    status = readPhysicalMem(pa, &result, sizeof(result), &bytesRead);
    if (status != LW_OK) {
        dprintf("flcndbg: ReadPhysical failed: address = 0x08x\n", (U032)pa);
        return 0;
    }

    return result;
}

//-----------------------------------------------------
// SYSMEM_RD08
//
//-----------------------------------------------------
LwU8 SYSMEM_RD08(LwU64 pa)
{
    LwU32 status;
    LwU8 result;
    LwU32 bytesRead;

    if (usingMods)
    {
        return SYSMEM_RD08Mods(pa);
    }

    status = readPhysicalMem(pa, &result, sizeof(result), &bytesRead);
    if (status != LW_OK) {
        dprintf("flcndbg: ReadPhysical failed: address = 0x08x\n", (U032)pa);
        return 0;
    }

    return result;
}

//-----------------------------------------------------
// SYSMEM_WR32
// - write to 64 bit physical address
//-----------------------------------------------------
void SYSMEM_WR32(LwU64 pa, LwU32 data)
{
    LwU32 status;
    LwU32 bytesRead;     //bytes actually read. 
  
    status = writePhysicalMem(pa, &data, sizeof(data), &bytesRead);
    if ( (status != LW_OK) || (bytesRead != sizeof(data)) )
    {
        dprintf("flcndbg: WritePhysical failed! address: 0x%08lx\n", pa);
    }
}

//-----------------------------------------------------
// opendir
//
//-----------------------------------------------------
DIR *opendir(const char *dirname)
{
    DIR *d;

    d = (DIR *)malloc(sizeof(DIR));
    if(d == NULL)
    {
        return(NULL);
    }
    d->h = ILWALID_HANDLE_VALUE;
    d->state = DIR_INIT;
    d->dirname = (char *)malloc(strlen(dirname)+1+2+1);
    if(d->dirname == NULL)
    {
        free(d);
        return(NULL);
    }
    sprintf(d->dirname, "%s/*", dirname);

    return(d);
}

//-----------------------------------------------------
// readdir
//
//-----------------------------------------------------
struct dirent *readdir(DIR *d)
{
    static struct dirent *e = NULL;
    WIN32_FIND_DATA data;

    if(e == NULL) 
    {
        e = (struct dirent *)malloc(sizeof(struct dirent));
        if(e == NULL)
        {
            return(NULL);
        }
    }

    switch(d->state)
    {
    case DIR_CLOSED:
        return(NULL);
        break;
    case DIR_INIT:
        if((d->h=FindFirstFile(d->dirname, &data)) == NULL)
        {
            return(NULL);
        }
        e->d_ino = 0;
        e->d_off = -1;
        e->d_reclen = 0;
        strcpy(e->d_name, data.cFileName);
        d->state = DIR_READING;
        return(e);
        break;
    case DIR_READING:
        if(FindNextFile(d->h, &data) == FALSE)
        {
            return(NULL);
        }
        e->d_ino = 0;
        e->d_off = -1;
        e->d_reclen = 0;
        strcpy(e->d_name, data.cFileName);
        return(e);
        break;
    }

    return(NULL);
}

//-----------------------------------------------------
// closedir
//
//-----------------------------------------------------
int closedir(DIR *d)
{
    if (ILWALID_HANDLE_VALUE != d->h)
    {
        FindClose(d->h);
    }
    free(d->dirname);
    free(d);

    return 0;
}

#define lowercase(c) (islower(c) ? (c) : tolower(c))

int strcasecmp(
    const char *s0,
    const char *s1
)
{
    int diff;

    while (((diff = (lowercase(*s0) - lowercase(*s1)))) == 0)
    {
        if (*s0 == 0)
            break;
        s0++;
        s1++;
    }

    return diff;
}

int strncasecmp(
    const char *s0,
    const char *s1,
    int n
)
{
    int diff;

    for ( ; n--; s0++, s1++)
    {
        diff = lowercase(*s0) - lowercase(*s1);
        if (diff)
            return diff;

        if (*s0 == 0)
            break;
    }

    return 0;
}

//-----------------------------------------------------
// osPciRead32
//
//-----------------------------------------------------
LwU32 osPciRead32(LwU16 DomainNumber, LwU8 BusNumber, LwU8 Device, LwU8 Function, LwU32* Buffer, LwU32 Offset)
{
    if(Offset > (0x100 - sizeof(*Buffer)))
        return LW_ERR_GENERIC;
    
    GetBusData(PCIConfiguration, BusNumber, Device, Function, Buffer, Offset, 4);
    return LW_OK;
}

//-----------------------------------------------------
// osPciRead16
//
//-----------------------------------------------------
LwU32 osPciRead16(LwU16 DomainNumber, LwU8 BusNumber, LwU8 Device, LwU8 Function, LwU16* Buffer, LwU32 Offset)
{
    if(Offset > (0x100 - sizeof(*Buffer)))
        return LW_ERR_GENERIC;
    
    GetBusData(PCIConfiguration, BusNumber, Device, Function, Buffer, Offset, 2);
    return LW_OK;
}

//-----------------------------------------------------
// osPciRead08
//
//-----------------------------------------------------
LwU32 osPciRead08(LwU16 DomainNumber, LwU8 BusNumber, LwU8 Device, LwU8 Function, LwU8* Buffer, LwU32 Offset)
{
    if(Offset > (0x100 - sizeof(*Buffer)))
        return LW_ERR_GENERIC;
    
    GetBusData(PCIConfiguration, BusNumber, Device, Function, Buffer, Offset, 1);
    return LW_OK;
}

//-----------------------------------------------------
// osCheckControlC
//
//-----------------------------------------------------
LwU32 osCheckControlC()
{
    LwU32 data = (LwU32) CheckControlC();

    return data;
}

//-----------------------------------------------------
// osGetSystemTime
//
//-----------------------------------------------------
LwU32 osGetSystemTime(LwU64 *dt)
{
  FILETIME ftUTC;
  FILETIME ftLocal;
  GetSystemTimeAsFileTime(&ftUTC);
  FileTimeToLocalFileTime(&ftUTC, &ftLocal);
  FileTimeToDosDateTime(&ftLocal,((LPWORD)dt)+1,((LPWORD)dt)+0);

  return 1;
}
