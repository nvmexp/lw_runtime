/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2005-2011 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// os.c
//
//*****************************************************

//
// includes
//
#include "lwwatch.h"
#include "lwtypes.h"
#include "os.h"
#include "chip.h"
#include "hal.h"
#include "lw_ref.h"
#include "lw40/dev_dac.h"
#include "i2c.h"
#include "fermi/gf100/dev_master.h"
#include "inc/exts.h"
#include "socbrdg.h"

#if !defined(MINIRM)
#include "br04.h"
#endif

//
// globals
//
PhysAddr   lwBar0 = 0;
PhysAddr   lwBar1 = 0;
PhysAddr   lwClassCodeRevId = 0;

PhysAddr   multiGpuBar0[8];
PhysAddr   multiGpuBar1[8];
U032    verboseLevel;
U032    usingMods = 0;
U032 pmc_boot0;
LW_MODE lwMode = MODE_NONE;

//
// This flag determines if we should log the communication because the extension
// and the engine in LWWATCH_OC_DEBUG_FILE
// 
int debugOCMFlag;

//-----------------------------------------------------
// osInit
// - common OS initialization
//-----------------------------------------------------
U032 osInit()
{
    U032 regValue;

    dprintf("\n"
            "******************************************************************\n"
            "*  WARNING                                              WARNING  *\n"
            "*                                                                *\n"
            "*                          INTERNAL TOOL                         *\n"
            "*                    Not for use outside LWPU                  *\n"
            "*                                                                *\n"
            "*  WARNING                                              WARNING  *\n"
            "******************************************************************\n");

    // Print out the date and time of compilation
    dprintf("flcndbg: Compiled %s at %s\n", __DATE__, __TIME__);

    // set up default verbose level
    verboseLevel = 1;

    if (IsTegra())
    {
        // init the hal layer
        if (LW_ERR_GENERIC == initLwWatchHal(pmc_boot0))
        {
            return LW_ERR_GENERIC;
        }

        if (pTegrasys[indexGpu].tegrasysInit(&TegraSysObj[indexGpu]) != LW_OK)
            return LW_ERR_GENERIC;

        if (IsSocBrdg())
        {
            // initialize the SOCBRDG here.
            if (pSocbrdg[indexGpu].socbrdgInit() != LW_OK)
                return LW_ERR_GENERIC;
        }
    }
    else
    {
        //
        // LW_PMC_BOOT_0 needs to be read before HAL is wired up, so bypass the 
        // GPU_REG_RD32 mechanism and just read straight from the GPU.
        //
        pmc_boot0 = REG_RD32(LW_PMC_BOOT_0);
        dprintf("flcndbg: LW_PMC_BOOT_0: 0x%08x\n", pmc_boot0);

        // init the hal layer
        if (LW_ERR_GENERIC == initLwWatchHal(pmc_boot0))
        {
            return LW_ERR_GENERIC;
        }

        if (pTegrasys[indexGpu].tegrasysInit(&TegraSysObj[indexGpu]) != LW_OK)
            return LW_ERR_GENERIC;

        // set up lwBar1
        if (lwBar1 == 0)
        {
            osInitBar1(&lwBar1);
        }

        //
        // set the crystal
        //
        if (IsLW17orBetter()) 
        {
            regValue = (GPU_REG_RD_DRF(_PEXTDEV, _BOOT_0, _STRAP_CRYSTAL1) << 1) | 
                        GPU_REG_RD_DRF(_PEXTDEV, _BOOT_0, _STRAP_CRYSTAL0);
        }
        else 
        {
            regValue = GPU_REG_RD_DRF(_PEXTDEV, _BOOT_0, _STRAP_CRYSTAL);
        }
    
        if (regValue & (LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL1_27000K << 1))
            hal.chipInfo.CrystalFreq = 27000000;
        else if (regValue == LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL_13500K)
            hal.chipInfo.CrystalFreq = 13500000;
        else if (regValue == LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL_14318180)
            hal.chipInfo.CrystalFreq = 14318180;
        else
            hal.chipInfo.CrystalFreq = 0;

#if !LWWATCHCFG_IS_PLATFORM(UNIX) && !LWWATCHCFG_IS_PLATFORM(OSX) && !defined(CLIENT_SIDE_RESMAN)
        br04ClearTopology();
#endif
    }
  
    //
    // XXX Remove me once DCBs are read correctly
    //
#if !defined(USERMODE)
    ICBEntry[0].CR.Access   = 0;
    ICBEntry[0].CR.WriteIdx = 0x3F;
    ICBEntry[0].CR.ReadIdx  = 0x3E;
    ICBEntry[1].CR.Access   = 0;
    ICBEntry[1].CR.WriteIdx = 0x37;
    ICBEntry[1].CR.ReadIdx  = 0x36;
    ICBEntry[2].CR.Access   = 0;
    ICBEntry[2].CR.WriteIdx = 0x51;
    ICBEntry[2].CR.ReadIdx  = 0x50;
    ICBEntry[3].CR.Access   = 0x7;
#endif

    if (getelw("LWW_CLASS_SDK") == NULL)
    {
        dprintf("lw: The environment variable LWW_CLASS_SDK was not found.\n"); 
        dprintf("lw: Set this to a local drive to accelerate parsing of class headers.\n");
        dprintf("lw: For example, LWW_CLASS_SDK = " CLASS_DIR_EXAMPLE "\n");
    }

    return LW_OK;
}

//-----------------------------------------------------
// osInitBar1
//
//-----------------------------------------------------
void osInitBar1(PhysAddr *bar1)
{
    if (IsLW10Arch() || IsLW20Arch() || IsLW30Arch() || IsLW40Arch())
    {
        // Pre-G8x case
        *bar1 = GPU_REG_RD32(LW_PBUS_PCI_LW_5);
    }
    else
    {
        // G8x+ case: LW_PCFG + LW_XVE_BAR1_LO
        *bar1 = GPU_REG_RD32(0x88014);
    }

    *bar1 &= ~(0xf);
}

//-----------------------------------------------------
// REG_RDCR
// - read an lw cr register
//-----------------------------------------------------
U008 REG_RDCR(U008 crReg, U032 crtcOffset)
{
    U032 indexFrom;
    U032 indexData;
    U008 crVal;

    // set up the indices
    indexFrom = 0x6013d4 + crtcOffset;
    indexData = 0x6013d5 + crtcOffset;

    // get val
    GPU_REG_WR08(indexFrom, crReg);
    crVal = GPU_REG_RD08(indexData);

    return crVal;
}

//-----------------------------------------------------
// REG_WRCR
// - write an lw cr register
//-----------------------------------------------------
VOID REG_WRCR(U008 crReg, U008 crVal, U032 crtcOffset)
{
    U032 indexFrom;
    U032 indexData;

    // set up the indices
    indexFrom = 0x6013d4 + crtcOffset;
    indexData = 0x6013d5 + crtcOffset;

    // set val
    GPU_REG_WR08(indexFrom, crReg);
    GPU_REG_WR08(indexData, crVal);
}

//-----------------------------------------------------
// TMDS_RD
// - read an lw TMDS register
//-----------------------------------------------------
U032 TMDS_RD(U032 Link, U032 index)
{
    U032 LinkOffset = 0;
    U032 data;

    if(Link == 1)
    {
        if(IsLW17orBetter())
          LinkOffset = 0x8;
        else
          LinkOffset = 0x2000;
    }
    else if(Link == 2)
      LinkOffset = 0x2000;

    GPU_REG_WR32(0x6808B0 + LinkOffset, (index | 0x10000));
    data = GPU_REG_RD32(0x6808B4 + LinkOffset);

    return data;
}

//-----------------------------------------------------
// TMDS_WR
// - write an lw TMDS register
//-----------------------------------------------------
VOID TMDS_WR(U032 Link, U032 index, U032 data)
{
    U032 LinkOffset = 0;

    if(Link == 1)
    {
        if(IsLW17orBetter())
          LinkOffset = 0x8;
        else
          LinkOffset = 0x2000;
    }
    else if(Link == 2)
      LinkOffset = 0x2000;

    GPU_REG_WR32(0x6808B0 + LinkOffset, (index | 0x10000));
    GPU_REG_WR32(0x6808B4 + LinkOffset, data);
    GPU_REG_WR32(0x6808B0 + LinkOffset, (index) );
    GPU_REG_WR32(0x6808B0 + LinkOffset, (index | 0x10000));
}

//-----------------------------------------------------------------------------
// osReadMemByType
// 
//-----------------------------------------------------------------------------
U032 osReadMemByType(PhysAddr address, void * buf, ULONG size, U032 *pSizer, MEM_TYPE memoryType)
{
    U032 i, status = LW_OK;
    ULONG Sizer;
   
    if (NULL == buf)
    {
        dprintf("lw: %s: NULL buffer passed!\n", __FUNCTION__);
        *pSizer = 0;
        return LW_ERR_GENERIC;
    }

    //
    // Hoping to return as much as asked
    // 
    *pSizer = size;
    
    //
    // Align if not so.
    // 
    if (size%4 != 0)
    {
        dprintf("lw: %s: size:  0x%08lx\n", __FUNCTION__, size);
        dprintf("lw: %s: pSizer: 0x%08x\n", __FUNCTION__, *pSizer);
        *pSizer = size - size%4;
        dprintf("lw: %s reading 0x%x bytes only.\n", __FUNCTION__, *pSizer);
    }
    
    switch (memoryType)
    {
        case SYSTEM_PHYS:
            status = readPhysicalMem(address, buf, size, pSizer);
            break;

        case SYSTEM_VIRT:
            status = readVirtMem(address, buf, size, &Sizer);
            *pSizer = Sizer;
            break;

        case FRAMEBUFFER:
            // GPU only
            if (pTegrasys[indexGpu].tegrasysDeviceOn(&TegraSysObj[indexGpu], "GPU", 0)==TRUE)
            {
                for (i = 0 ; i<*pSizer ; i += 4)
                    *((U032*)((U008*)buf+i)) = FB_RD32((U032)address+i);
            }
            else
            {
                dprintf("lw: GPU FB read failed. %s powered off/in reset\n",
                    "GPU");
                status = LW_ERR_GENERIC;
            }
            break;

        case REGISTER:
            if (pTegrasys[indexGpu].tegrasysDeviceOn(&TegraSysObj[indexGpu], "GPU", 0)==FALSE)
            {
                dprintf("lw: Register 0x%08x read failed. %s powered off/in reset\n",
                    (LwU32) address, "GPU");
            }
            else
            {
                for (i = 0 ; i<*pSizer ; i += 4)
                    *((U032*)((U008*)buf+i)) = GPU_REG_RD32((U032)address+i);
            }
            break;

        default:
            dprintf("lw: %s: Not implemented.\n", __FUNCTION__);
            *pSizer = 0;
            status = LW_ERR_GENERIC;
    }
    
    return status;
}

/*!
 * @brief Read/write from/to physical memory.
 *
 * If is_write == 0, reads from physical memory at the given address for
 * the given number of bytes and stores it in the buffer.
 * If is_write != 0, writes to physical memory at the given address for
 * the given number of bytes with the contents of buffer.
 * Note that function doesn't align addr by DWORD.
 * 
 * @param[in] addr          LwU64 address in physical memory to read/write.
 * @param[in, out] buffer   void * pointer to buffer, 
 *                          function reads/writes into/from this buffer.
 * @param[in] length        U032 number of bytes to read/write.
 * @param[in] is_write      U032 0 for read, otherwise, write
 * 
 * @return LW_OK on success, LW_ERR_GENERIC on failure.
 */
U032 readWriteSystem(LwU64 addr, void* buffer, U032 length, U032 is_write)
{
    LwU64  lwrrent_addr;
    char*  current;
    U032   bytes_left;
    U032   transfer_length;
    U032   bytes_transferred;
    const  U032 block_size = 0x1000;
    LwU32  status = LW_OK;

    if (buffer == NULL)
    {
        return LW_ERR_GENERIC;
    }

    lwrrent_addr = addr;
    current = (char*)buffer;
    bytes_left = length;

    // Loop through writing up to a block_size at a time
    while (bytes_left > 0)
    {
        // Check for early exit
        if (osCheckControlC())
        {
            status = LW_ERR_GENERIC;
            break;
        }

        transfer_length = min(block_size, bytes_left);
        if (is_write)
        {
            // Write physical memory
            writePhysicalMem(lwrrent_addr, current, transfer_length, &bytes_transferred);
            if (bytes_transferred != transfer_length)
            {
                dprintf("lw: %s writePhysicalMem wrote %d bytes instead of requested # %d\n",
                        __FUNCTION__, bytes_transferred, transfer_length);
            }
        }
        else
        {
            // Read physical memory
            status = readPhysicalMem(lwrrent_addr, current, transfer_length, &bytes_transferred);
            if (status != LW_OK)
            {
                dprintf("lw: %s: readPhysicalMem() failed\n", __FUNCTION__);
                break;
            }
        }
        // Update tracking
        lwrrent_addr += transfer_length;
        current += transfer_length;
        bytes_left -= transfer_length;
    }

    return status;
}

#if 0
//-----------------------------------------------------
// readSystem
//
//-----------------------------------------------------
U032 readSystem(LwU64 pa, void *buffer, U032 length)
{
    if (pMmu[indexGpu].mmuIsGpuIommuMapped(pa, NULL))
        return pFb[indexGpu].fbRead(pa, buffer, length);
    return readWriteSystem(pa, buffer, length, 0);
}

//-----------------------------------------------------
// writeSystem
//
//-----------------------------------------------------
U032 writeSystem(LwU64 pa, void* buffer, U032 length)
{
    if (pMmu[indexGpu].mmuIsGpuIommuMapped(pa, NULL))
        return pFb[indexGpu].fbWrite(pa, buffer, length);
    return readWriteSystem(pa, buffer, length, 1);
}
#endif

//-----------------------------------------------------
// DEV_REG_RD32
// - read an lw device register
//-----------------------------------------------------
U032 DEV_REG_RD32(PhysAddr reg, const char * const devName, U032 devIndex)
{
    U032  status;
    PhysAddr address;      // physical address to read. 
    U032  buffer = 0;   // address of an array of bytes to hold the data read. 
    U032  bytesRead;    // address of a variable to receive the number of bytes 
                        // actually read.
    PDEVICE_RELOCATION pDev;

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
        //Not yet supported
        assert(0);
        return 0xffffffff; //GPU_REG_RD32Mods(reg);    
    }

    pDev = tegrasysGetDeviceReloc(&TegraSysObj[indexGpu], devName, devIndex);
    assert(pDev);
    //dprintf("lw: %s pDev->start: 0x" PhysAddr_FMT " reg: 0x%08x\n", __FUNCTION__, pDev->start, reg);
    address = pDev->start + reg;

    // On SOCBRDG we need to access the Register through the sliding window
    if (IsSocBrdg())
    {
        // Get the GPU Tunnel address to read LW_PFB_PRI_MMU_CTRL
        address = pSocbrdg[indexGpu].socbrdgLwGetTunnelAddress(address);
    }

    if (pTegrasys[indexGpu].tegrasysDeviceOn(&TegraSysObj[indexGpu], devName, devIndex)==TRUE)
    {
        status = readPhysicalMem(address, &buffer, sizeof(buffer), &bytesRead);
        if (status != LW_OK)
        {
            dprintf("lw: readPhysicalMem failed! address: 0x%llx\n", address);
            return 0;
        }
    }
    else
    {
        dprintf("lw: Register 0x%08x read failed. %s(%d) powered off/in reset\n",
            (LwU32) address, devName, devIndex);
    }

    return buffer;
}

//-----------------------------------------------------
// DEV_REG_WR32
// - write an lw device register
//-----------------------------------------------------
VOID DEV_REG_WR32(PhysAddr reg, U032 data, const char * const devName, U032 devIndex)
{
    PhysAddr address;          // physical address to write. 
    U032  buffer = data;    // address of an array of bytes to hold the data write. 
    U032  size;             // number of bytes to write. 
    U032  bytesWritten;     // address of a variable to receive the number of 
                            // bytes actually written.
    PDEVICE_RELOCATION pDev;

    //
    // If using as lwwatchMods, then write data thru. Mods and return
    //
    if (usingMods) 
    {
        //Not yet supported
        assert(0);
        //GPU_REG_WR32Mods(reg, data);    
        return;
    }

    pDev = tegrasysGetDeviceReloc(&TegraSysObj[indexGpu], devName, devIndex);
    assert(pDev);
    address = pDev->start + reg;

    // On SOCBRDG we need to access the Register through the sliding window
    if (IsSocBrdg())
    {
        // Get the GPU Tunnel address to read LW_PFB_PRI_MMU_CTRL
        address = (LwU32)pSocbrdg[indexGpu].socbrdgLwGetTunnelAddress(address);
    }

    if (pTegrasys[indexGpu].tegrasysDeviceOn(&TegraSysObj[indexGpu], devName, devIndex)==TRUE)
    {
        size = sizeof(buffer);
        writePhysicalMem(address, &buffer, size, &bytesWritten);

        if (bytesWritten != size)
        {
            dprintf("lw: writePhysicalMem failed! address: 0x%llx\n", address);
        }
    }
    else
    {
        dprintf("lw: Register 0x%08x write failed. %s(%d) powered off/in reset\n",
            (LwU32) address, devName, devIndex);
    }
}

//-----------------------------------------------------
// GPU_REG_RD32
// - read an lw GPU register
//-----------------------------------------------------
U032 GPU_REG_RD32(PhysAddr reg)
{
    U032  buffer = 0;   // address of an array of bytes to hold the data read. 

    if (pTegrasys[indexGpu].tegrasysDeviceOn(&TegraSysObj[indexGpu], "GPU", 0)==TRUE)
    {
        buffer = REG_RD32(reg);
    }
    else
    {
        dprintf("lw: Register 0x%08x read failed. %s powered off/in reset\n",
            (LwU32) reg, "GPU");
    }

    return buffer;
}

//-----------------------------------------------------
// GPU_REG_WR32
// - write an lw GPU register
//-----------------------------------------------------
VOID GPU_REG_WR32(PhysAddr reg, U032 data)
{
    if (pTegrasys[indexGpu].tegrasysDeviceOn(&TegraSysObj[indexGpu], "GPU", 0)==TRUE)
    {
        REG_WR32(reg,data);
    }
    else
    {
        dprintf("lw: Register 0x%08x write failed. %s powered off/in reset\n",
            (LwU32) reg, "GPU");
    }
}
