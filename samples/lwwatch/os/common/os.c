/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2005-2018 by LWPU Corporation.  All rights reserved.  All
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
#include "utils/lwassert.h"
#include "os.h"
#include "chip.h"
#include "hal.h"
#include "fb.h"
#include "mmu.h"
#include "i2c.h"
#include "falctrace.h"
#include "inc/exts.h"
#include "tegrasys.h"
#include "gr.h"
#include "heap.h"

#include "ioaccess/ioaccess.h"

#include "br04.h"
#include "vgpu.h"

// Common for all architecture/implementations (So no HW header has to be included)
#define LW_PMC_BOOT_0_ARCHITECTURE                            28:24 /* R-IVF */
#define LW_PMC_BOOT_0_IMPLEMENTATION                          23:20 /* R-IVF */

//
// globals
//
PhysAddr   lwBar0 = 0;
PhysAddr   lwBar1 = 0;
PhysAddr   lwClassCodeRevId = 0;

PhysAddr   multiGpuBar0[MAX_GPUS];
PhysAddr   multiGpuBar1[MAX_GPUS];
LwU32      multiPmcBoot0[MAX_GPUS];
LwU32      grEngineId;
LwU32      verboseLevel;
LwU32      usingMods = 0;
LwU32      pmc_boot0;
LW_MODE    lwMode = MODE_NONE;
char       osname[10];
//
// This flag determines if we should log the communication because the extension
// and the engine in LWWATCH_OC_DEBUG_FILE
//
int debugOCMFlag;

IO_DEVICE gpuDevice[MAX_GPUS];
///-----------------------------------------------------
/// gpuAperture
/// - top level IO_APERTURE node.
///-----------------------------------------------------
IO_APERTURE gpuAperture[MAX_GPUS];
///-----------------------------------------------------
/// grApertures
/// - GR IO_APERTURE 1:1 mapped with gpuAperture.
///-----------------------------------------------------
GR_IO_APERTURE grApertures[MAX_GPUS];

void initRegAccess();

///-----------------------------------------------------
/// osInit
/// - common OS initialization
///-----------------------------------------------------
LW_STATUS osInit()
{
    // Print out the date and time of compilation
    dprintf("lw: Compiled %s at %s\n", __DATE__, __TIME__);

    // set up default verbose level
    verboseLevel = 1;

    if (!IsTegra())
    {
        ///
        /// LW_PMC_BOOT_0 needs to be read before HAL is wired up, so bypass the
        /// GPU_REG_RD32 mechanism and just read straight from the GPU.
        ///
        pmc_boot0 = osRegRd32(0x0 /* LW_PMC_BOOT_0 */);

        dprintf("lw: LW_PMC_BOOT_0: 0x%08x\n", pmc_boot0);
    }
    // init the hal layer
    if (LW_ERR_GENERIC == osInitHal())
    {
        return LW_ERR_GENERIC;
    }

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch mode bits
    if (isVirtual())
    {
        // Set LwwatchMode as we need some PF registers before this can be set in exts cmds
        setLwwatchMode(LW_TRUE);
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

    falctraceInit();

    return LW_OK;
}

///-----------------------------------------------------
/// osInitHal
/// - common OS HAL initialization (Used to initialize for MGPU_LOOP_START/END
///-----------------------------------------------------
LW_STATUS osInitHal()
{
    LW_STATUS status = LW_ERR_GENERIC;


    // init the hal layer
    if (LW_ERR_GENERIC == initLwWatchHal(pmc_boot0))
    {
        return LW_ERR_GENERIC;
    }

    if (pTegrasys[indexGpu].tegrasysInit(&TegraSysObj[indexGpu]) != LW_OK)
        return LW_ERR_GENERIC;

    if (!IsTegra())
    {
        // set up lwBar1
        if (lwBar1 == 0)
        {
            osInitBar1(&lwBar1);
        }

#if !LWWATCHCFG_IS_PLATFORM(UNIX) && !defined(CLIENT_SIDE_RESMAN)
        br04ClearTopology();
#endif

        ioaccessInitIOAperture(&gpuAperture[indexGpu], NULL, &gpuDevice[indexGpu], 0, 0x00FFFFFF); // size is from DRF_SIZE(LW_RSPACE);
        initRegAccess();
        if ((status = pGr[indexGpu].grConstructIoApertures(&grApertures[indexGpu], &gpuAperture[indexGpu])) != LW_OK)
            return status;
    }
    return LW_OK;
}

///-----------------------------------------------------
/// osInitBar1
///
///-----------------------------------------------------
void osInitBar1(PhysAddr *bar1)
{
    // G8x+ case: LW_PCFG + LW_XVE_BAR1_LO + (LW_XVE_BAR1_HI << 32)
    *bar1 = GPU_REG_RD32(0x88014) + ((LwU64)GPU_REG_RD32(0x88018) << 32);

    *bar1 &= ~(0xf);
}

///-----------------------------------------------------
/// osDestroyHal
///
///-----------------------------------------------------
void osDestroyHal()
{
    pGr[indexGpu].grDestroyIoApertures(&grApertures[indexGpu]);
}

///-----------------------------------------------------
/// REG_RDCR
/// - read an lw cr register
///-----------------------------------------------------
LwU8 REG_RDCR(LwU8 crReg, LwU32 crtcOffset)
{
    LwU32 indexFrom;
    LwU32 indexData;
    LwU8 crVal;

    // set up the indices
    indexFrom = 0x6013d4 + crtcOffset;
    indexData = 0x6013d5 + crtcOffset;

    // get val
    GPU_REG_WR08(indexFrom, crReg);
    crVal = GPU_REG_RD08(indexData);

    return crVal;
}

///-----------------------------------------------------
/// REG_WRCR
/// - write an lw cr register
///-----------------------------------------------------
void REG_WRCR(LwU8 crReg, LwU8 crVal, LwU32 crtcOffset)
{
    LwU32 indexFrom;
    LwU32 indexData;

    // set up the indices
    indexFrom = 0x6013d4 + crtcOffset;
    indexData = 0x6013d5 + crtcOffset;

    // set val
    GPU_REG_WR08(indexFrom, crReg);
    GPU_REG_WR08(indexData, crVal);
}

///-----------------------------------------------------
/// TMDS_RD
/// - read an lw TMDS register
///-----------------------------------------------------
LwU32 TMDS_RD(LwU32 Link, LwU32 index)
{
    LwU32 LinkOffset = 0;
    LwU32 data;

    if(Link == 1)
    {
        LinkOffset = 0x8;
    }
    else if(Link == 2)
      LinkOffset = 0x2000;

    GPU_REG_WR32(0x6808B0 + LinkOffset, (index | 0x10000));
    data = GPU_REG_RD32(0x6808B4 + LinkOffset);

    return data;
}

///-----------------------------------------------------
/// TMDS_WR
/// - write an lw TMDS register
///-----------------------------------------------------
void TMDS_WR(LwU32 Link, LwU32 index, LwU32 data)
{
    LwU32 LinkOffset = 0;

    if(Link == 1)
    {
        LinkOffset = 0x8;
    }
    else if(Link == 2)
      LinkOffset = 0x2000;

    GPU_REG_WR32(0x6808B0 + LinkOffset, (index | 0x10000));
    GPU_REG_WR32(0x6808B4 + LinkOffset, data);
    GPU_REG_WR32(0x6808B0 + LinkOffset, (index) );
    GPU_REG_WR32(0x6808B0 + LinkOffset, (index | 0x10000));
}

///-----------------------------------------------------------------------------
/// osReadMemByType
///
///-----------------------------------------------------------------------------
LW_STATUS osReadMemByType(PhysAddr address, void * buf, LwU64 size, LwU64 *pSizer, MEM_TYPE memoryType)
{
    LW_STATUS status = LW_OK;
    LwU64 i;

    if (NULL == buf)
    {
        dprintf("lw: %s: NULL buffer passed!\n", __FUNCTION__);
        *pSizer = 0;
        return LW_ERR_GENERIC;
    }

    ///
    /// Hoping to return as much as asked
    ///
    *pSizer = size;

    ///
    /// Align if not so.
    ///
    if (size%4 != 0)
    {
        dprintf("lw: %s: size:  0x%08" LwU64_fmtx "\n", __FUNCTION__, size);
        dprintf("lw: %s: pSizer: 0x%08" LwU64_fmtx "\n", __FUNCTION__, *pSizer);
        *pSizer = size - size%4;
        dprintf("lw: %s reading 0x%" LwU64_fmtx " bytes only.\n", __FUNCTION__, *pSizer);
    }

    switch (memoryType)
    {
        case SYSTEM_PHYS:
            status = readPhysicalMem(address, buf, size, pSizer);
            break;

        case SYSTEM_VIRT:
            status = readVirtMem(address, buf, size, pSizer);
            break;

        case FRAMEBUFFER:
            // GPU only
            if (pTegrasys[indexGpu].tegrasysDeviceOn(&TegraSysObj[indexGpu], "GPU", 0)==TRUE)
            {
                for (i = 0 ; i<*pSizer ; i += 4)
                    *((LwU32*)((LwU8*)buf+i)) = FB_RD32((LwU32)(address+i));
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
                    *((LwU32*)((LwU8*)buf+i)) = 
                        isVirtualWithSriov() ? GPU_REG_RD32_DIRECT((LwU32)address+i) : 
                                               GPU_REG_RD32((LwU32)address+i);
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
 * @param[in] length        LwU32 number of bytes to read/write.
 * @param[in] is_write      LwU32 0 for read, otherwise, write
 *
 * @return LW_OK on success, LW_ERR_GENERIC on failure.
 */
LW_STATUS readWriteSystem(LwU64 addr, void* buffer, LwU32 length, LwU32 is_write)
{
    LwU64  lwrrent_addr;
    char*  current;
    LwU64  bytes_left;
    LwU64  transfer_length;
    LwU64  bytes_transferred;
    const  LwU32 block_size = 0x1000;
    LW_STATUS  status = LW_OK;

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
                dprintf("lw: %s writePhysicalMem wrote %" LwU64_fmtu " bytes instead of requested # %" LwU64_fmtu "\n",
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


///-----------------------------------------------------
/// readSystem
///
///-----------------------------------------------------
LW_STATUS readSystem(LwU64 pa, void *buffer, LwU32 length)
{
    if (pMmu[indexGpu].mmuIsGpuIommuMapped(pa, NULL))
        return pFb[indexGpu].fbRead(pa, buffer, length);
    return readWriteSystem(pa, buffer, length, 0);
}

///-----------------------------------------------------
/// writeSystem
///
///-----------------------------------------------------
LW_STATUS writeSystem(LwU64 pa, void* buffer, LwU32 length)
{
    if (pMmu[indexGpu].mmuIsGpuIommuMapped(pa, NULL))
        return pFb[indexGpu].fbWrite(pa, buffer, length);
    return readWriteSystem(pa, buffer, length, 1);
}

///-----------------------------------------------------
/// DEV_REG_RD32
/// - read an lw device register
///-----------------------------------------------------
LwU32 DEV_REG_RD32(PhysAddr reg, const char * const devName, LwU32 devIndex)
{
    PhysAddr address;      // physical address to read.
    LwU32 data = 0;
    PDEVICE_RELOCATION pDev;

    ///
    /// If in dump mode, then read value from dump contents and return
    ///
    if (lwMode == MODE_DUMP)
    {
        return REG_RD32_DUMP(reg);
    }
    ///
    /// If using lwwatch as lwwatchMods, then read value from Mods and return
    ///
    if (usingMods)
    {
        //Not yet supported
        assert(0);
        return 0xffffffff; //GPU_REG_RD32Mods(reg);
    }

    pDev = tegrasysGetDeviceReloc(&TegraSysObj[indexGpu], devName, devIndex);
    assert(pDev);
    //dprintf("lw: %s pDev->start: " PhysAddr_FMT " reg: 0x%08x\n", __FUNCTION__, pDev->start, reg);
    address = pDev->start + reg;

    if (pTegrasys[indexGpu].tegrasysDeviceOn(&TegraSysObj[indexGpu], devName, devIndex)==TRUE)
    {
        data = RD_PHYS32(address);
    }
    else
    {
        dprintf("lw: Register 0x%08x read failed. %s(%d) powered off/in reset\n",
            (LwU32) address, devName, devIndex);
    }

    return data;
}

///-----------------------------------------------------
/// DEV_REG_WR32
/// - write an lw device register
///-----------------------------------------------------
void DEV_REG_WR32(PhysAddr reg, LwU32 data, const char * const devName, LwU32 devIndex)
{
    PhysAddr address;          // physical address to write.
    PDEVICE_RELOCATION pDev;

    ///
    /// If using as lwwatchMods, then write data thru. Mods and return
    ///
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

    if (pTegrasys[indexGpu].tegrasysDeviceOn(&TegraSysObj[indexGpu], devName, devIndex)==TRUE)
    {
        WR_PHYS32(address, data);
    }
    else
    {
        dprintf("lw: Register 0x%08x write failed. %s(%d) powered off/in reset\n",
            (LwU32) address, devName, devIndex);
    }
}

static LwU32 GPU_REG_RD32_COMMON(PhysAddr reg)
{
    LwU32 buffer = 0;   // address of an array of bytes to hold the data read.

    if (pTegrasys[indexGpu].tegrasysDeviceOn(&TegraSysObj[indexGpu], "GPU", 0)==TRUE)
    {
        buffer = pGr[indexGpu].grReadReg32(reg);
    }
    else
    {
        dprintf("lw: Register 0x%08x read failed. %s powered off/in reset\n",
            (LwU32) reg, "GPU");
    }

    return buffer;

}
///-----------------------------------------------------
/// GPU_REG_RD32
/// - read an lw GPU register
///-----------------------------------------------------
LwU32 GPU_REG_RD32(PhysAddr reg)
{
    if (isVirtualWithSriov())
    {
        return pfRegRead(reg);
    }
    return GPU_REG_RD32_DIRECT(reg);
}

void GPU_REG_WR32_COMMON(PhysAddr reg, LwU32 data)
{
    if (pTegrasys[indexGpu].tegrasysDeviceOn(&TegraSysObj[indexGpu], "GPU", 0)==TRUE)
    {
        pGr[indexGpu].grWriteReg32(reg, data);
    }
    else
    {
        dprintf("lw: Register 0x%08x write failed. %s powered off/in reset\n",
            (LwU32) reg, "GPU");
    }
}

///-----------------------------------------------------
/// GPU_REG_WR32
/// - write an lw GPU register
///-----------------------------------------------------
void GPU_REG_WR32(PhysAddr reg, LwU32 data)
{
    if (isVirtualWithSriov())
    {
        pfRegWrite(reg, data);
    }
    else
    {
        GPU_REG_WR32_COMMON(reg, data);
    }
}

#if !(LWWATCHCFG_IS_PLATFORM(UNIX))
LwU32 GPU_REG_RD32_DIRECT(PhysAddr reg)
{
    return GPU_REG_RD32_COMMON(reg);
}

void GPU_REG_WR32_DIRECT(PhysAddr reg, LwU32 data)
{
    GPU_REG_WR32_COMMON(reg, data);
}
#endif


///-----------------------------------------------------
/// GPU_REG_RD08
/// - read a byte from an lw GPU register
///-----------------------------------------------------
LwU8 GPU_REG_RD08(PhysAddr reg)
{
    LwU8 buffer = 0;   // address of an array of bytes to hold the data read.

    if (pTegrasys[indexGpu].tegrasysDeviceOn(&TegraSysObj[indexGpu], "GPU", 0)==TRUE)
    {

        buffer = pGr[indexGpu].grReadReg08(reg);
    }
    else
    {
        dprintf("lw: Register 0x%08x read failed. %s powered off/in reset\n",
            (LwU32) reg, "GPU");
    }

    return buffer;
}

///-----------------------------------------------------
/// GPU_REG_WR08
/// - write a byte to an lw GPU register
///-----------------------------------------------------
void GPU_REG_WR08(PhysAddr reg, LwU8 data)
{
    if (pTegrasys[indexGpu].tegrasysDeviceOn(&TegraSysObj[indexGpu], "GPU", 0)==TRUE)
    {
        pGr[indexGpu].grWriteReg08(reg, data);
    }
    else
    {
        dprintf("lw: Register 0x%08x write failed. %s powered off/in reset\n",
            (LwU32) reg, "GPU");
    }
}

///-----------------------------------------------------
/// IsAndroid
/// - Checks whether the current OS is Android
///-----------------------------------------------------
BOOL IsAndroid()
{
    return !strcmp(osname, "android");
}

#define IS_OFFSET_IN_APERTURE(pAperture, offset)                                            \
    ((pAperture)->baseAddress + (offset) > (pAperture)->baseAddress + (pAperture)->length)

#define NULL_AND_ADDRESS_RANGE_CHECK_FOR_WRITE(pAperture, offset) do {                      \
        if ((pAperture) == NULL) {                                                          \
            dprintf("lw: NULL address.\n");                                                 \
            return;                                                                         \
        }                                                                                   \
        if (IS_OFFSET_IN_APERTURE((pAperture), (offset))) {                                 \
            dprintf("lw: Address range out of sub-device address.\n");                      \
            return;                                                                         \
        }                                                                                   \
    } while(0)


#define NULL_AND_ADDRESS_RANGE_CHECK_FOR_READ(pAperture, offset) do {                       \
        if ((pAperture) == NULL) {                                                          \
            dprintf("lw: NULL address.\n");                                                 \
            return -1;                                                                      \
        }                                                                                   \
        if (IS_OFFSET_IN_APERTURE((pAperture), (offset))) {                                 \
            dprintf("lw: Address range out of sub-device address.\n");                      \
            return -1;                                                                      \
        }                                                                                   \
    } while(0)

///-----------------------------------------------------
/// _WriteReg08
/// - write a byte to an lw GPU register
///-----------------------------------------------------
void _WriteReg08(PIO_APERTURE pAperture, LwU32 addr, LwV8 value)
{
    NULL_AND_ADDRESS_RANGE_CHECK_FOR_WRITE(pAperture, addr);
    osRegWr08(pAperture->baseAddress + addr, value);
}

///-----------------------------------------------------
/// _ReadReg08
/// - Read a bytes from an lw GPU register
///-----------------------------------------------------
LwU8 _ReadReg08(PIO_APERTURE pAperture, LwU32 addr)
{
    NULL_AND_ADDRESS_RANGE_CHECK_FOR_READ(pAperture, addr);
    return osRegRd08(pAperture->baseAddress + addr);
}

///-----------------------------------------------------
/// _WriteReg32
/// - write 4 bytes to an lw GPU register
///-----------------------------------------------------
void _WriteReg32(PIO_APERTURE pAperture, LwU32 addr, LwV32 value)
{
    NULL_AND_ADDRESS_RANGE_CHECK_FOR_WRITE(pAperture, addr);
    osRegWr32(pAperture->baseAddress + addr, value);
}

///-----------------------------------------------------
/// _ReadReg32
/// - Read 4 bytes from an lw GPU register
///-----------------------------------------------------
LwU32 _ReadReg32(PIO_APERTURE pAperture, LwU32 addr)
{
    NULL_AND_ADDRESS_RANGE_CHECK_FOR_READ(pAperture, addr);
    return osRegRd32(pAperture->baseAddress + addr);
}

void initRegAccess()
{
    gpuDevice[indexGpu].pReadReg032Fn = _ReadReg32;
    gpuDevice[indexGpu].pWriteReg032Fn = _WriteReg32;

    gpuDevice[indexGpu].pReadReg016Fn = NULL;
    gpuDevice[indexGpu].pWriteReg016Fn = NULL;

    gpuDevice[indexGpu].pReadReg008Fn = _ReadReg08;
    gpuDevice[indexGpu].pWriteReg008Fn = _WriteReg08;

    ioaccessInitIOAperture(&gpuAperture[indexGpu], NULL, &gpuDevice[indexGpu], 0, 0x00FFFFFF); // size is from DRF_SIZE(LW_RSPACE);
}
