//*****************************************************
//
// lwwatch linux module Extension for gdb
// adimitrov@lwpu.com - 27.06.2007
// simLinux.c
// Linux OS dependent routines for MODS...
//*****************************************************

#define RM_PAGE_SIZE                                     4096

#define CLASS_CODE_VIDEO_CONTROLLER_VGA                  0x030000
#define CLASS_CODE_VIDEO_CONTROLLER_3D                   0x030200

#define ATTRIB_UC                                               1
#define PROTECT_READABLE                                 0x1
#define PROTECT_WRITEABLE                                0x2
#define PROTECT_EXELWTABLE                               0x4
#define PROTECT_READ_WRITE                              (PROTECT_READABLE | PROTECT_WRITEABLE)
#define LW_CONFIG_PCI_LW_7(i)                           (0x0000001C+(i)*4) /* R--4A */

#define LW_CONFIG_PCI_LW_4                               0x00000010 /* RW-4R */

#define MC_PARENT_DEVICE                                 1

#include "lwwatch.h"
#include "lwtypes.h"
#include "lwstatus.h"
#include "os.h"
#include "hwref/lwutil.h"
#include "lw_ref.h"
#include <stdarg.h>                 // varargs

#ifdef XP_UNIX
int vsscanf(const char *str, const char *format, va_list ap);
#endif

#define memcpy  lw_memcpy
#define memcmp  lw_memcmp
#define sprintf lw_sprintf
#define sscanf  lw_sscanf

#include "rm.h"
#include "exts.h"

// From mods
// Conflict between rh72 linux/types.h and sys/types.h
//#define _SYS_TYPES_H 1
#include "core/include/modsdrv.h"
#include "t12x/t124/project_relocation_table.h"

#include "chip.h" // isTegraHack

// Device register access array.
typedef union GPUHWREG
{
    volatile LwV8  Reg008[1];
    volatile LwV16 Reg016[1];
    volatile LwV32 Reg032[1];
} GPUHWREG;

const char * const LWWATCH_OC_DEBUG_FILE="lwwatch.txt";

//
// Here we keep the CPU virtual address for BAR0 as it was
// mapped by MODS for us.
//
static GPUHWREG *regBase = NULL;

#ifdef NO_MODS_GPU_DEVICE
typedef PciDevInfo ModsGpuSubdevice;

typedef struct ModsGpuDevice_t
{
    UINT32                  NumSubdevices; // number of chips, always 1+
    ModsGpuSubdevice       *Subdevices[MC_MAX_SUBDEVICES];
    struct ModsGpuDevice_t *pNext;
} ModsGpuDevice;
#endif

//
// We need a main and sub devices to collect data about
// the gpu we are working with.
//
static ModsGpuDevice *mainDev = 0;
static ModsGpuSubdevice *subDev = 0;

//------------------------------------------------------------------------------
// RM_BOOL LwWatchFindDevices
//
// Here we find the device that we will be working with and get some
// information about it. The most important is BAR0 CPU virtual address after
// it is mapped to such an address by MODS.
//------------------------------------------------------------------------------

RM_BOOL LwWatchFindDevices
(
    ModsGpuDevice ** ppDevices
)
{
    ModsGpuDevice * pLastDev;
    UINT32          Index;
    UINT32          ClassCodeIndex;
    RM_BOOL         bSuccess = FALSE;

#if 0
    DBG_PRINT_STRING(DBG_LEVEL_INFO, "LWRM: RmFindDevices:\n");

    RM_ASSERT(ppDevices != 0);
#endif

    *ppDevices      = 0;
    pLastDev        = 0;

    // Examine each PCI addressable video card, looking for LWpu devices.

    for (ClassCodeIndex = 0; ClassCodeIndex < 2; ++ClassCodeIndex)
    {
        UINT32 ThisClassCode = (ClassCodeIndex == 0) ? CLASS_CODE_VIDEO_CONTROLLER_VGA :
                                                       CLASS_CODE_VIDEO_CONTROLLER_3D;

        for (Index = 0; /* until all devices found or error */; ++Index)
        {
            UINT32 Domain, Bus, Device, Function;

            PciReturnCode PciRc = ModsDrvFindPciClassCode(ThisClassCode,
                Index, &Domain, &Bus, &Device, &Function);

            // Found a video controller.
            if (PCI_OK == PciRc)
            {
#if 0
                DBG_PRINT_STRING_VALUE(DBG_LEVEL_INFO,
                    "LWRM: graphics controller at domain   ", Domain);
                DBG_PRINT_STRING_VALUE(DBG_LEVEL_INFO,
                    "                             bus      ", Bus);
                DBG_PRINT_STRING_VALUE(DBG_LEVEL_INFO,
                    "                             device   ", Device);
                DBG_PRINT_STRING_VALUE(DBG_LEVEL_INFO,
                    "                             function ", Function);
#endif
                if (ModsDrvPciRd16(Domain, Bus, Device, Function, 0)
                        == LW_CONFIG_PCI_LW_0_VENDOR_ID_LWIDIA)
                {
                    // This is an LWpu device.

                    // Allocate the ModsGpuDevice and ModsGpuSubdevice structures.
                    ModsGpuDevice * pDevice;
                    ModsGpuSubdevice * pSubdevice;
                    UINT32 PciLw1;
                    BOOL   barIs64Bit = FALSE;
                    UINT32 BarOffset;
                    UINT32 i;
#define MODS_NUM_BARS 3
                    PHYSADDR   baseAddress[MODS_NUM_BARS];
                    UINT64     barSize[MODS_NUM_BARS];
                    UINT32     baseAddressLow  = 0;

                    pDevice = calloc(1, sizeof(ModsGpuDevice));
                    lw_memset(pDevice, 0, sizeof(ModsGpuDevice));
                    pSubdevice = calloc(1, sizeof(ModsGpuSubdevice));
                    lw_memset(pSubdevice, 0, sizeof(ModsGpuSubdevice));

                    pDevice->Subdevices[MC_PARENT_DEVICE] = pSubdevice;
                    pDevice->NumSubdevices = 1;

                    // Find the BAR locations and sizes.
                    BarOffset = LW_CONFIG_PCI_LW_4;

                    for (i = 0; i < MODS_NUM_BARS; i++)
                    {
                        baseAddressLow = ModsDrvPciRd32(Domain, Bus, Device, Function, BarOffset);
                        BarOffset += 4;

                        if (DRF_VAL(_CONFIG, _PCI_LW_5, _ADDRESS_TYPE, baseAddressLow) == LW_CONFIG_PCI_LW_5_ADDRESS_TYPE_64_BIT)
                        {
                            BarOffset += 4;
                            barIs64Bit = TRUE;
                        }
                        ModsDrvPciGetBarInfo(Domain, Bus, Device, Function, i, &baseAddress[i], &barSize[i]);
                    }

                    pSubdevice->PhysLwBase   = baseAddress[0];
                    pSubdevice->PhysFbBase   = baseAddress[1];
                    pSubdevice->PhysInstBase = baseAddress[2];
                    pSubdevice->LwApertureSize   = (UINT32) barSize[0];
                    pSubdevice->FbApertureSize   = (UINT32) barSize[1];
                    pSubdevice->InstApertureSize = (UINT32) barSize[2];

#if 0
                    DBG_PRINTF((DBG_MODULE_OS, DBG_LEVEL_INFO,
                        "LWRM: BAR0 base = 0x%X_%08X\n",
                        LwU64_HI32(pSubdevice->PhysLwBase),
                        LwU64_LO32(pSubdevice->PhysLwBase)));
                    DBG_PRINTF((DBG_MODULE_OS, DBG_LEVEL_INFO,
                        "           size = 0x%08X\n", pSubdevice->LwApertureSize));
                    DBG_PRINTF((DBG_MODULE_OS, DBG_LEVEL_INFO,
                        "LWRM: BAR1 base = 0x%X_%08X\n",
                        LwU64_HI32(pSubdevice->PhysFbBase),
                        LwU64_LO32(pSubdevice->PhysFbBase)));
                    DBG_PRINTF((DBG_MODULE_OS, DBG_LEVEL_INFO,
                        "           size = 0x%08X\n", pSubdevice->FbApertureSize));
                    DBG_PRINTF((DBG_MODULE_OS, DBG_LEVEL_INFO,
                        "LWRM: BAR2 base = 0x%X_%08X\n",
                        LwU64_HI32(pSubdevice->PhysInstBase),
                        LwU64_LO32(pSubdevice->PhysInstBase)));
                    DBG_PRINTF((DBG_MODULE_OS, DBG_LEVEL_INFO,
                        "           size = 0x%08X\n", pSubdevice->InstApertureSize));
#endif

                    // Make sure that the SBIOS assigned us valid base addresses
                    if ((0 == pSubdevice->PhysLwBase) || (0 == pSubdevice->PhysFbBase))
                    {
#if 0
                        DBG_PRINT_STRING(DBG_LEVEL_ERRORS,
                           "LWRM: *** LW BAR or FB BAR is zero.\n");
                        osFreeMem(pDevice);
                        osFreeMem(pSubdevice);
#endif
                        free(pDevice);
                        free(pSubdevice);
                        return FALSE;
                    }

                    // Enable the memory space and bus mastering, map the memory.
                    PciLw1 = ModsDrvPciRd32(Domain, Bus, Device, Function, LW_CONFIG_PCI_LW_1);

                    PciLw1 |=   DRF_DEF(_CONFIG, _PCI_LW_1, _MEMORY_SPACE, _ENABLED)
                              | DRF_DEF(_CONFIG, _PCI_LW_1, _BUS_MASTER, _ENABLED);

                    ModsDrvPciWr32(Domain, Bus, Device, Function, LW_CONFIG_PCI_LW_1, PciLw1);

                    if (barIs64Bit != TRUE)
                    {
                        // There is a hardware bug in BR02 when used with PCI-express
                        // GPUs that support 64-bit BARs in 32-bit native-AGP systems.
                        // To work around it, bang this unused bar register to zero.
                        // See bugs 118419 and 119644.
                        ModsDrvPciWr32(Domain, Bus, Device, Function, LW_CONFIG_PCI_LW_7(1), 0);
                    }

                    // Map the GPU registers
                    pSubdevice->LinLwBase =
                        ModsDrvMapDeviceMemory(pSubdevice->PhysLwBase,
                                               pSubdevice->LwApertureSize,
                                               ATTRIB_UC, PROTECT_READ_WRITE);
                    pSubdevice->InitDone |= MODSRM_INIT_BAR0;

                    if (pLastDev)
                    {
                        if (pSubdevice->SbiosBoot)
                        {
                            // Add the primary board to the beginning of the list.
                           pDevice->pNext = *ppDevices;
                           *ppDevices = pDevice;
                        }
                        else
                        {
                            // Add non-primary boards to the end of the list.
                            pLastDev->pNext = pDevice;
                            pLastDev = pDevice;
                        }
                    }
                    else
                    {
                        // First detected board, start the list.
                        *ppDevices = pDevice;
                        pLastDev = pDevice;
                    }
                    bSuccess = TRUE;
                }
            }
            else if (PCI_DEVICE_NOT_FOUND == PciRc)
            {
                // didn't find a device but need to make sure to check for all supported class codes
                break;
            }
            else
            {
                // Some other error.
//                DBG_PRINT_STRING(DBG_LEVEL_ERRORS,
//                    "LWRM: *** PCI error while searching for a device\n");
                return FALSE;
            }

        }
    }

    if (bSuccess == FALSE)
    {
        UINT32 miscRegBase, miscApertureSize;
        // Check if we are running on CheetAh
        if (ModsDrvGetSOCDeviceAperture(LW_DEVID_MISC, 0, (void **)&miscRegBase, &miscApertureSize))
        {
            assert(miscApertureSize >= 0x808);
            isTegraHack = ModsDrvMemRd32((UINT32 *)(miscRegBase + 0x804));
            lwBar0 = 0;
            dprintf("lw: CheetAh Device Detected: address: 0x70000804 buffer: 0x%08lx\n", isTegraHack);
            dprintf("lw: NOT using lwBar0; lwBar0 set to zero !\n");
            dprintf("lw: Only certain functions will work - others may CRASH!\n");
            bSuccess =  TRUE;
        }
    }

    return bSuccess;
} // LwWatchFindDevices

//-------------------------------------------------------------------
// initLwWatch
// Finds BAR0 for the GPU we are working with. This BAR0 is mapped
// to CPU virtual memory by MODS. lwBar0 points to the physical address
// and regBase to the virtual one.
//-------------------------------------------------------------------
void initLwWatch()
{
    ModsGpuDevice     *pDev    = NULL;
    ModsGpuSubdevice  *pSubDev = NULL;

    // If regBase address is determined this means initLwWatch
    // has already been called once.
    if (regBase != NULL || IsTegra())
    {
        lw_dprintf("initLwWatch() was already called once.\n");
        return;
    }

    // Get the info for the gpu.
    if (LwWatchFindDevices(&mainDev) == FALSE)
    {
        lw_dprintf("initLwWatch() could not find any supported devices\n");
        return;
    }

    if (IsTegra()) goto finish_init;

    if (mainDev == NULL)
    {
        // Couldn't find any devices, exit nicely instead of seg faulting
        lw_dprintf("initLwWatch() could not find any supported devices\n");
        return;
    }

    // The user has specified a BAR0 entry, try to use that one.
    if (lwBar0 != 0)
    {
        pDev = mainDev;
        while (pDev != NULL)
        {
            pSubDev = pDev->Subdevices[MC_PARENT_DEVICE];
            // Does it match the request?  If so, we're done!
            if (lwBar0 == pSubDev->PhysLwBase)
            {
                mainDev = pDev;
                subDev = pSubDev;
                break;
            }

            pDev = pDev->pNext;
        }
    }

    // If we couldn't find the desired subDev (or if the user didn't specify
    // one), choose the first GPU.
    if (subDev == NULL)
    {
        // We get the subdevice for the main device
        // it lives at index 1.
        subDev = mainDev->Subdevices[MC_PARENT_DEVICE];
    }

    // This field gives the BAR0 value as it was mapped
    // by MODS to CPU virtual memory.
    regBase = (GPUHWREG*)subDev->LinLwBase;
    lwBar0 = subDev->PhysLwBase;//virtToPhys(&regBase->Reg032[0]);
    lwBar1 = subDev->PhysFbBase;

finish_init:

#if LWWATCHCFG_IS_PLATFORM(UNIX)
    //
    // On linux sim we can use a global location for manuals and class files
    //
    if (getelw("LWW_MANUAL_SDK") == NULL)
    {
        setelw("LWW_MANUAL_SDK",
               "/home/scratch.jsmith_lw5x/lw50/sw/dev/gpu_drv/chips_a/drivers/resman/kernel/inc",
               0);
    }
    if (getelw("LWW_CLASS_SDK") == NULL)
    {
        setelw("LWW_CLASS_SDK",
               "/home/scratch.jsmith_lw5x/lw50/sw/dev/gpu_drv/chips_a/sdk/lwpu/inc/class",
               0);
    }
#endif

    osInit();
}

//-------------------------------------------------------------------------------------
// exitLwWatch -    On exit unmap the memory mapped during initialization.
//-------------------------------------------------------------------------------------
void exitLwWatch()
{
    if(regBase != NULL)
    {
        ModsDrvUnMapDeviceMemory(regBase);
    }

    osDestroyHal();

    // mainDev points to other devices which are never freed.  This is a memory
    // leak!
    if(mainDev != NULL)
    {
        free(mainDev);
    }

    // Other devices have subdevcies which are never freed.  This is a memory
    // leak!
    if(subDev != NULL)
    {
        free(subDev);
    }
}

//-------------------------------------------------------------------------------------
// GetExpression -  Get the next argument, if there is one.
//                  If the argument is symbolic, then find the value for the symbol
//                  Else, return 0.
//-------------------------------------------------------------------------------------

LwU32 GetExpression(const char *args)
{
    LwU64 val = 0;
    char *endp;

    endp = NULL;
    GetExpressionEx(args, &val, &endp);

    return (LwU32)val;
}

//-------------------------------------------------------------------------------------
// lw_strtoull -    Get a string with numbers and colwert the first to
//                  unsigned long long. Return the rest of the string.
//-------------------------------------------------------------------------------------
LwU64 lw_strtoull(const char* args, char** endp)
{
    int i = 0;
    int deg = 10;
    LwU64 res = 0L;

    if(*(args)=='0')
    {
        i++;
        if(*(args+1)=='x')
        {
            deg = 16;
            i++;
        }
        else
        {
            deg = 8;
        }
    }

    while(*(args+i)!=' ' && *(args+i)!=0)
    {
        res *= deg;
        if(deg==16)
        {
            if(*(args+i)>='a' && *(args+i)<='f')
                res += *(args+i)-'a'+10;
            else if(*(args+i)>='A' && *(args+i)<='F')
                res += *(args+i)-'A'+10;
            else
                res += *(args+i) - '0';
        }
        else
        {
            res += *(args+i) - '0';
        }
        i++;
    }

    *endp = (char*)(size_t)(args+i);

    return res;
}

//-------------------------------------------------------------------------------
// GetExpressionEx -  Check to see if there are two more arguments
//                    that can be taken.If so, get the first one and return TRUE.
//                    Else, return FALSE.
//-------------------------------------------------------------------------------

BOOL GetExpressionEx(const char *args, LwU64 *val, char **endp)
{
    if (!args || !args[0] || !endp)
        return 0;

    //*val = (LwU64)simple_strtoull(args, endp, 0);
    //*val = (LwU64)strtoull(args, endp, 0);
    *val = (LwU64)lw_strtoull(args, endp);

    if (*endp && **endp != '\0')
    {
        (*endp)++;
        return TRUE;
    }
    else
    {
        *endp = (char *)(size_t)args;
        return FALSE;
    }
}

//-----------------------------------------------------
// physToVirt
// - make an effort to find a valid virtual mapping for this address
//-----------------------------------------------------

LwU64 physToVirt(LwU64 physAddr, LwU32 flags)
{
    dprintf("physToVirt not implemented\n");
    return 0;
}

//-----------------------------------------------------
// virtToPhys
// copied from lw_get_phys_address in lw.c
//
//-----------------------------------------------------
LwU64 virtToPhys(LwU64 addr, LwU32 pid)
{
    dprintf("virtToPhys not implemented\n");
    return 0;
}

//-----------------------------------------------------
// readPhysicalMem
//-----------------------------------------------------
LW_STATUS readPhysicalMem(LwU64 address, void *buf, LwU64 size, LwU64 *pSizer)
{
    void *virtAddress = ModsDrvMapDeviceMemory(address, size, ATTRIB_UC, PROTECT_READ_WRITE);

#if 0
    if (((address <  subDev->PhysLwBase) ||
         (address >= subDev->PhysLwBase + subDev->LwApertureSize)) &&
        ((address <  subDev->PhysFbBase) ||
         (address >= subDev->PhysFbBase + subDev->FbApertureSize)) &&
        ((address <  subDev->PhysInstBase) ||
         (address >= subDev->PhysInstBase + subDev->InstApertureSize)) &&
        ((address < 0xA0000) ||
         (address > 0xBFFFF)))
    {
        return LW_ERR_GENERIC;
    }
#endif

    LwU64 virtAddress64 = (size_t)virtAddress;
    if (sizeof(virtAddress) == 4)
        virtAddress64 &= 0xffffffff;

    if (readVirtMem(virtAddress64, buf, size, pSizer) == LW_OK)
    {
        ModsDrvUnMapDeviceMemory((void*)virtAddress);
        return LW_OK;
    }

    ModsDrvUnMapDeviceMemory((void*)virtAddress);

    return LW_ERR_GENERIC;
}

//
// I think I'd prefer to call kdb_getstr to handle this, but that's not exported
// this is logically identical code, but uses exported symbols
//
LwU32 osGetInputLine(LwU8 *prompt, LwU8 *buffer, LwU32 size)
{
    int charsRead;

    dprintf("%s", prompt);

    charsRead = scanf("%s",buffer);

    return charsRead;
}

//-----------------------------------------------------
// osPerfDelay
//
//-----------------------------------------------------
void osPerfDelay(LwU32 MicroSeconds)
{
    ModsDrvSleep( MicroSeconds / 1000 );
    return;
}

//-----------------------------------------------------
// RD_PHYS32
// - read physical address
//-----------------------------------------------------
LwU32 RD_PHYS32(PhysAddr physAddr)
{
    LwU32 data;
    LwU64 size;
    if (readPhysicalMem(physAddr, &data, 4, &size) != LW_OK)
        data = (LwU32) -1;
    return data;
}

//-----------------------------------------------------
// WR_PHYS32
// - write physical address
//-----------------------------------------------------
void WR_PHYS32(PhysAddr physAddr, LwU32 data)
{
    LwU64 size;
    writePhysicalMem(physAddr, &data, 4, &size);
}

LW_STATUS writePhysicalMem(LwU64 address, void *buf, LwU64 size, LwU64 *pSizew)
{
    void *virtAddress = ModsDrvMapDeviceMemory(address, size, ATTRIB_UC, PROTECT_READ_WRITE);

#if 0
    if (((address <  subDev->PhysLwBase) ||
         (address >= subDev->PhysLwBase + subDev->LwApertureSize)) &&
        ((address <  subDev->PhysFbBase) ||
         (address >= subDev->PhysFbBase + subDev->FbApertureSize)) &&
        ((address <  subDev->PhysInstBase) ||
         (address >= subDev->PhysInstBase + subDev->InstApertureSize)) &&
        ((address < 0xA0000) ||
         (address > 0xBFFFF)))
    {
        return LW_ERR_GENERIC;
    }
#endif

    LwU64 virtAddress64 = (size_t)virtAddress;
    if (sizeof(virtAddress) == 4)
        virtAddress64 &= 0xffffffff;

    if (writeVirtMem(virtAddress64, buf, size, pSizew) == LW_OK)
    {
        ModsDrvUnMapDeviceMemory(virtAddress);
        return LW_OK;
    }

    ModsDrvUnMapDeviceMemory(virtAddress);

    return LW_ERR_GENERIC;
}

LW_STATUS readVirtMem(LwU64 address, void *buf, LwU64 size, LwU64 *pSizer)
{
    LwU64 i;
    LwU64 leftBytes;
    LwU64 bytesRead = 0;

    LwU32 tempVal;

    if ((buf == NULL) || (pSizer == NULL))
        return LW_ERR_GENERIC;

    // get the number of bytes that are not 4 bytes aligned
    // in the beginning
    leftBytes = address & 0x3;

    // if address is not 4 bytes aligned read the left bytes
    // in the begining
    if(leftBytes)
    {
        // read a whole 32 bit chunk
        tempVal = ModsDrvMemRd32((void*)(size_t)(address - leftBytes));
        // and leave the values that are interesting for us
        tempVal = tempVal >> (leftBytes*8);

        // put the bytes needed in the buffer
        while(leftBytes<4)
        {
            *((LwU8*)buf+bytesRead) = (LwU8)(tempVal & 0xFF);
            tempVal >>= 8;
            leftBytes++;
            bytesRead++;
        }
    }

    // go on and read 4 bytes pieces of data while they fit
    // in the size requested
    for (i = bytesRead; i+4 <= size; i+=4)
    {
        *((LwU32*)((LwU8*)buf+i)) = (LwU32)ModsDrvMemRd32((LwU32*)(size_t)(address+i));
    }

    // if the end is not 4 bytes aligned, read the left bytes
    if(i<size)
    {
        // read a 32 bit chunk
        tempVal = ModsDrvMemRd32((LwU32*)(size_t)(address+i));
        // and put the bytes needed in the buffer, skip the others
        while(i<size)
        {
            *((LwU8*)buf+i) = (LwU8)(tempVal & 0xFF);
            i++;
            tempVal >>= 8;
        }
    }
    *pSizer = size;

    return LW_OK;
}

LW_STATUS writeVirtMem(LwU64 address, void *buf, LwU64 size, LwU64 *pSizew)
{
    LwU64 i;

    if ((buf == NULL) || (pSizew == NULL))
        return LW_ERR_GENERIC;

    // if we are doing a single 32b write on a properly aligned addr boundary, execute as 32b mods write
    if ((size == 4) && (address == (address & ~0x3)))
    {
        ModsDrvMemWr32((void*)(size_t)address, *((LwU32*)buf));
    }
    else
    {
        for (i = 0; i < size; i++)
        {
            ModsDrvMemWr08(((LwU8*)(size_t)address+i), *((LwU8*)buf+i));
        }
    }
    *pSizew = size;

    return LW_OK;
}

LwU32 osRegRd32(PhysAddr off)
{
    if (regBase)
    {
        return ModsDrvMemRd32(&regBase->Reg032[off/4]);
    }
    else
    {
        return RD_PHYS32((PhysAddr)off);
    }
}

void osRegWr32(PhysAddr off, LwU32 data)
{
    if (regBase)
    {
        ModsDrvMemWr32(&regBase->Reg032[off/4], data);
    }
    else
    {
        WR_PHYS32((PhysAddr)off, data);
    }
}

LwU8 osRegRd08(PhysAddr off)
{
    assert(regBase);
    return ModsDrvMemRd08(&regBase->Reg008[off]);
}

void osRegWr08(PhysAddr off, LwU8 data)
{
    assert(regBase);
    ModsDrvMemWr08(&regBase->Reg008[off], data);
}

LW_STATUS FB_RD32(LwU32 off)
{
    dprintf("FB_RD32 not implemented\n");

    return LW_ERR_GENERIC;
}

// XXX? not properly implemented yet
LW_STATUS FB_RD32_64(LwU64 reg)
{
    return FB_RD32((LwU32)reg);
}

void FB_WR32(LwU32 off, LwU32 data)
{
    dprintf("FB_WR32 not implemented\n");
}

void lw_memcpy(void *dest, void *src, unsigned int bytes)
{
        ModsDrvMemCopy(dest, src, bytes);
}

void lw_memset(void *dest, int c, unsigned int bytes)
{
    unsigned int i;

    for(i = 0; i < bytes; i++)
    {
        ModsDrvMemWr08((LwU8*)dest+i, c);
    }
}

int lw_memcmp(const void *buf1, const void *buf2, unsigned int bytes)
{
    unsigned int i;
    LwU8 val1, val2;

    for(i = 0; i < bytes; i++)
    {
        // read a byte in a time and compare
        val1 = ModsDrvMemRd08((const LwU8*)buf1+i);
        val2 = ModsDrvMemRd08((const LwU8*)buf2+i);
        if (val1 > val2)
            return 1;

        if (val1 < val2)
            return -1;
    }

    return 0;
}

//
// Use kdb's printout functions
// these will keep the screen from scrolling too far as well as output
// to the correct cpu. (not sure how much I like the scrolling feature
// it does well for the help screen, but the vga register dumps end up
// with a prompt in the middle of a line).. hmm, kdb allows setting
// environment variables that can "fix" this. kdb defaults to pausing
// every 22 lines, but the command "set LINES <NN>" changes that.
//

int lw_dprintf(const char *format, ...)
{
    int chars_written = 0;
    va_list arglist;

    va_start(arglist, format);

    chars_written = ModsDrvVPrintf(4, format, arglist);

    va_end(arglist);

    ModsDrvFlushLogFile();

    return chars_written;
}

int lw_sprintf(char *str, const char *format, ...)
{
    int chars_written = 0;
    va_list arglist;

    va_start(arglist, format);
    chars_written = vsprintf(str, format, arglist);

    va_end(arglist);

    return chars_written;
}

int lw_sscanf(const char *str, const char *format, ...)
{
    int items_assigned = 0;
    va_list arglist;

    va_start(arglist, format);
#if XP_UNIX
    items_assigned = vsscanf(str, format, arglist);
#else
#undef sscanf
    items_assigned = sscanf(str, format, arglist);
#define sscanf lw_sscanf
#endif

    va_end(arglist);

    return items_assigned;
}

static void lw_swap(void *x, void *y, int size)
{
    while (size > 0)
    {
        char t = *(char *)x;
        *(char *)x = *(char *)y;
        *(char *)y = t;
        x = (char *)x + 1;
        y = (char *)y + 1;
        size--;
    }
}

static void hsort(void *p, int n, int size,
                  int (*cmp)(const void *, const void *),
                  void (*swap)(void *, void *, int))
{
    int i, j;
    char *v1, *v2, *root = p;

    if (n < 2) return;

    /*
     * In order for heap sort to work, the incoming data
     * need to be organized as a heap, first.
     */
    for (i = (n/2); i >= 0; i--)
    {
        for (j = 0; j < (n/2); j++)
        {
            v1 = root + (j * size);
            v2 = root + (j * size * 2) + size;
            if (cmp(v1, v2) < 0)
                swap(v1, v2, size);
            v2 += size;
            if (cmp(v1, v2) < 0 && (2*j+1) < (n-1))
                swap(v1, v2, size);
        }
    }

    /*
     * Now that we have a valid heap, we can perform the
     * actual sort; its complexity is O(n*log(n)). This
     * particular heap sort implementation wasn't geared
     * towards performance...
     */
    for (i = (n-1); i >= 0; i--)
    {
        swap(root, (root + (i * size)), size);
        for (j = 0; j < (i/2); j++)
        {
            v1 = root + (j * size);
            v2 = root + (j * size * 2) + size;
            if (cmp(v1, v2) < 0)
                swap(v1, v2, size);
            v2 += size;
            if (cmp(v1, v2) < 0 && (2*j+1) < (i-1))
                swap(v1, v2, size);
        }
    }
}

void lw_hsort(void *p, int n, int size,
                          int (*cmp)(const void *, const void *))
{
    hsort(p, n, size, cmp, lw_swap);
}

//-----------------------------------------------------
// SYSMEM_RD08
//
//-----------------------------------------------------
LwU8 SYSMEM_RD08(LwU64 pa)
{
    LwU8 buf;
    LwU64 sizer;

    readPhysicalMem(pa, &buf, 1, &sizer);

    return buf;
}

//-----------------------------------------------------
// SYSMEM_RD16
//
//-----------------------------------------------------
LwU16 SYSMEM_RD16(LwU64 pa)
{
    LwU16 buf;
    LwU64 sizer;

    readPhysicalMem(pa, &buf, 2, &sizer);

    return buf;
}

//-----------------------------------------------------
// SYSMEM_RD32
//
//-----------------------------------------------------
LwU32 SYSMEM_RD32(LwU64 pa)
{
    LwU32 buf;
    LwU64 sizer;

   if (readPhysicalMem(pa, &buf, 4, &sizer) != LW_OK)
       buf = (LwU32) -1;
   return buf;
}

//-----------------------------------------------------
//  SYSMEM_WR32
// - write to 64 bit physical address
//-----------------------------------------------------
void SYSMEM_WR32(LwU64 pa, LwU32 data)
{
    LwU64 size;
    writePhysicalMem(pa, &data, 4, &size);
}


//-----------------------------------------------------
// osPciRead32
//
//-----------------------------------------------------
LW_STATUS osPciRead32(LwU16 DomainNumber, LwU8 BusNumber, LwU8 Device, LwU8 Function, LwU32* Buffer, LwU32 Offset)
{
    if (NULL == Buffer)
    {
        return LW_ERR_GENERIC;
    }

    *Buffer = ModsDrvPciRd32(DomainNumber, BusNumber, Device, Function, Offset);
    return LW_OK;
}

//-----------------------------------------------------
// osPciRead32
//
//-----------------------------------------------------
LW_STATUS osPciRead16(LwU16 DomainNumber, LwU8 BusNumber, LwU8 Device, LwU8 Function, LwU16* Buffer, LwU32 Offset)
{
    if (NULL == Buffer)
    {
        return LW_ERR_GENERIC;
    }

    *Buffer = ModsDrvPciRd16(DomainNumber, BusNumber, Device, Function, Offset);
    return LW_OK;
}

//-----------------------------------------------------
// osPciRead32
//
//-----------------------------------------------------
LW_STATUS osPciRead08(LwU16 DomainNumber, LwU8 BusNumber, LwU8 Device, LwU8 Function, LwU8* Buffer, LwU32 Offset)
{
    if (NULL == Buffer)
    {
        return LW_ERR_GENERIC;
    }

    *Buffer = ModsDrvPciRd08(DomainNumber, BusNumber, Device, Function, Offset);
    return LW_OK;
}

LW_STATUS osPciWrite32(LwU16 DomainNumber, LwU8 BusNumber, LwU8 Device, LwU8 Function, LwU32 Data, LwU32 Offset)
{
    ModsDrvPciWr32(DomainNumber, BusNumber, Device, Function, Offset, Data);
    return LW_OK;
}

//-----------------------------------------------------
// osPciFindDevices
//
//-----------------------------------------------------
LW_STATUS osPciFindDevices(LwU16 DeviceId, LwU16 VendorId, osPciCallback callback)
{
    LwU32 i = 0;
    LwU32 domain, bus, device, func;

    while (PCI_OK == ModsDrvFindPciDevice(DeviceId, VendorId, i++, &domain, 
                                                &bus, &device, &func))
    {
        callback(domain, bus, device, func);
    }
    return LW_OK;
}

//-----------------------------------------------------
// osPciFindDevicesByClass
//
//-----------------------------------------------------
LW_STATUS osPciFindDevicesByClass(LwU32 classCode, osPciCallback callback)
{
    LwU32 i = 0;
    LwU32 domain, bus, device, func;

    while (PCI_OK == ModsDrvFindPciClassCode(classCode, i++, &domain, 
                                                &bus, &device, &func))
    {
        callback(domain, bus, device, func);
    }
    return LW_OK;
}

LW_STATUS osPciGetBarInfo
(
    LwU16 DomainNumber,
    LwU8 BusNumber,
    LwU8 Device,
    LwU8 Function,
    LwU8 BarIndex,
    LwU64 *BaseAddr,
    LwU64 *BarSize)
{
    if (PCI_OK != ModsDrvPciGetBarInfo(DomainNumber, BusNumber, Device,
                                        Function, BarIndex, BaseAddr,
                                        BarSize))
    {
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}

LW_STATUS osMapDeviceMemory
(
    LwU64 BaseAddr,
    LwU64 Size,
    MemProtFlags prot,
    void **ppBar
)
{
    LwU32 memProt = 0;
    if (NULL == ppBar)
    {
        return LW_ERR_GENERIC;
    }

    if (prot & MEM_PROT_EXEC)
    {
        memProt |= PROTECT_EXELWTABLE;
    }
    if (prot & MEM_PROT_READ)
    {
        memProt |= PROTECT_READABLE;
    }
    if (prot & MEM_PROT_WRITE)
    {
        memProt |= PROTECT_WRITEABLE;
    }

    *ppBar = ModsDrvMapDeviceMemory(BaseAddr, (size_t)Size, ATTRIB_UC, memProt);
    return LW_OK;
}

LW_STATUS osUnMapDeviceMemory(void *pBar, LwU64 BarSize)
{
    ModsDrvUnMapDeviceMemory(pBar);
    return LW_OK;
}



//-----------------------------------------------------
// osCheckControlC
//
//-----------------------------------------------------
LwU32 osCheckControlC()
{
    return 0;
}

#ifdef XP_PC

#define lowercase(c) (tolower(c))

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
#endif //XP_PC

void PMU_LOG(int in_lvl, const char *fmt, ...)
{
    int lvl = in_lvl;

    va_list Arguments;
    va_start(Arguments, fmt);

    if (lvl) dprintf(" ");
    while (lvl--) dprintf(">");
    dprintf(fmt, Arguments);

    va_end(Arguments);
}

void DPU_LOG(int in_lvl, const char *fmt, ...)
{
    int lvl = in_lvl;

    va_list Arguments;
    va_start(Arguments, fmt);

    if (lvl) dprintf(" ");
    while (lvl--) dprintf(">");
    dprintf(fmt, Arguments);

    va_end(Arguments);
}
