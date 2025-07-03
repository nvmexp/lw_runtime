/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#define _GNU_SOURCE
#if LWCPU_IS_X86
// Tell glibc to use mmap64 instead of mmap, to be able to map BARs > 4GB.
#define _FILE_OFFSET_BITS 64
#endif
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <pci/pci.h>
#if LWOS_IS_FREEBSD
#include <sys/sysctl.h>
#endif
#include "turing/tu102/dev_master.h"
#include "vgpu.h"
#include "chip.h"
#include "disp.h"
#include "t23x/t234/address_map_new.h"

#define PAGESIZE (getpagesize())
#define PAGEMASK (getpagesize() - 1)

#ifndef PCI_CAP_ID_EXP
#define PCI_CAP_ID_EXP 0x10
#endif

#ifndef PCI_CLASS_DISPLAY_3D
#define PCI_CLASS_DISPLAY_3D 0x302
#endif

#include "os.h"
#include "exts.h"

#define BYTE_ALIGNMENT 4

struct RAM_region {
    uint64_t start, end;
};

#define MAX_REGIONS 16
static struct RAM_region RAM_regions[MAX_REGIONS];

struct MEM_region {
    uint64_t base_address;
    uint32_t size;
    union {
        volatile void *pv;
        volatile uint8_t *p8;
        volatile uint32_t *p32;
    } mapping;
};

#define MAX_BARS 6
static struct MEM_region GPU_BARs[MAX_BARS];

static struct MEM_region current;


#define IS_IN_MEM_REGION(BAR, address) \
    ((address >= BAR.base_address) && (address < (BAR.base_address + BAR.size)))
#define IS_IN_RAM_REGION(RAM, address) \
    ((address >= RAM.start) && (address < RAM.end))

static struct pci_access *pci_bus = NULL;
static int fd = -1;
static int fd_cached = -1;

static int addr_is_RAM(uint64_t address)
{
    int i;
    for (i = 0; i < MAX_REGIONS; i++) {
        if (IS_IN_RAM_REGION(RAM_regions[i], address)) {
            return 1;
        }
    }
    return 0;
}

static volatile void *mmio_get_mapping(uint64_t address)
{
    uint64_t offset;
    int prot = PROT_READ | PROT_WRITE, flags = MAP_SHARED;
    int i, _fd;
    
    // On SoC, Disp engine is outside iGPU (bar0) while the manuals (reg offset) is shared with dGPU.
    if (pDisp[indexGpu].dispTranslateGpuRegAddrToSoc) {
        address = pDisp[indexGpu].dispTranslateGpuRegAddrToSoc(address);
    }

    for (i = 0; i < MAX_BARS; i++) {
        if (IS_IN_MEM_REGION(GPU_BARs[i], address)) {
            offset = address - GPU_BARs[i].base_address;
            return &GPU_BARs[i].mapping.p8[offset];
        }
    }

    if (!IS_IN_MEM_REGION(current, address)) {
        _fd = fd;
        if (addr_is_RAM(address))
            _fd = fd_cached;

        if (current.mapping.pv)
            munmap((void *)current.mapping.pv, current.size);

        current.base_address = (address & ~PAGEMASK);
        current.size = PAGESIZE;
        current.mapping.pv = mmap(NULL, current.size, prot, flags,
                _fd, current.base_address);
        if (current.mapping.pv == MAP_FAILED) {
            fprintf(stderr, "Unable to map 0x%x @ 0x%" PRIx64 "\n",
                    current.size, current.base_address);
            memset(&current, 0, sizeof(current));
            return NULL;
        }
    }

    offset = address - current.base_address;
    return &current.mapping.p8[offset];
}

static void MMIO_WR32(uint64_t address, uint32_t value)
{
    volatile uint32_t *mapping;

    if ((address % 4) != 0) {
        fprintf(stderr, "MMIO_WR32(): 0x%" PRIx64 " is unaligned!\n", address);
        return;
    }

    mapping = mmio_get_mapping(address);
    if (mapping) {
        *mapping = value;
    }
}

static uint32_t MMIO_RD32(uint64_t address)
{
    volatile uint32_t *mapping;

    if ((address % 4) != 0) {
        fprintf(stderr, "MMIO_RD32(): 0x%" PRIx64 " is unaligned!\n", address);
        return 0;
    }

    mapping = mmio_get_mapping(address);
    if (mapping) {
        return *mapping;
    }

    return 0;
}

static void MMIO_WR08(uint64_t address, uint8_t value)
{
    volatile uint8_t *mapping;

    mapping = mmio_get_mapping(address);
    if (mapping) {
        *mapping = value;
    }
}

static uint8_t MMIO_RD08(uint64_t address)
{
    volatile uint8_t *mapping;

    mapping = mmio_get_mapping(address);
    if (mapping) {
        return *mapping;
    }

    return 0;
}

#ifdef GPU_REG_RD32
#undef GPU_REG_RD32
#undef GPU_REG_WR32
#endif

#ifdef GPU_REG_RD08
#undef GPU_REG_RD08
#undef GPU_REG_WR08
#endif

LwU32 GPU_REG_RD32_DIRECT(PhysAddr reg)
{
    return MMIO_RD32(lwBar0 + reg);
}

#define GPU_REG_RD32(offset)  (((isVirtualWithSriov())) ? pfRegRead(offset) : GPU_REG_RD32_DIRECT(offset))

void GPU_REG_WR32_DIRECT(PhysAddr offset, LwU32 value)
{
    MMIO_WR32((lwBar0 + (offset)), (value));
}

#define GPU_REG_WR32(offset,value)\
     (((isVirtualWithSriov())) ? pfRegWrite(offset, value) : GPU_REG_WR32_DIRECT((offset), (value)))

#define GPU_REG_RD08(offset) MMIO_RD08(lwBar0 + (offset))
#define GPU_REG_WR08(offset,value) MMIO_WR08((lwBar0 + (offset)), (value))

#define GPU_FB_RD32(offset) MMIO_RD32(lwBar1 + (offset))
#define GPU_FB_WR32(offset,value) MMIO_WR32((lwBar1 + (offset)), (value))


static struct pci_dev *find_pci_device(u16 domain, u8 bus, u8 device, u8 function)
{
    struct pci_dev *dev;

    dev = pci_bus->devices;
    while (dev) {
        if ((dev->domain == domain) &&
                (dev->bus == bus) &&
                (dev->dev == device) &&
                (dev->func == function))
            break;
        dev = dev->next;
    }

    return dev;
}

static int find_RAM_regions(void)
{
#if LWOS_IS_LINUX
    char *path = "/proc/iomem";
    FILE *fp;
    char *line = NULL;
    char *ptr;
    size_t len = 0;
    ssize_t read;
    unsigned long long start, end;
    int n, consumed;
    int i = 0;

    fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "fopen(%s) failed (%s)!\n", path, strerror(errno));
        return 1;
    }

    while ((read = getline(&line, &len, fp)) != -1) {
        n = sscanf(line, "%Lx-%Lx : %n\n", &start, &end, &consumed);
        if (n != 2)
            continue;
        ptr = line + consumed;
        end++;
        if (strncmp(ptr, "System RAM", 10) == 0) {
            RAM_regions[i].start = start;
            RAM_regions[i].end = end;
            if (++i == MAX_REGIONS) {
                fprintf(stderr, "Reached RAM region table limit!\n");
                break;
            }
        }
    }
    if (line)
        free(line);

    fclose(fp);

    return 0;

#elif LWOS_IS_FREEBSD
    int ret;
    char *phys_seg_str = NULL;
    char *line, *tmp;
    uint32_t ram_region_index = 0;
    {
        size_t str_len = 0;

        ret = sysctlbyname("vm.phys_segs", NULL, &str_len, NULL, 0);
        if (ret < 0) {
            return 1;
        }

        phys_seg_str = malloc(str_len);
        if (phys_seg_str == NULL) {
            return 1;
        }

        ret = sysctlbyname("vm.phys_segs", phys_seg_str, &str_len, NULL, 0);
        if (ret < 0) {
            free(phys_seg_str);
            return 1;
        }
    }

    /*
     *  Parse a list of entries of the form:
     *
     *  SEGMENT 0:
     *
     *  start:     0x10000
     *  end:       0x9b000
     *  domain:    0
     *  free list: 0xffffffff81e30a10
     *
     *  SEGMENT 1:
     *  [...]
     */
    enum {
        SEGMENT,
        START,
        END,
        DOMAIN,
        FREE_LIST,
        RESTART
    } parse_state = SEGMENT;

    tmp = phys_seg_str;
    while ((line = strsep(&tmp, "\n")) != NULL) {
        if (line[0] == '\0') continue;

        switch (parse_state) {
        case SEGMENT:
            {
                uint32_t segment = 0;
                ret = sscanf(line, "SEGMENT %" SCNu32, &segment);
                if (ret != 1) {
                    fprintf(stderr, "Failed to parse '%s' as SEGMENT\n", line);
                    return 1;
                }
                if (segment != ram_region_index) {
                    fprintf(stderr, "Unexpected segment index %u\n", segment);
                    return 1;
                }
               break;
            }
        case START:
            {
                uint64_t start = 0;
                ret = sscanf(line, "start: %" SCNx64, &start);
                if (ret != 1) {
                    fprintf(stderr, "Failed to parse '%s' as START\n", line);
                    return 1;
                }
                RAM_regions[ram_region_index].start = start;
                break;
            }
        case END:
            {
                uint64_t end = 0;
                ret = sscanf(line, "end: %" SCNx64, &end);
                if (ret != 1) {
                    fprintf(stderr, "Failed to parse '%s' as END\n", line);
                    return 1;
                }
                RAM_regions[ram_region_index].end = end;
                break;
            }
        case DOMAIN:
        case FREE_LIST:
        case RESTART:
            /* Ignore these lines. */
            break;
        }
        parse_state++;
        if (parse_state == RESTART) {
            parse_state = SEGMENT;
            ram_region_index++;
            if (ram_region_index == MAX_REGIONS) {
                fprintf(stderr, "Reached RAM region table limit!\n");
                break;
            }
        }
    }

    free(phys_seg_str);

    return 0;

#else
#error Implement find_RAM_regions
#endif
}

static int start_pci(void)
{
    char *path = "/dev/mem";
    static int started = 0;

    if (started) {
        return 0;
    }

    started = 1;

    if (find_RAM_regions() != 0) {
        fprintf(stderr, "Unable to locate system RAM regions!\n");
        return 1;
    }

    fd = open(path, O_SYNC | O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "open(%s,UC) failed (%s)!\n", path, strerror(errno));
        return 1;
    }

    fd_cached = open(path, O_RDWR);
    if (fd_cached < 0) {
        fprintf(stderr, "open(%s,WB) failed (%s)!\n", path, strerror(errno));
        return 1;
    }

    pci_bus = pci_alloc();
    pci_init(pci_bus);
    pci_scan_bus(pci_bus);

    return 0;
}

static void stop_pci(void)
{
    if (pci_bus != NULL) {
        pci_cleanup(pci_bus);
    }

    if (fd >= 0) {
        close(fd);
    }

    if (fd_cached >= 0) {
        close(fd_cached);
    }
}

static void do_init(const char *arg)
{
    char init_args[19];

    if (arg != NULL) {
        lwBar0 = strtoul(arg, NULL, 0);
    }

    sprintf(init_args, PhysAddr_FMT, lwBar0);
    init(init_args);
}

static void print_help(FILE *stream)
{
    fprintf(stream, "\n");
    fprintf(stream, "Usage: lwwatch\n");
    fprintf(stream, "   or: lwwatch BAR0\n");
    fprintf(stream, "   or: lwwatch --dumpinit LWDEBUG-DUMPFILE\n");
    fprintf(stream, "   or for CheetAh Chips: lwwatch [BAR0] [BAR0_size] [chipId]\n");
    fprintf(stream, "\n");
    fprintf(stream, "Run lwwatch and point it at a GPU to inspect.\n");
    fprintf(stream, "By default, lwwatch will pick a GPU in the system.\n");
    fprintf(stream, "If a single argument is given, use the GPU indicated\n");
    fprintf(stream, "by BAR0.  If '--dumpinit' is specified, instead\n");
    fprintf(stream, "inspect the GPU information in the LwDebug dump file.\n");
    fprintf(stream, "\n");
    fprintf(stream, "From the lwwatch prompt, type 'help' for a command\n");
    fprintf(stream, "list.\n");
    fprintf(stream, "\n");
}

int main(int argc, char **argv)
{
    memset(&RAM_regions, 0, sizeof(RAM_regions));
    memset(&GPU_BARs, 0, sizeof(GPU_BARs));
    memset(&current, 0, sizeof(current));

    if ((argc > 1) &&
        ((strcmp(argv[1], "-h") == 0) ||
         (strcmp(argv[1], "-help") == 0) ||
         (strcmp(argv[1], "--help") == 0))) {
        print_help(stdout);
        return 0;
    }


    if (argc == 4) { 
       // For CheetAh SOC chip
       GPU_BARs[0].base_address = strtoul(argv[1], NULL, 0);
       GPU_BARs[0].size = strtoul(argv[2], NULL, 0);
       isTegraHack = strtoul(argv[3], NULL, 0);
       do_init(argv[1]);
    } else if ((argc == 3) && (strcmp(argv[1], "--dumpinit") == 0)) {
        dumpinit(argv[2]);
    } else if (argc == 2) {
        do_init(argv[1]);
    } else if (argc == 1) {
        do_init(NULL);
    } else {
        print_help(stderr);
        return 1;
    }

    main_loop(NULL);

    osDestroyHal();
    stop_pci();

    return 0;
}

void initLwWatch(void)
{
    struct pci_dev *dev;
    int prot = PROT_READ | PROT_WRITE, flags = MAP_SHARED;
    uint16_t command;
    uint32_t class, low;
    int i, j, k;
    LwBool is_virtual = LW_FALSE;

    if (start_pci() != 0) {
        return;
    }

    for (i = 0; i < 3; i++) {
        if (!GPU_BARs[i].mapping.p8)
            continue;
        if (munmap((void *)GPU_BARs[i].mapping.p8, GPU_BARs[i].size)) {
            fprintf(stderr,
                "Unable to unmap BAR%u: %s\n", i, strerror(errno));
            return;
        }
        GPU_BARs[i].mapping.p8 = NULL;
    }

    dev = pci_bus->devices;
    while (dev) {
        class = pci_read_word(dev, PCI_CLASS_DEVICE);

        if (class != PCI_CLASS_DISPLAY_VGA &&
            class != PCI_CLASS_DISPLAY_3D)
            goto next;

        if (pci_read_word(dev, PCI_VENDOR_ID) != 0x10de)
            goto next;

        pci_fill_info(dev, PCI_FILL_BASES);

        if ((dev->known_fields & PCI_FILL_BASES) == 0)
            goto next;

        for (j = 0, k = 0; j < MAX_BARS; j++) {
            low = pci_read_long(dev, PCI_BASE_ADDRESS_0 + j*4);
            if ((low & PCI_BASE_ADDRESS_SPACE) ==
                    PCI_BASE_ADDRESS_SPACE_MEMORY) {
                GPU_BARs[k].base_address = (dev->base_addr[j] & ~PAGEMASK);
                GPU_BARs[k].size = dev->size[j];
                if ((low & PCI_BASE_ADDRESS_MEM_TYPE_MASK) ==
                        PCI_BASE_ADDRESS_MEM_TYPE_64) {
                    GPU_BARs[k].base_address |= ((uint64_t)pci_read_long(dev,
                            PCI_BASE_ADDRESS_0 + (++j)*4) << 32);
                }
                k++;
            }
        }

        command = pci_read_word(dev, PCI_COMMAND);
        pci_write_word(dev, PCI_COMMAND, (command | PCI_COMMAND_MASTER |
                    PCI_COMMAND_MEMORY));

        if (lwBar0 == 0) {
            lwBar0 = GPU_BARs[0].base_address;
            break;
        } else if (lwBar0 == GPU_BARs[0].base_address)
            break;

next:
        dev = dev->next;
    }

    if ((lwBar0 == 0) && !dev) {
        fprintf(stderr, "No device is found; call 'init <BAR0>' to begin.\n");
        return;
    }

    // skip check on CheetAh as pci related var "dev" is NULL
    if ((!IsTegra()) && (lwBar0 != 0) && !dev) {
        fprintf(stderr, "BAR0 address invalid; Specified BAR0 offset not found.\n");
        return;
    }

    for (i = 0; i < 3; i++) {
        if (GPU_BARs[i].size == 0)
            continue;
        GPU_BARs[i].mapping.p8 = mmap(NULL, GPU_BARs[i].size, prot, flags,
                fd, GPU_BARs[i].base_address);
        if (GPU_BARs[i].mapping.p8 == MAP_FAILED) {
            fprintf(stderr,
                    "WARNING: Unable to map BAR%u (%s), accesses to that space will fail\n"
                    "         Base address = 0x%.16llx, size = 0x%.8x\n"
                    "         Use \"init <BAR0>\" to retry\n",
                    i, strerror(errno), (unsigned long long) GPU_BARs[i].base_address,
                    GPU_BARs[i].size );
            if (errno == EPERM) {
                fprintf(stderr,
                        "NOTE: EPERM error may be caused by kernel configuration.\n"
                        "      Check for CONFIG_IO_STRICT_DEVMEM, and either recompile\n"
                        "      kernel without that, or boot with \"iomem=relaxed\".\n");
            }
            memset(&GPU_BARs[i], 0, sizeof(struct MEM_region));
        }
    }

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch mode bits
    is_virtual = isVirtual();
    if (is_virtual)
    {
        // Set LwwatchMode as we need some PF registers before this can be set in exts cmds
        setLwwatchMode(LW_TRUE);
    }

    if(osInit() != LW_OK) {
        fprintf(stderr, "WARNING: Operating system specific initializations failed\n");
    }

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch mode bits
    if (is_virtual)
    {
        setLwwatchMode(LW_FALSE);
    }
}

LW_STATUS readPhysicalMem(LwU64 address, void *buffer, LwU64 bytes, LwU64 *read)
{
    uint64_t i;
    uint8_t unaligned;
    uint8_t tempBuffer[BYTE_ALIGNMENT];

    if ((buffer == NULL) || (read == NULL))
        return LW_ERR_GENERIC;

    *read     = 0;
    // Callwlate unaligned bytes and align the initial address
    unaligned = address &  (BYTE_ALIGNMENT - 1);
    address   = address & ~(BYTE_ALIGNMENT - 1);

    // If the address is not 4 byte aligned read the unaligned bytes first
    if (unaligned != 0) {
        // Read unaligned bytes from the aligned address
        *((uint32_t *)tempBuffer) = MMIO_RD32(address);
        for (i = unaligned; i < BYTE_ALIGNMENT && bytes; i++) {
            // Copy data starting from the unaligned offset
            *((uint8_t *)buffer + (*read)) = tempBuffer[i];
            (*read)++;
            bytes--;
        }
        address += BYTE_ALIGNMENT;
    }
   
    // Loop reading all the aligned bytes
    for (i = 0; bytes >= BYTE_ALIGNMENT; i++) {
        *((uint32_t *)((uint8_t *)buffer + (*read))) = MMIO_RD32(address);
        (*read) += BYTE_ALIGNMENT;
        address += BYTE_ALIGNMENT;
        bytes   -= BYTE_ALIGNMENT;
    }

    // Read any remaining unaligned bytes
    if (bytes != 0) {
        *((uint32_t *)tempBuffer) = MMIO_RD32(address);
        for (i = 0; bytes != 0; i++) {
            *((uint8_t *)buffer + (*read)) = tempBuffer[i];
            (*read)++;
            bytes--;
        }
    }

    return LW_OK;
}

LW_STATUS writePhysicalMem(LwU64 address, void *buffer, LwU64 bytes, LwU64 *written)
{
    uint64_t i;
    uint8_t unaligned;
    uint8_t tempBuffer[BYTE_ALIGNMENT];

    if ((buffer == NULL) || (written == NULL))
        return LW_ERR_GENERIC;

    *written  = 0;
    // Callwlate unaligned bytes and align the initial address
    unaligned = address &  (BYTE_ALIGNMENT - 1);
    address   = address & ~(BYTE_ALIGNMENT - 1);

    // If the address is not 4 byte aligned write the unaligned bytes first
    if (unaligned != 0) {
        // Read the 4 bytes first to keep the untouched data
        *((uint32_t *)tempBuffer) = MMIO_RD32(address);
        for (i = unaligned; i < BYTE_ALIGNMENT && bytes; i++) {
            tempBuffer[i] = *((uint8_t *)buffer + (*written));
            (*written)++;
            bytes--;
        }
        MMIO_WR32((address), *((uint32_t *)tempBuffer));
        address += BYTE_ALIGNMENT;
    }

    // Loop writing all the aligned bytes
    for (i = 0; bytes >= BYTE_ALIGNMENT; i++) {
        MMIO_WR32((address), *((uint32_t *)((uint8_t *)buffer + (*written))));
        (*written) += BYTE_ALIGNMENT;
        address    += BYTE_ALIGNMENT;
        bytes      -= BYTE_ALIGNMENT;
    }

    if (bytes != 0) {
        // Read the 4 bytes first to keep the untouched data
        *((uint32_t *)tempBuffer) = MMIO_RD32(address);
        for (i = 0; bytes != 0; i++) {
            tempBuffer[i] = *((uint8_t *)buffer + (*written));
            (*written)++;
            bytes--;
        }
        MMIO_WR32((address), *((uint32_t *)tempBuffer));
    }

    return LW_OK;
}

LwU32 osRegRd32(PhysAddr offset)
{
    if (lwMode == MODE_DUMP) {
        return REG_RD32_DUMP(offset);
    }

    if (lwBar0 == 0) {
        fprintf(stderr, "BAR0 address unknown; call 'init <BAR0>' to begin.\n");
        return 0;
    }
    return GPU_REG_RD32(offset);
}

void osRegWr32(PhysAddr offset, LwU32 data)
{
    if (lwBar0 == 0) {
        fprintf(stderr, "BAR0 address unknown; call 'init <BAR0>' to begin.\n");
        return;
    }
    GPU_REG_WR32(offset, data);
}

LwU8 osRegRd08(PhysAddr offset)
{
    if (lwBar0 == 0) {
        fprintf(stderr, "BAR0 address unknown; call 'init <BAR0>' to begin.\n");
        return 0;
    }
    return GPU_REG_RD08(offset);
}

void osRegWr08(PhysAddr offset, LwU8 data)
{
    if (lwBar0 == 0) {
        fprintf(stderr, "BAR0 address unknown; call 'init <BAR0>' to begin.\n");
        return;
    }
    GPU_REG_WR08(offset, data);
}

LwU32 RD_PHYS32(PhysAddr address)
{
    return MMIO_RD32(address);
}

void WR_PHYS32(PhysAddr address, LwU32 data)
{
    MMIO_WR32(address, data);
}

LwU32 FB_RD32(LwU32 offset)
{
    if (lwMode == MODE_DUMP) {
        return FB_RD32_DUMP(offset);
    }

    if (lwBar1 == 0) {
        fprintf(stderr, "BAR1 address unknown; call 'init <BAR0>' to begin.\n");
        return 0;
    }
    return GPU_FB_RD32(offset);
}

LwU32 FB_RD32_64(LwU64 offset)
{
    if ((lwBar1 & ~0xffffffffULL) != 0) {
        fprintf(stderr, "FB_RD32_64(): offset 0x%llx too large!\n", offset);
        return 0;
    }
    return FB_RD32((LwU32)offset);
}

void FB_WR32(LwU32 offset, LwU32 data)
{
    if (lwBar1 == 0) {
        fprintf(stderr, "BAR1 address unknown; call 'init <BAR0>' to begin.\n");
        return;
    }
    GPU_FB_WR32(offset, data);
}

LwU8 SYSMEM_RD08(LwU64 address)
{
    if ((address & ~0xffffffffULL) != 0) {
        fprintf(stderr, "SYSMEM_RD08(): address 0x%llx too large!\n", address);
        return 0;
    }
    return MMIO_RD08(address);
}

LwU32 SYSMEM_RD32(LwU64 address)
{
    if ((address & ~0xffffffffULL) != 0) {
        fprintf(stderr, "SYSMEM_RD32(): address 0x%llx too large!\n", address);
        return 0;
    }
    return MMIO_RD32(address);
}

void SYSMEM_WR32(LwU64 address, LwU32 data)
{
    if ((address & ~0xffffffffULL) != 0) {
        fprintf(stderr, "SYSMEM_WR32(): address 0x%llx too large!\n", address);
        return;
    }
    MMIO_WR32(address, data);
}

LW_STATUS osPciRead08(LwU16 domain, LwU8 bus, LwU8 device, LwU8 function, LwU8 *value, LwU32 offset)
{
    struct pci_dev *dev;
    int ret;
    if (!value)
        return LW_ERR_GENERIC;
    *value = 0;
    dev = find_pci_device(domain, bus, device, function);
    if (!dev)
        return LW_ERR_GENERIC;
    *value = pci_read_byte(dev, offset);
    return LW_OK;
}

LW_STATUS osPciRead16(LwU16 domain, LwU8 bus, LwU8 device, LwU8 function, LwU16 *value, LwU32 offset)
{
    struct pci_dev *dev;
    int ret;
    if (!value)
        return LW_ERR_GENERIC;
    *value = 0;
    dev = find_pci_device(domain, bus, device, function);
    if (!dev)
        return LW_ERR_GENERIC;
    *value = pci_read_word(dev, offset);
    return LW_OK;
}

LW_STATUS osPciRead32(LwU16 domain, LwU8 bus, LwU8 device, LwU8 function, LwU32 *value, LwU32 offset)
{
    struct pci_dev *dev;
    int ret;
    if (!value)
        return LW_ERR_GENERIC;
    *value = 0;
    dev = find_pci_device(domain, bus, device, function);
    if (!dev)
        return LW_ERR_GENERIC;
    *value = pci_read_long(dev, offset);
    return LW_OK;
}

LW_STATUS osPciWrite32(LwU16 domain, LwU8 bus, LwU8 device, LwU8 function, LwU32 Data, LwU32 offset)
{
    struct pci_dev *dev;
    int ret;

    dev = find_pci_device(domain, bus, device, function);
    if (!dev)
        return LW_ERR_GENERIC;
    pci_write_word(dev, offset, Data);
    return LW_OK;
}

LW_STATUS osPciFindDevices(LwU16 DeviceId, LwU16 VendorId, osPciCallback callback)
{
    struct pci_dev *dev;

    dev = pci_bus->devices;
    while (dev)
    {
        if (((LwU16) pci_read_word(dev, PCI_DEVICE_ID)) == DeviceId
                && ((LwU16) pci_read_word(dev, PCI_VENDOR_ID) == VendorId))
        {
            callback(dev->domain, dev->bus, dev->dev, dev->func);
        }
        dev = dev->next;
    }
    return LW_OK;
}

LW_STATUS osPciFindDevicesByClass(LwU32 classCode, osPciCallback callback)
{
    struct pci_dev *dev;

    dev = pci_bus->devices;
    while (dev)
    {
        if (pci_read_word(dev, PCI_CLASS_DEVICE) == classCode)
        {
            callback(dev->domain, dev->bus, dev->dev, dev->func);
        }
        dev = dev->next;
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
    if (NULL == BaseAddr || NULL == BarSize || BarIndex >= MAX_BARS)
    {
        return LW_ERR_GENERIC;
    }

    struct pci_dev *dev;
    int i, j;
    uint32_t low;

    for (dev = pci_bus->devices; dev; dev = dev->next)
    {
        if (dev->domain != DomainNumber || dev->bus != BusNumber
                || dev->dev != Device || dev->func != Function)
        {
            continue;
        }

        pci_fill_info(dev, PCI_FILL_BASES);

        if ((dev->known_fields & PCI_FILL_BASES) == 0)
            continue;

        for (i = 0, j = 0; j < MAX_BARS; i++)
        {
            low = pci_read_long(dev, PCI_BASE_ADDRESS_0 + (i * 4));

            if ((low & PCI_BASE_ADDRESS_SPACE) == PCI_BASE_ADDRESS_SPACE_MEMORY)
            {
                if (j == BarIndex)
                {
                    *BaseAddr = (dev->base_addr[i] & ~PAGEMASK);
                    *BarSize = dev->size[i];
                    if ((low & PCI_BASE_ADDRESS_MEM_TYPE_MASK) ==
                                    PCI_BASE_ADDRESS_MEM_TYPE_64)
                    {
                        *BaseAddr |= ((uint64_t)pci_read_long(dev,
                                PCI_BASE_ADDRESS_0 + (i + 1) * 4) << 32);
                    }

                    return LW_OK;
                }

                if ((low & PCI_BASE_ADDRESS_MEM_TYPE_MASK) ==
                        PCI_BASE_ADDRESS_MEM_TYPE_64)
                {
                    ++i;
                }
                ++j;
            }
        }
    }

    return LW_ERR_GENERIC;
}

LW_STATUS osMapDeviceMemory
(
    LwU64 BaseAddr,
    LwU64 Size,
    MemProtFlags prot,
    void **ppBar
)
{
    int flags = MAP_SHARED;
    int memProt = PROT_NONE;
    if (prot & MEM_PROT_EXEC)
        memProt |= PROT_EXEC;
    if (prot & MEM_PROT_READ)
        memProt |= PROT_READ;
    if (prot & MEM_PROT_WRITE)
        memProt |= PROT_WRITE;

    *ppBar = mmap(NULL, Size, memProt, flags, fd, BaseAddr);
    return LW_OK;
}

LW_STATUS osUnMapDeviceMemory(void *pBar, LwU64 BarSize)
{
    munmap(pBar, BarSize);
    return LW_OK;
}

