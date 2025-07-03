/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2007-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <fcntl.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <unistd.h>
#if defined(HWSNOOP_RPC)
#include <getopt.h>
#endif
#include <pci/pci.h>

#define PAGESIZE (getpagesize())
#define PAGEMASK (getpagesize() - 1)

#ifndef PCI_CAP_ID_EXP
#define PCI_CAP_ID_EXP 0x10
#endif

#ifndef PCI_CLASS_DISPLAY_3D
#define PCI_CLASS_DISPLAY_3D 0x302
#endif

#include <readline/readline.h>
#include <readline/history.h>

#include "os.h"
#include "exts.h"

#ifdef GPU_REG_RD32
#undef GPU_REG_RD32
#undef GPU_REG_WR32
#endif

#ifdef GPU_REG_RD08
#undef GPU_REG_RD08
#undef GPU_REG_WR08
#endif

#define GPU_REG_RD32(offset) PCI_RD32(lwBar0 + (offset))
#define GPU_REG_WR32(offset,value) PCI_WR32((lwBar0 + (offset)), (value))
#define GPU_REG_RD08(offset) PCI_RD08(lwBar0 + (offset))
#define GPU_REG_WR08(offset,value) PCI_WR08((lwBar0 + (offset)), (value))

#define GPU_FB_RD32(offset) PCI_RD32(lwBar1 + (offset))
#define GPU_FB_WR32(offset,value) PCI_WR32((lwBar1 + (offset)), (value))

#include "hwsnoop.h"
#include "hwsnoop_drf.h"
#include "hwsnoop_dma.h"

#if !defined(HWSNOOP_RPC)
#include <byteswap.h>
#define SWAP_BYTES_32(x) bswap_32(x)
#else
#define SWAP_BYTES_32(x) (x)
#endif

static int debug;

static hwsnoop_state_t *state;
static char *host;

static void PCI_WR32(PhysAddr address, uint32_t value)
{
    int ret;
    ret = hwsnoop_pci_write_long(state, address, value, 1);
    if (ret < 0) {
        if (debug) {
            fprintf(stderr, "hwsnoop_pci_write_long() failed (%d).\n",
                    hwsnoop_errno);
        }
    }
}

static uint32_t PCI_RD32(PhysAddr address)
{
    int ret;
    uint32_t value;
    ret = hwsnoop_pci_read_long(state, address, &value, 1);
    if (ret < 0) {
        if (debug) {
            fprintf(stderr, "hwsnoop_pci_read_long() failed (%d).\n",
                    hwsnoop_errno);
        }
        value = 0xffffffff;
    }
    return value;
}

static void PCI_WR08(PhysAddr address, uint8_t value)
{
    int ret;
    ret = hwsnoop_pci_write_byte(state, address, value, 1);
    if (ret < 0) {
        if (debug) {
            fprintf(stderr, "hwsnoop_pci_write_byte() failed (%d).\n",
                    hwsnoop_errno);
        }
    }
}

static uint8_t PCI_RD08(PhysAddr address)
{
    int ret;
    uint8_t value;
    ret = hwsnoop_pci_read_byte(state, address, &value, 1);
    if (ret < 0) {
        if (debug) {
            fprintf(stderr, "hwsnoop_pci_read_byte() failed (%d).\n",
                    hwsnoop_errno);
        }
        return 0xff;
    }
    return value;
}

static int PCI_RWxx(uint64_t address, void *buffer, uint64_t transfer_length,
        LwBool write_access, LwBool snoop_enable)
{
    uint32_t dma_flags;
    uint32_t page_table[2], page_table_entries;
    uint32_t bytes;
    uint8_t *aligned_buffer;
    uint64_t offset = 0;
    int ret;

    if (((uintptr_t)buffer & PAGEMASK) == 0)
        aligned_buffer = buffer;
    else {
        aligned_buffer = memalign(PAGESIZE, transfer_length);
        if (!aligned_buffer) {
            fprintf(stderr, "memalign() failed (%d).\n", errno);
            return -ENOMEM;
        }

        if (write_access)
            memcpy((void *)aligned_buffer, buffer, transfer_length);
    }

    dma_flags = HWSNOOP_DRF_DEF(_DMA, _FLAGS, _TYPE, _CONTIGUOUS);

    if (!write_access) {
        dma_flags = HWSNOOP_FLD_SET_DRF(_DMA,
                _FLAGS, _DIRECTION, _FROM_PCI, dma_flags);
    } else {
        dma_flags = HWSNOOP_FLD_SET_DRF(_DMA,
                _FLAGS, _DIRECTION, _TO_PCI, dma_flags);
    }

    if (!snoop_enable) {
        dma_flags = HWSNOOP_FLD_SET_DRF(_DMA,
                _FLAGS, _SNOOP_ENABLE, _DISABLED, dma_flags);
    } else {
        dma_flags = HWSNOOP_FLD_SET_DRF(_DMA,
                _FLAGS, _SNOOP_ENABLE, _ENABLED, dma_flags);
    }

    while (transfer_length > 0) {
        page_table[0] = (HWSNOOP_DRF_NUM(_DMA,
                    _TRANSFER, _PTE_ADJUST,
                    HWSNOOP_DRF_VAL(_DMA,
                        _TRANSFER, _PTE_ADJUST, address)) |
                HWSNOOP_DRF_NUM(_DMA,
                    _TRANSFER, _PTE_PFN_LOW,
                    HWSNOOP_DRF_VAL(_DMA,
                        _TRANSFER, _PTE_PFN_LOW, address)));

        if (HWSNOOP_DRF_VAL(_DMA,
                    _TRANSFER, _PTE_PFN_HIGH, address) > 0) {
            page_table[0] = HWSNOOP_FLD_SET_DRF(_DMA,
                    _TRANSFER, _PTE_WIDTH, _64BIT, page_table[0]);
            page_table[1] = HWSNOOP_DRF_VAL(_DMA,
                    _TRANSFER, _PTE_PFN_HIGH, address);
            page_table_entries = 2;
        } else {
            page_table[0] = HWSNOOP_FLD_SET_DRF(_DMA,
                    _TRANSFER, _PTE_WIDTH, _32BIT, page_table[0]);
            page_table_entries = 1;
        }

        bytes = min(PAGESIZE, transfer_length);

        ret = hwsnoop_pci_dma_transfer_pages(state, dma_flags, bytes,
                (aligned_buffer + offset), page_table,
                page_table_entries);
        if (ret < 0) {
            fprintf(stderr, "hwsnoop_pci_dma_transfer_pages() failed (%d).\n",
                    hwsnoop_errno);
            return ret;
        }

        address += bytes;
        offset += bytes;
        transfer_length -= bytes;
    }

    if (aligned_buffer != buffer) {
        if (!write_access) {
            transfer_length = offset;
            memcpy(buffer, (void *)aligned_buffer, transfer_length);
        }

        free(aligned_buffer);
    }

    return 0;
}

static struct option options[] = {
#if defined(HWSNOOP_RPC)
    { "host", required_argument, NULL, 'h' },
#endif
    { NULL,   0,                 NULL, 0   }
};

int main(int argc, char **argv)
{
    char init_args[19];
    int c, ret;
    char *optstr, *endptr;

    if (getelw("HWSNOOP_DEBUG"))
        debug = 1;

    optstr = "h:";
    while (1) {
        c = getopt_long(argc, argv, optstr, options, NULL);
        if (c < 0)
            break;

        switch (c) {
#if defined(HWSNOOP_RPC)
            case 'h':
                host = optarg;
                break;
#endif
            default:
                fprintf(stderr, "Bad command line option.\n");
                return 1;
        }
    }

    argc -= optind;

#if defined(HWSNOOP_RPC)
    if (!host)
        host = getelw("HWSNOOP_HOST");
    if (!host) {
        fprintf(stderr, "Bad host.\n");
        return 1;
    }
#endif

    if (argc > 1) {
        fprintf(stderr, "Bad command line.\n");
        exit(1);
    } else if (argc == 1) {
        lwBar0 = strtoul(argv[optind], &endptr, 0);
        if (*endptr != '\0') {
            fprintf(stderr, "Bad BAR0 address.\n");
            exit(1);
        }
    }

    ret = hwsnoop_init(host, &state);
    if (ret < 0) {
        fprintf(stderr, "failed to initialize libhwsnoop (%d).\n",
                hwsnoop_errno);
        return 1;
    }

    sprintf(init_args, PhysAddr_FMT, lwBar0);
    init(init_args);

    main_loop(NULL);

    hwsnoop_shutdown(state);
    osDestroyHal();

    return 0;
}

void initLwWatch(void)
{
    struct pci_access *bus;
    uint16_t class_device, command;
    struct pci_dev *dev;

    if (lwBar0 == 0) {
        bus = pci_alloc();
#if defined(HWSNOOP_RPC)
        bus->method_params[PCI_ACCESS_HWSNOOP] = host;
#endif
        pci_init(bus);
        pci_scan_bus(bus);

        dev = bus->devices;
        while (dev) {
            class_device = pci_read_word(dev, PCI_CLASS_DEVICE);

            if (class_device != PCI_CLASS_DISPLAY_VGA &&
                class_device != PCI_CLASS_DISPLAY_3D)
                goto next;

            if (pci_read_word(dev, PCI_VENDOR_ID) != 0x10de)
                goto next;

            pci_fill_info(dev, PCI_FILL_BASES);

            if ((dev->known_fields & PCI_FILL_BASES) == 0)
                goto next;

            command = pci_read_word(dev, PCI_COMMAND);
            pci_write_word(dev, PCI_COMMAND,
                    (command | (PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER)));

            lwBar0 = dev->base_addr[0];
            break;
next:
            dev = dev->next;
        }

        pci_cleanup(bus);
    }

    if (lwBar0 == 0) {
        fprintf(stderr, "BAR0 address unknown; call 'init <BAR0>' to begin.\n");
        return;
    }

    if ((GPU_REG_RD32(0x88000) & 0xffff) != 0x10de) {
        fprintf(stderr, "BAR0 address invalid; call 'init <BAR0>' to begin.\n");
        return;
    }

    osInit();
}

LW_STATUS readPhysicalMem(LwU64 address, void *buffer, LwU64 bytes, LwU64 *read)
{
    uint64_t i;
    int ret;

    if ((buffer == NULL) || (read == NULL))
        return LW_ERR_GENERIC;

    *read = 0;
    if (bytes < 8) {
        for (i = 0; i < bytes; i++) {
            *((uint8_t *)buffer + i) = PCI_RD08(address + i);
            (*read)++;
        }
    } else {
        ret = PCI_RWxx(address, buffer, bytes, LW_FALSE, LW_TRUE);
        if (ret < 0)
            return LW_ERR_GENERIC;
        *read = bytes;
    }

    return LW_OK;
}

LW_STATUS writePhysicalMem(LwU64 address, void *buffer, LwU64 bytes, LwU64 *written)
{
    uint64_t i;
    int ret;

    if ((buffer == NULL) || (written == NULL))
        return LW_ERR_GENERIC;

    *written = 0;
    if (bytes < 8) {
        for (i = 0; i < bytes; i++) {
            PCI_WR08((address + i), *((uint8_t *)buffer + i));
            (*written)++;
        }
    } else {
        ret = PCI_RWxx(address, buffer, bytes, LW_TRUE, LW_TRUE);
        if (ret < 0)
            return LW_ERR_GENERIC;
        *written = bytes;
    }

    return LW_OK;
}

LwU32 osRegRd32(PhysAddr offset)
{
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
    return PCI_RD32(address);
}

void WR_PHYS32(PhysAddr address, LwU32 data)
{
    PCI_WR32(address, data);
}

LwU32 FB_RD32(LwU32 offset)
{
    if (lwBar1 == 0) {
        fprintf(stderr, "BAR1 address unknown; call 'init <BAR0>' to begin.\n");
        return 0;
    }
    return GPU_FB_RD32(offset);
}

LwU32 FB_RD32_64(LwU64 offset)
{
    if ((lwBar1 & ~0xffffffffULL) != 0) {
        fprintf(stderr, "FB_RD32_64(): offset 0x%llx too large.\n", offset);
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
        fprintf(stderr, "SYSMEM_RD08(): address 0x%llx too large.\n", address);
        return 0;
    }
    return PCI_RD08(address);
}

LwU32 SYSMEM_RD32(LwU64 address)
{
    if ((address & ~0xffffffffULL) != 0) {
        fprintf(stderr, "SYSMEM_RD32(): address 0x%llx too large.\n", address);
        return 0;
    }
    return SWAP_BYTES_32(PCI_RD32(address));
}

void SYSMEM_WR32(LwU64 address, LwU32 data)
{
    if ((address & ~0xffffffffULL) != 0) {
        fprintf(stderr, "SYSMEM_WR32(): address 0x%llx too large.\n", address);
        return;
    }
    PCI_WR32(address, SWAP_BYTES_32(data));
}

LW_STATUS osPciRead08(LwU16 domain, LwU8 bus, LwU8 device, LwU8 function, LwU8 *value, LwU32 offset)
{
    int ret;
    if (!value)
        return LW_ERR_GENERIC;
    *value = 0;
    ret = hwsnoop_pci_cfg_read_byte(state, domain, bus, device,
            function, offset, value);
    if (ret < 0) {
        if (debug) {
            fprintf(stderr, "hwsnoop_pci_cfg_read_byte() failed (%d).\n",
                    hwsnoop_errno);
        }
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}

LW_STATUS osPciRead16(LwU16 domain, LwU8 bus, LwU8 device, LwU8 function, LwU16 *value, LwU32 offset)
{
    int ret;
    if (!value)
        return LW_ERR_GENERIC;
    *value = 0;
    ret = hwsnoop_pci_cfg_read_word(state, domain, bus, device,
            function, offset, value);
    if (ret < 0) {
        if (debug) {
            fprintf(stderr, "hwsnoop_pci_cfg_read_word() failed (%d).\n",
                    hwsnoop_errno);
        }
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}

LW_STATUS osPciRead32(LwU16 domain, LwU8 bus, LwU8 device, LwU8 function, LwU32 *value, LwU32 offset)
{
    int ret;
    if (!value)
        return LW_ERR_GENERIC;
    *value = 0;
    ret = hwsnoop_pci_cfg_read_long(state, domain, bus, device,
            function, offset, value);
    if (ret < 0) {
        if (debug) {
            fprintf(stderr, "hwsnoop_pci_cfg_read_long() failed (%d).\n",
                    hwsnoop_errno);
        }
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}

LW_STATUS osPciWrite32(LwU16 domain, LwU8 bus, LwU8 device, LwU8 function, LwU32 Data, LwU32 offset)
{
    fprintf(stderr, "hwsnoop osPciWrite32() STUB\n");
    return LW_ERR_GENERIC;
}


LW_STATUS osPciFindDevices(LwU16 DeviceId, LwU16 VendorId, osPciCallback callback)
{
    fprintf(stderr, "hwsnoop osPciFindDevices() STUB\n");
    return LW_ERR_GENERIC;
}

LW_STATUS osPciFindDevicesByClass(LwU32 classCode, osPciCallback callback)
{
    fprintf(stderr, "hwsnoop osPciFindDevicesByClass() STUB\n");
    return LW_ERR_GENERIC;
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
    return LW_ERR_GENERIC;
}

LW_STATUS osUnMapDeviceMemory(void *pBar, LwU64 BarSize)
{
    return LW_ERR_GENERIC;
}



LwU64 virtToPhys(LwU64 virtual_address, LwU32 pid)
{
    int ret;
    uint64_t physical_address;
    ret = hwsnoop_lookup_physical_address(state, pid, virtual_address,
            &physical_address, NULL);
    if (ret < 0) {
        if (debug) {
            fprintf(stderr, "hwsnoop_lookup_physical_address() failed (%d).\n",
                    hwsnoop_errno);
        }
        return 0;
    }
    return physical_address;
}

LwU32 GPU_REG_RD32_DIRECT(PhysAddr reg)
{
    return PCI_RD32(lwBar0 + (reg));
}

void GPU_REG_WR32_DIRECT(PhysAddr reg, LwU32 value)
{
    PCI_WR32(lwBar0 + (reg), value);
}
