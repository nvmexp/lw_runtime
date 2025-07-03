#ifndef __MOBILE_H__
#define __MOBILE_H__

#define _GNU_SOURCE
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

#ifndef PAGESIZE
#define PAGESIZE (getpagesize())
#endif

#define PAGEMASK (getpagesize() - 1)

#include "os.h"
#include "exts.h"
#include "vgpu.h"

#define DEV_LWGPU 0
#define DEV_NON_LWGPU 1

#ifdef GPU_REG_RD32
#undef GPU_REG_RD32
#undef GPU_REG_WR32
#endif

#ifdef GPU_REG_RD08
#undef GPU_REG_RD08
#undef GPU_REG_WR08
#endif

LwU32 GPU_REG_RD32_DIRECT(PhysAddr reg);
void GPU_REG_WR32_DIRECT(PhysAddr offset, LwU32 value);

#define GPU_REG_RD32(offset)  (((isVirtualWithSriov())) ? pfRegRead(offset) : GPU_REG_RD32_DIRECT(offset))
#define GPU_REG_WR32(offset,value)\
     (((isVirtualWithSriov())) ? pfRegWrite(offset, value) : GPU_REG_WR32_DIRECT((offset), (value)))
#define GPU_REG_RD08(offset) MMIO_RD08(lwBar0 + (offset))
#define GPU_REG_WR08(offset,value) MMIO_WR08((lwBar0 + (offset)), (value))

#define GPU_FB_RD32(offset) MMIO_RD32(lwBar1 + (offset))
#define GPU_FB_WR32(offset,value) MMIO_WR32((lwBar1 + (offset)), (value))

#define IS_IN_MEM_REGION(BAR, address) \
    ((address >= BAR.base_address) && (address < (BAR.base_address + BAR.size)))
#define IS_IN_RAM_REGION(RAM, address) \
    ((address >= RAM.start) && (address < RAM.end))

uint32_t MMIO_RD32(uint64_t address); // different for unix/android
void MMIO_WR32(uint64_t address, uint32_t value);
uint8_t MMIO_RD08(uint64_t address); // different for unix/android
void MMIO_WR08(uint64_t address, uint8_t value);
LW_STATUS readPhysicalMem(LwU64, void *, LwU64, LwU64*);
LW_STATUS writePhysicalMem(LwU64, void *, LwU64, LwU64*);

int mobile_init_args(int argc, char** argv);
ssize_t mobile_getline(char** lineptr, size_t* n, FILE* str);
void mobile_init_gpu_bars(char**);
void mobile_do_nothing();
int mobile_regOps(LwU64, uint32_t*, LwBool, int, int);
int find_RAM_regions( ssize_t (*getline)(char**, size_t*, FILE*) );

typedef struct {
    char arguments[19];

#define MAX_REGIONS 8
    struct RAM_region {
        LwU64 start, end;
    } RAM_regions[MAX_REGIONS];

#define MAX_BARS 6
    struct MEM_region {
        LwU64 base_address;
        uint32_t size;
        union {
            volatile uint8_t *p8;
            volatile uint32_t *p32;
        } mapping;
    } GPU_BARs[MAX_BARS], current;

    int fd, fd_cached;
} lww_main;

extern lww_main lwwatch;

typedef struct {
    LwU64 offset;
    uint32_t value;
    uint32_t isRead;
    uint32_t status;
    uint32_t devId;
    uint32_t alignment;
} RegOp;
#endif
