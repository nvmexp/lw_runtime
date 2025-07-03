/*
 * Copyright (c) 2015-2019, LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
 
#include <nn/nn_Common.h>
#include <nn/nn_Log.h>
#include <nn/os.h>
#include <nn/init.h>
#include <nn/nn_Assert.h>
#include <nn/fs.h>
#include <nn/lmem/lmem_ExpHeap.h>
#include <nn/tma/tma.h>
#include <nn/mem/mem_StandardAllocator.h>
#include <lw/lw_MemoryManagement.h>
#include "lwca.h"
#include "lwdaLWN.h"
#include "lwdaNNAllocator.h"
#include <lwn/lwn_Cpp.h>
#include "lwnutil.h"
#include <stdio.h>
#include "lwn/lwn_CppFuncPtrImpl.h"  //Code to set up LWN C function pointer interface
#include <lwn/lwn_FuncPtrInline.h>
#include <string.h>
#include <lwn/lwn_CppFuncPtr.h>
#include <lwn/lwn_CppMethods.h>

#define ARRAY_SIZE 16
#define SHARED_MEM_SIZE 100
#define DEBUG_LOG 0

#ifdef __aarch64__
const size_t heapSize = 512 * 1024 * 1024;
#else
const size_t heapSize = 128 * 1024 * 1024;
#endif

static const int POOL_SIZE = 64 <<20;

 extern "C" PFNLWNGENERICFUNCPTRPROC LWNAPIENTRY lwnBootstrapLoader(const char *name);
 //---------------------------------------------------------------------------
//  This unnamed namespace includes file system heap allocator and deallcator
// ---------------------------------------------------------------------------


template <class T>
const T align(const T val, const T align)
{
    return (val + align -1) & ~(align - 1);
}

 namespace{

     const int FsHeapSize = 512 * 1024;
     const int TlsHeapSize = 1 * 1024 * 1024;

     uint8_t  g_FsHeapBuffer[FsHeapSize];
     nn::lmem::HeapHandle g_FsHeap;
     uint8_t g_TlsHeapBuffer[TlsHeapSize];
     nn::util::TypedStorage<nn::mem::StandardAllocator, sizeof(nn::mem::StandardAllocator),
                            NN_ALIGNOF(nn::mem::StandardAllocator)> g_TlsAllocator;

     void FsInitHeap()
     {
         g_FsHeap = nn::lmem::CreateExpHeap(g_FsHeapBuffer, FsHeapSize, nn::lmem::CreationOption_DebugFill);
     }

     void* FsAllocate(size_t size)
     {
         return nn::lmem::AllocateFromExpHeap(g_FsHeap, size);
     }

     void FsDeallocate(void* p, size_t size)
     {
         NN_UNUSED(size);
         return nn::lmem::FreeToExpHeap(g_FsHeap, p);
     }

     void* TlsAlloc(size_t size, size_t alignment)
     {
         return nn::util::Get(g_TlsAllocator).Allocate(size, alignment);
     }

     void TlsDealloc(void* p, size_t size)
     {
         nn::util::Get(g_TlsAllocator).Free(p);
         NN_UNUSED(size);
     }
 }

using namespace lwn;
 extern "C" void nninitStartup()
 {
     const size_t MallocMemorySize = 64 * 1024 * 1024;
     uintptr_t address;
     nn::Result result = nn::os::SetMemoryHeapSize(heapSize);
     NN_ASSERT( result.IsSuccess() );
     result = nn::os::AllocateMemoryBlock( &address, MallocMemorySize );
     NN_ASSERT( result.IsSuccess() );
     nn::init::InitializeAllocator( reinterpret_cast<void*>(address), MallocMemorySize );

     // Set file system allocator and deallocator
     FsInitHeap();
     nn::fs::SetAllocator(FsAllocate, FsDeallocate);

     new(&nn::util::Get(g_TlsAllocator)) nn::mem::StandardAllocator(g_TlsHeapBuffer, sizeof(g_TlsHeapBuffer));
     nn::os::SetMemoryAllocatorForThreadLocal(TlsAlloc, TlsDealloc);
 }

void dumpHex(unsigned char *buf, unsigned long long size)
{
     for (unsigned long long index=0; index<size; index++) {
         if((index % 64) == 0) NN_LOG("\n");
         NN_LOG("%02X ", buf[index]);
     }
}

void fillHexSequence(unsigned char *buf, unsigned long long size)
{
     for (unsigned long long index=0; index<size; index++) {
         buf[index] = index % 0xFF;
     }
}

int compareHexSequence(unsigned char *buf1, unsigned char *buf2, unsigned long long size)
{
     for (unsigned long long index=0; index<size; index++) {
         if(buf1[index] != buf2[index]) {
             NN_LOG("%d - %02X != %02X", index, buf1[index], buf2[index]);
             return 0;
         }
     }

     return 1;
}


void printAsIntBuf(unsigned char *buf, int width, int height, short formatPixelSize)
{
    unsigned long long bufIndex=0;
    const char *printFormat;

    NN_LOG("\n=========================================================\n");

    // Dump hex if formatPixelSize is more than 8
    if(formatPixelSize>8) {
        dumpHex(buf, width*height*formatPixelSize);
        goto done;
    }

    switch(formatPixelSize) {
    case 1: printFormat="%02X "; break;
    case 2: printFormat="%04X "; break;
    case 4: printFormat="%08X "; break;
    case 8: printFormat="%016X "; break;
    default:
            printFormat="%08X"; break;
    }

    // Create long long type pixel and print pattern
    for(int pixelIndex=0; pixelIndex<width*height; pixelIndex++) {
        unsigned long long pixel = 0;

        for(short index=0; index<formatPixelSize; index++) {
            pixel |= buf[bufIndex] << index*8;
            bufIndex++;
        }

        if((pixelIndex % width) == 0) NN_LOG("\n");

        NN_LOG(printFormat, pixel);
    }

done:
    NN_LOG("\n=========================================================\n\n");
}

void generateTextureData (unsigned char *ptr, int texWidth, int texHeight, short formatPixelSize)
{
    // Fill hex sequence if formatPixelSize is more than 8
    if(formatPixelSize>8) {
        fillHexSequence(ptr, texWidth*texHeight*formatPixelSize);
        return;
    }

    // Create long long type pixel and fill the pattern
    for (unsigned short heightIndex = 0; heightIndex < texHeight; heightIndex++) {
       for (unsigned short widthIndex = 0; widthIndex < texWidth; ++widthIndex) {
           int index = formatPixelSize * (heightIndex * texWidth + widthIndex);

           ptr[index + 0] = widthIndex&0x00FF;

           if(formatPixelSize>1) {
               ptr[index + 1] = (widthIndex&0xFF00)>>8;
               if(formatPixelSize>2) {
                   ptr[index + 2] = heightIndex&0x00FF;
                   if(formatPixelSize>2) {
                       ptr[index + 3] = (heightIndex&0xFF00)>>8;
                   }
               }
           }
       }
    }
}

int dataPatternMatch (unsigned char *ptr1, unsigned char *ptr2, unsigned long long size, short ptr1PixelSize, short ptr2PixelSize)
{
    unsigned long long ptr1Index=0, ptr2Index=0;


     // Compare hex sequence if formatPixelSize is more than 8. The formats bigger than 8 bytes will have same Pixel size in LWCA.
    if(ptr1PixelSize>8) {
        return compareHexSequence(ptr1, ptr2, size*ptr1PixelSize);
    }

    // LWCA array can only access pixel of size 4, 8 or 16. If the pixel size smaller or not in that range. It upgrade the size to next level.
    // So we need to reconstruct the pixel based on ptr1PixelSize and ptr2PixelSize give by caller and comapre the values.
    for (unsigned long long pixelIndex=0; pixelIndex<size; ++pixelIndex) {
        unsigned long long pixel1 = 0;
        unsigned long long pixel2 = 0;
        short index;

        for(index=0; index<ptr1PixelSize; index++) {
            pixel1 |= ((unsigned long long) ptr1[ptr1Index]) << index*8;
            ptr1Index++;
        }

        for(index=0; index<ptr2PixelSize; index++) {
            pixel2 |= ((unsigned long long) ptr2[ptr2Index]) << index*8;
            ptr2Index++;
        }

        if(pixel1 != pixel2) {
             return 0;
        }
    }

    return 1;
}


void debug(DebugCallbackSource::Enum source, DebugCallbackType::Enum type, int id,
           DebugCallbackSeverity::Enum severity, const char *message, void* userParam)
{
   printf("LWN DEBUG ERROR: %s\n", message);
}

/* The LWCA kernel binary array generated using below kernel.
texture<uchar4,2> texUC;

extern "C" __global__ void 
Tex2DUC(uchar4 *py, size_t Pitch, int Width, int Height, float uOffset, float vOffset, int bNormalized) 
{
    uchar4 *p = py+blockIdx.x*Width;
    for ( int i = threadIdx.x; i < Width; i += blockDim.x ) {
        float u = (float) i+uOffset;
        float v = (float) blockIdx.x+vOffset;
        if ( bNormalized ) { u /= (float) Width; v /= (float) Height; }
        p[i] = tex2D( texUC, u, v );
    }
}
*/
#ifdef __aarch64__
static const unsigned long long __lwbin_sm_53_texture_kernel[] = {
0x00100001ba55ed50ull,0x0000000000000cf0ull,0x0000004801010001ull,0x00000000000003c0ull,
0x00000040000003bbull,0x0000003500040003ull,0x0000000000000000ull,0x0000000000002015ull,
0x0000000000000000ull,0x0000000000000bb6ull,0x0000000000000000ull,0x762e1cf000010a13ull,
0x34206e6f69737265ull,0x677261742e0a332eull,0x33355f6d73207465ull,0x7365726464612e0aull,
0x3620657a69735f73ull,0x6f6c6721f0002e34ull,0x7865742e206c6162ull,0x5578657420666572ull,
0x6165772e0a0a3b43ull,0x20636e75662e206bull,0x206d617261702e28ull,0xf50012203233622eull,
0x6c61767465725f07ull,0x4d61647563202930ull,0x260a28636f6c6c61ull,0x5f11001834362700ull,
0x00202c305f3f0016ull,0x0a7b0a290a31a20bull,0x25e100676765722eull,0x0a0a0a3b3e323c72ull,
0x920011752e766f6dull,0x730a3b3033202c31ull,0x0928002700004974ull,0x2c5d302b40008f5bull,
0x7465720a3b9f0028ull,0xfe1a00c70a7d0a3bull,0x746547636e754602ull,0x7475626972747441ull,
0x0e00230d00d27365ull,0xe80f06002b0f00ddull,0x6369766544687600ull,0x240e00e90e009c65ull,
0x32332f0000ea0f00ull,0x18002c311f0b002lwll,0x6547326e0117321full,0x1b05010e0e009e74ull,
0x6e00ad0f01050400ull,0x61707563634f1afeull,0x634178614d79636eull,0x636f6c4265766974ull,
0x6c754d726550736bull,0x7365636f72706974ull,0x003b0f00cd726f73ull,0x2500430f02dc0e16ull,
0x1f1e00430f02090eull,0x01b6331f2f008632ull,0x6c46687469579f97ull,0x44052801bf736761ull,
0x27004c0f01c80e00ull,0x1f004c0c3001d10full,0x0098331f38009832ull,0x07f644022f341f38ull,
0x20656c6269736976ull,0x54207972746e652eull,0x01e9435544327865ull,0xba0e001534367534ull,
0x00018b07001d0e01ull,0x1d321f001d0a0095ull,0x1d661d012d070900ull,0x351f09001d341f00ull,
0xb401383617090057ull,0x3c70252064657270ull,0x363185014a3b3e33ull,0x0012353c73722520ull,
0x0023662528005f00ull,0x8100123117016d02ull,0x393c647225203436ull,0x12016f646c230180ull,
0x5b202c344b001775ull,0x0100263b5d270131ull,0x32380025351d004full,0x2c320cf401cb3b5dull,
0x3b782e6469742520ull,0x65672e707465730aull,0x317025093233732eull,0x3b35720df3001f2lwll,
0x726220317025400aull,0x3b335f3642422061ull,0x742e617476630a0aull,0x350100940409476full,
0x3672006234642800ull,0x006361746325202lwll,0x2e6f6c2e6c756d81ull,0x001f2c3723001a73ull,
0x4574766330006100ull,0x7225094100350000ull,0x2000173712007864ull,0x11001a02012d6e72ull,
0x0000613618006266ull,0x0a1000c36e13002bull,0x3200373a3221009full,0x64723200cb343673ull,
0x64610a3b7200cc35ull,0x00790100ab732e64ull,0x0069326429001a02ull,0x0034326624003501ull,
0x2e64322e78657481ull,0x0950002100009c76ull,0x3911001e3872257bull,0x3151000630110023ull,
0x2c320a455b202c7dull,0x7d316694003e7b20ull,0x01c96c68730a3b5dull,0x008a0a00792c3723ull,
0x00f303013b2c3826ull,0x7312010a36317533ull,0xa0020018311d0188ull,0x9a33110018301d00ull,
0x0017341000170d00ull,0x0001b30303b53811ull,0x00815b20383000d1ull,0x0200240300b45d10ull,
0x317333005f020041ull,0x7215022d0600aa7dull,0x746c230230010119ull,0x32150230321c0230ull,
0x33c0019932130230ull,0x7d0a3b7465720a3aull,0x00000000000a0a0aull,0x0000004001010002ull,
0x00000000000008a8ull,0x0000000000000000ull,0x0000003500010007ull,0x0000000000000000ull,
0x0000000000000015ull,0x0000000000000000ull,0x0000000000000000ull,0x33010102464c457full,
0x0000000000000007ull,0x0000004600be0002ull,0x0000000000000000ull,0x0000000000000800ull,
0x00000000000005c0ull,0x0038004000350535ull,0x0001000900400003ull,0x7472747368732e00ull,
0x747274732e006261ull,0x746d79732e006261ull,0x746d79732e006261ull,0x78646e68735f6261ull,
0x666e692e766e2e00ull,0x2e747865742e006full,0x0043554432786554ull,0x6f666e692e766e2eull,
0x435544327865542eull,0x6168732e766e2e00ull,0x327865542e646572ull,0x2e766e2e00435544ull,
0x746e6174736e6f63ull,0x5544327865542e30ull,0x6e2e6c65722e0043ull,0x6174736e6f632e76ull,
0x327865542e30746eull,0x68732e0000435544ull,0x2e00626174727473ull,0x2e00626174727473ull,
0x2e006261746d7973ull,0x735f6261746d7973ull,0x766e2e0078646e68ull,0x6554006f666e692eull,
0x742e004355443278ull,0x327865542e747865ull,0x2e766e2e00435544ull,0x7865542e6f666e69ull,
0x766e2e0043554432ull,0x2e6465726168732eull,0x0043554432786554ull,0x6e2e004355786574ull,
0x6174736e6f632e76ull,0x327865542e30746eull,0x7261705f00435544ull,0x2e6c65722e006d61ull,
0x74736e6f632e766eull,0x7865542e30746e61ull,0x4942240043554432ull,0x545f5353454c444eull,
0x455346464f5f5845ull,0x0000000000000054ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x000800030000003aull,0x0000000000000000ull,0x0000000000000000ull,
0x0007000300000072ull,0x0000000000000000ull,0x0000000000000000ull,0x0008101200000032ull,
0x0000000000000000ull,0x0000000000000140ull,0x0000001a0000006lwll,0x0000000000000000ull,
0x0000000000000000ull,0x0000000300082304ull,0x0008120400000000ull,0x0000000000000003ull,
0x0000000300081104ull,0x0010070400000000ull,0xffffffff00000004ull,0xffffffffffffffffull,
0x00080a0400001502ull,0x0024014000000002ull,0x000c170400241903ull,0x0020000600000000ull,
0x000c17040011f000ull,0x001c000500000000ull,0x000c17040011f000ull,0x0018000400000000ull,
0x000c17040011f000ull,0x0014000300000000ull,0x000c17040011f000ull,0x0010000200000000ull,
0x000c17040011f000ull,0x0008000100000000ull,0x000c17040021f000ull,0x0000000000000000ull,
0x00ff1b030021f000ull,0x0000003000041d04ull,0x0000002800081c04ull,0x00041e0400000108ull,
0x0000000000000210ull,0x0000000000000164ull,0x0000000400000006ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x003fb400e3e007f6ull,
0x4c98078000870001ull,0xf0c8000002170000ull,0x4b6d038005470007ull,0x0020c400e3e007fdull,
0xe30000000000000full,0xf0c8000002570002ull,0x5cb8000000270a07ull,0x001fd840fec207f1ull,
0x4e007f8005470203ull,0x4f107f8005470204ull,0x5b30019800470206ull,0x1045c40007a01ff0ull,
0x3829000001f70008ull,0x5cb8000000072a02ull,0xd832059020770204ull,0x001fc400fea207f1ull,
0x5c1080000067000aull,0x4c10000000270000ull,0x5c1008000087ff08ull,0x011fd000fc2007f1ull,
0x4bd7810005070a09ull,0x4b63038005470007ull,0x36f0020080870504ull,0x001fd400fe2007f2ull,
0x1a17040005170a05ull,0x36f0020081070208ull,0x5c98078000970004ull,0x007ff4003fa007f2ull,
0x36f0040081870302ull,0xeedc200000070402ull,0xe2400ffff600000full,0x001f8000ffe007ffull,
0xe30000000007000full,0xe2400fffff87000full,0x50b0000000070f00ull,0x001f8000fc0007e0ull,
0x50b0000000070f00ull,0x50b0000000070f00ull,0x50b0000000070f00ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000300000001ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000040ull,0x0000000000000094ull,
0x0000000000000000ull,0x0000000000000001ull,0x0000000000000000ull,0x000000030000000bull,
0x0000000000000000ull,0x0000000000000000ull,0x00000000000000d4ull,0x00000000000000beull,
0x0000000000000000ull,0x0000000000000001ull,0x0000000000000000ull,0x0000000200000013ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000198ull,0x0000000000000078ull,
0x0000000200000002ull,0x0000000000000008ull,0x0000000000000018ull,0x7000000000000029ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000210ull,0x0000000000000038ull,
0x0000000000000003ull,0x0000000000000004ull,0x0000000000000000ull,0x7000000000000040ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000248ull,0x00000000000000a4ull,
0x0000000800000003ull,0x0000000000000004ull,0x0000000000000000ull,0x000000090000007aull,
0x0000000000000000ull,0x0000000000000000ull,0x00000000000002f0ull,0x0000000000000010ull,
0x0000000700000003ull,0x0000000000000008ull,0x0000000000000010ull,0x0000000100000064ull,
0x0000000000000002ull,0x0000000000000000ull,0x0000000000000300ull,0x0000000000000168ull,
0x0000000800000000ull,0x0000000000000004ull,0x0000000000000000ull,0x0000000100000032ull,
0x0000000000000006ull,0x0000000000000000ull,0x0000000000000480ull,0x0000000000000140ull,
0x0b00000300000003ull,0x0000000000000020ull,0x0000000000000000ull,0x0000000500000006ull,
0x0000000000000800ull,0x0000000000000000ull,0x0000000000000000ull,0x00000000000000a8ull,
0x00000000000000a8ull,0x0000000000000008ull,0x0000000500000001ull,0x0000000000000300ull,
0x0000000000000000ull,0x0000000000000000ull,0x00000000000002a8ull,0x00000000000002a8ull,
0x0000000000000008ull,0x0000000600000001ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000008ull
};
#else // !aarch64
static const unsigned long long __lwbin_sm_53_texture_kernel[] = {
0x00100001ba55ed50ull,0x0000000000001350ull,0x0000004001010002ull,0x0000000000000e30ull,
0x0000000000000000ull,0x0000003500010007ull,0x0000000000000000ull,0x0000000000000014ull,
0x0000000000000000ull,0x0000000000000000ull,0x33010101464c457full,0x0000000000000007ull,
0x0000004600be0002ull,0x00000dd000000000ull,0x0035013500000c40ull,0x0028000300200034ull,
0x68732e000001000aull,0x2e00626174727473ull,0x2e00626174727473ull,0x2e006261746d7973ull,
0x735f6261746d7973ull,0x766e2e0078646e68ull,0x742e006f666e692eull,0x327865542e747865ull,
0x2e766e2e00435544ull,0x7865542e6f666e69ull,0x766e2e0043554432ull,0x2e6465726168732eull,
0x0043554432786554ull,0x736e6f632e766e2eull,0x65542e32746e6174ull,0x6e2e004355443278ull,
0x6174736e6f632e76ull,0x327865542e30746eull,0x6c65722e00435544ull,0x736e6f632e766e2eull,
0x65542e30746e6174ull,0x2e00004355443278ull,0x6261747274736873ull,0x6261747274732e00ull,
0x6261746d79732e00ull,0x6261746d79732e00ull,0x2e0078646e68735full,0x006f666e692e766eull,
0x0043554432786554ull,0x65542e747865742eull,0x6e2e004355443278ull,0x542e6f666e692e76ull,
0x2e00435544327865ull,0x65726168732e766eull,0x5544327865542e64ull,0x6f632e766e2e0043ull,
0x2e32746e6174736eull,0x0043554432786554ull,0x65740074736e6f63ull,0x7865542400435578ull,
0x635f5f2443554432ull,0x78336d735f616475ull,0x5f6e725f7669645full,0x33665f7a74666f6eull,
0x4432786554240032ull,0x6475635f5f244355ull,0x645f78336d735f61ull,0x6f6e5f6e725f7669ull,
0x5f3233665f7a7466ull,0x68746170776f6c73ull,0x6e6f632e766e2e00ull,0x542e30746e617473ull,
0x5f00435544327865ull,0x722e006d61726170ull,0x6f632e766e2e6c65ull,0x2e30746e6174736eull,
0x0043554432786554ull,0x53454c444e494224ull,0x464f5f5845545f53ull,0x0000000054455346ull,
0x0000000000000000ull,0x0000000000000000ull,0x000000000000003aull,0x0009000300000000ull,
0x000000000000006lwll,0x0007000300000000ull,0x000001f00000008eull,0x00090022000000a8ull,
0x00000298000000b4ull,0x0009002200000468ull,0x00000000000000e3ull,0x0008000300000000ull,
0x0000000000000032ull,0x0009101200000700ull,0x0000000000000088ull,0x0000001a00000000ull,
0x0000000400082304ull,0x0008120400000000ull,0x0000000000000004ull,0x0000000400081104ull,
0x0008230400000000ull,0x0000000000000003ull,0x0000000300081204ull,0x0008110400000000ull,
0x0000000000000003ull,0x0000000600082304ull,0x0008120400000000ull,0x0000000000000006ull,
0x0000000600081104ull,0x0010070400000000ull,0xffffffff00000007ull,0xffffffffffffffffull,
0x00080a0400001502ull,0x001c014000000005ull,0x000c1704001c1903ull,0x0018000600000000ull,
0x000c17040011f000ull,0x0014000500000000ull,0x000c17040011f000ull,0x0010000400000000ull,
0x000c17040011f000ull,0x000c000300000000ull,0x000c17040011f000ull,0x0008000200000000ull,
0x000c17040011f000ull,0x0004000100000000ull,0x000c17040011f000ull,0x0000000000000000ull,
0x00ff1b030011f000ull,0x0000001800041d04ull,0x00000050000c1c04ull,0x000001e800000150ull,
0x0000026000041e04ull,0x000007060000015lwll,0x7fffffff3f800000ull,0x0000000080000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x001cfc00e22007f6ull,0x4c98078000870001ull,
0xf0c8000002170006ull,0xf0c8000002570002ull,0x081fec42fe200ff1ull,0x4b6d038005270607ull,
0x4e007f8005270200ull,0x4f107f8005270203ull,0x001c4400ffa007f0ull,0x5b30001800370200ull,
0xe30000000000000full,0x5cb8000000270a02ull,0x003fd800e7a00731ull,0x4cb8000005272a07ull,
0x4cb8000005372a04ull,0x4c58000005570202ull,0x001ff400fda017fdull,0xe260000016000040ull,
0x4b6503800567ff07ull,0xe24000000b80000full,0x0000f400fe0007f6ull,0x5c98078000c70008ull,
0x5c98078000770004ull,0x5cb8000000672a02ull,0x1001c401ffa017f6ull,0x4c58000005470202ull,
0xe260000011800040ull,0xd832057040870c02ull,0x011fc400fe2007f6ull,0x4c10000000270609ull,
0x4b63038005270907ull,0x36f0010080870303ull,0x001fc400fc2007f5ull,0x5c10000000670002ull,
0x36f0018081070403ull,0x4c18010005070202ull,0x0007f400fe4007f4ull,0x5c98078000970006ull,
0x36f0018081870503ull,0xeedc000000070203ull,0x001fc000ffe01ffdull,0xe2400ffff600000full,
0xe30000000007000full,0x4c10000000270607ull,0x1005c402fe40003dull,0x5cb8000000672a03ull,
0x4c58000005470304ull,0xd832057040270408ull,0x011fd800fe2007f1ull,0x4b63038005270707ull,
0x5c10000000670006ull,0x36f0040080870903ull,0x001fd000fe2007e1ull,0x36f0018081070403ull,
0x4c18010005070608ull,0x5c98078000770006ull,0x00fff4005fa007f2ull,0x36f0018081870503ull,
0xeedc000000070803ull,0xe2400ffff780000full,0x001fb400fec007ffull,0xe30000000007000full,
0x5c98078000270003ull,0x5c88000000470300ull,0x0000f400fe0007fdull,0xe24000000680000full,
0x5c5930000ff7040aull,0x5080000000470405ull,0x101fd800fec217f6ull,0x5180028800070a0bull,
0x5980028000b70505ull,0x59807f800057030lwll,0x001f9800feca07f6ull,0x5980018000c70a0bull,
0x5980060000570b0lwll,0x5980018000c70a03ull,0x001fd801ffe007f0ull,0x598006000057030lwll,
0xe32000000007000full,0x5c98078000370009ull,0x001fd800ffe007fdull,0xe260000000800040ull,
0xe32000000007000full,0x5c98078000470005ull,0x001fc400fea007f1ull,0x3828000001770503ull,
0x5c98078000970004ull,0x040000000ff70303ull,0x001fd400fc2007f5ull,0x3828000001770409ull,
0x1c0ffffffff7030dull,0x040000000ff7090lwll,0x001fb400fd8007f1ull,0x366803800fd70d07ull,
0x1c0ffffffff70c0bull,0x366820000fd70b07ull,0x001fc400ffa007f0ull,0x5c9807800ff80009ull,
0xe24000001008000full,0x30cc03ff80070409ull,0x001ff400fda007f6ull,0x30cc03ff8007050aull,
0x5c40320000a709ffull,0xe24000003700000full,0x001ff400fda007f6ull,0x02c8020800170509ull,
0x5b6503800ff70907ull,0xe24000003300000full,0x001fb400fda007f1ull,0x36b283ff80070487ull,
0x36b283ff8007058full,0x5090038020070017ull,0x001fb400fec007fdull,0xe24000003002000full,
0x0407fffffff70409ull,0x5b6520800ff7090full,0x001fb400fec007fdull,0xe24000002c81000full,
0x0407fffffff70509ull,0x5b6520000ff70907ull,0x001fb000fe2007fdull,0xe24000002880000full,
0x5b6303800ff70b07ull,0x5b6303800ff70d0full,0x001fc400fe2007f6ull,0x5c9807800ff80009ull,
0x010ffffffc00f009ull,0x32807fdf80000404ull,0x001fd800fcc007f4ull,0x32807fdf80010505ull,
0x1c00000004010909ull,0x16ec08000007030aull,0x001fc400068007f2ull,0x5c11000000a7050aull,
0x5080000000470a0bull,0x5c5930000ff70a0dull,0x001f9402fe2007fdull,0x1c0ffffff8170c05ull,
0x5180058800070d0lwll,0x5c1a0b8000470504ull,0x281fd880fec007f6ull,0x5980058000c70b0bull,
0x59807f8000b7040eull,0x5980020000e70d0lwll,0x001f9400fe2007f6ull,0x5980070000b70c0eull,
0x5980020000e70d04ull,0x38c2018007f70503ull,0x001fd800fea007f1ull,0x5980070000b7040lwll,
0x5c10000000370905ull,0x3828000001770c0aull,0x001fd800fec007fdull,0x040000000ff70a03ull,
0x5c10000000570303ull,0x1c0ffffffff70309ull,0x001fb401ffa007edull,0x366203800fe70907ull,
0xe24000001480000full,0x366903800fe70307ull,0x001ff400fda007fdull,0xe24000001100000full,
0x3663038000170307ull,0xe32000000008000full,0x001ff400fe0007edull,0x376303fffe870307ull,
0x0408000000070c0lwll,0xe32000000000000full,0x001fc440feae07f1ull,0x5998070000b70405ull,
0x5b6b03800ff7030full,0x040007fffff70505ull,0x001f8400fe2007f5ull,0x1c0000000207030aull,
0x0420080000070509ull,0x5990070000b70405ull,0x001fd000fe4007f4ull,0x5988070000b70404ull,
0x5c48000000a7090aull,0x5bbd838000470507ull,0x001f9800fec007f1ull,0x5b6b00800ff70a0full,
0x5c1200000ff70304ull,0x5b4501800047ff03ull,0x001fd800fec007f1ull,0x5090038021070007ull,
0x5c28000000370903ull,0x3828000000170305ull,0x001fd800fec007fdull,0x38a004000017ff04ull,
0x3cf8028000170404ull,0x5c47000000370403ull,0x001ffc00fe0007e6ull,0x5c10000000370503ull,
0x5c47020000c7030lwll,0xe32000000007000full,0x001ffc00fe0007f6ull,0x0408000000070c03ull,
0x0427f8000007030lwll,0xe32000000007000full,0x001fd800ffe007f0ull,0x5c180b8000c7050lwll,
0xe32000000007000full,0x0248020800270503ull,0x001fc000ffe007f0ull,0x0427f8000007030lwll,
0xe32000000007000full,0x024802080027050lwll,0x0000f400fe4007ffull,0xe32000000007000full,
0x010ffc000007f003ull,0x508000000057030lwll,0x001ffc00fe001fffull,0xe32000000007000full,
0x5c5810000057040lwll,0xe32000000007000full,0x001f8000fc0007ffull,0xe2400fffff07000full,
0x50b0000000070f00ull,0x50b0000000070f00ull,0x001f8000fc0007e0ull,0x50b0000000070f00ull,
0x50b0000000070f00ull,0x50b0000000070f00ull,0x0000000000000000ull,0x0000000000000000ull,
0x0000000000000000ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000300000001ull,
0x0000000000000000ull,0x000000aa00000034ull,0x0000000000000000ull,0x0000000000000001ull,
0x000000030000000bull,0x0000000000000000ull,0x0000012f000000deull,0x0000000000000000ull,
0x0000000000000001ull,0x0000000200000013ull,0x0000000000000000ull,0x0000008000000210ull,
0x0000000500000002ull,0x0000001000000004ull,0x7000000000000029ull,0x0000000000000000ull,
0x0000008000000290ull,0x0000000000000003ull,0x0000000000000004ull,0x7000000000000040ull,
0x0000000000000000ull,0x000000a800000310ull,0x0000000900000003ull,0x0000000000000004ull,
0x0000000900000090ull,0x0000000000000000ull,0x00000008000003b8ull,0x0000000800000003ull,
0x0000000800000004ull,0x0000000100000064ull,0x0000000000000002ull,0x0000000c000003c0ull,
0x0000000900000000ull,0x0000000000000004ull,0x000000010000007aull,0x0000000000000002ull,
0x00000160000003clwll,0x0000000900000000ull,0x0000000000000004ull,0x0000000100000032ull,
0x0000000000000006ull,0x0000070000000540ull,0x0f00000600000003ull,0x0000000000000020ull,
0x00000dd000000006ull,0x0000000000000000ull,0x0000006000000060ull,0x0000000400000005ull,
0x000003c000000001ull,0x0000000000000000ull,0x0000086c0000086lwll,0x0000000400000005ull,
0x0000000000000001ull,0x0000000000000000ull,0x0000000000000000ull,0x0000000400000006ull,
0x0000004801010001ull,0x0000000000000498ull,0x0000004000000490ull,0x0000003500040003ull,
0x0000000000000000ull,0x0000000000002014ull,0x0000000000000000ull,0x0000000000000e93ull,
0x0000000000000000ull,0x762e1cf000010a13ull,0x34206e6f69737265ull,0x677261742e0a332eull,
0x33355f6d73207465ull,0x7365726464612e0aull,0x3320657a69735f73ull,0x6f6c6721f0002e32ull,
0x7865742e206c6162ull,0x5578657420666572ull,0x6165772e0a0a3b43ull,0x20636e75662e206bull,
0x206d617261702e28ull,0xf80012203233622eull,0x6c61767465725f07ull,0x4d61647563202930ull,
0x260a28636f6c6c61ull,0x00165f1100180600ull,0xa20b00202c305f3full,0x722e0a7b0a290a31ull,
0x3c7225e100216765ull,0x6f6d0a0a0a3b3e32ull,0x2c31920011752e76ull,0x4974730a3b303320ull,
0x8f5b092800270000ull,0x00282c5d302b4000ull,0x0a3b7465720a3b9full,0x4602fe1a00c70a7dull,
0x7441746547636e75ull,0x7365747562697274ull,0x00dd0e00230d00d2ull,0x7600e80f06002b0full,
0x9c65636976654468ull,0x0e00240e00e90e00ull,0x311f0e002c0f00eaull,0x6e0117321f18002lwll,
0x0e0e009e74654732ull,0x0f010504001b0501ull,0x63634f1afe6e00adull,0x614d79636e617075ull,
0x4265766974634178ull,0x726550736b636f6lwll,0x6f727069746c754dull,0x00cd726f73736563ull,
0x0f01f20e16003b0full,0x2f0043311f250043ull,0xb6331f2f0043321full,0x46687469579f9701ull,
0x052801bf7367616lwll,0x004c0f01c80e0044ull,0x321f38004c311f2eull,0x38004c331f38004lwll,
0x7607f644022f341full,0x2e20656c62697369ull,0x6554207972746e65ull,0x0001e94355443278ull,
0x01ba0e001503005bull,0x09001d311f001d0eull,0x012d0709001d321full,0x001d341f001d661dull,
0x3617090057351f09ull,0x2064657270b40138ull,0x014a3b3e353c7025ull,0x3c73722520363185ull,
0x257b005f00001239ull,0x016e3b3e33313c66ull,0x2034366273002406ull,0x646c230181647225ull,
0x2c334b0186030170ull,0x263b5d2d01325b20ull,0x26321f0026301d00ull,0x33180026311d0000ull,
0x25341d00ad020026ull,0x25351d0025341e00ull,0x321d000070351f00ull,0x630a3b5d36b30026ull,
0x099c6f742e617476ull,0x3b1700ed00026e06ull,0x25202c3206f10281ull,0x3b782e6469617463ull,
0x2e6f6c2e6c756d0aull,0x001f2c3323001a73ull,0x3852003230317239ull,0x737000317425202lwll,
0x003265672e707465ull,0x001f2c3170250954ull,0x70254001f0003301ull,0x4242206172622031ull,
0x3000940a3b345f36ull,0x004b0000dd6e722eull,0x5300313666250941ull,0x0000f36464610a3bull,
0x3566280017010043ull,0x6e25202c35530074ull,0x0077010046070074ull,0x11001a0d00a46615ull,
0x640a3b3163001a37ull,0x66252054001a7669ull,0x00c2371300692c33ull,0x2c322300c2716523ull,
0xbf321500bf020152ull,0x00080a0a3b334000ull,0xf7381200663a322eull,0x202c393200c70800ull,
0x00007d34662b0018ull,0x3b32c00020030245ull,0x2e64322e7865740aull,0x6100220000e83476ull,
0x0086343172257b09ull,0x6101230100063511ull,0x0b585b202c7d3731ull,0x333200457b202c34ull,
0x2601b00200745d7dull,0x0a3b3363017f3831ull,0x39313303076c6873ull,0x3500303218001f2lwll,
0x3d391202132c3032ull,0x22019a3631753201ull,0x0018371d00187372ull,0x120018361d012202ull,
0xcc0200180d00b433ull,0x028c0304e3341100ull,0x825b20383000ee00ull,0x00250300cf5d1000ull,
0x7339006102004302ull,0x1400d50100ab7d31ull,0x6c2301b401016835ull,0x150276331d01b474ull,
0x620a3b32a501b733ull,0x0285696e752e6172ull,0x2e0001c60f01ce01ull,0x03020a0001c73131ull,
0x0401ae34662f001aull,0x1200b70100383212ull,0x0201ae341b008c32ull,0x35322f01ae0a0046ull,
0x1c031532120401aeull,0x1e01ae371501ae35ull,0x341d004d02017e32ull,0x18331d004c020018ull,
0x00180d00b4371200ull,0x0301ae311f024302ull,0x0200250101ae3713ull,0x35732f0061020043ull,
0x1d01ae341d1701aeull,0x720a3a34c0036534ull,0x0a0a0a7d0a3b7465ull,0x0000000000000000ull
};
#endif // !aarch64

static char s_poolStorage[POOL_SIZE] __attribute__((aligned(LWN_MEMORY_POOL_STORAGE_ALIGNMENT)));
static unsigned char texBuf_poolStorage[POOL_SIZE] __attribute__((aligned(LWN_MEMORY_POOL_STORAGE_ALIGNMENT)));
static unsigned char t_poolStorage[POOL_SIZE/2] __attribute__((aligned(LWN_MEMORY_POOL_STORAGE_ALIGNMENT)));
static const int COMMAND_MEMORY = 4096;
static char s_controlMemory[COMMAND_MEMORY];
static Device device;
static Queue queue;
static MemoryPool pool,poolGPU, textureBufferPool;
static uint64_t commandMemoryBegin;

unsigned int DeviceGet(DeviceInfo pname)
{
    int v;
    device.GetInteger(pname, &v);
    return v;
}

unsigned char lwdaCopyBuffer[POOL_SIZE] = {0};
unsigned char texdata[POOL_SIZE];
LWcontext ctx;
LWdevice dev;


void initLwn()
{
    ///--------------------------------------------------------------------------------
    // Initialize LWN driver interface.
    //--------------------------------------------------------------------------------
    DeviceGetProcAddressFunc getProcAddress = (DeviceGetProcAddressFunc) ((lwnBootstrapLoader)("lwnDeviceGetProcAddress"));
    lwnLoadCPPProcs(NULL, getProcAddress);

    DeviceBuilder deviceBuilder;
    deviceBuilder.SetDefaults();
    deviceBuilder.SetFlags(LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_2_BIT);
    if (!device.Initialize(&deviceBuilder)) {
        NN_SDK_LOG("Failed to init device");
    }

    lwnLoadCPPProcs(&device, getProcAddress);
    device.InstallDebugCallback(debug, NULL, LWN_TRUE);

    int majorVersion = DeviceGet(DeviceInfo::API_MAJOR_VERSION);
    int minorVersion = DeviceGet(DeviceInfo::API_MINOR_VERSION);
    if (majorVersion != LWN_API_MAJOR_VERSION || minorVersion < LWN_API_MINOR_VERSION) {
        NN_SDK_LOG("API version mismatch (application compiled with %d.%d, driver reports %d.%d).\n",
                LWN_API_MAJOR_VERSION, LWN_API_MINOR_VERSION, majorVersion, minorVersion);
    }
    NN_LOG("API version is compatible (application compiled with %d.%d, driver reports %d.%d).\n",
            LWN_API_MAJOR_VERSION, LWN_API_MINOR_VERSION, majorVersion, minorVersion);

    //--------------------------------------------------------------------------------
    // Initialize GPU queue, memory pool and command buffer builder.
    //--------------------------------------------------------------------------------
    MemoryPoolBuilder textureBuffer, commandBuffer, texGPUPoolBuilder;

    // Create a memory pool.
    commandBuffer.SetDevice(&device)
        .SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_CACHED)
        .SetStorage(s_poolStorage, sizeof(s_poolStorage));
    pool.Initialize(&commandBuffer);

    // Init GPU queue.
    QueueBuilder qb;
    qb.SetDevice(&device)
      .SetDefaults();
    queue.Initialize(&qb);

    // Create a memory pool to load the texture format into memory.
    textureBuffer.SetDevice(&device)
        .SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_CACHED)
        .SetStorage(texBuf_poolStorage, sizeof(texBuf_poolStorage));
    textureBufferPool.Initialize(&textureBuffer);

    // Create a memory pool for texture
    texGPUPoolBuilder.SetDevice(&device)
        .SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_CACHED| MemoryPoolFlags::GPU_CACHED) 
        .SetStorage(t_poolStorage, sizeof(t_poolStorage));
    poolGPU.Initialize(&texGPUPoolBuilder);
}

template<class T, class B>
struct ScopedLwnObject : public T
{
    explicit ScopedLwnObject(B &bld) : T()
    {
        this->Initialize(&bld);
    }

    ~ScopedLwnObject()
    {
        this->Finalize();
    }
};

int runTextureRest(Format lwnFormat, LWarray_format lwdaFormat, short numColorComponents, short bytesPerColorComponent, unsigned short texWidth, unsigned short texHeight, unsigned long long GpuPoolOffset)
{
    LWresult status;
    LWarray dImgArray = NULL;
    LWmodule mod = 0;
    LWfunction func = 0;
    LWtexref texref;
    LWevent pEvent;
    LWdeviceptr dstDevice = 0;
    size_t pitch;
    TextureView texView;
    LWDA_MEMCPY2D desc_copy = {0};

    BufferAddress poolGpuAddress;
    unsigned long poolOffset = 0;
    short formatPixelSize=numColorComponents*bytesPerColorComponent;
    short lwdaPixelLen;

    // Get buffer addresses
    poolGpuAddress = pool.GetBufferAddress();
    poolGpuAddress = textureBufferPool.GetBufferAddress();

    // Create a command buffer for GPU commands.
    ScopedLwnObject<CommandBuffer, Device> cmd(device);

    cmd.AddControlMemory(s_controlMemory, sizeof(s_controlMemory));
    poolOffset = align(poolOffset, (unsigned long)DeviceGet(DeviceInfo::COMMAND_BUFFER_COMMAND_ALIGNMENT));
    cmd.AddCommandMemory(&pool, poolOffset, COMMAND_MEMORY);
    commandMemoryBegin = poolOffset;
    poolOffset += COMMAND_MEMORY;

    // Begin recording the program init commands that will follow.
    cmd.BeginRecording();

    BufferBuilder bb;
    bb.SetDevice(&device);
    bb.SetDefaults();
    bb.SetStorage(&textureBufferPool, 0, formatPixelSize * texHeight * texWidth);

    ScopedLwnObject<Buffer, BufferBuilder> buffer(bb);

    ScopedLwnObject<Sync, Device> sync(device);

    //--------------------------------------------------------------------------------
    // Generate and upload the texure format into buffer and copy it into lwn texture.
    //--------------------------------------------------------------------------------

    unsigned char *lwnBufferPtr;
    lwnBufferPtr = (unsigned char*)buffer.Map();
    generateTextureData(texdata, texWidth, texHeight, formatPixelSize);
    CopyRegion copyRegion = { 0, 0, 0, texWidth, texHeight, 1 };
    memcpy(lwnBufferPtr, texdata, texWidth * texHeight * formatPixelSize);

#if DEBUG_LOG
    NN_LOG("Uploaded texture data:\n");
    printAsIntBuf(texdata, texWidth, texHeight, formatPixelSize);
#endif

    TextureBuilder tb;
    tb.SetDevice(&device)
        .SetDefaults()
        .SetTarget(TextureTarget::TARGET_2D)
        .SetFormat(lwnFormat)
        .SetSize2D(texWidth, texHeight);
    tb.SetStorage(&poolGPU, GpuPoolOffset);

    ScopedLwnObject<Texture, TextureBuilder> texture(tb);

    cmd.CopyBufferToTexture(buffer.GetAddress(), &texture, NULL, &copyRegion, 0);
    CommandHandle initCommands = cmd.EndRecording();
    queue.SubmitCommands(1, &initCommands);
    queue.FenceSync(&sync,SyncCondition::ALL_GPU_COMMANDS_COMPLETE , SyncFlagBits::FLUSH_FOR_CPU);
    poolGPU.IlwalidateMappedRange(GpuPoolOffset, texWidth * texHeight * formatPixelSize);
    queue.Finish();
    cmd.BeginRecording();

#if DEBUG_LOG
    NN_LOG("Texture in the lwn texture buffer:\n");
    printAsIntBuf(t_poolStorage + GpuPoolOffset, texWidth, texHeight, formatPixelSize);
#endif

    lwnBufferPtr = (unsigned char*)buffer.Map();
    for(int i=0;i < formatPixelSize*texHeight*texWidth;i++) {
        lwnBufferPtr[i] = 0;
    }
    cmd.CopyTextureToBuffer(&texture,NULL,&copyRegion,buffer.GetAddress(),0);
    initCommands = cmd.EndRecording();
    queue.SubmitCommands(1, &initCommands);
    queue.FenceSync(&sync,SyncCondition::ALL_GPU_COMMANDS_COMPLETE , SyncFlagBits::FLUSH_FOR_CPU);
    poolGPU.IlwalidateMappedRange(GpuPoolOffset,texWidth*texHeight*formatPixelSize);
    queue.Finish();

#if DEBUG_LOG
    NN_LOG("Copy back to lwn buffer:\n");
    printAsIntBuf(lwnBufferPtr, texWidth, texHeight, formatPixelSize);
#endif

    if(dataPatternMatch(texdata, lwnBufferPtr, texWidth*texHeight, formatPixelSize, formatPixelSize))
        NN_LOG("Copy back in lwn matches the texture pattern\n");
    else {
        NN_LOG("ERROR: Copy back in lwn do not matches the texture pattern\n");
        return -1;
    }

    //--------------------------------------------------------------------------------
    // Create a LWN pEvent from the lwnsync
    //--------------------------------------------------------------------------------
    status = lwEventCreateFromLWNSync(&pEvent, (LWNsync *)&sync, 0);
    if (status != LWDA_SUCCESS) {
        NN_LOG("Failed in lwEventCreateFromLWNSync %d\n", status);
        return -1;
    }

    //--------------------------------------------------------------------------------
    // Get LWCA array from a LWN texture and validate the incoming texture
    //--------------------------------------------------------------------------------

    texView.SetDefaults();
    memset(lwdaCopyBuffer, 0, texWidth * texHeight * sizeof(char)*formatPixelSize);
    status = lwLWNtextureGetArray ( &dImgArray, (LWNtexture*)&texture, (LWNtextureView*) &texView, LW_GRAPHICS_REGISTER_FLAGS_NONE);
    NN_LOG("lwLWNtextureGetArray status = %d\n", status);
    if (LWDA_SUCCESS != status) {
        NN_LOG("Failed to lwLWNtextureGetArray\n");
        return -1;
    }

    desc_copy.dstMemoryType = LW_MEMORYTYPE_HOST;
    desc_copy.dstHost = lwdaCopyBuffer;
    desc_copy.dstXInBytes = 0;
    desc_copy.dstY = 0;
    desc_copy.dstPitch = formatPixelSize*sizeof(char)*texWidth;
    desc_copy.srcMemoryType = LW_MEMORYTYPE_ARRAY;
    desc_copy.srcArray = dImgArray;
    desc_copy.srcXInBytes = 0;
    desc_copy.srcY = 0;
    desc_copy.WidthInBytes = formatPixelSize*texWidth*sizeof(char);
    desc_copy.Height = texHeight;

    status = lwStreamWaitEvent(NULL, pEvent, 0);
    NN_LOG("\nlwStreamWaitEvent status = %d\n", status);
    if (LWDA_SUCCESS != status) {
        NN_LOG("Failed in lwStreamWaitEvent\n");
        return -1;
    }

    status = lwMemcpy2DAsync(&desc_copy, 0);
    if (LWDA_SUCCESS != status) {
        NN_LOG("Failed to lwMemcpy2D, status= %d\n", status);
        return -1;
    }
    status = lwStreamSynchronize(NULL);
    NN_LOG("\nlwStreamSynchronize  status = %d\n", status);
    if (LWDA_SUCCESS != status) {
        NN_LOG("Failed in lwStreamSynchronized\n");
        return -1;
    }

#if DEBUG_LOG
    NN_LOG("The texture in LWCA-LWN interop image (at LWCA side):\n");
    printAsIntBuf(lwdaCopyBuffer, texWidth, texHeight, formatPixelSize);
#endif
    if(dataPatternMatch(texdata, lwdaCopyBuffer, texWidth*texHeight, formatPixelSize, formatPixelSize))
        NN_LOG("DMA coppy in LWCA matches the texture pattern\n");
    else {
        NN_LOG("ERROR: DMA coppy in LWCA do not matches the texture pattern\n");
        return -1;
    }

    //------------------------------------------------------------------------------------
    // Create a LWCA texture type using lwn texture and launch a kernel to verify the data
    //------------------------------------------------------------------------------------

    status = lwModuleLoadData(&mod, __lwbin_sm_53_texture_kernel);
    NN_LOG("\nlwModuleLoadData status = %d\n", status);
    if (LWDA_SUCCESS != status) {
        NN_LOG("Failed in lwModuleLoadData\n");
        return -1;
    }

    status = lwModuleGetFunction(&func, mod, "Tex2DUC");
    NN_LOG("lwModuleGetFunction status = %d\n", status);
    if (LWDA_SUCCESS != status) {
        NN_LOG("Failed in lwModuleGetFunction\n");
        return -1;
    }
    lwModuleGetTexRef(&texref, mod, "texUC");

    // LWCA array can only access pixel of size 4, 8 or 16. If the pixel size is smaller or not in that range. It upgrade the size to next level.
    lwdaPixelLen = (formatPixelSize<4)?4:formatPixelSize;

    if ( LWDA_SUCCESS != lwTexRefSetArray( texref, dImgArray, LW_TRSA_OVERRIDE_FORMAT ) )  { NN_LOG( "lwTexRefSetArray failed\n" ); return -1; }
    if ( LWDA_SUCCESS != lwTexRefSetFlags( texref, LW_TRSF_READ_AS_INTEGER) ) { NN_LOG( "lwTexRefSetFlags failed\n" ); return -1; }
    if ( LWDA_SUCCESS != lwTexRefSetAddressMode( texref, 0, LW_TR_ADDRESS_MODE_CLAMP) ) { NN_LOG( "lwTexRefSetAddressMode failed\n" ); return -1; }
    if ( LWDA_SUCCESS != lwTexRefSetAddressMode( texref, 1, LW_TR_ADDRESS_MODE_CLAMP) ) { NN_LOG( "lwTexRefSetAddressMode failed\n" ); return -1; }
    if ( LWDA_SUCCESS != lwTexRefSetFormat( texref, lwdaFormat, numColorComponents) ) { NN_LOG( "lwTexRefSetFormat failed\n" ); return -1; }
    if ( LWDA_SUCCESS != lwFuncSetBlockShape(func, 1, 1, 1) ) { NN_LOG( "lwFuncSetBlockShape failed\n" ); return -1; }
    if ( LWDA_SUCCESS != lwMemAllocPitch( &dstDevice, &pitch, texWidth*formatPixelSize, texHeight, lwdaPixelLen) ) { NN_LOG( "lwMemAllocPitch failed\n" ); return -1; }
    if ( LWDA_SUCCESS != lwMemsetD8( dstDevice, 0, texHeight*texWidth*formatPixelSize) ) { NN_LOG( "lwMemsetD8 failed\n" ); return -1; }

    {
        unsigned int offset = 0;
        offset += 0;                   lwParamSetv( func, offset, &dstDevice, sizeof(dstDevice));
        offset += sizeof(dstDevice); lwParamSeti( func, offset, formatPixelSize*texWidth);
        offset += sizeof(size_t);         lwParamSeti( func, offset, texWidth );
        offset += sizeof(int);         lwParamSeti( func, offset, 0 );
        offset += sizeof(int);         lwParamSetf( func, offset, (float) 0 );
        offset += sizeof(float);       lwParamSetf( func, offset, (float) 0 );
        offset += sizeof(float);       lwParamSeti( func, offset, 0 );
        offset += sizeof(int);         lwParamSetSize( func, offset );
    }
    if (LWDA_SUCCESS != lwParamSetTexRef( func, LW_PARAM_TR_DEFAULT, texref )){NN_LOG("Failed set texref\n");return -1;}

    if ( LWDA_SUCCESS != lwLaunchGrid(func, texHeight, 1) ) {
        NN_LOG( "lwLaunchGrid failed \n");
        return -1;
    }
    memset(lwdaCopyBuffer, 0, texWidth*texHeight*formatPixelSize);
    lwMemcpyDtoH(lwdaCopyBuffer,dstDevice,lwdaPixelLen*texWidth*texHeight);
    status = lwStreamSynchronize(NULL);

#if DEBUG_LOG
    NN_LOG("The texture seen by LWCA kernel:\n");
    printAsIntBuf(lwdaCopyBuffer, texWidth, texHeight, lwdaPixelLen);
#endif

    if(dataPatternMatch(texdata, lwdaCopyBuffer, texWidth*texHeight, formatPixelSize, lwdaPixelLen))
        NN_LOG("Kernel access passed!\n");
    else {
        NN_LOG("ERROR: Kernel access failed!\n");
        return -1;
    }
    status = lwEventDestroy(pEvent);
    if (LWDA_SUCCESS != status) {
       NN_LOG("Event Destroy Failed %d\n", status);
       return -1;
    }

    return 0;
}

int initLwda()
{
    LWresult status;

    //--------------------------------------------------------------------------------
    // Initilize LWCA driver and Create LWCA context
    //--------------------------------------------------------------------------------

    status = lwInit(0);
    if (LWDA_SUCCESS != status) {
        NN_LOG("Failed in lwInit\n");
        goto Error;
    }

    status = lwDeviceGet(&dev, 0);
    NN_LOG("lwDeviceGet status = %d\n", status);
    if (LWDA_SUCCESS != status) {
        NN_LOG("Failed to lwDeviceGet\n");
        goto Error;
    }

    status = lwCtxCreate(&ctx, 0, dev);
    NN_LOG("lwCtxCreate status = %d\n", status);
    if (LWDA_SUCCESS != status) {
        NN_LOG("Failed to lwCtxCreate\n");
        goto Error;
    }

    return 0;

Error:
    return -1;

}

typedef struct testData_t {
    Format lwnFormat;
    LWarray_format    lwdaFormat;
    short  numColorComponents;
    short  bytesPerColorComponent;
    short  texWidth;
    short  texHeight;
    unsigned long long offsetInPool;
}testData;

testData test_list[] = {
    { Format::RGBA8,   LW_AD_FORMAT_UNSIGNED_INT8, 4, 1, 32,  16,  0        },
    { Format::R8,      LW_AD_FORMAT_UNSIGNED_INT8, 1, 1, 32,  16,  0        },
};

namespace {

    const int firmwareMemorySize = 8 * 1024 * 1024;
    char g_FirmwareMemory[firmwareMemorySize] __attribute__((aligned(4096)));
    const int lwdaHeapSize = 512 * 1024 * 1024;
    char g_lwdaHeapBuffer[lwdaHeapSize];
    nn::mem::StandardAllocator  g_LwdaAllocator(g_lwdaHeapBuffer, sizeof(g_lwdaHeapBuffer));

    void *lwdaAllocateCallback(size_t size, size_t align, void *userPtr)
    {
        void  *address = NULL;

        address = g_LwdaAllocator.Allocate(size, align);
        if (!address)
        {
            NN_LOG("Failed to allocate memory.\n");
        }

        return (void *)(address);
    }

    void lwdaNNFreeCallback(void *address, void *userPtr)
    {
        g_LwdaAllocator.Free(address);
    }

    void* lwdaNNReallocateCallback(void *addr, size_t newSize, void *userPtr)
    {
        void  *address = NULL;

        address = g_LwdaAllocator.Reallocate(addr, newSize);

        if (!address)
        {
            NN_LOG("Failed to allocate memory\n");
        }

        return (void *)(address);
    }
}

extern "C" void nnMain()
{
    LWresult status;

    status = lwNNSetAllocator(lwdaAllocateCallback, lwdaNNFreeCallback, lwdaNNReallocateCallback, NULL);
    NN_LOG("lwNNSetAllocator() status = %d\n", status);
    if (LWDA_SUCCESS != status) {
        NN_LOG("Failed in lwNNSetAllocator\n");
        return;
    }

    lw::SetGraphicsAllocator(lwdaAllocateCallback, lwdaNNFreeCallback, lwdaNNReallocateCallback ,NULL);
    lw::SetGraphicsDevtoolsAllocator(lwdaAllocateCallback, lwdaNNFreeCallback, lwdaNNReallocateCallback ,NULL);
    lw::InitializeGraphics(g_FirmwareMemory, sizeof(g_FirmwareMemory));

    struct CleanStaticLwn
    {
        ~CleanStaticLwn()
        {
            pool.Finalize();
            poolGPU.Finalize();
            textureBufferPool.Finalize();
            queue.Finalize();
            device.Finalize();
        }
    } finalizer;

    if(initLwda() != 0)
        goto Error;

    initLwn();

    for(unsigned int index=0; index<sizeof(test_list)/sizeof(testData); index++) {
        if(runTextureRest(
           test_list[index].lwnFormat,
           test_list[index].lwdaFormat,
           test_list[index].numColorComponents,
           test_list[index].bytesPerColorComponent,
           test_list[index].texWidth,
           test_list[index].texHeight,
           test_list[index].offsetInPool) != 0) {

            NN_LOG("Test index %d fialed!\n", index);
            goto Error;
        }
        else
            NN_LOG("Test index %d passed!\n", index);
      
    }

    NN_LOG("\n\n\t\t&&&& lwn_lwda_texture_interop test PASSED!!\n");
    return;

Error:
    NN_LOG("\n\n\t\t&&&& lwn_lwda_texture_interop test FAILED!!\n");
}


