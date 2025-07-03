/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
// includes
//
#include "print.h"
#include "lwclass.h"

//-----------------------------------------------------
// _printBuffer - internal routine
//
//-----------------------------------------------------
VOID printBuffer(char *buffer, U032 length, LwU64 offset, U008 size)
{
    printBufferEx(buffer, length, offset, size, 0x00);
}

//-----------------------------------------------------
// _printBufferEx - internal routine
//
//-----------------------------------------------------
VOID printBufferEx(char *buffer, U032 length, LwU64 offset, U008 size, LwU64 base64)
{
    U032 i;
    U032 j;
    U032 *tmp32;
    U016 *tmp16;
    U032 left;

    if (size == 1) {
        for (i = 0; i < length/16; i++) {
            if (osCheckControlC())
                return;

            dprintf(LwU40_FMT ": ", base64 + i*16 + offset);
            for (j = 0; j < 16; j++) {
                dprintf("%02x ", buffer[i*16 + j] & 0xFF);
            }
            for (j = 0; j < 16; j++) {
                if (isprint(buffer[i*16 + j])) {
                    dprintf("%c", buffer[i*16 + j] & 0xFF);
                }
                else {
                    dprintf(".");
                }
            }
            dprintf("\n");
        }
        i *= 16;
        if (length % 16 != 0) {
            j = i;
            left = length % 16;

            dprintf(LwU40_FMT ": ", base64 + i + offset);
            for (i = 0; i < left; i++) {
                dprintf("%02x ", buffer[j+i] & 0xFF);
            }
            for (i = 0; i < (16 - left); i++) {
                dprintf("   ");
            }

            for (i = 0; i < left; i++) {
                if (isprint(buffer[j+i])) {
                    dprintf("%c", buffer[j+i] & 0xFF);
                }
                else {
                    dprintf(".");
                }
            }
            dprintf("\n");
        }
    }
    else if (size == 2) {
        for (i = 0; i < length/16; i++) {
            if (osCheckControlC())
                return;

            dprintf(LwU40_FMT ": ", base64 + i*16 + offset);
            tmp16 = (U016*)&(buffer[i*16]);
            for (j = 0; j < 8; j++) {
                dprintf("%04x ", tmp16[j] & 0xFFFF);
            }
            dprintf("\n");
        }
        i *= 16;
        if (length % 16 != 0) {
            dprintf(LwU40_FMT ": ", base64 + i + offset);
            tmp16 = (U016*)&(buffer[i]);
            left = ((length - i) / 2);
            for (i = 0; i < left; i++) {
                dprintf("%04x ", tmp16[i] & 0xFFFF);
            }
            dprintf("\n");
        }
    }
    else {
        for (i = 0; i < length/16; i++) {
            if (osCheckControlC())
                return;

            dprintf(LwU40_FMT ": ", base64 + i*16 + offset);
            tmp32 = (U032*)&(buffer[i*16]);
            for (j = 0; j < 4; j++) {
                dprintf("%08x ", tmp32[j]);
            }
            dprintf("\n");
        }
        i *= 16;
        if (length % 16 != 0) {
            dprintf(LwU40_FMT ": ", base64 + i + offset);
            tmp32 = (U032*)&(buffer[i]);
            left = ((length - i) / 4);
            for (i = 0; i < left; i++) {
                dprintf("%08x ", tmp32[i]);
            }
            dprintf("\n");
        }
    }
}

//-----------------------------------------------------
// printDataByType
// + print sizeInBytes bytes from addr.
//-----------------------------------------------------
VOID printDataByType(PhysAddr addr, U032 sizeInBytes, MEM_TYPE memoryType, U032 numColumns)
{
    U032 status;
    U032 i = 0;
    U032 bytesRead;
    U032 *buffer = 0;

    // 1 dw at least
    if (sizeInBytes < sizeof(U032))
        sizeInBytes = sizeof(U032);

    buffer = (U032*) malloc(sizeInBytes);

    if (buffer == NULL)
    {
        dprintf("lw: printDataByType - alloc failed!\n");
        return;
    }

    //
    // read the data in a buffer and print it out.
    //
    status = osReadMemByType(addr, buffer, sizeInBytes, &bytesRead, memoryType);

    if (status != LW_OK)
    {
        if (buffer) free(buffer);
        dprintf("lw: printDataByType - osReadMemByType failed!\n");
        return;
    }

    // XXX - make sure we're aligned
    for (i = 0; i < (sizeInBytes / 4); i++)
    {
        if ((i % numColumns) == 0)
        {
            PhysAddr offsetPrint = addr;

            if (memoryType == REGISTER)
            {
                offsetPrint += lwBar0;
            }
            dprintf("#" PhysAddr_FMT " ", offsetPrint);
            addr += numColumns*sizeof(U032);
        }

        dprintf("%08x ", buffer[i]);

        if (((i+1) % numColumns) == 0)
        {
            if (i != 0)
                dprintf("\n");
        }
    }
    dprintf("\n");

    if (buffer) free(buffer);
}

//-----------------------------------------------------
// printData
// + print sizeInBytes bytes from Physical address addr
//-----------------------------------------------------
VOID printData(PhysAddr addr, U032 sizeInBytes)
{
    printDataByType(addr, sizeInBytes, SYSTEM_PHYS, 4);
}

//-----------------------------------------------------
// printDataColumns
// + print sizeInBytes bytes from Physical address addr
//-----------------------------------------------------
VOID printDataColumns(PhysAddr addr, U032 sizeInBytes, U032 numColumns)
{
    printDataByType(addr, sizeInBytes, SYSTEM_PHYS, numColumns);
}

//-----------------------------------------------------
// printClassName
//
//-----------------------------------------------------
static U008 LwClassStrings[][33]={"LW01_NULL_OBJECT                ", /*    (0x00000000) */
                                  "LW01_CLASS                      ", /*    (0x00000001) */
                                  "LW01_CONTEXT_DMA_FROM_MEMORY    ", /*    (0x00000002) */
                                  "LW01_TIMER                      ", /*    (0x00000004) */
                                  "LW01_EVENT                      ", /*    (0x00000005) */
                                  "LW01_CONTEXT_ORDINAL            ", /*    (0x00000006) */
                                  "LW01_PATCHCORD_GAME_PORT        ", /*    (0x00000007) */
                                  "UNDEFINED_CLASS_Lw008           ", /*                 */
                                  "UNDEFINED_CLASS_Lw009           ", /*                 */
                                  "UNDEFINED_CLASS_Lw00a           ", /*                 */
                                  "UNDEFINED_CLASS_Lw00b           ", /*                 */
                                  "UNDEFINED_CLASS_Lw00c           ", /*                 */
                                  "UNDEFINED_CLASS_Lw00d           ", /*                 */
                                  "UNDEFINED_CLASS_Lw00e           ", /*                 */
                                  "UNDEFINED_CLASS_Lw00f           ", /*                 */
                                  "UNDEFINED_CLASS_Lw010           ", /*                 */
                                  "UNDEFINED_CLASS_Lw011           ", /*                 */
                                  "LW01_BETA_SOLID                 ", /*    (0x00000012) */
                                  "UNDEFINED_CLASS_Lw013           ", /*                 */
                                  "UNDEFINED_CLASS_Lw014           ", /*                 */
                                  "UNDEFINED_CLASS_Lw015           ", /*                 */
                                  "UNDEFINED_CLASS_Lw016           ", /*                 */
                                  "LW01_IMAGE_SOLID                ", /*    (0x00000017) */
                                  "LW01_IMAGE_PATTERN              ", /*    (0x00000018) */
                                  "LW01_CONTEXT_CLIP_RECTANGLE     ", /*    (0x00000019) */
                                  "UNDEFINED_CLASS_Lw01a           ", /*                 */
                                  "UNDEFINED_CLASS_Lw01b           ", /*                 */
                                  "LW01_RENDER_SOLID_LIN           ", /*    (0x0000001C) */
                                  "LW01_RENDER_SOLID_TRIANGLE      ", /*    (0x0000001D) */
                                  "LW01_RENDER_SOLID_RECTANGLE     ", /*    (0x0000001E) */
                                  "LW01_IMAGE_BLIT                 ", /*    (0x0000001F) */
                                  "UNDEFINED_CLASS_Lw020           ", /*                 */
                                  "LW01_IMAGE_FROM_CPU             ", /*    (0x00000021) */
                                  "UNDEFINED_CLASS_Lw022           ", /*                 */
                                  "UNDEFINED_CLASS_Lw023           ", /*                 */
                                  "UNDEFINED_CLASS_Lw024           ", /*                 */
                                  "UNDEFINED_CLASS_Lw025           ", /*                 */
                                  "UNDEFINED_CLASS_Lw026           ", /*                 */
                                  "UNDEFINED_CLASS_Lw027           ", /*                 */
                                  "UNDEFINED_CLASS_Lw028           ", /*                 */
                                  "UNDEFINED_CLASS_Lw029           ", /*                 */
                                  "UNDEFINED_CLASS_Lw02a           ", /*                 */
                                  "UNDEFINED_CLASS_Lw02b           ", /*                 */
                                  "UNDEFINED_CLASS_Lw02c           ", /*                 */
                                  "UNDEFINED_CLASS_Lw02d           ", /*                 */
                                  "UNDEFINED_CLASS_Lw02e           ", /*                 */
                                  "UNDEFINED_CLASS_Lw02f           ", /*                 */
                                  "LW01_NULL                       ", /*    (0x00000030) */
                                  "UNDEFINED_CLASS_Lw031           ", /*                 */
                                  "UNDEFINED_CLASS_Lw032           ", /*                 */
                                  "UNDEFINED_CLASS_Lw033           ", /*                 */
                                  "UNDEFINED_CLASS_Lw034           ", /*                 */
                                  "UNDEFINED_CLASS_Lw035           ", /*                 */
                                  "LW03_STRETCHED_IMAGE_FROM_CPU   ", /*    (0x00000036) */
                                  "LW03_SCALED_IMAGE_FROM_MEMORY   ", /*    (0x00000037) */
                                  "LW04_DVD_SUBPICTURE             ", /*    (0x00000038) */
                                  "LW03_MEMORY_TO_MEMORY_FORMAT    ", /*    (0x00000039) */
                                  "UNDEFINED_CLASS_Lw03a           ", /*                 */
                                  "UNDEFINED_CLASS_Lw03b           ", /*                 */
                                  "UNDEFINED_CLASS_Lw03c           ", /*                 */
                                  "LW01_CONTEXT_ERROR_TO_MEMORY    ", /*    (0x0000003E) */
                                  "LW01_MEMORY_PRIVILEGED          ", /*    (0x0000003F) */
                                  "LW01_MEMORY_USER                ", /*    (0x00000040) */
                                  "UNDEFINED_CLASS_Lw041           ", /*                 */
                                  "LW04_CONTEXT_SURFACES_2D        ", /*    (0x00000042) */
                                  "LW03_CONTEXT_ROP                ", /*    (0x00000043) */
                                  "LW04_CONTEXT_PATTERN            ", /*    (0x00000044) */
                                  "UNDEFINED_CLASS_Lw045           ", /*                 */
                                  "LW04_VIDEO_LUT_LWRSOR_DAC       ", /*    (0x00000046) */
                                  "LW04_VIDEO_OVERLAY              ", /*    (0x00000047) */
                                  "LW03_DX3_TEXTURED_TRIANGLE      ", /*    (0x00000048) */
                                  "LW05_VIDEO_LUT_LWRSOR_DAC       ", /*    (0x00000049) */
                                  "LW04_GDI_RECTANGLE_TEXT         ", /*    (0x0000004A) */
                                  "LW03_GDI_RECTANGLE_TEXT         ", /*    (0x0000004B) */
                                  "UNDEFINED_CLASS_Lw04c           ", /*                 */
                                  "LW03_EXTERNAL_VIDEO_DECODER     ", /*    (0x0000004D) */
                                  "LW03_EXTERNAL_VIDEO_DECOMPRESSOR", /*    (0x0000004E) */
                                  "LW01_EXTERNAL_PARALLEL_BUS      ", /*    (0x0000004F) */
                                  "LW03_EXTERNAL_MONITOR_BUS       ", /*    (0x00000050) */
                                  "LW03_EXTERNAL_SERIAL_BUS        ", /*    (0x00000051) */
                                  "LW04_CONTEXT_SURFACE_SWIZZLED   ", /*    (0x00000052) */
                                  "LW04_CONTEXT_SURFACES_3D        ", /*    (0x00000053) */
                                  "LW04_DX5_TEXTURED_TRIANGLE      ", /*    (0x00000054) */
                                  "LW04_DX6_MULTI_TEXTURE_TRIANGLE ", /*    (0x00000055) */
                                  "LW10_CELSIUS_PRIMITIVE          ", /*    (0x00000056) */
                                  "LW04_CONTEXT_COLOR_KEY          ", /*    (0x00000057) */
                                  "LW03_CONTEXT_SURFACE_0          ", /*    (0x00000058) */
                                  "LW03_CONTEXT_SURFACE_1          ", /*    (0x00000059) */
                                  "LW03_CONTEXT_SURFACE_2          ", /*    (0x0000005A) */
                                  "LW03_CONTEXT_SURFACE_3          ", /*    (0x0000005B) */
                                  "LW04_RENDER_SOLID_LIN           ", /*    (0x0000005C) */
                                  "LW04_RENDER_SOLID_TRIANGLE      ", /*    (0x0000005D) */
                                  "LW04_RENDER_SOLID_RECTANGLE     ", /*    (0x0000005E) */
                                  "LW04_IMAGE_BLIT                 ", /*    (0x0000005F) */
                                  "LW04_INDEXED_IMAGE_FROM_CPU     ", /*    (0x00000060) */
                                  "LW04_IMAGE_FROM_CPU             ", /*    (0x00000061) */
                                  "LW10_CONTEXT_SURFACES_2D        ", /*    (0x00000062) */
                                  "LW05_SCALED_IMAGE_FROM_MEMORY   ", /*    (0x00000063) */
                                  "LW05_INDEXED_IMAGE_FROM_CPU     ", /*    (0x00000064) */
                                  "LW05_IMAGE_FROM_CPU             ", /*    (0x00000065) */
                                  "LW05_STRETCHED_IMAGE_FROM_CPU   ", /*    (0x00000066) */
                                  "LW10_VIDEO_LUT_LWRSOR_DAC       ", /*    (0x00000067) */
                                  "LW01_CHANNEL_PIO                ", /*    (0x00000068) */
                                  "LW02_CHANNEL_PIO                ", /*    (0x00000069) */
                                  "LW03_CHANNEL_PIO                ", /*    (0x0000006A) */
                                  "LW03_CHANNEL_DMA                ", /*    (0x0000006B) */
                                  "LW04_CHANNEL_DMA                ", /*    (0x0000006C) */
                                  "LW04_CHANNEL_PIO                ", /*    (0x0000006D) */
                                  "LW10_CHANNEL_DMA                ", /*    (0x0000006E) */
                                  "UNDEFINED_CLASS_Lw06f           ", /*                 */
                                  "UNDEFINED_CLASS_Lw070           ", /*                 */
                                  "UNDEFINED_CLASS_Lw071           ", /*                 */
                                  "LW04_CONTEXT_BETA               ", /*    (0x00000072) */
                                  "UNDEFINED_CLASS_Lw073           ", /*                 */
                                  "UNDEFINED_CLASS_Lw074           ", /*                 */
                                  "UNDEFINED_CLASS_Lw075           ", /*                 */
                                  "LW04_STRETCHED_IMAGE_FROM_CPU   ", /*    (0x00000076) */
                                  "LW04_SCALED_IMAGE_FROM_MEMORY   ", /*    (0x00000077) */
                                  "UNDEFINED_CLASS_Lw078           ", /*                 */
                                  "UNDEFINED_CLASS_Lw079           ", /*                 */
                                  "LW10_VIDEO_OVERLAY              ", /*    (0x0000007A) */
                                  "LW10_TEXTURE_FROM_CPU           ", /*    (0x0000007B) */
                                  "LW15_VIDEO_LUT_LWRSOR_DAC       ", /*    (0x0000007C) */
                                  "UNDEFINED_CLASS_Lw07d           ", /*                 */
                                  "UNDEFINED_CLASS_Lw07e           ", /*                 */
                                  "UNDEFINED_CLASS_Lw07f           ", /*                 */
                                  "LW01_DEVICE_0                   ", /*    (0x00000080) */
                                  "LW01_DEVICE_1                   ", /*    (0x00000081) */
                                  "LW01_DEVICE_2                   ", /*    (0x00000082) */
                                  "LW01_DEVICE_3                   ", /*    (0x00000083) */
                                  "LW01_DEVICE_4                   ", /*    (0x00000084) */
                                  "LW01_DEVICE_5                   ", /*    (0x00000085) */
                                  "LW01_DEVICE_6                   ", /*    (0x00000086) */
                                  "LW01_DEVICE_7                   ", /*    (0x00000087) */
                                  "LW10_DVD_SUBPICTURE             ", /*    (0x00000088) */
                                  "LW10_SCALED_IMAGE_FROM_MEMORY   ", /*    (0x00000089) */
                                  "LW10_IMAGE_FROM_CPU             ", /*    (0x0000008A) */
                                  "UNDEFINED_CLASS_Lw08b           ", /*                 */
                                  "UNDEFINED_CLASS_Lw08c           ", /*                 */
                                  "UNDEFINED_CLASS_Lw08d           ", /*                 */
                                  "UNDEFINED_CLASS_Lw08e           ", /*                 */
                                  "UNDEFINED_CLASS_Lw08f           ", /*                 */
                                  "UNDEFINED_CLASS_Lw090           ", /*                 */
                                  "UNDEFINED_CLASS_Lw091           ", /*                 */
                                  "UNDEFINED_CLASS_Lw092           ", /*                 */
                                  "LW10_CONTEXT_SURFACES_3D        ", /*    (0x00000093) */
                                  "LW10_DX5_TEXTURED_TRIANGLE      ", /*    (0x00000094) */
                                  "LW10_DX6_MULTI_TEXTURE_TRIANGLE ", /*    (0x00000095) */
                                  "LW15_CELSIUS_PRIMITIVE          ", /*    (0x00000096) */
                                  "LW20_KELVIN_PRIMITIVE           ", /*    (0x00000097) */
                                  "UNDEFINED_CLASS_Lw098           ", /*                 */
                                  "LW17_CELSIUS_PRIMITIVE_HW       ", /*    (0x00000099) */
                                  "UNDEFINED_CLASS_Lw09a           ", /*                 */
                                  "UNDEFINED_CLASS_Lw09b           ", /*                 */
                                  "UNDEFINED_CLASS_Lw09c           ", /*                 */
                                  "UNDEFINED_CLASS_Lw09d           ", /*                 */
                                  "UNDEFINED_CLASS_Lw09e           ", /*                 */
                                  "LW15_IMAGE_BLIT                 ", /*    (0x0000009F) */
                                  "UNDEFINED_CLASS                 "
                                  };                                  /*    (>0x9F) */

U032 printClassName(U032 classNum)
{
    U032 status = LW_OK;

    if (classNum <= MAX_CLASS_NUMBER_SUPPORTED)
    {
        dprintf(" - %s\n", LwClassStrings[classNum]);
    }
    else
    {
        dprintf(" - ");
        switch (classNum)
        {
        case 0xff6:
            dprintf("LW_VIDEO_COLOR_KEY\n");
            break;
        case 0xff7:
            dprintf("LW_VIDEO_SCALER\n");
            break;
        case 0xff8:
            dprintf("LW_VIDEO_FROM_MEMORY\n");
            break;
        case 0xff9:
            dprintf("LW_VIDEO_COLORMAP\n");
            break;
        case 0xffa:
            dprintf("LW_VIDEO_SINK\n");
            break;
        case 0xffb:
            dprintf("LW_PATCHCORD_VIDEO\n");
            break;
        case 0xfff:
            dprintf("LW_CLASS\n");
            break;
        case 0x1774:
            dprintf("LW17_CHANNEL_MPEG\n");
            break;
        case 0x177a:
            dprintf("LW17_VIDEO_OVERLAY\n");
            break;
        case 0x317a:
            dprintf("LW31_VIDEO_OVERLAY\n");
            break;
        case 0x25a0:
            dprintf("LW25_MULTICHIP_VIDEO_SPLIT\n");
            break;
        case 0x177c:
            dprintf("LW17_VIDEO_LUT_LWRSOR_DAC\n");
            break;
        case 0x207c:
            dprintf("LW20_VIDEO_LUT_LWRSOR_DAC\n");
            break;
        case 0x307c:
            dprintf("LW30_VIDEO_LUT_LWRSOR_DAC\n");
            break;
        case 0x177e:
            dprintf("LW17_HOST_BLOAT\n");
            break;
        case 0x177f:
            dprintf("LW17_HOST_RELOCATION\n");
            break;
        case 0x1796:
            dprintf("LW17_CELSIUS_PRIMITIVE\n");
            break;
        case 0x1189:
            dprintf("LW11_SCALED_IMAGE_FROM_MEMORY\n");
            break;
        case 0x1196:
            dprintf("LW11_CELSIUS_PRIMITIVE\n");
            break;
        case 0x205f:
            dprintf("LW20_IMAGE_BLIT\n");
            break;
        case 0x206e:
            dprintf("LW20_CHANNEL_DMA\n");
            break;
        case 0x597:
        case 0x2597:
            dprintf("LW25_KELVIN_PRIMITIVE\n");
            break;
        case 0x35c:
        case 0x305c:
            dprintf("LW30_RENDER_SOLID_NT_LIN\n");
            break;
        case 0x362:
        case 0x3062:
            dprintf("LW30_CONTEXT_SURFACES_2D\n");
            break;
        case 0x364:
        case 0x3064:
            dprintf("LW30_INDEXED_IMAGE_FROM_CPU\n");
            break;
        case 0x366:
        case 0x3066:
            dprintf("LW30_STRETCHED_IMAGE_FROM_CPU\n");
            break;
        case 0x37a:
        case 0x307a:
            dprintf("LW30_VIDEO_OVERLAY\n");
            break;
        case 0x37b:
        case 0x307b:
            dprintf("LW30_TEXTURE_FROM_CPU\n");
            break;
        case 0x389:
        case 0x3089:
            dprintf("LW30_SCALED_IMAGE_FROM_MEMORY\n");
            break;
        case 0x38a:
        case 0x308a:
            dprintf("LW30_IMAGE_FROM_CPU\n");
            break;
        case 0x397:
        case 0x3097:
            dprintf("LW30_RANKINE_PRIMITIVE\n");
            break;
        case 0x697:
        case 0x3497:
            dprintf("LW34_RANKINE_PRIMITIVE\n");
            break;
        case 0x497:
        case 0x3597:
            dprintf("LW35_RANKINE_PRIMITIVE\n");
            break;
        case 0x357c:
            dprintf("LW35_VIDEO_LUT_LWRSOR_DAC\n");
            break;
        case 0x366e:
            dprintf("LW36_CHANNEL_DMA\n");
            break;
        case 0x406e:
            dprintf("LW40_CHANNEL_DMA\n");
            break;
        case 0x4075:
            dprintf("LW40_MOTION_SEARCH\n");
            break;
        case 0x4096:
            dprintf("LW40_RANKINE_PRIMITIVE\n");
            break;
        case 0x4097:
            dprintf("LW40_LWRIE_PRIMITIVE\n");
            break;
        case 0x4497:
            dprintf("LW44_LWRIE_PRIMITIVE\n");
            break;
        case 0x4176:
            dprintf("LW41_VIDEOPROCESSOR\n");
            break;
        case 0x502D:
            dprintf("LW50_TWOD_PRIMITIVE\n");
            break;
        case 0x5039:
            dprintf("LW50_MEMORY_TO_MEMORY_FORMAT\n");
            break;
        case 0x5062:
             dprintf("LW50_CONTEXT_SURFACES_2D\n");
             break;
        case 0x506E:
             dprintf("LW50_CHANNEL_DMA\n");
             break;
        case 0x506F:
             dprintf("LW50_GPFIFO_CHANNEL_DMA\n");
             break;
        case 0x5070:
             dprintf("LW50_DISPLAY\n");
             break;
        case 0x5071:
             dprintf("LW50_CAPTURE\n");
             break;
        case 0x5076:
             dprintf("LW50_VIDEO_PROCESSOR\n");
             break;
        case 0x5077:
             dprintf("LW50_CORE_CHANNEL_PIO\n");
             break;
        case 0x5078:
             dprintf("LW50_VIDEO_CHANNEL_PIO\n");
             break;
        case 0x5079:
             dprintf("LW50_AUDIO_CHANNEL_PIO\n");
             break;
        case 0x507A:
             dprintf("LW50_LWRSOR_CHANNEL_PIO\n");
             break;
        case 0x507B:
             dprintf("LW50_OVERLAY_IMM_CHANNEL_PIO\n");
             break;
        case 0x507C:
             dprintf("LW50_BASE_CHANNEL_DMA\n");
             break;
        case 0x507D:
             dprintf("LW50_CORE_CHANNEL_DMA\n");
             break;
        case 0x507E:
             dprintf("LW50_OVERLAY_CHANNEL_DMA\n");
             break;
        case 0x5089:
             dprintf("LW50_SCALED_IMAGE_FROM_MEMORY\n");
             break;
        case 0x5097:
             dprintf("LW50_TESLA_PRIMITIVE\n");
             break;
        case 0x50C0:
             dprintf("LW50_COMPUTE_PRIMITIVE\n");
             break;
        default:
            {
                // hacks for LW40 ctx dmas
                switch (classNum & 0xfff)
                {
                case 0x2:
                    dprintf("LW01_CONTEXT_DMA_FROM_MEMORY\n");
                    break;
                case 0x30:
                    dprintf("LW01_NULL\n");
                    break;
                default:
                    dprintf("Unknown Class\n");
                    status = LW_ERR_GENERIC;
                }
            }
        }
    }

    return status;
}
