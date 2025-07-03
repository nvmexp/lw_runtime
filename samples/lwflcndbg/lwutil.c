/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// vgupta@lwpu.com - Aug 18 2004
// FB register reading routines
//
//*****************************************************


#include "os.h" //for basic data types
#include "lwutil.h"

//----------------------------------------------------
// FB_RD_FIELD32
//
// - Reads value given a address and offsets like x= 2*32+20
// and y = 2*32+20
//
// - Running example
// x=2*32+16
// y=2*32+20
//
//----------------------------------------------------
U032 FB_RD_FIELD32
(
    U032 address,
    U032 x,
    U032 y
)
{
    U032 xoffset1, xoffset2;
    U032 yoffset1, yoffset2;
    U032 word;

    xoffset1 = x/32; //Running example: xoffset1 = 2
    xoffset2 = x%32; //Running example: xoffset2 = 16

    yoffset1 = y/32; //Running example: yoffset1 = 2
    yoffset2 = y%32; //Running example: yoffset2 = 20

    //
    // assert(y >= x)
    //
    if (!(yoffset2 >= xoffset2 && yoffset1 >=xoffset1 )) goto errorOut;

    //
    // Access spanning >1 dword not supported
    //
    if (xoffset1 != yoffset1) goto errorOut;
    if (yoffset2 - xoffset2 >=32 ) goto errorOut;

    //
    // Now read the word
    //
    word = FB_RD32(address+xoffset1*4);

    return (word>>xoffset2) & DRF_MASK((yoffset2):(xoffset2));

errorOut:
    dprintf("lw: FB_RD_FIELD not implemented for x = %x, y = %x.\n", x, y);
    return -1;
}

//----------------------------------------------------
// FB_RD_FIELD32_64
//
//----------------------------------------------------

U032 FB_RD_FIELD32_64
(
    LwU64 address,
    U032  x,
    U032  y
)
{
    //
    // 64BIT CAUTION: Should remove cast below when we completely transition to
    // 64-bit code
    //
    return FB_RD_FIELD32((U032)address, x, y);
}


//----------------------------------------------------
// DeviceIDToString
//
// - Gives the name of a board given the DeviceID
//
//----------------------------------------------------

VOID
DeviceIDToString
(
    U032 devid,
    char *name
)
{
    switch(devid)
    {
        case 0x0403:
            strcpy(name, "LWPU G84");
            break;
        case 0x014F:
            strcpy(name, "VdChip 6200");
            break;
        case 0x00F3:
            strcpy(name, "VdChip 6200");
            break;
        case 0x0221:
            strcpy(name, "VdChip 6200");
            break;
        case 0x0163:
            strcpy(name, "VdChip 6200 LE");
            break;
        case 0x0162:
            strcpy(name, "VdChip 6200SE TurboCache(TM)");
            break;
        case 0x0161:
            strcpy(name, "VdChip 6200 TurboCache(TM)");
            break;
        case 0x0160:
            strcpy(name, "VdChip 6500");
            break;
        case 0x0141:
            strcpy(name, "VdChip 6600");
            break;
        case 0x00F2:
            strcpy(name, "VdChip 6600");
            break;
        case 0x0140:
            strcpy(name, "VdChip 6600 GT");
            break;
        case 0x00F1:
            strcpy(name, "VdChip 6600 GT");
            break;
        case 0x0142:
            strcpy(name, "VdChip 6600 LE");
            break;
        case 0x00F4:
            strcpy(name, "VdChip 6600 LE");
            break;
        case 0x0143:
            strcpy(name, "VdChip 6600 VE");
            break;
        case 0x0147:
            strcpy(name, "VdChip 6700 XL");
            break;
        case 0x0041:
            strcpy(name, "VdChip 6800");
            break;
        case 0x00C1:
            strcpy(name, "VdChip 6800");
            break;
        case 0x0047:
            strcpy(name, "VdChip 6800 GS");
            break;
        case 0x00F6:
            strcpy(name, "VdChip 6800 GS");
            break;
        case 0x00C0:
            strcpy(name, "VdChip 6800 GS");
            break;
        case 0x0045:
            strcpy(name, "VdChip 6800 GT");
            break;
        case 0x00F9:
            strcpy(name, "VdChip 6800 Series GPU");
            break;
        case 0x00C2:
            strcpy(name, "VdChip 6800 LE");
            break;
        case 0x0040:
            strcpy(name, "VdChip 6800 Ultra");
            break;
        case 0x0043:
            strcpy(name, "VdChip 6800 XE");
            break;
        case 0x0048:
            strcpy(name, "VdChip 6800 XT");
            break;
        case 0x0218:
            strcpy(name, "VdChip 6800 XT");
            break;
        case 0x00C3:
            strcpy(name, "VdChip 6800 XT");
            break;
        case 0x01DF:
            strcpy(name, "VdChip 7300 GS");
            break;
        case 0x0393:
            strcpy(name, "VdChip 7300 GT");
            break;
        case 0x01D1:
            strcpy(name, "VdChip 7300 LE");
            break;
        case 0x01D3:
            strcpy(name, "VdChip 7300 SE");
            break;
        case 0x01DD:
            strcpy(name, "VdChip 7500 LE");
            break;
        case 0x0392:
            strcpy(name, "VdChip 7600 GS");
            break;
        case 0x02E1:
            strcpy(name, "VdChip 7600 GS");
            break;
        case 0x0391:
            strcpy(name, "VdChip 7600 GT");
            break;
        case 0x0394:
            strcpy(name, "VdChip 7600 LE");
            break;
        case 0x00F5:
            strcpy(name, "VdChip 7800 GS");
            break;
        case 0x0092:
            strcpy(name, "VdChip 7800 GT");
            break;
        case 0x0091:
            strcpy(name, "VdChip 7800 GTX");
            break;
        case 0x0291:
            strcpy(name, "VdChip 7900 GT/GTO");
            break;
        case 0x0290:
            strcpy(name, "VdChip 7900 GTX");
            break;
        case 0x0293:
            strcpy(name, "VdChip 7900 GX2");
            break;
        case 0x0294:
            strcpy(name, "VdChip 7950 GX2");
            break;
        case 0x0322:
            strcpy(name, "VdChip FX 5200");
            break;
        case 0x0321:
            strcpy(name, "VdChip FX 5200 Ultra");
            break;
        case 0x0323:
            strcpy(name, "VdChip FX 5200LE");
            break;
        case 0x0326:
            strcpy(name, "VdChip FX 5500");
            break;
        case 0x0312:
            strcpy(name, "VdChip FX 5600");
            break;
        case 0x0311:
            strcpy(name, "VdChip FX 5600 Ultra");
            break;
        case 0x0314:
            strcpy(name, "VdChip FX 5600XT");
            break;
        case 0x0342:
            strcpy(name, "VdChip FX 5700");
            break;
        case 0x0341:
            strcpy(name, "VdChip FX 5700 Ultra");
            break;
        case 0x0343:
            strcpy(name, "VdChip FX 5700LE");
            break;
        case 0x0344:
            strcpy(name, "VdChip FX 5700VE");
            break;
        case 0x0302:
            strcpy(name, "VdChip FX 5800");
            break;
        case 0x0301:
            strcpy(name, "VdChip FX 5800 Ultra");
            break;
        case 0x0331:
            strcpy(name, "VdChip FX 5900");
            break;
        case 0x0330:
            strcpy(name, "VdChip FX 5900 Ultra");
            break;
        case 0x0333:
            strcpy(name, "VdChip FX 5950 Ultra");
            break;
        case 0x0324:
            strcpy(name, "VdChip FX Go5200 64M");
            break;
        case 0x031A:
            strcpy(name, "VdChip FX Go5600");
            break;
        case 0x0347:
            strcpy(name, "VdChip FX Go5700");
            break;
        case 0x0167:
            strcpy(name, "VdChip Go 6200/6400");
            break;
        case 0x0168:
            strcpy(name, "VdChip Go 6200/6400");
            break;
        case 0x0148:
            strcpy(name, "VdChip Go 6600");
            break;
        case 0x00c8:
            strcpy(name, "VdChip Go 6800");
            break;
        case 0x00c9:
            strcpy(name, "VdChip Go 6800 Ultra");
            break;
        case 0x0098:
            strcpy(name, "VdChip Go 7800");
            break;
        case 0x0099:
            strcpy(name, "VdChip Go 7800 GTX");
            break;
        case 0x0298:
            strcpy(name, "VdChip Go 7900 GS");
            break;
        case 0x0299:
            strcpy(name, "VdChip Go 7900 GTX");
            break;
        case 0x0185:
            strcpy(name, "VdChip MX 4000");
            break;
        case 0x00FA:
            strcpy(name, "VdChip PCX 5750");
            break;
        case 0x00FB:
            strcpy(name, "VdChip PCX 5900");
            break;
        case 0x0110:
            strcpy(name, "GeForce2 MX/MX 400");
            break;
        case 0x0111:
            strcpy(name, "GeForce2 MX200");
            break;
        case 0x0200:
            strcpy(name, "GeForce3");
            break;
        case 0x0201:
            strcpy(name, "GeForce3 Ti200");
            break;
        case 0x0202:
            strcpy(name, "GeForce3 Ti500");
            break;
        case 0x0172:
            strcpy(name, "GeForce4 MX 420");
            break;
        case 0x0171:
            strcpy(name, "GeForce4 MX 440");
            break;
        case 0x0181:
            strcpy(name, "GeForce4 MX 440 with AGP8X");
            break;
        case 0x0173:
            strcpy(name, "GeForce4 MX 440-SE");
            break;
        case 0x0170:
            strcpy(name, "GeForce4 MX 460");
            break;
        case 0x0253:
            strcpy(name, "GeForce4 Ti 4200");
            break;
        case 0x0281:
            strcpy(name, "GeForce4 Ti 4200 with AGP8X");
            break;
        case 0x0251:
            strcpy(name, "GeForce4 Ti 4400");
            break;
        case 0x0250:
            strcpy(name, "GeForce4 Ti 4600");
            break;
        case 0x0280:
            strcpy(name, "GeForce4 Ti 4800");
            break;
        case 0x0282:
            strcpy(name, "GeForce4 Ti 4800SE");
            break;
        case 0x0203:
            strcpy(name, "Lwdqro DCC");
            break;
        case 0x0309:
            strcpy(name, "Lwdqro FX 1000");
            break;
        case 0x034E:
            strcpy(name, "Lwdqro FX 1100");
            break;
        case 0x00FE:
            strcpy(name, "Lwdqro FX 1300");
            break;
        case 0x00CE:
            strcpy(name, "Lwdqro FX 1400");
            break;
        case 0x0308:
            strcpy(name, "Lwdqro FX 2000");
            break;
        case 0x0338:
            strcpy(name, "Lwdqro FX 3000");
            break;
        case 0x00FD:
            strcpy(name, "Lwdqro PCI-E Series");
            break;
        case 0x00F8:
            strcpy(name, "Lwdqro FX 3400/4400");
            break;
        case 0x00CD:
            strcpy(name, "Lwdqro FX 3450/4000 SDI");
            break;
        case 0x004E:
            strcpy(name, "Lwdqro FX 4000");
            break;
        case 0x009D:
            strcpy(name, "Lwdqro FX 4500");
            break;
        case 0x029F:
            strcpy(name, "Lwdqro FX 4500 X2");
            break;
        case 0x032B:
            strcpy(name, "Lwdqro FX 500/FX 600");
            break;
        case 0x014E:
            strcpy(name, "Lwdqro FX 540");
            break;
        case 0x014C:
            strcpy(name, "Lwdqro FX 540 MXM");
            break;
        case 0x033F:
            strcpy(name, "Lwdqro FX 700");
            break;
        case 0x034C:
            strcpy(name, "Lwdqro FX Go1000");
            break;
        case 0x00CC:
            strcpy(name, "Lwdqro FX Go1400");
            break;
        case 0x031C:
            strcpy(name, "Lwdqro FX Go700");
            break;
        case 0x018A:
            strcpy(name, "Lwdqro LWS with AGP8X");
            break;
        case 0x032A:
            strcpy(name, "Lwdqro LWS 280 PCI");
            break;
        case 0x0165:
            strcpy(name, "Lwdqro LWS 285");
            break;
        case 0x017A:
            strcpy(name, "Lwdqro LWS");
            break;
        case 0x0113:
            strcpy(name, "Quadro2 MXR/EX");
            break;
        case 0x018B:
            strcpy(name, "Quadro4 380 XGL");
            break;
        case 0x0178:
            strcpy(name, "Quadro4 550 XGL");
            break;
        case 0x0188:
            strcpy(name, "Quadro4 580 XGL");
            break;
        case 0x025B:
            strcpy(name, "Quadro4 700 XGL");
            break;
        case 0x0259:
            strcpy(name, "Quadro4 750 XGL");
            break;
        case 0x0258:
            strcpy(name, "Quadro4 900 XGL");
            break;
        case 0x0288:
            strcpy(name, "Quadro4 980 XGL");
            break;
        case 0x028C:
            strcpy(name, "Quadro4 Go700");
            break;
        case 0x0295:
            strcpy(name, "LWPU VdChip 7950 GT");
            break;
        case 0x03D0:
            strcpy(name, "LWPU VdChip 6100 nForce 430");
            break;
        case 0x03D1:
            strcpy(name, "LWPU VdChip 6100 nForce 405");
            break;
        case 0x03D2:
            strcpy(name, "LWPU VdChip 6100 nForce 400");
            break;
        case 0x0241:
            strcpy(name, "LWPU VdChip 6150 LE");
            break;
        case 0x0242:
            strcpy(name, "LWPU VdChip 6100");
            break;
        case 0x0245:
            strcpy(name, "LWPU Lwdqro LWS 210S / LWPU VdChip 6150LE");
            break;
        case 0x029C:
            strcpy(name, "LWPU Lwdqro FX 5500");
            break;
        case 0x0191:
            strcpy(name, "LWPU VdChip 8800 GTX");
            break;
        case 0x0193:
            strcpy(name, "LWPU VdChip 8800 GTS");
            break;
        case 0x0400:
            strcpy(name, "LWPU VdChip 8600 GTS");
            break;
        case 0x0402:
            strcpy(name, "LWPU VdChip 8600 GT");
            break;
        case 0x0421:
            strcpy(name, "LWPU VdChip 8500 GT");
            break;
        case 0x0422:
            strcpy(name, "LWPU VdChip 8400 GS");
            break;
        case 0x0423:
            strcpy(name, "LWPU VdChip 8300 GS");
            break;
        case 0x00FC:
            strcpy(name, "LWPU Lwdqro FX 330 / LWPU VdChip PCX 5300");
            break;
        case 0x0604:
            strcpy(name, "LWPU VdChip 9800 GX2");
            break;
        default:
            sprintf(name, "Unknown_0x%04X", devid);
    }
}

int isDelim(char ch, char* delims)
{
  U032 i;

  for (i=0; i<strlen(delims); i++)
    if (ch == delims[i])
      return 1;

  return 0;
}

void skipDelims(char** input, char* delims)
{
  while (**input && isDelim(**input, delims))
    *input += 1;
}

char* struppr (char *a)
{
    char *ret = a;
    while (*a != '\0')
    {
        if (islower (*a))
            *a = (char)toupper (*a);
        ++a;
    }
    return ret;
}

