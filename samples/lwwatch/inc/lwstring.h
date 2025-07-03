/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
// Defines
//
#define lower(val)   (((val)>'A'&&(val)<'G')?('a'+(val)-'A'):(val))
#define hexalph(val) (((lower(val)>='a')&&(lower(val)<'g'))?1:0)  // valid hex alphachar
#define atoi(val)    (((val)>'9')?-1:(((val)<'0')?-1:(val)-'0'))
#define htoi(val)    (((val)<='9')?atoi(val):hexalph(val)?(10+(lower(val)-'a')):-1)

//
// hex string to long
//
static inline unsigned long hexstrtol(const char *str)
{
    unsigned long value = 0;
    int i = 0, len = strlen(str);

    // valid hex is 16 chars for 64-bit + "0x" == 10 chars
    if (len > 18)
    {
        dprintf("invalid hex string\n");
        return -1;
    }

    // verify this is a hex number, then skip over the beginning (0x)
    // otherwise, error out so the calling code can treat it as a decimal int
    if (str[0] != '0' || str[1] != 'x')
        return -1;

    str += 2;
    len -= 2;

    //
    // ok, add up the values of the individual characters,
    // starting at the least significant digit and moving up
    //
    while (len)
    {
        // value += htoi(str[len]) * (i*16);
        unsigned long tmp = htoi(str[len-1]);
        if (tmp == (unsigned long)-1)
            return tmp;
        value += tmp * ((unsigned long) 1<<(i*4));
        len--;
        i++;
    }

    // dprintf("colwerted %s to 0x%lx\n", str, value);

    return value;
}

//
// (decimal) string to long
//
static inline unsigned long strtol(const char *str)
{
    unsigned long value = 0;
    int i = 0, len = strlen(str);

    //
    // should there be a check on a valid max length?
    // ok, add up the values of the individual characters,
    // starting at the least significant digit and moving up
    //
    while (len)
    {
        // value += atoi(str[len]) * (i*16);
        unsigned long tmp = atoi(str[len-1]);
        if (tmp == (unsigned long)-1)
            return tmp;
        value += tmp * (i?i*10:1);
        len--;
        i++;
    }

    return value;
}

#ifndef tolower
static inline char tolower(char alpha)
{
    if (alpha < 'A' || alpha > 'Z')
    {
        return alpha;
    }
    return (char) alpha + 32;
}
#endif
