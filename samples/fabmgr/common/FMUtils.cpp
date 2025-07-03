/*
 *  Copyright 2018-2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

/**
 * @file Misc utility functions
 */

#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>

#include "FMUtils.h"
#include "lwtypes.h"
#include "lwCpuUuid.h"

std::string
FMUtils::colwertUuidToHexStr(const void *buf, int buflen)
{
    std::stringstream ss;

    ss << std::uppercase << std::hex << std::setfill('0');  // needs to be set only once
    const unsigned char *ptr = reinterpret_cast<const unsigned char *>(buf);
    for (int i = 0; (i < buflen) && (i < LW_UUID_LEN) ; i++) {
        if ((i == 4) || (i == 6) || (i == 8) || (i == 10))
        {
            ss << "-";
        }
        ss << std::setw(2) << static_cast<unsigned>(*ptr++);
    }
    return ss.str();
}
