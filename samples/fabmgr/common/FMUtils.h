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
#pragma once
#include <string>

/**
 * Misc utility functions
 */
class FMUtils {
public:
    /**
     * Colwert buffer to a string with each byte represented in hex
     * @param[in] buf       buffer to be printed in hex format
     * @param[in] buflen    size of buffer to be printed
     *
     * @returns string with each byte represented in hex
     */
    static std::string colwertUuidToHexStr(const void *buf, int buflen);
};

