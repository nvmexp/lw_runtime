#pragma once

/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>

inline static void Error(const char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg); fflush(stderr);
    exit(1);
}

inline static void Error(const char *msg, const char *arg)
{
    fprintf(stderr, "ERROR: ");
    fprintf(stderr, msg, arg);
    fprintf(stderr, "\n");
    fflush(stderr);
    exit(1);
}

inline static void Error(const char *msg, int arg)
{
    fprintf(stderr, "ERROR: ");
    fprintf(stderr, msg, arg);
    fprintf(stderr, "\n");
    fflush(stderr);
    exit(1);
}
