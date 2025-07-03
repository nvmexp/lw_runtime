/*
 * Copyright (c) 2022, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIIPC_CONFIG_H
#define INCLUDED_LWSCIIPC_CONFIG_H

#if defined(__QNX__)
/* enable config blob V2 structure */
//#define CONFIGBLOB_V2
/* enable checksum of config entry */
//#define CFGENTRY_CHKSUM
/* enable 64bit VUID */
//#define VUID_64BIT
#endif /* __QNX__ */

#endif /* INCLUDED_LWSCIIPC_CONFIG_H */

