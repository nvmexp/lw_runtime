/*
 * SPDX-FileCopyrightText: Copyright (c) 2005-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl2080/ctrl2080bios.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "ctrl/ctrl2080/ctrl2080base.h"

/* LW20_SUBDEVICE_XX bios-related control commands and parameters */

/*
 * LW2080_CTRL_CMD_BIOS_GET_IMAGE
 *
 * This command fills in the specified buffer with the contents of the
 * bios image for the specified GPU.
 *
 *   biosImageLength
 *     This parameter specifies the size in bytes of the buffer referenced
 *     by the biosImage parameter on entry.  It returns the actual number
 *     of bytes transferred to the biosImage buffer on exit.  The value
 *     specified by this parameter should be greater than or equal to 1.
 *  biosImage
 *     This parameter contains a pointer to the destination buffer into which
 *     the bios image is to be copied.  The size of this buffer is specified
 *     by the biosImageLength parameter.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW2080_CTRL_CMD_BIOS_GET_IMAGE (0x20800801) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BIOS_INTERFACE_ID << 8) | LW2080_CTRL_BIOS_GET_IMAGE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BIOS_GET_IMAGE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_BIOS_GET_IMAGE_PARAMS {
    LwU32 biosImageLength;
    LW_DECLARE_ALIGNED(LwP64 biosImage, 8);
} LW2080_CTRL_BIOS_GET_IMAGE_PARAMS;

/*
 * LW2080_CTRL_BIOS_INFO
 *
 * This structure represents a single 32bit bios value.  Clients
 * request a particular bios information value by specifying a unique bios
 * information index.
 *
 * Legal bus information index values are:
 *   LW2080_CTRL_BIOS_INFO_INDEX_REVISION
 *     This index is used to request the bios revision for the associated GPU.
 *   LW2080_CTRL_BIOS_INFO_INDEX_OEM_REVISION
 *     This index is used to request the OEM-specific bios revision for the
 *     associated GPU.
 *   LW2080_CTRL_BIOS_INFO_INDEX_TV_FORMAT_DEFAULT
 *     This index is used to request the default TV standard used by the
 *     bios for the associated GPU.
 *   LW2080_CTRL_BIOS_INFO_INDEX_PINSET_A
 *     This index is used to inquire whether PINSET_A connector exist
 *     for the associated GPU. (0 for no, 1 for yes.)
 *   LW2080_CTRL_BIOS_INFO_INDEX_PINSET_B
 *     This index is used to inquire whether PINSET_B connector exist
 *     for the associated GPU. (0 for no, 1 for yes.)
 *   LW2080_CTRL_BIOS_INFO_INDEX_STRICT_SKU_FLAG
 *     This index is used to inquire whether this SKU is subjected to strict
 *     testing. A value of one indicates that it is and zero that it is not.
 *   LW2080_CTRL_BIOS_INFO_INDEX_MXM_VERSION
 *     This index is used to request the MXM version.
 *   LW2080_CTRL_BIOS_INFO_INDEX_VBIOS_SELWRITY_TYPE
 *       This index is used to retrieve the VBIOS type
 *       The VBIOS type can be one of the following values:
 *          LW2080_CTRL_BIOS_INFO_SELWRITY_TYPE_ILWALID
 *          LW2080_CTRL_BIOS_INFO_SELWRITY_TYPE_UNSIGNED
 *          LW2080_CTRL_BIOS_INFO_SELWRITY_TYPE_LWIDIA_DEBUG
 *          LW2080_CTRL_BIOS_INFO_SELWRITY_TYPE_LWIDIA_RELEASE
 *          LW2080_CTRL_BIOS_INFO_SELWRITY_TYPE_LWIDIA_AE_DEBUG
 *          LW2080_CTRL_BIOS_INFO_SELWRITY_TYPE_PARTNER_DEBUG
 *          LW2080_CTRL_BIOS_INFO_SELWRITY_TYPE_PARTNER
 *          LW2080_CTRL_BIOS_INFO_SELWRITY_TYPE_XOC
 *   LW2080_CTRL_BIOS_INFO_INDEX_VBIOS_STATUS
 *       This index is used to retrieve the VBIOS status.
 *       The VBIOS status can be one of the following values:
 *          LW2080_CTRL_BIOS_INFO_STATUS_OK
 *              This is the value if all checks have passed.
 *          LW2080_CTRL_BIOS_INFO_STATUS_EXPIRED
 *              This status implies that the X509 certificate has expired.
 *          LW2080_CTRL_BIOS_INFO_STATUS_ILWALID
 *              This status implies that there is an error in VBIOS security
 *              checks.
 *          LW2080_CTRL_BIOS_INFO_STATUS_DEVID_MISMATCH
 *              This status implies that The PCI ID mismatches between the
 *              VBIOS and chip.
 *          LW2080_CTRL_BIOS_INFO_STATUS_ERR_ROMPACK_OFFSET
 *              This status implies that there was an error retrieving the
 *              ROMPack offset from the VBIOS.
 *          LW2080_CTRL_BIOS_INFO_STATUS_INSUFFICIENT_RESOURCES
 *              This status oclwrs when there is a memory allocation problem
 *          LW2080_CTRL_BIOS_INFO_STATUS_NBSI_INCOMPLETE
 *              This status oclwrs when NBSI object is incomplete.
 *          LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_CERT
 *              This status implies that the certificate data is invalid.
 *          LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_HASH
 *              This status implies that the hash data is invalid.
 *          LW2080_CTRL_BIOS_INFO_STATUS_VBIOS_HASH_NOT_STARTED
 *              This status implies that there is an unexpected flag set.
 *              VBIOS image hash should have started at this point.
 *          LW2080_CTRL_BIOS_INFO_STATUS_SELWRITY_BLOCK_NOT_FOUND
 *              This status oclwrs when getNbsiObjByType raises an error. It
 *              implies that the security block has not been found in the VBIOS
 *              image.
 *          LW2080_CTRL_BIOS_INFO_STATUS_FIRST_X509_NOT_FOUND
 *              This status implies that the first VBIOS certificate has not been
 *              found.
 *          LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_PEM_FORMAT
 *              This status implies that there was an error while reading the
 *              certificate. The PEM format is invalid.
 *          LW2080_CTRL_BIOS_INFO_STATUS_UNKNOWN_CERT_TYPE
 *              This status implies that certificate type is unknown.
 *          LW2080_CTRL_BIOS_INFO_STATUS_DUPLICATE_VENDOR_CERT_FOUND
 *              This status implies that a duplicate vendor certificate has been
 *              found while checking the X509 certificates.
 *          LW2080_CTRL_BIOS_INFO_STATUS_NO_PUBLIC_KEY
 *              This status implies that no public key exists in the X509.
 *              certificate.
 *          LW2080_CTRL_BIOS_INFO_STATUS_POINTER_PAST_SELWRITY_BLK
 *              This status implies that there was an error while checking X509
 *              certificates. The header pointer addresses a location which is
 *              past the security block.
 *          LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_HASH_HEADER_VERSION
 *              This status implies that the hash header version is different
 *              from what is expected.
 *          LW2080_CTRL_BIOS_INFO_STATUS_ERR_HASH_HEADER_FLAG_SET
 *              This status implies that the hash header flag is set which has
 *              caused the error.
 *          LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_SIGNATURE_HEADER_VER
 *              This status implies that the signature header version is
 *              unexpected.
 *          LW2080_CTRL_BIOS_INFO_STATUS_SIG_UNKNOWN_DIGEST_ALGO
 *              This status implies that the Digest algorithm in the signature
 *              is unknown.
 *          LW2080_CTRL_BIOS_INFO_STATUS_SIG_UNKNOWN_FORMAT
 *              This status implies that the signature format is unknown.
 *          LW2080_CTRL_BIOS_INFO_STATUS_SIG_ILWALID_SIZE
 *              This status implies that the signature and VBIOS hash sizes
 *              are invalid.
 *          LW2080_CTRL_BIOS_INFO_STATUS_SIG_VERIFICATION_FAILURE
 *              This status implies that there is a verification failure in the
 *              signature data.
 *          LW2080_CTRL_BIOS_INFO_STATUS_PRESERV_TABLE_HASH_NOT_STARTED
 *              This status implies that there is an unexpected flag set.
 *              Preservation Table hash should have started at this point.
 *          LW2080_CTRL_BIOS_INFO_STATUS_NO_EXPANSION_ROM
 *              This status implies that there was no expansion ROM found in
 *              the hash area.
 *          LW2080_CTRL_BIOS_INFO_STATUS_UNKNOWN_HASH_TYPE
 *              This status implies that the hash type was unknown and therefore
 *              not handled.
 *          LW2080_CTRL_BIOS_INFO_STATUS_POINTER_PAST_HASH_BLK
 *              This status implies that the hash entry goes past the PCI
 *              block size.
 *          LW2080_CTRL_BIOS_INFO_STATUS_CERT_VALIDITY_PERIOD_NOT_FOUND
 *              This status implies that the validity period was not found in
 *              the X509 certificate.
 *          LW2080_CTRL_BIOS_INFO_STATUS_CERT_OEM_NAME_NOT_FOUND
 *              This implies that the OEM name could not be located in the
 *              X509 certificate
 *          LW2080_CTRL_BIOS_INFO_STATUS_CERT_CHAIN_OF_TRUST_FAILURE
 *              This implies a verification failure in chain of trust in
 *              X509 certificate.
 *          LW2080_CTRL_BIOS_INFO_STATUS_NO_BIT_HEADER
 *              This implies that there was no BIT header found in the VBIOS
 *              image.
 *          LW2080_CTRL_BIOS_INFO_STATUS_NO_VBIOS_FOUND
 *              This implies that the system was unable to locate a VBIOS image.
 *          LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_PARAMS
 *              This implies that the function in context has received invalid
 *              parameters.
 *          LW2080_CTRL_BIOS_INFO_STATUS_NOT_SILICON_OR_EMULATION
 *              This implies that an emulation or a non-Silicon chip was
 *              detected while locating Expansion ROM images.
 *          LW2080_CTRL_BIOS_INFO_STATUS_LW_CONFIG_PCI_LW_20_READ_ERROR
 *              This denotes an error while reading GPU configuration from HAL.
 *          LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_PCI_ROM_SIG
 *              This implies that the PCI ROM signature is invalid. This means
 *              that the VBIOS image is invalid.
 *          LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_PCI_DATA_SIG
 *              This implies that the PCI data signature is invalid. This means
 *              that the VBIOS image is invalid.
 *          LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_PCI_HEADER
 *              This implies that the PCI Header is invalid.
 *          LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_CHECKSUM
 *              This implies that the there was a checksum error.
 *          LW2080_CTRL_BIOS_INFO_STATUS_NO_NBSI_BLOCK
 *              This implies that no NBSI block was found in the VBIOS image.
 *          LW2080_CTRL_BIOS_INFO_STATUS_CANNOT_MAP_TO_KERNEL_SPACE
 *              This implies that there was an error while mapping system BIOS
 *              into kernel space.
 *          LW2080_CTRL_BIOS_INFO_STATUS_IMAGE_EXCEEDS_PCIR_SIZE
 *              This implies that the PCIR cannot fit in the image size. This
 *              means that the image is invalid.
 *          LW2080_CTRL_BIOS_INFO_STATUS_PCIR_VENDOR_ID_MISMATCH
 *              This implies that the PCIR is invalid due to a vendor ID
 *              mismatch.
 *          LW2080_CTRL_BIOS_INFO_STATUS_PCIR_LEN_EXCEEDS_IMAGE
 *              This implies that PCIR exceeds past the image length.
 *          LW2080_CTRL_BIOS_INFO_STATUS_IMAGE_SIZE_OUT_OF_BOUNDS
 *              This implies that the source image size does lie between the
 *              bounds of the BIOS ROM code size.
 *          LW2080_CTRL_BIOS_INFO_STATUS_REGISTRY_NOT_FOUND
 *              This implies a read error from registry.
 *          LW2080_CTRL_BIOS_INFO_STATUS_VOLATILE_REGISTRY_NOT_FOUND
 *              This implies a read error from a volatile/session registry key
 *              value.
 *          LW2080_CTRL_BIOS_INFO_STATUS_GPUMGR_OBJECT_NOT_FOUND
 *              This implies that a valid cache could not be found in the GPU
 *              manager.
 *          LW2080_CTRL_BIOS_INFO_STATUS_GPUMGR_BUFFER_TOO_SMALL
 *              This implies that the input buffer is smaller than the copy
 *              of the VBIOS within the GPU manager.
 *          LW2080_CTRL_BIOS_INFO_STATUS_INSTANCE_NOT_FOUND
 *              This implies a read error while fetching BIOS from Instance
 *              Memory
 *          LW2080_CTRL_BIOS_INFO_STATUS_IMAGE_VERIFICATION_FAILURE
 *              This implies a verification/validation failure in the BIOS
 *              image.
 *          LW2080_CTRL_BIOS_INFO_STATUS_UNSET
 *              The status is initialized with this value. It is temporary and
 *              is active only when the security checks are being run.
 *          LW2080_CTRL_BIOS_INFO_STATUS_FWSECLIC_SB_FAILURE
 *              The status implies secure boot failure
 *  LW2080_CTRL_BIOS_INFO_INDEX_VBIOS_SELWRITY_EXPIRATION
 *       This index is used to get the VBIOS expiration date.
 *       The time is in seconds elapsed since 1970 Jan 1.
 *       If there is no expiration date the returned value is 0.
 * LW2080_CTRL_BIOS_INFO_INDEX_VBIOS_SELWRITY_CREATION
 *       This index is used to get the VBIOS creation date.
 *       The time is in seconds elapsed since 1970 Jan 1.
 * LW2080_CTRL_BIOS_INFO_INDEX_VBIOS_DEVICE_ID
 *       This index is used to get the device Id stored in the VBIOS.
 */
typedef struct LW2080_CTRL_BIOS_INFO {
    LwU32 index;
    LwU32 data;
} LW2080_CTRL_BIOS_INFO;

/* valid bios info index values */
#define LW2080_CTRL_BIOS_INFO_INDEX_REVISION                        (0x00000000)
#define LW2080_CTRL_BIOS_INFO_INDEX_OEM_REVISION                    (0x00000001)
#define LW2080_CTRL_BIOS_INFO_INDEX_TV_FORMAT_DEFAULT               (0x00000002)
#define LW2080_CTRL_BIOS_INFO_INDEX_PIN_SET_A                       (0x00000003)
#define LW2080_CTRL_BIOS_INFO_INDEX_PIN_SET_B                       (0x00000004)
#define LW2080_CTRL_BIOS_INFO_INDEX_STRICT_SKU_FLAG                 (0x00000005)
#define LW2080_CTRL_BIOS_INFO_INDEX_MXM_VERSION                     (0x00000006)
#define LW2080_CTRL_BIOS_INFO_INDEX_VBIOS_SELWRITY_TYPE             (0x00000007)
#define LW2080_CTRL_BIOS_INFO_INDEX_VBIOS_STATUS                    (0x00000008)
#define LW2080_CTRL_BIOS_INFO_INDEX_VBIOS_SELWRITY_EXPIRATION       (0x00000009)
#define LW2080_CTRL_BIOS_INFO_INDEX_VBIOS_SELWRITY_CREATION         (0x0000000a)
#define LW2080_CTRL_BIOS_INFO_INDEX_VBIOS_DEVICE_ID                 (0x0000000b)

/* This macro has been deprecated and will be removed */
#define LW2080_CTRL_BIOS_INFO_INDEX_VBIOS_SELWRITY_STATUS           (0x00000008)

/* Maximum number of bios infos that can be queried at once */
#define LW2080_CTRL_BIOS_INFO_MAX_SIZE                              (0x0000000F)

/* valid VBIOS security types values */
#define LW2080_CTRL_BIOS_INFO_SELWRITY_TYPE_ILWALID                 (0x00000000)
#define LW2080_CTRL_BIOS_INFO_SELWRITY_TYPE_UNSIGNED                (0x00000001)
#define LW2080_CTRL_BIOS_INFO_SELWRITY_TYPE_LWIDIA_DEBUG            (0x00000002)
#define LW2080_CTRL_BIOS_INFO_SELWRITY_TYPE_LWIDIA_RELEASE          (0x00000003)
#define LW2080_CTRL_BIOS_INFO_SELWRITY_TYPE_LWIDIA_AE_DEBUG         (0x00000004)
#define LW2080_CTRL_BIOS_INFO_SELWRITY_TYPE_PARTNER_DEBUG           (0x00000005)
#define LW2080_CTRL_BIOS_INFO_SELWRITY_TYPE_PARTNER                 (0x00000006)
#define LW2080_CTRL_BIOS_INFO_SELWRITY_TYPE_XOC                     (0x00000007)

/*
 * VBIOS Security and VBIOS Extraction Error codes
 * All codes start with LW2080_CTRL_BIOS_INFO_STATUS_
 */

/* valid VBIOS status values */
#define LW2080_CTRL_BIOS_INFO_STATUS_OK                             (0x00000000)
#define LW2080_CTRL_BIOS_INFO_STATUS_EXPIRED                        (0x00000001)
#define LW2080_CTRL_BIOS_INFO_STATUS_ILWALID                        (0x00000002)
#define LW2080_CTRL_BIOS_INFO_STATUS_DEVID_MISMATCH                 (0x00000003)
#define LW2080_CTRL_BIOS_INFO_STATUS_ERR_ROMPACK_OFFSET             (0x00000004)
#define LW2080_CTRL_BIOS_INFO_STATUS_INSUFFICIENT_RESOURCES         (0x00000005)
#define LW2080_CTRL_BIOS_INFO_STATUS_NBSI_INCOMPLETE                (0x00000006)
#define LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_CERT                   (0x00000007)
#define LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_HASH                   (0x00000008)
#define LW2080_CTRL_BIOS_INFO_STATUS_VBIOS_HASH_NOT_STARTED         (0x00000009)
#define LW2080_CTRL_BIOS_INFO_STATUS_SELWRITY_BLOCK_NOT_FOUND       (0x0000000a)
#define LW2080_CTRL_BIOS_INFO_STATUS_FIRST_X509_NOT_FOUND           (0x0000000b)
#define LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_PEM_FORMAT             (0x0000000c)
#define LW2080_CTRL_BIOS_INFO_STATUS_UNKNOWN_CERT_TYPE              (0x0000000d)
#define LW2080_CTRL_BIOS_INFO_STATUS_DUPLICATE_VENDOR_CERT_FOUND    (0x0000000e)
#define LW2080_CTRL_BIOS_INFO_STATUS_NO_PUBLIC_KEY                  (0x0000000f)
#define LW2080_CTRL_BIOS_INFO_STATUS_POINTER_PAST_SELWRITY_BLK      (0x00000010)
#define LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_HASH_HEADER_VERSION    (0x00000011)
#define LW2080_CTRL_BIOS_INFO_STATUS_ERR_HASH_HEADER_FLAG_SET       (0x00000012)
#define LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_SIGNATURE_HEADER_VER   (0x00000013)
#define LW2080_CTRL_BIOS_INFO_STATUS_SIG_UNKNOWN_DIGEST_ALGO        (0x00000014)
#define LW2080_CTRL_BIOS_INFO_STATUS_SIG_UNKNOWN_FORMAT             (0x00000015)
#define LW2080_CTRL_BIOS_INFO_STATUS_SIG_ILWALID_SIZE               (0x00000016)
#define LW2080_CTRL_BIOS_INFO_STATUS_SIG_VERIFICATION_FAILURE       (0x00000017)
#define LW2080_CTRL_BIOS_INFO_STATUS_PRESERV_TABLE_HASH_NOT_STARTED (0x00000018)
#define LW2080_CTRL_BIOS_INFO_STATUS_NO_EXPANSION_ROM               (0x00000019)
#define LW2080_CTRL_BIOS_INFO_STATUS_UNKNOWN_HASH_TYPE              (0x0000001a)
#define LW2080_CTRL_BIOS_INFO_STATUS_POINTER_PAST_HASH_BLK          (0x0000001b)
#define LW2080_CTRL_BIOS_INFO_STATUS_CERT_VALIDITY_PERIOD_NOT_FOUND (0x0000001c)
#define LW2080_CTRL_BIOS_INFO_STATUS_CERT_OEM_NAME_NOT_FOUND        (0x0000001d)
#define LW2080_CTRL_BIOS_INFO_STATUS_CERT_CHAIN_OF_TRUST_FAILURE    (0x0000001e)
#define LW2080_CTRL_BIOS_INFO_STATUS_NO_BIT_HEADER                  (0x0000001f)
#define LW2080_CTRL_BIOS_INFO_STATUS_NO_VBIOS_FOUND                 (0x00000020)
#define LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_PARAMS                 (0x00000021)
#define LW2080_CTRL_BIOS_INFO_STATUS_NOT_SILICON_OR_EMULATION       (0x00000022)
#define LW2080_CTRL_BIOS_INFO_STATUS_LW_CONFIG_PCI_LW_20_READ_ERROR (0x00000023)
#define LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_PCI_ROM_SIG            (0x00000024)
#define LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_PCI_DATA_SIG           (0x00000025)
#define LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_PCI_HEADER             (0x00000026)
#define LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_CHECKSUM               (0x00000027)
#define LW2080_CTRL_BIOS_INFO_STATUS_NO_NBSI_BLOCK                  (0x00000028)
#define LW2080_CTRL_BIOS_INFO_STATUS_CANNOT_MAP_TO_KERNEL_SPACE     (0x00000029)
#define LW2080_CTRL_BIOS_INFO_STATUS_IMAGE_EXCEEDS_PCIR_SIZE        (0x0000002a)
#define LW2080_CTRL_BIOS_INFO_STATUS_PCIR_VENDOR_ID_MISMATCH        (0x0000002b)
#define LW2080_CTRL_BIOS_INFO_STATUS_PCIR_LEN_EXCEEDS_IMAGE         (0x0000002c)
#define LW2080_CTRL_BIOS_INFO_STATUS_IMAGE_SIZE_OUT_OF_BOUNDS       (0x0000002d)
#define LW2080_CTRL_BIOS_INFO_STATUS_REGISTRY_NOT_FOUND             (0x0000002e)
#define LW2080_CTRL_BIOS_INFO_STATUS_VOLATILE_REGISTRY_NOT_FOUND    (0x0000002f)
#define LW2080_CTRL_BIOS_INFO_STATUS_GPUMGR_OBJECT_NOT_FOUND        (0x00000030)
#define LW2080_CTRL_BIOS_INFO_STATUS_GPUMGR_BUFFER_TOO_SMALL        (0x00000031)
#define LW2080_CTRL_BIOS_INFO_STATUS_INSTANCE_NOT_FOUND             (0x00000032)
#define LW2080_CTRL_BIOS_INFO_STATUS_IMAGE_VERIFICATION_FAILURE     (0x00000033)
#define LW2080_CTRL_BIOS_INFO_STATUS_UNSET                          (0x00000034)
#define LW2080_CTRL_BIOS_INFO_STATUS_FWSECLIC_SB_FAILURE            (0x00000035)

/* These set of values have been deprecated and will be removed. */
#define LW2080_CTRL_BIOS_INFO_SELWRITY_STATUS_OK                    (0x00000000)
#define LW2080_CTRL_BIOS_INFO_SELWRITY_STATUS_EXPIRED               (0x00000001)
#define LW2080_CTRL_BIOS_INFO_SELWRITY_STATUS_ILWALID               (0x00000002)
#define LW2080_CTRL_BIOS_INFO_SELWRITY_STATUS_DEVID_MISMATCH        (0x00000003)

/*
 * LW2080_CTRL_CMD_BIOS_GET_INFO
 *
 * This command returns bios information for the associated GPU.
 * Requests to retrieve bios information use a list of one or more
 * LW2080_CTRL_BIOS_INFO structures.
 *
 *   biosInfoListSize
 *     This field specifies the number of entries on the caller's
 *     biosInfoList.
 *   biosInfoList
 *     This field specifies a pointer in the caller's address space
 *     to the buffer into which the bios information is to be returned.
 *     This buffer must be at least as big as biosInfoListSize multiplied
 *     by the size of the LW2080_CTRL_BIOS_INFO structure.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW2080_CTRL_CMD_BIOS_GET_INFO                               (0x20800802) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BIOS_INTERFACE_ID << 8) | 0x2" */

typedef struct LW2080_CTRL_BIOS_GET_INFO_PARAMS {
    LwU32 biosInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 biosInfoList, 8);
} LW2080_CTRL_BIOS_GET_INFO_PARAMS;


/*
 * LW2080_CTRL_CMD_BIOS_GET_INFO_V2
 *
 * Like LW2080_CTRL_CMD_BIOS_GET_INFO but without the embedded pointer
 */
#define LW2080_CTRL_CMD_BIOS_GET_INFO_V2 (0x20800810) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BIOS_INTERFACE_ID << 8) | LW2080_CTRL_BIOS_GET_INFO_V2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BIOS_GET_INFO_V2_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW2080_CTRL_BIOS_GET_INFO_V2_PARAMS {
    LwU32                 biosInfoListSize;
    LW2080_CTRL_BIOS_INFO biosInfoList[LW2080_CTRL_BIOS_INFO_MAX_SIZE];
} LW2080_CTRL_BIOS_GET_INFO_V2_PARAMS;

/*
 * LW2080_CTRL_BIOS_NBSI
 *
 * LW2080_CTRL_BIOS_NBSI_MAX_REG_STRING_LENGTH
 *   This is the maximum length of a given registry string input (in characters).
 *
 * LW2080_CTRL_BIOS_NBSI_STRING_TYPE_ASCII
 *   This is a value indicating the format of a registry string is ascii.
 * LW2080_CTRL_BIOS_NBSI_STRING_TYPE_UNICODE
 *   This is a value indicating the format of a registry string is unicode.
 * LW2080_CTRL_BIOS_NBSI_STRING_TYPE_HASH
 *   This is a value indicating a registry string is actually a pre-hashed value.
 *
 * LW2080_CTRL_BIOS_NBSI_REG_STRING
 *   This is a structure used to store a registry string object.
 *   The members are as follows:
 *
 *   size
 *     This is the size (in bytes) of the data contained in the string. If this
 *     is greater than the maximum registry string length, an error will be
 *     returned.
 *   type
 *     This is the type of data contained in the registry string. It can be either
 *     ascii, unicode or a pre-hashed value.
 *   value
 *     This is the value of the string. Depending on the type, a different object
 *     will be used to access the data.
 */
#define LW2080_CTRL_BIOS_NBSI_MAX_REG_STRING_LENGTH (0x00000100)

#define LW2080_CTRL_BIOS_NBSI_STRING_TYPE_ASCII     (0x00000000)
#define LW2080_CTRL_BIOS_NBSI_STRING_TYPE_UNICODE   (0x00000001)
#define LW2080_CTRL_BIOS_NBSI_STRING_TYPE_HASH      (0x00000002)

#define LW2080_CTRL_BIOS_NBSI_MODULE_ROOT           (0x00000000)
#define LW2080_CTRL_BIOS_NBSI_MODULE_RM             (0x00000001)
#define LW2080_CTRL_BIOS_NBSI_MODULE_DISPLAYDRIVER  (0x00000002)
#define LW2080_CTRL_BIOS_NBSI_MODULE_VIDEO          (0x00000003)
#define LW2080_CTRL_BIOS_NBSI_MODULE_CPL            (0x00000004)
#define LW2080_CTRL_BIOS_NBSI_MODULE_D3D            (0x00000005)
#define LW2080_CTRL_BIOS_NBSI_MODULE_OGL            (0x00000006)
#define LW2080_CTRL_BIOS_NBSI_MODULE_PMU            (0x00000007)
#define LW2080_CTRL_BIOS_NBSI_MODULE_MODE           (0x00000008)
// this should equal the last NBSI_MODULE plus 1.
#define LW2080_CTRL_BIOS_NBSI_NUM_MODULES           (0x00000009)

//
// Never use this value! It's needed for DD/Video modules, but does not correspond
// to a valid NBSI hive!
//
#define LW2080_CTRL_BIOS_NBSI_MODULE_UNKNOWN        (0x80000000)

typedef struct LW2080_CTRL_BIOS_NBSI_REG_STRING {
    LwU32 size;
    LwU32 type;

    union {
        LwU8  ascii[LW2080_CTRL_BIOS_NBSI_MAX_REG_STRING_LENGTH];
        LwU16 unicode[LW2080_CTRL_BIOS_NBSI_MAX_REG_STRING_LENGTH];
        LwU16 hash;
    } value;
} LW2080_CTRL_BIOS_NBSI_REG_STRING;


/*
 * LW2080_CTRL_CMD_BIOS_GET_NBSI
 *
 * module
 *   This field specifies the given module per the MODULE_TYPES enum.
 * path
 *   This field specifies the full path and registry node name for a
 *   given NBSI object. This is a maximum of 255 unicode characters,
 *   but may be provided as ascii or a pre-formed hash per the type
 *   member. The size (in bytes) of the given string/hash should be
 *   provided in the size member.
 *
 *   NOTE: In the case of an incomplete path such as HKR, one may pass
 *   in simply the root node. E.g.:
 *   1.) Normal case: HKLM\Path\Subpath
 *   2.) Unknown case: HKR
 *   It is expected that all unknown/incomplete paths will be determined
 *   prior to NBSI programming! There is otherwise NO WAY to match
 *   the hash given by an incomplete path to that stored in NBSI!
 *
 * valueName
 *   This field specifies the registry name for a given NBSI object.
 *   This is a maximum of 255 unicode characters, but may be provided
 *   in ascii or a pre-formed hash per the type member. The size (in bytes)
 *   of the given string/hash should be provided in the size member.
 * retBuf
 *   This field provides a pointer to a buffer into which the value
 *   retrieved from NBSI may be returned
 * retSize
 *   This field is an input/output. It specifies the maximum size of the
 *   return buffer as an input, and the size of the returned data as an
 *   output.
 * errorCode
 *   This field is a return value. It gives an error code representing
 *   failure to return a value (as opposed to failure of the call).
 *   This obeys the following:
 *
 *   LW2080_CTRL_BIOS_GET_NBSI_SUCCESS
 *     The call has returned complete and valid data.
 *   LW2080_CTRL_BIOS_GET_NBSI_OVERRIDE
 *     The call returned complete and valid data which is expected to override
 *     any stored registry settings.
 *   LW2080_CTRL_BIOS_GET_NBSI_INCOMPLETE
 *     The call returned data, but the size of the return buffer was
 *     insufficient to contain it. The value returned in retSize represents
 *     the total size necessary (in bytes) to contain the data.
 *     if the size was non-0, the buffer is filled with the object contents up
 *     to that size. Can be used with retBufOffset to use multiple calls to get
 *     tables of very large size.
 *   LW2080_CTRL_BIOS_GET_NBSI_NOT_FOUND
 *     The call did not find a valid NBSI object for this key. This indicates
 *     NBSI has no opinion and, more importantly, any data returned is identical
 *     to data passed in.
 *
 * Possible return values are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_BIOS_GET_NBSI_SUCCESS         (0x00000000)
#define LW2080_CTRL_BIOS_GET_NBSI_OVERRIDE        (0x00000001)
#define LW2080_CTRL_BIOS_GET_NBSI_BAD_HASH        (0xFFFFFFFA)
#define LW2080_CTRL_BIOS_GET_NBSI_APITEST_SUCCESS (0xFFFFFFFB)
#define LW2080_CTRL_BIOS_GET_NBSI_BAD_TABLE       (0xFFFFFFFC)
#define LW2080_CTRL_BIOS_GET_NBSI_NO_TABLE        (0xFFFFFFFD)
#define LW2080_CTRL_BIOS_GET_NBSI_INCOMPLETE      (0xFFFFFFFE)
#define LW2080_CTRL_BIOS_GET_NBSI_NOT_FOUND       (0xFFFFFFFF)

#define LW2080_CTRL_CMD_BIOS_GET_NBSI             (0x20800803) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BIOS_INTERFACE_ID << 8) | LW2080_CTRL_BIOS_GET_NBSI_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BIOS_GET_NBSI_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW2080_CTRL_BIOS_GET_NBSI_PARAMS {
    LwU32                            module;
    LW2080_CTRL_BIOS_NBSI_REG_STRING path;
    LW2080_CTRL_BIOS_NBSI_REG_STRING valueName;
    LW_DECLARE_ALIGNED(LwP64 retBuf, 8);
    LwU32                            retSize;
    LwU32                            errorCode;
} LW2080_CTRL_BIOS_GET_NBSI_PARAMS;

#define LW2080_CTRL_CMD_BIOS_GET_NBSI_V2  (0x2080080e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BIOS_INTERFACE_ID << 8) | LW2080_CTRL_BIOS_GET_NBSI_V2_PARAMS_MESSAGE_ID" */

#define LW2080_BIOS_GET_NBSI_MAX_RET_SIZE (0x100)

#define LW2080_CTRL_BIOS_GET_NBSI_V2_PARAMS_MESSAGE_ID (0xEU)

typedef struct LW2080_CTRL_BIOS_GET_NBSI_V2_PARAMS {
    LwU32                            module;
    LW2080_CTRL_BIOS_NBSI_REG_STRING path;
    LW2080_CTRL_BIOS_NBSI_REG_STRING valueName;
    LwU8                             retBuf[LW2080_BIOS_GET_NBSI_MAX_RET_SIZE];
    LwU32                            retSize;
    LwU32                            errorCode;
} LW2080_CTRL_BIOS_GET_NBSI_V2_PARAMS;

/*
 * LW2080_CTRL_CMD_BIOS_GET_NBSI_OBJ
 *
 * globType
 *   This field specifies the glob type wanted
 *   0xffff: APItest... returns LW2080_CTRL_BIOS_GET_NBSI_APITEST_SUCCESS
 * globIndex
 *   Index for globType desired
 *      0 = best fit
 *      1..255 = actual index
 * globSource
 *   Index to nbsi directory sources used when getting entire directory
 *      0 = registry
 *      1 = VBIOS
 *      2 = SBIOS
 *      3 = ACPI
 * retBufOffset
 *   When making multiple calls to get the object (if retSize is too small)
 *   offset into real object (0=start of object)
 * retBuf
 *   This field provides a pointer to a buffer into which the object
 *   retrieved from NBSI may be returned
 * retSize
 *   This field is an input/output. It specifies the maximum size of the
 *   return buffer as an input, and the size of the returned data as an
 *   output.
 * totalObjSize
 *   This field is an output, where the total size of the object being
 *   retrieved is returned.
 * errorCode
 *   This field is a return value. It gives an error code representing
 *   failure to return a value (as opposed to failure of the call).
 *   This obeys the following:
 *
 *   LW2080_CTRL_BIOS_GET_NBSI_SUCCESS
 *     The call has returned complete and valid data.
 *   LW2080_CTRL_BIOS_GET_NBSI_OVERRIDE
 *     The call returned complete and valid data which is expected to override
 *     any stored registry settings.
 *   LW2080_CTRL_BIOS_GET_NBSI_INCOMPLETE
 *     The call returned data, but the size of the return buffer was
 *     insufficient to contain it. The value returned in retSize represents
 *     the total size necessary (in bytes) to contain the data.
 *   LW2080_CTRL_BIOS_GET_NBSI_NOT_FOUND
 *     The call did not find a valid NBSI object for this key. This indicates
 *     NBSI has no opinion and, more importantly, any data returned is identical
 *     to data passed in.
 *
 * Possible return values are:
 *   LW2080_CTRL_BIOS_GET_NBSI_SUCCESS
 *   LW2080_CTRL_BIOS_GET_NBSI_APITEST_NODIRACCESS
 *   LW2080_CTRL_BIOS_GET_NBSI_APITEST_SUCCESS
 *   LW2080_CTRL_BIOS_GET_NBSI_INCOMPLETE
 *   LW2080_CTRL_BIOS_GET_NBSI_BAD_TABLE
 *   LW2080_CTRL_BIOS_GET_NBSI_NO_TABLE
 *   LW2080_CTRL_BIOS_GET_NBSI_NOT_FOUND
 */
#define LW2080_CTRL_CMD_BIOS_GET_NBSI_OBJ (0x20800806) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BIOS_INTERFACE_ID << 8) | LW2080_CTRL_BIOS_GET_NBSI_OBJ_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BIOS_GET_NBSI_OBJ_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW2080_CTRL_BIOS_GET_NBSI_OBJ_PARAMS {
    LwU16 globType;
    LwU8  globIndex;
    LwU16 globSource;
    LwU32 retBufOffset;
    LW_DECLARE_ALIGNED(LwP64 retBuf, 8);
    LwU32 retSize;
    LwU32 totalObjSize;
    LwU32 errorCode;
} LW2080_CTRL_BIOS_GET_NBSI_OBJ_PARAMS;

#define GLOB_TYPE_GET_NBSI_DIR                       0xfffe
#define GLOB_TYPE_APITEST                            0xffff
#define GLOB_TYPE_GET_NBSI_ACPI_RAW                  0xfffd

/*
 * LW2080_CTRL_CMD_BIOS_GET_OEM_INFO
 *
 * This command returns OEM-specific bios information for the associated GPU.
 *
 *   vendorName
 *     This field returns the vendor name associated with the bios.
 *   selwreVendorName
 *     This field returns the vendor name associated with the bios.
 *     This vendor name is retrieved from the VBIOS X509 Vendor certificate
 *     and cannot be tampered. Available for Kepler and up.
 *   productName
 *     This field returns the product name associated with the bios.
 *   productRevision
 *     This field returns the product revision associated with the bios.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW2080_CTRL_CMD_BIOS_GET_OEM_INFO            (0x20800807) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BIOS_INTERFACE_ID << 8) | LW2080_CTRL_BIOS_GET_OEM_INFO_PARAMS_MESSAGE_ID" */

/* maximum length of parameter strings */
#define LW2080_CTRL_BIOS_OEM_VENDOR_NAME_LENGTH      (0x40)
#define LW2080_CTRL_BIOS_OEM_PRODUCT_NAME_LENGTH     (0x40)
#define LW2080_CTRL_BIOS_OEM_PRODUCT_REVISION_LENGTH (0x40)

#define LW2080_CTRL_BIOS_GET_OEM_INFO_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW2080_CTRL_BIOS_GET_OEM_INFO_PARAMS {
    LwU8 vendorName[LW2080_CTRL_BIOS_OEM_VENDOR_NAME_LENGTH];
    LwU8 selwreVendorName[LW2080_CTRL_BIOS_OEM_VENDOR_NAME_LENGTH];
    LwU8 productName[LW2080_CTRL_BIOS_OEM_PRODUCT_NAME_LENGTH];
    LwU8 productRevision[LW2080_CTRL_BIOS_OEM_PRODUCT_REVISION_LENGTH];
} LW2080_CTRL_BIOS_GET_OEM_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_BIOS_GET_SKU_INFO
 *
 * This command returns information about the current board SKU.
 * LW_ERR_ILWALID_OWNER will be returned if the call
 * isn't made with the OS as the administrator.
 *
 *  chipSKU
 *    This field returns the sku for the current chip.
 *  chipSKUMod
 *    This field returns the SKU modifier.
 *  project
 *    This field returns the Project (Board) number.
 *  projectSKU
 *    This field returns the Project (Board) SKU number.
 *  CDP
 *    This field returns the Collaborative Design Project Number.
 *  projectSKUMod
 *    This field returns the Project (Board) SKU Modifier.
 *  businessCycle
 *    This field returns the business cycle the board is associated with.
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_OWNER
 */
#define LW2080_CTRL_CMD_BIOS_GET_SKU_INFO (0x20800808) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BIOS_INTERFACE_ID << 8) | LW2080_CTRL_BIOS_GET_SKU_INFO_PARAMS_MESSAGE_ID" */

/* maximum length of parameter strings */


#define LW2080_CTRL_BIOS_GET_SKU_INFO_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW2080_CTRL_BIOS_GET_SKU_INFO_PARAMS {
    LwU32 BoardID;
    char  chipSKU[4];
    char  chipSKUMod[2];
    char  project[5];
    char  projectSKU[5];
    char  CDP[6];
    char  projectSKUMod[2];
    LwU32 businessCycle;
} LW2080_CTRL_BIOS_GET_SKU_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_BIOS_GET_POST_TIME

 * This command is used to get the GPU POST time (in milliseconds).
 * If the associated GPU is the master GPU this value will be recorded
 * by the VBIOS and retrieved from the KDA buffer.  If the associated
 * GPU is a secondaryGPU then this value will reflect the devinit
 * processing time.
 *
 * vbiosPostTime
 *   This parameter returns the vbios post time in msec.
 *
 * Possible return status values are
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW2080_CTRL_CMD_BIOS_GET_POST_TIME (0x20800809) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BIOS_INTERFACE_ID << 8) | LW2080_CTRL_CMD_BIOS_GET_POST_TIME_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_BIOS_GET_POST_TIME_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW2080_CTRL_CMD_BIOS_GET_POST_TIME_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 vbiosPostTime, 8);
} LW2080_CTRL_CMD_BIOS_GET_POST_TIME_PARAMS;


/*
 * LW2080_CTRL_BIOS_MAX_SUBIMAGES
 *      Maximum number of subimages inside a super image.
 */
#define LW2080_CTRL_BIOS_MAX_SUBIMAGES (0x08)

/*
 * LW2080_CTRL_BIOS_SOURCE_INFO
 *
 * This structure maintains debugging information recorded while attempting
 * to extract a VBIOS image for a single source.
 *
 *   status
 *     This value represents the status of extraction for each source.
 *     The status member is initialized to LW2080_CTRL_BIOS_INFO_STATUS_UNSET.
 *     Legal values are of the type LW2080_CTRL_BIOS_INFO_STATUS_*.
 *
 *   nImages
 *     This field indicates the number of subimages found.
 *
 *   bLastImage, offset, imageLength
 *     These fields record the position and length of each subimage along
 *     with the bLastImage boolean.
 *
 *   superImageLength
 *     This field records the superImageLength of the BIOS image for the
 *     current source.
 *
 *   romSig
 *     This field records The ROM signature for the BIOS.
 *     This field is valid only when there is an error indicated with by a
 *     status value of LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_PCI_ROM_SIG.
 *
 *   pciDataSig
 *     This field records the PCI Data Signature for the BIOS.
 *     This field is valid only when there is an error indicated by a
 *     status value of LW2080_CTRL_BIOS_INFO_STATUS_ILWALID_PCI_DATA_SIG.
 */
typedef struct LW2080_CTRL_BIOS_SOURCE_INFO {
    LwU8   status;

    LwU8   nImages;
    LwBool bLastImage[LW2080_CTRL_BIOS_MAX_SUBIMAGES];
    LwU32  offset[LW2080_CTRL_BIOS_MAX_SUBIMAGES];
    LwU32  imageLength[LW2080_CTRL_BIOS_MAX_SUBIMAGES];

    LwU32  superImageLength;
    LwU32  romSig;
    LwU32  pciDataSig;
} LW2080_CTRL_BIOS_SOURCE_INFO;

/*
 * LW2080_CTRL_BIOS_EXTRACTION_INFO
 *
 * This structure maintains debugging information recorded while attempting
 * to extract a VBIOS image for all possible sources.
 *
 *   source
 *     This parameter stores the source for the VBIOS image.  Legal values
 *     for this parameter include:
 *     LW2080_CTRL_BIOS_SRC_UNSET
 *         This value indicates the source is not set.  This is the default
 *         value.
 *     LW2080_CTRL_BIOS_SRC_NOT_FOUND
 *         This value indicates no image source was found.
 *     LW2080_CTRL_BIOS_SRC_REGISTRY
 *         This value indicates the image source is the registry.
 *     LW2080_CTRL_BIOS_SRC_SW
 *         This value indicates the image source is software.
 *     LW2080_CTRL_BIOS_SRC_GPUMGR_CACHE
 *         This value indicates the image source is the GPUMGR cache.
 *     LW2080_CTRL_BIOS_SRC_INSTANCE_MEMORY
 *         This value indicates the image source is instance memory.
 *     LW2080_CTRL_BIOS_SRC_ACPI_ROM
 *         This value indicates the image source is the ACPI ROM.
 *     LW2080_CTRL_BIOS_SRC_ROM
 *         This value indicates the image source is the ROM.
 *     LW2080_CTRL_BIOS_SRC_SBIOS
 *         This value indicates the image source is the SBIOS.
 *
 *   srcInfoReg
 *     This structure contains extraction state for the registry.
 *   srcInfoCachedVbios
 *     This structure contains extraction state for the VBIOS cache.
 *   srcInfoInstanceMem
 *     This structure contains extraction state for instance memory.
 *   srcInfoRom
 *     This structure contains extraction state for the ROM.
 *   srcInfoAcpiRom
 *     This structure contains extraction state for the ACPI ROM.
 *   srcInfoAcpiSbio
 *     This structure contains extraction state for the SBIOS.
 *
 *   statusExpansionRom
 *     This field stores the state of extraction for the expansion ROM.
 *     The status member is initialized to LW2080_CTRL_BIOS_INFO_STATUS_UNSET.
 *     Legal values are of the type LW2080_CTRL_BIOS_INFO_STATUS_*.
 *
 */
typedef struct LW2080_CTRL_BIOS_EXTRACTION_INFO {
    LwU8                         source;

    LW2080_CTRL_BIOS_SOURCE_INFO srcInfoReg;
    LW2080_CTRL_BIOS_SOURCE_INFO srcInfoCachedVbios;
    LW2080_CTRL_BIOS_SOURCE_INFO srcInfoInstanceMem;
    LW2080_CTRL_BIOS_SOURCE_INFO srcInfoRom;
    LW2080_CTRL_BIOS_SOURCE_INFO srcInfoAcpiRom;
    LW2080_CTRL_BIOS_SOURCE_INFO srcInfoSbios;

    LwU8                         statusExpansionRom;
} LW2080_CTRL_BIOS_EXTRACTION_INFO;

/* Valid VBIOS source values */
#define LW2080_CTRL_BIOS_SRC_UNSET               (0x00000000)
#define LW2080_CTRL_BIOS_SRC_NOT_FOUND           (0x00000001)
#define LW2080_CTRL_BIOS_SRC_REGISTRY            (0x00000002)
#define LW2080_CTRL_BIOS_SRC_SW                  (0x00000003)
#define LW2080_CTRL_BIOS_SRC_GPUMGR_CACHE        (0x00000004)
#define LW2080_CTRL_BIOS_SRC_INSTANCE_MEMORY     (0x00000005)
#define LW2080_CTRL_BIOS_SRC_ACPI_ROM            (0x00000006)
#define LW2080_CTRL_BIOS_SRC_ROM                 (0x00000007)
#define LW2080_CTRL_BIOS_SRC_SBIOS               (0x00000008)

/*
 * LW2080_CTRL_CMD_BIOS_GET_EXTRACTION_INFO
 *
 * This command returns debugging information for VBIOS extraction.
 *
 *   vbiosExtractionInfo
 *     Structure which contains debugging information collected during VBIOS
 *     extraction.  See description of LW2080_CTRL_BIOS_EXTRACTION_INFO
 *     for more information.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW2080_CTRL_CMD_BIOS_GET_EXTRACTION_INFO (0x2080080a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BIOS_INTERFACE_ID << 8) | LW2080_CTRL_BIOS_GET_EXTRACTION_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BIOS_GET_EXTRACTION_INFO_PARAMS_MESSAGE_ID (0xAU)

typedef struct LW2080_CTRL_BIOS_GET_EXTRACTION_INFO_PARAMS {
    LW2080_CTRL_BIOS_EXTRACTION_INFO vbiosExtractionInfo;
} LW2080_CTRL_BIOS_GET_EXTRACTION_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_BIOS_GET_UEFI_SUPPORT
 *
 * This function is used to give out the UEFI version, UEFI image presence and
 * Graphics Firmware Mode i.e. whether system is running in UEFI or not.
 *
 *   version
 *     This parameter returns the UEFI version.
 *
 *   flags
 *     This parameter indicates UEFI image presence and Graphics Firmware mode.
 *       LW2080_CTRL_BIOS_UEFI_SUPPORT_FLAGS_PRESENCE
 *         This field returns UEFI presence value. Legal values for this
 *         field include:
 *           LW2080_CTRL_BIOS_UEFI_SUPPORT_FLAGS_PRESENCE_NO
 *             This value indicates that UEFI image is not present.
 *           LW2080_CTRL_BIOS_UEFI_SUPPORT_FLAGS_PRESENCE_YES
 *             This value indicates that UEFI image is present.
 *           LW2080_CTRL_BIOS_UEFI_SUPPORT_FLAGS_PRESENCE_PLACEHOLDER
 *             This value indicates that there is a dummy UEFI placeholder,
 *             which can later be updated with a valid UEFI image.
 *           LW2080_CTRL_BIOS_UEFI_SUPPORT_FLAGS_PRESENCE_HIDDEN
 *             This value indicates that UEFI image is hidden.
 *       LW2080_CTRL_BIOS_UEFI_SUPPORT_FLAGS_RUNNING
 *         This field indicates the UEFI running value. Legal values for
 *         this parameter include:
 *           LW2080_CTRL_BIOS_UEFI_SUPPORT_FLAGS_RUNNING_FALSE
 *             This value indicates that UEFI is not running.
 *           LW2080_CTRL_BIOS_UEFI_SUPPORT_FLAGS_RUNNING_TRUE
 *             This value indicates that UEFI is running.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_READY
 *   LW_ERR_ILWALID_STATE
 */

#define LW2080_CTRL_CMD_BIOS_GET_UEFI_SUPPORT (0x2080080b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BIOS_INTERFACE_ID << 8) | LW2080_CTRL_BIOS_GET_UEFI_SUPPORT_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BIOS_GET_UEFI_SUPPORT_PARAMS_MESSAGE_ID (0xBU)

typedef struct LW2080_CTRL_BIOS_GET_UEFI_SUPPORT_PARAMS {
    LwU32 version;
    LwU32 flags;
} LW2080_CTRL_BIOS_GET_UEFI_SUPPORT_PARAMS;

/* Legal values for flags parameter */
#define LW2080_CTRL_BIOS_UEFI_SUPPORT_FLAGS_PRESENCE               1:0
#define LW2080_CTRL_BIOS_UEFI_SUPPORT_FLAGS_PRESENCE_NO          (0x00000000)
#define LW2080_CTRL_BIOS_UEFI_SUPPORT_FLAGS_PRESENCE_YES         (0x00000001)
#define LW2080_CTRL_BIOS_UEFI_SUPPORT_FLAGS_PRESENCE_PLACEHOLDER (0x00000002)
#define LW2080_CTRL_BIOS_UEFI_SUPPORT_FLAGS_PRESENCE_HIDDEN      (0x00000003)
#define LW2080_CTRL_BIOS_UEFI_SUPPORT_FLAGS_RUNNING                2:2
#define LW2080_CTRL_BIOS_UEFI_SUPPORT_FLAGS_RUNNING_FALSE        (0x00000000)
#define LW2080_CTRL_BIOS_UEFI_SUPPORT_FLAGS_RUNNING_TRUE         (0x00000001)

/*
 * LW2080_CTRL_CMD_BIOS_GET_IMAGE_HASH
 *
 * This command returns the VBIOS image hash.
 *
 *   imageHash
 *     Structure that stores the image hash.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_BIOS_GET_IMAGE_HASH                      (0x2080080c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BIOS_INTERFACE_ID << 8) | LW2080_CTRL_BIOS_GET_IMAGE_HASH_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BIOS_IMAGE_HASH_SIZE                         (0x10)
#define LW2080_CTRL_BIOS_GET_IMAGE_HASH_PARAMS_MESSAGE_ID (0xLW)

typedef struct LW2080_CTRL_BIOS_GET_IMAGE_HASH_PARAMS {
    LwU8 imageHash[LW2080_CTRL_BIOS_IMAGE_HASH_SIZE];
} LW2080_CTRL_BIOS_GET_IMAGE_HASH_PARAMS;

/*
 * LW2080_CTRL_CMD_BIOS_GET_BUILD_GUID
 *
 * This command returns the VBIOS build GUID which is unique per build
 * Bug 200291210 has more details about the control command requirement
 *
 *   buildGuid
 *     Structure that stores the vbios build guid
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_BIOS_GET_BUILD_GUID (0x2080080d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BIOS_INTERFACE_ID << 8) | LW2080_CTRL_BIOS_GET_BUILD_GUID_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BIOS_BUILD_GUID_SIZE    (0x10)
#define LW2080_CTRL_BIOS_GET_BUILD_GUID_PARAMS_MESSAGE_ID (0xDU)

typedef struct LW2080_CTRL_BIOS_GET_BUILD_GUID_PARAMS {
    LwU8 buildGuid[LW2080_CTRL_BIOS_BUILD_GUID_SIZE];
} LW2080_CTRL_BIOS_GET_BUILD_GUID_PARAMS;

/*!
 * Type of physical hardware register supported in VFIELD Table
 */
#define LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_TYPE_ILWALID        0x00000000   // Invalid register type
#define LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_TYPE_REG            0x00000001   // Normal LW register (at dword register "reg")
#define LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_TYPE_INDEX_REG      0x00000002   // Normal LW Index register (at index register "regindex" and data register "reg" with index "index")
#define LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_TYPE_IO_REG         0x00000003   // Normal I/O register
#define LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_TYPE_IO_INDEX_REG   0x00000004   // I/O Index register
#define LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_TYPE_IO_INDEX_A_REG 0x00000005   // I/O Index Head A register
#define LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_TYPE_IO_INDEX_B_REG 0x00000006   // I/O Index Head B register
#define LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_TYPE_MAX            0x00000007

// Known Field ID's
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_CRYSTAL                     0x00
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_TV_MODE                     0x01
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_FP_IFACE                    0x02
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_PANEL                       0x03
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_RAMCFG                      0x04  // Called VFIELDID_STRAP_MEM in the vbios.  Memory strap.
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_MEMSEL                      0x05  // Also memory strap.  Why do we need both???
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_UNUSED_0                    0x06  // Deprecated: VFIELD_ID_STRAP_THERMAL_BIN
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_SPEEDO                      0x07  // Speedo value
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_SPEEDO_VERSION              0x08  // Speedo version
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_IDDQ                        0x09  // IDDQ value for voltage rail 0
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_IDDQ_VERSION                0x0A  // IDDQ version
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_IDDQ_1                      0x0B  // IDDQ value for voltage rail 1
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_BOARD_BINNING               0x0C  // Speedo value for Board Binning
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_BOARD_BINNING_VERSION       0x0D  // Board Binning version
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_SRAM_VMIN                   0x0E  // SRAM Vmin
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_SRAM_VMIN_VERSION           0x0F  // SRAM Vmin version
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_BOOT_VMIN_LWVDD             0x10  // LWVDD Boot Vmin
#define LW2080_CTRL_BIOS_VFIELD_ID_ISENSE_VCM_OFFSET                 0x11  // ISENSE VCM Offset
#define LW2080_CTRL_BIOS_VFIELD_ID_ISENSE_DIFF_GAIN                  0x12  // ISENSE Differential Gain
#define LW2080_CTRL_BIOS_VFIELD_ID_ISENSE_DIFF_OFFSET                0x13  // ISENSE Differential Offset
#define LW2080_CTRL_BIOS_VFIELD_ID_ISENSE_CALIBRATION_VERSION        0x14  // ISENSE Calibration version. This is a common version for the 3 fields above
#define LW2080_CTRL_BIOS_VFIELD_ID_KAPPA                             0x15  // KAPPA fuse - Will link to fuse opt_kappa_info
#define LW2080_CTRL_BIOS_VFIELD_ID_KAPPA_VERSION                     0x16  // KAPPA version.
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_SPEEDO_1                    0x17  // SPEEDO_1
#define LW2080_CTRL_BIOS_VFIELD_ID_CPM_VERSION                       0x18  // Fuse OPT_CPM_REV
#define LW2080_CTRL_BIOS_VFIELD_ID_CPM_0                             0x19  // Fuse OPT_CPM0
#define LW2080_CTRL_BIOS_VFIELD_ID_CPM_1                             0x1A  // Fuse OPT_CPM1
#define LW2080_CTRL_BIOS_VFIELD_ID_CPM_2                             0x1B  // Fuse OPT_CPM2
#define LW2080_CTRL_BIOS_VFIELD_ID_ISENSE_VCM_COARSE_OFFSET          0x1C  // ISENSE VCM Coarse Offset
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_BOOT_VMIN_MSVDD             0x1D  // MSVDD Boot Vmin
#define LW2080_CTRL_BIOS_VFIELD_ID_KAPPA_VALID                       0x1E  // KAPPA fuse
#define LW2080_CTRL_BIOS_VFIELD_ID_IDDQ_LWVDD                        0x1F  // LWVDD IDDQ
#define LW2080_CTRL_BIOS_VFIELD_ID_IDDQ_MSVDD                        0x20  // MSVDD IDDQ
#define LW2080_CTRL_BIOS_VFIELD_ID_STRAP_SPEEDO_2                    0x21  // SPEEDO_2
#define LW2080_CTRL_BIOS_VFIELD_ID_OC_BIN                            0x22  // OC_BIN
#define LW2080_CTRL_BIOS_VFIELD_ID_LV_FMAX_KNOB                      0x23  // LV_FMAX_KNOB
#define LW2080_CTRL_BIOS_VFIELD_ID_MV_FMAX_KNOB                      0x24  // MV_FMAX_KNOB
#define LW2080_CTRL_BIOS_VFIELD_ID_HV_FMAX_KNOB                      0x25  // HV_FMAX_KNOB
#define LW2080_CTRL_BIOS_VFIELD_ID_PSTATE_VMIN_KNOB                  0x26  // PSTATE_VMIN_KNOB
#define LW2080_CTRL_BIOS_VFIELD_ID_ATEKAPPA0                         0x27  // ATEKAPPA0
#define LW2080_CTRL_BIOS_VFIELD_ID_ATEKAPPA1                         0x28  // ATEKAPPA1
#define LW2080_CTRL_BIOS_VFIELD_ID_ATEKAPPA2                         0x29  // ATEKAPPA2
#define LW2080_CTRL_BIOS_VFIELD_ID_ATEKAPPA_VALID                    0x2A  // ATEKAPPA_VALID
#define LW2080_CTRL_BIOS_VFIELD_ID_ATEKAPPA_REV                      0x2B  // ATEKAPPA_REV
#define LW2080_CTRL_BIOS_VFIELD_ID_ALT_MARKER0                       0x2C  // ALT_MARKER0
#define LW2080_CTRL_BIOS_VFIELD_ID_ALT_MARKER1                       0x2D  // ALT_MARKER1
#define LW2080_CTRL_BIOS_VFIELD_ID_ALT_MARKER2                       0x2E  // ALT_MARKER2
#define LW2080_CTRL_BIOS_VFIELD_ID_ALT_MARKER_REV                    0x2F  // ALT_MARKER_REV
#define LW2080_CTRL_BIOS_VFIELD_ID_ALT_MARKER0_CHOICE                0x30  // ALT_MARKER0_CHOICE
#define LW2080_CTRL_BIOS_VFIELD_ID_ALT_MARKER1_CHOICE                0x31  // ALT_MARKER1_CHOICE
#define LW2080_CTRL_BIOS_VFIELD_ID_ALT_MARKER2_CHOICE                0x32  // ALT_MARKER2_CHOICE
#define LW2080_CTRL_BIOS_VFIELD_ID_ILWALID                           0xFF
/*!
 * Base type of all vfield register segments
 */
typedef struct LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_SUPER {
    /*!
     * VFIELD_BIT_START: Lowest bit of the current segment
     */
    LwU8 lowBit;
    /*!
     * VFIELD_BIT_STOP: Highest bit of the current segment
     */
    LwU8 highBit;
} LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_SUPER;
typedef struct LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_SUPER *PLW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_SUPER;

/*!
 * Defines a register segment in the vfield table
 */
typedef struct LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_REG {
    LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_SUPER super;

    /*!
     * Register address to read for the current segment
     */
    LwU32                                          addr;
} LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_REG;
typedef struct LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_REG *PLW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_REG;

/*!
 * Defines indexed register segment in the vfield table
 */
typedef struct LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_INDEX_REG {
    LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_SUPER super;

    /*!
     * Register address to read for the current segment
     */
    LwU32                                          addr;
    /*!
     * Address of the indexed register to use
     */
    LwU32                                          regIndex;
    /*!
     * Index to store in "regIndex" to access the data at "addr"
     */
    LwU32                                          index;
} LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_INDEX_REG;
typedef struct LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_INDEX_REG *PLW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_INDEX_REG;

/*!
 * Union of all types of VFields lwrrently supported (LW_REG and LW_INDEX_REG)
 * See https://wiki.lwpu.com/engwiki/index.php/VBIOS/Data_Structures/Virtual_Register_Fields#Virtual_Field_Table
 */


typedef struct LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT {
    /*!
     * Type is one of LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_TYPE_*
     */
    LwU8 type;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_REG       reg;
        LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT_INDEX_REG indexReg;
    } data;
} LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT;
typedef struct LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT *PLW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT;

/*
 * LW2080_CTRL_CMD_BIOS_GET_PROJECT_INFO
 *
 * This command returns bios project information for the associated GPU.
 *  - projectId
 *  - partner
 *  - sessionid
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_BIOS_GET_PROJECT_INFO (0x2080080f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BIOS_INTERFACE_ID << 8) | LW2080_CTRL_BIOS_GET_PROJECT_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BIOS_MAX_PROJECTID_LENGTH 7   // 6  bytes + '\0'
#define LW2080_CTRL_BIOS_MAX_PARTNER_LENGTH   21  // 20 bytes + '\0'
#define LW2080_CTRL_BIOS_MAX_SESSIONID_LENGTH 21  // 20 bytes + '\0'

#define LW2080_CTRL_BIOS_GET_PROJECT_INFO_PARAMS_MESSAGE_ID (0xFU)

typedef struct LW2080_CTRL_BIOS_GET_PROJECT_INFO_PARAMS {
    LwU8 projectId[LW2080_CTRL_BIOS_MAX_PROJECTID_LENGTH];
    LwU8 partner[LW2080_CTRL_BIOS_MAX_PARTNER_LENGTH];
    LwU8 sessionId[LW2080_CTRL_BIOS_MAX_SESSIONID_LENGTH];
} LW2080_CTRL_BIOS_GET_PROJECT_INFO_PARAMS;

/* _ctrl2080bios_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

