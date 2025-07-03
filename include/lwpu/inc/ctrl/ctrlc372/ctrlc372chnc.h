/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrlc372/ctrlc372chnc.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "lwdisptypes.h"
#include "ctrl/ctrlc372/ctrlc372base.h"

#define LWC372_CTRL_MAX_POSSIBLE_HEADS   8
#define LWC372_CTRL_MAX_POSSIBLE_WINDOWS 32

#define LWC372_CTRL_CMD_IS_MODE_POSSIBLE (0xc3720101) /* finn: Evaluated from "(FINN_LWC372_DISPLAY_SW_CHNCTL_INTERFACE_ID << 8) | LWC372_CTRL_IS_MODE_POSSIBLE_PARAMS_MESSAGE_ID" */

/*
 * LWC372_CTRL_CMD_IS_MODE_POSSIBLE
 *
 * This command tells whether or not the specified display config is possible.
 * A config might not be possible if the display requirements exceed the GPU
 * capabilities.  Display requirements will be higher with more display
 * surfaces, higher resolutions, higher downscaling factors, etc.  GPU
 * capabilities are determined by clock frequencies, the width of data pipes,
 * amount of mempool available, number of thread groups available, etc.
 *
 * Inputs:
 *   head.headIndex
 *     This is the hardware index number for the head.  Only active heads
 *     should be included in the input structure.
 *
 *   head.maxPixelClkKHz
 *     This parameter specifies the pixel scanout rate (in KHz).
 *
 *   head.rasterSize
 *     This structure specifies the total width and height of the raster that
 *     is sent to the display.  (The width and height are also referred to as
 *     HTotal and VTotal, respectively.)
 *
 *   head.rasterBlankStart
 *     X specifies the pixel column where horizontal blanking begins;
 *     Y specifies the pixel row where vertical blanking begins.
 *
 *   head.rasterBlankEnd
 *     X specifies the pixel column where horizontal blanking ends;
 *     Y specifies the pixel row where vertical blanking ends.
 *
 *   head.rasterVertBlank2
 *     X and Y specify the pixel column/row where horizontal/vertical blanking
 *     ends on the second field of every pair for an interlaced raster.  This
 *     field is not used when the raster is progressive.
 *
 *   head.control.masterLockMode
 *   head.control.masterLockPin
 *   head.control.slaveLockMode
 *   head.control.slaveLockPin
 *     Heads that are raster locked or frame locked together will have
 *     synchronized timing.  For example, vblank will occur at the same time on
 *     all of the heads that are locked together.
 *
 *     "LockMode" tells if a head is raster locked, frame locked, or not locked.
 *
 *     "LockPin" tells which heads are in a group of locked heads.  There
 *     should be one master per group, and all slave heads that are locked to
 *     that master should have the same slaveLockPin number as the master's
 *     masterLockPin number.
 *
 *     Note: The LockModes and LockPins are used only if the min v-pstate is
 *     required (i.e., if LWC372_CTRL_IS_MODE_POSSIBLE_OPTIONS_NEED_MIN_VPSTATE
 *     is set).
 * 
 *   head.maxDownscaleFactorH
 *   head.maxDownscaleFactorV
 *     maxDownscaleFactorH and maxDownscaleFactorV represent the maximum amount
 *     by which the the composited image can be reduced in size, horizontally
 *     and vertically, respectively, multiplied by 0x400.  For example, if the
 *     scaler input width is 1024, and the scaler output width is 2048, the
 *     downscale factor would be 1024 / 2048 = 0.5, and multiplying by 0x400
 *     would give 512.
 *
 *   head.outputScalerVerticalTaps
 *     scalerVerticalTaps indicates the maximum number of vertical taps 
 *     allowed in the output scaler.
 * 
 *     Note that there are no #defines for tap values; the parameter is simply
 *     the number of taps (e.g., "2" for 2 taps).
 *
 *   head.bUpscalingAllowedV
 *     bUpscalingAllowed indicates whether or not the composited image can be
 *     increased in size, vertically.
 * 
 *   head.bOverfetchEnabled
 *     bOverfetchEnabled indicates whether or not the vertical overfetch is 
 *     enabled in postcomp scaler.
 *
 *   head.minFrameIdle.leadingRasterLines
 *     leadingRasterLines defines the number of lines between the start of the
 *     frame (vsync) and the start of the active region.  This includes Vsync,
 *     Vertical Back Porch, and the top part of the overscan border.  The
 *     minimum value is 2 because vsync and VBP must be at least 1 line each.
 *
 *   head.minFrameIdle.trailingRasterLines
 *     trailingRasterLines defines the number of lines between the end of the
 *     active region and the end of the frame.  This includes the bottom part
 *     of the overscan border and the Vertical Front Porch.
 *
 *   head.lut
 *     This parameter specifies whether or not the output LUT is enabled, and
 *     the size of the LUT.  The parameter should be an
 *     LWC372_CTRL_IMP_LUT_USAGE_xxx value.
 *
 *   head.lwrsorSize32p
 *     This parameter specifies the width of the cursor, in units of 32 pixels.
 *     So, for example, "8" would mean 8 * 32 = 256, for a 256x256 cursor.  Zero
 *     means the cursor is disabled.
 *
 *   head.bEnableDsc
 *     bEnableDsc indicates whether or not DSC is enabled
 *
 *   head.bYUV420Format
 *     This parameter indicates output format is YUV420.
 *     Refer to LWD_YUV420_Output_Functional_Description.docx for more details.
 *
 *   head.bIs2Head1Or
 *     This parameter specifies if the head operates in 2Head1Or mode.
 *     Refer to LWD_2_Heads_Driving_1_OR_Functional_Description.docx for more details.
 * 
 *   head.bDisableMidFrameAndDWCFWatermark
 *     WAR for bug 200508242. 
 *     In linux it is possible that there will be no fullscreen window visible 
 *     for a head. For these cases we would not hit dwcf or midframe watermarks 
 *     leading to fbflcn timing out waiting on ok_to_switch and forcing mclk 
 *     switch. This could lead to underflows. So if that scenario is caught (by
 *     Display Driver) bDisableMidFrameAndDWCFWatermark will be set to true and 
 *     IMP will exclude dwcf and midframe contribution from the "is mclk switch 
 *     guaranteed" callwlation for the bandwidth clients of that head.
 *
 *   window.windowIndex
 *     This is the hardware index number for the window.  Only active windows
 *     should be included in the input structure.
 *
 *   window.owningHead
 *     This is the hardware index of the head that will receive the window's
 *     output.
 * 
 *   window.formatUsageBound
 *     This parameter is a bitmask of all possible non-rotated mode data
 *     formats (LWC372_CTRL_FORMAT_xxx values).
 *
 *   window.rotatedFormatUsageBound
 *     This parameter is a bitmask of all possible rotated mode data formats
 *     (LWC372_CTRL_FORMAT_xxx values).
 *
 *   window.maxPixelsFetchedPerLine
 *     This parameter defines the maximum number of pixels that may need to be
 *     fetched in a single line for this window.  Often, this can be set to the
 *     viewportSizeIn.Width.  But if the window is known to be clipped, such
 *     that an entire line will never be fetched, then this parameter can be
 *     set to the clipped size (to improve the chances of the mode being
 *     possible, or possible at a lower v-pstate).
 * 
 *     In some cases, the value of this parameter must be increased by a few
 *     pixels in order to account for scaling overfetch, input chroma overfetch
 *     (420/422->444), and/or chroma output low pass filter overfetch
 *     (444->422/420).  This value is chip dependent; refer to the
 *     MaxPixelsFetchedPerLine parameter in lwdClass_01.mfs for the exact
 *     value.  In no case does the maxPixelsFetchedPerLine value need to exceed
 *     the surface width.
 *
 *   window.maxDownscaleFactorH
 *   window.maxDownscaleFactorV
 *     maxDownscaleFactorH and maxDownscaleFactorV represent the maximum amount
 *     by which the the window image can be reduced in size, horizontally and
 *     vertically, respectively, multiplied by 
 *     LWC372_CTRL_SCALING_FACTOR_MULTIPLIER. For example,
 *     if the scaler input width is 1024, and the scaler output width is 2048,
 *     the downscale factor would be 1024 / 2048 = 0.5, and multiplying by 
 *     LWC372_CTRL_SCALING_FACTOR_MULTIPLIER if 0x400 would give 512.
 *
 *   window.inputScalerVerticalTaps
 *     scalerVerticalTaps indicates the maximum number of vertical taps 
 *     allowed in the input scaler.
 *
 *     Note that there are no #defines for tap values; the parameter is simply
 *     the number of taps (e.g., "2" for 2 taps).
 *
 *   window.bUpscalingAllowedV
 *     bUpscalingAllowed indicates whether or not the composited image can be
 *     increased in size, vertically.
 *
 *   window.bOverfetchEnabled
 *     bOverfetchEnabled indicates whether or not the vertical overfetch is 
 *     enabled in precomp scaler.
 *
 *   window.lut
 *     This parameter specifies whether or not the input LUT is enabled, and
 *     the size of the LUT.  The parameter should be an
 *     LWC372_CTRL_IMP_LUT_USAGE_xxx value.
 *
 *   window.tmoLut
 *     This parameter specifies whether or not the tmo LUT is enabled, and
 *     the size of the LUT. This lut is used for HDR.  The parameter should be
 *     an LWC372_CTRL_IMP_LUT_USAGE_xxx value.
 *
 *   numHeads
 *     This is the number of heads in the "head" array of the
 *     LWC372_CTRL_IS_MODE_POSSIBLE_PARAMS struct.  Only active heads should be
 *     included in the struct.
 *
 *   numWindows
 *     This is the number of windows in the "window" array of the
 *     LWC372_CTRL_IS_MODE_POSSIBLE_PARAMS struct.  Only active windows should
 *     be included in the struct.
 *
 *   options
 *     This parameter specifies a bitmask for options.
 *
 *       LWC372_CTRL_IS_MODE_POSSIBLE_OPTIONS_GET_MARGIN
 *         tells IMP to callwlate worstCaseMargin and worstCaseDomain.
 *       LWC372_CTRL_IS_MODE_POSSIBLE_OPTIONS_NEED_MIN_VPSTATE
 *         tells IMP to callwlate and report the minimum v-pstate at which the
 *         mode is possible.
 *
 *   bUseCachedPerfState
 *     Indicates that RM should use cached values for the fastest
 *     available perf level (v-pstate for PStates 3.0 or pstate for
 *     PStates 2.0) and dispclk.  This feature allows the query call to
 *     execute faster, and is intended to be used, for example, during
 *     mode enumeration, when many IMP query calls are made in close
 *     succession, and perf conditions are not expected to change between
 *     query calls.  When IMP has not been queried recently, it is
 *     recommended to NOT use cached values, in case perf conditions have
 *     changed and the cached values no longer reflect the current
 *     conditions.
 *
 *   testMclkFreqKHz
 *     This is the mclk frequency specified by the client, in KHz.  RM will 
 *     use this value to compare with the minimum dramclk required by the 
 *     given mode.  The parameter will have value 0 if the client doesn't want 
 *     IMP query to consider this. This input is valid only on CheetAh and only
 *     for verification purposes on internal builds.
 *     For this input to work, client must set 
 *     LWC372_CTRL_IS_MODE_POSSIBLE_OPTIONS_NEED_MIN_VPSTATE in the
 *     "options" field.
 *
 * Outputs:
 *   bIsPossible
 *     This output tells if the specified mode can be supported.
 *
 *   minImpVPState
 *     minImpVPState returns the minimum v-pstate at which the mode is possible
 *     (assuming bIsPossible is TRUE).  This output is valid only on dGPU, and
 *     only if LWC372_CTRL_IS_MODE_POSSIBLE_OPTIONS_NEED_MIN_VPSTATE was set in
 *     the "options" field.
 *
 *     If the minimum v-pstate is required for a multi-head config, then
 *     masterLockMode, masterLockPin, slaveLockMode, and slaveLockPin must all
 *     be initialized.
 *   minPState
 *     minPState returns the pstate value corresponding to minImpVPState.  It
 *     is returned as the numeric value of the pstate (P0 -> 0, P1 -> 1, etc.).
 *     This output is valid only on dGPU, and only if
 *     LWC372_CTRL_IS_MODE_POSSIBLE_OPTIONS_NEED_MIN_VPSTATE was set
 *     in the "options" field.
 *
 *     Note that the pstate returned by minPstateForGlitchless is not
 *     necessarily sufficient to meet IMP requirements.  The pstate corresponds
 *     to the vpstate returned by minImpVPState, and this vpstate represents
 *     clocks that are sufficient for IMP requirements, but the pstate
 *     typically covers a range of frequencies (depending on the clock), and it
 *     is possible that only part of the range is sufficient for IMP.
 *
 *   minRequiredBandwidthKBPS
 *     minRequiredBandwidthKBPS returns the minimum bandwidth that must be
 *     allocated to display in order to make the mode possible (assuming
 *     bIsPossible is TRUE).  This output is valid only on CheetAh, and only if
 *     LWC372_CTRL_IS_MODE_POSSIBLE_OPTIONS_NEED_MIN_VPSTATE was set in the
 *     "options" field.
 * 
 *   floorBandwidthKBPS
 *     floorBandwidthKBPS returns the minimum mclk frequency that can support
 *     the mode, and allow glitchless mclk switch, multiplied by the width of
 *     the data pipe.  (This is an approximation of the bandwidth that can be
 *     provided by the min required mclk frequency, ignoring overhead.)  If the
 *     mode is possible, but glitchless mclk switch is not, floorBandwidthKBPS
 *     will be callwlated based on the maximum possible mclk frequency.  This
 *     output is valid only on CheetAh, and only if
 *     LWC372_CTRL_IS_MODE_POSSIBLE_OPTIONS_NEED_MIN_VPSTATE was set in the
 *     "options" field.
 *
 *   minRequiredHubclkKHz
 *     minRequiredHubclkKHz returns the minimum hubclk frequency that can 
 *     support the mode.  This output is valid only on CheetAh, and only if 
 *     LWC372_CTRL_IS_MODE_POSSIBLE_OPTIONS_NEED_MIN_VPSTATE was set in the
 *     "options" field.
 * 
 *   worstCaseMargin
 *     worstCaseMargin returns the ratio of available bandwidth to required
 *     bandwidth, multiplied by LW5070_CTRL_IMP_MARGIN_MULTIPLIER.  Available
 *     bandwidth is callwlated in the worst case bandwidth domain, i.e., the
 *     domain with the least available margin.  Bandwidth domains include the
 *     IMP-relevant clock domains, and possibly other virtual bandwidth
 *     domains such as AWP.
 *
 *     Note that IMP checks additional parameters besides the bandwidth margins
 *     but only the bandwidth margin is reported here, so it is possible for a
 *     mode to have a more restrictive domain that is not reflected in the
 *     reported margin result.
 *
 *     This result is not guaranteed to be valid if the mode is not possible.
 *
 *     Note also that the result is generally callwlated for the highest
 *     v-pstate possible (usually P0).  But if the _NEED_MIN_VPSTATE is
 *     specified, the result will be callwlated for the min possible v-pstate
 *     (or the highest possible v-pstate, if the mode is not possible).
 * 
 *     The result is valid only if
 *     LW5070_CTRL_IS_MODE_POSSIBLE_OPTIONS_GET_MARGIN is set in "options".
 *
 *   dispClkKHz
 *     This is the dispclk frequency selected by IMP for this mode. For dGPU,
 *     it will be one of the fixed frequencies from the list of frequencies 
 *     supported by the vbios.
 *
 *   worstCaseDomain
 *     Returns a short text string naming the domain for the margin returned in
 *     "worstCaseMargin".  See "worstCaseMargin" for more information.
 *
 * Possible status values returned are:
 *     LWOS_STATUS_SUCCESS
 *     LWOS_STATUS_ERROR_GENERIC
 */
#define LWC372_CTRL_IMP_LUT_USAGE_NONE   0
#define LWC372_CTRL_IMP_LUT_USAGE_257    1
#define LWC372_CTRL_IMP_LUT_USAGE_1025   2

typedef struct LWC372_CTRL_IMP_HEAD {
    LwU8  headIndex;

    LwU32 maxPixelClkKHz;

    struct {
        LwU32 width;
        LwU32 height;
    } rasterSize;

    struct {
        LwU32 X;
        LwU32 Y;
    } rasterBlankStart;

    struct {
        LwU32 X;
        LwU32 Y;
    } rasterBlankEnd;

    struct {
        LwU32 yStart;
        LwU32 yEnd;
    } rasterVertBlank2;

    struct {
        LW_DISP_LOCK_MODE masterLockMode;
        LW_DISP_LOCK_PIN  masterLockPin;
        LW_DISP_LOCK_MODE slaveLockMode;
        LW_DISP_LOCK_PIN  slaveLockPin;
    } control;

    LwU32  maxDownscaleFactorH;
    LwU32  maxDownscaleFactorV;
    LwU8   outputScalerVerticalTaps;
    LwBool bUpscalingAllowedV;
    LwBool bOverfetchEnabled;

    struct {
        LwU16 leadingRasterLines;
        LwU16 trailingRasterLines;
    } minFrameIdle;

    LwU8   lut;
    LwU8   lwrsorSize32p;

    LwBool bEnableDsc;

    LwBool bYUV420Format;

    LwBool bIs2Head1Or;

    LwBool bDisableMidFrameAndDWCFWatermark;
} LWC372_CTRL_IMP_HEAD;
typedef struct LWC372_CTRL_IMP_HEAD *PLWC372_CTRL_IMP_HEAD;

typedef struct LWC372_CTRL_IMP_WINDOW {
    LwU32  windowIndex;
    LwU32  owningHead;
    LwU32  formatUsageBound;
    LwU32  rotatedFormatUsageBound;
    LwU32  maxPixelsFetchedPerLine;
    LwU32  maxDownscaleFactorH;
    LwU32  maxDownscaleFactorV;
    LwU8   inputScalerVerticalTaps;
    LwBool bUpscalingAllowedV;
    LwBool bOverfetchEnabled;
    LwU8   lut;
    LwU8   tmoLut;
} LWC372_CTRL_IMP_WINDOW;
typedef struct LWC372_CTRL_IMP_WINDOW *PLWC372_CTRL_IMP_WINDOW;

#define LWC372_CTRL_IS_MODE_POSSIBLE_OPTIONS_GET_MARGIN       (0x00000001)
#define LWC372_CTRL_IS_MODE_POSSIBLE_OPTIONS_NEED_MIN_VPSTATE (0x00000002)

#define LWC372_CTRL_IS_MODE_POSSIBLE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWC372_CTRL_IS_MODE_POSSIBLE_PARAMS {
    LWC372_CTRL_CMD_BASE_PARAMS base;

    LwU8                        numHeads;
    LwU8                        numWindows;

    LWC372_CTRL_IMP_HEAD        head[LWC372_CTRL_MAX_POSSIBLE_HEADS];

    // C form: LWC372_CTRL_IMP_WINDOW window[LWC372_CTRL_MAX_POSSIBLE_WINDOWS];
    LWC372_CTRL_IMP_WINDOW      window[LWC372_CTRL_MAX_POSSIBLE_WINDOWS];

    LwU32                       options;

    LwU32                       testMclkFreqKHz;

    LwBool                      bIsPossible;

    LwU32                       minImpVPState;

    LwU32                       minPState;

    LwU32                       minRequiredBandwidthKBPS;

    LwU32                       floorBandwidthKBPS;

    LwU32                       minRequiredHubclkKHz;

    LwU32                       worstCaseMargin;

    LwU32                       dispClkKHz;

    char                        worstCaseDomain[8];

    LwBool                      bUseCachedPerfState;
} LWC372_CTRL_IS_MODE_POSSIBLE_PARAMS;
typedef struct LWC372_CTRL_IS_MODE_POSSIBLE_PARAMS *PLWC372_CTRL_IS_MODE_POSSIBLE_PARAMS;

/* valid format values */
#define LWC372_CTRL_FORMAT_RGB_PACKED_1_BPP                           (0x00000001)
#define LWC372_CTRL_FORMAT_RGB_PACKED_2_BPP                           (0x00000002)
#define LWC372_CTRL_FORMAT_RGB_PACKED_4_BPP                           (0x00000004)
#define LWC372_CTRL_FORMAT_RGB_PACKED_8_BPP                           (0x00000008)
#define LWC372_CTRL_FORMAT_YUV_PACKED_422                             (0x00000010)
#define LWC372_CTRL_FORMAT_YUV_PLANAR_420                             (0x00000020)
#define LWC372_CTRL_FORMAT_YUV_PLANAR_444                             (0x00000040)
#define LWC372_CTRL_FORMAT_YUV_SEMI_PLANAR_420                        (0x00000080)
#define LWC372_CTRL_FORMAT_YUV_SEMI_PLANAR_422                        (0x00000100)
#define LWC372_CTRL_FORMAT_YUV_SEMI_PLANAR_422R                       (0x00000200)
#define LWC372_CTRL_FORMAT_YUV_SEMI_PLANAR_444                        (0x00000400)
#define LWC372_CTRL_FORMAT_EXT_YUV_PLANAR_420                         (0x00000800)
#define LWC372_CTRL_FORMAT_EXT_YUV_PLANAR_444                         (0x00001000)
#define LWC372_CTRL_FORMAT_EXT_YUV_SEMI_PLANAR_420                    (0x00002000)
#define LWC372_CTRL_FORMAT_EXT_YUV_SEMI_PLANAR_422                    (0x00004000)
#define LWC372_CTRL_FORMAT_EXT_YUV_SEMI_PLANAR_422R                   (0x00008000)
#define LWC372_CTRL_FORMAT_EXT_YUV_SEMI_PLANAR_444                    (0x00010000)

/* valid impResult values */
#define LWC372_CTRL_IMP_MODE_POSSIBLE                                 0
#define LWC372_CTRL_IMP_NOT_ENOUGH_MEMPOOL                            1
#define LWC372_CTRL_IMP_REQ_LIMIT_TOO_HIGH                            2
#define LWC372_CTRL_IMP_VBLANK_TOO_SMALL                              3
#define LWC372_CTRL_IMP_HUBCLK_TOO_LOW                                4
#define LWC372_CTRL_IMP_INSUFFICIENT_BANDWIDTH                        5
#define LWC372_CTRL_IMP_DISPCLK_TOO_LOW                               6
#define LWC372_CTRL_IMP_ELV_START_TOO_HIGH                            7
#define LWC372_CTRL_IMP_INSUFFICIENT_THREAD_GROUPS                    8
#define LWC372_CTRL_IMP_ILWALID_PARAMETER                             9
#define LWC372_CTRL_IMP_UNRECOGNIZED_FORMAT                           10
#define LWC372_CTRL_IMP_UNSPECIFIED                                   11

/*
 * The callwlated margin is multiplied by a constant, so that it can be
 * represented as an integer with reasonable precision.  "0x400" was chosen
 * because it is a power of two, which might allow some compilers/CPUs to
 * simplify the callwlation by doing a shift instead of a multiply/divide.
 * (And 0x400 is 1024, which is close to 1000, so that may simplify visual
 * interpretation of the raw margin value.)
 */
#define LWC372_CTRL_IMP_MARGIN_MULTIPLIER                             (0x00000400)

/* scaling factor */
#define LWC372_CTRL_SCALING_FACTOR_MULTIPLIER                         (0x00000400)

#define LWC372_CTRL_CMD_NUM_DISPLAY_ID_DWORDS_PER_HEAD                2
#define LWC372_CTRL_CMD_MAX_SORS                                      4

#define LWC372_CTRL_CMD_IS_MODE_POSSIBLE_OR_SETTINGS                  (0xc3720102) /* finn: Evaluated from "(FINN_LWC372_DISPLAY_SW_CHNCTL_INTERFACE_ID << 8) | LWC372_CTRL_IS_MODE_POSSIBLE_OR_SETTINGS_PARAMS_MESSAGE_ID" */

/* 
 * LWC372_CTRL_CMD_IS_MODE_POSSIBLE_OR_SETTINGS
 *
 * This command tells us if output resource pixel clocks requested by client
 * is possible or not. Note that this will not be used for displayport sor as 
 * it will be handled by displayport library.
 * 
 * Inputs: 
 *   numHeads
 *     This is the number of heads in the "head" array of the
 *     LWC372_CTRL_IS_MODE_POSSIBLE_OR_SETTINGS_PARAMS struct.  Only active heads 
 *     should be included in the struct.
 *
 *   head.headIndex
 *     This is the hardware index number for the head.  Only an active head
 *     should be included in the input structure.
 *
 *   head.maxPixelClkKHz
 *     This parameter specifies the pixel scanout rate (in KHz).
 *
 *   head.displayId
 *     Array of displayId's associated with the head. This is limited by
 *     LWC372_CTRL_CMD_NUM_DISPLAY_ID_DWORDS_PER_HEAD.
 *
 *   sor.ownerMask
 *     Consists of a mask of all heads that drive this sor.
 *
 *   sor.protocol
 *     Defines the protocol of the sor in question.
 *
 *   sor.pixelReplicateMode
 *     Defines which pixel replication mode is requested. This can be off
 *     or X2 or X4 mode.
 *
 * Outputs:
 *   bIsPossible
 *     This tells us that the requested pixel clock can be supported.
 */


#define LWC372_CTRL_IS_MODE_POSSIBLE_DISPLAY_ID_SKIP_IMP_OUTPUT_CHECK (0xAAAAAAAA)

typedef struct LWC372_CTRL_IMP_OR_SETTINGS_HEAD {
    LwU8                               headIndex;
    LwU32                              maxPixelClkKhz;

    LW_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP outputResourcePixelDepthBPP;

    LwU32                              displayId[LWC372_CTRL_CMD_NUM_DISPLAY_ID_DWORDS_PER_HEAD];
} LWC372_CTRL_IMP_OR_SETTINGS_HEAD;
typedef struct LWC372_CTRL_IMP_OR_SETTINGS_HEAD *PLWC372_CTRL_IMP_OR_SETTINGS_HEAD;

#define LWC372_CTRL_CMD_SOR_OWNER_MASK_NONE                   (0x00000000)
#define LWC372_CTRL_CMD_SOR_OWNER_MASK_HEAD(i)                          (1 << i)

#define LWC372_CTRL_CMD_SOR_PROTOCOL_SINGLE_TMDS_A            (0x00000000)
#define LWC372_CTRL_CMD_SOR_PROTOCOL_SINGLE_TMDS_B            (0x00000001)
#define LWC372_CTRL_CMD_SOR_PROTOCOL_DUAL_TMDS                (0x00000002)
#define LWC372_CTRL_CMD_SOR_PROTOCOL_SUPPORTED                (0xFFFFFFFF)

#define LWC372_CTRL_IS_MODE_POSSIBLE_PIXEL_REPLICATE_MODE_OFF (0x00000000)
#define LWC372_CTRL_IS_MODE_POSSIBLE_PIXEL_REPLICATE_MODE_X2  (0x00000001)
#define LWC372_CTRL_IS_MODE_POSSIBLE_PIXEL_REPLICATE_MODE_X4  (0x00000002)

typedef struct LWC372_CTRL_IMP_OR_SETTINGS_SOR {
    LwU32 ownerMask;
    LwU32 protocol;
    LwU32 pixelReplicateMode;
} LWC372_CTRL_IMP_OR_SETTINGS_SOR;
typedef struct LWC372_CTRL_IMP_OR_SETTINGS_SOR *PLWC372_CTRL_IMP_OR_SETTINGS_SOR;

#define LWC372_CTRL_IS_MODE_POSSIBLE_OR_SETTINGS_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWC372_CTRL_IS_MODE_POSSIBLE_OR_SETTINGS_PARAMS {
    LWC372_CTRL_CMD_BASE_PARAMS      base;

    LwU32                            numHeads;

    LWC372_CTRL_IMP_OR_SETTINGS_HEAD head[LWC372_CTRL_MAX_POSSIBLE_HEADS];

    LWC372_CTRL_IMP_OR_SETTINGS_SOR  sor[LWC372_CTRL_CMD_MAX_SORS];

    LwBool                           bIsPossible;
} LWC372_CTRL_IS_MODE_POSSIBLE_OR_SETTINGS_PARAMS;
typedef struct LWC372_CTRL_IS_MODE_POSSIBLE_OR_SETTINGS_PARAMS *PLWC372_CTRL_IS_MODE_POSSIBLE_OR_SETTINGS_PARAMS;

#define LWC372_CTRL_CMD_VIDEO_ADAPTIVE_REFRESH_RATE (0xc3720103) /* finn: Evaluated from "(FINN_LWC372_DISPLAY_SW_CHNCTL_INTERFACE_ID << 8) | LWC372_CTRL_CMD_VIDEO_ADAPTIVE_REFRESH_RATE_PARAMS_MESSAGE_ID" */

/*
 * LWC372_CTRL_CMD_VIDEO_ADAPTIVE_REFRESH_RATE
 *
 * This control call is used by clients to inform RM about video adaptive refresh rate enable/disable.
 * Based on the state, RM will enable/disable supported low power features.
 *
 * Inputs:
 *   displayID
 *      displayId of panel on which video adaptive refresh rate is enabled/disabled.
 *
 *   bEnable
 *      LW_TRUE to enable video adaptive refresh rate mode.
 *      LW_FALSE to disable video adaptive refresh rate mode.
 *
 * Outputs:
 *   Possible status values returned are:
 *      LW_OK
 *      LW_ERR_NOT_SUPPORTED
 */

#define LWC372_CTRL_CMD_VIDEO_ADAPTIVE_REFRESH_RATE_PARAMS_MESSAGE_ID (0x3U)

typedef struct LWC372_CTRL_CMD_VIDEO_ADAPTIVE_REFRESH_RATE_PARAMS {
    LwU32  displayID;
    LwBool bEnable;
} LWC372_CTRL_CMD_VIDEO_ADAPTIVE_REFRESH_RATE_PARAMS;
typedef struct LWC372_CTRL_CMD_VIDEO_ADAPTIVE_REFRESH_RATE_PARAMS *PLWC372_CTRL_CMD_VIDEO_ADAPTIVE_REFRESH_RATE_PARAMS;


#define LWC372_CTRL_CMD_GET_ACTIVE_VIEWPORT_POINT_IN (0xc3720104) /* finn: Evaluated from "(FINN_LWC372_DISPLAY_SW_CHNCTL_INTERFACE_ID << 8) | LWC372_CTRL_CMD_GET_ACTIVE_VIEWPORT_POINT_IN_PARAMS_MESSAGE_ID" */

/*
 * LWC372_CTRL_CMD_GET_ACTIVE_VIEWPORT_POINT_IN
 *
 * This control call is used by clients to query the active viewport for the
 * provided window precallwlated at the beginning of each frame.
 *
 * Inputs:
 *   windowIndex
 *      Index of the window to be queried.  Must be connected to an active head.
 *
 * Outputs:
 *   activeViewportPointIn
 *      X and Y coordinates of the active viewport on the provided window for
 *      the most recent frame.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT if the window index is invalid
 *      LW_ERR_ILWALID_STATE if the window index isn't connected to a head
 *      LW_ERR_NOT_SUPPORTED
 */
#define LWC372_CTRL_CMD_GET_ACTIVE_VIEWPORT_POINT_IN_PARAMS_MESSAGE_ID (0x4U)

typedef struct LWC372_CTRL_CMD_GET_ACTIVE_VIEWPORT_POINT_IN_PARAMS {
    LWC372_CTRL_CMD_BASE_PARAMS base;

    LwU32                       windowIndex;

    struct {
        LwU32 x;
        LwU32 y;
    } activeViewportPointIn;
} LWC372_CTRL_CMD_GET_ACTIVE_VIEWPORT_POINT_IN_PARAMS;
typedef struct LWC372_CTRL_CMD_GET_ACTIVE_VIEWPORT_POINT_IN_PARAMS *PLWC372_CTRL_CMD_GET_ACTIVE_VIEWPORT_POINT_IN_PARAMS;

/* _ctrlc372chnc_h_ */
