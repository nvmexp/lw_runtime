//
// Copyright (c) 2020, LWPU CORPORATION. All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.
//

#ifndef IVCLIB_STATIC_ANALYSIS_H
#define IVCLIB_STATIC_ANALYSIS_H

// Inline annotations to suppress Partial Deviations (PDs) and False Positives
// (FPs) reported by Coverity (since version 2019.06). Enabled as follows:
// $ LW_BUILD_CONFIGURATION_IS_COVERITY=1 tmp

#ifdef LW_IS_COVERITY
// Disable MISRA C violations caused by the leading underscore in macro names.
_Pragma("coverity compliance block (deviate MISRA_C_2012_Rule_21_1) (deviate MISRA_C_2012_Rule_21_2)")

#define _str(_x) #_x
// The following _stan*() helpers should not be used outside of this file.
_Pragma("coverity compliance fp AUTOSAR_Cpp14_M16_0_6") // _STAN_M16_0_6_FP_HYP_3734
#define _stanFP(_id)    _Pragma(_str(coverity compliance fp _id))
_Pragma("coverity compliance fp AUTOSAR_Cpp14_M16_0_6") // _STAN_M16_0_6_FP_HYP_3734
#define _stanPD(_id)    _Pragma(_str(coverity compliance deviate _id))
#else
#define _stanFP(_id)
#define _stanPD(_id)
#endif

// All pending and approved FPs and PDs for ivclib shall be listed below.

// CERT PDs
#define _STAN_EXP32_C_PD_HYP_4977   _stanPD(CERT_EXP32_C)
#define _STAN_INT30_C_PD_HYP_4978   _stanPD(CERT_INT30_C)

// AUTOSAR PDs
#define _STAN_A7_2_2_PD_HYP_4847    _stanPD(AUTOSAR_Cpp14_A7_2_2)
#define _STAN_A7_2_3_PD_HYP_4848    _stanPD(AUTOSAR_Cpp14_A7_2_3)
#define _STAN_M17_0_2_PD_HYP_4858   _stanPD(AUTOSAR_Cpp14_M17_0_2)

// AUTOSAR FPs
#define _STAN_A9_6_1_FP_HYP_4836    _stanFP(AUTOSAR_Cpp14_A9_6_1)
#define _STAN_M16_0_6_FP_HYP_3734   _stanFP(AUTOSAR_Cpp14_M16_0_6)

// MISRA C PDs
#define _STAN_4_7_PD_HYP_4979       _stanPD(MISRA_C_2012_Directive_4_7)
#define _STAN_8_6_PD_HYP_4976       _stanPD(MISRA_C_2012_Rule_8_6)
#define _STAN_11_1_PD_HYP_4975      _stanPD(MISRA_C_2012_Rule_11_1)
#define _STAN_11_8_PD_HYP_4971      _stanPD(MISRA_C_2012_Rule_11_8)
#define _STAN_21_1_PD_HYP_4974      _stanPD(MISRA_C_2012_Rule_21_1)
#define _STAN_21_15_PD_HYP_4973     _stanPD(MISRA_C_2012_Rule_21_15)

#ifdef LW_IS_COVERITY
_Pragma("coverity compliance end_block MISRA_C_2012_Rule_21_1 MISRA_C_2012_Rule_21_2")
#endif

#endif // include guard
