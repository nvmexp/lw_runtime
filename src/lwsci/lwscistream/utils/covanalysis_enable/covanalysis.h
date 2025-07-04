//! \file
//! \brief LwSciStream macros for Coverity static analysis support
//!
//! \copyright
//! Copyright (c) 2019-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef COVANALYSIS_H
#define COVANALYSIS_H
// Static analysis macros enabled when coverity is in use.
//   These were based on those in lwos_static_analysis.h but don't depend
//   on LWOS and have support for more than just single line deviations.

// The defines themselves are coverity violations, so we manually add a
//   pragma around the code that defines the pragmas
_Pragma("coverity compliance block deviate AUTOSAR_CPP14_A16_0_1 'meta'");

// Utility to generate the pragma string
#define LWCOV_PRAGMA(str) _Pragma(#str)

// Generate full autosar rule name
//   Input should be underscore separated rule (e.g. A1_2_3 or M4_5_6)
#define LWCOV_AUTOSAR(rule) AUTOSAR_CPP14_##rule
#define LWCOV_CERTCPP(rule) CERT_##rule
#define LWCOV_CERTC(rule) CERT_##rule

// For all allowlist macros, the rule should be the full name of a rule,
//   as generated by above macro, and the comment should be quoted string
//   with one of the following forms:
//   * "Requested TID-123" : The Jira ID of a requested deviation
//   * "Approved TID-456"  : The Jira ID of an approved deviation
//   * "Bug 12345678"      : The LwBug number for a coverity bug

// Used to allowlist a single line of code. This will only affect the very
//   next line (not the next statement) so is only for very simple deviations.
#define LWCOV_ALLOWLIST_LINE(rule, comment) \
    LWCOV_PRAGMA(coverity compliance deviate rule comment)

// Used to allowlist a block of code with no include directives.
#define LWCOV_ALLOWLIST_BEGIN(rule, comment) \
    LWCOV_PRAGMA(coverity compliance block deviate rule comment)
#define LWCOV_ALLOWLIST_END(rule) \
    LWCOV_PRAGMA(coverity compliance end_block rule)

// Used to allowlist a block of code and any relwrsively included files
#define LWCOV_ALLOWLIST_BEGIN_INC(rule, comment) \
    LWCOV_PRAGMA(coverity compliance block(include) deviate rule comment)
#define LWCOV_ALLOWLIST_END_INC(rule) \
    LWCOV_PRAGMA(coverity compliance end_block(include) rule)

// End the block guarding the macro definitions
_Pragma("coverity compliance end_block AUTOSAR_CPP14_A16_0_1");

#endif // COVANALYSIS_H
