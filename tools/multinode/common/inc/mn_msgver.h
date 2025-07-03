/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#pragma once

// The 'major' version of messages is used to determine compatibility of messages
// between components.  Only tools compiled with the same major version are
// compatible.
#define MESSAGE_VERSION_MAJOR   1

// Tools compiled with the same 'minor' version are compatible and can communicate
// successfully, however some functionality not critical to internode communication
// may not be present
#define MESSAGE_VERSION_MINOR   0

