//
//  Copyright (c) 2020 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#pragma once

#include <utility>
#include <vector>

#include <ptxIR.h>
#include <ptxInstructions.h>

// Id of not fully supported types inside the ptxparser. This is needed for WAR to support these types properly
// during intrinsics' generation.
enum UnsupportedType
{
    UNSUPPORTED_TYPE_A16,
    UNSUPPORTED_TYPE_A32,
    UNSUPPORTED_TYPE_T32,
    UNSUPPORTED_TYPE_UNDEF
};

// Each of the two lists contains templates together with a potentially empty index types list containing unsupported types.
void getInstructionTemplates( ptxParseData                                                                  parseData,
                              std::vector<std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>>& stdTemplates,
                              std::vector<std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>>& extTemplates );
