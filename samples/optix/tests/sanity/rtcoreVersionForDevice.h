//
//  Copyright (c) 2021 LWPU Corporation.  All rights reserved.
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

#include <optix_types.h>

#include <map>
#include <string>

std::map<std::string, int> rtcoreVersionForDevice {
	{ "1e30", 10 }, // TU102GL [Lwdqro RTX 6000/8000]
	{ "1e02", 10 }, // TU102 [TITAN RTX
	{ "1e04", 10 }, // TU102 [VdChip RTX 2080 Ti]
	{ "1e07", 10 }, // TU102 [VdChip RTX 2080 Ti Rev. A]
	{ "1e2d", 10 }, // TU102 [VdChip RTX 2080 Ti Engineering Sample]
	{ "1e2e", 10 }, // TU102 [VdChip RTX 2080 Ti 12GB Engineering Sample]
	{ "1e30", 10 }, // TU102GL [Lwdqro RTX 6000/8000]
	{ "129e", 10 }, // Lwdqro RTX 8000
	{ "12ba", 10 }, // Lwdqro RTX 6000
	{ "1e36", 10 }, // TU102GL [Lwdqro RTX 6000]
	{ "1e78", 10 }, // TU102GL [Lwdqro RTX 6000/8000]
	{ "13d8", 10 }, // Lwdqro RTX 8000
	{ "13d9", 10 }, // Lwdqro RTX 6000
	{ "1e81", 10 }, // TU104 [VdChip RTX 2080 SUPER]
	{ "1e82", 10 }, // TU104 [VdChip RTX 2080]
	{ "1e84", 10 }, // TU104 [VdChip RTX 2070 SUPER]
	{ "1e87", 10 }, // TU104 [VdChip RTX 2080 Rev. A]
	{ "1e89", 10 }, // TU104 [VdChip RTX 2060]
	{ "1e90", 10 }, // TU104M [VdChip RTX 2080 Mobile]
	{ "1e91", 10 }, // TU104M [VdChip RTX 2070 SUPER Mobile / Max-Q]
	{ "1e93", 10 }, // TU104M [VdChip RTX 2080 SUPER Mobile / Max-Q]
	{ "1eb0", 10 }, // TU104GL [Lwdqro RTX 5000]
	{ "1eb1", 10 }, // TU104GL [Lwdqro RTX 4000]
	{ "1eb5", 10 }, // TU104GLM [Lwdqro RTX 5000 Mobile / Max-Q]
	{ "1eb6", 10 }, // TU104GLM [Lwdqro RTX 4000 Mobile / Max-Q]
	{ "1ec2", 10 }, // TU104 [VdChip RTX 2070 SUPER]
	{ "1ec7", 10 }, // TU104 [VdChip RTX 2070 SUPER]
	{ "1ed0", 10 }, // TU104BM [VdChip RTX 2080 Mobile]
	{ "1ed1", 10 }, // TU104BM [VdChip RTX 2070 SUPER Mobile / Max-Q]
	{ "1ed3", 10 }, // TU104BM [VdChip RTX 2080 SUPER Mobile / Max-Q]
	{ "1ef5", 10 }, // TU104GLM [Lwdqro RTX 5000 Mobile Refresh]
	{ "1f02", 10 }, // TU106 [VdChip RTX 2070]
	{ "1f06", 10 }, // TU106 [VdChip RTX 2060 SUPER]
	{ "1f07", 10 }, // TU106 [VdChip RTX 2070 Rev. A]
	{ "1f08", 10 }, // TU106 [VdChip RTX 2060 Rev. A]
	{ "1f10", 10 }, // TU106M [VdChip RTX 2070 Mobile]
	{ "1f11", 10 }, // TU106M [VdChip RTX 2060 Mobile]
	{ "1f12", 10 }, // TU106M [VdChip RTX 2060 Max-Q]
	{ "1f14", 10 }, // TU106M [VdChip RTX 2070 Mobile / Max-Q Refresh]
	{ "1f15", 10 }, // TU106M [VdChip RTX 2060 Mobile]
	{ "1f36", 10 }, // TU106GLM [Lwdqro RTX 3000 Mobile / Max-Q]
	{ "1f42", 10 }, // TU106 [VdChip RTX 2060 SUPER]
	{ "1f47", 10 }, // TU106 [VdChip RTX 2060 SUPER]
	{ "1f50", 10 }, // TU106BM [VdChip RTX 2070 Mobile / Max-Q]
	{ "1f51", 10 }, // TU106BM [VdChip RTX 2060 Mobile]
	{ "1f54", 10 }, // TU106BM [VdChip RTX 2070 Mobile]
	{ "1f55", 10 }, // TU106BM [VdChip RTX 2060 Mobile]
	{ "1f76", 10 }, // TU106GLM [Lwdqro RTX 3000 Mobile Refresh]
// AMPERE
	{ "2204", 20 }, // GA102 [VdChip RTX 3090]
	{ "2205", 20 }, // GA102 [VdChip RTX 3080 20GB]
	{ "2206", 20 }, // GA102 [VdChip RTX 3080]
	{ "1467", 20 }, // GA102 [VdChip RTX 3080]
	{ "146d", 20 }, // GA102 [VdChip RTX 3080 20GB]
	{ "2208", 20 }, // GA102 [VdChip RTX 3080 Ti]
	{ "220d", 20 }, // GA102 [VdChip RTX 3080 Lite Hash Rate]
	{ "222b", 20 }, // GA102 [VdChip RTX 3090 Engineering Sample]
	{ "222f", 20 }, // GA102 [VdChip RTX 3080 11GB / 12GB Engineering Sample]
	{ "2230", 20 }, // GA102GL [RTX A6000]
	{ "2231", 20 }, // GA102GL [RTX A5000]
	{ "2482", 20 }, // GA104 [VdChip RTX 3070 Ti]
	{ "2484", 20 }, // GA104 [VdChip RTX 3070]
	{ "146b", 20 }, // GA104 [VdChip RTX 3070]
	{ "14ae", 20 }, // GA104 [VdChip RTX 3070 16GB]
	{ "2486", 20 }, // GA104 [VdChip RTX 3060 Ti]
	{ "249c", 20 }, // GA104M [VdChip RTX 3080 Mobile / Max-Q 8GB/16GB]
	{ "249d", 20 }, // GA104M [VdChip RTX 3070 Mobile / Max-Q]
	{ "24ac", 20 }, // GA104 [VdChip RTX 30x0 Engineering Sample]
	{ "24ad", 20 }, // GA104 [VdChip RTX 3060 Engineering Sample]
	{ "24af", 20 }, // GA104 [VdChip RTX 3070 Engineering Sample]
	{ "24b0", 20 }, // GA104GL [RTX A4000]
	{ "24b6", 20 }, // GA104GLM [RTX A5000 Mobile]
	{ "24b7", 20 }, // GA104GLM [RTX A4000 Mobile]
	{ "24b8", 20 }, // GA104GLM [RTX A3000 Mobile]
	{ "24bf", 20 }, // GA104 [VdChip RTX 3070 Engineering Sample]
	{ "24dc", 20 }, // GA104M [VdChip RTX 3080 Mobile / Max-Q 8GB/16GB]
	{ "24dd", 20 }, // GA104M [VdChip RTX 3070 Mobile / Max-Q]
	{ "2501", 20 }, // GA106 [VdChip RTX 3060]
	{ "2503", 20 }, // GA106 [VdChip RTX 3060]
	{ "2504", 20 }, // GA106 [VdChip RTX 3060 Lite Hash Rate]
	{ "2520", 20 }, // GA106M [VdChip RTX 3060 Mobile / Max-Q]
	{ "252f", 20 }, // GA106 [VdChip RTX 3060 Engineering Sample]
	{ "2560", 20 }, // GA106M [VdChip RTX 3060 Mobile / Max-Q]
	{ "2583", 20 }, // GA107 [VdChip RTX 3050]
	{ "25a0", 20 }, // GA107M [VdChip RTX 3050 Ti Mobile]
	{ "25a2", 20 }, // GA107M [VdChip RTX 3050 Mobile]
	{ "25af", 20 }, // GA107 [VdChip RTX 3050 Engineering Sample]
	{ "25b5", 20 }, // GA107GLM [RTX A4 Mobile]
	{ "25b8", 20 }, // GA107GLM [RTX A2000 Mobile]
};