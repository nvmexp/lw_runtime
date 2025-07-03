//
//  Copyright (c) 2019 LWPU Corporation.  All rights reserved.
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
#include <Memory/DemandLoad/PagingManager.h>

#include <Util/MakeUnique.h>

namespace optix {

static const unsigned int MAX_GUTTER_WIDTH            = 8;
static const unsigned int MAX_TILE_WIDTH              = 64 + 2 * MAX_GUTTER_WIDTH;
static const unsigned int MAX_ELEMENT_SIZE            = 4 * sizeof( float );
static const unsigned int MAX_SOFTWARE_TILE_PAGE_SIZE = MAX_TILE_WIDTH * MAX_TILE_WIDTH * MAX_ELEMENT_SIZE;

static const unsigned int LWDA_SPARSE_TEXTURE_PAGE_SIZE = 65536;

const unsigned int PagingService::PAGE_SIZE_IN_BYTES = std::max( LWDA_SPARSE_TEXTURE_PAGE_SIZE, MAX_SOFTWARE_TILE_PAGE_SIZE );


std::unique_ptr<PagingService> createPagingService( Context* context, bool usePerTileCallbacks, unsigned int numVirtualPages )
{
    return makeUnique<PagingManager>( context, usePerTileCallbacks, numVirtualPages );
}

}  // namespace optix
