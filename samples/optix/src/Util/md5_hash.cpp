/***************************************************************************************************
 * Copyright (c) 2017, LWPU CORPORATION. All rights reserved.
 **************************************************************************************************/
/**
   \file
   \brief        MD5 hash value class.
*/

#include <Util/digest_md5.h>
#include <Util/md5_hash.h>

namespace MI {
namespace DIGEST {


MD5_hash::MD5_hash( MD5 const& generator )
{
    const char* raw_digest = generator.raw_digest();
    memcpy( &m_value[0], &raw_digest[0], 8 );
    memcpy( &m_value[1], &raw_digest[8], 8 );
}
}
}
