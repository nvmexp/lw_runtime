// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <ExelwtionStrategy/CommonRuntime.h>
#include <ExelwtionStrategy/FrameStatus.h>

// These can be used on the CPU for debugging
extern "C" void exit( int code );

using namespace cort;

//
// Variable token-based lookup.  Get ID and forward to bindless path.
//

// Texture size functions
CORT_OVERRIDABLE
unsigned int cort::Texture_getElement_token_txq_width( CanonicalState* state, unsigned short token, bool hwonly, bool swonly )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_txq_width( state, texid, hwonly, swonly );
}

CORT_OVERRIDABLE
unsigned int cort::Texture_getElement_token_txq_height( CanonicalState* state, unsigned short token, bool hwonly, bool swonly )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_txq_height( state, texid, hwonly, swonly );
}

CORT_OVERRIDABLE
unsigned int cort::Texture_getElement_token_txq_depth( CanonicalState* state, unsigned short token, bool hwonly, bool swonly )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_txq_depth( state, texid, hwonly, swonly );
}

CORT_OVERRIDABLE
uint3 cort::Texture_getElement_token_size( CanonicalState* state, unsigned short token, bool hwonly, bool swonly )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_size( state, texid, hwonly, swonly );
}

// Tex functions
CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tex_1d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tex_1d( state, texid, hwonly, swonly, x );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tex_2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tex_2d( state, texid, hwonly, swonly, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tex_3d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float z )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tex_3d( state, texid, hwonly, swonly, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tex_a1d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tex_a1d( state, texid, hwonly, swonly, a, x );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tex_a2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x, float y )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tex_a2d( state, texid, hwonly, swonly, a, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tex_lwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float z )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tex_lwbe( state, texid, hwonly, swonly, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tex_alwbe( CanonicalState* state,
                                                 unsigned short  token,
                                                 bool            hwonly,
                                                 bool            swonly,
                                                 unsigned int    a,
                                                 float           x,
                                                 float           y,
                                                 float           z )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tex_alwbe( state, texid, hwonly, swonly, a, x, y, z );
}


// TLD/"fetch" functions (linear memory only)
CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texfetch_1d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, int x )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texfetch_1d( state, texid, hwonly, swonly, x );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texfetch_2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, int x, int y )  // Not exposed in lwca
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texfetch_2d( state, texid, hwonly, swonly, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texfetch_3d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, int x, int y, int z )  // Not exposed in lwca
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texfetch_3d( state, texid, hwonly, swonly, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texfetch_a1d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, int x )  // Not exposed in lwca
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texfetch_a1d( state, texid, hwonly, swonly, a, x );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texfetch_a2d( CanonicalState* state,
                                                    unsigned short  token,
                                                    bool            hwonly,
                                                    bool            swonly,
                                                    unsigned int    a,
                                                    int             x,
                                                    int             y )  // Not exposed in lwca
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texfetch_a2d( state, texid, hwonly, swonly, a, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texfetch_2dms( CanonicalState* state,
                                                     unsigned short  token,
                                                     bool            hwonly,
                                                     bool            swonly,
                                                     unsigned int    s,
                                                     int             x,
                                                     int             y )  // Not exposed in lwca
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texfetch_2dms( state, texid, hwonly, swonly, s, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texfetch_a2dms( CanonicalState* state,
                                                      unsigned short  token,
                                                      bool            hwonly,
                                                      bool            swonly,
                                                      unsigned int    s,
                                                      unsigned int    a,
                                                      int             x,
                                                      int             y )  // Not exposed in lwca
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texfetch_a2dms( state, texid, hwonly, swonly, s, a, x, y );
}


// Mip level
CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texlevel_1d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float level )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texlevel_1d( state, texid, hwonly, swonly, x, level );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texlevel_2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float level )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texlevel_2d( state, texid, hwonly, swonly, x, y, level );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texlevel_3d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float z, float level )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texlevel_3d( state, texid, hwonly, swonly, x, y, z, level );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texlevel_a1d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x, float level )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texlevel_a1d( state, texid, hwonly, swonly, a, x, level );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texlevel_a2d( CanonicalState* state,
                                                    unsigned short  token,
                                                    bool            hwonly,
                                                    bool            swonly,
                                                    unsigned int    a,
                                                    float           x,
                                                    float           y,
                                                    float           level )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texlevel_a2d( state, texid, hwonly, swonly, a, x, y, level );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texlevel_lwbe( CanonicalState* state,
                                                     unsigned short  token,
                                                     bool            hwonly,
                                                     bool            swonly,
                                                     float           x,
                                                     float           y,
                                                     float           z,
                                                     float           level )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texlevel_lwbe( state, texid, hwonly, swonly, x, y, z, level );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texlevel_alwbe( CanonicalState* state,
                                                      unsigned short  token,
                                                      bool            hwonly,
                                                      bool            swonly,
                                                      unsigned int    a,
                                                      float           x,
                                                      float           y,
                                                      float           z,
                                                      float           level )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texlevel_alwbe( state, texid, hwonly, swonly, a, x, y, z, level );
}


// Mip grad
CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texgrad_1d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float dPdx, float dPdy )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texgrad_1d( state, texid, hwonly, swonly, x, dPdx, dPdy );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texgrad_2d( CanonicalState* state,
                                                  unsigned short  token,
                                                  bool            hwonly,
                                                  bool            swonly,
                                                  float           x,
                                                  float           y,
                                                  float           dPdx_x,
                                                  float           dPdx_y,
                                                  float           dPdy_x,
                                                  float           dPdy_y )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texgrad_2d( state, texid, hwonly, swonly, x, y, dPdx_x, dPdx_y, dPdy_x, dPdy_y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texgrad_3d( CanonicalState* state,
                                                  unsigned short  token,
                                                  bool            hwonly,
                                                  bool            swonly,
                                                  float           x,
                                                  float           y,
                                                  float           z,
                                                  float           dPdx_x,
                                                  float           dPdx_y,
                                                  float           dPdx_z,
                                                  float           dPdy_x,
                                                  float           dPdy_y,
                                                  float           dPdy_z )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texgrad_3d( state, texid, hwonly, swonly, x, y, z, dPdx_x, dPdx_y, dPdx_z, dPdy_x, dPdy_y, dPdy_z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texgrad_a1d( CanonicalState* state,
                                                   unsigned short  token,
                                                   bool            hwonly,
                                                   bool            swonly,
                                                   unsigned int    a,
                                                   float           x,
                                                   float           dPdx,
                                                   float           dPdy )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texgrad_a1d( state, texid, hwonly, swonly, a, x, dPdx, dPdy );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texgrad_a2d( CanonicalState* state,
                                                   unsigned short  token,
                                                   bool            hwonly,
                                                   bool            swonly,
                                                   unsigned int    a,
                                                   float           x,
                                                   float           y,
                                                   float           dPdx_x,
                                                   float           dPdx_y,
                                                   float           dPdy_x,
                                                   float           dPdy_y )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texgrad_a2d( state, texid, hwonly, swonly, a, x, y, dPdx_x, dPdx_y, dPdy_x, dPdy_y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texgrad_lwbe( CanonicalState* state,
                                                    unsigned short  token,
                                                    bool            hwonly,
                                                    bool            swonly,
                                                    float           x,
                                                    float           y,
                                                    float           z,
                                                    float           dPdx_x,
                                                    float           dPdx_y,
                                                    float           dPdy_x,
                                                    float           dPdy_y )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texgrad_lwbe( state, texid, hwonly, swonly, x, y, z, dPdx_x, dPdx_y, dPdy_x, dPdy_y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_texgrad_alwbe( CanonicalState* state,
                                                     unsigned short  token,
                                                     bool            hwonly,
                                                     bool            swonly,
                                                     unsigned int    a,
                                                     float           x,
                                                     float           y,
                                                     float           z,
                                                     float           dPdx_x,
                                                     float           dPdx_y,
                                                     float           dPdy_x,
                                                     float           dPdy_y )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_texgrad_alwbe( state, texid, hwonly, swonly, a, x, y, z, dPdx_x, dPdx_y, dPdy_x, dPdy_y );
}

// TLD4
CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tld4r_2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tld4r_2d( state, texid, hwonly, swonly, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tld4g_2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tld4g_2d( state, texid, hwonly, swonly, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tld4b_2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tld4b_2d( state, texid, hwonly, swonly, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tld4a_2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y )
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tld4a_2d( state, texid, hwonly, swonly, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tld4r_a2d( CanonicalState* state,
                                                 unsigned short  token,
                                                 bool            hwonly,
                                                 bool            swonly,
                                                 unsigned int    a,
                                                 float           x,
                                                 float           y )  // Not exposed in PTX
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tld4r_a2d( state, texid, hwonly, swonly, a, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tld4g_a2d( CanonicalState* state,
                                                 unsigned short  token,
                                                 bool            hwonly,
                                                 bool            swonly,
                                                 unsigned int    a,
                                                 float           x,
                                                 float           y )  // Not exposed in PTX
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tld4g_a2d( state, texid, hwonly, swonly, a, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tld4b_a2d( CanonicalState* state,
                                                 unsigned short  token,
                                                 bool            hwonly,
                                                 bool            swonly,
                                                 unsigned int    a,
                                                 float           x,
                                                 float           y )  // Not exposed in PTX
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tld4b_a2d( state, texid, hwonly, swonly, a, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tld4a_a2d( CanonicalState* state,
                                                 unsigned short  token,
                                                 bool            hwonly,
                                                 bool            swonly,
                                                 unsigned int    a,
                                                 float           x,
                                                 float           y )  // Not exposed in PTX
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tld4a_a2d( state, texid, hwonly, swonly, a, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tld4r_lwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float z )  // Not exposed in PTX
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tld4r_lwbe( state, texid, hwonly, swonly, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tld4g_lwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float z )  // Not exposed in PTX
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tld4g_lwbe( state, texid, hwonly, swonly, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tld4b_lwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float z )  // Not exposed in PTX
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tld4b_lwbe( state, texid, hwonly, swonly, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tld4a_lwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float z )  // Not exposed in PTX
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tld4a_lwbe( state, texid, hwonly, swonly, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tld4r_alwbe( CanonicalState* state,
                                                   unsigned short  token,
                                                   bool            hwonly,
                                                   bool            swonly,
                                                   unsigned int    a,
                                                   float           x,
                                                   float           y,
                                                   float           z )  // Not exposed in PTX
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tld4r_alwbe( state, texid, hwonly, swonly, a, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tld4g_alwbe( CanonicalState* state,
                                                   unsigned short  token,
                                                   bool            hwonly,
                                                   bool            swonly,
                                                   unsigned int    a,
                                                   float           x,
                                                   float           y,
                                                   float           z )  // Not exposed in PTX
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tld4g_alwbe( state, texid, hwonly, swonly, a, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tld4b_alwbe( CanonicalState* state,
                                                   unsigned short  token,
                                                   bool            hwonly,
                                                   bool            swonly,
                                                   unsigned int    a,
                                                   float           x,
                                                   float           y,
                                                   float           z )  // Not exposed in PTX
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tld4b_alwbe( state, texid, hwonly, swonly, a, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_token_tld4a_alwbe( CanonicalState* state,
                                                   unsigned short  token,
                                                   bool            hwonly,
                                                   bool            swonly,
                                                   unsigned int    a,
                                                   float           x,
                                                   float           y,
                                                   float           z )  // Not exposed in PTX
{
    int texid = Runtime_lookupIdVariableValue( state, token );
    return Texture_getElement_id_tld4a_alwbe( state, texid, hwonly, swonly, a, x, y, z );
}


//
// OptiX texture ID-based lookup.  Fetch the header and forward to HW or SW texture.
//

// Texture size functions
CORT_OVERRIDABLE
unsigned int cort::Texture_getElement_id_txq_width( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly )
{
    return Global_getTextureSamplerHeader( state, textureid ).width;
}

CORT_OVERRIDABLE
unsigned int cort::Texture_getElement_id_txq_height( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly )
{
    return Global_getTextureSamplerHeader( state, textureid ).height;
}

CORT_OVERRIDABLE
unsigned int cort::Texture_getElement_id_txq_depth( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly )
{
    return Global_getTextureSamplerHeader( state, textureid ).depth;
}

CORT_OVERRIDABLE
uint3 cort::Texture_getElement_id_size( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    return uint3( texture.width, texture.height, texture.depth );
}

// Tex functions
CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tex_1d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, float x )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tex_1d( texture, x );
    else
        return Texture_getElement_hw_tex_1d( texref, x );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tex_2d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, float x, float y )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tex_2d( texture, x, y );
    else
        return Texture_getElement_hw_tex_2d( texref, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tex_3d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, float x, float y, float z )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tex_3d( texture, x, y, z );
    else
        return Texture_getElement_hw_tex_3d( texref, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tex_a1d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, unsigned int a, float x )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tex_a1d( texture, a, x );
    else
        return Texture_getElement_hw_tex_a1d( texref, a, x );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tex_a2d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, unsigned int a, float x, float y )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tex_a2d( texture, a, x, y );
    else
        return Texture_getElement_hw_tex_a2d( texref, a, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tex_lwbe( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, float x, float y, float z )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tex_lwbe( texture, x, y, z );
    else
        return Texture_getElement_hw_tex_lwbe( texref, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tex_alwbe( CanonicalState* state,
                                              unsigned int    textureid,
                                              bool            hwonly,
                                              bool            swonly,
                                              unsigned int    a,
                                              float           x,
                                              float           y,
                                              float           z )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tex_alwbe( texture, a, x, y, z );
    else
        return Texture_getElement_hw_tex_alwbe( texref, a, x, y, z );
}


// TLD/"fetch" functions (linear memory only)
CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texfetch_1d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, int x )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texfetch_1d( texture, x );
    else
        return Texture_getElement_hw_texfetch_1d( texref, x );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texfetch_2d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, int x, int y )  // Not exposed in lwca
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texfetch_2d( texture, x, y );
    else
        return Texture_getElement_hw_texfetch_2d( texref, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texfetch_3d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, int x, int y, int z )  // Not exposed in lwca
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texfetch_3d( texture, x, y, z );
    else
        return Texture_getElement_hw_texfetch_3d( texref, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texfetch_a1d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, unsigned int a, int x )  // Not exposed in lwca
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texfetch_a1d( texture, a, x );
    else
        return Texture_getElement_hw_texfetch_a1d( texref, a, x );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texfetch_a2d( CanonicalState* state,
                                                 unsigned int    textureid,
                                                 bool            hwonly,
                                                 bool            swonly,
                                                 unsigned int    a,
                                                 int             x,
                                                 int             y )  // Not exposed in lwca
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texfetch_a2d( texture, a, x, y );
    else
        return Texture_getElement_hw_texfetch_a2d( texref, a, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texfetch_2dms( CanonicalState* state,
                                                  unsigned int    textureid,
                                                  bool            hwonly,
                                                  bool            swonly,
                                                  unsigned int    s,
                                                  int             x,
                                                  int             y )  // Not exposed in lwca
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texfetch_2dms( texture, s, x, y );
    else
        return Texture_getElement_hw_texfetch_2dms( texref, s, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texfetch_a2dms( CanonicalState* state,
                                                   unsigned int    textureid,
                                                   bool            hwonly,
                                                   bool            swonly,
                                                   unsigned int    s,
                                                   unsigned int    a,
                                                   int             x,
                                                   int             y )  // Not exposed in lwca
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texfetch_a2dms( texture, s, a, x, y );
    else
        return Texture_getElement_hw_texfetch_a2dms( texref, s, a, x, y );
}


// Mip level
CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texlevel_1d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, float x, float level )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texlevel_1d( texture, x, level );
    else
        return Texture_getElement_hw_texlevel_1d( texref, x, level );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texlevel_2d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, float x, float y, float level )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texlevel_2d( texture, x, y, level );
    else
        return Texture_getElement_hw_texlevel_2d( texref, x, y, level );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texlevel_3d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, float x, float y, float z, float level )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texlevel_3d( texture, x, y, z, level );
    else
        return Texture_getElement_hw_texlevel_3d( texref, x, y, z, level );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texlevel_a1d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, unsigned int a, float x, float level )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texlevel_a1d( texture, a, x, level );
    else
        return Texture_getElement_hw_texlevel_a1d( texref, a, x, level );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texlevel_a2d( CanonicalState* state,
                                                 unsigned int    textureid,
                                                 bool            hwonly,
                                                 bool            swonly,
                                                 unsigned int    a,
                                                 float           x,
                                                 float           y,
                                                 float           level )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texlevel_a2d( texture, a, x, y, level );
    else
        return Texture_getElement_hw_texlevel_a2d( texref, a, x, y, level );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texlevel_lwbe( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, float x, float y, float z, float level )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texlevel_lwbe( texture, x, y, z, level );
    else
        return Texture_getElement_hw_texlevel_lwbe( texref, x, y, z, level );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texlevel_alwbe( CanonicalState* state,
                                                   unsigned int    textureid,
                                                   bool            hwonly,
                                                   bool            swonly,
                                                   unsigned int    a,
                                                   float           x,
                                                   float           y,
                                                   float           z,
                                                   float           level )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texlevel_alwbe( texture, a, x, y, z, level );
    else
        return Texture_getElement_hw_texlevel_alwbe( texref, a, x, y, z, level );
}


// Mip grad
CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texgrad_1d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, float x, float dPdx, float dPdy )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texgrad_1d( texture, x, dPdx, dPdy );
    else
        return Texture_getElement_hw_texgrad_1d( texref, x, dPdx, dPdy );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texgrad_2d( CanonicalState* state,
                                               unsigned int    textureid,
                                               bool            hwonly,
                                               bool            swonly,
                                               float           x,
                                               float           y,
                                               float           dPdx_x,
                                               float           dPdx_y,
                                               float           dPdy_x,
                                               float           dPdy_y )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texgrad_2d( texture, x, y, dPdx_x, dPdx_y, dPdy_x, dPdy_y );
    else
        return Texture_getElement_hw_texgrad_2d( texref, x, y, dPdx_x, dPdx_y, dPdy_x, dPdy_y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texgrad_3d( CanonicalState* state,
                                               unsigned int    textureid,
                                               bool            hwonly,
                                               bool            swonly,
                                               float           x,
                                               float           y,
                                               float           z,
                                               float           dPdx_x,
                                               float           dPdx_y,
                                               float           dPdx_z,
                                               float           dPdy_x,
                                               float           dPdy_y,
                                               float           dPdy_z )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texgrad_3d( texture, x, y, z, dPdx_x, dPdx_y, dPdx_z, dPdy_x, dPdy_y, dPdy_z );
    else
        return Texture_getElement_hw_texgrad_3d( texref, x, y, z, dPdx_x, dPdx_y, dPdx_z, dPdy_x, dPdy_y, dPdy_z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texgrad_a1d( CanonicalState* state,
                                                unsigned int    textureid,
                                                bool            hwonly,
                                                bool            swonly,
                                                unsigned int    a,
                                                float           x,
                                                float           dPdx,
                                                float           dPdy )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texgrad_a1d( texture, a, x, dPdx, dPdy );
    else
        return Texture_getElement_hw_texgrad_a1d( texref, a, x, dPdx, dPdy );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texgrad_a2d( CanonicalState* state,
                                                unsigned int    textureid,
                                                bool            hwonly,
                                                bool            swonly,
                                                unsigned int    a,
                                                float           x,
                                                float           y,
                                                float           dPdx_x,
                                                float           dPdx_y,
                                                float           dPdy_x,
                                                float           dPdy_y )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texgrad_a2d( texture, a, x, y, dPdx_x, dPdx_y, dPdy_x, dPdy_y );
    else
        return Texture_getElement_hw_texgrad_a2d( texref, a, x, y, dPdx_x, dPdx_y, dPdy_x, dPdy_y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texgrad_lwbe( CanonicalState* state,
                                                 unsigned int    textureid,
                                                 bool            hwonly,
                                                 bool            swonly,
                                                 float           x,
                                                 float           y,
                                                 float           z,
                                                 float           dPdx_x,
                                                 float           dPdx_y,
                                                 float           dPdy_x,
                                                 float           dPdy_y )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texgrad_lwbe( texture, x, y, z, dPdx_x, dPdx_y, dPdy_x, dPdy_y );
    else
        return Texture_getElement_hw_texgrad_lwbe( texref, x, y, z, dPdx_x, dPdx_y, dPdy_x, dPdy_y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_texgrad_alwbe( CanonicalState* state,
                                                  unsigned int    textureid,
                                                  bool            hwonly,
                                                  bool            swonly,
                                                  unsigned int    a,
                                                  float           x,
                                                  float           y,
                                                  float           z,
                                                  float           dPdx_x,
                                                  float           dPdx_y,
                                                  float           dPdy_x,
                                                  float           dPdy_y )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_texgrad_alwbe( texture, a, x, y, z, dPdx_x, dPdx_y, dPdy_x, dPdy_y );
    else
        return Texture_getElement_hw_texgrad_alwbe( texref, a, x, y, z, dPdx_x, dPdx_y, dPdy_x, dPdy_y );
}

// TLD4
CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tld4r_2d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, float x, float y )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tld4r_2d( texture, x, y );
    else
        return Texture_getElement_hw_tld4r_2d( texref, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tld4g_2d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, float x, float y )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tld4g_2d( texture, x, y );
    else
        return Texture_getElement_hw_tld4g_2d( texref, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tld4b_2d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, float x, float y )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tld4b_2d( texture, x, y );
    else
        return Texture_getElement_hw_tld4b_2d( texref, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tld4a_2d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, float x, float y )
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tld4a_2d( texture, x, y );
    else
        return Texture_getElement_hw_tld4a_2d( texref, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tld4r_a2d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, unsigned int a, float x, float y )  // Not exposed in PTX
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tld4r_a2d( texture, a, x, y );
    else
        return Texture_getElement_hw_tld4r_a2d( texref, a, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tld4g_a2d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, unsigned int a, float x, float y )  // Not exposed in PTX
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tld4g_a2d( texture, a, x, y );
    else
        return Texture_getElement_hw_tld4g_a2d( texref, a, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tld4b_a2d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, unsigned int a, float x, float y )  // Not exposed in PTX
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tld4b_a2d( texture, a, x, y );
    else
        return Texture_getElement_hw_tld4b_a2d( texref, a, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tld4a_a2d( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, unsigned int a, float x, float y )  // Not exposed in PTX
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tld4a_a2d( texture, a, x, y );
    else
        return Texture_getElement_hw_tld4a_a2d( texref, a, x, y );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tld4r_lwbe( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, float x, float y, float z )  // Not exposed in PTX
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tld4r_lwbe( texture, x, y, z );
    else
        return Texture_getElement_hw_tld4r_lwbe( texref, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tld4g_lwbe( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, float x, float y, float z )  // Not exposed in PTX
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tld4g_lwbe( texture, x, y, z );
    else
        return Texture_getElement_hw_tld4g_lwbe( texref, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tld4b_lwbe( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, float x, float y, float z )  // Not exposed in PTX
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tld4b_lwbe( texture, x, y, z );
    else
        return Texture_getElement_hw_tld4b_lwbe( texref, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tld4a_lwbe( CanonicalState* state, unsigned int textureid, bool hwonly, bool swonly, float x, float y, float z )  // Not exposed in PTX
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tld4a_lwbe( texture, x, y, z );
    else
        return Texture_getElement_hw_tld4a_lwbe( texref, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tld4r_alwbe( CanonicalState* state,
                                                unsigned int    textureid,
                                                bool            hwonly,
                                                bool            swonly,
                                                unsigned int    a,
                                                float           x,
                                                float           y,
                                                float           z )  // Not exposed in PTX
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tld4r_alwbe( texture, a, x, y, z );
    else
        return Texture_getElement_hw_tld4r_alwbe( texref, a, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tld4g_alwbe( CanonicalState* state,
                                                unsigned int    textureid,
                                                bool            hwonly,
                                                bool            swonly,
                                                unsigned int    a,
                                                float           x,
                                                float           y,
                                                float           z )  // Not exposed in PTX
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tld4g_alwbe( texture, a, x, y, z );
    else
        return Texture_getElement_hw_tld4g_alwbe( texref, a, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tld4b_alwbe( CanonicalState* state,
                                                unsigned int    textureid,
                                                bool            hwonly,
                                                bool            swonly,
                                                unsigned int    a,
                                                float           x,
                                                float           y,
                                                float           z )  // Not exposed in PTX
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tld4b_alwbe( texture, a, x, y, z );
    else
        return Texture_getElement_hw_tld4b_alwbe( texref, a, x, y, z );
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_id_tld4a_alwbe( CanonicalState* state,
                                                unsigned int    textureid,
                                                bool            hwonly,
                                                bool            swonly,
                                                unsigned int    a,
                                                float           x,
                                                float           y,
                                                float           z )  // Not exposed in PTX
{
    TextureSampler texture = Global_getTextureSamplerHeader( state, textureid );
    uint64         texref  = texture.dd.texref;
    if( !hwonly && ( swonly || texref == TextureSampler::UseSoftwarePointer ) )
        return Texture_getElement_sw_tld4a_alwbe( texture, a, x, y, z );
    else
        return Texture_getElement_hw_tld4a_alwbe( texref, a, x, y, z );
}

/*
 * SW texture functions
 */

static inline int formatToLog2EltSize( int x )
{
    return ( x & 3 ) + ( x >> 4 );
}

static inline int formatToLog2SizeOfComponent( int x )
{
    return x & 3;
}

static inline int formatToLog2VectorSize( int x )
{
    return x >> 4;
}

union TexLoadB128
{
    float        f[4];
    unsigned int ui[4];
    int          i[4];
};

union TexLoadB64
{
    float          f[2];
    unsigned int   ui[2];
    int            i[2];
    unsigned short us[4];
    short          s[4];
};

union TexLoadB32
{
    float          f[1];
    unsigned int   ui[1];
    int            i[1];
    unsigned short us[2];
    short          s[2];
    unsigned char  ub[4];
    char           b[4];
};

union TexLoadB16
{
    unsigned short us[1];
    short          s[1];
    unsigned char  ub[2];
    char           b[2];
};

union TexLoadB8
{
    unsigned char ub[1];
    char          b[1];
};

#define FLT_EPSILON 1.192092896e-07F /* smallest such that 1.0+FLT_EPSILON != 1.0 */
#define FLT_MAX 3.402823466e+38F     /* max value */
#define FLT_MIN 1.175494351e-38F     /* min positive value */

float4 loadTexel( int format, const char* mem, bool colwertI2F, bool scaleI2F, int log2EltSize )
{
    using namespace lwca;
    const float suint   = scaleI2F ? ( 1.0f / (float)0xffffffff ) : 1.0f;
    const float sint    = scaleI2F ? ( 2.0f / (float)0xffffffff ) : 1.0f;
    const float sushort = scaleI2F ? ( 1.0f / (float)0xffff ) : 1.0f;
    const float sshort  = scaleI2F ? ( 2.0f / (float)0xffff ) : 1.0f;
    const float subyte  = scaleI2F ? ( 1.0f / (float)0xff ) : 1.0f;
    const float sbyte   = scaleI2F ? ( 2.0f / (float)0xff ) : 1.0f;

    float4  ret;
    float4* retF4 = &ret;
    uint4*  retI4 = (cort::uint4*)retF4;

    if( log2EltSize == 4 )
    {
        TexLoadB128 t = *( (TexLoadB128*)mem );
        if( format == TEX_FORMAT_FLOAT4 || !colwertI2F )
        {
            *retF4 = (const float4&)t;
        }
        else if( format == TEX_FORMAT_UNSIGNED_INT4 )
        {
            *retF4 = float4( suint * uint2float_rz( t.ui[0] ), suint * uint2float_rz( t.ui[1] ),
                             suint * uint2float_rz( t.ui[2] ), suint * uint2float_rz( t.ui[3] ) );
        }
        else if( format == TEX_FORMAT_INT4 )
        {
            *retF4 = float4( sint * int2float_rz( t.i[0] ), sint * int2float_rz( t.i[1] ),
                             sint * int2float_rz( t.i[2] ), sint * int2float_rz( t.i[3] ) );
        }
    }
    else if( log2EltSize == 3 )
    {
        TexLoadB64 t = *( (TexLoadB64*)mem );
        if( format == TEX_FORMAT_FLOAT2 )
        {
            retF4->x = t.f[0], retF4->y = t.f[1];
        }
        else if( format == TEX_FORMAT_UNSIGNED_INT2 )
        {
            if( colwertI2F )
                retF4->x = suint * uint2float_rz( t.ui[0] ), retF4->y = suint * uint2float_rz( t.ui[1] );
            else
                retF4->x = t.f[0], retF4->y = t.f[1];
        }
        else if( format == TEX_FORMAT_INT2 )
        {
            if( colwertI2F )
                retF4->x = sint * int2float_rz( t.i[0] ), retF4->y = sint * int2float_rz( t.i[1] );
            else
                retF4->x = t.f[0], retF4->y = t.f[1];
        }
        else if( format == TEX_FORMAT_UNSIGNED_SHORT4 )
        {
            if( colwertI2F )
                retF4->x = sushort * uint2float_rz( t.us[0] ), retF4->y = sushort * uint2float_rz( t.us[1] ),
                retF4->z = sushort * uint2float_rz( t.us[2] ), retF4->w = sushort * uint2float_rz( t.us[3] );
            else
                retI4->x = t.us[0], retI4->y = t.us[1], retI4->z = t.us[2], retI4->w = t.us[3];
        }
        else if( format == TEX_FORMAT_SHORT4 )
        {
            if( colwertI2F )
                retF4->x = sshort * int2float_rz( t.s[0] ), retF4->y = sshort * int2float_rz( t.s[1] ),
                retF4->z = sshort * int2float_rz( t.s[2] ), retF4->w = sshort * int2float_rz( t.s[3] );
            else
                retI4->x = t.s[0], retI4->y = t.s[1], retI4->z = t.s[2], retI4->w = t.s[3];
        }
    }
    else if( log2EltSize == 2 )
    {
        TexLoadB32 t = *( (TexLoadB32*)mem );
        if( format == TEX_FORMAT_FLOAT1 )
        {
            retF4->x = t.f[0];
        }
        else if( format == TEX_FORMAT_UNSIGNED_INT1 )
        {
            if( colwertI2F )
                retF4->x = suint * uint2float_rz( t.ui[0] );
            else
                retF4->x = t.f[0];
        }
        else if( format == TEX_FORMAT_INT1 )
        {
            if( colwertI2F )
                retF4->x = sint * int2float_rz( t.i[0] );
            else
                retF4->x = t.f[0];
        }
        else if( format == TEX_FORMAT_UNSIGNED_SHORT2 )
        {
            if( colwertI2F )
                retF4->x = sushort * uint2float_rz( t.us[0] ), retF4->y = sushort * uint2float_rz( t.us[1] );
            else
                retI4->x = t.us[0], retI4->y = t.us[1];
        }
        else if( format == TEX_FORMAT_SHORT2 )
        {
            if( colwertI2F )
                retF4->x = sshort * int2float_rz( t.s[0] ), retF4->y = sshort * int2float_rz( t.s[1] );
            else
                retI4->x = t.s[0], retI4->y = t.s[1];
        }
        else if( format == TEX_FORMAT_UNSIGNED_BYTE4 )
        {
            if( colwertI2F )
                retF4->x = subyte * uint2float_rz( t.ub[0] ), retF4->y = subyte * uint2float_rz( t.ub[1] ),
                retF4->z = subyte * uint2float_rz( t.ub[2] ), retF4->w = subyte * uint2float_rz( t.ub[3] );
            else
                retI4->x = t.ub[0], retI4->y = t.ub[1], retI4->z = t.ub[2], retI4->w = t.ub[3];
        }
        else if( format == TEX_FORMAT_BYTE4 )
        {
            if( colwertI2F )
                retF4->x = sbyte * int2float_rz( t.b[0] ), retF4->y = sbyte * int2float_rz( t.b[1] ),
                retF4->z = sbyte * int2float_rz( t.b[2] ), retF4->w = sbyte * int2float_rz( t.b[3] );
            else
                retI4->x = t.b[0], retI4->y = t.b[1], retI4->z = t.b[2], retI4->w = t.b[3];
        }
    }
    else if( log2EltSize == 1 )
    {
        TexLoadB16 t = *( (TexLoadB16*)mem );
        if( format == TEX_FORMAT_UNSIGNED_SHORT1 )
        {
            if( colwertI2F )
                retF4->x = sushort * uint2float_rz( t.us[0] );
            else
                retI4->x = t.us[0];
        }
        else if( format == TEX_FORMAT_SHORT1 )
        {
            if( colwertI2F )
                retF4->x = sshort * int2float_rz( t.s[0] );
            else
                retI4->x = t.s[0];
        }
        else if( format == TEX_FORMAT_UNSIGNED_BYTE2 )
        {
            if( colwertI2F )
                retF4->x = subyte * uint2float_rz( t.ub[0] ), retF4->y = subyte * uint2float_rz( t.ub[1] );
            else
                retI4->x = t.ub[0], retI4->y = t.ub[1];
        }
        else if( format == TEX_FORMAT_BYTE2 )
        {
            if( colwertI2F )
                retF4->x = sbyte * int2float_rz( t.b[0] ), retF4->y = sbyte * int2float_rz( t.b[1] );
            else
                retI4->x = t.b[0], retI4->y = t.b[1];
        }
    }
    else if( log2EltSize == 0 )
    {
        TexLoadB8 t = *( (TexLoadB8*)mem );
        if( format == TEX_FORMAT_UNSIGNED_BYTE1 )
        {
            if( colwertI2F )
                retF4->x = subyte * uint2float_rz( t.ub[0] );
            else
                retI4->x = t.ub[0];
        }
        else if( format == TEX_FORMAT_BYTE1 )
        {
            if( colwertI2F )
                retF4->x = sbyte * int2float_rz( t.b[0] );
            else
                retI4->x = t.b[0];
        }
    }
    return ret;
}

int nextTexelPos( int posI, int sizeI, bool repeat, bool mirror, bool oddCoord )
{
    if( repeat )
    {
        int pos2 = posI + 1;
        return pos2 >= sizeI ? 0 : pos2;
    }
    else if( mirror )
    {
        if( oddCoord )
        {
            int pos2 = posI - 1;
            return lwca::maxi( pos2, 0 );
        }
        else
        {
            int pos2 = posI + 1;
            return lwca::mini( pos2, sizeI - 1 );
        }
    }
    else
    {
        return lwca::mini( posI + 1, sizeI - 1 );
    }
}

void computeWeights1D( int posI, int posI2, float& w0, float& w1, float posF, bool mirror )
{
    float frac0 = posF - lwca::floorf( posF );
    // for mirror mode
    if( mirror && posI > posI2 )
        frac0 = 1.0f - frac0;
    w1        = frac0;
    w0        = 1.0f - frac0;
}

#if 0
float sanitizeCoord(float coordF)
{
  coordF = isnan(coordF) ? 0.f : coordF;

  if( cort_isfinite(coordF) )
    coordF = coordF > 0 ? FLT_MAX : -FLT_MAX;
  return coordF;
}
#endif

// do denormalization, wrapping and clamping
void computePositionLinear( int&   posI,
                            float& posF,
                            float  sizeF,
                            int    sizeI,
                            float  coordF,
                            bool   normalizedCoords,
                            bool   repeat,
                            bool   mirror,
                            bool   clamp,
                            bool&  clampToBorder,
                            bool&  oddCoord )
{
    using namespace lwca;
    oddCoord = false;
    if( !normalizedCoords || clamp )
    {
        if( normalizedCoords )
        {
            posF = coordF * sizeF - 0.5f;
        }
        else
        {
            posF = coordF - 0.5f;
        }

        float clampMaxF = sizeF - 1.0f;
        if( clampToBorder )
        {
            clampToBorder = posF < 0.0f || posF >= sizeF;
        }
        // Max should be first here to fix NaN to minimum value
        // clamp posF to be at least 0
        posF = maxf( 0.0f, posF );
        // clamp posF to be <= clampMaxF
        posF = minf( posF, clampMaxF );

        posI = float2int_rz( posF );
    }
    else
    {
        float coordFloorF = lwca::floorf( coordF );
        // use saturation mode to fix NaN and +-Inf to zero; ROUND_RZ to have result range [0; 1)
        posF = saturate( add_rz( coordF, -coordFloorF ) );
        posF = posF * sizeF - 0.5f;
        if( repeat )
        {
            // used ROUND_RZ to have result range [0, size)
            float wrapped = add_rz( sizeF, posF );
            posF          = posF < 0 ? wrapped : posF;
        }
        else if( mirror )
        {
            bool is_odd = ( float2int_rm( coordF ) & 1 ) != 0;
            oddCoord    = is_odd ^ ( posF < 0.f );
            posF        = lwca::fabs( posF );
            if( is_odd )
                posF = ( sizeF - FLT_EPSILON ) - posF;
        }

        posI = float2int_rz( posF );
    }
}

// do denormalization, wrapping and clamping
void computePositionNearest( int& posI, float& posF, float sizeF, int sizeI, float coordF, bool normalizedCoords, bool repeat, bool mirror, bool clamp, bool& clampToBorder )
{
    using namespace lwca;
    if( !normalizedCoords || clamp )
    {
        if( normalizedCoords )
        {
            posF = mul_rz( coordF, sizeF );
        }
        else
        {
            posF = coordF;
        }

        float clampMaxF = sizeF - 1.0f;
        if( clampToBorder )
        {
            clampToBorder = posF < 0.0f || posF >= sizeF;
        }
        // Max should be first here to fix NaN to minimum value
        // clamp posF to be at least 0
        posF = maxf( 0.0f, posF );
        // clamp posF to be <= clampMaxF
        posF = minf( posF, clampMaxF );

        posI = float2int_rz( posF );
    }
    else
    {
        float coordFloorF = lwca::floorf( coordF );
        // use saturation mode to fix NaN and +-Inf to zero; ROUND_RZ to have result range [0; 1)
        posF = saturate( add_rz( coordF, -coordFloorF ) );
        if( mirror )
        {
            bool is_odd = ( float2int_rm( coordF ) & 1 ) != 0;
            if( is_odd )
                posF = ( 1.0f - FLT_EPSILON ) - posF;
        }
        posF = mul_rz( posF, sizeF );
        posI = float2int_rz( posF );
    }
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_sw_tex_1d( cort::TextureSampler sampler, float coordXF )
{
    using namespace lwca;
    const bool  repeatX          = sampler.wrapMode0 == TEX_WRAP_REPEAT;
    const bool  mirrorX          = sampler.wrapMode0 == TEX_WRAP_MIRROR;
    const bool  clampX           = sampler.wrapMode0 & 1;  // TEX_WRAP_CLAMP_TO_EDGE || TEX_WRAP_CLAMP_TO_BORDER
    bool        clampToBorderX   = sampler.wrapMode0 == TEX_WRAP_CLAMP_TO_BORDER;
    const bool  normalizedCoords = sampler.normCoord;
    const int   log2EltSize      = formatToLog2EltSize( sampler.format );
    const bool  linear           = sampler.filterMode;
    const char* ptr              = sampler.dd.swptr;
    const int   sizeXI           = sampler.width;
    const float sizeXF           = int2float_rz( sizeXI );
    float       posXF;
    int         posXI;

    if( !linear )
    {
        computePositionNearest( posXI, posXF, sizeXF, sizeXI, coordXF, normalizedCoords, repeatX, mirrorX, clampX, clampToBorderX );
        if( clampToBorderX )
            return float4( 0, 0, 0, 0 );

        int offset = posXI << log2EltSize;
        return loadTexel( sampler.format, ptr + offset, sampler.normRet, true, log2EltSize );
    }
    else
    {
        bool oddXCoord;
        computePositionLinear( posXI, posXF, sizeXF, sizeXI, coordXF, normalizedCoords, repeatX, mirrorX, clampX,
                               clampToBorderX, oddXCoord );
        if( clampToBorderX )
            return float4( 0, 0, 0, 0 );

        int pos2XI = nextTexelPos( posXI, sizeXI, repeatX, mirrorX, oddXCoord );

        // Compute weights
        float wx0, wx1;
        computeWeights1D( posXI, pos2XI, wx0, wx1, posXF, mirrorX );

        // callwlate offsets for the next texels
        int offset0 = posXI << log2EltSize;
        int offset1 = pos2XI << log2EltSize;

        float4 val0 = loadTexel( sampler.format, ptr + offset0, true, false, log2EltSize );
        float4 val1 = loadTexel( sampler.format, ptr + offset1, true, false, log2EltSize );

        int    log2VSize  = formatToLog2VectorSize( sampler.format );
        int    colwertF2I = !sampler.normRet && ( sampler.format & 0xf ) != TEX_FORMAT_FLOAT1;
        float4 ret;
        ret.x        = val0.x * wx0 + val1.x * wx1;
        float4& retI = (float4&)ret;
        if( colwertF2I )
            retI.x = float2int_rz( ret.x );
        if( log2VSize == 1 )
        {
            ret.y = val0.y * wx0 + val1.y * wx1;
            if( colwertF2I )
                retI.y = float2int_rz( ret.y );
        }
        else if( log2VSize == 2 )
        {
            ret.y = val0.y * wx0 + val1.y * wx1;
            ret.z = val0.z * wx0 + val1.z * wx1;
            ret.w = val0.w * wx0 + val1.w * wx1;
            if( colwertF2I )
            {
                retI.y = float2int_rz( ret.y );
                retI.z = float2int_rz( ret.z );
                retI.w = float2int_rz( ret.w );
            }
        }
        return ret;
    }
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_sw_tex_2d( cort::TextureSampler sampler, float coordXF, float coordYF )
{
    using namespace lwca;
    const bool repeatX = sampler.wrapMode0 == TEX_WRAP_REPEAT, repeatY = sampler.wrapMode1 == TEX_WRAP_REPEAT;
    const bool mirrorX = sampler.wrapMode0 == TEX_WRAP_MIRROR, mirrorY = sampler.wrapMode1 == TEX_WRAP_MIRROR;
    const bool clampX = sampler.wrapMode0 & 1,
               clampY = sampler.wrapMode1 & 1;  // TEX_WRAP_CLAMP_TO_EDGE || TEX_WRAP_CLAMP_TO_BORDER
    bool        clampToBorderX = sampler.wrapMode0 == TEX_WRAP_CLAMP_TO_BORDER, clampToBorderY = sampler.wrapMode1 == TEX_WRAP_CLAMP_TO_BORDER;
    const bool  normalizedCoords = sampler.normCoord;
    const int   log2EltSize      = formatToLog2EltSize( sampler.format );
    const bool  linear           = sampler.filterMode;
    const char* ptr              = sampler.dd.swptr;
    const int   sizeXI = sampler.width, sizeYI = sampler.height;
    const float sizeXF = int2float_rz( sizeXI ), sizeYF = int2float_rz( sizeYI );
    float       posXF, posYF;
    int         posXI, posYI;

    if( !linear )
    {
        computePositionNearest( posXI, posXF, sizeXF, sizeXI, coordXF, normalizedCoords, repeatX, mirrorX, clampX, clampToBorderX );
        if( clampToBorderX )
            return float4( 0, 0, 0, 0 );
        computePositionNearest( posYI, posYF, sizeYF, sizeYI, coordYF, normalizedCoords, repeatY, mirrorY, clampY, clampToBorderY );
        if( clampToBorderY )
            return float4( 0, 0, 0, 0 );

        int offset = ( sizeXI * posYI + posXI ) << log2EltSize;
        return loadTexel( sampler.format, ptr + offset, sampler.normRet, true, log2EltSize );
    }
    else
    {
        bool oddXCoord, oddYCoord;
        computePositionLinear( posXI, posXF, sizeXF, sizeXI, coordXF, normalizedCoords, repeatX, mirrorX, clampX,
                               clampToBorderX, oddXCoord );
        if( clampToBorderX )
            return float4( 0, 0, 0, 0 );
        computePositionLinear( posYI, posYF, sizeYF, sizeYI, coordYF, normalizedCoords, repeatY, mirrorY, clampY,
                               clampToBorderY, oddYCoord );
        if( clampToBorderY )
            return float4( 0, 0, 0, 0 );

        int pos2XI = nextTexelPos( posXI, sizeXI, repeatX, mirrorX, oddXCoord );
        int pos2YI = nextTexelPos( posYI, sizeYI, repeatY, mirrorY, oddYCoord );

        // Compute weights
        float wx0, wx1, wy0, wy1;
        float weight00, weight01, weight10, weight11;
        computeWeights1D( posXI, pos2XI, wx0, wx1, posXF, mirrorX );
        computeWeights1D( posYI, pos2YI, wy0, wy1, posYF, mirrorY );

        weight00 = wy0 * wx0;
        weight01 = wy0 * wx1;
        weight10 = wy1 * wx0;
        weight11 = wy1 * wx1;

        // callwlate offsets for the next texels
        int offset00 = ( posYI * sizeXI + posXI ) << log2EltSize;
        int offset01 = ( posYI * sizeXI + pos2XI ) << log2EltSize;
        int offset10 = ( pos2YI * sizeXI + posXI ) << log2EltSize;
        int offset11 = ( pos2YI * sizeXI + pos2XI ) << log2EltSize;

        float4 val00 = loadTexel( sampler.format, ptr + offset00, true, false, log2EltSize );
        float4 val01 = loadTexel( sampler.format, ptr + offset01, true, false, log2EltSize );
        float4 val10 = loadTexel( sampler.format, ptr + offset10, true, false, log2EltSize );
        float4 val11 = loadTexel( sampler.format, ptr + offset11, true, false, log2EltSize );

        const int log2VSize  = formatToLog2VectorSize( sampler.format );
        const int colwertF2I = !sampler.normRet && ( sampler.format & 0xf ) != TEX_FORMAT_FLOAT1;
        float4    ret;
        ret.x        = val00.x * weight00 + val01.x * weight01 + val10.x * weight10 + val11.x * weight11;
        float4& retI = (float4&)ret;
        if( colwertF2I )
            retI.x = float2int_rz( ret.x );
        if( log2VSize == 1 )
        {
            ret.y = val00.y * weight00 + val01.y * weight01 + val10.y * weight10 + val11.y * weight11;
            if( colwertF2I )
                retI.y = float2int_rz( ret.y );
        }
        else if( log2VSize == 2 )
        {
            ret.y = val00.y * weight00 + val01.y * weight01 + val10.y * weight10 + val11.y * weight11;
            ret.z = val00.z * weight00 + val01.z * weight01 + val10.z * weight10 + val11.z * weight11;
            ret.w = val00.w * weight00 + val01.w * weight01 + val10.w * weight10 + val11.w * weight11;
            if( colwertF2I )
            {
                retI.y = float2int_rz( ret.y );
                retI.z = float2int_rz( ret.z );
                retI.w = float2int_rz( ret.w );
            }
        }
        return ret;
    }
}

CORT_OVERRIDABLE
float4 cort::Texture_getElement_sw_tex_3d( cort::TextureSampler sampler, float coordXF, float coordYF, float coordZF )
{
    using namespace lwca;
    const bool repeatX = sampler.wrapMode0 == TEX_WRAP_REPEAT, repeatY = sampler.wrapMode1 == TEX_WRAP_REPEAT,
               repeatZ = sampler.wrapMode2 == TEX_WRAP_REPEAT;
    const bool mirrorX = sampler.wrapMode0 == TEX_WRAP_MIRROR, mirrorY = sampler.wrapMode1 == TEX_WRAP_MIRROR,
               mirrorZ = sampler.wrapMode2 == TEX_WRAP_MIRROR;
    const bool clampX = sampler.wrapMode0 & 1, clampY = sampler.wrapMode1 & 1,
               clampZ            = sampler.wrapMode2 & 1;  // TEX_WRAP_CLAMP_TO_EDGE || TEX_WRAP_CLAMP_TO_BORDER
    bool clampToBorderX          = sampler.wrapMode0 == TEX_WRAP_CLAMP_TO_BORDER,
         clampToBorderY          = sampler.wrapMode1 == TEX_WRAP_CLAMP_TO_BORDER,
         clampToBorderZ          = sampler.wrapMode2 == TEX_WRAP_CLAMP_TO_BORDER;
    const bool  normalizedCoords = sampler.normCoord;
    const int   log2EltSize      = formatToLog2EltSize( sampler.format );
    const bool  linear           = sampler.filterMode;
    const char* ptr              = sampler.dd.swptr;
    const int   sizeXI = sampler.width, sizeYI = sampler.height, sizeZI = sampler.depth;
    const float sizeXF = int2float_rz( sizeXI ), sizeYF = int2float_rz( sizeYI ), sizeZF = int2float_rz( sizeZI );
    float       posXF, posYF, posZF;
    int         posXI, posYI, posZI;

    if( !linear )
    {
        computePositionNearest( posXI, posXF, sizeXF, sizeXI, coordXF, normalizedCoords, repeatX, mirrorX, clampX, clampToBorderX );
        if( clampToBorderX )
            return float4( 0, 0, 0, 0 );
        computePositionNearest( posYI, posYF, sizeYF, sizeYI, coordYF, normalizedCoords, repeatY, mirrorY, clampY, clampToBorderY );
        if( clampToBorderY )
            return float4( 0, 0, 0, 0 );
        computePositionNearest( posZI, posZF, sizeZF, sizeZI, coordZF, normalizedCoords, repeatZ, mirrorZ, clampZ, clampToBorderZ );
        if( clampToBorderZ )
            return float4( 0, 0, 0, 0 );

        int offset = ( ( sizeXI * sizeYI * posZI ) + ( sizeXI * posYI + posXI ) ) << log2EltSize;
        return loadTexel( sampler.format, ptr + offset, sampler.normRet, true, log2EltSize );
    }
    else
    {
        bool oddXCoord, oddYCoord, oddZCoord;
        computePositionLinear( posXI, posXF, sizeXF, sizeXI, coordXF, normalizedCoords, repeatX, mirrorX, clampX,
                               clampToBorderX, oddXCoord );
        if( clampToBorderX )
            return float4( 0, 0, 0, 0 );
        computePositionLinear( posYI, posYF, sizeYF, sizeYI, coordYF, normalizedCoords, repeatY, mirrorY, clampY,
                               clampToBorderY, oddYCoord );
        if( clampToBorderY )
            return float4( 0, 0, 0, 0 );
        computePositionLinear( posZI, posZF, sizeZF, sizeZI, coordZF, normalizedCoords, repeatZ, mirrorZ, clampZ,
                               clampToBorderZ, oddZCoord );
        if( clampToBorderZ )
            return float4( 0, 0, 0, 0 );

        int pos2XI = nextTexelPos( posXI, sizeXI, repeatX, mirrorX, oddXCoord );
        int pos2YI = nextTexelPos( posYI, sizeYI, repeatY, mirrorY, oddYCoord );
        int pos2ZI = nextTexelPos( posZI, sizeZI, repeatZ, mirrorZ, oddZCoord );

        // Compute weights
        float wx0, wx1, wy0, wy1, wz0, wz1;
        float weight000, weight001, weight010, weight011, weight100, weight101, weight110, weight111;
        computeWeights1D( posXI, pos2XI, wx0, wx1, posXF, mirrorX );
        computeWeights1D( posYI, pos2YI, wy0, wy1, posYF, mirrorY );
        computeWeights1D( posZI, pos2ZI, wz0, wz1, posZF, mirrorZ );

        weight000 = wx0 * wy0;
        weight001 = wx1 * wy0;
        weight010 = wx0 * wy1;
        weight011 = wx1 * wy1;

        weight100 = weight000 * wz1;
        weight101 = weight001 * wz1;
        weight110 = weight010 * wz1;
        weight111 = weight011 * wz1;

        weight000 = weight000 * wz0;
        weight001 = weight001 * wz0;
        weight010 = weight010 * wz0;
        weight011 = weight011 * wz0;

        // callwlate offsets for the next texels
        int sizeXYI   = sizeXI * sizeYI;
        int offset000 = ( ( sizeXYI * posZI ) + ( posYI * sizeXI + posXI ) ) << log2EltSize;
        int offset001 = ( ( sizeXYI * posZI ) + ( posYI * sizeXI + pos2XI ) ) << log2EltSize;
        int offset010 = ( ( sizeXYI * posZI ) + ( pos2YI * sizeXI + posXI ) ) << log2EltSize;
        int offset011 = ( ( sizeXYI * posZI ) + ( pos2YI * sizeXI + pos2XI ) ) << log2EltSize;
        int offset100 = ( ( sizeXYI * pos2ZI ) + ( posYI * sizeXI + posXI ) ) << log2EltSize;
        int offset101 = ( ( sizeXYI * pos2ZI ) + ( posYI * sizeXI + pos2XI ) ) << log2EltSize;
        int offset110 = ( ( sizeXYI * pos2ZI ) + ( pos2YI * sizeXI + posXI ) ) << log2EltSize;
        int offset111 = ( ( sizeXYI * pos2ZI ) + ( pos2YI * sizeXI + pos2XI ) ) << log2EltSize;

        float4 val000 = loadTexel( sampler.format, ptr + offset000, true, false, log2EltSize );
        float4 val001 = loadTexel( sampler.format, ptr + offset001, true, false, log2EltSize );
        float4 val010 = loadTexel( sampler.format, ptr + offset010, true, false, log2EltSize );
        float4 val011 = loadTexel( sampler.format, ptr + offset011, true, false, log2EltSize );

        float4 val100 = loadTexel( sampler.format, ptr + offset100, true, false, log2EltSize );
        float4 val101 = loadTexel( sampler.format, ptr + offset101, true, false, log2EltSize );
        float4 val110 = loadTexel( sampler.format, ptr + offset110, true, false, log2EltSize );
        float4 val111 = loadTexel( sampler.format, ptr + offset111, true, false, log2EltSize );

        int    log2VSize  = formatToLog2VectorSize( sampler.format );
        int    colwertF2I = !sampler.normRet && ( sampler.format & 0xf ) != TEX_FORMAT_FLOAT1;
        float4 ret;
        ret.x = val000.x * weight000 + val001.x * weight001 + val010.x * weight010 + val011.x * weight011
                + val100.x * weight100 + val101.x * weight101 + val110.x * weight110 + val111.x * weight111;
        float4& retI = (float4&)ret;
        if( colwertF2I )
            retI.x = float2int_rz( ret.x );
        if( log2VSize == 1 )
        {
            ret.y = val000.y * weight000 + val001.y * weight001 + val010.y * weight010 + val011.y * weight011
                    + val100.y * weight100 + val101.y * weight101 + val110.y * weight110 + val111.y * weight111;
            if( colwertF2I )
                retI.y = float2int_rz( ret.y );
        }
        else if( log2VSize == 2 )
        {
            ret.y = val000.y * weight000 + val001.y * weight001 + val010.y * weight010 + val011.y * weight011
                    + val100.y * weight100 + val101.y * weight101 + val110.y * weight110 + val111.y * weight111;
            ret.z = val000.z * weight000 + val001.z * weight001 + val010.z * weight010 + val011.z * weight011
                    + val100.z * weight100 + val101.z * weight101 + val110.z * weight110 + val111.z * weight111;
            ret.w = val000.w * weight000 + val001.w * weight001 + val010.w * weight010 + val011.w * weight011
                    + val100.w * weight100 + val101.w * weight101 + val110.w * weight110 + val111.w * weight111;
            if( colwertF2I )
            {
                retI.y = float2int_rz( ret.y );
                retI.z = float2int_rz( ret.z );
                retI.w = float2int_rz( ret.w );
            }
        }
        return ret;
    }
}
