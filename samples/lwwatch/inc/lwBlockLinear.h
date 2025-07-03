#ifndef __LWBLOCKLINEAR_H__
#define __LWBLOCKLINEAR_H__

//
// lwBlockLinear.h
//
// Structure definitions and utility functions for handling block-linear
// surfaces.
//
// Block linear surfaces were added in LW50, and are similar in spirit to
// tiled surfaces
//
// First, the image is decomposed into gobs.  A gob on LW50 is a 256-byte 2D
// image block, consisting of 4 rows with 64 bytes of image data in each row.
// Within a gob, pixels are stored in normal raster-scan order (step first in
// X, then in Y).
//
// Then, the gobs are arranged into blocks.  The purpose of a block is to
// arrange a collection of gobs into a 2D/3D sub-surface whose sizes are
// roughly equal in each dimension.  This will tend to minimize the number of
// page transitions as we walk through a surface during rasterization or
// texturing.  The desired size of a block is implementation dependent (may
// want larger blocks for wide vidmem channels).  The class interface allows a
// programmable block size.
//
// For example, if you had a large 2D surface with 32-bit pixels, each gob
// would be 16x4 pixels in size.  If you want to arrange blocks inside a
// single 4KB page, you would stack 8 gobs vertically and 2 gobs horizontally.
// This would give you a 32x32 pixel block.  If you had a similar 3D surface,
// you might want to stack 1 gob horizontally, 2 gobs vertically, and 8 gobs
// in depth, which would give you a 16x8x8 pixel block.
//
// Within each block, individual gobs are stored in normal raster-scan order
// (step first in X, then in Y, then in Z).  Blocks themselves are also stored
// in normal raster-scan order.
//
// If an image is smaller than a gob or a block, the size of the image is
// effectively padded out to the next boundary.  For example, 1D textures have
// only a single row, but there is no 1D-specific gob format.  Since each gob
// has four rows, 3/4 of each gob is wasted.  For small images, the driver
// should choose the gobs-per-block parameters to avoid storing completely
// unused gobs within in a block.  In the example above, if the 2D surface was
// only 16 pixels high, you would want to stack 4 gobs vertically and 4 gobs
// horizontally, which would yield a 64x16 block size.
//
// The gobs and blocks are arranged to represent a single 1D, 2D, or 3D
// surface.  Mipmapped and lwbemap textures, as well as 2D texture image
// arrays (where the texture is an array of separate 2D images), consist of
// multiple images.
//
// If the texture is mipmapped, the LODs of each face are stored
// conselwtively.  The class interface allows for only a single set of
// gobs-per-block parameters, which would correspond to those used for LOD
// zero.  For subsequent levels, the gobs-per-block parameters are computed
// as:
//
//   gobsPerBlockX(LOD n) = min(roundUpPow2(width(LOD n)/gobSizeX),
//                              gobsPerBlockX(LOD 0))
//   gobsPerBlockY(LOD n) = min(roundUpPow2(Height(LOD n)/gobSizeY),
//                              gobsPerBlockY(LOD 0))
//   gobsPerBlockZ(LOD n) = min(roundUpPow2(Depth(LOD n)),
//                              gobsPerBlockZ(LOD 0))
//
// Basically, the LOD zero gobs-per-block parameters are used, except if the
// LOD is smaller than a block in any dimension.  In that case, we instead use
// the number of gobs in that dimension, rounded up to the next power of two.
//
// After arranging all the mipmaps in the first image/face of the texture (if
// any), the subsequent image/face is stored immediately after the end of the
// current one.  It will at least be gob-aligned.
//

// Constants for fixed hardware gob sizes.
#define LW_BLOCK_LINEAR_LOG_GOB_WIDTH       6    /*    64 bytes (2^6) */
#define LW_BLOCK_LINEAR_LOG_GOB_HEIGHT      3    /* x   8 rows  (2^3) */
#define LW_BLOCK_LINEAR_GOB_SIZE            512  /* = 512 bytes (2^9) */

// Derived constants for fixed hardware gob sizes.
#define LW_BLOCK_LINEAR_GOB_WIDTH               \
        (1 << LW_BLOCK_LINEAR_LOG_GOB_WIDTH)
#define LW_BLOCK_LINEAR_GOB_HEIGHT              \
        (1 << LW_BLOCK_LINEAR_LOG_GOB_HEIGHT)

// LwBlockLinearLog2GobsPerBlock:  Holds the base2 logs of the size of a block
// (in gobs).
//
// IMPORTANT:  These parameters are necessary to interpret how block-linear
// memory is formatted.
typedef struct LwBlockLinearLog2GobsPerBlockRec {
    unsigned int        x, y, z;
} LwBlockLinearLog2GobsPerBlock;


// LwBlockLinearImageInfo:  Describes parameters for a given block linear
// image.  Includes both block linear formatting parameters and the overall
// size.
typedef struct LwBlockLinearImageInfoRec {

    // log2GobsPerBlock:  Holds the base2 logs of the number of gobs per block
    // in each dimension.
    LwBlockLinearLog2GobsPerBlock log2GobsPerBlock;

    // xBlocks, yBlocks, zBlocks:  Number of blocks in the image in the X, Y,
    // and Z dimensions.
    unsigned int    xBlocks, yBlocks, zBlocks;

    // offset:  Offset (in bytes) of this image from the surface base.  If the
    // surface is not mipmapped or an array, the offset is always zero.
    unsigned int    offset;

    // size:  Size (in bytes) of this image.
    unsigned int    size;

} LwBlockLinearImageInfo;


// structure of data necessary to callwlate block linear info for a texture
// used as a parameter to

typedef struct _LwBlockLinearTexParams {
    LwU32 dwBaseWidth;      // width of base texture in texels
    LwU32 dwBaseHeight;     // height of base texture in texels
    LwU32 dwBaseDepth;      // duh...
    LwU32 dwTexelSize;      // texel size in bytes
    LwU32 dwDimensionality;
    LwU32 dwLOD;            // LOD selection
    LwU32 dwFace;           // lwbemap face selection
    LwU32 dwFaceSize;       // bytes per face
    LwU32 dwBorderSize;     // border size in texels (?)
    LwU32 dwBlockWidthLog2;
    LwU32 dwBlockHeightLog2;
} LwBlockLinearTexParams;

// LW_SIZE_IN_BLOCKS:  Macro to transform a size <baseSize> into a block
// count, where blocks of size 2^<logBlockSize>.  The base size is padded up
// to the next block boundary, if needed.
#define LW_SIZE_IN_BLOCKS(baseSize, logBlockSize)      \
    ((baseSize + (1 << logBlockSize) - 1) >> logBlockSize)


// lwGetBlockLinearTexLevelInfo:  Return block linear information (via "pBlockLinearInfo")
// for the texture whose properties are specified in "pTexParams"
void lwGetBlockLinearTexLevelInfo (LwBlockLinearImageInfo *pBlockLinearInfo,    // out
                                   LwBlockLinearTexParams *pTexParams);         // in

// lwGetBlockLinearImageInfo:  Extract image information for a
// block-linear image of size <w> x <h> x <d>, with pixels of <pixelSize>
// bytes.
static LW_INLINE void
lwGetBlockLinearImageInfo(LwBlockLinearImageInfo *info,
                          unsigned int w, unsigned int h,
                          unsigned int d, unsigned int pixelSize)
{
    // Transform the width parameter from pixels to bytes.
    w *= pixelSize;

    // Transform the width and height parameters from bytes/pixels into gobs.
    w = LW_SIZE_IN_BLOCKS(w, LW_BLOCK_LINEAR_LOG_GOB_WIDTH);
    h = LW_SIZE_IN_BLOCKS(h, LW_BLOCK_LINEAR_LOG_GOB_HEIGHT);

    // Transform all the size parameters into blocks.
    w = LW_SIZE_IN_BLOCKS(w, info->log2GobsPerBlock.x);
    h = LW_SIZE_IN_BLOCKS(h, info->log2GobsPerBlock.y);
    d = LW_SIZE_IN_BLOCKS(d, info->log2GobsPerBlock.z);

    // Save away the number of blocks used in each dimension.
    info->xBlocks = w;
    info->yBlocks = h;
    info->zBlocks = d;

    // Compute the overall size of the surface, which is stored as a
    // collection of blocks.
    info->size = ((w * h * d) <<
                  (LW_BLOCK_LINEAR_LOG_GOB_WIDTH +
                   LW_BLOCK_LINEAR_LOG_GOB_HEIGHT +
                   info->log2GobsPerBlock.x +
                   info->log2GobsPerBlock.y +
                   info->log2GobsPerBlock.z));

    // A single image is by definition not offset from itself.
    info->offset = 0;
}

#endif // #ifndef __LWBLOCKLINEAR_H__

