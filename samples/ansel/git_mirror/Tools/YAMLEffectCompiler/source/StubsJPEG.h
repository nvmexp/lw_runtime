#pragma once

#include "jpeglib.h"
#include "turbojpeg.h"

struct jpeg_error_mgr * jpeg_std_error (struct jpeg_error_mgr *err)
{
    return nullptr;
}

void jpeg_CreateCompress (j_compress_ptr cinfo, int version, size_t structsize) { }
void jpeg_CreateDecompress (j_decompress_ptr cinfo, int version, size_t structsize) { }

void jpeg_destroy_compress (j_compress_ptr cinfo) {}
void jpeg_destroy_decompress (j_decompress_ptr cinfo) {}

void jpeg_stdio_dest (j_compress_ptr cinfo, FILE *outfile) {}
void jpeg_stdio_src (j_decompress_ptr cinfo, FILE *infile) {}

void jpeg_mem_dest (j_compress_ptr cinfo, unsigned char **outbuffer, unsigned long *outsize) {}
void jpeg_mem_src (j_decompress_ptr cinfo, const unsigned char *inbuffer, unsigned long insize) {}

void jpeg_set_defaults (j_compress_ptr cinfo) {}
void jpeg_set_colorspace (j_compress_ptr cinfo, J_COLOR_SPACE colorspace) {}
void jpeg_default_colorspace (j_compress_ptr cinfo) {}
void jpeg_set_quality (j_compress_ptr cinfo, int quality, boolean force_baseline) {}
void jpeg_set_linear_quality (j_compress_ptr cinfo, int scale_factor, boolean force_baseline) {}

void jpeg_start_compress (j_compress_ptr cinfo, boolean write_all_tables) {}
JDIMENSION jpeg_write_scanlines (j_compress_ptr cinfo, JSAMPARRAY scanlines, JDIMENSION num_lines) { return (JDIMENSION)0; }
void jpeg_finish_compress (j_compress_ptr cinfo) {}

boolean jpeg_start_decompress (j_decompress_ptr cinfo) { return (boolean)0; }
JDIMENSION jpeg_read_scanlines (j_decompress_ptr cinfo, JSAMPARRAY scanlines, JDIMENSION max_lines) { return (JDIMENSION)0; }
JDIMENSION jpeg_skip_scanlines (j_decompress_ptr cinfo, JDIMENSION num_lines) { return (JDIMENSION)0; }
void jpeg_crop_scanline (j_decompress_ptr cinfo, JDIMENSION *xoffset, JDIMENSION *width) {}
boolean jpeg_finish_decompress (j_decompress_ptr cinfo) { return (boolean)0; }

JDIMENSION jpeg_read_raw_data (j_decompress_ptr cinfo, JSAMPIMAGE data, JDIMENSION max_lines) { return (JDIMENSION)0; }

void jpeg_write_marker (j_compress_ptr cinfo, int marker, const JOCTET *dataptr, unsigned int datalen) {}
void jpeg_write_m_header (j_compress_ptr cinfo, int marker, unsigned int datalen) {}
void jpeg_write_m_byte (j_compress_ptr cinfo, int val) {}
void jpeg_write_tables (j_compress_ptr cinfo) {}
int jpeg_read_header (j_decompress_ptr cinfo, boolean require_image) { return 0; }

void jpeg_save_markers (j_decompress_ptr cinfo, int marker_code, unsigned int length_limit) {}

void jpeg_abort_compress (j_compress_ptr cinfo) {}
void jpeg_abort_decompress (j_decompress_ptr cinfo) {}



tjhandle tjInitDecompress(void) { return (tjhandle)0; }
int tjDecompressHeader3(tjhandle handle, const unsigned char *jpegBuf, unsigned long jpegSize, int *width, int *height, int *jpegSubsamp, int *jpegColorspace) { return 0; }
int tjDecompress2(tjhandle handle, const unsigned char *jpegBuf, unsigned long jpegSize, unsigned char *dstBuf, int width, int pitch, int height, int pixelFormat, int flags) { return 0; }
int tjDestroy(tjhandle handle) { return 0; }

