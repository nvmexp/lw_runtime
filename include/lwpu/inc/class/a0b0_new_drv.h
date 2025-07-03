// **************************************************************************
//
//       Copyright 1993-2011 LWPU, Corporation.  All rights reserved.
//
//     NOTICE TO USER:   The source code  is copyrighted under  U.S. and
//     international laws.  Users and possessors of this source code are
//     hereby granted a nonexclusive,  royalty-free copyright license to
//     use this code in individual and commercial software.
//
//     Any use of this source code must include,  in the user dolwmenta-
//     tion and  internal comments to the code,  notices to the end user
//     as follows:
//
//     LWPU, CORPORATION MAKES NO REPRESENTATION ABOUT THE SUITABILITY
//     OF  THIS SOURCE  CODE  FOR ANY PURPOSE.  IT IS  PROVIDED  "AS IS"
//     WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.  LWPU, CORPOR-
//     ATION DISCLAIMS ALL WARRANTIES  WITH REGARD  TO THIS SOURCE CODE,
//     INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGE-
//     MENT,  AND FITNESS  FOR A PARTICULAR PURPOSE.   IN NO EVENT SHALL
//     LWPU, CORPORATION  BE LIABLE FOR ANY SPECIAL,  INDIRECT,  INCI-
//     DENTAL, OR CONSEQUENTIAL DAMAGES,  OR ANY DAMAGES  WHATSOEVER RE-
//     SULTING FROM LOSS OF USE,  DATA OR PROFITS,  WHETHER IN AN ACTION
//     OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,  ARISING OUT OF
//     OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.
//
//     U.S. Government  End  Users.   This source code  is a "commercial
//     item,"  as that  term is  defined at  48 C.F.R. 2.101 (OCT 1995),
//     consisting  of "commercial  computer  software"  and  "commercial
//     computer  software  documentation,"  as such  terms  are  used in
//     48 C.F.R. 12.212 (SEPT 1995)  and is provided to the U.S. Govern-
//     ment only as  a commercial end item.   Consistent with  48 C.F.R.
//     12.212 and  48 C.F.R. 227.7202-1 through  227.7202-4 (JUNE 1995),
//     all U.S. Government End Users  acquire the source code  with only
//     those rights set forth herein.
//
// **************************************************************************

#ifndef __LWDEC_DRV_H_
#define __LWDEC_DRV_H_

// TODO: Many fields can be colwerted to bitfields to save memory BW
// TODO: Revisit reserved fields for proper alignment and memory savings

///////////////////////////////////////////////////////////////////////////////
// LWDEC(MSDEC 5) is a single engine solution, and seperates into VLD, MV, IQT,
//                MCFETCH, MC, MCC, REC, DBF, DFBFDMA, HIST etc unit.
//                The class(driver to HW) can mainly seperate into VLD parser
//                and Decoder part to be consistent with original design. And
//                the sequence level info usally set in VLD part. Later codec like
//                VP8 won't name in this way.
// MSVLD: Multi-Standard VLD parser.
//
#define ALIGN_UP(v, n)          (((v) + ((n)-1)) &~ ((n)-1))
#define LWDEC_ALIGN(value)      ALIGN_UP(value,256) // Align to 256 bytes
#define LWDEC_MAX_MPEG2_SLICE   65536 // at 4096*4096, macroblock count = 65536, 1 macroblock per slice

#define LWDEC_CODEC_MPEG1   0
#define LWDEC_CODEC_MPEG2   1
#define LWDEC_CODEC_VC1     2
#define LWDEC_CODEC_H264    3
#define LWDEC_CODEC_MPEG4   4
#define LWDEC_CODEC_DIVX    LWDEC_CODEC_MPEG4
#define LWDEC_CODEC_VP8     5

// AES encryption
enum
{
    AES128_NONE = 0x0,
    AES128_CTR = 0x1,
    AES128_CBC,
    AES128_ECB,
    AES128_OFB,
    AES128_CTR_LSB16B,
    AES128_CLR_AS_ENCRYPT,
    AES128_RESERVED = 0x7
};

enum
{
    AES128_CTS_DISABLE = 0x0,
    AES128_CTS_ENABLE = 0x1
};

enum
{
    AES128_PADDING_NONE = 0x0,
    AES128_PADDING_CARRY_OVER,
    AES128_PADDING_RFC2630,
    AES128_PADDING_RESERVED = 0x7
};

typedef enum
{
    ENCR_MODE_CTR64         = 0,
    ENCR_MODE_CBC           = 1,
    ENCR_MODE_ECB           = 2,
    ENCR_MODE_ECB_PARTIAL   = 3,
    ENCR_MODE_CBC_PARTIAL   = 4,
    ENCR_MODE_CLEAR_INTO_VPR = 5,     // used for clear stream decoding into VPR.
} ENCR_MODE;

// drm_mode configuration
//
// Bit 0:2  AES encryption mode
// Bit 3    CTS (CipherTextStealing) enable/disable
// Bit 4:6  Padding type
// Bit 7:7  Unwrap key enable/disable

#define AES_MODE_MASK           0x7
#define AES_CTS_MASK            0x1
#define AES_PADDING_TYPE_MASK   0x7
#define AES_UNWRAP_KEY_MASK     0x1

#define AES_MODE_SHIFT          0
#define AES_CTS_SHIFT           3
#define AES_PADDING_TYPE_SHIFT  4
#define AES_UNWRAP_KEY_SHIFT    7

#define AES_SET_FLAG(M, C, P)   ((M & AES_MODE_MASK) << AES_MODE_SHIFT) | \
                                ((C & AES_CTS_MASK) << AES_CTS_SHIFT) | \
                                ((P & AES_PADDING_TYPE_MASK) << AES_PADDING_TYPE_SHIFT)

#define AES_GET_FLAG(V, F)      ((V & ((AES_##F##_MASK) <<(AES_##F##_SHIFT))) >> (AES_##F##_SHIFT))

#define DRM_MODE_MASK           0x7f        // Bits 0:6  (0:2 -> AES_MODE, 3 -> AES_CTS, 4:6 -> AES_PADDING_TYPE)
#define AES_GET_DRM_MODE(V)      (V & DRM_MODE_MASK)

enum { DRM_MS_PIFF_CTR  =   AES_SET_FLAG(AES128_CTR, AES128_CTS_DISABLE, AES128_PADDING_CARRY_OVER) };
enum { DRM_MS_PIFF_CBC  =   AES_SET_FLAG(AES128_CBC, AES128_CTS_DISABLE, AES128_PADDING_NONE) };
enum { DRM_MARLIN_CTR   =   AES_SET_FLAG(AES128_CTR, AES128_CTS_DISABLE, AES128_PADDING_NONE) };
enum { DRM_MARLIN_CBC   =   AES_SET_FLAG(AES128_CBC, AES128_CTS_DISABLE, AES128_PADDING_RFC2630) };
enum { DRM_WIDEVINE     =   AES_SET_FLAG(AES128_CBC, AES128_CTS_ENABLE,  AES128_PADDING_NONE) };
enum { DRM_WIDEVINE_CTR =   AES_SET_FLAG(AES128_CTR, AES128_CTS_DISABLE, AES128_PADDING_CARRY_OVER) };
enum { DRM_ULTRA_VIOLET =   AES_SET_FLAG(AES128_CTR_LSB16B, AES128_CTS_DISABLE, AES128_PADDING_NONE) };
enum { DRM_NONE         =   AES_SET_FLAG(AES128_NONE, AES128_CTS_DISABLE, AES128_PADDING_NONE) };
enum { DRM_CLR_AS_ENCRYPT = AES_SET_FLAG(AES128_CLR_AS_ENCRYPT, AES128_CTS_DISABLE, AES128_PADDING_NONE)};

// Legacy codecs encryption parameters
typedef struct _lwdec_pass2_otf_s {
    unsigned int   wrapped_session_key[4];  // session keys
    unsigned int   wrapped_content_key[4];  // content keys
    unsigned int   initialization_vector[4];// Ctrl64 initial vector
    unsigned int   enable_encryption : 1;   // flag to enable/disable encryption
    unsigned int   key_increment : 6;       // added to content key after unwrapping
    unsigned int   encryption_mode : 4;
    unsigned int   reserved1 : 21;          // reserved
} lwdec_pass2_otf_s; // 0x10 bytes

typedef struct _lwdec_display_param_s
{
    int enableTFOutput                         : 1; //=1, enable dbfdma to output the display surface; if disable, then the following configure on tf is useless.
    //remap for VC1
    int VC1MapYFlag                            : 1;
    int MapYValue                              : 3;
    int VC1MapUVFlag                           : 1;
    int MapUVValue                             : 3;
    //tf
    int OutStride                              : 8;
    int TilingFormat                           : 3;
    int OutputStructure                        : 1;  //(0=frame, 1=field)
    int reserved0                              : 11;
    int OutputTop[2];                           // in units of 256
    int OutputBottom[2];                        // in units of 256
    //histogram
    int enableHistogram                     : 1; // enable histogram info collection.
    int HistogramStartX                      :12; // start X of Histogram window
    int HistogramStartY                      :12; // start Y of Histogram window
    int reserved1                            :7;
    int HistogramEndX                        :12; // end X of Histogram window
    int HistogramEndY                        :12; // end y of Histogram window
    int reserved2                            :8;
} lwdec_display_param_s;  // size 0x1c bytes

// H.264
typedef struct _lwdec_dpb_entry_s  // 16 bytes
{
    unsigned int index          : 7;    // uncompressed frame buffer index
    unsigned int col_idx        : 5;    // index of associated co-located motion data buffer
    unsigned int state          : 2;    // bit1(state)=1: top field used for reference, bit1(state)=1: bottom field used for reference
    unsigned int is_long_term   : 1;    // 0=short-term, 1=long-term
    unsigned int not_existing   : 1;    // 1=marked as non-existing
    unsigned int is_field       : 1;    // set if unpaired field or complementary field pair
    unsigned int top_field_marking : 4;
    unsigned int bottom_field_marking : 4;
    unsigned int output_memory_layout : 1;  // Set according to picture level output LW12/LW24 setting.
    unsigned int reserved       : 6;
    unsigned int FieldOrderCnt[2];      // : 2*32 [top/bottom]
    int FrameIdx;                       // : 16   short-term: FrameNum (16 bits), long-term: LongTermFrameIdx (4 bits)
} lwdec_dpb_entry_s;

typedef struct _lwdec_h264_pic_s
{
    lwdec_pass2_otf_s encryption_params;
    unsigned char eos[16];
    unsigned char explicitEOSPresentFlag;
    unsigned char reserved0[3];
    unsigned int stream_len;
    unsigned int slice_count;
    unsigned int mbhist_buffer_size;     // to pass buffer size of MBHIST_BUFFER

    // Driver may or may not use based upon need.
    // If 0 then default value of 1<<27 = 298ms @ 450MHz will be used in ucode.
    // Driver can send this value based upon resolution using the formula:
    // gptimer_timeout_value = 3 * (cycles required for one frame)
    unsigned int gptimer_timeout_value;

    // Fields from msvld_h264_seq_s
    int log2_max_pic_order_cnt_lsb_minus4;
    int delta_pic_order_always_zero_flag;
    int frame_mbs_only_flag;
    int PicWidthInMbs;
    int FrameHeightInMbs;
    int tileFormat; // 0: TBL; 1: KBL; 2: Tile16x16

    // Fields from msvld_h264_pic_s
    int entropy_coding_mode_flag;
    int pic_order_present_flag;
    int num_ref_idx_l0_active_minus1;
    int num_ref_idx_l1_active_minus1;
    int deblocking_filter_control_present_flag;
    int redundant_pic_cnt_present_flag;
    int transform_8x8_mode_flag;

    // Fields from mspdec_h264_picture_setup_s
    unsigned int pitch_luma;                    // Luma pitch
    unsigned int pitch_chroma;                  // chroma pitch

    unsigned int luma_top_offset;               // offset of luma top field in units of 256
    unsigned int luma_bot_offset;               // offset of luma bottom field in units of 256
    unsigned int luma_frame_offset;             // offset of luma frame in units of 256
    unsigned int chroma_top_offset;             // offset of chroma top field in units of 256
    unsigned int chroma_bot_offset;             // offset of chroma bottom field in units of 256
    unsigned int chroma_frame_offset;           // offset of chroma frame in units of 256
    unsigned int HistBufferSize;                // in units of 256

    unsigned int MbaffFrameFlag           : 1;  //
    unsigned int direct_8x8_inference_flag: 1;  //
    unsigned int weighted_pred_flag       : 1;  //
    unsigned int constrained_intra_pred_flag:1; //
    unsigned int ref_pic_flag             : 1;  // reference picture (nal_ref_idc != 0)
    unsigned int field_pic_flag           : 1;  //
    unsigned int bottom_field_flag        : 1;  //
    unsigned int second_field             : 1;  // second field of complementary reference field
    unsigned int log2_max_frame_num_minus4: 4;  //  (0..12)
    unsigned int chroma_format_idc        : 2;  //
    unsigned int pic_order_cnt_type       : 2;  //  (0..2)
    int pic_init_qp_minus26               : 6;  // : 6 (-26..+25)
    int chroma_qp_index_offset            : 5;  // : 5 (-12..+12)
    int second_chroma_qp_index_offset     : 5;  // : 5 (-12..+12)

    unsigned int weighted_bipred_idc      : 2;  // : 2 (0..2)
    unsigned int LwrrPicIdx               : 7;  // : 7  uncompressed frame buffer index
    unsigned int LwrrColIdx               : 5;  // : 5  index of associated co-located motion data buffer
    unsigned int frame_num                : 16; //
    unsigned int frame_surfaces           : 1;  // frame surfaces flag
    unsigned int output_memory_layout     : 1;  // 0: LW12; 1:LW24. Field pair must use the same setting.

    int LwrrFieldOrderCnt[2];                   // : 32 [Top_Bottom], [0]=TopFieldOrderCnt, [1]=BottomFieldOrderCnt
    lwdec_dpb_entry_s dpb[16];
    unsigned char WeightScale[6][4][4];         // : 6*4*4*8 in raster scan order (not zig-zag order)
    unsigned char WeightScale8x8[2][8][8];      // : 2*8*8*8 in raster scan order (not zig-zag order)

    // mvc setup info, must be zero if not mvc
    unsigned char num_inter_view_refs_lX[2];         // number of inter-view references
    char reserved1[14];                               // reserved for alignment
    signed char inter_view_refidx_lX[2][16];         // DPB indices (must also be marked as long-term)

    // lossless decode (At the time of writing this manual, x264 and JM encoders, differ in Intra_8x8 reference sample filtering)
    unsigned int lossless_ipred8x8_filter_enable        : 1;       // = 0, skips Intra_8x8 reference sample filtering, for vertical and horizontal predictions (x264 encoded streams); = 1, filter Intra_8x8 reference samples (JM encoded streams)
    unsigned int qpprime_y_zero_transform_bypass_flag   : 1;       // determines the transform bypass mode
    unsigned int reserved2                              : 30;      // kept for alignment; may be used for other parameters

    lwdec_display_param_s displayPara;
} lwdec_h264_pic_s;

// VC-1 Scratch buffer
typedef enum _vc1_fcm_e
{
    FCM_PROGRESSIVE = 0,
    FCM_FRAME_INTERLACE = 2,
    FCM_FIELD_INTERLACE = 3
} vc1_fcm_e;

typedef enum _syntax_vc1_ptype_e
{
    PTYPE_I       = 0,
    PTYPE_P       = 1,
    PTYPE_B       = 2,
    PTYPE_BI      = 3, //PTYPE_BI is not used to config register LW_CLWDEC_VLD_PIC_INFO_COMMON. field LW_CLWDEC_VLD_PIC_INFO_COMMON_PIC_CODING_VC1 is only 2 bits. I and BI pictures are configured with same value. Please refer to manual.
    PTYPE_SKIPPED = 4
} syntax_vc1_ptype_e;

// 7.1.1.32, Table 46 etc.
enum vc1_mvmode_e
{
    MVMODE_MIXEDMV                = 0,
    MVMODE_1MV                    = 1,
    MVMODE_1MV_HALFPEL            = 2,
    MVMODE_1MV_HALFPEL_BILINEAR   = 3,
    MVMODE_INTENSITY_COMPENSATION = 4
};

// 9.1.1.42, Table 105
typedef enum _vc1_fptype_e
{
    FPTYPE_I_I = 0,
    FPTYPE_I_P,
    FPTYPE_P_I,
    FPTYPE_P_P,
    FPTYPE_B_B,
    FPTYPE_B_BI,
    FPTYPE_BI_B,
    FPTYPE_BI_BI
} vc1_fptype_e;

// Table 43 (7.1.1.31.2)
typedef enum _vc1_dqprofile_e
{
    DQPROFILE_ALL_FOUR_EDGES  = 0,
    DQPROFILE_DOUBLE_EDGE     = 1,
    DQPROFILE_SINGLE_EDGE     = 2,
    DQPROFILE_ALL_MACROBLOCKS = 3
} vc1_dqprofile_e;

typedef struct _lwdec_vc1_pic_s
{
    lwdec_pass2_otf_s encryption_params;
    unsigned char eos[16];                    // to pass end of stream data separately if not present in bitstream surface
    unsigned char prefixStartCode[4];         // used for dxva to pass prefix start code.
    unsigned int  bitstream_offset;           // offset in words from start of bitstream surface if there is gap.
    unsigned char explicitEOSPresentFlag;     // to indicate that eos[] is used for passing end of stream data.
    unsigned char reserved0[3];
    unsigned int stream_len;
    unsigned int slice_count;
    unsigned int scratch_pic_buffer_size;

    // Driver may or may not use based upon need.
    // If 0 then default value of 1<<27 = 298ms @ 450MHz will be used in ucode.
    // Driver can send this value based upon resolution using the formula:
    // gptimer_timeout_value = 3 * (cycles required for one frame)
    unsigned int gptimer_timeout_value;

    // Fields from vc1_seq_s
    unsigned short FrameWidth;     // actual frame width
    unsigned short FrameHeight;    // actual frame height

    unsigned char profile;        // 1 = SIMPLE or MAIN, 2 = ADVANCED
    unsigned char postprocflag;
    unsigned char pulldown;
    unsigned char interlace;

    unsigned char tfcntrflag;
    unsigned char finterpflag;
    unsigned char psf;
    unsigned char tileFormat; // 0: TBL; 1: KBL; 2: Tile16x16

    // simple,main
    unsigned char multires;
    unsigned char syncmarker;
    unsigned char rangered;
    unsigned char maxbframes;

    // Fields from vc1_entrypoint_s
    unsigned char dquant;
    unsigned char panscan_flag;
    unsigned char refdist_flag;
    unsigned char quantizer;

    unsigned char extended_mv;
    unsigned char extended_dmv;
    unsigned char overlap;
    unsigned char vstransform;

    // Fields from vc1_scratch_s
    char refdist;
    char reserved1[3];               // for alignment

    // Fields from vld_vc1_pic_s
    vc1_fcm_e fcm;
    syntax_vc1_ptype_e ptype;
    int tfcntr;
    int rptfrm;
    int tff;
    int rndctrl;
    int pqindex;
    int halfqp;
    int pquantizer;
    int postproc;
    int condover;
    int transacfrm;
    int transacfrm2;
    int transdctab;
    int pqdiff;
    int abspq;
    int dquantfrm;
    vc1_dqprofile_e dqprofile;
    int dqsbedge;
    int dqdbedge;
    int dqbilevel;
    int mvrange;
    enum vc1_mvmode_e mvmode;
    enum vc1_mvmode_e mvmode2;
    int lumscale;
    int lumshift;
    int mvtab;
    int cbptab;
    int ttmbf;
    int ttfrm;
    int bfraction;
    vc1_fptype_e fptype;
    int numref;
    int reffield;
    int dmvrange;
    int intcompfield;
    int lumscale1;  //  type was char in ucode
    int lumshift1;  //  type was char in ucode
    int lumscale2;  //  type was char in ucode
    int lumshift2;  //  type was char in ucode
    int mbmodetab;
    int imvtab;
    int icbptab;
    int fourmvbptab;
    int fourmvswitch;
    int intcomp;
    int twomvbptab;
    // simple,main
    int rangeredfrm;

    // Fields from pdec_vc1_pic_s
    unsigned int   HistBufferSize;                  // in units of 256
    // frame buffers
    unsigned int   FrameStride[2];                  // [y_c]
    unsigned int   luma_top_offset;                 // offset of luma top field in units of 256
    unsigned int   luma_bot_offset;                 // offset of luma bottom field in units of 256
    unsigned int   luma_frame_offset;               // offset of luma frame in units of 256
    unsigned int   chroma_top_offset;               // offset of chroma top field in units of 256
    unsigned int   chroma_bot_offset;               // offset of chroma bottom field in units of 256
    unsigned int   chroma_frame_offset;             // offset of chroma frame in units of 256

    unsigned short CodedWidth;                      // entrypoint specific
    unsigned short CodedHeight;                     // entrypoint specific

    unsigned char  loopfilter;                      // entrypoint specific
    unsigned char  fastuvmc;                        // entrypoint specific
    unsigned char  output_memory_layout;            // picture specific
    unsigned char  ref_memory_layout[2];            // picture specific 0: fwd, 1: bwd
    unsigned char  reserved3[3];                    // for alignment

    lwdec_display_param_s displayPara;
} lwdec_vc1_pic_s;

// MPEG-2
typedef struct _lwdec_mpeg2_pic_s
{
    lwdec_pass2_otf_s encryption_params;
    unsigned char eos[16];
    unsigned char explicitEOSPresentFlag;
    unsigned char reserved0[3];
    unsigned int stream_len;
    unsigned int slice_count;

    // Driver may or may not use based upon need.
    // If 0 then default value of 1<<27 = 298ms @ 450MHz will be used in ucode.
    // Driver can send this value based upon resolution using the formula:
    // gptimer_timeout_value = 3 * (cycles required for one frame)
    unsigned int gptimer_timeout_value;

    // Fields from vld_mpeg2_seq_pic_info_s
    short FrameWidth;                   // actual frame width
    short FrameHeight;                  // actual frame height
    unsigned char picture_structure;    // 0 => Reserved, 1 => Top field, 2 => Bottom field, 3 => Frame picture. Table 6-14.
    unsigned char picture_coding_type;  // 0 => Forbidden, 1 => I, 2 => P, 3 => B, 4 => D (for MPEG-2). Table 6-12.
    unsigned char intra_dc_precision;   // 0 => 8 bits, 1=> 9 bits, 2 => 10 bits, 3 => 11 bits. Table 6-13.
    char frame_pred_frame_dct;          // as in section 6.3.10
    char concealment_motion_vectors;    // as in section 6.3.10
    char intra_vlc_format;              // as in section 6.3.10
    char tileFormat;                    // 0: TBL; 1: KBL; 2: Tile16x16;
    char reserved1;                     // always 0
    char f_code[4];                  // as in section 6.3.10

    // Fields from pdec_mpeg2_picture_setup_s
    unsigned short PicWidthInMbs;
    unsigned short  FrameHeightInMbs;
    unsigned int pitch_luma;
    unsigned int pitch_chroma;
    unsigned int luma_top_offset;
    unsigned int luma_bot_offset;
    unsigned int luma_frame_offset;
    unsigned int chroma_top_offset;
    unsigned int chroma_bot_offset;
    unsigned int chroma_frame_offset;
    unsigned int HistBufferSize;
    unsigned short output_memory_layout;
    unsigned short alternate_scan;
    unsigned short secondfield;
    /******************************/
    // Got rid of the union kept for compatibility with LWDEC1.
    // Removed field mpeg2, and kept rounding type.
    // LWDEC1 ucode is not using the mpeg2 field, instead using codec type from the methods.
    // Rounding type should only be set for Divx3.11.
    unsigned short rounding_type;
    /******************************/
    unsigned int MbInfoSizeInBytes;
    unsigned int q_scale_type;
    unsigned int top_field_first;
    unsigned int full_pel_fwd_vector;
    unsigned int full_pel_bwd_vector;
    unsigned char quant_mat_8x8intra[64];
    unsigned char quant_mat_8x8nonintra[64];
    unsigned int ref_memory_layout[2]; //0:for fwd; 1:for bwd

    lwdec_display_param_s displayPara;
} lwdec_mpeg2_pic_s;

// MPEG-4
typedef struct _lwdec_mpeg4_pic_s
{
    lwdec_pass2_otf_s encryption_params;
    unsigned char eos[16];
    unsigned char explicitEOSPresentFlag;
    unsigned char reserved2[3];     // for alignment
    unsigned int stream_len;
    unsigned int slice_count;
    unsigned int scratch_pic_buffer_size;

    // Driver may or may not use based upon need.
    // If 0 then default value of 1<<27 = 298ms @ 450MHz will be used in ucode.
    // Driver can send this value based upon resolution using the formula:
    // gptimer_timeout_value = 3 * (cycles required for one frame)
    unsigned int gptimer_timeout_value;

    // Fields from vld_mpeg4_seq_s
    short FrameWidth;                     // :13 video_object_layer_width
    short FrameHeight;                    // :13 video_object_layer_height
    char  vop_time_increment_bitcount;    // : 5 1..16
    char  resync_marker_disable;          // : 1
    char  tileFormat;                     // 0: TBL; 1: KBL; 2: Tile16x16
    char  reserved3;                      // for alignment

    // Fields from pdec_mpeg4_picture_setup_s
    int width;                              // : 13
    int height;                             // : 13

    unsigned int FrameStride[2];            // [y_c]
    unsigned int luma_top_offset;           // offset of luma top field in units of 256
    unsigned int luma_bot_offset;           // offset of luma bottom field in units of 256
    unsigned int luma_frame_offset;         // offset of luma frame in units of 256
    unsigned int chroma_top_offset;         // offset of chroma top field in units of 256
    unsigned int chroma_bot_offset;         // offset of chroma bottom field in units of 256
    unsigned int chroma_frame_offset;       // offset of chroma frame in units of 256

    unsigned int HistBufferSize;            // in units of 256, History buffer size

    int trd[2];                             // : 16, temporal reference frame distance (only needed for B-VOPs)
    int trb[2];                             // : 16, temporal reference B-VOP distance from fwd reference frame (only needed for B-VOPs)

    int divx_flags;                         // : 16 (bit 0: DivX interlaced chroma rounding, bit 1: Divx 4 boundary padding, bit 2: Divx IDCT)

    short vop_fcode_forward;                // : 1...7
    short vop_fcode_backward;               // : 1...7

    unsigned char interlaced;               // : 1
    unsigned char quant_type;               // : 1
    unsigned char quarter_sample;           // : 1
    unsigned char short_video_header;       // : 1

    unsigned char lwrr_output_memory_layout; // : 1 0:LW12; 1:LW24
    unsigned char ptype;                    // picture type: 0 for PTYPE_I, 1 for PTYPE_P, 2 for PTYPE_B, 3 for PTYPE_BI, 4 for PTYPE_SKIPPED
    unsigned char rnd;                      // : 1, rounding mode
    unsigned char alternate_vertical_scan_flag; // : 1

    unsigned char top_field_flag;           // : 1
    unsigned char reserved0[3];             // alignment purpose

    unsigned char intra_quant_mat[64];      // : 64*8
    unsigned char nonintra_quant_mat[64];   // : 64*8
    unsigned char ref_memory_layout[2];    //0:for fwd; 1:for bwd
    unsigned char reserved1[34];            // 256 byte alignemnt till now

    lwdec_display_param_s displayPara;
} lwdec_mpeg4_pic_s;

// VP8
enum VP8_FRAME_TYPE
{
    VP8_KEYFRAME = 0,
    VP8_INTERFRAME = 1
};

enum VP8_FRAME_SFC_ID
{
    VP8_GOLDEN_FRAME_SFC = 0,
    VP8_ALTREF_FRAME_SFC,
    VP8_LAST_FRAME_SFC,
    VP8_LWRR_FRAME_SFC
};

typedef struct _lwdec_vp8_pic_s
{
    lwdec_pass2_otf_s encryption_params;

    // Driver may or may not use based upon need.
    // If 0 then default value of 1<<27 = 298ms @ 450MHz will be used in ucode.
    // Driver can send this value based upon resolution using the formula:
    // gptimer_timeout_value = 3 * (cycles required for one frame)
    unsigned int gptimer_timeout_value;

    unsigned short FrameWidth;     // actual frame width
    unsigned short FrameHeight;    // actual frame height

    unsigned char keyFrame;        // 1: key frame; 0: not
    unsigned char version;
    unsigned char tileFormat; // 0: TBL; 1: KBL; 2: Tile16x16
    unsigned char errorConcealOn;  // 1: error conceal on; 0: off

    unsigned int  firstPartSize;   // the size of first partition(frame header and mb header partition)

    // ctx
    unsigned int   HistBufferSize;                  // in units of 256
    unsigned int   VLDBufferSize;                   // in units of 1
    // current frame buffers
    unsigned int   FrameStride[2];                  // [y_c]
    unsigned int   luma_top_offset;                 // offset of luma top field in units of 256
    unsigned int   luma_bot_offset;                 // offset of luma bottom field in units of 256
    unsigned int   luma_frame_offset;               // offset of luma frame in units of 256
    unsigned int   chroma_top_offset;               // offset of chroma top field in units of 256
    unsigned int   chroma_bot_offset;               // offset of chroma bottom field in units of 256
    unsigned int   chroma_frame_offset;             // offset of chroma frame in units of 256

    lwdec_display_param_s displayPara;

    // decode picture buffere related
    char lwrrent_output_memory_layout;
    char output_memory_layout[3];  // output LW12/LW24 setting. item 0:golden; 1: altref; 2: last

    unsigned char reserved1[4];

    // ucode return result
    unsigned int resultValue;      // ucode return the picture header info; includes copy_buffer_to_golden etc.
    unsigned int partition_offset[8];            // byte offset to each token partition (used for encrypted streams only)
} lwdec_vp8_pic_s; // size is 0xc0

#endif // __DRV_LWDEC_H_
