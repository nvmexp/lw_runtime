#ifndef OFA_DRV_H
#define OFA_DRV_H
// ===========================================================================
// CLASS LW_C9FA_OFA
//
//  a brief intro of OFA
// ===========================================================================



enum ofa_mode_t {
    MODE_EPISGM,
    MODE_PYDSGM,
    MODE_STEREO,
};


enum search_gridsize_t {
    SEARCH_GRIDSIZE_1,
    SEARCH_GRIDSIZE_2,
    SEARCH_GRIDSIZE_4,
    SEARCH_GRIDSIZE_8,
};


enum epipolar_direction_t {
    EPI_DIRECTION_FORWARD,
    EPI_DIRECTION_BACKWARD,
};

enum epipolar_ndisparity_t {
    EPI_NUM_DISPARITY_128,
    EPI_NUM_DISPARITY_256,
};

typedef struct {
    LwU32           stride                      : 16;   // surface pitch, in byte unit
    LwU32           block_height                : 7;    // BL mapping block height setting. 0: 8, 1: 16, 2: 32, 3: 64, 4: 128, 5: 256
    LwU32           bl_mode                     : 1;    // block linear mode: 0~gpu bl; 1~cheetah bl
    LwU32           reserved                    : 8;

    LwU32           buf_size                    ;       // surface size
} surface_cfg_s;                                        // 8 bytes

// ROI related control fields
// ROI: top-left position of ROI
// ROI width  = roi_end_x - roi_start_x + 1;
// ROI height = roi_end_y - roi_start_y + 1;
// Notes for ROI programming:
//   ROI width/roi_start_x needs to align to 32 pixels
//   ROI height/roi_start_y needs to align to 16 pixels
//   ROI width/height needs to follow frame size restrictions
//   ROI can't combine with grid size feature
typedef struct {
    // top-left position
    LwU32           start_x : 16;                   // ROI top-left x index (in pixel unit), available for all modes 
    LwU32           start_y : 16;                   // ROI top-left y index (in pixel unit), available for all modes

    // bottom-right position
    LwU32           end_x   : 16;                   // ROI bottom-right index (in pixel unit), available for all modes
    LwU32           end_y   : 16;                   // ROI bottom-right index (in pixel unit), available for all modes
} ofa_rect_s;                                       // 8 bytes

// This is the top level config structure passed by driver to ofa. It must be aligned
// to a 256B boundary. The first integer in this struct is a 'magic' number which driver
// must set to a specific value, and which the main Falcon app must check to see if it
// matches the expected value. This prevents running wrong combinations of traces/drvr 
// code and falcon code. Encoding is as follows:
//     Bit 31..16 = 0xc9fa
//     Bit 15..8  = Version number
//     Bit 7..0   = Revision number
// This number changes whenever there is a change to the class. If the change is small and
// backward compatible, only the revision number is incremented. If the change is major or
// not backward compatible, the revision number is reset to zero and the version number
// incremented. Falcon app should check if the upper 24 bits match the expected value, and
// terminate with error code "Lwc9faErrorOFABadMagic" if there is a mismatch. The value
// for the current class is defined as:

#define LW_OFA_DRV_MAGIC_VALUE 0xC9FA0105

typedef struct {
    // LwU32 magic number
    LwU32           magic                          ;    // magic number
    
    // LwU32 
    LwU32           frame_index                    ;    // frame index
    
    // LwU32, map to CTRL register
    LwU32           mode                        : 2;    // OFA working mode. 0: epipolar SGM optical flow; 1: pyramidal SGM optical flow; 2: stereo SGM mode
    LwU32           frame_mode                  : 2;    // OFA frame mode: 0:regular mode(whole frame); 1: ROI mode; 2: subframe mode
                      
    LwU32           num_sgmpass_minus1          : 2;    // number of total pass used in SGM. available for all modes
    LwU32           reserved_c_4                : 2; 
                      
    LwU32           bit_depth_minus8            : 4;    // support bit depth 8/10/12/16
    LwU32           reserved_c_1                : 4;
    
    // Notes:
    // HW only supports the x/y_gridsize_log2 combination that satisfy: |x_gridsize_log2 - y_gridsize_log2| <= 1
    // for HW pyramidal SGM mode, |x_gridsize_log2 - y_gridsize_log2| <= 1 && x/y_gridsize_log2 <= 2                  
    LwU32           x_gridsize_log2             : 2;    // grid size (mv/disparity granularity) in horizontal direction. see search_gridsize_t
    LwU32           reserved_c_2                : 2;    
                      
    LwU32           y_gridsize_log2             : 2;    // grid size (mv/disparity granularity) in vertical direction. see search_gridsize_t
    LwU32           reserved_c_3                :10;


    // LwU32, map to FRAME_SIZE register
    LwU32           width_minus1                :16;    // current frame width  (in pixel unit), minmimun 31, maximun 8191
    LwU32           height_minus1               :16;    // current frame height (in pixel unit), minmimun 31, maximun 8191

    // LwU32, map to EPI_CTRL register
    LwU32           direction                   : 1;    // search direction used by epiSGM mode. 0: away from epipole; 1: towards epipole. see epipolar_direction_t
    LwU32           reserved_ec_0               : 7;
        
    LwU32           num_disparity_log2_minus7   : 1;    // number of disparities used. for epiSGM/stereo mode. 0: 128 disparities; 1: 256 disparities. see epipolar_ndisparity_t
    LwU32           reserved_ec_1               :23;

    // map to EPIPOLE_X, EPIPOLE_Y, FUNDAMENTAL_M, ROTATION_M
    LwS32           epipole_x;                          // x coordinate of epipole position, in S17.3 format. used by epiSGM mode
    LwS32           epipole_y;                          // y coordinate of epipole position, in S17.3 format. used by epiSGM mode
    LwF32           fundamental_m[3][3];                // Fundamental matrix, coefficient in float format, stored in row major order. used by epiSGM mode
    LwF32           rotation_m[3][3];                   // homography matrix, coefficient in float format, stored in row major order. used by epiSGM mode

    // LwU32, map to CVC_CTRL
    LwU32           pyd_const_cost             : 5;     // const penalty cost used by pydSGM mode. only valid when reference pixel is outside of image
    LwU32           reserved_cvc_0             : 3;
                    
    LwU32           pyd_zero_hint              : 1;     // 0: HW use external hint; 1: HW use zero mv as hint, no external hint input required
                    
    LwU32           search_range_log2_minus7   : 1;     // maximun search range in horizontal/vertical direction. 0: 128 pixels; 1: 256 pixels. used by epiSGM mode
    LwU32           pyd_hint_width_1_2x        : 1;     // 0: default input hint width  (the same as output width);  1: 1/2 of default width
    LwU32           pyd_hint_height_1_2x       : 1;     // 0: default input hint height (the same as output height); 1: 1/2 of default height
    LwU32           pyd_hint_magnitude_scale_2x: 1;     // 0: default hint mv magnitude;   1: use 2x hint mv magnitude           
    LwU32           pyd_flow_width_1_2x        : 1;     // 0: default flow width ; 1: 1/2 of default flow width  (downsampled by HW)
    LwU32           pyd_flow_heigth_1_2x       : 1;     // 0: default flow height; 1: 1/2 of default flow height (downsampled by HW)
              
    LwU32           pyd_spatial_enable         : 1;     // enable/disable 3x3 spatial hint (center hint is always used even this flag is set false)
    LwU32           pyd_exthint1_enable        : 1;     // enable/disable external hint 1
    LwU32           pyd_exthint0_enable        : 1;     // enable/disable external hint 0
    LwU32           pyd_const1_enable          : 1;     // enable/disable const 1 hint 
    LwU32           pyd_const0_enable          : 1;     // enable/disable const 0 hint 
    LwU32           pyd_max_hint_per_grid      : 4;     // maximum hint/grid, used to drop hint when this threshold is met
    LwU32           pyd_exthint0_width_1_2x     : 1;    // 0: default external hint (surface 0) width  (the same as output width);  1: 1/2 of default width
    LwU32           pyd_exthint0_height_1_2x    : 1;    // 0: default external hint (surface 0) height (the same as output height); 1: 1/2 of default height
    LwU32           pyd_exthint0_magnitude_scale_2x: 1; // 0: default external hint (surface 0) mv magnitude;   1: use 2x external hint mv magnitude   
    LwU32           pyd_exthint1_width_1_2x     : 1;    // 0: default external hint (surface 1) width  (the same as output width);  1: 1/2 of default width
    LwU32           pyd_exthint1_height_1_2x    : 1;    // 0: default external hint (surface 1) height (the same as output height); 1: 1/2 of default height
    LwU32           pyd_exthint1_magnitude_scale_2x: 1; // 0: default external hint (surface 1) mv magnitude;   1: use 2x external hint mv magnitude   
    LwU32           pyd_spatial_force_3x3      : 1;     // 0: default: the feature is disabled, when hint/guide upsampling happened, HW will read 3x3 block, but only 2x2 among them are unique. 1: force 3x3 unique hint/guide block read
  
    LwU32           reserved_cvc_1             : 1;


    // LwU32, map to SGM_CTRL
    LwU32           cost_output_enable         : 1;     // enable output map output. available for all modes
    LwU32           subpixl_refine_enable      : 1;     // enable subpixel refinement. available for all modes
    LwU32           weighted_median_enable     : 1;     // enable/disable weighted median filter
    LwU32           weighted_median_kernel_size: 1;     // weighted median filter kernel size 0: 3x3, 1: 5x5. only valid when weighted_median_enable = 1
    LwU32           weighted_median_guide_enable: 2;    // enable/disalbe the guide image (8bpp only) read for weighted median filter. 0: disable, 1: use current image as guide. 2: use external guide image source, must allocate the guide image surface in this case. only valid when weighted_median_enable = 1
    LwU32           reserved_sc_0              :10;
                    
    LwU32           diag_path                  : 1;     // enable diagnoal path in SGM. available for all modes
    LwU32           reserved_sc_1              :15;

    // LwU32, map to SGM_PENALTY register
    // Notes:
    // Restrictions on penalty programming:
    // Stereo/Epipolar SGM mode: penalty_2 <= 217 - penalty_1 && penalty_2 > penalty_1; 
    // Pyramidal SGM mode: penalty_2 <= 108 && penalty_2 should > penalty_1;
    LwU32           penalty_1                  : 8;     // small penalty P1 used in SGM. available for all modes 
    LwU32           penalty_2                  : 8;     // large penalty P2 used in SGM. available for all modes
    
    LwU32           alpha                      : 2;     // coefficient of adaptive P2. adjusted P2 = -(1<<alpha) * pixel_diff + P2. available for all modes
    LwU32           reserved_sp_0              : 6;     // 
    LwU32           adaptive_penalty_2         : 1;     // enable/disable adaptive P2. available for all modes
    LwU32           reserved_sp_1              : 7;     // 
                                                        
    surface_cfg_s   lwrr_pic_surface;                   // current picture surface config
    surface_cfg_s   ref_pic_surface;                    // reference picture surface config
    surface_cfg_s   hint_mv_surface;                    // hint mv surface config, only required by pydSGM mode
    surface_cfg_s   winner_cost_surface;                // output cost surface config
    surface_cfg_s   winner_flow_surface;                // output flow/disparity surface config

    LwU32           temporary_buf_size         ;        // intermediate temporary info surface size.(reuqired for all modes, used by multi-pass SGM)
    LwU32           history_buf_size           ;        // intermediate history info surface size. (reuqired for all modes, used by SGM)

    LwU32           sgm_start_pass             : 2;     // start SGM pass, for HW verification purpose only
    LwU32           sgm_end_pass               : 2;     // end SGM pass,   for HW verification purpose only
    LwU32           reserved_sep_0             : 28;

    // Subframe & ROI related control fields
    //ROI: top-left position of ROI
    // ROI width  = roi_end_x - roi_start_x + 1;
    // ROI height = roi_end_y - roi_start_y + 1;
    // Notes for ROI programming:
    //   ROI width/roi_start_x needs to align to 32 pixels
    //   ROI height/roi_start_y needs to align to 16 pixels
    //   ROI width/height needs to follow frame size restrictions
    //   ROI can't combine with grid size feature
    LwU32           roi_start_x                :16;   // ROI top-left x index (in pixel unit), available for all modes 
    LwU32           roi_start_y                :16;   // ROI top-left y index (in pixel unit), available for all modes
    
    //ROI: bottom-right position of ROI
    LwU32           roi_end_x                  :16;   // ROI bottom-right index (in pixel unit), available for all modes
    LwU32           roi_end_y                  :16;   // ROI bottom-right index (in pixel unit), available for all modes
    
    // Subframe start/end y position need to align to 16 pixels
    // subframe_height = (subframe_end_y - subframe_start_y + 1), it needs to satisfy the following constraint:
    // it needs to satisfy the following constraints:
    //    subframe_start_y/subframe_height aligned to 16/32/64/128 pixels for grid y = 1/2/4/8 case
    //    for the last subframe, there is no subframe height alignment requirement (subframe_start_y requirement is still there)
    LwU32           subframe_start_y           :16;   // Subframe start y index (in pixel unit), available for all modes
    LwU32           subframe_end_y             :16;   // Subframe end   y index (in pixel unit), available for all modes
    
    LwU32           num_rois_minus1;                  // number of rois. Must be set to 0 if frame_mode=0
    LwU32           roi_offset;                       // offset where roi information is available. Driver has to use the same surface
                                                      // for ofa_cfg_s and roi rects and may chose any offset (aligned to 8) after
                                                      // ofa_cfg_s struct data


    // LwU32, map to SGM_DIAG_PENALTY register
    // Notes:
    // Restrictions on penalty programming:
    // Stereo/Epipolar SGM mode: penalty_2 <= 217 - penalty_1 && penalty_2 > penalty_1; 
    // Pyramidal SGM mode: penalty_2 <= 108 && penalty_2 should > penalty_1;
    LwU32           diag_penalty_1             : 8;     // small penalty P1 used in SGM for diagonal path. available for all modes
    LwU32           diag_penalty_2             : 8;     // large penalty P2 used in SGM for diagonal path. available for all modes
    LwU32           reserved_dp_0              : 16;

    // LwU32, map to PYD_HINT_WEIGHT0 register
    LwS32           pyd_spatial_center_hint_weight : 8; // weight for spatial center hint, range is [-128, 127], large weight to prioritize hint
    LwS32           slice_height_norm          : 23;    // slice_height_norm = ceil(65536/slice_height), this is used for HW to callwlate the y position in one slice.
    LwS32           reserved_hw_2              : 1;

    // LwS8[8], map to PYD_HINT_WEIGHT1/2 register
    // The mapping of index to hint location: 
    // 0: top-left, 1: top, 2: top-right, 3: left, 4: right, 5: bottom-left, 6: bottom, 7: bottom-right
    LwS8            pyd_spatial_neighbour_hint_weight[8]; // weight for spatial neigbhour hint, range is [-128, 127], large weight to prioritize hint

    // LwU32, map to PYD_HINT_WEIGHT3 register
    LwS32           pyd_const_hint0_weight     : 8;    // weight for const hint 0, range is [-128, 127], large weight to prioritize hint
    LwS32           pyd_const_hint1_weight     : 8;    // weight for const hint 0, range is [-128, 127], large weight to prioritize hint
    LwS32           pyd_external_hint0_weight  : 8;    // weight for external hint 0, range is [-128, 127], large weight to prioritize hint
    LwS32           pyd_external_hint1_weight  : 8;    // weight for external hint 0, range is [-128, 127], large weight to prioritize hint

    LwS32           pyd_const_hint0_mvx        : 16;   // mvx of const hint 0, in S10.5 format
    LwS32           pyd_const_hint0_mvy        : 16;   // mvy of const hint 0, in S10.5 format
    LwS32           pyd_const_hint1_mvx        : 16;   // mvx of const hint 1, in S10.5 format
    LwS32           pyd_const_hint1_mvy        : 16;   // mvy of const hint 1, in S10.5 format

    LwU8            pyd_const_cost_disable[4];         // disable const cost at boundary image (using padded pixels to callwlate the cost). 0~3: top, left, right, bottom

    surface_cfg_s   exthint0_mv_surface;               // external hint mv 0 surface config, only required by pydSGM mode
    surface_cfg_s   exthint1_mv_surface;               // external hint mv 1 surface config, only required by pydSGM mode
    surface_cfg_s   guide_pic_surface;                 // guide picture surface config
    LwU32           pyd_depth_hint_select      : 1;    // enable/disable depth based hint select
    LwU32           pyd_spatial_max_hint_per_grid  : 4;    // maximun spatial hint selected by depth based hint select algorithm. only valid when depth hint select is enabled
    LwU32           slice_height               : 9;    // pixels per slice except the last slice, 1~512, but need to be less than 128 grids
    LwU32           reserved_hw_3              : 16;   
    LwU32           temp_buff_compress         : 1;    // enable/disable temporary buffer data compression
    LwU32           pyd_ctrl_const_hint_first  : 1;
    surface_cfg_s   prev_guide_pic_surface;            // previous level guide picture surface config. used for depth based hint selection

    LwU8            hw_reset_only_pass0;               // reset HW submodules only at passs 0. This can be useful for perf for modules like RPC
    LwU8            reserved[3];
} ofa_cfg_s;                                          // 256 bytes

enum 
{
    SEL_CVC2SGM_COST    = 0x00000000,
    SEL_CVC2SGM_MV      = 0x00000001,
    SEL_CPF2SGM_LWR     = 0x00000002,
    SEL_CPF2SGM_HINT    = 0x00000003,
    SEL_DMA2SGM_LINE    = 0x00000004,
    SEL_DMA2SGM_LINE_M  = 0x00000005,
    SEL_DMA2SGM_TEMP    = 0x00000006,
    SEL_DMA2SGM_WIN     = 0x00000007,
    SEL_SGM2DMA_LINE    = 0x00000008,
    SEL_SGM2DMA_LINE_M  = 0x00000009,
    SEL_SGM2DMA_TEMP    = 0x0000000a,
    SEL_SGM2DMA_WIN     = 0x0000000b,
    SEL_CPF2RPF_HINT    = 0x0000000c
};

enum 
{
    SEL_RPF2RPC_FETCH   = 0x00000000,
    SEL_FBIF2EXT_RDAT   = 0x00000001,
    SEL_RPC2CVC_DATA0   = 0x00000002,
    SEL_RPC2CVC_DATA1   = 0x00000003,
    SEL_RPC2CVC_DATA2   = 0x00000004,
    SEL_RPC2CVC_DATA3   = 0x00000005,
    SEL_RPC2CVC_DATA4   = 0x00000006,
    SEL_RPC2CVC_DATA5   = 0x00000007,
    SEL_RPC2CVC_DATA6   = 0x00000008,
    SEL_EXT2FBIF_RWREQ  = 0x00000009
};

// Debug registers
// General guideline for using CRC interface debug feature:
// 1. Generate the golden CRC (using cmodel) for one level and one pass of the selected interface, e.g. level 4 - pass 3, rpc2cvc
// 2. program the golden crc to dbg_crc_partx_golden, and config the dbg_crc_lvl/pass_select, dbg_crc_intf_partx, set dbg_crc_enable = 1
// 3. kick off HW and the result could read from dgb_crc_comp_partx when finish processing the frame
typedef struct
{
    LwU32           dbg_crc_enable     : 1;     //Global enable flag for enable/disable interface crc callwlation in OFA HW
    LwU32           dbg_crc_intf_parta : 5;     //For parta to select which interface to compare crc. see PartACrcSelect for detailed control value for each interface
    LwU32           dbg_crc_intf_partb : 5;     //For partb to select which interface to compare crc. see PartBCrcSelect for detailed control value for each interface 
    LwU32           dbg_crc_lvl_select : 3;     //Specify the #level (0~4) from which the golden CRC is generated. the compare result will be set only at specified level
    LwU32           dbg_crc_pass_select: 2;     //Specify the #pass (0~2) from which the golden CRC is generated. the compare result will be set only at specified pass
    LwU32           reserved0          : 16;
    
    LwU32           dbg_crc_parta_golden[4];    //Golden crc values for part A
    LwU32           dbg_crc_partb_golden[4];    //Golden crc values for part B
            
    LwU32           dbg_crc_comp_parta : 4;     //Compare result for part A
    LwU32           dbg_crc_comp_partb : 4;     //Compare result for part B
    LwU32           dbg_ofa_cnt_ctrl   : 1;     //Enable/Disable perf counters
    LwU32           reserved1          : 23;   
    
    LwU32           dbg_engine_hw_cnt     ;     //Engine level HW cycles
    LwU32           dbg_sgm2dma_win_cnt   ;     //SGM to DMA winner packet count
    LwU32           dbg_sgm2dma_temp_cnt  ;     //SGM to DMA Temporaray packet count
    LwU32           dbg_cvc2sgm_cost_cnt  ;     //SGM to DMA winner packet count
    LwU32           dbg_cpf2cvc_pix_cnt   ;     //CPF to CVC pixel packet count
    LwU32           dbg_rpc_miss_cnt      ;     //RPF miss count
    LwU32           dbg_rpc2cvc_pix_cnt   ;     //RPF to CVC pixel packet count

    LwU8            reserved2[60];

} ofa_status_s;                                 // 128 Bytes


// Debug registers
// General guideline for using CRC interface debug feature:
// 1. Generate the golden CRC (using cmodel) for one level and one pass for all interfaces, e.g. level 4 - pass 3
// 2. program the golden crc to dbg_crc_partx_golden, and config the dbg_crc_lvl/pass_select, set dbg_crc_enable = 1
// 3. kick off HW and the compare result could read from LW_COFA_DBG_CRC_COMP_PART_A/B when finish processing the frame
// 4. raw HW CRC values could be read from LW_COFA_DBG_CRC_READ_PARTA/B, the interface idx needs to write through LW_COFA_DBG_CRC_READ_INTF_IDX, before reading the crc value 
typedef struct
{
    LwU32           dbg_crc_enable     : 1;     //Global enable flag for enable/disable interface crc callwlation in OFA HW
    LwU32           dbg_crc_init_enable: 1;     //Enable crc initialization
    LwU32           dbg_crc_comp_enable: 1;     //Enable crc golden comparision
    LwU32           dbg_crc_addr_comp  : 1;     //Enable crc callwlation for fbif address
    LwU32           dbg_crc_lvl_select : 3;     //Specify the #level (0~4) from which the golden CRC is generated. the compare result will be set only at specified level
    LwU32           dbg_crc_pass_select: 2;     //Specify the #pass (0~2) from which the golden CRC is generated. the compare result will be set only at specified pass
    LwU32           reserved0          : 24;
    
    LwU32           dbg_crc_parta_golden[14];   //Golden crc values for part A
    LwU32           dbg_crc_partb_golden[4];    //Golden crc values for part B
            
    LwU32           dbg_crc_comp_parta : 14;    //Compare result for part A
    LwU32           dbg_crc_comp_partb : 4;     //Compare result for part B
    LwU32           dbg_ofa_cnt_ctrl   : 1;     //Enable/Disable perf counters
    LwU32           reserved1          : 13;   
    
    LwU32           dbg_engine_hw_cnt     ;     //Engine level HW cycles
    LwU32           dbg_sgm2dma_win_cnt   ;     //SGM to DMA winner packet count
    LwU32           dbg_sgm2dma_temp_cnt  ;     //SGM to DMA Temporaray packet count
    LwU32           dbg_cvc2sgm_cost_cnt  ;     //SGM to DMA winner packet count
    LwU32           dbg_cpf2cvc_pix_cnt   ;     //CPF to CVC pixel packet count
    LwU32           dbg_rpc_miss_cnt      ;     //RPF miss count
    LwU32           dbg_rpc2cvc_pix_cnt   ;     //RPF to CVC pixel packet count

    LwU32           dbg_crc_parta_crc[14];      //HW crc values for part A
    LwU32           dbg_crc_partb_crc[4];       //HW crc values for part B

    LwU8            reserved[72];
} ofa_status_s_v1;                                 // 256 Bytes

typedef struct
{
    LwU8            weights[256];               //weights lookup table used by the weighted median filter
} ofa_weighted_median_weight_s;                 // 256 Bytes



#endif // OFA_DRV_H

