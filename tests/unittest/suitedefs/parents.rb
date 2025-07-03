#************************ BEGIN COPYRIGHT NOTICE ***************************#
#                                                                           #
#          Copyright (c) LWPU Corporation.  All rights reserved.          #
#                                                                           #
# All information contained herein is proprietary and confidential to       #
# LWPU Corporation.  Any use, reproduction, or disclosure without the     #
# written permission of LWPU Corporation is prohibited.                   #
#                                                                           #
#************************** END COPYRIGHT NOTICE ***************************#

#***************************************************************************#
#                            Module : parents                               #
#         Defintion for tree Struture of suites in Unit Test infra          #
#                    contains all, but leaf level nodes                     #
#***************************************************************************#

require 'suite_tree'
require 'unit_suite'

################################################################################
#                        Suite Creation Section                                #
################################################################################

# this is the root node
$unit = (SuiteTree.new("unit", TRUE)) 

$resman = (SuiteTree.new("resman", TRUE)) 

$arch = (SuiteTree.new("arch", FALSE)) 

	dir_dummy = (SuiteTree.new("dir_dummy", FALSE)) 

$kernel = (SuiteTree.new("kernel", TRUE)) 



dir_am = (SuiteTree.new("dir_am", FALSE)) 




dir_bif = (SuiteTree.new("dir_bif", FALSE)) 

	dir_bif_fermi = (SuiteTree.new("dir_bif_fermi", FALSE)) 
	dir_bif_lw = (SuiteTree.new("dir_bif_lw", FALSE)) 
	dir_bif_lw4 = (SuiteTree.new("dir_bif_lw4", FALSE)) 
	dir_bif_lw40 = (SuiteTree.new("dir_bif_lw40", FALSE)) 
	dir_bif_lw50 = (SuiteTree.new("dir_bif_lw50", FALSE)) 
	dir_bif_sims = (SuiteTree.new("dir_bif_sims", FALSE)) 



dir_bsp = (SuiteTree.new("dir_bsp", FALSE)) 

	dir_bsp_fermi = (SuiteTree.new("dir_bsp_fermi", FALSE)) 
	dir_bsp_g82 = (SuiteTree.new("dir_bsp_g82", FALSE)) 
	dir_bsp_g98 = (SuiteTree.new("dir_bsp_g98", FALSE)) 
	dir_bsp_gt212 = (SuiteTree.new("dir_bsp_gt212", FALSE)) 
	dir_bsp_lw = (SuiteTree.new("dir_bsp_lw", FALSE)) 



dir_btree = (SuiteTree.new("dir_btree", FALSE)) 

	dir_btree_lw = (SuiteTree.new("dir_btree_lw", FALSE)) 



dir_bus = (SuiteTree.new("dir_bus", FALSE)) 

	dir_bus_fermi = (SuiteTree.new("dir_bus_fermi", FALSE)) 
	dir_bus_lw = (SuiteTree.new("dir_bus_lw", FALSE)) 
	dir_bus_lw40 = (SuiteTree.new("dir_bus_lw40", FALSE)) 
	dir_bus_lw50 = (SuiteTree.new("dir_bus_lw50", FALSE)) 
	dir_bus_sims = (SuiteTree.new("dir_bus_sims", FALSE)) 



dir_ce = (SuiteTree.new("dir_ce", FALSE)) 

	dir_ce_fermi = (SuiteTree.new("dir_ce_fermi", FALSE)) 
	dir_ce_lw = (SuiteTree.new("dir_ce_lw", FALSE)) 
	dir_ce_lw50 = (SuiteTree.new("dir_ce_lw50", FALSE)) 



dir_cipher = (SuiteTree.new("dir_cipher", FALSE)) 

	dir_cipher_fermi = (SuiteTree.new("dir_cipher_fermi", FALSE)) 
	dir_cipher_g82 = (SuiteTree.new("dir_cipher_g82", FALSE)) 
	dir_cipher_g98 = (SuiteTree.new("dir_cipher_g98", FALSE)) 
	dir_cipher_gt212 = (SuiteTree.new("dir_cipher_gt212", FALSE)) 
	dir_cipher_igt21a = (SuiteTree.new("dir_cipher_igt21a", FALSE)) 
	dir_cipher_mcp77 = (SuiteTree.new("dir_cipher_mcp77", FALSE)) 
	dir_cipher_lw = (SuiteTree.new("dir_cipher_lw", FALSE)) 



dir_class = (SuiteTree.new("dir_class", FALSE)) 

	dir_class_lw = (SuiteTree.new("dir_class_lw", FALSE)) 



dir_clk = (SuiteTree.new("dir_clk", FALSE)) 

	dir_clk_fermi = (SuiteTree.new("dir_clk_fermi", FALSE)) 
	dir_clk_lw = (SuiteTree.new("dir_clk_lw", FALSE)) 
	dir_clk_lw40 = (SuiteTree.new("dir_clk_lw40", FALSE)) 
	dir_clk_lw50 = (SuiteTree.new("dir_clk_lw50", FALSE)) 
	dir_clk_sims = (SuiteTree.new("dir_clk_sims", FALSE)) 



dir_dac = (SuiteTree.new("dir_dac", FALSE)) 

	dir_dac_fermi = (SuiteTree.new("dir_dac_fermi", FALSE)) 
	dir_dac_lw = (SuiteTree.new("dir_dac_lw", FALSE)) 
	dir_dac_lw10 = (SuiteTree.new("dir_dac_lw10", FALSE)) 
	dir_dac_lw20 = (SuiteTree.new("dir_dac_lw20", FALSE)) 
	dir_dac_lw30 = (SuiteTree.new("dir_dac_lw30", FALSE)) 
	dir_dac_lw4 = (SuiteTree.new("dir_dac_lw4", FALSE)) 
	dir_dac_lw40 = (SuiteTree.new("dir_dac_lw40", FALSE)) 
	dir_dac_lw50 = (SuiteTree.new("dir_dac_lw50", FALSE)) 
	dir_dac_sims = (SuiteTree.new("dir_dac_sims", FALSE)) 
	dir_dac_v02 = (SuiteTree.new("dir_dac_v02", FALSE)) 



dir_devinit = (SuiteTree.new("dir_devinit", FALSE)) 

	dir_devinit_fcode = (SuiteTree.new("dir_devinit_fcode", FALSE)) 
	dir_devinit_lw = (SuiteTree.new("dir_devinit_lw", FALSE)) 
	dir_devinit_pcvbios = (SuiteTree.new("dir_devinit_pcvbios", FALSE)) 



dir_disp = (SuiteTree.new("dir_disp", FALSE)) 

	dir_disp_fermi = (SuiteTree.new("dir_disp_fermi", FALSE)) 
	dir_disp_lw = (SuiteTree.new("dir_disp_lw", FALSE)) 
	dir_disp_lw10 = (SuiteTree.new("dir_disp_lw10", FALSE)) 
	dir_disp_lw20 = (SuiteTree.new("dir_disp_lw20", FALSE)) 
	dir_disp_lw30 = (SuiteTree.new("dir_disp_lw30", FALSE)) 
	dir_disp_lw4 = (SuiteTree.new("dir_disp_lw4", FALSE)) 
	dir_disp_lw40 = (SuiteTree.new("dir_disp_lw40", FALSE)) 
	dir_disp_lw50 = (SuiteTree.new("dir_disp_lw50", FALSE)) 
	dir_disp_v02 = (SuiteTree.new("dir_disp_v02", FALSE)) 



dir_dma = (SuiteTree.new("dir_dma", FALSE)) 

	dir_dma_fermi = (SuiteTree.new("dir_dma_fermi", FALSE)) 
	dir_dma_lw = (SuiteTree.new("dir_dma_lw", FALSE)) 
	dir_dma_lw30 = (SuiteTree.new("dir_dma_lw30", FALSE)) 
	dir_dma_lw4 = (SuiteTree.new("dir_dma_lw4", FALSE)) 
	dir_dma_lw40 = (SuiteTree.new("dir_dma_lw40", FALSE)) 
	dir_dma_lw50 = (SuiteTree.new("dir_dma_lw50", FALSE)) 
	dir_dma_sims = (SuiteTree.new("dir_dma_sims", FALSE)) 



dir_extdev = (SuiteTree.new("dir_extdev", FALSE)) 

	dir_extdev_lw = (SuiteTree.new("dir_extdev_lw", FALSE)) 



dir_fan = (SuiteTree.new("dir_fan", FALSE)) 

	dir_fan_fermi = (SuiteTree.new("dir_fan_fermi", FALSE)) 
	dir_fan_lw = (SuiteTree.new("dir_fan_lw", FALSE)) 
	dir_fan_lw50 = (SuiteTree.new("dir_fan_lw50", FALSE)) 
	dir_fan_v02 = (SuiteTree.new("dir_fan_v02", FALSE)) 



dir_fb = (SuiteTree.new("dir_fb", TRUE)) 

	dir_fb_fermi = (SuiteTree.new("dir_fb_fermi", TRUE)) 
	dir_fb_kepler = (SuiteTree.new("dir_fb_kepler", FALSE)) 
	dir_fb_lw = (SuiteTree.new("dir_fb_lw", FALSE)) 
	dir_fb_lw10 = (SuiteTree.new("dir_fb_lw10", FALSE)) 
	dir_fb_lw30 = (SuiteTree.new("dir_fb_lw30", FALSE)) 
	dir_fb_lw40 = (SuiteTree.new("dir_fb_lw40", FALSE)) 
	dir_fb_lw50 = (SuiteTree.new("dir_fb_lw50", FALSE)) 
	dir_fb_sims = (SuiteTree.new("dir_fb_sims", FALSE)) 



dir_fifo = (SuiteTree.new("dir_fifo", TRUE)) 

	dir_fifo_fermi = (SuiteTree.new("dir_fifo_fermi", TRUE)) 
	dir_fifo_kepler = (SuiteTree.new("dir_fifo_kepler", FALSE)) 
	dir_fifo_lw = (SuiteTree.new("dir_fifo_lw", FALSE)) 
	dir_fifo_lw10 = (SuiteTree.new("dir_fifo_lw10", FALSE)) 
	dir_fifo_lw20 = (SuiteTree.new("dir_fifo_lw20", FALSE)) 
	dir_fifo_lw4 = (SuiteTree.new("dir_fifo_lw4", FALSE)) 
	dir_fifo_lw40 = (SuiteTree.new("dir_fifo_lw40", FALSE)) 
	dir_fifo_lw50 = (SuiteTree.new("dir_fifo_lw50", FALSE)) 
	dir_fifo_sims = (SuiteTree.new("dir_fifo_sims", FALSE)) 



dir_flcn = (SuiteTree.new("dir_flcn", FALSE)) 

	dir_flcn_fermi = (SuiteTree.new("dir_flcn_fermi", FALSE)) 
	dir_flcn_g98 = (SuiteTree.new("dir_flcn_g98", FALSE)) 
	dir_flcn_gt212 = (SuiteTree.new("dir_flcn_gt212", FALSE)) 
	dir_flcn_igt21a = (SuiteTree.new("dir_flcn_igt21a", FALSE)) 
	dir_flcn_lw = (SuiteTree.new("dir_flcn_lw", FALSE)) 
	dir_flcn_v04 = (SuiteTree.new("dir_flcn_v04", FALSE)) 



dir_fuse = (SuiteTree.new("dir_fuse", FALSE)) 

	dir_fuse_fermi = (SuiteTree.new("dir_fuse_fermi", FALSE)) 
	dir_fuse_lw = (SuiteTree.new("dir_fuse_lw", FALSE)) 
	dir_fuse_lw40 = (SuiteTree.new("dir_fuse_lw40", FALSE)) 
	dir_fuse_lw50 = (SuiteTree.new("dir_fuse_lw50", FALSE)) 



dir_gpu = (SuiteTree.new("dir_gpu", TRUE)) 

	dir_gpu_fermi = (SuiteTree.new("dir_gpu_fermi", FALSE)) 
	dir_gpu_lw = (SuiteTree.new("dir_gpu_lw", FALSE)) 
	dir_gpu_lw10 = (SuiteTree.new("dir_gpu_lw10", TRUE)) 
	dir_gpu_lw30 = (SuiteTree.new("dir_gpu_lw30", FALSE)) 
	dir_gpu_lw4 = (SuiteTree.new("dir_gpu_lw4", FALSE)) 
	dir_gpu_lw40 = (SuiteTree.new("dir_gpu_lw40", FALSE)) 
	dir_gpu_lw50 = (SuiteTree.new("dir_gpu_lw50", FALSE)) 
	dir_gpu_sims = (SuiteTree.new("dir_gpu_sims", FALSE)) 



dir_gr = (SuiteTree.new("dir_gr", FALSE)) 

	dir_gr_fermi = (SuiteTree.new("dir_gr_fermi", FALSE)) 
	dir_gr_kepler = (SuiteTree.new("dir_gr_kepler", FALSE)) 
	dir_gr_lw = (SuiteTree.new("dir_gr_lw", FALSE)) 
	dir_gr_lw10 = (SuiteTree.new("dir_gr_lw10", FALSE)) 
	dir_gr_lw20 = (SuiteTree.new("dir_gr_lw20", FALSE)) 
	dir_gr_lw4 = (SuiteTree.new("dir_gr_lw4", FALSE)) 
	dir_gr_lw40 = (SuiteTree.new("dir_gr_lw40", FALSE)) 
	dir_gr_lw50 = (SuiteTree.new("dir_gr_lw50", FALSE)) 
	dir_gr_sims = (SuiteTree.new("dir_gr_sims", FALSE)) 



dir_gsync = (SuiteTree.new("dir_gsync", FALSE)) 

	dir_gsync_lw = (SuiteTree.new("dir_gsync_lw", FALSE)) 



dir_gvi = (SuiteTree.new("dir_gvi", FALSE)) 

	dir_gvi_lw = (SuiteTree.new("dir_gvi_lw", FALSE)) 



dir_hal = (SuiteTree.new("dir_hal", FALSE)) 

	dir_hal_fermi = (SuiteTree.new("dir_hal_fermi", FALSE)) 
	dir_hal_kepler = (SuiteTree.new("dir_hal_kepler", FALSE)) 
	dir_hal_lw = (SuiteTree.new("dir_hal_lw", FALSE)) 
	dir_hal_lw40 = (SuiteTree.new("dir_hal_lw40", FALSE)) 
	dir_hal_lw50 = (SuiteTree.new("dir_hal_lw50", FALSE)) 
	dir_hal_sims = (SuiteTree.new("dir_hal_sims", FALSE)) 



dir_hbloat = (SuiteTree.new("dir_hbloat", FALSE)) 

	dir_hbloat_lw = (SuiteTree.new("dir_hbloat_lw", FALSE)) 
	dir_hbloat_lw17 = (SuiteTree.new("dir_hbloat_lw17", FALSE)) 
	dir_hbloat_lw44 = (SuiteTree.new("dir_hbloat_lw44", FALSE)) 



dir_heap = (SuiteTree.new("dir_heap", FALSE)) 

	dir_heap_lw = (SuiteTree.new("dir_heap_lw", FALSE)) 



dir_inforom = (SuiteTree.new("dir_inforom", FALSE)) 

	dir_inforom_fermi = (SuiteTree.new("dir_inforom_fermi", FALSE)) 
	dir_inforom_lw = (SuiteTree.new("dir_inforom_lw", FALSE)) 



dir_instmem = (SuiteTree.new("dir_instmem", FALSE)) 

	dir_instmem_lw = (SuiteTree.new("dir_instmem_lw", FALSE)) 
	dir_instmem_lw4 = (SuiteTree.new("dir_instmem_lw4", FALSE)) 
	dir_instmem_lw40 = (SuiteTree.new("dir_instmem_lw40", FALSE)) 
	dir_instmem_lw50 = (SuiteTree.new("dir_instmem_lw50", FALSE)) 
	dir_instmem_sims = (SuiteTree.new("dir_instmem_sims", FALSE)) 



dir_intr = (SuiteTree.new("dir_intr", FALSE)) 

	dir_intr_fermi = (SuiteTree.new("dir_intr_fermi", FALSE)) 
	dir_intr_lw = (SuiteTree.new("dir_intr_lw", FALSE)) 



dir_mc = (SuiteTree.new("dir_mc", FALSE)) 

	dir_mc_fermi = (SuiteTree.new("dir_mc_fermi", FALSE)) 
	dir_mc_lw = (SuiteTree.new("dir_mc_lw", FALSE)) 
	dir_mc_lw10 = (SuiteTree.new("dir_mc_lw10", FALSE)) 
	dir_mc_lw20 = (SuiteTree.new("dir_mc_lw20", FALSE)) 
	dir_mc_lw4 = (SuiteTree.new("dir_mc_lw4", FALSE)) 
	dir_mc_lw40 = (SuiteTree.new("dir_mc_lw40", FALSE)) 
	dir_mc_lw50 = (SuiteTree.new("dir_mc_lw50", FALSE)) 
	dir_mc_sims = (SuiteTree.new("dir_mc_sims", FALSE)) 



dir_me = (SuiteTree.new("dir_me", FALSE)) 

	dir_me_lw = (SuiteTree.new("dir_me_lw", FALSE)) 
	dir_me_lw40 = (SuiteTree.new("dir_me_lw40", FALSE)) 
	dir_me_lw50 = (SuiteTree.new("dir_me_lw50", FALSE)) 



dir_modular = (SuiteTree.new("dir_modular", FALSE)) 

	dir_modular_lw = (SuiteTree.new("dir_modular_lw", FALSE)) 



dir_mp = (SuiteTree.new("dir_mp", FALSE)) 

	dir_mp_g84 = (SuiteTree.new("dir_mp_g84", FALSE)) 
	dir_mp_lw = (SuiteTree.new("dir_mp_lw", FALSE)) 
	dir_mp_lw10 = (SuiteTree.new("dir_mp_lw10", FALSE)) 
	dir_mp_lw4 = (SuiteTree.new("dir_mp_lw4", FALSE)) 
	dir_mp_lw40 = (SuiteTree.new("dir_mp_lw40", FALSE)) 



dir_mpeg = (SuiteTree.new("dir_mpeg", FALSE)) 

	dir_mpeg_lw = (SuiteTree.new("dir_mpeg_lw", FALSE)) 
	dir_mpeg_lw10 = (SuiteTree.new("dir_mpeg_lw10", FALSE)) 
	dir_mpeg_lw30 = (SuiteTree.new("dir_mpeg_lw30", FALSE)) 
	dir_mpeg_lw40 = (SuiteTree.new("dir_mpeg_lw40", FALSE)) 
	dir_mpeg_lw50 = (SuiteTree.new("dir_mpeg_lw50", FALSE)) 



dir_msdec = (SuiteTree.new("dir_msdec", FALSE)) 

	dir_msdec_fermi = (SuiteTree.new("dir_msdec_fermi", FALSE)) 
	dir_msdec_g98 = (SuiteTree.new("dir_msdec_g98", FALSE)) 
	dir_msdec_gt212 = (SuiteTree.new("dir_msdec_gt212", FALSE)) 
	dir_msdec_lw = (SuiteTree.new("dir_msdec_lw", FALSE)) 



dir_msenc = (SuiteTree.new("dir_msenc", FALSE)) 

	dir_msenc_lw = (SuiteTree.new("dir_msenc_lw", FALSE)) 
	dir_msenc_v01 = (SuiteTree.new("dir_msenc_v01", FALSE)) 



dir_lwd = (SuiteTree.new("dir_lwd", FALSE)) 

	dir_lwd_lw = (SuiteTree.new("dir_lwd_lw", FALSE)) 



dir_perfctl = (SuiteTree.new("dir_perfctl", FALSE)) 

	dir_perfctl_kepler = (SuiteTree.new("dir_perfctl_kepler", FALSE)) 
	dir_perfctl_fermi = (SuiteTree.new("dir_perfctl_fermi", FALSE)) 
	dir_perfctl_lw = (SuiteTree.new("dir_perfctl_lw", FALSE)) 
	dir_perfctl_lw30 = (SuiteTree.new("dir_perfctl_lw30", FALSE)) 
	dir_perfctl_lw40 = (SuiteTree.new("dir_perfctl_lw40", FALSE)) 
	dir_perfctl_lw50 = (SuiteTree.new("dir_perfctl_lw50", FALSE)) 



dir_pg = (SuiteTree.new("dir_pg", FALSE)) 

	dir_pg_fermi = (SuiteTree.new("dir_pg_fermi", FALSE)) 
	dir_pg_lw = (SuiteTree.new("dir_pg_lw", FALSE)) 
	dir_pg_lw50 = (SuiteTree.new("dir_pg_lw50", FALSE)) 



dir_pmgr = (SuiteTree.new("dir_pmgr", FALSE)) 

	dir_pmgr_fermi = (SuiteTree.new("dir_pmgr_fermi", FALSE)) 
	dir_pmgr_lw = (SuiteTree.new("dir_pmgr_lw", FALSE)) 
	dir_pmgr_lw10 = (SuiteTree.new("dir_pmgr_lw10", FALSE)) 
	dir_pmgr_lw4 = (SuiteTree.new("dir_pmgr_lw4", FALSE)) 
	dir_pmgr_lw40 = (SuiteTree.new("dir_pmgr_lw40", FALSE)) 
	dir_pmgr_lw50 = (SuiteTree.new("dir_pmgr_lw50", FALSE)) 
	dir_pmgr_sims = (SuiteTree.new("dir_pmgr_sims", FALSE)) 
	dir_pmgr_v02 = (SuiteTree.new("dir_pmgr_v02", FALSE)) 



dir_pmu = (SuiteTree.new("dir_pmu", FALSE)) 

	dir_pmu_fermi = (SuiteTree.new("dir_pmu_fermi", FALSE)) 
	dir_pmu_lw = (SuiteTree.new("dir_pmu_lw", FALSE)) 
	dir_pmu_lw50 = (SuiteTree.new("dir_pmu_lw50", FALSE)) 



dir_rc = (SuiteTree.new("dir_rc", FALSE)) 

	dir_rc_lw = (SuiteTree.new("dir_rc_lw", FALSE)) 
	dir_rc_lw40 = (SuiteTree.new("dir_rc_lw40", FALSE)) 
	dir_rc_lw50 = (SuiteTree.new("dir_rc_lw50", FALSE)) 



dir_res = (SuiteTree.new("dir_res", FALSE)) 

	dir_res_lw = (SuiteTree.new("dir_res_lw", FALSE)) 



dir_seq = (SuiteTree.new("dir_seq", FALSE)) 

	dir_seq_fermi = (SuiteTree.new("dir_seq_fermi", FALSE)) 
	dir_seq_lw = (SuiteTree.new("dir_seq_lw", FALSE)) 
	dir_seq_lw10 = (SuiteTree.new("dir_seq_lw10", FALSE)) 
	dir_seq_lw40 = (SuiteTree.new("dir_seq_lw40", FALSE)) 
	dir_seq_lw50 = (SuiteTree.new("dir_seq_lw50", FALSE)) 



dir_smu = (SuiteTree.new("dir_smu", FALSE)) 

	dir_smu_lw = (SuiteTree.new("dir_smu_lw", FALSE)) 
	dir_smu_lw50 = (SuiteTree.new("dir_smu_lw50", FALSE)) 



dir_spb = (SuiteTree.new("dir_spb", FALSE)) 

	dir_spb_lw = (SuiteTree.new("dir_spb_lw", FALSE)) 



dir_ss = (SuiteTree.new("dir_ss", FALSE)) 

	dir_ss_lw = (SuiteTree.new("dir_ss_lw", FALSE)) 
	dir_ss_lw10 = (SuiteTree.new("dir_ss_lw10", FALSE)) 
	dir_ss_lw30 = (SuiteTree.new("dir_ss_lw30", FALSE)) 
	dir_ss_lw40 = (SuiteTree.new("dir_ss_lw40", FALSE)) 



dir_state = (SuiteTree.new("dir_state", FALSE)) 

	dir_state_lw = (SuiteTree.new("dir_state_lw", FALSE)) 



dir_stereo = (SuiteTree.new("dir_stereo", FALSE)) 

	dir_stereo_fermi = (SuiteTree.new("dir_stereo_fermi", FALSE)) 
	dir_stereo_lw = (SuiteTree.new("dir_stereo_lw", FALSE)) 
	dir_stereo_lw10 = (SuiteTree.new("dir_stereo_lw10", FALSE)) 
	dir_stereo_lw30 = (SuiteTree.new("dir_stereo_lw30", FALSE)) 
	dir_stereo_lw40 = (SuiteTree.new("dir_stereo_lw40", FALSE)) 
	dir_stereo_lw50 = (SuiteTree.new("dir_stereo_lw50", FALSE)) 



dir_sw = (SuiteTree.new("dir_sw", FALSE)) 

	dir_sw_lw = (SuiteTree.new("dir_sw_lw", FALSE)) 



dir_syscon = (SuiteTree.new("dir_syscon", FALSE)) 

	dir_syscon_lw = (SuiteTree.new("dir_syscon_lw", FALSE)) 



dir_thermctl = (SuiteTree.new("dir_thermctl", FALSE)) 

	dir_thermctl_fermi = (SuiteTree.new("dir_thermctl_fermi", FALSE)) 
	dir_thermctl_lw = (SuiteTree.new("dir_thermctl_lw", FALSE)) 
	dir_thermctl_lw30 = (SuiteTree.new("dir_thermctl_lw30", FALSE)) 
	dir_thermctl_lw40 = (SuiteTree.new("dir_thermctl_lw40", FALSE)) 
	dir_thermctl_lw50 = (SuiteTree.new("dir_thermctl_lw50", FALSE)) 



dir_tmr = (SuiteTree.new("dir_tmr", FALSE)) 

	dir_tmr_fermi = (SuiteTree.new("dir_tmr_fermi", FALSE)) 
	dir_tmr_lw = (SuiteTree.new("dir_tmr_lw", FALSE)) 
	dir_tmr_lw4 = (SuiteTree.new("dir_tmr_lw4", FALSE)) 
	dir_tmr_lw40 = (SuiteTree.new("dir_tmr_lw40", FALSE)) 
	dir_tmr_lw50 = (SuiteTree.new("dir_tmr_lw50", FALSE)) 
	dir_tmr_sims = (SuiteTree.new("dir_tmr_sims", FALSE)) 



dir_tvo = (SuiteTree.new("dir_tvo", FALSE)) 

	dir_tvo_lw = (SuiteTree.new("dir_tvo_lw", FALSE)) 



dir_vblank = (SuiteTree.new("dir_vblank", FALSE)) 

	dir_vblank_lw = (SuiteTree.new("dir_vblank_lw", FALSE)) 



dir_vic = (SuiteTree.new("dir_vic", FALSE)) 

	dir_vic_lw = (SuiteTree.new("dir_vic_lw", FALSE)) 
	dir_vic_v01 = (SuiteTree.new("dir_vic_v01", FALSE)) 



dir_video = (SuiteTree.new("dir_video", FALSE)) 

	dir_video_lw = (SuiteTree.new("dir_video_lw", FALSE)) 
	dir_video_lw10 = (SuiteTree.new("dir_video_lw10", FALSE)) 
	dir_video_lw30 = (SuiteTree.new("dir_video_lw30", FALSE)) 
	dir_video_lw40 = (SuiteTree.new("dir_video_lw40", FALSE)) 



dir_vp = (SuiteTree.new("dir_vp", FALSE)) 

	dir_vp_fermi = (SuiteTree.new("dir_vp_fermi", FALSE)) 
	dir_vp_g82 = (SuiteTree.new("dir_vp_g82", FALSE)) 
	dir_vp_g98 = (SuiteTree.new("dir_vp_g98", FALSE)) 
	dir_vp_gt212 = (SuiteTree.new("dir_vp_gt212", FALSE)) 
	dir_vp_lw = (SuiteTree.new("dir_vp_lw", FALSE)) 
	dir_vp_lw41 = (SuiteTree.new("dir_vp_lw41", FALSE)) 
	dir_vp_lw43 = (SuiteTree.new("dir_vp_lw43", FALSE)) 
	dir_vp_lw44 = (SuiteTree.new("dir_vp_lw44", FALSE)) 
	dir_vp_lw50 = (SuiteTree.new("dir_vp_lw50", FALSE)) 
	dir_vp_vgpu = (SuiteTree.new("dir_vp_vgpu", FALSE)) 

################################################################################
#                 Suite Addition Section                                       #
################################################################################

$unit.add($resman)


$resman.add($arch)

$arch.add(dir_dummy)


$resman.add($kernel)

$kernel.add(dir_am)




$kernel.add(dir_bif)

	dir_bif.add(dir_bif_fermi)
	dir_bif.add(dir_bif_lw)
	dir_bif.add(dir_bif_lw4)
	dir_bif.add(dir_bif_lw40)
	dir_bif.add(dir_bif_lw50)
	dir_bif.add(dir_bif_sims)



$kernel.add(dir_bsp)

	dir_bsp.add(dir_bsp_fermi)
	dir_bsp.add(dir_bsp_g82)
	dir_bsp.add(dir_bsp_g98)
	dir_bsp.add(dir_bsp_gt212)
	dir_bsp.add(dir_bsp_lw)



$kernel.add(dir_btree)

	dir_btree.add(dir_btree_lw)



$kernel.add(dir_bus)

	dir_bus.add(dir_bus_fermi)
	dir_bus.add(dir_bus_lw)
	dir_bus.add(dir_bus_lw40)
	dir_bus.add(dir_bus_lw50)
	dir_bus.add(dir_bus_sims)



$kernel.add(dir_ce)

	dir_ce.add(dir_ce_fermi)
	dir_ce.add(dir_ce_lw)
	dir_ce.add(dir_ce_lw50)



$kernel.add(dir_cipher)

	dir_cipher.add(dir_cipher_fermi)
	dir_cipher.add(dir_cipher_g82)
	dir_cipher.add(dir_cipher_g98)
	dir_cipher.add(dir_cipher_gt212)
	dir_cipher.add(dir_cipher_igt21a)
	dir_cipher.add(dir_cipher_mcp77)
	dir_cipher.add(dir_cipher_lw)



$kernel.add(dir_class)

	dir_class.add(dir_class_lw)



$kernel.add(dir_clk)

	dir_clk.add(dir_clk_fermi)
	dir_clk.add(dir_clk_lw)
	dir_clk.add(dir_clk_lw40)
	dir_clk.add(dir_clk_lw50)
	dir_clk.add(dir_clk_sims)



$kernel.add(dir_dac)

	dir_dac.add(dir_dac_fermi)
	dir_dac.add(dir_dac_lw)
	dir_dac.add(dir_dac_lw10)
	dir_dac.add(dir_dac_lw20)
	dir_dac.add(dir_dac_lw30)
	dir_dac.add(dir_dac_lw4)
	dir_dac.add(dir_dac_lw40)
	dir_dac.add(dir_dac_lw50)
	dir_dac.add(dir_dac_sims)
	dir_dac.add(dir_dac_v02)



$kernel.add(dir_devinit)

	dir_devinit.add(dir_devinit_fcode)
	dir_devinit.add(dir_devinit_lw)
	dir_devinit.add(dir_devinit_pcvbios)



$kernel.add(dir_disp)

	dir_disp.add(dir_disp_fermi)
	dir_disp.add(dir_disp_lw)
	dir_disp.add(dir_disp_lw10)
	dir_disp.add(dir_disp_lw20)
	dir_disp.add(dir_disp_lw30)
	dir_disp.add(dir_disp_lw4)
	dir_disp.add(dir_disp_lw40)
	dir_disp.add(dir_disp_lw50)
	dir_disp.add(dir_disp_v02)



$kernel.add(dir_dma)

	dir_dma.add(dir_dma_fermi)
	dir_dma.add(dir_dma_lw)
	dir_dma.add(dir_dma_lw30)
	dir_dma.add(dir_dma_lw4)
	dir_dma.add(dir_dma_lw40)
	dir_dma.add(dir_dma_lw50)
	dir_dma.add(dir_dma_sims)



$kernel.add(dir_extdev)

	dir_extdev.add(dir_extdev_lw)



$kernel.add(dir_fan)

	dir_fan.add(dir_fan_fermi)
	dir_fan.add(dir_fan_lw)
	dir_fan.add(dir_fan_lw50)
	dir_fan.add(dir_fan_v02)



$kernel.add(dir_fb)

	dir_fb.add(dir_fb_fermi)
	dir_fb.add(dir_fb_kepler)
	dir_fb.add(dir_fb_lw)
	dir_fb.add(dir_fb_lw10)
	dir_fb.add(dir_fb_lw30)
	dir_fb.add(dir_fb_lw40)
	dir_fb.add(dir_fb_lw50)
	dir_fb.add(dir_fb_sims)



$kernel.add(dir_fifo)

	dir_fifo.add(dir_fifo_fermi)
	dir_fifo.add(dir_fifo_kepler)
	dir_fifo.add(dir_fifo_lw)
	dir_fifo.add(dir_fifo_lw10)
	dir_fifo.add(dir_fifo_lw20)
	dir_fifo.add(dir_fifo_lw4)
	dir_fifo.add(dir_fifo_lw40)
	dir_fifo.add(dir_fifo_lw50)
	dir_fifo.add(dir_fifo_sims)



$kernel.add(dir_flcn)

	dir_flcn.add(dir_flcn_fermi)
	dir_flcn.add(dir_flcn_g98)
	dir_flcn.add(dir_flcn_gt212)
	dir_flcn.add(dir_flcn_igt21a)
	dir_flcn.add(dir_flcn_lw)
	dir_flcn.add(dir_flcn_v04)



$kernel.add(dir_fuse)

	dir_fuse.add(dir_fuse_fermi)
	dir_fuse.add(dir_fuse_lw)
	dir_fuse.add(dir_fuse_lw40)
	dir_fuse.add(dir_fuse_lw50)



$kernel.add(dir_gpu)

	dir_gpu.add(dir_gpu_fermi)
	dir_gpu.add(dir_gpu_lw)
	dir_gpu.add(dir_gpu_lw10)
	dir_gpu.add(dir_gpu_lw30)
	dir_gpu.add(dir_gpu_lw4)
	dir_gpu.add(dir_gpu_lw40)
	dir_gpu.add(dir_gpu_lw50)
	dir_gpu.add(dir_gpu_sims)



$kernel.add(dir_gr)

	dir_gr.add(dir_gr_fermi)
	dir_gr.add(dir_gr_kepler)
	dir_gr.add(dir_gr_lw)
	dir_gr.add(dir_gr_lw10)
	dir_gr.add(dir_gr_lw20)
	dir_gr.add(dir_gr_lw4)
	dir_gr.add(dir_gr_lw40)
	dir_gr.add(dir_gr_lw50)
	dir_gr.add(dir_gr_sims)



$kernel.add(dir_gsync)

	dir_gsync.add(dir_gsync_lw)



$kernel.add(dir_gvi)

	dir_gvi.add(dir_gvi_lw)



$kernel.add(dir_hal)

	dir_hal.add(dir_hal_fermi)
	dir_hal.add(dir_hal_kepler)
	dir_hal.add(dir_hal_lw)
	dir_hal.add(dir_hal_lw40)
	dir_hal.add(dir_hal_lw50)
	dir_hal.add(dir_hal_sims)



$kernel.add(dir_hbloat)

	dir_hbloat.add(dir_hbloat_lw)
	dir_hbloat.add(dir_hbloat_lw17)
	dir_hbloat.add(dir_hbloat_lw44)



$kernel.add(dir_heap)

	dir_heap.add(dir_heap_lw)



$kernel.add(dir_inforom)

	dir_inforom.add(dir_inforom_fermi)
	dir_inforom.add(dir_inforom_lw)



$kernel.add(dir_instmem)

	dir_instmem.add(dir_instmem_lw)
	dir_instmem.add(dir_instmem_lw4)
	dir_instmem.add(dir_instmem_lw40)
	dir_instmem.add(dir_instmem_lw50)
	dir_instmem.add(dir_instmem_sims)



$kernel.add(dir_intr)

	dir_intr.add(dir_intr_fermi)
	dir_intr.add(dir_intr_lw)



$kernel.add(dir_mc)

	dir_mc.add(dir_mc_fermi)
	dir_mc.add(dir_mc_lw)
	dir_mc.add(dir_mc_lw10)
	dir_mc.add(dir_mc_lw20)
	dir_mc.add(dir_mc_lw4)
	dir_mc.add(dir_mc_lw40)
	dir_mc.add(dir_mc_lw50)
	dir_mc.add(dir_mc_sims)



$kernel.add(dir_me)

	dir_me.add(dir_me_lw)
	dir_me.add(dir_me_lw40)
	dir_me.add(dir_me_lw50)



$kernel.add(dir_modular)

	dir_modular.add(dir_modular_lw)



$kernel.add(dir_mp)

	dir_mp.add(dir_mp_g84)
	dir_mp.add(dir_mp_lw)
	dir_mp.add(dir_mp_lw10)
	dir_mp.add(dir_mp_lw4)
	dir_mp.add(dir_mp_lw40)



$kernel.add(dir_mpeg)

	dir_mpeg.add(dir_mpeg_lw)
	dir_mpeg.add(dir_mpeg_lw10)
	dir_mpeg.add(dir_mpeg_lw30)
	dir_mpeg.add(dir_mpeg_lw40)
	dir_mpeg.add(dir_mpeg_lw50)



$kernel.add(dir_msdec)

	dir_msdec.add(dir_msdec_fermi)
	dir_msdec.add(dir_msdec_g98)
	dir_msdec.add(dir_msdec_gt212)
	dir_msdec.add(dir_msdec_lw)



$kernel.add(dir_msenc)

	dir_msenc.add(dir_msenc_lw)
	dir_msenc.add(dir_msenc_v01)



$kernel.add(dir_lwd)

	dir_lwd.add(dir_lwd_lw)



$kernel.add(dir_perfctl)

	dir_perfctl.add(dir_perfctl_kepler)
	dir_perfctl.add(dir_perfctl_fermi)
	dir_perfctl.add(dir_perfctl_lw)
	dir_perfctl.add(dir_perfctl_lw30)
	dir_perfctl.add(dir_perfctl_lw40)
	dir_perfctl.add(dir_perfctl_lw50)



$kernel.add(dir_pg)

	dir_pg.add(dir_pg_fermi)
	dir_pg.add(dir_pg_lw)
	dir_pg.add(dir_pg_lw50)



$kernel.add(dir_pmgr)

	dir_pmgr.add(dir_pmgr_fermi)
	dir_pmgr.add(dir_pmgr_lw)
	dir_pmgr.add(dir_pmgr_lw10)
	dir_pmgr.add(dir_pmgr_lw4)
	dir_pmgr.add(dir_pmgr_lw40)
	dir_pmgr.add(dir_pmgr_lw50)
	dir_pmgr.add(dir_pmgr_sims)
	dir_pmgr.add(dir_pmgr_v02)



$kernel.add(dir_pmu)

	dir_pmu.add(dir_pmu_fermi)
	dir_pmu.add(dir_pmu_lw)
	dir_pmu.add(dir_pmu_lw50)



$kernel.add(dir_rc)

	dir_rc.add(dir_rc_lw)
	dir_rc.add(dir_rc_lw40)
	dir_rc.add(dir_rc_lw50)



$kernel.add(dir_res)

	dir_res.add(dir_res_lw)



$kernel.add(dir_seq)

	dir_seq.add(dir_seq_fermi)
	dir_seq.add(dir_seq_lw)
	dir_seq.add(dir_seq_lw10)
	dir_seq.add(dir_seq_lw40)
	dir_seq.add(dir_seq_lw50)



$kernel.add(dir_smu)

	dir_smu.add(dir_smu_lw)
	dir_smu.add(dir_smu_lw50)



$kernel.add(dir_spb)

	dir_spb.add(dir_spb_lw)



$kernel.add(dir_ss)

	dir_ss.add(dir_ss_lw)
	dir_ss.add(dir_ss_lw10)
	dir_ss.add(dir_ss_lw30)
	dir_ss.add(dir_ss_lw40)



$kernel.add(dir_state)

	dir_state.add(dir_state_lw)



$kernel.add(dir_stereo)

	dir_stereo.add(dir_stereo_fermi)
	dir_stereo.add(dir_stereo_lw)
	dir_stereo.add(dir_stereo_lw10)
	dir_stereo.add(dir_stereo_lw30)
	dir_stereo.add(dir_stereo_lw40)
	dir_stereo.add(dir_stereo_lw50)



$kernel.add(dir_sw)

	dir_sw.add(dir_sw_lw)



$kernel.add(dir_syscon)

	dir_syscon.add(dir_syscon_lw)



$kernel.add(dir_thermctl)

	dir_thermctl.add(dir_thermctl_fermi)
	dir_thermctl.add(dir_thermctl_lw)
	dir_thermctl.add(dir_thermctl_lw30)
	dir_thermctl.add(dir_thermctl_lw40)
	dir_thermctl.add(dir_thermctl_lw50)



$kernel.add(dir_tmr)

	dir_tmr.add(dir_tmr_fermi)
	dir_tmr.add(dir_tmr_lw)
	dir_tmr.add(dir_tmr_lw4)
	dir_tmr.add(dir_tmr_lw40)
	dir_tmr.add(dir_tmr_lw50)
	dir_tmr.add(dir_tmr_sims)



$kernel.add(dir_tvo)

	dir_tvo.add(dir_tvo_lw)



$kernel.add(dir_vblank)

	dir_vblank.add(dir_vblank_lw)



$kernel.add(dir_vic)

	dir_vic.add(dir_vic_lw)
	dir_vic.add(dir_vic_v01)



$kernel.add(dir_video)

	dir_video.add(dir_video_lw)
	dir_video.add(dir_video_lw10)
	dir_video.add(dir_video_lw30)
	dir_video.add(dir_video_lw40)



$kernel.add(dir_vp)

	dir_vp.add(dir_vp_fermi)
	dir_vp.add(dir_vp_g82)
	dir_vp.add(dir_vp_g98)
	dir_vp.add(dir_vp_gt212)
	dir_vp.add(dir_vp_lw)
	dir_vp.add(dir_vp_lw41)
	dir_vp.add(dir_vp_lw43)
	dir_vp.add(dir_vp_lw44)
	dir_vp.add(dir_vp_lw50)
	dir_vp.add(dir_vp_vgpu)
