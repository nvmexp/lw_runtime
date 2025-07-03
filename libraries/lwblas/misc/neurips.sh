#
#This file was used to create all plots for the neurips presentation
#
#

# DATA_DIR should be a copy of data@10.110.40.147:~/lwtensor/neurips
DATA_DIR=./data/contraction

function gv100_rand1000_dataTypes {
    # data
    GV100_rand1000_fp64=${DATA_DIR}/11_13_20_14_49_50_rand1000_dddd_prm-dgx-04_0_0_prm-dgx-04_0b863789_0.dat
    GV100_rand1000_fp64_fp32=${DATA_DIR}/11_13_20_14_49_50_rand1000_ddds_prm-dgx-04_0_0_prm-dgx-04_0b863789_1.dat
    GV100_rand1000_fp16=${DATA_DIR}/11_13_20_14_49_50_rand1000_hhhs_prm-dgx-04_0_0_prm-dgx-04_0b863789_3.dat
    GV100_rand1000_fp32_fp16=${DATA_DIR}/11_13_20_14_49_50_rand1000_sssh_prm-dgx-04_0_0_prm-dgx-04_0b863789_4.dat
    GV100_rand1000_fp32=${DATA_DIR}/11_13_20_14_49_50_rand1000_ssss_prm-dgx-04_0_0_prm-dgx-04_0b863789_2.dat

    # create csv
    python selectData.py --logfile=${GV100_rand1000_fp16}      --label=fp16      --output=rand1000_hhhs.csv
    python selectData.py --logfile=${GV100_rand1000_fp32}      --label=fp32      --output=rand1000_ssss.csv
    python selectData.py --logfile=${GV100_rand1000_fp64}      --label=fp64      --output=rand1000_dddd.csv
    python selectData.py --logfile=${GV100_rand1000_fp32_fp16} --label=fp32+fp16 --output=rand1000_sssh.csv
    python selectData.py --logfile=${GV100_rand1000_fp64_fp32} --label=fp64+fp32 --output=rand1000_ddds.csv

    #create plot (combining all columns)
    python plot.py --output=rand1000_gv100.png --hideTicksX --xLimitUpper ${DATA_DIR}/tblis_8168_plat.csv rand1000_dddd.csv rand1000_ddds.csv rand1000_ssss.csv rand1000_sssh.csv rand1000_hhhs.csv --useGrid

    # cleanup temp files
    rm -f rand1000_dddd.csv rand1000_ddds.csv rand1000_ssss.csv rand1000_sssh.csv rand1000_hhhs.csv
}

function ga100_gv100_rand1000 {
    GA100_rand1000_fp64=${DATA_DIR}/11_13_20_23_24_52_rand1000_dddd_dt03.eth.cluster_0_0_dt03.eth.cluster_c_.dat
    GV100_rand1000_fp64=${DATA_DIR}/11_13_20_14_49_50_rand1000_dddd_prm-dgx-04_0_0_prm-dgx-04_0b863789_0.dat

    python selectData.py --logfile=${GV100_rand1000_fp64}  --label="GV100 (fp64)" --output=gv100_dddd.csv
    python selectData.py --logfile=${GA100_rand1000_fp64}  --label="GA100 (fp64)" --output=ga100_dddd.csv

    python plot.py --output=rand1000_ga100_gv100.png ${DATA_DIR}/tblis_8168_plat.csv gv100_dddd.csv ga100_dddd.csv --xLimitUpper --hideTicksX --useGrid

    rm -f ga100_dddd.csv gv100_dddd.csv
}

function ga100_complex_rand1000 {
    GA100_rand1000_cfp32_fp16=${DATA_DIR}/11_14_20_06_06_29_rand1000_ccct_dt03.eth.cluster_0_0_dt03.eth.cluster_c_.dat
    GA100_rand1000_cfp64=${DATA_DIR}/11_14_20_06_06_29_rand1000_zzzd_dt03.eth.cluster_0_0_dt03.eth.cluster_c_.dat
    python selectData.py --logfile=${GA100_rand1000_cfp32_fp16}  --label="GA100 (complex fp32+tf32)" --output=ga100_ccct.csv
    python selectData.py --logfile=${GA100_rand1000_cfp64}  --label="GA100 (complex fp64)" --output=ga100_zzzz.csv
    python plot.py --output=rand1000_ga100_complex.png ga100_zzzz.csv ga100_ccct.csv --xLimitUpper --hideTicksX --useGrid
}

function ga100_rand1000_dataTypes {
    # data
    GA100_rand1000_fp64=${DATA_DIR}/11_13_20_23_24_52_rand1000_dddd_dt03.eth.cluster_0_0_dt03.eth.cluster_c_.dat
    #GA100_rand1000_fp64=${DATA_DIR}/11_14_20_08_32_59_rand1000_dddd_pmajcher-dt2_0_0_pmajcher-dt2_668337e9_0.dat # <<< uses limited kernels (noticably slower)
    GA100_rand1000_bf16=${DATA_DIR}/11_13_20_16_27_35_rand1000_bbbs_dt03.eth.cluster_0_0_dt03.eth.cluster_c_.dat
    GA100_rand1000_fp32_tf32=${DATA_DIR}/11_13_20_16_27_35_rand1000_ssst_dt03.eth.cluster_0_0_dt03.eth.cluster_c_.dat
    #GA100_rand1000_fp32=${DATA_DIR}/

    # create csv
    python selectData.py --logfile=${GA100_rand1000_bf16}      --label=bf16 --output=bbbs.csv
    #python selectData.py --logfile=${GA100_rand1000_fp32}      --label=fp32      --output=rand1000_ssss.csv
    python selectData.py --logfile=${GA100_rand1000_fp32_tf32} --label=fp32+tf32 --output=ssst.csv
    python selectData.py --logfile=${GA100_rand1000_fp64}      --label=fp64      --output=dddd.csv

    #create plot (combining all columns)
    python plot.py --output=rand1000_ga100.png ${DATA_DIR}/tblis_8168_plat.csv dddd.csv ssst.csv bbbs.csv --xLimitUpper --hideTicksX --useGrid

    # cleanup temp files
    rm -f dddd.csv ssst.csv bbbs.csv
}

function gv100_rand1000_dl {
    # data
    MX_TF_PYT=${DATA_DIR}/rand1000_fp32_pyt_gv100_aad5603c.csv
    MX_TF_PYT_cache=${DATA_DIR}/rand1000_fp32_pyt_gv100_cache_aad5603c.csv
    MX_PYT_mxnet_cache=${DATA_DIR}/mxnet_fp32_pyt_gv100_sssh_cache.csv
    MX_PYT_mxnet=${DATA_DIR}/mxnet_fp32_pyt_gv100_sssh.csv

    #create plot
    python plot.py --yMax=14000 --output=rand1000_gv100_dl.png --hideTicksX ${MX_TF_PYT} --sortBy="MXNet (+lwTENSOR)"
    python plot.py --yMax=14000 --output=rand1000_gv100_cache_dl.png --hideTicksX ${MX_TF_PYT_cache} --sortBy="MXNet (+lwTENSOR + plancache)"

    python plot.py --yLabel="Speedup" --output=bert_transformer_gv100_dl.png --hideTicksX ${MX_PYT_mxnet} --sortBy="MXNet (+lwTENSOR)" --speedup --useGrid --yMax=9
    python plot.py --yLabel="Speedup" --output=bert_transformer_gv100_cache_dl.png --hideTicksX ${MX_PYT_mxnet_cache} --sortBy="MXNet (+lwTENSOR + plancache)" --speedup --useGrid --yMax=9
}

function tu_rand300_ew {
    # data
    OLD=${DATA_DIR}/11_13_20_radn300_ew_ssss_mhohnerbach-ws_121.dat
    NEW=${DATA_DIR}/11_13_20_radn300_ew_ssss_mhohnerbach-ws_develop.dat

    # create csv
    python selectData.py --logfile=${OLD}      --label="lwTENSOR 1.2.1" --output=old.csv
    python selectData.py --logfile=${NEW}      --label="lwTENSOR 1.2.2" --output=new.csv

    python plot.py --output=rand300_ew_tu.png --yLabel="GB/s" --hideTicksX --useGrid old.csv new.csv
}

ga100_gv100_rand1000
gv100_rand1000_dataTypes
ga100_rand1000_dataTypes
gv100_rand1000_dl
ga100_complex_rand1000 
tu_rand300_ew
