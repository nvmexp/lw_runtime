#OLD=11_30_20_16_30_36_gemvBenchmark_hhhh_anon-desktop_0_0_anon-desktop_a9965396_0.dat
#NEW=11_30_20_14_38_33_gemvBenchmark_hhhh_anon-desktop_0_0_anon-desktop_08095dd9_0.dat

NEW=11_30_20_14_38_33_gemvBenchmark_ssss_anon-desktop_0_0_anon-desktop_08095dd9_0.dat
OLD=11_30_20_16_30_36_gemvBenchmark_ssss_anon-desktop_0_0_anon-desktop_a9965396_0.dat

#NEW=11_30_20_14_38_33_reduction_largeK_hhhh_anon-desktop_0_0_anon-desktop_08095dd9_0.dat
#OLD=11_30_20_16_30_36_reduction_largeK_hhhh_anon-desktop_0_0_anon-desktop_a9965396_0.dat
#
#NEW=11_30_20_14_38_33_reduction_largeK_ssss_anon-desktop_0_0_anon-desktop_08095dd9_0.dat
#OLD=11_30_20_16_30_36_reduction_largeK_ssss_anon-desktop_0_0_anon-desktop_a9965396_0.dat

python ../../../misc/selectData.py --logfile data/${OLD} --label old --output old.csv
python ../../../misc/selectData.py --logfile data/${NEW} --label new --output new.csv
python ../../../misc/analyzeBenchmarks.py old.csv new.csv gemv_fp32.png
