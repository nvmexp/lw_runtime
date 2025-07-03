export TEGRA_TOP=$(pwd)
export OUTPUT=$TEGRA_TOP/complexity_lwsciipc.txt
bash cloc.sh complexity $TEGRA_TOP modules.txt $OUTPUT

echo "" >> $OUTPUT
echo "CHECK DATE:" >> $OUTPUT
date >> $OUTPUT
echo "" >> $OUTPUT

pushd $TEGRA_TOP/qnx/src
echo $(pwd) >> $OUTPUT
git log --oneline | head >> $OUTPUT
popd
echo "" >> $OUTPUT

pushd $TEGRA_TOP/gpu/drv
echo $(pwd) >> $OUTPUT
git log --oneline | head >> $OUTPUT
popd
echo "" >> $OUTPUT
