VERSION="0_2_2_x86_64_lwda_10_0"
rm -rf lwtensor
mkdir lwtensor
mkdir lwtensor/lib
mkdir lwtensor/include
mkdir lwtensor/include/lwtensor
mkdir lwtensor/samples

cp Doxyfile ./lwtensor/
cp ../include/lwtensor.h ./lwtensor/include/
#cp ../include/lwtensor/types.h ./lwtensor/include/lwtensor/
cp ./types.h ./lwtensor/include/lwtensor/
cp ../examples/contraction.lw ./lwtensor/samples/
cp ../examples/contraction_autotuning.lw ./lwtensor/samples/
cp ../examples/elementwise_binary.lw ./lwtensor/samples/
cp ../examples/elementwise_permute.lw ./lwtensor/samples/
cp ../examples/elementwise_trinary.lw ./lwtensor/samples/
cp ../examples/vectorization.lw ./lwtensor/samples/
cp ../examples/reduction.lw ./lwtensor/samples/
cp ../README_public.md ./lwtensor/README.md
cp license.pdf ./lwtensor/
cp Makefile ./lwtensor/samples
 
cp ../lib/liblwtensor.so ./lwtensor/lib/liblwtensor.so.10
cd lwtensor/lib
ln -s liblwtensor.so.10 liblwtensor.so
cd ../../
tar -cjf lwtensor_${VERSION}.tar.gz lwtensor
