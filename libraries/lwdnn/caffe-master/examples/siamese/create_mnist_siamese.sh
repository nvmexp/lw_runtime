#!/usr/bin/elw sh
# This script colwerts the mnist data into leveldb format.

EXAMPLES=./build/examples/siamese
DATA=./data/mnist

echo "Creating leveldb..."

rm -rf ./examples/siamese/mnist_siamese_train_leveldb
rm -rf ./examples/siamese/mnist_siamese_test_leveldb

$EXAMPLES/colwert_mnist_siamese_data.bin \
    $DATA/train-images-idx3-ubyte \
    $DATA/train-labels-idx1-ubyte \
    ./examples/siamese/mnist_siamese_train_leveldb
$EXAMPLES/colwert_mnist_siamese_data.bin \
    $DATA/t10k-images-idx3-ubyte \
    $DATA/t10k-labels-idx1-ubyte \
    ./examples/siamese/mnist_siamese_test_leveldb

echo "Done."
