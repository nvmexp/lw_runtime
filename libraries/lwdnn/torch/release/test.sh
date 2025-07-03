set -e

LUA=$(which luajit lua | head -n 1)

if [ ! -x "$LUA" ]
then
    echo "Neither luajit nor lua found in path"
    exit 1
fi

echo "Using Lua at:"
echo "$LUA"

#smoke tests
$LUA -lpaths     -e "print('paths loaded succesfully')"
$LUA -ltorch     -e "print('torch loaded succesfully')"
$LUA -lelw       -e "print('elw loaded succesfully')"
$LUA -ltrepl     -e "print('trepl loaded succesfully')"
$LUA -ldok       -e "print('dok loaded succesfully')"
$LUA -limage     -e "print('image loaded succesfully')"
$LUA -lcwrap     -e "print('cwrap loaded succesfully')"
$LUA -lgnuplot   -e "print('gnuplot loaded succesfully')"
$LUA -loptim     -e "print('optim loaded succesfully')"
$LUA -lsys       -e "print('sys loaded succesfully')"
$LUA -lxlua      -e "print('x$(basename $LUA) loaded succesfully')"
$LUA -largcheck  -e "print('argcheck loaded succesfully')"
$LUA -lgraph     -e "print('graph loaded succesfully')"
$LUA -lnn        -e "print('nn loaded succesfully')"
$LUA -lnngraph   -e "print('nngraph loaded succesfully')"
$LUA -lnnx       -e "print('nnx loaded succesfully')"
$LUA -lthreads   -e "print('threads loaded succesfully')"

th -ltorch -e "torch.test()"
th -lnn    -e "nn.test()"

if [ $(basename $LUA) = "luajit" ]
then
    $LUA -lsundown         -e "print('sundown loaded succesfully')"
    $LUA -lsignal          -e "print('signal loaded succesfully')"
    $LUA -lgraphicsmagick  -e "print('graphicsmagick loaded succesfully')"
    $LUA -lfftw3           -e "print('fftw3 loaded succesfully')"
    $LUA -laudio           -e "print('audio loaded succesfully')"
fi

# LWCA tests
set +e
path_to_lwcc=$(which lwcc)
path_to_lwidiasmi=$(which lwpu-smi)
set -e

if [ -x "$path_to_lwcc" ] || [ -x "$path_to_lwidiasmi" ]
then
    th -llwtorch -e "print('lwtorch loaded succesfully')"

    if [ $(basename $LUA) = "luajit" ];
    then
        th -llwdnn -e "print('lwdnn loaded succesfully')"
    fi
    th -llwtorch -e "lwtorch.test()"
    th -llwnn  -e "nn.testlwda()"
    th extra/lwdnn/test/test.lua
else
    echo "LWCA not found"
fi
