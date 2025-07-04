#!/usr/bin/elw bash

SKIP_RC=0
BATCH_INSTALL=0

THIS_DIR=$(cd $(dirname $0); pwd)
PREFIX=${PREFIX:-"${THIS_DIR}/install"}
TORCH_LUA_VERSION=${TORCH_LUA_VERSION:-"LUAJIT21"} # by default install LUAJIT21
# default is empty arch list
TORCH_LWDA_ARCH_LIST=

while getopts 'bsh:' x; do
    case "$x" in
        h)
            echo "usage: $0
This script will install Torch and related, useful packages into $PREFIX.

    -b      Run without requesting any user input (will automatically add PATH to shell profile)
    -s      Skip adding the PATH to shell profile
"
            exit 2
            ;;
        b)
            BATCH_INSTALL=1
            ;;
        s)
            SKIP_RC=1
            ;;
    esac
done


# Scrub an anaconda install, if exists, from the PATH.
# It has a malformed MKL library (as of 1/17/2015)
OLDPATH=$PATH
if [[ $(echo $PATH | grep anaconda) ]]; then
    export PATH=$(echo $PATH | tr ':' '\n' | grep -v "anaconda/bin" | grep -v "anaconda/lib" | grep -v "anaconda/include" | uniq | tr '\n' ':')
fi

echo "Prefix set to $PREFIX"

if [[ `uname` == 'Linux' ]]; then
    export CMAKE_LIBRARY_PATH=/opt/OpenBLAS/include:/opt/OpenBLAS/lib:$CMAKE_LIBRARY_PATH
fi
export CMAKE_PREFIX_PATH=$PREFIX

git submodule update --init --relwrsive

# If we're on OS X, use clang
if [[ `uname` == "Darwin" ]]; then
    # make sure that we build with Clang. LWCA's compiler lwcc
    # does not play nice with any recent GCC version.
    export CC=clang
    export CXX=clang++
fi

echo "Installing Lua version: ${TORCH_LUA_VERSION}"
mkdir -p install
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX="${PREFIX}" -DCMAKE_BUILD_TYPE=Release -DWITH_${TORCH_LUA_VERSION}=ON 2>&1 >>$PREFIX/install.log || exit 1
(make 2>&1 >>$PREFIX/install.log  || exit 1) && (make install 2>&1 >>$PREFIX/install.log || exit 1)
cd ..

# Check for a LWCA install (using lwcc instead of lwpu-smi for cross-platform compatibility)
path_to_lwcc=$(which lwcc)

# check if we are on mac and fix RPATH for local install
path_to_install_name_tool=$(which install_name_tool 2>/dev/null)
if [ -x "$path_to_install_name_tool" ]
then
   if [ ${TORCH_LUA_VERSION} == "LUAJIT21" ] || [ ${TORCH_LUA_VERSION} == "LUAJIT20" ] ; then
       install_name_tool -id ${PREFIX}/lib/libluajit.dylib ${PREFIX}/lib/libluajit.dylib
   else
       install_name_tool -id ${PREFIX}/lib/liblua.dylib ${PREFIX}/lib/liblua.dylib
   fi
fi

if [ -x "$path_to_lwcc" ] || [ -x "$path_to_lwidiasmi" ]
then
    echo "Found LWCA on your machine. Installing CMake 3.6 modules to get up-to-date FindLWDA"
    cd ${THIS_DIR}/cmake/3.6 && \
(cmake -E make_directory build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        && make install) && echo "FindLwda bits of CMake 3.6 installed" || exit 1
fi

setup_lua_elw_cmd=$($PREFIX/bin/luarocks path)
eval "$setup_lua_elw_cmd"

echo "Installing common Lua packages"
cd ${THIS_DIR}/extra/luafilesystem && $PREFIX/bin/luarocks make rockspecs/luafilesystem-1.6.3-1.rockspec || exit 1
cd ${THIS_DIR}/extra/penlight && $PREFIX/bin/luarocks make || exit 1
cd ${THIS_DIR}/extra/lua-cjson && $PREFIX/bin/luarocks make || exit 1

echo "Installing core Torch packages"
cd ${THIS_DIR}/extra/luaffifb && $PREFIX/bin/luarocks make                             || exit 1
cd ${THIS_DIR}/pkg/sundown   && $PREFIX/bin/luarocks make rocks/sundown-scm-1.rockspec || exit 1
cd ${THIS_DIR}/pkg/cwrap     && $PREFIX/bin/luarocks make rocks/cwrap-scm-1.rockspec   || exit 1
cd ${THIS_DIR}/pkg/paths     && $PREFIX/bin/luarocks make rocks/paths-scm-1.rockspec   || exit 1
cd ${THIS_DIR}/pkg/torch     && $PREFIX/bin/luarocks make rocks/torch-scm-1.rockspec   || exit 1
cd ${THIS_DIR}/pkg/dok       && $PREFIX/bin/luarocks make rocks/dok-scm-1.rockspec     || exit 1
cd ${THIS_DIR}/exe/trepl     && $PREFIX/bin/luarocks make                              || exit 1
cd ${THIS_DIR}/pkg/sys       && $PREFIX/bin/luarocks make sys-1.1-0.rockspec           || exit 1
cd ${THIS_DIR}/pkg/xlua      && $PREFIX/bin/luarocks make xlua-1.0-0.rockspec          || exit 1
cd ${THIS_DIR}/extra/nn      && $PREFIX/bin/luarocks make rocks/nn-scm-1.rockspec      || exit 1
cd ${THIS_DIR}/extra/graph   && $PREFIX/bin/luarocks make rocks/graph-scm-1.rockspec   || exit 1
cd ${THIS_DIR}/extra/nngraph && $PREFIX/bin/luarocks make                              || exit 1
cd ${THIS_DIR}/pkg/image     && $PREFIX/bin/luarocks make image-1.1.alpha-0.rockspec   || exit 1
cd ${THIS_DIR}/pkg/optim     && $PREFIX/bin/luarocks make optim-1.0.5-0.rockspec       || exit 1

if [ -x "$path_to_lwcc" ]
then
    echo "Found LWCA on your machine. Installing LWCA packages"
    cd ${THIS_DIR}/extra/lwtorch && $PREFIX/bin/luarocks make rocks/lwtorch-scm-1.rockspec || exit 1
    cd ${THIS_DIR}/extra/lwnn    && $PREFIX/bin/luarocks make rocks/lwnn-scm-1.rockspec    || exit 1
fi

# Optional packages
echo "Installing optional Torch packages"
cd ${THIS_DIR}/pkg/gnuplot          && $PREFIX/bin/luarocks make rocks/gnuplot-scm-1.rockspec
cd ${THIS_DIR}/exe/elw              && $PREFIX/bin/luarocks make
cd ${THIS_DIR}/extra/nnx            && $PREFIX/bin/luarocks make nnx-0.1-1.rockspec
cd ${THIS_DIR}/exe/qtlua            && $PREFIX/bin/luarocks make rocks/qtlua-scm-1.rockspec
cd ${THIS_DIR}/pkg/qttorch          && $PREFIX/bin/luarocks make rocks/qttorch-scm-1.rockspec
cd ${THIS_DIR}/extra/threads        && $PREFIX/bin/luarocks make rocks/threads-scm-1.rockspec
cd ${THIS_DIR}/extra/graphicsmagick && $PREFIX/bin/luarocks make graphicsmagick-1.scm-0.rockspec
cd ${THIS_DIR}/extra/argcheck       && $PREFIX/bin/luarocks make rocks/argcheck-scm-1.rockspec
cd ${THIS_DIR}/extra/audio          && $PREFIX/bin/luarocks make audio-0.1-0.rockspec
cd ${THIS_DIR}/extra/fftw3          && $PREFIX/bin/luarocks make rocks/fftw3-scm-1.rockspec
cd ${THIS_DIR}/extra/signal         && $PREFIX/bin/luarocks make rocks/signal-scm-1.rockspec

# Optional LWCA packages
if [ -x "$path_to_lwcc" ]
then
    echo "Found LWCA on your machine. Installing optional LWCA packages"
    cd ${THIS_DIR}/extra/lwdnn   && $PREFIX/bin/luarocks make lwdnn-scm-1.rockspec
    cd ${THIS_DIR}/extra/lwnnx   && $PREFIX/bin/luarocks make rocks/lwnnx-scm-1.rockspec
fi

export PATH=$OLDPATH # Restore anaconda distribution if we took it out.

if [[ $SKIP_RC == 1 ]]; then
  exit 0
fi


# Add C libs to LUA_CPATH
if [[ `uname` == "Darwin" ]]; then
    CLIB_LUA_CPATH=$PREFIX/lib/?.dylib
else
    CLIB_LUA_CPATH=$PREFIX/lib/?.so
fi

cat <<EOF >$PREFIX/bin/torch-activate
$setup_lua_elw_cmd
export PATH=$PREFIX/bin:\$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:\$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$PREFIX/lib:\$DYLD_LIBRARY_PATH
export LUA_CPATH='$CLIB_LUA_CPATH;'\$LUA_CPATH
EOF
chmod +x $PREFIX/bin/torch-activate

RC_FILE=0
DEFAULT=yes
if [[ $(echo $SHELL | grep bash) ]]; then
    RC_FILE=$HOME/.bashrc
elif [[ $(echo $SHELL | grep zsh) ]]; then
    RC_FILE=$HOME/.zshrc
else
    echo "

Non-standard shell $SHELL detected. You might want to
add the following lines to your shell profile:

. $PREFIX/bin/torch-activate
"
fi

WRITE_PATH_TO_PROFILE=0
if [[ $BATCH_INSTALL == 0 ]]; then
    if [ -f $RC_FILE ]; then
        echo "

Do you want to automatically prepend the Torch install location
to PATH and LD_LIBRARY_PATH in your $RC_FILE? (yes/no)
[$DEFAULT] >>> "
        read input
        if [[ $input == "" ]]; then
            input=$DEFAULT
        fi

        is_yes() {
            yesses={y,Y,yes,Yes,YES}
            if [[ $yesses =~ $1 ]]; then
                echo 1
            fi
        }

        if [[ $(is_yes $input) ]]; then
            WRITE_PATH_TO_PROFILE=1
        fi
    fi
else
    if [[ $RC_FILE ]]; then
        WRITE_PATH_TO_PROFILE=1
    fi
fi

if [[ $WRITE_PATH_TO_PROFILE == 1 ]]; then
    echo "

. $PREFIX/bin/torch-activate" >> $RC_FILE
    echo "

. $PREFIX/bin/torch-activate" >> $HOME/.profile

else
    echo "

Not updating your shell profile.
You might want to
add the following lines to your shell profile:

. $PREFIX/bin/torch-activate
"
fi
