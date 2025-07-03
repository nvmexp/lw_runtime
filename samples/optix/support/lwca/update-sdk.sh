#!/bin/bash
#
# Copyright (c) 2020, LWPU CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of LWPU CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
base=//sw/wsapps/raytracing/rtsdk/LWCA

list()
{
    p4 files "$base/aarch64/.../version.txt" | sed 's%.*/\([^/]*\)/version.txt.*%\1%'
}

# lwoptix is linked against lwdart_static
integrateStaticDriverRuntime()
{
    local libdir
    local libprefix
    local libsuffix
    local libfile

    if [ "$src" = "win64" ]; then
        # We only need the static library in lib
        libdir=lib/x64
        libprefix=
        libsuffix=lib
    else
        # On Linux, the static library is in lib64
        libdir=lib64
        libprefix=lib
        libsuffix=a
    fi
    # Static LWCA driver runtime
    libfile=$libdir/${libprefix}lwdart_static.${libsuffix}
    p4 integrate -q $base/$src/$toolkit/$libfile $tgt/$libfile || exit 1
}

# test_sanity is a whitebox test that needs to link against the dynamic runtime
integrateDynamicRuntime()
{
    local libdir
    local libprefix
    local libsuffix
    local libfile

    if [ "$src" = "win64" ]; then
        libdir=lib/x64
        libprefix=
        libsuffix=lib
    else
        libdir=lib64/stubs
        libprefix=lib
        libsuffix=so
    fi
    libfile=$libdir/${libprefix}lwca.${libsuffix}
    p4 integrate -q $base/$src/$toolkit/$libfile $tgt/$libfile || exit 1
}

integrate()
{
    local src=$1
    local tgt=$2
    local dir

    echo Integrating $src into $tgt

    # Revert any previous attempt
    p4 revert $tgt/... > /dev/null || exit 1
    p4 delete $tgt/... > /dev/null || exit 1
    rm -rf $tgt || exit 1

    # Integrate only the portions of the toolkit needed to build lwoptix
    for dir in bin include lwvm; do
        p4 integrate -q $base/$src/$toolkit/$dir/... $tgt/$dir/... || exit 1
    done
    # We don't need bin/*.dll
    p4 revert $tgt/bin/*.dll > /dev/null || exit 1
    # We don't need liblwvm-samples
    p4 revert $tgt/lwvm/liblwvm-samples/... > /dev/null || exit 1

    integrateStaticDriverRuntime
    integrateDynamicRuntime
}

integratePackageManifest()
{
    p4 revert packages.txt || exit 1
    p4 delete packages.txt || exit 1
    rm -f packages.txt || exit 1
    p4 integrate -q $base/packages/${toolkit}.txt packages.txt || exit 1
}

lwdaSpecificsMessage()
{
    echo
    echo Be sure to update the LWDA_VERSION variable in apps/optix/make/lwca-specifics.lwmk
    echo
}

update()
{
    integrate aarch64 Linux_aarch64 || exit 1
    integrate linux Linux_amd64 || exit 1
    integrate ppc64le Linux_ppc64le || exit 1
    integrate win64 Windows_amd64 || exit 1
    integratePackageManifest
    lwdaSpecificsMessage
}

if [ "$1" = "list" ]; then
    list || exit 1
elif [ "$1" = "update" ] && [ "X$2" != "X" ]; then
    toolkit=$2
    update || exit 1
else
    echo "Usage:"
    echo "    $0 list -- list available toolkits from //sw/wsapps/raytracing/rtsdk/LWCA"
    echo "    $0 update <toolkit> -- update to specified toolkit"
fi
