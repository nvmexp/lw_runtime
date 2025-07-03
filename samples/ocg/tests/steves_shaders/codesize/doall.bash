#!/bin/bash

ROOT=$(cygpath -u $(p4 client -o | sed -n '/^Root:/s/.*	//p'))

export BUILDDIR=$(cygpath -w $ROOT/sw)

CGCDIR=$ROOT/sw/cg/main/src/cglang/cgc

while [ $# -gt 0 ]; do
    case $1 in
    -build)
        make -C $CGCDIR -f Makefile
        (cd $ROOT/sw/main/apps/ocg/cop_parseasm; lwmake usecg lddm)
        ;;
    *)
        echo "Unknown arg $1" >&2
        ;;
    esac
    shift
done

if [ "$CGCDIR/bin/win32_debug/cgc.exe" -nt "$CGCDIR/Debug/cgc.exe" ]; then
    export CGC=$(cygpath -w $CGCDIR/bin/win32_debug/cgc.exe)
else
    export CGC=$(cygpath -w $CGCDIR/Debug/cgc.exe)
fi
if [ "$CGCDIR/bin/win32_lwonly_debug/cgc.exe" -nt "$CGC" ]; then
    export CGC=$(cygpath -w $CGCDIR/bin/win32_lwonly_debug/cgc.exe)
fi

for f in $(find . -iname doall.bat); do
    echo $(dirname $f):
    if [ -e $(dirname $f)/doall0.bat ]; then
        (cd $(dirname $f); ./doall0.bat)
    else
        (cd $(dirname $f); ./doall.bat)
    fi
done | tee results.txt
