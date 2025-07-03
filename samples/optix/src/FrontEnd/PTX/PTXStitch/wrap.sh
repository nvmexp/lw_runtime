#!/bin/sh

echo "Probing edges in CG"

lwcc -m32 -Xopencc -INLINE:list=on --ptx $1 > tmp.stdout 2> tmp.stderr
if [ "$?" -ne 0 ]
then
	echo 'Error in first compilation pass:'
	cat tmp.stdout tmp.stderr
	exit 1
fi

EDGES=`grep "^Inlining _Z..\?GROUP_" tmp.stderr | cut -d ' ' -f 6 | sed 's:)::g'`
EDGES=`echo $EDGES | sed 's: :,:g'`

echo "Re-compiling and skipping edges" $EDGES

OPENCC_FLAGS="-INLINE:skip=$EDGES" lwcc -m32 --ptx -o tmp.ptx $1
