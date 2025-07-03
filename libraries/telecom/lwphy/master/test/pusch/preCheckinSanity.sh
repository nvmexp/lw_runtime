#!/bin/bash

# Script to batch exelwtion of the PUSCH-multipipe example (lwphy_ex_pudsch_rx_multi_pipe) with test vectors of of the format: TV_lwphy_*.h5 
# See Usage for example run commands 

#########################
# Usage #
Usage() {
echo "Usage: $0 [-h] [-r] {<s|p> <build_dir> <TV_dir> <out_dir>}" >&2
echo
echo "   -h, --help           show this help text"
echo "   -r, --run            run exelwtable from <build_dir> TVs from <TV_dir>, s|p selects serial|paralle TV exelwtion, store results and logs in <out_dir>"
echo
echo "Suppose lwPHY dir contains the TVs then:"
echo "Example usage from build dir: ../test/pusch/preCheckinSanity.sh -r p \$PWD ../. results"
echo "Example usage from lwPHY dir: test/pusch/preCheckinSanity.sh -r p \$PWD/build \$PWD results"
exit 1
}

#########################
# Tests with failures #
DisplayResults() {
#export RESDIR="resultsSdkTvs201912210738"

# array to collect TC names of all failing TCs 
declare -a FAIL_TCNAME_ARR

# Report all tests with failures (note that some of the failures may be intentional)
echo "---------------------------------------------------------------"
echo "Build directory : $BUILDDIR"
echo "TV directory    : $TVDIR"
echo "Result directory: $RESDIR"

echo "Test summary    :" 
for TVFILE in "$RESDIR"/logs/*.log; do 
	if grep -q 'ERROR\|error\|EXCEPTION\|failed' $TVFILE; then
	    # Get file name without the path
	    TVFILENAME=$(basename -- "$TVFILE")
	    # Remove extension from file name
	    TCNAME="${TVFILENAME%.*}"
	    FAIL_TCNAME_ARR+=("$TCNAME")
	    #echo "$TCNAME"
	fi
done

if [ ${#FAIL_TCNAME_ARR[@]} -eq 0 ]; then
        echo "No failures detected" 
else
        echo "Tests with failures" 
        printf '%s\n' "${FAIL_TCNAME_ARR[@]}"
        #for each in "${FAIL_TCNAME_ARR[@]}"
        #do
	#    echo "$each"
        #done
fi
}

#########################
# Serial exelwtion #
Execute() {
export RESDIR=$(date +${RESDIRPREFIX}'%G%m%d%H%M')
#export RESDIR="results201912090933"
 
echo "Build directory : $BUILDDIR"
echo "TV directory    : $TVDIR"
echo "Result directory: $RESDIR"

rm -fr $RESDIR
mkdir $RESDIR
mkdir $RESDIR/logs

#if [ 1 -eq 0 ]; then
# Find all files with specified TV prefix and extension and run them
find $TVDIR -maxdepth 1 -type f -name 'TV_lwphy_*.h5' -printf "%P\n" |
     while IFS= read -r -d $'\n' TVFILE; do 
	TCNAME="${TVFILE%.*}"
        echo "--------------------------------Running Testcase: $TCNAME-------------------------------"

	$BUILDDIR/examples/pusch_rx_multi_pipe/lwphy_ex_pusch_rx_multi_pipe -i $TVDIR/$TVFILE -o $RESDIR/gpu_out_$TCNAME.h5 > >(tee -a $RESDIR/logs/$TCNAME.log) 2> >(tee -a $RESDIR/logs/$TCNAME.log >&2)
     done
#fi

DisplayResults
}

#########################
# Parallel exelwtion #
ExelwteParallel() {
export RESDIR=$(date +${RESDIRPREFIX}'%G%m%d%H%M')
#export RESDIR="resultsSdkTvs201912210738"
 
echo "Build directory : $BUILDDIR"
echo "TV directory    : $TVDIR"
echo "Result directory: $RESDIR"

rm -fr $RESDIR
mkdir $RESDIR
mkdir $RESDIR/logs

# array to collect pids of all background processes running test cases so that they may be waited upon
PID_ARR=""
# background processes run asynchronously, use a named pipe to collect outputs and display one they are all done
mkfifo OUTPIPE
RESULT=0

#if [ 1 -eq 0 ]; then
# Find all files with specified TV prefix and extension and run them
find $TVDIR -maxdepth 1 -type f -name 'TV_lwphy_*.h5' -printf "%P\n" |
     while IFS= read -r -d $'\n' TVFILE; do 
	(
	# get TV name only
	TCNAME="${TVFILE%.*}"
        echo "Running Testcase: $TCNAME"

	# Run test

	$BUILDDIR/examples/pusch_rx_multi_pipe/lwphy_ex_pusch_rx_multi_pipe -i $TVDIR/$TVFILE -o $RESDIR/gpu_out_$TCNAME.h5 > >(tee -a $RESDIR/logs/$TCNAME.log) 2> >(tee -a $RESDIR/logs/$TCNAME.log >&2) > OUTPIPE
        # $BUILDDIR/examples/pusch_rx_multi_pipe/lwphy_ex_pusch_rx_multi_pipe -i $TVDIR/$TVFILE -o $RESDIR/gpu_out_$TCNAME.h5 | tee $RESDIR/logs/$TCNAME.log > OUTPIPE
	) &
	PID_ARR="$PID_ARR $!"
     done

# wait for all background processes to finish/exit before moving forward 
for PID in $PID_ARR; do
    wait $PID || let "RESULT=1"
done

# display output from all background processes
cat OUTPIPE
# remove the named pipe
rm OUTPIPE

# check for error during test exelwtion
if [ "$RESULT" == "1" ]; then
    echo "Failure during exelwtion of one or more tests"
    exit 1
fi
#fi

DisplayResults
}


################################
# Check Options #
#echo "$0"
#echo "$1"
#echo "$2" 
#echo "$3"
#echo "$4"

while :
do
    case "$1" in
      -r | --run)
        shift 1
	if [ $# -eq 4 ]; then
           export BUILDDIR=$2 TVDIR=$3 RESDIRPREFIX=$4
           if [ "$1" == "s" ]; then
	      Execute 
           elif [ "$1" == "p" ]; then 
              ExelwteParallel 
	      #DisplayResults
           else
              echo "Error: Unknown option: $1" >&2
           fi
        else
           echo "Not enough input arguments" >&2
           Usage
        fi
        ;;
      -h | --help)
        Usage
        exit 0
	;;
      --) # End of all options
        shift
        break
        ;;
      -*)
        echo "Error: Unknown option: $1" >&2
        Usage
        exit 1 
        ;;
      *)  # No more options
        break
        ;;
    esac
done

