#! /bin/sh
# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2015 6WIND S.A.

# Load config options:
# - DPDK_CHECKPATCH_PATH
# - DPDK_CHECKPATCH_CODESPELL
# - DPDK_CHECKPATCH_LINE_LENGTH
# - DPDK_CHECKPATCH_OPTIONS
. $(dirname $(readlink -f $0))/load-devel-config

VALIDATE_NEW_API=$(dirname $(readlink -f $0))/check-symbol-change.sh

# Enable codespell by default. This can be overwritten from a config file.
# Codespell can also be enabled by setting DPDK_CHECKPATCH_CODESPELL to a valid path
# to a dictionary.txt file if dictionary.txt is not in the default location.
codespell=${DPDK_CHECKPATCH_CODESPELL:-enable}
length=${DPDK_CHECKPATCH_LINE_LENGTH:-80}

# override default Linux options
options="--no-tree"
if [ "$codespell" = "enable" ] ; then
    options="$options --codespell"
elif [ -f "$codespell" ] ; then
    options="$options --codespell"
    options="$options --codespellfile $codespell"
fi
options="$options --max-line-length=$length"
options="$options --show-types"
options="$options --ignore=LINUX_VERSION_CODE,ENOSYS,\
FILE_PATH_CHANGES,MAINTAINERS_STYLE,SPDX_LICENSE_TAG,\
VOLATILE,PREFER_PACKED,PREFER_ALIGNED,PREFER_PRINTF,\
PREFER_KERNEL_TYPES,PREFER_FALLTHROUGH,BIT_MACRO,CONST_STRUCT,\
SPLIT_STRING,LONG_LINE_STRING,C99_COMMENT_TOLERANCE,\
LINE_SPACING,PARENTHESIS_ALIGNMENT,NETWORKING_BLOCK_COMMENT_STYLE,\
NEW_TYPEDEFS,COMPARISON_TO_NULL"
options="$options $DPDK_CHECKPATCH_OPTIONS"

print_usage () {
	cat <<- END_OF_HELP
	usage: $(basename $0) [-h] [-q] [-v] [-nX|-r range|patch1 [patch2] ...]

	Run Linux kernel checkpatch.pl with DPDK options.
	The environment variable DPDK_CHECKPATCH_PATH must be set.

	The patches to check can be from stdin, files specified on the command line,
	latest git commits limited with -n option, or commits in the git range
	specified with -r option (default: "origin/main..").
	END_OF_HELP
}

check_forbidden_additions() { # <patch>
	res=0

	# refrain from new additions of rte_panic() and rte_exit()
	# multiple folders and expressions are separated by spaces
	awk -v FOLDERS="lib drivers" \
		-v EXPRESSIONS="rte_panic\\\( rte_exit\\\(" \
		-v RET_ON_FAIL=1 \
		-v MESSAGE='Using rte_panic/rte_exit' \
		-f $(dirname $(readlink -f $0))/check-forbidden-tokens.awk \
		"$1" || res=1

	# refrain from using compiler attribute without defining a common macro
	awk -v FOLDERS="lib drivers app examples" \
		-v EXPRESSIONS="__attribute__" \
		-v RET_ON_FAIL=1 \
		-v MESSAGE='Using compiler attribute directly' \
		-f $(dirname $(readlink -f $0))/check-forbidden-tokens.awk \
		"$1" || res=1

	# forbid variable declaration inside "for" loop
	awk -v FOLDERS='.' \
		-v EXPRESSIONS='for[[:space:]]*\\((char|u?int|unsigned|s?size_t)' \
		-v RET_ON_FAIL=1 \
		-v MESSAGE='Declaring a variable inside for()' \
		-f $(dirname $(readlink -f $0))/check-forbidden-tokens.awk \
		"$1" || res=1

	# refrain from new additions of 16/32/64 bits rte_atomicNN_xxx()
	awk -v FOLDERS="lib drivers app examples" \
		-v EXPRESSIONS="rte_atomic[0-9][0-9]_.*\\\(" \
		-v RET_ON_FAIL=1 \
		-v MESSAGE='Using rte_atomicNN_xxx' \
		-f $(dirname $(readlink -f $0))/check-forbidden-tokens.awk \
		"$1" || res=1

	# refrain from new additions of rte_smp_[r/w]mb()
	awk -v FOLDERS="lib drivers app examples" \
		-v EXPRESSIONS="rte_smp_(r|w)?mb\\\(" \
		-v RET_ON_FAIL=1 \
		-v MESSAGE='Using rte_smp_[r/w]mb' \
		-f $(dirname $(readlink -f $0))/check-forbidden-tokens.awk \
		"$1" || res=1

	# refrain from using compiler __sync_xxx builtins
	awk -v FOLDERS="lib drivers app examples" \
		-v EXPRESSIONS="__sync_.*\\\(" \
		-v RET_ON_FAIL=1 \
		-v MESSAGE='Using __sync_xxx builtins' \
		-f $(dirname $(readlink -f $0))/check-forbidden-tokens.awk \
		"$1" || res=1

	# refrain from using compiler __atomic_thread_fence()
	# It should be avoided on x86 for SMP case.
	awk -v FOLDERS="lib drivers app examples" \
		-v EXPRESSIONS="__atomic_thread_fence\\\(" \
		-v RET_ON_FAIL=1 \
		-v MESSAGE='Using __atomic_thread_fence' \
		-f $(dirname $(readlink -f $0))/check-forbidden-tokens.awk \
		"$1" || res=1

	# forbid use of experimental build flag except in examples
	awk -v FOLDERS='lib drivers app' \
		-v EXPRESSIONS='-DALLOW_EXPERIMENTAL_API allow_experimental_apis' \
		-v RET_ON_FAIL=1 \
		-v MESSAGE='Using experimental build flag for in-tree compilation' \
		-f $(dirname $(readlink -f $0))/check-forbidden-tokens.awk \
		"$1" || res=1

	# svg figures must be included with wildcard extension
	# because of png colwersion for pdf docs
	awk -v FOLDERS='doc' \
		-v EXPRESSIONS='::[[:space:]]*[^[:space:]]*\\.svg' \
		-v RET_ON_FAIL=1 \
		-v MESSAGE='Using explicit .svg extension instead of .*' \
		-f $(dirname $(readlink -f $0))/check-forbidden-tokens.awk \
		"$1" || res=1

	# links must prefer https over http
	awk -v FOLDERS='doc' \
		-v EXPRESSIONS='http://.*dpdk.org' \
		-v RET_ON_FAIL=1 \
		-v MESSAGE='Using non https link to dpdk.org' \
		-f $(dirname $(readlink -f $0))/check-forbidden-tokens.awk \
		"$1" || res=1

	return $res
}

check_experimental_tags() { # <patch>
	res=0

	cat "$1" |awk '
	BEGIN {
		lwrrent_file = "";
		ret = 0;
	}
	/^+++ b\// {
		lwrrent_file = $2;
	}
	/^+.*__rte_experimental/ {
		if (lwrrent_file ~ ".c$" ) {
			print "Please only put __rte_experimental tags in " \
				"headers ("lwrrent_file")";
			ret = 1;
		}
		if ($1 != "+__rte_experimental" || $2 != "") {
			print "__rte_experimental must appear alone on the line" \
				" immediately preceding the return type of a function."
			ret = 1;
		}
	}
	END {
		exit ret;
	}' || res=1

	return $res
}

check_internal_tags() { # <patch>
	res=0

	cat "$1" |awk '
	BEGIN {
		lwrrent_file = "";
		ret = 0;
	}
	/^+++ b\// {
		lwrrent_file = $2;
	}
	/^+.*__rte_internal/ {
		if (lwrrent_file ~ ".c$" ) {
			print "Please only put __rte_internal tags in " \
				"headers ("lwrrent_file")";
			ret = 1;
		}
		if ($1 != "+__rte_internal" || $2 != "") {
			print "__rte_internal must appear alone on the line" \
				" immediately preceding the return type of" \
				" a function."
			ret = 1;
		}
	}
	END {
		exit ret;
	}' || res=1

	return $res
}

number=0
range='origin/main..'
quiet=false
verbose=false
while getopts hn:qr:v ARG ; do
	case $ARG in
		n ) number=$OPTARG ;;
		q ) quiet=true ;;
		r ) range=$OPTARG ;;
		v ) verbose=true ;;
		h ) print_usage ; exit 0 ;;
		? ) print_usage ; exit 1 ;;
	esac
done
shift $(($OPTIND - 1))

if [ ! -f "$DPDK_CHECKPATCH_PATH" ] || [ ! -x "$DPDK_CHECKPATCH_PATH" ] ; then
	print_usage >&2
	echo
	echo 'Cannot execute DPDK_CHECKPATCH_PATH' >&2
	exit 1
fi

print_headline() { # <title>
	printf '\n### %s\n\n' "$1"
	headline_printed=true
}

total=0
status=0

check () { # <patch> <commit> <title>
	local ret=0
	headline_printed=false

	total=$(($total + 1))
	! $verbose || print_headline "$3"
	if [ -n "$1" ] ; then
		tmpinput=$1
	else
		tmpinput=$(mktemp -t dpdk.checkpatches.XXXXXX)
		trap "rm -f '$tmpinput'" INT

		if [ -n "$2" ] ; then
			git format-patch --find-renames \
			--no-stat --stdout -1 $commit > "$tmpinput"
		else
			cat > "$tmpinput"
		fi
	fi

	! $verbose || printf 'Running checkpatch.pl:\n'
	report=$($DPDK_CHECKPATCH_PATH $options "$tmpinput" 2>/dev/null)
	if [ $? -ne 0 ] ; then
		$headline_printed || print_headline "$3"
		printf '%s\n' "$report" | sed -n '1,/^total:.*lines checked$/p'
		ret=1
	fi

	! $verbose || printf '\nChecking API additions/removals:\n'
	report=$($VALIDATE_NEW_API "$tmpinput")
	if [ $? -ne 0 ] ; then
		$headline_printed || print_headline "$3"
		printf '%s\n' "$report"
		ret=1
	fi

	! $verbose || printf '\nChecking forbidden tokens additions:\n'
	report=$(check_forbidden_additions "$tmpinput")
	if [ $? -ne 0 ] ; then
		$headline_printed || print_headline "$3"
		printf '%s\n' "$report"
		ret=1
	fi

	! $verbose || printf '\nChecking __rte_experimental tags:\n'
	report=$(check_experimental_tags "$tmpinput")
	if [ $? -ne 0 ] ; then
		$headline_printed || print_headline "$3"
		printf '%s\n' "$report"
		ret=1
	fi

	! $verbose || printf '\nChecking __rte_internal tags:\n'
	report=$(check_internal_tags "$tmpinput")
	if [ $? -ne 0 ] ; then
		$headline_printed || print_headline "$3"
		printf '%s\n' "$report"
		ret=1
	fi

	if [ "$tmpinput" != "$1" ]; then
		rm -f "$tmpinput"
		trap - INT
	fi
	[ $ret -eq 0 ] && return 0

	status=$(($status + 1))
}

if [ -n "$1" ] ; then
	for patch in "$@" ; do
		# Subject can be on 2 lines
		subject=$(sed '/^Subject: */!d;s///;N;s,\n[[:space:]]\+, ,;s,\n.*,,;q' "$patch")
		check "$patch" '' "$subject"
	done
elif [ ! -t 0 ] ; then # stdin
	subject=$(while read header value ; do
		if [ "$header" = 'Subject:' ] ; then
			IFS= read next
			continuation=$(echo "$next" | sed -n 's,^[[:space:]]\+, ,p')
			echo $value$continuation
			break
		fi
	done)
	check '' '' "$subject"
else
	if [ $number -eq 0 ] ; then
		commits=$(git rev-list --reverse $range)
	else
		commits=$(git rev-list --reverse --max-count=$number HEAD)
	fi
	for commit in $commits ; do
		subject=$(git log --format='%s' -1 $commit)
		check '' $commit "$subject"
	done
fi
pass=$(($total - $status))
$quiet || printf '\n%d/%d valid patch' $pass $total
$quiet || [ $pass -le 1 ] || printf 'es'
$quiet || printf '\n'
exit $status
