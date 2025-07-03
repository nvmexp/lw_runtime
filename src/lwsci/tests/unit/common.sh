#!/bin/bash -e

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
EXECDIR="$( cd "$( dirname "$0" )" >/dev/null 2>&1 && pwd )"
PROJECT="$(basename "$( cd "$( dirname "$EXECDIR/../../.." )" >/dev/null 2>&1 && pwd )")"

echo $SCRIPTDIR
echo $EXECDIR
echo $PROJECT

## Set target and host IP address
TARGET=""
HOST=""
## Run VCast in the end of script
SHOW_GUI=false

show_help() {
    printf 'VectorCast test suite exelwtion script\n'
    printf '\n'
    printf 'Usage:\t./setup.sh [--target target_ip [--host host_ip]] [--gui] \n'
    printf '\n'
    printf 'Options:\n'
    printf ' --target\n\tSpecify target IP address, implies that test suite will be run on a remote device\n'
    printf ' --host\n\tSpecify host IP address, works only if --target is specified\n'
    printf ' --gui\n\tOpen VectorCAST gui with current environment after script exelwtion\n'
}

die() {
    printf '%s\n' "$1" >&2
    exit 1
}

mount_nfs() {
    set -u
    # add PWD to NFS exports if does not exist
    sudo grep -Fq "$PWD" /etc/exports || echo "$PWD *(rw,no_subtree_check,sync,no_root_squash,no_all_squash)" | sudo tee -a /etc/exports
    sudo exportfs -a
    # ssh-keygen -f "~/.ssh/known_hosts" -R $TARGET
    # ssh-keyscan -H $TARGET >> ~/.ssh/known_hosts
    sshpass -p root ssh -tt root@${TARGET} "mount | grep -Fq ${PWD} || fs-nfs3 -w delay=0 -w sync=hard ${HOST}:${PWD} ${PWD}"
    set +u
}

umount_nfs() {
    # Do not contaminate /etc/exports
    sudo sed -i '/$PWD/d' /etc/exports
    sudo exportfs -a
}

if [ -z $UNIT ]; then
    die 'ERROR: "UNIT" variable is not set.'
fi

# Parse command-line options
while :; do
    case $1 in
        -h|-\?|--help)
            show_help    # Display a usage synopsis.
            exit
            ;;
        --gui)
            show_gui=true
            ;;
        # Specify target IP address, implies that test suite is exelwted remotely on target device
        --target)
            if [ "$2" ]; then
                TARGET=$2
                shift
            else
                die 'ERROR: "--target" requires a non-empty option argument.'
            fi
            ;;
        --target=?*)
            TARGET=${1#*=} # Delete everything up to "=" and assign the remainder.
            ;;
        --target=)         # Handle the case of an empty --target=
            die 'ERROR: "--target" requires a non-empty option argument.'
            ;;
        # Specify host IP address, works when we specify --target
        --host)
            if [ "$2" ]; then
                HOST=$2
                shift
            else
                die 'ERROR: "--host" requires a non-empty option argument.'
            fi
            ;;
        --host=?*)
            HOST=${1#*=} # Delete everything up to "=" and assign the remainder.
            ;;
        --host=)         # Handle the case of an empty --host=
            die 'ERROR: "--host" requires a non-empty option argument.'
            ;;
        --)              # End of all options.
            shift
            break
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)               # Default case: No more options, so break out of the loop.
            break
    esac

    shift
done

export TARGET=$TARGET
export HOST=${HOST:-10.0.0.1}
export TEGRA_TOP=$SCRIPTDIR/../../../../../../
export GIT_TOP=$SCRIPTDIR/../../../../

if [ ! -d "$GIT_TOP/.git" ]; then
    die 'ERROR: "GIT_TOP does not point to git top directory. Please check the script.'
fi

## Export necessary paths
export VECTORCAST_DIR=$P4ROOT/sw/tools/VectorCAST/linux/2020sp5
export VECTOR_LICENSE_FILE=1787@sc-lic-25
export VCAST_PROJ_DIR=$EXECDIR
export VCAST_COMPILER_PATH=$_lw_qnx_base/host/linux/x86_64/usr/bin
export PATH=$VECTORCAST_DIR:$VCAST_COMPILER_PATH:$PATH

## build VC project
cd $VCAST_PROJ_DIR
## Clean-up first
rm -rf $UNIT CCAST_.CFG

#prepare OUTMIRROR
mkdir -p include/outmirror/tegra_top/core/include
mkdir -p include/outmirror/tegra_top_hidden/core/include
mkdir -p include/outmirror/tegra_top/core-private/include
cp -r $GIT_TOP/../../core/include include/outmirror/tegra_top/core/
cp -r $GIT_TOP/../../core/include include/outmirror/tegra_top_hidden/core/
cp -r $GIT_TOP/../../core-private/include include/outmirror/tegra_top/core-private/

# mount the project dir on target, not building for host
cd $EXECDIR
if [ ! -z $TARGET ]; then
    mount_nfs
    export _lw_qnx_base=${P4ROOT}/sw/tools/embedded/qnx/qnx700-ga6
    export QNX_HOST=$_lw_qnx_base/host/linux/x86_64/
    export QNX_TARGET=$_lw_qnx_base/target/qnx7/
    export PATH=$QNX_HOST/usr/bin/:$PATH
    $SCRIPTDIR/$PROJECT.csh target
else
    $SCRIPTDIR/$PROJECT.csh host
fi

if [ "$show_gui" = true ]; then
    # start VectorCast
    $VECTORCAST_DIR/vcastqt -e $UNIT &
fi
