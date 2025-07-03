#!/bin/bash

SRIOV_ENABLED_GPUS=(20b0: 20be: 20bf: 20f1:)

# Sanity check
usage()
{
    cat >&2 <<EOF
Usage:
    /usr/lib/lwpu/sriov-manage <-e|-d> <ssss:bb:dd.f|ALL>
        Enable/disable Virtual functions on the GPU specified by the PCI_ID=ssss:bb:dd.f.
        If 'ALL' is specified in place of PCI_ID, operate on all the lwpu GPUs in the system.

    /usr/lib/lwpu/sriov-manage -h
        Print this help.
EOF
}

lwidia_vfs_supported()
{
    local gpu=$1
    local gpu_devid

    gpu_devid=$(lspci -s "$gpu" -nm | awk '{print gensub("\"","","g",$4":"$7)}')

    for sriov_enabled_gpu in "${SRIOV_ENABLED_GPUS[@]}"; do
        [[ "$gpu_devid" =~ ^$sriov_enabled_gpu ]] && return 0
    done

    local RMSetSriovMode
    if ! [ -r /proc/driver/lwpu/params ]; then
        echo "Lwpu driver not loaded. Cannot continue."
        return 1
    fi
    RMSetSriovMode=$(awk '$1=="RegistryDwords:"{
        RegistryDwords_str=gensub("\\\"", "", "g", $2);
        c = split(RegistryDwords_str, RegistryDwords, ";");
        for (i = 1; i <= c; i++)
            if (RegistryDwords[i] ~ /^RMSetSriovMode=/) {
                RMSetSriovMode = gensub("RMSetSriovMode=", "", "g", RegistryDwords[i]);
            }
    }
    END {print RMSetSriovMode}
    ' /proc/driver/lwpu/params)

    if ! (( RMSetSriovMode )); then
        # RMSetSriovMode not set
        return 1
    fi

    return 0
}

sanitize_gpu_sbdf()
{
    local gpu=$1
    gpu=$(lspci -D -d 10de: -s "$gpu" | awk '{print $1}') # Easy way of formatting.
    [ "$gpu" ] || exit 1
    echo "$gpu"
}

kernel_support_check()
{
    [ -e /sys/bus/pci/drivers/pci-pf-stub ] || modprobe pci-pf-stub || exit 1
}

input_sanity_check()
{
    local gpu=$1

    device_has_sriov "$gpu" || exit 0
    lwidia_vfs_supported "$gpu" || exit 0

    if [ "$(setpci -s "$gpu" b.b)" != 03 ]; then # class code
        echo "This is not a VGA or 3D controller. Exiting." >&2
        exit 1
    fi
}

unbind_from_existing_driver()
{
    local gpu=$1
    local existing_driver_name
    local existing_driver
    [ -e "/sys/bus/pci/devices/$gpu/driver" ] || return 0
    existing_driver=$(readlink -f "/sys/bus/pci/devices/$gpu/driver")
    existing_driver_name=$(basename "$existing_driver")
    if [ "$existing_driver_name" == lwpu ]; then
        get_lwidia_unbind_lock "$gpu" || return 1
    fi
    echo "$gpu" > "$existing_driver/unbind"
}

bind_to_lwidia_driver()
{
    local gpu=$1
    echo "$gpu" > /sys/bus/pci/drivers/lwpu/bind
    resume_services
}

get_lwidia_unbind_lock()
{
    local unbindlock_file="/proc/driver/lwpu/gpus/$1/unbindLock"
    local unbindlock=0
    [ -e "$unbindlock_file" ] || return 1
    stop_services
    echo 1 > "$unbindlock_file"
    read -r unbindlock < "$unbindlock_file"
    [ "$unbindlock" == 1 ] && return 0
    echo "Cannot obtain unbindLock for $1" >&2
    resume_services
    return 1
}

# Hypervisor specific code starts.
#KVM_SPECIFIC_CODE_STARTS
if [ -d /sys/devices/virtual/misc/kvm ]; then
    stop_hypervisor_services()
    {
        return 0
    }

    resume_hypervisor_services()
    {
        return 0
    }

    unbind_from_hypervisor_passthrough_backend()
    {
        [ -e "/sys/bus/pci/drivers/vfio-pci/$1" ] && echo "$1" > /sys/bus/pci/drivers/vfio-pci/unbind
        [ -e "/sys/bus/pci/drivers/lwpu/$1" ] && echo "$1" > /sys/bus/pci/drivers/lwpu/unbind
    }

    setup_vf_kernel_driver_binding()
    {
        local vf=$1
        # Bind VF to lwpu driver
        if ! [ -e "/sys/bus/pci/drivers/lwpu/$vf" ]; then
            if [ -e "/sys/bus/pci/devices/$vf/driver" ]; then
                echo "$vf" > "/sys/bus/pci/devices/$vf/driver/unbind"
            fi
            echo "$vf" > "/sys/bus/pci/drivers/lwpu/bind"
        fi
    }
fi
#KVM_SPECIFIC_CODE_ENDS
# Hypervisor specific code ends.

stop_services()
{
    stop_hypervisor_services
}

resume_services()
{
    # Make sure that the driver is loaded on the GPU
    lwpu-smi -pm 1 &>/dev/null
    resume_hypervisor_services
}

unbind_from_passthrough_backend()
{
    unbind_from_hypervisor_passthrough_backend "$@"
}

device_has_sriov()
{
    [ -e "/sys/bus/pci/devices/$1/sriov_numvfs" ] && return 0
    return 1
}

get_totalvfs()
{
    cat "/sys/bus/pci/devices/$1/sriov_totalvfs"
}

get_numvfs()
{
    cat "/sys/bus/pci/devices/$1/sriov_numvfs"
}

set_numvfs()
{
    local gpu=$1
    local numvfs=$2

    # unbind from existing driver
    unbind_from_existing_driver "$gpu" || return 1
    # bind to pci-pf-stub
    lspci -n -s "$gpu" | awk '{gsub(":", " ", $3); print $3}' > /sys/bus/pci/drivers/pci-pf-stub/new_id
    [ -e "/sys/bus/pci/drivers/pci-pf-stub/$gpu" ] || echo "$gpu" > /sys/bus/pci/drivers/pci-pf-stub/bind
    # set numvfs
    (( numvfs )) && sleep 0.5
    echo "$numvfs" > "/sys/bus/pci/devices/$gpu/sriov_numvfs"
    # unbind from pci-pf-stub
    echo "$gpu" > /sys/bus/pci/drivers/pci-pf-stub/unbind
    lspci -n -s "$gpu" | awk '{gsub(":", " ", $3); print $3}' > /sys/bus/pci/drivers/pci-pf-stub/remove_id
    # bind to lwpu driver
    bind_to_lwidia_driver "$gpu"
}

enable_vfs()
{
    local gpu=$1

    if [ "${gpu,,}" == all ]; then
        enable_all
        exit
    fi

    gpu=$(sanitize_gpu_sbdf "$gpu") || exit 1
    input_sanity_check "$gpu"
    kernel_support_check

    # GPU already has VFs. Nothing to do.
    if [ "$(get_numvfs "$gpu")" != 0 ]; then
        echo "GPU at $gpu already has VFs enabled."
        return 0
    fi

    echo "Enabling VFs on $gpu"
    # Set numvfs via the pci-pf-stub driver
    set_numvfs "$gpu" "$(get_totalvfs "$gpu")"

    local sb=${gpu/00.0}
    lspci -D -s "$sb" | while read -r vf _; do
        [ "$vf" != "$gpu" ] && setup_vf_kernel_driver_binding "$vf"
    done

}

disable_vfs()
{
    local gpu=$1
    if [ "${gpu,,}" == all ]; then
        disable_all
        exit
    fi
    local vf

    gpu=$(sanitize_gpu_sbdf "$gpu") || exit 1
    input_sanity_check "$gpu"
    kernel_support_check

    # GPU has no VFs. Nothing to do.
    if [ "$(get_numvfs "$gpu")" == 0 ]; then
        echo "GPU at $gpu already has VFs disabled."
        return 0
    fi

    echo "Disabling VFs on $gpu"
    # Unbind any existing VFs from its owned driver
    local sb=${gpu/00.0}
    lspci -D -s "$sb" | while read -r vf _; do
        [ "$vf" != "$gpu" ] && unbind_from_passthrough_backend "$vf"
    done

    # Set numvfs via the pci-pf-stub driver
    set_numvfs "$gpu" 0
}

enable_all()
{
    for gpu in $(lspci -Dd 10de: -s 0.0 -n | awk '$2 ~ /030[02]/{print $1}'); do
        "$0" -e "$gpu" || exit
    done
}

disable_all()
{
    for gpu in $(lspci -Dd 10de: -s 0.0 -n | awk '$2 ~ /030[02]/{print $1}'); do
        "$0" -d "$gpu" || exit
    done
}

while getopts "e:d:h" OPTION; do
    case $OPTION in
        h) usage; exit;;
        e) operation=enable_vfs; gpu="$OPTARG";;
        d) operation=disable_vfs; gpu="$OPTARG";;
        *) usage; exit 1;;
    esac
done

if [ "$operation" ]; then
    $operation "$gpu"
    exit
fi

usage
exit 1
