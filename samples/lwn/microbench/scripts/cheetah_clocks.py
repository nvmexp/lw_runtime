#!/usr/bin/elw python2.7

import getopt
import os
import sys
import subprocess

def adb_shell(c):
    cmd = "adb shell \"{0}\"".format(c)
    print(cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    p.wait()
    return list(p.stdout)

# Returns possible clock rates (in Hz) for a given bus (gbus/emc)
def possible_rates(bus):
    if bus == "cpu":
        stdout = adb_shell("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies".format(bus))
    else:
        stdout = adb_shell("cat /d/clock/{0}/possible_rates".format(bus))
    return [int(r) * 1000 for r in stdout[0].split(" ") if r.isdigit()]

# Set FPLW (fast) clock rate (rate in Hz)
#
# TODO with low clocks, "someone" sets cluster back to LP from G, even
# though we successfully force it to G here.
#
# This turns off a whole bunch of things, so don't expect your system
# to behave "like a production system" after running this.
def force_cpu_rate(rate):
    cmds = ['stop ussrd',
            'echo 0 > /sys/module/qos/parameters/enable' # disable QoS
            '# Disable auto hotplug and auto cluster switch',
            'echo 0 > /sys/devices/system/cpu/cpuquiet/tegra_cpuquiet/enable',
            'echo 1 > /sys/kernel/cluster/immediate',
            'echo 1 > /sys/kernel/cluster/force',
            '# Enable the FCPU cluster',
            'echo G > /sys/kernel/cluster/active',
            '# Enable all 4 CPUs if needed (Optional step).  ',
            'hotplug 1 1',
            'hotplug 2 1',
            'hotplug 3 1',
            '# Prevent cpuquiet from dynamically enabling/disabling cores (by switching to the userspace governor)',
            'echo userspace > /sys/devices/system/cpu/cpuquiet/lwrrent_governor',
            '# Disable EDP',
            'echo 0 > /d/edp/vdd_cpu/edp_reg_override',
            '# Prevent cpufreq from dynamically adjusting cores',
            'echo "userspace" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor',
            'echo "userspace" > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor',
            'echo "userspace" > /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor',
            'echo "userspace" > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor',
            '# Relax the max cpu cap to be atleast equal to the required cpu speed. ',
            'echo $KHZ > /sys/module/cpu_tegra/parameters/cpu_user_cap',
            '# Set the required speed',
            'echo $KHZ > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed',
            'echo $KHZ > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed',
            'echo $KHZ > /sys/devices/system/cpu/cpu2/cpufreq/scaling_setspeed',
            'echo $KHZ > /sys/devices/system/cpu/cpu3/cpufreq/scaling_setspeed']
    if rate:
        khz = str(rate / 1000)
        for cmd in cmds:
            c = cmd.replace("$KHZ", khz)
            adb_shell(c)

# Round rate to one of the available rates.  The user is asked to
# specify rates in MHz but rate*1000000 or rate*1000 won't match the
# actual rates the HW can run on.
def find_closest_mhz_match(rates, rate):
    for r in rates:
        if (r / 1000000) == (rate / 1000000):
            return r
    print "Failed to set clock rate to {0} - not a possible frequency.".format(rate)
    print "Use one of the allowed rates:"
    print ", ".join([str(r/1000000) for r in rates])
    sys.exit(1)


def set_rate(bus, rate):
    if rate:
        rates = possible_rates(bus)
        rate = find_closest_mhz_match(rates, rate)
        new_rate = None
        if bus == "cpu":
            force_cpu_rate(rate)
            new_rate = int(adb_shell("cat /d/clock/cpu/rate")[0])
        else:
            adb_shell("echo {1} > /d/clock/override.{0}/rate".format(bus, rate))
            adb_shell("echo 1 > /d/clock/override.{0}/state".format(bus))

            adb_shell("echo {1} > /d/clock/cap.{0}/rate".format(bus, rate))
            adb_shell("echo 1 > /d/clock/cap.{0}/state".format(bus))
            new_rate = int(adb_shell("cat /d/clock/{0}/rate".format(bus))[0])
        if new_rate != rate:
            print "Failed to set {0} rate to {1}.  New rate was set to {2}".format(bus, rate, new_rate)
            print "Use one of the allowed rates:"
            print ", ".join([str(r) for r in rates])
            sys.exit(1)
    else:
        if bus != "cpu":
            adb_shell("echo 0 > /d/clock/cap.{0}/state".format(bus))
            adb_shell("echo 0 > /d/clock/floor.{0}/state".format(bus))

def parse_rates(rate_str, bus):
    rates = possible_rates(bus)
    if rate_str == "shmoo":
        if bus == "cpu":
            return sorted([r for r in rates if r >= 306000000], reverse=True)
        else:
            return sorted(rates, reverse=True)
    else:
        # Adjust MHz input to actual clock rate in Hz
        rates_in_hz = [int(a) * 1000000 for a in rate_str.split(',')]
        return [find_closest_mhz_match(rates, r) for r in rates_in_hz]
