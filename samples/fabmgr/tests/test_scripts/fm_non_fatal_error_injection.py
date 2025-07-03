import sys
sys.path.append("./../utils")
import shlex
import pexpect
import time
import random
from fm_test_utils import *
from fm_process_start_stop_test import *
from p2p_write_ce_bandwidth_test import *

def injectNonFatalError(lwswitch_bdf_str):
	#clear dmesg before injecting error
	child = pexpect.spawn("dmesg -c")

	deviceID = getDeviceId()

	#injecting the non fatal error
	if deviceID == WILLOW_ID:
		lwpex_app = app_dir + "lwpex2/lwpex2" 
		args = "--bdf " + lwswitch_bdf_str + " -r W:0:8D0040:F0094 -i 1"
		cmd = lwpex_app + " " + args
		child = pexpect.spawn(cmd)
		msg = child.read()
	elif deviceID == LIMEROCK_ID:
		#inject non fatal error
		pass

 	#wait for 6 seconds
	time.sleep(NON_FATAL_ERROR_REPORTING_CNT)

	# look for non fatal error in dmesg after injecting error
	cmd = 'dmesg > /tmp/dmesg.txt; grep -i "fatal" /tmp/dmesg.txt | wc -l'
	child = pexpect.spawn('''/bin/bash -c "''' + cmd + '''"''')
	out = str(child.read(), "utf-8").split("\n")[0]

	if int(out) > 0:
		return True
	else:
		print("Non fatal error not found")
		return False


def performNonFatalErrorTest():
	"""
		i)	  Ensure FM is still running
		ii)   Inject non fatal error
		iii)  FM does error recovery
		iv)   Run p2p bandwidth
	"""

	child = pexpect.spawn('/bin/bash -c "ps -ax | grep [n]v-fabricmanager"')

	fm_proc = str(child.read(),'utf-8').split("\n")
	if len(fm_proc) < 2:
		#start FM
		fm = runFabricManager()
		if waitForFabricManagerReady(fm) == True:
			print("Fabricmanager done with configuring switches....")

	if injectNonFatalError(getLwswitchBdfStr()) == False:
		print("injecting non fatal error was not successful")
		return False
	
	#wait for error recovery
	time.sleep(TIME_TO_SLEEP)
	
	if runP2pCeBandwidthTest() == False:
		return False

	return True

def main():
	coloredPrint(bcolors.BOLD, "Running non fatal error test....") 
	#Error recovery test 
	if performNonFatalErrorTest():
		coloredPrint(bcolors.OKGREEN, "Non fatal error Test successful")
	else: 
		coloredPrint(bcolors.FAIL, "Non fatal error Test failed")
		return 


if __name__ == '__main__':
	main()