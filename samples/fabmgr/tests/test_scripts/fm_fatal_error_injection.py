import sys
sys.path.append("./../utils")
import shlex
import pexpect
import time
import random
from fm_test_utils import *
from fm_process_start_stop_test import *
from p2p_write_ce_bandwidth_test import *

def injectFatalError(lwswitch_bdf_str):
	#Inject fatal error using lwpex2

	#clear dmesg before injecting error
	child = pexpect.spawn("dmesg -c")

	deviceID = getDeviceId()

	#injecting the fatal error
	if deviceID == WILLOW_ID:
		lwpex_app = app_dir + "lwpex2/lwpex2" 
		args = "--bdf " + lwswitch_bdf_str + " -r W:0:856710:FFFF -i 1"
		cmd = lwpex_app + " " + args
		child = pexpect.spawn(cmd)
		msg = child.read()
	elif deviceID == LIMEROCK_ID:
		#inject fatal error
		pass

	#wait for 1 second
	time.sleep(1)

	# look for fatal error in dmesg after injecting error
	cmd = 'dmesg > /tmp/dmesg.txt; grep -i "fatal" /tmp/dmesg.txt | wc -l'
	child = pexpect.spawn('''/bin/bash -c "''' + cmd + '''"''')
	out = str(child.read(), "utf-8").split("\n")[0]

	if int(out) > 0:
		return True
	else:
		print("Fatal error not found")

	return False

def performFatalErrorTest():
	"""
	 Fatal Error Test
	 	i)    Stop FM if running
		ii)   Inject fatal error
		iii)  Reset hardware using lwpu-smi
		iv)   Run FM
		v)    Run p2p bandwidth
	"""	

	if stopFabricManagerIfRunning() == -1:
		print("Could not stop running fabricmanager instance....")
		return False

	if injectFatalError(getLwswitchBdfStr()) == False:
		print("injecting fatal error was not successful")
		return False

	if performFabricReset() == False:
		print("Fabric reset failed")

	fm = runFabricManager()

	if fm == 0:
		return 0

	if waitForFabricManagerReady(fm) == True:
		pass
	else:
		print("Fabricmanager did not Successfully configure the switches")
		return False

	time.sleep(5)
	
	if runP2pCeBandwidthTest() == False:
		return False

	return True

def main():

	coloredPrint(bcolors.BOLD, "Running fatal error test....")
	#Fatal error test
	if performFatalErrorTest() == True:
		coloredPrint(bcolors.OKGREEN, "Fatal Error Test successful")
	else:
		coloredPrint(bcolors.FAIL, "Fatal Error Test failed")
		return

if __name__ == '__main__':
	main()