import sys
sys.path.append("./../utils")
import random
import pexpect
from fm_test_utils import *
from fm_process_start_stop_test import *

def runP2pCeBandwidthTest():
	"""
		p2p bandwidth test. Lwrrently testing only Memcpy_DtoD_Write_CE_Bandwidth
	"""
	p2p_app = app_dir + "p2p_bandwidth/p2p_bandwidth"
	#execute p2p
	child = pexpect.spawn(p2p_app + " -t Memcpy_DtoD_Write_CE_Bandwidth")

	#check if process still running
	res = child.expect(["ENABLED", pexpect.EOF, pexpect.TIMEOUT])
	if res == 0:
		res = child.expect(["100", pexpect.EOF, pexpect.TIMEOUT])
		if res == 0:
			return True
		else:
			return False
	else:
		return False


	return 0


def main():
	coloredPrint(bcolors.BOLD, "Running p2p bandwidth test Memcpy_DtoD_Write_CE_Bandwidth...")

	if isFabricManagerRunning() != True:
		fm = runFabricManager()

		if fm == 0:
			return 0

		if waitForFabricManagerReady(fm) == True:
			coloredPrint(bcolors.OKGREEN, "Fabricmanager done with configuring switches....")
		else:
			coloredPrint(bcolors.FAIL, "Fabricmanager did not Successfully configure the switches")
			return

	#p2p bandwidth test
	if runP2pCeBandwidthTest():
		coloredPrint(bcolors.OKGREEN, "P2P bandwidth is successful")
	else:
		coloredPrint(bcolors.FAIL, "P2P bandwidth test failed")

if __name__ == '__main__':
	main()