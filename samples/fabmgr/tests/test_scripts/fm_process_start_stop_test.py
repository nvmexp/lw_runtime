import sys
sys.path.append("./../utils")
import shlex
import time
import random
from fm_test_utils import *

def killFabricmanager(pid):
	"""
		Stop fabricmanager instance if running
	"""

	#kill fm process
	child = pexpect.spawn("kill -9 " + pid)

	msg = str(child.read(), 'utf-8')

	if len(msg) > 0:
		#process could not be killed or some error
		print(msg)	
		return False

	return True

def stopFabricManagerIfRunning():
	#check if fabricmanager is already running
	child = pexpect.spawn('/bin/bash -c "ps -ax | grep [n]v-fabricmanager"')

	fm_proc = str(child.read(),'utf-8').split("\n")

	for i in range(len(fm_proc)-1):
		if len(fm_proc) == 1:
			continue
		fm_proc[i] = fm_proc[i].strip()
		pid = fm_proc[i].split(" ")[0]
		# fabricmanager process is running
		ret = killFabricmanager(pid)
		if ret == False:
			return -1

	time.sleep(10)

	return 1

def isFabricManagerRunning():

	#check if fabricmanager is already running
	child = pexpect.spawn('/bin/bash -c "ps -ax | grep [n]v-fabricmanager"')

	fm_proc = str(child.read(),'utf-8').split("\n")

	if len(fm_proc) > 1:
		return True

	return False

def runFabricManager():
	"""
		Start fabricmanager instance
	"""
	fm_app_dir = '/usr/bin/lw-fabricmanager'

	if stopFabricManagerIfRunning() == -1:
		print("Could not stop running fabricmanager instance....")
		return 0

	#wait for 2 seconds before starting fm again
	time.sleep(2)

	fm = pexpect.spawn(fm_app_dir + " -c /usr/share/lwpu/lwswitch/fabricmanager.cfg")
	return fm


def waitForFabricManagerReady(fm):
	'''
		Check if fabricmanager process has finished configuring switches
	'''
	# it could take 40 more seconds
	waitTime = 40.0
	start = time.time()

	res = fm.expect(["Successfully configured all the available GPUs and LWSwitches", pexpect.EOF, pexpect.TIMEOUT], timeout=waitTime)
	if res == 0:
		return True

	return False


def main():
	coloredPrint(bcolors.BOLD, "Running fabricmanager .....")

	fm = runFabricManager()

	if fm == 0:
		return 0

	if waitForFabricManagerReady(fm) == True:
		coloredPrint(bcolors.OKGREEN, "Fabricmanager done with configuring switches....")
	else:
		coloredPrint(bcolors.FAIL, "Fabricmanager did not Successfully configure the switches")
		return

if __name__ == '__main__':
	main()