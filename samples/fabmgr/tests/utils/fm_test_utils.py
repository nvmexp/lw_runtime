import pexpect
import random
import sys
import telnetlib
import time
from pmodel_explorer16 import DgxProductModelExplorer16
from pmodel_explorer8 import DgxProductModelExplorer8

app_dir = './../apps/'
TIME_TO_SLEEP = 10
LOG_FILE = '/var/log/fabricmanager.log'
NON_FATAL_ERROR_REPORTING_CNT = 6
WILLOW_ID = '1ac'
LIMEROCK_ID = '1af'

class bcolors:
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def coloredPrint(color, msg):
	print(color + msg + bcolors.ENDC)	


def getLwswitchBdfStr():
	""" 
		This function gets the bdf of lwswitches with the help of lspci command
	"""

	lwswitch_bdf_strs = []
	child = pexpect.spawn("/bin/bash -c 'lspci | grep -i lwpu'")

	msg = str(child.read(), 'utf-8').split("\n")

	for i in range(len(msg)):
		if "bridge" in msg[i].lower():
			bdf = msg[i].split(" ")[0]
			lwswitch_bdf_strs.append(bdf)

	randindex = random.randint(0,len(lwswitch_bdf_strs)-1)

	return lwswitch_bdf_strs[randindex]


def getDeviceId():
	""" 
		This function gets the device ID of Lwswitches, can be used to differentiate
		between generations (Willow/Limerock)
	"""

	deviceId = 0
	child = pexpect.spawn("/bin/bash -c 'lspci | grep -i lwpu'")

	msg = str(child.read(), 'utf-8').split("\n")

	for i in range(len(msg)):
		if "bridge" in msg[i].lower():
			deviceId = msg[i].split(" ")[5]
			return deviceId[:-1]


def performFabricReset():
	""" 
		Perform fabric reset using lwpu-smi. Return true if reset is successful
	"""
	child = pexpect.spawn("lwpu-smi -r")

	res = child.expect(["All done", pexpect.EOF, pexpect.TIMEOUT])
	if res == 0:
		return True

	return False

def ExelwteCommandServerQueryCmd(cmd, delay=0.5):
    queryCmd = "/query " + cmd + "\r\n"
    quitCmd = "/quit" + "\r\n"
    tn = telnetlib.Telnet('localhost', 17000)
    # execute the actual command
    tn.write(queryCmd.encode('ascii'))
    # close the telnet session. give sometime for previous command to finish
    sleep(delay) # Time in seconds.
    tn.write(quitCmd.encode('ascii'))
    return (tn.read_all().decode('ascii'))

def GetDgxProductModelObj(pmodel):
    if pmodel == "Explorer16":
        return DgxProductModelExplorer16()
    if pmodel == "Explorer8":
        return DgxProductModelExplorer8()
    if pmodel == "Hgx2":
    	return Hgx2ProductModel()
    #default case
    return None