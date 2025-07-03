import sys
sys.path.append("./../utils/")
from pmodel_explorer16 import DgxProductModelExplorer16
from pmodel_explorer8 import DgxProductModelExplorer8
import pexpect
from fm_test_utils import *
from fm_process_start_stop_test import *

def getConfigInfo(config):

	num_gpus, num_switches, num_lwlinks, num_trunk_link_conns = 0,0,0,0

	child = pexpect.spawn("/bin/bash -c 'lspci | grep -i lwpu'")
	msg = str(child.read(), 'utf-8').split("\n")

	for i in range(len(msg)):
		if "bridge" in msg[i].lower():
			num_switches += 1
		elif "controller" in msg[i].lower():
			num_gpus += 1

	#find number of lwlinks by looking at the fabricmanager.log
	cmd = "cat " + LOG_FILE
	child = pexpect.spawn('''/bin/bash -c "''' + cmd + ''' | grep 'Number of connections'"''')

	msg = str(child.read(), 'utf-8').split("\n")[0]
	num_lwlinks = int(msg.split(":")[-1])


	#get lwlink trunk connection info
	cmd = "cat " + LOG_FILE
	child = pexpect.spawn('''/bin/bash -c "''' + cmd + ''' | grep trunk"''')
	msg = child.read()

	if msg:
		intra_node_trunk_link_conn = int(str(msg, 'utf-8').split("\n")[0].split(":")[-1])
		inter_node_trunk_link_conn = int(str(msg, 'utf-8').split("\n")[1].split(":")[-1])

		num_trunk_link_conns = intra_node_trunk_link_conn + inter_node_trunk_link_conn

	return num_gpus, num_switches, num_lwlinks, num_trunk_link_conns


def isAllLWLinkConnsTrained(num_lwlinks):
	cmd = "cat " + LOG_FILE
	child = pexpect.spawn('''/bin/bash -c "''' + cmd + ''' | grep 'connection is not trained to ACTIVE' | wc -l"''')
	msg = str(child.read(), "utf-8").split("\n")[0]

	if int(msg) == 0:
		return True

	return False
  
def main():
	coloredPrint(bcolors.BOLD, "Starting test basic sanity")
	config = sys.argv[1]

	cmd = 'cat /dev/null > ' + LOG_FILE
	child = pexpect.spawn('/bin/bash -c "cat /dev/null > ' + LOG_FILE + '"')

	fm = runFabricManager()

	if fm == 0:
		return 0

	if waitForFabricManagerReady(fm) == True:
		coloredPrint(bcolors.OKGREEN, "Fabricmanager done with configuring switches....")
	else:
		coloredPrint(bcolors.FAIL, "Fabricmanager did not Successfully configure the switches")
		return

	model = GetDgxProductModelObj(config);

	if model == None:
		print("Provided model name is incorrect")

	num_gpus, num_switches, num_lwlinks, num_trunk_link_conns = getConfigInfo(config)

	#basic assertion check to see if number of GPUs, switches and lwlinks are matching
	assert num_gpus == model.NumGpus(), "Number of GPUs found does not match the " + config + " configuration"
	assert num_switches == model.NumLWSwitches(), "Number of LWSwitches found does not match the " + config + " configuration"
	assert num_lwlinks == model.NumLWLinkConns(), "Number of LWLinks found does not match the " + config + " configuration"
	if config == "Explorer16":
		assert num_trunk_link_conns == model.NumLWLinkTrunkConns(), "Number of trunk LWLinks found does not match the " + config + " configuration"

	coloredPrint(bcolors.OKGREEN, "Successfully completed basic sanity Test...")

	if isAllLWLinkConnsTrained(num_lwlinks) == True:
		coloredPrint(bcolors.OKGREEN, "All LWLink Connections trained Successfully...")
	else:
		coloredPrint(bcolors.FAIL, "All LWLink Connections not trained Successfully...")

if __name__ == "__main__":
    main()
