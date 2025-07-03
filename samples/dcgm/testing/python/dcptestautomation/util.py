import sys
from subprocess import PIPE, Popen

###############################################################################################
#
# Utility function to execute bash commands from this script
# Prints the stdout to screen and returns a return code from the shell
#
###############################################################################################
def exelwteBashCmd(cmd, prnt):
    """
    Exelwtes a shell command as a separated process, return stdout, stderr and returncode
    """
    ret_line = ''

    print "exelwteCmd: \"%s\"" % str(cmd)
    try:
        result = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
        ret_line = result.stdout.readline()
        while True:
            line = result.stdout.readline()
            if prnt:
                print line.strip('\n')
            if not line:
                break
            sys.stdout.flush()
        (stdout, stderr) = result.communicate()
        if stdout:
            print stdout#pass
        if stderr:
            print stderr
    except Exception as msg:
        print "Failed with: %s" % msg

    return result.returncode, ret_line.strip()

###############################################################################################
#
# This function removes cleans up all the dependent libraries
# It also uninstalls any existing installation of datacenter gpu manager
# Returns success in the end.
#
###############################################################################################
def removeDependencies(prnt):
    #Remove existing installation files and binaries
    ret = exelwteBashCmd("echo {0} | sudo pip uninstall pandas".format('y'), prnt)
    print ("sudo pip uninstall pandas returned: ", ret[0])
    #if module is not installed, this returns 1, so check for both values
    if (ret[0] in [0, 1]):
        ret = exelwteBashCmd("echo {0} | sudo pip uninstall wget".format('y'), prnt)
        print ("sudo pip uninstall wget returned: ", ret[0])

    if (ret[0] in [0, 1]):
        ret = exelwteBashCmd("echo {0} | sudo pip uninstall xlwt".format('y'), prnt)
        print ("sudo pip uninstall xlwt returned: ", ret[0])

    if (ret[0] in [0, 1]):
        ret = exelwteBashCmd("echo {0} | sudo pip uninstall xlrd".format('y'), prnt)
        print ("sudo pip uninstall xlrd returned: ", ret[0])

    if ret[0] in [0, 1]:
        print "\nRemoveDependencies returning 0"
        return 0

    print ("\nReturning: ", ret[0])
    return ret[0]


def installDependencies(prnt):
    ret = 0
    #Install all dependent libraries
    ret = exelwteBashCmd("sudo pip install pandas", prnt)
    if ret[0] == 0:
        ret = exelwteBashCmd("sudo pip install wget", prnt)

    if ret[0] == 0:
        ret = exelwteBashCmd("sudo pip install xlwt", prnt)

    if ret[0] == 0:
        ret = exelwteBashCmd("sudo pip install xlrd", prnt)

    print "InstallDependencies returning: ", ret[0]
    return ret[0]
