import argparse
import util
#Copy the dcgm data in csv file

def main(cmdArgs):
    metrics = cmdArgs.metrics
    time = cmdArgs.time
    gpuid_list = cmdArgs.gpuid_list
    download_bin = cmdArgs.download_bin

    ret = util.removeDependencies(True)
    if ret == 0:
        ret = util.installDependencies(True)

    if ret == 0:
        if download_bin:
            cmd = 'python run_validate_dcgm.py -m {0} -t {1} -d -i {2}'\
                    .format(metrics, time, gpuid_list)
        else:
            cmd = 'python run_validate_dcgm.py -m {0} -t {1} -i {2}'\
                    .format(metrics, time, gpuid_list)
        ret = util.exelwteBashCmd(cmd, True)

    print "\nTests are done, removing dependencies"

    ret = util.removeDependencies(False)

    print "\n All Done"

def parseCommandLine():
    parser = argparse.ArgumentParser(description="Validation of dcgm metrics")
    parser.add_argument("-m", "--metrics", required=True, help="Metrics to be validated \
            E.g. \"1009\", etc")
    parser.add_argument("-i", "--gpuid_list", required=False, default='0', \
            help="comma separated gpu id list starting from 0, eg \"0,1,2\"")
    parser.add_argument("-t", "--time", required=True, help="time in seconds")
    parser.add_argument("-d", "--download_bin", action='store_true', required=False, default=False,\
            help="If specified, download new binaries")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # Parsing command line options
    cmdArgs = parseCommandLine()

    main(cmdArgs)
