import csv
import argparse
import re

class ParseDcgmProftesterSingleMetric:
    "class for parsing a single metric"

    def __init__(self):
        self.data_lines_lines = ['PcieTxBytes', 'PcieRxBytes', 'GrActivity:', 'SmActivity', \
                                 'SmActivity:', 'SmOclwpancy', 'SmOclwpancy:', \
                                 'TensorEngineUtil', 'DramUtil', 'Fp64EngineUtil', \
                                 'Fp32EngineUtil', 'Fp16EngineUtil']
        self.d_cpu_count = {}

    def getDataString(self, metric):
        if metric == '1001':
            return 'GrActivity'
        elif metric == '1002':
            return 'SmActivity'
        elif metric == '1003':
            return 'SmOclwpancy'
        elif metric == '1009':
            return 'PcieTxBytes'
        elif metric == '1010':
            return 'PcieRxBytes'
        else:
            return 'None'

    def getFirstIndex(self, metric):
        if metric == '1010' or metric == '1009':
            return 11
        elif metric == '1001':
            return 8
        elif metric == '1002':
            return 9
        elif metric == '1003':
            return 9

    def parseAndWriteToCsv(self, fName, metric, gpu_index):
        csvFileName = 'dcgmProfTester' + '_' + str(metric) + '_gpu'+ str(gpu_index) + '.csv'
        sample_num = 0
        f = open(fName, 'r+')
        lines = f.readlines() # read all lines at once
        metric_label = self.getDataString(metric)

        with open(csvFileName, 'wb') as csvFile:
            fieldnames = ['Sample Number', metric_label]
            writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
            writer.writeheader()

            i = 0
            pattern = re.compile('\d+(\.\d+)?')
            while i < len(lines):
                try:
                    line = lines[i]
                    row = line.split()
                    if row == [] or row[0] not in self.data_lines_lines:
                        print "Skipping non data row: " + str(row)
                    elif pattern.match(row[1]):
                        #print ("Row[1]", row[1] + ' i[' + str(i) +']' )
                        sample_num = sample_num + 1
                        val = float(lines[i].split()[1])
                        dict_row = {'Sample Number': sample_num, metric_label: val}
                        writer.writerow(dict_row)
                    i = i + 1
                except IndexError:
                    print "Excepting non data row: " + str(row)
                    i = i+1
                    pass
                except StopIteration:
                    pass
        print "Outside loop"

def main(cmdArgs):
    fName = cmdArgs.fileName
    metric = cmdArgs.metric
    gpu_index = cmdArgs.gpu_index
    #parse and output the data
    po = ParseDcgmProftesterSingleMetric()
    po.parseAndWriteToCsv(fName, metric, gpu_index)


def parseCommandLine():
    parser = argparse.ArgumentParser(description="Parse logs from dcgmLogs into a csv")
    parser.add_argument("-f", "--fileName", required=True, help="fielName of the \
                        file to be parsed and outputted to csv")
    parser.add_argument("-m", "--metric", required=True, help="metric for which the data is being \
                        analyzed")
    parser.add_argument("-i", "--gpu_index", required=False, default='0', help="metric for which \
                        the data is being analyzed")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    cmdArgs = parseCommandLine()
    main(cmdArgs)
