import json
import os
import GcovAggregator
import argparse

class GcovFile(object):
    def __init__(self, path):
        self.m_path = path
        self.m_exelwtedLines = 0
        self.m_exelwtableLines = 0
        self.m_totalExelwtions = 0
        self.m_nonExelwtableLines = 0

    def CountFileLines(self):
        with open(self.m_path, 'r') as f:
            for count, line in enumerate(f):
                separated = line.split(':')
                execCount = GcovAggregator.get_gcov_count(separated[0].strip())
                if execCount < 0:
                    self.m_nonExelwtableLines += 1
                else:
                    self.m_exelwtableLines += 1
                    if execCount > 0:
                        self.m_exelwtedLines += 1
                        self.m_totalExelwtions += execCount

    def GetExelwtableLines(self):
        return self.m_exelwtableLines

    def GetExelwtedLines(self):
        return self.m_exelwtedLines

    def GetNonExelwtedLines(self):
        return self.m_exelwtableLines - self.m_exelwtedLines

    def GetTotalExelwtions(self):
        return self.m_totalExelwtions

    def GetReportString(self, csv_mode):
        if not self.m_exelwtableLines:
            return

        pcnt = 100 * float(self.m_exelwtedLines) / float(self.m_exelwtableLines)
        avgExec = float(self.m_totalExelwtions) / float(self.m_exelwtableLines)

        if not csv_mode:
            return "\t%s : %d of %d lines exelwted for %.2f%%. Average exelwtions per line = %.2f" %\
                    (self.m_path, self.m_exelwtedLines, self.m_exelwtableLines, pcnt, avgExec)
        else:
            return "%s,%d,%d,%.2f,%.2f" % (self.m_path, self.m_exelwtedLines, self.m_exelwtableLines, pcnt, avgExec)

    def PrintReport(self, csv_mode):
        print(self.GetReportString(csv_mode))

class GcovAnalyzer(object):
    def __init__(self, gcovDir):
        self.m_gcovFiles = {}
        self.m_gcovDir = gcovDir
        self.m_totalLines = 0
        self.m_totalExelwted = 0
        self.m_totalExelwtions = 0
        self.m_totalFiles = 0

    def ParseGcovFiles(self):
        files = GcovAggregator.find_matching_files_in_path(self.m_gcovDir, '.gcov')
        self.m_totalFiles = 0
        for gcovFile in files:
            self.m_gcovFiles[gcovFile] = GcovFile(files[gcovFile])
            self.m_gcovFiles[gcovFile].CountFileLines()
            self.m_totalLines += self.m_gcovFiles[gcovFile].GetExelwtableLines()
            self.m_totalExelwted += self.m_gcovFiles[gcovFile].GetExelwtedLines()
            self.m_totalExelwtions += self.m_gcovFiles[gcovFile].GetTotalExelwtions()
            self.m_totalFiles += 1

    def PrintCoverageReport(self, csv_mode):
        if not csv_mode:
            print("DCGM Coverage Report\n")
        else:
            print("Filename,Exelwted Lines,Exelwtable Lines, Pcnt Exelwted, Avg Exelwtions Per Line")

        info = {}
        for gcovFile in self.m_gcovFiles:
            info[self.m_gcovFiles[gcovFile].GetNonExelwtedLines()] = self.m_gcovFiles[gcovFile].GetReportString(csv_mode)

        for i in sorted(info.keys()):
            print(info[i])

        pcnt = 100 * float(self.m_totalExelwted) / float(self.m_totalLines)
        avgExec = 100 * float(self.m_totalExelwtions) / float(self.m_totalLines)
        if not csv_mode:
            print("\nTotal Project : %d of %d lines exelwted for %.2f%% in %d files. \n" %\
                    (self.m_totalExelwted, self.m_totalLines, pcnt, self.m_totalFiles))
        else:
            print ("Summary,%d,%d,%.2f,%.2f" % (self.m_totalExelwted, self.m_totalLines, pcnt, avgExec))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv-mode', dest='csv_mode', action="store_true", default=False,
                        help='Prints comma separated values instead of human-readable output.')
    parser.add_argument('-d', '--dir', dest='gcov_dir', required=True, type=str,
                        help='Tells the analyzer what directory to find the .gcov files in.')
    args = parser.parse_args()
    analyzer = GcovAnalyzer(args.gcov_dir)
    analyzer.ParseGcovFiles()
    analyzer.PrintCoverageReport(args.csv_mode)


if __name__ == '__main__':
    main()
