#! /usr/bin/python

# Quick and dirty solution to compare RF XML outputs.
# Example:
# ./rf_diff.py http://10.21.13.168:8080/job/Goldenrod%20Tests%20Kepler/35/robot/report/output.xml http://10.21.13.168:8080/job/Goldenrod%20Tests%20Maxwell/34/robot/report/output.xml
# The script works only for full results files from test_all.

import sys;
import operator;
import urllib2;
import xml.etree.ElementTree as ET;
import datetime;
import argparse;

# timing differences of < 300ms and <5%, are ignored as noise.
tolerance = 0.300;
tolerance_percent = 0.05;

enable_timing = False;

def collectResults(node):
    results = {};
    for child in node:
        if(child.tag == "test"):
            result_set = {};
            testName = child.attrib["name"];
            result_set[0] = child[3].get("status");
            if (enable_timing):
                starttime = datetime.datetime.strptime(child[3].get("starttime"),"%Y%m%d %H:%M:%S.%f");
                endtime = datetime.datetime.strptime(child[3].get("endtime"),"%Y%m%d %H:%M:%S.%f");
                elapsed = endtime - starttime;
                result_set[1] = elapsed;
            results[testName] = result_set;
    return results;

def getResults(content):
    root = ET.fromstring(content);

    bugsNode = root[0][1];
    lwstomersNode = root[0][2];
    motionBlurNode = root[0][3];
    postprocessingNode = root[0][4];
    primeNode = root[0][5];
    sampleBinariesNode = root[0][6];
    sampleTracesNode = root[0][7];
    unitTestsNode = root[0][8];

    assert(bugsNode.attrib["name"] == 'Bugs');
    assert(lwstomersNode.attrib["name"] == 'Lwstomers');
    assert(motionBlurNode.attrib["name"] == 'Motion Blur');
    assert(postprocessingNode.attrib["name"] == 'Postprocessing');
    assert(primeNode.attrib["name"] == 'Prime');
    assert(sampleBinariesNode.attrib["name"] == 'Sample Binaries');
    assert(sampleTracesNode.attrib["name"] == 'Sample Traces');
    assert(unitTestsNode.attrib["name"] == 'Unit Tests');

    bugsResult = collectResults(bugsNode);
    lwstomersResult = collectResults(lwstomersNode);
    motionBlurResult = collectResults(motionBlurNode);
    postprocessingResult = collectResults(postprocessingNode);
    primeResult = collectResults(primeNode);
    sampleBinariesResult = collectResults(sampleBinariesNode);
    sampleTracesResult = collectResults(sampleTracesNode);
    unitTestsResult = collectResults(unitTestsNode);

    result = {};
    result.update(bugsResult);
    result.update(lwstomersResult);
    result.update(motionBlurResult);
    result.update(postprocessingResult);
    result.update(primeResult);
    result.update(sampleBinariesResult);
    result.update(sampleTracesResult);
    result.update(unitTestsResult);

    return result;

def compare(firstResult, secondResult):
    zippedList = [];
    if (len(firstResult) != len(secondResult)):
        firstKeys = set(firstResult.keys())
        secondKeys = set(secondResult.keys())
        firstDiff = firstKeys - secondKeys;
        secondDiff = secondKeys - firstKeys;
        print("Tests in first xml but not in second:");
        for x in firstDiff:
            print x;
        print("Tests in second xml but not in first:");
        for x in secondDiff:
            print x;

        intersection = firstKeys & secondKeys;

        firstResultFiltered = dict((k,firstResult[k]) for k in intersection);
        secondResultFiltered = dict((k,secondResult[k]) for k in intersection);
        zippedList = zip(firstResultFiltered.items(), secondResultFiltered.items());
    else:
        firstResult = sorted(firstResult.items(), key=operator.itemgetter(0))
        secondResult = sorted(secondResult.items(), key=operator.itemgetter(0))
        zippedList = zip(firstResult, secondResult);

    for x in zippedList:
        testName = x[0][0];
        assert(x[0][0] == x[1][0]);
        firstResult = x[0][1];
        secondResult = x[1][1];
        if(firstResult[0] != secondResult[0]):
            print("\033[1;31m%s %s %s\033[1;m" % (testName, firstResult[0], secondResult[0]));
        elif (enable_timing):
            if (firstResult[0] == "PASS"):
                diff = (secondResult[1] - firstResult[1]).total_seconds();
                if ( firstResult[1].total_seconds() > 0 ):
                   relative_change = abs(diff) / firstResult[1].total_seconds();
                else:
                   relative_change = 0;

                if (secondResult[1].total_seconds() > 0):
                   speedup = firstResult[1].total_seconds() / secondResult[1].total_seconds();
                else:
                   speedup = 0;

                if ((abs(diff) < tolerance) and (relative_change < tolerance_percent)):
                    print("%s %s %s - Timing: (%.3fs,%.3fs) %.3f sec" % (testName, firstResult[0], secondResult[0],firstResult[1].total_seconds(),secondResult[1].total_seconds(),diff));
                elif (secondResult[1] < firstResult[1]):
                    print("\033[1;32m%s %s %s - Timing: (%.3fs,%.3fs) %.3f sec, %.2fx speedup\033[1;m" % (testName, firstResult[0], secondResult[0],firstResult[1].total_seconds(),secondResult[1].total_seconds(),diff,speedup));
                else:
                    print("\033[1;33m%s %s %s - Timing: (%.3fs,%.3fs) %.3f sec, %.2fx speedup\033[1;m" % (testName, firstResult[0], secondResult[0],firstResult[1].total_seconds(),secondResult[1].total_seconds(),diff,speedup));
        else:
            print("%s %s %s" % (testName, firstResult[0], secondResult[0]));

# Example: "http://10.21.13.168:8080/job/Goldenrod%20Tests%20Kepler/35/robot/report/output.xml"

def getFileString(fileName):
    if(fileName.startswith("http://")):
        response = urllib2.urlopen(fileName);
        return response.read();
    tmp = open(fileName);
    return tmp.read();

def main():
    global enable_timing;

    parser = argparse.ArgumentParser(description='Compare output.xml from RobotFramework reports.');
    parser.add_argument('-t','--timing',dest='timing',action='store_true',default=False,help='Enable Timing comparison.');
    parser.add_argument('file1',help='First output.xml for comparison (reference for timing)');
    parser.add_argument('file2',help='Second output.xml for comparison (under test)');
    args = parser.parse_args();

    enable_timing = args.timing;
    firstFile = args.file1;
    secondFile = args.file2;

    firstString = getFileString(firstFile);
    secondString = getFileString(secondFile);

    firstResult = getResults(firstString);
    secondResult = getResults(secondString);

    compare(firstResult, secondResult);

main();
