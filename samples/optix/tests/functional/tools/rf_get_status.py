#!/usr/bin/python

# Quick and dirty script to extract the status from a test from an RF output file.
# Useful to programatically process single test results.
# Usage:
#  ./rf_get_status.py output.xml test_name0

import sys;
import operator;
import xml.etree.ElementTree as ET;
import datetime;
import argparse;

def printRelwrsive(node, offset):
    print offset, node.tag, node.attrib
    for child in node:
        if child is not None:
            printRelwrsive(child, offset+' ');

def getStatus(testName, content):
    root = ET.fromstring(content);
    #printRelwrsive(root,'')
    test_nodes = root.findall('.//test');
    if test_nodes is None:
        print 'No test node found.';
        return '';    
    for t in test_nodes:
        if t.attrib['name'] == testName:
            st = t.find('status');
            if st is None:
                print 'No status node found for this test.';
                return '';
            s = st.attrib['status'];
            return  s;

def getFileString(fileName):
    if(fileName.startswith("http://")):
        response = urllib2.urlopen(fileName);
        return response.read();
    tmp = open(fileName);
    return tmp.read();

def main():
    parser = argparse.ArgumentParser(description='Parse output.xml to get status of test');
    parser.add_argument('filename',help='output.xml file to parse');
    parser.add_argument('testname',help='Name of the test to get the status');
    args = parser.parse_args();

    string = getFileString(args.filename);
    result = getStatus(args.testname,string);
    print result;

main();
