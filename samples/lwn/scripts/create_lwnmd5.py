#!/bin/python

#
# Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
#

# Script for pulling latest gold image md5s from REST for Lwntest.
# Can be called from the command-line, or getMd5String can be called
# directly from another script (by inputing a list of argument strings)
#
# Example usage:
#   python create_lwnmd5.py --branch=dev_a
#
# If not --branch is specified, the script will use --tegratop location
# to find a .repo directory and determine the branch from the manifest.xml
# file.  If no --tegratop location is specified, the script will attempt
# to walk up the directory tree until a parent directory containing a
# .repo directory is found.  If the script can't automatically find that
# directory, it will error out.


import argparse
import os
import pyodbc
import sys
import xml.etree.ElementTree as xmlElemTree

# Verbosity for logMsg
IS_VERBOSE = False

# This is a list possible SQL drivers to use on the client
# to perform the pyodbc database queries.  This is only a
# couple of possible drivers, obviously not an exhaustive
# list.  If you have an installed driver not on this list,
# add it here to be able to query from REST.  This script
# will go through this list and see if it matches any of
# the installed drivers on the machine, and use the first
# one it finds.
# The ODBC Driver 13 for SQL server can be installed from
# here: https://www.microsoft.com/en-us/download/details.aspx?id=53339
POSSIBLE_DRIVERS = ["SQL Server Native Client 11.0",
                    "ODBC Driver 13 for SQL Server"]


def logMsg(msg):
    """
    Log a message to stdout if IS_VERBOSE is true.

    msg -- The message string to log to stdout.

    Returns nothing.
    """

    if IS_VERBOSE:
        print msg


def logErrorMsg(errMsg):
    """
    Log an error message to stderr

    errMsg -- The error message string to log to stderr.

    Returns nothing.
    """

    sys.stderr.write("error: %s\n" % errMsg)


def getMd5String(branch, os, gpu,
                 mode=None, verbose=False,
                 *positionals, **kwargs):
    """
    Query the MD5 from REST and returns the result string.

    branch -- The code branch to query the MD5 strings.
    os -- The operating system to query from.
    gpu -- The GPU from which to query from.
    mode -- The mode to query from.
            Possible values: None, 'fp16'
            If value is None, then standard MD5s will be
            queried.
    verbose -- Whether to log diagnostic output messages to
              stdout.

    positionals -- Any extra positional arguments; unused.
    kwargs -- Any extra dictionary arguments; unused.

    Returns md5 string of all tests if successful, otherwise None.
    """

    if os == 'hos':
        testName = "Lwntest"

        # These strings can be found in
        # http://hqlwmbrest01:9000/api/connection-string.axd
        restHost = "hqlwmbrest01"
        restDb = "MB_GL_REST"
        restPort = 9000

        if mode == "fp16":
            testName += "_FP16"
        elif mode is not None:
            logErrorMsg("Invalid 'mode' %s" % mode)
            return None
    elif os == "windows":
        testName = "Lwntest Sanity 64"

        restHost = "hqlwogl02"
        restDb = "OGL_DevA"
        restPort = 8083

        if mode is not None:
            logErrorMsg("Windows does not allow for different modes.")
            return None
    else:
        logErrorMsg("Invalid os '%s'." % os)
        return None

    query = \
        """SELECT Description, REPLACE(LOWER(PixelHash),'-','')
           FROM ImageInflectiolwiew
           INNER JOIN (
               SELECT Id = MAX(ResultInflectionId)
               FROM ImageInflectiolwiew
               WHERE Disposition='pass'
                   AND TestName='{testName}'
                   AND Gpu='{gpu}'
                   AND Os='{os}'
                   AND Branch='{branch}'
               GROUP BY Description
           ) Latest ON ResultInflectionId = Latest.Id
           ORDER BY Description""" \
        .format(testName=testName, gpu=gpu,
                os=os, branch=branch)

    if verbose:
        logMsg("Connecting to %s\n" % (restHost))
        logMsg("Exelwting query: {0}\n".format(query))

    return queryRest(query, restHost, restPort, restDb)


def queryRest(query, restHost, restPort, restDb,
              username="restuser", password="readonly"):

    """
    Performs the actual query to REST for the LWN MD5 test values.

    query -- The database query as a string
    restHost -- The server host name.
    restPort -- The REST port of the server.
    restDb -- The REST database name.
    username -- The REST username.
    password -- The REST password.

    Returns the lwnmd5 contents as a string.
    """

    # Find the list of compatible drivers installed on the client's machine
    # for the DB queries.
    installedDrivers = [driver for driver in POSSIBLE_DRIVERS if driver in
                        pyodbc.drivers()]

    if len(installedDrivers) == 0:
        logErrorMsg("No installed SQL drivers.")
        return

    # Take the first one that's installed.
    driver = installedDrivers[0]

    logMsg("Trying driver '%s'..." % driver)

    conn = "DRIVER={{{0}}};SERVER={1};PORT={2};DATABASE={3};UID={4};PWD={5}" \
        .format(driver, restHost, restPort, restDb, username, password)

    result = ""
    with pyodbc.connect(conn) as con:
        lwr = con.execute(query)
        return "\n".join(" ".join(row) for row in lwr) + "\n"


def getGitBranch(repoDir):
    """
    Queries the XML manifest from the manifest.xml in the folder repoDir.

    repoDir -- The .repo directory path.

    Returns the branch name as a string, or None if error.
    """
    # Determine the remote branch in the XML repo manifest.  We are looking to
    # extract the branch name from the "revision" field from this this XML
    # node:
    #
    #   <default remote="origin" revision="<branch name>"/>
    #

    manifestName = "manifest.xml"
    manifestPath = os.path.join(repoDir, manifestName)

    logMsg("Trying to read repo from '%s'" % manifestPath)

    tree = xmlElemTree.parse(manifestPath)

    root = tree.getroot()

    remoteBranch = root.find("default").get("revision")

    if remoteBranch is None:
        logErrorMsg("Could not determine branch from %s.  The script looks "
                    "for an XML element named 'default' with attribue "
                    "'revision' in the manifest.xml to determine branch."
                    % manifestPath)

    return remoteBranch

if __name__ == "__main__":
    # Parse command line options

    defaultGpu = "t210"
    defaultOs = "hos"

    usageMessage = """%(prog)s [options]

    This script allows developers to pull the MD5 text file from REST.
    This MD5 text file can be used in conjunction with lwntest running
    with the '-g' option.  Various options allow for specifying which
    branches, OS, mode, and other configurations from which to pull the MD5
    file.  If exelwted with NO options, then the script will default to
    GPU := '{0}', and OS := '{1}', and will attempt to determine the branch
    from a manifest.xml file in the .repo directory from the current working
    directory's ancestors. --branch can be used to specify the branch
    directly, or --tegratop can be used to specify the directory containing
    the .repo/manifest.xml file from which to query the branch.
    """ \
    .format(
        defaultGpu,
        defaultOs)

    parser = argparse.ArgumentParser(usage=usageMessage)
    parser.add_argument(
        "--gpu",
        dest='gpu',
        default="t210",
        help="Query images for this GPU (defaults to '{0}').  For Windows, "
             "please include GPU name from REST server.".format(defaultOs),
        action="store")
    parser.add_argument(
        "--os",
        dest='os',
        default="hos",
        choices=["hos", "windows"],
        help="Query images for this OS (defaults to '{0}')".format(defaultOs),
        action="store")
    parser.add_argument(
        "--mode",
        dest='mode',
        default=None,
        choices=['fp16'],
        help="Specify the test configuration of the MD5 file to obtain. "
             "Leaving this argument off will pull normal, non-fp16 mode. "
             "Note: FP16 is not available when OS == 'windows'.",
        action="store")
    parser.add_argument(
        "--branch",
        dest='branch',
        default=None,
        help="Query images for this branch.  Allows "
             "release branches, such as 'rel-hovi-hr21'.",
        action="store")
    parser.add_argument(
        "--tegratop",
        dest="tegratop",
        default=None,
        help="Specify directory where the .repo is located (used to "
             "determine Git branch name).  This is only required if the "
             "--branch argument is not set or the script can't automatically "
             "determine the .repo location by walking up the directory tree.",
        action="store")
    parser.add_argument(
        "--output_file",
        dest="output_file",
        default=None,
        help="Output to a file instead of to stdout. Filename must be "
             "specified.",
        action="store")
    parser.add_argument(
        "--verbose",
        dest="verbose",
        default=False,
        help="When enabled, will output logging information to stdout. "
             "Default is non-verbose.",
        action="store_true")

    args = parser.parse_args()

    if args.verbose:
        IS_VERBOSE = True

    if args.branch is None:
        # Need to figure out branch by reading the .repo information.
        # We do this by walking up the parent directories until we
        # reach the root of the tree (base drive) and then stop.  If
        # no .repo directory is found, we report the error and exit.
        if args.tegratop is None:
            # Move up until we find the repo dir.
            lwrrWalkDir = os.path.realpath(os.getcwd())
            repoDir = None

            while os.path.split(lwrrWalkDir)[1]:
                if os.path.isdir(os.path.join(lwrrWalkDir, '.repo')):
                    repoDir = lwrrWalkDir
                    break

                # Go to parent directory
                lwrrWalkDir = os.path.dirname(lwrrWalkDir)

            if repoDir is None:
                logErrorMsg("Can not locate .repo folder location by walking "
                            "up directory tree.  Alternatively, use "
                            "--branch or manually specify .repo location "
                            "using --tegratop.")
                exit(1)

            args.tegratop = repoDir

        # Set the branch name by querying the manifest in the .repo directory.
        args.branch = getGitBranch(os.path.join(args.tegratop, ".repo"))

    md5String = getMd5String(**vars(args))

    # Output the MD5 results.
    if md5String is not None:
        if args.output_file is not None:
            with open(args.output_file, 'w') as f:
                f.write(md5String)
        else:
            sys.stdout.write(md5String)
    else:
        logErrorMsg("Script did not run successfully. "
                    "Please check error messages.")
        exit(1)
