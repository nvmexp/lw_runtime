#!python
#
# This script traverses the directory tree and removes LWNCFG blocks
# from the source files.
#
# Usage: rm_lwncfg_blocks.py <directory> <Config1> [<Config2>..<ConfigN>]
#
# directory: The root directory where the traversal starts
# Config: A configuration that is used with LWNCFG.
#
# The script will then remove any block like:
#     #if LWNCFG(Config)
#     ...
#     #endif
#
# Logical operators in the #if line are not supported and will raise an
# exception. LWNCFG expressions that need to be filtered out by this script
# should not be used with logical operators like &&, ||, !. These LWNCFG
# should also not be part of #elif expressions or a #defines.


import sys
import os
import re

from os.path import join


class BlockFilter(object):

    NoChange, EnteredLwnCfg, LeftLwnCfg = range(3)

    def __init__(self, cfg_list):

        self.cfg_list = cfg_list
        self.depth = 0
        # Regex that matches any line starting with #if LWNCFG(...) where the LWNCFG is not
        # concatenated with another expression using a logical operators like '&&' or '||'.
        self.regex_list = [re.compile("#if\s*LWNCFG\(" + cfg +"\)[^&^|]*$") for cfg in cfg_list]


    def findBlock(self, line):
        '''
        Parses the line to find the beginning or the end of a LWNCFG block.
        If we are already inside a LWNCFG block, the depth value is updated
        if other #if..#endif blocks are detected.
        The depth value indicates how many blocks we entered including the
        LWNCFG block.
        Return  values are:
        EnteredLwnCfg if a new LWNCFG block is entered
        LeftLwnCfg if the current LWNCFG block is left either due to a #else or a #endif
        NoChange otherwise
        '''

        # We only care about preprocessor statements
        if line[0] != '#':
            return self.NoChange

        if self.depth == 0:
            # Search for the beginning of a LWNCFG block. Nested LWNCFG blocks
            # are not supported. Any #if..#endif block inside a LWNCFG block will
            # treated the same way.
            for regex in self.regex_list:
                m = regex.match(line)
                if m:
                    self.depth = 1
                    return self.EnteredLwnCfg

            # check if a LWNCFG expression exists in the current line
            if any("LWNCFG(" + cfg +")" in line for cfg in self.cfg_list):
                # A LWNCFG expression was found in the current line that was not
                # recognized by the regex. This indicates an unsupported use of
                # LWNCFG like using it with a logical operator.
                raise RuntimeError("LWCFG expression at line " + str(self.line_number) + " could not be handled by script!")

        elif self.depth > 0:
            # A new #if block starts inside a LWNCFG block
            if line[1:3] == "if":
                self.depth += 1

            # A #if block ends
            if line[1:6] == "endif":
                self.depth -= 1

            # A LWNCFG block ended either by an #endif or by an #else.
            if self.depth == 0 or (self.depth == 1 and line[1:5] == "else"):
                return self.LeftLwnCfg

        else:
            # This should never happen
            raise RuntimeError("#endif with no matching #if found (Depth = " + str(self.depth) + ") at line " + str(self.line_number) + ".")

        return self.NoChange


    def removeLWNCFG(self, file):
        '''
        Parses file and remove all LWNCFG blocks that match a condition
        in self.cfg_list.
        '''

        self.depth = 0

        in_lwncfg_block = False

        with open(file, 'r+') as f:
            source = f.readlines()

            # Check if the file contains any of the LWNCFG keywords. If not it does not
            # need to be processed.
            if not any(cfg in src for cfg in self.cfg_list for src in source):
                return

            # Delete file content
            f.seek(0)
            f.truncate(0)

            self.line_number = 0

            for line in source:
                self.line_number += 1

                # Search for preprocessor statements that start or end a
                # block and update the self.depth accordingly.
                lwncfg_block_change = self.findBlock(line)

                if lwncfg_block_change == self.EnteredLwnCfg:
                    in_lwncfg_block = True
                    continue
                elif lwncfg_block_change == self.LeftLwnCfg:
                    in_lwncfg_block = False
                    continue

                if not in_lwncfg_block:
                    # Write line to output file
                    f.write(line)


# Main
if __name__ == '__main__':

    if len(sys.argv) < 3:
        print "Usage: rm_lwcfg_blocks.py <Path to root> <list of LWNCFG to use>"
        sys.exit()

    lwncfg = [ x for x in sys.argv[2:]]
    src_ext = ['c', 'cpp', 'h', 'hpp']

    bf = BlockFilter(lwncfg)

    for root, dirs, files in os.walk(sys.argv[1]):
        for f in files:
            sname = f.split('.')

            if (len(sname) == 2) and (sname[1] in src_ext):
                file_path = os.path.join(root, f)
                file_path = '/'.join(file_path.split('\\'))
                try:
                    bf.removeLWNCFG(file_path)
                except IOError:
                    print "ERROR: Failed to open file " + file_path
                except RuntimeError as error:
                    print "ERROR while processing file " + file_path + " " + str(error)

