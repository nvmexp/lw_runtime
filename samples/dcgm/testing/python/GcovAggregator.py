import os
import shutil
import json
import subprocess
import logger

global g_gcovAggregator
g_gcovAggregator = None

filesToOmit = [ 'stl_map.h',
            'postypes.h',
            'sstream',
            'vector.tcc',
            'stlheap.h',
            'ctype_inline.h',
            'stl_pair.h',
            'stl_multimap.h',
            'fstream.tcc',
            'stl_list.h',
            'stl_iterator_base_types.h',
            'stl_iterator.h',
            'stl_iterator_base_funcs.h',
            'stl_function.h',
            'unordered_map',
            'basic_ios.h',
            'basic_string.h',
            'stl_stack.h',
            'iosfwd',
            'stream_iterator.h',
            'fstream',
            'stl_set.h',
            'ios_base.h',
            'stl_construct.h',
            'ostream',
            'new',
            'exceptions.h',
            'stl_algobase.h',
            'deque.tcc',
            'stl_algo.h',
            'stdlib.h',
            'istream',
            'stl_deque.h',
            'stl_queue.h',
            'istream.tcc',
            'list.tcc',
            'iostream',
            'ostream.tcc',
            'streambuf',
            'sstream.tcc',
            'cmath',
            ]

################################################################################
### Find all files with the specified extension in the path
################################################################################
def find_matching_files_in_path(path, extension):
    matching_files = {}
    extLen = -1 * len(extension)
    for r, d, files in os.walk(path):
        for filename in files:
            # Check for filenames that are too short to match the extension
            if len(filename) + extLen < 0: # Add extLen because it's negative
                continue

            if filename[extLen:] == extension:
                matching_files[filename] = os.path.join(r, filename)
        break # do not look relwrsively

    return matching_files

################################################################################
### Turns the string in the .gcov file into a count. '-' means does not exist, 
### which we will represent with -1, '#####' means 0, and other than that it
### should be an int
################################################################################
def get_gcov_count(val):
    try:
        if val == '-':
            return -1
        elif val == '#####':
            return 0
        elif val == '=====':
            return 0
        return int(val)
    except ValueError as e:
        return 0

################################################################################
### Translate the count into a length-formatted string for a .gcov file.
### Basically, the numbers should fit into a 7 character string, with leading
### spaces for padding. O is specified with '#####' and does not exist is '-'.
################################################################################
def make_gcov_count(count):
    if count < 0:
        return '      -'
    elif count == 0:
        return '  #####'
    else:
        count_str = str(count)
        pad_len = 7 - len(count_str)
        final = ''
        for i in range(pad_len):
            final += ' '
        return "%s%s" % (final, count_str)


class GcovAggregator(object):
    ################################################################################
    ### 
    ################################################################################
    def __init__(self, work_dir, gcov_dir=None):
        if work_dir[0] == '/':
            self.m_workDir = work_dir
        else:
            self.m_workDir = "%s/%s" % (os.getcwd(), work_dir)

        # If no gcov dir is specified, default to making a dir called gcov inside the working directory
        if gcov_dir:
            if gcov_dir[0] == '/':
                self.m_gcovDir = gcov_dir
            else:
                self.m_gcovDir = "%s/%s" % (os.getcwd(), gcov_dir)
        else:
            self.m_gcovDir = "%s/%s" % (self.m_workDir, "gcovDir")

        if not os.path.isdir(self.m_workDir):
            raise Exception("Working directory %s does not exist!")

        self.m_gcovBinary = "%s/%s" % (self.m_workDir, "gcov")

        # gcov binary must be present in the work dir 
        if not os.path.isfile(self.m_gcovBinary):
            raise Exception("Couldn't find gcov binary in the working directory. %s not found" % self.m_gcovBinary)

        # gcov binary must also be exelwtable
        if not os.access(self.m_gcovBinary, os.X_OK):
            raise Exception("gcov binary %s is not exelwtable!" % self.m_gcovBinary)
        
        # If the gcov directory doesn't exist, attempt to create it
        self.m_dstGcovFiles = {}
        if not os.path.isdir(self.m_gcovDir):
            os.mkdir(self.m_gcovDir)
        else:
            self.m_dstGcovFiles = find_matching_files_in_path(self.m_gcovDir, '.gcov')

        # Load a list of the DCGM files we care about so we know where the file we're looking for is
        with open('dcgm_files.json', 'r') as f:
            self.m_dcgmFiles = json.load(f)

        os.elwiron['GCOV_PREFIX'] = self.m_workDir
        os.elwiron['GCOV_PREFIX_STRIP'] = '100'

        # Since we're in the _out subdir testing, the .gcno files should be 1 directory
        # above us.
        self.CopyGcnoFiles('..')

        self.m_exclusions = filesToOmit

    ################################################################################
    ###  Copies the .gcno files from the specified directory to the working directory
    ################################################################################
    def CopyGcnoFiles(self, fromdir):
        gcnos = find_matching_files_in_path(fromdir, '.gcno')
        for filename in gcnos:
            dst_file = "%s/%s" % (self.m_workDir, filename)
            shutil.copyfile(gcnos[filename], dst_file)

    ################################################################################
    ### Removes all the .gcda files from the working directory so we are clean for
    ### the next run
    ################################################################################
    def CleanFromRun(self):
        # Remove all the .gcda files from the working directory
        self.ExelwteCommand("rm -f %s/*.gcda" % self.m_workDir)
        self.ExelwteCommand("rm -f %s/*.gcov" % self.m_workDir)

    ################################################################################
    ### Exelwtes the command with arguments specified in arg_list
    ################################################################################
    def ExelwteCommand(self, cmd):
        runner = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # Right now we are ignoring the output, but we will capture it so it isn't printed to the
        # console
        (output, error) = runner.communicate()
        if runner.returncode:
           logger.warning("Command %s exited with non-zero exit code %d. Stderr: %s" % (cmd, runner.returncode, error))

    ################################################################################
    ### Compares two lines from two different .gcov files. If they are the same line
    ### number, then it gets the total count and returns 0 and a line with the
    ### combined information.
    ### .gcov lines are in the format <times exelwted>:<line number>:<line text>
    ### If the lines do not agree, then 1, None is returned if line 2 is lower and
    ### -1, None is returned if line 1 is lower
    ################################################################################
    def CombineCoverageLines(self, line1, line2):
        # split a maximum of twice to preserve all text in the final position
        one = line1.split(':', 2)
        two = line2.split(':', 2)

        linenum1 = int(one[1].strip())
        linenum2 = int(two[1].strip())
        if linenum1 == linenum2:
            # These refer to the same line
            if linenum1 == 0:
                # We only need to process at index 1 and above. Nothing to do
                return 0, line1

            # Combine the lines
            count = get_gcov_count(one[0].strip()) + get_gcov_count(two[0].strip())
            one[0] = make_gcov_count(count)
            line = "%s:%s:%s" % (one[0], one[1], one[2])
                
            return 0, line
        elif linenum1 > linenum2:
            return 1, None
        else:
            return -1, None

    ################################################################################
    ### Combines the line counts from two gcov files and overwrites the destination
    ### file with the result
    ################################################################################
    def CombineGcovFilesAndWrite(self, src_file, dst_file):
        with open(src_file, 'r') as f:
            src_lines = f.readlines()
        with open(dst_file, 'r') as f:
            dst_lines = f.readlines()

        srcI = 0
        dstI = 0
        lines = []

        # Iterate over each line in the two files
        while srcI < len(src_lines) and dstI < len(dst_lines):
            res, line = self.CombineCoverageLines(src_lines[srcI], dst_lines[dstI])
            if res == 1:
                dstI += 1
            elif res == -1:
                srcI += 1
            else:
                srcI += 1
                dstI += 1
                lines.append(line)

        # Write the new coverage file
        with open(dst_file, 'w') as f:
            for line in lines:
                if line[-1] != '\n':
                    f.write("%s\n" % line)
                else:       
                    f.write(line)
    
    ################################################################################
    ### Finds all of the .gcov files in the work dir and either copies them to the 
    ### total coverage directory or combines the information from the two into
    ### the file in the total coverage directory
    ################################################################################
    def CombineCoverageFiles(self):
        src_files = find_matching_files_in_path(self.m_workDir, '.gcov')

        for filename in src_files:
            # Do not process excluded files; files are excluded because they're part of the STL.
            if filename in self.m_exclusions:
                continue

            if filename not in self.m_dstGcovFiles:
                dst_file = "%s/%s" % (self.m_gcovDir, filename)
                shutil.copyfile(src_files[filename], dst_file)
                self.m_dstGcovFiles[filename] = dst_file
            else:
                self.CombineGcovFilesAndWrite(src_files[filename], self.m_dstGcovFiles[filename])

    ################################################################################
    ### Determines which DCGM file this is a .gcda for, and returns None if no 
    ### path can be found
    ################################################################################
    def GetDcgmPath(self, filename):
        sansExtension = filename[:-5]
        dotCpp = "%s.cpp" % (sansExtension)
        if dotCpp in self.m_dcgmFiles:
            return self.m_dcgmFiles[dotCpp]

        dotC = "%s.c" % (sansExtension)
        if dotC in self.m_dcgmFiles:
            return self.m_dcgmFiles[dotC]

        dotH = "%s.h" % (sansExtension)
        if dotH in self.m_dcgmFiles:
            return self.m_dcgmFiles[dotH]

        dotHpp = "%s.hpp" % (sansExtension)
        if dotHpp in self.m_dcgmFiles:
            return self.m_dcgmFiles[dotHpp]

        return None

    ################################################################################
    ### Update the coverage files based on one run of a process. This is the entry
    ### point into generating and aggregating coverage files from main.py which is
    ### the integration tests.
    ################################################################################
    def UpdateCoverage(self):
        gcdas = find_matching_files_in_path(self.m_workDir, '.gcda')
        cwd = os.getcwd()
        os.chdir(self.m_workDir)
        for filename in gcdas:
            path = self.GetDcgmPath(filename)
            if not path:
                continue
   
            pos = path.find('lwvs/plugin_src')
            if pos < 3 and pos >= 0:
                slashPos = path.rfind('/')
                self.ExelwteCommand('./gcovLwvs -s %s -o . %s' % (path[:slashPos], path[slashPos+1:]))
            else:
                self.ExelwteCommand('./gcov %s' % (path))

        self.CombineCoverageFiles()

        self.CleanFromRun()

        os.chdir(cwd)
