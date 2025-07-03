import io
import sys
import os

# Use this when you need to make a new change to a driver release branch that does not already have a dedicated Ansel github branch.
# For example, here are the steps you would take in order to update the build of Ansel that is in the old R421 driver branch:
#     1) Check if the dedicated Ansel release branch exists for R421: r421_release.
#           If this branch exists, skip all other steps and simply check out that branch and make your changes as normal.
#     2) Go to perforce, and look at the properties of the latest LwCamera dll submitted in R421 - lets say it has the file version: 7.0.543.0
#           NOTE - if the final number in the Ansel build version is not 0 (for example 7.0.543.654), then this branch has already been marked as a branch off of another build, and you cannot use this script to re-mark it, and thus will skip step 5 here.
#     3) Find the git SHA commit that Ansel build 543 was built from, and checkout that commit. If the final number in the file version is not 0, such as 7.0.543.654, then you will instead check out the commit for the last number, build 654.
#     4) Create branch r421_release from this commit.
#     5) Run this script with the argument '543' in order to mark all future builds as a branch off of build 543, and commit this change:
#           branch_version_buildnumber.cmd 543
#     6) In TeamCity, edit all 3 "release", "Effect compilers (release)", and "System test (release)" build configurations:
#           Under "Version Control Settings", add "+:r421_release" as a new line in "Branch Filter".
#     7) In TeamCity, edit "Other Branches" build configuration:
#           Under "Version Control Settings", add "-:r421_release" (that is with a '-' instead of a '+') as a new line in "Branch Filter".
#     8) Continue to make your changes as normal.

__author__ = 'jingham'


def replace_lines_in_file(file_path, repl):
    with io.open(file_path, 'r', newline=None) as file:
        lines = file.readlines()
    with io.open(file_path, 'w', newline=None) as file:
        lines_out = []
        for line in lines:
            for key, value in repl.iteritems():
                if line.startswith(key):
                    line = line[0:len(key)] + value + line[-1:]
            lines_out.append(line)

        file.writelines(lines_out)


def main():
    if len(sys.argv) != 2:
        print "Please specify the [build-number] that this branch is being created from as the arguments to this script!"
        exit(1)

    build_number = sys.argv[1]
    commaEnd = build_number + ', ANSEL_BUILD_NUMBER'
    dotEnd = build_number + '.ANSEL_BUILD_NUMBER)'

    my_dir = os.path.dirname(os.path.realpath(__file__))
    full_path = os.path.join(my_dir, 'ShaderMod/include/AnselVersionInfo.h')

    replacements = {'	ANSEL_VERSION_MAJOR, ANSEL_VERSION_MINOR, ': commaEnd, '	STRINGIZE(ANSEL_VERSION_MAJOR.ANSEL_VERSION_MINOR.': dotEnd }

    replace_lines_in_file(full_path, replacements)


main()
