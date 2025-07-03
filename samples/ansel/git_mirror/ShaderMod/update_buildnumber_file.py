import io
import sys
import os

__author__ = 'hfannar'


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
    if len(sys.argv) != 3:
        print "Please specify [build-number] [hash] as the arguments to this script!"
        exit(1)

    build_number = sys.argv[1]
    short_hash = sys.argv[2][0:8]

    my_dir = os.path.dirname(os.path.realpath(__file__))
    full_path = os.path.join(my_dir, 'include/AnselBuildNumber.h')

    replacements = {'#define ANSEL_COMMIT_HASH 0x': short_hash, '#define ANSEL_BUILD_NUMBER ': build_number }

    replace_lines_in_file(full_path, replacements)


main()
