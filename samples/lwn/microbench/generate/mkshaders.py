#!/usr/bin/python
import sys
import os.path
import argparse

def error(p, str):
    p.print_help()
    print ("")
    print ("ERROR: "+str)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', help='Path to manifest xml')
    parser.add_argument('-o', '--output', help='Output .h file')
    parser.add_argument('filename', nargs='*')

    args = parser.parse_args()

    if len(args.filename) < 1:
        error(parser, "Must specify input filename")

    inputFile = args.filename[0]
    with open(args.output, "wt") as outf:
        shader_name = os.path.splitext(os.path.basename(args.filename[0]))
        outf_macro_name = "__" + os.path.basename(args.output).replace('.', "_") + "_"

        outf.write("#ifndef " + outf_macro_name + "\n")
        outf.write("#define " + outf_macro_name + "\n\n")
        outf.write("// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        outf.write("// !!! AUTOMATICALLY GENERATED - DO NOT EDIT !!!\n")
        outf.write("// !!! Please refer to README.txt on how to  !!!\n")
        outf.write("// !!! generate such files                   !!!\n")
        outf.write("// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
        outf.write('static const char *shader_' + shader_name[0] + " =\n")
        lines = []
        for line in open(inputFile):
            quotes_escaped = line.replace('"', '\\"').rstrip()
            lines.append("    \"{}\\n\"".format(quotes_escaped))
        outf.write("\n".join(lines) + ";\n")
        outf.write("\n#endif /* " + outf_macro_name + " */\n")

if __name__ == "__main__":
    main()
