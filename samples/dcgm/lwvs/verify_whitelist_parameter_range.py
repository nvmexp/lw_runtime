import re
import sys

def VerifyWhitelistParameterRanges(whitelistFile):
    pattern = r"(?P<value>[\d\.]+), (?P<min>[\d\.]+), (?P<max>[\d\.]+)"
    f = open(whitelistFile)
    lines = f.readlines()

    errorCount = 0
    print("Verifying parameter ranges in whitelist file...")
    for i, line in enumerate(lines):
        match = re.search(pattern, line)
        if match:
            val = float(match.group('value'))
            min_val = float(match.group('min'))
            max_val = float(match.group('max'))
            if val < min_val or val > max_val:
                errorCount += 1
                print("Line %s: invalid range or value: %s" % (i+1, line.rstrip()))
    
    if errorCount:
        print("Errors found. Please fix errors before committing.")
        sys.exit(1)
    print("Success!")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Script called with args: %s" % (sys.argv[1:]))
        print("Invalid arguments. Script should be called with path to whitelist file only.")
        sys.exit(1)
    VerifyWhitelistParameterRanges(sys.argv[1])
