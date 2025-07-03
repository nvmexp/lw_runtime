from collections import defaultdict

class AnselTelemetryErrorString:
    def __init__(self, eCode = 0, eAddr = 0, baseAddr = 0, stacktrace = []):
        self.eCode = eCode
        self.eAddr = eAddr
        self.baseAddr = baseAddr
        self.stacktrace = stacktrace

def _getNumber(s, left, right, base = 10):
    lloc = s.find(left)
    if lloc == -1:
        return -1, s
    s = s[lloc+1:]
    rloc = s.find(right)
    if rloc == -1:
        return -1, s
    number = int(s[:rloc], base)
    return number, s[rloc+1:]

def _parseAnselTelemetryErrorString(s):
    #s is a string of the form "eCode=..., eAddr=..., baseAddr=..., stacktrace=...,...,.......
    eCode, s = _getNumber(s, '=', ',', 16)
    eAddr, s = _getNumber(s, '=', ',', 16)
    baseAddr, s = _getNumber(s, '=', ',', 16)

    st0, s = _getNumber(s, '=', ',', 16)
    strace = [st0]
    for i in range(14):
        loc, s = _getNumber(s, '', ',', 16)
        strace.append(loc)
    loc, s = _getNumber(s, '', '"', 16)
    strace.append(loc)
    return AnselTelemetryErrorString(eCode, eAddr, baseAddr, strace)

DEFAULT_EXCEPTION_PREFIXES = [s + ": top level exception handler exelwted (" for s in ["init", "exelwtePostProcessing"]]

def _parsePrefixes(lines, prefixes):
    #returns a map from prefix -> lines matching the prefix
    linesToParse = defaultdict(list)
    for line in lines:
        for prefix in prefixes:
            location = line.find(prefix)
            if location != -1:
                linesToParse[prefix].append(line)
    return linesToParse

def parseErrorstrings(filename, prefixes = DEFAULT_EXCEPTION_PREFIXES):
    """
    Parses the errorstrings from LwCamera telemetry into a colwenient data representation
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        linesToParse = _parsePrefixes(lines, prefixes)
        errorstrings = {prefix: map(lambda s: _parseAnselTelemetryErrorString(s[s.find(prefix)+len(prefix):]), strings) for prefix, strings in linesToParse.items()}
        return errorstrings

def printFrequencies(parsed_errstrs, metrics):
    #parsed_errstrs is a list of AnselTelemetryErrorString
    #metrics is a list of pairs of (title, key), where key is a lambda taking an AnselTelemetryErrorString
    for prefix, errorstrings in parsed_errstrs.items():
        print 'PREFIX: "' + prefix +'"'
        for title, key in metrics:
            print "=" * len(title) + "\n" + title + "\n"+ "="*len(title)
            d = defaultdict(type(key(errorstrings[0])))
            for err in errorstrings:
                d[key(err)] += 1

            for k, v in sorted(d.items(), key = lambda kv: kv[1], reverse=True):
                print hex(k), v


if __name__=='__main__':
    metrics = [ ("Exception Address",(lambda e: e.eAddr - e.baseAddr)), 
                ("Stack Trace[0]", (lambda e: e.stacktrace[0] - e.baseAddr)),
                ("Error Code", (lambda e: e.eCode))
              ]
    errstrs = parseErrorstrings("omnisci-ansel-table-20181018T215325643Z.csv")
    printFrequencies(errstrs, metrics)
