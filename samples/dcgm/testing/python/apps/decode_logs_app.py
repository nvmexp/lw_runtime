import app_runner
import performance_stats
import option_parser
import os
import utils
import re
import utils
import logger
from contextlib import nested

class DecodeLogsApp(app_runner.AppRunner):
    paths = {
            "Linux_32bit": "./apps/x86/decode_logs",
            "Linux_64bit": "./apps/amd64/decode_logs",
            "Linux_ppc64le": "./apps/ppc64le/decode_logs",
            "Linux_aarch64": "./apps/aarch64/decode_logs",
            "Windows_64bit": "./apps/amd64/decode_logs.exe"
            }
    def __init__(self, input_fname, out_fname=True, patched_fname=True, stats_fname=False, dvs_stats_fname=True, delete_input_fname=True):
        """
        Decodes trace log to out_fname file (if true picks predefined name)
        .patched_fname Can also use turn release into debug logs when patched_fname is not None (if true picks predefined output name)
        .stats_fname   Can also callwlate perf stats of the file (if true picks predefined output name)
        """
        self.input_fname = input_fname
        self.delete_input_fname = delete_input_fname
        self.out_fname = out_fname
        if self.out_fname is True:
            self.out_fname = self.input_fname + ".decoded.txt"

        self.patched_fname = patched_fname
        if self.patched_fname is True:
            self.patched_fname = self.input_fname + ".patched.txt"
        if option_parser.options.no_dcgm_trace_patching:
            # Code patching disabled by a user switch
            self.patched_fname = None

        self.stats_fname = stats_fname
        if self.stats_fname is True:
            self.stats_fname = self.input_fname + ".stats.txt"

        self.dvs_stats_fname = dvs_stats_fname
        if self.dvs_stats_fname is True:
            self.dvs_stats_fname = self.input_fname + ".dvs_stats.txt"

        path = os.path.join(utils.script_dir, DecodeLogsApp.paths[utils.platform_identifier])
        super(DecodeLogsApp, self).__init__(path, ["-i", self.input_fname, "-o", self.out_fname])
            
    def _process_finish(self, stdout_buf, stderr_buf):
        super(DecodeLogsApp, self)._process_finish(stdout_buf, stderr_buf)
       
        if self.patched_fname:
            patch_decoded_release_trace_log(self.out_fname, self.patched_fname)

        if self.stats_fname or self.dvs_stats_fname:
            src = self.patched_fname
            if not src:
                src = self.out_fname # when patching disabled use original file to generate stats

            self.stats = performance_stats.PerformanceStats(src)
            if self.stats_fname:
                self.stats.write_to_file(self.stats_fname)
            if self.dvs_stats_fname:
                self.stats.write_to_file_dvs(self.dvs_stats_fname)

        # we don't keep the encrypted file
        if self.delete_input_fname:
            os.remove(self.input_fname)

        # if we have unpatched and patched we keep the patched only, since it has dcgm source substituted in place of rm calls
        if self.patched_fname and self.out_fname:
            os.remove(self.out_fname)
        

    def __str__(self):
        return "DecodeLogsApp " + super(DecodeLogsApp, self).__str__()

@utils.cache()
def load_src_database(fname):
    result = dict()
    line_regex = re.compile("""^([^-:]+)[-:](\d+)[-:](.*)""")
    with open(fname) as f:
        for line in f:
            line = line[:-1] # crop \n
            if line == "--":
                continue
            match = line_regex.match(line);
            file_name = match.group(1)
            file_line_nb = int(match.group(2))
            contents = match.group(3)
            result.setdefault(file_name, dict())[file_line_nb] = contents

    return result

default_decode_db_fname = os.path.join(utils.script_dir, "data/dcgm_decode_db.txt")
def patch_decoded_release_trace_log(fname_in, fname_out, decode_db_fname=default_decode_db_fname):
    """
    Loads decode_db file and maps release log print statements to dcgm source code
    and tries to regenerate the log as if it was created from debug dcgm

    """
    decode_db = load_src_database(decode_db_fname)
    (TYPE_RM_CALL, TYPE_PRINT, TYPE_ENTRY_CALL) = range(3)

    # regex that matches
    # DEBUG:	[tid 7918]	[0.006449s - dmal/rm/rm_compute.c:(fn name in debug):15]	a55a0010 20800131 ## 0
    # and groups into
    # (1)            (2)     (3)         (4)                  (5)                (6)    (7)
    regex_dcgm_trace = re.compile("""^([A-Z]+):\t\[tid (\d+)\]\t\[(\d+\.\d+)s - ([^\]]+):?(\w+)?:([0-9]+)\]\t(.*)""")
    # Matches RM call in source file
    regex_src_rmcall = re.compile(""".*dcgmRmCall\(.+, *(.+), *(LW[A-F0-9]{4}_CTRL_CMD_[A-Z0-9_]+)""")
    # Matches PRINT_ call in source file 
    regex_src_print  = re.compile(""" *PRINT_[A-Z]+\("(.*?)", *"(.*?)" *[,)]""")
    # Matches RM call in release trace log
    regex_trace_release_rmcall = re.compile("^([0-9a-f]+) ([0-9a-f]+)( ## 0?x?([0-9a-f]+))?$")
    # Matches all number formatting strings like %u %llU %08X etc.
    regex_fmt_to_pct_s = re.compile("%-?0?[0-9]?l?l?[xXuUpd]")

    # Generates a regex expression that can extract information back from string generated with format string
    @utils.cache()
    def fmt_to_regex(fmt):
        fmt = re.sub("\(", "\(", fmt)
        fmt = re.sub("\)", "\)", fmt)
        fmt = re.sub("%-?0?[0-9]?l?l?[ud]", "(-?\d+)", fmt)
        fmt = re.sub("%-?0?[0-9]?l?l?[xXU]", "([a-fA-F0-9]+)", fmt)
        fmt = re.sub("%p", "(0x[a-fA-F0-9]+)", fmt)
        fmt = re.sub("%-?[0-9]?s", "(.*)", fmt)
        return re.compile(fmt)

    with nested(open(fname_out, "w"), open(fname_in)) as (fout, fin):
        try:
            for line in fin:
                line = line.strip()
                match = regex_dcgm_trace.match(line)
                if not match:
                    # not a kind of line that this parser can handle. Leave as is
                    fout.write("%s\n" % (line))
                    continue

                match_level  = match.group(1)
                match_tid    = match.group(2)
                match_tstamp = match.group(3)
                match_fname  = match.group(4)
                match_fn     = match.group(5)
                match_fnb    = int(match.group(6))
                match_msg    = match.group(7)
                match_msg_new = None

                # encompass with loop so that we could easily skip to
                # routine that handles writing to file by just calling break
                while True:
                    if match_fn is not None:
                        break # it contains function name it's already debug
                    
                    # Match src line
                    # 
                    # match_fnb points to the source line that printed the message.
                    # If the call is multi-line then match_fnb points to the last line and we need to search
                    # back to find the first line of the call.
                    try:
                        src_type = None
                        for i in xrange(4):
                            src_line = decode_db[match_fname][match_fnb - i]
                            if src_line.find("PRINT_" + match_level + "(") != -1:
                                # look at 2 lines since dbg_fmt might be in the second line
                                src_line += decode_db[match_fname][match_fnb - i + 1]
                                src_type = TYPE_PRINT
                                break
                            if regex_src_rmcall.match(src_line):
                                src_type = TYPE_RM_CALL
                                break
                            # TODO TYPE_ENTRY_CALL
                    except KeyError:
                        # missing src file
                        break

                    if src_type == TYPE_RM_CALL:
                        # Get information about rm call from src file
                        src_rmcall_match = regex_src_rmcall.match(src_line)
                        (objtype, call_name) = src_rmcall_match.groups()
                        (handle1, handle2, tmp, retcode) = regex_trace_release_rmcall.match(match_msg).groups()
                        if retcode is None:
                            retcode = ""
                        else:
                            retcode = " returned 0x%s" % retcode
                        match_msg_new =  "dcgmRmCall(%s %s, %s, ...)%s" % (objtype, handle1, call_name, retcode)
                    elif src_type == TYPE_PRINT:
                        # Get information about dbg_fmt from src file
                        src_print_match = regex_src_print.match(src_line)
                        if src_print_match is None:
                            break
                        (rel_fmt, dbg_fmt) = src_print_match.groups()
                        dbg_fmt = regex_fmt_to_pct_s.sub("%s", dbg_fmt)
                        parsed = fmt_to_regex(rel_fmt).match(match_msg).groups()
                        match_msg_new = dbg_fmt % parsed
                        match_msg_new = match_msg_new.replace("\\t", "\t")
                    break
                if match_msg_new is None:
                    fout.write("%s\n" % (line))
                else:
                    fout.write("%s:\t[tid %s]\t[%ss - %s:%d]\t%s\n" % 
                                (match_level, match_tid, match_tstamp, match_fname, match_fnb, match_msg_new))
        except Exception, e:
            # TODO add automatic check and warn before failing
            logger.error("Failed to apply patching to LWML trace log. Please make sure that the data/version.txt matches driver version of source driver")
            raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print "Usage <decode_db> <input decoded trace log> <output decoded trace log>"
        print "Provided args %s" % (sys.argv)
        sys.exit(1)
    (ignore, decode_db_fname, fname_in, fname_out) = sys.argv # pylint: disable=unbalanced-tuple-unpacking
    patch_decoded_release_trace_log(fname_in, fname_out, decode_db_fname)
    print "Log %s patched and written into %s" % (fname_in, fname_out)
