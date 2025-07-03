import os
import string

import app_runner
import dcgm_structs
import dcgm_agent_internal
import utils
import test_utils
import logger
import option_parser
import decode_logs_app

class LwidiaSmiApp(app_runner.AppRunner):
    # TODO add option to also run just compiled lwpu-smi
    paths = {
            "Linux_32bit": "lwpu-smi", # it should be in the path
            "Linux_64bit": "lwpu-smi", # it should be in the path
            "Linux_ppc64le": "lwpu-smi", # it should be in the path
            "Linux_aarch64": "lwpu-smi", # it should be in the path
            "Windows_64bit": os.path.join(os.getelw("ProgramFiles", "C:/Program Files"), "LWPU Corporation/LWSMI/lwpu-smi.exe")
            }
    forbidden_strings = [
            # None of this error codes should be ever printed by lwpu-smi
            "Unknown Error",
            "Uninitialized",
            "Invalid Argument",
            "Already Initialized",
            "Insufficient Size",
            "Insufficient External Power",
            "Driver Not Loaded",
            "Timeout",
            "Interrupt Request Issue",
            "LWML Shared Library Not Found",
            "Function Not Found",
            "Corrupted infoROM",
            "ERR!", # from non-verbose output
            "(null)", # e.g. from printing %s from null ptr
            ]
    def __init__(self, args=None):
        path = LwidiaSmiApp.paths[utils.platform_identifier]
        self.decode_logs = None
        self.output_filename = None
        super(LwidiaSmiApp, self).__init__(path, args)
        
        if not test_utils.noLogging:
            self.trace_fname = os.path.join(logger.log_dir, "app_%03d_trace.log" % (self.process_nb))
            self.elw["__LWML_DBG_FILE"] = self.trace_fname
            self.elw["__LWML_DBG_LVL"] = test_utils.loggingLevel
        else:
            self.trace_fname = None

    def append_switch_filename(self, filename=None):
        """
        Appends [-f | --filename] switch to args.
        If filename is None than filename is generated automatically

        """
        # Skip this part if --no-logging option is used
        if logger.log_dir is None:
            return None

        if filename is None:
            filename = os.path.join(logger.log_dir, "app_%03d_filename_output.txt" % (self.process_nb))

        self.args.extend(["-f", filename])
        self.output_filename = filename

        return filename
    
    def _process_finish(self, stdout_buf, stderr_buf):
        super(LwidiaSmiApp, self)._process_finish(stdout_buf, stderr_buf)
       
        # Skip this part if --no-logging option is used
        if logger.log_dir is None:
            return
         
        # decode trace log automatically
        if self.trace_fname and os.path.exists(self.trace_fname):
            self.decode_logs = decode_logs_app.DecodeLogsApp(self.trace_fname)
            self.decode_logs.run()

        # TODO, debug builds can print to stderr.  We can check for release build here
        #assert self.stderr_lines == [], "lwpu-smi printed something to stderr. It shouldn't ever do that!"

        # Verify that lwpu smi doesn't print any strings that should never be printed on a working system
        stdout = string.join(self.stdout_lines, "\n")
        for forbidden_text in LwidiaSmiApp.forbidden_strings:
            assert stdout.find(forbidden_text) == -1, "lwpu-smi printed \"%s\", this should never happen!" % forbidden_text

    def __str__(self):
        return "lwpu-smi" + super(LwidiaSmiApp, self).__str__()

class LwidiaSmiAppPerfRun(object):
    def __init__(self, args=[], human_readable_fname=True, dvs_fname=True, iterations=5):
        self.apps = [LwidiaSmiApp(args) for i in range(iterations)]
        self.human_readable_fname = human_readable_fname
        self.dvs_fname = dvs_fname
        self.stats = None

    def run(self):
        for app in self.apps:
            app.run()

        # Skip this part if --no-logging option is used
        if logger.log_dir is None:
            return

        if test_utils.noLogging:
            return

        first_run = self.apps[0]
        last_run = self.apps[-1]
        if last_run.decode_logs is None:
            return
        
        last_stats = last_run.decode_logs.stats
        for i in range(len(self.apps) - 1):
            if self.apps[i].decode_logs is not None:
                last_stats.combine_stat(self.apps[i].decode_logs.stats)

        if self.human_readable_fname is True:
            self.human_readable_fname = os.path.join(logger.log_dir, "app_%03d-%03d.avg_stats.txt") % (first_run.process_nb, last_run.process_nb)
        if self.human_readable_fname:
            last_stats.write_to_file(self.human_readable_fname)

        if self.dvs_fname is True:
            self.dvs_fname = os.path.join(logger.log_dir, "app_%03d-%03d.avg_dvs_stats.txt") % (first_run.process_nb, last_run.process_nb)
        if self.dvs_fname:
            last_stats.write_to_file_dvs(self.dvs_fname)

        self.stats = last_stats
