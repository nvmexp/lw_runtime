import os
import string
import app_runner
import utils
import test_utils
import logger
import option_parser
import decode_logs_app

class DcgmiApp(app_runner.AppRunner):
    # Including future supported architectures
    paths = {
            "Linux_32bit": "./apps/x86/dcgmi",
            "Linux_64bit": "./apps/amd64/dcgmi",
            "Linux_ppc64le": "./apps/ppc64le/dcgmi",
            "Linux_aarch64": "./apps/aarch64/dcgmi",
            "Windows_64bit": "./apps/amd64/dcgmi.exe"
            }
    forbidden_strings = [
            # None of this error codes should be ever printed by lwcmi
            "Unknown Error",
            "Uninitialized",
            "Invalid Argument",
            "Already Initialized",
            "Insufficient Size",
            "Driver Not Loaded",
            "Timeout",
            "DCGM Shared Library Not Found",
            "Function Not Found",
            "(null)", # e.g. from printing %s from null ptr
            ]
    def __init__(self, args=None):
        path = DcgmiApp.paths[utils.platform_identifier]
        self.dcgmi = None
        self.output_filename = None
        super(DcgmiApp, self).__init__(path, args)
        
        if not test_utils.noLogging:
            self.trace_fname = os.path.join(logger.log_dir, "app_%03d_dcgm_trace.log" % (self.process_nb))
            self.elw["__DCGM_DBG_FILE"] = self.trace_fname
            self.elw["__DCGM_DBG_LVL"] = test_utils.loggingLevel
        else:
            self.trace_fname = None
   
    def _process_finish(self, stdout_buf, stderr_buf):
        super(DcgmiApp, self)._process_finish(stdout_buf, stderr_buf)
       
        # Skip this part if --no-logging option is used
        if logger.log_dir is None:
            return

        # decode trace log automatically
        if self.trace_fname and os.path.exists(self.trace_fname):
            self.decode_logs = decode_logs_app.DecodeLogsApp(self.trace_fname)
            self.decode_logs.run()         

        # Verify that lw_hostengine doesn't print any strings that should never be printed on a working system
        stdout = string.join(self.stdout_lines, "\n")
        for forbidden_text in DcgmiApp.forbidden_strings:
            assert stdout.find(forbidden_text) == -1, "dcgmi printed \"%s\", this should never happen!" % forbidden_text

    def __str__(self):
        return "dcgmi" + super(DcgmiApp, self).__str__()
