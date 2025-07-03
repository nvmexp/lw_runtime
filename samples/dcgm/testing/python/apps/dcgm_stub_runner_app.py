import os
import string
import app_runner
import utils
import test_utils
import logger
import option_parser
import decode_logs_app
import datetime
import subprocess

class DcgmStubRunnerApp(app_runner.AppRunner):
    # Including future supported architectures
    paths = {
            "Linux_32bit": "./apps/x86/stub_library_test",
            "Linux_64bit": "./apps/amd64/stub_library_test",
            "Linux_ppc64le": "./apps/ppc64le/stub_library_test",
            "Linux_aarch64": "./apps/aarch64/stub_library_test",
            "Windows_64bit": "./apps/amd64/stub_library_test.exe"
            }
    forbidden_strings = [
            # None of this error codes should be ever printed by lw-hostengine
            "Unknown Error",
            "Uninitialized",
            "Invalid Argument",
            "(null)", # e.g. from printing %s from null ptr
            ]
    def __init__(self, args=None):
        path = DcgmStubRunnerApp.paths[utils.platform_identifier]
        self.stub = None
        self.output_filename = None
        super(DcgmStubRunnerApp, self).__init__(path, args)
        
        if not test_utils.noLogging:
            self.lwml_trace_fname = os.path.join(logger.log_dir, "app_%03d_lwml_trace.log" % (self.process_nb))
            self.elw["__LWML_DBG_FILE"] = self.lwml_trace_fname
            self.elw["__LWML_DBG_LVL"] = test_utils.loggingLevel

            self.dcgm_trace_fname = os.path.join(logger.log_dir, "app_%03d_dcgm_trace.log" % (self.process_nb))
            self.elw["__DCGM_DBG_FILE"] = self.dcgm_trace_fname
            self.elw["__DCGM_DBG_LVL"] = test_utils.loggingLevel
        else:
            self.lwml_trace_fname = None
            self.dcgm_trace_fname = None

   
    def _process_finish(self, stdout_buf, stderr_buf):
        super(DcgmStubRunnerApp, self)._process_finish(stdout_buf, stderr_buf)
       
        # Skip this part if --no-logging option is used
        if logger.log_dir is None:
            return

		# decode trace log automatically
        if self.lwml_trace_fname and os.path.exists(self.lwml_trace_fname):
            self.decode_logs = decode_logs_app.DecodeLogsApp(self.lwml_trace_fname)
            self.decode_logs.run()
        if self.dcgm_trace_fname and os.path.exists(self.dcgm_trace_fname):
            self.decode_logs = decode_logs_app.DecodeLogsApp(self.dcgm_trace_fname)
            self.decode_logs.run()
            
    
        # Verify that stub_library_test doesn't print any strings that should never be printed
        stdout = string.join(self.stdout_lines, "\n")
        for forbidden_text in DcgmStubRunnerApp.forbidden_strings:
            assert stdout.find(forbidden_text) == -1, "stub_library_test printed \"%s\", this should never happen!" % forbidden_text

    def __str__(self):
        return "stub_library_test" + super(DcgmStubRunnerApp, self).__str__()
    
    def stdout(self):
        stdout = string.join(self.stdout_lines, "\n")
        return stdout
    