import os
import string
import app_runner
import utils
import test_utils
import logger
import option_parser
import decode_logs_app

class TestDcgmUnittestsApp(app_runner.AppRunner):
    # Including future supported architectures
    paths = {
            "Linux_32bit": "./apps/x86/testdcgmunittests",
            "Linux_64bit": "./apps/amd64/testdcgmunittests",
            "Linux_ppc64le": "./apps/ppc64le/testdcgmunittests",
            "Linux_aarch64": "./apps/aarch64/testdcgmunittests",
            "Windows_64bit": "./apps/amd64/testdcgmunittests.exe"
            }
    forbidden_strings = [
            # None of this error codes should be ever printed by testlwcmunittests
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
        path = TestDcgmUnittestsApp.paths[utils.platform_identifier]
        self.lw_hostengine = None
        self.output_filename = None
        super(TestDcgmUnittestsApp, self).__init__(path, args)
        
        if not test_utils.noLogging:
            self.trace_fname = os.path.join(logger.log_dir, "app_%03d_trace.log" % (self.process_nb))
            self.elw["__DCGM_DBG_FILE"] = self.trace_fname
            self.elw["__DCGM_DBG_LVL"] = test_utils.loggingLevel
        else:
            self.trace_fname = None
   
    def _process_finish(self, stdout_buf, stderr_buf):
        super(TestDcgmUnittestsApp, self)._process_finish(stdout_buf, stderr_buf)
       
        # Skip this part if --no-logging option is used
        if logger.log_dir is None:
            return

        # decode trace log automatically
        if self.trace_fname and os.path.exists(self.trace_fname):
            self.decode_logs = decode_logs_app.DecodeLogsApp(self.trace_fname)
            self.decode_logs.run()         

        # Verify that lw_hostengine doesn't print any strings that should never be printed on a working system
        stdout = string.join(self.stdout_lines, "\n")
        for forbidden_text in TestDcgmUnittestsApp.forbidden_strings:
            assert stdout.find(forbidden_text) == -1, "testdcgmunittests printed \"%s\", this should never happen!" % forbidden_text

        if self.retvalue() != 0:
            stderr = string.join(self.stderr_lines, "\n")
            logger.warning("testdcgmunittests returned %d" % self.retvalue())
            logger.warning("stdout:\n%s\n" % stdout)
            logger.warning("stderr:\n%s\n" % stderr)

    def __str__(self):
        return "lw_hostengine" + super(TestDcgmUnittestsApp, self).__str__()
