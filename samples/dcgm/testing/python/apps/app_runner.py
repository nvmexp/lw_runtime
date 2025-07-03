import subprocess
import os
import threading
import string
import datetime

import logger
import option_parser
import utils
import test_utils

default_timeout = 10.0 # 10s

class AppRunner(object):
    """
    Class for running command line applications. It handles timeouts, logging and reading stdout/stderr.
    Stdout and stderr of an application is also stored in log output dir in files process_<NB>_stdout/stderr.txt

    If application finished with non 0 error code you need to mark it with .validate() function. Otherwise testing
    framework will fail a subtest. AppRunner is also "validated" when .terminate() is called.

    You can access all lines read so far (or when the application terminates all lines printed) from attributes
    .stdout_lines
    .stderr_lines

    You can see how long the application ran for +/- some minimal overhead (must run() for time to be accurate:
    .runTime

    # Sample usage
    app = AppRunner("lwpu-smi", ["-l", "1"])
    app.run(timeout=2.5)
    print string.join(app.stdout_lines, "\n")

    Notes: AppRunner works very closely with test_utils SubTest environment. SubTest at the end of the test
           checks that all applications finished successfully and kills applications that didn't finish by
           the end of the test.
    """
    RETVALUE_TERMINATED = "Terminated"
    RETVALUE_TIMEOUT = "Terminated - Timeout"

    _processes = []               # Contains list of all processes running in the background
    _processes_not_validated = [] # Contains list of processes that finished with non 0 error code 
                                  #     and were not marked as validated
    _process_nb = 0

    def __init__(self, exelwtable, args=None, cwd=None, elw=None):
        self.exelwtable = exelwtable
        if args is None:
            args = []
        self.args = args
        self.cwd = cwd
        if elw is None:
            elw = dict()
        self.elw = elw

        self._timer = None              # to implement timeout
        self._subprocess = None
        self._retvalue = None           # stored return code or string when the app was terminated
        self._lock = threading.Lock()   # to implement thread safe timeout/terminate
        self.stdout_lines = []          # buff that stores all app's output
        self.stderr_lines = []
        self._logfile_stdout = None
        self._logfile_stderr = None
        self._is_validated = False
        self._info_message = False
        
        self.process_nb = AppRunner._process_nb
        AppRunner._process_nb += 1

    def run(self, timeout=default_timeout):
        """
        Run the application and wait for it to finish. 
        Returns the app's error code/string
        """
        self.start(timeout)
        return self.wait()
    
    def start(self, timeout=default_timeout):
        """
        Begin exelwting the application.
        The application may block if stdout/stderr buffers become full.
        This should be followed by self.terminate() or self.wait() to finish exelwtion.
        Exelwtion will be forcefully terminated if the timeout expires.
        If timeout is None, then this app will never timeout.
        """
        assert self._subprocess is None

        logger.debug("Starting " + str(self))

        
        elw = self._create_subprocess_elw()
        if utils.is_linux():
            if os.path.exists(self.exelwtable):
                # On linux, for binaries inside the package (not just commands in the path) test that they have +x
                # e.g. if package is extracted on windows and copied to Linux, the +x privileges will be lost
                assert os.access(self.exelwtable, os.X_OK), "Application binary %s is not exelwtable! Make sure that the testing archive has been correctly extracted." % (self.exelwtable)
        self.startTime = datetime.datetime.now()
        self._subprocess = subprocess.Popen(
                [self.exelwtable] + self.args, 
                stdin=None, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                cwd=self.cwd, 
                elw=elw)
        AppRunner._processes.append(self) # keep track of running processe
        # Start timeout if we want one
        self._timer = None
        if timeout is not None:
            self._timer = threading.Timer(timeout, self._trigger_timeout)
            self._timer.start()

        if not test_utils.noLogging:
            def args_to_fname(args):
                # crop each argument to 16 characters and make sure the output string is no longer than 50 chars
                # Long file names are hard to read (hard to find the extension of the file)
                # Also python sometimes complains about file names being too long.
                #   IOError: [Errno 36] File name too long
                return string.join(map(lambda x: utils.string_to_valid_file_name(x)[:16], self.args), "_")[:50]
            shortname = os.path.basename(self.exelwtable) + "_" + args_to_fname(self.args)
            stdout_fname = os.path.relpath(os.path.join(
                logger.log_dir, "app_%03d_%s_stdout.txt" % (self.process_nb, shortname)))
            stderr_fname = os.path.relpath(os.path.join(
                logger.log_dir, "app_%03d_%s_stderr.txt" % (self.process_nb, shortname)))
            # If the app fails, this message will get printed. If it succeeds it'll get popped in _process_finish
            self._info_message = logger.info("Starting %s...\nstdout in %s\nstderr in %s" % (
                str(self)[:64], # cut the string to make it more readable
                stdout_fname, stderr_fname), defer=True)
            self._logfile_stdout = open(stdout_fname, "w")
            self._logfile_stderr = open(stderr_fname, "w")

    def _process_finish(self, stdout_buf, stderr_buf):
        """
        Logs return code/string and reads the remaining stdout/stderr.

        """
        logger.debug("Application %s returned with status: %s" % (self.exelwtable, self._retvalue))
        self.runTime = datetime.datetime.now() - self.startTime

        self._split_and_log_lines(stdout_buf, self.stdout_lines, self._logfile_stdout)
        self._split_and_log_lines(stderr_buf, self.stderr_lines, self._logfile_stderr)

        if self._logfile_stdout:
            self._logfile_stdout.close()
        if self._logfile_stderr:
            self._logfile_stderr.close()
        AppRunner._processes.remove(self)
        if self._retvalue != 0 and self._retvalue != AppRunner.RETVALUE_TERMINATED:
            AppRunner._processes_not_validated.append(self)
        else:
            self._is_validated = True
            logger.pop_defered(self._info_message)

    def wait(self):
        """
        Wait for application to finish and return the app's error code/string

        """
        if self._retvalue is not None:
            return self._retvalue

        logger.debug("Waiting for application %s to finish" % str(self))
        stdout_buf, stderr_buf = self._subprocess.communicate()
        if self._timer is not None:
            self._timer.cancel()

        with self._lock:                   # set ._retvalue in thread safe way. Make sure it wasn't set by timeout already
            if self._retvalue is None:
                self._retvalue = self._subprocess.returncode
                self._process_finish(stdout_buf, stderr_buf)

        return self._retvalue

    def poll(self):
        if self._retvalue is None:
            self._retvalue = self._subprocess.poll()
            if self._retvalue is not None:
                stdout_buf = self._read_all_remaining(self._subprocess.stdout)
                stderr_buf = self._read_all_remaining(self._subprocess.stderr)
                self._process_finish(stdout_buf, stderr_buf)
                
        return self._retvalue

    def _trigger_timeout(self):
        """
        Function called by timeout routine. Kills the app in a thread safe way.

        """
        logger.warning("App %s with pid %d has timed out. Killing it." % (self.exelwtable, self.getpid()))
        with self._lock: # set ._retvalue in thread safe way. Make sure that app wasn't terminated already
            if self._retvalue is not None:
                return self._retvalue

            self._subprocess.kill()
            stdout_buf = self._read_all_remaining(self._subprocess.stdout)
            stderr_buf = self._read_all_remaining(self._subprocess.stderr)
            self._retvalue = AppRunner.RETVALUE_TIMEOUT
            self._process_finish(stdout_buf, stderr_buf)

            return self._retvalue

    def _create_subprocess_elw(self):
        ''' Merge additional elw with current elw '''
        elw = os.elwiron.copy()
        for key in self.elw:
            elw[key] = self.elw[key]
        return elw

    def validate(self):
        """
        Marks the process that finished with error code as validated - the error was either expected or handled by the caller
        If process finished with error but wasn't validated one of the subtest will fail.

        """
        assert self.retvalue() != None, "This function shouldn't be called when process is still running"

        if self._is_validated:
            return
        self._is_validated = True
        self._processes_not_validated.remove(self)
        logger.pop_defered(self._info_message)

    def terminate(self):
        """
        Forcfully terminates the application and return the app's error code/string.

        """
        with self._lock: # set ._retvalue in thread safe way. Make sure that app didn't timeout
            if self._retvalue is not None:
                return self._retvalue

            if self._timer is not None:
                self._timer.cancel()
            self._subprocess.kill()
            
            stdout_buf = self._read_all_remaining(self._subprocess.stdout)
            stderr_buf = self._read_all_remaining(self._subprocess.stderr)
            self._retvalue = AppRunner.RETVALUE_TERMINATED
            self._process_finish(stdout_buf, stderr_buf)
            
            return self._retvalue

    def signal(self, signal):
        """
        Send a signal to the process
        """
        self._subprocess.send_signal(signal)

    def _read_all_remaining(self, stream):
        """
        Return a string representing the entire remaining contents of the specified stream
        This will block if the stream does not end
        Should only be called on a terminated process
        """
        out_buf = ""

        while True:
            rawline = stream.readline()
            if rawline == "":
                break
            else:
                out_buf += rawline

        return out_buf

    def _split_and_log_lines(self, input_string, buff, log_file):
        """
        Splits string into lines, removes '\\n's, and appends to buffer & log file

        """
        lines = input_string.splitlines()

        for i in xrange(len(lines)):
            lines[i] = string.rstrip(lines[i], "\n\r")
            if log_file:
                log_file.write(lines[i])
                log_file.write("\n")
            buff.append(lines[i])

    def stdout_readtillmatch(self, match_fn):
        """
        Blocking function that reads input until function match_fn(line : str) returns True.
        If match_fn didn't match anything function raises EOFError exception
        """
        logger.debug("stdout_readtillmatch called", caller_depth=1)

        while True:
            rawline = self._subprocess.stdout.readline()
            if rawline == "":
                break
            else:
                rawline = string.rstrip(rawline, "\n\r")
                if self._logfile_stdout:
                    self._logfile_stdout.write(rawline)
                    self._logfile_stdout.write("\n")
                self.stdout_lines.append(rawline)

            if match_fn(rawline) is True:
                return 
        raise EOFError("Process finished and requested match wasn't found")

    def retvalue(self):
        """
        Returns code/string if application finished or None otherwise.

        """
        if self._subprocess.poll() is not None:
            self.wait()
        return self._retvalue

    def getpid(self):
        """
        Returns the pid of the process

        """
        
        return self._subprocess.pid

    def __str__(self):
        return ("AppRunner #%d: %s %s (cwd: %s; elw: %s)" %
                (self.process_nb, self.exelwtable, string.join(self.args, " "), self.cwd, self.elw))
    def __repr__(self):
        return str(self)

    @classmethod
    def clean_all(cls):
        """
        Terminate all processes that were created using this class and makes sure that all processes that were spawned were validated.

        """
        import test_utils
        def log_output(message, process):
            """
            Prints last 10 lines of stdout and stderr for faster lookup
            """
            logger.info("%s: %s" % (message, process))
            logger.info("Last 10 lines of stdout")
            with logger.IndentBlock():
                for line in process.stdout_lines[-10:]:
                    logger.info(line)
            logger.info("Last 10 lines of stderr")
            with logger.IndentBlock():
                for line in process.stderr_lines[-10:]:
                    logger.info(line)

        with test_utils.SubTest("not terminated processes", quiet=True):
            assert AppRunner._processes == [], "Some processes were not terminated by previous test: " + str(AppRunner._processes)
        for process in AppRunner._processes[:]:
            log_output("Unterminated process", process)
            process.terminate()
        with test_utils.SubTest("not validated processes", quiet=True):
            for process in AppRunner._processes_not_validated:
                log_output("Process returned %s ret code" % process.retvalue(), process)
            assert AppRunner._processes_not_validated == [], "Some processes failed and were not validated by previous test: " + str(AppRunner._processes_not_validated)
        AppRunner._processes_not_validated = []
