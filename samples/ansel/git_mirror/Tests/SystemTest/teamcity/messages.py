# coding=utf-8
import sys
import datetime

if sys.version_info < (3, ):
    # Python 2
    text_type = unicode  # flake8: noqa
else:
    # Python 3
    text_type = str


_quote = {"'": "|'", "|": "||", "\n": "|n", "\r": "|r", '[': '|[', ']': '|]'}

def escape_value(value):
    return "".join(_quote.get(x, x) for x in value)


class TeamcityServiceMessages(object):
    def __init__(self, output=sys.stdout, now=datetime.datetime.now, encoding='auto'):
        if sys.version_info < (3, ) or not hasattr(output, 'buffer'):
            self.output = output
        else:
            self.output = output.buffer
        self.now = now

        if encoding and encoding != 'auto':
            self.encoding = encoding
        elif getattr(output, 'encoding', None) or encoding == 'auto':
            # Default encoding to 'utf-8' because it sucks if we fail with a
            # `UnicodeEncodeError` simply because LANG didn't get propagated to
            # a subprocess or something and sys.stdout.encoding is None
            self.encoding = getattr(output, 'encoding', None) or 'utf-8'
        else:
            self.encoding = None

    def escapeValue(self, value):
        return escape_value(value)

    def message(self, messageName, **properties):
        timestamp = self.now().strftime("%Y-%m-%dT%H:%M:%S.") + "%03d" % (self.now().microsecond / 1000)
        message = ("##teamcity[%s timestamp='%s'" % (messageName, timestamp))

        for k in sorted(properties.keys()):
            value = properties[k]
            if value is None:
                continue

            message += (" %s='%s'" % (k, self.escapeValue(value)))

        message += ("]\n")

        if self.encoding and isinstance(message, text_type):
            message = message.encode(self.encoding)

        # Python may buffer it for a long time, flushing helps to see real-time result
        self.output.write(message)
        self.output.flush()

    def _single_value_message(self, messageName, value):
        message = ("##teamcity[%s '%s']\n" % (messageName, self.escapeValue(value)))

        if self.encoding and isinstance(message, text_type):
            message = message.encode(self.encoding)

        # Python may buffer it for a long time, flushing helps to see real-time result
        self.output.write(message)
        self.output.flush()

    def blockOpened(self, name, flowId=None):
        self.message('blockOpened', name=name, flowId=flowId)

    def blockClosed(self, name, flowId=None):
        self.message('blockClosed', name=name, flowId=flowId)

    def block(self, name, flowId=None):
        import teamcity.context_managers as cm
        return cm.block(self, name=name, flowId=flowId)

    def compilationStarted(self, compiler):
        self.message('compilationStarted', compiler=compiler)

    def compilationFinished(self, compiler):
        self.message('compilationFinished', compiler=compiler)

    def compilation(self, compiler):
        import teamcity.context_managers as cm
        return cm.compilation(self, compiler=compiler)

    def testSuiteStarted(self, suiteName, flowId=None):
        self.message('testSuiteStarted', name=suiteName, flowId=flowId)

    def testSuiteFinished(self, suiteName, flowId=None):
        self.message('testSuiteFinished', name=suiteName, flowId=flowId)

    def testSuite(self, name):
        import teamcity.context_managers as cm
        return cm.testSuite(self, name=name)

    def testStarted(self, testName, captureStandardOutput=None, flowId=None):
        self.message('testStarted', name=testName, captureStandardOutput=captureStandardOutput, flowId=flowId)

    def testFinished(self, testName, testDuration=None, flowId=None):
        if testDuration is not None:
            duration_ms = testDuration.days * 86400000 + \
                testDuration.seconds * 1000 + \
                int(testDuration.microseconds / 1000)
            self.message('testFinished', name=testName, duration=str(duration_ms), flowId=flowId)
        else:
            self.message('testFinished', name=testName, flowId=flowId)

    def test(self, testName, captureStandardOutput=None, testDuration=None, flowId=None):
        import teamcity.context_managers as cm
        return cm.test(self, testName=testName, captureStandardOutput=captureStandardOutput, testDuration=testDuration, flowId=flowId)

    def testIgnored(self, testName, message='', flowId=None):
        self.message('testIgnored', name=testName, message=message, flowId=flowId)

    def testFailed(self, testName, message='', details='', flowId=None):
        self.message('testFailed', name=testName, message=message, details=details, flowId=flowId)

    def testStdOut(self, testName, out, flowId=None):
        self.message('testStdOut', name=testName, out=out, flowId=flowId)

    def testStdErr(self, testName, out, flowId=None):
        self.message('testStdErr', name=testName, out=out, flowId=flowId)

    def publishArtifacts(self, path, flowId=None):
        self._single_value_message('publishArtifacts', path)

    def progressMessage(self, message):
        self._single_value_message('progressMessage', message)

    def progressStart(self, message):
        self._single_value_message('progressStart', message)

    def progressFinish(self, message):
        self._single_value_message('progressFinish', message)

    def progress(self, message):
        import teamcity.context_managers as cm
        return cm.progress(self, message=message)

    def buildProblem(self, description, identity):
        self.message('buildProblem', description=description, identity=identity)

    def buildStatus(self, status, text):
        self.message('buildStatus', status=status, text=text)

    def setParameter(self, name, value):
        self.message('setParameter', name=name, value=value)

    def buildStatisticLinesCovered(self, linesCovered):
        self.message('buildStatisticValue', key='CodeCoverageAbsLCovered', value=str(linesCovered))

    def buildStatisticTotalLines(self, totalLines):
        self.message('buildStatisticValue', key='CodeCoverageAbsLTotal', value=str(totalLines))

    def buildStatisticLinesUncovered(self, linesUncovered):
        self.message('buildStatisticValue', key='CodeCoverageAbsLUncovered', value=str(linesUncovered))

    def enableServiceMessages(self, flowId=None):
        self.message('enableServiceMessages', flowId=flowId)

    def disableServiceMessages(self, flowId=None):
        self.message('disableServiceMessages', flowId=flowId)

    def serviceMessagesDisabled(self, flowId=None):
        import teamcity.context_managers as cm
        return cm.serviceMessagesDisabled(self, flowId=flowId)

    def serviceMessagesEnabled(self, flowId=None):
        import teamcity.context_managers as cm
        return cm.serviceMessagesEnabled(self, flowId=flowId)

    def importData(self, typeID, pathToXMLFile):
        self.message('importData', type=typeID, path=pathToXMLFile)

    def lwstomMessage(self, text, status, errorDetails='', flowId=None):
        self.message('message', text=text, status=status, errorDetails=errorDetails, flowId=flowId)

