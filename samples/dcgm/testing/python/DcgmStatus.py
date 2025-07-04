
import pydcgm
import dcgm_agent
import dcgm_structs

class DcgmStatus:
    def __init__(self):
        self.handle = dcgm_agent.dcgmStatusCreate()
        self.errors = []

    def __del__(self):
        dcgm_agent.dcgmStatusDestroy(self.handle)

    '''
    Take any errors stored in our handle and update self.errors with them
    '''
    def UpdateErrors(self):
        errorCount = dcgm_agent.dcgmStatusGetCount(self.handle)
        if errorCount < 1:
            return

        for i in range(errorCount):
            self.errors.append(dcgm_agent.dcgmStatusPopError(self.handle))

    '''
    Throw an exception if any errors are stored in our status handle

    The exception text will contain all of the errors
    '''
    def ThrowExceptionOnErrors(self):
        #Make sure we've captured all errors before looking at them
        self.UpdateErrors()

        if len(self.errors) < 1:
            return

        errorString = "Errors: "
        for value in self.errors:
            errorString += "\"%s\"" % value
            raise dcgm_structs.DCGMError(value.status)

