'''
This class implements product specific values for Explorer16(DGX-2)
'''
class DgxProductModelExplorer16:
    # returns the number of GPUs expected on the specified DGX model
    def NumGpus(self):
        return 16

    # returns the number of LWSwitches expected on the specified DGX model
    def NumLWSwitches(self):
        return 12

    # returns the number of LWLink connections expected on the specified DGX model
    def NumLWLinkConns(self):
        return 144

    # returns the number of LWLink trunk connections
    def NumLWLinkTrunkConns(self):
        return 48