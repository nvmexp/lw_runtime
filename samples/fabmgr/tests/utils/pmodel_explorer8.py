'''
This class implements product specific values for Explorer8 initial prototype machines (8 Gpus, 3 LWSwitches)
'''
class DgxProductModelExplorer8:
    # returns the number of GPUs expected on the specified DGX model
    def NumGpus(self):
        return 8

    # returns the number of LWSwitches expected on the specified DGX model
    def NumLWSwitches(self):
        return 3

    # returns the number of LWLink connections expected on the specified DGX model
    def NumLWLinkConns(self):
        return 48

    # returns the number of LWLink trunk connections
    def NumLWLinkTrunkConns(self):
        return 0            