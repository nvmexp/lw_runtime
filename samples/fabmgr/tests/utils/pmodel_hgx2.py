'''
This class implements product specific values for single baseboard(HGX-2)
'''
class Hgx2ProductModel:
    # returns the number of GPUs expected on the specified HGX model
    def NumGpus(self):
        return 8

    # returns the number of LWSwitches expected on the specified HGX model
    def NumLWSwitches(self):
        return 6

    # returns the number of LWLink connections expected on the specified HGX model
    def NumLWLinkConns(self):
        return 96

    # returns the number of LWLink trunk connections
    def NumLWLinkTrunkConns(self):
        return 48