from DcgmReader import DcgmReader
from json import dumps as toJson
from os import elwiron
from socket import socket, AF_INET, SOCK_DGRAM
from time import sleep
import dcgm_fields
import logging

class DcgmJsonReader(DcgmReader):

    ###########################################################################
    def ColwertFieldIdToTag(self, fieldId):
        return self.m_fieldIdToInfo[fieldId].tag

    ###########################################################################
    def PrepareJson(self, gpuId, obj):
        '''
        Receive an object with measurements turn it into an equivalent JSON. We
        add the GPU UUID first.
        '''
        uuid = self.m_gpuIdToUUId[gpuId]
        # This mutates the original object, but it shouldn't be a problem here
        obj['gpu_uuid'] = uuid
        return toJson(obj)

    ###########################################################################
    def LwstomDataHandler(self, fvs):
        for gpuId in fvs.keys():
            # We don't need the keys because each value has a `fieldId`
            # So just get the values
            gpuData = fvs[gpuId].values()

            # Get the values from FV (which is a list of values)
            valuesListOfLists = [datum.values for datum in gpuData]

            # We only want the last measurement
            lastValueList = [l[-1] for l in valuesListOfLists]

            # Turn FV into a colwentional Python Object which can be colwerted to JSON
            outObject = {self.ColwertFieldIdToTag(i.fieldId): i.value for i in lastValueList}
            outJson = self.PrepareJson(gpuId, outObject)

            self.LwstomJsonHandler(outJson)

    ###########################################################################
    def LwstomJsonHandler(self, outJson):
        '''
        This method should be overriden by subclasses to handle the JSON objects
        received.
        '''
        logging.warn('LwstomJsonHandler has not been overriden')
        logging.info(outJson)
