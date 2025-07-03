#!/usr/bin/python
import LWPU.Hadoop.Streaming.Streaming
from LWPU.SHIMDB.Dictionary.Titles import TitlesRef, DRSProfilesRef, ShortNamesRef
import json

class MyStreaming(LWPU.Hadoop.Streaming.Streaming.Streaming):

  def __init__(self):
    super(MyStreaming, self).__init__()
    self.Parser = json.JSONDecoder()

  def Mapper(self, data):
    if data:
      # JSON Parse
      try:
        JSONDolwment = self.Parser.decode(data.value[0])
      except:
        return
      JSONDolwment['Version'] = 1
      self.emit({'Key': JSONDolwment['deviceId'], 'Value': JSONDolwment})

  def Reducer(self, data):
    if data:
      JSONDolwment = data.value
      self.emit(
        [
          JSONDolwment.get('clientVer', ''),
          JSONDolwment.get('eventId', ''),
          JSONDolwment.get('eventName', ''),
          JSONDolwment.get('deviceId', ''),
          JSONDolwment.get('userId', ''),
          JSONDolwment.get('sessionId', ''),
          JSONDolwment.get('serverTs', ''),
          JSONDolwment.get('clientTs', ''),
          JSONDolwment.get('adjustedTs', ''),
          JSONDolwment.get('location', {}).get('isInternal', ''),
          JSONDolwment.get('location', {}).get('country', ''),
          JSONDolwment.get('location', {}).get('region', ''),
          JSONDolwment.get('location', {}).get('city', ''),
          JSONDolwment.get('location', {}).get('zipcode', ''),
          JSONDolwment.get('location', {}).get('timezone', ''),
          JSONDolwment.get('location', {}).get('latitude', ''),
          JSONDolwment.get('location', {}).get('longitude', ''),
          JSONDolwment.get('location', {}).get('isp', ''),
          JSONDolwment.get('location', {}).get('domain', ''),
          JSONDolwment.get('location', {}).get('netspeed', ''),
          DRSProfilesRef.get(JSONDolwment.get('parameters', {}).get('drsProfileName', '').lower(), {}).get(JSONDolwment.get('parameters', {}).get('drsAppName', '').lower(), 0) or ShortNamesRef.get(JSONDolwment.get('parameters', {}).get('appExeName', '').lower(), 0),
          JSONDolwment.get('parameters', {}).get('appExeName', ''),
          JSONDolwment.get('parameters', {}).get('drsProfileName', ''),
          JSONDolwment.get('parameters', {}).get('drsAppName', ''),
          JSONDolwment.get('parameters', {}).get('screenResolutionX', ''),
          JSONDolwment.get('parameters', {}).get('screenResolutionY', ''),
          JSONDolwment.get('parameters', {}).get('colorBufferFormat', ''),
          JSONDolwment.get('parameters', {}).get('depthBufferFormat', ''),
          JSONDolwment.get('parameters', {}).get('kindOfShot', ''),
          JSONDolwment.get('parameters', {}).get('colorRange', ''),
          JSONDolwment.get('parameters', {}).get('highresMultiplier', ''),
          JSONDolwment.get('parameters', {}).get('quality360resolution', ''),
          JSONDolwment.get('parameters', {}).get('fov', ''),
          JSONDolwment.get('parameters', {}).get('roll', ''),
          JSONDolwment.get('parameters', {}).get('lwrrentCameraPosX', ''),
          JSONDolwment.get('parameters', {}).get('lwrrentCameraPosY', ''),
          JSONDolwment.get('parameters', {}).get('lwrrentCameraPosZ', ''),
          JSONDolwment.get('parameters', {}).get('lwrrentCameraRotX', ''),
          JSONDolwment.get('parameters', {}).get('lwrrentCameraRotY', ''),
          JSONDolwment.get('parameters', {}).get('lwrrentCameraRotZ', ''),
          JSONDolwment.get('parameters', {}).get('lwrrentCameraRotW', ''),
          JSONDolwment.get('parameters', {}).get('originalCameraPosX', ''),
          JSONDolwment.get('parameters', {}).get('originalCameraPosY', ''),
          JSONDolwment.get('parameters', {}).get('originalCameraPosZ', ''),
          JSONDolwment.get('parameters', {}).get('originalCameraRotX', ''),
          JSONDolwment.get('parameters', {}).get('originalCameraRotY', ''),
          JSONDolwment.get('parameters', {}).get('originalCameraRotZ', ''),
          JSONDolwment.get('parameters', {}).get('originalCameraRotW', ''),
          JSONDolwment.get('parameters', {}).get('specialEffectsMode', ''),
          JSONDolwment.get('parameters', {}).get('effectName', '').replace('\n', ','),
          JSONDolwment.get('parameters', {}).get('allUserConstants', '').replace('\n', ','),
          JSONDolwment.get('parameters', {}).get('uiMode', ''),
          JSONDolwment.get('parameters', {}).get('errorType', ''),
          JSONDolwment.get('parameters', {}).get('errorCode', ''),
          JSONDolwment.get('parameters', {}).get('errorString', ''),
          JSONDolwment.get('parameters', {}).get('sourceFilename', ''),
          JSONDolwment.get('parameters', {}).get('sourceLine', ''),
          JSONDolwment.get('parameters', {}).get('captureStateOnError', ''),
          JSONDolwment.get('parameters', {}).get('gamepadMapping', ''),
          JSONDolwment.get('parameters', {}).get('gamepadProductId', ''),
          JSONDolwment.get('parameters', {}).get('gamepadVendorId', ''),
          JSONDolwment.get('parameters', {}).get('gamepadVersionNumber', ''),
          JSONDolwment.get('parameters', {}).get('gamepadUsedForCameraOperationDuringThisSession', ''),
          JSONDolwment.get('parameters', {}).get('gamepadUsedForUIInteractionDuringThisSession', ''),
          JSONDolwment.get('parameters', {}).get('timeElapsedSinceSessionStarted', ''),
          JSONDolwment.get('parameters', {}).get('timeElapsedSinceCaptureStarted', '')
        ]
      )

RawAnselToTSV = MyStreaming()
RawAnselToTSV.run()
