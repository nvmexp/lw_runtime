#!/usr/bin/elw python
# -*- coding: utf-8 -*-

import os
import re
import sys
import uuid
import glob
import time
import ctypes
import random
import getpass
import itertools
import threading
import unittest
from ctypes import *
from sets import Set

import Utils
from teamcity import is_running_under_teamcity
from teamcity.unittestpy import TeamcityTestRunner
from Config import *
from Utils import *
from google import protobuf
from BusMessage_pb2 import *
from ipc_pb2 import *

from ToolsTests import *
from YAMLFXCTests import *

LogFile = 'smoketests.log'
generateGoldens = False

class AllowlistingMode:
	ENABLED = 1
	APPROVED = 2
	DEFAULT = 3

# This base class does not perform testing itself, but it helps setting up the testing environment:
# * installs Ansel from the artifact/ folder if running on TeamCity
# * sets up modding allowlisting
# * joins to the MessageBus as "Ansel/LwCameraControl" end-point
# * starts the graphics process to carry the testing on
# * implements functions to post a message on the MessageBus as well as await for it
@for_all_test_methods(catch_all_errors)
class SmokeTestBase:
	@classmethod
	def setUpClass(cls, ProcessToTestOnImageName, ProcessToTestOnPath, ProcessToTestOnWorkDir, GoldenPath, Resolution, Allowlisting=AllowlistingMode.APPROVED):
		def onBusMessage(msg, size):
			def ParseMessage(msg):
				if msg.HasField('request'):
					return msg.request.ListFields()[0][1]
				elif msg.HasField('response'):
					return msg.response.ListFields()[0][1]
			message = msg[:size]
			msg = BusMessage()
			msg.ParseFromString(message)
			if msg.source_system == 'Ansel' and msg.source_module == 'LwCamera':
				if msg.HasField('generic'):
					lwCameraMsg = AnselIPCMessage()
					lwCameraMsg.ParseFromString(msg.generic.data)
					specificMsg = ParseMessage(lwCameraMsg)
					if specificMsg != None:
						with cls.messagesLock:
							cls.messages.append(specificMsg)

		if TestArguments.override == "False":
			cls.AnselTargetPath = TestArguments.defaultFolder
		elif TestArguments.override == "ProgFiles":
			cls.AnselTargetPath = 'C:/Program Files/LWPU Corporation/Ansel'
		else:
			appFolder = os.path.dirname(ProcessToTestOnPath)
			cls.AnselTargetPath = appFolder + '/LwCamera'

		cls.AnselToolsPath = cls.AnselTargetPath + '/'
		cls.LwCameraEnablePath = cls.AnselToolsPath + 'LwCameraEnable.exe'
		cls.HighresBlender32Path = cls.AnselToolsPath + 'HighresBlender32.exe'
		cls.HighresBlender64Path = cls.AnselToolsPath + 'HighresBlender64.exe'
		cls.SphericalEquirect32Path = cls.AnselToolsPath + 'SphericalEquirect32.exe'
		cls.SphericalEquirect64Path = cls.AnselToolsPath + 'SphericalEquirect64.exe'
		cls.LwImageColwert32Path = cls.AnselToolsPath + 'LwImageColwert32.exe'
		cls.LwImageColwert64Path = cls.AnselToolsPath + 'LwImageColwert64.exe'
		cls.YAMLFXC32Path = cls.AnselToolsPath + 'YAMLFXC32.exe'
		cls.YAMLFXC64Path = cls.AnselToolsPath + 'YAMLFXC64.exe'

		cls.goldenPath = GoldenPath
		[cls.xResolution, cls.yResolution] = Resolution
		cls.messages = []
		cls.messagesLock = threading.Lock()	
		# try .. except is needed because unittest framework will not call tearDownClass
		# if setUpClass raises an exception
		try:
			# now load MessageBusClientDll
			if sys.maxsize > 2**32:
				cls.mbClient = cdll.LoadLibrary(MessageBusClientDll64)
			else:
				cls.mbClient = cdll.LoadLibrary(MessageBusClientDll32)
			# allowlist the Ansel integrated app for allowlisting without restrictions
			# this will allow us to use 'Testing' shader which is not in the approved list
			if Allowlisting == AllowlistingMode.ENABLED:
				allowlisting_args = 'ANSEL_FREESTYLE_MODE=ENABLED'
			elif Allowlisting == AllowlistingMode.APPROVED:
				allowlisting_args = 'ANSEL_FREESTYLE_MODE=APPROVED_ONLY'
			elif Allowlisting == AllowlistingMode.DEFAULT:
				allowlisting_args = '-dANSEL_FREESTYLE_MODE'
			else:
				raise Exception('Failed to set modding allowlisting mode (%s)' % str(Allowlisting))
			allowlist_process = run_detached_process([D3DRegPath, allowlisting_args], '.')
			retcode = allowlist_process.wait()
			if retcode != 0:
				raise Exception('Failed to allowlist the test app (%s)' % allowlisting_args)
			print ("Using %s as a client" % cls.mbClient)
			cls.onBusMessageCallbackFunc = CFUNCTYPE(None, POINTER(c_char), c_uint32)
			cls.onBusMessageCallback = cls.onBusMessageCallbackFunc(onBusMessage)
			cls.mbClient.joinMessageBus('Ansel', 'LwCameraControl', cls.onBusMessageCallback)
			print('Current user: %s' % getpass.getuser())
			cls.startTime = time.time()
			# try killing test process first
			cls.processImageName = ProcessToTestOnImageName
			os.system("taskkill /f /im %s" % ProcessToTestOnImageName)
			# enable IPC
			cls.ipcEnabledRegkeyValue = get_regkey_value("HKEY_LWRRENT_USER\\Software\\LWPU Corporation\\Ansel", "IPCenabled")
			if cls.ipcEnabledRegkeyValue == '0':
				print ('Enabling IPC ...')
				set_regkey_value(r"Software\LWPU Corporation\Ansel", "IPCenabled", "1")
			
			# install ansel
			if is_running_under_teamcity():
				# Check whether we're trying to copy files into the DriverStore
				if TestArguments.canModifyDefaultFolder == False and TestArguments.override == "False":
					#TODO: Detect team city case earlier, and use program files as the install destination, instead of the defaultFolder 
					# (but I think this is a fairly low priority change at the moment since we control the build and test system for TeamCity)
					raise Exception('Cannot install Ansel to default folder (DriverStore).. Tests with TeamCity can only be run on non-DCH systems at the moment')
				else:
					print ('Installing the LwCamera package to - ' + cls.AnselTargetPath)
					install_ansel(cls.AnselTargetPath)
			else:
				print('Skipping Ansel installation, because TeamCity is not detected')

			# Check whether we're trying to copy files into the DriverStore
			if TestArguments.canModifyDefaultFolder == False and TestArguments.override == "False":
				# Note: Assumes that the path to program files is in C: (so the tests will fail if the Program Files folder is not at C:\Program Files)
				copy_testing_shader('C:/Program Files/LWPU Corporation/Ansel/')
			else:
				copy_testing_shader(cls.AnselTargetPath)
			# run test app and ipc tool
			cls.ProcessToTestOnPath = ProcessToTestOnPath
			cls.testProcess = run_detached_process(ProcessToTestOnPath, ProcessToTestOnWorkDir)
			# wait for the graphics app to startup
			sleep(5)
			cls.testProcess.poll()
			if cls.testProcess.returncode is not None:
				raise Exception('Test process exited unexpectedly')
			# now read its output ('LwCamera found' or 'LwCamera not found')
			if 'LwCamera' not in get_process_modules(ProcessToTestOnImageName):
				raise Exception('LwCamera was not loaded by the test process')
			else:
				print('LwCamera was found in the process')
		except Exception as e:
			print('Exception raised during SmokeTest startup:\n%s' % e)
			if hasattr(cls, 'testProcess') and cls.testProcess is not None:
				cls.shutdownTestProcess()
			raise

	@classmethod
	def tearDownClass(cls):
		if cls.ipcEnabledRegkeyValue == '0':
			print('reverting IPCenabled')
			set_regkey_value(r"Software\LWPU Corporation\Ansel", "IPCenabled", "0")
		else:
			print('not reverting IPCenabled')
		cls.endTime = time.time()
		print('SmokeTest took ' + str(cls.endTime - cls.startTime) + ' seconds')
		cls.mbClient.leaveMessageBus()
		cls.shutdownTestProcess()

	@classmethod
	def shutdownTestProcess(cls):
		print('Closing graphics process')
		# try closing the app gracefully by sending WM_CLOSE
		os.system("taskkill /im %s" % cls.processImageName)
		sleep(1)
		# kill it just in case
		if 'testProcess' in cls.__dict__:
			cls.testProcess.kill()
		os.system("taskkill /f /im %s" % cls.processImageName)

	def PostMessage(self, msg):
		busMessage = BusMessage()
		busMessage.uniqueid = '0123-4567-89AB-CDEF'
		busMessage.source_system = 'Ansel'
		busMessage.source_module = 'LwCameraControl'
		busMessage.generic.data = msg.SerializeToString()
		busMessageSerialized = busMessage.SerializeToString()
		self.__class__.mbClient.postMessage(busMessageSerialized, len(busMessageSerialized))

	def PostRequest(self, msg, requestFieldName):
		lwCameraMsg = AnselIPCMessage()
		lwCameraRequest = AnselIPCRequest()
		getattr(lwCameraRequest, requestFieldName).MergeFrom(msg)
		lwCameraMsg.request.MergeFrom(lwCameraRequest)
		self.PostMessage(lwCameraMsg)

	def GetMessage(self, msgIpcType):
		with self.__class__.messagesLock:
			for msg in self.__class__.messages:
				if type(msg) is msgIpcType:
					self.__class__.messages.remove(msg)
					return msg
		return None

	def AwaitMessage(self, msgIpcType, timeout, debug=False):
		start_time = time.time()
		read_ipc_timeout = 0.01
		while time.time() - start_time < timeout:
			if debug:
				print ('Received messages:\n', self.__class__.messages)
			msg = self.GetMessage(msgIpcType)
			if msg is not None:
				return msg
			sleep(read_ipc_timeout)
		return None

# This is a Python context manager to manage the Ansel session.
# It is useful to do testing inside an Ansel session like this:
# 		# just try entering and exiting Ansel
#		with AnselSession(self, AnselSession.ANSEL):
#			pass
#			# some other commands go here
#
class AnselSession:
	ANSEL = 1
	MODSONLY = 2
	def __init__(self, testSuite, mode):
		self.testSuite = testSuite
		self.mode = mode		
	def __enter__(self):
		pauseApp = True if self.mode == AnselSession.ANSEL else False
		response = self.testSuite.setAnselEnabled(enabled=True, pauseApplication = pauseApp, leaveFilters=None)
		self.testSuite.assertEqual(response.status, kOkAnsel if self.mode == AnselSession.ANSEL else kOkModsOnly)
	def __exit__(self, type, value, traceback):
		leaveFilters = False if self.mode == AnselSession.ANSEL else True
		response = self.testSuite.setAnselEnabled(enabled=False, 
			pauseApplication = False, 
			leaveFilters = leaveFilters)
		self.testSuite.assertEqual(response.status, kOk)

# This base class implements the majority of tests hapenning here
# It is agnostic to the graphics process to run the tests on.
# It is meant to be inherited by child classes which would specify:
# the actual process name, 
# a path to goldens, 
# expected resolution, 
# allowlisting mode
@for_all_test_methods(catch_all_errors)
class SmokeTest(SmokeTestBase):
	@classmethod
	def setUpClass(cls, ProcessToTestOnImageName, ProcessToTestOnPath, ProcessToTestOnWorkDir, GoldenPath, Resolution, Allowlisting=AllowlistingMode.APPROVED):
		SmokeTestBase.setUpClass(ProcessToTestOnImageName, ProcessToTestOnPath, ProcessToTestOnWorkDir, GoldenPath, Resolution, Allowlisting)
	@classmethod
	def tearDownClass(cls):
		SmokeTestBase.tearDownClass()

	# method decorator to skip certain tests if we're running DX9
	def skipIfNoHDR(func):
		def wrapper(self):
			if self.__class__.processImageName in NoHDRTestProcessNames:
				raise unittest.SkipTest("Not supported by DX9 test app")
			return func(self)
		return wrapper

	def setAnselEnabled(self, enabled, pauseApplication, leaveFilters):
		# we use IpcVersionResponse as a source of the current IPC version
		ipcVersionResponse = IpcVersionResponse()
		request = SetAnselEnabledRequest()
		request.major = ipcVersionResponse.major
		request.minor = ipcVersionResponse.major
		request.patch = ipcVersionResponse.patch
		request.enabled = enabled
		if pauseApplication is not None:
			request.pauseApplication = pauseApplication
		if leaveFilters is not None:
			request.leaveFiltersEnabled = leaveFilters
		self.PostRequest(request, 'setAnselEnabledRequest')
		response = self.AwaitMessage(SetAnselEnabledResponse, 2.0)
		self.assertIsNotNone(response)
		return response

	def takeRegularScreenshot(self, isExr=False, generateThumbnail=False, golden=None, psnr=None, generateGoldens=False):
		request = CaptureShotRequest()
		request.type = kRegular
		request.isExr = isExr
		request.generateThumbnail = generateThumbnail
		self.PostRequest(request, 'captureShotRequest')
		response = self.AwaitMessage(CaptureShotProcessingFinishedResponse, 2.0)
		self.assertIsNotNone(response)
		self.assertEqual(response.status, kOk)
		if golden is not None and psnr is not None:
			self.assertTrue(compare_images(response.absoluteFilePath, golden, '.', psnr, generateGoldens))
		return response

	def captureSuperResolution(self, mult, enhance=False, isExr=False, generateThumbnail=False, golden=None, psnr=None, generateGoldens=False):
		request = CaptureShotRequest()
		request.type = kHighres
		request.isExr = isExr
		request.highresMultiplier = mult
		request.generateThumbnail = generateThumbnail
		self.PostRequest(request, 'captureShotRequest')
		response = self.AwaitMessage(CaptureShotProcessingFinishedResponse, 300.0)
		self.assertIsNotNone(response)
		self.assertEqual(response.status, kOk)
		if golden is not None and psnr is not None:
			self.assertTrue(compare_images(response.absoluteFilePath, golden, '.', psnr, generateGoldens))
		return response

	def capture360(self, resolution, type, isExr=False, generateThumbnail=False, golden=None, psnr=None, generateGoldens=False):
		request = CaptureShotRequest()
		request.type = kPanorama360Mono if type == 'mono' else kPanorama360Stereo
		request.isExr = isExr
		request.generateThumbnail = generateThumbnail
		request.horizontal360Resolution = resolution
		self.PostRequest(request, 'captureShotRequest')
		response = self.AwaitMessage(CaptureShotProcessingFinishedResponse, 300.0)
		self.assertIsNotNone(response)
		self.assertEqual(response.status, kOk)
		if golden is not None and psnr is not None:
			self.assertTrue(compare_images(response.absoluteFilePath, golden, '.', psnr, generateGoldens))
		return response

	def captureRegularStereo(self, isExr=False, generateThumbnail=False, golden=None, psnr=None, generateGoldens=False):
		request = CaptureShotRequest()
		request.type = kRegularStereo
		request.isExr = isExr
		request.generateThumbnail = generateThumbnail
		self.PostRequest(request, 'captureShotRequest')
		response = self.AwaitMessage(CaptureShotProcessingFinishedResponse, 300.0)
		self.assertIsNotNone(response)
		self.assertEqual(response.status, kOk)
		if golden is not None and psnr is not None:
			self.assertTrue(compare_images(response.absoluteFilePath, golden, '.', psnr, generateGoldens))
		return response

	def getFilterList(self):
		self.PostRequest(GetFilterListRequest(), 'getFilterListRequest')
		response = self.AwaitMessage(GetFilterListResponse, 2.0)
		self.assertIsNotNone(response)
		self.assertEqual(len(response.filterIdList), len(response.filterNameList))
		return dict(zip(response.filterNameList, response.filterIdList))

	def getStackInfo(self):
		self.PostRequest(GetStackInfoRequest(), 'getStackInfoRequest')
		response = self.AwaitMessage(GetStackInfoResponse, 2.0)
		self.assertIsNotNone(response)
		return response

	def setFilter(self, filterId, stackIdx):
		request = SetFilterRequest()
		request.filterId, request.stackIdx = filterId, stackIdx
		self.PostRequest(request, 'setFilterRequest')
		response = self.AwaitMessage(SetFilterResponse, 2.0)
		self.assertIsNotNone(response)
		self.assertEqual(response.status, kOk)
		if response.HasField('filterProperties'):
			self.assertEqual(response.filterProperties.filterId, filterId)
		return response

	def insertFilter(self, filterId, stackIdx):
		request = InsertFilterRequest()
		request.filterId, request.stackIdx = filterId, stackIdx
		self.PostRequest(request, 'insertFilterRequest')
		response = self.AwaitMessage(InsertFilterResponse, 2.0)
		self.assertIsNotNone(response)
		self.assertEqual(response.status, kOk)
		if response.HasField('filterProperties'):
			self.assertEqual(response.filterProperties.filterId, filterId)
		return response
		
	def setFilterAndAttributes(self, filterId, stackIdx, controlIds, values):
		self.assertEqual(len(controlIds), len(values)) # Must provide a value for every ID and vice versa
		request = SetFilterAndAttributesRequest()
		request.filterId = filterId
		request.stackIdx = stackIdx
		for id in controlIds:
			request.floatControlIds.append(id)
		for val in values:
			request.floatValues.append(val)
		self.PostRequest(request, 'setFilterAndAttributesRequest')
		response = self.AwaitMessage(SetFilterAndAttributesResponse, 2.0)
		self.assertIsNotNone(response)
		self.assertEqual(response.setFilterResponse.status, kOk)
		if response.setFilterResponse.HasField('filterProperties'):
			self.assertEqual(response.setFilterResponse.filterProperties.filterId, filterId)

	def removeFilter(self, stackIdx):
		request = RemoveFilterRequest()
		request.stackIdx = stackIdx
		self.PostRequest(request, 'removeFilterRequest')
		response = self.AwaitMessage(RemoveFilterResponse, 2.0)
		self.assertIsNotNone(response)
		self.assertEqual(response.status, kOk)
		return response

	def moveFilter(self, newIndices):
		request = MoveFilterRequest()
		request.desiredStackIndices.extend(newIndices)
		self.PostRequest(request, 'moveFilterRequest')
		response = self.AwaitMessage(MoveFilterResponse, 2.0)
		self.assertIsNotNone(response)
		return response

	def thumbnailExists(self, response):
		timestamp_re = re.compile('- (\d{2}\.\d{2}\.\d{2}\.\d{2})')
		match = timestamp_re.search(response.absoluteFilePath)
		self.assertIsNotNone(match)
		timestamp = match.group(1)
		head, tail = os.path.split(response.absoluteFilePath)
		filelist = glob.glob(os.path.join(head, '*%s*Thumbnail*' % timestamp))
		return filelist

	def setFov(self, fov):
		request = SetFOVRequest()
		request.fov = fov
		self.PostRequest(request, 'setFOVRequest')		
		response = self.AwaitMessage(SetFOVResponse, 2.0)
		self.assertIsNotNone(response)
		self.assertEqual(response.status, kOk)

	def setRoll(self, roll):
		request = SetRollRequest()
		request.roll = roll
		self.PostRequest(request, 'setRollRequest')		
		response = self.AwaitMessage(SetRollResponse, 2.0)
		self.assertIsNotNone(response)
		self.assertEqual(response.status, kOk)

	def setFilterAttribute(self, filterId, stackIdx, controlId, value):
		request = SetFilterAttributeRequest()
		request.filterId = filterId
		request.stackIdx = stackIdx
		request.controlId = controlId
		request.floatValue.append(value)
		self.PostRequest(request, 'setFilterAttributeRequest')
		response = self.AwaitMessage(SetFilterAttributeResponse, 2.0)
		self.assertIsNotNone(response)

	def resetEntireStack(self):
		self.PostRequest(ResetEntireStackRequest(), 'resetEntireStackRequest')
		response = self.AwaitMessage(ResetEntireStackResponse, 2.0)
		self.assertIsNotNone(response)
		self.assertEqual(response.status, kOk)

	def assertStackIsEmpty(self):
		response = self.getStackInfo()
		# depending on how this test is being run (as a part of the test suite or individually)
		# and if Ansel was used before this request or not,
		# the result might be either an empty list of filters or 
		# a list of one item - 'None'. Both cases mean the stack is empty and are fine.

		self.assertLess(len(response.filterIds), 2)
		if len(response.filterIds) == 1:
			self.assertEqual(response.filterIds[0], 'None')

	def setHighQuality(self, isHighIQEnabled):
		request = SetHighQualityRequest()
		request.setting = isHighIQEnabled
		self.PostRequest(request, 'setHighQualityRequest')		
		response = self.AwaitMessage(SetHighQualityResponse, 2.0)
		self.assertIsNotNone(response)
		self.assertEqual(response.status, kOk)

	# ------------------------
	# GENERAL TESTS START HERE
	# ------------------------
	def test_step000_checkIpcIsEnabled(self):
		ipc_enabled = get_regkey_value(r"HKEY_LWRRENT_USER\Software\LWPU Corporation\Ansel", "IPCenabled", default="1")
		self.assertEqual(ipc_enabled, "1")

	def test_step001_checkAnselVersion(self):
		self.PostRequest(IpcVersionRequest(), 'ipcVersionRequest')
		response = self.AwaitMessage(IpcVersionResponse, 2.0)
		self.assertIsNotNone(response)
		print('Ansel IPC version: %d.%d.%d' % (response.major, response.minor, response.patch))

	def test_step002_enableDisableAnsel(self):
		# just try entering and exiting Ansel
		with AnselSession(self, AnselSession.ANSEL):
			pass

	def test_step003_enableDisableModding(self):
		# just try entering and exiting Ansel/modding
		with AnselSession(self, AnselSession.MODSONLY):
			pass

	def test_step004_tryIncorrectIpcVersion(self):
		# we use IpcVersionResponse as a source of the current IPC version
		ipcVersionResponse = IpcVersionResponse()
		request = SetAnselEnabledRequest()
		request.major = ipcVersionResponse.major + 1
		request.minor = ipcVersionResponse.major
		request.patch = ipcVersionResponse.patch
		request.enabled = True
		self.PostRequest(request, 'setAnselEnabledRequest')
		response = self.AwaitMessage(SetAnselEnabledResponse, 2.0)
		self.assertIsNotNone(response)
		self.assertEqual(response.status, kIncompatibleVersion)
		return response

	def test_step005_tryBackwardsCompatibility(self):
		# we use IpcVersionResponse as a source of the current IPC version
		ipcVersionResponse = IpcVersionResponse()
		request = SetAnselEnabledRequest()
		request.major = ipcVersionResponse.major
		request.minor = ipcVersionResponse.major + 1
		request.patch = ipcVersionResponse.patch
		for enabled, status in [(True, kOkAnsel), (False, kOk)]:
			request.enabled = enabled
			self.PostRequest(request, 'setAnselEnabledRequest')
			response = self.AwaitMessage(SetAnselEnabledResponse, 2.0)
			self.assertIsNotNone(response)
			self.assertEqual(response.status, status)

	def test_step006_enableEnabledAnsel(self):
		# try enabling Ansel, while in Ansel mode already
		with AnselSession(self, AnselSession.ANSEL):
			response = self.setAnselEnabled(True, True, None)
			self.assertEqual(response.status, kAlreadyEnabled)

	def test_step007_enableEnabledModding(self):
		# try enabling Modding, while in Modding mode already
		with AnselSession(self, AnselSession.MODSONLY):
			response = self.setAnselEnabled(True, False, None)
			self.assertEqual(response.status, kAlreadyEnabled)

	def test_step008_disableDisabledAnsel(self):
		# try disabling Ansel, while Ansel is not enabled
		response = self.setAnselEnabled(False, None, False)
		self.assertEqual(response.status, kAlreadyDisabled)

	def test_step009_upgradeModsToAnsel(self):
		# enable modding, then enable Ansel
		with AnselSession(self, AnselSession.MODSONLY):
			response = self.setAnselEnabled(True, True, None)
			self.assertEqual(response.status, kOkAnsel)
		
	def test_step010_checkAnselVersionAnsel(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.test_step001_checkAnselVersion()

	def test_step011_checkAnselVersionModding(self):
		with AnselSession(self, AnselSession.MODSONLY):
			self.test_step001_checkAnselVersion()

	def test_step012_checkAnselIsAvailable(self):
		self.PostRequest(IsAnselAvailableRequest(), 'isAnselAvailableRequest')
		response = self.AwaitMessage(IsAnselAvailableResponse, 2.0)
		self.assertIsNotNone(response)
		self.assertTrue(response.available)	

	def test_step013_checkAnselIsAvailableAnsel(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.test_step012_checkAnselIsAvailable()

	def test_step014_checkAnselIsAvailableModding(self):
		with AnselSession(self, AnselSession.MODSONLY):
			self.test_step012_checkAnselIsAvailable()

	def test_step015_checkFeatureSet(self):
		self.PostRequest(GetFeatureSetRequest(), 'getFeatureSetRequest')
		response = self.AwaitMessage(GetFeatureSetResponse, 2.0)
		self.assertIsNotNone(response)
		self.assertTrue(response.sdkDetected)
		self.assertTrue(response.modsAvailable)
		if self.__class__.ProcessToTestOnPath == DX11ProcessToTestOnPath32:
			self.assertFalse(response.restyleAvailable)
		if self.__class__.ProcessToTestOnPath == DX11ProcessToTestOnPath64:
			self.assertTrue(response.restyleAvailable)

	def test_step016_checkFeatureSetAnsel(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.test_step015_checkFeatureSet()

	def test_step017_checkFeatureSetModding(self):
		with AnselSession(self, AnselSession.MODSONLY):
			self.test_step015_checkFeatureSet()

	def test_step018_check360Range(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.PostRequest(Get360ResolutionRangeRequest(), 'get360ResolutionRangeRequest')		
			response = self.AwaitMessage(Get360ResolutionRangeResponse, 2.0)
			self.assertIsNotNone(response)
			self.assertTrue(response.minimumXResolution > 0)
			self.assertTrue(response.minimumXResolution < response.maximumXResolution)
			print('360 panorama range is %d...%d pixels wide' % (response.minimumXResolution, response.maximumXResolution))

	def test_step019_checkRollRange(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.PostRequest(GetRollRangeRequest(), 'getRollRangeRequest')		
			response = self.AwaitMessage(GetRollRangeResponse, 2.0)
			self.assertIsNotNone(response)
			self.assertTrue(response.minRoll < response.maxRoll)
			print('Camera roll range is %f...%f degrees' % (response.minRoll, response.maxRoll))

	def test_step020_checkFovRange(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.PostRequest(GetFOVRangeRequest(), 'getFOVRangeRequest')		
			response = self.AwaitMessage(GetFOVRangeResponse, 2.0)
			self.assertIsNotNone(response)
			self.assertTrue(response.minFov > 0)
			self.assertTrue(response.maxFov < 180)
			self.assertTrue(response.minFov < response.maxFov)
			print('Camera fov range is %f...%f degrees' % (response.minFov, response.maxFov))

	def test_step021_getFov(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.PostRequest(GetLwrrentFOVRequest(), 'getLwrrentFOVRequest')		
			response = self.AwaitMessage(GetLwrrentFOVResponse, 2.0)
			self.assertIsNotNone(response)
			self.assertTrue(response.fov > 0)
			self.assertTrue(response.fov < 180)
			print ('Initial FOV is: %f' % response.fov)

	def test_step022_setFov(self):
		with AnselSession(self, AnselSession.ANSEL):
			# get fov range
			self.PostRequest(GetFOVRangeRequest(), 'getFOVRangeRequest')		
			response = self.AwaitMessage(GetFOVRangeResponse, 2.0)
			self.assertIsNotNone(response)
			# and try setting every fov with increments of 10 degrees
			for fov in range(int(response.minFov), int(response.maxFov), 10):
				self.setFov(fov)

	def test_step023_setFovFailure(self):
		with AnselSession(self, AnselSession.ANSEL):
			# get fov range
			self.PostRequest(GetFOVRangeRequest(), 'getFOVRangeRequest')		
			response = self.AwaitMessage(GetFOVRangeResponse, 2.0)
			self.assertIsNotNone(response)
			# and try setting fovs outside of allowed range
			for fov in [response.minFov - 5, response.maxFov + 5]:
				request = SetFOVRequest()
				request.fov = fov
				self.PostRequest(request, 'setFOVRequest')		
				response = self.AwaitMessage(SetFOVResponse, 2.0)
				self.assertIsNotNone(response)
				self.assertEqual(response.status, kOutOfRange)

	def test_step024_checkFovRestored(self):
		# decrease FOV by 10 degrees, close Ansel session and check if fov is reset to
		# the original value
		with AnselSession(self, AnselSession.ANSEL):
			self.PostRequest(GetLwrrentFOVRequest(), 'getLwrrentFOVRequest')		
			response = self.AwaitMessage(GetLwrrentFOVResponse, 2.0)
			self.assertIsNotNone(response)
			fov = response.fov
			self.setFov(fov - 10.0)
		with AnselSession(self, AnselSession.ANSEL):
			time.sleep(0.5)
			self.PostRequest(GetLwrrentFOVRequest(), 'getLwrrentFOVRequest')		
			response = self.AwaitMessage(GetLwrrentFOVResponse, 2.0)
			self.assertIsNotNone(response)
			self.assertEqual(response.fov, fov)

	def test_step025_setRoll(self):
		with AnselSession(self, AnselSession.ANSEL):
			# get roll range
			self.PostRequest(GetRollRangeRequest(), 'getRollRangeRequest')		
			response = self.AwaitMessage(GetRollRangeResponse, 2.0)
			self.assertIsNotNone(response)
			# and try setting every roll with increments of 10 degrees
			for rollValue in range(int(response.minRoll), int(response.maxRoll), 10):
				self.setRoll(rollValue)

	def test_step026_setRollFailure(self):
		with AnselSession(self, AnselSession.ANSEL):
			# get roll range
			self.PostRequest(GetRollRangeRequest(), 'getRollRangeRequest')		
			response = self.AwaitMessage(GetRollRangeResponse, 2.0)
			self.assertIsNotNone(response)
			# and try setting rolls outside of allowed range
			for rollValue in [response.minRoll - 45, response.maxRoll + 45]:
				request = SetRollRequest()
				request.roll = rollValue
				self.PostRequest(request, 'setRollRequest')		
				response = self.AwaitMessage(SetRollResponse, 2.0)
				self.assertIsNotNone(response)
				self.assertEqual(response.status, kOutOfRange)

	def getSettings(self, mode):
		with AnselSession(self, mode):
			self.PostRequest(GetSettingsRequest(), 'getSettingsRequest')		
			response = self.AwaitMessage(GetSettingsResponse, 2.0)
			self.assertIsNotNone(response)

	def test_step027_getSettingsAnsel(self):
		self.getSettings(AnselSession.ANSEL)

	def test_step028_getSettingsModding(self):
		self.getSettings(AnselSession.MODSONLY)

	def test_step029_getGameSpecificControls(self):
		# the test app contains three game specific controls:
		# two checkboxes and a slider
		# we collect control types here and check that they are as expected
		# we use Set to not bother with the ordering of the controls
		with AnselSession(self, AnselSession.ANSEL):
			self.PostRequest(GetGameSpecificControlsRequest(), 'getGameSpecificControlsRequest')
			controlTypes = Set()
			for i in range(3):
				response = self.AwaitMessage(AddUIElementRequest, 2.0)
				self.assertIsNotNone(response)
				controlTypes.add(response.controlType)
			self.assertEqual(controlTypes, Set([kControlSlider, kControlBoolean]))

	def test_step030_isAnselModdingAvailableRequest(self):
		# we check IsAnselModdingAvailableResponse is in line with
		# GetFeatureSetResponse
		self.PostRequest(GetFeatureSetRequest(), 'getFeatureSetRequest')
		getFeatureSetResponse = self.AwaitMessage(GetFeatureSetResponse, 2.0)
		self.assertIsNotNone(getFeatureSetResponse)
		self.PostRequest(IsAnselModdingAvailableRequest(), 'isAnselModdingAvailableRequest')		
		response = self.AwaitMessage(IsAnselModdingAvailableResponse, 2.0)
		self.assertIsNotNone(response)
		self.assertEqual(response.status, kOk if getFeatureSetResponse.modsAvailable else kDisabled)

	def test_step031_isAnselSDKIntegrationAvailableRequest(self):
		# we check IsAnselSDKIntegrationAvailableResponse is in line with
		# GetFeatureSetResponse
		self.PostRequest(GetFeatureSetRequest(), 'getFeatureSetRequest')
		getFeatureSetResponse = self.AwaitMessage(GetFeatureSetResponse, 2.0)
		self.assertIsNotNone(getFeatureSetResponse)
		self.PostRequest(IsAnselSDKIntegrationAvailableRequest(), 'isAnselSDKIntegrationAvailableRequest')		
		response = self.AwaitMessage(IsAnselSDKIntegrationAvailableResponse, 2.0)
		self.assertIsNotNone(response)
		self.assertEqual(response.status, kOk if getFeatureSetResponse.sdkDetected else kFailed)

	def test_step032_enableGridOfThirds(self):
		# create a session, enable and disable grid of thirds
		with AnselSession(self, AnselSession.ANSEL):
			request = SetGridOfThirdsEnabledRequest()
			for enabled in [True, False]:
				request.enabled = enabled
				self.PostRequest(request, 'setGridOfThirdsEnabledRequest')
				response = self.AwaitMessage(SetGridOfThirdsEnabledResponse, 2.0)
				self.assertIsNotNone(response)
				self.assertEqual(response.status, kOk)

	def test_step033_getProcessInfoRequest(self):
		# check GetProcessInfoRequest and that the process path returned is
		# as expected
		self.PostRequest(GetProcessInfoRequest(), 'getProcessInfoRequest')
		response = self.AwaitMessage(GetProcessInfoResponse, 2.0)
		self.assertIsNotNone(response)
		self.assertEqual(response.status, kOk)
		self.assertEqual(os.path.abspath(response.processPath), os.path.abspath(self.__class__.ProcessToTestOnPath))

	def test_step034_getAnselEnabledRequest(self):
		def assertAnselStatus(status):
			self.PostRequest(GetAnselEnabledRequest(), 'getAnselEnabledRequest')
			response = self.AwaitMessage(GetAnselEnabledResponse, 2.0)
			self.assertIsNotNone(response)
			self.assertEqual(response.enabled, status)
		assertAnselStatus(False)
		with AnselSession(self, AnselSession.ANSEL):
			assertAnselStatus(True)
		assertAnselStatus(False)

	def test_step035_getAnselShotPermissionsResponse(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.PostRequest(GetAnselShotPermissionsRequest(), 'getAnselShotPermissionsRequest')
			response = self.AwaitMessage(GetAnselShotPermissionsResponse, 2.0)
			self.assertIsNotNone(response)
			# in the test app we use, we know that:
			# * it's Ansel integrated
			# * all capture types are allowed
			self.assertTrue(response.isIntegrationDetected)
			self.assertTrue(response.isHDRAllowed if not self.__class__.processImageName in NoHDRTestProcessNames else not response.isHDRAllowed)
			for shotType in [kRegular, kRegularStereo, kHighres, kPanorama360Mono, kPanorama360Stereo]:
				self.assertTrue(response.isShotAllowed[shotType])

	def getScreenResolutionRequest(self, mode):
		# we check GetScreenResolutionRequest here
		# it should return kOk status and the screen buffer size
		# which we unfortunately have to hard code here
		with AnselSession(self, mode):
			self.PostRequest(GetScreenResolutionRequest(), 'getScreenResolutionRequest')
			response = self.AwaitMessage(GetScreenResolutionResponse, 2.0)
			self.assertIsNotNone(response)
			self.assertEqual(response.status, kOk)
			self.assertEqual([response.xResolution, response.yResolution], [self.__class__.xResolution, self.__class__.yResolution])

	def test_step036_getScreenResolutionRequestAnsel(self):
		self.getScreenResolutionRequest(AnselSession.ANSEL)

	def test_step037_getScreenResolutionRequestMods(self):
		self.getScreenResolutionRequest(AnselSession.MODSONLY)

	def test_step038_getHighresResolutionListRequest(self):
		# we check GetHighresResolutionListRequest here
		# which should 
		# * only work while in Ansel session
		# * have more than one available resolution
		with AnselSession(self, AnselSession.ANSEL):
			self.PostRequest(GetHighresResolutionListRequest(), 'getHighresResolutionListRequest')
			response = self.AwaitMessage(GetHighresResolutionListResponse, 2.0)
			self.assertIsNotNone(response)
			self.assertGreater(len(response.resolutions), 1)

	def test_step039_setHighQuality(self):
		with AnselSession(self, AnselSession.ANSEL):
			# Enable High Quality
			self.setHighQuality(1)
			response = self.takeRegularScreenshot(golden=self.__class__.goldenPath + "\\GoldenHighQuality.png", psnr=35, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))
			# Disable High Quality
			self.setHighQuality(0)
			response = self.takeRegularScreenshot(golden=self.__class__.goldenPath + "\\GoldenRegular.png", psnr=35, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))

	def test_step050_checkLocalization(self):
		# we check localization here
		# we try every supported language - (langid, sublangid) pair
		# we check some filter name and its attribute against hardcoded
		# list of strings
		localeToLangSublandIds = { "en-US": [0x09, 0x01], "de-DE": [0x07, 0x01], "es-ES": [0x0a, 0x03], "es-MX": [0x0a, 0x02], "fr-FR": [0x0c, 0x01], 
							"it-IT": [0x10, 0x01], "ru-RU": [0x19, 0x01], "zh-CHS": [0x04, 0x02], "zh-CHT": [0x04, 0x01], "ja-JP": [0x11, 0x01], 
							"cs-CZ": [0x05, 0x01], "da-DK": [0x06, 0x01], "el-GR": [0x08, 0x01], "en-UK": [0x09, 0x02], "fi-FI": [0x0b, 0x01], 
							"hu": [0x0e, 0x01],	"ko-KR": [0x12, 0x01], "nl-NL": [0x13, 0x01], "nb-NO": [0x14, 0x01], "pl": [0x15, 0x01], 
							"pt-PT": [0x16, 0x02], "pt-BR": [0x16, 0x01], "sl-SI": [0x24, 0x01], "sk-SK": [0x1b, 0x01], "sv-SE": [0x1d, 0x01], 
							"th-TH": [0x1e, 0x01], "tr-TR": [0x1f, 0x01] }

		blackAndWhiteFilterNames = { "en-US": u"Black & White", "ru-RU": u"Чёрно-белый", "fr-FR": u"Noir et blanc", "de-DE": u"Schwarzweiß", "it-IT": u"Bianco e nero",
							"es-ES": u"Blanco y negro", "es-MX": u"Blanco y negro", "zh-CHS": u"黑白", "zh-CHT": u"黑白", "ja-JP": u"グレースケール", "cs-CZ": u"Černobílá",
							"da-DK": u"Sort og hvid", "el-GR": u"Ασπρόμαυρο", "en-UK": u"Black & White", "fi-FI": u"Mustavalkoinen", "hu": u"Fekete-fehér", "ko-KR": u"블랙 앤 화이트",
							"nl-NL": u"Zwart-wit", "nb-NO": u"Svart-hvitt", "pl": u"Czarno-białe", "pt-PT": u"Preto e branco", "pt-BR": u"Preto e branco", "sl-SI": u"Črno-belo",
							"sk-SK": u"Čiernobiele", "sv-SE": u"Svartvit", "th-TH": u"ดำและขาว", "tr-TR": u"Siyah & Beyaz" }

		blackAndWhiteIntensity = { "en-US": u"Intensity", "cs-CZ": u"Intenzita", "da-DK": u"Intensitet", "de-DE": u"Intensität", "el-GR": u"Ένταση", "en-UK": u"Intensity",
							"es-ES": u"Intensidad", "es-MX": u"Intensidad", "fi-FI": u"Voimakkuus", "fr-FR": u"Intensité", "hu": u"Intenzitás", "it-IT": u"Intensità", "ja-JP": u"明度",
							"ko-KR": u"강도", "nl-NL": u"Intensiteit", "nb-NO": u"Intensitet", "pl": u"Intensywność", "pt-PT": u"Intensidade", "pt-BR": u"Intensidade",
							"ru-RU": u"Насыщенность", "sk-SK": u"Intenzita", "sl-SI": u"Intenzivnost", "sv-SE": u"Intensitet", "th-TH": u"ความเข้ม", "tr-TR": u"Yoğunluk", "zh-CHS": u"强度",
							"zh-CHT": u"強度" }

		self.assertEqual(len(localeToLangSublandIds), len(blackAndWhiteFilterNames))
		self.assertEqual(len(blackAndWhiteIntensity), len(blackAndWhiteFilterNames))
		# work in progress
		def setLanguage(locale):
			request = SetLangIdRequest()
			request.lang, request.subLang = tuple(localeToLangSublandIds[locale])
			self.PostRequest(request, 'setLangIdRequest')
			response = self.AwaitMessage(SetLangIdResponse, 2.0)
			self.assertIsNotNone(response)
			self.assertEqual(response.status, kOk)

		with AnselSession(self, AnselSession.ANSEL):
			filterNameToId = self.getFilterList()
			blackAndWhiteFilterId = filterNameToId[blackAndWhiteFilterNames['en-US']]

			for locale in localeToLangSublandIds:
				setLanguage(locale)
				filterIdToName = dict((v, k) for k, v in self.getFilterList().iteritems())
				name = filterIdToName[blackAndWhiteFilterId]
				# check filter name is translated
				self.assertEqual(filterIdToName[blackAndWhiteFilterId], blackAndWhiteFilterNames[locale])
				# check filter attribute is translated
				response = self.setFilter(blackAndWhiteFilterId, 0)
				self.assertEqual(response.filterProperties.controls[0].displayName, blackAndWhiteIntensity[locale])
				self.setFilter('None', 0)

		# reset locale to en-US at the end of the test
		with AnselSession(self, AnselSession.ANSEL):
			setLanguage('en-US')

	# ------------------------
	# CAPTURE TESTS START HERE
	# ------------------------

	def test_step100_regularScreenshot(self):
		# create a session and take a regular screenshot
		with AnselSession(self, AnselSession.ANSEL):
			response = self.takeRegularScreenshot(golden=self.__class__.goldenPath + "\\GoldenRegular.png", psnr=35, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))

	def test_step101_anotherRegularScreenshot(self):
		self.test_step100_regularScreenshot()

	def test_step102_regularScreenshotPair(self):
		# create a session and take two regular screenshots in a row
		with AnselSession(self, AnselSession.ANSEL):
			self.takeRegularScreenshot(golden=self.__class__.goldenPath + "\\GoldenRegular.png", psnr=35, generateGoldens=generateGoldens)
			self.takeRegularScreenshot(golden=self.__class__.goldenPath + "\\GoldenRegular.png", psnr=35, generateGoldens=generateGoldens)

	def test_step103_highres2x(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.captureSuperResolution(2, enhance=False, golden=self.__class__.goldenPath + "\\GoldenHighres2x.jpg", psnr=35, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))

	def test_step104_highres2xThumbnail(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.captureSuperResolution(2, enhance=False, generateThumbnail=True)
			self.assertTrue(self.thumbnailExists(response))

	def test_step105_highres2xEnhance(self):
		# even though the test app doesn't have any screenspace effects which could
		# make 'Enhance' option valuable, we still test that it works in principle
		with AnselSession(self, AnselSession.ANSEL):
			response = self.captureSuperResolution(2, enhance=True, golden=self.__class__.goldenPath + "\\GoldenHighres2xEnhance.jpg", psnr=30, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))

	def test_step106_highres2xFovRoll(self):
		# we test if super resolution shot captures fov and roll changes
		with AnselSession(self, AnselSession.ANSEL):
			self.setFov(100)
			self.setRoll(45)
			sleep(0.1)
			self.captureSuperResolution(2, enhance=False, golden=self.__class__.goldenPath + "\\GoldenHighres2xFovRoll.jpg", psnr=25, generateGoldens=generateGoldens)

	def test_step107_highres4x(self):
		# we test if we can capture 4x super resolution shot
		with AnselSession(self, AnselSession.ANSEL):
			response = self.captureSuperResolution(4, enhance=False, golden=self.__class__.goldenPath + "\\GoldenHighres4x.jpg", psnr=25, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))

	def test_step108_highres4xEnhance(self):
		# even though the test app doesn't have any screenspace effects which could
		# make 'Enhance' option valuable, we still test that it works in principle
		with AnselSession(self, AnselSession.ANSEL):
			response = self.captureSuperResolution(4, enhance=True, golden=self.__class__.goldenPath + "\\GoldenHighres4xEnhance.jpg", psnr=25, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))

	def test_step109_360mono_4096(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.capture360(4096, 'mono', golden=self.__class__.goldenPath + "\\Golden360Mono4096.jpg", psnr=25, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))

	def test_step110_360mono_4096_Thumbnail(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.capture360(4096, 'mono', generateThumbnail=True)
			self.assertTrue(self.thumbnailExists(response))

	def test_step111_360mono_8192(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.capture360(8192, 'mono', golden=self.__class__.goldenPath + "\\Golden360Mono8192.jpg", psnr=25, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))

	def test_step112_360stereo_4096(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.capture360(4096, 'stereo', golden=self.__class__.goldenPath + "\\Golden360Stereo4096.jpg", psnr=25, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))

	def test_step113_360stereo_4096_Thumbnail(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.capture360(4096, 'stereo', generateThumbnail=True)
			self.assertTrue(self.thumbnailExists(response))

	def test_step114_360stereo_8192(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.capture360(8192, 'stereo', golden=self.__class__.goldenPath + "\\Golden360Stereo8192.jpg", psnr=25, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))

	def test_step115_stereo(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.captureRegularStereo(golden=self.__class__.goldenPath + "\\GoldenStereo.jpg", psnr=25, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))

	def test_step116_stereoThumbnail(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.captureRegularStereo(generateThumbnail=True)
			self.assertTrue(self.thumbnailExists(response))
	
	@skipIfNoHDR
	def test_step117_regularExr(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.takeRegularScreenshot(isExr=True, generateThumbnail=True, golden=self.__class__.goldenPath + "\\GoldenRegularExr.exr", psnr=20, generateGoldens=generateGoldens)
			self.assertTrue(self.thumbnailExists(response))

	@skipIfNoHDR
	def test_step118_regularStereoExr(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.captureRegularStereo(isExr=True, golden=self.__class__.goldenPath + "\\GoldenRegularStereoExr.exr", psnr=20, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))

	@skipIfNoHDR
	def test_step119_regularStereoExrThumbnail(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.captureRegularStereo(isExr=True, generateThumbnail=True)
			self.assertTrue(self.thumbnailExists(response))

	@skipIfNoHDR
	def test_step120_360MonoExr(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.capture360(4096, 'mono', isExr=True, golden=self.__class__.goldenPath + "\\Golden360MonoExr.exr", psnr=20, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))

	@skipIfNoHDR
	def test_step121_360MonoExrThumbnail(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.capture360(4096, 'mono', isExr=True, generateThumbnail=True)
			self.assertTrue(self.thumbnailExists(response))

	@skipIfNoHDR
	def test_step122_360StereoExr(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.capture360(4096, 'stereo', isExr=True, golden=self.__class__.goldenPath + "\\Golden360StereoExr.exr", psnr=20, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))

	@skipIfNoHDR
	def test_step123_360StereoExr(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.capture360(4096, 'stereo', isExr=True, generateThumbnail=True)
			self.assertTrue(self.thumbnailExists(response))

	@skipIfNoHDR
	def test_step124_highresExr(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.captureSuperResolution(2, isExr=True, golden=self.__class__.goldenPath + "\\GoldenHighres2xExr.exr", psnr=25, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))

	@skipIfNoHDR
	def test_step125_highresExrEnhance(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.captureSuperResolution(2, isExr=True, golden=self.__class__.goldenPath + "\\GoldenHighres2xExrEnhance.exr", psnr=25, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))

	@skipIfNoHDR
	def test_step126_regularExrNoThumbnail(self):
		with AnselSession(self, AnselSession.ANSEL):
			response = self.takeRegularScreenshot(isExr=True, golden=self.__class__.goldenPath + "\\GoldenRegularExr.exr", psnr=20, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))

	@skipIfNoHDR
	def test_step127_RawExrIntensity(self):
		with AnselSession(self, AnselSession.ANSEL):
			responseexr = self.takeRegularScreenshot(isExr=True, golden=self.__class__.goldenPath + "\\GoldenRegularExrIntensity.exr", psnr=20, generateGoldens=generateGoldens)
			self.assertIsNotNone(responseexr)
			responsepng = self.takeRegularScreenshot(isExr=False, golden=self.__class__.goldenPath + "\\GoldenRegularIntensity.png", psnr=20, generateGoldens = generateGoldens)
			self.assertIsNotNone(responsepng)

			if not generateGoldens:
				self.assertTrue(compare_intensity_ranges(responseexr.absoluteFilePath, responsepng.absoluteFilePath, '.'))
			else:
				self.assertTrue(compare_intensity_ranges(self.__class__.goldenPath+"\\GoldenRegularExrIntensity.exr", self.__class__.goldenPath+"\\GoldenRegularIntensity.png", "."))

	# ------------------------
	# FILTER TESTS START HERE
	# ------------------------

	def setEachFilter(self, mode):
		# we try to set each filter in the slot 0 of the stack
		with AnselSession(self, mode):
			filterIdToName = dict((v, k) for k, v in self.getFilterList().iteritems())
			for filterId in filterIdToName:
				if mode == AnselSession.MODSONLY and TestArguments.override != "False" and string.find(filterId, 'BlacknWhite') >= 0:
					# Black and White filter will fail the hash check in this case, so pick a different filter
					continue
				response = self.setFilter(filterId, 0)
				self.assertEqual(response.status, kOk)
			# reset stack
			response = self.setFilter('None', 0)
			self.assertEqual(response.status, kOk)

	def test_step300_setEachFilter(self):
		self.setEachFilter(AnselSession.ANSEL)

	def test_step301_setEachFilter(self):
		self.setEachFilter(AnselSession.MODSONLY)

	def test_step302_setBlackAndWhite(self):
		with AnselSession(self, AnselSession.ANSEL):
			filterNameToId = self.getFilterList()
			blackAndWhiteFilterId = filterNameToId['Black & White']
			self.setFilter(blackAndWhiteFilterId, 0)
			if TestArguments.override == "False":
				goldenPath = self.__class__.goldenPath + "\\GoldenRegularBnW.png"
			elif TestArguments.override == "ProgFiles":
				goldenPath = self.__class__.goldenPath + "\\GoldenRegularBnW-green.png"
			else:
				goldenPath = self.__class__.goldenPath + "\\GoldenRegularBnW-red.png"
			response = self.takeRegularScreenshot(golden=goldenPath, psnr=35, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))
			self.setFilter('None', 0)

	def setTestingFilter(self, mode):
		with AnselSession(self, mode):
			self.setFilter(self.getFilterList()['Testing'], 0)
			self.setFilter('None', 0)

	def test_step303_setTestingFilter(self):
		self.setTestingFilter(AnselSession.ANSEL)

	def test_step304_setTestingFilter(self):
		self.setTestingFilter(AnselSession.MODSONLY)

	def test_step305_testDepth(self):
		if self.__class__.processImageName == VulkanProcessToTestOnImageName:
			raise unittest.SkipTest("Not supported by Vulkan test app")

		with AnselSession(self, AnselSession.ANSEL):
			testFilterId = self.getFilterList()['Testing']
			self.setFilter(testFilterId, 0)
			self.setFilterAttribute(testFilterId, stackIdx=0, controlId=0, value=1.0)
			response = self.takeRegularScreenshot(golden=self.__class__.goldenPath + "\\GoldenRegularDepth.png", psnr=35, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))
			self.setFilterAttribute(testFilterId, stackIdx=0, controlId=0, value=0.0)
			self.setFilter('None', 0)

	def test_step306_testRegularCaptureState(self):
		with AnselSession(self, AnselSession.ANSEL):
			testFilterId = self.getFilterList()['Testing']
			self.setFilter(testFilterId, 0)
			self.setFilterAttribute(testFilterId, stackIdx=0, controlId=1, value=1.0)
			response = self.takeRegularScreenshot(golden=self.__class__.goldenPath + "\\GoldenRegularMultiPartState.png", psnr=35, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))
			self.setFilterAttribute(testFilterId, stackIdx=0, controlId=1, value=0.0)
			self.setFilter('None', 0)

	def test_step307_testHighresCaptureState(self):
		with AnselSession(self, AnselSession.ANSEL):
			testFilterId = self.getFilterList()['Testing']
			self.setFilter(testFilterId, 0)
			self.setFilterAttribute(testFilterId, stackIdx=0, controlId=1, value=1.0)
			response = self.captureSuperResolution(2, enhance=False, golden=self.__class__.goldenPath + "\\GoldenHighres2xMultiPartState.jpg", psnr=25, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))
			self.setFilterAttribute(testFilterId, stackIdx=0, controlId=1, value=0.0)
			self.setFilter('None', 0)

	def test_step308_test360MonoCaptureState(self):
		with AnselSession(self, AnselSession.ANSEL):
			testFilterId = self.getFilterList()['Testing']
			self.setFilter(testFilterId, 0)
			self.setFilterAttribute(testFilterId, stackIdx=0, controlId=1, value=1.0)
			response = self.capture360(4096, 'mono', golden=self.__class__.goldenPath + "\\Golden360Mono4096MultiPartState.jpg", psnr=20, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))
			self.setFilterAttribute(testFilterId, stackIdx=0, controlId=1, value=0.0)
			self.setFilter('None', 0)

	def test_step309_test360StereoCaptureState(self):
		with AnselSession(self, AnselSession.ANSEL):
			testFilterId = self.getFilterList()['Testing']
			self.setFilter(testFilterId, 0)
			self.setFilterAttribute(testFilterId, stackIdx=0, controlId=1, value=1.0)
			response = self.capture360(4096, 'stereo', golden=self.__class__.goldenPath + "\\Golden360Stereo4096MultiPartState.jpg", psnr=20, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))
			self.setFilterAttribute(testFilterId, stackIdx=0, controlId=1, value=0.0)
			self.setFilter('None', 0)

	def test_step310_testRegularStereoCaptureState(self):
		with AnselSession(self, AnselSession.ANSEL):
			testFilterId = self.getFilterList()['Testing']
			self.setFilter(testFilterId, 0)
			self.setFilterAttribute(testFilterId, stackIdx=0, controlId=1, value=1.0)
			response = self.captureRegularStereo(golden=self.__class__.goldenPath + "\\GoldenStereoMultiPartState.jpg", psnr=25, generateGoldens=generateGoldens)
			self.setFilterAttribute(testFilterId, stackIdx=0, controlId=1, value=0.0)
			self.setFilter('None', 0)

	def test_step311_testScreenSize(self):
		with AnselSession(self, AnselSession.ANSEL):
			testFilterId = self.getFilterList()['Testing']
			self.setFilter(testFilterId, 0)
			self.setFilterAttribute(testFilterId, stackIdx=0, controlId=2, value=1.0)
			response = self.takeRegularScreenshot(golden=self.__class__.goldenPath + "\\GoldenRegularScreenSizeState.png", psnr=35, generateGoldens=generateGoldens)
			self.setFilterAttribute(testFilterId, stackIdx=0, controlId=2, value=0.0)
			self.setFilter('None', 0)

	def test_step312_testResetEntireStackAnsel(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.resetEntireStack()

	def test_step313_testGetStackInfoNoFiltersAnsel(self):
		with AnselSession(self, AnselSession.MODSONLY):
			self.assertStackIsEmpty()

	def test_step314_testResetEntireStackMods(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.resetEntireStack()

	def test_step315_testGetStackInfoNoFiltersMods(self):
		with AnselSession(self, AnselSession.MODSONLY):
			self.assertStackIsEmpty()

	def getStackInfoWithFilters(self, mode):
		with AnselSession(self, mode):
			MaxLen = 5
			filterIds = dict((v, k) for k, v in self.getFilterList().iteritems()).keys()
			self.assertGreater(len(filterIds), MaxLen - 1)
			tryAgain = True
			while tryAgain == True:
				filters = random.sample(filterIds, MaxLen)
				tryAgain = False
				for f in filters:
					if mode == AnselSession.MODSONLY and TestArguments.override != "False" and string.find(f, 'BlacknWhite') >= 0:
						# Black and White filter will fail the hash check in this case, so pick a set of filters that doesn't include it
						tryAgain = True
			for k in range(1, MaxLen + 1):
				# we set a set of filters on the stack in the known order defined as filters[:k]
				for i in range(k):
					self.setFilter(filters[i], i)
				# we get the stack info and compare the filter list - it should be the same
				response = self.getStackInfo()
				self.assertEqual(response.filterIds, filters[:k])
				# we reset entire stack (remove all filters)
				self.resetEntireStack()
				# we check that the stack is indeed empty after reset
				self.assertStackIsEmpty()
	
	def test_step316_testGetStackInfoWithFiltersAnsel(self):
		self.getStackInfoWithFilters(AnselSession.ANSEL)

	def test_step317_testGetStackInfoWithFiltersMods(self):
		self.getStackInfoWithFilters(AnselSession.MODSONLY)

	def insertFilterSuccess(self, mode):
		with AnselSession(self, mode):
			MaxLen = 5
			filterIds = dict((v, k) for k, v in self.getFilterList().iteritems()).keys()
			self.assertGreater(len(filterIds), MaxLen - 1)
			tryAgain = True
			while tryAgain == True:
				tryAgain = False
				filters = random.sample(filterIds, MaxLen)
				for f in filters:
					if mode == AnselSession.MODSONLY and TestArguments.override != "False" and string.find(f, "BlacknWhite") >= 0:
						# Black and White filter will fail the hash check in this case, so pick a set of filters that doesn't include it
						tryAgain = True
			for k in range(2, MaxLen - 1):
				for filter_permutation in itertools.permutations(filters, k):
					filter1 = filter_permutation[0]
					self.setFilter(filter1, 0)
					# wait until it is actually set
					for filt_id in filter_permutation[1:]:
						self.insertFilter(filt_id, 0)
					expectedStackList = list(filter_permutation[1:])
					expectedStackList.reverse()
					expectedStackList.append(filter1)
					response = self.getStackInfo()
					self.assertEqual(response.filterIds, expectedStackList)
					# reset stack afterwards
					self.resetEntireStack()
					# we check that the stack is indeed empty after reset
					self.assertStackIsEmpty()

	def test_step318_insertFilterSuccessAnsel(self):
		self.insertFilterSuccess(AnselSession.ANSEL)

	def test_step319_insertFilterSuccessMods(self):
		self.insertFilterSuccess(AnselSession.MODSONLY)

	def insertFilterFailure(self, mode):
		with AnselSession(self, mode):
			filterIds = dict((v, k) for k, v in self.getFilterList().iteritems()).keys()
			# we need at least 2 filters to perform the test
			self.assertGreater(len(filterIds), 1)
			i = 0
			tryAgain = True
			while tryAgain == True:
				tryAgain = False
				if mode == AnselSession.MODSONLY and TestArguments.override != "False" and string.find(filterIds[i], "BlacknWhite") >= 0:
					i = i + 1
					tryAgain = True
			self.setFilter(filterIds[i], 0)
			# try inserting at the end (not allowed) and after the end of the stack (also, not allowed)
			request = InsertFilterRequest()
			request.filterId, request.stackIdx = filterIds[i], 1
			self.PostRequest(request, 'insertFilterRequest')
			response = self.AwaitMessage(InsertFilterResponse, 2.0)
			self.assertIsNotNone(response)
			self.assertEqual(response.status, kIlwalidRequest)
			request.filterId, request.stackIdx = filterIds[i], 1
			self.PostRequest(request, 'insertFilterRequest')
			response = self.AwaitMessage(InsertFilterResponse, 2.0)
			self.assertIsNotNone(response)
			self.assertEqual(response.status, kIlwalidRequest)
			# reset stack afterwards
			self.resetEntireStack()
			self.assertStackIsEmpty()

	def test_step320_insertFilterFailureAnsel(self):
		self.insertFilterFailure(AnselSession.ANSEL)

	def test_step321_insertFilterFailureMods(self):
		self.insertFilterFailure(AnselSession.MODSONLY)

	def removeFilterSuccess(self, mode):
		with AnselSession(self, mode):
			filterIds = dict((v, k) for k, v in self.getFilterList().iteritems()).keys()
			# we need at least 2 filters to perform the test
			self.assertGreater(len(filterIds), 2)
			self.setFilter(filterIds[0], 0)
			self.removeFilter(0)
			self.assertStackIsEmpty()

	def test_step322_testRemoveFilterSuccess(self):
		self.removeFilterFailure(AnselSession.ANSEL)

	def test_step323_testRemoveFilterSuccess(self):
		self.removeFilterFailure(AnselSession.MODSONLY)

	def removeFilterFailure(self, mode):
		with AnselSession(self, mode):
			for i in range(2):
				request = RemoveFilterRequest()
				request.stackIdx = 0
				self.PostRequest(request, 'removeFilterRequest')
				response = self.AwaitMessage(RemoveFilterResponse, 2.0)
				self.assertIsNotNone(response)
				self.assertEqual(response.status, kIlwalidRequest)

	def test_step324_testRemoveFilterFailureAnsel(self):
		self.removeFilterFailure(AnselSession.ANSEL)

	def test_step325_testRemoveFilterFailureMods(self):
		self.removeFilterFailure(AnselSession.MODSONLY)

	def moveFilterSuccess(self, mode):
		with AnselSession(self, mode):
			# try every possible move sequence and check that filters are actually
			# in the expected order
			MaxLen = 5
			NumFilt = 2
			filterIds = dict((v, k) for k, v in self.getFilterList().iteritems()).keys()
			self.assertGreater(len(filterIds), MaxLen - 1)
			tryAgain = True
			while tryAgain == True:
				tryAgain = False
				filters = random.sample(filterIds, MaxLen)
				for f in filters:
					if mode == AnselSession.MODSONLY and TestArguments.override != "False" and string.find(f, "BlacknWhite") >= 0:
						# Black and White filter will fail the hash check in this case, so pick a set of filters that doesn't include it
						tryAgain = True
			for k in range(NumFilt, MaxLen):
				for permutation in itertools.permutations(range(k), k):
					for i in range(k):
						self.setFilter(filters[i], i)
					response = self.moveFilter(permutation)
					self.assertEqual(response.status, kOk)
					expectedStackList = [filters[:k][i] for i in permutation]
					response = self.getStackInfo()
					self.assertEqual(response.filterIds, expectedStackList)
					# reset stack afterwards
					self.resetEntireStack()
					self.assertStackIsEmpty()

	def test_step326_testMoveFilterSuccessAnsel(self):
		self.moveFilterFailure(AnselSession.ANSEL)

	def test_step327_testMoveFilterSuccessMods(self):
		self.moveFilterFailure(AnselSession.MODSONLY)

	def moveFilterFailure(self, mode):
		with AnselSession(self, mode):
			filterIds = dict((v, k) for k, v in self.getFilterList().iteritems()).keys()
			self.assertGreater(len(filterIds), 2)
			index = 0
			for i in range(3):
				tryAgain = True
				while tryAgain == True:
					tryAgain = False
					if mode == AnselSession.MODSONLY and TestArguments.override != "False" and string.find(filterIds[index], "BlacknWhite") >= 0:
						# Black and White filter will fail the hash check in this case, so pick a set of filters that doesn't include it
						index = index + 1
						tryAgain = True
				self.setFilter(filterIds[index], i)
				index = index + 1
			response = self.moveFilter([0, 0, 0])
			self.assertEqual(response.status, kIlwalidRequest)
			response = self.moveFilter([0, 1, 2])
			self.assertEqual(response.status, kOk)
			response = self.moveFilter([1, 1, 2])
			self.assertEqual(response.status, kIlwalidRequest)
			response = self.moveFilter([0, 1, 3])
			self.assertEqual(response.status, kIlwalidRequest)
			# reset stack afterwards
			self.resetEntireStack()
			self.assertStackIsEmpty()

	def test_step328_testMoveFilterFailureAnsel(self):
		self.moveFilterFailure(AnselSession.ANSEL)

	def test_step329_testMoveFilterFailureMods(self):
		self.moveFilterFailure(AnselSession.MODSONLY)
	
	def test_step330_testSetFilterAndAttributes(self):
		with AnselSession(self, AnselSession.ANSEL):
			testFilterId = self.getFilterList()['Stickers']
			self.setFilterAndAttributes(testFilterId, stackIdx=0, controlIds={1,2}, values={0.3,0.4})
			response = self.takeRegularScreenshot(golden=self.__class__.goldenPath + "\\GoldenSetFilterAndAttributes.png", psnr=35, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))
			self.setFilter('None', 0)


	# ------------------------
	# INPUT HANDLING TESTS START HERE
	# ------------------------

	def rotateCameraTest(self, right, down):
		def getSignedNumber(number, bitLength):
			mask = (2 ** bitLength) - 1
			if number & (1 << (bitLength - 1)):
				return number | ~mask
			else:
				return number & mask		
		WM_LBUTTONDOWN = 0x0201
		WM_LBUTTONUP = 0x0202
		WM_MOUSEMOVE = 0x0200
		response = self.takeRegularScreenshot()
		centerShot = response.absoluteFilePath
		# move camera:
		# first press left mouse button
		request = InputEventRequest()
		request.isDeltaCoords, request.wParam, request.lParam = True, 0, 0
		request.message = WM_LBUTTONDOWN
		self.PostRequest(request, 'inputEventRequest')
		# wait a bit - Ansel doesn't send responses for high-frequency requests like
		# InputEventRequest
		sleep(0.1)
		# now move mouse, do it twice
		request.lParam = getSignedNumber((int(right & 0xffff) & 0xffffffff) | (((down & 0xffff) << 16) & 0xffffffff), 32)
		request.message = WM_MOUSEMOVE
		for i in range(2):
			self.PostRequest(request, 'inputEventRequest')
			sleep(0.1)
		# release left mouse button
		request.isDeltaCoords, request.wParam, request.lParam = True, 0, 0
		request.message = WM_LBUTTONUP
		sleep(0.1)
		# take second screenshot and callwlate global motion vector
		response = self.takeRegularScreenshot()
		self.assertFalse(compare_images(centerShot, response.absoluteFilePath, '.', 32, False))

	def test_step400_moveRotateCameraDown(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.rotateCameraTest(0, 100)

	def test_step401_moveRotateCameraUp(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.rotateCameraTest(0, -100)

	def test_step402_moveRotateCameraRight(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.rotateCameraTest(100, 0)

	def test_step403_moveRotateCameraLeft(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.rotateCameraTest(-100, 0)

	def gamepadTest(self, axis, value):
		def setAxisValue(request, axis, value):
			if axis == 0:
				request.leftStickXValue = value
			elif axis == 1:
				request.leftStickYValue = value
			elif axis == 2:
				request.rightStickXValue = value
			elif axis == 3:
				request.rightStickYValue = value
			elif axis == 4:
				request.leftTriggerValue = value 
			elif axis == 5:
				request.rightTriggerValue = value

		response = self.takeRegularScreenshot()
		centerShot = response.absoluteFilePath
		request = InputEventRequest()
		# move axis 'axis' by value 'value' 
		request.wParam, request.lParam = 0, 0
		request.message = 0
		setAxisValue(request, axis, value)
		self.PostRequest(request, 'inputEventRequest')
		# wait a bit - Ansel doesn't send responses for high-frequency requests like
		# InputEventRequest
		sleep(0.2)
		setAxisValue(request, axis, 0.0)
		sleep(0.2)
		response = self.takeRegularScreenshot()
		self.assertFalse(compare_images(centerShot, response.absoluteFilePath, '.', 25, False))
		
	def test_step404_moveGamepadLeftStickLeft(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.gamepadTest(0, -0.5)

	def test_step405_moveGamepadLeftStickRight(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.gamepadTest(0, 0.5)

	def test_step406_moveGamepadRightStickUp(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.gamepadTest(3, 0.4)

	def test_step407_moveGamepadRightStickDown(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.gamepadTest(3, -0.4)

	def test_step408_moveGamepadRightStickLeft(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.gamepadTest(2, -0.4)

	def test_step409_moveGamepadRightStickRight(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.gamepadTest(2, 0.4)

	def test_step410_moveGamepadLeftTrigger(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.gamepadTest(4, 0.5)

	def test_step411_moveGamepadRightTrigger(self):
		with AnselSession(self, AnselSession.ANSEL):
			self.gamepadTest(5, 0.5)

# This is a base class to implement tests which do not require golden images and
# implement security testing (with ANSEL_FREESTYLE_MODE=APPROVED_ONLY)
@for_all_test_methods(catch_all_errors)
@for_all_assert_methods(count_asserts)
class SmokeTestModdingSelwrityBase(SmokeTestBase):
	@classmethod
	def setUpClass(cls, ProcessToTestOnImageName, ProcessToTestOnPath, ProcessToTestOnWorkDir):
		SmokeTestBase.setUpClass(ProcessToTestOnImageName, ProcessToTestOnPath, ProcessToTestOnWorkDir, None, [0, 0], Allowlisting=AllowlistingMode.APPROVED)
	@classmethod
	def tearDownClass(cls):
		SmokeTestBase.tearDownClass()

	# we just copy methods from SmokeTest base class
	setAnselEnabled = SmokeTest.__dict__['setAnselEnabled']
	getFilterList = SmokeTest.__dict__['getFilterList']
	setEachFilter = SmokeTest.__dict__['setEachFilter']
	setFilter = SmokeTest.__dict__['setFilter']
	setFilterAttribute = SmokeTest.__dict__['setFilterAttribute']

	def test_setEachFilterAnsel(self):
		self.setEachFilter(AnselSession.ANSEL)

	def test_setEachFilterModding(self):
		self.setEachFilter(AnselSession.MODSONLY)

	@unittest.skip('Skip this test for now')
	def test_noTestingFilterIsAvailable(self):
		with AnselSession(self, AnselSession.MODSONLY):
			filters = self.getFilterList()
			self.assertFalse('TestingUnselwre' in filters)

# This is a base class to implement Ansel SDK backwards compatibility tests
@for_all_test_methods(catch_all_errors)
@for_all_assert_methods(count_asserts)
class SmokeTestBackwardsCompatibilityBase(SmokeTestBase):
	@classmethod
	def setUpClass(cls, ProcessToTestOnImageName, ProcessToTestOnPath, ProcessToTestOnWorkDir):
		SmokeTestBase.setUpClass(ProcessToTestOnImageName, ProcessToTestOnPath, ProcessToTestOnWorkDir, SmokeTestImagesDX11, [1024, 768], Allowlisting=AllowlistingMode.APPROVED)
	@classmethod
	def tearDownClass(cls):
		SmokeTestBase.tearDownClass()

	# we just copy methods from SmokeTest base class
	setAnselEnabled = SmokeTest.__dict__['setAnselEnabled']
	setFov = SmokeTest.__dict__['setFov']
	setRoll = SmokeTest.__dict__['setRoll']
	captureSuperResolution = SmokeTest.__dict__['captureSuperResolution']
	
	def test_enableDisableAnsel(self):
		# just try entering and exiting Ansel
		with AnselSession(self, AnselSession.ANSEL):
			pass

	def test_highres2xFovRoll(self):
		# we test if super resolution shot captures fov and roll changes
		with AnselSession(self, AnselSession.ANSEL):
			self.setFov(100)
			self.setRoll(45)
			sleep(0.1)
			self.captureSuperResolution(2, enhance=False, golden=self.__class__.goldenPath + "\\GoldenHighres2xFovRoll.jpg", psnr=25, generateGoldens=generateGoldens)

# This is a class to implement Ansel NoSDK tests
# It should be run on an app that does not implement the Ansel SDK
@for_all_test_methods(catch_all_errors)
@for_all_assert_methods(count_asserts)
class SmokeTestAnselNoSDK(SmokeTestBase):
	@classmethod
	def setUpClass(cls, ProcessToTestOnImageName, ProcessToTestOnPath, ProcessToTestOnWorkDir, GoldenPath, Resolution):
		SmokeTestBase.setUpClass(ProcessToTestOnImageName, ProcessToTestOnPath, ProcessToTestOnWorkDir, GoldenPath, Resolution, Allowlisting=AllowlistingMode.APPROVED)
	@classmethod
	def tearDownClass(cls):
		SmokeTestBase.tearDownClass()

	# method decorator to skip certain tests if we're running DX9
	def skipIfCantOverride(func):
		def wrapper(self):
			if TestArguments.override != "False":
				raise unittest.SkipTest("Not supported by app with NoSDK")
			return func(self)
		return wrapper

	# we just copy methods from SmokeTest base class
	setAnselEnabled = SmokeTest.__dict__['setAnselEnabled']
	getFilterList = SmokeTest.__dict__['getFilterList']
	setFilter = SmokeTest.__dict__['setFilter']
	setFilterAttribute = SmokeTest.__dict__['setFilterAttribute']
	takeRegularScreenshot = SmokeTest.__dict__['takeRegularScreenshot']
	thumbnailExists = SmokeTest.__dict__['thumbnailExists']

	def test_enableDisableAnsel(self):
		# just try entering and exiting Ansel
		with AnselSession(self, AnselSession.MODSONLY):
			pass

	@skipIfCantOverride
	def test_setBlackAndWhite(self):
		with AnselSession(self, AnselSession.MODSONLY):
			filterNameToId = self.getFilterList()
			blackAndWhiteFilterId = filterNameToId['Black & White']
			self.setFilter(blackAndWhiteFilterId, 0)
			sleep(0.25) # Give Ansel NoSDK time to exit lightweight mode & apply the filter
			response = self.takeRegularScreenshot(golden=self.__class__.goldenPath + "\\GoldenRegularBnW.png", psnr=35, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))
			self.setFilter('None', 0)
	
	def test_setColor(self):
		with AnselSession(self, AnselSession.MODSONLY):
			filterNameToId = self.getFilterList()
			colorFilterId = filterNameToId['Color']
			self.setFilter(colorFilterId, 0)
			sleep(0.25) # Give Ansel NoSDK time to exit lightweight mode & apply the filter
			response = self.takeRegularScreenshot(golden=self.__class__.goldenPath + "\\GoldenRegularColor.png", psnr=35, generateGoldens=generateGoldens)
			self.assertFalse(self.thumbnailExists(response))
			self.setFilter('None', 0)

# The following specializations of the SmokeTest class are here
# to run SmokeTest test suite on 32 bit and 64 bit test applications
# * DX11 32 bit
# * DX11 64 bit
# * DX9 32 bit
# * DX9 64 bit
# * Vulkan 32 bit
# * Vulkan 64 bit
# * DX11 32 bit with APPROVED_ONLY to test filter security features
# * DX11 64 bit with APPROVED_ONLY to test filter security features
# * DX11 32 bit with Ansel SDK 1.1, 1.2, 1.3, 1.4 (1.5 is already tested in the tests above)
# * DX11 64 bit with Ansel SDK 1.1, 1.2, 1.3, 1.4 (1.5 is already tested in the tests above)

# A note about TestCaseFactory function. It generates a class, which implements a Python unittest.TestCase and is considered 
# a separate test suite in this Ansel testing framework
#
# TestSuiteName = TestCaseFactory('TestSuiteName, BaseClass, skipMessage=None, catchAllErrors=True, countAsserts=True, args...) 

# is identical to 
#
# @unittest.skip(skipMessage) <- if skipMessage is not None
# @for_all_assert_methods(count_asserts) <- if countAsserts is True
# @for_all_test_methods(catch_all_errors) <- if catchAllErrors is True
# class TestSuiteName(unittest.TestCase, BaseClass):
#		def setUpClass(cls):
#			BaseClass.setUpClass(args...)
#		def tearDownClass(cls):
#			BaseClass.tearDownClass()


SmokeTest32DX11 = TestCaseFactory('SmokeTest32DX11', SmokeTest, None, True, True,
											DX11ProcessToTestOnImageName, DX11ProcessToTestOnPath32[AnselSDKLatestVersion], 
											DX11ProcessToTestOnWorkDir[AnselSDKLatestVersion], SmokeTestImagesDX11, Resolution=[1024, 768])

SmokeTest64DX11 = TestCaseFactory('SmokeTest64DX11', SmokeTest, None, True, True,
											DX11ProcessToTestOnImageName, DX11ProcessToTestOnPath64[AnselSDKLatestVersion], 
											DX11ProcessToTestOnWorkDir[AnselSDKLatestVersion], SmokeTestImagesDX11, Resolution=[1024, 768])

SmokeTest32DX9 = TestCaseFactory('SmokeTest32DX9', SmokeTest, None, True, True,
											DX9ProcessToTestOnImageName, DX9ProcessToTestOnPath32, DX9ProcessToTestOnWorkDir, SmokeTestImagesDX9, Resolution=[640, 480])

SmokeTest64DX9 = TestCaseFactory('SmokeTest64DX9', SmokeTest, None, True, True,
											DX9ProcessToTestOnImageName, DX9ProcessToTestOnPath64, DX9ProcessToTestOnWorkDir, SmokeTestImagesDX9, Resolution=[640, 480])

SmokeTest32Vulkan = TestCaseFactory('SmokeTest32Vulkan', SmokeTest, None, True, True,
											VulkanProcessToTestOnImageName, VulkanProcessToTestOnPath32, VulkanProcessToTestOnWorkDir, SmokeTestImagesVulkan, Resolution=[800, 600])

SmokeTest64Vulkan = TestCaseFactory('SmokeTest64Vulkan', SmokeTest, None, True, True,
											VulkanProcessToTestOnImageName, VulkanProcessToTestOnPath64, VulkanProcessToTestOnWorkDir, SmokeTestImagesVulkan, Resolution=[800, 600])

SmokeTest32DX12 = TestCaseFactory('SmokeTest32DX12', SmokeTest, None, True, True,
											DX12ProcessToTestOnImageName, DX12ProcessToTestOnPath32, DX12ProcessToTestOnWorkDir32, SmokeTestImagesDX12, Resolution=[800,600])

SmokeTest64DX12 = TestCaseFactory('SmokeTest64DX12', SmokeTest, None, True, True,
											DX12ProcessToTestOnImageName, DX12ProcessToTestOnPath64, DX12ProcessToTestOnWorkDir64, SmokeTestImagesDX12, Resolution=[800,600])

SmokeTest32DX11Selwrity = TestCaseFactory('SmokeTest32DX11Selwrity', SmokeTestModdingSelwrityBase, None, True, True,
											DX11ProcessToTestOnImageName,
											DX11ProcessToTestOnPath32[AnselSDKLatestVersion], 
											DX11ProcessToTestOnWorkDir[AnselSDKLatestVersion])

SmokeTest64DX11Selwrity = TestCaseFactory('SmokeTest64DX11Selwrity', SmokeTestModdingSelwrityBase, None, True, True,
											DX11ProcessToTestOnImageName, 
											DX11ProcessToTestOnPath64[AnselSDKLatestVersion], 
											DX11ProcessToTestOnWorkDir[AnselSDKLatestVersion])

SmokeTest32DX11BackwardsCompatibilitySDK11 = TestCaseFactory('SmokeTest32DX11BackwardsCompatibilitySDK11', SmokeTestBackwardsCompatibilityBase,  None, True, True,
																DX11ProcessToTestOnImageName, 
																DX11ProcessToTestOnPath32['11'], 
																DX11ProcessToTestOnWorkDir['11'])

SmokeTest64DX11BackwardsCompatibilitySDK11 = TestCaseFactory('SmokeTest64DX11BackwardsCompatibilitySDK11', SmokeTestBackwardsCompatibilityBase,  None, True, True,
																DX11ProcessToTestOnImageName, 
																DX11ProcessToTestOnPath64['11'], 
																DX11ProcessToTestOnWorkDir['11'])

SmokeTest32DX11BackwardsCompatibilitySDK12 = TestCaseFactory('SmokeTest32DX11BackwardsCompatibilitySDK12', SmokeTestBackwardsCompatibilityBase,  None, True, True,
																DX11ProcessToTestOnImageName, 
																DX11ProcessToTestOnPath32['12'], 
																DX11ProcessToTestOnWorkDir['12'])

SmokeTest64DX11BackwardsCompatibilitySDK12 = TestCaseFactory('SmokeTest64DX11BackwardsCompatibilitySDK12', SmokeTestBackwardsCompatibilityBase,  None, True, True,
																DX11ProcessToTestOnImageName, 
																DX11ProcessToTestOnPath64['12'], 
																DX11ProcessToTestOnWorkDir['12'])

SmokeTest32DX11BackwardsCompatibilitySDK13 = TestCaseFactory('SmokeTest32DX11BackwardsCompatibilitySDK13', SmokeTestBackwardsCompatibilityBase,  None, True, True,
																DX11ProcessToTestOnImageName, 
																DX11ProcessToTestOnPath32['13'], 
																DX11ProcessToTestOnWorkDir['13'])

SmokeTest64DX11BackwardsCompatibilitySDK13 = TestCaseFactory('SmokeTest64DX11BackwardsCompatibilitySDK13', SmokeTestBackwardsCompatibilityBase,  None, True, True,
																DX11ProcessToTestOnImageName, 
																DX11ProcessToTestOnPath64['13'], 
																DX11ProcessToTestOnWorkDir['13'])

SmokeTest32DX11BackwardsCompatibilitySDK14 = TestCaseFactory('SmokeTest32DX11BackwardsCompatibilitySDK14', SmokeTestBackwardsCompatibilityBase,  None, True, True,
																DX11ProcessToTestOnImageName, 
																DX11ProcessToTestOnPath32['14'], 
																DX11ProcessToTestOnWorkDir['14'])

SmokeTest64DX11BackwardsCompatibilitySDK14 = TestCaseFactory('SmokeTest64DX11BackwardsCompatibilitySDK14', SmokeTestBackwardsCompatibilityBase,  None, True, True,
																DX11ProcessToTestOnImageName, 
																DX11ProcessToTestOnPath64['14'], 
																DX11ProcessToTestOnWorkDir['14'])

SmokeTest32DX11BackwardsCompatibilitySDK15 = TestCaseFactory('SmokeTest32DX11BackwardsCompatibilitySDK15', SmokeTestBackwardsCompatibilityBase,  None, True, True,
																DX11ProcessToTestOnImageName, 
																DX11ProcessToTestOnPath32['15'], 
																DX11ProcessToTestOnWorkDir['15'])

SmokeTest64DX11BackwardsCompatibilitySDK15 = TestCaseFactory('SmokeTest64DX11BackwardsCompatibilitySDK15', SmokeTestBackwardsCompatibilityBase,  None, True, True,
																DX11ProcessToTestOnImageName, 
																DX11ProcessToTestOnPath64['15'], 
																DX11ProcessToTestOnWorkDir['15'])

#Note that the NoSDK tests use APICs as their test app, because they require a freestyle-enabled app without ansel integrated.
SmokeTest64DX11NoSDK = TestCaseFactory('SmokeTest64DX11NoSDK', SmokeTestAnselNoSDK, None, True, True,
                                            DX11NoSDKProcessToTestOnImageName, DX11NoSDKProcessToTestOnPath64,
                                            DX11NoSDKProcessToTestOnWorkDir64, SmokeTestImagesDX11NoSDK, Resolution = [1920, 1080])

SmokeTest32DX11NoSDK = TestCaseFactory('SmokeTest32DX11NoSDK', SmokeTestAnselNoSDK, None, True, True,
                                            DX11NoSDKProcessToTestOnImageName, DX11NoSDKProcessToTestOnPath32,
                                            DX11NoSDKProcessToTestOnWorkDir32, SmokeTestImagesDX11NoSDK, Resolution = [1920, 1080])

SmokeTest64DX9NoSDK = TestCaseFactory('SmokeTest64DX9NoSDK', SmokeTestAnselNoSDK, None, True, True,
                                            DX9NoSDKProcessToTestOnImageName, DX9NoSDKProcessToTestOnPath64,
                                            DX9NoSDKProcessToTestOnWorkDir64, SmokeTestImagesDX9NoSDK, Resolution = [1920, 1080])

SmokeTest32DX9NoSDK = TestCaseFactory('SmokeTest32DX9NoSDK', SmokeTestAnselNoSDK, None, True, True,
                                            DX9NoSDKProcessToTestOnImageName, DX9NoSDKProcessToTestOnPath32,
                                            DX9NoSDKProcessToTestOnWorkDir32, SmokeTestImagesDX9NoSDK, Resolution = [1920, 1080])

try:
	if __name__ == '__main__':
		argv = TestArguments.args
		if is_running_under_teamcity():
			runner = TeamcityTestRunner()
			unittest.main(testRunner=runner, argv=argv, exit=False)
			print ('Assertions statistics: %d passed, %d failed (%d total)' % (Utils.AssertionPassCounter, Utils.AssertionFailCounter, Utils.AssertionCounter))
		else:
			with open(LogFile, "w") as f:
				runner = unittest.TextTestRunner(f, verbosity=2)
				unittest.main(testRunner=runner, argv=argv, exit=False)
				print ('Assertions statistics: %d passed, %d failed (%d total)' % (Utils.AssertionPassCounter, Utils.AssertionFailCounter, Utils.AssertionCounter))
finally:
	pass


