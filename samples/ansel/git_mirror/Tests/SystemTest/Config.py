#!/usr/bin/python
import os
import sys
import argparse
import string

#TODO: Also fix ToolsTests.py to use TestArguments to get to LwCamera folder path
# specify paths here
parser = argparse.ArgumentParser(description='Run the Ansel tests.', epilog='Should only be run via one of the run*.bat files')
parser.add_argument('-override', default="False", choices=["False", "ProgFiles", "AppFolder"])
parser.add_argument('-defaultFolder')
parser.add_argument('args', nargs=argparse.REMAINDER)
TestArguments = parser.parse_args()
TestArguments.args.insert(0, sys.argv[0])

if string.find(TestArguments.defaultFolder, 'System32\\DriverStore\\FileRepository') != -1:
	TestArguments.canModifyDefaultFolder = False
else:
	TestArguments.canModifyDefaultFolder = True

scriptDir = os.path.dirname(os.path.realpath(__file__))
D3DRegPath = 'package-links/d3dreg/d3dreg.exe'
AnselSourcePath = scriptDir + '/artifacts'

# Only use these for ToolsTests (Other Smoke tests get their paths in the setUpClass function for the individual Smoke Tests)
# Note: ToolsTests can only be run on the default paths. They cannot be run from the app folder.
# This is because the ToolsTests don't really have a separate folder for themselves. 
# The tools are simply triggered from where the driver installs them (which is the default folder).
AnselToolsPath = TestArguments.defaultFolder + '/'
LwCameraEnablePath = AnselToolsPath + 'LwCameraEnable.exe'
HighresBlender32Path = AnselToolsPath + 'HighresBlender32.exe'
HighresBlender64Path = AnselToolsPath + 'HighresBlender64.exe'
SphericalEquirect32Path = AnselToolsPath + 'SphericalEquirect32.exe'
SphericalEquirect64Path = AnselToolsPath + 'SphericalEquirect64.exe'
LwImageColwert32Path = AnselToolsPath + 'LwImageColwert32.exe'
LwImageColwert64Path = AnselToolsPath + 'LwImageColwert64.exe'
YAMLFXC32Path = AnselToolsPath + 'YAMLFXC32.exe'
YAMLFXC64Path = AnselToolsPath + 'YAMLFXC64.exe'

VulkanProcessToTestOnImageName = 'VkBallsAnsel.exe'
VulkanProcessToTestOnWorkDir = 'package-links/AnselIntegrationTestAppVulkan'
VulkanProcessToTestOnPath32 = VulkanProcessToTestOnWorkDir + '/Win32/Release/' + VulkanProcessToTestOnImageName
VulkanProcessToTestOnPath64 = VulkanProcessToTestOnWorkDir + '/x64/Release/' + VulkanProcessToTestOnImageName
DX9ProcessToTestOnImageName = 'AnselIntegrationTestApp9.exe'
DX9ProcessToTestOnWorkDir = 'package-links/AnselIntegrationTestAppDX9'
DX9ProcessToTestOnPath32 = DX9ProcessToTestOnWorkDir + '/Win32/release/' + DX9ProcessToTestOnImageName
DX9ProcessToTestOnPath64 = DX9ProcessToTestOnWorkDir + '/x64/release/' + DX9ProcessToTestOnImageName
DX11NoSDKProcessToTestOnImageName = 'TslGame.exe'
DX11NoSDKProcessToTestOnWorkDir32 = 'package-links/AnselNoSDKTestApp11/Win32'
DX11NoSDKProcessToTestOnWorkDir64 = 'package-links/AnselNoSDKTestApp11/x64'
DX11NoSDKProcessToTestOnPath32 = DX11NoSDKProcessToTestOnWorkDir32 + '/' + DX11NoSDKProcessToTestOnImageName
DX11NoSDKProcessToTestOnPath64 = DX11NoSDKProcessToTestOnWorkDir64 + '/' + DX11NoSDKProcessToTestOnImageName
DX9NoSDKProcessToTestOnImageName = 'SC2.exe'
DX9NoSDKProcessToTestOnWorkDir32 = 'package-links/AnselNoSDKTestApp9/Win32'
DX9NoSDKProcessToTestOnWorkDir64 = 'package-links/AnselNoSDKTestApp9/x64'
DX9NoSDKProcessToTestOnPath32 = DX9NoSDKProcessToTestOnWorkDir32 + '/' + DX9NoSDKProcessToTestOnImageName
DX9NoSDKProcessToTestOnPath64 = DX9NoSDKProcessToTestOnWorkDir64 + '/' + DX9NoSDKProcessToTestOnImageName
DX12ProcessToTestOnImageName = 'AnselIntegrationTestApp12.exe' # Multipart Shots don't work properly for this app, but it's deterministic, so it should still pass all tests.
DX12ProcessToTestOnWorkDir32 = 'package-links/AnselIntegrationTestAppDX12/Win32/Release'
DX12ProcessToTestOnWorkDir64 = 'package-links/AnselIntegrationTestAppDX12/x64/Release'
DX12ProcessToTestOnPath32 = DX12ProcessToTestOnWorkDir32 + '/' + DX12ProcessToTestOnImageName
DX12ProcessToTestOnPath64 = DX12ProcessToTestOnWorkDir64 + '/' + DX12ProcessToTestOnImageName
# This marks DX9, DX12, and Vulkan processes as the ones that don't have HDR rendering, so HDR tests will be skipped
NoHDRTestProcessNames = [DX9ProcessToTestOnImageName, VulkanProcessToTestOnImageName]

DX11ProcessToTestOnImageName = 'AnselIntegrationTestApp.exe'
# change this when the new Ansel SDK is out
AnselSDKLatestVersion = '16'
DX11ProcessToTestOnWorkDir = {}
DX11ProcessToTestOnPath32 = {}
DX11ProcessToTestOnPath64 = {}
for sdk in ['11', '12', '13', '14', '15', AnselSDKLatestVersion]:
	DX11ProcessToTestOnWorkDir[sdk] = 'package-links/AnselIntegrationTestAppDX11-SDK%s/src' % sdk
	DX11ProcessToTestOnPath32[sdk] = DX11ProcessToTestOnWorkDir[sdk] + '/Win32/release/' + DX11ProcessToTestOnImageName
	DX11ProcessToTestOnPath64[sdk] = DX11ProcessToTestOnWorkDir[sdk] + '/x64/release/' + DX11ProcessToTestOnImageName
MessageBusClientDll32 = 'package-links/MessageBusClientDll/bin/MessageBusClientDll32.dll'
MessageBusClientDll64 = 'package-links/MessageBusClientDll/bin/MessageBusClientDll64.dll'
CompareTool = 'package-links/LwImageCompare/bin/LwImageCompare64.exe'
ColwertTool = 'package-links/imagemagick/colwert.exe'
IdentifyTool = 'package-links/imagemagick/identify.exe'
ListDlls32 = 'package-links/listdlls/listdlls.exe'
ListDlls64 = 'package-links/listdlls/listdlls64.exe'
Handle32 = 'package-links/handle/handle.exe'
Handle64 = 'package-links/handle/handle64.exe'
Xcorr232 = 'package-links/xcorr2/xcorr2_32.exe'
Xcorr264 = 'package-links/xcorr2/xcorr2_64.exe'
# golden images
HighresBlenderMediumTiles = 'package-links/SystemTestsHighresBlenderMediumTiles'
HighresBlenderEnhanceTiles = 'package-links/SystemTestsHighresBlenderEnhanceTiles'
HighresBlenderExrTiles = 'package-links/SystemTestsHighresBlenderExrTiles'
SphericalEquirectTiles = 'package-links/SystemTestsSphericalEquirectTiles'
SphericalEquirectTilesExr = 'package-links/SystemTestsSphericalEquirectTilesExr'
LwImageColwertTiles = 'package-links/SystemTestsLwImageColwertTiles'
SmokeTestImagesVulkan = 'package-links/SmokeTestImagesVulkan'
SmokeTestImagesDX9 = 'package-links/SmokeTestImagesDX9'
SmokeTestImagesDX9NoSDK = 'package-links/SmokeTestImagesDX9NoSDK'
SmokeTestImagesDX11 = 'package-links/SmokeTestImagesDX11'
SmokeTestImagesDX11NoSDK = 'package-links/SmokeTestImagesDX11NoSDK'
SmokeTestImagesDX12 = 'package-links/SmokeTestImagesDX12'
#imagemagick identify format constants
IdentifyFormat = "-format"
IntensityFormatSeparator = " "
IntensityFormatArg = "%[fx:minima.intensity]"+IntensityFormatSeparator+"%[fx:maxima.intensity]"
