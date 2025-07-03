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

LogFile = 'ToolsTests.log'
generateGoldens = False

#@unittest.skip("disabling these test for now")
@for_all_test_methods(catch_all_errors)
@for_all_assert_methods(count_asserts)
class ToolsTests(unittest.TestCase):
	# HighresBlender tests
	def HighresBlenderRun(self, HighresBlenderPath, HighresBlenderOptions=[], goldenFileName='golden.jpg', ExactFileMatch=False, TilesPath=HighresBlenderMediumTiles, ResultFileExt='.jpg'):
		temp_path = tempfile.mkdtemp()
		result_path = temp_path + '/result'
		hb_run_list = [HighresBlenderPath, '--output', result_path] # we generate output always
		hb_run_list.extend(HighresBlenderOptions)
		test_process = run_detached_process(hb_run_list, TilesPath)
		retcode = test_process.wait()
		self.assertEqual(retcode, 0)
		result_file = result_path + ResultFileExt
		golden_file = TilesPath + '\\' + goldenFileName
		if ExactFileMatch == True:
			self.assertTrue(compare_files(result_file, golden_file))
		else:
			images_are_the_same = compare_images(result_file, golden_file, '.', 50, generateGoldens)
			self.assertTrue(images_are_the_same)
		if not generateGoldens:
			rmtree(temp_path)
	# test default multithreaded blending for 32-bit version
	def test_HighresBlender32MultithreadedNonConservative(self):
		self.HighresBlenderRun(HighresBlender32Path)
	# test default multithreaded blending for 64-bit version
	def test_HighresBlender64MultithreadedNonConservative(self):
		self.HighresBlenderRun(HighresBlender64Path)
	# test singlethreaded blending for 32-bit version
	def test_HighresBlender32SinglethreadedNonConservative(self):
		self.HighresBlenderRun(HighresBlender32Path, ['--threads', '1'])
	# test singlethreaded blending for 64-bit version
	def test_HighresBlender64SinglethreadedNonConservative(self):
		self.HighresBlenderRun(HighresBlender64Path, ['--threads', '1'])
	# test multithreaded memory-conservative blending for 32-bit version
	def test_HighresBlender32MultithreadedConservative(self):
		self.HighresBlenderRun(HighresBlender32Path, ['--conserve'])
	# test multithreaded memory-conservative blending for 64-bit version
	def test_HighresBlender64MultithreadedConservative(self):
		self.HighresBlenderRun(HighresBlender64Path, ['--conserve'])
	# test singlethreaded memory-conservative blending for 32-bit version
	def test_HighresBlender32SinglethreadedConservative(self):
		self.HighresBlenderRun(HighresBlender32Path, ['--conserve', '--threads', '1'])
	# test singlethreaded memory-conservative blending for 64-bit version
	def test_HighresBlender64SinglethreadedConservative(self):
		self.HighresBlenderRun(HighresBlender64Path, ['--conserve', '--threads', '1'])
	def test_HighresBlender64MultithreadedConservativeTestTags(self):
		self.HighresBlenderRun(HighresBlender64Path, ['--conserve', '--make', 'LWpu', '--model', 'LwCamera', '--type', 'SuperResolution', '--software', 'AnselIntegrationTestApp'], goldenFileName='golden-tagged.jpg', ExactFileMatch=True)
	# test multithreaded memory-conservative blending for 64-bit version with frequency transfer enhancement
	def test_HighresBlender32MultithreadedEnhance(self):
		self.HighresBlenderRun(HighresBlender32Path, 
			['--conserve', 
			'--make', 'LWPU', 
			'--model', 'Ansel', 
			'--type', 'SuperResolution', 
			'--software', '"The Witness"',
			'--freq-transfer-alpha', '0.75', 
			'--freq-transfer-input', 'regular.bmp'], 
			TilesPath=HighresBlenderEnhanceTiles)
	# test multithreaded memory-conservative blending for 64-bit version with frequency transfer enhancement
	def test_HighresBlender64MultithreadedEnhance(self):
		self.HighresBlenderRun(HighresBlender64Path, 
			['--conserve', 
			'--make', 'LWPU', 
			'--model', 'Ansel', 
			'--type', 'SuperResolution', 
			'--software', '"The Witness"', 
			'--freq-transfer-alpha', '0.75', 
			'--freq-transfer-input', 'regular.bmp'],			
			TilesPath=HighresBlenderEnhanceTiles)
	# test multithreaded memory-conservative blending for 64-bit version EXR
	def test_HighresBlender32MultithreadedExr(self):
		self.HighresBlenderRun(HighresBlender32Path, 
			['--conserve',
			'--make', 'LWPU', 
			'--model', 'Ansel', 
			'--type', 'SuperResolution', 
			'--software', '"The Witness"'], 
			TilesPath=HighresBlenderExrTiles,
			ResultFileExt='.exr',
			goldenFileName='golden.exr')
	# test multithreaded memory-conservative blending for 64-bit version EXR
	def test_HighresBlender64MultithreadedExr(self):
		self.HighresBlenderRun(HighresBlender64Path, 
			['--conserve',
			'--make', 'LWPU', 
			'--model', 'Ansel', 
			'--type', 'SuperResolution', 
			'--software', '"The Witness"'],			
			TilesPath=HighresBlenderExrTiles,
			ResultFileExt='.exr',
			goldenFileName='golden.exr')
	# test multithreaded memory-conservative blending for 64-bit version EXR
	def test_HighresBlender32MultithreadedExrEnhance(self):
		self.HighresBlenderRun(HighresBlender32Path, 
			['--conserve',
			'--make', 'LWPU', 
			'--model', 'Ansel', 
			'--type', 'SuperResolution', 
			'--software', '"The Witness"',
			'--freq-transfer-alpha', '0.75',
			'--freq-transfer-input', 'regular.exr'], 
			TilesPath=HighresBlenderExrTiles,
			ResultFileExt='.exr',
			goldenFileName='golden-enhance.exr')
	# test multithreaded memory-conservative blending for 64-bit version EXR
	def test_HighresBlender64MultithreadedExrEnhance(self):
		self.HighresBlenderRun(HighresBlender64Path, 
			['--conserve',
			'--make', 'LWPU', 
			'--model', 'Ansel', 
			'--type', 'SuperResolution', 
			'--software', '"The Witness"',
			'--freq-transfer-alpha', '0.75',
			'--freq-transfer-input', 'regular.exr'],			
			TilesPath=HighresBlenderExrTiles,
			ResultFileExt='.exr',
			goldenFileName='golden-enhance.exr')		

	# SphericalEquirect tests
	def SphericalEquirectRun(self, SphericalEquirectPath, SphericalEquirectOptions=[], DatasetPath=SphericalEquirectTiles, ResultFileExt='.jpg', goldenFileName='golden.jpg', ExactFileMatch=False):
		temp_path = tempfile.mkdtemp()
		result_path = temp_path + '/result' + ResultFileExt
		se_run_list = [SphericalEquirectPath, 'captures.txt', result_path] # we generate output always
		se_run_list.extend(SphericalEquirectOptions)
		test_process = run_detached_process(se_run_list, DatasetPath)
		retcode = test_process.wait()
		self.assertEqual(retcode, 0)
		golden_file = DatasetPath + '/' + goldenFileName
		if ExactFileMatch == True:
			self.assertTrue(compare_files(result_path, golden_file, generateGoldens))
		else:
			images_are_the_same = compare_images(result_path, golden_file, '.', 50, generateGoldens)
			self.assertTrue(images_are_the_same)
		rmtree(temp_path)
	# test multithreaded stitching for 32-bit version
	def test_SphericalEquirect32Multithreaded(self):
		self.SphericalEquirectRun(SphericalEquirect32Path)
	# test multithreaded stitching for 64-bit version
	def test_SphericalEquirect64Multithreaded(self):
		self.SphericalEquirectRun(SphericalEquirect64Path)
	# test multithreaded stitching for 32-bit version
	def test_SphericalEquirect32MultithreadedExr(self):
		self.SphericalEquirectRun(SphericalEquirect32Path, DatasetPath=SphericalEquirectTilesExr, goldenFileName='golden.exr', ResultFileExt='.exr')
	# test multithreaded stitching for 64-bit version
	def test_SphericalEquirect64MultithreadedExr(self):
		self.SphericalEquirectRun(SphericalEquirect64Path, DatasetPath=SphericalEquirectTilesExr, goldenFileName='golden.exr', ResultFileExt='.exr')
	# test multithreaded stitching for 32-bit version + adding GPano panorama tags (XMP)
	def test_SphericalEquirect32MultithreadedGPano(self):
		self.SphericalEquirectRun(SphericalEquirect32Path, ['--360'])
	# test multithreaded stitching for 64-bit version + adding GPano panorama tags (XMP)
	def test_SphericalEquirect64MultithreadedGPano(self):
		self.SphericalEquirectRun(SphericalEquirect64Path, ['--360'])
	# test singlethreaded stitching for 32-bit version
	def test_SphericalEquirect32Singlethreaded(self):
		self.SphericalEquirectRun(SphericalEquirect32Path, ['--threads', '1'])
	# test singlethreaded stitching for 64-bit version
	def test_SphericalEquirect64Singlethreaded(self):
		self.SphericalEquirectRun(SphericalEquirect64Path, ['--threads', '1'])
	def test_SphericalEquirect64SinglethreadedTestTags(self):
		self.SphericalEquirectRun(SphericalEquirect64Path, ['--threads', '1', '--360', '--make', 'LWpu', '--model', 'LwCamera', '--type', '360Mono', '--software', 'AnselIntegrationTestApp'], goldenFileName='golden-tagged.jpg', ExactFileMatch=True)

	# LwImageColwert tests
	def LwImageColwertRun(self, LwImageColwertPath, GoldenImage, LwImageColwertOptions, ResultExt, ExactFileMatch=False):
		temp_path = tempfile.mkdtemp()
		result_path = temp_path + '/result.' + ResultExt
		ic_run_list = [LwImageColwertPath]
		ic_run_list.extend(LwImageColwertOptions) # we generate output always
		ic_run_list.extend([result_path])
		test_process = run_detached_process(ic_run_list, LwImageColwertTiles)
		retcode = test_process.wait()
		self.assertEqual(retcode, 0)
		golden_file = LwImageColwertTiles + '/' + GoldenImage
		if ExactFileMatch == True:
			self.assertTrue(compare_files(result_path, golden_file, generateGoldens))
		else:
			images_are_the_same = compare_images(result_path, golden_file, '.', 50, generateGoldens)
			self.assertTrue(images_are_the_same)
		rmtree(temp_path)
	# test horizontal appending for 32-bit version
	def test_LwImageColwert32AppendHorizontally(self):
		self.LwImageColwertRun(LwImageColwert32Path, 'golden-horizontal.jpg', ['--append-horizontally', 'left.jpg', 'right.jpg'], 'jpg')
	# test horizontal appending for 64-bit version
	def test_LwImageColwert64AppendHorizontally(self):
		self.LwImageColwertRun(LwImageColwert64Path, 'golden-horizontal.jpg', ['--append-horizontally', 'left.jpg', 'right.jpg'], 'jpg')
	# test vertical appending for 32-bit version
	def test_LwImageColwert32AppendVertically(self):
		self.LwImageColwertRun(LwImageColwert32Path, 'golden-vertical.jpg', ['--append-vertically', 'left.jpg', 'right.jpg'], 'jpg')
	# test vertical appending for 64-bit version
	def test_LwImageColwert64AppendVertically(self):
		self.LwImageColwertRun(LwImageColwert64Path, 'golden-vertical.jpg', ['--append-vertically', 'left.jpg', 'right.jpg'], 'jpg')
	# test vertical appending for 64-bit version
	def test_LwImageColwert64AppendVerticallyTestTags(self):
		self.LwImageColwertRun(LwImageColwert64Path, 'golden-vertical-tagged.jpg', ['--append-vertically', 'left.jpg', 'right.jpg', '--make', 'LWpu', '--model', 'LwCamera', '--type', '360Stereo', '--software', 'AnselIntegrationTestApp'], 'jpg', ExactFileMatch=True)

try:
	if __name__ == '__main__':
		if is_running_under_teamcity():
			runner = TeamcityTestRunner()
			unittest.main(testRunner=runner)
		else:
			with open(LogFile, "w") as f:
				runner = unittest.TextTestRunner(f)
				unittest.main(testRunner=runner, exit=False)
				print ('Assertions statistics: %d passed, %d failed (%d total)' % (Utils.AssertionPassCounter, Utils.AssertionFailCounter, Utils.AssertionCounter))
finally:
	pass
