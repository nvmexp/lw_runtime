# -*- coding: utf-8 -*-

import os
import re
import sys
import six
import uuid
import ntpath
import _winreg
import filecmp
import fnmatch
import inspect
import unittest
import tempfile
import traceback
import threading
import cStringIO
from shutil import move
from shutil import rmtree
from time import sleep
from subprocess import Popen, PIPE, STDOUT, check_output
from shutil import copy, copyfile, copytree, rmtree
from Config import *

AssertionCounter = 0
AssertionPassCounter = 0
AssertionFailCounter = 0

# member function decorator to catch all exceptions and print stacktrace
def catch_all_errors(fn):
	def wrapped(self, *args, **kwargs):
		try:
			fn(self, *args, **kwargs)
		except unittest.SkipTest:
			pass
		except:
			print "Unexpected error:", sys.exc_info()[0]
			traceback.print_exc(file=sys.stdout)
			raise
	return wrapped

# class decorator to apply another decorator to all test methods
def for_all_test_methods(decorator):
	def decorate(cls):
		for attr in cls.__dict__: # there's propably a better way to do this
			if attr.startswith('test_') and callable(getattr(cls, attr)):
				setattr(cls, attr, decorator(getattr(cls, attr)))
		return cls
	return decorate

# count asserts decorator
def count_asserts(fn):
	def wrapped(*args, **kwargs):
		global AssertionCounter, AssertionPassCounter, AssertionFailCounter
		AssertionCounter += 1
		try:
			ret = fn(*args, **kwargs)
			AssertionPassCounter += 1
			return ret
		except AssertionError:
			AssertionFailCounter += 1
			raise
	return wrapped

# class decorator to apply another decorator to all assert methods (should start with 'assert')
def for_all_assert_methods(decorator):
	def decorate(cls):
		for c in inspect.getmro(cls):
			for attr in c.__dict__: # there's propably a better way to do this
				if callable(getattr(cls, attr)) and attr.startswith('assert'):
					setattr(cls, attr, decorator(getattr(cls, attr)))
		return cls
	return decorate

def TestCaseFactory(name, BaseClass, skipMessage=None, catchAllErrors=True, countAsserts=True, *args, **kwargs):
	def setUpClass(cls):
		BaseClass.setUpClass(*args, **kwargs)

	def tearDownClass(cls):
		BaseClass.tearDownClass()

	newclass = type(name, (unittest.TestCase, BaseClass), {})
	setattr(newclass, 'setUpClass', classmethod(setUpClass))
	setattr(newclass, 'tearDownClass', classmethod(tearDownClass))
	if skipMessage is not None and isinstance(skipMessage, six.string_types):
		newclass = unittest.skip(skipMessage)(newclass)
	if catchAllErrors:
		newclass = for_all_test_methods(catch_all_errors)(newclass)
	if countAsserts:
		newclass = for_all_assert_methods(count_asserts)(newclass)
	return newclass

def get_process_modules(ProcessName):
	return check_output([ListDlls64, '-accepteula', ProcessName])

def set_regkey_value(regpath, name, value):
	try:
		key = _winreg.OpenKey(_winreg.HKEY_LWRRENT_USER, regpath, 0, _winreg.KEY_ALL_ACCESS)
		_winreg.SetValueEx(key, name, 0, _winreg.REG_SZ, value)
		_winreg.CloseKey(key)
		return True
	except WindowsError as e:
		print ('Exception in set_regkey_value: %s' % str(e))
		return False

def get_regkey_value(path, name="", start_key = None, default=None):
	if isinstance(path, str):
		path = path.split("\\")
	if start_key is None:
		start_key = getattr(_winreg, path[0])
		return get_regkey_value(path[1:], name, start_key)
	else:
		subkey = path.pop(0)
	with _winreg.OpenKey(start_key, subkey) as handle:
		assert handle
		if path:
			return get_regkey_value(path, name, handle)
		else:
			desc, i = None, 0
			while not desc or desc[0] != name:
				desc = _winreg.EnumValue(handle, i)
				i += 1
			return desc[1]
	return default

# return a list of PIDs keeping a handle on a certain path
def get_pids_for_subtree_handles(path):
	pids = set()
	if 'win' in sys.platform:
		getpids = [Handle64, '-accepteula', path]
		pid_re = re.compile('pid: *(\d+)')
	with tempfile.TemporaryFile() as f:
		p = Popen(getpids, stdout=f, stderr=STDOUT)
		p.wait()
		f.seek(0)
		for line in f.readlines():
			match = pid_re.search(line.strip())
			if match:
				pids.add(int(match.group(1)))
		if os.getpid() in pids:
			pids.remove(os.getpid())
	return pids

# return a list of PIDs which have dlls mapped from the path
def get_pids_for_subtree_dll(image_name):
	pids = set()
	if 'win' in sys.platform:
		getpids = [ListDlls64, '-accepteula', '-d', image_name]
		pid_re = re.compile('pid: *(\d+)')
	with tempfile.TemporaryFile() as f:
		p = Popen(getpids, stdout=f, stderr=STDOUT)
		p.wait()
		f.seek(0)
		for line in f.readlines():
			match = pid_re.search(line.strip())
			if match:
				pids.add(int(match.group(1)))
		if os.getpid() in pids:
			pids.remove(os.getpid())
	return pids

# return a list of PIDs which have dlls mapped from the path
def get_pids_for_subtree_dlls(path):
	pids = set()
	matches = []
	for root, dirnames, filenames in os.walk(path):
		for filename in fnmatch.filter(filenames, '*.dll'):
			matches.append(os.path.join(root, filename))
	matches = [ntpath.basename(x) for x in matches]
	print(matches)
	for match in matches:
		pids.update(get_pids_for_subtree_dll(match))
	return pids

def copy_testing_shader(AnselTargetPath):
	copy(SmokeTestImagesDX11 + '\\Testing.yaml', AnselTargetPath + '/')
	copy(SmokeTestImagesDX11 + '\\Testing.yfx', AnselTargetPath + '/')

def copytree_flatten(sourcePath, targetPath):
	if (not os.path.exists(targetPath)):
		os.makedirs(targetPath)
	for d, dirs, files in os.walk(sourcePath):
		for f in files:
			copyfile(os.path.join(d, f), os.path.join(targetPath, f))

def install_ansel(AnselTargetPath):
	# kill all processes keeping a lock on the target path
	pids = get_pids_for_subtree_handles(AnselTargetPath) | get_pids_for_subtree_dlls(AnselTargetPath)
	print("List of PIDs keeping locks on the Ansel intallation path: '%s'" % str(pids))
	for pid in pids:
		print('Killing PID %d' % pid)
		os.system("taskkill /F /PID %d" % pid)
	# delete target path
	errorsOnRemoveTree = False
	def handle_error(function, path, excinfo):
		errorsOnRemoveTree = True
		print('rmtree error: ' + str(function))
		print('rmtree error: ' + str(path))
		print('rmtree error: ' + str(excinfo))
	rmtree(AnselTargetPath, onerror=handle_error, ignore_errors=True)
	if errorsOnRemoveTree:
		print('There were some errors while cleaning up the target directory')
	else:
		print('Cleaned up target directory')
	# copy artifacts to the target path
	copytree_flatten(AnselSourcePath, AnselTargetPath)
	print('Ansel installed')

# run detached process
def run_detached_process(processPath, workingDir):
	return Popen(processPath, close_fds=True, cwd=workingDir)

# run piped process
def run_piped_process(processPath, workingDir, out, err=STDOUT):
	return Popen(processPath, stdin=PIPE, stdout=out, stderr=err, cwd=workingDir)

def get_global_motion_vector(image1, image2, workDir='.'):
	xcorrTool = run_piped_process([Xcorr264, image1, image2], workDir, PIPE, None)
	output = xcorrTool.stdout.read()
	vector_re = re.compile('(-?\d+),\s*(-?\d+)')
	match = vector_re.search(output)
	if match is not None:
		return (int(match.group(1)), int(match.group(2)))
	else:
		raise Exception("xcorr2 tool didn't respond as expected")
# compare images using PSNR metric with a threshold
# uses ImageMagick compare
def compare_images(image, golden, workDir, threshold, generateGolden=False):
	print('comparing %s and %s' % (image, golden))
	if generateGolden:
		print("Renaming '%s' to '%s'" % (image, golden))
		move(image, golden)
		return True

	compareTool = run_piped_process([CompareTool, image, golden], workDir, PIPE)
	output = compareTool.stdout.read().strip()
	print('output ' + output)
	if output == '1.#INF' or output == '0':
		return True
	psnr = float(output)
	if psnr > threshold:
		return True
	return False

def compare_files(file1, file2, generateGolden=False):
	print('comparing %s and %s' % (file1, file2))
	if not generateGolden:
		return filecmp.cmp(file1, file2)
	else:
		move(file1, file2)
		return True

def compare_intensity_ranges(image1, image2, workDir):
	print('comparing intensity ranges of %s and %s' % (image1, image2))
	identifyTool1 = run_piped_process([IdentifyTool, IdentifyFormat, IntensityFormatArg, image1], workDir, PIPE)
	identifyTool2 = run_piped_process([IdentifyTool, IdentifyFormat, IntensityFormatArg, image2], workDir, PIPE)

	output1 = identifyTool1.stdout.read().strip().split(IntensityFormatSeparator)
	output2 = identifyTool2.stdout.read().strip().split(IntensityFormatSeparator)
	floats1 = map(float, output1)
	floats2 = map(float, output2)
	print(" ".join(map(str, [floats1, floats2])))
	return (floats1[1] - floats1[0]) >= (floats2[1] - floats2[0])

class InputStreamChunker(threading.Thread):
	'''
	Threaded object / code that mediates reading output from a stream,
	detects "separation markers" in the stream and spits out chunks
	of original stream, split when ends of chunk are encountered.

	Results are made available as a list of filled file-like objects
	(your choice). Results are accessible either "asynchronously"
	(you can poll at will for results in a non-blocking way) or
	"synchronously" by exposing a "subscribe and wait" system based
	on threading.Event flags.

	Usage:
	- instantiate this object
	- give our input pipe as "stdout" to other subprocess and start it:
		Popen(..., stdout = th.input, ...)
	- (optional) subscribe to data_available event
	- pull resulting file-like objects off .data
	  (if you are "messing" with .data from outside of the thread,
	   be lwrteous and wrap the thread-unsafe manipulations between:
	   obj.data_unoclwpied.clear()
	   ... mess with .data
	   obj.data_unoclwpied.set()
	   The thread will not touch obj.data for the duration and will
	   block reading.)

	License: Public domain
	Absolutely no warranty provided
	'''
	def __init__(self, delimiter = None, outputObjConstructor = None):
		'''
		delimiter - the string that will be considered a delimiter for the stream
		outputObjConstructor - instanses of these will be attached to self.data array
		 (intantiator_pointer, args, kw)
		'''
		super(InputStreamChunker,self).__init__()

		self._data_available = threading.Event()
		self._data_available.clear() # parent will .wait() on this for results.
		self._data = []
		self._data_unoclwpied = threading.Event()
		self._data_unoclwpied.set() # parent will set this to true when self.results is being changed from outside
		self._r, self._w = os.pipe() # takes all inputs. self.input = public pipe in.
		self._stop = False
		if not delimiter: delimiter = str(uuid.uuid1())
		self._stream_delimiter = [l for l in delimiter]
		self._stream_roll_back_len = ( len(delimiter)-1 ) * -1
		if not outputObjConstructor:
			self._obj = (cStringIO.StringIO, (), {})
		else:
			self._obj = outputObjConstructor
	@property
	def data_available(self):
		'''returns a threading.Event instance pointer that is
		True (and non-blocking to .wait() ) when we attached a
		new IO obj to the .data array.
		Code consuming the array may decide to set it back to False
		if it's done with all chunks and wants to be blocked on .wait()'''
		return self._data_available
	@property
	def data_unoclwpied(self):
		'''returns a threading.Event instance pointer that is normally
		True (and non-blocking to .wait() ) Set it to False with .clear()
		before you start non-thread-safe manipulations (changing) .data
		array. Set it back to True with .set() when you are done'''
		return self._data_unoclwpied
	@property
	def data(self):
		'''returns a list of input chunkes (file-like objects) captured
		so far. This is a "stack" of sorts. Code consuming the chunks
		would be responsible for disposing of the file-like objects.
		By default, the file-like objects are instances of cStringIO'''
		return self._data
	@property
	def input(self):
		'''This is a file descriptor (not a file-like).
		It's the input end of our pipe which you give to other process
		to be used as stdout pipe for that process'''
		return self._w
	def flush(self):
		'''Normally a read on a pipe is blocking.
		To get things moving (make the subprocess yield the buffer,
		we inject our chunk delimiter into self.input

		This is useful when primary subprocess does not write anything
		to our in pipe, but we need to make internal pipe reader let go
		of the pipe and move on with things.
		'''
		os.write(self._w, ''.join(self._stream_delimiter))
	def stop(self):
		self._stop = True
		self.flush() # reader has its teeth on the pipe. This makes it let go for for a sec.
		os.close(self._w)
		self._data_available.set()
	def __del__(self):
		try:
			self.stop()
		except:
			pass
		try:
			del self._w
			del self._r
			del self._data
		except:
			pass
	def run(self):
		''' Plan:
		- We read into a fresh instance of IO obj until marker encountered.
		- When marker is detected, we attach that IO obj to "results" array
		  and signal the calling code (through threading.Event flag) that
		  results are available
		- repeat until .stop() was called on the thread.
		'''
		marker = ['' for l in self._stream_delimiter] # '' is there on purpose
		tf = self._obj[0](*self._obj[1], **self._obj[2])
		while not self._stop:
			l = os.read(self._r, 1)
			trash_str = marker.pop(0)
			marker.append(l)
			if marker != self._stream_delimiter:
				tf.write(l)
			else:
				# chopping off the marker first
				tf.seek(self._stream_roll_back_len, 2)
				tf.truncate()
				tf.seek(0)
				self._data_unoclwpied.wait(5) # seriously, how much time is needed to get your items off the stack?
				self._data.append(tf)
				self._data_available.set()
				tf = self._obj[0](*self._obj[1], **self._obj[2])
		os.close(self._r)
		tf.close()
		del tf
