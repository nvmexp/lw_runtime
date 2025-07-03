#!/usr/bin/elw python
# -*- coding: utf-8 -*-

import os
import re
import csv
import sys
import gzip
import time
import thread
import sqlite3
import fnmatch
import requests
import datetime
import threading
import subprocess
from time import strftime
from pytz import timezone
from subprocess import check_call
from BasicBotPlugin import *
from rtmbot.core import Job

WorkDir = os.path.dirname(os.path.realpath(__file__)) + '/../workdir'
DumpTelemetryScript = WorkDir + '/launch.pl'
# set all these appropriately
ClickhouseServer = None
AnselBotUsername = None
AnselBotPassword = None
GitBotUsername = None
GitBotPassword = None

SqlShotTypeHighLevelStat = "SELECT kindofshot, COUNT(*) FROM ansel WHERE kindofshot != 'NONE' AND kindofshot != '' AND isinternal != 'True' AND eventname = 'CaptureStarted' GROUP BY kindofshot"
SqlShotTypeShotsTotalStat = "SELECT COUNT(*) FROM ansel WHERE kindofshot != 'NONE' AND kindofshot != '' AND isinternal != 'True' AND eventname = 'CaptureStarted'"
SqlShotsByGames = "SELECT drsprofilename, COUNT(*) FROM ansel WHERE kindofshot != 'NONE' AND kindofshot != '' AND isinternal != 'True' AND eventname = 'CaptureStarted' GROUP BY drsprofilename ORDER BY COUNT(*) DESC"
SqlShotsByCountry30 = "SELECT country, count(*) FROM ansel WHERE  kindofshot != 'NONE' AND kindofshot != '' AND isinternal != 'True' AND eventname = 'CaptureStarted' GROUP BY country ORDER BY COUNT(*) DESC LIMIT 30"
SqlHighresMult = "SELECT highresmultiplier, count(*) FROM ansel WHERE kindofshot = 'HIGHRES' AND isinternal != 'True' AND eventname = 'CaptureStarted' GROUP BY highresmultiplier ORDER BY COUNT(*) DESC"
Sql360Quality = "SELECT quality360resolution, count(*) FROM ansel WHERE (kindofshot = 'MONO_360' OR kindofshot = 'STEREO_360') AND isinternal != 'True' AND eventname = 'CaptureStarted' GROUP BY quality360resolution ORDER BY COUNT(*) DESC LIMIT 10"

def as_block(text):
	return '```' + text + '```'

def find_all_files(directory, regexp):
	matches = []
	mask_re = re.compile(regexp)
	for root, dirnames, filenames in os.walk(directory):
		for filename in filenames:
			if mask_re.match(filename):
				matches.append(filename)
	return matches

def find_all_directories(directory, regexp):
	matches = []
	mask_re = re.compile(regexp)
	for root, dirnames, filenames in os.walk(directory):
		for dirname in dirnames:
			if mask_re.match(dirname):
				matches.append(dirname)
	return matches

def get_git_file_listing(request):
	TeamcityBuildId = 'Ansel_Release'
	LinesToInclude = 10
	try:
		params = request.split()
		if len(params) == 7:
			BuildRestRequest = 'http://teamcity/app/rest/builds?locator=buildType:(id:%s),count:200' % TeamcityBuildId
			r = requests.get(BuildRestRequest, auth=(AnselBotUsername, AnselBotPassword))
			if r.status_code == 200:
				re_buildId = re.compile('<build id="(\d+)" buildTypeId="%s" number="%s.*?"' % (TeamcityBuildId, params[6]))
				match = re_buildId.search(r.text)
				if match:
					ChangesRestRequest = 'http://teamcity/app/rest/changes?locator=buildType:(id:%s),build:(id:%s)' % (TeamcityBuildId, match.group(1))
					r = requests.get(ChangesRestRequest, auth=(AnselBotUsername, AnselBotPassword))
					if r.status_code == 200:
						re_changeId = re.compile('<change id="\d+" version="(.*?)"')
						match = re_changeId.search(r.text)
						if match:
							git_hash = match.group(1)
							lineNo = int(params[3])
							lineNoStart = lineNo - LinesToInclude / 2
							lineNoEnd = lineNo + LinesToInclude / 2
							gitCmdLine = ['git', 'blame', '-L', '%d,%d' % (lineNoStart, lineNoEnd), git_hash, '--', params[1]]
							git_blame_strings = subprocess.check_output(gitCmdLine, stderr=subprocess.STDOUT, cwd='%s/Ansel' % WorkDir)
							return as_block(git_blame_strings)
						else:
							return as_block("Couldn't get GIT hash from Teamcity, sorry")
					else:
						return as_block("Couldn't get changes list from Teamcity, sorry (REST API answers with %d, not 200)" % r.status_code)
				else:
					return as_block("Couldn't get Teamcity build id, sorry")
			else:
				return as_block('Teamcity REST API answers with %d, not 200, try another time' % r.status_code)
		else:
			msg = 'usage: show file <filename relative to GIT repo root> line <line number> in build <build number without git hash>\n'
			msg += '\t\t%smonitoring only release builds' % self.magic_string()
			msg += '\t\t%sshow file README.md line 5 in build 170' % self.magic_string()
			return as_block(msg)
	except Exception as e:
		return as_block(str(e))

class AnselBotJob(threading.Thread):
	def __init__(self, func, channel, request):
		threading.Thread.__init__(self)
		self.func = func
		self.result = [channel, None] # None will be filled with text that this job generated
		self.request = request
		self.finished = False
		self.job_started_at = None

	def time(self):
		if self.job_started_at == None:
			return datetime.timedelta(seconds=0)
		else:
			return datetime.datetime.now() - self.job_started_at

	def run(self):
		try:
			if self.request is not None:
				self.result[1] = self.func(self.request)
			else:
				self.result[1] = self.func()
		except Exception as e:
			self.result[1] = as_block(str(e))
		self.finished = True

class AnselTelemetryHandler(Job):
	def __init__(self, interval):
		super(AnselTelemetryHandler, self).__init__(interval)
		self.SqlFormat = 'PrettyCompactNoEscapes'
		self.jobs = []
		self.jobs_lock = threading.Lock()

	def run(self, slack_client):
		ret = []
		try:
			with self.jobs_lock:
				unfinished_jobs = [x for x in self.jobs if not x.finished]
				finished_jobs = [x for x in self.jobs if x.finished]
				for job in finished_jobs:
					# always post a snippet from a dedicated job
					one_meg = 1024 * 1024
					if len(job.result[1]) > one_meg:
						job.result[0] = job.result[0][:one_meg]
					res = slack_client.api_call("files.upload", filename=str(time.time()) + '.txt', title='response', content=job.result[1], channels=job.result[0])
					if res['ok'] != True:
						ret.append([job.result[0], 'Error while posting a snippet'])
				self.jobs = unfinished_jobs
		except Exception as e:
			print(e)
		finally:
			return ret

	def submit_job(self, func, channel, request):
		with self.jobs_lock:
			job = AnselBotJob(func, channel, request)
			job.start()
			self.jobs.append(job)

	def set_use_format_request(self, request):
	    if request == 'pretty':
		self.SqlFormat = 'PrettyCompactNoEscapes'
	    elif request == 'csv':
		self.SqlFormat = 'CSVWithNames'
	    elif request == 'tsv':
		self.SqlFormat = 'TabSeparatedWithNames'
	    else:
		return 'No such format [%s]' % request
	    return 'Using [%s] now' % self.SqlFormat
    
	def get_sql_request(self, request):
		request = request.replace('&lt', '<')
		request = request.replace('&gt', '>')
		request += ' FORMAT ' + self.SqlFormat
		r = requests.post(ClickhouseServer, data=request)
		if r.status_code == 200:
			return r.text
		else:
			return 'Clickhouse returns %d, not 200\n%s' % (r.status_code, r.text)

	def get_shot_type_stats(self):
		return self.get_sql_request(SqlShotTypeHighLevelStat)

	def get_shots_by_game_stats(self):
		return self.get_sql_request(SqlShotsByGames)

	def get_highres_stats(self):
		return self.get_sql_request(SqlHighresMult)

	def get_360_stats(self):
		return self.get_sql_request(Sql360Quality)

	def get_country_stats(self):
		return self.get_sql_request(SqlShotsByCountry30)

class GitPullJob(Job):
	def __init__(self, interval):
		super(GitPullJob, self).__init__(interval)

	def run(self, slack_client):
		gitCmdLine = ['git', 'pull']
		subprocess.check_output(gitCmdLine, stderr=subprocess.STDOUT, cwd='%s/Ansel' % WorkDir)
		return []

# In order for a decorator to work properly
# all methods in a class should be called with a name starting with 'answer'
class AnselPlugin(BasicBotPlugin):
	def register_jobs(self):
		self.anselTelemetryHandler = AnselTelemetryHandler(1)
		self.gitPullJob = GitPullJob(600) # update git every 10 minutes
		self.jobs.append(self.anselTelemetryHandler)
		self.jobs.append(self.gitPullJob)

	def channel(self):
		return 'ct-ansel'
	def magic_string(self):
		return 'hey bot, '

	@reply_to('sup')
	def answer_sup(self, channel):
		self.send_output(channel, as_block('sup'))

	@reply_to('help')
	def answer_help(self, channel):
		msg = 'Prepend all messages addressed to the bot with "%s"\n' % self.magic_string()
		msg += 'sup - test if bot is alive\n'
		msg += 'time - print time\n'
		msg += 'shot type stats - shot type statistics\n'
		msg += 'shots by country - country statistics\n'
		msg += 'shots by title - title statistics\n'
		msg += 'highres stats - highres multiplier statistics\n'
		msg += '360 stats - 360 quality statistics\n'
		msg += 'use format [pretty,csv,tsv] - change SQL output format style (persistent)\n'
		msg += 'sql <sql request> - arbitrary SQL request (see https://clickhouse.yandex/reference_en.html )\n'
		msg += 'show file <filename relative to GIT repo root> line <line number> in build <build number without git hash>\n'
		msg += 'stfu - terminate the bot\n'
		self.send_output(channel, as_block(msg))

	@reply_to('time')
	def answer_time(self, channel):
	    try:
		tz_reyk = timezone('Atlantic/Reykjavik')
		tz_msk = timezone('Europe/Moscow')
		tz_pst = timezone('US/Pacific')
		tz_cst = timezone('US/Central')
		msg = '%-20s: %s\n' % ('Santa Clara:', datetime.datetime.now(tz_pst).strftime('%Y-%m-%d %H:%M:%S (GMT%z)'))
		msg += '%-20s: %s\n' % ('Austin:', datetime.datetime.now(tz_cst).strftime('%Y-%m-%d %H:%M:%S (GMT%z)'))
		msg += '%-20s: %s\n' % ('Reykjavik:', datetime.datetime.now(tz_reyk).strftime('%Y-%m-%d %H:%M:%S (GMT%z)'))
		msg += '%-20s: %s\n' % ('Moscow:', datetime.datetime.now(tz_msk).strftime('%Y-%m-%d %H:%M:%S (GMT%z)'))
		self.send_output(channel, as_block(msg))
	    except Exception as e:
		print(e)

	@reply_to("shot type stats")
	def answer_shot_type_stats(self, channel):
		ah = self.anselTelemetryHandler
		ah.submit_job(ah.get_shot_type_stats, channel, None)

	@reply_to("shots by country")
	def answer_shots_by_country(self, channel):
		ah = self.anselTelemetryHandler
		ah.submit_job(ah.get_country_stats, channel, None)

	@reply_to("shots by title")
	def answer_shots_by_titles(self, channel):
		ah = self.anselTelemetryHandler
		ah.submit_job(ah.get_shots_by_game_stats, channel, None)

	@reply_to("highres stats")
	def answer_highres_stats(self, channel):
		ah = self.anselTelemetryHandler
		ah.submit_job(ah.get_highres_stats, channel, None)

	@reply_to("360 stats")
	def answer_360_stats(self, channel):
		ah = self.anselTelemetryHandler
		ah.submit_job(ah.get_360_stats, channel, None)

	@reply_to("sql")
	def answer_sql_stats(self, request, channel):
		ah = self.anselTelemetryHandler
		ah.submit_job(ah.get_sql_request, channel, request)

	@reply_to("use format")
	def answer_use_format(self, request, channel):
		ah = self.anselTelemetryHandler
		ah.submit_job(ah.set_use_format_request, channel, request)

	@reply_to("show")
	def answer_git_file_listing(self, request, channel):
		ah = self.anselTelemetryHandler
		ah.submit_job(get_git_file_listing, channel, request)

	@reply_to("stfu")
	def answer_terminate(self, channel):
		self.send_output(channel, as_block('terminating, bye'))
		quit()

