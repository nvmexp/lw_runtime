# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from rtmbot.core import Plugin
from slackclient import SlackClient

def reply_to(tag_name):
	def reply_to_decorator(func):
		def func_wrapper(object, name, channel):
			if name.startswith(tag_name):
				stripped_name = name[len(tag_name):].strip()
				if not stripped_name or func.func_code.co_argcount == 1:
					func(object, channel)
				else:
					func(object, stripped_name, channel)
			else:
			    return False
		return func_wrapper
	return reply_to_decorator

class BasicBotPlugin(Plugin):
	def get_channel_id(self):
		channels_call = self.slack_client.api_call("channels.list")
		if channels_call['ok']:
			channels = channels_call['channels']
			for c in channels:
				if c['name'] == self.channel():
					print('Found channel for monitoring')
					return c['id']
			if self.channel_id_to_look_at == '':
				raise LookupError("Unable to find the channel")
		else:
			raise LookupError("Unable to find the channel")

	def process_message(self, data):
		if not hasattr(self, 'channel_id_to_look_at'):
			self.channel_id_to_look_at = self.get_channel_id()
		is_dm = data['channel'].startswith('D')
		if is_dm and data['user'] == self.slack_client.server.userid:
		    return
		print(data)
		if data['channel'] == self.channel_id_to_look_at or is_dm:
			if data['type'] == "message" and not ('subtype' in data and data['subtype'] == 'message_changed'):
				if is_dm or (data['text'].lower().startswith(self.magic_string().lower()) == True and not is_dm):
					response = data['text'].strip()
					if not is_dm:
					    response = response[len(self.magic_string()):].strip()
					found_handler = False
					for methodName in [method for method in dir(self) if callable(getattr(self, method))]:
						if methodName.startswith("answer"):
    						    if getattr(self, methodName)(response, data['channel']) != False:
		    					found_handler = True
					if found_handler == False:
					    self.send_output(data['channel'], u"```Can't handle that ¯\_(ツ)_/¯```")
							

	def send_output(self, channel, msg):
		self.outputs.append([channel, msg])
