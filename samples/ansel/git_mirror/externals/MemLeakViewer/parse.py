from __future__ import print_function

import cgi
import re
import collections
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--in',			dest='in_filename', help='input filename')
	options = parser.parse_args()

	in_filename = "vld_output.txt"
	if options.in_filename is not None:
		in_filename = options.in_filename

	with open(in_filename) as f_vld_output:
		vld_data = f_vld_output.read()

	re_prog_leak = re.compile("---------- Block .*?\n\n", flags=re.DOTALL | re.M)
	leak_matches = re_prog_leak.findall(vld_data)
	print("Found %d leaks! Generating report." % (len(leak_matches)))

	LK_HEADER_BLOCKID = 0	# Block idx
	LK_HEADER_ADDRESS = 1	# Address
	LK_HEADER_NUMBYTES = 2	# Num bytes
	LK_HEADER_HASH = 3		# Hash
	LK_HEADER_COUNT = 4		# Count
	re_prog_leak_header = re.compile("^---------- Block ([0-9]+) at (0x[0-9ABCDEF]+): ([0-9]+) bytes ----------\n  Leak Hash: (0x[0-9ABCDEF]+), Count: ([0-9]+)", flags=re.M)

	LK_CALLSTACK_FILE = 0	# File (if available)
	LK_CALLSTACK_LINE = 1	# Line in file (if available)
	LK_CALLSTACK_BIN = 2	# Binary
	LK_CALLSTACK_FUNC = 3	# Function
	LK_CALLSTACK_OFFSET = 4	# Offset (if available)
	re_prog_file_func_offs = re.compile("[ ]{4}(?:(.*) \(([0-9]*)\): )?(.*)!(.*)\(\)(?: \+ (.*) bytes)*")

	def parse_leak_info(leak_info):
		leak_header_parsed = re_prog_leak_header.search(leak_info)
		leak_info_parsed = re_prog_file_func_offs.findall(leak_info)
		return leak_header_parsed.groups(), leak_info_parsed

	leak_header, leak_parsed = parse_leak_info(leak_matches[0])

	callstack_root = collections.OrderedDict()

	VLD_DBG_INFO_NAME = "__vld_info__"

	total_mem_leak = 0
	for m_idx, leak_match in enumerate(leak_matches):
		lwr_callstack_node = callstack_root
		leak_header, leak_parsed = parse_leak_info(leak_match)

		found_main = True

		# if "ilwoke_main" is present - we need to shift stack towards that
		for f_match in leak_parsed:
			if f_match[3] == "ilwoke_main":
				found_main = False
				break

		leak_num_bytes = int(leak_header[LK_HEADER_NUMBYTES]) * int(leak_header[LK_HEADER_COUNT])
		f_num = len(leak_parsed)
		for f_idx, f_match in enumerate(reversed(leak_parsed)):

			# Skip first entries to common colwergence point
			# which is typically `ilwoke_main` function
			if (not found_main) and (f_match[LK_CALLSTACK_FUNC] == "ilwoke_main"):
				found_main = True
				continue
			if (not found_main):
				continue

			key = f_match[LK_CALLSTACK_BIN]+":"+f_match[LK_CALLSTACK_FUNC]+":"+f_match[LK_CALLSTACK_OFFSET]

			if f_idx == f_num - 1:
				# Check if allocation from the same place was already recorded
				key_new = key
				keys_similar = 0
				while key_new in lwr_callstack_node:
					keys_similar += 1;
					key_new = "%s_%d" % (key, keys_similar)
				key = key_new

			if not key in lwr_callstack_node:
				lwr_callstack_node[key] = collections.OrderedDict()
			lwr_callstack_node = lwr_callstack_node[key]

			if not VLD_DBG_INFO_NAME in lwr_callstack_node:
				lwr_callstack_node[VLD_DBG_INFO_NAME] = {}	# Internal, order doesn't matter
				lwr_callstack_node[VLD_DBG_INFO_NAME]["num_bytes_total"] = 0
				lwr_callstack_node[VLD_DBG_INFO_NAME]["num_leaks_total"] = 0

				lwr_callstack_node[VLD_DBG_INFO_NAME]["file"]		= f_match[LK_CALLSTACK_FILE]
				lwr_callstack_node[VLD_DBG_INFO_NAME]["line"]		= f_match[LK_CALLSTACK_LINE]
				lwr_callstack_node[VLD_DBG_INFO_NAME]["binary"]		= f_match[LK_CALLSTACK_BIN]
				lwr_callstack_node[VLD_DBG_INFO_NAME]["function"]	= f_match[LK_CALLSTACK_FUNC]
				lwr_callstack_node[VLD_DBG_INFO_NAME]["offset"]		= f_match[LK_CALLSTACK_OFFSET]

			lwr_callstack_node[VLD_DBG_INFO_NAME]["num_bytes_total"] += leak_num_bytes
			lwr_callstack_node[VLD_DBG_INFO_NAME]["num_leaks_total"] += 1
		lwr_callstack_node["block"]		= leak_header[LK_HEADER_BLOCKID]
		lwr_callstack_node["address"]	= leak_header[LK_HEADER_ADDRESS]
		lwr_callstack_node["numBytes"]	= leak_header[LK_HEADER_NUMBYTES]
		lwr_callstack_node["hash"]		= leak_header[LK_HEADER_HASH]
		lwr_callstack_node["count"]		= leak_header[LK_HEADER_COUNT]
		total_mem_leak += leak_num_bytes

	UI_WIDGET_CLASSNAME = "UITreeView"
	UI_WIDGET_NODE_CLASSNAME = "UITreeViewNode"

	with open("vld_output_stack.html", "w") as f_vld_output_stack:
		def print_fmt_dict(file, dict_obj):
			file.write("<ul>\n")
			for k, v in dict_obj.items():
				if k == VLD_DBG_INFO_NAME:
					# Skip this special debug info storage
					continue
				if isinstance(v, dict):
					if VLD_DBG_INFO_NAME in v:
						v_info = v[VLD_DBG_INFO_NAME]
						file.write("<li><span class=\"leakEntry\">")
						file.write("<span class=\"leakedBin\">%s</span>" % (cgi.escape(v_info["binary"])))
						file.write(":&nbsp;<span class=\"leakedFunc\">%s</span>" % (cgi.escape(v_info["function"])))
						file.write(":&nbsp;<span class=\"leakedOffset\">%s</span>" % (cgi.escape(v_info["offset"])))
						if v_info["num_leaks_total"] > 1:
							file.write(" <span class=\"leakedBytesSummary\"><span class=\"leakedBytes\">%d</span> <span class=\"leakedBytesText\">bytes</span>, <span class=\"leakedNum\">%d</span> <span class=\"leakedNumText\">leaks</span></span>\n" % (v_info["num_bytes_total"], v_info["num_leaks_total"]))
						else:
							file.write(" <span class=\"leakedBytesSummary\"><span class=\"leakedBytes\">%d</span> <span class=\"leakedBytesText\">bytes</span></span>\n" % (v_info["num_bytes_total"]))
						file.write("</span>")
					else:
						file.write("<li>%s\n" % (k))

					print_fmt_dict(file, v)
					file.write("</li>")
				else:
					if k == "count":
						# Do not print explicit "Count" field
						continue
					elif k == "numBytes":
						if "count" in dict_obj:
							file.write("<li>%s : %s x <span class=\"leakedBytesNode\">%s</span> bytes</li>\n" % (k, dict_obj["count"], v))
						else:
							file.write("<li>%s : <span class=\"leakedBytesNode\">%s</span></li>\n" % (k, v))
					else:
						file.write("<li>%s : %s</li>\n" % (k, v))
			file.write("\n</ul>\n")

		# HTML header
		f_vld_output_stack.write("<!DOCTYPE html>\n<html lang=\"en\">\n")
		f_vld_output_stack.write("<head><meta charset=\"UTF-8\"><script src=\"treeview.js\"></script><link rel=\"stylesheet\" type=\"text/css\" href=\"treeview_memleaks.css\"></head>\n")
		f_vld_output_stack.write("<body>\n")
		f_vld_output_stack.write("<ul class=\"%s\"><ul class=\"%s\">\n" % (UI_WIDGET_CLASSNAME, UI_WIDGET_NODE_CLASSNAME))

		# Mem leaks callstack
		print_fmt_dict(f_vld_output_stack, callstack_root)

		# HTML footer
		f_vld_output_stack.write("</ul></ul>\n")
		f_vld_output_stack.write("\n</body>\n</html>")

	with open("vld_output_pretty.html", "w") as f_vld_output_pretty:
		f_vld_output_pretty.write("<!DOCTYPE html>\n<html lang=\"en\">\n<head><meta charset=\"UTF-8\"></head>\n<body>")
		for m_idx, match in enumerate(leak_matches):
			f_vld_output_pretty.write("<details>\n<summary>%d</summary>\n<pre>\n" % (m_idx))
			f_vld_output_pretty.write("%s" % (match))
			f_vld_output_pretty.write("\n</pre>\n</details>")
		f_vld_output_pretty.write("</body>\n</html>")
