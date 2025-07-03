#!/bin/bash

# The script reads an input file with a module list and runs the cloc tool
# on those directories and captures the C code and header code lines for each
# of these directories and stores that in a file given by the user. It also
# callwlates the total count.

func=$1
qnx_root=$2
input_file=$3
output_file=$4
top_result_count=$5
tmp_file="/tmp/cloc_tmp"

function print_data
{
	echo "$1 $2 $3 $4" >> ${tmp_file}
}

# Starts from here

echo "QNX Root:" ${qnx_root}
echo "Input file:" ${input_file}
echo "Output file:" ${output_file}

rm -f ${tmp_file}
rm -f ${output_file}

function get_complexity
{
	print_data "Module_Name" "Modified_Complexity" "Regular_Complexity" "Lines in function"

	while IFS='' read -r line || [[ -n "$line" ]]; do
		# Skip a blank line
		if [[ -z $line ]]; then
			continue;
		fi

		FILES=${qnx_root}/${line}/*.c
		for file in $FILES; do
			if [[ -z $top_result_count ]]; then
				complexity=`pmccabe ${file}  | sort -nr`
			else
				complexity=`pmccabe ${file}  | sort -nr | head -${top_result_count}`
			fi
			IFS=$'\n'
			for result in ${complexity}; do
				modified_complexity=`echo ${result} | awk '{print $1}'`
				regular_complexity=`echo ${result} | awk '{print $2}'`
				lines_in_func=`echo ${result} | awk '{print $5}'`
				func_name=`echo ${result} | awk '{print $7}'`
				print_data "${file}:$func_name" "$modified_complexity" "$regular_complexity" "$lines_in_func"
			done
		done
	done < ${input_file}

	column -t ${tmp_file} > ${output_file}
	rm -f ${tmp_file}
}

function get_cloc
{
	c_total=0
	h_total=0
	c_loc=0
	h_loc=0

	print_data "Module_Name" "C_Code_Lines" "Header_Lines"

	while IFS='' read -r line || [[ -n "$line" ]]; do
		# Skip a blank line
		if [[ -z $line ]]; then
			continue;
		fi

		# Get C line count
		c_line=`cloc ${qnx_root}/${line} 2> /dev/null | grep -w "C" | grep -v "Header"`
		if [[ -z $c_line ]]; then
			c_loc=0;
		else
			c_loc=`echo $c_line | cut -d " " -f 5`
		fi
		c_total=$(( c_total + c_loc ))

		# Get Header lines count
		h_line=`cloc ${qnx_root}/${line} 2> /dev/null | grep "Header"`
		if [[ -z $h_line ]]; then
			h_loc=0;
		else
			h_loc=`echo $h_line | cut -d " " -f 6`
		fi
		h_total=$(( h_total + h_loc ))

		print_data "$line" "$c_loc" "$h_loc"
	done < ${input_file}

	print_data "Total" "$c_total" "$h_total"

	column -t ${tmp_file} > ${output_file}
	rm -f ${tmp_file}
}

if [[ $func == "cloc" ]]; then
	get_cloc
elif [[ $func == "complexity" ]]; then
	get_complexity
fi
