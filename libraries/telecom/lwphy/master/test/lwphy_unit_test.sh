#!/bin/bash

bin_file=./build/examples/error_correction/lwphy_ex_ldpc
if [ -f "${bin_file}" ]; then
	for tv in $(find ./ -name "*ldpc_BG1*.h5"); do
		echo "${bin_file} -i ${tv}   -n 10 -p 8"
		${bin_file} -i ${tv} -n 10 -p 8
	done

	for tv in $(find ./ -name "*ldpc_BG2*.h5"); do
		echo "${bin_file} -i ${tv}   -n 10 -p 8 -g 2"
		${bin_file} -i ${tv} -n 10 -p 8 -g 2
	done
fi

bin_file=./build/examples/pusch_rx_multi_pipe/lwphy_ex_pusch_rx_multi_pipe
if [ -f "${bin_file}" ]; then
	for tv in $(find ./ -name "*pusch*.h5"); do
		echo "${bin_file} -i ${tv}"
		${bin_file} -i ${tv}
	done
fi

bin_file=./build/examples/pucch_receiver/pucch_receiver
if [ -f "${bin_file}" ]; then
	for tv in $(find ./ -name "*pucch*.h5"); do
		echo "${bin_file} -i ${tv} 20"
		${bin_file} -i ${tv} 20
	done
fi

bin_file=./build/examples/pdsch_tx/lwphy_ex_pdsch_tx
if [ -f "${bin_file}" ]; then
	for tv in $(find ./ -name "*pdsch*.h5"); do
		echo "${bin_file} ${tv} 20 0  //non-AAS mode"
		${bin_file} ${tv} 20 0

		echo "${bin_file} ${tv} 20 1  //AAS mode"
		${bin_file} ${tv} 20 1
	done
fi

bin_file=./build/examples/pdcch/embed_pdcch_tf_signal
if [ -f "${bin_file}" ]; then
	for tv in $(find ./ -name "*pdcch*.h5"); do
		echo "${bin_file} ${tv}"
		${bin_file} ${tv}
	done
fi

bin_file=./build/examples/ss/testSS
if [ -f "${bin_file}" ]; then
	for tv in $(find ./ -name "*SS*.h5"); do
		echo "${bin_file} ${tv}"
		${bin_file} ${tv}
	done
fi
