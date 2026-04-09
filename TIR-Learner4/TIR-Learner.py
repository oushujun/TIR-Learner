#!/usr/app/env python3
# -*- coding: utf-8 -*-
# Kenji Gerhardt (kenji.gerhardt@gmail.com)
# Tianyu (Sky) Lu (tianyu@lu.fm)
# Dec. 8, 2025

import argparse
import os

VERSION = "v1.0"
INFO = "by Kenji Gerhardt, released under GPLv3"

#Options
def options():
	parser = argparse.ArgumentParser(prog="TIR-Learner",
									 description="TIR-Learner is an ensemble pipeline for Terminal Inverted Repeat "
												 "(TIR) transposable elements annotation in eukaryotic genomes")
												 
	parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {VERSION} {INFO}")

	parser.add_argument("-g", "-f", "--genome_file", help="Genome file in fasta format",
						type=str, required=True)

	parser.add_argument("-s", "--species", default = None, help="Check for homologous TIRs based on one of the following: \
																\"maize\", \"rice\"")		
																
	parser.add_argument("-l", "--length", help="Max length of TIR (Optional)", type=int, default=5000)
	
	parser.add_argument("-p", "--processors",
						help="Number of parallel processes. Default 1.", type=int, default=1)
						
	parser.add_argument("-o", "--directory", help=f"Where to place results. \
						Also used as the working directory. \
						Will be created if it does not exist. \
						Default location is {os.path.join(os.getcwd(), 'TIR_Learner_working_directory')}.",
						type=str, default='TIR_Learner_working_directory')
						
	parser.add_argument('--skip_tirvish', action = 'store_true', 
						help = 'Skip running TIRvish or omit existing TIRvish results from post-processing.')
	parser.add_argument('--skip_grf', action = 'store_true', 
						help = 'Skip running GRF or omit existing GRF results from post-processing.')
	
	parser.add_argument('--existing_tirvish', default = None, 
						help = 'Supply an existing TIRvish output JSON from a previous TIR-Learner run.')
	parser.add_argument('--existing_grf', default = None, 
						help = 'Supply an existing GRF output JSON from a previous TIR-Learner run.')
						
	# see prog_const for what additional args are acceptable

	parsed_args = parser.parse_args()
	
	return parsed_args

#Manager to handle option parsing and invoke the rest of the program. The true main behavior of TIR-Learner is in app/main.py
def main():	
	parsed_args = options()
	
	ok_species = ['rice', 'maize']
	genome_file = parsed_args.genome_file
	directory = parsed_args.directory
	
	threads = parsed_args.processors
	species = parsed_args.species
	
	tir_max_length = parsed_args.length
	
	ok_to_continue = True
	
	if species is not None:
		species = species.lower()
		if species == 'others':
			species = None  # 'others' means no species-specific filtering (EDTA compatibility)
		elif species not in ok_species:
			print(f'Your supplied species {species} is not in the acceptable species list, which is:')
			for s in ok_species:
				print(f'\t{s}')
			ok_to_continue = False
						
	if ok_to_continue:	
		if threads == 1:
			os.environ['OMP_NUM_THREADS'] = '1'
		else:
			#We handle most of the parallelization in python, leave exactly 2 threads / CNN process
			os.environ['OMP_NUM_THREADS'] = '2'
		os.environ["KERAS_BACKEND"] = "torch"  # use pytorch as keras backend
		os.environ["KMP_WARNINGS"] = '0'  # mute all OpenMP warnings
		
		os.environ["CUDA_VISIBLE_DEVICES"] = "," #force torch to use CPU, not GPU

	
		print('Loading program resources. This usually takes about a minute.')
		from app.main import newTL
		print('Program resources loaded.')
		print('')
	
		skip_t = parsed_args.skip_tirvish
		skip_g = parsed_args.skip_grf
		exist_t = parsed_args.existing_tirvish
		exist_g = parsed_args.existing_grf
	
		TIRLearner_instance = newTL(
								genome_file_path = genome_file,
								TIR_length = tir_max_length,
								processors = threads,
								species = species,
								wd = directory,
								skip_tirvish = skip_t,
								skip_grf = skip_g, 
								existing_tirvish = exist_t,
								existing_grf = exist_g
								)
		TIRLearner_instance.run()

if __name__ == "__main__":
	main()
