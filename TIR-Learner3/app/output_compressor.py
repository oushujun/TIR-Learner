import subprocess
import os

def compress(input_file, threads = 1):
	if os.path.exists(f'{input_file}.gz'):
		print(f'{input_file} gzip was already found. Nothing to do.')
	else:
		try:
			print(f'Attempting to compress {input_file} with pigz...')
			proc = subprocess.call(['pigz', '-6', '-k', '-p', str(threads), input_file])
		except:
			print('pigz compressor not found. Defaulting to gzip.')
			try:
				print(f'Attempting to compress {input_file} with gzip...')
				proc = subprocess.call(['gzip', '-k', '-6', input_file])
			except:
				print('gzip compressor not found. The file will be left uncompressed.')
