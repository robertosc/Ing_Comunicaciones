import numpy as np
import matplotlib.pyplot as plt
import math
import os
import binascii
import sys
from statistics import mean
from scipy.signal import hilbert
from numpy import array, sign, zeros


def file2bin(input_file, output_bin):
	'''
		Creates a binary file as output from a given input file with no particular format.
			Inputs:
				-input_file: [str] source file to transfer (any extension will work [eg: .jpg, .png, .mp3, .pptx, ...])
				-output_bin: [str] binary file created from the input file codification.

			Outputs:
				-bin_string: [str] binary string with the information of the input file.
	'''

	# Definition of files handler.
	writer = open(output_bin, 'w')

	# reading the contents of the input file 
	with open(input_file, 'rb') as f:
		content = f.read()
	# Changing the contents from the input file format to hex codification.
	bin_string = binascii.hexlify(content)

	bin_string = bin(int(bin_string, 16))[2:]
	writer.write(f'{bin_string}')
	print(f'[INFO] Hex file successfully created as: {output_bin}')

	# Closing files.
	f.close()
	writer.close()
		
	return bin_string

def bin2file(bin_string, output_file):

	'''
	Converts from an input hex file to a selected format output file. 
		Inputs:
			-bin_string: [str] binary string received from an external source.
			-output_file: [str] result of the re transformation of binary data to a particular file.
	'''

	with open(output_file, 'wb') as fout:
		# Re-transforming the binary string to data
		bin_string = hex( int(bin_string, 2) )[2:]

		fout.write(binascii.unhexlify(bin_string))
		fout.close()

	print(f'[INFO] New file created succesfully. Name: {output_file}')

def extension_handler(input_file):
	'''
		This function gets the extension from the input file and creates the output_file with the same extension
			Inputs:
				-input_file: [str] hardcoded input file from the main function (including its extension)
			Outputs: 
				-output_bin: [str] path and name (hardcoded) to the output binary string to transmit
				-output_file [str] path and name (with the same extension as the input) to the received file.
	'''
	# Splitting the path given as input
	path = os.path.split(input_file)
	output_path = path[0]
	extension = path[1]
	print(extension)
	# Getting the extension of the file given as input
	extension = extension.split('.')
	extension = extension[1]

	# Output paths definition
	output_bin = os.path.join(output_path, 'output_bin.txt')
	output_file = os.path.join(output_path, f'output_file.{extension}')

	return output_bin, output_file

def channelError(syndrome):
	H = np.array([[1, 0, 0],[ 0, 1, 0],[ 0, 0, 1],[ 1, 1, 0],[ 0, 1, 1],[ 1, 0, 1]])
	dim = len(H)
	errorArray = np.zeros(dim)
	counter = 0
	errorLocation = 0
	for row in H:
		comparison = row == syndrome
		if comparison.all():
			errorLocation = counter
		counter = counter+1
	if errorLocation != 0:
		errorArray[errorLocation] = 1
		
	return errorArray

def channelEncoder(string_bits):
	""" Channel encoder """
	bit_sequence = list(string_bits)
	bit_sequence = [int(bit) for bit in bit_sequence]

	#En encoder_output se tendra el result final cuando se termine la codificacion de todos los conjuntos de k bits
	encoder_output = np.empty([1,6], int)
	#Define la matriz generadora G
	G = np.array([[1, 1, 0, 1, 0, 0], [ 0, 1, 1, 0, 1, 0], [ 1, 0, 1, 0, 0, 1]])
	#Limites para recorrer el arreglo
	k = 3
	n = 0
	m = 3

	#Guarda la cantidad de zeroes agregados en la bit_sequence en caso de que no sea múltiplo de k = 3
	zeroes = 0

	while(len(bit_sequence)%k != 0): #Revisa que no vayan a quedar vectores con menos de k elementos
		bit_sequence.append(0)
		zeroes+=1

	#Recorre todo el arreglo de entrada
	while n < len(bit_sequence):
		#Se obtienen los k = 3 bits para codificar
		encoder_input = np.array([bit_sequence[n:m]])
		multiplicacion = encoder_input@G
			
		for i in multiplicacion[0]:
			multiplicacion[0][i] = multiplicacion[0][i] % 2

		if n == 0: #Inicizaliza el np.array
			encoder_output= multiplicacion
		else: #apila los demas resultados
			encoder_output = np.vstack([encoder_output, multiplicacion])
		n+=k
		m+=k

	string = np.array_str(encoder_output)
	string = string.replace('[', '')
	string = string.replace(']', '')
	string = string.replace(' ', '')
	string = string.replace('\n', '')

	return encoder_output, zeroes, string_bits

def channelDecoder(encoder_output, string):
	""" Channel decoder """
	H = np.array([[1, 0, 0],[ 0, 1, 0],[ 0, 0, 1],[ 1, 1, 0],[ 0, 1, 1],[ 1, 0, 1]])

	# To go through all the 6 bits received. 6 bits are obtained from the matrix multiplication m*G
	n = 0
	m = 6

	sindromes = np.empty([1,3], int)
	decoder_output = []
	# Recorremos todos los bits codificados para hallar el sindrome con S = vH
	while n < encoder_output.shape[0]:
		multiplicacion_sindromes = np.remainder(encoder_output[n]@H, 2) #Multiplicacion de matrices y modulo 2

		if (n==0):
			sindromes = multiplicacion_sindromes 
		else:
			sindromes = np.vstack([sindromes, multiplicacion_sindromes])

		# Simple error correction with syndrome
		for i in range(m): #
			if (H[i] == multiplicacion_sindromes).all():

				if encoder_output[n][i] == 1:
					encoder_output[n][i] = 0
				else:
					encoder_output[n][i] = 1

		# Get the last 3 bits which are packaged according to the G matrix
		decoder_output.extend(encoder_output[n][3:6].tolist())
		# get the next 6 bits
		n+=1

	decoder_outputStr = ''.join(str(bit) for bit in decoder_output)
	diff = len(decoder_outputStr) - len(string)
	if diff != 0:
		decoder_outputStr = decoder_outputStr[:-diff]
		
	return decoder_outputStr, sindromes

def modulation(string, M, Ns):
	Amp_out = []
	#Paso de vectores a simbolos
	for i in range(0, len(string)):
		symb = ''.join(str(e) for e in string[i])
		ampl = 1+int(symb, 2)/(M-2)

		# Sampling
		for j in range(Ns):
			Amp_out.append(ampl) 

	i = 0
	ordered_symb = []
	while i < len(Amp_out):
		ordered_symb.append(Amp_out[i])
		i+=Ns
		
	return Amp_out, ordered_symb

def ASK(symbols, M, frecuency):
	
	sampling = 100

	Ts = 1/frecuency
	n = len(symbols)

	fs = sampling/Ts
	#Time vectors
	full_time = np.linspace(0, n*Ts, n*sampling)
	tp = np.linspace(0, Ts, sampling)
	
	#Signal vector
	signal = np.zeros(n*sampling)

	#Carrier signal
	carrier = np.cos(2*np.pi*frecuency*tp)

	for i,j in enumerate(symbols):
		signal[i*sampling:(i+1)*sampling] = j*(math.sqrt(2/Ts))*carrier
	
	span = 10000

	plt.plot(full_time[0:span], signal[0:span])
	plt.show()
	return signal

def DemASK(signal, frecuency):
	#Carrier signal
	sampling = 100
	Ts = 1/frecuency
	tp = np.linspace(0, Ts, sampling)
	analytic_signal = hilbert(signal) #Esta función crea el contorno superior de la señal 
	amplitud_envelope = np.abs(analytic_signal)
	plt.plot(signal, label='signal')
	plt.plot(amplitud_envelope, label='envelope')
	plt.show()
	
	symbol_array = [] #Arreglo donde se guardarán los valores de los símbolos extraidos
	
	average_sampling = 5
	i = 0
	while i < (len(amplitud_envelope)/sampling):
		average = 0
		for j in range((i*sampling)+average_sampling, (i+1)*sampling, average_sampling):
			average = average + amplitud_envelope[j]
		i+=1
		average = average/((sampling/average_sampling)-1)
		symbol = average/((math.sqrt(2/Ts)))
		symbol_array.append(symbol)
	return symbol_array
				
def demodulation(Amplitud_received, M, Ns, Pn):
	str_out = ''
	# Seq xn(k) grouped by Ns
	xnk = [Amplitud_received[i:i+Ns] for i in range(0, len(Amplitud_received), Ns)]

	# Prom of the Ns samples for each seq in xn(k)
	yn = [mean(xnk[i]) for i in range(len(xnk))]

	ai = [1+num/(M-2) for num in range(2**M)]
	#print(ai)

	#Asignación de medias a simbolos
	an = [min(ai, key=lambda x:abs(x-yn[i])) for i in range(len(yn))]

	for i in range(len(an)):
		symb = int((an[i]-1)*(M-2))
		bits = format(symb, '0' + str(M)+ 'b')
		str_out += bits
		
	demod_out = list(str_out)
	demod_out = [int(bit) for bit in str_out]
		

	while(len(demod_out)%6 != 0): #Checks for number of elements and adds 0 as desired
		demod_out.append(0)

	# Retransforming into NP array for decoding stage
	package = [demod_out[n:n+6] for n in range(0, len(demod_out), 6)]
		
	temp = []
	for i in package:
		temp.append(np.array(i))
	temp = np.asarray(temp)

	return temp, an 

def AWGN(signal, Pn):
	# Additive White Gaussian Noise
	noise = np.random.normal(0, math.sqrt(Pn),len(signal))
	for i in range(len(signal)):
		signal[i] += noise[i]
	return signal

def main():

	# Input file for the simulation
	try:
		input_file = sys.argv[1]
	except IndexError:
		print('\t[ERROR] You need to add the input file (text file/image/audio file/ ...) as arg')
		print('\t\tEg: python modulacion.py input.txt')
		exit()
	# setting the output files to the same location and extension as the input file
	output_bin, output_file = extension_handler(input_file)


	# Transforming the input file to binary
	bin_string = file2bin(input_file, output_bin)

	# Channel encoder calling
	trans_bin, zeroes, string = channelEncoder(bin_string)

	# Modulation calling
	Pn = 500		#AWGN Power
	Ns = 12
	M = 6			#M-PAM, es fijo

	#Se obtienen los simbolos
	Amp_out, ordered_symb = modulation(trans_bin, M, Ns)

	#Se modula en ASK
	signal = ASK(Amp_out, M, 50)

	#Se añade ruido
	Amp_out = AWGN(signal, Pn)

	symbol_array = DemASK(Amp_out, 100)

	demod_out_AWGN, an = demodulation(symbol_array, M, Ns, Pn)
	lista_err = []
	errores_nodecoder = 0

	for i in range(len(trans_bin)):
		for j in range(6):
			if(trans_bin[i][j] != demod_out_AWGN[i][j]):
				errores_nodecoder +=1
				if(i != 0):
					lista_err.append(((i+1)*6)+j+1)
				else:
					lista_err.append(j)

	lista3_err = []
	errores_nodecoder3 = 0

	for i in range(len(trans_bin)):
		for j in range(3,6):
			if(trans_bin[i][j] != demod_out_AWGN[i][j]):
				errores_nodecoder3 +=1
				if(i != 0):
					lista3_err.append(((i)*3)+j-3)
				else:
					lista3_err.append(j-2)

	print("Errores generados en todo el vector v en el canal por el ruido blanco: ", errores_nodecoder)

	print("Errores generados unicamente en mensaje m en el canal por el ruido blanco: ", errores_nodecoder3)

	rec_bin1, s2 = channelDecoder(demod_out_AWGN, string)

	errores2 = 0
	lista2_err = []
	for j in range(0, len(string)):
		if (rec_bin1[j] != string[j]):
			errores2+=1
			lista2_err.append(j)
	print("Errores recibidos después del decodificador: ",errores2)
	bin2file(rec_bin1,output_file)

if __name__ == '__main__':
	main()
