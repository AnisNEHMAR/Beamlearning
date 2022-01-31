# mu32power.py python program example for MegaMicro Mu32 transceiver 
#
# Copyright (c) 2022 Distalsense
# Author: bruno.gas@distalsense.com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Run the Mu32 system during one second for getting and ploting signals comming from
8 activated microphones 

Documentation is available on https://distalsense.io

Please, note that the following packages should be installed before using this program:
	> pip install libusb1
	> pip install matplotlib
"""

welcome_msg = '-'*20 + '\n' + 'Mu32 power program\n \
Copyright (C) 2022  Distalsense\n \
This program comes with ABSOLUTELY NO WARRANTY; for details see the source code\'.\n \
This is free software, and you are welcome to redistribute it\n \
under certain conditions; see the source code for details.\n' + '-'*20

import logging
import numpy as np
import queue
import matplotlib.pyplot as plt
from mu32.core import Mu32, mu32log

mu32log.setLevel( logging.INFO )

"""
Power signal queue
"""
signal_q = queue.Queue()


def main():

	print( welcome_msg )

	try:
		mu32 = Mu32()
		mu32.run( 
			mems=(0, 1, 2, 3,4,5,6),							# activated mems
			post_callback_fn=my_process_function, 		# the user defined data processing function
			callback_fn=my_callback_fn
		)
	except:
		print( 'aborting' )


def my_callback_fn( mu32: Mu32, data ):
	"""
	Compute energy (mean power) on transfered frame and push it in the queue
	"""
	signal = data.reshape( mu32.buffer_length, mu32.mems_number )
	signal = np.float32( signal )
	mean_power = np.sum( signal**2, axis=0 ) / mu32.buffer_length

	signal_q.put( mean_power )


def my_process_function( mu32: Mu32 ):
	"""
	The data processing function is called  after the acquisition process has finished.
	Plot signal mean power comming from activated MEMs 
	"""

	q_size = signal_q.qsize()
	if q_size== 0:
		raise Exception( 'No received data !' )

	print( 'got %d transfer buffers from %d microphones (recording time is: %f s)' % (q_size, mu32.mems_number, q_size*mu32.buffer_length/mu32.sampling_frequency) )	

	"""
	get queued signals from Mu32
	"""
	signal = []
	for _ in range( q_size ):
		signal = np.append( signal, signal_q.get( block=False ) )

	signal = signal.reshape( q_size, mu32.mems_number )

	"""
	plot mems signals 
	"""
	fig, axs = plt.subplots( mu32.mems_number )
	fig.suptitle('Mems energy')
	time = np.array( [t for t in range( q_size )] ) * mu32.buffer_length / mu32.sampling_frequency
	for s in range( mu32.mems_number ):
		X = signal[:,s]
		axs[s].plot( time, X )
		axs[s].set( xlabel='time in seconds', ylabel='mic %d' % s )

	plt.show()



if __name__ == "__main__":
	main()