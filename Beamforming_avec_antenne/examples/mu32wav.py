# mu32wav.py python program example for MegaMicro Mu32 transceiver 
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
Run the Mu32 system during some seconds and records signals comming from
2 activated microphones.

Documentation is available on https://distalsense.io

Please, note that the following packages should be installed before using this program:
	> pip install libusb1
    > pip install matplotlib
    > pip install sounddevice
"""

welcome_msg = '-'*20 + '\n' + 'Mu32 wav program\n \
Copyright (C) 2022  distalsense\n \
This program comes with ABSOLUTELY NO WARRANTY; for details see the source code\'.\n \
This is free software, and you are welcome to redistribute it\n \
under certain conditions; see the source code for details.\n' + '-'*20

import logging
import numpy as np
import wave
import sounddevice as sd
import matplotlib.pyplot as plt
from core import Mu32, mu32log

mu32log.setLevel( logging.INFO )

def main():

    print( welcome_msg )
    
    try:
        mu32 = Mu32()
        mu32.run(
            
		    post_callback_fn=my_callback_end_function, 	# the user defined data processing function
		    mems=(0,1,2,3,4,5,6),                                # activated mems
            duration=10                                  # recording time
        )
    except Exception as e:
        print( str( e ) )



def my_callback_end_function( mu32: Mu32 ):
    """
    The data processing function is called  after the acquisition process has finished.
    It records signals from the two microphones that have been activated in the Mu32.run() function 
    """
    q_size = mu32.signal_q.qsize()
    if q_size== 0:
        raise Exception( 'No received data !' )

    print( 'got %d transfer buffers from %d microphones (recording time is: %f s)' % (q_size, mu32.mems_number, q_size*mu32.buffer_length/mu32.sampling_frequency) )	

    """
    Open wavfile and write header
    """
    MEMS=(0, 1, 2,3,4,5,6)
    MEMS_NUMBER = len( MEMS )
    
    signal=[]

    for _ in range( q_size ):
        signal = np.append( signal, mu32.signal_q.get( block=False ) )
    samples_number = int( len( signal )/MEMS_NUMBER )
    signal = signal.reshape( samples_number, MEMS_NUMBER )

    wav_filename = ['test0.wav','test1.wav','test2.wav','test3.wav','test4.wav','test5.wav','test6.wav']
    for i in range (len(wav_filename)):
        wavfile = wave.open( wav_filename[i], mode='wb' )
        wavfile.setnchannels(1)
        wavfile.setsampwidth(1)
        wavfile.setframerate( mu32.sampling_frequency )

        """
        get queued signals from Mu32 and save them in wavfile
        """

        
        for _ in range( mu32.signal_q.qsize() ):
            wavfile.writeframesraw( np.int8( signal[:,i] ))
            print(signal.shape)
            
        wavfile.close()


