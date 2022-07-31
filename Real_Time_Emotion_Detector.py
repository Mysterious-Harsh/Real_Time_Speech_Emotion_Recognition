from turtle import color
import librosa
import librosa.display
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import IPython.display as ipd  # To play sound in the notebook
import pyaudio
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from matplotlib.animation import FuncAnimation
from keras.models import load_model
import noisereduce as nr
import math
import os, sys
from threading import Thread

matplotlib.use( 'MacOSX' )
plt.style.use( 'dark_background' )
plt.rc( 'figure', titlesize=16 )
plt.rc( 'axes', labelsize=12 )


class emotion_detector:
	# ['angry', 'calm', 'disgust', 'fear', 'happy', 'sad', 'surprise']
	def __init__( self, model_path, sample_rate=44100, threshold=122 ):
		self.MODEL = load_model( model_path )
		EMOTIONS = { 0: 'Angry', 1: 'Calm', 2: 'Disgust', 3: 'Fear', 4: 'Happy', 5: 'Sad', 6: 'Surprise' }
		self.ENC = OneHotEncoder()
		self.ENC.fit_transform(
		    [ [ 'Angry' ], [ 'Calm' ], [ 'Disgust' ], [ 'Fear' ], [ 'Happy' ], [ 'Sad' ], [ 'Surprise' ] ]
		    )
		self.THRESHOLD = threshold
		self.FORMAT = pyaudio.paFloat32
		self.CHANNELS = 1
		self.RATE = sample_rate
		self.sr = self.RATE
		self.CHUNK = 1024 * 4
		self.COLORS = [ 'red', 'lightblue', 'peru', 'grey', 'green', 'yellow', 'lightpink' ]
		self.EMOTION_LIST = list( EMOTIONS.values() )
		self.emotion = "Silence"
		self.predictions = [ 0, 0, 0, 0, 0, 0, 0 ]
		self.LENGTH = 121212
		self.AUDIO = pyaudio.PyAudio()
		self.FRAME_LENGTH = 2048
		self.HOP_LENGTH = 512
		self.RECORD_SECONDS = 2.6
		self.audio = ( np.sin( np.pi * np.arange( self.RECORD_SECONDS ) * 54 / self.sr ) ).astype( np.float32 )

		self.SCALER = StandardScaler()
		self.FIG, self.AXES = plt.subplots( 1, 2, figsize=( 14, 8 ), tight_layout=True )
		self.FIG.canvas.mpl_connect( 'close_event', self.stop_stream )
		self.stop_flag = False
		self.emo_color = { self.EMOTION_LIST[ i ]: self.COLORS[ i ] for i in range( len( self.EMOTION_LIST ) ) }
		self.emo_color[ "Silence" ] = 'white'

	def __analyser( self, frame ):
		self.AXES[ 0 ].clear()
		self.AXES[ 1 ].clear()

		self.AXES[ 0 ].set_ylim( 0, 1.2 )
		self.AXES[ 1 ].set_ylim( -1, 1 )

		plt.suptitle( "\n\n\n" + self.emotion.capitalize(), va='center', fontweight="bold" )
		self.AXES[ 0 ].bar( self.EMOTION_LIST, self.predictions, color=self.COLORS )

		librosa.display.waveshow( y=self.audio, sr=self.sr, ax=self.AXES[ 1 ], color=self.emo_color[ self.emotion ] )

	def __extract_features( self, audio, sr ):
		rms = []
		mfcc = []
		mel = []

		# Fetch the sample rate.
		normalizedsound = librosa.util.normalize( audio )

		# Trim silence from the beginning and the end.
		trimmed_audio, index = librosa.effects.trim( y=normalizedsound, top_db=30 )

		final_audio = np.pad( trimmed_audio, ( 0, self.LENGTH - len( trimmed_audio ) ), 'constant' )
		# Noise reduction.
		# final_audio = nr.reduce_noise(y=padded_audio,
		#                               sr=self.RATE)  #updated 03/03/22

		f1 = librosa.feature.rms(
		    y=final_audio, frame_length=self.FRAME_LENGTH, hop_length=self.HOP_LENGTH
		    ).T  # Energy - Root Mean Square

		f2 = librosa.feature.melspectrogram(
		    y=final_audio, sr=sr, n_fft=self.FRAME_LENGTH, hop_length=self.HOP_LENGTH
		    ).T

		f3 = librosa.feature.mfcc( y=final_audio, sr=sr, n_mfcc=40, hop_length=self.HOP_LENGTH ).T  # MFCC

		# Filling the data lists

		rms.append( self.SCALER.fit_transform( f1 ) )
		mel.append( self.SCALER.fit_transform( f2 ) )
		mfcc.append( self.SCALER.fit_transform( f3 ) )

		f_rms = np.asarray( rms )
		f_mel = np.asarray( mel )
		f_mfccs = np.asarray( mfcc )

		# Concatenating all features to 'X' variable.
		features = np.concatenate( ( f_rms, f_mel, f_mfccs ), axis=2 )
		return features

	def __emotion( self, audio_features ):
		predictions = self.MODEL.predict( audio_features, use_multiprocessing=True )
		# print(predictions)

		max_emo = self.ENC.inverse_transform( predictions )
		pred_list = list( predictions )
		predictions = np.squeeze( np.array( pred_list ).tolist(), axis=0 )
		# print(predictions)

		return predictions, max_emo[ 0 ][ 0 ]

	def list_devices( self ):
		print( "----------------------record device list---------------------" )
		info = self.AUDIO.get_host_api_info_by_index( 0 )
		numdevices = info.get( 'deviceCount' )
		for i in range( 0, numdevices ):
			if ( self.AUDIO.get_device_info_by_host_api_device_index( 0, i ).get( 'maxInputChannels' ) ) > 0:
				print(
				    "Input Device id ", i, " - ",
				    self.AUDIO.get_device_info_by_host_api_device_index( 0, i ).get( 'name' )
				    )

		print( "-------------------------------------------------------------" )

		index = int( input() )
		return index

	def silence( self, audio ):
		threshold = ( sum( audio ) / len( audio ) )
		# print(threshold)
		return ( sum( audio ) / len( audio ) ) < self.THRESHOLD

	def __start( self, file=None, device_index=None ):

		if file != None:
			try:
				audio_data, self.sr = librosa.load( self.file )
				ipd.display( ipd.Audio( data=audio_data, rate=sr ) )
				n = len( audio_data ) / self.LENGTH
				if n < 1:
					self.audio = audio_data
					features = self.__extract_features( audio_data, sr )
					self.predictions, self.emotion = self.__emotion( features )

				else:
					for i in range( math.floor( n ) ):
						self.audio = audio_data[ self.LENGTH * i : self.LENGTH * ( i + 1 ) ]
						features = self.__extract_features( audio, sr )
						self.predictions, self.emotion = self.__emotion( features )
						print( self.predictions, self.emotion )

					else:
						self.audio = audio_data[ self.LENGTH * i : ]
						features = self.__extract_features( audio, sr )
						self.predictions, self.emotion = self.__emotion( features )
						print( self.predictions, self.emotion )

			except Exception as e:
				print( e )

		else:
			if device_index == None:
				print( "Missing Device Index Or File !" )
				sys.exit( 1 )
				# index = self.list_devices()
			print( "recording via index " + str( device_index ) )

			self.STREAM = self.AUDIO.open(
			    format=self.FORMAT,
			    channels=self.CHANNELS,
			    rate=self.RATE,
			    input=True,
			    input_device_index=device_index,
			    frames_per_buffer=self.CHUNK
			    )
			self.sr = self.RATE
			# try:
			while not self.stop_flag:
				# print("recording started")
				Recordframes = []

				for i in range( 0, int( self.RATE / self.CHUNK * self.RECORD_SECONDS ) ):
					data = self.STREAM.read( self.CHUNK, exception_on_overflow=False )
					Recordframes.append( data )
				# print ("recording stopped")
				self.audio = np.frombuffer( b''.join( Recordframes ), dtype=np.float32 )
				# ipd.display(ipd.Audio(data=audio, rate=self.RATE))
				# time.sleep(5)
				if self.silence( Recordframes[ -1 ] ):
					# print("Silence Detected !")
					self.emotion = "Silence"
					self.predictions = [ 0, 0, 0, 0, 0, 0, 0 ]
				else:
					features = self.__extract_features( self.audio, self.RATE )
					self.predictions, self.emotion = self.__emotion( features )
					# print(emotion)

		print( "Main Thread Terminated !" )

	def start_stream( self, file=None, device_index=None ):
		print( "Stream Started !" )
		self.main_thread = Thread( target=self.__start, args=( file, device_index ) )
		self.main_thread.start()

		self.anim = FuncAnimation( fig=self.FIG, func=self.__analyser, interval=1 )
		plt.show()

	def stop_stream( self, event=None ):
		print( "Stream Stoped !" )
		self.stop_flag = True
		self.main_thread.join()
		self.STREAM.stop_stream()
		self.STREAM.close()
		self.AUDIO.terminate()


ed = emotion_detector( 'Models/SER.hdf5' )
device_index = ed.list_devices()
ed.start_stream( device_index=device_index )

RAVDESS = "Data/RAVDESS/audio_speech_actors_01-24/"
TESS = "Data/TESS/"
datafiles = []
for i in os.listdir( TESS ):
	datafiles.append( TESS + i )

for i in os.listdir( RAVDESS ):
	if os.path.isdir( RAVDESS + i ):
		for j in os.listdir( RAVDESS + i ):
			datafiles.append( RAVDESS + i + '/' + j )
