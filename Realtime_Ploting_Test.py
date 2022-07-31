from distutils.command.clean import clean
import time
from turtle import clear
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import random
from functools import partial
from threading import Thread


class ani:

	def __init__( self ) -> None:

		self.fig, self.axes = plt.subplots( 1, 2, figsize=( 8, 6 ), tight_layout=True )
		self.fig.canvas.mpl_connect( 'close_event', self.close )
		plt.style.use( "seaborn" )

		self.l1 = [ i if i < 20 else 20 for i in range( 100 ) ]
		self.l2 = [ i if i < 85 else 85 for i in range( 100 ) ]
		self.l3 = [ i if i < 30 else 30 for i in range( 100 ) ]
		self.l4 = [ i if i < 65 else 65 for i in range( 100 ) ]
		self.palette = list( reversed( sns.color_palette( "seismic", 4 ).as_hex() ) )
		self.y1, self.y2, self.y3, self.y4 = [ 100, 100, 100, 100 ]
		self.close_flag = False

	def close( self, event ):
		print( event )
		self.close_flag = True

	def animate( self, i ):

		# self.y1, self.y2, self.y3, self.y4 = self.get_values()
		# self.axes[ 0 ].clear()
		self.axes[ 0 ].set_ylim( 0, 100 )
		self.axes[ 0 ].bar(
		    [ "one", "two", "three", "four" ], sorted( [ self.y1, self.y2, self.y3, self.y4 ] ), color=self.palette
		    )

	def get_values( self ):
		while not self.close_flag:
			self.y1 = random.choice( self.l1 )
			self.y2 = random.choice( self.l2 )
			self.y3 = random.choice( self.l3 )
			self.y4 = random.choice( self.l4 )

			time.sleep( 1 )
		print( "Stoped !" )

	def start( self ):
		self.new_thread = Thread( target=self.get_values )

		self.new_thread.start()

		plt.title( "Animated Bars", color=( "blue" ) )
		anim = animation.FuncAnimation( fig=self.fig, func=self.animate, interval=1 )
		# anim.save("bar2.gif", writer="imagemagick")
		plt.show()
		self.new_thread.join()


a = ani()

a.start()
