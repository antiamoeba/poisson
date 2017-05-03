#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from scipy import misc
import IPython


class PanoramaViewer:
	def __init__(self, image_name):
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111)
		plt.axis('off')
		plt.title('Use left and right arrows to pan. Any key to quit.')
		self.image = misc.imread(image_name)[:,:,:3]
		self.offset = 0
		self.width = self.image.shape[0] * 3  # Viewer width
		self.canvas = self.ax.imshow(self.image[:,self.offset:self.offset+self.width])

		self.callback_id_press = self.fig.canvas.mpl_connect('key_press_event', self.onArrowDown)
		plt.show()
		return

	def onArrowDown(self, event):
		if event.key == "left" or event.key == "right":
			if event.key == "left":
				self.offset -= 50
			elif event.key == "right":
				self.offset += 50
			if self.offset < 0:
				self.offset += self.image.shape[1]
			elif self.offset > self.image.shape[1]:
				self.offset -= self.image.shape[1]
			new_image = self.image[:,self.offset:self.offset+self.width]
			if new_image.shape[1] < self.width:
				new_image = np.append(new_image, self.image[:,:(self.width - new_image.shape[1])], axis=1)
			self.canvas.set_data(new_image)
			plt.draw()
		else:
			plt.close()
		return






if __name__ == "__main__":
	print ""
	PanoramaViewer("output/image1.png")
