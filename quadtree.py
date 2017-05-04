from scipy.ndimage import variance
import numpy as np
import matplotlib.pyplot as plt

def displayImage(image, title=None):
    if len(image.shape) == 2:
        image = np.dstack([image for _ in range(3)])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis('off')
    ax.imshow(image)
    plt.show()
    return ax
class QuadTree:
    def __init__(self, img, min_var=0.001, min_size=1):
        self.leaves = dict()
        self.img = img
        self.min_var = min_var
        self.min_size = min_size
        self.root = NodeBox((0,0), img.shape, self)
    def getIndex(self, query):
        return self.root.get(query)
    
class NodeBox:
    def __init__(self, point_tl, point_br, qt):
        self.point_tl = point_tl
        self.point_br = point_br
        self.qt = qt
        self.leaf = False
        segment = qt.img[point_tl[0]:point_br[0], point_tl[1]:point_br[1]]
        if segment.shape[0] * segment.shape[1] < qt.min_size:
            self.leaf = True
            self.qt.leaves[self] = len(self.qt.leaves)
            return
        curr_var = variance(segment)
        if curr_var < qt.min_var:
            self.leaf = True
            self.qt.leaves[self] = len(self.qt.leaves)
            return
        #build children
        midpoint = (int((point_tl[0] + point_br[0])/2), int((point_tl[1] + point_br[1])/2))
        if abs(point_tl[0] - point_br[0]) == 1:
            self.tl = NodeBox(point_tl, (point_br[0], midpoint[1]), qt)
            self.br = NodeBox((point_tl[0],midpoint[1]), point_br, qt)
        elif abs(point_tl[1] - point_br[1]) == 1:
            self.tl = NodeBox(point_tl, (midpoint[0], point_br[1]), qt)
            self.br = NodeBox((midpoint[0],point_tl[1]), point_br, qt)
        else:
            self.tl = NodeBox(point_tl, midpoint, qt)
            self.br = NodeBox(midpoint, point_br, qt)
            self.tr = NodeBox((point_tl[0], midpoint[1]), (midpoint[0], point_br[1]), qt)
            self.bl = NodeBox((midpoint[0], point_tl[1]), (point_br[0], midpoint[1]), qt)
    def contains(self, query):
        if query[0] >= self.point_tl[0] and query[0] < self.point_br[0] and query[1] >= self.point_tl[1] and query[1] < self.point_br[1]:
            return True
        return False
    def get(self, query):
        if self.leaf == True:
            return self.qt.leaves[self], self
        else:
            if self.tl.contains(query):
                return self.tl.get(query)
            if self.br.contains(query):
                return self.br.get(query)
            if self.tr.contains(query):
                return self.tr.get(query)
            if self.bl.contains(query):
                return self.bl.get(query)