import numpy as np
import os
import time
import multiprocessing as mp

from dataset import GraphStack, writeJSON

class States(object):
    def __init__(self, g):

        self.xMax = None
        self.xMin = None
        self.yMax = None
        self.yMin = None

        self.zMax = None
        self.zMin = None
        self.disp = None

        self.diag = []
        self.beamLen = []

        self.getStats(g)

    def getStats(self, g):

        nodes, edges, nodesTar, edgesTar = g

        allPts = np.concatenate((nodes.reshape((-1, 3)), edges.reshape((-1, 3))), axis=0)
        tarPts = np.concatenate((nodesTar.reshape((-1, 3)), edgesTar.reshape((-1, 3))), axis=0)

        self.xMax, self.yMax, _ = np.amax(allPts, axis=0)
        self.xMin, self.yMin, _ = np.amin(allPts, axis=0)

        _, _, self.zMax = np.amax(tarPts, axis=0)
        _, _, self.zMin = np.amin(tarPts, axis=0)

        self.diag = (((self.xMax - self.xMin) ** 2) * ((self.yMax - self.yMin) ** 2)) ** .5
        self.disp = float(max(abs(self.zMax), abs(self.zMin)))

        for i in range(len(edges)):
            e = edges[i]
            p0 = e[3:6]
            p1 = e[27:30]
            v = p1 - p0
            dist = float(np.sqrt(np.sum(v * v)))
            self.beamLen += [dist]

def getGraph(graphs, i):

    nodes = graphs.nodes[i, 0]
    edges = graphs.edges[i, 0]
    nodesLast = graphs.nodes[i, -1] + graphs.nodes[i, -1]
    edgesLast = graphs.edgesTar[i, 0] + graphs.edges[i, -1]

    return nodes, edges, nodesLast, edgesLast

def getPoints(graph):

    edges = edges[:,:39]
    nodes = nodes[:,:27]

    edgeCount = edges.shape[0]
    nodeCount = nodes.shape[0]

def main():

    stats = []
    data = GraphStack().load("./data")

    for i in range(data.count):
        g = getGraph(data, i)
        stats += [States(g)]

    area = []
    beamLen = []
    disp = []
    for g in stats:
        area += [g.diad]
        disp += [g.disp]
        beamLen += g.beamLen

    writeJSON("area.json", area)
    writeJSON("disp.json", disp)
    writeJSON("beamLen.json", beamLen)


if __name__ == "__main__":
    main()