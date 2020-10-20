import numpy as np
import math as m
import copy
import json
import torch
import random as r
import multiprocessing as mp

from parameters import *

class GraphStack(object):
    def __init__(self, input=None):

        # information
        self.count = None
        self.trainRatio = TRAIN_RATIO
        self.type = None # "torch", "numpy", "list"

        # constant attributes
        self.edgeNumType = None
        self.nodeNumType = None
        self.edgeCenter = None
        self.nodeCenter = None

        # variable attributes
        self.adjMat = None
        self.nodes = None
        self.edges = None
        self.nodesTar = None
        self.edgesTar = None

        # noise
        self.nodesCumSumDelta = None
        self.edgesCumSumDelta = None
        self.cumSumInit = False
        
        if(isinstance(input, list)):
            if(isinstance(input[0], dict)):
                self.fromDicts(input)
            else:
                self.fromGraphs(input)

    def __repr__(self):
        
        info = "dataset of %d trials" % self.count
        
        adjMatSize = "adjacency matrices size: " + str(self.adjMat.shape)
        nodesSize = "nodes matrices size: " + str(self.nodes.shape)
        edgesSize = "edges matrices size: " + str(self.edges.shape)

        message = '\n'.join([info, adjMatSize, nodesSize, edgesSize])

        return message

    def __len__(self):

        return self.count

    ###########################################################################
    # creation
    ###########################################################################

    def fromDicts(self, trial):

        # constant attributes
        self.edgeNumType = trial[0][EDGE_NUMTYPE_TOKEN]
        self.nodeNumType = trial[0][NODE_NUMTYPE_TOKEN]
        self.edgeCenter = trial[0][EDGE_CEN_TOKEN]
        self.nodeCenter = trial[0][NODE_CEN_TOKEN]

        # variable attributes
        frames = len(trial)
        nodeCount = len(trial[0][NODES_TOKEN])
        edgeCount = len(trial[0][EDGES_TOKEN])
        nodeLen = len(self.nodeNumType)
        edgeLen = len(self.edgeNumType)
        self.adjMat = np.zeros((frames, edgeCount, nodeCount))
        self.nodes = np.zeros((frames, nodeCount, nodeLen))
        self.edges = np.zeros((frames, edgeCount, edgeLen))
        self.nodesTar = np.zeros((frames, nodeCount, nodeLen))
        self.edgesTar = np.zeros((frames, edgeCount, edgeLen))

        for i in range(len(trial)):
            frame = trial[i]
            nodes = np.asarray(frame[NODES_TOKEN])
            edges = np.asarray(frame[EDGES_TOKEN])
            nodesTar = np.asarray(frame[NODES_TAR_TOKEN])
            edgesTar = np.asarray(frame[EDGES_TAR_TOKEN])
            adjMat = np.asarray(frame[ADJ_MATRIX_TOKEN])

            self.adjMat[i] = adjMat
            self.nodes[i] = nodes
            self.edges[i] = edges
            self.nodesTar[i] = nodesTar
            self.edgesTar[i] = edgesTar

        # cast type
        self.adjMat = self.adjMat.astype(NUMPY_DTYPE_BIN)
        self.nodes = self.nodes.astype(NUMPY_DTYPE)
        self.edges = self.edges.astype(NUMPY_DTYPE)
        self.nodesTar = self.nodesTar.astype(NUMPY_DTYPE)
        self.edgesTar = self.edgesTar.astype(NUMPY_DTYPE)
        
        # update self information
        self.count = 1
        self.type = "numpy"

    def fromGraphs(self, stack):
        
        # constant attributes
        self.edgeNumType = stack[0].edgeNumType
        self.nodeNumType = stack[0].nodeNumType
        self.edgeCenter = stack[0].edgeCenter
        self.nodeCenter = stack[0].nodeCenter

        # variable attributes
        self.adjMat = np.concatenate([graph.adjMat for graph in stack], axis=0).astype(NUMPY_DTYPE_BIN)
        self.nodes = np.concatenate([graph.nodes for graph in stack], axis=0).astype(NUMPY_DTYPE)
        self.edges = np.concatenate([graph.edges for graph in stack], axis=0).astype(NUMPY_DTYPE)
        self.nodesTar = np.concatenate([graph.nodesTar for graph in stack], axis=0).astype(NUMPY_DTYPE)
        self.edgesTar = np.concatenate([graph.edgesTar for graph in stack], axis=0).astype(NUMPY_DTYPE)
        self.count = len(self.adjMat)
        self.type = "numpy"

    ###########################################################################
    # type
    ###########################################################################

    def isTorch(self):
        return self.type == "torch"

    def isNumpy(self):
        return self.type == "numpy"

    def isList(self):
        return self.type == "list"

    def toTorch(self, device=DEVICE):
        
        self.toType("torch", device)

    def toNumpy(self):

        self.toType("numpy")

    def toList(self):

        self.toType("list")

    def toType(self, tarType, device=DEVICE):
        
        if(tarType == "torch"):
            if(device == DEVICE_GPU): adjMatType = TORCH_DTYPE
            else: adjMatType = TORCH_DTYPE_BIN
        if(tarType == "numpy"):
            adjMatType = NUMPY_DTYPE_BIN
        self.adjMat = changeType(self.adjMat, tarType, device, dType=adjMatType)
        self.nodes = changeType(self.nodes, tarType, device)
        self.edges = changeType(self.edges, tarType, device)
        self.nodesTar = changeType(self.nodesTar, tarType, device)
        self.edgesTar = changeType(self.edgesTar, tarType, device)

        if(self.cumSumInit):
            self.nodesCumSumDelta = changeType(self.nodesCumSumDelta, tarType, device)
            self.edgesCumSumDelta = changeType(self.edgesCumSumDelta, tarType, device)
        
        self.type = tarType

    def toDevice(self, device):
        
        self.adjMat = self.adjMat.to(device)
        self.nodes = self.nodes.to(device)
        self.edges = self.edges.to(device)
        self.nodesTar = self.nodesTar.to(device)
        self.edgesTar = self.edgesTar.to(device)

        if(self.cumSumInit):
            self.nodesCumSumDelta = self.nodesCumSumDelta.to(device)
            self.edgesCumSumDelta = self.edgesCumSumDelta.to(device)

    ###########################################################################
    # util
    ###########################################################################

    def shuffle(self):

        if(self.type == "torch"):
            newIndices = torch.randperm(self.count)
            self.adjMat = self.adjMat[newIndices]
            self.nodes = self.nodes[newIndices]
            self.edges = self.edges[newIndices]
            self.nodesTar = self.nodesTar[newIndices]
            self.edgesTar = self.edgesTar[newIndices]
        elif(self.type == "numpy"):
            np.random.shuffle(self.adjMat)
            np.random.shuffle(self.nodes)
            np.random.shuffle(self.edges)
            np.random.shuffle(self.nodesTar)
            np.random.shuffle(self.edgesTar)
        elif(self.type == "list"):
            r.shuffle(self.adjMat)
            r.shuffle(self.nodes)
            r.shuffle(self.edges)
            r.shuffle(self.nodesTar)
            r.shuffle(self.edgesTar)
    
    def cut(self, index):

        self.adjMat = self.adjMat[:index]
        self.nodes = self.nodes[:index]
        self.edges = self.edges[:index]
        self.nodesTar = self.nodesTar[:index]
        self.edgesTar = self.edgesTar[:index]

        self.count = len(self.adjMat)

    def _copy(self, copyInfo=True, copyVal=True):

        new = GraphStack()

        if(copyInfo):
            # information
            new.count = self.count
            new.trainRatio = self.trainRatio
            new.type = self.type

            # constant attributes
            new.edgeNumType = copy.deepcopy(self.edgeNumType)
            new.nodeNumType = copy.deepcopy(self.nodeNumType)
            new.edgeCenter = copy.deepcopy(self.edgeCenter)
            new.nodeCenter = copy.deepcopy(self.nodeCenter)

        if(copyVal):
            # variable attributes
            if(self.isNumpy()):
                new.adjMat = np.copy(self.adjMat)
                new.nodes = np.copy(self.nodes)
                new.edges = np.copy(self.edges)
                new.nodesTar = np.copy(self.nodesTar)
                new.edgesTar = np.copy(self.edgesTar)
            elif(self.isTorch()):
                new.adjMat = self.adjMat.clone()
                new.nodes = self.nodes.clone()
                new.edges = self.edges.clone()
                new.nodesTar = self.nodesTar.clone()
                new.edgesTar = self.edgesTar.clone()
            elif(self.isList()):
                new.adjMat = copy.deepcopy(self.adjMat.clone())
                new.nodes = copy.deepcopy(self.nodes.clone())
                new.edges = copy.deepcopy(self.edges.clone())
                new.nodesTar = copy.deepcopy(self.nodesTar.clone())
                new.edgesTar = copy.deepcopy(self.edgesTar.clone())

        return new

    def swapNodeEdge(self):

        if(self.isNumpy()):
            dim = self.adjMat.ndim
            self.adjMat = np.swapaxes(self.adjMat, dim - 2, dim - 1)
        elif(self.isTorch()):
            dim = self.adjMat.dim
            self.adjMat = torch.transpose(self.adjMat, dim - 2, dim - 1)
        
        self.nodes, self.edges = self.edges, self.nodes
        self.nodesTar, self.edgesTar = self.edgesTar, self.nodesTar
        self.nodeNumType, self.edgeNumType = self.edgeNumType, self.nodeNumType
        self.nodeCenter, self.edgeCenter = self.edgeCenter, self.nodeCenter

    def rotateCopy(self, angles):

        # set matrices to appropriate sizes
        copies = len(angles)
        adjMats = np.expand_dims(self.adjMat, axis=0)
        adjMats = np.repeat(adjMats, copies, axis=0)

        nodes = np.expand_dims(self.nodes, axis=0)
        nodes = np.repeat(nodes, copies, axis=0)

        edges = np.expand_dims(self.edges, axis=0)
        edges = np.repeat(edges, copies, axis=0)

        nodesTar = np.expand_dims(self.nodesTar, axis=0)
        nodesTar = np.repeat(nodesTar, copies, axis=0)

        edgesTar = np.expand_dims(self.edgesTar, axis=0)
        edgesTar = np.repeat(edgesTar, copies, axis=0)

        # rotate to eliminate biases
        for i in range(copies):
            angle = angles[i]
            trans = Transformer(angle)
            nodesNew, edgesNew, nodesTarNew, edgesTarNew = self.transform(trans)
            nodes[i] = nodesNew
            edges[i] = edgesNew
            nodesTar[i] = nodesTarNew
            edgesTar[i] = edgesTarNew

        # dump
        self.adjMat = adjMats
        self.nodes = nodes
        self.edges = edges
        self.nodesTar = nodesTar
        self.edgesTar = edgesTar

    def transform(self, trans):

        frameCount = self.nodes.shape[0]

        nodesNew = np.copy(self.nodes)
        edgesNew = np.copy(self.edges)
        nodesTarNew = np.copy(self.nodesTar)
        edgesTarNew = np.copy(self.edgesTar)

        for i in range(frameCount):
            nodesNew[i] = np.asarray(trans.transform(self.nodes[i], self.nodeNumType))
            edgesNew[i] = np.asarray(trans.transform(self.edges[i], self.edgeNumType))
            nodesTarNew[i] = np.asarray(trans.transform(self.nodesTar[i], self.nodeNumType))
            edgesTarNew[i] = np.asarray(trans.transform(self.edgesTar[i], self.edgeNumType))

        return nodesNew, edgesNew, nodesTarNew, edgesTarNew

    def _transferView(self, other, start=None, end=None):

        if(start == None): start = 0
        if(end == None): end = self.count

        other.adjMat = self.adjMat[start:end]
        other.nodes = self.nodes[start:end]
        other.edges = self.edges[start:end]
        other.nodesTar = self.nodesTar[start:end]
        other.edgesTar = self.edgesTar[start:end]

        other.count = end - start

    def _foldShift(self, fold):

        chunkRatio = 1 / fold[1]

        mutPerTrial = ROT_PER_TRIAL
        if(USE_MIRROR): mutPerTrial *= 2
        trials = int(self.count / mutPerTrial)

        positions = [int(chunkRatio * i * trials * mutPerTrial) for i in range(fold[1] + 1)]
        positions[-1] = trials * mutPerTrial

        trainPairs = []
        for i in range(fold[1]):
            new = (positions[i], positions[i + 1])
            trainPairs += [new]
        testPair = trainPairs.pop(fold[1] - fold[0] - 1)
        
        self.adjMat = self._rePos(self.adjMat, trainPairs, testPair)
        self.nodes = self._rePos(self.nodes, trainPairs, testPair)
        self.edges = self._rePos(self.edges, trainPairs, testPair)
        self.nodesTar = self._rePos(self.nodesTar, trainPairs, testPair)
        self.edgesTar = self._rePos(self.edgesTar, trainPairs, testPair)
        
    def _rePos(self, matrix, trainPairs, testPair):
        
        if(isinstance(matrix, np.ndarray)): mode = "numpy"
        elif(isinstance(matrix, torch.Tensor)): mode = "torch"
        temp = []
        for start, end in trainPairs: # train
            temp += [matrix[start: end]]
        temp += [matrix[testPair[0]:testPair[1]]] # test

        if(mode == "torch"):
            result = torch.cat(temp)
        elif(mode == "numpy"):
            result = np.concatenate(temp)

        return result

    def splitData(self, perc=None, fold=(0, 5)):

        if(perc): slicePerc = perc
        else: slicePerc = self.trainRatio

        self._foldShift(fold)

        mutPerTrial = ROT_PER_TRIAL
        if(USE_MIRROR): mutPerTrial *= 2
        trials = self.count / mutPerTrial
        slicePos = int(round(trials * slicePerc)) * mutPerTrial
        
        slice0 = self._copy(copyVal=False)
        slice1 = self._copy(copyVal=False)

        self._transferView(slice0, end=slicePos)
        self._transferView(slice1, start=slicePos)

        return slice0, slice1

    def calDelta(self):

        self.nodesTar = self.nodesTar - self.nodes
        self.edgesTar = self.edgesTar - self.edges

    def preNormDelta(self, norm):
        # pytorch only
        
        oldDevice = self.nodesTar.device
        # nodes target
        # cast to the normalizer's device
        self.nodesTar = self.nodesTar.to(norm.device)
        # normalize output
        self.nodesTar = norm.normNodeOut(self.nodesTar)
        # cast to original device
        self.nodesTar = self.nodesTar.to(oldDevice)
        torch.cuda.empty_cache()

        # edges
        self.edgesTar = self.edgesTar.to(norm.device)
        self.edgesTar = norm.normEdgeOut(self.edgesTar)
        self.edgesTar = self.edgesTar.to(oldDevice)
        torch.cuda.empty_cache()

    def toSingleGraph(self):

        def to3D(mat):

            newShape = (-1, mat.shape[-2], mat.shape[-1])
            newMat = mat.reshape(newShape)
            
            return newMat

        self.adjMat = to3D(self.adjMat)
        self.nodes = to3D(self.nodes)
        self.edges = to3D(self.edges)
        self.nodesTar = to3D(self.nodesTar)
        self.edgesTar = to3D(self.edgesTar)

        if(self.cumSumInit):
            self.nodesCumSumDelta = to3D(self.nodesCumSumDelta)
            self.edgesCumSumDelta = to3D(self.edgesCumSumDelta)

        self.count = self.adjMat.shape[0]

    def iterMode(self):
        # hardcoded
        self.adjMat = self.adjMat[:, 0, :, :] # 3D
        self.nodes = self.nodes[:, 0, :, :] # 3D
        self.edges = self.edges[:, 0, :, :] # 3D

        self.nodesTar = self.nodesTar[:, :SIM_INCREMENT, :, :]
        self.edgesTar = self.edgesTar[:, :SIM_INCREMENT, :, :]

    def getCumSumDelta(self):
        # only workable with in 4D mode

        assert(len(self.adjMat.shape) == 4)
        repeat = [0] * self.adjMat.shape[1]
        self.nodesCumSumDelta = self.nodes - self.nodes[:,repeat,:,:]
        self.edgesCumSumDelta = self.edges - self.edges[:,repeat,:,:]

        self.cumSumInit = True

    def cutLast(self):
        # only  works with 4D data

        assert(len(self.adjMat.shape) == 4)
        self.adjMat = self.adjMat[:, :-1, :, :]
        self.nodes = self.nodes[:, :-1, :, :]
        self.edges = self.edges[:, :-1, :, :]
        self.nodesTar = self.nodesTar[:, :-1, :, :]
        self.edgesTar = self.edgesTar[:, :-1, :, :]

    ###########################################################################
    # display information
    ###########################################################################

    def showDataType(self):

        if(self.isList()):
            adjMatType = "int"
            nodesType = "float"
            edgesType = "float"
            nodesTarType = "float"
            edgesTarType = "float"
        elif(self.isNumpy() or self.isTorch()):
            adjMatType = self.adjMat.dtype
            nodesType = self.nodes.dtype
            edgesType = self.edges.dtype
            nodesTarType = self.nodesTar.dtype
            edgesTarType = self.edgesTar.dtype
        
        print("==============================")
        print("dataset types:")
        print("%s mode" % self.type)
        print("adjacency matrix: %s" % adjMatType)
        print("nodes: %s" % nodesType)
        print("edges: %s" % edgesType)
        print("nodes target: %s" % nodesTarType)
        print("edges target: %s" % edgesTarType)
        print("==============================")

    def showDataLengths(self):

        print("==============================")
        print("dataset sizes:")
        print("adjacency matrix: %s" % str(self.adjMat.shape))
        print("nodes: %s" % str(self.nodes.shape))
        print("edges: %s" % str(self.edges.shape))
        print("nodes target: %s" % str(self.nodesTar.shape))
        print("edges target: %s" % str(self.edgesTar.shape))
        print("==============================")

    ###########################################################################
    # getData
    ###########################################################################

    def getNodes(self, perc=NORM_PERC):

        shape = self.nodes.shape
        newShape = (-1, shape[-1])
        nodes = self.nodes.reshape(newShape)

        end = int(len(nodes) * perc)
        if(perc != 1.0): nodes = nodes[:end,:]

        return nodes

    def getNodesTar(self, perc=NORM_PERC):

        shape = self.nodesTar.shape
        newShape = (-1, shape[-1])
        nodesTar = self.nodesTar.reshape(newShape)

        end = int(len(nodesTar) * perc)
        if(perc != 1.0): nodesTar = nodesTar[:end,:]

        return nodesTar
    
    def getEdges(self, perc=NORM_PERC):

        shape = self.edges.shape
        newShape = (-1, shape[-1])
        edges = self.edges.reshape(newShape)

        end = int(len(edges) * perc)
        if(perc != 1.0): edges = edges[:end,:]

        return edges

    def getEdgesTar(self, perc=NORM_PERC):

        shape = self.edgesTar.shape
        newShape = (-1, shape[-1])
        edgesTar = self.edgesTar.reshape(newShape)

        end = int(len(edgesTar) * perc)
        if(perc != 1.0): edgesTar = edgesTar[:end,:]

        return edgesTar

    def getPairs(self, mode, perc=NORM_PERC):
        
        # check type
        if(self.isList()): self.toNumpy()
        
        if(self.isNumpy()):
            pairFunc = GraphStack.getPairNumpy
            tpFunc = np.swapaxes
            dim = self.adjMat.ndim
        elif(self.isTorch()):
            pairFunc = GraphStack.getPairTorch
            tpFunc = torch.transpose
            dim = self.adjMat.dim()
        
        # check input and swap aliases
        if(mode == 'n'):
            adjMat = self.adjMat
            nodes = self.nodes
            edges = self.edges
        elif(mode == 'e'):
            adjMat = tpFunc(self.adjMat, dim - 2, dim - 1)
            nodes = self.edges
            edges = self.nodes
        
        pairs, _, _ = pairFunc(adjMat, nodes, edges)


        end = int(len(nodes) * perc)
        if(perc != 1.0): 
            pairs = tuple([item[:end,:] for item in pairs])

        return pairs

    @staticmethod
    def getPairNumpy(adjMat, nodes, edges, mode=None):

        dim = adjMat.ndim
        # copmute node-node adjacency
        adjMat_t = np.swapaxes(adjMat, dim - 2, dim - 1)
        nn = np.matmul(adjMat_t, adjMat)
        # set diagonal to 0
        diag = np.einsum('...ii->...i', nn)
        diag *= 0
        # check node-node pairs
        pairs =  np.nonzero(nn)
        mat = list(pairs[:-2])
        n0ID = tuple(mat + [pairs[-2]])
        n1ID = tuple(mat + [pairs[-1]])
        # check shared edge between nodes
        n0Adj, n1Adj = adjMat_t[n0ID], adjMat_t[n1ID]
        eCom = np.nonzero(n0Adj * n1Adj)
        eComID = tuple(mat + [eCom[-1]])

        # get items
        n0 = nodes[n0ID]
        n1 = nodes[n1ID]
        eCom = edges[eComID]

        # get map
        if(mode == 'n' or mode == 'e' or mode == None):
            pairMap = n0ID
        else:
            pairMap = eComID
        
        return (eCom, n0, n1), pairMap

    @staticmethod
    def getPairTorch(adjMat, nodes, edges, mode=None):

        dim = adjMat.dim()
        # copmute node-node adjacency
        adjMat_t = torch.transpose(adjMat, dim - 2, dim - 1)
        nn = torch.matmul(adjMat_t, adjMat)
        # set diagonal to 0
        diag = torch.einsum("...ii->...i", (nn,))
        diag *= 0
        # check node-node pairs
        pairs = tuple(torch.t(torch.nonzero(nn)))
        mat = list(pairs[:-2])
        n0ID = tuple(mat + [pairs[-2]])
        n1ID = tuple(mat + [pairs[-1]])
        # check shared edge between nodes
        n0Adj, n1Adj = adjMat_t[n0ID], adjMat_t[n1ID]
        eCom = tuple(torch.t(torch.nonzero(n0Adj * n1Adj)))
        eComID = tuple(mat + [eCom[-1]])

        # get items
        n0 = nodes[n0ID]
        n1 = nodes[n1ID]
        eCom = edges[eComID]

        # get map
        if(mode == 'n' or mode == 'e' or mode == None):
            pairMap = n0ID
        else:
            pairMap = eComID

        # get adjacency type
        n0ModeID = tuple(list(eComID) + [pairs[-2]])
        n1ModeID = tuple(list(eComID) + [pairs[-1]])
        n0Mode = adjMat[n0ModeID]
        n1Mode = adjMat[n1ModeID]

        return (eCom, n0, n1), pairMap, (n0Mode, n1Mode)

    ###########################################################################
    # save/load
    ###########################################################################

    def save(self, folder):
    
        # save datastructure information
        info = {}
        info[NODE_NUMTYPE_TOKEN] = self.nodeNumType
        info[EDGE_NUMTYPE_TOKEN] = self.edgeNumType
        info[NODE_CEN_TOKEN] = self.nodeCenter
        info[EDGE_CEN_TOKEN] = self.edgeCenter
        writeJSON(folder + '/' + INFO_TOKEN + JSON_SUFFIX, info)

        # save matrices
        np.save(folder + '/' + ADJ_MATRIX_TOKEN, self.adjMat)
        np.save(folder + '/' + NODES_TOKEN, self.nodes)
        np.save(folder + '/' + EDGES_TOKEN, self.edges)
        np.save(folder + '/' + NODES_TAR_TOKEN, self.nodesTar)
        np.save(folder + '/' + EDGES_TAR_TOKEN, self.edgesTar)

        # save cumsum if applicable
        if(self.cumSumInit):
            np.save(folder + '/' + EDGES_TOKEN + CUMSUM_TOKEN, self.edgesCumSumDelta)
            np.save(folder + '/' + NODES_TOKEN + CUMSUM_TOKEN, self.nodesCumSumDelta)

    def load(self, path):
        
        # load datastructure information
        info = readJSON(path + '/' + INFO_TOKEN + JSON_SUFFIX)
        self.nodeNumType = info[NODE_NUMTYPE_TOKEN]
        self.edgeNumType = info[EDGE_NUMTYPE_TOKEN]
        self.nodeCenter = info[NODE_CEN_TOKEN]
        self.edgeCenter = info[EDGE_CEN_TOKEN]

        # get cumsum if possible
        if(CUMSUM_TOKEN in info):
            self.cumSumInit = info[CUMSUM_TOKEN]
        else:
            self.cumSumInit = False

        # load matrices
        self.adjMat = np.load(path + '/' + ADJ_MATRIX_TOKEN + NUMPY_SUFFIX)
        self.nodes = np.load(path + '/' + NODES_TOKEN + NUMPY_SUFFIX)
        self.edges = np.load(path + '/' + EDGES_TOKEN + NUMPY_SUFFIX)
        self.nodesTar = np.load(path + '/' + NODES_TAR_TOKEN + NUMPY_SUFFIX)
        self.edgesTar = np.load(path + '/' + EDGES_TAR_TOKEN + NUMPY_SUFFIX)

        if(self.cumSumInit):
            self.edgesCumSumDelta = np.load(path + '/' + EDGES_TOKEN + CUMSUM_TOKEN + NUMPY_SUFFIX)
            self.nodesCumSumDelta = np.load(path + '/' + NODES_TOKEN + CUMSUM_TOKEN + NUMPY_SUFFIX)
        
        self.trainRatio = TRAIN_RATIO
        self.count = len(self.adjMat)
        self.type = "numpy"

        return self

    def saveJSON(self, path):
        
        self.toSingleGraph()
        count = self.adjMat.shape[0]
        graphs = []
        printed = False

        if(PARALLEL):
            done = 0
            last10 = 0
            with mp.Pool(processes=WORKERS) as pool:
                for graph in pool.imap_unordered(self.makeJSON, range(count)):
                    graphs += [graph]
                    done += 1
                    perc = int(done / count * 100) // 10
                    if(last10 != perc):
                        print("%d0" % perc + chr(37)) # % character
                        last10 = perc
        else:
            for i in range(count):
                graphs += [makeJSON(i)]
        
        graphs[0][NODE_CEN_TOKEN] = self.nodeCenter
        graphs[0][EDGE_CEN_TOKEN] = self.edgeCenter
        graphs[0][NODE_NUMTYPE_TOKEN] = self.nodeNumType
        graphs[0][EDGE_NUMTYPE_TOKEN] = self.edgeNumType

        writeJSON(path, graphs)

    def makeJSON(self, index):

        new ={}
        new[ADJ_MATRIX_TOKEN] = self.adjMat[index].tolist()
        new[NODES_TOKEN] = self.nodes[index].tolist()
        new[EDGES_TOKEN] = self.edges[index].tolist()
        new[NODES_TAR_TOKEN] = self.nodesTar[index].tolist()
        new[EDGES_TAR_TOKEN] = self.edgesTar[index].tolist()
        
        return new

class SimpleStack(object):
    def __init__(self, adjMat=None, nodes=None, edges=None, nodesTar=None, edgesTar=None):
        
        self.adjMat = adjMat
        self.nodes = nodes
        self.edges = edges
        self.nodesTar = nodesTar
        self.edgesTar = edgesTar

        self.adjMatOut = None
        self.nodesOut = None
        self.edgesOut = None

        self.adjMatSeq = []
        self.nodesSeq = []
        self.edgesSeq = []

        self.count = None
        
        inpMat = isinstance(adjMat, np.ndarray) or \
                 isinstance(adjMat, torch.Tensor)
        if(inpMat and len(adjMat.shape) < 3): self.count = 1
        elif(inpMat): self.count = adjMat.shape[0]
        else: self.count = None

    def getGraph(self, index):

        adjMat, nodes, edges = None, None, None

        if(self.adjMat and self.nodes and self.edges):
            adjMat = self.adjMat[index]
            nodes = self.nodes[index]
            edges = self.edges[index]
        
        return adjMat, nodes, edges

    def getTarget(self, index):

        nodesTar, edgesTar = None, None
        
        if(self.nodesTar and self.edgesTar):
            nodesTar = self.nodesTar[index]
            edgesTar = self.edgesTar[index]

        return nodesTar, edgesTar

    def addDelta(self, norm):

        adjMat = self.adjMatOut.detach()
        nodes = self.nodes.detach() + norm.invertNodeOut(self.nodesOut.detach())
        edges = self.edges.detach() + norm.invertEdgeOut(self.edgesOut.detach())
        
        self.adjMat = adjMat
        self.nodes = nodes
        self.edges = edges
        
        del self.adjMatOut, self.nodesOut, self.edgesOut
        self.adjMatOut = None
        self.nodesOut = None
        self.edgesOut = None

    def fixAdj(self):

        analizer = MatchMaker()
        refNodes = self.nodes
        refEdges = self.edges
        newAdjMat = analizer.matchAnalysis(self.adjMat, refNodes, refEdges)
        self.adjMat = newAdjMat

    def dumpSeq(self):

        self.adjMatSeq += [self.adjMat]
        self.nodesSeq += [self.nodes]
        self.edgesSeq += [self.edges]

    def setSizes(self, adjSize, nodesSize, edgesSize, mode="torch"):


        if(mode == "torch"): initFunc = torch.zeros
        elif(mode == "numpy"): initFunc = np.zeros
        
        self.adjMat = initFunc(adjSize)
        self.nodes = initFunc(nodesSize)
        self.edges = initFunc(edgesSize)

    def loadJSONinit(self, arg, mode="torch", batchFlag=False):
        
        if(not batchFlag): # read and initializefrom file
            json = readJSON(arg)
            if(isinstance(json, list) and len(json) > 0):
                init = json[0]
            else:
                init = json
        else: init = arg # initialize from arg

        return self.initFromData(init)

    def initFromData(self, init, mode="torch"):

        if(mode == "torch"): initFunc = torch.tensor
        elif(mode == "numpy"): initFunc = np.asarray
        
        self.adjMat = initFunc(init[ADJ_MATRIX_TOKEN])
        self.nodes = initFunc(init[NODES_TOKEN])
        self.edges = initFunc(init[EDGES_TOKEN])

        return self

    def saveJSONSeq(self, path, lastOnly=False):
        
        # batch mode: [step, (graph, node, feature)]
        # single mode: [step, (nodes, feature)]
        assert(len(self.adjMatSeq) == len(self.nodesSeq) == len(self.edgesSeq))

        graphs = []
        
        if(lastOnly): steps = [0, -1]
        else: steps = range(len(self.adjMatSeq))

        for i in steps:
            graph = {}
            graph[ADJ_MATRIX_TOKEN] = self.adjMatSeq[i].tolist()
            graph[NODES_TOKEN] = self.nodesSeq[i].tolist()
            graph[EDGES_TOKEN] = self.edgesSeq[i].tolist()
            graphs += [graph]

        writeJSON(path, graphs)

    def copy(self):

        if(isinstance(self.adjMat, torch.Tensor)):

            adjMat, nodes, edges = None, None, None
            nodesTar, edgesTar = None, None

            if(isinstance(self.adjMat, torch.Tensor)): 
                adjMat = self.adjMat.clone()
            if(isinstance(self.nodes, torch.Tensor)): 
                nodes = self.nodes.clone()
            if(isinstance(self.edges, torch.Tensor)): 
                edges = self.edges.clone()
            if(isinstance(self.nodesTar, torch.Tensor)): 
                nodesTar = self.nodesTar.clone()
            if(isinstance(self.edgesTar, torch.Tensor)): 
                edgesTar = self.edgesTar.clone()

        new = SimpleStack(adjMat, nodes, edges, nodesTar, edgesTar)

        return new

    def toType(self, tarType, device=DEVICE):
        
        if(tarType == "torch"):
            if(device == DEVICE_GPU): adjMatType = TORCH_DTYPE
            else: adjMatType = TORCH_DTYPE_BIN
        if(tarType == "numpy"):
            adjMatType = NUMPY_DTYPE_BIN
        self.adjMat = changeType(self.adjMat, tarType, device, dType=adjMatType)
        self.nodes = changeType(self.nodes, tarType, device)
        self.edges = changeType(self.edges, tarType, device)
    
        self.type = tarType
    
    def to(self, device):

        if(isinstance(self.adjMat, torch.Tensor)): 
            self.adjMat = self.adjMat.to(device)
        if(isinstance(self.nodes, torch.Tensor)): 
            self.nodes = self.nodes.to(device)
        if(isinstance(self.edges, torch.Tensor)): 
            self.edges = self.edges.to(device)
        # don't need these during output
        """
        if(isinstance(self.nodesTar, torch.Tensor)): 
            self.nodesTar = self.nodesTar.to(device)
        if(isinstance(self.edgesTar, torch.Tensor)): 
            self.edgesTar = self.edgesTar.to(device)
        """
        
        return self

class Transformer(object):
    def __init__(self, angle=0):
        # initialize 0 translation and rotation
        self.trans = (0., 0., 0.)
        self.rot = (0., 0., 0.) # in radian
        # matrices
        self.transMatrix = None
        self.rotZMatrix = None
        self.stressRotZMatrix = None
        # initialize transformation matrix
        self.setTrans((0., 0., 0.))
        # using only the rotation matrix about z
        self.setRotZ(0.)

        if(angle != 0):
            self.setRotZ(angle)

    def setTrans(self, vec):
        # set a translation
        self.trans = (vec[0], vec[1], vec[2])
        self.transMatrix = \
            np.array([[1., 0., 0., self.trans[0]],
                      [0., 1., 0., self.trans[1]],
                      [0., 0., 1., self.trans[2]],
                      [0., 0., 0.,            1.]])
        
    def setRotZ(self, rad):
        # set a rotation about the z axis
        self.rot = (self.rot[0], self.rot[1], rad)
        self.rotZMatrix = \
            np.array([[m.cos(self.rot[2]), -1 * m.sin(self.rot[2]), 0., 0.],
                      [m.sin(self.rot[2]),      m.cos(self.rot[2]), 0., 0.],
                      [                0.,                      0., 1., 0.],
                      [                0.,                      0., 0., 1.]])
        self.stressRotZMatrix = \
            np.array([[m.cos(-self.rot[2]), -1 * m.sin(-self.rot[2]), 0.],
                      [m.sin(-self.rot[2]),      m.cos(-self.rot[2]), 0.],
                      [                0.,                      0., 1.]])
    
    def appTransGeo(self, vec):
        # apply the translation matrix to the input vector
        temp = np.array([vec[0], vec[1], vec[2], 1.])
        temp = np.matmul(self.transMatrix, temp)
        temp = [temp[0] / temp[3],
                temp[1] / temp[3],
                temp[2] / temp[3]]
        
        return temp

    def appRotZGeo(self, vec):
        # apply a rotation about the z axis
        temp = np.array([vec[0], vec[1], vec[2], 1.])
        temp = np.matmul(self.rotZMatrix, temp)
        temp = [temp[0] / temp[3],
                temp[1] / temp[3],
                temp[2] / temp[3]]
        
        return temp

    def appRotZMat(self, vec):
        initStress = np.array([[vec[0], vec[3], vec[4]],
                               [     0, vec[1], vec[5]],
                               [     0,      0, vec[2]]])
        temp = np.matmul(self.stressRotZMatrix.transpose(), initStress)
        temp = np.matmul(temp, self.stressRotZMatrix)

        return [temp[0][0], temp[1][1], temp[2][2], 
                temp[0][1], temp[0][2], temp[1][2]]

    def transform(self, matrix, numType):
        matrixNew = []
        for row in matrix:
            index, temp = 0, []
            while (index < len(row)):
                # rotate geometry
                nType = numType[index]
                if(POINT_TOKEN in nType or VECTOR_TOKEN in nType):
                    temp += self.appRotZGeo(row[index: index + COOR_LEN])
                    index += COOR_LEN
                # rotate stress
                elif(STRESS_TOKEN in nType):
                    temp += self.appRotZMat(row[index: index + STRESS_LEN])
                    index += STRESS_LEN
                # other types, skip and do nothing
                else:
                    temp += [row[index]]
                    index += 1
            matrixNew += [temp]

        return matrixNew

class MatchMaker(object):
    counter = 0
    _edgeFaces = [[ 2,  4,  1,  3],
                  [ 9, 11, 10, 12]]
    _nodeFaces = [[ 1,  5,  3,  7],
                  [ 2,  6,  1,  5],
                  [ 4,  8,  2,  6],
                  [ 3,  7,  4,  8]]
    _edgeFaceIndices = "None"
    _nodeFaceIndices = "None"

    def __init__(self):

        if(isinstance(self._edgeFaceIndices, str)):
            self._getIndices()

    def _getIndices(self):

        shift = np.arange(COOR_LEN)

        # edge
        # construct matrix
        faces = np.asarray(MatchMaker._edgeFaces) * COOR_LEN
        faces = faces[:,:,None]
        MatchMaker._edgeFaceIndices = np.repeat(faces, COOR_LEN, axis=-1) + shift
        # flip face orientation for matching convenience
        left = np.take(MatchMaker._edgeFaceIndices, (0, 1), axis=-2)
        right = np.take(MatchMaker._edgeFaceIndices, (2, 3), axis=-2)
        ori = np.concatenate((left, right), axis=-2)
        rev = np.concatenate((right, left), axis=-2)
        MatchMaker._edgeFaceIndices = np.concatenate((rev, ori), axis=0)

        # node
        # construct matrix
        faces = np.asarray(MatchMaker._nodeFaces) * COOR_LEN
        faces = faces[:,:,None]
        MatchMaker._nodeFaceIndices = np.repeat(faces, COOR_LEN, axis=-1) + shift
        
    def _getNodeFaces(self, nodes):

        faces = np.take(nodes, self._nodeFaceIndices, axis=-1)

        return faces

    def _getEdgeFaces(self, edges):

        faces = np.take(edges, self._edgeFaceIndices, axis=-1)

        return faces

    def _matchFaces(self, nodeFaces, edgeFaces):

        # counter index starts from 1 because 0 is "disconnected" in the 
        # adjacency matrix
        matchMode, counter = None, 1
        # loop through each possible connection
        for i in range(len(nodeFaces)):
            nodeFace = nodeFaces[i]

            for j in range(len(edgeFaces)):
                edgeFace = edgeFaces[j]
                # check face difference
                diff = nodeFace - edgeFace
                dist = np.sqrt(np.sum(np.square(diff), axis=-1))
                dist = np.sum(dist) # 1D distance

                if(dist < EPSILON): # update if found identical
                    matchMode = counter
                    break
                counter += 1
            
            if(matchMode != None): # early termination
                break
        
        # if no match was found, then return -1
        if(matchMode == None):
            matchMode = -1

        return matchMode
    
    def matchAnalysis(self, adjMat, nodes, edges):

        pairsIndices = np.nonzero(adjMat)
        
        pairEdges = edges[pairsIndices[0]]
        pairNodes = nodes[pairsIndices[1]]

        match = self.match(pairNodes, pairEdges)

        newAdjMat = np.copy(adjMat)
        newAdjMat[pairsIndices] = match

        assert(not np.any(newAdjMat == -1))

        return newAdjMat

    def match(self, nodes, edges):

        count = len(nodes)
        nodeFaces = self._getNodeFaces(nodes)
        edgeFaces = self._getEdgeFaces(edges)

        # compare
        match = []
        for i in range(count):
            nodeF = nodeFaces[i]
            edgeF = edgeFaces[i]
            
            match += [self._matchFaces(nodeF, edgeF)]

        return match

    def getPointPairs(self, adjMat, nodes, edges):

        self._checkType(adjMat)
        # get adjacency pairs
        adj = self._getAdjPairs(adjMat, nodes, edges)
        pairNodes, pairEdges, adjMode = adj

        # get index maps based on adjcency mode
        nodeIndices, edgeIndices = self._getFaceIndices(adjMode)
        # get the preceeding row index
        count = len(adjMode)
        if(isinstance(adjMat, np.ndarray)): row = np.arange(count)
        elif(isinstance(adjMat, torch.Tensor)): row = torch.arange(count)
        
        # retrieve points
        nodeOldShape = nodeIndices.shape
        edgeOldShape = edgeIndices.shape

        nodeIndices = nodeIndices.reshape((count, -1))
        edgeIndices = edgeIndices.reshape((count, -1))
        row = row.reshape((count, -1))
        nodePoints = pairNodes[row,nodeIndices].reshape(nodeOldShape)
        edgePoints = pairEdges[row,edgeIndices].reshape(edgeOldShape)
        
        return nodePoints, edgePoints
    
    def _checkType(self, ref):

        refTorch = isinstance(ref, torch.Tensor)
        refNumpy = isinstance(ref, np.ndarray)
        selfTorch = isinstance(MatchMaker._edgeFaceIndices, torch.Tensor)
        selfNumpy = isinstance(MatchMaker._edgeFaceIndices, np.ndarray)

        if(refTorch and selfNumpy):
            device = ref.device
            MatchMaker._edgeFaceIndices = torch.from_numpy(MatchMaker._edgeFaceIndices).to(device)
            MatchMaker._nodeFaceIndices = torch.from_numpy(MatchMaker._nodeFaceIndices).to(device)
        elif(refNumpy and selfTorch):
            MatchMaker._edgeFaceIndices = MatchMaker._edgeFaceIndices.numpy()
            MatchMaker._nodeFaceIndices = MatchMaker._nodeFaceIndices.numpy()

    def _getAdjPairs(self, adjMat, nodes, edges):

        # use different nonzero funciton based on input type
        if(isinstance(adjMat, np.ndarray)):
            pairIndices = np.nonzero(adjMat)
        elif(isinstance(adjMat, torch.Tensor)):
            pairIndices = tuple(torch.t(torch.nonzero(adjMat)))

        # adjMode is 1-indexed to avoid using 0(nonadj), therefore we convert
        # it to 0 index for later uses' convenience
        adjMode = adjMat[pairIndices] - 1

        # decompose indices
        indexPrefix = pairIndices[:-2]
        indexEdge = pairIndices[-2]
        indexNode = pairIndices[-1]
        # assemble indices
        pairEdgeIndex = tuple(list(indexPrefix) + [indexEdge])
        pairNodeIndex = tuple(list(indexPrefix) + [indexNode])
        # get items
        pairEdges = edges[pairEdgeIndex]
        pairNodes = nodes[pairNodeIndex]

        return pairNodes, pairEdges, adjMode

    def _getFaceIndices(self, adjMode):

        nodeFaceIndex = adjMode // len(self._edgeFaceIndices)
        edgeFaceIndex = adjMode % len(self._edgeFaceIndices)

        if(isinstance(adjMode, torch.Tensor)):
            edgeFaceIndex = edgeFaceIndex.type(torch.int)
            nodeFaceIndex = nodeFaceIndex.type(torch.int)
        elif(isinstance(adjMode, np.ndarray)):
            edgeFaceIndex = edgeFaceIndex.astype(np.int)
            nodeFaceIndex = nodeFaceIndex.astype(np.int)

        edgeFaceIndex = MatchMaker._edgeFaceIndices[edgeFaceIndex.long()].long()
        nodeFaceIndex = MatchMaker._nodeFaceIndices[nodeFaceIndex.long()].long()

        return nodeFaceIndex, edgeFaceIndex

    def calDist(self, adjMat, nodes, edges):

        nodePts, edgePts = self.getPointPairs(adjMat, nodes, edges)

        diffVec = nodePts - edgePts
        diffSq = diffVec * diffVec
        diffSqSum = torch.sum(diffSq, -1)
        dist = torch.sqrt(diffSqSum)

        return dist

class NoiseAgent(object):
    def __init__(self):
        """
        self.nodeSD = None
        self.edgeSD = None

        if(isinstance(nodes, np.ndarray)):
            if(NOISE_METHOD == "SD"):
                self.nodeSD = np.std(nodes, 0)
                self.edgeSD = np.std(edges, 0)
            elif(NOISE_METHOD == "MAX"):
                self.nodeSD = np.max(nodes, 0)
                self.edgeSD = np.max(edges, 0)
            elif(NOISE_METHOD == "PERC"):
                self.nodeSD = np.percentile(np.abs(nodes), 95, axis=0)
                self.edgeSD = np.percentile(np.abs(edges), 95, axis=0)
        elif(isinstance(nodes, torch.Tensor)):
            if(NOISE_METHOD == "SD"):
                self.nodeSD = torch.std(nodes, 0)
                self.edgeSD = torch.std(edges, 0)
            elif(NOISE_METHOD == "MAX"):
                self.nodeSD = torch.max(nodes, 0)[0]
                self.edgeSD = torch.max(edges, 0)[0]
            elif(NOISE_METHOD == "PERC"):
                self.nodeSD = np.percentile(torch.abs(nodes).numpy(), 95, axis=0)
                self.edgeSD = np.percentile(torch.abs(edges).numpy(), 95, axis=0)
                self.nodeSD = torch.from_numpy(self.nodeSD).float()
                self.edgeSD = torch.from_numpy(self.edgeSD).float()
        """

    def genNoise(self, nodesScale, edgesScale, noise=NOISE):

        #self._checkType(nodesScale)

        nodeShape = nodesScale.shape
        edgeShape = edgesScale.shape


        if(isinstance(nodesScale, np.ndarray)):
            nodeNoise = np.random.normal(np.zeros(nodeShape), noise)
            edgeNoise = np.random.normal(np.zeros(edgeShape), noise)
        
        elif(isinstance(nodesScale, torch.Tensor)):
            device = nodesScale.device
            nodeNoise = torch.normal(mean=torch.zeros(nodeShape), std=noise)
            nodeNoise = nodeNoise.to(device)
            edgeNoise = torch.normal(mean=torch.zeros(edgeShape), std=noise)
            edgeNoise = edgeNoise.to(device)

        nodeNoise *= nodesScale
        edgeNoise *= edgesScale

        return nodeNoise, edgeNoise

    def _checkType(self, other):

        selfNumpy = isinstance(self.nodeSD, np.ndarray)
        selfTorch = isinstance(self.nodeSD, torch.Tensor)
        otherNumpy = isinstance(other, np.ndarray)
        otherTorch = isinstance(other, torch.Tensor)

        if(selfNumpy and otherTorch):
            self.nodeSD = torch.from_numpy(self.nodeSD)
            self.edgeSD = torch.from_numpy(self.edgeSD)
        elif(selfTorch and otherNumpy):
            self.nodeSD = self.nodeSD.numpy()
            self.edgeSD = self.edgeSD.numpy()

    def noiseLevel(self, delta, deltaTar, scale=False, normFunc=False):

        # invert delta if need be
        if(normFunc):
            deltaOri = normFunc(delta)
        else:
            deltaOri = delta

        diff = deltaTar - deltaOri

        scalePos = scale.clone()
        scalePos[scalePos == 0] = 1

        if(isinstance(scale, torch.Tensor)): # scale
            noiseScale = torch.abs(diff) / scalePos
        
        avgScale = torch.mean(noiseScale)
        #avgScale = torch.mean(noiseScale, 0)

        return avgScale

def changeType(val, newType, device=DEVICE, dType=None):
    
    # check original type
    if(isinstance(val, torch.Tensor)):
        oldType = "torch"
    elif(isinstance(val, np.ndarray)):
        oldType = "numpy"
    elif(isinstance(val, list)):
        oldType = "list"
    else:
        return val
    
    # convert type
    if(newType == "numpy"):
        if(oldType == "torch"):
            new = val.numpy()
        elif(oldType == "list"):
            new = np.asarray(val)
        elif(oldType == "numpy"):
            new = val
        # change type if need be
        if(dType): new = new.astype(dType)
        else: new = new.astype(NUMPY_DTYPE)
    
    elif(newType == "torch"):
        if(oldType == "torch"):
            new = val.to(device)
        elif(oldType == "list"):
            new = torch.Tensor(val).to(device)
        elif(oldType == "numpy"):
            new = torch.from_numpy(val).to(device)
        # chage type if need be
        if(dType): new = new.type(dType)
        else: new = new.type(TORCH_DTYPE)
    
    elif(newType == "list"):
        if(oldType == "torch"):
            new = val.tolist()
        elif(oldType == "list"):
            new = val
        elif(oldType == "numpy"):
            new = val.tolist()

    return new

def sliceMD(matrix, start, end):
    # slice at the last dimension of a matrix
    
    dim = len(matrix.shape)

    if(start == None or end == None): return None
    if(start == end): return None

    if(dim == 1):
        sliced = matrix[start:end]
    elif(dim == 2):
        sliced = matrix[:, start:end]
    elif(dim == 3):
        sliced = matrix[:, :, start:end]
    elif(dim == 4):
        sliced = matrix[:, :, :, start:end]
    elif(dim == 5):
        sliced = matrix[:, :, :, :, start:end]

    return sliced

def readFile(path):

    with open(path, "rt") as f:
        content = f.read()
    
    return content

def readJSON(path):

    with open(path) as f:
        data = json.load(f)
    return data

def writeJSON(path, data):

    with open(path, 'w') as output:
            json.dump(data, output)

def writeTxt(path, data):

    with open(path, "wt") as f:
        f.write(data)

def readTxt(path):

    with open(path, "rt") as f:
        data = f.read()
    
    return data

def test():
    dataset = GraphStack().load(DATA_FOLDER)
    dataset.showDataLengths()

if __name__ == "__main__":
    test()