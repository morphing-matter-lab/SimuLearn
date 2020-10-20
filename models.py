import math
import torch
import torch.nn as nn

from parameters import *
from dataset import GraphStack, SimpleStack, MatchMaker

def getMLP(inLen, outLen, width, override=None):
    # generates an MLP model
    # parameters:
    #   inLen: int, the length of the input vector
    #   outLen: int, the length of the output vector
    # return:
    #   nn model, the MLP
    if(not override):
        lastLenLast = 2 ** math.ceil(math.log2(outLen)) * width
        lastLenFirst = 2 ** math.ceil(math.log2(inLen)) * width
        lastLen = int(max(lastLenFirst, lastLenLast))
    else:
        lastLen = override
    # print("model %d-%d-...-%d-%d" % (inLen, lastLen * 8, lastLen, outLen))
    return nn.Sequential(nn.Linear(inLen, lastLen * 8),
                         nn.ReLU(),
                         nn.Linear(lastLen * 8, lastLen * 4),
                         nn.ReLU(),
                         nn.Linear(lastLen * 4, lastLen * 2),
                         nn.ReLU(),
                         nn.Linear(lastLen * 2, lastLen),
                         nn.ReLU(),
                         nn.Linear(lastLen, outLen),).to(DEVICE)

class GNBlock(torch.nn.Module):
    def __init__(self, norm, width=1, scale=1, overrides=None, 
                 nodeIntVecLen=NODE_INT_VEC_LEN, 
                 edgeIntVecLen=EDGE_INT_VEC_LEN):

        super(GNBlock, self).__init__()

        # nodes
        self.nodeInt = None
        self.nodeUpdate = None
        # edges
        self.edgeInt = None
        self.nodeUpdate = None

        if(overrides == None or len(overrides) != 4): overrides = []
        self.getModels(norm, width, scale, overrides, 
                       nodeIntVecLen=nodeIntVecLen,
                       edgeIntVecLen=edgeIntVecLen)

    def getModels(self, norm, width=1, scale=1, overrides=None, 
                  nodeIntVecLen=NODE_INT_VEC_LEN, 
                  edgeIntVecLen=EDGE_INT_VEC_LEN):

        # get sizes
        latScale = scale - 1
        nodeLen = norm.nodeNormedLen + norm.nodeLen * latScale
        edgeLen = norm.edgeNormedLen + norm.edgeLen * latScale
        nodeIntBase = norm.nodeLen * 2 + norm.edgeLen
        edgeIntBase = norm.edgeLen * 2 + norm.nodeLen
        
        nodeIntLen = norm.nodeIntNormedLen + nodeIntBase * latScale
        edgeIntLen = norm.edgeIntNormedLen + edgeIntBase * latScale

        if(PROP_DISLOCATION):
            nodeIntLen += 16 * 2
            edgeIntLen += 16 * 2
        # nodes
        self.nodeInt = getMLP(nodeIntLen, nodeIntVecLen,
                              width, override=overrides[1]) # 0
        self.nodeUpdate = getMLP(nodeLen + nodeIntVecLen, norm.nodeLen,
                              width, override=overrides[3]) # 1
        # edges
        self.edgeInt = getMLP(edgeIntLen, edgeIntVecLen,
                              width, override=overrides[0]) # 2
        self.edgeUpdate = getMLP(edgeLen + edgeIntVecLen, norm.edgeLen,
                              width, override=overrides[2]) # 3
        
        self.initWeights()

    def initWeights(self):
        i = 0
        for module in self.modules():
            if(isinstance(module, nn.Linear)):
                i += 1
                module.weight.data.normal_(0., .02)
                module.bias.data.fill_(0.)

    def forward(self, adjMat, nodes, edges, norm):

        # get interaction pairs
        nodeIntPairs, nodeMap, nodeIntMode = self.getIntPairs(adjMat, nodes, 
                                                              edges, 'n')
        edgeIntPairs, edgeMap, edgeIntMode = self.getIntPairs(adjMat, nodes,
                                                              edges, 'e')
        # normalize
        nodeIntNormed = norm.normNodeInt(nodeIntPairs)
        edgeIntNormed = norm.normEdgeInt(edgeIntPairs)
        if(PROP_DISLOCATION):
            nodeIntModeCat = self.getModeVec(nodeIntMode)
            edgeIntModeCat = self.getModeVec(edgeIntMode)
            # assemble
            nodeIntNormed = tuple(list(nodeIntNormed) + [nodeIntModeCat])
            edgeIntNormed = tuple(list(edgeIntNormed) + [edgeIntModeCat])
        # assemble
        nodeIntCat = torch.cat(nodeIntNormed, -1)
        edgeIntCat = torch.cat(edgeIntNormed, -1)
        # compute interacction vectors
        nodeIntVec = self.nodeInt(nodeIntCat)
        edgeIntVec = self.edgeInt(edgeIntCat)
        # sum interaction vectors
        nodeIntLat = self.convolveIntVec(nodes, nodeIntVec, nodeMap)
        edgeIntLat = self.convolveIntVec(edges, edgeIntVec, edgeMap)

        # concatenate interaction vector with element vector
        nodeNormed = norm.normNode(nodes)
        edgeNormed = norm.normEdge(edges)
        nodeUpdateCat = torch.cat((nodeNormed, nodeIntLat), -1)
        edgeUpdateCat = torch.cat((edgeNormed, edgeIntLat), -1)
        # compute output
        nodesOut = self.nodeUpdate(nodeUpdateCat)
        edgesOut = self.edgeUpdate(edgeUpdateCat)
        
        return nodesOut, edgesOut

    def getIntPairs(self, adjMat, nodes, edges, mode):

        if(mode[0] == 'n'):
            a = adjMat
            n = nodes
            e = edges
        elif(mode[0] == 'e'):
            dim = len(adjMat.shape)
            a = torch.transpose(adjMat, dim - 2, dim - 1)
            n = edges
            e = nodes
        
        pairs, pairMap, pairMode = GraphStack.getPairTorch(a, n, e, mode)

        return pairs, pairMap, pairMode

    def convolveIntVec(self, tar, vecs, intMap):

        # get new shape
        newShape = list(tar.shape)
        newShape[-1] = list(vecs.shape)[-1]
        newShape = tuple(newShape)

        container = torch.zeros(newShape).to(DEVICE)

        container.index_put_(intMap, vecs, accumulate=True)

        return container

    def getModeVec(self, mode):

        mode0, mode1 = mode

        mode0Vec = torch.zeros((len(mode0), 16))
        mode1Vec = torch.zeros((len(mode1), 16))

        row = torch.arange(len(mode0), dtype=torch.long).reshape((-1, 1))
        mode0Col = mode0.long().reshape((-1, 1))
        mode1Col = mode1.long().reshape((-1, 1))

        mode0Vec[row, mode0Col] = 1
        mode1Vec[row, mode1Col] = 1

        modeVec = torch.cat((mode0Vec, mode1Vec), -1).to(DEVICE)
        
        return modeVec

class DoubleGN(torch.nn.Module):
    def __init__(self, norm, width=1, modelSize=None, 
                 nodeIntVecLen=NODE_INT_VEC_LEN, 
                 edgeIntVecLen=EDGE_INT_VEC_LEN):

        super(DoubleGN, self).__init__()
        
        self.GN1 = None
        if(DOUBLE_GN): self.GN2 = None

        self.getModels(norm, width, modelSize, 
                       nodeIntVecLen=nodeIntVecLen, 
                       edgeIntVecLen=edgeIntVecLen)

    def getModels(self, norm, width=1, modelSize=None, 
                  nodeIntVecLen=NODE_INT_VEC_LEN, 
                  edgeIntVecLen=EDGE_INT_VEC_LEN):

        if(modelSize):
            size1 = modelSize
            size2 = [modelSize[i] * 2 for i in range(len(modelSize))]
        else:
            size1 = [512, 512, 512, 256]
            size2 = [1024, 1024, 1024, 512]

        # GNs
        self.GN1 = GNBlock(norm, width, 1, size1, 
                           nodeIntVecLen=nodeIntVecLen, 
                           edgeIntVecLen=edgeIntVecLen)
        if(DOUBLE_GN): self.GN2 = GNBlock(norm, width, 2, size2, 
                                          nodeIntVecLen=nodeIntVecLen, 
                                          edgeIntVecLen=edgeIntVecLen)

    def showSizes(self):
        
        print(self)

    def forward(self, graphs, env=False, norm=False):
        
        adjMat = graphs.adjMat
        nodes = graphs.nodes
        edges = graphs.edges
        
        if(not norm): norm = env.norm
        else: norm = norm
        
        nodesLat, edgesLat = self.GN1.forward(adjMat, nodes, edges,
                                              norm)
        if(not DOUBLE_GN):
            graphs.adjMatOut = adjMat.clone().detach()
            graphs.nodesOut = nodesLat
            graphs.edgesOut = edgesLat

        else:
            # double mode features
            nodesCat = torch.cat((nodes, nodesLat), -1)
            edgesCat = torch.cat((edges, edgesLat), -1)
            nodesOut, edgesOut = self.GN2.forward(adjMat, nodesCat, edgesCat,
                                                  norm)
        
            graphs.adjMatOut = adjMat.clone().detach()
            graphs.nodesOut = nodesOut
            graphs.edgesOut = edgesOut
        
        return graphs.adjMatOut, graphs.nodesOut, graphs.edgesOut

    def calLoss(self, graphs, env):

        # unpack graphs
        nodesOut, edgesOut = graphs.nodesOut, graphs.edgesOut
        nodesTar, edgesTar = graphs.nodesTar, graphs.edgesTar

        # compute loss for node and edge delta
        output = torch.cat((nodesOut.flatten(), edgesOut.flatten()), 0)
        target = torch.cat((nodesTar.flatten(), edgesTar.flatten()), 0)

        output *= TARGET_GRAD_SCALE
        target *= TARGET_GRAD_SCALE

        env.optimizer.zero_grad()
        loss = env.lossFunc(output, target)

        env.loss = loss
        
        env.lossVal = loss.item()
        env.numCount = len(output)
        env.graphCount = graphs.count
        
        # compute dislocation loss

        if(CAL_DISLOCATION):
            adjMat = graphs.adjMatOut
            nodesNew = graphs.nodes + env.norm.invertNodeOut(graphs.nodesOut)
            edgesNew = graphs.edges + env.norm.invertEdgeOut(graphs.edgesOut)

            ptDist = env.mm.calDist(adjMat, nodesNew, edgesNew)
            ptDistOut = ptDist.flatten() * DISLOC_GRAD_SCALE
            ptDistTar = torch.zeros(ptDistOut.shape).to(DEVICE)

            ptLoss = env.lossFunc(ptDistOut, ptDistTar)

            env.ptLoss = ptLoss

            env.ptDist = torch.sum(ptDist).item()
            env.ptCount = len(ptDistOut)

    def update(self, env):
        
        env.loss.backward(retain_graph=PROP_DISLOCATION)
        
        if(PROP_DISLOCATION):
            env.ptLoss.backward()
            
        env.optimizer.step()

    def save(self, path):

        torch.save(self.state_dict(), path)

    def load(self, path):

        stateDict = torch.load(path, map_location=DEVICE)
        self.load_state_dict(stateDict)