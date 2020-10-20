import numpy as np
import copy
import json
import time as t
import torch
import torch.tensor

from parameters import *
from dataset import *

class Normalizer(object):
    def __init__(self, func, dataType):

        self.func = func
        self.type = dataType
        self.device = DEVICE
        self.len = None # int, the input vector length that this normalizer 
                          # can handle
        self.normedLen = None
    
    def trans(self, vec):
        assert False

    def toDevice(self, device):

        self.device = device
        dictForm = vars(self)

        for key in dictForm:
            if(isinstance(dictForm[key], Normalizer)):
                dictForm[key].toDevice(device)
            elif(isinstance(dictForm[key], torch.Tensor)):
                dictForm[key] = dictForm[key].to(device)
        
    def changeType(self, newType, device=DEVICE):

        self.type = newType
        self.device = device

        dictForm = vars(self)
        for key in dictForm:
            if(isinstance(dictForm[key], Normalizer)):
                dictForm[key].changeType(newType, device)
            else:
                dictForm[key] = changeType(dictForm[key], newType, device)
            
    def toJSON(self, level=0):

        selfDict = copy.deepcopy(vars(self))

        # check item type, if it is np array, convert to list
        for key in selfDict:
            if(isinstance(selfDict[key], Normalizer)):
                selfDict[key] = selfDict[key].toJSON(level + 1)
            elif(isinstance(selfDict[key], torch.device)):
                selfDict[key] = None
            else:
                selfDict[key] = changeType(selfDict[key], "list")
        
        return selfDict

class PCA(Normalizer):
    def __init__(self, threshold=None, matrix=0, device=DEVICE):

        super(PCA, self).__init__("PCA", device)

        self.threshold = threshold # float, the percentage of information to retain
        self.sliceID = None # int, the index to cut off at
        
        self.vectors = None # torch tensor, the eigen vector of PCA
        self.values = None # torch tensor, the eigen values

        self.device = device

        if(not(isinstance(matrix, int))):
            inLen, outLen = self.getPC(matrix)
            self.len = inLen
            self.normedLen = outLen
            self.toDevice(self.device)

    def getPC(self, matrix):

        if(len(matrix.shape) != 2):
            assert False
        
        
        if(isinstance(matrix, np.ndarray)):
            self.getPCANumpy(matrix)
        elif(isinstance(matrix, torch.Tensor)):
            self.getPCATorch(matrix)

        self.setThreshold(self.threshold)
        
        inLen = matrix.shape[1]
        normedLen = self.getOutputLen()

        return inLen, normedLen

    def setThreshold(self, threshold):

        self.threshold = threshold

        # decide how many vectors to use based on the threshold
        infoComp = self.cumSum()
        infoCompNumpy = infoComp.detach().cpu().numpy()
        self.sliceID = np.searchsorted(infoCompNumpy, self.threshold) + 1
        self.sliceID = min(int(self.sliceID), len(self.values))

    def getOutputLen(self):
        
        return self.sliceID

    def cumSum(self):

        if(isinstance(self.values, np.ndarray)):
            return np.cumsum(self.values, 0) / np.sum(self.values)
        elif(isinstance(self.values, torch.Tensor)):
            return  torch.cumsum(self.values, 0) / torch.sum(self.values)

    def getPCANumpy(self, matrix):
        # get eigen values and vectors
        cov = np.cov(matrix.T)
        values, vectors = np.linalg.eig(cov)
        
        # sort vectors based on values
        sortIndices = np.argsort(values, axis = 0)[::-1]
        values /= np.sum(values)
        self.values = values[sortIndices]
        self.vectors = vectors[:,sortIndices]

    def getPCATorch(self, matrix, rowVar=False):

        # fix orientation
        if not rowVar and matrix.shape[0] != 1: matrix = matrix.t()
        
        # get covariance matrix
        factor = 1 / (matrix.shape[1] - 1)
        cov = matrix.matmul(matrix.t()) * factor
        # get eigen values and vectors
        values, vectors = torch.eig(cov, eigenvectors=True)
        # sort vectors based on values
        values = values[:,0] # get rid of trash
        values /= torch.sum(values)
        sortIndices = torch.sort(values, 0, True)[1] # large to small
        self.values = values[sortIndices].float().to(self.device)
        self.vectors = vectors[:,sortIndices].float().to(self.device)

    def trans(self, matrix):

        #(A.matmul(B))^T = B^T.matmul(A^T) 
        # v->B, m->A
        #mT = matrix.t() # a trans
        vectors = self.vectors[:, :self.sliceID]
        #vT = vectors.t() # b trans

        #result = vT.matmul(mT).t()
        if(isinstance(matrix, np.ndarray)):
            result = np.matmul(matrix, vectors)
        elif(isinstance(matrix, torch.Tensor)):
            result = matrix.matmul(vectors)
        
        return result

class PCANormalizer_indi(Normalizer):
    def __init__(self, threshold=None, matrix=0, numType=[], device=DEVICE):
        
        super(PCANormalizer_indi, self).__init__("PCA_indi", device)

        self.threshold = threshold # float, the percentage of information to retain

        self.geoStart = None
        self.geoEnd = None
        self.geoPCA = None # PCA, for geometrical terms
        self.stressStart = None
        self.stressEnd = None
        self.stressPCA = None # PCA, for stress terms
        self.bcStart = None
        self.bcEnd = None
        self.bcPCA = None # PCA, for boundary condition terms

        self.device = device

        if(not isinstance(matrix, int)):
            inLen, outLen = self.getPCA(matrix, numType, device)
            self.len = inLen
            self.normedLen = outLen
            self.toDevice(self.device)

    def getPCA(self, matrix, numType, device):
        # get the sub-PCA components
        # parameters:
        #   matrix:  the matrix of data to process
        #   numType: List<String>, the number type indicator

        inLen = len(numType)

        geoStart, geoEnd = None, None
        stressStart, stressEnd = None, None
        bcStart, bcEnd = None, None
        wasGeo, wasStress, wasBC = False, False, False
        # loop to check index
        for i in range(len(numType)):
            curType = numType[i]
            isGeo = (curType.startswith(POINT_TOKEN) or curType.startswith(VECTOR_TOKEN))
            isStress = (curType.startswith(STRESS_TOKEN))
            isBC = (curType.startswith(BC_TOKEN))
            # geometry
            if(not wasGeo and isGeo): wasGeo, geoStart = True, i
            if(wasGeo and not isGeo): wasGeo, geoEnd = False, i
            # stress
            if(not wasStress and isStress): wasStress, stressStart = True, i
            if(wasStress and not isStress): wasStress, stressEnd = False, i
            # boundary condition
            if(not wasBC and isBC): wasBC, bcStart = True, i
            if(wasBC and not isBC): wasBC, bcEnd = False, i
        
        # check last
        if(geoEnd == None and geoStart != None): geoEnd = len(numType)
        if(stressEnd == None and stressStart != None): stressEnd = len(numType)
        if(bcEnd == None and bcStart != None): bcEnd = len(numType)

        # save index
        self.geoStart = geoStart
        self.geoEnd = geoEnd
        self.stressStart = stressStart
        self.stressEnd = stressEnd
        self.bcStart = bcStart
        self.bcEnd = bcEnd
        
        # get PCA
        # geometry
        if(geoStart != None and geoEnd != None): 
            geoPart = matrix[:, geoStart:geoEnd]
            self.geoPCA = PCA(self.threshold, geoPart, device)
        else:
            self.geoPDCA = None
        # stress
        if(stressStart != None and stressEnd != None):
            stressPart = matrix[:, stressStart:stressEnd]
            self.stressPCA = PCA(self.threshold, stressPart, device)
        else: 
            self.stressPCA = None
        # boundary condition
        if(bcStart != None and bcEnd != None):
            bcPart = matrix[:, bcStart:bcEnd]
            self.bcPCA = PCA(self.threshold, bcPart, device)
        else:
            self.bcPCA = None

        outLen = self.getOutputLen()

        return inLen, outLen

    def setThreshold(self, threshold):

        self.threshold = threshold

        # decide how many vectors to use based on the threshold
        if(isinstance(self.geoPCA, PCA)):
            self.geoPCA.setThreshold(threshold)
        if(isinstance(self.stressPCA, PCA)):
            self.stressPCA.setThreshold(threshold)
        if(isinstance(self.bcPCA, PCA)):
            self.bcPCA.setThreshold(threshold)

    def getOutputLen(self):

        # geometry
        if(isinstance(self.geoPCA, PCA)):
            geoLen = self.geoPCA.getOutputLen()
        else:
            geoLen = 0
        # stress
        if(isinstance(self.stressPCA, PCA)):
            stressLen = self.stressPCA.getOutputLen()
        else:
            stressLen = 0
        # boundary condition
        if(isinstance(self.bcPCA, PCA) and PCA_BC):
            bcLen = self.bcPCA.getOutputLen()
        elif(isinstance(self.bcPCA, PCA) and not PCA_BC):
            bcLen = self.bcPCA.len
        else:
            bcLen = 0
        
        outLen = geoLen + stressLen + bcLen

        return outLen

    def trans(self, matrix):

        # slice into parts
        geo = sliceMD(matrix, self.geoStart, self.geoEnd)
        stress = sliceMD(matrix, self.stressStart, self.stressEnd)
        bc = sliceMD(matrix, self.bcStart, self.bcEnd)

        # transform individual parts
        all = []
        if(self.geoPCA):
            all += [self.geoPCA.trans(geo)]
        if(self.stressPCA):
            all += [self.stressPCA.trans(stress)]
        if(self.bcPCA and PCA_BC):
            all += [self.bcPCA.trans(bc)]
        elif(self.bcPCA and not PCA_BC):
            all += [bc]
        
        # put parts back together
        if(isinstance(matrix, np.ndarray)):
            result = np.concatenate(all, axis=-1)
        elif(isinstance(matrix, torch.Tensor)):
            result = torch.cat(all, -1)
        
        return result

class PCANormalizer_all(Normalizer):
    def __init__(self, threshold=None, matrix=0, device=DEVICE):

        super(PCANormalizer_all, self).__init__("PCA_all", device)

        self.threshold = threshold # float, the percentage of information to retain
        self.sliceID = None # int, the index to cut off at
        
        self.vectors = None # torch tensor, the eigen vector of PCA
        self.values = None # torch tensor, the eigen values

        self.device = device

        if(not(isinstance(matrix, int))):
            inLen, outLen = self.getPC(matrix)
            self.len = inLen
            self.normedLen = outLen
            self.toDevice(self.device)

    def getPC(self, matrix):

        if(len(matrix.shape) != 2):
            assert False
        
        inLen = matrix.shape[1]
        
        if(isinstance(matrix, np.ndarray)):
            self.getPCANumpy(matrix)
        elif(isinstance(matrix, torch.Tensor)):
            self.getPCATorch(matrix)

        self.setThreshold(self.threshold)

        outLen = self.getOutputLen()
        
        return inLen, outLen

    def setThreshold(self, threshold):

        self.threshold = threshold

        # decide how many vectors to use based on the threshold
        infoComp = self.cumSum()
        infoCompNumpy = infoComp.detach().cpu().numpy()
        self.sliceID = np.searchsorted(infoCompNumpy, self.threshold) + 1
        self.sliceID = min(int(self.sliceID), self.len)

    def getOutputLen(self):

        return self.sliceID

    def cumSum(self):
        return torch.cumsum(self.values, 0) / torch.sum(self.values)

    def getPCANumpy(self, matrix):
        # get eigen values and vectors
        cov = np.cov(matrix.T)
        values, vectors = np.linalg.eig(cov)
        
        # sort vectors based on values
        sortIndices = np.argsort(values, axis = 0)[::-1]
        values /= np.sum(values)
        self.values = values[sortIndices]
        self.vectors = vectors[:,sortIndices]

    def getPCATorch(self, matrix, rowVar=False):

        # fix orientation
        if not rowVar and matrix.shape[0] != 1: matrix = matrix.t()
        
        # get covariance matrix
        factor = 1 / (matrix.shape[1] - 1)
        cov = matrix.matmul(matrix.t()) * factor
        # get eigen values and vectors
        values, vectors = torch.eig(cov, True)

        # sort vectors based on values
        values = values[:,0] # get rid of trash
        values /= torch.sum(values)
        sortIndices = torch.sort(values, 0, True)[1] # large to small
        self.values = values[sortIndices].float().to(self.device)
        self.vectors = vectors[:,sortIndices].float().to(self.device)

    def trans(self, matrix):
        #(A.matmul(B))^T = B^T.matmul(A^T) 
        # v->B, m->A
        #mT = matrix.t() # a trans
        vectors = self.vectors[:, :self.sliceID]
        #vT = vectors.t() # b trans

        #result = vT.matmul(mT).t()
        if(isinstance(matrix, np.ndarray)):
            result = np.matmul(matrix, vectors)
        elif(isinstance(matrix, torch.Tensor)):
            result = torch.matmul(matrix, vectors)
        
        return result

class MeanVarNormalizer(Normalizer):
    def __init__(self, matrix=0, device=DEVICE):

        super(MeanVarNormalizer, self).__init__("meanVar", device)

        self.percentile = VAR_PERCENTILE # Float between 0 to 100
        self.mean = None # torch tensor, the mean of the vector (cols)
        self.var = None # torch tensor, the number to normalize to -1 with

        self.device = device

        if(not isinstance(matrix, int)): # construct if matrix is supplied
            inLen, outLen = self.getMeanVar(matrix)
            self.len = inLen
            self.normedLen = outLen
            self.toDevice(self.device)

    def getMeanVar(self, matrix):
        # analyze the mean and lower/upper percentiles of the input matrix and 
        # save them for future transformations
        # parameters:
        #   matrix: 2D list, the whole dataset to learn parameters from

        # check type
        if(isinstance(matrix, list)): npMatrix = np.array(matrix)
        elif(isinstance(matrix, torch.Tensor)): npMatrix = matrix.cpu().numpy()
        elif(isinstance(matrix, np.ndarray)): npMatrix = matrix

        # get informaiton
        self.len = npMatrix.shape[-1]
        self.mean = np.mean(npMatrix, 0) # across rows

        oldDType = npMatrix.dtype
        npMatrix = npMatrix.astype(np.float16) # save mem
        mean16 = self.mean.astype(np.float16) # save mem

        if(USE_PERC): # percentile stretch
            centered = np.abs(npMatrix - mean16)
            self.var = np.percentile(centered, self.percentile, axis=0).astype(oldDType)

        else: # std stretch
            centered = npMatrix - self.mean
            self.var = np.std(centered, axis=0)


        # check and fix 0 in lower and upper numbers, otherwise they will cause
        # divide-by-zero errors later
        for i in range(self.len):
            if(self.var[i] == 0): self.var[i] = 1

        # convert type
        if(isinstance(matrix, list)):
            self.mean = self.mean.tolist()
            self.var = self.var.tolist()
        elif(isinstance(matrix, torch.Tensor)):
            self.mean = torch.from_numpy(self.mean).float().to(self.device)
            self.var = torch.from_numpy(self.var).float().to(self.device)
        elif(isinstance(matrix, np.ndarray)):
            pass
        
        inLen = npMatrix.shape[-1]
        outLen = inLen

        return inLen, outLen

    def trans(self, matrix):
        #  apply the transformation stored in this normalizer
        # parameter:
        #   matrix: up to a pytorch 3D tensor, the data matrix to normalize
        # return:
        #   the transformed matrix as an pytorch tensor
        
        return (matrix - self.mean) / self.var

    def invert(self, matrix):
        #  invert the transformation stored in this normalizer
        # parameter:
        #   matrix: a pytorch 2D tensor, the data matrix to normalize
        # return:
        #   the inversely transformed matrix as an pytorch tensor

        return (matrix * self.var) + self.mean

class GeoOffsetNormalizer(Normalizer):
    def __init__(self, numType=None, device=DEVICE):

        super(GeoOffsetNormalizer, self).__init__("geoOff", device)

        self.indexPairs = [] # list of tuples, pairs of indices to normalize
                             # position with
        
        self.device = device

        if(numType != None): # construct if a numType vector is supplied
            inLen, outLen = self.getIndexPairs(numType)
            self.len = inLen
            self.normedLen = outLen
            self.toDevice(self.device)
    
    def getIndexPairs(self, numType):
        # check the number type and create a list of normalization coordinates
        # parameter:
        #   numType: a vector of characters describing the number type
        # return:
        
        # loop through the vector with a while loop
        index, pairs = 0, []
        while (index < len(numType)):
            # read an point number, check consequent values for legitimacy

            if(numType[index] == POINT_TOKEN):
                # check the following terms for point data
                valid = True
                for j in range(1, COOR_LEN):
                    if(numType[index + j] != POINT_TOKEN):
                        valid = False
                        break
                
                if(valid):
                    pairs += [tuple([index + j for j in range(COOR_LEN)])]
                    index += COOR_LEN
                    continue # have already shifted index, skip to next loop
            
            # either the current numtype is not point or it failed the check
            index += 1
        
        self.indexPairs = pairs

        inLen = len(numType)
        outLen = inLen

        return inLen, outLen

    def changeType(self, newType, device=DEVICE):
        
        super(GeoOffsetNormalizer, self).changeType(newType, device)
        
        if(newType == "torch"): dType = torch.int32
        elif(newType == "numpy"): dType = np.int32
        else: dType = None
            
        self.indexPairs = changeType(self.indexPairs, newType, device,
                                     dType=dType)

    def trans(self, matrix, center):
        # takes a vector and reposition it with the new center coordinates
        # parameters:
        #   matrix: up to torch 3D Tensor, the matrix to process
        #   center: the new center point position
        # return:
        #   the transformed matrix

        # check input vector length and center coordinate length
        # checking the input type at the same time

        shape = (matrix.shape)
        dim = len(shape)

        # transformation matrix
        if(isinstance(matrix, np.ndarray)):
            trans = np.zeros(shape, dtype=matrix.dtype)
        else:
            trans = torch.zeros(shape, dtype=matrix.dtype).to(self.device)
        # set transformation matrix elements
        for pair in self.indexPairs:
            start = pair[0]
            if(dim == 1):
                trans[start:start + COOR_LEN] = center
            elif(dim == 2):
                trans[:, start:start + COOR_LEN] = center
            elif(dim == 3):
                trans[:, :, start:start + COOR_LEN] = center
            elif(dim == 4):
                trans[:, :, :, start:start + COOR_LEN] = center
            else:
                assert False, "dimension error"

        return matrix - trans

class ElementNormalizer(Normalizer):
    def __init__(self, elemsIn=0, elemsOut=0, numType=0, center=0, device=DEVICE):

        super(ElementNormalizer, self).__init__("elemNorm", device)
        
        self.center = None

        # input
        self.inGeo = None
        self.inVar = None
        self.inPCA = None
        self.inPostVar = None

        # output
        self.outVar = None
        
        self.device = device

        if(not isinstance(elemsIn, int)):
            inLen, outLen = self.getNorms(elemsIn, elemsOut, numType, center, device)
            self.len = inLen
            self.normedLen = outLen
            self.toDevice(self.device)
    
    def getNorms(self, elemsIn, elemsOut, numType, center, device):

        inVecLen = len(numType)

        # input
        # geometrical offset
        self.center = tuple([i for i in range(center[0],center[1])])
        self.inGeo = GeoOffsetNormalizer(numType, device)
        # mean variance normalizer
        centerIn = self.getCenter(elemsIn)
        elemsIn = self.inGeo.trans(elemsIn, centerIn)
        self.inVar = MeanVarNormalizer(elemsIn, device)
        # PCA normalizer
        elemsIn = self.inVar.trans(elemsIn)
        if(PCA_ALL):
            self.inPCA = PCANormalizer_all(PCA_THRESHOLD, elemsIn, device)
        else:
            self.inPCA = PCANormalizer_indi(PCA_THRESHOLD, elemsIn, numType, device)
        # post PCA mean variance normalizer
        if(POST_PCA_MEAN_VAR):
            elemsIn = self.inPCA.trans(elemsIn)
            self.inPostVar = MeanVarNormalizer(elemsIn, device)
        
        # output, the target is already normalized in delta mode
        # mean variance
        if(USE_DELTA):
            # delta mode
            self.outVar = MeanVarNormalizer(elemsOut, device)
        else:
            centerOut = self.getCenter(elemsOut)
            centeredOut = self.inGeo.trans(elemsOut, centerOut)
            self.outVar = MeanVarNormalizer(centeredOut, device)

        # lengths
        inLen = len(numType)
        if(USE_PCA):
            outLen = self.inPCA.getOutputLen()
        else:
            outLen = inVecLen
        
        return inLen, outLen

    def getCenter(self, matrix):

        shape = matrix.shape
        dim = len(shape)

        if(isinstance(matrix, np.ndarray)):
            cen = np.take(matrix, self.center, axis=dim - 1)
        else:
            if(isinstance(self.center, torch.Tensor)): indices = self.center
            else: indices = torch.Tensor(self.center)
            indices = indices.long().to(self.device)
            cen = torch.index_select(matrix, dim - 1, indices)
        
        return cen

    def normInput(self, matrix):

        dim = len(matrix.shape)
        isLatent = matrix.shape[-1] != self.len
        
        if(not isLatent): # base case, no latent vector
            center = self.getCenter(matrix)
            normed = self.inGeo.trans(matrix, center)
            normed = self.inVar.trans(normed)
            if(USE_PCA):
                normed = self.inPCA.trans(normed)
                if(POST_PCA_MEAN_VAR):
                    normed = self.inPostVar.trans(normed)
        else: # with latent feature
            latentLen = matrix.shape[-1] - self.len
            if(isinstance(matrix, torch.Tensor)): # Pytorch
                source, latent = torch.split(matrix,
                                             [self.len, latentLen],
                                             dim - 1)
                normed = self.normInput(source)
                normed = torch.cat((normed, latent), dim - 1)
            elif(isinstance(matrix, np.ndarray)): # Numpy
                source, latent = np.split(matrix,
                                          [self.len],
                                          dim - 1)
                normed = self.normInput(source)
                normed = np.concatenate((normed, latent), axis=-1)
        
        return normed

    def normOutput(self, matrix):
        
        return self.outVar.trans(matrix)

    def invertOutput(self, matrix):

        return self.outVar.invert(matrix)

    def _testNormTime(self, matrix):

        print("now: get center")
        s = t.time()
        for i in range(1000): center = self.getCenter(matrix)
        e = t.time()
        print("get center time: %.3f" % (e - s))
        print("now: geo reposition")
        s = t.time()
        for i in range(1000): reposed = self.inGeo.trans(matrix, center)
        e = t.time()
        print("geo reposition time: %.3f" % (e - s))
        print("now: mean var")
        s = t.time()
        for i in range(1000): normed = self.inVar.trans(reposed)
        e = t.time()
        print("mean var time: %.3f" % (e - s))
        if(USE_PCA):
            print("now: PCA")
            s = t.time()
            for i in range(1000): reduced = self.inPCA.trans(normed)
            e = t.time()
            print("PCA time: %.3f" % (e - s))
            
            if(POST_PCA_MEAN_VAR):
                print("now: post PCA mean var")
                s = t.time()
                for i in range(1000): last = self.inPostVar.trans(reduced)
                e = t.time()
                print("post PCA mean var time: %.3f" % (e - s))

class InteractionNormalizer(Normalizer):
    def __init__(self, intPair=0, numType=0, center=0, device=DEVICE):
        
        super(InteractionNormalizer, self).__init__("intNorm", device)

        self.nodeLen = None
        self.edgeLen = None
        self.center = None
        
        # input node
        self.inNodeGeo = None
        self.inNodeVar = None
        self.inNodePCA = None
        self.inNodePostVar = None

        # input edge
        self.inEdgeGeo = None
        self.inEdgeVar = None
        self.inEdgePCA = None
        self.inEdgePostVar = None

        self.device = device

        if(not isinstance(intPair, int)):
            inLen, outLen = self.getNorms(intPair, numType, center, device)
            self.len = inLen
            self.normedLen = outLen
            self.toDevice(self.device)

    def getNorms(self, intPair, numType, center, device):

        # unpack
        # omit the second node because it's a duplicate of the first in a
        #  different order
        edges, nodes, _ = intPair
        eType, nType, _ = numType
        
        inNodeLen, inEdgeLen = len(nType), len(eType)

        self.center = tuple([i for i in range(center[0],center[1])])
        self.inNodeGeo = GeoOffsetNormalizer(nType, device)
        self.inEdgeGeo = GeoOffsetNormalizer(eType, device)
        # mean variance normalizer
        centerIn = self.getCenter(edges)
        nodes = self.inNodeGeo.trans(nodes, centerIn)
        edges = self.inEdgeGeo.trans(edges, centerIn)
        self.inNodeVar = MeanVarNormalizer(nodes, device)
        self.inEdgeVar = MeanVarNormalizer(edges, device)
        # PCA normalizer
        nodes = self.inNodeVar.trans(nodes)
        edges = self.inEdgeVar.trans(edges)
        if(PCA_ALL):
            self.inNodePCA = PCANormalizer_all(PCA_THRESHOLD, nodes, device)
            self.inEdgePCA = PCANormalizer_all(PCA_THRESHOLD, edges, device)
        else:
            self.inNodePCA = PCANormalizer_indi(PCA_THRESHOLD, nodes, nType,
                                                device)
            self.inEdgePCA = PCANormalizer_indi(PCA_THRESHOLD, edges, eType,
                                                device)
        # post PCA mean variance normalizer
        nodes = self.inNodePCA.trans(nodes)
        edges = self.inEdgePCA.trans(edges)
        self.inNodePostVar = MeanVarNormalizer(nodes, device)
        self.inEdgePostVar = MeanVarNormalizer(edges, device)

        # set lengths
        # input
        self.nodeLen = inNodeLen
        self.edgeLen = inEdgeLen
        inLen = inEdgeLen + inNodeLen + inNodeLen
        # output
        if(USE_PCA):
            nodeOutLen = self.inNodePCA.getOutputLen()
            edgeOutLen = self.inEdgePCA.getOutputLen()
            outLen = edgeOutLen + nodeOutLen + nodeOutLen
        else:
            outLen = inEdgeLen + inNodeLen + inNodeLen

        return inLen, outLen

    def getCenter(self, matrix):

        shape = matrix.shape
        dim = len(shape)

        if(isinstance(matrix, np.ndarray)):
            cen = np.take(matrix, self.center, axis=dim - 1)
        else:
            if(isinstance(self.center, torch.Tensor)): indices = self.center
            else: indices = torch.Tensor(self.center)
            indices = indices.long().to(self.device)
            cen = torch.index_select(matrix, dim - 1, indices)
        
        return cen

    def norm(self, data, latLen=0):
        
        tupleMode = isinstance(data, tuple)
        # split data into chunks
        splitted = self.splitData(data, latLen)
        edge, edgeLat, node0, node0Lat, node1, node1Lat = splitted
        
        # normalize offset
        center = self.getCenter(edge)
        edge = self.inEdgeGeo.trans(edge, center)
        node0 = self.inNodeGeo.trans(node0, center)
        node1 = self.inNodeGeo.trans(node1, center)
        # mean variance
        edge = self.inEdgeVar.trans(edge)
        node0 = self.inNodeVar.trans(node0)
        node1 = self.inNodeVar.trans(node1)
        # PCA
        if(USE_PCA):
            edge = self.inEdgePCA.trans(edge)
            node0 = self.inNodePCA.trans(node0)
            node1 = self.inNodePCA.trans(node1)
            if(POST_PCA_MEAN_VAR):
                edge = self.inEdgePostVar.trans(edge)
                node0 = self.inNodePostVar.trans(node0)
                node1 = self.inNodePostVar.trans(node1)

        # assemble input back together
        chunks = (edge, edgeLat, node0, node0Lat, node1, node1Lat)
        assembled = self.assembleData(chunks, tupleMode)

        return assembled

    def splitData(self, data, latLen):

        splitted = isinstance(data, tuple)
        # check input type and mode
        if(splitted): # tuple mode
            mode = "numpy" if isinstance(data[0], np.ndarray) else "torch"
        else: # matrix mode
            mode = "numpy" if isinstance(data, np.ndarray) else "torch"
        if(mode == "numpy"):
            inpSplit, latSplit = self.splitInputNumpy, self.splitLatentNumpy
        elif(mode == "torch"):
            inpSplit, latSplit = self.splitInputTorch, self.splitLatentTorch
        # split input
        edge, node0, node1 = None, None, None
        edgeLat, node0Lat, node1Lat = None, None, None
        if(splitted): edge, node0, node1 = data
        else: edge, node0, node1 = inpSplit(data, latLen)
        # check latent feature
        edgeIsLat = not (edge.shape[-1] == self.edgeLen)
        node0IsLat = not (node0.shape[-1] == self.nodeLen)
        node1IsLat = not (node1.shape[-1] == self.nodeLen)
        if(edgeIsLat): edge, edgeLat = latSplit(edge, self.edgeLen)
        if(node0IsLat): node0, node0Lat = latSplit(node0, self.nodeLen)
        if(node1IsLat): node1, node1Lat = latSplit(node1, self.nodeLen)

        return (edge, edgeLat, node0, node0Lat, node1, node1Lat)

    def splitInputNumpy(self, matrix, latLen):

        break0 = 0
        break1 = break0 + self.edgeLen + latLen[0]
        break2 = break1 + self.nodeLen + latLen[1]
        break3 = break2 + self.nodeLen + latLen[2]

        edgeId  = np.arange(break0, break1, dtype=np.int16)
        node0Id = np.arange(break1, break2, dtype=np.int16)
        node1Id = np.arange(break2, break3, dtype=np.int16)

        edge  = np.take(matrix, edgeId, axis=-1)
        node0 = np.take(matrix, node0Id, axis=-1)
        node1 = np.take(matrix, node1Id, axis=-1)

        return edge, node0, node1

    def splitInputTorch(self, matrix, latLen):

        edgeLen = self.edgeLen + latLen[0]
        node0Len = self.nodeLen + latLen[1]
        node1Len = self.nodeLen + latLen[2]

        splitSize = [edgeLen, node0Len, node1Len]

        edge, node0, node1 = torch.split(matrix, splitSize, -1)

        return edge, node0, node1

    def splitLatentNumpy(self, matrix, elemLen):
        
        break0 = 0
        break1 = elemLen
        break2 = matrix.shape[-1]

        elemId = np.arange(break0, break1, dtype=np.int16)
        latId  = np.arange(break1, break2, dtype=np.int16)

        elem = np.take(matrix, elemId, axis=-1)
        lat  = np.take(matrix,  latId, axis=-1)
        
        return elem, latent

    def splitLatentTorch(self, matrix, elemLen):

        latLen = matrix.shape[-1] - elemLen
        splitSize = [elemLen, latLen]

        elem, lat = torch.split(matrix, splitSize, -1)

        return elem, lat

    def assembleData(self, chunks, tupleMode):

        if(isinstance(chunks[0], np.ndarray)): mode = "numpy"
        elif(isinstance(chunks[0], torch.Tensor)): mode = "torch"

        isLatent = isinstance(chunks[1], np.ndarray) or \
                   isinstance(chunks[1], torch.Tensor)
        
        if(isLatent):
            assembled = chunks
        else:
            assembled = (chunks[0], chunks[2], chunks[4])
        if(not tupleMode):
            if(mode == "numpy"):
                assembled = np.concatenate(assembled, axis=-1)
            elif(mode == "torch"):
                assembled = torch.cat(assembled, -1)
        
        return assembled

class GridNorm(object):
    def __init__(self, data=0, mode="torch", device=DEVICE):

        self.type = mode
        self.device = device

        # int, length of input data
        self.nodeLen = None # length of input joint vectors
        self.edgeLen = None # length of input beam vectors
        # int, length of output (normalized) data
        self.nodeNormedLen = None # length of input joint vectors
        self.edgeNormedLen = None # length of input beam vectors
        self.nodeIntNormedLen = None # 
        self.edgeIntNormedLen = None #

        # normalizers
        self.nodeNorm = None
        self.edgeNorm = None
        self.nodeIntNorm = None
        self.edgeIntNorm = None

        if(not isinstance(data, int)):
            if(data.type != mode): data.toType(mode, device)
            self.getNorms(data, device)

    def getNorms(self, data, device):

        # elements
        nodes, edges = data.getNodes(), data.getEdges()
        nodesTar, edgesTar = data.getNodesTar(), data.getEdgesTar()
        nodesType, edgesType = data.nodeNumType, data.edgeNumType
        nodesCen, edgesCen = data.nodeCenter, data.edgeCenter
        self.nodeNorm = ElementNormalizer(nodes, nodesTar, nodesType, nodesCen,
                                          device=device)
        self.edgeNorm = ElementNormalizer(edges, edgesTar, edgesType, edgesCen,
                                          device=device)

        # interactions
        nodeIntIn = data.getPairs('n')
        nodeIntNumtype = (data.edgeNumType, data.nodeNumType, data.nodeNumType)
        self.nodeIntNorm = InteractionNormalizer(nodeIntIn, nodeIntNumtype,
                                                 edgesCen, device=device)
        edgeIntIn = data.getPairs('e')
        edgeIntNumtype = (data.nodeNumType, data.edgeNumType, data.edgeNumType)
        self.edgeIntNorm = InteractionNormalizer(edgeIntIn, edgeIntNumtype,
                                                   nodesCen, device=device)

        # set Lengths
        self.nodeLen = len(data.nodeNumType)
        self.edgeLen = len(data.edgeNumType)
        self.nodeNormedLen = self.nodeNorm.normedLen
        self.edgeNormedLen = self.edgeNorm.normedLen
        self.nodeIntNormedLen = self.nodeIntNorm.normedLen
        self.edgeIntNormedLen = self.edgeIntNorm.normedLen
    
    def normNode(self, matrix):

        return self.nodeNorm.normInput(matrix)

    def normEdge(self, matrix):

        return self.edgeNorm.normInput(matrix)

    def normNodeOut(self, matrix):
        
        return self.nodeNorm.normOutput(matrix)

    def normEdgeOut(self, matrix):
        
        return self.edgeNorm.normOutput(matrix)
    
    def normNodeInt(self, matrix, latLen=None):
        
        return self.nodeIntNorm.norm(matrix)

    def normEdgeInt(self, matrix,latLen=None):
        
        return self.edgeIntNorm.norm(matrix)
    
    def invertNodeOut(self, matrix):

        return self.nodeNorm.invertOutput(matrix)

    def invertEdgeOut(self, matrix):

        return self.edgeNorm.invertOutput(matrix)
    
    def showLengths(self):

        print("==============================")
        print("normalizer lengths:")
        print("node length: %d" % self.nodeLen)
        print("edge length: %d" % self.edgeLen)
        print("normalized node length: %d" % self.nodeNormedLen)
        print("normalized edge length: %d" % self.edgeNormedLen)
        print("normalized node interaction length: %d" % self.nodeIntNormedLen)
        print("normalized edge interaction length: %d" % self.edgeIntNormedLen)
        print("==============================")

    def toDevice(self, device):

        self.device = device
        dictForm = vars(self)

        for key in dictForm:
            if(isinstance(dictForm[key], Normalizer)):
                dictForm[key].toDevice(device)
            elif(isinstance(dictForm[key], torch.Tensor)):
                dictForm[key] = dictForm[key].to(device)

    def toType(self, newType, device=DEVICE):

        self.type = newType
        self.device = device
        dictForm = vars(self)

        for key in dictForm:
            if(isinstance(dictForm[key], Normalizer)):
                dictForm[key].changeType(newType, device)
            else:
                dictForm[key] = changeType(dictForm[key], newType, device)

    def save(self, path):
        # saves the current normalizer to a .json file
        # parameters:
        #   path: the path to save the normalizer to

        # convert this normalizer to a dictionary
        dictForm = {}
        dictSelf = copy.deepcopy(vars(self))
        for key in dictSelf:
            # for the sub-normalizers, convert them into dictionaries too
            if(isinstance(dictSelf[key], Normalizer)):
                dictForm[key] = dictSelf[key].toJSON()
            elif(isinstance(dictSelf[key], torch.device)):
                dictForm[key] = None
            else: # ordinary datatype, directly dump into the dictionary
                dictForm[key] = changeType(dictSelf[key], "list")
            
        writeJSON(path, dictForm)

    @staticmethod
    def load(path, mode="torch", device=DEVICE):
        
        dictForm = readJSON(path) # read raw dictionary form from file
        norm = GridNorm()

        normDictForm = vars(norm)
        for key in dictForm:
            if(isinstance(dictForm[key], dict)):
                normDictForm[key] = _normLoader(dictForm[key])
            else:
                normDictForm[key] = dictForm[key]

        norm.toType(mode, device)

        return norm

def _normLoader(dictForm):

    normType = dictForm["func"]

    if(normType == "geoOff"):
        norm = GeoOffsetNormalizer()
    elif(normType == "meanVar"):
        norm = MeanVarNormalizer()
    elif(normType == "PCA"):
        norm = PCA()
    elif(normType == "PCA_indi"):
        norm = PCANormalizer_indi()
    elif(normType == "PCA_all"):
        norm = PCANormalizer_all()
    elif(normType == "elemNorm"):
        norm = ElementNormalizer()
    elif(normType == "intNorm"):
        norm = InteractionNormalizer()

    normDictForm = vars(norm)
    for key in dictForm:
        if(isinstance(dictForm[key], dict)):
            normDictForm[key] = _normLoader(dictForm[key])
        else:
            normDictForm[key] = dictForm[key]

    return norm

def _testNormTime():
    print("testing normalization time 1000")
    print("loading data...")
    d = GraphStack().load(DATA_FOLDER)
    d.toType("torch", DEVICE_CPU)
    print("done!")

    print("making normalizer...")
    norm = GridNorm(d, mode="torch", device=DEVICE_CPU)
    norm.toType("torch", device=DEVICE_GPU)
    print("done!")

    print("getting graph nodes")
    nodes = d.getNodes().to(DEVICE_GPU)
    print("done!")

    print("input nodes:", nodes.shape, nodes.device)
    print('=' * 80)
    print("testing normalization now...")
    norm.nodeNorm._testNormTime(nodes)

def _testNorm():
    # load data
    d = GraphStack().load(DATA_FOLDER)
    d.toType("numpy")
    norm = GridNorm(d, mode="torch")
    # test change type
    norm.toType("numpy")
    norm.toType("list")
    norm.toType("torch")
    # test save and load
    path = DATA_FOLDER + '/' + NORM_TOKEN + JSON_SUFFIX
    path0 = DATA_FOLDER + '/' + NORM_TOKEN + "_0" + JSON_SUFFIX
    path1 = DATA_FOLDER + '/' + NORM_TOKEN + "_1" + JSON_SUFFIX
    norm.save(path0)
    norm2 = GridNorm.load(path0)
    norm2.save(path1)
    norm3 = GridNorm.load(path1)
    # test normalization functions
    # nodes
    nodes = d.getNodes()
    nodesNormed = norm3.normNode(nodes)
    # edges
    edges = d.getEdges()
    edgesNormed = norm3.normEdge(edges)
    # nodes target
    nodesTar = d.getNodesTar()
    nodesTarNormed = norm3.normNodeOut(nodesTar)
    nodesTarInv = norm3.invertNodeOut(nodesTarNormed)
    # edges target
    edgesTar = d.getEdgesTar()
    edgesTarNormed = norm3.normEdgeOut(edgesTar)
    edgesTarInv = norm3.invertEdgeOut(edgesTarNormed)
    # nodes interaction
    nodeInt = d.getPairs('n')
    nodeIntNormed = norm3.normNodeInt(nodeInt)
    # edges interaction
    edgeInt = d.getPairs('e')
    edgeIntNormed = norm3.normEdgeInt(edgeInt)
    
    print("ok")

def testMeanVar():
    data = GraphStack().load(DATA_FOLDER)
    nodes = data.getNodes()
    nodes = torch.from_numpy(nodes)
    #mv = MeanVarNormalizer(nodes)
    print(torch.mean(nodes, 0))
    print(torch.std(nodes, 0))
    print(torch.std(nodes - torch.mean(nodes, 0), 0))
    pass

    
if __name__ == "__main__":
    #_testNormTime()
    testMeanVar()