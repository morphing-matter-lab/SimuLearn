import os
import numpy as np
import math as m
import random
import json
import multiprocessing as mp

from parameters import *
from inpParser import *
from inputGen import IDString
from dataset import GraphStack, MatchMaker, readJSON, writeJSON, readFile
import mirror

class GridParser(InpParser):
    def __init__(self, inp):
        self.modelStart = None
        self.modelEnd = None
        self.jointStart = None
        self.beamStart = None
        self.findModelInfoIndices(inp)
        super(GridParser, self).__init__(inp)

    def findModelInfoIndices(self, inp):
        i = 0

        for line in inp.splitlines():

            if(line == MODEL_START_TOKEN):
                self.modelStart = i
            elif(line == JOINT_START_TOKEN):
                self.jointStart = i
            elif(line == BEAM_START_TOKEN):
                self.beamStart = i
            elif(line == MODEL_END_TOKEN):
                self.modelEnd = i
                break
            i += 1

        assert(self.modelEnd > self.beamStart and \
               self.beamStart > self.jointStart and \
               self.jointStart > self.modelStart)

    def adjMat(self):

        jointCount = self.beamStart - self.jointStart - 2
        beamCount = self.modelEnd - self.beamStart - 2
        aMatrix = np.zeros((beamCount, jointCount), dtype=int)

        for line in self.inp[self.beamStart + 2:self.modelEnd]:
            items = line.split(',')
            edgeID, startID, endID = int(items[0]), int(items[1]), int(items[2])
            aMatrix[edgeID][startID] = 1
            aMatrix[edgeID][endID] = 1
        
        return aMatrix

    def sampFrames(self):
        
        result = {}
        for line in self.inp[self.beamStart + 2:self.modelEnd]:
            items = line.split(',')
            edgeID, actStartFrame, actEndFrame = int(items[0]), int(items[6]), int(items[7])
            result[edgeID] = [actStartFrame, actEndFrame]

        return result

    def actRatio(self):

        # get vertices
        vertices = {}
        for line in self.inp[self.jointStart + 2:self.beamStart]:
            if(line == BEAM_START_TOKEN): break
            items = line.split(',')
            vID = int(items[0])
            x, y, z = float(items[1]), float(items[2]), float(items[3])
            vertices[vID] = [x, y, z]
        
        ratio = {}
        # get edge lengths
        for line in self.inp[self.beamStart + 2:self.modelEnd]:
            items = line.split(',')
            eID, start, end, actLen = int(items[0]), int(items[1]), int(items[2]), float(items[5])
            startPt, endPt = vertices[start], vertices[end]

            dist = sum([((startPt[i] - endPt[i]) ** 2) \
                        for i in range(len(startPt))]) ** .5

            ratio[eID] = actLen / dist

        return ratio

class GridTrial(Trial):
    def __init__(self, inputFolder, nodeFileFolder, name):
        super(GridTrial, self).__init__(inputFolder, nodeFileFolder, name)
        # graph information
        self.blocks = {} # Dictionary, contains the blocks to extract
                         # information for
        self.fixedPt = np.zeros(3) # np array point, hard-coded
        self.fixedPtID = None # IDString, the fixed point index
        self.adjMat = None # 2D Matrix, (edge, node) binary
        self.actRatio = None # Map[beamID] = actuator ratio
        self.edgeSampFrames = None # 

        self.nodes = None # 3D Matrix (timestep, n_nodes, len_features)
        self.edges = None # 3D Matrix (timestep, n_edges, len_features)
        self.nodesTarget = None # 3D Matrix (timestep, n_nodes, len_features)
        self.edgesTarget = None # 3D Matrix (timestep, n_edges, len_features)

        self.edgeNumType = None
        self.nodeNumType = None
        self.edgeCen = None
        self.nodeCen = None

        self.edgeShape = "beam: "
        self.nodeShape = "joint: "

        self.mirrored = False
        self.inStage2 = False
        self.angles = self.sampleAngles(ROT_PER_TRIAL)

    def getBlocks(self):
        # get all blocks in this grid

        blocks = {}
        # nodes
        for nID in self.model.nMap:
            info = IDString.breakID(nID)
            blockType = info["blockType"]
            index = info["index"]

            if(blockType == BEAM_INDEX): # beam
                name = "beam_" + str(index)
                if(name not in blocks): blocks[name] = BeamSimple(blockType, index)
            elif(blockType == JOINT_INDEX): # joint
                name = "joint_" + str(index)
                if(name not in blocks): blocks[name] = JointSimple(blockType, index)

            blocks[name].expandNode(nID)

        # elements
        for eID in self.model.eMap:
            info = IDString.breakID(eID)
            blockType = info["blockType"]
            index = info["index"]

            if(blockType == BEAM_INDEX): # beam
                name = "beam_" + str(index)
                if(name not in blocks): blocks[name] = BeamSimple(blockType, index)
            elif(blockType == JOINT_INDEX): # joint
                name = "joint_" + str(index)
                if(name not in blocks): blocks[name] = JointSimple(blockType, index)

            blocks[name].expandElem(eID)
        
        # add beam extra information
        for blockName in blocks:
            block = blocks[blockName]
            if(block.type == BEAM_INDEX):
                block.insertFrames = self.edgeSampFrames[block.id]
                block.actRatio = self.actRatio[block.id]
        
        self.blocks = blocks

    def getBlockSamp(self):
        # get the sampling IDStrings of blocks

        for blockID in self.blocks:
            block = self.blocks[blockID]
            block.getSamp(self)

    def extract(self):

        beamFeatureLen, jointFeatureLen = 0, 0
        beamCount, jointCount = 0, 0

        # loop over blocks
        for blockID in self.blocks:
            block = self.blocks[blockID]
            
            # get length
            if(block.type == BEAM_INDEX):
                beamFeatureLen = len(block.numType)
                beamCount += 1
            elif(block.type == JOINT_INDEX):
                jointFeatureLen = len(block.numType)
                jointCount += 1

            # get initial condition
            frames = [block.samp(self)]
            # for each time step, extract information for this block
            for step in range(self.frames): frames += [block.samp(self, step)]
            # save input/output data to attributes
            block.sampFrames = np.array(frames)
        
        # construct input matrices
        stepCount = self.frames if self.inStage2 else self.frames
        nodes = np.zeros((stepCount, jointCount, jointFeatureLen))
        edges = np.zeros((stepCount, beamCount, beamFeatureLen))
        nodesTar = np.zeros((stepCount, jointCount, jointFeatureLen))
        edgesTar = np.zeros((stepCount, beamCount, beamFeatureLen))

        for step in range(stepCount):
            for blockID in self.blocks:
                block = self.blocks[blockID]

                curFrame = block.sampFrames[step]
                if(step < self.frames):
                    nextFrame = block.sampFrames[step + 1]
                else:
                    nextFrame = np.copy(curFrame)

                if(block.type == BEAM_INDEX):
                    edges[step][block.id] = curFrame
                    edgesTar[step][block.id] = nextFrame

                if(block.type == JOINT_INDEX):
                    nodes[step][block.id] = curFrame
                    nodesTar[step][block.id] = nextFrame
        
        self.nodes = nodes
        self.nodesTarget = nodesTar
        self.edges = edges
        self.edgesTarget = edgesTar

        self.getUtil()

    def getUtil(self):

        # get data utilities
        beamNumType, jointNumType = None, None
        beamCen, jointCen = None, None
        for blockID in self.blocks:
            blockType = self.blocks[blockID].type
            if(blockType == BEAM_INDEX):
                beamNumType = self.blocks[blockID].numType
                beamCen = self.blocks[blockID].center
            elif(blockType == JOINT_INDEX):
                jointNumType = self.blocks[blockID].numType
                jointCen = self.blocks[blockID].center
            if(beamCen != None and jointCen != None):
                break
        
        self.edgeNumType = beamNumType
        self.nodeNumType = jointNumType
        self.edgeCen = beamCen
        self.nodeCen = jointCen

    def extractAndWrite(self):
        # extract data
        self.extract()
        self.matchAnalysis() # update topological info into adj matrix
        self.writeTempData()

        # mirror
        if(USE_MIRROR):
            self.getMirrored()
            self.mirror() # to mirror
            # extract again
            self.extract()
            self.matchAnalysis()
            self.writeTempData()
            self.mirror() # to original

    def writeTempData(self):

        # save files
        data = []
        
        frames = self.frames if self.inStage2 else self.frames
        for step in range(frames):
            datum = {}
            datum[ADJ_MATRIX_TOKEN] = self.adjMat.tolist()
            datum[EDGE_NUMTYPE_TOKEN] = self.edgeNumType
            datum[NODE_NUMTYPE_TOKEN] = self.nodeNumType
            datum[NODE_CEN_TOKEN] = self.nodeCen
            datum[EDGE_CEN_TOKEN] = self.edgeCen
            datum[NODES_TOKEN] = self.nodes[step].tolist()
            datum[EDGES_TOKEN] = self.edges[step].tolist()
            datum[NODES_TAR_TOKEN] = self.nodesTarget[step].tolist()
            datum[EDGES_TAR_TOKEN] = self.edgesTarget[step].tolist()
            datum[ROT_ANGS_TOKEN] = self.angles
            data += [datum]
        
        # make file name
        if(self.inStage2): path = EXT_FOLDER + STAGE_TWO_FOLDER + '/'
        else: path = EXT_FOLDER + '/'
        name = self.name
        if(self.mirrored): name += MIRROR_TOKEN
        path += name + JSON_SUFFIX
        
        writeJSON(path, data)
    
    def sampleAngles(self, count):

        numbers = []
        cellAngle = m.pi * 2 / count
        for i in range(count):
            angle = cellAngle * (i + random.random())
            numbers += [angle]
        
        return numbers
    
    def countData(self):

        self.edgeShape += str(self.edges.shape) + ", "
        self.nodeShape += str(self.nodes.shape) + ", "
        
        msg = self.name + ", " + self.nodeShape + self.edgeShape

        return msg

    def matchAnalysis(self):

        analizer = MatchMaker()
        refNodes = self.nodes[0] # first time step (initial)
        refEdges = self.edges[0]
        newAdjMat = analizer.matchAnalysis(self.adjMat, refNodes, refEdges)
        self.adjMat = newAdjMat

    def getMirrored(self):

        mirror.getMirrorMap(self)
        # flip model
        mirror.mirrorGeo(self)
        mirror.mirrorStress(self)

    def mirror(self):

        self.mirrored = not self.mirrored

        # model
        # nodes
        m = self.model
        m.nMap, m.nMapMirrored = m.nMapMirrored, m.nMap
        m.nodes, m.nodesMirrored = m.nodesMirrored, m.nodes
        # elements
        m.eMap, m.eMapMirrored = m.eMapMirrored, m.eMap
        m.elemStress, m.elemStressMirrored = m.elemStressMirrored, m.elemStress

        # trial
        self.stress, self.stressMirrored = self.stressMirrored, self.stress
        self.coor, self.coorMirrored = self.coorMirrored, self.coor

        # always perfrom match analysis after mirroring
        self.matchAnalysis()

    def transStage2(self):

        # write last step in sequence into initial
        lastGeo = self.coor[-1,:,:]
        assert(self.model.nodes.shape == lastGeo.shape)
        self.model.nodes = lastGeo
        # read 2nd stage subsequent steps
        stage2coorPath = FIL_FOLDER + STAGE_TWO_FOLDER + '/' + self.name + \
                         '_' + STAGE_TWO_TOKEN + COOR_FILE_SUFFIX
        
        tarShape = (STAGE_TWO_INCREMENT, self.model.nCount, COOR_LEN)
        self.coor = Trial.loadCoor(stage2coorPath, tarShape)

        # update solver
        self.updateSolver(STAGE_TWO_INCREMENT, STAGE_TWO_FREQ)

        # drop stress in all blocks
        for blockID in self.blocks:
            block = self.blocks[blockID]
            block.numType = []
            block.geoSamp = []
            block.matSamp = []
            self.blocks[blockID].getSamp(self, sampStress=False)

        # dump shape
        self.edgeShape += str(self.edges.shape) + ", "
        self.nodeShape += str(self.nodes.shape) + ", "

        # set flag
        self.inStage2 = True

    @staticmethod
    def compileData(groups):
        # compile extracted files into a stack

        fileNames, index = groups

        graphs = []

        for i in range(len(fileNames)):
            fileName = fileNames[i]
            graph = GridTrial._permutate(fileName)
            
            if(USE_MIRROR):
                mirrorName = fileName.replace(               JSON_SUFFIX,
                                              MIRROR_TOKEN + JSON_SUFFIX)
                
                mirs = GridTrial._permutate(mirrorName)
                graph = GraphStack([graph, mirs])

            # put together

            graphs += [graph]
            
        graphStack = GraphStack(graphs)

        return graphStack, index

    @staticmethod
    def _permutate(fileName):

        data = readJSON(fileName)
        graph = GraphStack(data)
        # rotate trials
        rotAngles = data[0][ROT_ANGS_TOKEN]
        graph.rotateCopy(rotAngles)
        if(USE_DELTA): graph.calDelta()

        return graph

class BeamSimple(ElemBlockSimple):
    def __init__(self, blockType, index):

        name = str(blockType) + BEAM_TOKEN + '_' + str(index)
        self.numType = []
        self.insertFrames = []

        if(SAMP_CORNER):
            center = (0, COOR_LEN) # first 3 numbers
        else:
            if(SAMP_FIVE_FRAMES):
                center = (COOR_LEN * 6, COOR_LEN * 7) # third frame center
            else:
                center = (COOR_LEN * 3, COOR_LEN * 4) # seond frame center
        
        self.center = center
        self.cenID = None

        self.actRatio = None
        
        super(BeamSimple, self).__init__(blockType, index, name)
    
    def getSectionGeoSamp(self, lenSection):
        # takes a section frame along the beam and returns the center, x, y
        # sampling point IDString
        cenWid = int(m.floor(self.nWid / 2))
        cenHei = int(m.floor(self.nHeight / 2))
        ptCenID = IDString.getID(self.type, NODE_INDEX, self.id, 
                                 lenSection, cenWid, cenHei)
        ptXID = IDString.getID(self.type, NODE_INDEX, self.id,
                               lenSection, self.nWid - 1, cenHei)
        ptYID = IDString.getID(self.type, NODE_INDEX, self.id, 
                               lenSection, cenWid, self.nHeight - 1)
        
        sampID = [(ptCenID, None), (ptXID, ptCenID), (ptYID, ptCenID)]
        numType = [POINT_TOKEN] * COOR_LEN + [VECTOR_TOKEN] * (COOR_LEN * 2)
        # (vector end, vector start)
        return sampID, numType

    def getSectionGeoSamp_pt(self, lenSection):
        # takes a section frame along the beam and returns the center, x, y
        # sampling point IDString

        ptBot0 = IDString.getID(self.type, NODE_INDEX, self.id, 
                                 lenSection, 0, 0)
        ptBot1 = IDString.getID(self.type, NODE_INDEX, self.id, 
                                 lenSection, self.nWid - 1, 0)

        ptTop0 = IDString.getID(self.type, NODE_INDEX, self.id, 
                                 lenSection, 0, self.nHeight - 1)
        ptTop1 = IDString.getID(self.type, NODE_INDEX, self.id, 
                                 lenSection, self.nWid - 1, self.nHeight - 1)
        
        sampID = [(ptBot0, None), (ptBot1, None), (ptTop0, None), (ptTop1, None)]
        numType = [POINT_TOKEN] * (COOR_LEN * len(sampID))
        # (vector end, vector start)
        return sampID, numType

    def getGeoSamp(self):
        # get geometrical sampling indices

        # sample at 5 cross sections: both ends, center, and inserts
        insert0, insert1 = self.insertFrames
        nCenter = int(m.floor(self.nLen / 2))
        if(insert0 == insert1): # no actuator
            insert0 = int((nCenter + 0) / 2)
            insert1 = int((nCenter + self.nLen - 1) / 2)

        # list all frames
        head = [0]
        actStart = [insert0]
        middle = [nCenter]
        actEnd = [insert1]
        tail = [self.nLen - 1]
        
        if(SAMP_FIVE_FRAMES): samp = head + actStart + middle + actEnd + tail
        else: samp = head + middle + tail
        
        
        if(SAMP_CORNER):
            # add beam center point
            cenWid = int(m.floor(self.nWid / 2))
            cenHei = int(m.floor(self.nHeight / 2))
            ptCenID = IDString.getID(self.type, NODE_INDEX, self.id, 
                                     nCenter, cenWid, cenHei)
            self.geoSamp += [(ptCenID, None)]
            self.numType += [POINT_TOKEN] * COOR_LEN

        for nLen in samp:
            if(SAMP_CORNER):
                sampIDPairs, numType = self.getSectionGeoSamp_pt(nLen)
            else:
                sampIDPairs, numType = self.getSectionGeoSamp(nLen)

            self.geoSamp += sampIDPairs
            self.numType += numType
        
    def getSectionMatSamp(self, lenSection, side):
        
        # get offsets
        offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
        if(side == -1): elemLenIndex = lenSection - 1
        else: elemLenIndex = lenSection
        
        GPPairs, numType = [], [] # Gaussian point pairs
        for offset in offsets:
            # width
            if(offset[0] == 0): eWid, nWid = 1, 1 # inner offset to avoid shell
            else: eWid, nWid = self.eWid - 2, self.nWid - 2 # likewise
            # height
            if(offset[1] == 0): eHei, nHei = 0, 0
            else: eHei, nHei = self.eHeight - 1, self.nHeight - 1
            
            elemID = IDString.getID(self.type, ELEM_INDEX, self.id, elemLenIndex, 
                                    eWid, eHei)
            vertexID = IDString.getID(self.type, NODE_INDEX, self.id, lenSection,
                                      nWid, nHei)
            
            GPPairs += [(elemID, vertexID)]
            numType += [STRESS_TOKEN] * STRESS_LEN
        
        return GPPairs, numType
    
    def getMatSamp(self):
        # get geometrical sampling indices

        # sample at 5 cross sections: both ends, center, and inserts
        insert0, insert1 = self.insertFrames
        nCenter = int(m.floor(self.nLen / 2))
        if(insert0 == insert1): # no actuator
            insert0 = int((nCenter + 0) / 2)
            insert1 = int((nCenter + self.nLen - 1) / 2)
        
        # list all frames
        head = [(0, 1)]
        actStart = [(insert0, -1), (insert0, 1)]
        middle = [(nCenter, -1), (nCenter, 1)]
        actEnd = [(insert1, -1), (insert1, 1)]
        tail = [(self.nLen - 1, -1)]

        if(SAMP_FIVE_FRAMES): samp = head + actStart + middle + actEnd + tail
        else: samp = head + middle + tail
        
        for nLen, side in samp:
            sampIDPairs, numType = self.getSectionMatSamp(nLen, side)
            self.matSamp += sampIDPairs
            self.numType += numType

    def getBCSamp(self):

        cenLen = int(m.floor(self.nLen / 2))
        cenWid = int(m.floor(self.nWid / 2))
        cenHei = int(m.floor(self.nHeight / 2))
        ptCenID = IDString.getID(self.type, NODE_INDEX, self.id, 
                                 cenLen, cenWid, cenHei)

        self.cenID = ptCenID

        self.numType += [BC_TOKEN] + [BC_TOKEN + VECTOR_TOKEN] * COOR_LEN

    def getSamp(self, trial, sampStress=SAMP_STRESS):
        # get sampling information for itself

        self.getGeoSamp() # [(NodeIDString, NodeIDString)], [(target, anchor)]
        if(sampStress):
            self.getMatSamp() # [(ElemIDString, NodeIDString)], [(voxel, vertex)]
        self.getBCSamp()

    def sampGeo(self, trial, step): # init

        # convert IDStrings to model indices
        geoIDs = [(trial.model.nMap.get(tarID, None), 
                   trial.model.nMap.get(srcID, None)) \
                  for tarID, srcID in self.geoSamp]
        
        # find step target
        if(step == None): tarMat = trial.model.nodes # initial
        else: tarMat = trial.coor[step] # subsequent
        # load into frame
        frame = []
        for tarID, srcID in geoIDs:
            if(srcID != None): # vector
                frame += list(tarMat[tarID] - tarMat[srcID])
            else: # point
                frame += list(tarMat[tarID])
        
        return frame
        
    def sampMat(self, trial, step): # init

        # convert IDStrings to model indices
        matIDs = [(trial.model.eMap[elemID], trial.model.nMap[vertexID]) \
                  for elemID, vertexID in self.matSamp]
        # find step target
        frame = []
        if(step == None): # initial
            for elemID, _ in matIDs:
                frame += list(trial.model.elemStress[elemID])
        else: # subsequent
            for elemID, vertexID in matIDs: # extract material
                elemVertices = trial.model.elems[elemID].tolist()
                gpID = GP_MAP[elemVertices.index(vertexID)]
                frame += list(trial.stress[step][elemID][gpID])

        return frame
        
    def sampBC(self, trial, step):

        frame = []

        # beam active/passive indicator
        if(abs(self.actRatio) < EPSILON): ratio = 0
        else: ratio = self.actRatio
        
        frame += [ratio]
        
        # offset
        nID = trial.model.nMap[self.cenID]
        if(step == None): # initial
            frame += list(trial.fixedPt - trial.model.nodes[nID])
        else: # subsequent
            frame += list(trial.fixedPt - trial.coor[step][nID])
        
        return frame

    def samp(self, trial, step=None):

        geoFrame = self.sampGeo(trial, step)
        matFrame = self.sampMat(trial, step)
        bcFrame = self.sampBC(trial, step)

        frame = geoFrame + matFrame + bcFrame

        return frame

class JointSimple(ElemBlockSimple):
    def __init__(self, blockType, index):
        name = str(blockType) + JOINT_TOKEN + '_' + str(index)
        self.numType = []
        self.cenID = None
        self.center = (0, COOR_LEN)
        self.isFixed = False
        super(JointSimple, self).__init__(blockType, index, name)

    def getGeoSamp(self):

        cenLen = int(m.floor(self.nLen / 2))
        cenWid = int(m.floor(self.nWid / 2))
        cenHei = int(m.floor(self.nHeight / 2))

        ptCenID = IDString.getID(self.type, NODE_INDEX, self.id, 
                                 cenLen, cenWid, cenHei)

        pt00 = IDString.getID(self.type, NODE_INDEX, self.id, 
                              0, 0, cenHei)
        pt10 = IDString.getID(self.type, NODE_INDEX, self.id,
                              self.nLen - 1, 0, cenHei)
        pt11 = IDString.getID(self.type, NODE_INDEX, self.id,
                              self.nLen - 1, self.nWid -1, cenHei)
        pt01 = IDString.getID(self.type, NODE_INDEX, self.id,
                              0, self.nWid - 1, cenHei)
        ptTop = IDString.getID(self.type, NODE_INDEX, self.id, 
                               cenLen, cenWid, self.nHeight - 1)

        pairs = [(ptCenID, None), (pt00, ptCenID), (pt10, ptCenID), 
                 (pt11, ptCenID), (pt01, ptCenID), (ptTop, ptCenID)]
        
        self.numType += [POINT_TOKEN] * COOR_LEN + [VECTOR_TOKEN] * (COOR_LEN * 5)
        self.geoSamp = pairs

    def getGeoSamp_pt(self):
    
        if(SAMP_CORNER):
            # add beam center point
            cenLen = int(m.floor(self.nLen / 2))
            cenWid = int(m.floor(self.nWid / 2))
            cenHei = int(m.floor(self.nHeight / 2))

            ptCenID = IDString.getID(self.type, NODE_INDEX, self.id, 
                                    cenLen, cenWid, cenHei)
            self.geoSamp += [(ptCenID, None)]
            self.numType += [POINT_TOKEN] * COOR_LEN


        for k in range(2): # height
            for i in range(2): # length
                for j in range(2): # width

                    vertexID = IDString.getID(self.type, NODE_INDEX, self.id,
                                                i * (self.nLen - 1), 
                                                j * (self.nWid - 1), 
                                                k * (self.nHeight - 1))

                    self.numType += [POINT_TOKEN] * COOR_LEN
                    self.geoSamp += [(vertexID, None)]
    
    def getMatSamp(self):
        # to implement
        for k in range(2): # height
            for i in range(2): # length
                for j in range(2): # width

                    elemID = IDString.getID(self.type, ELEM_INDEX, self.id,
                                            i * (self.eLen - 1), 
                                            j * (self.eWid - 1), 
                                            k * (self.eHeight - 1))
                    
                    vertexID = IDString.getID(self.type, NODE_INDEX, self.id,
                                              i * (self.nLen - 1), 
                                              j * (self.nWid - 1), 
                                              k * (self.nHeight - 1))

                    self.numType += [STRESS_TOKEN] * STRESS_LEN
                    self.matSamp += [(elemID, vertexID)]

    def getBCSamp(self, trial):

        cenLen = int(m.floor(self.nLen / 2))
        cenWid = int(m.floor(self.nWid / 2))
        cenHei = int(m.floor(self.nHeight / 2))
        ptCenID = IDString.getID(self.type, NODE_INDEX, self.id, 
                                 cenLen, cenWid, cenHei)

        self.cenID = ptCenID
        cenPt = trial.model.nodes[trial.model.nMap[ptCenID]]

        self.isFixed = trial.fixedPt[0] - cenPt[0] < EPSILON and \
                       trial.fixedPt[1] - cenPt[1] < EPSILON and \
                       trial.fixedPt[2] - cenPt[2] < EPSILON

        self.numType += [BC_TOKEN] + [BC_TOKEN + VECTOR_TOKEN] * COOR_LEN

    def getSamp(self, trial, sampStress=SAMP_STRESS):
        if(SAMP_CORNER):
            self.getGeoSamp_pt() # [(NodeIDString, NodeIDString)], [(target, anchor)]
        else:
            self.getGeoSamp()
        if(sampStress and JOINT_SAMP_STRESS):
            self.getMatSamp() # [(ElemIDString, NodeIDString)], [(voxel, vertex)]
        self.getBCSamp(trial)
        
    def sampGeo(self, trial, step): # init

        # convert IDStrings to model indices
        geoIDs = [(trial.model.nMap.get(tarID, None), 
                   trial.model.nMap.get(srcID, None)) \
                  for tarID, srcID in self.geoSamp]
        
        # find step target
        if(step == None): tarMat = trial.model.nodes # initial
        else: tarMat = trial.coor[step] # increment
        # load into frame
        frame = []
        for tarID, srcID in geoIDs:
            if(srcID != None): # vector
                frame += list(tarMat[tarID] - tarMat[srcID])
            else: # point
                frame += list(tarMat[tarID])
        
        return frame
        
    def sampMat(self, trial, step): # init

        # convert IDStrings to model indices
        matIDs = [(trial.model.eMap[elemID], trial.model.nMap[vertexID]) \
                  for elemID, vertexID in self.matSamp]
        # find step target
        frame = []
        if(step == None): # initial
            for elemID, _ in matIDs:
                frame += list(trial.model.elemStress[elemID])
        else: # increment
            for elemID, vertexID in matIDs: # extract material
                elemVertices = trial.model.elems[elemID].tolist()
                gpID = GP_MAP[elemVertices.index(vertexID)]
                frame += list(trial.stress[step][elemID][gpID])

        return frame
        
    def sampBC(self, trial, step):

        frame = []

        # beam active/passive indicator
        if(self.isFixed): frame += [1]
        else: frame += [0]
        
        # offset
        nID = trial.model.nMap[self.cenID]
        if(step == None):
            frame += list(trial.fixedPt - trial.model.nodes[nID])
        else:
            frame += list(trial.fixedPt - trial.coor[step][nID])
        return frame

    def samp(self, trial, step=None):

        geoFrame = self.sampGeo(trial, step)
        matFrame = self.sampMat(trial, step)
        bcFrame = self.sampBC(trial, step)

        frame = geoFrame + matFrame + bcFrame

        return frame

def listFiles(inp, out):

    # read .inp and .fil file names
    inpNames = os.listdir(inp)
    resultFileNames = os.listdir(out)
    if(TWO_STAGE): 
        s2resultFiles = os.listdir(out + STAGE_TWO_FOLDER)
    
    # check valid .inp and .fil pairs
    validTrials = []
    for fileName in inpNames:
        if (not fileName.endswith(INP_SUFFIX)): continue
        # strip name from .inp file
        name = fileName[:fileName.find(INP_SUFFIX)]
        # check if coor and stress file exist
        stressFileName = name + STRESS_FILE_SUFFIX
        coorFileName = name + COOR_FILE_SUFFIX
        coorExist = coorFileName in resultFileNames
        stressExist = stressFileName in resultFileNames

        # check stage 1 output
        valid = False
        if(SAMP_STRESS and coorExist and stressExist): valid = True
        elif(not SAMP_STRESS and coorExist): valid = True
        # check stage 2 output
        if(TWO_STAGE):
            coorFileNameS2 = name + '_' + STAGE_TWO_TOKEN + COOR_FILE_SUFFIX
            if(coorFileNameS2 not in s2resultFiles):
                valid = False
        
        if(valid):
            validTrials += [name]

    return validTrials

def extractFile(name):

    timeStart = time.time()
    try:
        # construct trial
        trial = GridTrial(INP_FOLDER, FIL_FOLDER, name) # initialize
        parser = GridParser(readFile(trial.inpPath)) # parser
        trial.model = parser.parseAll() # model
        trial.adjMat = parser.adjMat() # adj matrix
        trial.actRatio = parser.actRatio() # actuator ratio
        trial.edgeSampFrames = parser.sampFrames() # sampling frames
        inc, freq = parser.parseSolver() # get solver status

        # get sampling information
        trial.updateSolver(inc, freq) # solver
        trial.getBlocks() # generate blocks to extract for
        trial.getBlockSamp() # get extraction IDs
        trial.loadResults() # load FEA results

        # extract data
        trial.extractAndWrite()
        if(TWO_STAGE):
            trial.transStage2()
            trial.extractAndWrite()

        msg = trial.countData()
        success = 1
    
    except:
        msg = "file extraction failed: %s, " % name
        success = 0
    
    timeEnd = time.time()
    timeElapsed = timeEnd - timeStart
    
    return msg, timeElapsed, success

def makeFolders(path):
    # checks and creates the folder required by the SimuLearn program

    # required folders
    tarFolders = [INP_FOLDER, FIL_FOLDER, EXT_FOLDER, DATA_FOLDER, # data
                  MODEL_FOLDER, REF_OUT_FOLDER] # ML

    # get folders in workspace
    print('=' * 80)
    print("checking folders...")
    curFolders = []
    for folder in os.listdir("./"):
        if(os.path.isdir(folder)):
            curFolders += [folder]
    
    # check if required folders exist in workspace and create
    for folder in tarFolders:
        name = folder[2:]
        if(name not in curFolders):
            os.mkdir(folder)
            print("folder created: %s" % name)
        if(TWO_STAGE and STAGE_TWO_FOLDER[1:] not in os.listdir(folder)):
            nameStage2 = folder + STAGE_TWO_FOLDER
            os.mkdir(nameStage2)
            print("folder created: %s" % nameStage2)

    print("done!")

def extraction(validTrials):

    print('=' * 80)
    print("extraction start, processing %d files" % len(validTrials))
    print("name, beam data, joint data (step, count, len)...(progress)...time")
    
    finished, succeeded  = 0, 0
    if(PARALLEL):
        with mp.Pool(processes=WORKERS) as pool:
            for message, timeElapsed, success in \
                    pool.imap_unordered(extractFile, validTrials):

                finished += 1
                succeeded += success
                perc = (finished / len(validTrials)) * 100
                progress = "%.2f" % perc + chr(37) # % character
                print(message + "(%s)...%.2f" % (progress, timeElapsed))
    else:
        for name in validTrials:
            message, timeElapsed, success = extractFile(name)
            finished += 1
            perc = (finished / len(validTrials)) * 100
            succeeded += success
            progress = "%.2f" % perc + chr(37) # % character
            print(message + ", (%s)...%.2f" % (progress, timeElapsed))
    
    print("extraction finished! Overall success rate: %.3f" % (succeeded / finished))

def compilation(fileNames, tarFolder):

    print('='* 80)
    print("compilation start, processing %d files" % len(fileNames))
    random.shuffle(fileNames) # shuffle to elimnate order bias
    orderFileName = tarFolder + '/' + ORDER_TOKEN + JSON_SUFFIX
    writeJSON(orderFileName, fileNames)

    if(PARALLEL):
        # split into groups
        groups = []
        groupSize = int(len(fileNames) / WORKERS)
        for i in range(WORKERS):
            if(i < WORKERS - 1):
                new = fileNames[i * groupSize:(i + 1) * groupSize]
            else:
                new = fileNames[i * groupSize:] # last group
            groups += [(new, i)]
        
        # parallel stack
        stacks = {}
        with mp.Pool(processes=WORKERS) as pool:
            for stack, i in pool.imap_unordered(GridTrial.compileData, groups):
                stacks[i] = stack
        stacks = [stacks[i] for i in range(WORKERS)]
        compiled = GraphStack(stacks)
    
    else:
        # single stack
        stacks = GridTrial.compileData((fileNames, 0))
        compiled = GraphStack(stacks)
    
    print("writing file...")
    compiled.save(tarFolder)
    # compile files
    print("compilation finished!")

def listFolder(folder, suffix):

    fileNames = []
    for name in os.listdir(folder):
        isFile = name.endswith(suffix)
        notMirror = not name.endswith(MIRROR_TOKEN + suffix)
        fullName = folder + '/' + name
        if(isFile and notMirror): fileNames += [fullName]

    if(CUT_DATA == -1):
        fileNames = fileNames
    else:
        fileNames = fileNames[:CUT_DATA]

    return fileNames

def extractBatch():
    timeStart = time.time()

    if(EXTRACT):
        validTrials = listFiles(INP_FOLDER, FIL_FOLDER) # check for files
        if(len(validTrials) != 0):
            extraction(validTrials)
    
    if(COMPILE):
        fileNames = listFolder(EXT_FOLDER, JSON_SUFFIX)
        if(len(fileNames) != 0):
            print("stage 1 compilation")
            compilation(fileNames, DATA_FOLDER)
        
        if(TWO_STAGE):
            print("stage 2 compilation")
            fileNames = listFolder(EXT_FOLDER + STAGE_TWO_FOLDER, JSON_SUFFIX)
            compilation(fileNames, DATA_FOLDER + STAGE_TWO_FOLDER)
    
    timeEnd = time.time()
    print("execution time: %.3f seconds" % (timeEnd - timeStart))

if __name__ == "__main__":
    makeFolders("./")
    extractBatch()