import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import time

from dataNormalizer import GridNorm as Normalizer
from dataset import readJSON, sliceMD, SimpleStack
from models import DoubleGN as Model
from parameters import *

class Simulator(object):
    def __init__(self):
        self.stage1 = None
        self.stage2 = None

        self.nodeSplitter = None
        self.edgeSplitter = None

        self._getStages()
        self._getTrans()

    def _getStages(self):

        # stage 1
        norm1 = Normalizer.load(STAGE_ONE_NORM, mode="torch", device=DEVICE)
        model1 = Model(norm1, width=STAGE_ONE_MODEL_WIDTH, modelSize=STAGE_ONE_MODEL_SHAPE,
                       nodeIntVecLen=STAGE_ONE_NODE_INT_VEC_LEN, 
                       edgeIntVecLen=STAGE_ONE_EDGE_INT_VEC_LEN)
        model1.load(STAGE_ONE_MODEL)
        model1.eval()
        model1.to(DEVICE)
        self.stage1 = Stage(model1, norm1, SIM_INCREMENT)
        # stage 2
        norm2 = Normalizer.load(STAGE_TWO_NORM, mode="torch", device=DEVICE)
        model2 = Model(norm2, width=STAGE_TWO_MODEL_WIDTH, modelSize=STAGE_TWO_MODEL_SHAPE,
                       nodeIntVecLen=STAGE_TWO_NODE_INT_VEC_LEN,
                       edgeIntVecLen=STAGE_TWO_EDGE_INT_VEC_LEN)
        model2.load(STAGE_TWO_MODEL)
        model2.eval()
        model2.to(DEVICE)
        self.stage2 = Stage(model2, norm2, STAGE_TWO_INCREMENT)

    def _getTrans(self):

        info = readJSON(INFO_FILE)

        self.edgeSplitter = self._parseType(info[EDGE_NUMTYPE_TOKEN])
        self.nodeSplitter = self._parseType(info[NODE_NUMTYPE_TOKEN])

    def _parseType(self, types):
        
        geo, bc = [], []
        for i in range(len(types)):
            t = types[i]
            if(t == POINT_TOKEN or t == VECTOR_TOKEN):
                geo += [i]
            elif(t[0] == BC_TOKEN):
                bc += [i]
        gS, gE = min(geo), max(geo) + 1
        bcS, bcE = min(bc), max(bc) + 1

        return Splitter(gS, gE, bcS, bcE)

    def trainsit(self, graphs):

        graphs.nodes = self.nodeSplitter.strip(graphs.nodes)
        graphs.edges = self.edgeSplitter.strip(graphs.edges)

        return graphs

    def forward(self, graphs):
        graphs.dumpSeq()
        stage1Out = self.stage1.forward(graphs)
        stage2In = self.trainsit(stage1Out)
        stage2Out = self.stage2.forward(stage2In)
        stage2Out = stage1Out

        return stage2Out

class Splitter(object):
    def __init__(self, gS, gE, bcS, bcE):
        self.geoStart = gS
        self.geoEnd = gE
        self.bcStart = bcS
        self.bcEnd = bcE

    def strip(self, matrix):

        geo = sliceMD(matrix, self.geoStart, self.geoEnd)
        bc = sliceMD(matrix, self.bcStart, self.bcEnd)
        
        if(isinstance(matrix, np.ndarray)):
            cat = np.concatenate
        elif(isinstance(matrix, torch.Tensor)):
            cat = torch.cat

        out = cat((geo, bc), -1)

        return out

class Stage(object):
    def __init__(self, model, norm, iterations):
        self.model = model
        self.iterations = iterations
        self.norm = norm

    def forward(self, graphs):

        for _ in range(self.iterations):
            self.model.forward(graphs, norm=self.norm)
            graphs.addDelta(self.norm)
            graphs.dumpSeq()

        return graphs

def forwardSim(inputName, outputName):
    s0 = time.time()
    graphs = SimpleStack().loadJSONinit(INPUT_NAME)
    s1 = time.time()
    print("load input time: %.3f " % (s1 - s0))

    sim = Simulator()
    s2 = time.time()
    print("load model time: %.3f " % (s2 - s1))

    graphs = sim.forward(graphs)
    s3 = time.time()
    print("forward simulation time: %.3f " % (s3 - s2))

    graphs.saveJSONSeq(OUTPUT_NAME)
    s4 = time.time()
    print("save output time: %.3f " % (s4 - s3))

    print("done!")

if __name__ == "__main__":
    forwardSim(INPUT_NAME, OUTPUT_NAME)