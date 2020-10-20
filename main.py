import numpy as np
import torch
import copy
from torch.utils.data import Dataset, DataLoader
import json
import time

from parameters import *
from dataset import GraphStack, SimpleStack, MatchMaker, NoiseAgent, writeTxt, writeJSON
from dataNormalizer import GridNorm as Normalizer
from models import DoubleGN as Model
from dataParser import makeFolders

# ================== calsses ====================

class CustomLoader(object):
    def __init__(self, data, batchSize, flat=True, noise=NOISE):

        if(flat): data.toSingleGraph()

        self.data = data
        self.count = len(data)
        self.batchSize = batchSize
        self.keys = None
        self.noise = noise

        if(noise != 0):
            self.noiseAgent = NoiseAgent()

    def shuffle(self):

        self.keys = torch.randperm(self.count)

    def load(self):

        self.shuffle()
        curIndex = 0
        i = 0
        while (curIndex < self.count):
            
            nexIndex = curIndex + self.batchSize
            curKeys = self.keys[curIndex:nexIndex]

            adjMats = self.data.adjMat[curKeys].to(DEVICE)
            nodes = self.data.nodes[curKeys].to(DEVICE)
            edges = self.data.edges[curKeys].to(DEVICE)
            nodesTar = self.data.nodesTar[curKeys].to(DEVICE)
            edgesTar = self.data.edgesTar[curKeys].to(DEVICE)
            
            if(self.noise != 0):
                nodeNoiseScale = self.data.nodesCumSumDelta[curKeys].to(DEVICE)
                edgeNoiseScale = self.data.edgesCumSumDelta[curKeys].to(DEVICE)
                nodes, edges = self._addNoise(nodes, edges, 
                                              nodeNoiseScale, edgeNoiseScale, 
                                              self.noise)

            graphs = SimpleStack(adjMats, nodes, edges, nodesTar, edgesTar)

            curIndex = nexIndex
            i += adjMats.shape[0]

            yield i, graphs

            del adjMats, nodes, edges, nodesTar, edgesTar
            del graphs

    def _addNoise(self, nodes, edges, nodeNoiseScale, edgeNoiseScale, noise):

        if(NOISE == 0): return nodes, edges # bypass

        nodeNoise, edgeNoise = self.noiseAgent.genNoise(nodeNoiseScale,
                                                        edgeNoiseScale, 
                                                        noise)

        nodesNew = nodes + nodeNoise
        edgesNew = edges + edgeNoise

        return nodesNew, edgesNew

class MLDataset(Dataset):
    def __init__(self, data, noise=0):
        self.data = data # the dataset it holds
        self.noise = noise
        self.noiseAgent = NoiseAgent()

        if(NOISE != 0):
            nodesTar = data.getNodesTar()
            edgesTar = data.getEdgesTar()
            self.noiseAgent = NoiseAgent(nodesTar, edgesTar)
        
    def __getitem__(self, item):
        # returns:
        #   adj_mat: adjacency matrix, shape: (n_batch, n_edges, n_nodes)
        #   nodes: nodes matrix, shape: (n_batch, n_nodes, nodeLen)
        #   edges: edges matrix, shape: (n_batch, n_nodes, nodeLen)
        #   nodes_target: nodes target matrix, shape: (n_batch, n_nodes, nodeLen)
        #   edges_target: edges target matrix, shape: (n_batch, n_edges, edgeLen)

        adjMat = self.data.adjMat[item].to(DEVICE)
        nodes = self.data.nodes[item].to(DEVICE)
        edges = self.data.edges[item].to(DEVICE)
        nodesTar = self.data.nodesTar[item].to(DEVICE)
        edgesTar = self.data.edgesTar[item].to(DEVICE)
        
        if(self.noise != 0):
            nodes, edges = self._addNoise(nodes, edges)

        #graphs = SimpleStack(adjMat, nodes, edges, nodesTar, edgesTar)

        return adjMat, nodes, edges, nodesTar, edgesTar

    def _addNoise(self, nodes, edges):

        if(NOISE == 0): return nodes, edges # bypass

        nodeNoise, edgeNoise = self.noiseAgent.genNoise(nodes, edges, self.noise)

        nodesNew = nodes + nodeNoise
        edgesNew = edges + edgeNoise

        return nodesNew, edgesNew
    
    def __len__(self):
        return self.data.count

class Environment(object):
    def __init__(self, trainLoader=None, testLoader=None, iterLoader=None, model=None, 
                 lossFunc=None, optimizer=None, norm=None, epochs=MAX_EPOCH,
                 modelFolder=None, batchSize =BATCH_SIZE, trainCount=None,
                 testCount=None, stage=1, increments=SIM_INCREMENT, noise=NOISE):
        
        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.iterLoader = iterLoader
        self.model = model
        self.norm = norm
        self.noise = noise

        self.trainRef = None
        self.testRef = None
        self.stage = stage
        self.simIter = increments

        self.optimizer = optimizer
        self.lossFunc = lossFunc
        self.loss = None

        self.lossLog = LossLog()

        self.epochs = epochs
        self.modelFolder = modelFolder

        self.batchSize = batchSize
        self.trainCount = trainCount
        self.testCount = testCount

        # batch-wise variables
        self.lossVal = 0
        self.numCount = 0
        self.graphCount = 0
        self.ptDist = 0
        self.ptCount = 0

        self.nodeNoiseLevel = 0
        self.edgeNoiseLevel = 0

        self.iterResult = []

        if(self.trainLoader):
            trainRef = self.getRefFromLoader(self.trainLoader)
            self.trainRef = RefTest(trainRef, self, stage=self.stage, mode="train")
        if(self.testLoader):
            testRef = self.getRefFromLoader(self.testLoader)
            self.testRef = RefTest(testRef, self, stage=self.stage, mode="test")

        if(CAL_DISLOCATION or PROP_DISLOCATION):
            self.mm = MatchMaker()
            self.ptLoss = None
    
    def batchReset(self):

        del self.loss
        self.loss = None
        if(CAL_DISLOCATION or PROP_DISLOCATION):
            del self.ptLoss
            self.ptLoss = None
        
        self.lossVal = 0
        self.lossPerNum = 0
        self.graphCount = 0
        self.ptDist = 0
        self.ptCount = 0

        lossLog = self.lossLog._genEpochReport()

        return lossLog

    def getRefFromLoader(self, loader):

        adjMat = loader.data.adjMat[0].clone()
        nodes = loader.data.nodes[0].clone()
        edges = loader.data.edges[0].clone()
        nodesDelta = loader.data.nodesTar[:SIM_INCREMENT].clone()
        edgesDelta = loader.data.edgesTar[:SIM_INCREMENT].clone()
        
        # fix target output
        steps, nodesCount, nodeLen = nodesDelta.shape
        _    , edgesCount, edgeLen = edgesDelta.shape
        nodesTar = torch.zeros((steps + 1, nodesCount, nodeLen))
        edgesTar = torch.zeros((steps + 1, edgesCount, edgeLen))
        nodesLast, edgesLast = nodes.clone(), edges.clone()
        nodesTar[0] = nodesLast
        edgesTar[0] = edgesLast
        for i in range(steps):
            nodesDeltaDev = nodesDelta[i].to(DEVICE)
            edgesDeltaDev = edgesDelta[i].to(DEVICE)
            nodesDeltaInvert = self.norm.invertNodeOut(nodesDeltaDev)
            edgesDeltaInvert = self.norm.invertEdgeOut(edgesDeltaDev)
            nodesTar[i + 1] = nodesLast + nodesDeltaInvert.cpu()
            edgesTar[i + 1] = edgesLast + edgesDeltaInvert.cpu()
            nodesLast = nodesTar[i + 1]
            edgesLast = edgesTar[i + 1]

        initGraph = SimpleStack(adjMat=adjMat, nodes=nodes, edges=edges, 
                                nodesTar=nodesTar, edgesTar=edgesTar)

        return initGraph

    def epochReset(self, epoch=0):

        self.trainRef.evaluate(epoch)
        self.testRef.evaluate(epoch)
        lossLog = self.lossLog.epochClean()
        
        return lossLog

    def updateLoss(self, mode):

        if(mode == TRAIN_TOKEN):
            self.lossLog.trainLoss += self.lossVal
            self.lossLog.trainGraphCount += self.graphCount
            self.lossLog.trainNumCount += self.numCount
            self.lossLog.trainDisLoc += self.ptDist
            self.lossLog.trainPointCount += self.ptCount
        elif(mode == TEST_TOKEN):
            self.lossLog.testLoss += self.lossVal
            self.lossLog.testGraphCount += self.graphCount
            self.lossLog.testNumCount += self.numCount
            self.lossLog.testDisLoc += self.ptDist
            self.lossLog.testPointCount += self.ptCount
        else:
            assert(False)

    def iterTest(self):

        self.model.eval()
        torch.no_grad()

        lossVal = [0] * SIM_INCREMENT
        ptDist = [0] * SIM_INCREMENT
        ptCount = [0] * SIM_INCREMENT
        graphCount = [0] * SIM_INCREMENT
        
        for batch, graphs in self.iterLoader.load():
            # iteratively simulate
            nodesTarAll, edgesTarAll = graphs.nodesTar, graphs.edgesTar
            
            for i in range(self.simIter):
                nodesTar, edgesTar = nodesTarAll[:,i,:,:], edgesTarAll[:,i,:,:]
                graphs.nodesTar, graphs.edgesTar = nodesTar, edgesTar

                self.model.forward(graphs, norm=self.norm)
                
                self.model.calLoss(graphs, self)

                lossVal[i] += self.lossVal
                ptDist[i] += self.ptDist
                ptCount[i] += self.ptCount
                graphCount[i] += self.graphCount

                graphs.addDelta(self.norm)
                
                self.batchReset()
        
            lossPerGraph = [round(lossVal[i] / graphCount[i], 2) for i in range(self.simIter)]
            avgDisLoc = [round(ptDist[i] / ptCount[i], 2) for i in range(self.simIter)]
            if(batch % LOG_OUT_INTERVAL == 0):
                print("lossPerGraph:", lossPerGraph, ", avgDisLoc:", avgDisLoc, end="\r")
            
        result = {}
        result["lossPerGraph"] = lossPerGraph
        result["avgDisLoc"] = avgDisLoc
        self.iterResult += [result]
        print("lossPerGraph:", lossPerGraph, ", avgDisLoc:", avgDisLoc)
        
        writeJSON("./iterTest.json", self.iterResult)

        self.model.train()
        torch.enable_grad()

class LossLog(object):
    def __init__(self):

        # log for a single epoch
        # train
        self.trainLoss = 0 # Float, sum
        self.trainGraphCount = 0 # Int, sum
        self.trainNumCount = 0 # Int, sum
        self.trainDisLoc = 0 # Float, sum
        self.trainPointCount = 0 # Int, sum

        # test
        self.testLoss = 0 # Float, sum
        self.testGraphCount = 0 # Int, sum
        self.testNumCount = 0 # Int, sum
        self.testDisLoc = 0 # Float, sum
        self.testPointCount = 0 # Int, sum

        # iter
        self.iterLoss = 0 # Float, sum
        self.iterGraphCount = 0 # Int, sum
        self.iterNumCount = 0 # Int, sum
        self.iterDisLoc = 0 # Float, sum
        self.iterPointCount = 0 # Int, sum

        # log for epochs
        self.LossLog = [] # string

    def epochClean(self):

        lossLog = self._genEpochReport()
        self.LossLog += [lossLog]

        # reset epoch logs
        # train
        self.trainLoss = 0 # Float, sum
        self.trainGraphCount = 0 # Int, sum
        self.trainNumCount = 0 # Int, sum
        self.trainDisLoc = 0 # Float, sum
        self.trainPointCount = 0 # Int, sum

        # test
        self.testLoss = 0 # Float, sum
        self.testGraphCount = 0 # Int, sum
        self.testNumCount = 0 # Int, sum
        self.testDisLoc = 0 # Float, sum
        self.testPointCount = 0 # Int, sum

        return lossLog

    def _genEpochReport(self):

        calPt = CAL_DISLOCATION or PROP_DISLOCATION
        
        # compute loss
        # train
        graphCount = self.trainGraphCount if self.trainGraphCount != 0 else 1
        numCount = self.trainNumCount if self.trainNumCount != 0 else 1

        trainPerGraph = self.trainLoss / graphCount
        trainPerNum = self.trainLoss / numCount
        trainDisLoc = -1
        testPerGraph = -1
        testPerNum = -1
        testDisLoc = -1

        if(VALIDATION):
            graphCount = self.testGraphCount if self.testGraphCount != 0 else 1
            numCount = self.testNumCount if self.testNumCount != 0 else 1

            testPerGraph = self.testLoss / graphCount
            testPerNum = self.testLoss / numCount
            if(calPt):
                trainPtCount = self.trainPointCount if self.trainPointCount != 0 else 1
                testPtCount = self.testPointCount if self.testPointCount != 0 else 1
                trainDisLoc = self.trainDisLoc / trainPtCount
                testDisLoc = self.testDisLoc / testPtCount
        else:
            if(calPt):
                trainPtCount = self.trainPointCount if self.trainPointCount != 0 else 1
                trainDisLoc = self.trainDisLoc / trainPtCount

        trainMsg = "train loss: Graph: %.3f, Num: %.4f, DisLoc: %.3f" % \
                   (trainPerGraph, trainPerNum, trainDisLoc)
    
        testMsg = "test loss: Graph: %.3f, Num: %.4f, DisLoc: %.3f" % \
                   (testPerGraph, testPerNum, testDisLoc)

        lossLog = trainMsg + " | " + testMsg

        return lossLog

class RefTest(object):
    def __init__(self, initGraph, env, stage=1, mode="train"):

        self.graphs = initGraph
        self.stage = stage
        self.mode = mode

        self.model = env.model
        self.iteration = env.simIter
        self.norm = env.norm

    def evaluate(self, epoch, output=True):
        
        self.model.eval()
        torch.no_grad()

        graphs = self.graphs.copy().to(DEVICE)
        graphs.dumpSeq() # init

        for _ in range(self.iteration):
            self.model.forward(graphs, norm=self.norm)
            graphs.addDelta(self.norm)
            graphs.dumpSeq()

        if(output):
            # get save path
            name = "%s_epoch_%d" % (self.mode, epoch) + JSON_SUFFIX
            path = REF_OUT_FOLDER + '/'
            if(self.stage == 2): path += STAGE_TWO_FOLDER + '/'
            fullName = path + name

            graphs.saveJSONSeq(fullName)

        self.model.train()
        torch.enable_grad()

# ========================= functions ==============================

def getData(path, fold=(0, 5), trainPerc=TRAIN_RATIO):

    # load and split
    allData = GraphStack().load(path)
    trainData, testData = allData.splitData(fold=fold, perc=TRAIN_RATIO)
    if(CUT_LAST):
        trainData.cutLast()
        testData.cutLast()

    iterLoader = testData._copy()
    iterLoader.iterMode()

    # convert to single graph mode
    trainData.getCumSumDelta()
    testData.getCumSumDelta()
    
    # cast data to cpu
    trainData.toTorch(device=DEVICE_CPU)
    testData.toTorch(device=DEVICE_CPU)
    iterLoader.toTorch(device=DEVICE_CPU)

    return trainData, testData, iterLoader

def getNormalizer(makeNew, data=None, mode="torch", saveFolder=MODEL_FOLDER,
                  device=DEVICE):
    # get the normalizer
    # parameters:
    #   read: bool, indicating if the program should load from a previous save
    #   data: a graph dataset as a list of dictionaries
    # return:
    #   a Grid2x2Normalizer instance as the normalizer to use
    
    savePath = saveFolder + '/' + NORMALIZER_FILENAME

    if(makeNew):
        # craete and save normalizer
        norm = Normalizer(data, mode=data.type, device=DEVICE_CPU)
        norm.save(savePath)
        norm.toDevice(device)
    else:
        # load from file
        norm = Normalizer.load(savePath, mode=mode, device=device)

    return norm

def getModelSavePath(modelFolder, epoch):
    # returns the filepath of the given epoch save file
    # parameter:
    #   epoch: int, the epoch index to format into the path
    #   id: int, the number of the GN block
    # reutrn:
    #   a file path based on the input epoch number

    path = modelFolder + "/model_%d.pth" % epoch

    return path

def getModel(norm, modelNum=-1, width=1, modelFolder=MODEL_FOLDER, modelSizes=None,
             device=DEVICE):
    # get the ML model
    # parameters:
    #   PRE_TRAINED: bool, indicating if the program should load from a previous save
    #   norm: the normalizer to use (which contains all length information)
    # return:
    #   an ML model instance as the normalizer to use

    model = Model(norm, width=width, modelSize=modelSizes)

    if(modelNum != -1):
        # load from save file
        modelPath = getModelSavePath(modelFolder, modelNum)
        model.load(modelPath)

    else:
        # create new
        model = Model(norm, width=width, modelSize=modelSizes)
    
    return model

def getMLEnv(dataPath=DATA_FOLDER, modelPath=MODEL_FOLDER, modelNum=-1, 
             makeNorm=True, widthFactor=WIDTH_FACTOR, batchSize=BATCH_SIZE,
             lr=LEARNING_RATE, epochs=MAX_EPOCH, stage=1, inc=SIM_INCREMENT,
             fold=(0, 5), trainRatio=TRAIN_RATIO, noise=NOISE, modelSize=None):
    # returns the items in the ML workflow
    # parameters:
    #   dataPath: the path to the data to use
    # return:
    #   dataLoader:
    #   model:
    #   lossFunc: the loss function
    #   optimizer: 
    #   norm: the normalizer

    # initialize data
    print("loading data from %s/..." % dataPath)
    timeStart = time.time()
    trainData, testData, iterData = getData(dataPath, fold=fold, trainPerc=trainRatio)
    timeEnd = time.time()
    print("\tdataset size: train: %d, test: %d" % (trainData.count,
                                                   testData.count))
    print("\tdone!...%.3f" % (timeEnd - timeStart))
    
    # make normalizer
    print("making data normalizer...")
    timeStart = time.time()
    norm = getNormalizer(makeNorm, trainData, saveFolder=modelPath)
    timeEnd = time.time()
    print("\tdone!...%.3f" % (timeEnd - timeStart))

    # initialize model
    print("initializing model...")
    timeStart = time.time()
    model = getModel(norm, modelNum=modelNum, width=widthFactor, 
                     modelFolder=modelPath, modelSizes=modelSize)
    timeEnd = time.time()
    print("\ttrained models will be saved to %s/" % modelPath)
    print("\tdone!...%.3f" % (timeEnd - timeStart))

    # initialize data loader
    print("initializing data loader...")
    timeStart = time.time()
    trainLoader = CustomLoader(trainData, batchSize, noise=noise)
    testLoader = CustomLoader(testData, batchSize, noise=0)
    iterLoader = CustomLoader(iterData, batchSize, flat=False, noise=0)
    timeEnd = time.time()
    print("\tdone!...%.3f" % (timeEnd - timeStart))

    # pre-normalize delta output
    print("pre-normalizing output data...")
    timeStart = time.time()
    trainData.preNormDelta(norm)
    testData.preNormDelta(norm)
    iterData.preNormDelta(norm)
    timeEnd = time.time()
    print("\tdone!...%.3f" % (timeEnd - timeStart))

    # initialize loss function and optimizer
    print("initializing loss function and optimizer...")
    timeStart = time.time()
    lossFunc = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("\tdone!...%.3f" % (time.time() - timeStart))

    env = Environment(trainLoader=trainLoader, testLoader=testLoader, iterLoader=iterLoader,
                      model=model, lossFunc=lossFunc, optimizer=optimizer,
                      norm=norm, epochs=epochs, modelFolder=modelPath,
                      batchSize=batchSize, trainCount=trainData.count,
                      testCount=testData.count, stage=stage, increments=inc, noise=noise)
    
    return env

def addNoise(nodes, edges):

    nodeNoise =  torch.normal(mean=torch.zeros(nodes.shape), std=NOISE).to(DEVICE)
    edgeNoise = torch.normal(mean=torch.zeros(edges.shape), std=NOISE).to(DEVICE)

    nodesOut = nodes + nodeNoise
    edgesOut = edges + edgeNoise

    return nodesOut, edgesOut

def compute(graphs, env, mode):

    # compose simple graph
    if(mode == TRAIN_TOKEN):
        env.model.train()
        torch.enable_grad()
    elif(mode == TEST_TOKEN):
        env.model.eval()
        torch.no_grad()
    
    env.model(graphs, env) # results stored in graphs
    
    if(mode == FORWARD_TOKEN): return None # shortcut

    env.model.calLoss(graphs, env)
    
    if(mode == TEST_TOKEN): return None # shortcut
    
    env.model.update(env)

    return None

# ========================= modes ==============================

def trainMode(env):

    # print separator
    print('=' * 80)
    log = ""
    torch.cuda.empty_cache()
    
    for epoch in range(env.epochs):
        
        # train
        timeStart = time.time()
        tick = 0
        #for _, (adjMat, nodes, edges, nodesTar, edgesTar) in enumerate(env.trainLoader):
        for _, graphs in env.trainLoader.load():
            #graphs = SimpleStack(adjMat, nodes, edges, nodesTar, edgesTar)
            tick += 1
            compute(graphs, env, TRAIN_TOKEN)

            env.updateLoss(TRAIN_TOKEN)
            lossLog = env.batchReset()
            torch.cuda.empty_cache()

            if(tick % LOG_OUT_INTERVAL == 0):
                print("Train batch progress: %d / %d, %s" % \
                      (tick * env.batchSize, env.trainCount, lossLog), end="\r")

        
        timeEnd = time.time()
        trainTime = timeEnd - timeStart

        # test
        timeStart = time.time()
        tick = 0
        
        if(VALIDATION):
            
            for _, graphs in env.testLoader.load():
                #graphs = SimpleStack(adjMat, nodes, edges, nodesTar, edgesTar)
                tick += 1
                compute(graphs, env, TEST_TOKEN)
                
                env.updateLoss(TEST_TOKEN)
                lossLog = env.batchReset()
                torch.cuda.empty_cache()

                if(tick % LOG_OUT_INTERVAL == 0):
                    print("Test batch progress: %d / %d, %s" % \
                          (tick * env.batchSize, env.testCount, lossLog), end="\r")
        
        timeEnd = time.time()
        testTime = timeEnd - timeStart

        # save model
        if((epoch + 1) % SAVE_INTERVAL == 0):
            env.model.save(getModelSavePath(env.modelFolder, epoch))
        
        # generate epoch report
        lossLog = env.epochReset(epoch)
        epochNum = '0' * (len(str(MAX_EPOCH)) - len(str(epoch))) + str(epoch)
        epochLog = "Epoch: %s, LR: %.6f, %s, Train Time: %d, Test Time: %d" % \
                   (epochNum, LEARNING_RATE, lossLog, trainTime, testTime)
        print(epochLog)
        env.iterTest()
        log += epochLog + '\n'
        writeTxt(env.modelFolder + '/' + LOG_FILENAME, log)

    return log

# ========================= main ==============================

def stageSwitch(stage, modelNum, epochs, width=WIDTH_FACTOR, fold=(0, 5)):

    dataFolder = DATA_FOLDER
    modelFolder = MODEL_FOLDER
    if(stage == 2):
        dataFolder += STAGE_TWO_FOLDER
        modelFolder += STAGE_TWO_FOLDER
    env = getMLEnv(dataPath=dataFolder, modelPath=modelFolder, 
                   modelNum=modelNum, epochs=epochs, widthFactor=width,
                   makeNorm=NEW_NORM, stage=stage, inc=SIM_INCREMENT,
                   fold=fold, noise=NOISE)
    
    return env

def main():

    print("loading data and components...")
    timeStart = time.time()
    env = stageSwitch(TRAIN_STAGE, MODEL_NUM, MAX_EPOCH)
    timeEnd = time.time()
    print("done!...%.3fsec" % (timeEnd - timeStart))

    log = trainMode(env)

    return log

def scheduler():
    # stage 1
    logs = ["noise: %.3f, disLocProp: %s, disLoc: %.3f, target:%.3f" % \
            (NOISE, PROP_DISLOCATION, DISLOC_GRAD_SCALE, TARGET_GRAD_SCALE)]
    print(logs)
    
    fold = (0, 5)
    env = stageSwitch(1, MODEL_NUM, STAGE_TWO_MAX_EPOCH, STAGE_TWO_WIDTH, fold=fold)
    log = trainMode(env)
    log += [log]

    writeJSON("./cross_validaton_log.json", logs)

if __name__ == "__main__":
    makeFolders("./")
    #scheduler()
    main()