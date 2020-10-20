import torch
from time import time, localtime, asctime
from dataNormalizer import GridNorm as Normalizer
from dataset import readJSON, sliceMD, SimpleStack, readFile, writeJSON
from models import DoubleGN as Model
from parameters import *
from forward import Simulator

def singleInit(data, mode="torch"):

    graph = SimpleStack().initFromData(data, mode="numpy")
    graph.fixAdj()

    return graph

def batchInit(dicts, mode="torch"):

    graphs = []
    
    count = len(dicts)
    ref = dicts[0]
    adjShape = (count, len(ref[ADJ_MATRIX_TOKEN]), len(ref[ADJ_MATRIX_TOKEN][0]))
    nodesShape = (count, len(ref[NODES_TOKEN]), len(ref[NODES_TOKEN][0]))
    edgesShape = (count, len(ref[EDGES_TOKEN]), len(ref[EDGES_TOKEN][0]))

    graphs = SimpleStack()
    graphs.setSizes(adjShape, nodesShape, edgesShape, mode="numpy")

    for i in range(count):
        graph = SimpleStack().initFromData(dicts[i], mode="numpy")
        graph.fixAdj()
        graphs.adjMat[i] = graph.adjMat
        graphs.nodes[i] = graph.nodes
        graphs.edges[i] = graph.edges

    return graphs

def graphsInit(source, mode="torch"):

    json = readJSON(source)
    if(isinstance(json, list)):
        graphs = batchInit(json, mode)
    elif(isinstance(json, dict)):
        graphs = singleInit(json, mode)
    else:
        assert False, "unsupported data"
    
    graphs.toType("torch")

    return graphs

def singelForward(graphs, sim):

    graphs = sim.forward(graphs)

    return graphs

def batchForward(graphs, sim, setting, ranking):
    
    # simulate
    graphs = sim.forward(graphs)
    # compare result
    nodesFinal, edgesFinal = graphs.nodesSeq[-1], graphs.edgesSeq[-1]
    goalPts = torch.tensor(setting["goalPts"]).to(DEVICE)
    perf = checkGoals(setting["goals"], nodesFinal, edgesFinal, goalPts)
    # rank actions
    ranked = rank(perf, setting["actions"])
    writeJSON(ranking, ranked)

    return graphs

def checkGoals(goals, nodes, edges, goalPts):

    objectiveGoals = []
    for goal in goals: 
        sub, tar = goal
        sub = getPoint(sub, nodes, edges, goalPts)
        tar = getPoint(tar, nodes, edges, goalPts)
        dists = dist(sub, tar).tolist()
        objectiveGoals += [dists]

    return objectiveGoals

def rank(perf, actions):

    rankedResult = []
    for i in range(len(perf)): # objective #
        temp = []
        for j in range(len(perf[i])): # permutation #
            temp += [(j, perf[i][j], actions[j])]
        ranked = sorted(temp, key=lambda x: x[1])
        rankedResult += [ranked]

    return rankedResult

def dist(pt1, pt2):

    dist = torch.sqrt(torch.sum((pt1 - pt2) ** 2, -1))
    
    return dist

def getPoint(tag, nodes, edges, goals):

    indicator = tag[0]
    if(indicator == 'n'): # node
        pts = nodes[:, tag[1], tag[2] * COOR_LEN:(tag[2] + 1) * COOR_LEN]
    elif(indicator == 'e'): # edges
        pts = edges[:, tag[1], tag[2] * COOR_LEN:(tag[2] + 1) * COOR_LEN]
    elif(indicator == 'g'): # goal
        pts = goals[tag[1]]

    return pts

def main(source, target=None, setting=None, ranking=None):

    msg = [asctime(localtime())]

    # load input
    msg += ["loading input design..."]
    s = time()
    graphs = graphsInit(source)
    e = time()
    msg += ["\tfinished in %.3f seconds" % (e - s)]

    # load model
    msg += ["loading simulator..."]
    s = time()
    sim = Simulator()
    e = time()
    msg += ["\tfinished in %.3f seconds" % (e - s)]

    # forward simulate
    msg += ["running simulation..."]
    s = time()
    if(setting == None): # single graph forward
        graphs = singelForward(graphs, sim)
    else: # batch graph forward
        setting = eval(readFile(setting))
        graphs = batchForward(graphs, sim, setting, ranking)
    e = time()
    msg += ["\tfinished in %.3f seconds" % (e - s)]

    # save result
    msg += ["writting result..."]
    s = time()
    if(target == None):
        target = source.replace(".json", "_result.json")
    graphs.saveJSONSeq(target)
    e = time()
    msg += ["\tfinished in %.3f seconds" % (e - s)]

    msg = '\n'.join(msg)
    print(msg)

    return msg

if __name__ == "__main__":
    main("C:/CHI_design_tool/batch/conduit.json", 
         "C:/CHI_design_tool/batch/conduit_result.json",
         "C:/CHI_design_tool/batch/setting.json",
         "C:/CHI_design_tool/batch/ranking.json")