import numpy
from parameters import *
from inputGen import IDString

def getMirrorMap(trial):

    # for each node in the index map, flip axis
    nLen = {}
    for blockID in trial.blocks:
        nLen[blockID] = trial.blocks[blockID].nLen
    newNodeMap = _flipAxial(trial.model.nMap, nLen)
    
    # likewise for elem map
    eLen = {}
    for blockID in trial.blocks:
        eLen[blockID] = trial.blocks[blockID].eLen
    newElemMap = _flipAxial(trial.model.eMap, eLen)

    trial.model.nMapMirrored = newNodeMap
    trial.model.eMapMirrored = newElemMap

def _flipAxial(mapOld, nLen):

    newMap = {}

    for ID in mapOld:
        info = IDString.breakID(ID)
        # get parent block info
        blockType = "beam" if info["blockType"] == BEAM_INDEX else "joint"
        index = info["index"]
        parentBlockID = blockType + '_' + str(index)
        parentNodeLen = nLen[parentBlockID]

        # compose new ID
        frameOld = info["frame"]
        frameNew = (parentNodeLen - frameOld) - 1 # -1 due to 0-index
        newID = IDString.getID(info["blockType"], info["selfType"], 
                               info["index"],
                               frameNew, info["row"], info["layer"])
        newMap[newID] = mapOld[ID]

    return newMap

def mirrorGeo(trial):

    mirror = np.asarray([-1, 1, 1])
    # initial
    trial.model.nodesMirrored = trial.model.nodes * mirror
    # steps
    trial.coorMirrored = trial.coor * mirror

def mirrorStress(trial):

    mirror = np.asarray([1, 1, 1,  -1, -1, 1])
    # initial
    trial.model.elemStressMirrored = trial.model.elemStress * mirror
    # steps
    trial.stressMirrored = trial.stress * mirror