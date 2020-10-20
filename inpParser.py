import numpy as np

from parameters import *
from inputGen import *

###############################################################################
# tokens
###############################################################################

class Trial(object):
    def __init__(self, inpDirPath, filDirPath, name):

            self.name = name # string, the name of this file

            # file paths, strings
            self.inpFolder = inpDirPath # folder of this trial's input file
            self.filFolder = filDirPath # folder of this trial's nodal files
            self.inpPath = inpDirPath + '/' + name + INP_SUFFIX
            self.coorPath = filDirPath + '/' + name + COOR_FILE_SUFFIX
            self.stressPath = filDirPath + '/' + name + STRESS_FILE_SUFFIX

            # model and simulation result
            self.model = None # Model object, containing all geometric info
            self.coor = None # numpy 3D array, shape: (frame, index, Point3D)
            self.coorMirrored = None
            self.stress = None # numpy 3D array, shape:
            self.stressMirrored = None
            # (frame, index, element gaussian point index, stress values)
            #     element index

            # solver information
            self.inc = None # int, number of increments in this step
            self.freq = None # int, interval between nodal file output
            self.frames = None # int, number of available frames (inc / freq)

    def __repr__(self):
            return "|Trial Name: %s|" % self.name

    def updateSolver(self, inc, freq):
        # updates the solver parameters of this trial
        # parameters:
        #   inc: see self.inc
        #   freq: see self.freq
        #   frames: see self.frames

        self.inc = inc
        self.freq = freq
        self.frames = int(inc / freq)

    def loadResults(self):
        # load the result files (coor, stress) for this trial
        # [:, 1:] to trim the index column

        # coor
        coorShape = (self.frames, self.model.nCount, COOR_LEN)
        self.coor = Trial.loadCoor(self.coorPath, coorShape)

        # stress
        stressShape = (self.frames, self.model.eCount, GP_LEN, STRESS_LEN)
        self.stress = Trial.loadStress(self.stressPath, stressShape)

    @staticmethod
    def loadCoor(path, tarShape):

        if(path.endswith(CSV_SUFFIX)): # csv
            coor = np.genfromtxt(path, delimiter=",").astype(np.float)
        elif(path.endswith(NUMPY_SUFFIX)): # numpy
            coor = np.load(path)

        coor = coor[:, 1:] # drop index
        coor = coor.reshape(tarShape)

        return coor

    @staticmethod
    def loadStress(path, tarShape):

        if(path.endswith(CSV_SUFFIX)): # csv
            stress = np.genfromtxt(path, delimiter=",").astype(np.float)
        elif(path.endswith(NUMPY_SUFFIX)): # numpy
            stress = np.load(path)

        stress = stress.reshape(tarShape)

        return stress
    
class InpParser(object):
    def __init__(self, inp):
        self.inp = inp.splitlines()# list, lines in the input file
        # Primitives ==========================================================
        # int, indexing of the first or last line of the part
        # nodes
        self.nMapStart = None # int, node map code start
        self.nMapEnd = None # int, node map end end
        self.nodeStart = None # int, node declaration start
        self.nodeEnd = None # int, node declaration end
        # elements
        self.eMapStart = None # int, element map code start
        self.eMapEnd = None # int, element map code end
        self.elemStart = None # int, element declaration start
        self.elemEnd = None # int, element declaration end
        # Primitive sets ======================================================
        # list of int, indexing of the first or last line of the part
        # the length of start and end should have the same length
        # elsets
        self.elsetStart = [] # list of int, elset code starts
        self.elsetEnd = [] # list of intm elset code ends
        # nsets
        self.nsetStart = [] # list of int, nset code starts
        self.nsetEnd = [] # list of int, nset code ends
        # boundary conditions =================================================
        # list of int, indexing of the first or last line of the part
        # the length of start and end should have the same length
        self.bcStart = [] # list of int, bc code starts
        self.bcEnd = [] # list of int, bc code ends
        # solver configurations ===============================================
        self.inc = None # int, increment specification line
        self.freq = None # node file output frequency specification line
        # initialization
        self.getCodeIndices() # initialize the index variables listed above
        self.stripAsterisks() # strips * from inp
        # sanity checks
        
        assert(len(self.elsetStart) == len(self.elsetEnd)) # elsets
        assert(len(self.nsetStart) == len(self.nsetEnd)) # nsets
        assert(len(self.bcStart) == len(self.bcEnd)) # boundary conditions
    
    def parseAll(self):
        # parse all data in the input file
        # return:
        #   a Model object containing all informaiton of the input model

        # nodes
        nMap = self.parseNodeMap()
        nodes = self.parseNodes()
        # elems
        eMap = self.parseElemMap()
        elems = self.parseElems()
        # sets
        elsets = self.parseElsets()
        nsets = self.parseNsets()
        # boundary conditions
        bcFixed, bcStress = self.parseBCs()

        return Model(nMap, nodes, eMap, elems, 
                     elsets, nsets, bcFixed, bcStress)

    def getCodeIndices(self):
        # get the indices of the start and end of different code blocks

        # nodes
        self.nMapStart = self.inp.index(NODE_MAP_START_TOKEN)
        self.nMapEnd = self.inp.index(NODE_MAP_END_TOKEN)
        self.nodeStart = self.inp.index(NODE_START_TOKEN)
        self.nodeEnd = self.inp.index(NODE_END_TOKEN)
        # elements
        self.eMapStart = self.inp.index(ELEM_MAP_START_TOKEN)
        self.eMapEnd = self.inp.index(ELEM_MAP_END_TOKEN)
        self.elemStart = self.inp.index(ELEM_START_TOKEN)
        self.elemEnd = self.inp.index(ELEM_END_TOLKEN)

        # elsets, nsets, bcs
        # on-the-fly searching for efficiency
        inElset = False
        for i in range(self.elemEnd, len(self.inp)):
            line = self.inp[i]
            # elsets
            if(line == ELEM_SET_START_TOLKEN):
                if(inElset == False):
                    inElset = True
                    self.elsetStart += [i]
            elif(line == ELEM_SET_END_TOKEN):
                if(inElset == True):
                    inElset = False
                    self.elsetEnd += [i]
            # nsets
            elif(line == NODE_SET_START_TOKEN): self.nsetStart += [i]
            elif(line == NODE_SET_END_TOKEN): self.nsetEnd += [i]
            # boundary conditions
            elif(line == BC_START_TOKEN): self.bcStart += [i]
            elif(line == BC_END_TOKEN): self.bcEnd +=[i]
        
        # solver specifications, start search from the end of bc
        for i in range(self.bcEnd[-1], len(self.inp)):
            line = self.inp[i]
            if(INCREMENT_TOKEN in line): self.inc = i # increment
            if(FREQ_TOKEN in line): self.freq = i # output frequency

    def stripAsterisks(self):
        # strip the preceeding asterisks in the input file lines

        for i in range(len(self.inp)):
            stripped = self.inp[i] # initialize as the whole line
            for j in range(len(self.inp[i])):
                if(self.inp[i][j] == '*'):
                    continue # skip *
                else:
                    stripped = self.inp[i][j:]
                    break # dump at first non-*

            self.inp[i] = stripped

    def parseNodeMap(self):
        # parse the node map from the input file
        # return:
        #   a dictionary that map IDString to input file indices

        # strip header and footer
        nMapCode = self.inp[self.nMapStart + 1: self.nMapEnd]
        # save pairs into a dictionary
        nMap = {}
        for line in nMapCode:
            for pair in line.split(", "):
                # split pairs into raw ID and target ID
                items = pair.split(':')
                rawID, tarID = items[0], int(items[1])
                # -1 to convert from 1-indexing to 0-indexing
                nMap[rawID] = tarID - 1

        return nMap
    
    def parseNodes(self):
        # parse all nodes in the model
        # return:
        #   a dictionary that map 0-based input file index to coordinates

        nodeCode = self.inp[self.nodeStart + 1: self.nodeEnd] # strip code

        nodes = np.zeros((len(nodeCode), COOR_LEN))
        

        for line in nodeCode:
            # get items
            items = line.split(", ")
            # -1 to convert from 1-indexing to 0-indexing
            index = int(items[0]) - 1
            nodes[index] = [float(item) for item in items[1:]]
        
        return nodes

    def parseElemMap(self):
        # parse the element map from the input file
        # return:
        #   a dictionary that map IDString to input file indices

        
        eMapCode = self.inp[self.eMapStart + 1: self.eMapEnd] # strip

        # save pairs into a dictionary
        eMap = {}
        for line in eMapCode:
            for pair in line.split(", "):
                # split pairs into raw ID and target ID
                items = pair.split(':')
                rawID, tarID = items[0], int(items[1])
                # -1 to convert from 1-indexing to 0-indexing
                eMap[rawID] = tarID - 1

        return eMap

    def parseElems(self):
        # parse all elements in the model
        # return:
        #   a dictionary that map 0-based input file index to vertex indices

        elemCode = self.inp[self.elemStart + 1: self.elemEnd] # strip

        elems = np.zeros((len(elemCode), GP_LEN))

        for line in elemCode:
            # get items
            items = line.split(", ")
            # -1 to convert from 1-indexing to 0-indexing
            index = int(items[0]) - 1
            elems[index] = [int(item) - 1 for item in items[1:]] # to 0-indexing
        
        return elems

    def parseElsets(self):
        # parse all element sets in the model
        # return:
        #   a dictionary that map set names to 0-based node indices

        elsetCount = len(self.elsetStart)

        elsets = {}
        for i in range(elsetCount):
            # slice code from input file
            start, end = self.elsetStart[i], self.elsetEnd[i]
            code = self.inp[start + 1:end]

            setName = code[0].split(", ")[1][len("elset="):] # name of elset
            # add elements of this set to a list
            elems = []
            for j in range(1, len(code)):
                new = []
                for num in code[j].split(", "):
                    if(num.isnumeric()): # avoid empty string
                        # -1 to convert from 1-indexing to 0-indexing
                        new += [int(num) - 1]
                        
                elems += new
        
            elsets[setName] = elems
        
        return elsets

    def parseNsets(self):
        # parse all node sets in the model
        # return:
        #   a dictionary that map set names to 0-based node indices

        nsetCount = len(self.nsetStart)

        nsets = {}
        for i in range(nsetCount):
            # slice code from input file
            start, end = self.nsetStart[i], self.nsetEnd[i]
            code = self.inp[start + 1:end]

            setName = code[0].split(", ")[1][len("nset="):] # name of elset
            # add elements of this set to a list
            nodes = []
            for j in range(1, len(code)):
                # -1 to convert from 1-indexing to 0-indexing
                nodes += [int(item) - 1 for item in code[j].split(", ")]
        
            nsets[setName] = nodes
        
        return nsets

    def parseBCs(self):
        # =======WARNING: POTENTIAL HARDCODING=======
        # parse all node sets in the model
        # return:
        #   bcFixed: a dictionary that map set names to fixed bool
        #   bcStress: a dictionary that map set names to a stress field

        bcCount = len(self.bcStart)
        # tokens
        nameToken = "Name: "
        typeToken = "Type: "
        stressToken = "Stress"
        fixedToken = "Displacement/Rotation"

        bcStress, bcFixed = {}, {}
        for i in range(bcCount):
            # slice code from input file
            start, end = self.bcStart[i], self.bcEnd[i]
            code = self.inp[start + 1:end]

            # get name
            nameStart = code[0].find(nameToken) + len(nameToken)
            nameEnd = code[0].find(typeToken) - 1
            name = code[0][nameStart:nameEnd]
            # get type
            typeStart = code[0].find(typeToken) + len(typeToken)
            bcType = code[0][typeStart:]
            # get items in the third row
            items = code[2].split(", ")
            target = items[0]

            # stress field
            if(bcType == stressToken):
                stress = [float(item) for item in items[1:]]
                bcStress[target] = stress
            # fixed node
            elif(bcType == fixedToken):
                bcFixed[target] = True

        return bcFixed, bcStress

    def parseSolver(self):
        # parse solver information from the input file
        # return:
        #   inc: the increment as described in the input file
        #   freq: the data output frequency as described in the input file

        # increment
        incLine = self.inp[self.inc]
        inc = int(incLine[incLine.find(INCREMENT_TOKEN) + len(INCREMENT_TOKEN):])
        # frequency
        freqLine = self.inp[self.freq]
        freq = int(freqLine[freqLine.find(FREQ_TOKEN) + len(FREQ_TOKEN):])
        
        return inc, freq

class ElemBlockSimple(ElemBlock):
    def __init__(self, blockType, index, name):
        super(ElemBlockSimple, self).__init__(blockType, index)
        self.name = name # string, name of this block
        self.geoSamp = [] # list of IDStrings to sample for
                                     # geometric information
        self.matSamp = [] # list of 
            # (element IDStrings, gaussian point indices)
            # to sample for mateiral/stress information
        self.sampFrames = None 

    def __repr__(self):
        return "ElemBlock: %s" % self.name

    def expandNode(self, nID):
        # expand the width, length, and height based on the input IDString
        # parameter:
        #   nID: the node IDString to evaluate (extend) with this block

        info = IDString.breakID(nID)
        
        frameLen = info["frame"] + 1
        if(frameLen > self.nLen):
            self.nLen = frameLen # because of 0-index

        frameWid = info["row"] + 1
        if(frameWid > self.nWid):
            self.nWid = frameWid # because of 0-index

        frameHei = info["layer"] + 1
        if(frameHei > self.nHeight):
            self.nHeight = frameHei # because of 0-index

    def expandElem(self, eID):
        # expand the width, length, and height based on the input IDString
        # parameter:
        #   eID: the element IDString to evaluate (extend) with this block

        info = IDString.breakID(eID)
        
        frameLen = info["frame"] + 1
        if(frameLen > self.eLen):
            self.eLen = frameLen # because of 0-index

        frameWid = info["row"] + 1
        if(frameWid > self.eWid):
            self.eWid = frameWid # because of 0-index

        frameHei = info["layer"] + 1
        if(frameHei > self.eHeight):
            self.eHeight = frameHei # because of 0-index
 
    def getGeoSampID(self):
        # to implement, gets the sampling point indices and stores them at
        # self.geoSamplingIndices
        assert(False) # throws warning if this function is not overwritten

    def getMatSampID(self):
        # to implement, get the sampling point indices and stores them at
        # self.matSamplingIndices
        assert(False) # throws warning if this function is not overwritten

class Model(object):
    def __init__(self, nMap, nodes, eMap, elems, elsets, nsets, 
                 bcFixed, bcStress):
        # counts
        self.nCount = len(nodes) # int, the number of nodes in the model
        self.eCount = len(elems) # int, the number of elements in the model
        # model    
        self.nMap = nMap # dictionary, node IDString to index map
        self.nMapMirrored = None
        self.nodes = nodes # np array, (index, Point3D)
        self.nodesMirrored = None
        self.eMap = eMap # dictionary, element IDString to index map
        self.eMapMirrored = None
        self.elems = elems # np array, (index, vertex ids)
        self.elsets = elsets # dictionary, element set name to element indices
        self.nsets = nsets # dictionary, node set name to element indices
        self.bcStress = bcStress # dictionary, target set ID to stress field
        self.bcFixed = bcFixed # dictionary, target set ID to bool(True)
        self.elemStress = self.getElemStress() # np array, (index, stress field)
        self.elemStressMirrored = None
    
    def dictToArray(self, dictIn):
        # convert a dictionary into an numpy array
        # parameters:
        #   dictIn: the dictionary to process, the keys must be 0-based 
        #           consecutive integers
        # return: 
        #   the converted matrix, shape: (len(keys), len(data))

        # initialize an empty array
        arrayLen = len(dictIn)
        arrayWid = len(dictIn[list(dictIn.keys())[0]])
        array = np.zeros((arrayLen, arrayWid))

        for key in dictIn: # loop through dictionary to dump data
            for i in range(len(dictIn[key])):
                array[key][i] = dictIn[key][i]
        
        return array

    def getElemStress(self):
        # initialize a matrix of stress field to the elements
        # return:
        #   a numpy array, shape(len(elems), len(stressfield))

        # initialize an empty array
        array = np.zeros((self.eCount, STRESS_LEN))

        for tarElset in self.bcStress: # loop through bc sets
            targets = self.elsets[tarElset]
            field = self.bcStress[tarElset]

            for tar in targets: # for each target voxel

                # write stress field
                array[tar] = field
        
        return array