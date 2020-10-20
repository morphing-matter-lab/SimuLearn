import math as m
import time as time

from parameters import *

###############################################################################
# Model classes
###############################################################################

class IDString(object):
    def __init__(self):
        pass
    
    @staticmethod
    def getID(blockType, selfType, elemID, frame, row, layer):
        # create an IDString for this object
        # return:
        #   string, the IDString of this object

        # format ABCCCDDDEEEFFF
        #   A -> element parent block type, 1:beam, 2:joint
        #   B -> element type, 0:node, 1:element (voxel), 2:nset, 3:elset
        # CCC -> element index
        # DDD -> frame index (length)
        # EEE -> row index (width)
        # FFF -> layer index (height)
        A   = str(blockType)
        B   = str(selfType)
        CCC = (INDEX_LEN - len(str(elemID))) * '0' + str(elemID)
        DDD = (INDEX_LEN - len(str(frame))) * '0' + str(frame)
        EEE = (INDEX_LEN - len(str(row))) * '0' + str(row)
        FFF = (INDEX_LEN - len(str(layer))) * '0' + str(layer)
        
        return A + B + CCC + DDD + EEE + FFF

    @staticmethod
    def breakID(id):
        # inverse function of getID
        # parameter:
        #   the IDString to process
        # return:
        #   a dictionary containing useful information

        information = {}
        information["blockType"]  = int(id[0])
        information["selfType"]  = int(id[1])
        information["index"] = int(id[2 + 0 * INDEX_LEN: 2 + 1 * INDEX_LEN])
        information["frame"] = int(id[2 + 1 * INDEX_LEN: 2 + 2 * INDEX_LEN])
        information["row"]   = int(id[2 + 2 * INDEX_LEN: 2 + 3 * INDEX_LEN])
        information["layer"] = int(id[2 + 3 * INDEX_LEN:])
        
        return information

class Primitive(object):
    def __init__(self, id):
        self.id = id # IDString, ID of this primitive
        self.inpIndex = None # int, index of this primitive in the input file

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

class Node(Primitive):
    def __init__(self, id, coor):
        super(Node, self).__init__(id)
        self.coor = coor # Point3D, point 3D
        self.interface = False # boolean, indicates if this node is at the 
                               # contacting edge between blocks
    
    def __hash__(self):
        # hashing based on coordinates
        return hash((self.coor[0], self.coor[1], self.coor[2]))
    
    def __eq__(self, other):
        # equality based on coordinates
        if(isinstance(other, Node)):
            eq = self.coor[0] == other.coor[0] and \
                 self.coor[1] == other.coor[1] and \
                 self.coor[2] == other.coor[2]
        elif(isinstance(other, list) and len(other) == 3):
            eq = self.coor[0] == other[0] and \
                 self.coor[1] == other[1] and \
                 self.coor[2] == other[2]
        else:
            eq = False
        
        return eq

class Voxel(Primitive):
    def __init__(self, id, vertices):
        super(Voxel, self).__init__(id)
        self.vertices = vertices # list, IDStrings of vertex nodes

    def __hash__(self):
        return hash(set(self.vertices))
        
    def __eq__(self, other):
        return set(self.vertices) == set(other.vertices)

class PrimitiveSet(object):
    def __init__(self, name, count):
        self.name = name # string, name of this set
        self.count = count # int, number of primitives in this set

class Elset(PrimitiveSet):
    def __init__(self, name, elems, blockID=None):
        super(Elset, self).__init__(name, len(elems))
        self.elems = elems # list, IDstrings of elements
        
class Nset(PrimitiveSet):
    def __init__(self, name, nodes, blockID=None):
        super(Nset, self).__init__(name, len(nodes))
        self.nodes = nodes # list, IDStrings of nodes in this set

class Part(object):
    def __init__(self, blocks, id, elemType, aName, iName, material):
        # names for input file
        self.id = id # string, name of part
        self.elemType = elemType # string, element type of elements
        self.assemblyName = aName # string, name of assembly
        self.instanceName = iName # string, name of instance
        # FEA parameters
        self.material = material # Mateiral object, material for this part
        self.boundaryConditions = [] # list, containing all boundary conditions
        # geometries and elements
        self.blocks = blocks # list, containing all of its blocks
        self.nodes = {} # map, node[nodeID] = Node class object
        self.elems = {} # map, elems[elemID] = Voxel class object
        self.elsets = {} # map, elsets[elsetID] = Elset class objecct
        self.nsets = {} # map, nsets[nsetID] = Nset class object
        self.nMap = {} # map, nMap[rawNodeID] = targetNodeID
        # nMap connects culled points to surviving points
    
    def objectListToDict(self, objects):
        # convert element list into dictionarys
        # parameter:
        #   objects: the list of objects for conversion
        # return:
        #   dictionary: the converted dict, dict[object id] = object
        
        # convert set to list
        if(isinstance(objects, set)): objects = list(objects)
        # check input type
        if (isinstance(objects[0], Primitive)):
            useId = True
        else: useId = False
        
        # convert to dictionary
        dictionary = {}
        for obj in objects:
            if (useId): dictionary[obj.id] = obj # primitives
            else: dictionary[obj.name] = obj # sets
        
        return dictionary

class ElemBlock(object):
    def __init__(self, blockType, index, elemLenCount=0, elemWidCount=0, 
                 elemHeiCount=0):
        self.type = blockType # int, type of this block
        self.id = index # int, index of this block
        # geometry: 3D information: elements
        self.elems = [] # list, element IDstrings of this block
        self.eLen = elemLenCount # int, element count along its length
        self.eWid = elemWidCount # int, element count along its width
        self.eHeight = elemHeiCount # int, element count along its height
        # geometry: 3D information: nodes
        self.nodes = [] # list, node IDstrings of this block
        self.nLen = elemLenCount + 1 # int, node count along its length
        self.nWid = elemWidCount + 1 # int, node count along its width
        self.nHeight = elemHeiCount + 1 # int, node count along its height

class Material(object):
    def __init__(self, name, properties):
        self.name = name # string, name of the material
        self.properties = properties # string, used for the input file

###############################################################################
# Abaqus FEA solver classes
###############################################################################

class BoundaryCondition(object):
    def __init__(self, BCType, name, target):
        self.type = BCType # string, type of this boundary condition
        self.name = name # string, name of this boundary condition
        self.target = target # IDString, target nset/elset to apply to

class Solver(object):
    def __init__(self, incCount, saveFreq):
        self.incCount = incCount # int, number of increments to perform 
        self.step = 1. # float, total steps
        self.increment = self.step / incCount # float, increment step size
        self.saveFreq = saveFreq # int, interval between filesaves
        self.code = self.genCode() # string, code for input file

    def genCode(self):
        # =======WARNING: POTENTIAL HARDCODING=======
        # generated the gcode for the solver
        # return:
        #   the input file code of this solver

        code = ["**",
                "** STEP: Shrinking",
                "**",
                "*Step, name=shrinking, nlgeom=Yes, inc=%d" % (self.incCount),
                "*Static, direct",
                "%.2f, %.1f," % (self.increment, self.step),
                "**",
                "** Name: BodyForce  Type: Gravity",
                "*Dload,amplitude=gravity",
                ", GRAV, 1.96e+3, 0., 0., -1.",
                "**",
                "**",
                "** OUTPUT REQUEST",
                "**",
                "*Restart, write, frequency=0",
                "**",
                "** FIELD OUTPUT: F-Output-1",
                "**",
                "*Output, field, variable=PRESELECT",
                "**",
                "**HISTORY OUTPUT: H-Output-1",
                "**",
                "*Output, history, variable=PRESELECT",
                "**",
                "*FILE FORMAT, ASCII",
                "*node file, frequency=%d" % self.saveFreq,
                "U, COORD",
                "*El file, frequency=%d" % self.saveFreq,
                "S, COORD",
                "*End Step"]

        return "\n".join(code)

class InpCompiler(object):
    def __init__(self):
        self.lineLen = LINE_LEN # int, the maximum number of IDStrings to use in
                               # each line
        self.code = [] # list of strings, blocks of input file codes
                       # change into a string when compiled

    def write(self, path):
        # save the current code in the compiler into a gcode file
        # parameter:
        #   path: the path to save input file to
        
        with open(path, "wt") as f:
            f.write(self.code)

    def listIINDEX_LEN(self, inList):
        # convert a long list (1D) to a shorter list (~2D), each row in the
        # list containing #lineLen of items
        # parameter:
        #   inList: the list of strings to convert
        # return:
        #    the shorter, wrapped list of srtings

        idLines = []
        lineCount = int(m.ceil(len(inList) / self.lineLen))
            
        for i in range(lineCount):
            start, end = i * self.lineLen, (i + 1) * self.lineLen
            idLines += [", ".join(inList[start:end])]
        
        return idLines # returns a list of lines (of IDStrings)

    def compile(self, part, solver):
        # compile a input file for the current scene
        # parameter:
        #   part: the Part object (the model) to compile for
        #   solver: the Solver object to use for simulation

        # header code
        self.addHeader(part)
        # part code
        self.addPart(part)
        # assembly code
        self.addAssembly(part)
        # boundary condition code
        self.addBoundaryConditions(part)
        # add solver
        self.addSolver(solver)
        # final compile
        self.code = '\n'.join(self.code)

    def addHeader(self, part):
        # add header to the code
        # parameter:
        #   part: the Part object (the model) to compile for

        code = []
        code += ["*Heading"]
        code += [self.addTimeStamp()]
        code += ["**",
                 "*Preprint, echo=NO, model=NO,history=NO,contact=NO",
                 "**"]
        code += [self.addModelnfo(part)]

        self.code += ['\n'.join(code)]
    
    def addPart(self, part):
        # add part information to the code
        # parameter:
        #   part: the Part object (the model) to compile for

        # part code header
        header = ["**", 
                  "**PARTS",
                  "**",
                  "*Part, name=%s" % part.id,
                  "**"]
        
        code = []
        code += [self.addNodeMap(part)]
        code += [self.addNodes(part)]
        code += [self.addElementMap(part)]
        code += [self.addElements(part)]
        code += [self.addAllNset(part)]
        code += [self.addAllElset(part)]
        code += [self.addSection(part.material)]

        footer = ["**",
                  "*End Part",
                  "**"]

        self.code += ["\n".join(header + code + footer)]

    def addAssembly(self, part):
        # add assembly information to the code
        # parameter:
        #   part: the Part object (the model) to compile for

        header = ["**",
                  "**ASSEMBLY",
                  "**", 
                  "*Assembly, name=%s" % part.assemblyName,
                  "**"]

        code = []
        code += [self.addInstance(part)]
        code += [self.addBCElsets(part)]
        code += [self.addBCNsets(part)]

        footer = ["**",
                  "*End Assembly",
                  "**",
                  "**ASSEMBLY END",
                  "**"]
        
        self.code += ['\n'.join(header + code + footer)]

    def addBoundaryConditions(self, part):
        # add boundary condition information to the code
        # parameter:
        #   part: the Part object (the model) to compile for

        header = ["**",
                  "**BOUNDARY CONDITIONS",
                  "**"]
        
        code = []
        code += [self.addGravity()]
        code += [self.addMaterial(part.material)]
        code += [self.addFixed(part)]
        code += [self.addStressFields(part)]
        
        footer = ["**",
                  "**BOUNDARY CONDITIONS END",
                  "**"]

        self.code += ["\n".join(header + code + footer)]

    def addSolver(self, solver):
        # add solver information to the code
        # parameter:
        #   solver: the Solver object (the model) to compile for

        code = ["** ---------------------------------------------------",
                solver.code]

        self.code += ['\n'.join(code)]

    ###########################################################################
    # for header
    ###########################################################################

    def addTimeStamp(self):
        # generate a time stamp
        # return:
        #   string of time stamp

        date = time.localtime()
        
        curTime = "%d:%d:%d, %d/%d, %d" % (date.tm_hour, date.tm_min,
                                          date.tm_sec,  date.tm_mon, 
                                          date.tm_mday, date.tm_year)
        code = "**Created time: %s" % curTime 
        
        return code
    
    def addModelnfo(self, part):
        # generate model design information
        # parameter:
        #   part: the Part object (the model) to compile for
        # return:
        #   string that describes the design

        header = ["**",
                  modelStartToken]

        code = []
        for line in part.genModelInfo():
            code += ["**" + line]
        
        footer = [modelEndToken,
                  "**"]

        return '\n'.join(header + code + footer)
    

    ###########################################################################
    # for parts
    ###########################################################################

    def addNodeMap(self, part):
        # generate node map information
        # parameter:
        #   part: the Part object (the model) to compile for
        # return:
        #   string that encodes the node map

        header = ["**",
                  nMapStartToken]
        
        code = []
        for rawID in part.nMap:
            inpIndex = part.nodes[part.nMap[rawID]].inpIndex
            code += ["%s:%d" % (rawID, inpIndex)]
        
        # list into lines-in-lnegth and add **
        code = self.listIINDEX_LEN(code)
        for i in range(len(code)):
            code[i] = "**" + code[i]
        
        footer = [nMapEndToken,
                  "**"]
        
        return '\n'.join(header + code + footer)
    
    def addNodes(self, part):
        # generate node information
        # parameter:
        #   part: the Part object (the model) to compile for
        # return:
        #   string that encodes the node indices and coordinates

        header = ["**",
                  "*Node",
                  nodeStartToken]
        
        code = []
        for nodeID in part.nodes:
            inpIndex = part.nodes[nodeID].inpIndex
            coor = part.nodes[nodeID].coor
            code += ["%d, %f, %f, %f" % (inpIndex, coor[0], coor[1], coor[2])]
        
        footer = [nodeEndToken,
                  "**"]

        return '\n'.join(header + code + footer)

    def addElementMap(self, part):
        # generate element map information
        # parameter:
        #   part: the Part object (the model) to compile for
        # return:
        #   string that encodes the element map

        header = ["**",
                  eMapStartToken]
        
        code = []
        for elemID in part.elems:
            code += ["%s:%d" % (elemID, part.elems[elemID].inpIndex)]
        
        # list into lines-in-lnegth and add **
        code = self.listIINDEX_LEN(code)
        for i in range(len(code)):
            code[i] = "**" + code[i]
        
        footer = [eMapEndToken,
                  "**"]
        
        return '\n'.join(header + code + footer)

    def addElements(self, part):
        # generate element information
        # parameter:
        #   part: the Part object (the model) to compile for
        # return:
        #   string that encodes the indices and vertices

        header = ["**",
                  "*Element, type=%s" % part.elemType,
                  elemStartToken]
        
        code = []
        for elemID in part.elems:
            # get voxel input file index
            inpIndex = [str(part.elems[elemID].inpIndex)]
            # translate vertex index from IDString to input file index
            nodeIndex = []
            for nodeID in part.elems[elemID].vertices:
                nodeIndex += [str(part.nodes[nodeID].inpIndex)]
            
            code += [", ".join(inpIndex + nodeIndex)]
        
        footer = [elemEndToken,
                  "**"]

        return '\n'.join(header + code + footer)

    def addAllNset(self, part):
        # generate an all-nset information
        # parameter:
        #   part: the Part object (the model) to compile for
        # return:
        #   string that encodes a nset of all nodes

        header = ["**",
                  nsetStartToken,
                  "*Nset, nset=allNodes"]
        
        allNodes = [str(part.nodes[nodeID].inpIndex) for nodeID in part.nodes]
        code = self.listIINDEX_LEN(allNodes)

        footer = [nsetEndToken,
                  "**"]

        return '\n'.join(header + code + footer) 

    def addAllElset(self, part):
        # generate an all-elset information
        # parameter:
        #   part: the Part object (the model) to compile for
        # return:
        #   string that encodes an elset of all elements

        header = ["**",
                  elsetStartToken,
                  "*Elset, elset=allElems"]
        
        allElems = [str(part.elems[elemID].inpIndex) for elemID in part.elems]
        code = self.listIINDEX_LEN(allElems)

        footer = [elsetEndToken,
                  "**"]

        return '\n'.join(header + code + footer)
    
    def addSection(self, material):
        # generate an cross-section information
        # parameter:
        #   part: the Part object (the model) to compile for
        # return:
        #   string that describes the section material assignment

        code = ["**", 
                "**Section",
                "*Solid Section, elset=allElems, material=%s" % material.name,
                ","]

        return '\n'.join(code)
    
    ###########################################################################
    # for assembly
    ###########################################################################
    
    def addInstance(self, part):
        # generate an instance information
        # parameter:
        #   part: the Part object (the model) to compile for
        # return:
        #   string that describes the instance

        code = ["**",
                "*Instance, name=%s, part=%s" % (part.instanceName, part.id),
                "*End Instance",
                "**"]

        return '\n'.join(code)

    def addBCElsets(self, part):
        # generate elset information (to apply boundary conditions on)
        # parameter:
        #   part: the Part object (the model) to compile for
        # return:
        #   string that describes the elsets

        code = []
        # create elsets in the assembly (bot/top of beams/joins)
        iName = part.instanceName
        for elsetID in part.elsets:
            code += ["****Elset code start",
                     "*Elset, elset=%s, instance=%s" % (elsetID, iName)]
            
            # get input file index list for current element set
            elset = []
            for elemID in part.elsets[elsetID].elems:
                elset += [str(part.elems[elemID].inpIndex)]
            
            code += self.listIINDEX_LEN(elset)
            code += ["****Elset code end",
                     "**"]

        # create a whole body elset
        code += ["****Elset code start",
                 "*Elset, elset=WholeBody, instance=%s" % (iName)]
        whole = []
        for elemID in part.elems:
            whole += [str(part.elems[elemID].inpIndex)]
        
        code += self.listIINDEX_LEN(whole)
        code += ["****Elset code end",
                 "**"]

        return '\n'.join(code)

    def addBCNsets(self, part):
        # generate nset information (to apply boundary conditions on)
        # parameter:
        #   part: the Part object (the model) to compile for
        # return:
        #   string that describes the nsets

        code = []
        # create Nsets in the assembly
        iName = part.instanceName
        for nsetID in part.nsets:
            code += ["****Nset code start",
                     "*Nset, nset=%s, instance=%s" % (nsetID, iName)]
            
            # get input file index list for current element set
            nset = []
            for nodeID in part.nsets[nsetID].nodes:
                nset += [str(part.nodes[part.nMap[nodeID]].inpIndex)]
                
            code += self.listIINDEX_LEN(nset)
            code += ["****Nset code end",
                     "**"]
        
        return '\n'.join(code)

    ###########################################################################
    # for boundary conditions
    ###########################################################################
    
    def addGravity(self):
        # =======WARNING: POTENTIAL HARDCODING=======
        # generate gravity information for the whole simulation
        # return:
        #   string that describes the gravity

        code = ["**",
                "**Gravity",
                "*Amplitude, name=gravity, definition=EQUALLY SPACED, fixed interval=1",
                "             1.,             1.	",
                "**"]
        
        return '\n'.join(code)

    def addMaterial(self, material):
        # generate material information
        # parameters:
        #   material: the material to add to the input file
        # return:
        #   string that describes the material

        code = ["**",
                "**MATERIALS",
                "*Material, name=%s" % material.name,
                material.properties,
                "**"]
        
        return '\n'.join(code)

    def addFixed(self, part):
        # =======WARNING: POTENTIAL HARDCODING=======
        # generate fixed-node boundary condition information
        # parameter:
        #   part: the Part to compile for
        # return:
        #   string that describes the boundary condition

        header = ["**"]

        code = []
        for bc in part.boundaryConditions:
            if (bc.type != "fixed"): continue
            code += ["****BC code start",
                     "**Name: %s Type: Displacement/Rotation" % bc.name,
                     "*Boundary",
                     "%s, 1, 1" % bc.target,
                     "%s, 2, 2" % bc.target,
                     "%s, 3, 3" % bc.target,
                     "%s, 4, 4" % bc.target,
                     "%s, 5, 5" % bc.target,
                     "%s, 6, 6" % bc.target]

        footer = ["****BC code end",
                  "**"]
        
        return '\n'.join(header + code + footer)
    
    def addStressFields(self, part):
        # =======WARNING: POTENTIAL HARDCODING=======
        # generate stress field boundary condition information
        # parameter:
        #   part: the Part to compile for
        # return:
        #   string that describes the boundary condition

        header = ["**"]

        code = []
        for bc in part.boundaryConditions:
            if(bc.type != "stress"): continue
            # get stress field
            lx, ly, lz = bc.stress[0], bc.stress[1], bc.stress[2]
            mx, my, mz = bc.stress[3], bc.stress[4], bc.stress[5]
            # convert into code
            code += ["****BC code start",
                     "**Name: %s Type: Stress" % bc.name,
                     "*Initial Conditions, type=STRESS",
                     "%s, %f, %f, %f, %f, %f, %f" % (bc.target, lx, ly, lz, 
                                                                  mx, my, mz),
                     "****BC code end"]
        
        footer = ["**"]

        return '\n'.join(header + code + footer)