__author__ = "Humphrey"
__version__ = "2019.02.02"

import rhinoscriptsyntax as rs
import math as m
import time
import copy

epsilon = 0.0001
nLen = 3 # digits to use in index coords
lineLen = 10 # maximum number of elements in a line in the input file

width = 7.2
height = 4
widthCount = 8
heightCount = 4
lengthCountFactor = 2
path = "C:/Users/humph/OneDrive/Desktop/CHI_design_tool/conduit.json"
genMode = 1
simulate = True
graph = 1

###############################################################################
# Model classes
###############################################################################

class IDString(object):
    def __init__(self):
        pass
    
    @staticmethod
    def getID(blockType, selfType, elemID, frame, row, layer):
        # format ABCCCDDDEEEFFF
        #   A -> element parent block type, 1:beam, 2:joint
        #   B -> element type, 0:node, 1:element (voxel), 2:nset, 3:elset
        # CCC -> element index
        # DDD -> frame index (length)
        # EEE -> row index (width)
        # FFF -> layer index (height)
        A   = str(blockType)
        B   = str(selfType)
        CCC = (nLen - len(str(elemID))) * '0' + str(elemID)
        DDD = (nLen - len(str(frame))) * '0' + str(frame)
        EEE = (nLen - len(str(row))) * '0' + str(row)
        FFF = (nLen - len(str(layer))) * '0' + str(layer)
        
        return A + B + CCC + DDD + EEE + FFF

    @staticmethod
    def breakID(id):
        # inverse function of getID, returns a dicitonary of information
        # parameter:
        #   IDString
        # return:
        #   a dictionary of informations as listed below
        
        information = {}
        information["blockType"]  = int(id[0])
        information["selfType"]  = int(id[1])
        information["index"] = int(id[2 + 0 * nLen: 2 + 1 * nLen])
        information["frame"] = int(id[2 + 1 * nLen: 2 + 2 * nLen])
        information["row"]   = int(id[2 + 2 * nLen: 2 + 3 * nLen])
        information["layer"] = int(id[2 + 3 * nLen:])
        
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
            assert(False)
        
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
        self.blockID = blockID # int, index of parent block

class Nset(PrimitiveSet):
    def __init__(self, name, nodes, blockID=None):
        super(Nset, self).__init__(name, len(nodes))
        self.nodes = nodes # list, IDStrings of nodes in this set
        self.blockID = blockID # int, index of parent block

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
        # nMap connects culled points to existing points
    
    def objectListToDict(self, objects):
        # convert element list into dictionarys
        # parameter:
        #   object: dict[IDString/name] = primitive/set
        # return:
        #   a dictionary of the versioan of the input objects
        
        # convert set to list
        if(isinstance(objects, set)): objects = list(objects)
        # check input type
        if (isinstance(objects[0], Primitive)):
            useID = True
        else: useID = False
        
        # convert to dictionary
        dictionary = {}
        for obj in objects:
            if (useID): dictionary[obj.id] = obj
            else: dictionary[obj.name] = obj
        
        return dictionary

class ElemBlock(object):
    def __init__(self, blockType, index, elemLenCount, elemWidCount, 
                 elemHeiCount):
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
        self.name = name
        self.properties = properties

###############################################################################
# Abaqus FEA solver classes
###############################################################################

###############################################################################
# SimuLearn specific classes
###############################################################################

class GridMesh(Part):
    def __init__(self, beams, joints, width, height):
        
        # input file info
        name = "grid2x2" # string, name of this gridmesh
        elemType = "C3D8" # string, the element type to use
        aName = "gridAssembly" # string, assembly name
        iName = "gridInstance" # string, instance name
        material = PLA() # Material, the material to use for this model
        super(GridMesh, self).__init__(beams + joints, name, elemType, 
                                       aName, iName, material)
        # blocks
        self.beams = beams # list, Beam objects in this grid
        self.joints = joints # list, Joint objects in this grid
        # geometry: dimensions
        self.width = width # float, beam width used for this grid
        self.height = height # float, beam height used for this grid
    
    def cullNodes(self, nodeSoup):
        # culls duplicated points and create an indexing map
        # parameter:
        #   nodeSoup: a list of Nodes to process
        # return:
        #   nodeSet: a set of unique nodes
        #   indexMap: a dictionary mapping an IDString to another
        
        indexMap = {}
        nodeInterface, nodeInterfaceID = [], []
        nodeSet = set()
        # joints first
        for node in nodeSoup:
            info = IDString.breakID(node.id)
            if ((info["blockType"] == 1 and not node.interface) or\
                info["blockType"] == 2):
                # new node, point to self
                nodeSet.add(node)
                indexMap[node.id] = node.id
                # add interfaces to a temporary list
                if(node.interface):
                    nodeInterface += [node.coor]
                    nodeInterfaceID += [node.id]
            else:
                # existing node, point to previously seen one
                CPID = rs.PointArrayClosestPoint(nodeInterface, node.coor)
                indexMap[node.id] = nodeInterfaceID[CPID]
        return nodeSet, indexMap
    
    def genNodes(self):
        # generate a clean set of nodes to itself and create an indexing map
        
        # create node soup
        nodeSoup = []
        for block in self.joints + self.beams:
            nodeSoup += block.genNodes()
        
        # cull points
        nSet, indexMap = self.cullNodes(nodeSoup)

        # convert to dictionary
        self.nodes = self.objectListToDict(nSet)
        self.nMap = indexMap

    def genElements(self):
        # generate elements: 8-vertices cubic voxel
        
        voxels = []
        for block in self.blocks:
            voxels += block.genVoxels(self.nMap)
        # convert to dictionary
        self.elems = self.objectListToDict(voxels)
    
    def genElsets(self):
        # generate element sets (with boundary conditions)
        
        # boundary condition: stress field
        elsets, bcs = [], []
        for block in self.blocks:
            newElsets, newBCs = block.genElsets() # TODO: need to consider returning more than 2 blocks
            
            # add to lists
            elsets += newElsets
            bcs += newBCs

        # convert to dictionary
        self.elsets = self.objectListToDict(elsets)
        self.boundaryConditions += bcs
    
    def genNsets(self):
        # generate nsets (with boundary conditions)
        
        # boundary condition: fixed points
        nsets, bcs = [], []
        for joint in self.joints:
            if(joint.fixed):
                nset = joint.genNset() # returns a list
                nsets += [nset]
                name = "fixed_" + nset.name
                bcs += [FixedBC(name, nset.name)]
        
        # convert to dictionary
        self.nsets = self.objectListToDict(nsets)
        self.boundaryConditions += bcs
    
    def shift(self):
        # shift the whole model to recenter it around the origin
        
        # find center
        for block in self.blocks:
            if(block.type == 1): continue # skip beams
            if(block.fixed):
                shift = rs.VectorReverse(block.cenCoor)
       
        # shift nodes
        for nodeID in self.nodes:
            self.nodes[nodeID].coor = rs.VectorAdd(self.nodes[nodeID].coor,
                                                  shift)
        
        # shift corner points
        for block in self.blocks:
            for i in range(len(block.corners)):
                block.corners[i] = rs.VectorAdd(block.corners[i], shift)
        
        # shift surfaces
        for block in self.blocks:
            block.surface = rs.MoveObject(block.surface, shift)
    
    def assignIndex(self):
        # assign input file indices to primitives
        
        # assign input file index to nodes
        i = 1
        for nodeID in self.nodes:
            self.nodes[nodeID].inpIndex = i
            i += 1

        # assign input file index to elements
        i = 1
        for elemID in self.elems:
            self.elems[elemID].inpIndex = i
            i += 1
    
    def genModelInfo(self):
        # generate the model design information for the input file
        # return:
        #   the model design as a string
        
        # TODO: need to change this implementation to also save sampling frame index (along the length)
        
        # vertices information
        vHeader = ["Joints",
                   "vertexID, x, y, z"]
        vCode = []
        for joint in self.joints:
            vCode += ["%d, %f, %f, %f" % (joint.id, 
                                          joint.cenCoor[0],
                                          joint.cenCoor[1],
                                          joint.cenCoor[2])]
        # edge information
        bHeader = ["Beams",
                   "beamID, startVertexID, endVertexID, actStress, conStress, "\
                   "actuatorLength(posUp_negBot), actuator first end, "\
                   "actuator second end"]
        bCode = []
        for beam in self.beams:
            if (beam.side): sideFactor = 1.
            else: sideFactor = -1.
            bCode += ["%d, %d, %d, %f, %f, %f, %d, %d" % (beam.id,
                                                      beam.startID,
                                                      beam.endID,
                                                      beam.actStress,
                                                      beam.conStress,
                                                      beam.actLen * sideFactor,
                                                      beam.actStart,
                                                      beam.actEnd)]
        
        return vHeader + vCode + bHeader + bCode

class PlanarBlock(ElemBlock):
    def __init__(self, blockType, id, voxLenCount, voxWidCount, voxHeiCount,
                 width, height, srf):
        super(PlanarBlock, self).__init__(blockType, id, voxLenCount, 
                                          voxWidCount, voxHeiCount)
        # geometry: 3D information
        self.corners = [] # list of Point3D
        self.surface = srf # surface, planar surface of this block
        self.width = width # float, width of this block
        self.height = height # float, height of this block
    
    def genSurface(self):
        # generate the 2D surface of the block from its corner points
        
        self.surface = orientSrf(rs.AddSrfPt(self.corners))

    def genNodes(self):
        # generate the nodes used by this block
        # return:
        #   a list of Nodes of this block
        
        nodes = []
        for i in range(self.nLen):
            u = i / self.eLen
            for j in range(self.nWid):
                v = j / self.eWid
                
                # get reference point coordinates
                coor = rs.SurfaceParameter(self.surface, (u, v))
                refPt = rs.EvaluateSurface(self.surface, coor[0], coor[1])
                
                # compose a node for each layer height
                for k in range(self.nHeight):
                    # get node informaiton
                    z = (k / self.eHeight - .5) * self.height
                    pt = rs.VectorCreate([refPt[0], refPt[1], z], 
                                         [0., 0., 0.])
                    index = IDString.getID(self.type, 0, self.id, i, j, k) # 0:n

                    # save node
                    nodes += [Node(index, pt)]
                    self.nodes += [index]
        
        return nodes

    def genVoxels(self, nMap):
        # generate the voxels for this block
        # parameter:
        #   nMap: a dictionary the maps an node IDString to another
        # return:
        #   a list of Voxels
        
        voxels = []
        t = self.type
        bID = self.id

        for i in range(self.eLen):
            for j in range(self.eWid):
                for k in range(self.eHeight):
                    # get voxel vertices in ccw-bottom-up order
                    vRawID = [IDString.getID(t, 0, bID, i + 1, j + 1, k    ),
                              IDString.getID(t, 0, bID, i    , j + 1, k    ),
                              IDString.getID(t, 0, bID, i    , j    , k    ),
                              IDString.getID(t, 0, bID, i + 1, j    , k    ),
                              IDString.getID(t, 0, bID, i + 1, j + 1, k + 1),
                              IDString.getID(t, 0, bID, i    , j + 1, k + 1),
                              IDString.getID(t, 0, bID, i    , j    , k + 1),
                              IDString.getID(t, 0, bID, i + 1, j    , k + 1)]
                    vertices = [nMap[rawID] for rawID in vRawID]
                    # construct element
                    index = IDString.getID(t, 1, bID, i, j, k) # 1:voxel
                    # save element
                    voxels += [Voxel(index, vertices)]
                    self.elems += [index]
        
        return voxels

    def genNset(self):
        # generate node sets for this block (likely joints)
        # return:
        #   an Nset for this block (for fixed-end assignments)
        
        cenNodes = []

        for nodeID in self.nodes:
            info = IDString.breakID(nodeID)
            layer = info["layer"]
            if((layer - (self.eHeight / 2)) < epsilon): cenNodes += [nodeID]

        return Nset("fixedCenter", cenNodes, self.id)

class Beam(PlanarBlock):
    def __init__(self, id, elemLenCount, elemWidCount, elemHeiCount, width,
                 height, edge, srf, actStress, conStress, actLen, startID, endID):
        super(Beam, self).__init__(1, id, elemLenCount, elemWidCount,
                                    elemHeiCount, width, height, srf)
        # TODO: need to change implemntation to take actuator length as input and stress as a constant
        
        # geometry: design informaiton
        self.startID = startID # the start vertex (joint) index of this beam
        self.endID = endID # the end vertex (joint) index of this beam
        # geometry: skeleton
        self.edge = None # curve, center line of this beam
        self.start = None # Point3D, start point of edge
        self.end = None # Point3D, end point of edge
        self.mid = None # Point3D, mid point of edge
        self.vec = None # Vector3D, normalized vector from start to end
        self.angle = None # float(0-360), angle w.r.t. the x vector
        self.updateSkeleton(edge)
        # geometry: actuator
        self.actLen = None # the length of the actuator
        self.side = None # boolean, true if stress is at top side of the beam
        self.actStartPt = None # Point3D, start of the actuator (on skeleton)
        self.actEndPt = None # Point3D, end of the actuator (on skeleton)
        self.actStart = None # int, the actuator start frame index
        self.actEnd = None # int, the actuator end frame index
        self.updateActuator(actLen)
        
        # TODO: need to change implementation (stress field should be a constant)
        # initial condition: stress field
        self.actStress = actStress # float, stress value along beam direction
        self.conStress = conStress # float, stress value along beam direction
    
    def updateSkeleton(self, edge):
        # update geometrical information for this beam
        # parameter:
        #   edge: line, the edge assigned to this beam
        
        self.edge = edge
        self.start = rs.CurveStartPoint(edge)
        self.end = rs.CurveEndPoint(edge)
        self.mid = rs.CurveMidPoint(edge)
        self.vec = rs.VectorUnitize(rs.VectorCreate(self.end, self.start))
        self.angle = vectorAngleIn2D(self.vec)
    
    def updateActuator(self, actLen):
        # update the actuator info of this beam (orienting from local to global
        # coordinate system)
        # parameter:
        #   a float as the length of the actuator
        
        self.actLen = abs(actLen)
        if(actLen >= 0): side = True # positive, stress to top side
        else: side = False # negative, stress to bottom side
        self.side = side
        
        # get end points of actuator skeleton
        shiftVec = rs.VectorScale(self.vec, self.actLen * .5)
        self.actStartPt = rs.PointAdd(self.mid, -shiftVec)
        self.actEndPt = rs.PointAdd(self.mid, shiftVec)
    
    def sortCorners(self):
        # sort the corner points into order
        
        ordered = [None, None, None, None]
        edgeVec = rs.VectorCreate(self.end, self.start)
        norm = rs.VectorUnitize([edgeVec[1], -1 * edgeVec[0], 0])
        
        for pt in self.corners:
            # check side
            refVec = rs.VectorCreate(pt, self.mid)
            if(rs.VectorDotProduct(norm, refVec) >= 0): side = 0
            else: side = 1
            
            # check end
            startDist = rs.Distance(pt, self.start)
            endDist = rs.Distance(pt, self.end)
            if(startDist <= endDist): end = 0
            else: end = 1
            
            # fill pt into order
            if(side == 0 and end == 0): ordered[0] = pt
            elif(side == 0 and end == 1): ordered[1] = pt
            elif(side == 1 and end == 1): ordered[2] = pt
            elif(side == 1 and end == 0): ordered[3] = pt
            
        self.corners = ordered
    
    def genNodes(self):
        # generate the nodes of this beam
        # return:
        #   a list of Nodes
        
        # get node count along the longitudinal direction
        beamLen = rs.SurfaceArea(self.surface)[0] / self.width
        lengthCount = beamLen / ((self.width / self.eWid) * lengthCountFactor)
        self.eLen = int(int(round(lengthCount / 2)) * 2)
        self.nLen = self.eLen + 1
        
        # generate nodes
        nodes = super(Beam, self).genNodes()
        
        startMinDist, endMinDist = None, None
        startMinFrame, endMinFrame = None, None
        for node in nodes:
            # check for interface nodes
            info = IDString.breakID(node.id)
            localCoor = (info["frame"], info["row"], info["layer"])
            node.interface = self.nodeIsInterface(localCoor)
            
            # check actuator ends
            startDist = rs.Distance(self.actStartPt, node.coor)
            endDist = rs.Distance(self.actEndPt, node.coor)
            if(startMinDist == None or startDist < startMinDist):
                startMinDist = startDist
                startMinFrame = IDString.breakID(node.id)["frame"]
            if(endMinDist == None or endDist < endMinDist):
                endMinDist = endDist
                endMinFrame = IDString.breakID(node.id)["frame"]
        
        if(startMinFrame > endMinFrame): # check frame order
            startMinFrame, endMinFrame = endMinFrame, startMinFrame
        self.actStart = max(startMinFrame, 1)
        self.actEnd = min(endMinFrame, self.nLen - 2)
        return nodes
    
    def nodeIsInterface(self, coor): 
        # checking if a node is at the interface to another block
        # parameters:
        #   coor: coor as a tuple
        # return:
        #   a boolean indicating the interface conditions
        
        lenInterface = coor[0] == 0 or coor[0] == (self.nLen - 1)
        # widInterface = coor[1] == 0 or coor[1] == (self.nWid - 1)
        # heiInterface = coor[2] == 0 or coor[2] == (self.nHeight - 1)

        return lenInterface
    
    def orientStress(self, stress, add):
        # orients a planar stress field from the beam's local to global space
        # parameter:
        #   stress: the stress field (six numbers) at its local space
        #   add: float, 
        #   [sigma11, sigma22, sigma33, sigma12, sigma13, sigma23]
        # return:
        #   the oriented stress field in global space
        
        # theta is the negative value of the beam.angle (w.r.t. x-axis)
        # sigma11 = .5 * stress + .5 * stress * (2 * ((m.cos(theta)) ** 2) - 1)
        
        ang = -1 * (self.angle + add)
        sigma11 = stress * (m.cos(m.radians(ang)) ** 2)
        sigma22 = stress - sigma11
        sigma12 = -1. * stress * (m.sin(m.radians(ang)) * m.cos(m.radians(ang)))
        
        return [sigma11, sigma22, 0., sigma12, 0., 0.]
    
    def genElsets(self):
        # generate the element sets and initial conditions in this block 
        # (actuator and constraint, with and without intial stress)
        # return:
        #   a list of new Elsets and a list of new BCs
        
        elemsTop, elemsBot = [], []
        halfHeight = int(self.eHeight / 2)
        
        # assign voxels to top/bottom
        for elemID in self.elems:
            info = IDString.breakID(elemID)
            
            # identify bottom / top
            layerIndex = info["layer"]
            if (layerIndex >= halfHeight): elemsTop += [elemID]
            else: elemsBot += [elemID]
        
        # assign to actuator/constraint group
        elemsAct, elemsCon, elemsToCheck = [], [], []
        if (self.side): elemsConSide, elemsActSide = elemsBot, elemsTop # top act
        else: elemsConSide, elemsActSide = elemsTop, elemsBot # bot act
        
        for elemID in elemsConSide:
           info = IDString.breakID(elemID)
           rowIndex = info["row"]
           if(rowIndex == 0 or rowIndex == (self.eWid - 1)): # shell
               elemsAct += [elemID]
           else: # infill
               elemsCon += [elemID]
        
        # check if an element is between the actuator interval
        for elemID in elemsActSide:
            info = IDString.breakID(elemID)
            frameIndex = info["frame"]
            rowIndex = info["row"]
            # assign to actuator or constraint group
            if(rowIndex == 0 or rowIndex == (self.eWid - 1)): # shell
                elemsAct += [elemID]
            elif(self.actEnd > frameIndex >= self.actStart): # actuator
                elemsAct += [elemID]
            else: # infill
                elemsCon += [elemID]
        
        # construct elsets
        actElsetName = "%s_%d_act" % ("beam", self.id)
        conElsetName = "%s_%d_con" % ("beam", self.id)
        actElset = Elset(actElsetName, elemsAct, self.id)
        conElset = Elset(conElsetName, elemsCon, self.id)
        
        # construct initial conditions
        actStress = self.orientStress(self.actStress, 0)
        conStress = self.orientStress(self.conStress, 90)
        actBCName = "stress_" + actElsetName
        conBCName = "stress_" + conElsetName
        actStrBC = StressFieldBC(actBCName, actElsetName, actStress)
        conStrBC = StressFieldBC(conBCName, conElsetName, conStress)
        
        return [actElset, conElset], [actStrBC, conStrBC]

class Joint(PlanarBlock):
    def __init__(self, id, voxLenCount, voxWidCount, voxHeiCount,
                 width, height, center, srf, stress, fixed, edges):
         super(Joint, self).__init__(2, id, voxLenCount, voxWidCount, 
                                     voxHeiCount, width, height, srf)
         # geometry: skeleton
         self.center = center # Point3D, center point of this joint
         # boundary condition: fixed joint
         self.fixed = fixed # boolean, indicate wheter this joint is the center
         self.cenCoor = None # Point3D, central node at the middle center
         self.edges = edges # list of indices points to the connected beams
         self.stress = stress # float, stress value

    def sortCorners(self):
        # sort the corner points in CCW order
        
        # sort along a circle at the joint
        prms = []
        circle = rs.AddCircle(self.center, self.width * .5)
        # get curve CP parameter
        for pt in self.corners:
            prm = rs.CurveClosestPoint(circle, pt)
            prms += [prm]
        # sort with curve CP parameter
        temp = [pt for pt in self.corners]
        pairs = zip(temp, prms)
        pairs.sort(key = lambda pair: pair[1])
        result = [pair[0] for pair in pairs]
        self.corners = result

    def genNodes(self):
        # generate the nodes of this joint
        # return:
        #   a list of Nodes
        
        # generate nodes
        nodes = super(Joint, self).genNodes()

        # check for interface nodes
        for node in nodes:
            info = IDString.breakID(node.id)
            localCoor = (info["frame"], info["row"], info["layer"])
            node.interface = self.nIsInterface(localCoor)
        
        # get center coordinate, 0:node
        cenID = IDString.getID(self.type, 0, self.id, int(self.eLen / 2),
                                                      int(self.eWid / 2),
                                                      int(self.eHeight / 2))
        for node in nodes:
            if(node.id == cenID):
                self.cenCoor = node.coor
        
        return nodes

    def nIsInterface(self, coor):
        # checking if a node is at the interface to another block
        # parameters:
        #   coor: coor as a tuple
        # return:
        #   a boolean indicating the interface conditions
        
        # coor as a tuple
        lenInterface = coor[0] == 0 or coor[0] == (self.nLen - 1)
        widInterface = coor[1] == 0 or coor[1] == (self.nWid - 1)
        # heiInterface = coor[2] == 0 or coor[2] == (self.nHeight - 1)

        return lenInterface or widInterface

    def genElsets(self):
        # generate the element sets in this block (single, no stress)
        # return:
        #   a list of new elsets and a list of new initial conditions (stress)
        
        # construct elsets
        ID = "%s_%d_con" % ("joint", self.id)
        elsetAll = Elset(ID, self.elems, self.id)
        
        # assign initial condition
        stress = [self.stress, self.stress, 0., 0., 0., 0.]
        stressName = "stress_" + ID
        jointStrBC = StressFieldBC(stressName, elsetAll.name, stress)
        
        return [elsetAll], [jointStrBC]

###############################################################################
# Utilities
###############################################################################

def orientSrf(srf):
    # orient a surface so that its normal always points up
    # parameter:
    #   the surface to orient
    # return:
    #   the oriented surface with its normal pointing at positive Z
    
    uDom = rs.SurfaceDomain(srf, 0)
    vDom = rs.SurfaceDomain(srf, 1)
    uCen = (uDom[0] + uDom[1]) * .5
    vCen = (vDom[0] + vDom[1]) * .5
    
    norm = rs.VectorUnitize(rs.SurfaceNormal(srf, [uCen, vCen]))
    dotProd = rs.VectorDotProduct(norm, [0., 0., 1.])
    if(dotProd < 0.): rs.FlipSurface(srf, True)

    return srf

def doubleOffset(edge, width):
    # offset and edge on both sides ( in the xy plane)
    # parameters:
    #   edge: the edge (as a curve or line) to offset
    #   width: the distance of offset
    # return:
    #   two edges on both sides, offsetted with the specified width
    
    # get vector
    start = rs.CurveStartPoint(edge)
    end = rs.CurveEndPoint(edge)
    vec = rs.VectorUnitize(rs.VectorCreate(end, start))
    # get offset direction reference point
    normal = [vec[1], -vec[0], 0.]
    refRight = rs.VectorAdd(start, normal)
    refLeft = rs.VectorAdd(start, rs.VectorReverse(normal))
    
    # offset curve
    edgeRight = rs.OffsetCurve(edge, refRight, width * .5)
    edgeLeft  = rs.OffsetCurve(edge,  refLeft, width * .5)

    return edgeRight, edgeLeft # both are lists

def vectorAngleIn2D(vector):
    # get the angle of a vector w.r.t. the global x-axis
    # parameter:
    #   the vector to measure
    # return:
    #   the angle w.r.t. the globla x-axis in degrees
    
    # check the angle against x and y vectors
    unitVec = rs.VectorUnitize(vector)
    xAbsAngle = rs.VectorAngle([1., 0., 0.], unitVec)
    yAbsAngle = rs.VectorAngle([0., 1., 0.], unitVec)
    
    # check posi-negativity
    if(yAbsAngle >=90.): angle = -xAbsAngle
    else: angle = xAbsAngle
    
    return angle % 360.

def outputElements(gridMesh):
    # output the elements as boxes
    # parameter:
    #   the GridMesh model to output
    # return:
    #   a list of boxes representing the elements of the model
    
    # cubic voxel elements
    output = [rs.AddBox([gridMesh.nodes[nodeID].coor \
                        for nodeID in gridMesh.elems[elemID].vertices]) \
                        for elemID in gridMesh.elems]
    
    return output

def outputElsets(gridMesh):
    # output the element sets as groups of boxes
    # parameter:
    #   the GridMesh model to output
    # return:
    #   a list of boxes representing the elements of the model
    
    # cubic voxel elements
    elems = [rs.AddBox([gridMesh.nodes[nodeID].coor \
                        for nodeID in gridMesh.elems[voxID].vertices])\
                        for voxID in gridMesh.elems]
    elemIDs = [elemID for elemID in gridMesh.elems]
    elemsMap = {}
    for i in range(len(elemIDs)): elemsMap[elemIDs[i]] = elems[i]
    
    act, con = [], []
    for elsetID in gridMesh.elsets:
        setElems = [elemsMap[elemID] for elemID in gridMesh.elsets[elsetID].elems]
        if(elsetID.endswith("con")): con += setElems
        elif(elsetID.endswith("act")): act += setElems
            
    return act, con

def outputNodes(gridMesh):
    # output the elements as points
    # parameter:
    #   the GridMesh model to output
    # return:
    #   a list of Point3D representing the nodes of the model
    
    output = [gridMesh.nodes[nodeID].coor for nodeID in gridMesh.nodes]
    
    return output

def outputBeamSrfs(gridMesh):
    # output the beams as 2D surfaces
    # parameter:
    #   the GridMesh model to output
    # return:
    #   a list of surfaces representing the beams of the model
    
    output = [block.surface for block in gridMesh.beams]
    
    return output

def outputJointSrfs(gridMesh):
    # output the joints as 2D surfaces
    # parameter:
    #   the GridMesh model to output
    # return:
    #   a list of surfaces representing the joints of the model
    
    output = [block.surface for block in gridMesh.joints]
    
    return output

def outputCorners(gridMesh):
    # output the corner points of the model as points
    # parameter:
    #   the GridMesh model to output
    # return:
    #   a list of points representing the block corners of the model
    output = []
    for block in gridMesh.blocks:
        output += block.corners
    #output = [rs.AddPolyline(block.corners + [block.corners[0]]) \
    #          for block in gridMesh.blocks]
    
    return output

###############################################################################
# json
###############################################################################

def getJSON(graph):
    
    # find center
    center = [0, 0, 0]
    
    json = {}
    json["adjMat"] = getAdjMat(graph)
    json["nodes"] = getNodes(graph, center)
    json["edges"] = getEdges(graph, center)
    
    return json

def getAdjMat(graph):
    
    adjMat = [[0] * len(graph.vertices) for i in range(len(graph.edges))]
    for edge in graphs.adjEV:
        for vertex in graphs.adjEV[edge]:
            adjMat[edge][vertex] = 1
    
    return adjMat

def getNodes(graph, center):
    
    nodes = []
    for node in graph.vertices:
        nodeVec = getNodeVec(node, center)
        nodes += [nodeVec]
    
    return nodes

def getEdges(graph):
    
    edges = []
    for edge in graph.edges:
        edgeVec = getEdgeVec(edge, center)
        edges += [edgeVec]
        
    return edges

def getEdgeVec(edge):
    
    # get geometry
    # get center
    # get three frames
    
    # get stress fields
    
    # get bc
    # get actuator length assignment
    # get vector pointing to center
    
    return [42]

def getNodeVec(node, center):
    
    # get geometry
    # get center
    # get eight corners
    
    # get bc
    # get fixed end condition
    # get vector pointing to center
    
    return 42

###############################################################################
# Main
###############################################################################

def genBlocks(graph, actStr, conStr, w, h, wCount, hCount):
    # generate a GridMesh model with the input edges and joints
    # parameters:
    #   jointCens: list of Point3D, the vertices of the graph
    #   edges: list of lines, the edges of the graph
    #   cells: list of surfaces, the surrounded cells of the graph
    #   fixedID: int, the index of hte fixed joint in the graph
    #   bStr: float, the stress field (local) assigned to the beams
    #   bLen: list of floats, the actuator length assignment (pos-top, neg-bot)
    #   w: float, the width of the beams
    #   h: float, the height of the beams
    #   wCount: int, the number of elements along the width direction of beams
    #   hCount: int, the number of elements along the height direction of beams
    # return:
    #   a GridMesh model constructed with the input graph
    
    beams, joints = [], []
    # construct beams
    lCount = -1 # initialize as -1
    vertices = [v.pt for v in graph.vertices]
    
    for i in range(len(graph.edges)):
        e = graph.edges[i]
        # get start and end point index
        startID = rs.PointArrayClosestPoint(vertices, e.start)
        endID = rs.PointArrayClosestPoint(vertices, e.end)
        edge = rs.AddLine(e.start, e.end)
        srf = rs.AddSrfPt(e.corners)
        beams += [Beam(i, lCount, wCount, hCount, w, h, edge, srf,
                       actStr, conStr, e.actLen, startID, endID)]
        
    # construct joints
    lCount = wCount # same element count on length and width directions
    for i in range(len(graph.vertices)):
        v = graph.vertices[i]
        # check if the joint is the fixed joint
        isFixed = False
        if (i == graph.fixed):
            isFixed = True
        cen = v.pt
        srf = rs.AddSrfPt(v.corners)
        adj = graph.adjVE[i]
        joints += [Joint(i, lCount, wCount, hCount, w, h, v.pt, srf,
                         conStress, isFixed, adj)]
    
    return GridMesh(beams, joints, w, h)

def genSim(gridMesh):
    # generate an input file for Abaqus with the input informaiton
    # parameters:
    #   gridMesh: the GridMesh to generate input file for
    # reuturns:
    #   beamSrf: a list of 2D surfaces of the beams
    #   jointSrt: a list of 2D surfaces of hte joints
    
    # generate elements in the part
    # gridMesh.genBlockSrfs() # assign surface
    if(simulate):
        #=======================================================================
        json = {}
        gridMesh.genNodes() # generate nodes and index map
        gridMesh.genElements() # generate voxels
        gridMesh.genElsets() # generate element sets
    
    return beamSrf, jointSrf

actStress = 0.158
conStress = 0.118

if((genMode == 0 or genMode == 1) and graph != None):
    savePath = path
    gridMesh = genBlocks(graph, actStress, conStress, width, height, widthCount,
                         heightCount)
    beamSrfs, jointSrfs = genSim(gridMesh, savePath)