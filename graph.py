"""Provides a scripting component.
    Inpu=ts:
        x: The x script variable
        y: The y script variable
    Output:
        a: The a output variable"""
        
__author__ = "humph"
__version__ = "2019.04.08"
        
import math
import rhinoscriptsyntax as rs
        
epsilon = 0.0001

class Corner(object):
    def __init__(self, e0, e1, cen, ang):
        self.e0 = e0
        self.e1 = e1
        self.cen = cen
        self.ang = ang
    
    def split(self, maxAdd):
        
        add = min(int(round(self.ang / 90)), maxAdd + 1)
        angStep = self.ang / add
        
        edges = []
        for i in range(add + 1):
            newE = rs.RotateObject(self.e0, self.cen, angStep * i, copy=True)
            edges += [newE]
        
        newCorners = []
        for i in range(add):
            newC = Corner(edges[i], edges[i + 1], self.cen, angStep)
            newCorners += [newC]
        
        return newCorners
        
    def pt(self):
        
        #print(self.ang)
        e0ref = rs.RotateObject(self.e0, self.cen, self.ang * 0.01, copy=True)
        e0ref = rs.CurveMidPoint(e0ref)
        e0off = rs.OffsetCurve(self.e0, e0ref, wid * .5)
        
        e1ref = rs.RotateObject(self.e1, self.cen, self.ang * -0.01, copy=True)
        e1ref = rs.CurveMidPoint(e1ref)
        e1off = rs.OffsetCurve(self.e1, e1ref, wid * .5)
        
        intPt = rs.LineLineIntersection(e0off, e1off)[0]
        
        return intPt

class HalfedgeBase(object):
    def __init__(self, nxt=None, prev=None, twin=None, edge=None,
                 isOutline=False, isReal=True, face=None, vertex=None):
        self.next = nxt
        self.prev = prev
        self.twin = twin
        self.edge = edge
        self.isOutline = isOutline
        self.isReal = isReal
        self.face = face
        self.vertex = vertex
        
    def vec(self):
        
        return rs.VectorSubtract(self.twin.vertex.pt, self.vertex.pt)
    
    def render(self):
        
        edge = self.edge.edge
        base = self.vertex.pt
        rotated = rs.RotateObject(edge, base, -1, copy=True)
        refPt = rs.CurveMidPoint(rotated)
        off = rs.OffsetCurve(edge, refPt, 5)[0]
        
        cen = rs.CurveMidPoint(off)
        vec = rs.VectorUnitize(rs.PointSubtract(self.twin.vertex.pt, base))
        vec = rs.VectorScale(vec, 5)
        
        return cen, vec

class VertexBase(object):
    def __init__(self, pt=None, deg=None, halfedge=None):
        self.pt = pt
        self.deg = deg
        self.halfedge = halfedge
        
    def adjHalfedges(self):
        start = self.halfedge
        adj = [start]
        next = start.twin.next
        
        while next != start:
            adj += [next]
            next = next.twin.next
            
        return adj

class EdgeBase(object):
    def __init__(self, edge=None, halfedge=None, start=None, end=None, mid=None):
        self.edge = edge
        self.halfedge = halfedge
        self.start = start
        self.end = end
        self.mid = mid

class FaceBase(object):
    def __init__(self, face=None, area=None, halfedge=None):
        self.face = face
        self.halfedge = halfedge
        if(area == None): self.area = area
        else: self.area = rs.SurfaceArea(face)[0]

    def adjHalfedges(self):
        start = self.halfedge
        adj = [start]
        next = start.next
        
        while next != start:
            adj += [next]
            next = next.next
            
        return adj

class Halfedge(HalfedgeBase):
    def __init__(self, e, v, index):
        super(Halfedge, self).__init__(edge=e, vertex=v)
        self.index = index
    
    def __eq__(self, other):
        return isinstance(other, Halfedge) and self.index == other.index

class Vertex(VertexBase):
    def __init__(self, p, index):
        super(Vertex, self).__init__(pt=p)
        self.index = index
        self.corners = None
        self.srf = None
        self.axis = None
        
    def __eq__(self, other):
        return isinstance(other, Vertex) and self.index == other.index
        
    def setCorners(self, corners):
        
        # sort along curve
        pairs = []
        unitCir = rs.AddCircle(self.pt, 1)
        for c in corners: 
            # get points
            tar = c.pt()
            line = rs.AddLine(self.pt, tar)
            intPrm = rs.CurveCurveIntersection(unitCir, line)[0][5]
            pairs += [(tar, intPrm)]
            
        pairs = sorted(pairs, key=lambda x:x[1])
        self.corners = [pair[0] for pair in pairs]

class Edge(EdgeBase):
    def __init__(self, e, index):
        eStart = rs.CurveStartPoint(e)
        eEnd = rs.CurveEndPoint(e)
        eMid = rs.CurveMidPoint(e)
        super(Edge, self).__init__(edge=e, start=eStart, end=eEnd, mid=eMid)
        self.index = index
        self.corners = None
        self.srf = None
        self.actLen = 0
        
    def __eq__(self, other):
        return isinstance(other, Edge) and self.index == other.index

class Face(FaceBase):
    def __init__(self, f, index):
        super(Face, self).__init__(face=f)
        self.index = index
        
    def __eq__(self, other):
        return isinstance(other, Face) and self.index == other.index
    
class Graph(object):
    def __init__(self, edges):
        self.edges = edges
        self.vertices = None
        self.halfedges = None
        self.faces = None
        self.outline = None
        self.adjVE = {} # dict[vertexID] = [edgeID, edgeID, ...]
        self.adjEV = {} # dict[edgeID] = [vertexID, vertexID, ...]
        self.fixed = None
        self.wasted = None
        self.getVertices()
        self.getAdj()
        self.getHalfEdges()
        self.getFaces()
    
    def getVertices(self):
        
        # get unique vertices
        vertices = []
        for edge in self.edges:
            vertices += [rs.CurveStartPoint(edge), rs.CurveEndPoint(edge)]
        vertices = rs.CullDuplicatePoints(vertices, epsilon)
                
        # construct vertices
        self.vertices = []
        for i in range(len(vertices)):
            self.vertices += [Vertex(vertices[i], i)]
    
    def getAdj(self):
                
        VE, EV = {}, {}
        # get vertex-edge adjacency
        for i in range(len(self.vertices)):
                    
            v = self.vertices[i]
            # check edge-vertex adjacency
            for j in range(len(self.edges)):
                e = self.edges[j]
                CP = rs.EvaluateCurve(e, rs.CurveClosestPoint(e, v.pt))
                CPDist = rs.Distance(v.pt, CP)
                if(CPDist < epsilon):
                    VE[i] = VE.get(i, []) + [j]
                    EV[j] = EV.get(j, []) + [i]
                    
            # sort vertex adjaceny edges
            vCircle = rs.AddCircle(v.pt, 1)
            intPrm = [rs.CurveCurveIntersection(vCircle, edge)[0][5]\
                        for edge in [self.edges[eID] for eID in VE[i]]]
            pairs = [(VE[i][index], intPrm[index]) for index in range(len(VE[i]))]
            pairs = sorted(pairs, key=lambda x:x[1])
            VE[i] = [pairs[index][0] for index in range(len(intPrm))]
            v.deg = len(VE[i])
                    
        self.adjVE = VE
        self.adjEV = EV
    
    def getHalfEdges(self):
        
        HE, edges, heID = [], [], 0
        for i in range(len(self.edges)):
            # make edge
            e = self.edges[i]
            edge = Edge(e, i)
            # make halfedge
            points = [self.vertices[vID] for vID in self.adjEV[i]]
            he0 = Halfedge(edge, points[0], heID)
            he1 = Halfedge(edge, points[1], heID + 1)
            he0.twin, he1.twin = he1, he0
            heID += 2
            # assign halfedge
            points[0].halfedge = he0
            points[1].halfedge = he1
            edge.halfedge = he0
            # dump
            HE += [he0, he1]
            edges += [edge]
                
        for edge in edges:
            he0 = edge.halfedge
            he1 = edge.halfedge.twin
            # halfedge 0
            # find next edge
            he0NextAdj = self.adjVE[he1.vertex.index]
            he0NEID = (he0NextAdj.index(edge.index) - 1) % len(he0NextAdj)
            he0NEID = self.adjVE[he1.vertex.index][he0NEID]
            he0NE = edges[he0NEID]
            # find next halfedges
            he0NEhe0 = he0NE.halfedge
            he0NEhe1 = he0NEhe0.twin
            # find correct halfedge
            if(he0NEhe0.vertex == he1.vertex): 
                he0.next = he0NEhe0
                he0NEhe0.prev = he0
            else: 
                he0.next = he0NEhe1
                he0NEhe1.prev = he0
            # halfedge 1
            # find next edge
            he1NextAdj = self.adjVE[he0.vertex.index]
            he1NEID = (he1NextAdj.index(edge.index) - 1) % len(he1NextAdj)
            he1NEID = self.adjVE[he0.vertex.index][he1NEID]
            he1NE = edges[he1NEID]
            # find next halfedges
            he1NEhe0 = he1NE.halfedge
            he1NEhe1 = he1NEhe0.twin
            # find correct halfedge
            if(he1NEhe0.vertex == he0.vertex): 
                he1.next = he1NEhe0
                he1NEhe0.prev = he1
            else: 
                he1.next = he1NEhe1
                he1NEhe1.prev = he1
                    
        self.halfedges = HE
        self.edges = edges
    
    def getFaces(self):
        faces = self.recursiveFaces()
        # sort cells
        pairs = [(f, f.area) for f in faces]
        pairs = sorted(pairs, key=lambda x:x[1])
        faces = [pair[0] for pair in pairs]
        self.outline = faces.pop()
        self.faces = faces
        
        for he in self.outline.adjHalfedges():
            he.isOutline = True
    
    def recursiveFaces(self, heIndices=None, fIndex=0):
        if(heIndices == None): # input check
            heIndices = set(range(len(self.halfedges)))
                    
        if(len(heIndices) == 0): # base case
            return []
        else: # recursion case
            # get consecutive halfedges
            faceIDs = [heIndices.pop()]
            nextID = self.halfedges[faceIDs[0]].next.index
            while (nextID != faceIDs[0]):
                heIndices.remove(nextID)
                faceIDs += [nextID]
                nextID = self.halfedges[nextID].next.index
                        
            # convert into polyline surface
            corners = [self.halfedges[id].vertex.pt for id in faceIDs]
            corners += [corners[0]]
            outline = rs.AddPolyline(corners)
            face = Face(rs.AddPlanarSrf(outline)[0], fIndex)
            face.halfedge = self.halfedges[faceIDs[0]]
            for id in faceIDs: self.halfedges[id].face = face
            
            return [face] + self.recursiveFaces(heIndices, fIndex + 1)
    
    def getCorners(self):
        
        result = []
        # get joint corners
        for v in self.vertices:
            corners = []
            for he in v.adjHalfedges():
                # get angle
                vec0 = he.vec()
                vec1 = he.prev.twin.vec()
                vec0Ang = vectorAngleIn2D(vec0)
                vec1Ang = vectorAngleIn2D(vec1)
                ang = (vec1Ang - vec0Ang) % 360 # in degrees
                # construct corner
                newCorner = Corner(he.edge.edge, he.prev.edge.edge, he.vertex.pt, ang)
                corners += [newCorner]
                
            corners = self.expandCorners(corners)
            
            v.setCorners(corners)
            v.srf = rs.AddSrfPt(v.corners)
            
        # get edge corners
        for e in self.edges:
            corners = []
            # head
            vHead = e.halfedge.vertex
            cHead = vHead.corners
            for i in range(len(cHead)):
                line = rs.AddLine(cHead[i], cHead[(i + 1)% len(cHead)])
                intEvent = rs.CurveCurveIntersection(e.edge, line)
                if(intEvent != None):
                    corners += [cHead[i], cHead[(i + 1)% len(cHead)]]
            # tail
            vTail = e.halfedge.twin.vertex
            cTail = vTail.corners
            for i in range(len(cTail)):
                line = rs.AddLine(cTail[i], cTail[(i + 1)% len(cTail)])
                intEvent = rs.CurveCurveIntersection(e.edge, line)
                if(intEvent != None):
                    corners += [cTail[i], cTail[(i + 1)% len(cTail)]]
                    
            corners = corners[1:] + [corners[0]]
            corners.reverse()
            e.corners = corners
            e.srf = rs.AddSrfPt(e.corners)
    
    def expandCorners(self, corners, depth=0):
        if(len(corners) >= 4 or depth >= 4): # base case
            return corners
        # supplement, recursion case
        # find max
        maxC, maxA, maxIndex = corners[0], corners[0].ang, 0
        for i in range(len(corners)):
            c = corners[i]
            if(c.ang > maxA):
                maxC, maxA, maxIndex = c, c.ang, i
        #split max into multiple
        toAdd = 4 - len(corners)
        
        corners = corners[:maxIndex] + maxC.split(toAdd) + corners[maxIndex + 1:]
        
        return self.expandCorners(corners, depth + 1)
    
    def setActLens(self, actLens):
        
        for line in actLens:
            if(not ':' in line): continue
            items = line.split(':')
            try:
                id, actLen = int(items[0]), float(items[1])
                edgeLen = rs.CurveLength(self.edges[id].edge)
                setLen = abs(actLen * edgeLen)
                if(setLen != 0 and setLen < minActLen): setLen = minActLen
                
                if(actLen < 0): setLen *= -1
                
                self.edges[id].actLen = setLen
                print(actLen, setLen)
            except: continue
    
    def setWasted(self, edges):
        
        wasted = []
        for i in range(len(edges)): wasted += [Edge(edges[i], i)]
        self.wasted = wasted
    
    def setFixed(self, fixedID):
        
        if(isinstance(fixedID, int) or \
           (isinstance(fixedID, str) and fixedID.isnumeric())):
            self.fixed = int(fixedID)
        if(genMode == 0): # random
            nodes = [v.pt for v in self.vertices]
            self.fixed = rs.PointArrayClosestPoint(nodes, fixedID)
    
    def showHalfedges(self):
    
        base, vec = [], []
        for he in self.halfedges:
            
            pt, v = he.render()
            base += [pt]
            vec += [v]
            
        return base, vec
    
    def output(self):
        
        jointCens, jointSrfs, jointAxes = [], [], []
        edges, edgeSrfs, actLens = [], [], []
        cells = []
        
        for v in self.vertices:
            jointCens += [v.pt]
            jointSrfs += [v.srf]
            jointAxes += [v.axis]
        for e in self.edges:
            edges += [e.edge]
            edgeSrfs += [e.srf]
            actLens += [e.actLen]
        for f in self.faces:
            cells += [f.face]
        fixed = self.fixed
        
        return jointAxes, jointSrfs, jointCens, edges, edgeSrfs, actLens, cells, fixed

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
    if(yAbsAngle >= 90.): angle = -xAbsAngle
    else: angle = xAbsAngle
    
    return angle % 360.

def flattenCurves(curves):
            
    flat = []
            
    for curve in curves:
        if(curve == None): continue
        segments = rs.ExplodeCurves(curve)
        if(len(segments) == 0): segments = [curve]
        flat += segments
        
    return flat

def clean(edges):
    nodes = []
    for edge in edges:
        nodes += [rs.CurveStartPoint(edge), rs.CurveEndPoint(edge)]
    nodes = rs.CullDuplicatePoints(nodes, epsilon)
            
    nodes, edges, cleaned = cleanEdges(nodes, edges)
            
    return edges, cleaned

def cleanEdges(nodes, edges, cleaned=None):
    # written as a recursive function
    if(cleaned == None): cleaned = []
    # check degree
    nodesDegree = []
    # get node neighbor
    for node in nodes:
        neighbor = 0
        for edge in edges:
            start, end = rs.CurveStartPoint(edge), rs.CurveEndPoint(edge)
            dist = min(rs.Distance(start, node), rs.Distance(end, node))
            if(dist < epsilon): neighbor += 1
        nodesDegree += [neighbor]
            
    if(1 in nodesDegree or 0 in nodesDegree): clean = False
    else: clean = True
            
    if(clean):
        # base case
        return nodes, edges, cleaned
    else:
        # remove non-cleaned edges and vertices
        newNodes = []
        for i in range(len(nodes)):
            if(nodesDegree[i] <= 1):
                for j in range(len(edges)):
                    start, end = rs.CurveStartPoint(edges[j]), rs.CurveEndPoint(edges[j])
                    dist = min(rs.Distance(start, nodes[i]), rs.Distance(end, nodes[i]))
                    if(dist < epsilon):
                        cleaned += [edges.pop(j)]
                        break
            else: newNodes += [nodes[i]]
        # call recursion
        return cleanEdges(newNodes, edges, cleaned)

if(len(edges) != 0 and edges[0] != None):
    edgesFlat, edgesInvalid = clean(flattenCurves(edges))
    
    graph = Graph(edgesFlat)
    graph.setWasted(edgesInvalid)
    graph.setActLens(actLens)
    graph.setFixed(fixedID)
    graph.getCorners()
    
    jointAxes, jointSrfs, jointCens, edges, edgeSrfs, actLens, cells, fixed = \
        graph.output()