# Valeria Gomez - 261136869

# My program crashes and becomes unresponsive as soon as I do just a few operations on the cube. This is why the object I 
# submitted lacks creativity and doesn't look like much. However, I tried to include as many operations as I could before the 
# program crashed (bevel, extrude, inset, split edge, erase edge, and flip edge). I hope this is enough to show my code works :)

import numpy as np

# Nones initially and updated in HalfedgeMesh.build(), since we only have the vertex positions and face vertex indices
class Vertex:
    def __init__(self, point):
        self.point = point
        self.halfedge = None
        self.index = None

class Halfedge:
    def __init__(self):
        self.vertex = None # source vertex --> vertex we started at NOT where we point to
        self.twin = None
        self.next = None
        self.prev = None # previous halfedge
        self.edge = None
        self.face = None
        self.index = None

class Edge:
    def __init__(self):
        self.halfedge = None # any of the two
        self.index = None

class Face:
    def __init__(self, vertices):
        self.vertices = vertices
        self.halfedge = None
        self.index = None

class HalfedgeMesh:
    def __init__(self, vertices, faces):
        self.vertices = np.array([Vertex(pos) for pos in vertices])
        self.halfedges = []
        self.edges = []
        self.faces = []
        for face_vertex_ids in faces: # for each face in the 'faces' array
            face_vertices = [self.vertices[id] for id in face_vertex_ids] 
                # for each vertex ID in this face, join them in an array and store under 
                # faces_vertices
            self.faces.append(Face(face_vertices))  # create a Face class with that set of vertices 
        self.build()

    # Convenience functions to create new elements
    def new_vertex(self, point):
        vertex = Vertex(point)
        vertex.index = len(self.vertices)
        self.vertices = np.append(self.vertices, vertex)
        return vertex

    def new_face(self, vertices):
        face = Face(vertices)
        face.index = len(self.faces)
        self.faces = np.append(self.faces, face)
        return face

    def new_edge(self):
        edge = Edge()
        edge.index = len(self.edges)
        self.edges = np.append(self.edges, edge)
        return edge

    def new_halfedge(self):
        he = Halfedge()
        he.index = len(self.halfedges)
        self.halfedges = np.append(self.halfedges, he)
        return he

    '''
    Given HalfedgeMesh object (potentially with quads or ngons), return a tuple of numpy arrays (vertices, triangles, 
        triangle_to_face) for rendering.
    vertices: (n, 3) array of vertex positions
    triangles: (m, 3) array of vertex indices forming triangles
    triangle_to_face: (m,) array of face indices corresponding to each triangle (needed for face selection especially) 
        [tri_index] -> face_index
    '''
    def get_vertices_and_triangles(self):
        vertices = [vertex.point for vertex in self.vertices]
        triangles = []
        triangle_to_face = [] # map from triangle to face, {to generalize to n-gons}
        i = 0
        for face in self.faces:
            if len(face.vertices) == 3:
                triangles.append([vertex.index for vertex in face.vertices])
                triangle_to_face.append(i)
            else:
                # implement simple ear clipping algorithm
                triangles_vertices = triangulate([vertex for vertex in face.vertices])
                for triangle_vertices_triple in triangles_vertices:
                    triangles.append(triangle_vertices_triple)
                    triangle_to_face.append(i)
            i += 1
        return np.array(vertices), np.array(triangles), triangle_to_face


    def find_twin(self, he):
        source = he.vertex.index
        destination = he.next.vertex.index
        twin = None

        for halfedge in self.halfedges :
            if (halfedge.vertex.index == destination) and (halfedge.next.vertex.index == source) :
                twin = halfedge;
                break
        
        return twin
    
    def halfEdgeSetup(self, curFace, n):

        firstID = -1
        lastID = -1
        
        for i in range(0, n) :  # Repeat the process for each edge necessary
            curHEdge = self.new_halfedge()
            curHEdge.face = curFace                 # setting the .face attribute
            curHEdge.vertex = curFace.vertices[i]   # setting the .vertex attribute (source vertex)         
            curFace.vertices[i].halfedge = curHEdge # setting up the .halfedge attribute (for the vertex)
            newEdge = self.new_edge()               # initializing a new edge
            np.append(self.edges, newEdge)          # appending the edge to self.edges  
            newEdge.halfedge = curHEdge             # setting the .halfedge attribute (for the edge)
            curHEdge.edge = newEdge                 # setting the .edge attribute 
            np.append(self.halfedges, curHEdge)     # append this halfedge in the self.halfedges array

            if (i == 0): 
                curFace.halfedge = curHEdge         # setting up the .halfedge attribute (for curface)
                firstID = curHEdge.index

            if (i == n-1):
                lastID = curHEdge.index
            
        # Linking all halfedges from curface together
        faceHalfedges = self.halfedges[firstID : lastID + 1]

        for i in range(len(faceHalfedges)):
            curHEdge = faceHalfedges[i]
            if i == len(faceHalfedges) - 1 :
                curHEdge.next = faceHalfedges[0]
                curHEdge.prev = faceHalfedges[i-1]
            elif i == 0 :
                curHEdge.prev = faceHalfedges[-1]
                curHEdge.next = faceHalfedges[i+1]
            else :
                curHEdge.next = faceHalfedges[i+1]
                curHEdge.prev = faceHalfedges[i-1]


    def edgeIndicesBuilder (self, curFace, n) :
        # Creating the indices for each edge of a face with n vertices, returns them in form of a dictionary
        edgesDict = {}
        edgeArray = []

        for i in range(0, n):
            if i == n-1 :
                curEdge = [curFace.vertices[i].index, curFace.vertices[0].index] # the final edge loops back to the beginning
                edgeArray.append(curEdge)
            else :
                curEdge = [curFace.vertices[i].index, curFace.vertices[i+1].index] # each edge connects the current vertex with the next
                edgeArray.append(curEdge)

        edgesDict [curFace.index] = edgeArray


    # Build the halfedge data structure from the vertex positions and face vertex indices stored in self.vertices and self.faces
    # This is essential for all following objectives to work
    def build(self):
        # TODO: Objective 1: build the halfedge data structure

        self.halfedges = []

        for vertex in self.vertices : 
            # initialize the indices of each vertex
            vertex.index = np.where(self.vertices == vertex) [0][0] # index of the vertex = its index on the array

        for i in range(len(self.faces)) : # iterating over each face in self.faces
            curFace = self.faces[i]
            curFace.index = i
    
            # Count how many vertices this face has
            numOfVerts = len(curFace.vertices)

            # Create edges between vertices of the same face and store the edges inside the dictionary
            self.edgeIndicesBuilder (curFace, numOfVerts)

            # Setting up the half-edges for each face (depending on the face's number of vertices)
            self.halfEdgeSetup(curFace, numOfVerts)

        # Using my helper function to find the twin for each half edge
        for i in range(len(self.halfedges)): 
            curHedge = self.halfedges[i]
            curTwin = self.find_twin(curHedge)
            curHedge.twin = curTwin
            curTwin.twin = curHedge

        # DEBUGGING
        # print(edgesDict)
        # for i in range(len(self.halfedges)): 
        #     print(i , " -- halfedge index: ", self.halfedges[i].index)
        #     print(i , " -- source vertex : ", self.halfedges[i].vertex.index)
        #     print(i , " -- destination vertex : " , self.halfedges[i].next.vertex.index)
        #     print(i , " -- twin : ", self.halfedges[i].twin.index)
        #     print(i , " -- face : ", self.halfedges[i].face.index)
        #     print()
        # self.sanity_check()
        pass

    # Given a face, loop over its halfedges he in order to update face.vertices and he.face after some operation
    def update_he_vertices_around_face(self, face):
        he = face.halfedge
        vertices = []
        while True:
            vertices.append(he.vertex)
            he = he.next
            he.face = face # update he face
            if he.index == face.halfedge.index:
                break
        face.vertices = vertices # update face vertices

    # Given an edge, with both sides being triangles, flip the edge
    #           v1                              v1
    #           /\                              /\
    #          /  \                            / |\
    #         /    \                          /  | \
    #        /______\                        /   |  \
    #          edge      -> flip edge ->         |
    #        \      /                        \   |  /
    #         \    /                          \  | /
    #          \  /                            \ |/
    #           \/                              \/

    def flip_edge(self, edge):
        # TODO: Objective 3a: flip the edge (only if both sides are triangles)

        face1 = edge.halfedge.face
        face2 = edge.halfedge.twin.face
        
        if (len(face1.vertices) == 3) and (len(face2.vertices) == 3) :
            hEdge1 = edge.halfedge
            hEdge1_prev = hEdge1.prev
            hEdge1_next = hEdge1.next

            e = hEdge1.edge

            hEdge2 = edge.halfedge.twin
            hEdge2_prev = hEdge2.prev
            hEdge2_next = hEdge2.next

            # We will end up creating a new halfedge / edge from x to y
            y = hEdge2.prev.vertex  
            x = hEdge1.prev.vertex  

            # Updating the halfedge attributes
            hEdge1.vertex = y 
            hEdge1.next = hEdge1_prev
            hEdge1.prev = hEdge2_next
            hEdge1.prev.face = face1
            face1.halfedge = hEdge1

            y.halfedge = hEdge1

            hEdge2.vertex = x
            hEdge2.next = hEdge2_prev
            hEdge2.prev = hEdge1_next
            hEdge2.prev.face = face2
            face2.halfedge = hEdge2
            hEdge2.edge = e

            x.halfedge = hEdge2

            # Updating the face.vertices attributes
            face1_1 = face1.halfedge.vertex
            face1_2 = face1.halfedge.next.vertex
            face1_3 = face1.halfedge.prev.vertex
            face1.vertices = [face1_1, face1_2, face1_3]

            face2_1 = face2.halfedge.vertex
            face2_2 = face2.halfedge.next.vertex
            face2_3 = face2.halfedge.prev.vertex
            face2.vertices = [face2_1, face2_2, face2_3]

            self.update_indices()
            self.sanity_check()
        else :
            pass


    # Given an edge, with both sides being triangles, split the edge in the middle, creating a new vertex and connecting
    # it to the facing corners of the 2 triangles
    #           v1                              v1
    #           /\                              /\
    #          /  \                            / |\
    #         /    \                          /  | \
    #        /______\                        /   |  \
    #          edge      -> split edge ->    ---v2---
    #        \      /                        \   |  /
    #         \    /                          \  | /
    #          \  /                            \ |/
    #           \/                              \/

    def findMidpoint(self, p0, p1):
        midX = (p0[0] + p1[0]) / 2
        midY = (p0[1] + p1[1]) / 2
        midZ = (p0[2] + p1[2]) / 2
        return [midX, midY, midZ]

    def split_edge(self, edge):
        # TODO: Objective 3b: split the edge (only if both sides are triangles)

        face1 = edge.halfedge.face
        face2 = edge.halfedge.twin.face
        
        if (len(face1.vertices) == 3) and (len(face2.vertices) == 3) :
            selectedHEdge = edge.halfedge

            # Identifying the vertices in the selected edge
            p0 = selectedHEdge.vertex
            p0Coord = p0.point
            p1 = selectedHEdge.twin.vertex
            p1Coord = p1.point

            # Identifying the two other vertices (in the corners of the polygon)
            a = selectedHEdge.prev.vertex
            b = selectedHEdge.twin.prev.vertex

            # Calculating the midpoint of the selected edge
            midpoint = self.findMidpoint(p0Coord, p1Coord)
            mid = self.new_vertex(midpoint)
            
            # Modifying the two current faces
            face1.vertices = [p0, mid, a]
            face2.vertices = [p0, mid, b]

            # Creating two new faces
            face3 = self.new_face([b, p1, mid])
            face4 = self.new_face([a, mid, p1])

            # Creating the 3 new edges 
            e1 = self.new_edge()
            e2 = self.new_edge()
            e3 = self.new_edge()
            
            
            # Creating new halfedges

            hEdge11 = self.new_halfedge()   # b --> mid
            hEdge11.vertex = b
            hEdge11.prev = selectedHEdge.twin.next
            hEdge11.edge = e1
            hEdge11.face = face2
            
            hEdge12 = self.new_halfedge()   # mid --> b
            hEdge12.vertex = mid
            hEdge12.next = selectedHEdge.twin.prev
            hEdge12.edge = e1
            hEdge12.face = face3

            hEdge11.twin = hEdge12
            hEdge12.twin = hEdge11

            hEdge21 = self.new_halfedge()   # p1 --> mid
            hEdge21.vertex = p1
            hEdge21.next = hEdge12
            hEdge21.prev = hEdge12.next
            hEdge21.edge = e2
            hEdge21.face = face3

            hEdge22 = self.new_halfedge()   # mid --> p1
            hEdge22.vertex = mid
            hEdge22.next = selectedHEdge.next 
            hEdge22.edge = e2
            hEdge22.face = face4

            hEdge21.twin = hEdge22
            hEdge22.twin = hEdge21

            hEdge31 = self.new_halfedge()   # a --> mid
            hEdge31.vertex = a
            hEdge31.next = hEdge22
            hEdge31.prev = hEdge22.next
            hEdge31.edge = e3
            hEdge31.face = face4

            hEdge32 = self.new_halfedge()   # mid --> a
            hEdge32.vertex = mid
            hEdge32.next = selectedHEdge.prev
            hEdge32.prev = selectedHEdge
            hEdge32.edge = e3
            hEdge32.face = face1

            hEdge31.twin = hEdge32
            hEdge32.twin = hEdge31


            # Modifying selected halfedge and its twin
            selectedHEdge.next  = hEdge32
            selectedHEdge.face = face1

            selectedHEdge.twin.vertex = mid
            selectedHEdge.twin.prev = hEdge11
            selectedHEdge.twin.face = face2

            # Filling out some attributes for other halfedges 
            hEdge11.next = selectedHEdge.twin
            hEdge12.prev = hEdge21
            hEdge22.prev = hEdge31
            
            # Setting up the attributes for the new edges 
            e1.halfedge = hEdge12
            e2.halfedge = hEdge22
            e3.halfedge = hEdge32

            # Setting up the attributes for the new faces 
            face1.halfedge = selectedHEdge
            face2.halfedge = hEdge11
            face3.halfedge = hEdge21
            face4.halfedge = hEdge31

            # Setting up the halfedge for the new middle vertex 
            mid.halfedge = selectedHEdge.twin

            self.update_indices()
            
            # print("p0 : ", p0.index)
            # print("p1 : ", p1.index)
            # print("A : ", a.index)
            # print("B : ", b.index)

            # print("face1.vertices")
            # for vertex in face1.vertices :
            #     print(vertex.index)
                
            # print("face2.vertices")
            # for vertex in face1.vertices :
            #     print(vertex.index)

            # print("face3.vertices")
            # for vertex in face1.vertices :
            #     print(vertex.index)

            # print("face4.vertices")
            # for vertex in face1.vertices :
            self.sanity_check()

        else : 
            pass

    # Given an edge, dissolve (erase) the edge, merging the 2 faces on its sides
    def erase_edge(self, edge):
        # TODO: Objective 3c: erase the edge
        hEdge = edge.halfedge
        hEdge1 = hEdge.next 
        hEdge2 = hEdge.prev
        hEdge3 = hEdge.twin.next
        hEdge4 = hEdge.twin.prev
        face1 = hEdge.face
        face2 = hEdge.twin.face
        verticesNew = face1.vertices

        # print("verticesNew (face1.vertices) : ")
        for vertex in verticesNew:
            print(vertex.index)

        for vertex in face2.vertices :
            if not (vertex in verticesNew):     # add the vertices from face 2 to verticesNew (avoid repeating)
                verticesNew.append(vertex)
                # print("vertex in f2 not in f1 : ", vertex.index)
        
        # print("verticesNew (all edges) : ")
        # for vertex in verticesNew:
        #     print(vertex.index)
        
        # face1.vertices = verticesNew
        
        # print("face1.vertices : ")
        # print("face1.vertices.length : ", len(face1.vertices))
        # for vertex in face1.vertices:
        #     print(vertex.index)
        
        # print("end of erase_edge()")
        # Linking all the remaining half-edges together
        hEdge1.next = hEdge2
        hEdge2.next = hEdge3
        hEdge3.next = hEdge4
        hEdge4.next = hEdge1
        hEdge1.prev = hEdge4
        hEdge2.prev = hEdge1
        hEdge3.prev = hEdge2
        hEdge4.prev = hEdge3  
        # Making them all point to face1 (the one we won't delete)
        hEdge1.face = face1
        hEdge2.face = face1
        hEdge3.face = face1
        hEdge4.face = face1

        self.faces = np.delete(self.faces, face2.index)
        self.edges = np.delete(self.edges, edge.index)
        self.halfedges = np.delete(self.halfedges, hEdge.index)
        self.update_indices()
        self.halfedges = np.delete(self.halfedges, hEdge.twin.index)
        self.update_indices()

        start = face1.halfedge
        cur = face1.halfedge.next
        orderedVerts = [start.vertex]
        while cur.vertex.index != start.vertex.index :
            # print("orderedVert :")
            # print(v for v in orderedVerts)
            orderedVerts.append(cur.vertex)
            cur = cur.next

        face1.vertices = orderedVerts
        self.update_indices()
        self.sanity_check()

    def duplicate_hEdge(self, h) : 
        duplicatedH = self.new_halfedge()
        duplicatedH.edge = h.edge
        duplicatedH.face = h.face
        duplicatedH.next = h.next
        duplicatedH.prev = h.prev
        duplicatedH.twin = h.twin
        duplicatedH.vertex = h.vertex
        return duplicatedH


    # Since extrude_face, inset_face, bevel_face, all update connectivity in the exact same way, implement this topological
    # operation as a separate function (no need to call it inside the other functions, that's already done)
    def extrude_face_topological(self, face):
        # TODO: Objective 4a: implement the extrude operation,
        # Note that for each edge of the face, you need to create a new face (and accordingly new halfedges, edges, vertices)
        # Hint: Count the number of elements you need to create per each new face, Creating them all before updating connectivity may make it easier

        vertices = face.vertices
        newVertices = []
        edgesBetweenVerts = []  # Edges delimiting the new face
        connectingEdges = []    # Between A and A'
        newFaceHedges = []
        connectingHedges = []

        sideFaces = []

        numberOfVerts = len(vertices)
        # print("numberOfVerts : ", numberOfVerts)    # good -- 3

        startingHedge = face.halfedge
        oldFaceHedges = []

        for i in range(numberOfVerts):
            oldFaceHedges.append(startingHedge)
            startingHedge = startingHedge.next 

        duplicatedOldFaceHedges = []
        for h in oldFaceHedges:
            duplicatedOldFaceHedges.append(self.duplicate_hEdge(h))

        for i in range(numberOfVerts) :
            newVertices.append(self.new_vertex(vertices[i].point))  # Duplicating the vertices of the current face

            # For a face of i vertices, create a total of 2*i new edges
            edgesBetweenVerts.append(self.new_edge())    # Creating a new edge for each of these new vertices (to connect to each other)
            connectingEdges.append(self.new_edge())    # Creating a new edge for each vertex (to connect to the other face)

        # print("There are this many vertices after duplicating : ", len(self.vertices))  # displays good value -- 11
        # for v in self.vertices :
        #     print(v.index)
        # print()

        # For a face of i vertices, there are 4*i new halfedges to be added
        for i in range(0, 2 * numberOfVerts):
            newFaceHedges.append(self.new_halfedge())       # 2*i halfedges for the new i-sided face we created
            connectingHedges.append(self.new_halfedge())    # 2*i halfedges to connect the old vertices to the new ones

        topFace = self.new_face(newVertices)    # Creating a new face connecting the new vertices
        topFaceID = topFace.index

        # Creating the halfedges 
        for i in range(numberOfVerts):      
            a = vertices[i]
            # print("a.index : ", a.index)
            aPrime = newVertices[i]
            # print("aPrime.index : ", aPrime.index)

            if i == (numberOfVerts - 1) :
                b = vertices[0]
                bPrime = newVertices[0]
            else : 
                b = vertices[i+1]
                bPrime = newVertices[i+1]

            # print("b.index : ", b.index)
            # print("bPrime.index : ", bPrime.index)
            
            # Creating a face (the side faces that connect the old and new face)
            newSideFace = self.new_face([aPrime, a, bPrime, b])
            sideFaces.append(newSideFace)

            e1 = connectingEdges[i]                  # edge between A and A'
            hEdge11 = connectingHedges[2*i]          # picking the one halfedge from the ones we have created
            # print("the following are edges in connectingHedges : ")
            # print("hEdge11.index : ",  hEdge11.index)
            hEdge11.vertex = a                       # setting the starting vertex as A (old face)
            hEdge12 = connectingHedges[(2*i) + 1]    # picking the another halfedge from the ones we have created
            # print("hEdge12.index : ",  hEdge12.index)
            hEdge12.vertex = aPrime                  # setting the starting vertex as A' (new face)
            hEdge12.face = newSideFace
            newSideFace.halfedge = hEdge12

            e1.halfedge = hEdge11
            hEdge11.edge = e1
            hEdge12.edge = e1

            hEdgeToA = None
            hEdgeFromA = None
            for h in duplicatedOldFaceHedges :
                if h.vertex.index == a.index :
                    hEdgeFromA = h
                if h.next.vertex.index == a.index :
                    hEdgeToA = h
            
            hEdge11.prev = hEdgeToA
            hEdgeToA.next = hEdge11
            hEdge12.next = hEdgeFromA
            hEdgeFromA.prev = hEdge12
            hEdgeFromA.face = hEdge12.face

            hEdge11.twin = hEdge12
            hEdge12.twin = hEdge11

            # Creating halfedges between A' and one of its neighbors
            e2 = edgesBetweenVerts[i]   # edge from A' --> B'
            # print("the following are edges in newFaceHedges : ")
            hEdge21 = newFaceHedges[2*i]
            # print("hEdge21.index : ",  hEdge21.index)
            hEdge21.vertex = aPrime
            aPrime.halfedge = hEdge21
            # hEdge21.next = 
            # hEdge21.prev = 
            hEdge21.face = topFace
            hEdge21.edge = e2
            if (i==0):
                topFace.halfedge = hEdge21

            e2.halfedge = hEdge21

            hEdge22 = newFaceHedges[(2*i) + 1]  # edge from B' --> A'
            # print("hEdge22.index : ",  hEdge22.index)
            hEdge22.vertex = bPrime
            bPrime.halfedge = hEdge22
            hEdge22.edge = e2
            hEdge22.face = newSideFace
            hEdge22.next = hEdge12
            hEdge12.prev = hEdge22

            hEdge21.twin = hEdge22 
            hEdge22.twin = hEdge21

        # Updating the remaining next / previous / twin attributes for the halfedges
        for h in connectingHedges : 
            if h.next == None : 
                if h.face == None :
                    curFace = h.prev.face
                    # print("h.prev : ", h.prev.vertex.index,  ", ",  h.vertex.index)
                    h.face = curFace
                else : 
                    curFace = h.face

                # print("h.index : ", h.index)
                # print("h.face : ", h.face.index)
                curStart = h.vertex # A
                destinationIndex = -1

                # print("curStart.index : ", curStart.index)

                faceHEdge = curFace.halfedge
                nextDestination = faceHEdge.vertex   # C'

                # print(" -- (finding match) curFace.vertices :")
                # for v in curFace.vertices:
                #     print(v.index)

                for i in range(len(curFace.vertices)):  # Find the destination vertex of the current halfedge
                    curV = curFace.vertices[i]
                    # print("curV.index : ", curV.index)
                    if curV.index == curStart.index :
                        destinationIndex = i-1  # This works because of the way I have implemented the face.vertices array
                        break
                
                curDestination = curFace.vertices[destinationIndex] # A'
                match = None

                # print("\ntrying to find the match for the .next attribute : ")
                for hEdge in newFaceHedges + connectingHedges : 
                    # print("hEdge.vertex.index (", curDestination.index, ") : ", hEdge.vertex.index)
                    # print("hEdge.twin.vertex.index (", nextDestination.index, ") : ", hEdge.twin.vertex.index)
                    # print()
                    if (hEdge.vertex.index == curDestination.index) and (hEdge.twin.vertex.index == nextDestination.index):
                        # print("there's a match!")
                        match = hEdge
                        break
                
                if (match.face.index == curFace.index):
                    h.next = match

                faceHEdge.prev = match

        for cur in newFaceHedges :  # Connecting the halfedges in the top face (duplicated vertices)
            if (cur.face.index == topFaceID) :  # Needed or not ??
                curStart = cur.vertex
                hEdgeFromStart = None
                hEdgeToStart = None

                for h in newFaceHedges: # looking at every 
                    if (curStart.index == h.vertex.index) and ((h.face.index == topFaceID)) : 
                        hEdgeFromStart = h
                    if (curStart.index == h.vertex.index) and ((h.face.index != topFaceID)) :
                        hEdgeToStart = h.twin
                
                hEdgeFromStart.prev = hEdgeToStart
                hEdgeToStart.next = hEdgeFromStart

        for h in newFaceHedges: 
            if h.prev == None : 
                curFace = h.face
                firstHedge = curFace.halfedge
                h.prev = firstHedge.next.next

        for sideFace in sideFaces:
            start = sideFace.halfedge
            cur = sideFace.halfedge.next
            orderedVerts = [start.vertex]
            while cur.vertex.index != start.vertex.index :
                # print("orderedVert :")
                # print(v for v in orderedVerts)
                orderedVerts.append(cur.vertex)
                cur = cur.next
            sideFace.vertices = orderedVerts


    def inset_face(self, face, t): # t=0, no inset, t=1, full inset (face shrinks to barycenter)
        # TODO: Objective 4b: implement the inset operation,
        numberOfVerts = len(face.vertices)
        faceToModify = self.faces[-numberOfVerts - 1]   # This is the face created by the extrude_face_topological function

        barycenter = np.mean([vertex.point for vertex in face.vertices], axis=0)
        for vertex in faceToModify.vertices:
            vertex.point = vertex.point * (1 - t) + barycenter * t
        pass


    def extrude_face(self, face, t): # t=0, no extrusion, t<0, inwards, t>0 outwards
        # TODO: Objective 4b: implement the extrude operation,
        # Translate the selected face along its normal

        numberOfVerts = len(face.vertices)
        faceToModify = self.faces[-numberOfVerts - 1]   # This is the face created by the extrude_face_topological function

        # Calculate the normal of the face using three vertices to compute two vectors
        p0 = face.vertices[0].point
        p1 = face.vertices[1].point
        p2 = face.vertices[2].point

        v0 = p1 - p0
        v1 = p2 - p0

        n = np.cross(v0, v1)
        n = n / np.linalg.norm(n)

        for vertex in faceToModify.vertices:
            vertex.point = vertex.point + n * t

    def bevel_face(self, face, tx, ty): # ty for the normal extrusion, tx for the scaling (tangentially)
        # TODO: Objective 4B: implement the bevel operation,

        numberOfVerts = len(face.vertices)
        faceToModify = self.faces[-numberOfVerts - 1]   # This is the face created by the extrude_face_topological function

        # Calculate the normal of the face using three vertices to compute two vectors
        p0 = face.vertices[0].point
        p1 = face.vertices[1].point
        p2 = face.vertices[2].point

        v0 = p1 - p0
        v1 = p2 - p0

        n = np.cross(v0, v1)
        n = n / np.linalg.norm(n)

        for vertex in faceToModify.vertices:
            vertex.point = vertex.point * (1 - ty) + n * tx

    def scale_face(self, face, t): # t=0, no inset, t=1, full inset (face shrinks to barycenter)
        barycenter = np.mean([vertex.point for vertex in face.vertices], axis=0)
        for vertex in face.vertices:
            vertex.point = vertex.point * (1 - t) + barycenter * t

    # need to update HalfedgeMesh indices after deleting elements
    def update_indices(self):
        for i, vertex in enumerate(self.vertices):
            vertex.index = i
        for i, face in enumerate(self.faces):
            face.index = i
        for i, edge in enumerate(self.edges):
            edge.index = i
        for i, he in enumerate(self.halfedges):
            he.index = i

    # Helpful for debugging after each local operation
    def sanity_check(self):
        for i,f in enumerate(self.faces):
            if f.halfedge is None:
                print('Face {} has no halfedge'.format(i))
            if f.index != i:
                print('Face {} has wrong index'.format(i))
            for v in f.vertices:
                if v is None:
                    print('Face {} has a None vertex'.format(i))

        for i,e in enumerate(self.edges):
            if e.halfedge is None:
                print('Edge {} has no halfedge'.format(i))
            if e.index != i:
                print('Edge {} has wrong index'.format(i))

        for i,v in enumerate(self.vertices):
            if v.halfedge is None:
                print('Vertex {} has no halfedge'.format(i))
            if v.index != i:
                print('Vertex {} has wrong index'.format(i))

        for i,he in enumerate(self.halfedges):
            if he.vertex is None:
                print('Halfedge {} has no vertex'.format(i))
            if he.index != i:
                print('Halfedge {} has wrong index'.format(i))
            if he.face is None:
                print('Halfedge {} has no face'.format(i))
            if he.edge is None:
                print('Halfedge {} has no edge'.format(i))
            if he.next is None:
                print('Halfedge {} has no next'.format(i))
            if he.prev is None:
                print('Halfedge {} has no prev'.format(i))
            if he.twin is None:
                print('Halfedge {} has no twin'.format(i))

# THIS JUST RETURNS THE VERTICES, DOESN'T CHANGE FACES OR ANYTHING
def triangulate(vertices):
    start = vertices[0]
    returnArray = []

    # print("start of triangulate")
    # print("vertices array : ")
    # for v in vertices:
    #     print(v.index)

    for i in range(0, len(vertices), 1):
        # print("loop number : ", i)
        
        v2 = vertices[i+1]
        # print("v2 : ", v2.index)

        if v2.index == vertices[-1].index :
            break
            
        v3 = vertices[i+2]
        # print("v3 : ", v3.index)

        curTriangle = [start.index, v2.index, v3.index]
        # print("curTriangle : ")
        # for vertex in curTriangle:
        #     print(vertex)
        returnArray.append(curTriangle)

    # for triplet in returnArray:
    #     print(triplet[0])
    #     print(triplet[1])
    #     print(triplet[2])
    #     print()
    # print("end of triangulate")

    return returnArray