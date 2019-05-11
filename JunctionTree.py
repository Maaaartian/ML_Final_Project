import operator
from itertools import permutations
import numpy as np
import sys
import pandas as pd

rng = np.random

class junctionTree:

    def __init__(self, name):
        self.edges = edges
        self.grps = None
        self.V = None
        self.weightedEdges = None




    def edgeCount(self,edges):

        countEdges = {}
        for key,val in edges.items():
            countEdges[key] = len(val)

        sorted_d = sorted(countEdges.items(), key=operator.itemgetter(1))
        return sorted_d
    def findGroups(self):

        groups = {}
        num = 1

        sortedEd = self.edgeCount(self.edges)
        # print(sortedEd)

        while len(sortedEd)>0:
            temp = sortedEd.pop(0)

            # get node name and its neighbors
            node = temp[0]
            neigh = self.edges[node]
            if len(neigh)>0:
                # add new group node
                tempGrp = []
                tempGrp.append(node)
                tempGrp.extend(neigh)

                # add edge
                # nodesNum = len(neigh)
                # if nodesNum == 2:
                #     one = neigh[0]
                #     two = neigh[1]
                #
                #     if two not in self.edges[one]:
                #         self.edges[one].append(two)
                #         self.edges[two].append(one)

                for i in neigh:
                    for j in neigh:
                       if i != j:
                           if i not in self.edges[j]:
                               self.edges[i].append(j)
                               self.edges[j].append(i)





                groups[num] = tempGrp
                num+=1


                # remove this node from the edges
                self.edges.pop(node)

                # remove the current node from all its neighbor

                for i in neigh:
                    getList = self.edges[i]
                    getList.remove(node)
                    self.edges[i] = getList
            self.grps = groups
        return groups
    def maximalClique(self,eliminationClique):
        maximalClique = {}
        num = 1
        for k,eC in eliminationClique.items():
            isContained = False
            for k2,fC in eliminationClique.items():
                if set(eC).issubset(set(fC)) and len(eC) < len(fC):
                    isContained = True
            if not isContained:
                maximalClique[num] = eliminationClique[k]
                num+=1
        return maximalClique
    def weidhtedClique(self,maximalClique):
        graph = {}
        perm = permutations(maximalClique.keys(),2)

        for eachP in perm:
            n1 = maximalClique[eachP[0]]
            n2 = maximalClique[eachP[1]]

            tempWeight = 0
            for i in n1:
                if i in n2:
                    tempWeight+=1
            graph[(eachP[0],eachP[1])] = tempWeight

        return graph,maximalClique
    def pickMinWeightedKey(self,vertices,weights, isVisited):
        maxV = -10
        minKey = None
        for key in vertices:
            if weights[key] > maxV and isVisited[key] == False:
                maxV = weights[key]
                minKey = key
        # isVisited[minKey] = True

        return minKey
    def primsForMST(self,graph,edges,roots):
        print("Start --> Finding Maximum Spanning Tree")
        listOfV = [i for i in range(self.V)]
        # print(listOfV)
        weights = {}
        parent = {}
        isVisited = [False]*self.V

        for i in range(self.V):
            weights[i] = -1

        root = roots-1
        parent[root] = -1
        weights[root] = 0

        for cont in range(self.V):

            u= self.pickMinWeightedKey(listOfV,weights,isVisited)

            isVisited[u] = True

            for v in range(self.V):
                if graph[u][v] > 0 and isVisited[v] == False and weights[v] < graph[u][v]:
                    weights[v] = graph[u][v]
                    parent[v] = u

        print("Ended --> Finding Maximum Spanning Tree")
        print()
        return parent,weights











if __name__=='__main__':
    # edges = {"x1": ['x2', 'x3', 'x5'],
    #          "x2": ['x1', 'x3', 'x6', 'x4'],
    #          "x3": ['x1', 'x5', 'x2', 'x4'],
    #          "x4": ['x2', 'x3'],
    #          "x5": ['x1', 'x3'],
    #          "x6": ['x2']}

    # edges = {"x1": ['x2', 'x3'],
    #          "x2": ['x1','x4'],
    #          "x3" : ['x1','x4'],
    #          "x4" : ['x2','x3']}
    #

    # The way data looks like

    # Read the CSV File
    data = pd.read_csv("test_parameters.csv")
    data = data.values
    totalVertices = []
    # print(data)

#     Create the adjacency list from data

    edges = {}
    for i in range(len(data)):
        totalVertices.append(i)
        temp = []
        for j in range(len(data[0])):
            if data[i][j] != 0 and i!=j:
                temp.append(j)
        if len(temp)>0:
            edges[i] = temp
    # print(edges)

    # edges = {0 :  [1, 2, 5, 7, 8],
    #          1: [0, 3, 4, 9],
    #          2: [0, 3, 4, 5, 6, 8],
    #          3:[1, 2, 4],
    #          4:[1, 2, 3],
    #          5: [8, 0, 2],
    #          6:[2],
    #          7:[0],
    #          8:[0,2,5],
    #          9:[1]
    #
    #          }
#     # Initialization of Junction Tree (Instance)
    JT = junctionTree(edges)

    #
    # Find the cliques in the Graph
    grps =  JT.findGroups()
    print("Clique From Graph")
    print(grps)
    print()

    # Find Maximal Cliques
    maximalC = JT.maximalClique(grps)
    p = sorted(maximalC.items(),key = lambda c:len(c[1]))

    # Reassign the Keys of sorted Maximal Clique
    newMaximal = {}
    ct = 1
    for i in range(len(p)):
        newMaximal[ct] = p[i][1]
        ct+=1

    print("maximal Clique")
    print(newMaximal)
    print()

    # Assign root with the maximum elements which is at the end of the dict
    root = ct-1






    # unitTest = {1: [0, 5], 2: [4, 3]}
    weightedClique,clique = JT.weidhtedClique(newMaximal)


    # # print(clique)
    # # print(weightedClique)
    # # # JT.findWeights()
    #
    V = len(maximalC.keys())
    # print(V)
    JT.V = V
    # print(V)
    graph = [[0 for i in range(V)] for j in range(V)]
    for key,val in weightedClique.items():
        i = key[0]-1
        j = key[1]-1
        graph[i][j] = val
    #
    # Test Case
    # graph = [ [0, 2, 0, 6, 0],
    #         [2, 0, 3, 8, 5],
    #         [0, 3, 0, 0, 7],
    #         [6, 8, 0, 0, 9],
    #         [0, 5, 7, 9, 0]]
    #
    # testMst = {}
    # for i in range(len(graph)):
    #     for j in range(len(graph[0])):
    #         if graph[i][j] > 0:
    #             testMst[(i,j)] = graph[i][j]
    # print(testMst)
    print(clique)

    parent,weights = JT.primsForMST(graph,weightedClique,root)

    print(parent)
    pare = {}
    for key,val in parent.items():
        if val == -1:
            pare[key+1] = -1
        else:
            pare[key+1] = val+1
    print(pare)
    # parentNodes = []
    # childNodes = []
    # for i in range(1,len(pare)+1):
    #     # print(pare[i],"--->", i)
    #     if pare[i] not in  parentNodes and pare[i] != -1:
    #         parentNodes.append(pare[i])
    #
    print("Printing Junction Tree Instructions")
    print()

    eliminationQueue = []
    junctionTreeInTermsOfNodes = {}
    for key,val in pare.items():
        if val != -1:
            childSet = set(newMaximal[key])
            parentSet = set(newMaximal[val])
            # print(parentSet,childSet)
            commanEl = (parentSet.intersection(childSet))
            # junctionTreeInTermsOfNodes[list(childSet - commanEl)] = commanEl
            vertex = list(childSet - commanEl)

            for i in vertex:
                junctionTreeInTermsOfNodes[i] = list(commanEl)
            eliminationQueue.append(list(childSet - commanEl)[0])
            print(list(childSet - commanEl),"in terms of ",commanEl)
    print("sum of ", set(newMaximal[root]))
    for i in newMaximal[root]:
        eliminationQueue.append(i)
    print("Elimination Order",eliminationQueue)




    setRoot = set(newMaximal[root])
    for i in newMaximal[root]:
        vertex = set([i])
        parentSet = setRoot - vertex
        junctionTreeInTermsOfNodes[i] = list(parentSet)


    # print(totalVertices)

    for i in totalVertices:
        if junctionTreeInTermsOfNodes.get(i) == None:
            junctionTreeInTermsOfNodes[i] = []
    print("Junction Tree In Terms Of Nodes",junctionTreeInTermsOfNodes)


    #
    #
    #
    #
    #
    #



    # print(graph)

