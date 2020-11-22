import numpy as np
import json
import math

# dataSet = np.loadtxt('../data/test.txt')
# dataSet = dataSet.astype(int)

dataSet = np.array([
    [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1],
    [3, 3, 3, 1, 3, 3, 1, 1, 3, 3, 3, 1],
    [2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2],
    [2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 2],
    [2, 3, 2, 3, 2, 2, 3, 3, 1, 2, 3, 1],
    [2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2]])

with open('../data/deDomain.txt') as f:
    domain = json.load(f)

with open('../data/dataDesc.txt') as f:
    desc = json.load(f)

# print attributes of data
headerConstant = []
for atr in desc:
    headerConstant.append(str(atr[0]))
print(headerConstant)

# print trainset
dataSet = np.transpose(dataSet)
print(dataSet)

# calculate entropy of Attributes


def findEntropyAttri(attributeColumn, trainSet, header):
    count = {}
    atriEntropy = 0

    for row in trainSet:
        label = row[attributeColumn]
        if label not in count:
            count[label] = 0
        count[label] += 1

    print(count)

    for label in count:
        # print("label", label, "has", count[label], "customers")
        entropyEle = findEntropyElement(label, attributeColumn, trainSet)
        atriEntropy += calcEntropyAttributes(
            count[label], len(trainSet), entropyEle)
        # print("---------------------------------")

    print("Attribute of", header[attributeColumn],
          "has entropy of", atriEntropy, )
    # print("---------------------------------")
    return atriEntropy

# finding the entropy for currentLabel


def findEntropyElement(currentLabel, attributeColumn, trainSet):
    customers = []
    countClassLabelSet = {}

    # Find cutomers match with current currentLabel
    for row in trainSet:
        label = row[attributeColumn]
        if label == currentLabel:
            customers.append(row)

    countClassLabelSet = countClassLabel(customers)

    # find entropy of current currentLabel
    entropy = 0
    for label in countClassLabelSet:
        entropy += calcEntropy(countClassLabelSet[label], len(customers))

    print("entropy of currentLabel(label)", currentLabel, "is", entropy)
    return entropy

# Find Entropy of log base 2


def calcEntropy(numerator, denominator):
    entropy = (-1)*(numerator/denominator)*(math.log2(numerator/denominator))
    return entropy

# Find entropy for current Attricute


def calcEntropyAttributes(numerator, denominator, entropyEle):
    entropy = (numerator/denominator)*entropyEle
    return entropy

# find number of H&L Risk in dataset


def countClassLabel(trainSet):
    countClassLabel = {}
    riskColumn = 0
    for row in trainSet:
        label = row[riskColumn]
        if label not in countClassLabel:
            countClassLabel[label] = 0
        countClassLabel[label] += 1

    return countClassLabel

# Find Entropy of S


def findEntropy_S(trainSet):
    entropyS = 0
    countClassLabelDict = countClassLabel(trainSet)
    for label in countClassLabelDict:
        # print("Risk", label, "has", countClassLabelDict[label], "customers")
        entropyS += calcEntropy(countClassLabelDict[label], len(trainSet))

    print("Entorpy of S is", entropyS)
    print("----------------------------")
    return entropyS

# find best attribute as a node


def findNode(trainset, header):
    entropyS = findEntropy_S(trainset)
    nodeAttri = ""
    highestGain = 0
    if entropyS == 0:
        nodeAttri = "RISK"
        return nodeAttri

    i = 1  # i=0 is Risk column
    while i < len(header):  # itterate attri column
        print("Attribute of", header[i])
        atriEntropy = findEntropyAttri(i, trainset, header)
        Gain = entropyS - atriEntropy
        print("Attribute of", header[i], "has Gain of", Gain)
        print("<><><><><><><><><><><><><><><><>")
        if Gain > highestGain:
            highestGain = Gain
            nodeAttri = header[i]

        i += 1
    if len(header) == 1:
        return "RISK"
    elif highestGain == 0:
        return header[1]
    return nodeAttri


def nodeData(trainSet, header):
    node = findNode(trainSet, header)
    print(" ")
    print("<<<========== select", node, "as a node ==========>>>")
    attriRemove = header.index(node)
    count = {}
    final = []
    for row in trainSet:
        label = row[attriRemove]
        count[str(label)] = None

    final.append(count)
    final.insert(0, node)
    print(final)

    return final


class TreeNode:
    def __init__(self, attriName, elements):
        self.attriName = attriName
        self.children = elements
        self.parent = None

    def add_child(self, child, label):
        child.parent = self
        if child.attriName == "RISK":
            if "1" in child.children:
                self.children[label] = 1
            elif "2" in child.children:
                self.children[label] = 2
        else:
            self.children[label] = child


def buildTree(trainSet, header, result):
    newBranches = nodeData(trainSet, header)
    newNode = TreeNode(newBranches[0], newBranches[1])
    print(newNode.attriName, " , ", newNode.children)

    headerCopy = header.copy()
    newTrainSet = []
    for label in newNode.children:
        print("Current Node", newBranches[0], "'s label ", int(label))
        if newNode.children[label] == None:

            # Find new trainSet for sub branchs
            for row in trainSet:
                currentLabel = row[headerCopy.index(newNode.attriName)]
                if int(label) == currentLabel:
                    row = np.delete(row, headerCopy.index(newNode.attriName))
                    newTrainSet.append(row)

            if newBranches[0] == "RISK":
                leaf = TreeNode("RISK", {"1": None})
                newNode.add_child(leaf, label)
                continue

            # Find whethere all RISK column has same label
            clLabel = {newTrainSet[0][0]}
            for row in newTrainSet:
                clLabel.add(row[0])

            # Remove worked attribute from header
            if newNode.attriName in header:
                header.remove(newNode.attriName)

            if len(clLabel) == 1: #current trainSet has same RISK label
                labelV = 0
                for x in clLabel:
                    labelV = x
                leaf = TreeNode("RISK", {str(labelV): None})
                newNode.add_child(leaf, label)
                print ("Leaf node is", leaf.attriName, ",", leaf.children)
                print ("Updated node is", newNode.attriName, ",", newNode.children)
            else:
                print(newTrainSet)
                print(header)
                childNode, result = buildTree(newTrainSet, header, result)
                newNode.add_child(childNode, label)
                if label in newNode.children:
                    temp = TreeNode(newNode.attriName, newNode.children)
                    temp.children[label] = childNode.attriName
                    print ("Updated node is", temp.attriName, ",", temp.children)

            print(newTrainSet)
            newTrainSet.clear()
            print(header)
            print(newNode.attriName, " , ", newNode.children)
        else:
            return
    result.insert(0, [newNode.attriName, newNode.children])
    return newNode, result 

result = []
tree, result = buildTree(dataSet, headerConstant, result)
print (result)


# def printTree(node, result):
#     result.append({node.attriName: node.children})
#     print( node.attriName, ", ", node.children, "in ptTree")
#     for child in node.children:
#         if node.children[child] != 1 and node.children[child] != 2:
#             printTree(node.children[child], result)
#             return result

# result = []
# finalResult = printTree(tree, result)
# print (finalResult)

with open('../data/random.txt', 'w') as f:
    json.dump(result, f)
