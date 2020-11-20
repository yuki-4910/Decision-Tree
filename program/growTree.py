import numpy as np
import json
import math

# trainSet = np.loadtxt('data/test.txt')

trainSet = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
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
header = []
for atr in desc:
    header.append(str(atr[0]))

# print trainset
trainSet = np.transpose(trainSet)
print(trainSet)

# calculate entropy of Attributes
def findEntropyAttri(attributeColumn, trainSet):
    count = {}
    atriEntropy = 0

    for row in trainSet:
        label = row[attributeColumn]
        if label not in count:
            count[label] = 0
        count[label] += 1

    print(count)

    for label in count:
        print("label", label, "has", count[label], "customers")
        entropyEle = findEntropyElement(label, attributeColumn, trainSet)
        atriEntropy += calcEntropyAttributes(
            count[label], len(trainSet), entropyEle)
        print("---------------------------------")

    print("Attribute of", header[attributeColumn], "has entropy of", atriEntropy, )
    print("---------------------------------")
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
        print("Risk", label, "has", countClassLabelDict[label], "customers")
        entropyS += calcEntropy(countClassLabelDict[label], len(trainSet))

    print("Entorpy of S is", entropyS)
    print("----------------------------")
    return entropyS

# find best attribute as a node
def findNode(trainset):
    entropyS = findEntropy_S(trainset)
    nodeAttri = ""
    highestGain = 0
    if entropyS == 0:
        nodeAttri = "RISK"
        return nodeAttri

    i = 1 # i=0 is Risk column
    while i < len(header): #itterate attri column
        print("Attribute of", header[i])
        atriEntropy = findEntropyAttri(i, trainSet)
        Gain = entropyS - atriEntropy
        print("Attribute of", header[i], "has Gain of", Gain)
        print("<><><><><><><><><><><><><><><><>")
        if Gain > highestGain:
            highestGain = Gain
            nodeAttri = header[i]

        i += 1

    return nodeAttri


def nodeData(trainSet):
    node = findNode(trainSet)
    print("<<<========== select", node, "as a node ==========>>>")
    attriRemove = header.index(node)
    count = {}
    final = []
    for row in trainSet:
        label = row[attriRemove]
        if str(label) not in count:
            count[str(label)] = 0
        count[str(label)] += 1

    final.append(count)
    final.insert(0, node)
    print(final)

    with open('../data/random.txt', 'w') as f:
        json.dump(final, f)

    return final


newBranches = nodeData(trainSet)
currentTrainSet = trainSet.copy()
newTrainSet = []
headerCopy = header.copy()
for label in newBranches[1]:
    print(int(label))
    for row in currentTrainSet:
        currentLabel = row[headerCopy.index(newBranches[0])]
        if int(label) == currentLabel:
            row = np.delete(row, headerCopy.index(newBranches[0]))
            newTrainSet.append(row)

    print(newTrainSet)
    if newBranches[0] in header:
        header.remove(newBranches[0])
    print(header)
    nodeData(newTrainSet)
    newTrainSet.clear()
