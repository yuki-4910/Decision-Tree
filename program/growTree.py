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
container = []
header = []
for atr in desc:
    header.append(str(atr[0]))

# print trainset
trainSet = np.transpose(trainSet)
print(trainSet)

# calculate entropy of Attributes


def findEntropyAttri(attribute, trainSet):
    count = {}
    atriEntropy = 0

    for row in trainSet:
        label = row[attribute]
        if label not in count:
            count[label] = 0
        count[label] += 1

    print(count)

    for label in count:
        print("label", label, "has", count[label], "customers")
        entropyEle = findEntroyElement(label, attribute, trainSet)
        atriEntropy += calcEntropyAttributes(
            count[label], len(trainSet), entropyEle)
        print("---------------------------------")

    print("Attribute of", header[attribute], "has entropy of", atriEntropy, )
    print("---------------------------------")
    return atriEntropy

# finding the entropy for specific element


def findEntroyElement(element, attribute, trainSet):
    customers = []
    countClassLabelSet = {}

    # Find cutomers match with current element
    for row in trainSet:
        label = row[attribute]
        if label == element:
            customers.append(row)

    label, countClassLabelSet = countClassLabel(customers)

    # find entropy of current element
    entropy = 0
    for label in countClassLabelSet:
        entropy += calcEntropy(countClassLabelSet[label], len(customers))

    print("entropy of element(label)", element, "is", entropy)
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


def countClassLabel(array):
    countClassLabel = {}
    classLabel = 0
    for row in array:
        label = row[classLabel]
        if label not in countClassLabel:
            countClassLabel[label] = 0
        countClassLabel[label] += 1

    return label, countClassLabel

# Find Entropy of S


def findEntropy_S(array):
    entropyS = 0
    label, countClassLabelSet = countClassLabel(array)
    for label in countClassLabelSet:
        print("Risk", label, "has", countClassLabelSet[label], "customers")
        entropyS += calcEntropy(countClassLabelSet[label], len(array))

    print("Entorpy of S is", entropyS)
    print("----------------------------")
    return entropyS

# find best attribute as a node


def findNode(trainset):
    entropyS = findEntropy_S(trainset)
    nodeAttri = ""
    highestGain = 0
    i = 1
    while i < len(header):
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
    print("select", node, "as a node")
    attriRemove = header.index(node)
    count = {}
    final = []
    for row in trainSet:
        label = row[attriRemove]
        if str(label) not in str(count):
            count[str(label)] = 0
        count[str(label)] += 1

    final.append(count)
    final.insert(0, node)
    print(final)

    with open('../data/random.txt', 'w') as f:
        json.dump(final, f)

    return final


newBranches = nodeData(trainSet)
newTrainSet = []
headerCopy = header.copy()
for label in newBranches[1]:
    print(int(label))
    for row in trainSet:
        element = row[headerCopy.index(newBranches[0])]
        if int(label) == element:
            row = np.delete(row, headerCopy.index(newBranches[0]))
            newTrainSet.append(row)

    print(newTrainSet)
    if newBranches[0] in header:
        header.remove(newBranches[0])
    print(header)
    nodeData(newTrainSet)

    newTrainSet.clear()
