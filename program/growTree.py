import numpy as np
import json
import math

with open ('data/test.txt') as f:
    file = json.load(f)

print (file)

trainSet = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
    [3, 3, 3, 1, 3, 3, 1, 1, 3, 3, 3, 1],
    [2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2],
    [2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 2],
    [2, 3, 2, 3, 2, 2, 3, 3, 1, 2, 3, 1],
    [2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2]])


domain = [["RISK", [1, 2]], ["AGE", [1, 2, 3]], ["CRED_HIS", [1, 2]],
          ["INCOME", [1, 2]], ["RACE", [1, 2, 3]], ["HEALTH", [1, 2]]]
# print (m[0][1][1])

L = {"RISK": {"1": "L", "2": "H"}, "AGE": {"1": "youth", "2": "mid_age", "3": "senior"}, "CRED_HIS": {"1": "poor", "2": "good"},
     "INCOME": {"1": "low", "2": "high"}, "RACE": {"1": "white", "2": "asian", "3": "black"}, "HEALTH": {"1": "normal", "2": "poor"}}

# print attributes of data
header = []
for atr in domain:
    header.append(atr[0])
# print(header)

# print trainset
trainSet = np.transpose(trainSet)
print(trainSet)

# calculate entropy of Attributes
def findEntropyAttri(attribute):
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
        entropyEle = findEntroyElement(label, attribute)
        atriEntropy += calcEntropyAttributes(count[label], len(trainSet), entropyEle)
        print ("---------------------------------")
    
    print ("Attribute of", header[attribute], "has entropy of", atriEntropy, )
    print ("---------------------------------")
    return atriEntropy

# finding the entropy for specific element
def findEntroyElement(element, attribute):
    customers = []
    countClassLabelSet = {}
    # Find cutomers match with current element
    for row in trainSet:
        label = row[attribute]
        if label == element:
            customers.append(row)
    # print(customers)
    
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

    print ("Entorpy of S is", entropyS)
    print ("----------------------------")
    return entropyS

# find best attribute as a node
def findNode(trainset):
    entropyS = findEntropy_S(trainset)
    nodeAttri = ""
    highestGain = 0
    i = 1
    while i < len(header):
        print ("Attribute of", header[i])
        atriEntropy = findEntropyAttri(i)
        Gain = entropyS - atriEntropy
        print ("Attribute of", header[i], "has Gain of", Gain)
        print ("<><><><><><><><><><><><><><><><>")
        if Gain > highestGain:
            highestGain = Gain
            nodeAttri = header[i]
        i += 1

    print ("select", nodeAttri, "as a node")

findNode(trainSet)
