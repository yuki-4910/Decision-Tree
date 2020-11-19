import json
import numpy as np


def predict(tr, r, li):
    if type(tr) != list:
        return tr
    a = tr.copy()
    while type(a) == list:
        for i in range(len(li)):
            if a[0] == li[i][0]:
                break
        a = a[1][str(r[i-1])]
    return a


def accuracy(tr, m, li):
    ac = 0
    for i in range(np.size(m,axis=1)):
        if predict(tr, m[1:,i], li) == m[0,i]:
                   ac += 1
    return ac/np.size(m,axis=1)


def prunePhase(tr, m1, m2, li):
    mx = accuracy(tr, m2, li)
    a = prune(tr, m1, li)
    qu = [mx]
    while a != 'nil':
        if len(a) == 3: #prune occurs at the grandchildren level or lower
            b = tr
        else:          #at the children level
            b = a[1]
        acc = accuracy(b, m2, li)
        qu.append(acc)
        if acc >= mx:
            mx = acc
        else:
            restore(a)
            b = tr  #necessary if b is a label, redundant if it is a list
        a = prune(b, m1, li)
    print(qu)
    return b

def prune(tr, m, li): 
    if type(tr) != list:
        return 'nil'
    if len(tr) == 3:
        return 'nil'
    lc = True #lc tells if all children are leaves
    for v in tr[1]:
        if type(tr[1][v]) == list:
            #construct sub-training set to be associated with tr[1][v]
            for i in range(1,len(li)):
                if li[i][0] == tr[0]:
                    break
            ind = np.nonzero(m[i] == int(v))
            a = np.delete(m,i,axis=0)
            b = li.copy()
            b.pop(i) 
            r = prune(tr[1][v],a[:,ind[0]],b)
            if type(r) == list:#A descendent of tr[1][v] (exclusive) becomes leaf
                if len(r) == 3:#The descendent is a grandchild or lower
                    return r
                else:          #The descendent is a child
                    tr[1][v] = r[1] #The descendent becomes a leaf now
                    return [r[0], tr, v]
            else:  #r = 'nil'
                lc = False
    if lc == True:
        return [tr, majority(m[0])]
    else:
        return 'nil'

def restore(a):
    if len(a) == 3: 
        g = a[2]
        a[0].append('N')
        a[1][1][g] = a[0]
    else:
        a[0].append('N')
        
def majority(a):
    b = set(a)
    c = list(a)
    mx = 0
    for d in b:
        if c.count(d) > mx:
            mx = c.count(d)
            who = d
    return int(who)


def dispTree(tr,ls,sp,sb,f):
    if type(tr) != list:
        f.write(ls['RISK'][str(tr)])
        return
    f.write(tr[0]+'\n') #attribute selected in current node
    j = 0
    for v in tr[1]:#values in the domain of this attribute
        sp1 = sp.copy()
        sb1 = sb.copy()
        j += 1
        for k in range(len(sp1)):
            f.write(' '+sb1[k])
            for p in range(sp1[k]):
                f.write(' ')
        f.write(' !--'+ls[tr[0]][v]+'--')
        if list == type(tr[1][v]):
            sk = len(ls[tr[0]][v]) + 5
            sp1.append(sk)
            if j < len(tr[1]):
                sb1.append('!')
            else:
                sb1.append(' ')
            dispTree(tr[1][v],ls,sp1,sb1,f)
        else:
            f.write(ls['RISK'][str(tr[1][v])]+'\n')

def showIt(fname):
    with open('../data/deDomain.txt') as f:
        m = json.load(f)
    with open('../data/'+fname) as f:
        tr = json.load(f)
    with open('../data/treePicPruned.txt', 'w') as f:
        f.write('')
    with open('../data/treePicPruned.txt','a+') as f:
        dispTree(tr,m,[],[],f)
                      
def main(fname):
    #with open('../data/treeFileFull.txt') as f:
    #    tr = json.load(f)
    m1 = np.loadtxt('../data/train.txt', dtype=int)
    m2 = np.loadtxt('../data/test.txt', dtype=int)
    with open('../data/dataDesc.txt') as f:
        e = json.load(f)
    with open('../data/'+fname) as f:
        tr = json.load(f)
    a = prunePhase(tr, m1, m2, e)
    with open('../data/treeFilePruned.txt','w') as f:
        json.dump(a,f)
    showIt('treeFilePruned.txt')
    return 'treeFilePruned.txt'
