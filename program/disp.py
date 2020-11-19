import numpy as np
import json


#Draw the (sub)decision tree rooted at tr. ls recovers the english
#names for the values in each domain, pre gives the number of blank
#spaces preceding each level, and f is the file where the drawing
#will be stored
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
    msg = input('Plot un-pruned tree. Plot (Y/N)?')
    if msg == 'Y':
        with open('../data/deDomain.txt') as f:
            m = json.load(f)
        with open('../data/'+fname) as f:
            tr = json.load(f)
        with open('../data/treePicFull.txt', 'w') as f:
            f.write('')
        with open('../data/treePicFull.txt','a+') as f:
            dispTree(tr,m,[],[],f)


    
