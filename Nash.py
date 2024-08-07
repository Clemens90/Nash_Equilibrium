'''
Created on 12.07.2024

@author: clemens
'''
import numpy as np
from numpy.lib._npyio_impl import save
from numpy._core.fromnumeric import size
import itertools
import time

def nextTip(tip):
    l = len(tip)
    for i in range(l-2,-1,-1):
        if tip[i]>0:
            tip[i] = tip[i]-1
            sum = 0
            for j in range(i+2,l):
                sum = sum+tip[j]
                tip[j]=0
            tip[i+1]=tip[i+1]+1+sum
            return True
    return False

def IsTipConcentrated(tip,min):
    for i in range(0,len(tip)):
        if tip[i]>0 and tip[i]<min:
            return False
    return True

def AllTips(length, total):
    tip = [total]
    for i in range(length-1):
        tip.append(0)
    ret = [tip.copy()]
    while nextTip(tip):
        ret.append(tip.copy())
    return ret

def ConcentratedTips(length, total, min):
    if min > total:
        return []
    tip = [total]
    for i in range(length-1):
        tip.append(0)
    ret = [tip.copy()]
    while nextTip(tip):
        if IsTipConcentrated(tip, min):
            ret.append(tip.copy())
    return ret

#scores is an array with scores[i] being the score of team i
#ret is an array with the evaluation of each team. E.g. if one team has the highest score it gets 1 and the others 0
def EvalEnd(scores):
    max = scores[0]
    max_indices = [0]
    for n in range(1,len(scores)):
        if scores[n] == max:
            max_indices.append(n)
        if scores[n]>max:
            max = scores[n]
            max_indices = [n]
    ret = [0]*len(scores)
    for n in max_indices:
        ret[n] = 1/len(max_indices)
    
    return ret

#Quoten is an array with the percentages of tip i as a number in [0,1]
#score is an array with the current score of each team
#Tips is a 2d array with all tips that we consider. Tips[i] is the tip for Team i which is again an array
#eval is an evaluation function that gives the value after this round.
#E.g. for the last round eval should give 1 for the best score
#It returns an array where ret[k] are the odds for team k 
def Getodds(Quoten, scores, tips, eval):
    nPlayers = len(tips)
    ret = [0]*nPlayers
    
    #length is the maximal length that makes sense to consider
    length = len(Quoten)
    for tip in tips:
        if len(tip)<length:
            length = len(tip)
            
    #we iterate over all possible results
    for k in range(length):
        #we compute the new score if tip k is correct
        newscores = scores.copy()
        for n in range(nPlayers):
            newscores[n] += tips[n][k]
        
        evaluation = eval(newscores)
        for n in range(nPlayers):
            ret[n] += Quoten[k]*evaluation[n]
    
    #we also have the case that none of these results happen:
    sQuoten = sum(Quoten[0:length])
    evaluation = eval(scores)
    for n in range(nPlayers):
        ret[n] += (1-sQuoten)*evaluation[n]
    
    return ret
    

#tipsArray is a 3d array. tipsArray[k] is the array of tips for team k.
#Let n be the number of teams.
#ret[i0][i1]...[i(n-1)][k] are the odds for team k when team l plays their tip number il, i.e. tipsArray[l][il]
def GetAllOdds(Quoten, scores, tipsArray, eval):
    n = len(tipsArray)
    sizes=[]
    for k in range(n):
        sizes.append(len(tipsArray[k]))
    tupleSize = tuple(sizes)
    ret = np.ndarray(shape=tupleSize,dtype='O')
    for idx in itertools.product(*[range(s) for s in sizes]):
        tips = []
        for k in range(n):
            tips.append(tipsArray[k][idx[k]])
        ret[idx] = Getodds(Quoten, scores, tips, eval)
            
    return ret
        
#finds and prints the worst case scenario for each tip of team k
def PrintWorstCase(oddsAll, tips, k):
    numberTips = oddsAll.shape
    nPlayers = len(numberTips)
    best = [1]*len(tips[k])
    counterTips = [None]*len(tips[k])
    for idx in itertools.product(*[range(s) for s in numberTips]):
        if oddsAll[idx][k]<best[idx[k]]:
            best[idx[k]] = oddsAll[idx][k]
            counterTips[idx[k]] = []
            for l in range(nPlayers):
                if l!=k:
                    counterTips[idx[k]].append(tips[l][idx[l]])
    
    for idx, x in enumerate(tips[k]):
        print('{}: {:.1f}%, Counter tips: {}'.format(x, best[idx]*100,counterTips[idx])) 
    
#we have n players
#odds is a n-d array where odds[i0,...,i(n-1)] is an array whose l-th entry are the odds for team l 
#distributions is a 2-d array where distributions[l] is the distribution of the tips for team l
#returns an array whose l-th entry is the expected value of the l-th tip of team k
def ExpectedValues(k,odds,distributions):
    numberTips = odds.shape
    #numberTips is an array of the form [a,b] with a tips for Team 1 and b tips for Team 2
    nPlayers = len(numberTips)
    exp=[0]*len(distributions[k])
    #We compute an array that tells us the indices we actually only need to look at.
    reducedIndices = []
    for l in range(nPlayers):
        if l==k:
            reducedIndices.append(range(len(distributions[k])))
        else:
            temp = []
            for idelem, elem in enumerate(distributions[l]):
                if elem>0:
                    temp.append(idelem)
            reducedIndices.append(temp)
    #Iterates over the odds array
    #for idx in itertools.product(*[range(s) for s in numberTips]):
    for idx in itertools.product(*reducedIndices):
        prodDistr = 1
        for l in range(nPlayers):
            if l != k:
                prodDistr *= distributions[l][idx[l]]
        exp[idx[k]] += odds[idx][k]*prodDistr
    return exp

#we have n players
#odds is a n-d array where odds[i0,...,i(n-1)] is an array whose l-th entry are the odds for team l 
#distributions is a 2-d array where distributions[l] is the distribution of the tips for team l
#returns an array whose l-th entry is the expected value of the l-th tip of team k
def ExpectedValuesFast(k, odds, Eval, distributions, tips, scores):
    numberTips = []
    for dist in distributions:
        numberTips.append(len(dist))
    #numberTips is an array of the form [a,b,...] with a tips for Team 1 and b tips for Team 2
    nPlayers = len(numberTips)
    exp=[0]*len(distributions[k])
    #We compute an array that tells us the indices we actually only need to look at.
    reducedIndices = []
    for l in range(nPlayers):
        if l==k:
            reducedIndices.append(range(len(distributions[k])))
        else:
            temp = []
            for idelem, elem in enumerate(distributions[l]):
                if elem>0:
                    temp.append(idelem)
            reducedIndices.append(temp)
    #Iterates over the odds array
    #for idx in itertools.product(*[range(s) for s in numberTips]):
    for idx in itertools.product(*reducedIndices):
        prodDistr = 1
        tip = []
        for l in range(nPlayers):
            tip.append(tips[l][idx[l]])
            if l != k:
                prodDistr *= distributions[l][idx[l]]
        if odds[idx]==None:
            odds[idx] = Getodds(Quoten, scores, tip, Eval)
        exp[idx[k]] += odds[idx][k]*prodDistr
    return exp

#lambda1 decides when we update our distribution. 1 is the standard in which case we increase the distribution for any tip that has higher expected value than our current mixed strategy.
#lambda2 is a multiplicative factor for how much we change the distribution. 1 works fine.
#returns a new array containing the updated distribution
def UpdateDistribution(distribution, expectedvalues, lambda1, lambda2):
    newDistribution = distribution.copy()
    totalExpect = 0
    for i in range(len(distribution)):
        totalExpect += distribution[i]*expectedvalues[i]
    
    for i in range(len(distribution)):
        if expectedvalues[i]>totalExpect*lambda1:
            newDistribution[i]+= (expectedvalues[i]-totalExpect)*lambda2
    Normalize(newDistribution)
    return newDistribution
    


def Normalize(arr):
    sumArr = sum(arr)
    for i in range(len(arr)):
        arr[i] /= sumArr
        
def RemoveSmall(arr, eps):
    for i in range(len(arr)):
        if(arr[i]<eps):
            arr[i]=0
    Normalize(arr)
    
def RemoveAllButN(arr, n):
    temp = arr.copy()
    temp.sort()
    cutoff = temp[-n]
    for idel, el in enumerate(arr):
        if el<cutoff:
            arr[idel] = 0
    Normalize(arr)
    
#returns an array with all the expected values of the mixed strategies
#the distributions array is replaced with the updated one.
#I included reasonable default values
def Iterate(oddsAll, distributions, speedUp = 0, N_remove = 20, eps_remove = 0.0001, lambda1 = 1, lambda2 = 1):
    nPlayers = len(distributions)
    expArray = [0]*nPlayers
    for n in range(nPlayers):
        exp = ExpectedValues(n, oddsAll, distributions)
        temp = UpdateDistribution(distributions[n], exp, lambda1, lambda2)
        
        if speedUp == 1:
            RemoveAllButN(temp, N_remove)
        if speedUp == 2:
            RemoveSmall(temp, eps_remove)
            
                
        #compute the expectation for the current mixed strategy
        for idx, x in enumerate(temp):
            expArray[n]+=x*exp[idx]
        
        distributions[n] = temp
    
    return expArray

#returns an array with all the expected values of the mixed strategies
#the distributions array is replaced with the updated one.
#I included reasonable default values
def IterateFast(odds, Eval, distributions, tips, score, speedUp = 0, N_remove = 20, eps_remove = 0.0001, lambda1 = 1, lambda2 = 1):
    nPlayers = len(distributions)
    expArray = [0]*nPlayers
    for n in range(nPlayers):
        exp = ExpectedValuesFast(n, odds, Eval, distributions, tips, score)
        temp = UpdateDistribution(distributions[n], exp, lambda1, lambda2)
        
        if speedUp == 1:
            RemoveAllButN(temp, N_remove)
        if speedUp == 2:
            RemoveSmall(temp, eps_remove)
            
                
        #compute the expectation for the current mixed strategy
        for idx, x in enumerate(temp):
            expArray[n]+=x*exp[idx]
        
        distributions[n] = temp
    
    return expArray

def FindNashEquilibrium(oddsAll, tips, Print = False, PrintNumber = 3, speedUp = 0, N_remove = 20, eps_remove = 0.0001, lambda1 = 1, lambda2 = 1, changeDist = 0.001, maxNumber = 1000):
    numberTips = oddsAll.shape
    nPlayers = len(numberTips)
    distributions = []
    for i in range (nPlayers):
        distributions.append(NewArray1000(numberTips[i]))
    
    for counter in range(maxNumber):
        #print(counter)
        distrOld = distributions.copy()
        #maybe it is better to have lambda2 depend on the difference in the distribution?
        expectedVals = Iterate(oddsAll, distributions, speedUp, N_remove, eps_remove, lambda1, lambda2)
        if MaxDiffArrays2D(distributions, distrOld)<changeDist:
            if Print:
                PrintTop(distributions, tips, expectedVals, PrintNumber)
                print('')
            return expectedVals
    
    if Print:
        PrintTop(distributions, tips, expectedVals, PrintNumber)
        print('')
    return expectedVals

def FindNashEquilibriumFast(odds, Eval, Quoten, score, tips, Print = False, PrintNumber = 3, speedUp = 0, N_remove = 20, eps_remove = 0.0001, lambda1 = 1, lambda2 = 1, changeDist = 0.001, maxNumber = 1000):
    numberTips = []
    for t in tips:
        numberTips.append(len(t))
    nPlayers = len(numberTips)
    distributions = []
    for i in range (nPlayers):
        distributions.append(NewArray1000(numberTips[i]))
    
    for counter in range(maxNumber):
        #print(counter)
        distrOld = distributions.copy()
        #maybe it is better to have lambda2 depend on the difference in the distribution?
        expectedVals = IterateFast(odds, Eval, distributions, tips, score, speedUp, N_remove, eps_remove, lambda1, lambda2)
        if MaxDiffArrays2D(distributions, distrOld)<changeDist:
            if Print:
                PrintTop(distributions, tips, expectedVals, PrintNumber)
                print('')
            return expectedVals
    
    if Print:
        PrintTop(distributions, tips, expectedVals, PrintNumber)
        print('')
    return expectedVals

#we assume that newElement is an array of elementary 
def AddNewArray(ArrayOld, newElement):
    for elem in ArrayOld:
        if len(elem) == len(newElement) and elem==newElement:
            return 0
    
    ArrayOld.append(newElement)
    return 1

def PossibleScores(scoresArray, total = 5):
    nPlayers = len(scoresArray[0])
    ret = []
    sizes=[]
    for k in range(nPlayers):
        sizes.append(total+1)
    tupleSize = tuple(sizes)
    for scores in scoresArray:
        for idx in itertools.product(*[range(s) for s in sizes]):
            temp = scores.copy()
            for i in range(nPlayers):
                temp[i] += idx[i]
            AddNewArray(ret, temp)
    
    return ret

#Normalize by always assuming that scores[0] = 0
def NormalizeScores(scoresArray):
    ret = []
    nPlayers = len(scoresArray[0])
    for scores in scoresArray:
        temp1 = scores.copy()
        temp2 = temp1[0]
        for i in range(nPlayers):
            temp1[i]-=temp2
        AddNewArray(ret, temp1)
    return ret

def NRoundsToGo(quotenThisRound, quotenOtherRounds, scoresArray, roundsToGo, tips, Print = True, PrintAll = False, PrintNumber = 3, speedUp = 0, N_remove = 20, eps_remove = 0.0001, lambda1 = 1, lambda2 = 1, changeDist = 0.001, maxNumber = 1000, total = 5):
    if roundsToGo==0:
        ret = []
        for scores in scoresArray:
            ret.append(EvalEnd(scores))
        return ret
    else:
        scoresArrayNew = PossibleScores(scoresArray)
        scoresArrayNewNormalized = NormalizeScores(scoresArrayNew)
        scoresArrayNewNormalized.sort()
        temp = NRoundsToGo(quotenOtherRounds,quotenOtherRounds, scoresArrayNewNormalized, roundsToGo-1, tips, PrintAll, PrintAll, PrintNumber, speedUp, N_remove, eps_remove, lambda1, lambda2, changeDist, maxNumber, total)
        #Now we define a new eval-function
        def eval(score):
            #first we normalize scores
            temp2 = score.copy()
            temp3 = temp2[0]
            for i in range(len(temp2)):
                temp2[i]-=temp3
            #then we look it up
            for idel, el in enumerate(scoresArrayNewNormalized):
                if el==temp2:
                    return temp[idel]
            return None
        
        ret = []
        for scores in scoresArray:
            #check if it is trivial
            tempScores = scores.copy()
            tempScores.sort()
            if tempScores[-1]>tempScores[-2]+total*roundsToGo:
                if Print:
                    print('{} rounds to go, current score: {}'.format(roundsToGo, scores))
                    print('Team {} wins no matter what!'.format(scores.index(tempScores[-1])+1))
                    print('')
                tempRet = [0]*len(scores)
                tempRet[scores.index(tempScores[-1])] = 1
                ret.append(tempRet)
            else:
                oddsAll = GetAllOdds(quotenThisRound, scores, tips, eval)
                if Print==True:
                    print('{} rounds to go, current score: {}'.format(roundsToGo, scores))
                ret.append(FindNashEquilibrium(oddsAll, tips, Print, PrintNumber, speedUp, N_remove, eps_remove, lambda1, lambda2, changeDist, maxNumber))
        return ret
    
def NRoundsToGoFast(quotenThisRound, quotenOtherRounds, scoresArray, roundsToGo, tips, Print = True, PrintAll = False, PrintNumber = 3, speedUp = 0, N_remove = 20, eps_remove = 0.0001, lambda1 = 1, lambda2 = 1, changeDist = 0.001, maxNumber = 1000, total = 5):
    if roundsToGo==0:
        ret = []
        for scores in scoresArray:
            ret.append(EvalEnd(scores))
        return ret
    else:
        scoresArrayNew = PossibleScores(scoresArray)
        scoresArrayNewNormalized = NormalizeScores(scoresArrayNew)
        scoresArrayNewNormalized.sort()
        temp = NRoundsToGoFast(quotenOtherRounds,quotenOtherRounds, scoresArrayNewNormalized, roundsToGo-1, tips, PrintAll, PrintAll, PrintNumber, speedUp, N_remove, eps_remove, lambda1, lambda2, changeDist, maxNumber, total)
        #Now we define a new eval-function
        def eval(score):
            #first we normalize scores
            temp2 = score.copy()
            temp3 = temp2[0]
            for i in range(len(temp2)):
                temp2[i]-=temp3
            #then we look it up
            for idel, el in enumerate(scoresArrayNewNormalized):
                if el==temp2:
                    return temp[idel]
            return None
        
        ret = []
        for scores in scoresArray:
            #check if it is trivial
            tempScores = scores.copy()
            tempScores.sort()
            if tempScores[-1]>tempScores[-2]+total*roundsToGo:
                if Print:
                    print('{} rounds to go, current score: {}'.format(roundsToGo, scores))
                    print('Team {} wins no matter what!'.format(scores.index(tempScores[-1])+1))
                    print('')
                tempRet = [0]*len(scores)
                tempRet[scores.index(tempScores[-1])] = 1
                ret.append(tempRet)
            else:
                #define empty oddsall
                sizes=[]
                for k in range(len(tips)):
                    sizes.append(len(tips[k]))
                tupleSize = tuple(sizes)
                odds = np.ndarray(shape=tupleSize,dtype='O')
                if Print==True:
                    print('{} rounds to go, current score: {}'.format(roundsToGo, scores))
                ret.append(FindNashEquilibriumFast(odds, eval, quotenThisRound,scores, tips, Print, PrintNumber, speedUp, N_remove, eps_remove, lambda1, lambda2, changeDist, maxNumber))
        return ret

def PrintTop(distributions, tips, expectedValue, number=5):
    nPlayers = len(distributions)
    for n in range(nPlayers):
        print('Team {}: Win percentage: {:.1f}%'.format(n+1, expectedValue[n]*100))
        temp1 = distributions[n].copy()
        temp2 = tips[n].copy()
        list1, list2 = (list(t) for t in zip(*sorted(zip(temp1, temp2))))
        for i in range(len(list1)-1, len(list1)-number-1,-1):
            print('{}: {:.1f}%'.format(list2[i],list1[i]*100))
            
def PrintThreshold(distributions, tips, threshold = 0.05):
    nPlayers = len(distributions)
    for n in range(nPlayers):
        print('Team {}'.format(n+1))
        temp1 = distributions[n].copy()
        temp2 = tips[n].copy()
        list1, list2 = (list(t) for t in zip(*sorted(zip(temp1, temp2))))
        for i in range(len(list1)-1, -1,-1):
            if list1[i]>threshold:
                print('{}: {:.1f}%'.format(list2[i],list1[i]*100))

def NewArray1000(length):
    ret = [1]
    for i in range(length-1):
        ret.append(0)
    return ret

def NewArrays1000(length, number):
    temp = NewArray1000(length)
    ret = []
    for i in range(number):
        ret.append(temp.copy())
    return ret

def NewArrayRand(length):
    ret = []
    for i in range(length):
        ret.append(np.random.Generator.uniform(0, 1, None))
    return Normalize(ret)

def MaxDiffArrays(arr1, arr2):
    assert len(arr1)==len(arr2)
    diff = 0
    for idx, x in enumerate(arr1):
        if abs(x- arr2[idx])>diff:
            diff = abs(x- arr2[idx])
    return diff

def MaxDiffArrays2D(arr1, arr2):
    assert len(arr1)==len(arr2)
    maxdiff = 0
    for idx, x in enumerate(arr1):
        temp = MaxDiffArrays(x, arr2[idx])
        if temp>maxdiff:
            maxdiff = temp
    return maxdiff

##################################################################################################

##################################################################################################

scores = [0,-10]
#Quoten =[0.2,0.1,0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01]
Quoten = [0.155,0.129,0.114,0.096,0.082,0.058,0.0405,0.0405,0.04,0.036]

length = 5
total = 5
Tipps1 = AllTips(length, total)
Tipps2 = AllTips(length, total)
#Tipps3 = ConcentratedTips(length, total, 2)
tips = [Tipps1,Tipps2]
scoresArray = []
for i in range(21):
    scoresArray.append(scores.copy())
    #print(scores)
    #oddsAll = GetAllOdds(Quoten, scores, tips, EvalEnd)
    #print(FindNashEquilibrium(oddsAll, tips, False))
    scores[1] += 1


t = time.time()
NRoundsToGo(Quoten, Quoten, scoresArray, 3, tips, True, True, 3, speedUp = 0, N_remove = 20)

tips2 = [AllTips(length, total), AllTips(length, total), AllTips(length, total)]
#NRoundsToGo(Quoten, Quoten, [[2,1,0]], 1, tips2, True, True, PrintNumber=3, speedUp = 0, N_remove = 10)
print(time.time()-t)