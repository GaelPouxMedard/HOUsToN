import numpy as np
import cvxpy as cp
import random
import multiprocessing
import tqdm
from scipy.special import erfc, erf
from copy import deepcopy as copy
import sympy as sy
import time
import sparse
from sklearn.metrics import f1_score, roc_auc_score

np.random.seed(1111)
random.seed(1111)


def gaussian(reference_time, time_interval, bandwidth):
    ''' RBF kernel for Hawkes process.
        @param:
            1.reference_time: np.array, entries larger than 0.
            2.time_interval: float/np.array, entry must be the same.
            3. bandwidth: np.array, entries larger than 0.
        @rtype: np.array
    '''
    numerator = - (time_interval - reference_time) ** 2 / (2 * bandwidth ** 2)
    return np.exp(numerator)

def gaussian_integ(reference_time, bandwidth, max_time):
    ''' g_theta for DHP
        @param:
            2. timeseq: 1-D np array time sequence before current time
            3. base_intensity: float
            4. reference_time: 1-D np.array
            5. bandwidth: 1-D np.array
        @rtype: np.array, shape(3,)
    '''
    fac = (2*np.pi)**0.5 * bandwidth * 0.5
    div = (2 * bandwidth ** 2) ** 0.5
    results = fac * ( erfc( (reference_time - max_time) / div) - erfc( reference_time / div) )

    return results

# Kernels
def logS(ti, tj, alphaji, kernel, means=None, sigs=None):
    if kernel=="exp":
        return -alphaji[0]*(ti-tj)
    elif kernel=="ray":
        return -alphaji[0]*((ti-tj)**2) / 2
    elif kernel == "rbf":
        try:
            return -alphaji.dot(gaussian_integ(means, sigs, ti-tj))
        except:
            k = 0.
            for k_i in range(len(means)):
                k += -alphaji[k_i]*gaussian_integ(means[k_i], sigs[k_i], ti-tj)
            return k

def H(ti, tj, alphaji, kernel, means=None, sigs=None):
    if kernel == "exp":
        return alphaji[0]
    elif kernel == "ray":
        return alphaji[0]*(ti-tj)
    elif kernel == "rbf":
        try:
            return alphaji.dot(gaussian(means, ti-tj, sigs))
        except:
            k = 0.
            for k_i in range(len(means)):
                k += alphaji[k_i]*gaussian(means[k_i], ti-tj, sigs[k_i])
            return k


# Fit tools
def likelihood(alpha, dicUrls, T, kernel, means, sigs):
    L=0

    for u in dicUrls:
        vecCascTime=[]
        vecCascUsr=[]
        for (usri, ti) in dicUrls[u]:
            vecCascTime.append(ti)
            vecCascUsr.append(usri)
        vecCascTime = np.array(vecCascTime)
        vecCascUsr = np.array(vecCascUsr)

        for it in range(len(vecCascTime)):
            ti = vecCascTime[it]
            usri = vecCascUsr[it]

            usrInfTi = vecCascUsr[vecCascTime < ti]
            timeInfTi = vecCascTime[vecCascTime < ti]

            usrSupT = vecCascUsr[vecCascTime > T]
            timeSupT = (vecCascTime[vecCascTime > T] != 0).astype(int) * T

            #sumHinfTi = sum(alpha[usrInfTi, usri, 0])
            #logSinfTi = sum(-(ti-timeInfTi)*alpha[usrInfTi, usri, 0])
            #logSsupTi = sum(-(timeSupT-ti)*alpha[usri, usrSupT, 0])

            logSinfTi, logSsupTi, sumHinfTi = 0., 0., 0.
            if len(usrInfTi) != 0:
                for usrj, tj in zip(usrInfTi, timeInfTi):
                    sumHinfTi += H(ti, tj, alpha[usrj, usri], kernel, means, sigs)
                    logSinfTi += logS(ti, tj, alpha[usrj, usri], kernel, means, sigs)
            if len(usrSupT) != 0:
                for usrj, tj in zip(usrSupT, timeSupT):
                    logSsupTi += logS(tj, ti, alpha[usri, usrj], kernel, means, sigs)

            L += logSinfTi + logSsupTi + np.log(sumHinfTi+1e-6)

    return L

def likelihoodCVXUsrIndiv(alphai, dicUrls, cascUsr, T, usri, kernel, means, sigs):
    L=0

    for u in cascUsr[usri]:
        vecCascTime=[]
        vecCascUsr=[]
        ti = -1
        for (usr, timei) in dicUrls[u]:
            vecCascTime.append(timei)
            vecCascUsr.append(usr)
            if usr==usri:
                ti = timei

        vecCascTime = np.array(vecCascTime)
        vecCascUsr = np.array(vecCascUsr)

        usrInfTi = vecCascUsr[vecCascTime < ti]
        timeInfTi = vecCascTime[vecCascTime < ti]

        sumHinfTi, logSinfTi, logSsupTi = 0., 0., 0.
        for usrInfTi_i in range(len(usrInfTi)):
            if ti<=T:
                sumHinfTi += H(ti, timeInfTi[usrInfTi_i], alphai[usrInfTi[usrInfTi_i]], kernel, means, sigs)  #alphai[usrInfTi[usrInfTi_i]]
                logSinfTi += logS(ti, timeInfTi[usrInfTi_i], alphai[usrInfTi[usrInfTi_i]], kernel, means, sigs)  #-(ti - timeInfTi[usrInfTi_i]) * (alphai[usrInfTi[usrInfTi_i]])
            else:
                pass
                #logSsupTi += logS(T, timeInfTi[usrInfTi_i], alphai[usrInfTi[usrInfTi_i]], kernel, means, sigs)  #-(T - timeInfTi[usrInfTi_i]) * (alphai[usrInfTi[usrInfTi_i]])

        L += logSinfTi
        L += cp.log(sumHinfTi + 1e-100)
        L += logSsupTi

    return L

# Manipulate data files
def save(folder, filename, alphaInfer, usrToInt, cascToInt, kernel):
    np.save(folder+filename+f"_Alpha_{kernel}", alphaInfer)
    with open(folder+filename+"_int_to_users.txt", "w+", encoding="utf-8") as f:
        for u in usrToInt:
            f.write(f"{usrToInt[u]}\t{u}\n")
    with open(folder+filename+"_int_to_cascade.txt", "w+", encoding="utf-8") as f:
        for c in cascToInt:
            f.write(f"{cascToInt[c]}\t{c}\n")

def readObservations(dataFile, outputFolder, threshold=1e20):
    # Data where each line is: index\ttime\twords\tnode_id\tcascade_id\n
    observations = []
    wdToIndex, index = {}, 0
    with open(dataFile, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            l = line.replace("\n", "").split("\t")
            i_doc = int(l[0])
            timestamp = float(l[1])
            words = l[2].split(",")
            usr = int(l[3])
            casc = int(l[4])
            uniquewords, cntwords = np.unique(words, return_counts=True)
            for un in uniquewords:
                if un not in wdToIndex:
                    wdToIndex[un] = index
                    index += 1
            uniquewords = [wdToIndex[un] for un in uniquewords]
            uniquewords, cntwords = np.array(uniquewords, dtype=int), np.array(cntwords, dtype=int)

            tup = (i_doc, timestamp, (uniquewords, cntwords), usr, casc)
            observations.append(tup)

            if i>threshold:
                break
    with open(outputFolder+"indexWords.txt", "w+", encoding="utf-8") as f:
        for wd in wdToIndex:
            f.write(f"{wdToIndex[wd]}\t{wd}\n")

    obs = np.array(observations, dtype=object)
    observations = {}
    for casc in set(list(obs[:, 4])):
        usr, time = obs[obs[:, 4]==casc, 3], obs[obs[:, 4]==casc, 1]
        observations[casc] = list(zip(usr, time))
    return observations


# Treat the data

def getObsUsr(obs):
    obsUsr={}
    for c in obs:
        for (u, t) in obs[c]:
            try:
                obsUsr[u].append((c, t))
            except:
                obsUsr[u]=[(c, t)]

    return obsUsr

def removeUsers(obs, obsUsr, seuil=0):
    usrToRem = set()
    for u in obsUsr:
        if len(obsUsr[u])<=seuil:
            usrToRem.add(u)
    for u in obs:
        indToRem=[]
        for i in range(len(obs[u])):
            if obs[u][i][0] in usrToRem:
                indToRem.append(i)
        for i in sorted(indToRem, reverse=True):
            del obs[u][i]
    for u in list(obs.keys()):
        if len(obs[u])==0:
            del obs[u]

    return obs

def treatData(data):
    obs = {}
    obsUsr = {}
    usrToInt = {}
    cascToInt = {}
    intToUsr = {}

    numUsr = 0
    numCasc = 0
    for casc in data:
        for (u,t) in data[casc]:
            if u not in usrToInt:
                usrToInt[u] = numUsr
                intToUsr[numUsr]=u
                numUsr+=1

            if casc not in cascToInt:
                cascToInt[casc] = numCasc
                numCasc+=1

            if cascToInt[casc] not in obs: obs[cascToInt[casc]] = []
            obs[cascToInt[casc]].append((usrToInt[u],t))

            if usrToInt[u] not in obsUsr: obsUsr[usrToInt[u]] = []
            obsUsr[usrToInt[u]].append((cascToInt[casc],t))

    return obs, obsUsr, usrToInt, cascToInt

def getCascUsr(obs):
    cascUsr = {}
    for u in obs:
        for (usri, ti) in obs[u]:
            if usri not in cascUsr: cascUsr[usri] = set()
            cascUsr[usri].add(u)
    return cascUsr

def getNeverInteractingNodes(obsUsr):
    nonVusG, nonVusD = [], []
    for u in obsUsr:
        setCascU = set()
        for c,t in obsUsr[u]:
            setCascU.add(c)
        for v in obsUsr:
            setCascV = set()
            for c,t in obsUsr[v]:
                setCascV.add(c)

            if setCascU - setCascV == setCascU:
                nonVusG.append(u)
                nonVusD.append(v)

    nonVusG, nonVusD = np.array(nonVusG, dtype=int), np.array(nonVusD, dtype=int)

    return nonVusG, nonVusD


# Main loop

def fitNode(args):
    node, func, obs, cascUsr, T, N, kernel, K, means, sigs = args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9]
    alphai = cp.Variable((N, K))
    L = func(alphai, obs, cascUsr, T, node, kernel, means, sigs)
    objective = cp.Maximize(func(alphai, obs, cascUsr, T, node, kernel, means, sigs))
    constraints = [alphai >= 0, alphai <= 2]
    try:
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(solver=cp.ECOS)
        except:
            result = prob.solve(solver=cp.SCS)
    except Exception as e:
        print(e)
        alphai.value = np.zeros((N, K))

    return alphai, node

def getAlpha(obs=None, datafile="", save_data=True, outputfolder="", filename="_", processes=4, kernel="exp", K=1, means=None, sigs=None, printprog=False):
    # obs must be formatted as follows: {cascade_id: [(user, time), (user, time), ...]}
    if __name__ == '__main__':
        if obs is None:
            obs = readObservations(datafile, outputfolder)
        obsUsr = getObsUsr(obs)
        if printprog: print("Treat data - Remove useless usrs")  # Remove users that have seen less that seuil cascades
        obs = removeUsers(obs, obsUsr, seuil=0)
        if printprog: print("Treat data - Normalize")
        obs, obsUsr, usrToInt, cascToInt = treatData(obs)  # Reorder user and cascade numbers
        if printprog: print("Treat data - Cascade usr")
        cascUsr = getCascUsr(obs)  # Get the set of cascades a user has seen
        if printprog: print("End treat data")


        # If no common cascade between two nodes, then alpha_ij=0
        neverInteractingG, neverInteractingD = getNeverInteractingNodes(obsUsr)

        # Maximum observation time
        maxT = -1
        for c in obs:
            for (u, t) in obs[c]:
                if t > maxT:
                    maxT = t
        T = maxT
        if printprog: print("Obs length:", T)

        N = len(obsUsr)
        NCasc = len(obs)
        if printprog: print("# users:", N)
        if printprog: print("# cascades:", NCasc)

        # Likelihood test (fully random connexions)
        alpha = np.random.random((N, N, K))
        L = 0
        L = likelihood(alpha, obs, T, kernel, means, sigs)
        if printprog: print("L random", L)

        # Fit
        if printprog: print("Fit")
        alphaInfer = np.zeros((N, N, K))
        with multiprocessing.Pool(processes=processes) as p:
            with tqdm.tqdm(total=N) as progress:
                args = [(usri, likelihoodCVXUsrIndiv, obs, cascUsr, T, N, kernel, K, means, sigs) for usri in cascUsr]
                for i, res in enumerate(p.imap(fitNode, args)):
                    progress.update()
                    usri = res[1]
                    alphai = res[0].value
                    alphaInfer[:, usri] = alphai

        for i in range(len(alphaInfer)):
            alphaInfer[i, i] = 0
        for i,j in zip(neverInteractingG,neverInteractingD):
            alphaInfer[i, j] = 0
        optValue = likelihood(alphaInfer, obs, T, kernel, means, sigs)
        if printprog: print("Optimal value:", optValue)

        if save_data:
            save(outputfolder, filename, alphaInfer, usrToInt, cascToInt, kernel)

        return alphaInfer




varChange = 1
if __name__ == "__main__":
    datafile, outputfolder = "data/Synth/Synth_events.txt", "data/Synth/"
    import sys,os
    try:
        outputfolder = sys.argv[1]
        datafile = sys.argv[2]
        threshold = int(sys.argv[3])
    except Exception as e:
        outputfolder = "output_BL/Synth/"
        datafile = f"data/Synth/Synth_PL_OL=0.0_wdsPerObs=5_vocPerClass=100_events.txt"
        threshold = 10000

    kernel = "exp"
    K = 1
    if kernel == "rbf":
        means = np.array([3, 7, 11])[:K]
        sigs = np.array([0.5, 0.5, 0.5])[:K]
    else:
        means = None
        sigs = None
    filename = datafile.replace("_events.txt", "").replace("data/Synth/", "")
    filename = filename.replace("Synth_", "Synth_NetRate_")

    load = False
    if not load:
        obs = readObservations(datafile, outputfolder, threshold = threshold)
        obsUsr = getObsUsr(obs)
        al = getAlpha(obs = copy(obs), datafile=datafile, outputfolder=outputfolder, filename=filename, save_data=True, kernel=kernel, K=K, means=means, sigs=sigs)
        np.save(datafile.replace("data", "output_BL").replace("_events.txt", "").replace("Synth_", "Synth_NetRate_"), al)
    else:
        al = np.load(datafile.replace("data", "output_BL").replace("_events.txt", "").replace("Synth_", "Synth_NetRate_")+".npy")

    print(outputfolder)
    intToUsr = {}
    with open(outputfolder+filename+"_int_to_users.txt", "r") as f:
        for line in f:
            i, u = line.replace("\n", "").split("\t")
            intToUsr[int(i)] = int(u)

    N = 1000  # Has to be larger
    K = 1
    tabvsctrue = []
    alinftot = np.zeros((N, N))
    for u in range(len(al)):
        a = al[u]
        for v in list(a.nonzero()[0]):
            if u!=v:
                alinftot[intToUsr[v], intToUsr[u]] = a[v][0]

    # Save results
    tabMAEs = []
    tabNMAEs = []
    labTrueF1Tot, labInfF1Tot = [], []
    tabErr = []
    lfls = [f for f in os.listdir("data/Synth/") if datafile.replace("_events.txt", "").replace("data/Synth/", "")+"_alpha_c" in f]
    with open("output_BL/results_Netrate_"+datafile.replace("_events.txt", "").replace("data/Synth/", "")+".txt", "w+") as f:
        for ctrue_file in lfls:
            altrue_deso = sparse.load_npz(f"data/Synth/{ctrue_file}").todense()
            altrue = np.array(altrue_deso).reshape(altrue_deso.shape[:2])
            if "PolBlogs" in ctrue_file:
                N_true = 700
            else:
                N_true = np.max([500, altrue.shape[0]])

            altrue = sparse.COO(altrue)
            alinftot = sparse.COO(alinftot)
            altrue.shape = alinftot.shape = (N_true,N_true)
            altrue = altrue.todense()
            alinftot = alinftot.todense()


            labTrueF1 = (altrue.flatten()>0).astype(int)
            labInfF1 = (alinftot.flatten()>0).astype(int)
            labTrueF1Tot += list(labTrueF1)
            labInfF1Tot += list(labInfF1)

            z = altrue == 0
            nnz = altrue.nonzero()
            MAE = np.mean(np.abs(alinftot-altrue))
            MAEZ = np.mean(np.abs(alinftot[z]-altrue[z]))
            MAENNZ = np.mean(np.abs(alinftot[nnz]-altrue[nnz]))
            NMAENNZ = np.mean(np.abs(alinftot[nnz]-altrue[nnz])/altrue[nnz])
            tabMAEs.append(MAENNZ)
            tabNMAEs.append(NMAENNZ)
            tabErr += list(np.abs(alinftot[nnz]-altrue[nnz]))
            print(MAENNZ, NMAENNZ)
            f.write(f"{ctrue_file}\t{MAENNZ}\t{NMAENNZ}\t{MAE}\t{MAEZ}\n")
        avgMAE, stdMAE = np.mean(tabMAEs), np.std(tabMAEs)
        avgNMAE, stdNMAE = np.mean(tabNMAEs), np.std(tabNMAEs)
        MAETot = np.mean(tabErr)
        F1_glob = f1_score(labTrueF1Tot, labInfF1Tot)
        AUC = roc_auc_score(labTrueF1Tot, labInfF1Tot)
        print()
        print(avgMAE, stdMAE, avgNMAE, stdNMAE)
        print(MAETot, np.shape(tabErr))
        print(F1_glob, AUC)
        print()
        f.write(f"\nMean MAE = {avgMAE} pm {stdMAE}\nMean MAE = {avgNMAE} pm {stdNMAE}\n")
        f.write(f"F1 = {F1_glob}\tAUC = {AUC}\n")


