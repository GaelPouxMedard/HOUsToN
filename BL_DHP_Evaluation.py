import bz2
import pickle
import os
import re
from BL_DHP_utils import *
from scipy.spatial.distance import jensenshannon
import multiprocessing
import tqdm
import time


def loadFit(folder, file, r):
    try:
        data = bz2.BZ2File(folder+file, "rb")
        data = pickle.load(data)
    except:
        particles = []
        with open(folder+file, "r", encoding="utf-8") as f:
            for line in f.read().replace("\n ", " ").split("\n"):
                indic = line.split("\t")[0]
                if indic=="Particle":
                    try:
                        particles.append(p)
                    except:
                        pass
                    _, pIter, weight, docs2cluster_ID = line.split("\t")
                    p = Particle(float(weight))
                    p.docs2cluster_ID = eval(docs2cluster_ID)
                elif indic=="Cluster":
                    _, c, alpha0, alpha, likTxt, word_count = line.replace("\n", "").split("\t")
                    c = int(c)
                    alpha0 = np.array(eval(alpha0.replace("    ", " ").replace("   ", " ").replace("  ", " ").replace(" ", ", ")))
                    alpha = np.array(eval(alpha.replace("    ", " ").replace("   ", " ").replace("  ", " ").replace(" ", ", ")))
                    likTxt = float(likTxt)
                    word_count = int(word_count)

                    cluster = Cluster(c, 1, alpha0)
                    cluster.alpha = alpha
                    cluster.txtLikelihood = likTxt
                    cluster.word_count = word_count
                    p.clusters[c]=cluster
        data = particles

    return data

def loadData(folder, file, r):
    news_items = []

    file = file.replace("_particles_compressed.pklbz2", "").replace("_particles.txt", "").replace(f"_r={r}", "").replace(f"_{r}", "")
    for i in reversed(range(100)):
        file = file.replace(f"_runDS={i}", "")
    with open(folder+file+"_events.txt", "r") as f:
        for i, line in enumerate(f):
            l = line.replace("\n", "").split("\t")
            clusTemp = int(float(l[0]))
            clusTxt = int(float(l[1]))
            timestamp = float(l[2])
            words = l[3].split(",")
            uniquewords, cntwords = np.unique(words, return_counts=True)
            uniquewords, cntwords = np.array(uniquewords, dtype=int), np.array(cntwords, dtype=int)

            tup = (i, timestamp, (uniquewords, cntwords), np.sum(cntwords), clusTemp, clusTxt)
            news_items.append(tup)
    with open(folder+file+"_lamb0.txt", "r") as f:
        lamb0 = float(f.read().replace("\n", ""))

    means = np.loadtxt(folder+file+"_means.txt")
    sigs = np.loadtxt(folder+file+"_sigs.txt")
    try:
        alpha = np.load(folder+file+"_alpha.npy")
    except:
        alpha = None

    return news_items, lamb0, means, sigs, alpha

def getParamsFile(file):
    nbClasses = int(re.findall("(?<=_nbclasses=)(.*)(?=_lg=)", file)[0])
    lg = int(re.findall("(?<=_lg=)(.*)(?=_overlapvoc=)", file)[0])
    overlap_voc = float(re.findall("(?<=_overlapvoc=)(.*)(?=_overlaptemp=)", file)[0])
    overlap_temp = float(re.findall("(?<=_overlaptemp=)(.*)(?=_r=)", file)[0])
    r = float(re.findall("(?<=_r=)(.*)(?=_percrandomizedclus=)", file)[0])
    perc_rand = float(re.findall("(?<=_percrandomizedclus=)(.*)(?=_vocperclass=)", file)[0])
    vocPerClass = int(re.findall("(?<=_vocperclass=)(.*)(?=_wordsperevent=)", file)[0])
    wordsPerEvent = int(re.findall("(?<=_wordsperevent=)(.*)(?=_run=)", file)[0])
    run = int(re.findall("(?<=_run=)(.*)(?=_runDS=)", file)[0])
    runDS = int(re.findall("(?<=_runDS=)(.*)(?=_particles)", file)[0])

    return nbClasses, lg, overlap_voc, overlap_temp, r, perc_rand, vocPerClass, wordsPerEvent, run, runDS



def getResultOneFile(args):
    file, folderFit, folderData, params = args
    nbClasses, lg, overlap_voc, overlap_temp, r, perc_rand, vocPerClass, wordsPerEvent, run, runDS = params
    try:
        news_items, lamb0, means, sigs, alpha_true = loadData(folderData, file, r)
        particles = loadFit(folderFit, file, r)
    except:
        return ""
    if len(particles[0].docs2cluster_ID)%1000==2:  # To avoid considering incomplete particles & correspondance len(doc2clusid)=i+1 + on save à i%1000==1
        return ""

    scoresConfMat = confMat(news_items[:len(particles[0].docs2cluster_ID)+1], particles)
    scoresCompDist = compDists(news_items, alpha_true, particles, means, sigs)
    txt = ""
    for pIter in scoresConfMat:
        res = scoresConfMat[pIter]
        resDist = scoresCompDist[pIter]
        txt += f"{nbClasses, lg, overlap_voc, overlap_temp, r, perc_rand, vocPerClass, wordsPerEvent, run, runDS, pIter}\t{res}\t{resDist}\n"

    return txt

def computeResults():
    folderFit = "output/Synth/"
    folderData = "data/Synth/"

    listfiles = os.listdir(folderFit)
    nbFilesSeen = 0
    with open("results.txt", "w+") as f_ov:
        for file in listfiles:
            nbFilesSeen += 1
            params = getParamsFile(file)
            nbClasses, lg, overlap_voc, overlap_temp, r, perc_rand, vocPerClass, wordsPerEvent, run, runDS = params
            if perc_rand!=0. or run!=0 or runDS != 0:
                continue
                pass
            print(nbFilesSeen*100./len(listfiles), "% -", file)
            args = file, folderFit, folderData, params
            txt = getResultOneFile(args)
            f_ov.write(txt)
            print(txt.split("\n")[0])

def computeResultsMultiprocess(processes=6, loop=False):
    folderFit = "output/Synth/"
    folderData = "data/Synth/"

    filesSeen = []
    with open("filesSeen.txt", "r") as f:
        for line in f:
            filesSeen.append(line.replace("\n", ""))

    txtTot = ""
    with open("results.txt", "a+") as f_ov:
        while True:
            print("Computing the metrics")
            listfiles = os.listdir(folderFit)
            listfiles = [file for file in listfiles if "_particle" in file]
            args = [(file, folderFit, folderData, getParamsFile(file)) for file in listfiles if file not in filesSeen]
            with multiprocessing.Pool(processes=processes) as p:
                with tqdm.tqdm(total=len(args)) as progress:
                    for i, res in enumerate(p.imap(getResultOneFile, args)):
                        txt = res
                        f_ov.write(txt)
                        #print(txt.split("\n")[0])
                        txtTot += txt
                        if txt != "":
                            with open("filesSeen.txt", "a+") as f_filesseen:
                                f_filesseen.write(args[i][0]+"\n")
                                filesSeen.append(args[i][0])
                        progress.update()

            if not loop:
                break
            time.sleep(10)

    with open("results_fin.txt", "w+") as f_ov:
        f_ov.write(txtTot)



def confMat(news_events, particles):
    from sklearn.metrics import homogeneity_completeness_v_measure, normalized_mutual_info_score, adjusted_rand_score
    scores = {}

    for pIter, p in enumerate(particles):
        try:
            arrLabsTrueTxt, arrLabsTrueTmp = [], []
            for index, t, _, clusTemp, clusTxt in news_events:
                arrLabsTrueTxt.append(clusTxt)
                arrLabsTrueTmp.append(clusTemp)
        except:
            arrLabsTrueTxt, arrLabsTrueTmp = [], []
            for index, t, _, _, clusTemp, clusTxt in news_events:
                arrLabsTrueTxt.append(clusTxt)
                arrLabsTrueTmp.append(clusTemp)
        arrLabsInf = np.array(p.docs2cluster_ID)

        try:
            K = len(set(arrLabsInf))
            NMITxt = normalized_mutual_info_score(arrLabsTrueTxt, arrLabsInf)
            AdjRandTxt = adjusted_rand_score(arrLabsTrueTxt, arrLabsInf)
            NMI_last = normalized_mutual_info_score(arrLabsTrueTmp[-500:], arrLabsInf[-500:])
            AdjRandTmp = adjusted_rand_score(arrLabsTrueTmp, arrLabsInf)
            VmeasTxt = homogeneity_completeness_v_measure(arrLabsTrueTxt, arrLabsInf)
            VmeasTmp = homogeneity_completeness_v_measure(arrLabsTrueTmp, arrLabsInf)
            LogL = p.log_update_prob
            #print("K =", K, "\tNMI txt =", NMITxt, "\tNMI tmp =", NMITmp, "\tAdjrand txt =", AdjRandTxt, "\tAdjrand tmp =", AdjRandTmp, "\tVmeas txt =", VmeasTxt, "\tVmeas tmp =", VmeasTmp, "\tLik =", LogL)
        except Exception as e:
            print(e)
            K = np.nan
            NMITxt = np.nan
            AdjRandTxt = np.nan
            NMI_last = np.nan
            AdjRandTmp = np.nan
            VmeasTxt = (np.nan, np.nan, np.nan)
            VmeasTmp = (np.nan, np.nan, np.nan)
            LogL = np.nan
            pass

        scores[pIter] = [K, NMITxt, NMI_last, AdjRandTxt, AdjRandTmp, VmeasTxt, VmeasTmp, LogL]
    return scores

def compDists(news_events, alphaTrue, particles, means, sigs):
    # news_event = [(i, timestamp, (uniquewords, cntwords), np.sum(cntwords), clusTemp, clusTxt)]
    dt = np.linspace(0, np.max(means)+3*np.max(sigs), 1000)
    errTot, divTot = 0., 0.
    JSTot, divJSTot = 0., 0.
    avgErrParts = {}
    BL = np.ones((len(alphaTrue), len(alphaTrue), len(means)))/len(means)
    JSTotBL, divJSTotBL = 0., 0.
    for pIter, p in enumerate(particles):
        try:
            arrLabsTrueTxt, arrLabsTrueTmp = [], []
            for index, t, _, clusTemp, clusTxt in news_events[:-1]:
                arrLabsTrueTxt.append(clusTxt)
                arrLabsTrueTmp.append(clusTemp)
        except:
            arrLabsTrueTxt, arrLabsTrueTmp = [], []
            for index, t, _, _, clusTemp, clusTxt in news_events[:-1]:
                arrLabsTrueTxt.append(clusTxt)
                arrLabsTrueTmp.append(clusTemp)
        arrLabsInf = np.array(p.docs2cluster_ID)

        absErr, div = 0., 0.
        un, cnt = np.unique(arrLabsInf, return_counts=True)
        donotconsider = un[cnt<0]#
        for cTrue, cInf in zip(arrLabsTrueTmp, arrLabsInf):
            if cInf in donotconsider:
                pass
                continue
            err = np.sum(np.abs(alphaTrue[cTrue,cTrue] - p.clusters[cInf].alpha))
            absErr += err
            div += len(alphaTrue[-1,-1]) # [:-1] bc l'écart sur la dernière valeur est redondant

            distTrue = triggering_kernel(alphaTrue[cTrue,cTrue], means, dt, sigs, donotsum=True)
            distInf = triggering_kernel(p.clusters[cInf].alpha, means, dt, sigs, donotsum=True)
            #print(alphaTrue[cTrue,cTrue], p.clusters[cInf].alpha)
            errJS = jensenshannon(distTrue, distInf)
            JSTot += errJS
            divJSTot += 1

            distTrue = triggering_kernel(alphaTrue[cTrue,cTrue], means, dt, sigs, donotsum=True)
            distInf = triggering_kernel(BL[0, 0], means, dt, sigs, donotsum=True)
            errJSBL = jensenshannon(distTrue, distInf)
            JSTotBL += errJSBL
            divJSTotBL += 1

        errTot += absErr
        divTot += len(arrLabsInf)*(len(alphaTrue[-1,-1]))
        '''
        errClus = []
        maxc = -1
        maxl = -1
        #print(np.unique(np.array(news_events, dtype=object)[:, 4], return_counts=True))
        clus, cnt = np.unique(p.docs2cluster_ID, return_counts=True)
        clus = clus[np.argpartition(cnt, -5)[-5:]]
        cnt = cnt[np.argpartition(cnt, -5)[-5:]]
        for c in clus:
            contentClus = np.where(np.array(p.docs2cluster_ID, dtype=int)==int(c))[0]
            #print(contentClus)

            if len(contentClus)>maxl:
                maxc = c
                maxl = len(contentClus)
            absErr, div = 0., 0.
            for index in contentClus:
                err = np.sum(np.abs(alphaTrue[news_events[index][4],news_events[index][4]] - p.clusters[c].alpha))
                absErr += err
                div += len(alphaTrue[-1,-1]) # [:-1] bc l'écart sur la dernière valeur est redondant

                distTrue = triggering_kernel(alphaTrue[news_events[index][4],news_events[index][4]], means, dt, sigs, donotsum=True)
                distInf = triggering_kernel(p.clusters[c].alpha, means, dt, sigs, donotsum=True)
                errJS = jensenshannon(distTrue, distInf)
                JSTot += errJS
                divJSTot += 1

            errTot += absErr
            divTot += len(contentClus)*(len(alphaTrue[-1,-1]))

            errClus.append((len(contentClus), absErr/(div+1e-20)))
            '''
        if divTot != 0 and pIter==0 and False:
            print("Particle", pIter, "Err", errTot/divTot, JSTot/divJSTot, JSTotBL/divJSTotBL, len(p.clusters))#, errClus, alphaTrue[0,0], alphaTrue[1,1], p.clusters[maxc].alpha)
        MAE = errTot/(divTot+1e-20)
        MJS = JSTot/(divJSTot+1e-20)
        MJSBL = JSTotBL/(divJSTotBL+1e-20)
        avgErrParts[pIter] = (MAE, MJS, MJSBL)

    return avgErrParts


if __name__ == "__main__":
    pass

