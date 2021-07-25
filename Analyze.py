import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import os
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, f1_score, roc_auc_score
import sparse
import re
import pickle
import networkx as nx
import multidict
from wordcloud import WordCloud
import itertools
from scipy.stats import sem
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'

limObs = 55000

def getFrequencyDictForText(words):
    fullTermsDict = multidict.MultiDict()
    tmpDict = {}

    # making dict for counting frequencies
    for text in words:
        if re.match("a|the|an|the|to|in|for|of|or|by|with|is|on|that|be", text):
            continue
        val = tmpDict.get(text, 0)
        tmpDict[text.lower()] = val + 1
    for key in tmpDict:
        fullTermsDict.add(key, tmpDict[key])
    return fullTermsDict

def makeImage(words):
    #alice_mask = np.array(Image.open("alice_mask.png"))
    text = getFrequencyDictForText(words)

    x, y = np.ogrid[:1000, :1000]
    mask = (x - 500) ** 2 + (y - 500) ** 2 > 500 ** 2
    mask = 255 * mask.astype(int)
    wc = WordCloud(background_color="white", max_words=500, mask=mask, colormap="cividis")
    # generate word cloud
    wc.generate_from_frequencies(text)

    return wc

def readObservations(name):
    observations = []
    wdToIndex, index = {}, 0
    nodeToInt, cascToInt = {}, {}
    index_node, index_casc = 0, 0
    with open(name, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            l = line.replace("\n", "").split("\t")
            i_doc = int(l[0])
            timestamp = float(l[1])
            words = l[2].split(",")
            node = int(l[3])
            cascadeNumber = int(l[4])
            try:  # If Synth data
                clus = int(l[5])
            except:
                clus = -1
            uniquewords, cntwords = np.unique(words, return_counts=True)
            for un in uniquewords:
                if un not in wdToIndex:
                    wdToIndex[un] = index
                    index += 1
            uniquewords = [wdToIndex[un] for un in uniquewords]
            uniquewords, cntwords = np.array(uniquewords, dtype=int), np.array(cntwords, dtype=int)

            if node not in nodeToInt:
                nodeToInt[node] = index_node
                index_node += 1
            if cascadeNumber not in cascToInt:
                cascToInt[cascadeNumber] = index_casc
                index_casc += 1

            tup = (i_doc, timestamp, (uniquewords, cntwords), nodeToInt[node], cascToInt[cascadeNumber], clus)
            observations.append(tup)

    return observations, wdToIndex, nodeToInt, cascToInt

def readResultsBaselines():
    resBL = {}
    for net in ["PL", "ER", "PolBlogs"]:
        resBL[net]={}
        for IDHP in ["", "_IDHP"]:
            resTmp = {"NMI": {}, "ARI": {}}
            with open(f"output_BL/results{IDHP}_Synth_{net}_OL=0.0_wdsPerObs=5_vocPerClass=100.txt", "r") as f:
                for line in f:
                    time, nbObs, NMI, NMI_last, ARI = line.replace("\n", "").split("\t")
                    if int(nbObs)>limObs or int(nbObs)<10:
                        continue
                    resTmp["NMI"][int(nbObs)] = float(NMI)
                    resTmp["ARI"][int(nbObs)] = float(ARI)
            if IDHP=="": txtIDHP = "DHP"
            if IDHP=="_IDHP":
                txtIDHP = "IDHP"
                continue

            resBL[net][txtIDHP] = resTmp

    return resBL

def readParticle(name):
    particles = []
    with open(name+"_particles.txt", "r") as f:
        for line in f:
            tab = line.replace("\n", "").split("\t")
            if tab[0] == "Particle":
                entry, num_part, weight, docs2cluster_ID, docs2cluster_index = tab
                num_part = int(num_part)
                weight = float(weight)
                docs2cluster_ID = eval(docs2cluster_ID)
                docs2cluster_index = eval(docs2cluster_index)
                particles.append({})
                particles[-1]["num_part"] = num_part
                particles[-1]["clusters"] = {}
                particles[-1]["docs2cluster_ID"] = docs2cluster_ID
                particles[-1]["docs2cluster_index"] = docs2cluster_index

            elif tab[0] == "Cluster":
                entry, num_clus, likTxt, word_count, text = tab
                num_clus = int(num_clus)
                alpha = pickle.load(open(name+f"_alphas_{particles[-1]['num_part']}_{num_clus}.pkl","rb"))
                likTxt = float(likTxt)
                word_count = int(word_count)
                text = text.replace("[", "").replace("]", "").split(" ")
                particles[-1]["clusters"][num_clus] = {}
                particles[-1]["clusters"][num_clus]["alpha"] = alpha
                particles[-1]["clusters"][num_clus]["text"] = text
                particles[-1]["clusters"][num_clus]["likTxt"] = likTxt
    return particles

def evaluate(observations, part, base_name, base_name_fit, r):
    intToUsr = {}
    if "Synth" in base_name_fit:
        with open("output/Synth/"+base_name_fit+"indexNodes.txt", "r") as f:
            for line in f:
                i, u = line.replace("\n", "").split("\t")
                intToUsr[int(i)] = int(u)
        base_name_fit = base_name_fit.replace("data/Synth/", "").replace("_events.txt", "")
    else:
        with open("output/Memetracker/"+base_name_fit+"indexNodes.txt", "r") as f:
            for line in f:
                i, u = line.replace("\n", "").split("\t")
                intToUsr[int(i)] = int(u)
        base_name_fit = base_name_fit.replace("data/Memetracker/", "").replace("_events.txt", "")


    tabRes = []
    observations = np.array(observations, dtype=object)
    nbObs = len(part["docs2cluster_index"])

    clusTrue = observations[part["docs2cluster_index"], -1]
    clusInf = part["docs2cluster_ID"]
    NMI = normalized_mutual_info_score(clusTrue, clusInf)
    ARI = adjusted_rand_score(clusTrue, clusInf)
    nbClus = len(part["clusters"])

    if "Memetracker" in base_name_fit:
        with open("figures/results_memetracker.txt", "a+") as f:
            f.write(f"{NMI}\t{ARI}\t{nbClus}\n")
        print("NMI", NMI, "ARI", ARI, "# clus", nbClus)
        return NMI, ARI, -1, -1, -1, -1, -1, -1, nbClus, -1, -1
    alloc = np.array(part["docs2cluster_ID"])
    labTrueF1Tot, labInfF1Tot = [], []
    tabErrTot = []
    listErrs = []
    listNErrs = []
    for c in part["clusters"]:
        N = 1000  # Has to be larger than # nodes
        K = 1
        tabvsctrue = []
        tabF1 = []
        tabTrueF1Tmp, tabInfF1Tmp = [], []
        tabErrTmp = []
        alinftot = np.zeros((N, N))
        for u in part["clusters"][c]["alpha"]:
            a = part["clusters"][c]["alpha"][u].todense()
            for v in list(a.nonzero()[0]):
                if u!=v:
                    alinftot[intToUsr[v], intToUsr[u]] = a[v][0]

        pop = len(alloc[alloc == c])
        if pop <= len(alloc)*0.05:
            continue

        lfls = [f for f in os.listdir("data/Synth/") if base_name.replace("_events", "")+"_alpha_c" in f]
        for ctrue_file in lfls:
            altrue_deso = sparse.load_npz(f"data/Synth/{ctrue_file}").todense()
            altrue = np.array(altrue_deso).reshape(altrue_deso.shape[:2])
            if "PolBlogs" in ctrue_file:
                N_true = 700
            else:
                N_true = np.max([500, altrue.shape[0]])

            altrue[altrue<1e-3] = 0
            alinftot[alinftot<1e-3] = 0

            altrue = sparse.COO(altrue)
            alinftot = sparse.COO(alinftot)
            altrue.shape = alinftot.shape = (N_true,N_true)
            altrue = altrue.todense()
            alinftot = alinftot.todense()

            z = altrue == 0
            nnz = altrue.nonzero()
            MAE = np.mean(np.abs(alinftot-altrue))
            MAEZ = np.mean(np.abs(alinftot[z]-altrue[z]))
            MAENNZ = np.mean(np.abs(alinftot[nnz]-altrue[nnz]))
            NMAENNZ = np.mean(np.abs((altrue[nnz] - alinftot[nnz])/altrue[nnz]))

            labTrueF1 = (altrue.flatten()>0).astype(int)
            labInfF1 = (alinftot.flatten()>0).astype(int)
            tabTrueF1Tmp.append(labTrueF1)
            tabInfF1Tmp.append(labInfF1)
            tabErrTmp.append(np.abs(alinftot[nnz]-altrue[nnz]))
            F1clus = f1_score(labTrueF1, labInfF1)

            tabvsctrue.append((MAE, MAENNZ, MAEZ, NMAENNZ))
            tabF1.append(F1clus)
        tabvsctrue = np.array(tabvsctrue)
        tabF1 = np.array(tabF1)
        tabTrueF1Tmp = np.array(tabTrueF1Tmp, dtype=object)
        tabInfF1Tmp = np.array(tabInfF1Tmp, dtype=object)
        tabErrTmp = np.array(tabErrTmp, dtype=object)
        try:
            c_true = np.where(tabvsctrue[:, 1] == np.min(tabvsctrue[:, 1]))[0][0]
            select = np.sum(tabvsctrue, axis=1) == np.min(np.sum(tabvsctrue, axis=1))
            tabvsctrue = tabvsctrue[select]
            F1 = tabF1[select]
            labTrueF1 = tabTrueF1Tmp[select][0]
            labInfF1 = tabInfF1Tmp[select][0]
            err = tabErrTmp[select][0]
            labTrueF1Tot += list(labTrueF1)
            labInfF1Tot += list(labInfF1)
            tabErrTot += list(err)
            listErrs.append(tabvsctrue[0, 1])
            listNErrs.append(tabvsctrue[0, 3])
        except:
            c_true = 0
            F1 = -1
        tabRes.append((c, c_true, pop, tabvsctrue, F1))
        #print("Clus", c, "Pop", pop, "MinErr", tabvsctrue)
    try:
        F1_glob = f1_score(labTrueF1Tot, labInfF1Tot)
        AUC = roc_auc_score(labTrueF1Tot, labInfF1Tot)
    except Exception as e:
        print(e)
        F1_glob = -1
        AUC = -1
    MAE = np.mean(listErrs)
    NMAE = np.mean(listNErrs)
    MAESTD = sem(listErrs)
    NMAESTD = sem(listNErrs)
    MAETot = np.mean(tabErrTot)
    print("NMI", NMI, "ARI", ARI, "# clus", nbClus, "AUC-net", AUC, "MAE", MAE, "MAE-tot", MAETot, "F1_glob", F1_glob)
    return NMI, ARI, F1_glob, AUC, MAETot, MAE, MAESTD, NMAE, NMAESTD, nbClus, nbObs, tabRes



def computeAllResults():
    part_num = 4
    arrNet = ["ER", "PolBlogs", "PL", ]
    arrR = [0., 1.]
    dicRes = {}
    dicResChangeNet = {}
    dicResOVerlaps = {}

    allFilesSynth = os.listdir("output/Synth/")
    for network in arrNet:
        dicRes[network] = {}
        base_name = f"Synth_{network}_OL=0.0_wdsPerObs=5_vocPerClass=100_events"
        observations, wdToIndex, nodeToInt, cascToInt = readObservations("data/Synth/"+base_name+".txt")
        for r in arrR:
            base_name_fit = base_name + f"_r={r}_theta0=0.1_particlenum={part_num}_run_0_"
            files = [f for f in allFilesSynth if base_name_fit in f and "particles.txt" in f]
            dicResXP = {}
            for file in files:
                if "done" in file:
                    continue
                print(file)
                t_snapshot = re.findall("(?<=_t=)(.*)(?=h_)", file)[0]
                particles = readParticle("output/Synth/"+file.replace("h_particles.txt", "h"))
                t_snapshot = float(t_snapshot)
                dicResXP[t_snapshot] = {}
                for p in particles:
                    dicResXP[t_snapshot][p["num_part"]] = {}
                    NMI, ARI, F1_glob, AUC, MAETot, MAE, MAESTD, NMAE, NMAESTD, nbClus, nbObs, tabRes = evaluate(observations, p, base_name, base_name_fit, r)
                    dicResXP[t_snapshot][p["num_part"]]["NMI"] = NMI
                    dicResXP[t_snapshot][p["num_part"]]["ARI"] = ARI
                    dicResXP[t_snapshot][p["num_part"]]["F1_glob"] = F1_glob
                    dicResXP[t_snapshot][p["num_part"]]["AUC"] = AUC
                    dicResXP[t_snapshot][p["num_part"]]["MAETot"] = MAETot
                    dicResXP[t_snapshot][p["num_part"]]["MAE"] = MAE
                    dicResXP[t_snapshot][p["num_part"]]["MAESTD"] = MAESTD
                    dicResXP[t_snapshot][p["num_part"]]["NMAE"] = NMAE
                    dicResXP[t_snapshot][p["num_part"]]["NMAESTD"] = NMAESTD
                    dicResXP[t_snapshot][p["num_part"]]["nbClus"] = nbClus
                    dicResXP[t_snapshot][p["num_part"]]["nbObs"] = nbObs
                    dicResXP[t_snapshot][p["num_part"]]["clusters"] = {}
                    for c, c_true, pop, tabvsctrue, F1 in tabRes:
                        dicResXP[t_snapshot][p["num_part"]]["clusters"][c] = {}
                        dicResXP[t_snapshot][p["num_part"]]["clusters"][c]["err"] = tabvsctrue
                        dicResXP[t_snapshot][p["num_part"]]["clusters"][c]["F1"] = F1
                        dicResXP[t_snapshot][p["num_part"]]["clusters"][c]["pop"] = pop
                        dicResXP[t_snapshot][p["num_part"]]["clusters"][c]["LikTxt"] = p["clusters"][c]["likTxt"]
            dicRes[network][r] = dicResXP

    pickle.dump(dicRes, open("output/Synth/results_synth.pkl","wb"))



def plotNetworksSynth():
    part_num = 4
    arrNet = ["ER", "PL", "PolBlogs"]
    arrR = [1.]
    allFilesSynth = os.listdir("output/Synth/")
    for network in arrNet:
        base_name = f"Synth_{network}_OL=0.0_wdsPerObs=5_vocPerClass=100_events"
        observations, wdToIndex, nodeToInt, cascToInt = readObservations("data/Synth/"+base_name+".txt")
        for r in arrR:
            base_name_fit = base_name + f"_r={r}_theta0=0.1_particlenum={part_num}_run_0_"
            files = [f for f in allFilesSynth if base_name_fit in f and "particles.txt" in f]
            intToUsr = {}
            with open("output/Synth/"+base_name_fit+"indexNodes.txt", "r") as f:
                for line in f:
                    i, u = line.replace("\n", "").split("\t")
                    intToUsr[int(i)] = int(u)
            indToWd = {}
            with open("output/Synth/"+base_name_fit+"indexWords.txt", "r") as f:
                for line in f:
                    i, x = line.replace("\n", "").split("\t")
                    indToWd[int(i)] = x

            maxt, selectfile = 0, ""
            for file in files:
                if "done" in file:
                    continue
                t_snapshot = re.findall("(?<=_t=)(.*)(?=h_)", file)[0]
                t_snapshot_num = float(t_snapshot)
                if t_snapshot_num>maxt:
                    maxt=t_snapshot_num
                    selectfile = file
            for file in [selectfile]:
                particles = readParticle("output/Synth/"+file.replace("h_particles.txt", "h"))
                p = particles[0]
                print(file)
                NMI, ARI, F1_glob, AUC, MAETot, MAE, MAESTD, NMAE, NMAESTD, nbClus, nbObs, tabRes = evaluate(observations, p, base_name, base_name_fit, r)

                for i_lay, layout_func in enumerate([nx.spring_layout]):
                    colors1 = iter([plt.cm.Set1(i) for i in range(10) if i!=5]*2)
                    colors2 = iter([plt.cm.Set1(i) for i in range(10) if i!=5]*2)
                    # Plot the graph
                    fac = 6
                    plt.figure(figsize=(len(tabRes)*fac, 3*fac))
                    for i_c, (c, c_true, pop, tabvsctrue, F1) in enumerate(tabRes):
                        plt.subplot(3, len(tabRes), i_c+1)
                        words = []
                        inds = np.array(p["docs2cluster_index"], dtype=int)
                        inds = inds[np.array(p["docs2cluster_ID"], dtype=int)==int(c)]
                        if len(inds)<100:
                            continue
                        for o in np.array(observations, dtype=object)[inds, 2]:
                            for wd, cnt in zip(o[0], o[1]):
                                try:
                                    for _ in range(cnt):
                                        words.append(indToWd[wd])
                                except:
                                    continue
                        #print(list(np.unique(words, return_counts=True)[1]))
                        wc = makeImage(words)
                        plt.imshow(wc, interpolation="bilinear")
                        plt.title(f"Pop = {len(inds)}/{len(p['docs2cluster_ID'])}")
                        plt.tight_layout()
                        plt.axis("off")
                        plt.tight_layout()

                        N = 1000
                        plt.subplot(3, len(tabRes), 2*len(tabRes) + i_c+1)
                        altrue_deso = sparse.load_npz(f"data/Synth/Synth_{network}_OL=0.0_wdsPerObs=5_vocPerClass=100_alpha_c{c_true}.npz").todense()
                        altrue = np.array(altrue_deso).reshape(altrue_deso.shape[:2])

                        A = np.zeros((N, N))
                        for u in p["clusters"][c]["alpha"]:
                            a = p["clusters"][c]["alpha"][u].todense()
                            for v in list(a.nonzero()[0]):
                                if u!=v:
                                    A[intToUsr[v], intToUsr[u]] = a[v][0]
                        if "PolBlogs" in file:
                            thres = 1e-2
                            rig = 0.1
                        else:
                            thres = 1e-1
                            rig = 0.05
                        A[A<thres] = 0
                        altrue[altrue<thres] = 0

                        if "PolBlogs" in file:
                            N_true = 700
                        else:
                            N_true = np.max([500, altrue.shape[0]])
                        altrue = sparse.COO(altrue)
                        A = sparse.COO(A)
                        altrue.shape = A.shape = (N_true,N_true)
                        altrue = altrue.todense()
                        A = A.todense()

                        g = nx.from_numpy_matrix(altrue, create_using=nx.DiGraph)
                        isolate = list(nx.isolates(g))
                        g.remove_nodes_from(isolate)
                        #small_components = sorted(nx.connected_components(g), key=len)[:-1]
                        #g.remove_nodes_from(itertools.chain.from_iterable(small_components))
                        if i_lay==0:
                            layout = layout_func(g, k=rig)
                        else:
                            layout = layout_func(g)

                        widths = nx.get_edge_attributes(g, 'weight')
                        nodelist = g.nodes()
                        edge_color_base=to_rgb(next(colors2))
                        try:
                            maxw = (np.max(list(widths.values()))-thres)**1
                        except:
                            maxw=1.
                        edge_color = np.array([[edge_color_base[0], edge_color_base[1], edge_color_base[2], (w-thres)**1/(maxw+1e-20)] for w in widths.values()])
                        nx.draw_networkx_nodes(g, layout, nodelist=nodelist, node_size=10, node_color='k')
                        nx.draw_networkx_edges(g, layout, edgelist=widths.keys(), width=1., edge_color=edge_color)
                        #nx.draw(g, pos=layout, node_size=5, width=0.2, edge_color=next(colors2), node_color="k")
                        plt.tight_layout()


                        plt.subplot(3, len(tabRes), len(tabRes)+i_c+1)
                        g = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
                        g.remove_nodes_from(isolate)
                        #g.remove_nodes_from(itertools.chain.from_iterable(small_components))

                        widths = nx.get_edge_attributes(g, 'weight')
                        nodelist = g.nodes()
                        edge_color_base=to_rgb(next(colors1))
                        try:
                            maxw = (np.max(list(widths.values()))-thres)**1
                        except:
                            maxw=1.
                        edge_color = np.array([[edge_color_base[0], edge_color_base[1], edge_color_base[2], (w-thres)**1/(maxw+1e-20)] for w in widths.values()])
                        nx.draw_networkx_nodes(g, layout, nodelist=nodelist, node_size=10, node_color='k')
                        nx.draw_networkx_edges(g, layout, edgelist=widths.keys(), width=1., edge_color=edge_color)
                        #nx.draw(g, pos=layout, node_size=5, width=0.2, edge_color=next(colors1), node_color="k")
                        plt.tight_layout()



                    plt.tight_layout()
                    if str(i_lay) not in os.listdir("figures/networks_synth/"): os.mkdir(f"figures/networks_synth/{i_lay}/")
                    plt.savefig(f"figures/networks_synth/{i_lay}/{network}-{r}-{float(maxt)}.png", dpi=300)
                    plt.savefig(f"figures/networks_synth/{i_lay}/{network}-{r}-{float(maxt)}.pdf")
                    plt.close('all')
    return

def plotNetworksReal():
    part_num = 4
    arrNet = ["Memetracker"]
    arrR = [1.]
    allFilesSynth = os.listdir("output/Memetracker/")
    trentemin = True
    txtTrentemin = ""
    if trentemin == True:
        txtTrentemin = "_30min"
    base_name = f"Memetracker{txtTrentemin}_events"
    observations, wdToIndex, nodeToInt, cascToInt = readObservations("data/Memetracker/"+base_name+".txt")
    for r in arrR:
        base_name_fit = base_name + f"_r={r}_theta0=0.01_particlenum={part_num}_run_0_"
        files = [f for f in allFilesSynth if base_name_fit in f and "particles.txt" in f]
        intToUsr = {}
        with open("output/Memetracker/"+base_name_fit+f"indexNodes.txt", "r") as f:
            for line in f:
                i, u = line.replace("\n", "").split("\t")
                intToUsr[int(i)] = int(u)
        intToNodemm = {}
        with open(f"data/Memetracker/intToNodes{txtTrentemin}.txt", "r") as f:
            for line in f:
                i, x = line.replace("\n", "").split("\t")
                intToNodemm[int(i)] = x
        nodeToTypemm = {}
        lgDoc = 8357589
        lim = lgDoc//5
        with open("data/Memetracker/clust-qt08080902w3mfq5.txt", encoding="utf-8") as f:
            [f.readline() for _ in range(6)]
            for i, line in enumerate(f):
                if line == '\n': continue
                tab = line.replace("\n", "").split("\t")
                if tab[0]=='' and tab[1] == '':
                    node = tab[5].replace("http://", "").replace("https://", "")
                    node = node[:node.find("/")]
                    type = tab[4]
                    if node in intToNodemm.values():
                        nodeToTypemm[node] = type
                if i>lim: break
        nodeToType = {i: nodeToTypemm[intToNodemm[intToUsr[i]]] for i in intToUsr}

        indToWd = {}
        with open("output/Memetracker/"+base_name_fit+f"indexWords.txt", "r") as f:
            for line in f:
                i, x = line.replace("\n", "").split("\t")
                indToWd[i] = x
        indToWd2 = {}
        with open(f"data/Memetracker/intToWds{txtTrentemin}.txt", "r") as f:
            for line in f:
                i, x = line.replace("\n", "").split("\t")
                indToWd2[i] = x


        indToWdFin = {}
        for ind in indToWd:
            indToWdFin[int(ind)] = indToWd2[indToWd[ind]]
        indToWd = indToWdFin
        for file in files:
            t_snapshot = re.findall("(?<=_t=)(.*)(?=h_)", file)[0]
            particles = readParticle("output/Memetracker/"+file.replace("h_particles.txt", "h"))
            p = particles[0]
            evaluate(observations, p, "Memetracker_30min_events_r=1.0_theta0=0.01_particlenum=4_run_0_", "Memetracker_30min_events_r=1.0_theta0=0.01_particlenum=4_run_0_", r)

            # Find the layout
            N = 1000
            A_sum = np.zeros((N, N))
            allClus = False  # ================
            for c in p["clusters"]:
                if c not in [3, 12, 20, 21, 38, 39, 59, 65, 78, 84, 86, 89, 96, 105, 113, 141, 196]:
                    if not allClus:
                        continue
                for u in p["clusters"][c]["alpha"]:
                    a = p["clusters"][c]["alpha"][u].todense()
                    for v in list(a.nonzero()[0]):
                        if u!=v:
                            A_sum[intToUsr[v], intToUsr[u]] += a[v][0]
            A_sum[A_sum<1e-3] = 0
            A_sum = A_sum.reshape(A_sum.shape[:2])
            g_sum = nx.from_numpy_matrix(A_sum, create_using=nx.DiGraph)
            isolate_sum = list(nx.isolates(g_sum))
            g_sum.remove_nodes_from(isolate_sum)


            for i_lay, layout_func in enumerate([nx.spring_layout]):
                colors1 = iter([plt.cm.Set1(i) for i in range(10) if i!=5 and i!=0]*200)
                if layout_func==nx.spring_layout:
                    layout = layout_func(g_sum, k=0.5)
                else:
                    layout = layout_func(g_sum)

                bbx, bby = [0., 0.], [0., 0.]
                for x,y in layout.values():
                    if x>bbx[1]: bbx[1] = x
                    if x<bbx[0]: bbx[0] = x
                    if y>bby[1]: bby[1] = y
                    if y<bby[0]: bby[0] = y
                bby[1] *= 1.2
                # Plot the graph
                fac = 6
                for c in p["clusters"]:
                    if c not in [3, 12, 20, 21, 38, 39, 59, 65, 78, 84, 86, 89, 96, 105, 113, 141, 196]:
                        if not allClus:
                            continue
                    print(c)
                    plt.figure(figsize=(fac, 2*fac))
                    plt.subplot(2, 1, 1)
                    words = []
                    inds = np.array(p["docs2cluster_index"], dtype=int)
                    inds = inds[np.array(p["docs2cluster_ID"], dtype=int)==int(c)]
                    if len(inds)<100:
                        continue
                    for o in np.array(observations, dtype=object)[inds, 2]:
                        for wd, cnt in zip(o[0], o[1]):
                            try:
                                for _ in range(cnt):
                                    words.append(indToWd[wd])
                            except:
                                continue
                    #print(list(np.unique(words, return_counts=True)[1]))
                    wc = makeImage(words)
                    plt.imshow(wc, interpolation="bilinear")
                    plt.title(f"Pop = {len(inds)}/{len(p['docs2cluster_ID'])}")
                    plt.tight_layout()
                    plt.axis("off")

                    plt.subplot(2, 1, 2)
                    N = 1000
                    K=1
                    A = np.zeros((N,N,K))

                    for u in p["clusters"][c]["alpha"]:
                        a = p["clusters"][c]["alpha"][u].todense()
                        for v in list(a.nonzero()[0]):
                            if u!=v:
                                A[intToUsr[v], intToUsr[u]] += a[v][0]
                    A[A<1e-3] = 0
                    A = A.reshape(A.shape[:2])
                    g = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
                    isolate = list(nx.isolates(g))
                    #g.remove_nodes_from(isolate)
                    g.remove_nodes_from(isolate_sum)
                    #small_components = sorted(nx.connected_components(g), key=len)[:-1]
                    #g.remove_nodes_from(itertools.chain.from_iterable(small_components))

                    widths = nx.get_edge_attributes(g, 'weight')
                    nodelist = g.nodes()
                    edge_color_base=to_rgb(next(colors1))
                    try:
                        maxw = np.max(list(widths.values()))
                    except:
                        maxw=1.
                    edge_color = np.array([[edge_color_base[0], edge_color_base[1], edge_color_base[2], (w+0.1)**2/(maxw+1e-20+0.1)**2] for w in widths.values()])
                    nodesBlogs = [u for u in nodelist if nodeToType[u]=="B"]
                    nodesMedia = [u for u in nodelist if nodeToType[u]=="M"]
                    nodesize = 20
                    nx.draw_networkx_nodes(g, layout, nodelist=nodesBlogs, node_size=nodesize*1.3, node_color='k')
                    nx.draw_networkx_nodes(g, layout, nodelist=nodesBlogs, node_size=nodesize, node_color='b')
                    nx.draw_networkx_nodes(g, layout, nodelist=nodesMedia, node_size=nodesize*1.3, node_color='k')
                    nx.draw_networkx_nodes(g, layout, nodelist=nodesMedia, node_size=nodesize, node_color='r')
                    nx.draw_networkx_edges(g, layout, edgelist=widths.keys(), width=1., edge_color=edge_color)

                    nodeInterest = np.where(np.sum(A, axis=0)==np.max(np.sum(A, axis=0)))[0][0]
                    plt.plot(layout[nodeInterest][0], layout[nodeInterest][1], "^", color="chartreuse", markersize=12, label=f"Most influenced node: {intToNodemm[intToUsr[nodeInterest]]}")
                    un, cnt = np.unique(np.array(observations, dtype=object)[inds, 3], return_counts=True)
                    nodeInterest = un[cnt==np.max(cnt[cnt!=np.max(cnt)])][0]
                    plt.plot(layout[nodeInterest][0], layout[nodeInterest][1], "*", color="chartreuse", markersize=12, label=f"2nd most frequent node: {intToNodemm[intToUsr[nodeInterest]]}")

                    plt.xlim(np.array(bbx)*1.1)
                    plt.ylim(np.array(bby)*1.1)
                    plt.legend(loc='upper center', ncol=1)
                    plt.tight_layout()
                    if str(i_lay) not in os.listdir("figures/networks_rw/"): os.mkdir(f"figures/networks_rw/{i_lay}/")
                    plt.savefig(f"figures/networks_rw/{i_lay}/Memetracker_{t_snapshot}_{c}.png", dpi=600)
                    plt.savefig(f"figures/networks_rw/{i_lay}/Memetracker_{t_snapshot}_{c}.pdf")
                    plt.close('all')

def plotMetrics():
    dicRes = pickle.load(open("output/Synth/results_synth.pkl","rb"))
    resBL = readResultsBaselines()

    arrRes = {}
    for net in dicRes:
        for r in dicRes[net]:
            lab = f"{net} - r={r}"
            if net == "PolBlogs":
                lab = f"Blogs - r={r}"
            arrTmpNMI = []
            arrTmpARI = []
            arrTmpF1 = []
            arrTmpAUC = []
            arrTmpMAETot = []
            arrTmpMAE = []
            arrTmpMAESTD = []
            arrTmpNMAE = []
            arrTmpNMAESTD = []
            arrTmpnbObs = []
            for t in dicRes[net][r]:
                part = 0
                t = float(t)
                NMI = dicRes[net][r][t][part]["NMI"]
                ARI = dicRes[net][r][t][part]["ARI"]
                F1_glob = dicRes[net][r][t][part]["F1_glob"]
                AUC = dicRes[net][r][t][part]["AUC"]
                MAETot = dicRes[net][r][t][part]["MAETot"]
                MAE = dicRes[net][r][t][part]["MAE"]
                MAESTD = dicRes[net][r][t][part]["MAESTD"]
                NMAE = dicRes[net][r][t][part]["NMAE"]
                NMAESTD = dicRes[net][r][t][part]["NMAESTD"]
                nbObs = dicRes[net][r][t][part]["nbObs"]

                if nbObs<10 or nbObs>limObs: continue

                arrTmpNMI.append(NMI)
                arrTmpARI.append(ARI)
                arrTmpF1.append(F1_glob)
                arrTmpAUC.append(AUC)
                arrTmpMAETot.append(MAETot)
                arrTmpMAE.append(MAE)
                arrTmpMAESTD.append(MAESTD)
                arrTmpNMAE.append(NMAE)
                arrTmpNMAESTD.append(NMAESTD)
                arrTmpnbObs.append(nbObs)

            arrTmpNMI = [x for _, x in sorted(zip(arrTmpnbObs, arrTmpNMI))]
            arrTmpARI = [x for _, x in sorted(zip(arrTmpnbObs, arrTmpARI))]
            arrTmpF1 = [x for _, x in sorted(zip(arrTmpnbObs, arrTmpF1))]
            arrTmpAUC = [x for _, x in sorted(zip(arrTmpnbObs, arrTmpAUC))]
            arrTmpMAETot = [x for _, x in sorted(zip(arrTmpnbObs, arrTmpMAETot))]
            arrTmpMAE = [x for _, x in sorted(zip(arrTmpnbObs, arrTmpMAE))]
            arrTmpMAESTD = [x for _, x in sorted(zip(arrTmpnbObs, arrTmpMAESTD))]
            arrTmpNMAE = [x for _, x in sorted(zip(arrTmpnbObs, arrTmpNMAE))]
            arrTmpNMAESTD = [x for _, x in sorted(zip(arrTmpnbObs, arrTmpNMAESTD))]
            arrTmpnbObs = [x for x, _ in sorted(zip(arrTmpnbObs, arrTmpARI))]
            arrRes[lab] = {}
            arrRes[lab]["nbObs"] = arrTmpnbObs
            arrRes[lab]["NMI"] = arrTmpNMI
            arrRes[lab]["ARI"] = arrTmpARI
            arrRes[lab]["F1"] = arrTmpF1
            arrRes[lab]["AUC"] = arrTmpAUC
            arrRes[lab]["MAE"] = arrTmpMAE
            arrRes[lab]["MAETot"] = arrTmpMAETot
            arrRes[lab]["MAESTD"] = arrTmpMAESTD
            arrRes[lab]["NMAE"] = arrTmpNMAE
            arrRes[lab]["NMAESTD"] = arrTmpNMAESTD

    plotAll = False
    if plotAll:
        fac = 4
        plt.figure(figsize=(3*fac, 3*fac))
        plt.subplot(3, 3, 1)
        for lab in arrRes:
            plt.plot(arrRes[lab]["nbObs"], arrRes[lab]["NMI"], label=lab)
            plt.xlabel("# observations")
            plt.ylabel("NMI")
            plt.ylim([0,1])
            plt.legend()

        plt.subplot(3, 3, 2)
        for lab in arrRes:
            plt.plot(arrRes[lab]["nbObs"], arrRes[lab]["ARI"], label=lab)
            plt.xlabel("# observations")
            plt.ylabel("Adj.RI")
            plt.ylim([0,1])
            plt.legend()

        plt.subplot(3, 3, 4)
        for lab in arrRes:
            plt.plot(arrRes[lab]["nbObs"], arrRes[lab]["F1"], label=lab)
            plt.xlabel("# observations")
            plt.ylabel("F1")
            plt.ylim([0,1])
            plt.legend()

        plt.subplot(3, 3, 5)
        for i, lab in enumerate(arrRes):
            plt.plot(arrRes[lab]["nbObs"], arrRes[lab]["MAE"], label=lab, c=f"C{i}")
            tabMAE = np.array(arrRes[lab]["MAE"])
            tabSTD = np.array(arrRes[lab]["MAESTD"])
            plt.fill_between(arrRes[lab]["nbObs"], tabMAE-tabSTD, tabMAE+tabSTD, color=f"C{i}", alpha=0.3)
            plt.xlabel("# observations")
            plt.ylabel("MAE")
            plt.ylim([0,1])
            plt.legend()


        plt.subplot(3, 3, 6)
        for lab in arrRes:
            plt.plot(arrRes[lab]["nbObs"], arrRes[lab]["AUC"], label=lab)
            plt.xlabel("# observations")
            plt.ylabel("AUC - ROC")
            plt.ylim([0,1])
            plt.legend()

        plt.subplot(3, 3, 7)
        for i, lab in enumerate(arrRes):
            print(arrRes[lab].keys())
            plt.plot(arrRes[lab]["nbObs"], arrRes[lab]["NMAE"], label=lab, c=f"C{i}")
            tabMAE = np.array(arrRes[lab]["NMAE"])
            tabSTD = np.array(arrRes[lab]["NMAESTD"])
            plt.fill_between(arrRes[lab]["nbObs"], tabMAE-tabSTD, tabMAE+tabSTD, color=f"C{i}", alpha=0.3)
            plt.xlabel("# observations")
            plt.ylabel("MAE")
            plt.ylim([0,1])
            plt.legend()



        plt.tight_layout()
        plt.savefig("figures/Results.pdf")
        #plt.show()

    else:
        fac = 3
        plt.figure(figsize=(3*fac, 2*fac))

        plt.subplot(2, 3, 1)
        colors = iter([plt.cm.Set1(i) for i in range(20) if i!=5])
        for lab in arrRes:
            if "PL" in lab:
                if "r=1" in lab:
                    plt.plot(arrRes[lab]["nbObs"], arrRes[lab]["NMI"], label="Ouston", c=next(colors))
                    print("NMI", lab, arrRes[lab]["nbObs"][-1], arrRes[lab]["NMI"][-1])
                    print("ARI", lab, arrRes[lab]["nbObs"][-1], arrRes[lab]["ARI"][-1])
                if "r=0" in lab:
                    plt.plot(arrRes[lab]["nbObs"], arrRes[lab]["NMI"], label="TopicCascade", c=next(colors))
                    print("NMI", lab, arrRes[lab]["nbObs"][-1], arrRes[lab]["NMI"][-1])
                    print("ARI", lab, arrRes[lab]["nbObs"][-1], arrRes[lab]["ARI"][-1])
        for IDHP in resBL["PL"]:
            plt.plot(resBL["PL"][IDHP]["NMI"].keys(), resBL["PL"][IDHP]["NMI"].values(), label=IDHP, c=next(colors))
            print("NMI PL", "IDHP:", IDHP, np.array(list(resBL["PL"][IDHP]["NMI"].keys()))[-1], np.array(list(resBL["PL"][IDHP]["NMI"].values()))[-1])
            print("ARI PL", "IDHP:", IDHP, np.array(list(resBL["PL"][IDHP]["ARI"].keys()))[-1], np.array(list(resBL["PL"][IDHP]["ARI"].values()))[-1])
        plt.xlabel("# observations")
        plt.ylabel("NMI")
        #plt.ylim([0,1])
        #plt.semilogx()
        plt.legend()
        plt.title("PL")
        plt.tight_layout()
        print()

        plt.subplot(2, 3, 2)
        colors = iter([plt.cm.Set1(i) for i in range(20) if i!=5])
        for lab in arrRes:
            if "ER" in lab:
                if "r=1" in lab:
                    plt.plot(arrRes[lab]["nbObs"], arrRes[lab]["NMI"], label="Ouston", c=next(colors))
                    print("NMI", lab, arrRes[lab]["nbObs"][-1], arrRes[lab]["NMI"][-1])
                    print("ARI", lab, arrRes[lab]["nbObs"][-1], arrRes[lab]["ARI"][-1])
                if "r=0" in lab:
                    plt.plot(arrRes[lab]["nbObs"], arrRes[lab]["NMI"], label="TopicCascade", c=next(colors))
                    print("NMI", lab, arrRes[lab]["nbObs"][-1], arrRes[lab]["NMI"][-1])
                    print("ARI", lab, arrRes[lab]["nbObs"][-1], arrRes[lab]["ARI"][-1])
        for IDHP in resBL["ER"]:
            plt.plot(resBL["ER"][IDHP]["NMI"].keys(), resBL["ER"][IDHP]["NMI"].values(), label=IDHP, c=next(colors))
            print("NMI ER", "IDHP:", IDHP, np.array(list(resBL["ER"][IDHP]["NMI"].keys()))[-1], np.array(list(resBL["ER"][IDHP]["NMI"].values()))[-1])
            print("ARI ER", "IDHP:", IDHP, np.array(list(resBL["ER"][IDHP]["ARI"].keys()))[-1], np.array(list(resBL["ER"][IDHP]["ARI"].values()))[-1])
        plt.xlabel("# observations")
        plt.ylabel("NMI")
        #plt.ylim([0,1])
        #plt.semilogx()
        plt.legend()
        plt.title("ER")
        plt.tight_layout()
        print()

        plt.subplot(2, 3, 3)
        colors = iter([plt.cm.Set1(i) for i in range(20) if i!=5])
        for lab in arrRes:
            if "Blogs" in lab:
                if "r=1" in lab:
                    plt.plot(arrRes[lab]["nbObs"], arrRes[lab]["NMI"], label="Ouston", c=next(colors))
                    print("NMI", lab, arrRes[lab]["nbObs"][-1], arrRes[lab]["NMI"][-1])
                    print("ARI", lab, arrRes[lab]["nbObs"][-1], arrRes[lab]["ARI"][-1])
                if "r=0" in lab:
                    plt.plot(arrRes[lab]["nbObs"], arrRes[lab]["NMI"], label="TopicCascade", c=next(colors))
                    print("NMI", lab, arrRes[lab]["nbObs"][-1], arrRes[lab]["NMI"][-1])
                    print("ARI", lab, arrRes[lab]["nbObs"][-1], arrRes[lab]["ARI"][-1])
        for IDHP in resBL["PolBlogs"]:
            plt.plot(resBL["PolBlogs"][IDHP]["NMI"].keys(), resBL["PolBlogs"][IDHP]["NMI"].values(), label=IDHP, c=next(colors))
            print("NMI PolBlogs", "IDHP:", IDHP, np.array(list(resBL["PolBlogs"][IDHP]["NMI"].keys()))[-1], np.array(list(resBL["PolBlogs"][IDHP]["NMI"].values()))[-1])
            print("ARI PolBlogs", "IDHP:", IDHP, np.array(list(resBL["PolBlogs"][IDHP]["ARI"].keys()))[-1], np.array(list(resBL["PolBlogs"][IDHP]["ARI"].values()))[-1])
        plt.xlabel("# observations")
        plt.ylabel("NMI")
        #plt.ylim([0,1])
        #plt.semilogx()
        plt.legend()
        plt.title("Blogs")
        plt.tight_layout()
        print()

        plt.subplot(2, 1, 2)
        colors = [plt.cm.Set1(i) for i in range(20) if i!=5]
        for lab in arrRes:
            ind = 0
            l = ""
            if "PL" in lab:
                ind = 0
                l = "PL"
            elif "ER" in lab:
                ind = 1
                l = "ER"
            elif "Blogs" in lab:
                ind = 2
                l = "Blogs"

            if "r=1" in lab:
                plt.plot(arrRes[lab]["nbObs"], arrRes[lab]["MAETot"], "-", label=f"Ouston - {l}", c=colors[ind])
                arrRes[lab]["MAETot"] = np.array(arrRes[lab]["MAETot"])
                #plt.fill_between(arrRes[lab]["nbObs"], arrRes[lab]["MAE"]+arrRes[lab]["MAESTD"], arrRes[lab]["MAE"]-arrRes[lab]["MAESTD"], alpha=0.3, color=colors[ind])
                print("MAETot", lab, arrRes[lab]["MAETot"][-1])
                print("F1", lab, arrRes[lab]["F1"][-1])
                print("AUC", lab, arrRes[lab]["AUC"][-1])
            if "r=0" in lab:
                plt.plot(arrRes[lab]["nbObs"], arrRes[lab]["MAETot"], "--", label=f"TopicCascade - {l}", c=colors[ind])
                arrRes[lab]["MAETot"] = np.array(arrRes[lab]["MAETot"])
                #plt.fill_between(arrRes[lab]["nbObs"], arrRes[lab]["MAE"]+arrRes[lab]["MAESTD"], arrRes[lab]["MAE"]-arrRes[lab]["MAESTD"], alpha=0.3, color=colors[ind])
                print("MAETot", lab, arrRes[lab]["MAETot"][-1])
                print("F1", lab, arrRes[lab]["F1"][-1])
                print("AUC", lab, arrRes[lab]["AUC"][-1])
        plt.xlabel("# observations")
        plt.ylabel("MAE")
        #plt.ylim([0,1])
        plt.legend()
        plt.tight_layout()

        plt.tight_layout()
        plt.savefig("figures/Results.pdf")
        plt.show()


import sys
try:
    choice = sys.argv[1]
except:
    choice = "1"

if choice=="1":
    #computeAllResults()
    plotMetrics()

if choice=="2":
    plotNetworksSynth()

if choice=="3":
    with open("figures/results_memetracker.txt", "w+") as f:
        f.write(f"")
    plotNetworksReal()


