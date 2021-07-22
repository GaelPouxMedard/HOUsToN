import numpy as np
import datetime

import string
punctuations = list(string.punctuation)
import spacy
#spacy_nlp = spacy.load('en_core_web_trf')
spacy_nlp = spacy.load('en_core_web_sm')
toreject = ["DET", "AUX", "PUNCT", "ADP", "CCONJ", "PRON", "SCONJ", "PART"]

def preprocess_spacy(txt):
    txt = spacy_nlp(txt)
    txt = " ".join([w.lemma_.lower() for w in txt if w.pos_ not in toreject]).replace("\n", " ")
    txt.replace("  ", " ")
    return txt

def timeStamp(d):
    date = datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S")
    timestamp = datetime.datetime.timestamp(date)/(3600/2)  # 30 mins
    return timestamp



lgDoc = 8357589
lim = lgDoc//5

# i, t, text, u, casc, c
observations = {}
wdsToInt = {}
intToWds = {}
nodesToInt = {}
intToNodes = {}
cascToInt = {}
clusToInt = {}
occurence_node = {}
occurence_wd = {}
nodesCasc = {}
num_wds, num_nodes, num_casc, num_clus = 0, 0, 0, 0
cur_casc, cur_clus = 0, 0
index = 0

with open("clust-qt08080902w3mfq5.txt", encoding="utf-8") as f:
    [f.readline() for _ in range(6)]
    for i, line in enumerate(f):
        if i%(8357589//100)==0: print(i*100/np.min([lim, lgDoc]), "%")
        if line == '\n':
            continue
        tab = line.replace("\n", "").split("\t")
        if tab[0]!='':  # 1st level
            m = tab[3]
            phraseClus = tab[2]
            if phraseClus not in clusToInt:
                clusToInt[phraseClus] = num_clus
                num_clus += 1
            cur_clus = clusToInt[phraseClus]

        else:
            if tab[1] != '':  # 2nd level
                occurence_node_casc = {}
                casc = tab[3]
                lengthCasc = tab[2]
                #print("===", lengthCasc)
                if casc not in cascToInt:
                    cascToInt[casc] = num_casc
                    nodesCasc[cascToInt[casc]] = []
                    num_casc += 1
                cur_casc = cascToInt[casc]
                cur_txt = []
                txtcasc = preprocess_spacy(casc)
                for wd in txtcasc.split(" "):
                    if wd not in wdsToInt:
                        wdsToInt[wd] = num_wds
                        intToWds[num_wds] = wd
                        occurence_wd[num_wds] = 0
                        num_wds += 1
                    occurence_wd[wdsToInt[wd]] += 1
                    cur_txt.append(wdsToInt[wd])
            else:  # 3rd level
                ts = timeStamp(tab[2])
                type = tab[4]
                node = tab[5].replace("http://", "").replace("https://", "")
                node = node[:node.find("/")]
                if node not in nodesToInt:
                    nodesToInt[node] = num_nodes
                    intToNodes[num_nodes] = node
                    num_nodes += 1
                cur_node = nodesToInt[node]
                if node not in occurence_node_casc: occurence_node_casc[node] = 1
                else: continue  # Consider only 1st appearance of meme

                if cur_node not in occurence_node: occurence_node[cur_node] = 0
                occurence_node[cur_node] += 1

                if cur_node not in observations: observations[cur_node] = []
                nodesCasc[cur_casc].append(cur_node)
                observations[cur_node].append((ts, cur_node, cur_txt, cur_casc, cur_clus))
                index += 1

        if i%(8357589//100)==0:
            if len(occurence_wd) != 0:
                occ = np.array(list(sorted(occurence_node.values())))
                occwds = np.array(list(sorted(occurence_wd.values())))
                print(occ)
                print("====", len(occ), np.min(occ), np.max(occ), np.mean(occ), np.median(occ))
                print("====", num_wds, np.min(occwds), np.max(occwds), np.mean(occwds), np.median(occwds))

        if i>lim:
            break

occ = np.array(list(sorted(occurence_node.values())))
occwds = np.array(list(sorted(occurence_wd.values())))
try:
    thres = np.array(list(sorted(list(occ))))[-500]
except:
    thres = 0
print(len(nodesCasc))
selected_nodes = []
for node in occurence_node:
    if occurence_node[node]<=thres:
        del observations[node]
    else:
        selected_nodes.append(node)

k = list(nodesCasc.keys())
for c in k:
    tmp = [r for r in nodesCasc[c] if r in selected_nodes]
    if len(tmp)<=1:
        del nodesCasc[c]

print(len(nodesCasc))

vals = []
for node in observations:
    for v in observations[node]:
        if v[3] in nodesCasc:
            vals.append(v)

vals = list(sorted(vals, key=lambda x: x[0]))
print("Times", np.array(vals, dtype=object)[:, 0])

newWdToInt, newNodeToInt = {}, {}
numNode, numWds = 0, 0
with open("Memetracker_30min_events.txt", "w+", encoding="utf-8") as f:
    for i, v in enumerate(vals):
        ts, node, txt, casc_number, clus = v
        if intToNodes[node] not in newNodeToInt:
            newNodeToInt[intToNodes[node]] = numNode
            numNode += 1
        node = newNodeToInt[intToNodes[node]]
        newtxt = []
        for wd in txt:
            if intToWds[wd] not in newWdToInt:
                newWdToInt[intToWds[wd]] = numWds
                numWds += 1
            wd = newWdToInt[intToWds[wd]]
            newtxt.append(str(wd))
        content = ",".join(newtxt)
        txt = str(i)+"\t"+str(ts)+"\t"+content+"\t"+str(node)+"\t"+str(casc_number)+"\t"+str(clus)+"\n"
        f.write(txt)


with open("intToNodes_30min.txt", "w+") as f:
    for node in newNodeToInt:
        f.write(str(newNodeToInt[node])+"\t"+str(node)+"\n")
with open("intToWds_30min.txt", "w+") as f:
    for wd in newWdToInt:
        f.write(str(newWdToInt[wd])+"\t"+str(wd)+"\n")

vals = np.array(vals, dtype=object)
print("Wds", numWds)
