import numpy as np
import matplotlib.pyplot as plt
import sparse
from scipy import stats
import networkx as nx

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'


def gaussian(x, mu, sig):
    return (np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))/(2 * np.pi * np.power(sig, 2.)) ** 0.5

def gaussian_kernel(dt, means, sigs, alpha):
    k = gaussian(dt[:, None], means[None, :], sigs[None, :]).dot(alpha)
    return k

def exp_kernel(dt, alpha):
    fac = alpha[0]  # 0 car K=1 pour exp kernel
    k = np.exp(-fac*dt)
    return k

def constant_kernel(dt, alpha):
    k = np.zeros((len(dt)))+alpha
    return k

def genPL(n, alpha, kmin=3, kmax=None):
    forceSimpleGraph = True
    if kmax is None:
        if n**0.5>kmin+1:
            kmax=n**0.5  # Guarantees an uncorrelated network
        else:
            kmax=n

    seq=[]
    for u in range(n):
        k=kmax+1
        while k>kmax:
            r = np.random.random()
            k = int(kmin*(1.-r)**(-1./(alpha-1.)))

        seq.append(k)

    degSeq = seq
    dictGraph = {}
    degSeqMR=[]
    for i in range(len(degSeq)):  # Initializes the graph dictionary
        dictGraph[i] = []
        for j in range(degSeq[i]):  # Builds Malloy-Reed degree seq
            degSeqMR.append(i)

    if len(degSeqMR)%2!=0:  # To have an even number of stubs
        degSeqMR.append(np.random.randint(0, len(degSeq)-1))

    np.random.shuffle(degSeqMR)

    i=0
    while i <= len(degSeqMR)-2:
        if (forceSimpleGraph and not ((degSeqMR[i+1] in dictGraph[degSeqMR[i]]) or degSeqMR[i]==degSeqMR[i+1])) or not forceSimpleGraph:  # Rejects self loops and multiedges
            dictGraph[degSeqMR[i]].append(degSeqMR[i+1])
            dictGraph[degSeqMR[i+1]].append(degSeqMR[i])

        elif forceSimpleGraph and ((degSeqMR[i+1] in dictGraph[degSeqMR[i]]) or degSeqMR[i]==degSeqMR[i+1]) and i<len(degSeqMR)-3:  # If multiedge or self-loop : exchange two indices
            if i+2 == len(degSeqMR)-2:  # To avoid a loop if the last nodes block the configuration network generation
                del degSeqMR[i]
                del degSeqMR[i+1]
            else:
                r=np.random.randint(i+2, len(degSeqMR)-1)
                degSeqMR[i], degSeqMR[r] = degSeqMR[r], degSeqMR[i]
            i-=2

        elif degSeq[degSeqMR[i]]==2 and degSeqMR[i]==degSeqMR[i+1] and forceSimpleGraph and i<len(degSeqMR)-3:  # If m_node = 2 and it did a self loop : avoids having k_node=0
            r=np.random.randint(i+2, len(degSeqMR)-1)
            degSeqMR[i], degSeqMR[r] = degSeqMR[r], degSeqMR[i]
            i-=2



        i+=2

    row_ind, col_ind, data = [], [], []
    for u in dictGraph:
        for v in dictGraph[u]:
            row_ind.append(u)
            col_ind.append(v)
            data.append(1.)
    A = sparse.COO((data, (row_ind, col_ind))).tocsr()
    return A

def genER_nm(n, m):
    graphDict = {}

    for i in range(n):  # Initializes the dictionary
        graphDict[i] = []

    for i in range(m):
        u = round(np.random.random() * (n - 1))
        v = round(np.random.random() * (n - 1))

        if not (u == v or v in graphDict[u]):  # Avoid loops (while u==v) and multiple edges (if v, u are already neighbours)
            graphDict[u].append(v)
            graphDict[v].append(u)

    row_ind, col_ind, data = [], [], []
    for u in graphDict:
        for v in graphDict[u]:
            row_ind.append(u)
            col_ind.append(v)
            data.append(1.)
    A = sparse.COO((data, (row_ind, col_ind))).tocsr()
    return A

def readPolBlogs(filename):
    g = {}
    N = 0
    with open(filename, "r") as f:
        for line in f:
            u,v = line.replace("\n", "").split("\t")
            u,v = int(u)-1, int(v)-1
            if u not in g: g[u]={}
            if v not in g[u]: g[u][v]=0
            g[u][v] = 1.
            if u>N: N=u+1
            if v>N: N=v+1

    A = np.zeros((N,N))
    row_ind, col_ind, data = [], [], []
    for u in g:
        for v in g[u]:
            row_ind.append(u)
            col_ind.append(v)
            data.append(1.)
            A[u,v] = 1
    A = sparse.COO(A).tocsr()
    return A, N

def weightGraph(A, K, kernel):
    u, v = A.nonzero()[0:2]
    print(A.shape)
    c_u, c_v, c_k, d = [], [], [], []
    for i in range(len(A.data)):
        for k in range(K):
            c_k.append(k)
            c_u.append(u[i])
            c_v.append(v[i])
            rnd = np.random.random()
            d.append(rnd)
        #if kernel=="rbf":  # A PEUT ETRE ENLEVER
        #    d[-K:] = d[-K:]/np.sum(d[-K:])
    d = np.array(d)
    s = list(A.shape)
    s.append(K)
    return sparse.COO([c_u, c_v, c_k],d, shape=s)

def plot_hist(hist, A):
    plt.ion()
    clock_max = np.max(np.array(hist, dtype=object)[:, 1])
    for t in np.linspace(0, clock_max, 100):
        plt.clf()
        hist_actif = np.array(hist, dtype=object)
        hist_actif = hist_actif[hist_actif[:, 1]>t-10]
        hist_actif = hist_actif[hist_actif[:, 1]<t]
        for u in range(N):
            plt.plot(np.cos(2*np.pi*u/N), np.sin(2*np.pi*u/N), "ob")
            plt.text(np.cos(2*np.pi*u/N), np.sin(2*np.pi*u/N), f"{u}")
        for (v2, t2, c2) in hist_actif:
            shift = (c2-(C-1)/2)/(2*3)
            plt.plot(1.1*np.cos(2*np.pi*(v2+shift)/N), 1.1*np.sin(2*np.pi*(v2+shift)/N), "o", color=f"C{c2}", alpha=np.exp(-1.*(t-t2)))

        for c2 in range(len(A)):
            A_net = A[c2]
            shift = (c2-(C-1)/2)/(2*10)
            for u,v,k in zip(*A_net.nonzero()[:3]):
                plt.plot([np.cos(2*np.pi*(u+shift)/N),np.cos(2*np.pi*(v+shift)/N)], [np.sin(2*np.pi*(u+shift)/N),np.sin(2*np.pi*(v+shift)/N)], "-k", linewidth=1*A_net[u,v,k], color=f"C{c2}")
        plt.text(-1, 1, f"t={round(t, 2)}")
        plt.xlim([-1.2, 1.2])
        plt.ylim([-1.2, 1.2])
        plt.pause(0.1)

def get_infection_intertime(kernel, alpha, means=None, sigs=None):
    if kernel == "exp":
        a = alpha[0]
        t = np.random.exponential(1./a)
    if kernel == "rbf":
        t_ch = [np.random.normal(m, s) for m,s in zip(means, sigs)]
        pr = alpha.todense()
        pr /= np.sum(pr)
        t = np.random.choice(t_ch, p=pr)

    return t

def generate_cascades(A, T, nbCascPerInfo, kernel, means, sigs, nodesSubNet, nbSeeds, clock=None):
    allHists = []
    if clock is None:
        clock = [0. for _ in range(C)]  # To separate correctly cascades, the clock never resets
        clock_max = [T for c in range(C)]
    else:
        clock_max = [clock[c]+T for c in range(C)]
    for cascnumber in range(nbCascPerInfo):
        print(cascnumber+1, "/", nbCascPerInfo)
        ind_hist = 0
        hist = []
        node_status = [["Sus" for n in range(N)] for c in range(C)]
        listInfs = [[] for _ in range(C)]
        for c in range(C):
            clock_max[c] += T
        noInfNodes=False
        for _ in range(nbSeeds):
            for c in range(C):
                n_rnd = np.random.choice(nodesSubNet[c])
                node_status[c][n_rnd] = "Inf"
                listInfs[c].append(n_rnd)
                hist.append((n_rnd, clock[c]+10, c, cascnumber))
        hist = np.array(hist)

        while True:
            hist = hist[hist[:, 1].argsort()]
            try:
                u, clock_tmp, c, casc = hist[ind_hist]
                u = int(u)
                c = int(c)
                clock[int(c)] = clock_tmp
            except Exception as e:
                print("No more infected nodes")
                break

            if clock[c] > clock_max[c]:
                print("Max T reached")
                break

            neighs = set(list(A[c][u].nonzero()[0]))
            for v in neighs:
                v_inf = hist[hist[:, 2]==c]
                v_inf = v_inf[v_inf[:, 3]==cascnumber]
                v_inf = v_inf[v_inf[:, 0]==v]
                inf_time = clock[c] + get_infection_intertime(kernel, A[c][u,v], means, sigs)
                if len(v_inf)!=0:  # Only the 1st infection time counts
                    if v_inf[0][1]>inf_time:
                        index = np.where((hist == v_inf[0]).all(axis=1))  # Index of hist row matching v_inf, even if axis=1
                        hist = np.delete(hist, index, axis=0)  # Delete row indexed
                        hist = np.vstack((hist, (v, inf_time, c, cascnumber)))
                        hist = np.array(list(sorted(hist, key=lambda x: x[1])))
                else:
                    hist = np.vstack((hist, (v, inf_time, c, cascnumber)))

            ind_hist += 1

        hist = [(int(v),float(t),int(c),int(casc)) for v,t,c,casc in hist if t<clock_max[int(c)]]
        allHists.append(hist)

    hist = []
    for h in allHists:
        hist += h
    return hist, clock

def add_text_to_hist(hist, words_per_obs, voc_per_class, overlap):
    nbClasses = len(set(list(np.array(hist, dtype=object)[:, 0])))
    voc_clusters = [np.array(list(range(int(voc_per_class)))) + c*voc_per_class for c in range(nbClasses)]

    # Overlap
    for c in range(nbClasses):
        voc_clusters[c] -= int(c*voc_per_class*overlap)

    # Associate a fraction of vocabulary to each observation
    observations = []
    for i, (u,t,c,casc) in enumerate(hist):
        text = np.random.choice(voc_clusters[c], size=words_per_obs)
        observations.append((i, t, text, u, casc, c))
    observations = np.array(observations, dtype=object)
    return observations

def save(folder, name, observations, A, means, sigmas, lamb0, kernel, overlap, words_per_obs, voc_per_class, changeNet=None):
    observations = np.array(list(sorted(observations, key= lambda x: x[1])))
    txtChangeNet = ""
    if changeNet is not None:
        txtChangeNet += "_changenet="+str(changeNet)
    name += f"_OL={overlap}_wdsPerObs={words_per_obs}_vocPerClass={voc_per_class}"+txtChangeNet
    with open(folder+name+"_events.txt", "w+") as f:
        for i, (index, time, text, u, casc_number, clus) in enumerate(observations):
            content = ",".join(map(str, list(text)))
            txt = str(i)+"\t"+str(time)+"\t"+content+"\t"+str(u)+"\t"+str(casc_number)+"\t"+str(clus)+"\n"
            f.write(txt)

    with open(folder+name+"_kernel.txt", "w+") as f:
        f.write(str(kernel)+"\n\n")
        f.write(str(lamb0)+"\n\n")
        if means is None or sigmas is None:
            f.write("-1\n\n-1\n")
        else:
            for m in means:
                f.write(str(m)+"\n")
            f.write("\n")
            for s in sigs:
                f.write(str(s)+"\n")
    for c in range(len(A)):
        sparse.save_npz(folder+name+"_alpha_c"+str(c), A[c])

seed = 1121
np.random.seed(seed)

folder = "data/Synth/"

T = 100
nbCascPerInfo = 1500
kernel = "exp"

words_per_obs = 5
voc_per_class = 100
overlap = 0.
nbSeeds = 2

for net in ["ER", "PolBlogs", "PL"]:
    print(net)
    overlap = 0.
    name = "Synth_"+net
    N = 500
    C = 5
    K_net = 1
    A = []
    nodesSubNet = {}
    for c in range(C):
        if net == "PL":
            A_net = genPL(N, 3.25, kmin=1)  # kmin=1, alpha=3.25 et N=1000 donne des comps de 250 noeuds
        elif net=="ER":
            A_net = genER_nm(N, int(1.2*N))
            ch = np.random.choice(list(range(N)), size=300, replace=False)
            A_net = A_net.todense()
            for u in range(N):
                if u not in ch:
                    try:
                        A_net[:, u] *= 0
                        A_net[u, :] *= 0
                    except:
                        pass
            A_net = sparse.COO(A_net)
        elif net=="PolBlogs":
            A_net, N = readPolBlogs("data/Synth/web-polblogs.mtx")
            ch = np.random.choice(list(range(N)), size=300, replace=False)
            A_net = A_net.todense()
            for u in range(N):
                if u not in ch:
                    A_net[:, u] *= 0
                    A_net[u, :] *= 0
            A_net = sparse.COO(A_net)

        A_net = weightGraph(A_net, K_net, kernel)
        nodesSubNet[c] = list(range(N))

        A_net = A_net.todense()
        g = nx.convert_matrix.from_numpy_matrix(A_net[:, :, 0])
        components = nx.connected_components(g)

        largest_component = max(components, key=len)
        nodesSubNet[c] = list(largest_component)
        print("# nodes subnet:", len(largest_component))
        #print(nodesSubNet[c])
        for i in range(N):
            if i not in largest_component:
                try:
                    A_net[:, i] *= 0
                    A_net[i, :] *= 0
                except Exception as e:
                    print(e)
                    pass
        #print(A_net.reshape((N,N)))
        print(len(A_net), len(A_net[A_net!=0].flatten()))
        A_net = sparse.COO(A_net)
        A.append(A_net)

    if kernel == "rbf":
        means = np.array([3, 7, 11])
        sigs = np.array([0.5, 0.5, 0.5])
        means, sigs = means[:K_net], sigs[:K_net]
    else:
        means, sigs = None, None
    lamb0 = 0.01

    hist, clock = generate_cascades(A, T, nbCascPerInfo, kernel, means, sigs, nodesSubNet, nbSeeds)
    observations = add_text_to_hist(hist, words_per_obs, voc_per_class, overlap)
    save(folder, name, observations, A, means, sigs, lamb0, kernel, overlap, words_per_obs, voc_per_class)
    pause()

    if net=="PL":
        # Vary network in time
        N = 20
        C = 5
        K_net = 1
        A = []
        T = 20
        nbCascPerInfo = 10000
        nodesSubNet = {c: list(range(N)) for c in range(C)}
        nbSeeds = 1
        for c in range(C):
            A_net = genPL(N, 3.25, kmin=2)
            A_net = weightGraph(A_net, K_net, kernel)
            A_net = sparse.COO(A_net)
            A.append(A_net)
        clock = [0. for _ in range(C)]
        changeNet = 1
        observations = []
        for casc in range(nbCascPerInfo):
            print("==", casc, changeNet*nbCascPerInfo//5)
            if casc > changeNet*nbCascPerInfo//5:
                for i in range(len(A)):
                    A[i] = A[i].todense()
                    A[i] *= (np.random.random(A[i].shape)-0.5)/2 + 1
                    A[i][A[i]>1] = 1
                    A[i] = sparse.COO(A[i])

                nametmp = name+ f"_OL={overlap}_wdsPerObs={words_per_obs}_vocPerClass={voc_per_class}_change={changeNet}_t={np.round(clock, 2)}"
                for c in range(len(A)):
                    sparse.save_npz(folder+nametmp+"_alpha_c"+str(c), A[c])
                changeNet += 1
            hist, clock = generate_cascades(A, T, 1, kernel, means, sigs, nodesSubNet, nbSeeds, clock)
            hist = np.array(hist, dtype=object)
            hist = list(hist)
            observations += hist

        observations = add_text_to_hist(observations, words_per_obs, voc_per_class, overlap)
        save(folder, name, observations, A, means, sigs, lamb0, kernel, overlap, words_per_obs, voc_per_class, changeNet="All")
        pause()

        hist, clock = generate_cascades(A, T, nbCascPerInfo, kernel, means, sigs, nodesSubNet, nbSeeds)
        for overlap in [0.1, 0.25, 0.5, 0.75]:  # Generate overlaps
            observations = add_text_to_hist(hist, words_per_obs, voc_per_class, overlap)
            save(folder, name, observations, A, means, sigs, lamb0, kernel, overlap, words_per_obs, voc_per_class)

    #plot_hist(hist, A)




