import sys
import os
import numpy as np
import time
from copy import deepcopy as copy
from utils import *
import sparse
from sklearn.metrics import normalized_mutual_info_score
import pickle

'''
from memory_profiler import profile
import gc
fp = open("memory_profiler_Norm.log", "a")
@profile(stream=fp, precision=5)

import inspect
for name, obj in inspect.getmembers(cluster.expr_l[nodei]):
	s = asizeof(obj) / (1024*1024)
	if s>0.:
		print("=================", name, s, "=================")
'''

class Dirichlet_Hawkes_Process(object):
	"""docstring for Dirichlet Hawkes Prcess"""
	def __init__(self, particle_num, base_intensity, theta0, vocabulary_size, r, number_nodes, number_cascades, horizon):
		super(Dirichlet_Hawkes_Process, self).__init__()
		self.r = r
		self.K=1
		self.particle_num = particle_num
		self.base_intensity = base_intensity
		self.theta0 = theta0
		self.vocabulary_size = vocabulary_size
		self.number_nodes = number_nodes
		self.number_cascades=number_cascades
		self.horizon = horizon
		self.particles = []
		for i in range(particle_num):
			self.particles.append(Particle(weight = 1.0 / self.particle_num))

		self.active_interval = None

	def sequential_monte_carlo(self, doc, threshold):
		# Set relevant time interval
		tu = doc.timestamp-self.horizon
		T = doc.timestamp
		self.active_interval = [tu, T]

		particles = []
		for particle in self.particles:
			particles.append(self.particle_sampler(particle, doc))

		self.particles = particles

		# Resample particules whose weight is below the given threshold
		self.particles = self.particles_normal_resampling(self.particles, threshold)

	def particle_sampler(self, particle, doc):
		# Sample cluster label
		particle, selected_cluster_index = self.sampling_cluster_label(particle, doc)
		if not (self.r>-1e-5 and self.r<1e-5):
			# Update the triggering kernel
			particle.clusters[selected_cluster_index].alpha[doc.node] = self.parameter_estimation(particle, selected_cluster_index, doc.node)
		# Calculate the weight update probability
		particle.log_update_prob = self.calculate_particle_log_update_prob(particle, selected_cluster_index, doc)
		return particle

	def sampling_cluster_label(self, particle, doc):
		if len(particle.clusters) == 0: # The first document is observed
			particle.cluster_num_by_now += 1
			selected_cluster_index = particle.cluster_num_by_now
			selected_cluster = Cluster(index = selected_cluster_index, num_nodes=self.number_nodes, K=self.K)
			selected_cluster.add_document(doc)
			particle.clusters[selected_cluster_index] = selected_cluster #.append(selected_cluster)
			particle.docs2cluster_ID.append(selected_cluster_index)
			particle.docs2cluster_index.append(doc.index)
			particle.active_clusters[selected_cluster_index] = [doc]

		else: # A new document arrives
			active_cluster_indexes = [0] # Zero for new cluster
			active_cluster_rates = [self.base_intensity**self.r]
			cls0_log_dirichlet_multinomial_distribution = log_dirichlet_multinomial_distribution(doc.word_distribution, doc.word_distribution, doc.word_count, doc.word_count, self.vocabulary_size, self.theta0)
			active_cluster_textual_probs = [cls0_log_dirichlet_multinomial_distribution]
			# Update list of relevant timestamps
			particle.active_clusters = self.update_active_clusters(particle, doc)

			# Posterior probability for each cluster
			for active_cluster_index in particle.active_clusters:
				activeTup = np.array(particle.active_clusters[active_cluster_index], dtype=object)
				active_cluster_indexes.append(active_cluster_index)


				if not (self.r>-1e-5 and self.r<1e-5):
					rate = self.base_intensity**self.r  # Background intensity process

					if doc.node in particle.clusters[active_cluster_index].alpha:
						alphaNode = particle.clusters[active_cluster_index].alpha[doc.node].todense()
						for d in activeTup:
							if d.casc == doc.casc:
								t = d.timestamp
								u = int(d.node)
								alpha = alphaNode[u]
								rate += H(doc.timestamp, t, alpha)
				else:
					rate = 0

				# Dirichlet-Survival prior
				active_cluster_rates.append(rate)

				# Language model likelihood
				cls_word_distribution = particle.clusters[active_cluster_index].word_distribution + doc.word_distribution
				cls_word_count = particle.clusters[active_cluster_index].word_count + doc.word_count
				cls_log_dirichlet_multinomial_distribution = log_dirichlet_multinomial_distribution(cls_word_distribution, doc.word_distribution, cls_word_count, doc.word_count, self.vocabulary_size, self.theta0)
				active_cluster_textual_probs.append(cls_log_dirichlet_multinomial_distribution)

			#print(1, list(active_cluster_rates))
			#print(2, list(active_cluster_textual_probs))
			# Posteriors to probabilities
			active_cluster_rates = np.array(active_cluster_rates)
			active_cluster_rates[active_cluster_rates<1e-20] = 1e-20
			active_cluster_logrates = self.r*np.log(np.array(active_cluster_rates)+1e-100)
			cluster_selection_probs = active_cluster_logrates + active_cluster_textual_probs # in log scale
			cluster_selection_probs = cluster_selection_probs - np.max(cluster_selection_probs) # prevent overflow
			cluster_selection_probs = np.exp(cluster_selection_probs)
			cluster_selection_probs = cluster_selection_probs / np.sum(cluster_selection_probs)

			# Random cluster selection
			selected_cluster_array = multinomial(exp_num = 1, probabilities = cluster_selection_probs)
			selected_cluster_index = np.array(active_cluster_indexes)[np.nonzero(selected_cluster_array)][0]

			# New cluster drawn
			if selected_cluster_index == 0:
				particle.cluster_num_by_now += 1
				selected_cluster_index = particle.cluster_num_by_now
				selected_cluster = Cluster(index = selected_cluster_index, num_nodes=self.number_nodes, K = self.K)
				selected_cluster.add_document(doc)
				particle.clusters[selected_cluster_index] = selected_cluster
				particle.docs2cluster_ID.append(selected_cluster_index)
				particle.docs2cluster_index.append(doc.index)
				particle.active_clusters[selected_cluster_index] = [doc]

			# Existing cluster drawn
			else:
				selected_cluster = particle.clusters[selected_cluster_index]
				selected_cluster.add_document(doc)
				particle.docs2cluster_ID.append(selected_cluster_index)
				particle.docs2cluster_index.append(doc.index)
				particle.active_clusters[selected_cluster_index].append(doc)

		return particle, selected_cluster_index

	def parameter_estimation(self, particle, selected_cluster_index, node, forcefit = False):
		active_tup = np.array(particle.active_clusters[selected_cluster_index], dtype=object)

		# Observation is alone in the cluster => the cluster is new => random initialization of alpha
		# Note that it cannot be a previously filled cluster since it would have 0 chance to get selected (see sampling_cluster_label)
		if node not in particle.clusters[selected_cluster_index].num_docs_node:
			alpha = sparse.zeros(shape=(particle.clusters[selected_cluster_index].num_nodes, self.K))
			return alpha

		particle.clusters[selected_cluster_index] = update_cluster_likelihoods(active_tup, particle.clusters[selected_cluster_index])
		alpha = update_triggering_kernel_optim(particle.clusters[selected_cluster_index], node, self.r, forcefit)

		return alpha

	def update_active_clusters(self, particle, doc):
		tu = self.active_interval[0]
		keys = list(particle.active_clusters.keys())
		for cluster_index in keys:
			activTup = particle.active_clusters[cluster_index]
			active_timeseq = [d for d in activTup if d.timestamp > self.active_interval[1]-self.horizon]

			particle.active_clusters[cluster_index] = active_timeseq

			node = doc.node
			keysi = list(particle.clusters[cluster_index].coeffH.keys())
			if node in keysi:
				obsHi = np.array(particle.clusters[cluster_index].coeffH[node], dtype=object)
				if len(obsHi)!=0:
					obsHi = obsHi[obsHi[:, 0]>tu]
					particle.clusters[cluster_index].coeffH[node] = list(obsHi)
				else:
					particle.clusters[cluster_index].coeffH[node] = []
				keysj = list(particle.clusters[cluster_index].coeffS[node].keys())
				for nodej in keysj:
					obsij = np.array(particle.clusters[cluster_index].coeffS[node][nodej])
					if len(obsij) != 0:
						obsij = obsij[obsij[:, 0]>tu]
						if len(obsij) != 0:
							particle.clusters[cluster_index].coeffS[node][nodej] = list(obsij)
						else:
							del particle.clusters[cluster_index].coeffS[node][nodej]
					else:
						del particle.clusters[cluster_index].coeffS[node][nodej]
		return particle.active_clusters
	
	def calculate_particle_log_update_prob(self, particle, selected_cluster_index, doc):
		cls_word_distribution = particle.clusters[selected_cluster_index].word_distribution
		cls_word_count = particle.clusters[selected_cluster_index].word_count
		doc_word_distribution = doc.word_distribution
		doc_word_count = doc.word_count

		log_update_prob = log_dirichlet_multinomial_distribution(cls_word_distribution, doc_word_distribution, cls_word_count, doc_word_count, self.vocabulary_size, self.theta0)

		return log_update_prob

	def particles_normal_resampling(self, particles, threshold):
		weights = []; log_update_probs = []
		for particle in particles:
			weights.append(particle.weight)
			log_update_probs.append(particle.log_update_prob)
		weights = np.array(weights)
		log_update_probs = np.array(log_update_probs)
		log_update_probs = log_update_probs - np.max(log_update_probs) # Prevents overflow
		update_probs = np.exp(log_update_probs)
		weights = weights * update_probs
		weights = weights / np.sum(weights) # normalization
		resample_num = len(np.where(weights + 1e-5 < threshold)[0])

		if resample_num == 0: # No need to resample particle, but still need to assign the updated weights to particles
			for i, particle in enumerate(particles):
				particle.weight = weights[i]
			return particles
		else:
			#print("RESAMPLING")
			remaining_particles = [particle for i, particle in enumerate(particles) if weights[i] + 1e-5 > threshold ]
			resample_probs = weights[np.where(weights + 1e-5 > threshold)]
			resample_probs = resample_probs/np.sum(resample_probs)
			remaining_particle_weights = weights[np.where(weights + 1e-5 > threshold)]
			for i,_ in enumerate(remaining_particles):
				remaining_particles[i].weight = remaining_particle_weights[i]

			resample_distribution = multinomial(exp_num = resample_num, probabilities = resample_probs)
			if not resample_distribution.shape: # The case of only one particle left
				for _ in range(resample_num):
					new_particle = copy(remaining_particles[0])
					remaining_particles.append(new_particle)
			else: # The case of more than one particle left
				for i, resample_times in enumerate(resample_distribution):
					for _ in range(resample_times):
						new_particle = copy(remaining_particles[i])
						remaining_particles.append(new_particle)

			# Normalize the particle weights
			update_weights = np.array([particle.weight for particle in remaining_particles]); update_weights = update_weights / np.sum(update_weights)
			for i, particle in enumerate(remaining_particles):
				particle.weight = update_weights[i]

			self.particles = None
			return remaining_particles

def getArgs(args):
	import re
	dataFile, outputFolder, r, nbRuns, theta0, particle_num, printRes = [None]*7
	for a in args:
		print(a)
		try: dataFile = re.findall("(?<=data_file=)(.*)(?=)", a)[0]
		except: pass
		try: outputFolder = re.findall("(?<=destination=)(.*)(?=)", a)[0]
		except: pass
		try: r = re.findall("(?<=r=)(.*)(?=)", a)[0]
		except: pass
		try: theta0 = float(re.findall("(?<=theta0=)(.*)(?=)", a)[0])
		except: pass
		try: particle_num = int(re.findall("(?<=number_particles=)(.*)(?=)", a)[0])
		except: pass
		try: printRes = bool(re.findall("(?<=print_progress=)(.*)(?=)", a)[0])
		except: pass

	if "-help" in " ".join(args):
		txt = "Mandatory parameters: data_file, output_folder\n" \
			  "data_file: relative path to the file containing events. The file must be formatted so taht each line follows this syntax: " \
			  "observation_index(int)[TAB]timestamp(float)[TAB]textual content(comma separated strings)[TAB]node_index(int)[TAB]cascade_number(int)[TAB]true_cluster(optional, int)[end_line]\n" \
			  "output_folder: where to save output files\n\n" \
			  "Optional parameters: r, theta0, particle_num, printRes\n" \
			  "r (default 1): what version of the Powered Dirichlet Process to use for its survival version. r=1 is the regular Dirichlet process, and r=0 reduces Ouston to TopicCascade.\n" \
			  "theta0 (default 0.01): hyperparameter for the language model's Dirichlet prior\n" \
			  "particle_num (default 4): how many particles the SMC algorithm will use\n" \
			  "printRes (default True): whether to print progress every 100 treated observations. If the true cluster is provided, also displays NMI. If the true network is " \
			  "present in the input file directory, also attempts to compute the MAE on networks edges.\n"
		sys.exit(txt)
	if dataFile is None:
		sys.exit("Enter a valid value for data_file. Enter -help for details.")
	if outputFolder is None:
		sys.exit("Enter a valid value for output_folder. Enter -help for details.")
	if r is None: print("r value not found; defaulted to 1"); r="1"
	if theta0 is None: print("theta0 value not found; defaulted to 0.01"); theta0=0.01
	if particle_num is None: print("particle_num value not found; defaulted to 4"); particle_num=4
	if printRes is None: print("printRes value not found; defaulted to True"); printRes=True

	curdir = os.curdir+"/"
	for folder in outputFolder.split("/"):
		if folder not in os.listdir(curdir) and folder != "":
			os.mkdir(curdir+folder+"/")
		curdir += folder+"/"
	rarr = []
	for rstr in r.split(","):
		rarr.append(float(rstr))
	return dataFile, outputFolder, rarr, theta0, particle_num, printRes

def parse_newsitem_2_doc(news_item, vocabulary_size):
	index = news_item[0]
	timestamp = news_item[1]
	word_id = news_item [2][0]
	count = news_item[2][1]
	word_distribution = np.zeros(vocabulary_size)
	word_distribution[word_id] = count
	word_count = np.sum(count)
	node = news_item[3]
	cascNum = news_item[4]
	try:
		trueClus = news_item[5]
	except:
		trueClus = -1
	doc = Document(index, timestamp, word_distribution, word_count, node, cascNum, trueClus)
	return doc

def readObservations(dataFile, outputFolder, nameOut):
	observations = []
	wdToIndex, index = {}, 0
	nodeToInt, cascToInt = {}, {}
	index_node, index_casc = 0, 0
	with open(dataFile, "r", encoding="utf-8") as f:
		for i, line in enumerate(f):
			l = line.replace("\n", "").split("\t")
			i_doc = int(l[0])
			timestamp = float(l[1])
			words = l[2].split(",")
			node = l[3]
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
	with open(outputFolder+nameOut+"_indexWords.txt", "w+", encoding="utf-8") as f:
		for wd in wdToIndex:
			f.write(f"{wdToIndex[wd]}\t{wd}\n")
	with open(outputFolder+nameOut+"_indexNodes.txt", "w+", encoding="utf-8") as f:
		for n in nodeToInt:
			f.write(f"{nodeToInt[n]}\t{n}\n")
	with open(outputFolder+nameOut+"_indexCascades.txt", "w+", encoding="utf-8") as f:
		for c in cascToInt:
			f.write(f"{cascToInt[c]}\t{c}\n")

	V = len(wdToIndex)
	N = len(nodeToInt)
	C = len(cascToInt)
	return observations, V, N, C

def writeParticles(DHP, outputFolder, nameOut, news_item=None):
	def getLikTxt(cluster, theta0=None):
		cls_word_distribution = np.array(cluster.word_distribution, dtype=int)
		cls_word_count = int(cluster.word_count)

		vocabulary_size = len(cls_word_distribution)
		if theta0 is None:
			theta0 = 0.01

		priors_sum = theta0*vocabulary_size  # ATTENTION SEULEMENT SI THETA0 EST SYMMETRIQUE !!!!
		log_prob = 0

		cnt = np.bincount(cls_word_distribution)
		un = np.arange(len(cnt))

		log_prob += gammaln(priors_sum)
		log_prob += gammaln(cls_word_count+1)
		log_prob += gammaln(un + theta0).dot(cnt)  # ATTENTION SEULEMENT SI THETA0 EST SYMMETRIQUE !!!!

		log_prob -= gammaln(cls_word_count + priors_sum)
		log_prob -= vocabulary_size*gammaln(theta0)
		log_prob -= gammaln(cls_word_count+1)

		return log_prob
	if news_item is not None:
		timesave = news_item[1]
	else:
		timesave = None
	if timesave is None:
		nameOut += "_done"
	else:
		nameOut += f"_t={np.round(timesave, 1)}h"
	with open(outputFolder+nameOut+"_particles.txt", "w+") as f:
		for pIter, p in enumerate(DHP.particles):
			f.write(f"Particle\t{pIter}\t{p.weight}\t{p.docs2cluster_ID}\t{p.docs2cluster_index}\n")
			for c in p.clusters:
				likTxt = getLikTxt(p.clusters[c], theta0 = DHP.theta0[0])
				pickle.dump(p.clusters[c].alpha, open(outputFolder+nameOut+f"_alphas_{pIter}_{c}.pkl","wb"))
				f.write(f"Cluster\t{c}\t{likTxt}\t{p.clusters[c].word_count}\t[")
				V = len(p.clusters[c].word_distribution)
				for iwdd, wdd in enumerate(p.clusters[c].word_distribution):
					f.write(str(wdd))
					if iwdd != V:
						f.write(" ")
					else:
						f.write("]")
				f.write("\n")

def saveParts(DHP, outputFolder, nameOut, news_item=None):
	while True:
		try:
			writeParticles(DHP, outputFolder, nameOut, news_item)
			break
		except Exception as e:
			print(i, e)
			time.sleep(10)
			continue

def run_fit(observations, dataFile, outputFolder, nameOut, lamb0, r=1., theta0=None, particle_num=4, printRes=False, vocabulary_size=None, number_nodes=None, number_cascades=None, horizon=1e20):
	"""
	observations = ([array int] index_obs, [array float] timestamp, ([array int] unique_words, [array int] count_words), [opt, int] temporal_cluster, [opt, int] textual_cluster)
	outputFolder = Output folder for the results
	nameOut = Name of the file to which _particles_compressed.pklbz2 will be added
	lamb0 = base intensity
	r = exponent parameter of the Powered Dirichlet process; defaults to 1. (standard Dirichlet process)
	theta0 = value of the language model symmetric Dirichlet prior
	particle_num = number of particles used in the Sequential Monte-Carlo algorithm
	printRes = whether to print the results according to ground-truth (optional parameters of observations and alpha)
	"""

	if vocabulary_size is None:
		allWds = set()
		for a in observations:
			for w in a[2][0]:
				allWds.add(w)
		vocabulary_size = len(list(allWds))+2
	if theta0 is None: theta0 = 1.

	particle_num = particle_num
	base_intensity = lamb0
	theta0 = np.array([theta0 for _ in range(vocabulary_size)])
	threshold = 1.0 / (particle_num*2.)

	DHP = Dirichlet_Hawkes_Process(particle_num = particle_num, base_intensity = base_intensity, theta0 = theta0,
								   vocabulary_size = vocabulary_size,
								   r=r, number_nodes=number_nodes,
								   number_cascades=number_cascades, horizon=horizon)

	t = time.time()
	lastsavetime = 0
	lgObs = len(observations)
	for i, news_item in enumerate(observations):
		doc = parse_newsitem_2_doc(news_item = news_item, vocabulary_size = vocabulary_size)
		DHP.sequential_monte_carlo(doc, threshold)
		if i%100==1 and printRes:
			print(horizon, observations[-1][1]-observations[0][1])
			print(f'r={r} - Handling document {i}/{lgObs} (t={np.round(news_item[1]-observations[0][1], 1)}) - Average time : {np.round((time.time()-t)*1000/(i), 0)}ms - '
				  f'Remaining time : {np.round((time.time()-t)*(len(observations)-i)/(i*3600), 2)}h - Elapsed time : {np.round((time.time()-t)/3600, 2)}h - '
				  f'ClusTot={DHP.particles[0].cluster_num_by_now}')
			evaluate(observations, DHP.particles[0], outputFolder, nameOut, dataFile, DHP.r)

		# Takes snapshots
		if np.round(news_item[1]-observations[0][1], 1)>lastsavetime+horizon or i%1000==1:
			print("SAVING")
			saveParts(DHP, outputFolder, nameOut, news_item)
			lastsavetime = news_item[1]

	# Updates every node parameters one last time and save
	for p in range(len(DHP.particles)):
		for c in DHP.particles[p].clusters:
			for node in DHP.particles[p].clusters[c].alpha:
				DHP.particles[p].clusters[c].alpha[node] = DHP.parameter_estimation(DHP.particles[p], c, node, forcefit=True)
	saveParts(DHP, outputFolder, nameOut)

	evaluate(observations, DHP.particles[0], outputFolder, nameOut, dataFile, DHP.r)

def evaluate(observations, part, outputfolder, nameOut, nameIn, r):
	intToUsr = {}
	with open(outputfolder+nameOut+"_indexNodes.txt", "r") as f:
		for line in f:
			i, u = line.replace("\n", "").split("\t")
			intToUsr[int(i)] = int(u)
	inFolder = nameIn[:nameIn.rfind("/")]
	nameIn = nameIn[nameIn.rfind("/"):].replace("_events.txt", "")

	tabRes = []
	observations = np.array(observations, dtype=object)
	clusTrue = observations[part.docs2cluster_index, -1]
	clusInf = part.docs2cluster_ID
	NMI = normalized_mutual_info_score(clusTrue, clusInf)
	NMI_last = normalized_mutual_info_score(clusTrue[-500:], clusInf[-500:])
	nbClus = len(part.clusters)
	print("NMI", NMI)

	if "Memetracker" in nameIn or r==0.:
		return
	alloc = np.array(part.docs2cluster_ID)
	for c in part.clusters:
		N = 1000  # Has to be larger
		K = 1
		tabvsctrue = []
		alinftot = np.zeros((N, N))
		for u in part.clusters[c].alpha:
			a = part.clusters[c].alpha[u].todense()
			for v in list(a.nonzero()[0]):
				if u!=v:
					alinftot[intToUsr[v], intToUsr[u]] = a[v][0]
		pop = len(alloc[alloc == c])
		if pop <= len(alloc)*0.05:
			continue

		for ctrue_file in [f for f in os.listdir(inFolder) if nameIn+"_alpha_c" in f and "done" not in f]:
			altrue_deso = sparse.load_npz(inFolder+f"/{ctrue_file}").todense()
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

			z = altrue == 0
			nnz = altrue.nonzero()
			MAE = np.mean(np.abs(alinftot-altrue))
			MAEZ = np.mean(np.abs(alinftot[z]-altrue[z]))
			MAENNZ = np.mean(np.abs(alinftot[nnz]-altrue[nnz]))

			tabvsctrue.append((MAE, MAENNZ, MAEZ))
		tabvsctrue = np.array(tabvsctrue, dtype=object)
		try:
			c_true = np.where(tabvsctrue[:, 1] == np.min(tabvsctrue[:, 1]))[0][0]
			tabvsctrue = tabvsctrue[np.sum(tabvsctrue, axis=1) == np.min(np.sum(tabvsctrue, axis=1))]
		except:
			c_true = 0
			pass
		tabRes.append((NMI, nbClus, c, c_true, pop, tabvsctrue))
		print("Clus", c, "Pop", pop, "MinErr", tabvsctrue)
	return tabRes

if __name__ == '__main__':
	if len(sys.argv)>2:
		# python Ouston.py data_file="data/Synth/Synth_PL_OL=0.0_wdsPerObs=5_vocPerClass=100_events.txt" destination="output/Synth/" theta0=0.1 number_particles=1 print_progress=True
		dataFile, outputFolder, arrR, theta0, particle_num, printRes = getArgs(sys.argv)
	else:
		#data/Synth/Synth_PL_OL=0.0_wdsPerObs=5_vocPerClass=100_events.txt
		#data/Memetracker/Memetracker_events.txt
		dataFile = "data/Memetracker/Memetracker_30min_events.txt"
		outputFolder = "output/Memetracker/"
		arrR = [0.]
		nbRuns = 1
		theta0 = 0.1
		particle_num = 4
		printRes = True
	K = 1
	if "Memetracker" in dataFile:
		horizon = 2*24*7  # Memetracker_30min time unit: 30min... ; horizon = 1 week
	else:
		horizon = 10000
	lamb0 = 0.01
	np.random.seed(1111)

	t = time.time()
	i = 0

	for r in arrR:
		name = f"{dataFile[dataFile.rfind('/'):].replace('.txt', '')}_r={r}_theta0={theta0}_particlenum={particle_num}"
		observations, V, N, C = readObservations(dataFile, outputFolder, name)
		run_fit(observations, dataFile, outputFolder, name, lamb0, r=r, theta0=theta0, particle_num=particle_num, printRes=printRes, vocabulary_size=V, number_nodes=N, number_cascades=C, horizon=horizon)


