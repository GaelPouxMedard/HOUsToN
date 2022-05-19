from __future__ import print_function
from __future__ import division

import datetime
import pickle
import bz2

import numpy as np

from BL_DHP_utils import *
import copyreg as copy_reg
import types
from copy import deepcopy as copy
import time

from BL_DHP_Evaluation import compDists, confMat

np.random.seed(1111)

def _pickle_method(m):
	if m.im_self is None:
		return getattr, (m.im_class, m.im_func.func_name)
	else:
		return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


class Dirichlet_Hawkes_Process(object):
	"""docstring for Dirichlet Hawkes Prcess"""
	def __init__(self, particle_num, base_intensity, theta0, alpha0, vocabulary_size, sample_num, r):
		super(Dirichlet_Hawkes_Process, self).__init__()
		self.r = r
		self.particle_num = particle_num
		self.base_intensity = base_intensity
		self.theta0 = theta0
		self.alpha0 = alpha0
		self.vocabulary_size = vocabulary_size
		self.horizon = 2*24*7
		self.sample_num = sample_num
		self.particles = []
		for i in range(particle_num):
			self.particles.append(Particle(weight = 1.0 / self.particle_num))

		self.active_interval = None


	def sequential_monte_carlo(self, doc, threshold):
		# Set relevant time interval
		T = doc.timestamp
		tu = T-self.horizon
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
		# Update the triggering kernel
		particle.clusters[selected_cluster_index].alpha = self.parameter_estimation(particle, selected_cluster_index)
		# Calculate the weight update probability
		particle.log_update_prob = self.calculate_particle_log_update_prob(particle, selected_cluster_index, doc)
		return particle

	def sampling_cluster_label(self, particle, doc):
		if len(particle.clusters) == 0: # The first document is observed
			particle.cluster_num_by_now += 1
			selected_cluster_index = particle.cluster_num_by_now
			selected_cluster = Cluster(index = selected_cluster_index, num_samples=self.sample_num, alpha0=self.alpha0)
			selected_cluster.add_document(doc)
			particle.clusters[selected_cluster_index] = selected_cluster #.append(selected_cluster)
			particle.docs2cluster_ID.append(selected_cluster_index)
			particle.active_clusters[selected_cluster_index] = [doc.timestamp]
			self.active_cluster_logrates = {0:0, 1:0}

		else: # A new document arrives
			active_cluster_indexes = [0] # Zero for new cluster
			active_cluster_rates = [self.base_intensity**self.r]
			cls0_log_dirichlet_multinomial_distribution = log_dirichlet_multinomial_distribution(doc.word_distribution, doc.word_distribution, \
																								 doc.word_count, doc.word_count, self.vocabulary_size, self.theta0)
			active_cluster_textual_probs = [cls0_log_dirichlet_multinomial_distribution]
			# Update list of relevant timestamps
			particle.active_clusters = self.update_active_clusters(particle)

			# Posterior probability for each cluster
			for active_cluster_index in particle.active_clusters:
				timeseq = particle.active_clusters[active_cluster_index]
				active_cluster_indexes.append(active_cluster_index)
				time_intervals = doc.timestamp - np.array(timeseq)
				alpha = particle.clusters[active_cluster_index].alpha
				rate = triggering_kernel(alpha, time_intervals)

				# Powered Dirichlet-Hawkes prior
				active_cluster_rates.append(rate)

				# Language model likelihood
				cls_word_distribution = particle.clusters[active_cluster_index].word_distribution + doc.word_distribution
				cls_word_count = particle.clusters[active_cluster_index].word_count + doc.word_count
				cls_log_dirichlet_multinomial_distribution = log_dirichlet_multinomial_distribution(cls_word_distribution, doc.word_distribution, \
																									cls_word_count, doc.word_count, self.vocabulary_size, self.theta0)
				active_cluster_textual_probs.append(cls_log_dirichlet_multinomial_distribution)

			# Posteriors to probabilities
			#print(1, list(active_cluster_rates))
			#print(2, list(active_cluster_textual_probs))
			active_cluster_logrates = self.r*np.log(np.array(active_cluster_rates)+1e-100)
			self.active_cluster_logrates = {c: active_cluster_logrates[i+1] for i, c in enumerate(particle.active_clusters)}
			self.active_cluster_logrates[0] = active_cluster_logrates[0]
			cluster_selection_probs = active_cluster_logrates + active_cluster_textual_probs # in log scale
			cluster_selection_probs = cluster_selection_probs - np.max(cluster_selection_probs) # prevent overflow
			cluster_selection_probs = np.exp(cluster_selection_probs)
			cluster_selection_probs = cluster_selection_probs / np.sum(cluster_selection_probs)
			#print(cluster_selection_probs)

			# Random cluster selection
			pNewOrExisting = np.array([cluster_selection_probs[0], np.sum(cluster_selection_probs[1:])])
			pNewOrExisting = pNewOrExisting/np.sum(pNewOrExisting)
			newOrExisting = multinomial(exp_num = 1, probabilities = pNewOrExisting)

			newOrExisting = np.nonzero(newOrExisting)[0]
			if newOrExisting == 0:
				cluster_selection_probs[0] = 1
				cluster_selection_probs[1:] = 0
			elif newOrExisting == 1:
				cluster_selection_probs[0] = 0
				cluster_selection_probs[1:]= cluster_selection_probs[1:]/np.sum(cluster_selection_probs[1:])

			try:
				selected_cluster_array = multinomial(exp_num = 1, probabilities = cluster_selection_probs)
			except Exception as e:
				print(2, e, cluster_selection_probs)
				pause()
			selected_cluster_index = np.array(active_cluster_indexes)[np.nonzero(selected_cluster_array)][0]

			# New cluster drawn
			if selected_cluster_index == 0:
				particle.cluster_num_by_now += 1
				selected_cluster_index = particle.cluster_num_by_now
				self.active_cluster_logrates[selected_cluster_index] = self.active_cluster_logrates[0]
				selected_cluster = Cluster(index = selected_cluster_index, num_samples=self.sample_num, alpha0=self.alpha0)
				selected_cluster.add_document(doc)
				particle.clusters[selected_cluster_index] = selected_cluster
				particle.docs2cluster_ID.append(selected_cluster_index)
				particle.active_clusters[selected_cluster_index] = [doc.timestamp]

			# Existing cluster drawn
			else:
				selected_cluster = particle.clusters[selected_cluster_index]
				selected_cluster.add_document(doc)
				particle.docs2cluster_ID.append(selected_cluster_index)
				particle.active_clusters[selected_cluster_index].append(doc.timestamp)

		return particle, selected_cluster_index

	def parameter_estimation(self, particle, selected_cluster_index):
		timeseq = np.array(particle.active_clusters[selected_cluster_index])

		# Observation is alone in the cluster => the cluster is new => random initialization of alpha
		# Note that it cannot be a previously filled cluster since it would have 0 chance to get selected (see sampling_cluster_label)
		if len(timeseq)==1:
			alpha = dirichlet(self.alpha0)
			return alpha

		T = self.active_interval[1]
		particle.clusters[selected_cluster_index] = update_cluster_likelihoods(timeseq, particle.clusters[selected_cluster_index], self.base_intensity, T)
		alpha = update_triggering_kernel_optim(particle.clusters[selected_cluster_index])
		return alpha

	def update_active_clusters(self, particle):
		tu = self.active_interval[0]
		keys = list(particle.active_clusters.keys())
		for cluster_index in keys:
			timeseq = particle.active_clusters[cluster_index]
			active_timeseq = [t for t in timeseq if t > tu]
			particle.active_clusters[cluster_index] = active_timeseq
			if len(active_timeseq)==0 and False:
				del particle.active_clusters[cluster_index]  # If no observation is relevant anymore, the cluster has 0 chance to get chosen => we remove it from the calculations
				del particle.clusters[cluster_index].alphas
				del particle.clusters[cluster_index].log_priors
				del particle.clusters[cluster_index].likelihood_samples
				del particle.clusters[cluster_index].triggers
				del particle.clusters[cluster_index].integ_triggers
		return particle.active_clusters

	def calculate_particle_log_update_prob(self, particle, selected_cluster_index, doc):
		cls_word_distribution = particle.clusters[selected_cluster_index].word_distribution
		cls_word_count = particle.clusters[selected_cluster_index].word_count
		doc_word_distribution = doc.word_distribution
		doc_word_count = doc.word_count

		log_update_prob = log_dirichlet_multinomial_distribution(cls_word_distribution, doc_word_distribution, cls_word_count, doc_word_count, self.vocabulary_size, self.theta0)

		return log_update_prob

	def particles_normal_resampling(self, particles, threshold):
		#print('\nparticles_normal_resampling')
		weights = []; log_update_probs = []
		for particle in particles:
			weights.append(particle.weight)
			log_update_probs.append(particle.log_update_prob)
		weights = np.array(weights)
		log_update_probs = np.array(log_update_probs)
		log_update_probs = log_update_probs - np.max(log_update_probs) # prevent overflow
		update_probs = np.exp(log_update_probs)
		weights = weights * update_probs
		weights = weights / np.sum(weights) # normalization
		resample_num = len(np.where(weights + 1e-5 < threshold)[0])

		if resample_num == 0: # No need to resample particle, but still need to assign the updated weights to particles
			for i, particle in enumerate(particles):
				particle.weight = weights[i]
			return particles
		else:
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

def parse_newsitem_2_doc(news_item, vocabulary_size):
	''' convert (id, timestamp, word_distribution, word_count) to the form of document
	'''
	#print(news_item)
	index = news_item[0]
	timestamp = news_item[1] # / 3600.0 # unix time in hour
	word_id = news_item[2][0]
	count = news_item[2][1]
	word_distribution = np.zeros(vocabulary_size)
	word_distribution[word_id] = count
	word_count = np.sum(count)
	doc = Document(index, timestamp, word_distribution, word_count)
	# assert doc.word_count == np.sum(doc.word_distribution)
	return doc

def readData(folder, name):
	observations = []
	with open(folder+name+"_events.txt", "r") as f:
		for i, line in enumerate(f):
			l = line.replace("\n", "").split("\t")

			timestamp = float(l[1])
			words = l[2].split(",")
			try:  # If Synth data
				clusTemp = clusTxt = int(l[5])
			except:
				clusTemp = clusTxt = -1

			uniquewords, cntwords = np.unique(words, return_counts=True)
			uniquewords, cntwords = np.array(uniquewords, dtype=int), np.array(cntwords, dtype=int)

			tup = (i, timestamp, (uniquewords, cntwords), clusTemp, clusTxt)
			observations.append(tup)


	return observations

def run_fit_synth(params):
	r, folder, folderOut, name, lamb0, sample_num, theta0, save, particle_num = params
	nameOut = name

	observations = readData(folder, name)

	import matplotlib.pyplot as plt
	observations = np.array(observations, dtype=object)
	#plt.plot(observations[:2000, 1], observations[:2000, -1]/10, "o", markersize=3)
	#plt.show()

	run_fit(observations, folderOut, nameOut, lamb0, r=r, sample_num=sample_num, particle_num=particle_num, printRes=True, theta0=theta0, save=save)

def run_fit(observations, folderOut, nameOut, lamb0, r=1., theta0=1., alpha0 = None, sample_num=2000, particle_num=8, printRes=False, alphaTrue=None, save=True):
	"""
	observations = ([array int] index_obs, [array float] timestamp, ([array int] unique_words, [array int] count_words), [opt, int] temporal_cluster, [opt, int] textual_cluster)
	folderOut = Output folder for the results
	nameOut = Name of the file to which _particles_compressed.pklbz2 will be added
	lamb0 = base intensity
	means, sigs = means and sigmas of the gaussian RBF kernel
	r = exponent parameter of the Powered Dirichlet process; defaults to 1. (standard Dirichlet process)
	theta0 = value of the language model symmetric Dirichlet prior
	alpha0 = symmetric Dirichlet prior from which samples used in Gibbs sampling are drawn (estimation of alpha)
	sample_num = number of samples used in Gibbs sampling
	particle_num = number of particles used in the Sequential Monte-Carlo algorithm
	printRes = whether to print the results according to ground-truth (optional parameters of observations and alpha)
	alphaTrue = ground truth alpha matrix used to generate the observations from gaussian RBF kernel
	"""

	particle_num = particle_num
	allWds = set()
	for a in observations:
		for w in a[2][0]:
			allWds.add(w)
	vocabulary_size = len(allWds)+2

	base_intensity = lamb0
	if theta0 is None: theta0 = 1.
	theta0 = np.array([theta0 for _ in range(vocabulary_size)])
	if alpha0 is None: alpha0 = 1.
	alpha0 = np.array([alpha0])
	sample_num = sample_num
	threshold = 1.0 / (particle_num*1.5)
	with open(f"output_BL/results_{nameOut}.txt", "w+") as f: f.write("")

	DHP = Dirichlet_Hawkes_Process(particle_num = particle_num, base_intensity = base_intensity, theta0 = theta0,
								   alpha0 = alpha0, vocabulary_size = vocabulary_size,
								   sample_num = sample_num, r=r)

	t = time.time()


	for i, news_item in enumerate(observations):
		doc = parse_newsitem_2_doc(news_item = news_item, vocabulary_size = vocabulary_size)
		DHP.sequential_monte_carlo(doc, threshold)

		if i%100==1 and printRes:
			print(f'r={r} - Handling document {i}/{len(observations)} - t={np.round(news_item[1]-observations[1][1], 1)}h - Average time : {np.round((time.time()-t)*1000/(i), 0)}ms - '
				  f'Remaining time : {np.round((time.time()-t)*(len(observations)-i)/(i*3600), 2)}h - '
				  f'ClusTot={DHP.particles[0].cluster_num_by_now} - ActiveClus = {len(DHP.particles[0].active_clusters)}')
			un, cnt = np.unique(np.array(observations, dtype=object)[:i + 2, 4], return_counts=True)
			#print("True labels:", un, "\tCounts:", cnt, "\t")
			print("[# clus, NMI, NMI last 500 obs, ARI]:", confMat(observations[:i + 1], [DHP.particles[0]])[0][0:4])

		if i%1000==1:
			while True:
				try:
					with bz2.BZ2File(folderOut+nameOut+'_particles_compressed.pklbz2', 'w') as sfile:
						pickle.dump(DHP.particles, sfile)
					if save:
						with open(f"output_BL/results_{nameOut}.txt", "a+") as f:
							res = confMat(observations[:i + 1], [DHP.particles[0]])[0]
							f.write(f"{news_item[1]}\t{i}\t{res[1]}\t{res[2]}\t{res[3]}\n")
					break
				except Exception as e:
					print(i, e)
					time.sleep(10)
					continue


	while True:
		try:
			with bz2.BZ2File(folderOut+nameOut+'_particles_compressed.pklbz2', 'w') as sfile:
				pickle.dump(DHP.particles, sfile)
			break
		except Exception as e:
			time.sleep(10)
			print(e)
			continue

if __name__ == '__main__':
	import sys
	try:
		folderData = sys.argv[1]
		file = sys.argv[2]
	except Exception as e:
		print("=====", e)
		folderData = "Memetracker"
		file = f"Memetracker_30min"

	folder=f"data/{folderData}/"
	folderOut=f"output_BL/{folderData}/"
	lamb0 = 0.01
	sample_num = 2000
	theta0=0.1
	save = True
	particle_num = 4

	np.random.seed(1111)

	r = 1.
	params = (r, folder, folderOut, file, lamb0, sample_num, theta0, save, particle_num)
	run_fit_synth(params)