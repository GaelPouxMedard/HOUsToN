import numpy as np
from scipy.special import erfc, gammaln
from scipy.stats import dirichlet as dirsp
from copy import deepcopy as copy

class Document(object):
	def __init__(self, index, timestamp, word_distribution, word_count):
		super(Document, self).__init__()
		self.index = index
		self.timestamp = timestamp
		self.word_distribution = np.array(word_distribution, dtype=int)
		self.word_count = np.array(word_count, dtype=int)
		
class Cluster(object):
	def __init__(self, index, num_samples, alpha0):# alpha, word_distribution, documents, word_count):
		super(Cluster, self).__init__()
		self.index = index
		self.alpha = None
		self.word_distribution = None
		self.word_count = 0
		self.likelihood_samples = np.zeros((num_samples), dtype=np.float)
		self.likelihood_samples_sansLambda = np.zeros((num_samples), dtype=np.float)
		self.triggers = np.zeros((num_samples), dtype=np.float)
		self.integ_triggers = np.ones((num_samples), dtype=np.float)  # Ones pcq premiere obs incluse

		alphas = []; log_priors = []

		for _ in range(num_samples):
			alpha = np.array([np.nan])
			log_prior = 0
			while np.isnan(alpha).any():
				alpha = np.random.random((1))
				alpha[alpha<1e-15] = 1e-20
				log_prior = 1.
			alphas.append(np.array(alpha))
			log_priors.append(log_prior)

		self.alphas = np.array(alphas)
		self.log_priors = np.array(log_priors)

	def add_document(self, doc):
		if self.word_distribution is None:
			self.word_distribution = np.copy(doc.word_distribution)
		else:
			self.word_distribution += doc.word_distribution
		self.word_count += doc.word_count

	def __repr__(self):
		return 'cluster index:' + str(self.index) + '\n' +'word_count: ' + str(self.word_count) \
		+ '\nalpha:' + str(self.alpha)+"\n"

class Particle(object):
	"""docstring for Particle"""
	def __init__(self, weight):
		super(Particle, self).__init__()
		self.weight = weight
		self.log_update_prob = 0
		self.clusters = {} # can be store in the process for efficient memory implementation, key = cluster_index, value = cluster object
		self.docs2cluster_ID = [] # the element is the cluster index of a sequence of document ordered by the index of document
		self.active_clusters = {} # dict key = cluster_index, value = list of timestamps in specific cluster (queue)
		self.cluster_num_by_now = 0

	def __repr__(self):
		return 'particle document list to cluster IDs: ' + str(self.docs2cluster_ID) + '\n' + 'weight: ' + str(self.weight)
		

def dirichlet(prior):
	''' Draw 1-D samples from a dirichlet distribution to multinomial distritbution. Return a multinomial probability distribution.
		@param:
			1.prior: Parameter of the distribution (k dimension for sample of dimension k).
		@rtype: 1-D numpy array
	'''
	return np.random.dirichlet(prior).squeeze()

def multinomial(exp_num, probabilities):
	''' Draw samples from a multinomial distribution.
		@param:
			1. exp_num: Number of experiments.
			2. probabilities: multinomial probability distribution (sequence of floats).
		@rtype: 1-D numpy array
	'''
	return np.random.multinomial(exp_num, probabilities).squeeze()

def EfficientImplementation(tn, reference_time, bandwidth, epsilon = 1e-5):
	''' return the time we need to compute to update the triggering kernel
		@param:
			1.tn: float, current document time
			2.reference_time: list, reference_time for triggering_kernel
			3.bandwidth: int, bandwidth for triggering_kernel
			4.epsilon: float, error tolerance
		@rtype: float
	'''
	max_ref_time = max(reference_time)
	max_bandwidth = max(bandwidth)
	tu = tn - ( max_ref_time + np.sqrt( -2 * max_bandwidth * np.log(0.5 * epsilon * np.sqrt(2 * np.pi * max_bandwidth**2)) ))
	return tu

def log_dirichlet_PDF(alpha, alpha0):
	return dirsp.logpdf(alpha, alpha0)

def RBF_kernel_actualrbf(reference_time, time_interval, bandwidth):
	''' RBF kernel for Hawkes process.
		@param:
			1.reference_time: np.array, entries larger than 0.
			2.time_interval: float/np.array, entry must be the same.
			3. bandwidth: np.array, entries larger than 0.
		@rtype: np.array
	'''
	numerator = - (time_interval - reference_time) ** 2 / (2 * bandwidth ** 2)
	denominator = (2 * np.pi * bandwidth ** 2 ) ** 0.5
	return np.exp(numerator) / denominator

def RBF_kernel(time_interval, alpha):
	''' RBF kernel for Hawkes process.
		@param:
			1.reference_time: np.array, entries larger than 0.
			2.time_interval: float/np.array, entry must be the same.
			3. bandwidth: np.array, entries larger than 0.
		@rtype: np.array
	'''

	return np.exp(-alpha*time_interval)

def triggering_kernel(alpha, time_intervals, donotsum=False):
	''' triggering kernel for Hawkes porcess.
		@param:
			1. alpha: np.array, entres larger than 0
			2. reference_time: np.array, entries larger than 0.
			3. time_intervals: float/np.array, entry must be the same.
			4. bandwidth: np.array, entries larger than 0.
		@rtype: np.array
	'''
	#if len(alpha) != len(reference_time):
		#raise Exception("length of alpha and length of reference time must equal")
	time_intervals = time_intervals.reshape(-1, 1)

	if len(alpha.shape) == 3:
		RBF = RBF_kernel(time_intervals, alpha)
		if donotsum:
			return alpha*RBF.T
		#return np.sum(np.sum(alpha * RBF_kernel(reference_time, time_intervals, bandwidth), axis = 1), axis = 1)
		return np.sum(np.sum(alpha*RBF, axis = 1), axis = 1)
	else:
		RBF = RBF_kernel(time_intervals, alpha)
		#return np.sum(np.sum(alpha * RBF_kernel(reference_time, time_intervals, bandwidth), axis = 0), axis = 0)
		if donotsum:
			return RBF*alpha
		return RBF.dot(alpha).sum()

def g_theta_actualgtheta(timeseq, reference_time, bandwidth, max_time):
	''' g_theta for DHP
		@param:
			2. timeseq: 1-D np array time sequence before current time
			3. base_intensity: float
			4. reference_time: 1-D np.array
			5. bandwidth: 1-D np.array
		@rtype: np.array, shape(3,)
	'''
	timeseq = timeseq.reshape(-1, 1)
	timeseq = np.array(timeseq)
	results = 0.5 * ( erfc( (- reference_time) / (2 * bandwidth ** 2) ** 0.5) - erfc( (max_time - timeseq - reference_time) / (2 * bandwidth ** 2) **0.5) )

	#return np.sum(results, axis = 0)
	return np.ones((len(results))).dot(results)

def g_theta(timeseq, max_time, alpha):
	''' g_theta for DHP
		@param:
			2. timeseq: 1-D np array time sequence before current time
			3. base_intensity: float
			4. reference_time: 1-D np.array
			5. bandwidth: 1-D np.array
		@rtype: np.array, shape(3,)
	'''
	timeseq = timeseq.reshape(-1, 1)
	timeseq = np.array(timeseq)
	results = np.exp(-alpha*(timeseq)) - np.exp(-alpha*(max_time))

	#return np.sum(results, axis = 0)
	return np.ones((len(results))).dot(results)

def update_cluster_likelihoods(timeseq, cluster, base_intensity, max_time):
	alphas = cluster.alphas
	Lambda_0 = base_intensity * max_time
	#alphas_times_gtheta = np.sum(alphas * g_theta(timeseq, reference_time, bandwidth, max_time), axis = 1) # shape = (sample number,)
	alphas_times_gtheta = alphas.dot(g_theta(np.array([timeseq[-1]]), max_time, alphas))

	time_intervals = timeseq[-1] - timeseq[:-1]
	#time_intervals = time_intervals[time_intervals>0]
	alphas_res = copy(alphas.reshape(-1, 1, alphas.shape[-1]))

	#cluster.triggers += triggering_kernel(alphas_res, reference_time, time_intervals, bandwidth)
	cluster.triggers = triggering_kernel(alphas_res, time_intervals)

	cluster.integ_triggers += alphas_times_gtheta
	#cluster.likelihood_samples += -Lambda_0 - cluster.integ_triggers + np.log(cluster.triggers+1e-100)
	cluster.likelihood_samples_sansLambda += np.log(cluster.triggers+1e-100)
	cluster.likelihood_samples = -Lambda_0 - cluster.integ_triggers + cluster.likelihood_samples_sansLambda

	return copy(cluster)

def update_triggering_kernel_optim(cluster):
	''' procedure of triggering kernel for SMC
		@param:
			1. timeseq: list, time sequence including current time
			2. alphas: 2-D np.array with shape (sample number, length of alpha)
			3. reference_time: np.array
			4. bandwidth: np.array
			5. log_priors: 1-D np.array with shape (sample number,), p(alpha, alpha_0)
			6. base_intensity: float
			7. max_time: float
		@rtype: 1-D numpy array with shape (length of alpha0,)
	'''
	alphas = cluster.alphas
	log_priors = cluster.log_priors
	logLikelihood = cluster.likelihood_samples
	log_update_weight = log_priors + logLikelihood
	log_update_weight = log_update_weight - np.max(log_update_weight)
	update_weight = np.exp(log_update_weight)

	#update_weight[update_weight<np.mean(update_weight)]=0.  # Removes noise of obviously unfit alpha samples

	sumUpdateWeight = update_weight.dot(np.ones((len(update_weight))))
	#sumUpdateWeight = np.sum(update_weight)
	update_weight = update_weight / sumUpdateWeight

	#update_weight = update_weight.reshape(-1,1)
	#alpha = np.sum(update_weight * alphas, axis = 0)
	alpha = update_weight.dot(alphas)
	#print(np.max(logLikelihood), np.min(logLikelihood), np.mean(logLikelihood))
	#print(np.max(update_weight), np.min(update_weight), np.mean(update_weight), update_weight)
	#print(len(update_weight[update_weight>0]), alpha)

	return alpha

def log_dirichlet_multinomial_distribution(cls_word_distribution, doc_word_distribution, cls_word_count, doc_word_count, vocabulary_size, priors):
	''' compute the log dirichlet multinomial distribution
		@param:
			1. cls_word_distribution: 1-D numpy array, including document word_distribution
			2. doc_word_distribution: 1-D numpy array
			3. cls_word_count: int, including document word_distribution
			4. doc_word_count: int
			5. vocabulary_size: int
			6. priors: 1-d np.array
		@rtype: float
	'''

	#arrones = np.ones((len(priors)))
	#priors_sum = np.sum(priors)
	#priors_sum = priors.dot(arrones)
	priors_sum = priors[0]*len(priors)  # ATTENTION SI PRIOR[0] SEULEMENT SI THETA0 EST SYMMETRIQUE !!!!
	log_prob = 0
	log_prob += gammaln(cls_word_count - doc_word_count + priors_sum)
	log_prob -= gammaln(cls_word_count + priors_sum)

	#log_prob += np.sum(gammaln(cls_word_distribution + priors))
	#log_prob -= np.sum(gammaln(cls_word_distribution - doc_word_distribution + priors))

	#log_prob += gammaln(cls_word_distribution + priors).dot(arrones)
	#log_prob -= gammaln(cls_word_distribution - doc_word_distribution + priors).dot(arrones)

	cnt = np.bincount(cls_word_distribution)
	un = np.arange(len(cnt))
	log_prob += gammaln(un + priors[0]).dot(cnt)  # ATTENTION SI PRIOR[0] SEULEMENT SI THETA0 EST SYMMETRIQUE !!!!

	cnt = np.bincount(cls_word_distribution-doc_word_distribution)
	un = np.arange(len(cnt))
	log_prob -= gammaln(un + priors[0]).dot(cnt)

	return log_prob
