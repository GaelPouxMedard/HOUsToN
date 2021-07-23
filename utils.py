import os
import pickle
import random
import sys

import numpy as np
import sparse
from scipy.special import erfc, gammaln
from scipy.stats import dirichlet as dirsp
from copy import deepcopy as copy
import cvxpy as cp
import gc
import multiprocessing as mp
import subprocess
import platform
import psutil
from pympler.asizeof import asizeof

class Document(object):
	def __init__(self, index, timestamp, word_distribution, word_count, node, casc, trueClus=-1):
		super(Document, self).__init__()
		self.index = index
		self.timestamp = timestamp
		self.word_distribution = np.array(word_distribution, dtype=int)
		self.word_count = np.array(word_count, dtype=int)
		self.node = node
		self.casc = casc
		self.trueClus = trueClus

	def __repr__(self):
		return f"Doc: {self.node, self.timestamp}"

class Cluster(object):
	def __init__(self, index, num_nodes, K):
		super(Cluster, self).__init__()
		self.index = index
		self.num_nodes = num_nodes
		self.word_distribution = None
		self.word_count = 0
		self.K = K

		self.alpha = {}

		self.coeffS = {}
		self.coeffH = {}
		self.num_docs_node = {}

	def add_document(self, doc):
		if self.word_distribution is None:
			self.word_distribution = np.copy(doc.word_distribution)
			self.word_count += doc.word_count
		else:
			self.word_distribution += doc.word_distribution
			self.word_count += doc.word_count

		if doc.node not in self.alpha:
			self.alpha[doc.node] = sparse.zeros((self.num_nodes,1))
		if doc.node not in self.coeffS:
			self.coeffS[doc.node] = {}
			self.coeffH[doc.node] = []

		if doc.node not in self.num_docs_node: self.num_docs_node[doc.node] = 0
		self.num_docs_node[doc.node] += 1

	def __repr__(self):
		return 'cluster index:' + str(self.index) + '\n' +'word_count: ' + str(self.word_count) \
			   + '\nalpha:' + str(self.alpha)+"\n"

class Particle(object):
	"""docstring for Particle"""
	def __init__(self, weight):
		super(Particle, self).__init__()
		self.weight = weight
		self.log_update_prob = 0
		self.clusters = {} # can be stored in the process for efficient memory implementation, key = cluster_index, value = cluster object
		self.docs2cluster_ID = [] # the element is the cluster index of a sequence of document ordered by the index of document
		self.docs2cluster_index = [] # the element is the cluster index of a sequence of document ordered by the index of document
		self.active_clusters = {} # dict key = cluster_index, value = list of timestamps in specific cluster (queue)
		self.cluster_num_by_now = 0

	def __repr__(self):
		return 'particle document list to cluster IDs: ' + str(self.docs2cluster_ID) + '\n' + 'weight: ' + str(self.weight)


def multinomial(exp_num, probabilities):
	''' Draw samples from a multinomial distribution.
		@param:
			1. exp_num: Number of experiments.
			2. probabilities: multinomial probability distribution (sequence of floats).
		@rtype: 1-D numpy array
	'''
	return np.random.multinomial(exp_num, probabilities).squeeze()

def logS(ti, tj, alphaji):
	#if kernel=="exp":
	return -alphaji[0]*(ti-tj)

def H(ti, tj, alphaji):
	#if kernel == "exp":
	return alphaji[0]

def update_cluster_likelihoods(tupsPrec, cluster):
	if len(tupsPrec)<=1:
		return cluster
	doci = tupsPrec[-1]
	ti, nodei = doci.timestamp, doci.node

	arrH, coeffH = [], []
	seen = False

	for docj in reversed(tupsPrec[:-1]):
		tj, nodej = docj.timestamp, docj.node
		if nodei==nodej: continue
		if doci.casc != docj.casc: continue

		#if kernel=="exp":
		if nodej not in arrH:  # We only consider the last observation of each node to distinguish cascades
			if nodej not in cluster.coeffS[nodei]: cluster.coeffS[nodei][nodej]=[]
			cluster.coeffS[nodei][nodej].append([ti, ti-tj])
			arrH.append(nodej)
			coeffH.append(1.)
			seen = True

	if seen:
		cluster.coeffH[nodei].append([ti, arrH, coeffH])

	return cluster

def buildExpression(cluster, node):
	if len(cluster.coeffS[node])==0: return -1, cp.Variable(), dict(), False
	expr = cp.Constant(0.)
	vars = cp.Variable((len(cluster.coeffS[node]), cluster.K))
	nodeToInt = {}
	intToNode = {}
	arrcoeffS = []
	for i, nodej in enumerate(cluster.coeffS[node]):
		arrcoeffS.append(-np.sum(np.array(cluster.coeffS[node][nodej], dtype=object)[:, 1]))
		#expr += -cluster.coeffS[node][nodej]*vars[i]
		nodeToInt[nodej] = i
		intToNode[i] = nodej
	arrcoeffS = np.array(arrcoeffS)
	expr += arrcoeffS@vars
	for _, inds, coeffs in cluster.coeffH[node]:
		tmp = 0.
		msk = np.zeros((len(cluster.coeffS[node])))
		for i in range(len(inds)):
			#tmp += vars[nodeToInt[inds[i]]]*coeffs[i]
			msk[nodeToInt[inds[i]]] += coeffs[i]
		#expr += cp.log(tmp)
		expr += cp.log(msk@vars)
	return expr, vars, nodeToInt, True

def update_triggering_kernel_optim(cluster, node, r, forcefit=False):
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

	n = cluster.num_docs_node[node]
	if not forcefit and not ((n<5 and n%1==0) or (n<10 and n%2==0) or (n<30 and n%3==0) or (n<100 and n%10==0) or (n<200 and n%20==0) or (n<500 and n%30==0) or (n>500 and n%50==0)):
		return cluster.alpha[node]
	if not forcefit and r==0. and np.random.random()<0.999:
		pass
		#return cluster.alpha[node]

	# To avoid storing (memory heavy) expression for each node
	expr, vars, nodeToInt, hasSmthgToFit = buildExpression(cluster, node)
	if not hasSmthgToFit:
		return cluster.alpha[node]
	objective = cp.Maximize(expr)
	constraints = [vars >= 0, vars <= 1]

	prob = cp.Problem(objective, constraints)

	prob.solve(cp.SCS)#, max_iters=10000)#, gpu=True, use_indirect=True, verbose=True)
	vals = vars.value

	K = cluster.K
	c, d = [], []
	if prob.status=="optimal_inaccurate":
		print("=== OPTIMAL INACCURATE === (may be a node without parent candidate)")
	#print(expr)
	#print(vals.flatten())
	if vals is None or prob.status=="optimal_inaccurate":
		return cluster.alpha[node]
	for nodej in nodeToInt:
		val = vals[nodeToInt[nodej]]
		if val is None:
			continue
		for k in range(len(val)):
			c.append([nodej, k])
			if val[k]<1e-5: val[k]=0.
			d.append(val[k])
	c = list(zip(*c))
	alpha = sparse.COO(c, d, shape=(cluster.num_nodes, K))

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
