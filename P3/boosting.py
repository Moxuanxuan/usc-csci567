import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		H = 0
		for i in range(self.T):
			H = H + self.betas[i]*self.clfs_picked[i].predict(features)
		H[H>=0] = 1
		H[H<0] = -1
		H = H.flatten().tolist()
		return H
		########################################################
		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		X = np.array(features)
		y = np.array(labels)
		N = X.shape[0]
		y = y.reshape(N,1)
		D = np.ones((N,1))/N
		for i in range(self.T):
			m = 10
			for clf in self.clfs:
				pred_y = np.array(clf.predict(features))
				m_sub = (y != pred_y).reshape(1, N).dot(D).flatten()
				if m_sub < m:
					m = m_sub
					clf_min = clf
			self.clfs_picked.append(clf_min)
			e = m
			beta = (1/2)*np.log((1-e)/e)
			self.betas.append(beta)
			pred_y = np.array(clf_min.predict(features))
			D_pre = ((y == pred_y)*np.exp(-beta)).flatten()
			D_pre[D_pre==0] = np.exp(beta)
			D = np.multiply(D, D_pre.reshape(N,1))
			D = D/D.sum()
		############################################################
		
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	