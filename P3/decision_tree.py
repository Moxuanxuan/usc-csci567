import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		
		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●
				               
				      branches = [[2,2], [4,0]]
			'''
			########################################################
			bran = np.array(branches)
			C,B = bran.shape
			pbran = bran.sum(axis=0)/bran.sum()
			pclass = bran/bran.sum(axis = 0)
			logpclass = np.log2(pclass)
			logpclass[np.isinf(logpclass)] = 0
			H = -np.multiply(pclass, logpclass).sum(axis=0)
			cond_H = np.multiply(pbran, H).sum()
			return cond_H
			########################################################
			
		cond_entro = []
		X = np.array(self.features)
		for idx_dim in range(len(self.features[0])):
		############################################################
			subx = X[:,idx_dim].flatten()
			attrs = np.unique(subx)
			labels = np.unique(self.labels)
			B = len(attrs)
			C = len(labels)
			branches = np.zeros((C,B))
			for i in range(B):
				for j in range(C):
					branches[j,i] = ((subx == attrs[i])*(self.labels == labels[j])).sum()
			branches = branches.tolist()
			cond_entro.append(conditional_entropy(branches))
		############################################################




		############################################################
		self.dim_split = cond_entro.index(np.min(cond_entro))
		self.feature_uniq_split = np.unique(X[:,self.dim_split]).tolist()
		B = len(self.feature_uniq_split)
		y = np.array(self.labels)
		for i in range(B):
			subx =  X[:,self.dim_split].flatten()
			boolvec = (subx == self.feature_uniq_split[i])
			childfeatures = X[boolvec,:]
			childfeatures = np.delete(childfeatures, self.dim_split, axis=1).tolist()
			childlabels = y[boolvec].tolist()
			childNode = TreeNode(childfeatures,childlabels,self.num_cls)
			childcheck = np.array(childfeatures)
			if len(childfeatures[0])==0:
				childNode.splittable = False
			# if ((childcheck-childcheck[0])==0).all():
			# 	childNode.splittable = False
			self.children.append(childNode)
		############################################################




		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			feature = feature[:self.dim_split]+feature[self.dim_split+1:]
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



