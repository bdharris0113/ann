import numpy as np
#from videoSupport import *
from scipy import optimize


class nnet():
	def __init__(self,Lambda=0):
		#this will be static and should be changed before training
		#to fit desired network
		#ex if there are 4 input values, input_layer_nodes = 4
		#hidden_lay means number of nodes per layer (right now only one layer)
		self.input_layer_nodes = 3
		self.hidden_layer_nodes = 1
		self.output_layer_nodes = 1

		#weights
		#init the weights should be randomized 
		self.w1 = np.random.randn(self.input_layer_nodes, self.hidden_layer_nodes)
		self.w2 = np.random.randn(self.hidden_layer_nodes,self.output_layer_nodes)

		#regularization param
		self.Lambda = Lambda

	def forward(self,x):
		#propagate data through network
		#x: input_data, y_hat: calc output
		self.z2 = np.dot(x,self.w1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2,self.w2)
		self.y_hat = self.sigmoid(self.z3)
		return self.y_hat	

	def sigmoid(self,z):
		return 1 / (1+np.exp(-z))

	def sigmoid_prime(self,z):
		f = self.sigmoid(z)
		return f*(1-f)

	def cost_func(self,X,y):
		#compute cost / difference between expected & calc output given input
		#X: input matrix, y: expected output matrix

		self.y_hat = self.forward(X) #calc output
		J = 0.5*sum((y-self.y_hat)**2)/X.shape[0] + (self.Lambda/2)*(sum(self.w1**2)+sum(self.w2**2))	#actual difference
		return J

	def cost_func_prime(self,x,y):
		#compute deriv w/ respect to w1,w2
		#x: input_data, y: expected output, y_hat: calc output
		self.y_hat = self.forward(x)

		delta3 = np.multiply(-(y-self.y_hat), self.sigmoid_prime(self.z3))
		dj_dw2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.w2

		delta2 = np.dot(delta3, self.w2.T)*self.sigmoid_prime(self.z2)
		dj_dw1 = np.dot(x.T,delta2)/X.shape[0] + self.Lambda*self.w1

		return dj_dw1,dj_dw2

	def compute_grad(self,X,y):
		dj_dw1,dj_dw2 = self.cost_func_prime(X,y)
		return np.concatenate((dj_dw1.ravel(),dj_dw2.ravel()))

	def get_params(self):
		return np.concatenate((self.w1.ravel(),self.w2.ravel()))

	def set_params(self,params):
		#set w1,w2 using vector
		w1_start = 0
		w1_end = self.hidden_layer_nodes*self.input_layer_nodes
		self.w1 = np.reshape(params[w1_start:w1_end],(self.input_layer_nodes,self.hidden_layer_nodes))
		w2_end = w1_end + self.hidden_layer_nodes*self.output_layer_nodes
		self.w2 = np.reshape(params[w1_end:w2_end],(self.hidden_layer_nodes,self.output_layer_nodes))


class trainer():
	def __init__(self,ann):
		self.ann = ann

	def cost_func_wrapper(self,params,X,y):
		self.ann.setParams(params)
		cost = self.ann.costFunction(X,y)
		grad = self.ann.compute_grad(X,y)

	def callback(self,params):
		self.ann.set_params(params)
		self.j.append(self.ann.cost_func(self.X,self.y))

	def train(self,X,y):
		#internal var for callback / print
		self.X = X
		self.y = y

		#empty list to store costs
		self.j = []

		params0 = self.ann.get_params()
		options = {'maxiter':200,'disp':True}

		_res = optimize.minimize(self.cost_func_wrapper,params0,jac=True,method='BFGS',args=(X,y),options=options,callback=self.callback)

		self.ann.set_params(_res.x)
		self.optimizeationResults = _res

