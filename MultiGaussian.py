import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt 

### Read Data and preprocessing, keeping the 200 features with the biggest variances and dropping the rest
def read(images, labels):
	X = np.matrix(images)
	Y = np.matrix(labels)
	variances = np.zeros(784)
	rankings = np.zeros(784)
	bigV  = []
	
	for i in range(784):
		elements = []
		for j in range(len(X)):
			elements.append(X[j].item(i))
		variances[i] = np.var(elements)

	rankings = variances.argsort()
	for i in range(200):
		bigV.append(rankings[i+584])
			
	Xnew = np.hstack((X[:,bigV[0]],X[:,bigV[1]]))
	for i in range(len(bigV)-2):
		Xnew = np.hstack((Xnew, X[:,bigV[i+2]]))


	return Xnew, Y

## Estimate parameters
def parameters(X, Y):
	prior, mu, cov = [], [], []
	

	for i in range(10):
		indices = [idx for idx, Y in enumerate(Y) if Y==[i]]
		samples = [X[i] for i in indices] 
		labels = [Y[i] for i in indices]
		### prior probability
		prior.append(len(labels) / len(Y))
		### sample mean 
		sum_samples = np.zeros((1, 200))
		for j in range(len(samples)):
			sum_samples += samples[j]
		mu.append(sum_samples.T/len(samples))
		### covariance matrix
		sum_matrices = np.zeros((200, 200))
		for k in range(len(samples)):
			sum_matrices += np.dot(samples[k].T-mu[i], (samples[k].T-mu[i]).T)
		cov.append(sum_matrices/len(samples) + 0.1*np.identity(200))
	return prior, mu, cov 

## Modeling
def gaussian_classifier(input_image, prior, mu, cov):
	x = input_image
	results = []
	for j in range(10):
		results.append(math.log(prior[j]) - 
			0.5*np.dot(np.dot((x-mu[j]).T, np.linalg.inv(cov[j])),x-mu[j]) - 
				0.5*math.log(np.linalg.norm(cov[j])))
	return results.index(max(results))


## calculating error rates
def error(test_sample, labels):
	correct = 0
	for x in range(len(test_sample)):
		if test_sample[x] is np.asscalar(labels[x]):
			correct += 1
	return (1-(correct/float(len(test_sample))))

## splitting the data into the training set and the testing set
def split(images,labels, ratio):
	trainingX, testingX, trainingY, testingY = [],[],[],[]
	for i in range(10000):
		if np.random.random_sample()<ratio:
			trainingX.append(images[i])
			trainingY.append(labels[i])
		else:
			testingX.append(images[i])
			testingY.append(labels[i])
	return trainingX, testingX, trainingY, testingY


### Classification
def main():

	X, Y = read(sio.loadmat('hw1data.mat')['X'], sio.loadmat('hw1data.mat')['Y'])
	accuracy = []
	splits = [0.5, 0.6, 0.7, 0.8, 0.9]
	for i in range(5):
		trainingX, testingX, trainingY, testingY = split(X, Y, (i+5)/10)
		prior, mu, cov = parameters(trainingX, trainingY)
		results = []
		for x in range(len(testingX)):
			input_image = testingX[x].T
			result = (gaussian_classifier(input_image, prior, mu, cov))
			results.append(result)

		error_rate = error(results, testingY)
		accuracy.append(1-error_rate)
		print(error_rate)
    
    ## creating the plot for question 5.3 
	plt.plot(splits, accuracy)
	plt.title('split ratios VS accuracy ratios for Gaussian Classifier')
	plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9])
	plt.xlabel('split ratio')
	plt.ylabel('accuracy ratio')
	plt.legend()
	plt.show()

main()





