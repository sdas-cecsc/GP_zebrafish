
# coding: utf-8

# In[1]:

import numpy as np
from scipy import ndimage
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import gaussian_process
from sklearn.cross_validation import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from itertools import izip
import csv

debug_level=0
def debug(message,d_level=0):
	#greater level means more details
	if d_level == 0 and debug_level >= 0:
		print message
	if d_level == 1 and debug_level >= 1:
		print message
	if d_level == 2 and debug_level >= 2:
		print message
def error(message):
	print message

num_images = 319
images = np.empty([1,20000])


# In[2]:

for i in range(1,num_images+1):
	filename = "gray_images_all/image" + str(i) + ".jpg"
	img = ndimage.imread(filename)
	if (i == 1):
		images = img.flatten()
	else:
		images = np.vstack((images, img.flatten()))
		
images = images.astype(float)
debug(('images.shape: '+str(images.shape)),0)
debug(('images.min: '+str(images.min())),1)
debug(('images.max: '+str(images.max())),1)


# In[3]:

scaler = preprocessing.StandardScaler().fit(images)
images_scaled = scaler.transform(images)
debug(('images_scaled.shape: '+str(images_scaled.shape)),1)
debug(('images_scaled.min: '+str(images_scaled.min())),1)
debug(('images_scaled.max: '+str(images_scaled.max())),1)


# In[4]:
debug('Appying PCA to the data...',0)
thresh_components = 10
n_components = thresh_components
while(True):
	pca0 = decomposition.PCA(n_components=n_components)
	pca0.fit(images_scaled)
	if pca0.explained_variance_ratio_.sum() > 0.9:
		break
	n_components += thresh_components


# In[5]:

pca = decomposition.PCA(n_components=n_components)
pca.fit(images_scaled)
data = pca.transform(images_scaled)
debug(( 'Data Shape:'+str( data.shape)),0)
debug(( 'data.min: '+str(data.min())),1)
debug(( 'data.max: '+str(data.max())),1)


# In[6]:

debug(( 'End PCA to the data with n_components_ = '+str(pca.n_components_)),0)
debug(( 'pca.explained_variance_ratio_.sum: '+str(pca.explained_variance_ratio_.sum())),2)
# In[7]:

n_dependencies_ub = 31

for n_dependencies in range(2,n_dependencies_ub):
	debug('-------------------------------------------------\n',0)
	debug(('n_dependencies :'+str( n_dependencies)+'\n'),0)

	# In[8]:

	for i in range(len(data)-n_dependencies):
		newrow = []
		for j in range(n_dependencies):
			newrow = np.hstack((newrow, data[i+j]))
		if (i == 0):
			X = newrow
		else:
			X = np.vstack((X, newrow))


	# In[9]:

	print 'X :',X.shape
	ylist = []
	for i in range(n_dependencies,len(data)):
		ylist.append(data[i])
	y = np.asarray(ylist)
	debug(( 'y :'+str( y.shape)),2)


	# In[10]:

	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=0)
	test_size = 0.2
	if X.shape[0] == y.shape[0]:
		split = int(round((1-test_size)*X.shape[0]))
		X_train, X_test = X[:split], X[split:X.shape[0]]
		y_train, y_test = y[:split], y[split:y.shape[0]]
		debug(( 'X_train.shape: '+str(X_train.shape)),0)
		debug(( 'Y_train.shape: '+str(y_train.shape)),0)
		debug(( 'X_test.shape: '+str(X_test.shape)),0)
		debug(( 'y_test.shape: '+str(y_test.shape)),0)
	else:
		error('X, y shapes do not match')


	# In[11]:

	gp = gaussian_process.GaussianProcess(regr='constant',
										  corr='squared_exponential',
										  theta0=2e-2,
										  thetaL=1e-4,
										  thetaU=1e-1)
	debug('Learing the model...',0)
	gp.fit(X_train, y_train)


	# In[12]:

	debug(( 'X_test[0].shape: '+str(X_test[0].shape)),1)
	debug(( 'X_test[0][:pca.n_components_].shape: '+str(X_test[0][:pca.n_components_].shape)),1)


	# In[13]:
	debug('Testing the model...',0)
	sigma = []
	y_pred_list=np.zeros((y_test.shape[0],pca.n_components_))
	debug(( 'y_pred_list'+str(y_pred_list.shape)),1)
	for i in range(len(X_test)-n_dependencies):
		if(i == 0):
			X_test_sample = X_test[0]
			debug(( 'X_test_sample for i=0: '+str(X_test_sample.shape)),2)
		elif(i<n_dependencies):
			end = (n_dependencies-i)*pca.n_components_
			X_test_sample = np.hstack((X_test[i][:end]))
			debug(( 'X_test['+str(i)+']'+str(X_test[i][:end].shape)),2)
			for j in range(i):
				debug((  'y_pred_list['+str(j)+']'+str(y_pred_list[j].shape)),2)
				X_test_sample = np.hstack((X_test_sample,y_pred_list[j]))
				debug((  '--X_test_sample  for i='+str(i)+': '+str(X_test_sample.shape)),2)
			debug(('X_test_sample for i='+str(i)+': '+str(X_test_sample.shape)),2)
		else:
			strt = (i-n_dependencies)
			debug((  'y_pred_list['+str(strt)+':'+str(i)+']'+str(y_pred_list[strt:i].shape)),2)
			X_test_sample = np.hstack((y_pred_list[strt]))
			for j in y_pred_list[strt+1:i]:
				debug(str(j),2)
				X_test_sample = np.hstack((X_test_sample,j))
			debug((  'X_test_sample for i='+str(i)+': '+str(X_test_sample.shape)),2)
		y_test_pred, mse_pred = gp.predict(X_test_sample, eval_MSE=True)
		debug((  'y_test_pred'+str(y_test_pred.shape)),2)
		y_pred_list[i] = y_test_pred[0]
		sigma.append(np.sqrt(mse_pred))
	y_pred = np.asarray(y_pred_list)
	y_pred_size = y_pred.shape[0]
	debug((  'y_pred.shape: '+str(y_pred.shape)),0)
	debug((  'y_test.shape: '+str(y_test.shape)),0)

	# In[14]:

	inv_y_pred = pca.inverse_transform(y_pred)
	debug((  'inv_y_pred.shape: '+str(inv_y_pred.shape)),0)
	inv_y_test = pca.inverse_transform(y_test)
	debug((  'inv_y_test.shape: '+str(inv_y_test.shape)),0)


	# In[15]:

	transformed_y_pred = scaler.inverse_transform(inv_y_pred)
	#original_y_test = scaler.inverse_transform(inv_y_test[n_dependencies:])
	original_y_test = scaler.inverse_transform(inv_y_test)
	debug((  'transformed_y_pred.shape: '+str(transformed_y_pred.shape)),1)
	debug((  'original_y_test.shape: '+str(original_y_test.shape)),1)


	# In[16]:

	debug((  'inverted :'+str( inv_y_pred[0].min())+str(inv_y_pred[0].max())),2)
	debug((  'transformed :'+str( transformed_y_pred[0].min())+str( transformed_y_pred[0].max())),2)
	debug((  'images.shape: '+str(images.shape)),2)
	debug((  'y_pred_size: '+str(y_pred_size)),0)

	# In[17]:

	pred_images = transformed_y_pred.reshape((y_pred_size, 100, 200))
	#original_image = original_y_test.reshape((y_pred_size, 100, 200))
	original_images = images[-y_pred_size:].reshape((y_pred_size, 100, 200))


	# In[18]:

	debug(('pred_images[0].shape: '  +str(pred_images[0].shape)),2)
	debug(( 'original_images[0].shape: ' +str(original_images[0].shape)),2)


	# In[19]:

	rmse_lds = np.asarray(sigma)
	debug(( 'rmse_lds.shape: ' +str(rmse_lds.shape)),1)
	debug((  'min rmse_lds:'+str( rmse_lds.min())),1)
	debug((  'max rmse_lds:'+str( rmse_lds.max())),1)
	debug((  'mean rmse_lds:'+str( rmse_lds.mean())),1)


	# In[25]:

	mmscalar = preprocessing.MinMaxScaler((0,255))
	images2 = []
	for pred_img in pred_images:
		images2.append(mmscalar.fit_transform(pred_img))
	output_images = np.asarray(images2).reshape((y_pred_size, 100, 200))
	debug((  'output images :'+str(output_images.min())+str( output_images.max())),1)
	debug((  'original images :'+str( original_images.min())+str( original_images.max())),1)


	# In[26]:

	rmse_hds_list = []
	for pred_img, orig_img in izip(output_images, original_images):
		rmse_hds_list.append(mean_squared_error(orig_img, pred_img)**0.5)
	rmse_hds = np.asarray(rmse_hds_list)
	debug(( 'rmse_hds.shape: ' +str(rmse_hds.shape)),0)
	debug((  'min rmse_hds:'+str( rmse_hds.min())),1)
	debug((  'max rmse_hds:'+str( rmse_hds.max())),1)
	debug((  'mean rmse_hds:'+str( rmse_hds.mean())),1)

#	with open('rmse_hds.csv','a') as csvfile:
#		csvwriter = csv.writer(csvfile, delimiter=',')
#		csvwriter.writerow(rmse_hds)
	# In[ ]:

	info = [n_dependencies, rmse_lds.min(), rmse_lds.max(), rmse_lds.mean(), rmse_hds.min(), rmse_hds.max(), rmse_hds.mean()]
	with open('rmse_info_analysis.csv','a') as csvfile:
		csvwriter = csv.writer(csvfile, delimiter=',')
		csvwriter.writerow(info)
#	error_ = abs(original_images - pred_images)
#	root_path='output_images/'+str(n_dependencies)
#	if not os.path.exists(root_path):
#		os.makedirs(root_path)
#	for i in range(y_pred_size):
#		fig = plt.figure()
#		a=fig.add_subplot(1,3,1)
#		orig = plt.imshow(original_images[i], cmap = cm.Greys_r)
#		a.set_title('Orig')
#		a=fig.add_subplot(1,3,2)
#		pred = plt.imshow(pred_images[i], cmap = cm.Greys_r)
#		a.set_title('Pred')
#		a=fig.add_subplot(1,3,3)
#		pred = plt.imshow(error_[i], cmap = cm.Greys_r)
#		a.set_title('Error')
#		fig.savefig((root_path+'/output_'+str(i)+'.png'))
#		fig.clear()
		    
