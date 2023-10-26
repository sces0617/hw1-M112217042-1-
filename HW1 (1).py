#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re, csv, numpy as np, sys


# In[2]:


i = 0

with open('adult/adult1.test','w', encoding='utf-8') as csvfile:
    
    with open('adult/adult.test', 'r', newline='') as filein:
        for line in filein:
            flage = 0
            
            for i in range(0,len(line)):
                if(line[i] == '?'):
                    flage+=1
            
            if flage == 0:
                csvfile.write(line)
                #print(line)


# In[3]:


import pandas as pd
import sklearn.datasets as datasets
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer


# In[4]:


adult = pd.read_csv('adult/adult1.data', header=None)
adult_test = pd.read_csv('adult/adult1.test', skiprows=[0], header=None)

x = adult.iloc[:,:-1]
y = adult.iloc[:,-1:]

test_x = adult_test.iloc[:,:-1]
test_y = adult_test.iloc[:,-1:]

#print(x, y, test_x, test_y)


# In[5]:


from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore')
    
ohe.fit(x)

X_train_ohe = ohe.transform(x).toarray()
test_X_ohe = ohe.transform(test_x).toarray()
print(test_X_ohe)


# In[6]:


from sklearn.tree._tree import TREE_LEAF

def prune_index(inner_tree, index, threshold):
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are shildren, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)


# In[7]:


import matplotlib.pyplot as plt


# In[7]:


ID3_tree = tree.DecisionTreeClassifier(random_state = 0, criterion='entropy',
                                       max_depth=5).fit(X_train_ohe,y)

fig, axes = plt.subplots(nrows = 1,ncols = 1, figsize = (3,3), dpi=1000)
tree.plot_tree(ID3_tree, class_names=np.unique(y).astype('str'),filled = True)
#plt.savefig('tree.png', format='png', bbox_inches = "tight")
plt.show()


# In[15]:


cart_tree = tree.DecisionTreeClassifier(random_state = 5, criterion='gini', splitter = 'best',ccp_alpha = 0.0006, max_leaf_nodes=44,
                                        max_depth = 20).fit(X_train_ohe, y)   #, min_samples_leaf = .0015, min_samples_split = .001

fig, axes = plt.subplots(nrows = 1,ncols = 1, figsize = (3,3), dpi=1000)
tree.plot_tree(cart_tree, class_names=np.unique(y).astype('str'),filled = True)
#plt.savefig('tree.png', format='png', bbox_inches = "tight")
plt.show()


# In[15]:


C45_tree = tree.DecisionTreeClassifier( criterion='entropy', min_samples_split = 10, 
                                      max_depth = 5, min_samples_leaf=12).fit(X_train_ohe, y) 

#path = C45_tree.cost_complexity_pruning_path(X_train_ohe, y)
fig, axes = plt.subplots(nrows = 1,ncols = 1, figsize = (3,3), dpi=1000)
tree.plot_tree(C45_tree, class_names=np.unique(y).astype('str'),filled = True)
#plt.savefig('tree.png', format='png', bbox_inches = "tight")
plt.show()


# In[ ]:





# In[34]:


C50_tree = tree.DecisionTreeClassifier(random_state = 6, criterion='entropy', min_impurity_decrease=.01,
                                       min_samples_split = .05).fit(X_train_ohe, y) #, max_depth = 5  max_leaf_nodes = 10,, min_samples_leaf=.01
fig, axes = plt.subplots(nrows = 1,ncols = 1, figsize = (3,3), dpi=1000)
tree.plot_tree(C50_tree, class_names=np.unique(y).astype('str'),filled = True)
#plt.savefig('tree.png', format='png', bbox_inches = "tight")
plt.show()


# In[9]:


from sklearn import metrics
from sklearn.metrics import classification_report


# In[133]:


train_y_ID3 = ID3_tree.predict(X_train_ohe)
text_y_ID3 = ID3_tree.predict(test_X_ohe)
accuracy1 = metrics.accuracy_score(y,train_y_ID3)
accuracy = metrics.accuracy_score(test_y, text_y_ID3)

print(accuracy1)
print(accuracy)
print(classification_report(test_y, text_y_ID3))


# In[99]:





# In[16]:


train_y_cart = cart_tree.predict(X_train_ohe)
text_y_cart = cart_tree.predict(test_X_ohe)
accuracy1 = metrics.accuracy_score(y,train_y_cart)
accuracy = metrics.accuracy_score(test_y, text_y_cart)
print(accuracy1)
print(accuracy)
print(classification_report(test_y, text_y_cart))


# In[35]:


train_y_C50 = C50_tree.predict(X_train_ohe)
text_y_C50 = C50_tree.predict(test_X_ohe)
accuracy1 = metrics.accuracy_score(y,train_y_C50)
accuracy = metrics.accuracy_score(test_y, text_y_C50)
print(accuracy1)
print(accuracy)
print(classification_report(test_y, text_y_C50))


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


fig, ax = plt.subplots()
clf = tree.DecisionTreeClassifier(random_state=0, criterion='gini', splitter = 'best')
path = clf.cost_complexity_pruning_path(X_train_ohe, y)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")


# In[ ]:


clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train_ohe, y)
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(clfs[-1].tree_.node_count, ccp_alphas[-1]))


# In[ ]:


clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()


# In[ ]:





# In[ ]:




