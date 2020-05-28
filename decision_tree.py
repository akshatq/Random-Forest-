import numpy as np 
import random


class DecisionTree(object):
    def __init__(self, max_depth):
        # Initializing the tree as an empty dictionary or list, as preferred
        self.tree = {}
        self.max_depth = max_depth
        pass
    
    @staticmethod
    def entropy(class_y):

        np.seterr(divide='ignore', invalid='ignore')
        p=np.bincount(class_y)/sum(np.bincount(class_y))
        return sum(np.log2(p)*p*(-1))

    @staticmethod
    def partition_classes(X, y, split_attribute, split_val):
     
        X_left,X_right,y_left,y_right=[],[],[],[]
        
        left=[i for i in range(len(X)) if X[i][split_attribute] <= split_val]
        right=[i for i in range(len(X)) if  X[i][split_attribute] > split_val]  
        
        X_left=[X[i] for i in left]
        y_left=[y[i] for i in left]
        X_right=[X[i] for i in right]
        y_right=[y[i] for i in right]
        
        return (X_left, X_right, y_left, y_right)

    @staticmethod    
    def information_gain(previous_y, current_y):
        # Inputs:
        #   previous_y: the distribution of original labels (0's and 1's)
        #   current_y:  the distribution of labels after splitting based on a particular
        #               split attribute and split value
        
        info_gain = 0
    
        pl=len(current_y[0])/len(previous_y)
        pr=len(current_y[1])/len(previous_y)
        
        info_gain=DecisionTree.entropy(previous_y) - (DecisionTree.entropy(current_y[0])*pl + DecisionTree.entropy(current_y[1])*pr)
    
        return info_gain
        
    @staticmethod        
    def best_split(X, y):
        # Inputs:
        #   X       : Data containing all attributes in array format with numeric values only.
        #             For categorical variables please ensure to have one hot encoding done before hand
        #   y       : labels
    
    
        split_attribute = 0
        split_value = 0
        X_left, X_right, y_left, y_right = [], [], [], []
    
        col_len=len(X[0])
        cols=np.arange(len(X[0]))
        opti_attri=int(np.sqrt(col_len)+0.5)
        att_list=[]
        
        for i in range(opti_attri):
            q=random.choices(cols,k=1)
            att_list.append(q[0])
            cols= np.setdiff1d(cols,q)
    
    
        ig_list=[]
        split_val=[]
        
    
        for index1 in att_list:
            y_list=[]
            temp_list=[d[index1] for d in X]
            split_v=np.mean(temp_list)      
            X_left, X_right, y_left, y_right = DecisionTree.partition_classes(X, y, index1, split_v)
            split_val.append(split_v)
            y_list.append(y_left)
            y_list.append(y_right)
            ig=DecisionTree.information_gain(y, y_list)
            ig_list.append(ig)
          
        ind=ig_list.index(max(ig_list))    
        split_attribute=att_list[ind]
        split_value=split_val[ind]
        
        X_left, X_right, y_left, y_right = DecisionTree.partition_classes(X, y, split_attribute, split_value) 
        
        return split_attribute, split_value,X_left, X_right, y_left, y_right
          



	
    def learn(self, X, y, par_node = {}, depth=0):


        self.tree=par_node
                                 
        def most_frequent(List): 
            return max(set(List), key = List.count)
        
        def tree_split(node, depth=0):
            
            self.tree=par_node
            (X_left,y_left),(X_right,y_right)=node['groups']
            left, right = node['groups']
            del(node['groups'])
            
	# checking for a no split
            if y_left==[] or y_right==[]:
                node['left'] = node['right'] = most_frequent(y_left + y_right)
                return
            
	# checking for max depth
            if depth >= self.max_depth:
                node['left'], node['right'] = most_frequent(y_left), most_frequent(y_right)
                return
   
	# processing left key
            split_attribute, split_value,X_left1, X_right1, y_left1, y_right1=DecisionTree.best_split(X_left, y_left)
            data1=[(X_left1,y_left1),(X_right1,y_right1)]
            node['left']={'index':split_attribute, 'value':split_value, 'groups':data1}  
            #self.tree=par_node
            tree_split(node['left'], depth+1)

    # processing Right key
            split_attribute, split_value,X_left2, X_right2, y_left2, y_right2=DecisionTree.best_split(X_right, y_right)
            data2=[(X_left2,y_left2),(X_right2,y_right2)]
            node['right']={'index':split_attribute, 'value':split_value, 'groups':data2}
            #self.tree=par_node
            tree_split(node['right'], depth+1)

        
        split_attribute, split_value,X_left, X_right, y_left, y_right=DecisionTree.best_split(X, y)
        data=[(X_left,y_left),(X_right,y_right)]
        par_node={'index':split_attribute, 'value':split_value, 'groups':data}
        tree_split(par_node,depth=0)
        return            
	   

      
        #############################################

    def classify(self, record):

        tmp1=self.tree
        
        def cl1(tmp,record):
            if record[tmp['index']] < tmp['value']:
                if isinstance(tmp['left'], dict):
                    return cl1(tmp['left'],record)
                else:
                    return tmp['left']
            else:
                if isinstance(tmp['right'], dict):
                    return cl1(tmp['right'],record)
                else:
                    return tmp['right']
        
        return cl1(tmp1,record)
        #############################################
















