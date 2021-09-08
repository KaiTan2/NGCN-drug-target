# -*- coding: utf-8 -*-

import numpy as np
# from sets import Set
import pickle
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from utils2 import *
from tflearn.activations import relu
from sklearn.model_selection import train_test_split, StratifiedKFold
import sys
from optparse import OptionParser
from numpy import linalg as la
from sklearn.preprocessing import minmax_scale,scale,normalize

parser = OptionParser()
parser.add_option("-d", "--d", default=1024, help="The embedding dimension d")  # Data Dimensionality
parser.add_option("-n", "--n", default=1, help="global norm to be clipped") 
parser.add_option("-k", "--k", default=512, help="The dimension of project matrices k")  # The dimensions of the k
parser.add_option("-t", "--t", default="o", help="Test scenario")
parser.add_option("-r", "--r", default="ten", help="positive negative ratio")

(opts, args) = parser.parse_args()


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


def row_normalize(a_matrix, substract_self_loop):
    if substract_self_loop == True:
        np.fill_diagonal(a_matrix, 0)
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1) + 1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix
def construct_la(A):
    np.fill_diagonal(A,1)
    D = np.diag(np.sum(A,axis=1))
    v, Q = la.eig(D)
    # print(v)
    V = np.diag(v ** (-0.5))
    # print(V)
    T = Q * V * la.inv(Q)
    new_matrix = np.matmul(np.matmul(T,A),T)
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix

# load network
network_path = 'data/'

drug_drug = np.loadtxt('feature/Similarity_Drug.txt')
#drug_drug = drug_drug[:708,:708]
# print 'loaded drug chemical', check_symmetric(drug_chemical), np.shape(drug_chemical)

protein_protein = np.loadtxt('feature/Similarity_Protein.txt')
# print 'loaded protein sequence', check_symmetric(protein_sequence), np.shape(protein_sequence)

# normalize network for mean pooling aggregation 
#drug_drug = row_normalize(drug_drug, True)
drug_drug = normalize(drug_drug)
drug_drug = construct_la(drug_drug)
#protein_protein = row_normalize(protein_protein, True)
protein_protein = normalize(protein_protein)
protein_protein = construct_la(protein_protein)


drug = np.loadtxt('feature/drug4_vector_d200.txt') 
protein = np.loadtxt('feature/protein4_vector_d400.txt')

#drug = row_normalize(drug,False)
#protein = row_normalize(protein,False)
drug = scale(drug)
protein = scale(protein)
# define computation graph 
num_drug = len(drug_drug)  
num_protein = len(protein_protein)  

dim_drug = 200  
dim_protein = 400
dim_pred = 100
dim_pass = int(opts.d)


class Model(object):
    def __init__(self):
        self._build_model()

    def _build_model(self):
        # inputs
        self.drug_drug = tf.placeholder(tf.float32, [num_drug, num_drug])

        self.protein_protein = tf.placeholder(tf.float32, [num_protein, num_protein])

        self.drug_protein = tf.placeholder(tf.float32, [num_drug, num_protein])

        self.drug_protein_mask = tf.placeholder(tf.float32, [num_drug, num_protein])

        # features
        # 初始节点嵌入，通过随机数生成
        self.drug_embedding = tf.placeholder(tf.float32, [708, dim_drug])
        self.protein_embedding = tf.placeholder(tf.float32, [1512, dim_protein])

        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.drug_embedding))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.protein_embedding))

        # feature passing weights (maybe different types of nodes can use different weights)
        W0 = weight_variable([dim_drug, dim_drug])  # 
        W1 = weight_variable([dim_protein, dim_protein])
        W2 = weight_variable([dim_drug, dim_drug])  #  1024+1024 * 1024
        W3 = weight_variable([dim_protein, dim_protein])
        W4 = weight_variable([dim_drug, dim_drug])  #1024+1024 * 1024
        W5 = weight_variable([dim_protein, dim_protein])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W1))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W2))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W3))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W4))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W5))

      
        # passing 1 times (can be easily extended to multiple passes)
        drug_vector1 = tf.nn.l2_normalize(relu(tf.matmul(  
            tf.matmul(self.drug_drug,self.drug_embedding), W0)), dim=1) 
        drug_vector2 = tf.nn.l2_normalize(relu(tf.matmul(  
            drug_vector1, W2)), dim=1)
        drug_vector3 = tf.nn.l2_normalize(relu(tf.matmul(  
            drug_vector2, W4)), dim=1)
        protein_vector1 = tf.nn.l2_normalize(relu(tf.matmul(
            tf.matmul(self.protein_protein,self.protein_embedding), W1)), dim=1)  
        protein_vector2 = tf.nn.l2_normalize(relu(tf.matmul(
            protein_vector1, W3)), dim=1)
        protein_vector3 = tf.nn.l2_normalize(relu(tf.matmul(
            protein_vector2, W5)), dim=1)


        self.drug_representation = drug_vector1 
        self.protein_representation = protein_vector1 

        # reconstructing networks

        self.drug_protein_reconstruct = bi_layer(self.drug_representation, self.protein_representation, sym=False,
                                                 dim_pred=dim_pred)
        tmp = tf.multiply(self.drug_protein_mask,
                          (self.drug_protein_reconstruct - self.drug_protein)) 
        self.drug_protein_reconstruct_loss = tf.reduce_sum(tf.multiply(tmp, tmp))  

        # multiply    matmul

        self.l2_loss = tf.add_n(tf.get_collection("l2_reg"))

        self.loss = self.drug_protein_reconstruct_loss + self.l2_loss


graph = tf.get_default_graph()
with graph.as_default():
    model = Model()
    learning_rate = tf.placeholder(tf.float32, [])
    total_loss = model.loss
    dti_loss = model.drug_protein_reconstruct_loss

    optimize = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimize.compute_gradients(total_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, int(opts.n))
    optimizer = optimize.apply_gradients(
        zip(gradients, variables))

    eval_pred = model.drug_protein_reconstruct


def train_and_evaluate(DTItrain, DTIvalid, DTItest, graph, verbose=True, num_steps=4000):
    drug_protein = np.zeros((num_drug, num_protein))
    mask = np.zeros((num_drug, num_protein))
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1
    protein_drug = drug_protein.T

    drug_protein_normalize = row_normalize(drug_protein, False)
    protein_drug_normalize = row_normalize(protein_drug, False)

    lr = 0.001

    best_valid_aupr = 0
    best_valid_auc = 0
    test_aupr = 0
    test_auc = 0
    best_aupr = 0
    best_auc = 0
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()  # tf.global_variables_initializer  initialize_all_variables()
        for i in range(num_steps):
            _, tloss, dtiloss, results = sess.run([optimizer, total_loss, dti_loss, eval_pred],
                                                  feed_dict={model.drug_drug: drug_drug,
                                                             model.protein_protein:protein_protein,
                                                             model.drug_embedding:drug,
                                                             model.protein_embedding:protein,
                                                             model.drug_protein:drug_protein,
                                                             model.drug_protein_mask: mask,
                                                             learning_rate: lr})
            # every 25 steps of gradient descent, evaluate the performance, other choices of this number are possible
           
            if i % 25 == 0 and verbose == True:
                print('step', i, 'total and dtiloss', tloss, dtiloss)

                pred_list = []
                ground_truth = []
                for ele in DTIvalid:  
                    pred_list.append(results[ele[0], ele[1]])  
                    ground_truth.append(ele[2])
                print(pred_list)
                valid_auc = roc_auc_score(ground_truth, pred_list)  
                valid_aupr = average_precision_score(ground_truth, pred_list)  
                for ele in DTItest:
                    pred_list.append(results[ele[0], ele[1]])
                    ground_truth.append(ele[2])
                if valid_aupr >= best_valid_aupr:  
                    best_valid_aupr = valid_aupr  
                    best_valid_auc = valid_auc
                    pred_list = []
                    ground_truth = []
                    for ele in DTItest:
                        pred_list.append(results[ele[0], ele[1]])
                        ground_truth.append(ele[2])
                    test_auc = roc_auc_score(ground_truth, pred_list)
                    test_aupr = average_precision_score(ground_truth, pred_list)
                if test_aupr > best_aupr:
                    best_aupr = test_aupr
                print('valid auc aupr,', valid_auc, valid_aupr, 'test auc aupr', test_auc, test_aupr)
    return best_valid_auc, best_valid_aupr, test_auc, test_aupr,best_aupr


test_auc_round = []
test_aupr_round = []
best_aupr_round = []
for r in range(5):
    print('sample round', r + 1)
    if opts.t == 'o':  
        dti_o = np.loadtxt(network_path + 'mat_drug_protein_homo_protein_drug.txt')
    else:
        dti_o = np.loadtxt(network_path + 'mat_drug_protein_' + opts.t + '.txt')

    whole_positive_index = []  
    whole_negative_index = [] 
    for i in range(np.shape(dti_o)[0]): 
        for j in range(np.shape(dti_o)[1]):
            if int(dti_o[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(dti_o[i][j]) == 0:
                whole_negative_index.append([i, j])


        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),size=1 * len(whole_positive_index), replace=False)  # 10倍负样本
        #negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)), size=len(whole_negative_index),replace=False)



    data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
    count = 0
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1 
        count += 1
    for i in negative_sample_index:
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1

    if opts.t == 'unique':
        whole_positive_index_test = []
        whole_negative_index_test = []
        for i in range(np.shape(dti_o)[0]):
            for j in range(np.shape(dti_o)[1]):
                if int(dti_o[i][j]) == 3:
                    whole_positive_index_test.append([i, j])
                elif int(dti_o[i][j]) == 2:
                    whole_negative_index_test.append([i, j])

        if opts.r == 'ten':
            negative_sample_index_test = np.random.choice(np.arange(len(whole_negative_index_test)),
                                                          size=10 * len(whole_positive_index_test), replace=False)
        elif opts.r == 'all':
            negative_sample_index_test = np.random.choice(np.arange(len(whole_negative_index_test)),
                                                          size=whole_negative_index_test, replace=False)
        else:
            print('wrong positive negative ratio')
            break
        data_set_test = np.zeros((len(negative_sample_index_test) + len(whole_positive_index_test), 3), dtype=int)
        count = 0
        for i in whole_positive_index_test:
            data_set_test[count][0] = i[0]
            data_set_test[count][1] = i[1]
            data_set_test[count][2] = 1
            count += 1
        for i in negative_sample_index_test:
            data_set_test[count][0] = whole_negative_index_test[i][0]
            data_set_test[count][1] = whole_negative_index_test[i][1]
            data_set_test[count][2] = 0
            count += 1

        DTItrain = data_set
        DTItest = data_set_test
        rs = np.random.randint(0, 1000, 1)[0]
        DTItrain, DTIvalid = train_test_split(DTItrain, test_size=0.05,
                                              random_state=rs)
        v_auc, v_aupr, t_auc, t_aupr = train_and_evaluate(DTItrain=DTItrain, DTIvalid=DTIvalid, DTItest=DTItest,
                                                          graph=graph, num_steps=1000)  #

        test_auc_round.append(t_auc)
        test_aupr_round.append(t_aupr)

        np.savetxt('test_auc', test_auc_round)
        np.savetxt('test_aupr', test_aupr_round)

    else:
        test_auc_fold = []
        test_aupr_fold = []
        best_aupr_fold = []
        rs = np.random.randint(0, 1000, 1)[0]
        #kf = StratifiedKFold(data_set[:, 2], n_splits=10, shuffle=True, random_state=rs)
        kf = StratifiedKFold(n_splits=10,random_state=rs,shuffle=True)

        for train_index, test_index in kf.split(data_set[:,:2],data_set[:,2]):  # .split(data_set[:,2])
            DTItrain, DTItest = data_set[train_index], data_set[test_index]
            DTItrain, DTIvalid = train_test_split(DTItrain, test_size=0.05, random_state=rs)

            v_auc, v_aupr, t_auc, t_aupr, best_aupr = train_and_evaluate(DTItrain=DTItrain, DTIvalid=DTIvalid, DTItest=DTItest,
                                                              graph=graph, num_steps=1000)
            test_auc_fold.append(t_auc)
            test_aupr_fold.append(t_aupr)
            best_aupr_fold.append(best_aupr)
        test_auc_round.append(np.mean(test_auc_fold))
        test_aupr_round.append(np.mean(test_aupr_fold))
        best_aupr_round.append(np.mean(best_aupr_fold))
        np.savetxt('test_auc1', test_auc_round)
        np.savetxt('test_aupr1', test_aupr_round)
        np.savetxt('best_qupr1',best_aupr_round)
