# entire code has been implemented in python 2.7

import pickle
import numpy as np
from scipy import spatial
import tensorflow as tf
import math
import tensorflow_func as tf_func
import copy
from sklearn.metrics.pairwise import cosine_similarity
dictionary, embeddings = pickle.load(open('word2vec_relation_extracted_word_sentences.model', 'rb'))

entity_1_A = []
entity_2_A = []
relation_1_A = []
entity_1_B = []
entity_2_B = []
relation_1_B = []
embedding_size = 50

with open('emnlp2013_ml.txt') as f:
    for line in f:
        sentence = line.split()
        entity_1_A.append(sentence[1])
	entity_2_A.append(sentence[3])
	relation_1_A.append(sentence[2])
	entity_1_B.append(sentence[4])
	entity_2_B.append(sentence[6])
	relation_1_B.append(sentence[5])
		
embeddings_array_entity1_A = []
embeddings_array = []
for words in entity_1_A:
    entity_words = words.split(" ")
    for word in entity_words:
        try:
		word_id = dictionary[word]
	except KeyError:
		print("Error")
		word_id = dictionary['the']
        v1 = embeddings[word_id]
	embeddings_array.append(v1)
    if(len(embeddings_array) > 1):
		ans = np.mean(embeddings_array , axis = 0 )
    else:
		ans = embeddings_array[0]	
    embeddings_array_entity1_A.append(ans)
    embeddings_array = []
	
embeddings_array_entity2_A = []
embeddings_array = []
for words in entity_2_A:
    entity_words = words.split(" ")
    for word in entity_words:
	try:
	        word_id = dictionary[word]
	except KeyError:
		word_id = dictionary['the']
		print("Error")
	v1 = embeddings[word_id]
	embeddings_array.append(v1)
    if(len(embeddings_array) > 1):
		ans = np.mean(embeddings_array , axis = 0 )
    else:
		ans = embeddings_array[0]	
    embeddings_array_entity2_A.append(ans)
    embeddings_array = []
	
embeddings_array_relation_1_A = []
embeddings_array = []
for words in relation_1_A:
    entity_words = words.split(" ")
    for word in entity_words:
	try:
	        word_id = dictionary[word]
	except KeyError:
		print("Error")
		word_id = dictionary['the']
	v1 = embeddings[word_id]
	embeddings_array.append(v1)
    if(len(embeddings_array) > 1):
		ans = np.mean(embeddings_array , axis = 0 )
    else:
		ans = embeddings_array[0]	
    embeddings_array_relation_1_A.append(ans)
    embeddings_array = []
	
embeddings_array_entity1_B = []
embeddings_array = []
for words in entity_1_B:
    entity_words = words.split(" ")
    for word in entity_words:
	try:
	        word_id = dictionary[word]
	except KeyError:
		print("Error")
		word_id = dictionary['the']
	v1 = embeddings[word_id]
	embeddings_array.append(v1)
    if(len(embeddings_array) > 1):
		ans = np.mean(embeddings_array , axis = 0 )
    else:
		ans = embeddings_array[0]	
    embeddings_array_entity1_B.append(ans)
    embeddings_array = []
	
embeddings_array_entity2_B = []
embeddings_array = []
for words in entity_2_B:
    entity_words = words.split(" ")
    for word in entity_words:
	try:
	        word_id = dictionary[word]
	except KeyError:
		print("Error")
		word_id = dictionary['the']
	v1 = embeddings[word_id]
	embeddings_array.append(v1)
    if(len(embeddings_array) > 1):
		ans = np.mean(embeddings_array , axis = 0 )
    else:
		ans = embeddings_array[0]	
    embeddings_array_entity2_B.append(ans)
    embeddings_array = []
	
embeddings_array_relation_1_B = []
embeddings_array = []
for words in relation_1_B:
    entity_words = words.split(" ")
    for word in entity_words:
	try:
	        word_id = dictionary[word]
	except KeyError:
		print("Error")
		word_id = dictionary['the']
	v1 = embeddings[word_id]
      	embeddings_array.append(v1)
    if(len(embeddings_array) > 1):
		ans = np.mean(embeddings_array , axis = 0 )
    else:
		ans = embeddings_array[0]	
    embeddings_array_relation_1_B.append(ans)
    embeddings_array = []

embeddings_relation = copy.deepcopy(embeddings_array_relation_1_A)
embeddings_entity2 = copy.deepcopy(embeddings_array_entity2_A)
embeddings_entity1 = copy.deepcopy(embeddings_array_entity1_A)

embeddings_relation_B = copy.deepcopy(embeddings_array_relation_1_B)
embeddings_entity2_B = copy.deepcopy(embeddings_array_entity2_B)
embeddings_entity1_B = copy.deepcopy(embeddings_array_entity1_B)
graph = tf.Graph()

words_dict = [0 for i in range(4074)]

with graph.as_default():

    relations_embedding_tensor_A = tf.placeholder(tf.float32 , shape = [len(embeddings_relation) , embedding_size])
    entity1_embedding_tensor_A = tf.placeholder(tf.float32 , shape = [len(embeddings_relation) , embedding_size])
    entity2_embedding_tensor_A = tf.placeholder(tf.float32 , shape = [len(embeddings_relation) , embedding_size])
    
    weights = tf.stack(np.loadtxt('results_weights.txt'))

    #Code adopted from stackoverflow for repeating matrix elements
    jdx = tf.range(embedding_size)
    jdx = tf.tile(jdx , [len(embeddings_array_relation_1_A)]) #Goes from 0,1,2...127 and repeats again as 0,1,2...127 till the len(embeddings_array_relation) ie 727 times in total
    
    idx = tf.range(len(embeddings_array_relation_1_A))
    idx = tf.reshape(idx , [-1,1])
    idx = tf.tile( idx , [1,embedding_size])
    idx = tf.reshape(idx , [-1]) # Goes from 0,0,0...128 times followed by 1,1,...128 times and so on till the (len(embeddings_array_relation) i.e 727 times in total
    
    new_weights_matrix = tf.gather(weights , jdx) 
    new_weights_matrix = tf.transpose(new_weights_matrix)
    new_embedding_dimension_matrix = tf.gather(relations_embedding_tensor_A , idx)
    
    print("New embedding dimension matrix is/" , new_embedding_dimension_matrix.get_shape() , new_weights_matrix.get_shape())
    #Code adopted from stackoverflow for batch_matmul
    new_weights_transpose = tf.transpose(new_weights_matrix)
    new_weights_transpose_as_matrix_batch = tf.expand_dims(new_weights_transpose,2)
    
    print(new_embedding_dimension_matrix.get_shape())
    
    print(new_weights_transpose_as_matrix_batch.get_shape())
    new_embeddings_dimension_matrix_batch = tf.expand_dims(new_embedding_dimension_matrix , 1)
    print(new_embeddings_dimension_matrix_batch.get_shape())
    result = tf.matmul(tf.to_float(new_weights_transpose_as_matrix_batch) , tf.to_float(new_embeddings_dimension_matrix_batch))
    print("Shape is" , result.get_shape())
    
    #Compute the V matrix 
    V = tf.matmul(result , tf.to_float(new_weights_transpose_as_matrix_batch)) #dimension : nd * d * 1
    print("V shape is" , V.get_shape())
    V = tf.reshape(V , [ len(embeddings_array_relation_1_A) , embedding_size , embedding_size])
    
    # e1 -> reshape to (n , 1 , d). then do batch_matmul e1 and V to get final tensor of shape( n , 1 , d). Reshape e2 into ( n , d, 1). batch_mul of the final tensor of shape( n , 1, d) and e2. Final tensor would be of shape n * 1 * 1. reshape to n * 1. 
    
    entity1_embedding = tf.reshape(entity1_embedding_tensor_A , [len(embeddings_array_relation_1_A) , 1 , embedding_size ])
    entity1_multiply_V = tf.matmul( entity1_embedding , V)
    print(entity1_multiply_V.get_shape())
    entity2_embedding = tf.reshape(entity2_embedding_tensor_A , [len(embeddings_array_relation_1_A) , embedding_size , 1])
    entity1_V_entity2transpose = tf.matmul( entity1_multiply_V , entity2_embedding)
    entity1_V_entity2transpose = tf.reshape(entity1_V_entity2transpose , [len(embeddings_array_relation_1_A) , 1])
    print("Resultant shape is " , entity1_V_entity2transpose)
    
    # Initialize a random weight of dimension 1 * |S|
    w2 = np.loadtxt('result_W2.txt')
    w2 = tf.reshape(w2, [1 , len(w2)])
     
    b = tf.stack(np.loadtxt('results_bias.txt'))
    
#     print w2.get_shape()
#     resultant_matrix_A = tf.matmul(tf.to_float(entity1_V_entity2transpose) , tf.to_float(w2)) + tf.to_float(b)
#     print("Resultant_matrix_A shape " , resultant_matrix_A.get_shape())
#     resultant_matrix_A = tf.nn.sigmoid(resultant_matrix_A)
    
    # Do for tensor B
    relations_embedding_tensor_B = tf.placeholder(tf.float32 , shape = [len(embeddings_relation_B) , embedding_size])
    entity1_embedding_tensor_B = tf.placeholder(tf.float32 , shape = [len(embeddings_relation_B) , embedding_size])
    entity2_embedding_tensor_B = tf.placeholder(tf.float32 , shape = [len(embeddings_relation_B) , embedding_size])
    
    #Code adopted from stackoverflow link shared by Akshay for repeating matrix elements
    jdx_B = tf.range(embedding_size)
    jdx_B = tf.tile(jdx_B , [len(embeddings_array_relation_1_B)]) #Goes from 0,1,2...127 and repeats again as 0,1,2...127 till the len(embeddings_array_relation) ie 727 times in total
    
    idx_B = tf.range(len(embeddings_array_relation_1_B))
    idx_B = tf.reshape(idx_B , [-1,1])
    idx_B = tf.tile( idx_B , [1,embedding_size])
    idx_B = tf.reshape(idx_B , [-1]) # Goes from 0,0,0...128 times followed by 1,1,...128 times and so on till the (len(embeddings_array_relation) i.e 727 times in total
    
    new_weights_matrix_B = tf.gather(weights , jdx_B) 
    new_weights_matrix_B = tf.transpose(new_weights_matrix_B)
    new_embedding_dimension_matrix_B = tf.gather(relations_embedding_tensor_B , idx_B)
    
    print("New embedding dimension matrix is/" , new_embedding_dimension_matrix_B.get_shape() , new_weights_matrix_B.get_shape())
    #Code adopted from stackoverflow link shared by Akshay for batch_matmul
    new_weights_transpose_B = tf.transpose(new_weights_matrix_B)
    new_weights_transpose_as_matrix_batch_B = tf.expand_dims(new_weights_transpose_B,2)
    
    print(new_embedding_dimension_matrix_B.get_shape())
    
    print(new_weights_transpose_as_matrix_batch_B.get_shape())
    new_embeddings_dimension_matrix_batch_B = tf.expand_dims(new_embedding_dimension_matrix_B , 1)
    print(new_embeddings_dimension_matrix_batch_B.get_shape())
    result_B = tf.matmul(tf.to_float(new_weights_transpose_as_matrix_batch_B) , tf.to_float(new_embeddings_dimension_matrix_batch_B))
    print("Shape is" , result_B.get_shape())
    
    #Compute the V matrix 
    V_B = tf.matmul(result_B , tf.to_float(new_weights_transpose_as_matrix_batch_B)) #dimension : nd * d * 1
    print("V shape is" , V_B.get_shape())
    V_B = tf.reshape(V_B , [ len(embeddings_array_relation_1_B) , embedding_size , embedding_size])
    
    # e1 -> reshape to (n , 1 , d). then do batch_matmul e1 and V to get final tensor of shape( n , 1 , d). Reshape e2 into ( n , d, 1). batch_mul of the final tensor of shape( n , 1, d) and e2. Final tensor would be of shape n * 1 * 1. reshape to n * 1. 
    
    entity1_embedding_B = tf.reshape(entity1_embedding_tensor_B , [len(embeddings_array_relation_1_B) , 1 , embedding_size ])
    entity1_multiply_V_B = tf.matmul( entity1_embedding_B , V_B)
    print(entity1_multiply_V_B.get_shape())
    entity2_embedding_B = tf.reshape(entity2_embedding_tensor_B , [len(embeddings_array_relation_1_B) , embedding_size , 1])
    entity1_V_entity2transpose_B = tf.matmul( entity1_multiply_V_B , entity2_embedding_B)
    entity1_V_entity2transpose_B = tf.reshape(entity1_V_entity2transpose_B , [len(embeddings_array_relation_1_B) , 1])
    print("Resultant shape is " , entity1_V_entity2transpose_B)
    
    init = tf.global_variables_initializer()

with tf.Session(graph=graph) as sess:
    init.run()
    feed_dict = {relations_embedding_tensor_A : embeddings_relation , entity1_embedding_tensor_A :embeddings_entity1, entity2_embedding_tensor_A : embeddings_entity2, relations_embedding_tensor_B : embeddings_relation_B , entity1_embedding_tensor_B :embeddings_entity1_B, entity2_embedding_tensor_B : embeddings_entity2_B}
    A,B = sess.run([entity1_V_entity2transpose,entity1_V_entity2transpose_B],feed_dict=feed_dict)

    similarity = cosine_similarity(A,B)
    print len(similarity)
    print len(similarity[0])
    similarity = np.diagonal(similarity)
    print similarity

    score = (similarity > 0.5).sum()
    
    score /= 1.0 * len(similarity)
    print("score",score)

