# entire code has been implemented by ourselves in python 2.7

import pickle
import numpy as np
from scipy import spatial
import tensorflow as tf
import math
import tensorflow_func as tf_func
from nltk.tokenize import word_tokenize
import copy
dictionary, embeddings = pickle.load(open('word2vec_relation_extracted_word_sentences.model', 'rb'))

embedding_size = 50
entitity_1 = []
entity_2 = []
sentence_index = []
relation =[]
count = 0
with open('relation_extracted_file_50001_100000.txt') as f:
    for line in f:
        sentence = line.split("\t")
        entitity_1.append(sentence[0])
        entity_2.append(sentence[1])
        relation.append(sentence[2])
        count += 1
        if count % 10000 == 0:
            break
        
#Store  all the entity1 embeddings in a list called embeddings_array_entity1
embeddings_array_entity1 = []
embeddings_array = []
for words in entitity_1:
    entity_words = word_tokenize(words.decode('utf-8'))
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
    embeddings_array_entity1.append(ans)
    embeddings_array = []

#Store all the entity2 embeddings in a list called embeddings_array_entity2    
embeddings_array_entity2 = []
embeddings_array = []
for words in entity_2:
    entity_words = word_tokenize(words.decode('utf-8'))
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
    embeddings_array_entity2.append(ans)
    embeddings_array = []

#Store all the relation embeddings in a list called embeddings_array_relation    
embeddings_array_relation = []
embeddings_array = []
for words in relation:
    entity_words = word_tokenize(words.decode('utf-8'))
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
    embeddings_array_relation.append(ans)
    embeddings_array = []
    
print(len(embeddings_array_entity1) , len(embeddings_array_entity2) , len(embeddings_array_relation) , len(sentence_index)) 

embeddings_relation = copy.deepcopy(embeddings_array_relation)
embeddings_entity2 = copy.deepcopy(embeddings_array_entity2)
embeddings_entity1 = copy.deepcopy(embeddings_array_entity1)

# Form an actual label matrix which would be a 1 hot vector of dimension n * |S| where |S| is the no. of words in the vocab. Take w2 as tf.Variable of dimension 1 * |S|. 
# Then multiply above two to get n * |S|. Run softmax
words_dict = []
count = 0
with open('relation_extracted_file_sentences_50001_100000.txt') as f:
    for line in f:
        words_temp = word_tokenize(line.decode('utf-8'))
        for i in words_temp:
            if( i in words_dict):
                continue
            else :
                words_dict.append(i)
        count += 1
        if count % 10000 == 0:
            break
#print("Words length is " , len(words_dict))

one_hot_vector = [ 0 for i in range(len(words_dict))]
one_hot_vector_list = [one_hot_vector for i in range(len(embeddings_array_relation))] #Shape is 5000 * 4074(unique words)
#print("one hot vector is" , one_hot_vector)
#print("ONe hot vector list is" , one_hot_vector_list[0][5])
count = 0
with open('relation_extracted_file_sentences_50001_100000.txt') as f:
    for line in  f:
        sentence = line.split("\t")
        words = word_tokenize(sentence[0].decode('utf-8'))
        for word in words:
            word_dict_id = words_dict.index(word)
            if(one_hot_vector_list[count][int(word_dict_id)] != 1):
                one_hot_vector_list[count][int(word_dict_id)] = 1
        count = count + 1
        if count % 10000 == 0:
            break

print("Length of one_hot_vector_lsit is" , len(one_hot_vector_list) , len(one_hot_vector_list[0]))

graph = tf.Graph()
batch_size = 5000

with graph.as_default():

    relations_embedding_tensor = tf.placeholder(tf.float32 , shape = [batch_size , embedding_size])
    entity1_embedding_tensor = tf.placeholder(tf.float32 , shape = [batch_size , embedding_size])
    entity2_embedding_tensor = tf.placeholder(tf.float32 , shape = [batch_size , embedding_size])
    labels_tensor = tf.placeholder(tf.float32 , shape = [batch_size , len(one_hot_vector_list[0])])
    
    weights = tf.Variable(
            tf.random_normal([embedding_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)),name = "weights")
    
    #Code adopted from stackoverflow for repeating matrix elements
    jdx = tf.range(embedding_size)
    jdx = tf.tile(jdx , [batch_size]) #Goes from 0,1,2...127 and repeats again as 0,1,2...127 till the len(embeddings_array_relation) ie 727 times in total
    
    idx = tf.range(batch_size)
    idx = tf.reshape(idx , [-1,1])
    idx = tf.tile( idx , [1,embedding_size])
    idx = tf.reshape(idx , [-1]) # Goes from 0,0,0...128 times followed by 1,1,...128 times and so on till the (len(embeddings_array_relation) i.e 727 times in total
    
    new_weights_matrix = tf.gather(weights , jdx) 
    new_weights_matrix = tf.transpose(new_weights_matrix)
    new_embedding_dimension_matrix = tf.gather(relations_embedding_tensor , idx)
    
    print("New embedding dimension matrix is/" , new_embedding_dimension_matrix.get_shape() , new_weights_matrix.get_shape())
    #Code adopted from stackoverflow for batch_matmul
    new_weights_transpose = tf.transpose(new_weights_matrix)
    new_weights_transpose_as_matrix_batch = tf.expand_dims(new_weights_transpose,2)
    
    print(new_embedding_dimension_matrix.get_shape())
    
    print(new_weights_transpose_as_matrix_batch.get_shape())
    new_embeddings_dimension_matrix_batch = tf.expand_dims(new_embedding_dimension_matrix , 1)
    print(new_embeddings_dimension_matrix_batch.get_shape())
    result = tf.matmul(new_weights_transpose_as_matrix_batch , new_embeddings_dimension_matrix_batch)
    print("Shape is" , result.get_shape())
    
    #Compute the V matrix 
    V = tf.matmul(result , new_weights_transpose_as_matrix_batch) #dimension : nd * d * 1
    print("V shape is" , V.get_shape())
    V = tf.reshape(V , [ batch_size , embedding_size , embedding_size])
    
    # e1 -> reshape to (n , 1 , d). then do batch_matmul e1 and V to get final tensor of shape( n , 1 , d). Reshape e2 into ( n , d, 1). batch_mul of the final tensor of shape( n , 1, d) and e2. Final tensor would be of shape n * 1 * 1. reshape to n * 1. 
    
    entity1_embedding = tf.reshape(entity1_embedding_tensor , [batch_size , 1 , embedding_size ])
    entity1_multiply_V = tf.matmul( entity1_embedding , V)
    print(entity1_multiply_V.get_shape())
    entity2_embedding = tf.reshape(entity2_embedding_tensor , [batch_size , embedding_size , 1])
    entity1_V_entity2transpose = tf.matmul( entity1_multiply_V , entity2_embedding)
    entity1_V_entity2transpose = tf.reshape(entity1_V_entity2transpose , [batch_size , 1])
    print("Resultant shape is " , entity1_V_entity2transpose)
    
    
    # Initialize a random weight of dimension 1 * |S|
    w2 = tf.Variable(
            tf.random_normal([1, len(words_dict)],
                                stddev=1.0 / len(words_dict)),name = "w2")
    
    b = tf.Variable(tf.zeros([len(words_dict)]),name = 'bias')
    
    
    resultant_matrix = tf.matmul(entity1_V_entity2transpose , w2) + b
    
    #Calcuate the loss
#     softmax = tf.nn.softmax(resultant_matrix)
    loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = resultant_matrix, labels = tf.to_float(labels_tensor)),1))
#     loss = tf.reduce_mean(tf_func.cross_entropy_loss( tf.to_float(softmax) , tf.to_float(one_hot_vector_list)))
    optimizer = tf.train.GradientDescentOptimizer(0.1)    
    grads = optimizer.compute_gradients(loss)
        # Gradient Clipping
    clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
    app = optimizer.apply_gradients(clipped_grads)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


num_steps = 1001
with tf.Session(graph=graph) as sess:
    init.run()
    average_loss = 0
    for step in range(num_steps):
        print("Starting step :", step)
        start = (step*5000)%len(embeddings_relation)
        end = ((step+1)*5000)%len(embeddings_relation)
        if end < start:
            start -= end
            end = len(embeddings_relation)
        batch_relation, batch_entity1, batch_entity2, batch_labels = embeddings_relation[start:end], embeddings_entity1[start:end], embeddings_entity2[start:end], one_hot_vector_list[start:end]
        feed_dict = {relations_embedding_tensor : batch_relation , entity1_embedding_tensor :batch_entity1, entity2_embedding_tensor : batch_entity2, labels_tensor: batch_labels}
        _ , loss_val = sess.run([app , loss], feed_dict=feed_dict)
        print("Computing loss")
        print("weights is " , sess.run(weights))
        average_loss += loss_val
        if ( step % 5 == 0 ):
            if ( step > 0 ):
                average_loss /= 5
                print("Average loss at step : " , step , ":" , average_loss)
            average_loss = 0

    np.savetxt('results_weights.txt',weights.eval())
    np.savetxt('result_W2.txt',w2.eval())
    np.savetxt('results_bias.txt',b.eval())





