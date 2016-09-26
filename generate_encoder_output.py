# -*- coding: utf-8 -*-
"""
Created on Wed Mar 9 11:32:21 2016

@author: Bing Liu (liubing@cmu.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import control_flow_ops
import tensorflow as tf

def get_multilayer_perceptron_regularizers(output_projection):
    weight_hidden, bias_hidden, weight_out, bias_out = output_projection
    regularizers = tf.nn.l2_loss(weight_hidden) + tf.nn.l2_loss(bias_hidden) + tf.nn.l2_loss(weight_out) + tf.nn.l2_loss(bias_out) 
    return regularizers
    
def multilayer_perceptron_with_initialized_W(_X, output_projection, forward_only=False):
    #with variable_scope.variable_scope("intent_MLP"):
    weight_hidden, bias_hidden, weight_out, bias_out = output_projection
      #Hidden layer with RELU activation
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, weight_hidden), bias_hidden))

    if not forward_only:
        layer_1 = tf.nn.dropout(layer_1, 0.5)
                                                  
    output = tf.matmul(layer_1, weight_out) + bias_out
      #regularizers = tf.nn.l2_loss(weight_hidden) + tf.nn.l2_loss(bias_hidden) + tf.nn.l2_loss(weight_out) + tf.nn.l2_loss(bias_out) 
    return output

def get_linear_transformation_regularizers(output_projection):
    weight_out, bias_out = output_projection
    regularizers = tf.nn.l2_loss(weight_out) + tf.nn.l2_loss(bias_out) 
    return regularizers
    
def linear_transformation_with_initialized_W(_X, output_projection, forward_only=False):
    #with variable_scope.variable_scope("intent_MLP"):
    weight_out, bias_out = output_projection                                                  
    output = tf.matmul(_X, weight_out) + bias_out
    return output

def _extract_argmax_and_embed(embedding, DNN_at_output, output_projection, forward_only=False, update_embedding=True):
  def loop_function(prev, _):
    if DNN_at_output is True:
      prev = multilayer_perceptron_with_initialized_W(prev, output_projection, forward_only=forward_only)
    else:
      prev = linear_transformation_with_initialized_W(prev, output_projection, forward_only=forward_only)
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    return prev, emb_prev
  return loop_function

def rnn_with_output_feedback(cell, inputs, 
                             targets1, targets1_num_symbols, target1_emb_size, target1_output_projection, 
                             targets2, targets2_num_symbols, target2_emb_size, target2_output_projection, 
                             word_emb_size, DNN_at_output, 
                             zero_intent_thres=0,
                             sequence_length=None, dtype=None, 
                             train_with_true_label=True, use_predicted_output=False):
  '''
  zero_intent_thres:  int, the intent contribution to context remain zero before this thres, 
                      and linear increase to 1 after that.
  '''
  if not isinstance(cell, rnn_cell.RNNCell):
    raise TypeError("cell must be an instance of RNNCell")
  if not isinstance(inputs, list):
    raise TypeError("inputs must be a list")
  if not isinstance(targets1, list):
    raise TypeError("targets1 must be a list")
  if not isinstance(targets2, list):
    raise TypeError("targets2 must be a list")
  if not inputs:
    raise ValueError("inputs must not be empty")  
  if not dtype:
    raise ValueError("dtype must be provided, which is to used in defining intial RNN state")
    
  encoder_outputs = []
  intent_embedding = variable_scope.get_variable("intent_embedding",[targets1_num_symbols, target1_emb_size])
  tag_embedding = variable_scope.get_variable("tag_embedding",[targets2_num_symbols, target2_emb_size])
  # use predicted label if use_predicted_output during inference, use true label during training
  # To choose to always use predicted label, disable the if condition
  intent_loop_function = _extract_argmax_and_embed(intent_embedding, DNN_at_output, target1_output_projection, forward_only=use_predicted_output) #if use_predicted_output else None
  tagging_loop_function = _extract_argmax_and_embed(tag_embedding, DNN_at_output, target2_output_projection, forward_only=use_predicted_output)
  intent_targets = [array_ops.reshape(math_ops.to_int64(x), [-1]) for x in targets1]   
  intent_target_embeddings = list()
  intent_target_embeddings = [embedding_ops.embedding_lookup(intent_embedding, target) for target in intent_targets]  
  tag_targets = [array_ops.reshape(math_ops.to_int64(x), [-1]) for x in targets2]   
  tag_target_embeddings = list()
  tag_target_embeddings = [embedding_ops.embedding_lookup(tag_embedding, target) for target in tag_targets]  
  
  if inputs[0].get_shape().ndims != 1:
    (fixed_batch_size, input_size) = inputs[0].get_shape().with_rank(2)
    if input_size.value is None:
      raise ValueError(
            "Input size (second dimension of inputs[0]) must be accessible via "
            "shape inference, but saw value None.")
  else:
    fixed_batch_size = inputs[0].get_shape().with_rank_at_least(1)[0]

  if fixed_batch_size.value:
    batch_size = fixed_batch_size.value
  else:
    batch_size = array_ops.shape(inputs[0])[0]

  state = cell.zero_state(batch_size, dtype)
  zero_output = array_ops.zeros(
      array_ops.pack([batch_size, cell.output_size]), inputs[0].dtype)
  zero_output.set_shape(
      tensor_shape.TensorShape([fixed_batch_size.value, cell.output_size]))

  if sequence_length is not None:  # Prepare variables
    sequence_length = math_ops.to_int32(sequence_length)
    min_sequence_length = math_ops.reduce_min(sequence_length)
    max_sequence_length = math_ops.reduce_max(sequence_length)

#  prev_cell_output = zero_output
  zero_intent_embedding = array_ops.zeros(
        array_ops.pack([batch_size, target1_emb_size]), inputs[0].dtype)
  zero_intent_embedding.set_shape(
          tensor_shape.TensorShape([fixed_batch_size.value, target1_emb_size])) 
  zero_tag_embedding = array_ops.zeros(
        array_ops.pack([batch_size, target2_emb_size]), inputs[0].dtype)
  zero_tag_embedding.set_shape(
          tensor_shape.TensorShape([fixed_batch_size.value, target2_emb_size])) 

  encoder_outputs = list() 
  intent_logits = list()
  tagging_logits = list()
  sampled_intent_embeddings = list()             
  sampled_tag_embeddings = list()                                             
        
  for time, input_ in enumerate(inputs):
    # Bing: introduce output label embeddings as addtional input
    # if feed_previous (during testing):
    #     Use loop_function
    # if NOT feed_previous (during training):
    #     Use true target embedding
    if time == 0:
      current_intent_embedding = zero_intent_embedding
      current_tag_embedding = zero_tag_embedding


    if time > 0: variable_scope.get_variable_scope().reuse_variables()
        
    # here we introduce a max(0, t-4)/sequence_length intent weight
    thres = zero_intent_thres
    if time <= thres:
      intent_contribution = math_ops.to_float(0)
    else:
      intent_contribution = tf.div(math_ops.to_float(time - thres), math_ops.to_float(sequence_length))
#      intent_contribution = math_ops.to_float(1)
     
    x = rnn_cell._linear([tf.transpose(tf.transpose(current_intent_embedding)*intent_contribution), current_tag_embedding, input_], word_emb_size, True)
    call_cell = lambda: cell(x, state)
    
    
    # pylint: enable=cell-var-from-loop
    if sequence_length is not None:
      (output_fw, state) = rnn._rnn_step(
          time, sequence_length, min_sequence_length, max_sequence_length,
          zero_output, state, call_cell, cell.state_size)
    else:
      (output_fw, state) = call_cell()

    encoder_outputs.append(output_fw)     

    if use_predicted_output:
      intent_logit, current_intent_embedding = intent_loop_function(output_fw, time) 
      tagging_logit, current_tag_embedding = tagging_loop_function(output_fw, time) 
    else:
      if train_with_true_label is True:
        intent_logit = multilayer_perceptron_with_initialized_W(output_fw, target1_output_projection, forward_only=use_predicted_output)
        tagging_logit = multilayer_perceptron_with_initialized_W(output_fw, target2_output_projection, forward_only=use_predicted_output)
        current_intent_embedding = intent_target_embeddings[time]
        current_tag_embedding = tag_target_embeddings[time]
      else:    
        intent_logit, current_intent_embedding = intent_loop_function(output_fw, time)
        tagging_logit, current_tag_embedding = tagging_loop_function(output_fw, time) 
      # prev_symbols.append(prev_symbol)
    if time == 0:
      current_intent_embedding = zero_intent_embedding
      current_tag_embedding = zero_tag_embedding
    sampled_intent_embeddings.append(current_intent_embedding) 
    sampled_tag_embeddings.append(current_tag_embedding) 

    intent_logits.append(intent_logit)
    tagging_logits.append(tagging_logit)
      
  return encoder_outputs, state, intent_logits, tagging_logits, sampled_intent_embeddings, sampled_tag_embeddings
  
def generate_embedding_RNN_output_joint(encoder_inputs, cell,
                                        num_encoder_symbols,
                                        intent_targets,
                                        num_intent_target_symbols,
                                        tagging_targets,
                                        num_tagging_target_symbols,
                                        word_emb_size,
                                        dtype=dtypes.float32,
                                        scope=None, initial_state_attention=False,
                                        sequence_length=None,
                                        bidirectional_rnn=False,
                                        DNN_at_output=False, 
                                        dnn_hidden_layer_size=200,
                                        output_emb_size=50,
                                        trainable=True,
                                        train_with_true_label=True,
                                        zero_intent_thres=0,
                                        forward_only=False):
  """
  Generate RNN state outputs with word embeddings as inputs
  """
  with variable_scope.variable_scope(scope or "generate_embedding_RNN_output_joint"):
    encoder_cell = cell
    embedding = variable_scope.get_variable("word_embedding", [num_encoder_symbols, word_emb_size], trainable=trainable)
    encoder_embedded_inputs = list()
    encoder_embedded_inputs = [embedding_ops.embedding_lookup(embedding, encoder_input) for encoder_input in encoder_inputs]    

    if DNN_at_output is True:
        with variable_scope.variable_scope("intent_DNN"):
          bias_start = 0.0
          n_hidden = dnn_hidden_layer_size
          intent_target_weight_hidden = variable_scope.get_variable("Weight_Hidden", [encoder_cell.output_size, n_hidden])         
          intent_target_bias_hidden = variable_scope.get_variable("Bias_Hidden", [n_hidden],
                                                      initializer=init_ops.constant_initializer(bias_start))                                                  
          intent_target_weight_out = variable_scope.get_variable("Weight_Out", [n_hidden, num_intent_target_symbols])
          intent_target_1bias_out = variable_scope.get_variable("Bias_Out", [num_intent_target_symbols],
                                                      initializer=init_ops.constant_initializer(bias_start)) 
          intent_target_output_projection = (intent_target_weight_hidden, intent_target_bias_hidden, intent_target_weight_out, intent_target_1bias_out)
          intent_target_regularizers = get_multilayer_perceptron_regularizers(intent_target_output_projection)
        with variable_scope.variable_scope("tag_DNN"):
          bias_start = 0.0
          n_hidden = dnn_hidden_layer_size
          tagging_target_weight_hidden = variable_scope.get_variable("Weight_Hidden", [encoder_cell.output_size, n_hidden])         
          tagging_target_bias_hidden = variable_scope.get_variable("Bias_Hidden", [n_hidden],
                                                      initializer=init_ops.constant_initializer(bias_start))                                                  
          tagging_target_weight_out = variable_scope.get_variable("Weight_Out", [n_hidden, num_tagging_target_symbols])
          tagging_target_1bias_out = variable_scope.get_variable("Bias_Out", [num_tagging_target_symbols],
                                                      initializer=init_ops.constant_initializer(bias_start)) 
          tagging_target_output_projection = (tagging_target_weight_hidden, tagging_target_bias_hidden, tagging_target_weight_out, tagging_target_1bias_out)
          tagging_target_regularizers = get_multilayer_perceptron_regularizers(tagging_target_output_projection)
    else:
        with variable_scope.variable_scope("intent_linear"):
          bias_start = 0.0                                                
          intent_target_weight_out = variable_scope.get_variable("Weight_Out", [encoder_cell.output_size, num_intent_target_symbols])
          intent_target_bias_out = variable_scope.get_variable("Bias_Out", [num_intent_target_symbols],
                                                      initializer=init_ops.constant_initializer(bias_start)) 
          intent_target_output_projection = (intent_target_weight_out, intent_target_bias_out)         
          intent_target_regularizers = get_linear_transformation_regularizers(intent_target_output_projection)
        with variable_scope.variable_scope("tag_linear"):
          bias_start = 0.0                                                
          tagging_target_weight_out = variable_scope.get_variable("Weight_Out", [encoder_cell.output_size, num_tagging_target_symbols])
          tagging_target_bias_out = variable_scope.get_variable("Bias_Out", [num_tagging_target_symbols],
                                                      initializer=init_ops.constant_initializer(bias_start)) 
          tagging_target_output_projection = (tagging_target_weight_out, tagging_target_bias_out)         
          tagging_target_regularizers = get_linear_transformation_regularizers(tagging_target_output_projection)

    intent_target_emb_size = output_emb_size
    tagging_target_emb_size = output_emb_size
    
    outputs = rnn_with_output_feedback( encoder_cell, encoder_embedded_inputs, 
                                        intent_targets, num_intent_target_symbols, intent_target_emb_size, intent_target_output_projection, 
                                        tagging_targets, num_tagging_target_symbols, tagging_target_emb_size, tagging_target_output_projection,
                                        word_emb_size, DNN_at_output, zero_intent_thres=zero_intent_thres,
                                        sequence_length=sequence_length, dtype=dtype,
                                        train_with_true_label=train_with_true_label,
                                        use_predicted_output=forward_only)

    
    encoder_outputs, encoder_state, intent_logits, tagging_logits, sampled_intent_embeddings, sampled_tag_embeddings = outputs
    
    intent_results = (intent_logits, intent_target_regularizers, sampled_intent_embeddings)
    tagging_results = (tagging_logits, tagging_target_regularizers, sampled_tag_embeddings)
    
  return encoder_outputs, encoder_state, encoder_embedded_inputs, intent_results, tagging_results