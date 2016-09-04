# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:32:21 2016

@author: Bing Liu (liubing@cmu.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import zip     # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import tensor_shape
import tensorflow as tf

def _step(time, sequence_length, min_sequence_length, max_sequence_length, zero_logit, generate_logit):
  # Step 1: determine whether we need to call_cell or not
  empty_update = lambda: zero_logit
  logit = control_flow_ops.cond(
      time < max_sequence_length, generate_logit, empty_update)

  # Step 2: determine whether we need to copy through state and/or outputs
  existing_logit = lambda: logit

  def copy_through():
    # Use broadcasting select to determine which values should get
    # the previous state & zero output, and which values should get
    # a calculated state & output.
    copy_cond = (time >= sequence_length)
    return math_ops.select(copy_cond, zero_logit, logit)

  logit = control_flow_ops.cond(
      time < min_sequence_length, existing_logit, copy_through)
  logit.set_shape(logit.get_shape())
  return logit
  
def _reverse_seq(input_seq, lengths):
  """Reverse a list of Tensors up to specified lengths.

  Args:
    input_seq: Sequence of seq_len tensors of dimension (batch_size, depth)
    lengths:   A tensor of dimension batch_size, containing lengths for each
               sequence in the batch. If "None" is specified, simply reverses
               the list.

  Returns:
    time-reversed sequence
  """
  if lengths is None:
    return list(reversed(input_seq))

  input_shape = tensor_shape.matrix(None, None)
  for input_ in input_seq:
    input_shape.merge_with(input_.get_shape())
    input_.set_shape(input_shape)

  # Join into (time, batch_size, depth)
  s_joined = array_ops.pack(input_seq)

  # TODO(schuster, ebrevdo): Remove cast when reverse_sequence takes int32
  if lengths is not None:
    lengths = math_ops.to_int64(lengths)

  # Reverse along dimension 0
  s_reversed = array_ops.reverse_sequence(s_joined, lengths, 0, 1)
  # Split again into list
  result = array_ops.unpack(s_reversed)
  for r in result:
    r.set_shape(input_shape)
  return result


def linear_transformation(_X, input_size, n_class):
    with variable_scope.variable_scope("linear"):
      bias_start = 0.0
      weight_out = variable_scope.get_variable("Weight_out", [input_size, n_class])         
      bias_out = variable_scope.get_variable("Bias_out", [n_class],
                                                  initializer=init_ops.constant_initializer(bias_start))

      output = tf.matmul(_X, weight_out) + bias_out
      #regularizers = tf.nn.l2_loss(weight_hidden) + tf.nn.l2_loss(bias_hidden) + tf.nn.l2_loss(weight_out) + tf.nn.l2_loss(bias_out) 
    return output

def get_linear_transformation_regularizers():
    with variable_scope.variable_scope("linear"):
      weight_out = variable_scope.get_variable("Weight_out")         
      bias_out = variable_scope.get_variable("Bias_out")    
      regularizers = tf.nn.l2_loss(weight_out) + tf.nn.l2_loss(bias_out) 
    return regularizers                                        
                                                  
                                                  
def multilayer_perceptron(_X, input_size, n_hidden, n_class, forward_only=False):
    with variable_scope.variable_scope("DNN"):
      bias_start = 0.0
      weight_hidden = variable_scope.get_variable("Weight_Hidden", [input_size, n_hidden])         
      bias_hidden = variable_scope.get_variable("Bias_Hidden", [n_hidden],
                                                  initializer=init_ops.constant_initializer(bias_start))
      #Hidden layer with RELU activation
      layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, weight_hidden), bias_hidden))

      if not forward_only:
          layer_1 = tf.nn.dropout(layer_1, 0.5)
                                                  
      weight_out = variable_scope.get_variable("Weight_Out", [n_hidden, n_class])
      bias_out = variable_scope.get_variable("Bias_Out", [n_class],
                                                  initializer=init_ops.constant_initializer(bias_start))  
      output = tf.matmul(layer_1, weight_out) + bias_out
      #regularizers = tf.nn.l2_loss(weight_hidden) + tf.nn.l2_loss(bias_hidden) + tf.nn.l2_loss(weight_out) + tf.nn.l2_loss(bias_out) 
    return output

def get_multilayer_perceptron_regularizers():
    with variable_scope.variable_scope("DNN"):
      weight_hidden = variable_scope.get_variable("Weight_Hidden")         
      bias_hidden = variable_scope.get_variable("Bias_Hidden")

      weight_out = variable_scope.get_variable("Weight_Out")
      bias_out = variable_scope.get_variable("Bias_Out")  
      regularizers = tf.nn.l2_loss(weight_hidden) + tf.nn.l2_loss(bias_hidden) + tf.nn.l2_loss(weight_out) + tf.nn.l2_loss(bias_out) 
    return regularizers

def generate_sequence_output(encoder_outputs, 
                  encoder_state,
                  num_decoder_symbols,
                  sequence_length,
                num_heads=1,
                dtype=dtypes.float32,
                use_attention=True,
                loop_function=None,
                scope=None,
                DNN_at_output=False,
                forward_only=False):
  with variable_scope.variable_scope(scope or "non-attention_RNN"):
    attention_encoder_outputs = list()
    sequence_attention_weights = list()
    
    # copy over logits once out of sequence_length
    if encoder_outputs[0].get_shape().ndims != 1:
      (fixed_batch_size, output_size) = encoder_outputs[0].get_shape().with_rank(2)
    else:
      fixed_batch_size = encoder_outputs[0].get_shape().with_rank_at_least(1)[0]

    if fixed_batch_size.value: 
      batch_size = fixed_batch_size.value
    else:
      batch_size = array_ops.shape(encoder_outputs[0])[0]
    if sequence_length is not None:
      sequence_length = math_ops.to_int32(sequence_length)
    if sequence_length is not None:  # Prepare variables
      zero_logit = array_ops.zeros(
          array_ops.pack([batch_size, num_decoder_symbols]), encoder_outputs[0].dtype)
      zero_logit.set_shape(
          tensor_shape.TensorShape([fixed_batch_size.value, num_decoder_symbols]))
      min_sequence_length = math_ops.reduce_min(sequence_length)
      max_sequence_length = math_ops.reduce_max(sequence_length)
  
    for time, input_ in enumerate(encoder_outputs):
      if time > 0: variable_scope.get_variable_scope().reuse_variables()
      
      if not DNN_at_output:  
        generate_logit = lambda: linear_transformation(encoder_outputs[time], output_size, num_decoder_symbols)
      else:
        generate_logit = lambda: multilayer_perceptron(encoder_outputs[time], output_size, 200, num_decoder_symbols, forward_only=forward_only)
      # pylint: enable=cell-var-from-loop
      if sequence_length is not None:
        logit = _step(
            time, sequence_length, min_sequence_length, max_sequence_length, zero_logit, generate_logit)
      else:
        logit = generate_logit
      attention_encoder_outputs.append(logit)   
    if DNN_at_output:  
      regularizers = get_multilayer_perceptron_regularizers()
    else:
      regularizers = get_linear_transformation_regularizers()
  return attention_encoder_outputs, sequence_attention_weights, regularizers


def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.op_scope(logits + targets + weights, name,
                    "sequence_loss_by_example"):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            logit, target)
      else:
        crossent = softmax_loss_function(logit, target)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  with ops.op_scope(logits + targets + weights, name, "sequence_loss"):
    cost = math_ops.reduce_sum(sequence_loss_by_example(
        logits, targets, weights,
        average_across_timesteps=average_across_timesteps,
        softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, dtypes.float32)
    else:
      return cost

  
def generate_task_output(encoder_outputs, additional_inputs, encoder_state, targets,sequence_length, num_decoder_symbols, weights,
                       buckets, softmax_loss_function=None,
                       per_example_loss=False, name=None, use_attention=False, scope=None, DNN_at_output=False, 
                       intent_results=None, 
                       tagging_results=None, 
                       train_with_true_label=True,
                       use_local_context=False,
                       forward_only=False):
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))

  all_inputs = encoder_outputs + targets + weights
  with ops.op_scope(all_inputs, name, "model_with_buckets"):
    if scope == 'intent':
        logits, regularizers, sampled_intents = intent_results
        sampled_tags = list()
    elif scope == 'tagging':
        logits, regularizers, sampled_tags = tagging_results
        sampled_intents = list()
    elif scope == 'lm':
      with variable_scope.variable_scope(scope + "_generate_sequence_output", reuse=None):
        task_inputs = []
        if use_local_context:
          print ('lm task: use sampled_tag_intent_emb as local context')
          task_inputs = [array_ops.concat(1, [additional_input, encoder_output]) for additional_input, encoder_output in zip(additional_inputs, encoder_outputs)]
        else:
          task_inputs = encoder_outputs        

        logits, _, regularizers = generate_sequence_output(task_inputs, 
                                                            encoder_state,
                                                            num_decoder_symbols,
                                                            sequence_length,
                                                            use_attention=use_attention,
                                                            DNN_at_output=DNN_at_output,
                                                            forward_only=forward_only)
      
        sampled_tags = list()
        sampled_intents = list()             
      
    if per_example_loss is None:
      assert len(logits) == len(targets)
      # We need to make target and int64-tensor and set its shape.
      bucket_target = [array_ops.reshape(math_ops.to_int64(x), [-1]) for x in targets]
      crossent = sequence_loss_by_example(
            logits, bucket_target, weights,
            softmax_loss_function=softmax_loss_function)
    else:
      assert len(logits) == len(targets)
      bucket_target = [array_ops.reshape(math_ops.to_int64(x), [-1]) for x in targets]
      crossent = sequence_loss(
            logits, bucket_target, weights,
            softmax_loss_function=softmax_loss_function)
      crossent_with_regularizers = crossent + 1e-4 * regularizers

  return logits, sampled_tags, sampled_intents, crossent_with_regularizers, crossent
