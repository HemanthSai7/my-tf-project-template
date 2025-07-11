import math
import tensorflow as tf

def get_num_batches(
    nsamples,
    batch_size,
    drop_remainders=True,
):
    if nsamples is None or batch_size is None:
        return None
    if drop_remainders:
        return math.floor(float(nsamples) / float(batch_size))
    return math.ceil(float(nsamples) / float(batch_size))

def log10(x):
    return tf.math.log(x) / tf.math.log(10.0)

def get_conv_length(input_length, kernel_size, padding, strides):
    length = input_length

    length = tf.cast(length, tf.float32)
    kernel_size = tf.cast(kernel_size, tf.float32)
    strides = tf.cast(strides, tf.float32)
    
    if padding == "same":
        length = tf.math.ceil(length / strides)
    elif padding == "valid":
        length = (length - kernel_size ) / strides + 1.0
            
    return tf.cast(length, tf.int32)

def get_conv_length_py(input_length: int, kernel_size: int, padding: str, strides: int) -> int:
    if input_length is None:
        return None
    
    length = float(input_length)
    
    if padding == "same":
        length = math.ceil(length / float(strides))
    elif padding == "valid":
        length = math.floor((length - float(kernel_size) + 1.0) / float(strides))
            
    return int(length)
