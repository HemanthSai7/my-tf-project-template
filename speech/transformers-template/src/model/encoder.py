# Imports

import tensorflow as tf

__all__ = ["Encoder"]

class Encoder(tf.keras.Model):
    """
    Base Encoder class
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)