# Imports

import tensorflow as tf

__all__ = ["Decoder"]

class Decoder(tf.keras.Model):
    """
    Define decoder here
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)