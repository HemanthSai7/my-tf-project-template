from src.speech_featurizer import SpeechFeaturizer
from src.tokenizer import CharacterTokenizer
from src.dataset import get

import os
import string
import tensorflow as tf

logger = tf.get_logger()

def prepare_featurizers(
    config,
):
    speech_config = config.speech_config
    feature_extractor = SpeechFeaturizer(**dict(speech_config))

    chars = string.ascii_lowercase + " " # This is character vocab
    tokenizer = CharacterTokenizer(chars, model_max_length=None)
    
    return feature_extractor, tokenizer

def prepare_training_datasets(
    config,
    speech_featurizer: SpeechFeaturizer,
    tokenizer: CharacterTokenizer,
):
    train_dataset = get(
        tokenizer=tokenizer,
        speech_featurizer=speech_featurizer,
        dataset_config=config.data_config["train_dataset_config"],
    )
    valid_dataset = get(
        tokenizer=tokenizer,
        speech_featurizer=speech_featurizer,
        dataset_config=config.data_config["eval_dataset_config"],
    )

    return train_dataset, valid_dataset

def prepare_training_dataloaders(
    train_dataset,
    valid_dataset,
    strategy,
    global_batch_size,
    shapes,
):
    global_batch_size *= strategy.num_replicas_in_sync
    train_data_loader = train_dataset.create(batch_size=global_batch_size, padded_shapes=shapes)
    valid_data_loader = valid_dataset.create(batch_size=global_batch_size, padded_shapes=shapes)

    return train_data_loader, valid_data_loader, global_batch_size

def prepare_testing_datasets(
    config,
    speech_featurizer: SpeechFeaturizer,
    tokenizer: CharacterTokenizer,
):
    test_dataset = get(
        speech_featurizer=speech_featurizer,
        tokenizer=tokenizer,
        dataset_config=config.data_config["test_dataset_config"],
    )
    
    return test_dataset

def prepare_testing_dataloaders(
    test_dataset,
    strategy,
    global_batch_size,
    shapes,
):
    global_batch_size *= strategy.num_replicas_in_sync
    test_data_loader = test_dataset.create(batch_size=global_batch_size, padded_shapes=shapes)

    return test_data_loader, global_batch_size

