import tensorflow as tf
from src.model import Model
from src.helpers import prepare_featurizers, prepare_testing_datasets
from src.dataset import get_shape
from src.configs import Config
from omegaconf import OmegaConf, DictConfig
from src.utils import data_util
import numpy as np
import hydra


@hydra.main(config_path="config", config_name="config")
def main(
        config: DictConfig, 
        checkpoint_path: str = "model.h5",
        audio_path = "audio.wav"
):
    config = Config(OmegaConf.to_container(config, resolve=True), training=True)
    speech_featurizer, tokenizer = prepare_featurizers(config)

    test_dataset = prepare_testing_datasets(
        config,
        speech_featurizer=speech_featurizer,
        tokenizer=tokenizer,
    )

    shapes = get_shape(
        config,
        test_dataset,
    )
    model = Model(**config.model_config, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer)
    # model.add_featurizers(speech_featurizer, tokenizer)
    model.make(**shapes, batch_size=1)
    model.load_weights(checkpoint_path)
    
    audio = data_util.read_raw_audio(audio_path)
    audio = tf.expand_dims(audio, 1)  # Add batch dimension if needed
    features = speech_featurizer(audio)
    text = model.recognize_single(features)
    print("Predicted:", text)

if __name__ == "__main__":
    main()