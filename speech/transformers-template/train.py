from src.dataset import get, get_shape
from src.helpers import prepare_featurizers, prepare_training_datasets, prepare_training_dataloaders
from src.model import Model
from src.configs import Config
from src.loss import MaskedCrossEntropyLoss
from src.utils import env_util

from omegaconf import DictConfig, OmegaConf
from IPython.display import Audio

import os
import jiwer
import hydra
import tensorflow as tf

logger = tf.get_logger()


@hydra.main(config_path="config", config_name="config")
def main(
    config: DictConfig,
    batch_size: int = None,
    spx: int = 1,
):
    config = Config(OmegaConf.to_container(config, resolve=True), training=True)
    
    tf.keras.backend.clear_session()
    env_util.setup_seed()
    strategy = env_util.setup_strategy(config.learning_config["running_config"]["devices"])
    batch_size = batch_size or config.learning_config["running_config"]["batch_size"]
    
    speech_featurizer, tokenizer = prepare_featurizers(config)

    train_dataset, valid_dataset = prepare_training_datasets(
        config,
        speech_featurizer=speech_featurizer,
        tokenizer=tokenizer,
    )

    shapes = get_shape(
        config,
        train_dataset,
        valid_dataset,
    )

    train_data_loader, valid_data_loader, global_batch_size = prepare_training_dataloaders(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        strategy=strategy,
        global_batch_size=batch_size,
        shapes=shapes,
    )

    for batch in train_data_loader:
        print("Batch input keys:", batch[0].keys())
        print("Batch target keys:", batch[1].keys())
        print("Length of shifted_right_text_inputs:", batch[0]["shifted_right_text_inputs"])
        print("text_targets dtype:", batch[1]["text_targets"])
        print("text_targets shape:", batch[1]["text_targets"].shape)
        # print(data)
        # print(data[0]["audio_inputs"][0])
        # plot(data[0]["audio_inputs"][0].numpy())
        # print(data[0]["shifted_right_text_inputs"].shape)
        # print(tokenizer.batch_decode(data[0]["shifted_right_text_inputs"].numpy().tolist()))
        # print(data[1]["text_targets"][0])
        # print(tokenizer.batch_decode(data[1]["text_targets"].numpy().tolist()))
        break

    with strategy.scope():
        model = Model(**config.model_config, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer)
        model.make(**shapes, batch_size=global_batch_size)
        model.summary(expand_nested=False)

        if config.learning_config["pretrained"]:
            model.load_weights(config.learning_config["pretrained"], by_name=True)
        model.compile(
            optimizer=tf.keras.optimizers.get(config.learning_config["optimizer_config"]),
            loss=MaskedCrossEntropyLoss(global_batch_size=global_batch_size, ignore_class=tokenizer.pad_token_id),
            run_eagerly=False,
        )

        # outputs = model({"audio_inputs": batch[0]["audio_inputs"], "shifted_right_text_inputs": batch[0]["shifted_right_text_inputs"]}, training=False)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(**config.learning_config["running_config"]["checkpoint"], verbose=1),
        tf.keras.callbacks.BackupAndRestore(config.learning_config["running_config"]["states_dir"]),
        tf.keras.callbacks.TensorBoard(**config.learning_config["running_config"]["tensorboard"]),
        tf.keras.callbacks.CSVLogger(config.learning_config["running_config"]["csv_logger"]),
    ]

    model.fit(
        train_data_loader,
        epochs=config.learning_config["running_config"]["num_epochs"],
        validation_data=valid_data_loader,
        steps_per_epoch=train_dataset.total_steps,
        validation_steps=valid_dataset.total_steps if valid_data_loader else None,
        callbacks=callbacks,
        verbose=1,
    )

if __name__ == "__main__":
    main()