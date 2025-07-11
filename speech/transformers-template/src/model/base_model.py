from src.utils import file_util, data_util
from src.schemas import TrainInput

import tensorflow as tf

logger = tf.get_logger()


class BaseModel(tf.keras.Model):
    def __init__(self, tokenizer=None, **kwargs):
        super(BaseModel, self).__init__(**kwargs)
        self._tfasr_metrics = {}
        self.loss_metric = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)
        self._tfasr_metrics["loss"] = self.loss_metric
        self.tokenizer = tokenizer

        # self.wer_metric = WERMetric(decode_fn=self._decode_for_metric, name="wer")
        # self.cer_metric = CERMetric(decode_fn=self._decode_for_metric, name="cer")
        # self._tfasr_metrics["wer"] = self.wer_metric
        # self._tfasr_metrics["cer"] = self.cer_metric

    @property
    def metrics(self):
        return list(self._tfasr_metrics.values())
    
    def save(
        self,
        filepath: str,
        overwrite: bool = True,
        include_optimizer: bool = True,
        save_format: str = None,
        signatures: dict = None,
        options: tf.saved_model.SaveOptions = None,
        save_traces: bool = True,
    ):
        with file_util.save_file(filepath) as path:
            super(BaseModel, self).save(
                filepath=filepath,
                overwrite=overwrite,
                include_optimizer=include_optimizer,
                save_format=save_format,
                signatures=signatures,
                options=options,
                save_traces=save_traces,
            )

    def save_weights(
        self,
        filepath: str,
        overwrite: bool = True,
        save_format: str = None,
        options: tf.saved_model.SaveOptions = None,
    ):
        with file_util.save_file(filepath) as path:
            super(BaseModel, self).save_weights(filepath=path, overwrite=overwrite, save_format=save_format, options=options)

    def load_weights(
            self,
            filepath,
            by_name=False,
            skip_mismatch=False,
            options=None,
    ):
        with file_util.read_file(filepath) as path:
            super().load_weights(filepath=path, by_name=by_name, skip_mismatch=skip_mismatch, options=options)


    def make(
            self,
            audio_input_shape = [None],
            shifted_right_text_input_shape = [None],
            batch_size = None,
            **kwargs
    ):
        audio_inputs = tf.keras.Input(shape=audio_input_shape, batch_size=batch_size, dtype=tf.float32)
        shifted_right_text_inputs = tf.keras.Input(shape=shifted_right_text_input_shape, batch_size=batch_size, dtype=tf.int32)

        outputs = self(
            TrainInput(
                audio_inputs=audio_inputs,
                shifted_right_text_inputs=shifted_right_text_inputs,
            ),
            training=False,
        )
        return outputs
    
    def compile(
        self,
        loss,
        optimizer,
        run_eagerly=None,
        **kwargs,
    ):
        optimizer = tf.keras.optimizers.get(optimizer)
        super().compile(optimizer=optimizer, loss=loss, run_eagerly=run_eagerly, **kwargs)

    def call(self, inputs, training=False, mask=None):
        raise NotImplementedError("The call method is not implemented in the base model.")

    def _train_step(self, data):
        x = data[0]
        y = data[1]["text_targets"]

        with tf.GradientTape() as tape:
            tape.watch(x["audio_inputs"])
            outputs = self(x, training=True)
            tape.watch(outputs)
            y_pred = outputs
            # logger.info("============================================================")
            # logger.info(
            #     # f"Y pred: {y_pred}, Y true: {y}, "
            #     # f"Y pred shape: {y_pred.shape}, Y true shape: {y.shape}, "
            #     f"Y pred: {self.tokenizer.batch_decode(tf.argmax(y_pred, axis=-1).numpy().tolist())}, "
            #     f"Y true: {self.tokenizer.batch_decode(y.numpy().tolist())}, X: {x['audio_inputs'].shape}"
            # )
            loss = self.compute_loss(x, y, y_pred)

            # loss = tf.debugging.check_numerics(loss, "Loss is NaN or Inf")

            gradients = tape.gradient(loss, self.trainable_variables)

            # for i, grad in enumerate(gradients):
            #     if grad is not None:
            #         gradients[i] = tf.debugging.check_numerics(grad, f"Gradient for var {i} is NaN or Inf")
            # logger.info("Gradient norms:", [tf.norm(g) if g is not None else 0 for g in gradients[:5]])

        self.loss_metric.update_state(loss)
        
        return gradients
    
    def train_step(self, data):
        gradients  = self._train_step(data)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {m.name: m.result() for m in self.metrics}
    
    def _test_step(self, data):
        x = data[0]
        y = data[1]["text_targets"]
        outputs = self(x, training=False)
        y_pred = outputs
        loss = self.compute_loss(x, y, y_pred)

        self.loss_metric.update_state(loss)
        return loss
    
    def test_step(self, data):
        self._test_step(data)
        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, data):
        """Clean predict step that uses recognize for autoregressive generation."""
        x = data[0]
        y = data[1]["text_targets"] if len(data) > 1 and "text_targets" in data[1] else None
        
        audio_inputs = x["audio_inputs"]

        predicted_ids = self.recognize(audio_inputs, model_max_length=None, beam_width=50)
        
        # Get actual batch size from predicted_ids to ensure consistency
        actual_batch_size = tf.shape(predicted_ids)[0]
        
        if y is not None:
            # Decode ground truth to text
            def decode_labels(labels_tensor):
                decoded = self.tokenizer.batch_decode(labels_tensor.numpy().tolist(), skip_special_tokens=True)
                return tf.constant(decoded, dtype=tf.string)
            
            labels = tf.py_function(
                func=decode_labels,
                inp=[y],
                Tout=tf.string
            )
            # Set the shape explicitly
            labels.set_shape([None])
        else:
            # Create empty strings if no ground truth
            labels = tf.fill([actual_batch_size], "")

        # Remove padding and decode greedy results
        def clean_and_decode(sequences):
            """Remove padding tokens and decode sequences."""
            cleaned_sequences = []
            for seq in sequences.numpy():
                # Remove padding tokens
                if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                    # Find first pad token position
                    pad_mask = seq != self.tokenizer.pad_token_id
                    if tf.reduce_any(pad_mask):
                        # Keep tokens up to first pad token
                        valid_length = tf.reduce_sum(tf.cast(pad_mask, tf.int32))
                        cleaned_sequences.append(seq[:valid_length].tolist())
                    else:
                        cleaned_sequences.append([])
                else:
                    cleaned_sequences.append(seq.tolist())
            decoded = self.tokenizer.batch_decode(cleaned_sequences, skip_special_tokens=True)
            return tf.constant(decoded, dtype=tf.string)
        
        greedy_decoding = tf.py_function(
            func=clean_and_decode,
            inp=[predicted_ids],
            Tout=tf.string
        )
        # Set the shape explicitly
        greedy_decoding.set_shape([None])

        print(f"Greedy decoding: {greedy_decoding}")
        print(f"Labels: {labels}")
        print(f"Batch size: {actual_batch_size}")
        print(f"Predicted IDs shape: {tf.shape(predicted_ids)}")

        # Beam search decoding (placeholder for now)
        beam_search_decoding = tf.fill([actual_batch_size], "")
        
        # Stack results: [truth, greedy, beam_search]
        return tf.stack([labels, greedy_decoding, beam_search_decoding], axis=-1)
    
    # --------------------------------------------- TFLITE -----------------------------------

    def recognize(self, signal: tf.Tensor, model_max_length: int = None):
        raise NotImplementedError("The recognize method is not implemented in the base model.")
    
    def recognize_tflite(
        self,
        signal: tf.Tensor,
        predicted: tf.Tensor,
    ):
        raise NotImplementedError("The recognize_tflite method is not implemented in the base model.")
        
    
    def make_tflite_function(beam_width: int = 0):
        raise NotImplementedError("The make_tflite_function method is not implemented in the base model.")