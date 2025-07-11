from src.model import Encoder, Decoder ,BaseModel
from typing import Union, Optional, List

import tensorflow as tf

logger = tf.get_logger()

__all__ = ["Model"]

@tf.keras.utils.register_keras_serializable(package=__name__)
class Model(BaseModel):
    
    """
    Define model here.
    """
    
    # --------------------------------------------- GREEDY SEARCH ---------------------------------------------

    def _perform_greedy_batch(
        self,
        encoder_outputs: tf.Tensor,
        max_length: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
    ):
        """
        Greedy decode a batch of encoder outputs.
        Returns: [batch, max_length] int32 tensor (padded)
        """
        batch_size = tf.shape(encoder_outputs)[0]
        decoded = tf.TensorArray(tf.int32, size=batch_size, dynamic_size=False, clear_after_read=False)

        def condition(batch, decoded):
            return tf.less(batch, batch_size)

        def body(batch, decoded):
            output = self._perform_greedy(
                encoder_outputs[batch],
                max_length,
                bos_token_id,
                eos_token_id,
                pad_token_id,
            )
            # Pad output to max_length
            current_length = tf.shape(output)[0]
            padded_output = tf.pad(
                output, 
                [[0, max_length - current_length]], 
                constant_values=pad_token_id
            )
            decoded = decoded.write(batch, padded_output)
            return batch + 1, decoded

        # Remove the duplicate while_loop call
        batch, decoded = tf.while_loop(
            condition,
            body,
            [tf.constant(0, dtype=tf.int32), decoded],
        )
        
        return decoded.stack()  # [batch_size, max_length]
    
    def _perform_greedy(
        self,
        encoder_outputs: tf.Tensor,
        max_length: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
    ):
        """
        Greedy decode a single sample (no batch dimension).
        Returns: [sequence_length] int32 tensor
        """
        step = tf.constant(0, dtype=tf.int32)
        decoder_input = tf.expand_dims([bos_token_id], 0)  # [1, 1]
        finished = tf.constant(False)
        generated = tf.TensorArray(
            tf.int32, 
            size=max_length, 
            dynamic_size=False, 
            clear_after_read=False
        )

        # tf.print("BOS token ID:", bos_token_id)
        # tf.print("EOS token ID:", eos_token_id)
        # tf.print("PAD token ID:", pad_token_id)

        def condition(step, decoder_input, finished, generated):
            return tf.logical_and(step < max_length, tf.logical_not(finished))

        def body(step, decoder_input, finished, generated):
            embedded = self.text_embedding(decoder_input)
            text_mask = tf.cast(tf.not_equal(decoder_input, pad_token_id), tf.float32)
            decoder_out = self.decoder(
                inputs=[embedded, tf.expand_dims(encoder_outputs, 0)],
                mask=[text_mask, None],
                training=False,
            )
            logits = self.final_dense(decoder_out, training=False)
            next_token = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)  # [1]

            # tf.print("Step:", step, "Generated token:", next_token[0])
            # tf.print("Token logits (top 5):", tf.nn.top_k(logits[0, -1, :], k=5))

            generated = generated.write(step, next_token[0])

            is_eos = tf.equal(next_token[0], eos_token_id)
            is_bos_late = tf.logical_and(tf.equal(next_token[0], bos_token_id), step > 0)

            max_reasonable_length = tf.minimum(
                tf.cast(tf.shape(encoder_outputs)[0] * 2, tf.int32),  # 2x encoder length
                max_length - 1  # Leave room for potential EOS
            )
            is_too_long = step >= max_reasonable_length

            finished = tf.logical_or(tf.logical_or(is_eos, is_bos_late), is_too_long)

            # tf.print("Is EOS?", is_eos, "Is BOS late?", is_bos_late, "Finished?", finished)

            # Only write and update input if not finished
            decoder_input = tf.cond(
                finished,
                lambda: decoder_input,  # Keep current input if finished
                lambda: tf.concat([decoder_input, tf.expand_dims(next_token, 1)], axis=1)  # Append token if not finished
            )
    
            return step + 1, decoder_input, finished, generated

        step, decoder_input, finished, generated = tf.while_loop(
            condition,
            body,
            [step, decoder_input, finished, generated],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([1, None]),
                tf.TensorShape([]),
                tf.TensorShape(None)
            ]
        )

        # tf.print("Final step:", step, "Finished:", finished)

        actual_length = tf.minimum(step, max_length)
        generated_tokens = generated.gather(tf.range(actual_length))
        # tf.print(f"Generated tokens shape: {tf.shape(generated_tokens)}")
        # tf.print(f"Generated tokens: {generated_tokens}")
        return generated_tokens
    
    # --------------------------------------------- BEAM SEARCH ---------------------------------------------
    
    def _perform_beam_search_batched(
        self,
        encoder_outputs: tf.Tensor,
        max_length: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        beam_size: int = 5,
        patience: float = 1.0,
    ):
        """
        Batched beam search decoding for multiple inputs.
        Returns: [batch_size, sequence_length] int32 tensor
        """
        batch_size = tf.shape(encoder_outputs)[0]
        
        # Initialize sequences with BOS token: [batch, beam, 1]
        sequences = tf.fill([batch_size, beam_size, 1], bos_token_id)
        scores = tf.zeros([batch_size, beam_size], dtype=tf.float32)
        finished = tf.zeros([batch_size, beam_size], dtype=tf.bool)
        step = tf.constant(0, dtype=tf.int32)

        def condition(step, sequences, scores, finished):
            return tf.logical_and(
                step < max_length,
                tf.logical_not(tf.reduce_all(finished))
            )

        def body(step, sequences, scores, finished):
            # Flatten sequences for batch processing: [batch*beam, seq_len]
            current_seq_len = tf.shape(sequences)[2]
            flat_sequences = tf.reshape(sequences, [-1, current_seq_len])
            
            # Embed sequences
            embedded = self.text_embedding(flat_sequences)
            
            # Expand encoder outputs for each beam: [batch*beam, enc_len, d_model]
            enc_expanded = tf.repeat(encoder_outputs, beam_size, axis=0)
            
            # Create mask for padding
            text_mask = tf.cast(tf.not_equal(flat_sequences, pad_token_id), tf.float32)
            
            # Decoder forward pass
            decoder_out = self.decoder(
                inputs=[embedded, enc_expanded],
                mask=[text_mask, None],
                training=False,
            )
            
            # Get logits and log probabilities
            logits = self.final_dense(decoder_out)  # [batch*beam, seq_len, vocab]
            log_probs = tf.nn.log_softmax(logits[:, -1, :])  # [batch*beam, vocab]
            
            # Reshape back to batch dimension: [batch, beam, vocab]
            log_probs = tf.reshape(log_probs, [batch_size, beam_size, self.vocab_size])
            
            # Add current scores: [batch, beam, vocab]
            scores_expanded = tf.expand_dims(scores, axis=2)  # [batch, beam, 1]
            total_scores = scores_expanded + log_probs  # [batch, beam, vocab]
            
            # Flatten for top-k selection: [batch, beam*vocab]
            flat_scores = tf.reshape(total_scores, [batch_size, -1])
            
            # Get top-k scores and indices
            topk_scores, topk_indices = tf.math.top_k(flat_scores, k=beam_size)
            
            # Compute beam and token indices
            beam_indices = topk_indices // self.vocab_size  # [batch, beam]
            token_indices = topk_indices % self.vocab_size   # [batch, beam]
            
            # Gather sequences based on beam indices
            def gather_sequences(batch_idx):
                return tf.gather(sequences[batch_idx], beam_indices[batch_idx])
            
            gathered_sequences = tf.map_fn(
                gather_sequences, 
                tf.range(batch_size), 
                fn_output_signature=tf.TensorSpec([beam_size, None], dtype=tf.int32),
                parallel_iterations=10
            )
            
            # Append new tokens: [batch, beam, seq_len+1]
            new_tokens = tf.expand_dims(token_indices, axis=2)  # [batch, beam, 1]
            sequences = tf.concat([gathered_sequences, new_tokens], axis=2)
            
            # Update scores
            scores = topk_scores
            
            # Update finished status
            new_finished = tf.equal(token_indices, eos_token_id)
            
            def gather_finished(batch_idx):
                return tf.gather(finished[batch_idx], beam_indices[batch_idx])
            
            gathered_finished = tf.map_fn(
                gather_finished,
                tf.range(batch_size),
                fn_output_signature=tf.TensorSpec([beam_size], dtype=tf.bool),
                parallel_iterations=10
            )
            
            finished = tf.logical_or(gathered_finished, new_finished)
            
            return step + 1, sequences, scores, finished

        # Run the while loop
        step, sequences, scores, finished = tf.while_loop(
            condition,
            body,
            [step, sequences, scores, finished],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, beam_size, None]),  # sequences can grow
                tf.TensorShape([None, beam_size]),
                tf.TensorShape([None, beam_size])
            ]
        )
        
        # Select best sequence for each batch item
        best_indices = tf.argmax(scores, axis=1, output_type=tf.int32)  # [batch]
        
        # Gather best sequences
        def gather_best(batch_idx):
            return sequences[batch_idx, best_indices[batch_idx]]
        
        best_sequences = tf.map_fn(
            gather_best,
            tf.range(batch_size, dtype=tf.int32),
            fn_output_signature=tf.TensorSpec([None], dtype=tf.int32),
            parallel_iterations=10
        )
        
        return best_sequences

    def _perform_beam_batch(self, encoder_outputs, max_length, bos_token_id, eos_token_id, pad_token_id, beam_size=5):
        """
        Updated beam batch method using the new batched beam search.
        """
        # Use the batched beam search directly
        best_sequences = self._perform_beam_search_batched(
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            beam_size=beam_size
        )
        
        # Pad all sequences to max_length
        batch_size = tf.shape(best_sequences)[0]
        
        def pad_sequence(seq):
            seq_len = tf.shape(seq)[0]
            # Truncate if too long
            seq = tf.cond(
                seq_len > max_length,
                lambda: seq[:max_length],
                lambda: seq
            )
            # Pad if too short
            padding_needed = tf.maximum(0, max_length - tf.shape(seq)[0])
            return tf.pad(seq, [[0, padding_needed]], constant_values=pad_token_id)
        
        padded_sequences = tf.map_fn(
            pad_sequence,
            best_sequences,
            fn_output_signature=tf.TensorSpec([None], dtype=tf.int32),
            parallel_iterations=10
        )
        
        return padded_sequences
        
    # --------------------------------------------- INFERENCE METHODS ---------------------------------------------

    def recognize(self, signal: tf.Tensor, model_max_length: int = None, beam_width: int = 1):
        if model_max_length is None:
            duration = (((tf.shape(signal)[1] - 1) * 160) + 400) // 16000
            model_max_length = tf.cast(duration * 25, tf.int32)

        audio_mask = tf.cast(tf.reduce_any(tf.not_equal(signal, 0.0), axis=-1), tf.float32)
        encoder_outputs = self.encoder(
            inputs=signal,
            training=False,
            mask=audio_mask,
        )

        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        if beam_width == 1:
            decoded = self._perform_greedy_batch(
                encoder_outputs, model_max_length, bos_token_id, eos_token_id, pad_token_id,
            )
        else:
            decoded = self._perform_beam_batch(
                encoder_outputs, model_max_length, bos_token_id, eos_token_id, pad_token_id, beam_width
            )

        return decoded

    # --------------------------------------------- TFLITE ---------------------------------------------

    def recognize_tflite(
        self,
        signal: tf.Tensor,
        predicted: tf.Tensor,
    ):
        if self.speech_featurizer is None:
            logger.info(f"Speech featurizer is not set. Log-mel-spectrogram/MFCC will be used.")
            features = signal
        else:
            logger.info(f"Using speech featurizer: {self.speech_featurizer.feature_type}")
            features = self.speech_featurizer(signal)
        
        
    def make_tflite_function(self, beam_width: int = 0):
        tflite_func = self.recognize_tflite

        if self.speech_featurizer is not None:
            input_shape = [None]
        else:
            input_shape = [498, 80, 1]

        return tf.function(
            tflite_func,
            input_signature=[
                tf.TensorSpec(input_shape, dtype=tf.float32),
                tf.TensorSpec([], dtype=tf.int32)
            ]
        )