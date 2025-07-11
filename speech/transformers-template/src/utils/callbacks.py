import jiwer
import tensorflow as tf

class ASRLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, tokenizer, val_data, pad_token_id=0, max_samples_to_log=3):
        super().__init__()
        self.tokenizer = tokenizer
        self.val_data = val_data  # Should be a tf.data.Dataset or similar
        self.pad_token_id = pad_token_id
        self.max_samples_to_log = max_samples_to_log

    def decode_batch(self, token_ids_batch):
        text_list = []
        for single_example_token_ids in token_ids_batch:
            actual_ids = [int(idx) for idx in single_example_token_ids if idx != self.pad_token_id]
            if not actual_ids:
                text_list.append("")
                continue
            try:
                text = self.tokenizer.decode(actual_ids, skip_special_tokens=True)
            except Exception:
                text = "[decoding error]"
            text_list.append(text)
        return text_list

    def on_epoch_end(self, epoch, logs=None):
        total_wer = 0.0
        total_cer = 0.0
        total_samples = 0
        logged = 0

        for batch in self.val_data:
            x = batch[0]
            y_true = batch[1]["text_targets"].numpy()
            y_pred_logits = self.model(x, training=False)
            y_pred_ids = tf.argmax(y_pred_logits, axis=-1).numpy()
            
            true_texts = self.decode_batch(y_true)
            pred_texts = self.decode_batch(y_pred_ids)

            for t, p in zip(true_texts, pred_texts):
                total_wer += jiwer.wer(t, p)
                total_cer += jiwer.cer(t, p)
                total_samples += 1

                if logged < self.max_samples_to_log:
                    print(f"\n[Sample {logged+1}]")
                    print("Target:   ", t)
                    print("Predicted:", p)
                    logged += 1

        avg_wer = total_wer / total_samples if total_samples else 0.0
        avg_cer = total_cer / total_samples if total_samples else 0.0
        print(f"\nEpoch {epoch+1} - Validation WER: {avg_wer:.4f}, CER: {avg_cer:.4f}")