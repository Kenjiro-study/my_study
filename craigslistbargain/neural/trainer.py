from cocoa.neural.trainer import Trainer as BaseTrainer

class Trainer(BaseTrainer):
    ''' Cocoa から継承した訓練プロセスを制御するクラス '''

    def _run_batch(self, batch, dec_state=None, enc_state=None):
        encoder_inputs = batch.encoder_inputs
        decoder_inputs = batch.decoder_inputs
        targets = batch.targets
        lengths = batch.lengths
        #tgt_lengths = batch.tgt_lengths

        # NegotiationModel 内で forward() メソッドを実行する
        if hasattr(self.model, 'context_embedder'):
            context_inputs = batch.context_inputs
            title_inputs = batch.title_inputs
            desc_inputs = batch.desc_inputs

            outputs, attns, dec_state = self.model(encoder_inputs,
                    decoder_inputs, context_inputs, title_inputs,
                    desc_inputs, lengths, dec_state, enc_state)
        # NMT Model 内で forward() メソッドを実行する
        else:
            outputs, attns, dec_state = self.model(encoder_inputs,
                  decoder_inputs, lengths, dec_state, enc_state)

        return outputs, attns, dec_state
