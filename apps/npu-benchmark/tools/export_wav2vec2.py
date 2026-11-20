"""Export the six-second English CTC encoder with a fixed 16 kHz waveform input."""

import argparse
import torch
from transformers import Wav2Vec2ForCTC


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform):
        return self.model(waveform).logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("revision")
    parser.add_argument("output")
    args = parser.parse_args()
    network = Wav2Vec2ForCTC.from_pretrained(args.checkpoint, revision=args.revision, attn_implementation="eager")
    torch.onnx.export(Encoder(network).eval(), torch.zeros(1, 96000), args.output, opset_version=17,
                      input_names=["waveform"], output_names=["logits"])
