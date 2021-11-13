#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

See <https://arxiv.org/abs/1910.10683>

The T5 agent can be instantiated as simply `-m t5`
"""
import torch
from typing import Optional, Dict, Any, Tuple

# from transformers import T5ForConditionalGeneration


from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from transformers import LEDTokenizer, LEDModel, LEDForConditionalGeneration
import torch.nn.functional as F

# try:
#     from transformers.models.t5.modeling_t5 import T5Stack
# except ModuleNotFoundError:
#     # Prior versions of transformers package do not have T5Stack
# T5Stack = object

from parlai.agents.hugging_face.hugging_face import HF_VERSION
from parlai.agents.hugging_face.dict import LEDDictionaryAgent

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch, TorchAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel

from parlai.agents.transformer.modules import create_embeddings


def build_led(opt: Opt) -> AutoModelForSeq2SeqLM:
    mname = 'allenai/led-large-16384'
    print(f"Loading model from {opt['led_model_arch']}")
    model = LEDForConditionalGeneration.from_pretrained(
        opt["led_model_arch"],
        dropout=opt["led_dropout"],
        gradient_checkpointing=True,
        use_cache=False,
    )
    model.gradient_checkpointing_enable()
    return model
    # return T5ForConditionalGeneration.from_pretrained(
    #     opt['t5_model_arch'], dropout_rate=opt['t5_dropout']
    # )


class LedAgent(TorchGeneratorAgent):
    @classmethod
    def add_cmdline_args(cls, parser, partial_opt=None):
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group('LED Args')
        group.add_argument(
            '--led-model-arch',
            type=str,
            default='allenai/led-large-16384',
            choices=["allenai/led-base-16384", "allenai/led-large-16384"],
        )

        group.add_argument(
            '--led-dropout', type=float, default=0.1, help='Dropout for LED'
        )

        return parser

    def build_model(self):
        model = ParlaiLEDModel(self.opt, self.dict)
        return model

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overridden if a more complex dictionary is required.
        """
        return LEDDictionaryAgent

    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate an output with beam search.

        Use HF's built-in generation to perform beam search.
        """
        bad_words_ids = None
        if self.beam_block_list is not None:
            bad_words_ids = [
                gram for _, ngram in self.beam_block_list.items() for gram in ngram
            ]

        method = self.opt.get('inference', 'greedy')
        global_attention_mask = torch.zeros_like(batch.text_vec)
        global_attention_mask[:, 0] = 1

        generation_params = {
            'input_ids': batch.text_vec,
            'max_length': max_ts,
            'min_length': self.beam_min_length,
            'do_sample': self.opt['inference'] in ['topk', 'topp'],
            'early_stopping': None,
            'num_beams': beam_size,
            'temperature': self.temperature,
            'top_k': self.opt['topk'] if method in ['topk', 'delayedbeam'] else None,
            'top_p': self.opt['topp'] if method == 'nucleus' else None,
            'repetition_penalty': None,
            'bad_words_ids': bad_words_ids if bad_words_ids else None,
            'bos_token_id': self.START_IDX,
            'pad_token_id': self.NULL_IDX,
            'eos_token_id': self.END_IDX,
            'length_penalty': self.opt['beam_length_penalty'],
            'no_repeat_ngram_size': self.beam_block_ngram,
            'num_return_sequences': None,
            'attention_mask': batch.text_vec != self.NULL_IDX,
            'global_attention_mask': global_attention_mask,
            'decoder_start_token_id': self.NULL_IDX,
        }

        # if self.opt['t5_generation_config']:
        #     config = TASK_CONFIGS[self.opt['t5_generation_config']]
        #     config.pop('prefix', None)
        #     generation_params.update(config)
        # if overrides:
        #     generation_params.update(overrides)
        # print("GENERATING")
        # print(batch.text_vec)
        # print(global_attention_mask)
        # print("-"*50)
        outputs = self.model.model.generate(**generation_params)
        # print(outputs)
        # tokenizer = LEDTokenizer.from_pretrained('allenai/led-large-16384')
        # print(f"GENERATING")
        # print(outputs[0])
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        outputs = [(outputs[i], 0) for i in range(outputs.size(0))]
        # print(outputs)
        return outputs, []
        # return outputs


class ParlaiLEDModel(TorchGeneratorModel):
    def __init__(self, opt, dictionary):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = self.pad_idx
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)
        self.model = build_led(opt)

        # self.encoder = self.model.encoder
        self.encoder = ParlaiLEDEncoder(opt, self.model.get_encoder(), self.pad_idx)
        # self.decoder = self.model.decoder
        self.decoder = ParlaiLEDDecoder(opt, self.model.get_decoder(), self.pad_idx)

        # self.config = self.model.decoder.config
        # self.lm_head = torch.nn.Linear(
        #     self.config.n_embd, self.config.vocab_size, bias=False
        # )
        # embedding_size = 1024
        # self.embeddings = create_embeddings(
        #     dictionary, embedding_size, self.pad_idx
        # )

        # self.lm_head, self.decoder.transformer.wte

    # def output(self, tensor: torch.Tensor) -> torch.Tensor:
    #     """
    #     Compute output logits.

    #     Override standard TGM output to _not_ prevent generation of BOS.
    #     """
    #     # project back to vocabulary
    #     # output = F.linear(tensor, self.embeddings.weight)
    #     output = self.model(tensor)

    #     return output
    def output(self, decoder_output):
        # output = F.linear(decoder_output, self.embeddings.weight)
        # print(output)
        # return output
        # return F.linear(decoder_output, )
        return self.model.lm_head(decoder_output)

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(
        self, incremental_state: Dict[int, dict], inds: torch.Tensor
    ) -> Dict[int, dict]:
        """
        Not *quite* sure how to reconcile this with HF.
        """
        return {}


class ParlaiLEDEncoder(torch.nn.Module):
    def __init__(self, opt: Opt, encoder: object, padding_idx: Optional[int] = None):
        super().__init__()
        self.encoder = encoder
        self.padding_idx = padding_idx

    def forward(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param LongTensor[batch,seqlen] positions:
            Positions for input IDs
        :param LongTensor[batch,seqlen] segments:
            If provided, additionally adds ``segments`` as extra embedding features.
        """
        # if not self.paralleled:
        #     self.stack.parallelize()
        mask = input != self.padding_idx
        outputs = self.encoder.forward(
            input, attention_mask=mask, output_hidden_states=False
        )
        for k in outputs:
            if torch.is_tensor(outputs[k]):
                outputs[k] = outputs[k].to(input.device)
        return outputs[0], mask


class ParlaiLEDDecoder(torch.nn.Module):
    def __init__(self, opt: Opt, decoder: object, padding_idx: Optional[int] = None):
        super().__init__()
        self.decoder = decoder
        self.padding_idx = padding_idx

    def forward(
        self, input: torch.LongTensor, encoder_state: Tuple[Any], incr_state=None
    ):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            The incremental state: a dictionary whose keys index the layers and whose
            values contain the incremental state for each layer.
        """
        encoder_output, encoder_mask = encoder_state

        mask = input != self.padding_idx
        mask[:, 0] = True  # first token is pad

        outputs = self.decoder(
            input_ids=input,
            attention_mask=mask,
            encoder_hidden_states=encoder_output.to(input.device),
            encoder_attention_mask=encoder_mask.to(input.device),
        )
        return outputs[0].to(input.device), incr_state


# class ParlaiLEDEncoder(torch.nn.Module):
#     def __init__(self, opt: Opt, encoder: object, padding_idx: Optional[int] = None):
#         super().__init__()
#         self.stack = encoder
#         self.padding_idx = padding_idx

#     @set_device
#     def forward(
#         self,
#         input: torch.LongTensor,
#         positions: Optional[torch.LongTensor] = None,
#         segments: Optional[torch.LongTensor] = None,
#     ) -> Tuple[torch.Tensor, torch.BoolTensor]:
#         """
#         Forward pass.

#         :param LongTensor[batch,seqlen] input:
#             The input IDs
#         :param LongTensor[batch,seqlen] positions:
#             Positions for input IDs
#         :param LongTensor[batch,seqlen] segments:
#             If provided, additionally adds ``segments`` as extra embedding features.
#         """

#         mask = input != self.padding_idx
#         outputs = self.stack(input, attention_mask=mask, output_hidden_states=False)
#         for k in outputs:
#             if torch.is_tensor(outputs[k]):
#                 outputs[k] = outputs[k].to(input.device)
#         return outputs[0], mask


# class ParlaiLEDDecoder(torch.nn.Module):
#     def __init__(self, opt: Opt, decoder: object, padding_idx: Optional[int] = None):
#         super().__init__()
#         self.stack = decoder
#         self.padding_idx = padding_idx

#     @set_device
#     def forward(
#         self, input: torch.LongTensor, encoder_state: Tuple[Any], incr_state=None
#     ):
#         """
#         Forward pass.

#         :param LongTensor[batch,seqlen] input:
#             The decoder inputs (partial or full decoded token IDs).
#         :param encoder_state:
#             Output from the encoder module forward pass.
#         :param incr_state:
#             The incremental state: a dictionary whose keys index the layers and whose
#             values contain the incremental state for each layer.
#         """

#         encoder_output, encoder_mask = encoder_state

#         mask = input != self.padding_idx
#         mask[:, 0] = True  # first token is pad

#         outputs = self.stack(
#             input_ids=input,
#             attention_mask=mask,
#             encoder_hidden_states=encoder_output.to(input.device),
#             encoder_attention_mask=encoder_mask.to(input.device),
#         )
#         return outputs[0].to(input.device), incr_state


# class ParlaiLEDModel(TorchGeneratorModel):
#     """
#     Wrap T5 in ParlAI.
#     """

#     def __init__(self, opt, dictionary):
#         self.pad_idx = dictionary[dictionary.null_token]
#         self.start_idx = self.pad_idx
#         self.end_idx = dictionary[dictionary.end_token]
#         super().__init__(self.pad_idx, self.start_idx, self.end_idx)
#         self.model = build_led(opt)
#         self.encoder = ParlaiLEDEncoder(opt, self.model.get_encoder(), self.pad_idx)
#         self.decoder = ParlaiLEDDecoder(opt, self.model.get_decoder(), self.pad_idx)

#     @set_device
#     def _get_initial_forced_decoder_input(self, bsz: int, inputs: torch.LongTensor):
#         """
#         Return initial input to the decoder.

#         :param bsz:
#             batchsize
#         :param inputs:
#             inputs to decode

#         :return initial_input:
#             initial input for the decoder.
#         """
#         inputs = torch.cat([self.START.detach().expand(bsz, 1), inputs], 1)
#         return inputs

#     @set_device
#     def reorder_encoder_states(self, encoder_states, indices):
#         """
#         Reorder the encoder states.

#         See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
#         """
#         enc, mask = encoder_states
#         if not torch.is_tensor(indices):
#             indices = torch.LongTensor(indices).to(enc.device)
#         enc = torch.index_select(enc, 0, indices)
#         mask = torch.index_select(mask, 0, indices)
#         return enc, mask

#     def reorder_decoder_incremental_state(
#         self, incremental_state: Dict[int, dict], inds: torch.Tensor
#     ) -> Dict[int, dict]:
#         """
#         Not *quite* sure how to reconcile this with HF.
#         """
#         return {}

#     @set_device
#     def output(self, tensor):
#         """
#         Compute output logits.
#         """
#         # Taken directly from HuggingFace
#         # Rescale output before projecting on vocab
#         # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
#         # tensor = tensor * (self.model.model_dim ** -0.5)
#         lm_logits = self.model.lm_head(tensor)
#         return lm_logits
