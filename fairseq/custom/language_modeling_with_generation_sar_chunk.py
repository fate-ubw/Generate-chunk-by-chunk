# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os

from fairseq.data import (
    data_utils,
    TokenBlockDataset,
    ChunkedDataset,
)
from fairseq.tasks import register_task
# from fairseq.tasks.language_modeling import LanguageModelingTask
# from fairseq.custom.sequence_penalty_loss import SequencePenaltyCriterion
# import torch
from fairseq.custom.language_modeling_with_generation import LanguageModelingWithGenerationTask

@register_task('language_modeling_with_generation_sar_chunk')
class LanguageModelingWithGenerationTask_Sarchunk(LanguageModelingWithGenerationTask):
    
    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args, dictionary, output_dictionary=output_dictionary, targets=targets)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """rewrite the load_dataset adding AddChunkStampDataset

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

        dataset = TokenBlockDataset(
            dataset, dataset.sizes, self.args.tokens_per_sample,
            pad=self.dictionary.pad(), eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode, include_targets=True,
        )
        # False
        add_eos_for_other_targets = self.args.sample_break_mode is not None and self.args.sample_break_mode != 'none'
        # ar test
        self.datasets[split] = ChunkedDataset(
                dataset, dataset.sizes, self.dictionary, self.output_dictionary, max_chunklength = 5,
                add_eos_for_other_targets = add_eos_for_other_targets, shuffle = True,
                targets=self.targets, add_bos_token=self.args.add_bos_token,
                restore_way = 'Stay_half_chunk', chunk_option= 'NAR_insingleword'
        )
