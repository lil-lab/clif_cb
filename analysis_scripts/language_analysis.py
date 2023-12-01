from __future__ import annotations

from absl import app
from config.data_config import DataConfig, TokenizerConfig
from data import loading
from data.dataset_split import DatasetSplit
from learning.training import load_training_data
from typing import TYPE_CHECKING
from nltk import wordpunct_tokenize

if TYPE_CHECKING:
    from data.dataset import DatasetCollection
    from typing import Dict, List, Set

DATASETS: List[str] = [
    '1_29', '1_30', '1_31', '2_1', '2_2', '2_3', '2_5', '2_7'
]


def main(argv):
    print('Loading original datasets')
    original_data: DatasetCollection = load_training_data(DataConfig(
        TokenizerConfig()),
                                                          debug=True,
                                                          val_only=False)

    longest_sent: int = 0

    all_vocabs: Set[str] = set()
    vocabularies: Dict[str, Dict[str, int]] = {'original': dict()}
    lengths: Dict[str, Dict[int, int]] = {'original': dict()}

    for ex in original_data.static_datasets[
            DatasetSplit.TRAIN].instruction_examples:
        tokens = wordpunct_tokenize(ex.instruction.lower())
        length = len(tokens)

        if length not in lengths['original']:
            lengths['original'][length] = 0
        lengths['original'][length] += 1

        if length > longest_sent:
            longest_sent = length
        for tok in tokens:
            if tok not in vocabularies['original']:
                vocabularies['original'][tok] = 0
            vocabularies['original'][tok] += 1

            all_vocabs.add(tok)
    del original_data

    print('Loading recorded datasets')
    recorded_data: DatasetCollection = loading.load_recorded_data(
        DataConfig(TokenizerConfig()),
        DATASETS,
        'game_recordings/',
        use_ips={dataset: False
                 for dataset in DATASETS})

    for did, dat in recorded_data.online_datasets[DatasetSplit.TRAIN].items():
        vocabularies[did] = dict()
        lengths[did] = dict()
        for example in dat.instruction_examples:
            has_pos = False
            has_neg = False

            for step in example.target_action_sequence:
                pol = step.feedback_annotation.feedback.polarity()

                if pol < 0:
                    has_neg = True
                if pol > 0:
                    has_pos = True

            if has_pos and has_neg:
                fb = 'mixed'
            elif has_pos:
                fb = 'pos'
            elif has_neg:
                fb = 'neg'
            else:
                fb = 'none'

            tokens = wordpunct_tokenize(example.instruction.lower())
            length = len(tokens)

            print(
                f'{length}\t{fb}\t{did}\t{len(example.target_action_sequence)}\t{example.example_id.split("-")[-1]}\t{example.example_id}\t{example.instruction}'
            )

            if length not in lengths[did]:
                lengths[did][length] = 0
            lengths[did][length] += 1

            if length > longest_sent:
                longest_sent = length
            for tok in tokens:
                if tok not in vocabularies[did]:
                    vocabularies[did][tok] = 0
                vocabularies[did][tok] += 1

                all_vocabs.add(tok)

    for did in ['original'] + DATASETS:
        with open(f'lengths_{did}.txt', 'w') as ofile:
            ofile.write(f'{length}\t{did} freq\n')
            for length in range(longest_sent + 1):
                freq: int = 0
                if length in lengths[did]:
                    freq = lengths[did][length]
                ofile.write(f'{length}\t{freq}\n')


#        with open(f'vocabs_{did}.txt', 'w') as ofile:
#            ofile.write(f'tok\t{did} freq\n')
#            for tok in sorted(list(all_vocabs)):
#                freq: int = 0
#                if tok in vocabularies[did]:
#                    freq = vocabularies[did][tok]
#                ofile.write(f'{tok}\t{freq}\n')

if __name__ == '__main__':
    app.run(main)
