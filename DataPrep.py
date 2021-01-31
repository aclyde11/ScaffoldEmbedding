from rdkit import Chem
from tqdm import tqdm
import re
import os
import random

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""

import argparse
import glob
import sys
import gc
import os
import codecs
import torch
from onmt.utils.logging import init_logger, logger

import onmt.inputters as inputters
import onmt.opts as opts


# MARKDOWN boilerplate

# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
class MarkdownHelpFormatter(argparse.HelpFormatter):
    """A really bare-bones argparse help formatter that generates valid markdown.
    This will generate something like:
    usage
    # **section heading**:
    ## **--argument-one**
    ```
    argument-one help text
    ```
    """

    def _format_usage(self, usage, actions, groups, prefix):
        return ""

    def format_help(self):
        print(self._prog)
        self._root_section.heading = '# Options: %s' % self._prog
        return super(MarkdownHelpFormatter, self).format_help()

    def start_section(self, heading):
        super(MarkdownHelpFormatter, self) \
            .start_section('### **%s**' % heading)

    def _format_action(self, action):
        if action.dest == "help" or action.dest == "md":
            return ""
        lines = []
        lines.append('* **-%s %s** ' % (action.dest,
                                        "[%s]" % action.default
                                        if action.default else "[]"))
        if action.help:
            help_text = self._expand_help(action)
            lines.extend(self._split_lines(help_text, 80))
        lines.extend(['', ''])
        return '\n'.join(lines)


class MarkdownHelpAction(argparse.Action):
    """ MD help action """

    def __init__(self, option_strings,
                 dest=argparse.SUPPRESS, default=argparse.SUPPRESS,
                 **kwargs):
        super(MarkdownHelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.formatter_class = MarkdownHelpFormatter
        parser.print_help()
        parser.exit()


class DeprecateAction(argparse.Action):
    """ Deprecate action """

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(option_strings, dest, nargs=0,
                                              help=help, **kwargs)

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.mdhelp is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise argparse.ArgumentTypeError(msg)

def check_existing_pt_files(opt):
    """ Checking if there are existing .pt files to avoid tampering """
    # We will use glob.glob() to find sharded {train|valid}.[0-9]*.pt
    # when training, so check to avoid tampering with existing pt files
    # or mixing them up.
    for t in ['train', 'valid', 'vocab']:
        pattern = opt.save_data + '.' + t + '*.pt'
        if glob.glob(pattern):
            sys.stderr.write("Please backup existing pt file: %s, "
                             "to avoid tampering!\n" % pattern)
            sys.exit(1)

def add_md_help_argument(parser):
    """ md help parser """
    parser.add_argument('-md', action=MarkdownHelpAction,
                        help='print Markdown-formatted help text and exit.')


def preprocess_opts(parser):
    """ Pre-procesing options """
    # Data options
    group = parser.add_argument_group('Data')
    group.add_argument('-data_type', default="text",
                       help="""Type of the source input.
                       Options are [text|img].""")

    group.add_argument('-train_src', required=True,
                       help="Path to the training source data")
    group.add_argument('-train_tgt', required=True,
                       help="Path to the training target data")
    group.add_argument('-valid_src', required=True,
                       help="Path to the validation source data")
    group.add_argument('-valid_tgt', required=True,
                       help="Path to the validation target data")

    group.add_argument('-src_dir', default="",
                       help="Source directory for image or audio files.")

    group.add_argument('-save_data', required=True,
                       help="Output file for the prepared data")

    group.add_argument('-max_shard_size', type=int, default=0,
                       help="""Deprecated use shard_size instead""")

    group.add_argument('-shard_size', type=int, default=1000000,
                       help="""Divide src_corpus and tgt_corpus into
                       smaller multiple src_copus and tgt corpus files, then
                       build shards, each shard will have
                       opt.shard_size samples except last shard.
                       shard_size=0 means no segmentation
                       shard_size>0 means segment dataset into multiple shards,
                       each shard has shard_size samples""")

    # Dictionary options, for text corpus

    group = parser.add_argument_group('Vocab')
    group.add_argument('-src_vocab', default="",
                       help="""Path to an existing source vocabulary. Format:
                       one word per line.""")
    group.add_argument('-tgt_vocab', default="",
                       help="""Path to an existing target vocabulary. Format:
                       one word per line.""")
    group.add_argument('-features_vocabs_prefix', type=str, default='',
                       help="Path prefix to existing features vocabularies")
    group.add_argument('-src_vocab_size', type=int, default=50000,
                       help="Size of the source vocabulary")
    group.add_argument('-tgt_vocab_size', type=int, default=50000,
                       help="Size of the target vocabulary")

    group.add_argument('-src_words_min_frequency', type=int, default=0)
    group.add_argument('-tgt_words_min_frequency', type=int, default=0)

    group.add_argument('-dynamic_dict', action='store_true',
                       help="Create dynamic dictionaries")
    group.add_argument('-share_vocab', action='store_true',
                       help="Share source and target vocabulary")

    # Truncation options, for text corpus
    group = parser.add_argument_group('Pruning')
    group.add_argument('-src_seq_length', type=int, default=50,
                       help="Maximum source sequence length")
    group.add_argument('-src_seq_length_trunc', type=int, default=0,
                       help="Truncate source sequence length.")
    group.add_argument('-tgt_seq_length', type=int, default=50,
                       help="Maximum target sequence length to keep.")
    group.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                       help="Truncate target sequence length.")
    group.add_argument('-lower', action='store_true', help='lowercase data')

    # Data processing options
    group = parser.add_argument_group('Random')
    group.add_argument('-shuffle', type=int, default=1,
                       help="Shuffle data")
    group.add_argument('-seed', type=int, default=3435,
                       help="Random seed")

    group = parser.add_argument_group('Logging')
    group.add_argument('-report_every', type=int, default=100000,
                       help="Report status every this many sentences")
    group.add_argument('-log_file', type=str, default="",
                       help="Output logs to a file under this path.")

    # Options most relevant to speech
    group = parser.add_argument_group('Speech')
    group.add_argument('-sample_rate', type=int, default=16000,
                       help="Sample rate.")
    group.add_argument('-window_size', type=float, default=.02,
                       help="Window size for spectrogram in seconds.")
    group.add_argument('-window_stride', type=float, default=.01,
                       help="Window stride for spectrogram in seconds.")
    group.add_argument('-window', default='hamming',
                       help="Window type for spectrogram generation.")

    # Option most relevant to image input
    group.add_argument('-image_channel_size', type=int, default=3,
                       choices=[3, 1],
                       help="""Using grayscale image can training
                       model faster and smaller""")



def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_md_help_argument(parser)
    preprocess_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    check_existing_pt_files(opt)

    return opt


def build_save_in_shards_using_shards_size(src_corpus, tgt_corpus, fields,
                                           corpus_type, opt):
    """
    Divide src_corpus and tgt_corpus into smaller multiples
    src_copus and tgt corpus files, then build shards, each
    shard will have opt.shard_size samples except last shard.

    The reason we do this is to avoid taking up too much memory due
    to sucking in a huge corpus file.
    """

    with codecs.open(src_corpus, "r", encoding="utf-8") as fsrc:
        with codecs.open(tgt_corpus, "r", encoding="utf-8") as ftgt:
            logger.info("Reading source and target files: %s %s."
                        % (src_corpus, tgt_corpus))
            src_data = fsrc.readlines()
            tgt_data = ftgt.readlines()

            num_shards = int(len(src_data) / opt.shard_size)
            for x in range(num_shards):
                logger.info("Splitting shard %d." % x)
                f = codecs.open(src_corpus + ".{0}.txt".format(x), "w",
                                encoding="utf-8")
                f.writelines(
                        src_data[x * opt.shard_size: (x + 1) * opt.shard_size])
                f.close()
                f = codecs.open(tgt_corpus + ".{0}.txt".format(x), "w",
                                encoding="utf-8")
                f.writelines(
                        tgt_data[x * opt.shard_size: (x + 1) * opt.shard_size])
                f.close()
            num_written = num_shards * opt.shard_size
            if len(src_data) > num_written:
                logger.info("Splitting shard %d." % num_shards)
                f = codecs.open(src_corpus + ".{0}.txt".format(num_shards),
                                'w', encoding="utf-8")
                f.writelines(
                        src_data[num_shards * opt.shard_size:])
                f.close()
                f = codecs.open(tgt_corpus + ".{0}.txt".format(num_shards),
                                'w', encoding="utf-8")
                f.writelines(
                        tgt_data[num_shards * opt.shard_size:])
                f.close()

    src_list = sorted(glob.glob(src_corpus + '.*.txt'))
    tgt_list = sorted(glob.glob(tgt_corpus + '.*.txt'))

    ret_list = []

    for index, src in enumerate(src_list):
        logger.info("Building shard %d." % index)
        dataset = inputters.build_dataset(
            fields, opt.data_type,
            src_path=src,
            tgt_path=tgt_list[index],
            src_dir=opt.src_dir,
            src_seq_length=opt.src_seq_length,
            tgt_seq_length=opt.tgt_seq_length,
            src_seq_length_trunc=opt.src_seq_length_trunc,
            tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
            dynamic_dict=opt.dynamic_dict,
            sample_rate=opt.sample_rate,
            window_size=opt.window_size,
            window_stride=opt.window_stride,
            window=opt.window,
            image_channel_size=opt.image_channel_size
        )

        pt_file = "{:s}.{:s}.{:d}.pt".format(
            opt.save_data, corpus_type, index)

        # We save fields in vocab.pt seperately, so make it empty.
        dataset.fields = []

        logger.info(" * saving %sth %s data shard to %s."
                    % (index, corpus_type, pt_file))
        torch.save(dataset, pt_file)

        ret_list.append(pt_file)
        os.remove(src)
        os.remove(tgt_list[index])
        del dataset.examples
        gc.collect()
        del dataset
        gc.collect()

    return ret_list


def build_save_dataset(corpus_type, fields, opt):
    """ Building and saving the dataset """
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src_corpus = opt.train_src
        tgt_corpus = opt.train_tgt
    else:
        src_corpus = opt.valid_src
        tgt_corpus = opt.valid_tgt

    if (opt.shard_size > 0):
        return build_save_in_shards_using_shards_size(src_corpus,
                                                      tgt_corpus,
                                                      fields,
                                                      corpus_type,
                                                      opt)

    # For data_type == 'img' or 'audio', currently we don't do
    # preprocess sharding. We only build a monolithic dataset.
    # But since the interfaces are uniform, it would be not hard
    # to do this should users need this feature.
    dataset = inputters.build_dataset(
        fields, opt.data_type,
        src_path=src_corpus,
        tgt_path=tgt_corpus,
        src_dir=opt.src_dir,
        src_seq_length=opt.src_seq_length,
        tgt_seq_length=opt.tgt_seq_length,
        src_seq_length_trunc=opt.src_seq_length_trunc,
        tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
        dynamic_dict=opt.dynamic_dict,
        sample_rate=opt.sample_rate,
        window_size=opt.window_size,
        window_stride=opt.window_stride,
        window=opt.window,
        image_channel_size=opt.image_channel_size)

    # We save fields in vocab.pt seperately, so make it empty.
    dataset.fields = []

    pt_file = "{:s}.{:s}.pt".format(opt.save_data, corpus_type)
    logger.info(" * saving %s dataset to %s." % (corpus_type, pt_file))
    torch.save(dataset, pt_file)

    return [pt_file]


def build_save_vocab(train_dataset, fields, opt):
    """ Building and saving the vocab """
    fields = inputters.build_vocab(train_dataset, fields, opt.data_type,
                                   opt.share_vocab,
                                   opt.src_vocab,
                                   opt.src_vocab_size,
                                   opt.src_words_min_frequency,
                                   opt.tgt_vocab,
                                   opt.tgt_vocab_size,
                                   opt.tgt_words_min_frequency)

    # Can't save fields, so remove/reconstruct at training time.
    vocab_file = opt.save_data + '.vocab.pt'
    torch.save(inputters.save_fields_to_vocab(fields), vocab_file)


def main():
    opt = parse_args()

    if (opt.max_shard_size > 0):
        raise AssertionError("-max_shard_size is deprecated, please use \
                             -shard_size (number of examples) instead.")

    init_logger(opt.log_file)
    logger.info("Extracting features...")

    src_nfeats = inputters.get_num_features(
        opt.data_type, opt.train_src, 'src')
    tgt_nfeats = inputters.get_num_features(
        opt.data_type, opt.train_tgt, 'tgt')
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(opt.data_type, src_nfeats, tgt_nfeats)

    logger.info("Building & saving training data...")
    train_dataset_files = build_save_dataset('train', fields, opt)

    logger.info("Building & saving validation data...")
    build_save_dataset('valid', fields, opt)

    logger.info("Building & saving vocabulary...")
    build_save_vocab(train_dataset_files, fields, opt)




def randomize_smi(smi):
    random_equivalent_smiles = Chem.MolFromSmiles(Chem.MolToSmiles(smi, doRandom=True))
    return random_equivalent_smiles

class SmileTokenizer():
    def __init__(self):
        self.pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)
    def __call__(self, smi):
        tokens = [token for token in self.regex.findall(smi)]
        assert smi == ''.join(tokens)
        return ' '.join(tokens)

def tokenzie_smile(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

def train_test_split(datafile, file_base_paths='data', sampler = 1.0, test_size = 0.001, rseed=42):
    if not os.path.exists(file_base_paths):
        os.makedirs(file_base_paths)

    random.seed(rseed)
    tokenizer = SmileTokenizer()
    with open(datafile, 'r') as fin:
        with open(f"{file_base_paths}/src-train.txt", 'w') as src_train:
            with open(f"{file_base_paths}/tgt-train.txt", 'w') as tgt_train:
                with open(f"{file_base_paths}/src-val.txt", 'w') as src_val:
                    with open(f"{file_base_paths}/tgt-val.txt", 'w') as tgt_val:
                        for line in tqdm(fin):
                            if random.random() <= sampler:
                                molecule, scaffold = line.strip().split("\t")
                                molecule_tokens, scaffold_tokens = tokenizer(molecule), tokenizer(scaffold)

                                if random.random() > test_size: # goes into train
                                    src_train.write(f"{scaffold_tokens}\n")
                                    tgt_train.write(f"{molecule_tokens}\n")
                                else:
                                    src_val.write(f"{scaffold_tokens}\n")
                                    tgt_val.write(f"{molecule_tokens}\n")



if __name__ == '__main__':
    filename = "data/savi1to10_extended.txt"
    train_test_split(filename, file_base_paths='data/data_subsample', sampler=0.1)
    main()
