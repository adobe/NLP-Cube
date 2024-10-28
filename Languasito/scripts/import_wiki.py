# distutils: language = c++

import sys
import optparse
import os
import time

import tqdm
import string
import re
from collections import defaultdict


def _get_all_files(base_path):
    all_files = []
    for path, subdirs, files in os.walk(base_path):
        for name in files:
            fname = os.path.join(path, name)
            if not fname.endswith('.'):
                all_files.append(fname)
    return all_files


class Node:
    left = None
    right = None
    key = None
    value = None

    def __init__(self, key):
        self.key = key


class SortedDict:
    def __init__(self):
        self._root = None
        self._len = 0

    def __getitem__(self, key):
        return self._find_key(key, self._root).value

    def _find_key(self, key, node):
        while True:
            if node is None:
                raise KeyError
            if node.key == key:
                return node
            elif key < node.key:
                node = node.left
            else:
                node = node.right

    def _find_create_key(self, key, node, parent):
        while True:
            if node is None:
                if key < parent.key:
                    parent.left = Node(key)
                    self._len += 1
                    return parent.left
                else:
                    parent.right = Node(key)
                    self._len += 1
                    return parent.right
            if node.key == key:
                return node
            elif key < node.key:
                parent = node
                node = node.left
            else:
                parent = node
                node = node.right

    def __setitem__(self, key, value):
        if self._root is None:
            self._len = 1
            self._root = Node(key)
            self._root.value = value
        else:
            node = self._find_create_key(key, self._root, None)
            node.value = value

    def __contains__(self, key):
        try:
            _ = self._find_key(key, self._root)
            return True
        except Exception as _:
            return False

    def get(self, key, default=None):
        try:
            node = self._find_key(key, self._root)
            return node.value
        except:
            return default

    def __len__(self):
        return self._len

    def _walk(self, node, keys):
        if node is None or node.key is None:
            return
        n_list = []
        n_list.append(node)
        index = 0
        while index < len(n_list):
            node = n_list[index]
            keys.append(node.key)
            if node.left is not None:
                n_list.append(node.left)
            if node.right is not None:
                n_list.append(node.right)
            index += 1

    def __iter__(self):
        key_list = []
        self._walk(self._root, key_list)
        return key_list.__iter__()


def _update_maps(text, wordpair_counts, vocab, window=5):
    half_w = window // 2
    lines = text.split('\n')
    f_lines = []
    for line in lines:
        if not line.startswith('<doc id') and not line.startswith('</doc'):
            f_lines.append(line)
    text = '\n'.join(f_lines)
    s = text.replace('\n', ' ')
    s = ''.join(filter(lambda x: x.isalpha() or x.isspace(), s))
    words = s.replace('\n', ' ').split()
    for center in range(len(words)):
        dst_w = words[center]
        if dst_w not in vocab:
            continue
        dst_i = vocab[dst_w]
        if len(dst_w) < 2 or len(dst_w) > 20:
            continue
        if dst_i not in wordpair_counts:
            new_dict = defaultdict()
            wordpair_counts[dst_i] = new_dict
        else:
            new_dict = wordpair_counts[dst_i]

        for ii in range(max(0, center - half_w), min(len(words), center + half_w)):
            if ii != center:
                src_w = words[ii]
                if src_w not in vocab:
                    continue
                src_i = vocab[src_w]
                if len(src_w) < 2 or len(src_w) > 20:
                    continue
                new_dict[src_i] = new_dict.get(src_i, 0) + 1


def process_batch(files, map_train, map_dev, vocab, thread_id, params):
    sys.stdout.write(f'Thread {thread_id} started (waiting 3 seconds to settle)\n')
    sys.stdout.flush()
    time.sleep(3)
    for ii in range(len(files)):
        if (ii + 1) % params.ratio == 0:
            c_map = map_dev
        else:
            c_map = map_train
        f = open(files[ii], encoding='utf-8')
        text = f.read()
        f.close()
        _update_maps(text, c_map, vocab)
        if (ii + 1) % 10 == 0:
            sys.stdout.write(
                f"Thread {thread_id} processed {ii + 1}/{len(files)} files with ts={len(map_train)}, ds={len(map_dev)}\n")
            sys.stdout.flush()

    sys.stdout.write(
        f"Thread {thread_id} processed {ii + 1}/{len(files)} files with ts={len(map_train)}, ds={len(map_dev)}\n")
    sys.stdout.flush()
    sys.stdout.write(f'Thread {thread_id} finished\n')
    sys.stdout.flush()
    import joblib
    joblib.dump(map_train, f'm_t_{thread_id}')
    joblib.dump(map_dev, f'm_d_{thread_id}')


def _merge(dst_map, src_map):
    for w in src_map:
        if w not in dst_map:
            dst_map[w] = src_map[w]
        else:
            src = src_map[w]
            dst = dst_map[w]
            for w in src:
                dst[w] = dst.get(w, 0) + src[w]


def _build_vocab(files, threshold):
    vocab = defaultdict()
    pgb = tqdm.tqdm(range(len(files)))
    pgb.set_description("Building vocab lookup")
    for ii in pgb:
        text = open(files[ii]).read()

        lines = text.split('\n')
        f_lines = []
        for line in lines:
            if not line.startswith('<doc id') and not line.startswith('</doc'):
                f_lines.append(line)
        text = '\n'.join(f_lines)

        s = text.replace('\n', ' ')
        s = ''.join(filter(lambda x: x.isalpha() or x.isspace(), s))
        words = s.replace('\n', ' ').split()
        for w in words:
            vocab[w] = vocab.get(w, 0) + 1

    filtered_vocab = defaultdict()
    for w in vocab:
        if vocab[w] >= threshold:
            filtered_vocab[w] = len(filtered_vocab)
    return filtered_vocab


def _process(params):
    all_files = _get_all_files(params.wiki_base)

    # build vocab
    vocab = _build_vocab(all_files, params.threshold)
    # from sortedcontainers import SortedDict

    print("Updating word pairs")
    split_files = []
    bs = len(all_files) // params.threads
    for ii in range(params.threads - 1):
        split_files.append(all_files[ii * bs:ii * bs + bs])
    split_files.append(all_files[(params.threads - 1) * bs:])
    import multiprocessing as mp

    map_train = []
    map_dev = []
    thread_list = []
    for ii in range(len(split_files)):
        map_train.append(defaultdict(int))
        map_dev.append(defaultdict(int))
        # thread = threading.Thread(target=process_batch, args=(split_files[ii], map_train[-1], map_dev[-1], ii, params))
        thread = mp.Process(target=process_batch,
                            args=(split_files[ii], map_train[-1], map_dev[-1], vocab, ii, params))
        thread.start()
        thread_list.append(thread)
    index = 0
    import joblib
    for thread in thread_list:
        thread.join()
        map_train[index] = joblib.load(f'm_t_{index}')
        map_dev[index] = joblib.load(f'm_d_{index}')
        index += 1

    # merge data
    sys.stdout.write('Merging data\n')
    map_train_m = defaultdict()
    map_dev_m = defaultdict()
    index = 0
    for m_train, m_dev in zip(map_train, map_dev):
        index += 1
        sys.stdout.write(f'Merging {index}/{len(map_train)} - {len(m_train)}, {len(m_dev)}... ')
        sys.stdout.flush()
        _merge(map_train_m, m_train)
        _merge(map_dev_m, m_dev)
        sys.stdout.write('done\n')
        sys.stdout.flush()

    map_train = map_train_m
    map_dev = map_dev_m

    f_dev = open(params.dev_file, 'w')
    f_train = open(params.train_file, 'w')
    w_list = ['' for _ in vocab]
    for w in vocab:
        w_list[vocab[w]] = w
    for src_w in map_train:
        tmp = map_train[src_w]
        for dst_w in tmp:
            count = tmp[dst_w]
            if count > params.threshold:
                f_train.write(
                    f'{w_list[dst_w]}\t{w_list[src_w]}\t{count}\n')  # inversion between src and dst is intended

    for src_w in map_dev:
        tmp = map_dev[src_w]
        for dst_w in tmp:
            count = tmp[dst_w]
            if count > params.threshold:
                f_dev.write(f'{w_list[dst_w]}\t{w_list[src_w]}\t{count}\n')

    f_train.close()
    f_dev.close()


if __name__ == '__main__':

    parser = optparse.OptionParser()
    parser.add_option('--wiki', action='store', dest='wiki_base')
    parser.add_option('--train', action='store', dest='train_file')
    parser.add_option('--dev', action='store', dest='dev_file')
    parser.add_option('--ratio', action='store', default=100, type='int', dest='ratio',
                      help='train/dev ration (default=100)')
    parser.add_option('--threads', action='store', default=36, type='int', dest='threads',
                      help='train/dev ration (default=36)')
    parser.add_option('--threshold', action='store', default=10, type='int', dest='threshold',
                      help='keep threshold (default=10)')

    (params, _) = parser.parse_args(sys.argv)

    if params.wiki_base and params.train_file and params.dev_file:
        _process(params)
    else:
        parser.print_help()
