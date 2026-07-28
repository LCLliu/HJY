import pickle
import shutil
import tempfile
import os
from pathlib import Path
import gzip
import csv
import json
from abc import *
from .utils import *
from config import RAW_DATASET_ROOT_FOLDER

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def load_ratings_df(self):
        pass

    @abstractmethod
    def maybe_download_raw_dataset(self):
        pass

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        self._attach_explicit_unlearning_split(dataset)
        return dataset

    def _attach_explicit_unlearning_split(self, dataset):
        forget_path = getattr(self.args, 'forget_interactions_path', None)
        retain_path = getattr(self.args, 'retain_interactions_path', None)
        metadata_path = getattr(self.args, 'split_metadata_path', None)

        if not forget_path and not retain_path and not metadata_path:
            return dataset

        split_metadata = {}
        metadata_base_dir = None
        if metadata_path:
            with open(metadata_path, 'r') as f:
                split_metadata = json.load(f)
            metadata_base_dir = os.path.dirname(os.path.abspath(metadata_path))
            files = split_metadata.get('files', {})
            forget_path = self._resolve_split_file(
                metadata_base_dir, forget_path,
                self._metadata_file_value(files, 'forget_interactions')
            )
            retain_path = self._resolve_split_file(
                metadata_base_dir, retain_path,
                self._metadata_file_value(files, 'retain_interactions')
            )

        train = dataset['train']
        forget_rows = self._metadata_rows(split_metadata, 'forget_interactions')
        retain_rows = self._metadata_rows(split_metadata, 'retain_interactions')
        if not forget_rows:
            forget_rows = self._load_interaction_file(forget_path) if forget_path else []
        if not retain_rows:
            retain_rows = self._load_interaction_file(retain_path) if retain_path else []

        if forget_rows:
            forget_train = self._rows_to_user_sequences(
                forget_rows, train, split_name='forget'
            )
            if retain_rows:
                retain_train = self._rows_to_user_sequences(
                    retain_rows, train, split_name='retain'
                )
            else:
                retain_train = self._derive_retain_from_forget(train, forget_rows)

            dataset['forget_train'] = forget_train
            dataset['retain_train'] = retain_train
            dataset['forget_set'] = self._rows_to_interaction_set(forget_rows)
            dataset['retain_set'] = self._rows_to_interaction_set(
                retain_rows if retain_rows else self._sequences_to_rows(retain_train, 'retain')
            )

        if retain_rows and not forget_rows:
            retain_train = self._rows_to_user_sequences(
                retain_rows, train, split_name='retain'
            )
            dataset['retain_train'] = retain_train
            dataset['retain_set'] = self._rows_to_interaction_set(retain_rows)

        if metadata_path:
            dataset['split_metadata'] = split_metadata
            files = split_metadata.get('files', {})
            dataset['overlap_retain_set'] = self._load_optional_split_set(
                metadata_base_dir, self._metadata_file_value(files, 'overlap_retain_interactions'), split_metadata,
                'overlap_retain_interactions'
            )
            dataset['semantic_neighbor_retain_set'] = self._load_optional_split_set(
                metadata_base_dir, self._metadata_file_value(files, 'semantic_neighbor_retain'), split_metadata,
                'semantic_neighbor_retain'
            )
            dataset['collaborative_neighbor_retain_set'] = self._load_optional_split_set(
                metadata_base_dir, self._metadata_file_value(files, 'collaborative_neighbor_retain'), split_metadata,
                'collaborative_neighbor_retain'
            )

        dataset['explicit_unlearning_split'] = bool(forget_path or retain_path or forget_rows or retain_rows)
        dataset['unlearning_split_diagnostics'] = self._split_diagnostics(
            dataset, forget_rows, retain_rows
        )
        return dataset

    def _resolve_split_file(self, base_dir, explicit_path, metadata_filename):
        if explicit_path:
            return explicit_path
        if not metadata_filename:
            return None
        return (metadata_filename if os.path.isabs(metadata_filename)
                else os.path.join(base_dir, metadata_filename))

    def _split_aliases(self, key):
        aliases = {
            'forget_interactions': ['forget_interactions', 'forget', 'forget_set'],
            'retain_interactions': ['retain_interactions', 'retain', 'retain_set'],
            'overlap_retain_interactions': ['overlap_retain_interactions', 'overlap_retain', 'overlap'],
            'semantic_neighbor_retain': ['semantic_neighbor_retain', 'semantic_retain', 'semantic_neighbors'],
            'collaborative_neighbor_retain': [
                'collaborative_neighbor_retain',
                'collaborative_retain',
                'collaborative_neighbors',
            ],
        }
        return aliases.get(key, [key])

    def _metadata_file_value(self, files, key):
        if not isinstance(files, dict):
            return None
        for alias in self._split_aliases(key):
            value = files.get(alias)
            if isinstance(value, str):
                return value
        return None

    def _metadata_rows(self, metadata, key):
        if not isinstance(metadata, dict):
            return []
        for alias in self._split_aliases(key):
            value = metadata.get(alias)
            if isinstance(value, list):
                return [self._normalize_interaction_row(row) for row in value]
            if isinstance(value, dict):
                for inner_key in ['interactions', key, *self._split_aliases(key)]:
                    if isinstance(value.get(inner_key), list):
                        return [self._normalize_interaction_row(row) for row in value[inner_key]]
        return []

    def _load_optional_split_set(self, base_dir, filename, metadata=None, key=None):
        rows = self._metadata_rows(metadata, key) if key else []
        if rows:
            return self._rows_to_interaction_set(rows)
        if not filename:
            return []
        path = filename if os.path.isabs(filename) else os.path.join(base_dir, filename)
        if not os.path.exists(path):
            return []
        rows = self._load_interaction_file(path)
        return self._rows_to_interaction_set(rows)

    def _split_diagnostics(self, dataset, forget_rows, retain_rows):
        forget_positions = {
            (int(row['user_id']), row.get('position'))
            for row in forget_rows
            if row.get('position') is not None
        }
        retain_positions = {
            (int(row['user_id']), row.get('position'))
            for row in retain_rows
            if row.get('position') is not None
        }
        retained_forget_positions = len(forget_positions & retain_positions)
        retained_forget_count = self._count_forget_interactions_in_retain_train(
            dataset.get('train', {}),
            dataset.get('retain_train', {}),
            forget_rows,
        )
        return {
            'num_forget_interactions': len(forget_rows),
            'num_retain_interactions': len(retain_rows),
            'retain_train_loaded_from_split': bool(retain_rows or forget_rows),
            'retain_train_excludes_forget_interactions': (
                retained_forget_positions == 0 and retained_forget_count == 0
            ),
            'retained_forget_position_count': retained_forget_positions,
            'forgotten_interactions_in_retain_train': retained_forget_count,
            'forgotten_interaction_positions_in_retain_split': retained_forget_positions,
            'retain_train_fallback_full_train': not bool(retain_rows or forget_rows),
        }

    def _count_forget_interactions_in_retain_train(self, train, retain_train, forget_rows):
        count = 0
        for row in forget_rows:
            uid = int(row['user_id'])
            iid = int(row['item_id'])
            pos = row.get('position')
            retain_seq = [int(i) for i in retain_train.get(uid, [])]
            if pos is not None:
                full_seq = [int(i) for i in train.get(uid, [])]
                full_count = sum(1 for item in full_seq if item == iid)
                retain_count = sum(1 for item in retain_seq if item == iid)
                if retain_count >= full_count and full_count > 0:
                    count += 1
            elif iid in retain_seq:
                count += 1
        return count

    def _load_interaction_file(self, path):
        if path is None:
            return []
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Interaction split file not found: {path}')

        ext = os.path.splitext(path)[1].lower()
        if ext == '.csv':
            with open(path, newline='') as f:
                return [self._normalize_interaction_row(row)
                        for row in csv.DictReader(f)]
        if ext == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in ['interactions', 'forget_interactions', 'retain_interactions']:
                    if key in data:
                        data = data[key]
                        break
            if not isinstance(data, list):
                raise ValueError(f'JSON split must contain a list of interactions: {path}')
            return [self._normalize_interaction_row(row) for row in data]

        raise ValueError(f'Unsupported interaction split format: {path}')

    def _normalize_interaction_row(self, row):
        def parse_optional_int(value):
            if value is None or value == '' or value == 'null':
                return None
            return int(value)

        def first_nonempty(*values):
            for value in values:
                if value is not None and value != '' and value != 'null':
                    return value
            return None

        uid = first_nonempty(row.get('uid'), row.get('user_id'))
        iid = first_nonempty(row.get('iid'), row.get('item_id'))
        position = row.get('position', row.get('sequence_index'))

        out = {
            'interaction_id': row.get('interaction_id', ''),
            'forget_id': row.get('forget_id', ''),
            'uid': int(uid),
            'iid': int(iid),
            'user_id': int(uid),
            'item_id': int(iid),
            'rating': row.get('rating', ''),
            'timestamp': row.get('timestamp', ''),
            'position': parse_optional_int(position),
            'sequence_index': parse_optional_int(row.get('sequence_index', position)),
            'raw_index': parse_optional_int(row.get('raw_index')),
            'split_name': row.get('split_name', ''),
        }
        for key, value in row.items():
            if key not in out:
                out[key] = value
        return out

    def _rows_to_user_sequences(self, rows, train, split_name):
        by_user = {uid: [] for uid in train.keys()}
        train_lookup = self._build_train_lookup(train)

        for row in rows:
            uid = row['user_id']
            iid = row['item_id']
            pos = row.get('position')
            if uid not in train:
                raise ValueError(f'{split_name} interaction has unknown user_id={uid}')
            if pos is not None:
                if pos < 0 or pos >= len(train[uid]) or train[uid][pos] != iid:
                    raise ValueError(
                        f'{split_name} interaction does not match train sequence: '
                        f'user_id={uid}, item_id={iid}, position={pos}'
                    )
                resolved_pos = pos
            else:
                positions = train_lookup.get((uid, iid), [])
                if not positions:
                    raise ValueError(
                        f'{split_name} interaction not found in train: '
                        f'user_id={uid}, item_id={iid}'
                    )
                resolved_pos = positions.pop(0)
            by_user[uid].append((resolved_pos, iid))

        return {
            uid: [iid for _, iid in sorted(items, key=lambda x: x[0])]
            for uid, items in by_user.items()
        }

    def _derive_retain_from_forget(self, train, forget_rows):
        forget_positions = {uid: set() for uid in train.keys()}
        train_lookup = self._build_train_lookup(train)
        for row in forget_rows:
            uid = row['user_id']
            iid = row['item_id']
            pos = row.get('position')
            if pos is None:
                positions = train_lookup.get((uid, iid), [])
                if not positions:
                    raise ValueError(
                        f'forget interaction not found in train: user_id={uid}, item_id={iid}'
                    )
                pos = positions.pop(0)
            forget_positions[uid].add(pos)

        return {
            uid: [iid for pos, iid in enumerate(seq) if pos not in forget_positions[uid]]
            for uid, seq in train.items()
        }

    def _build_train_lookup(self, train):
        lookup = {}
        for uid, seq in train.items():
            for pos, iid in enumerate(seq):
                lookup.setdefault((uid, iid), []).append(pos)
        return {key: list(value) for key, value in lookup.items()}

    def _sequences_to_rows(self, sequences, split_name):
        rows = []
        for uid, seq in sequences.items():
            for pos, iid in enumerate(seq):
                rows.append({
                'user_id': uid,
                'uid': uid,
                'item_id': iid,
                'iid': iid,
                'position': pos,
                'sequence_index': pos,
                'split_name': split_name,
            })
        return rows

    def _rows_to_interaction_set(self, rows):
        return [
            {
                'user_id': row['user_id'],
                'uid': row.get('uid', row['user_id']),
                'item_id': row['item_id'],
                'iid': row.get('iid', row['item_id']),
                'position': row.get('position'),
                'sequence_index': row.get('sequence_index', row.get('position')),
                'interaction_id': row.get('interaction_id', ''),
                'forget_id': row.get('forget_id', ''),
                'split_name': row.get('split_name', ''),
                **{
                    k: v for k, v in row.items()
                    if k not in {'user_id', 'item_id', 'position', 'split_name'}
                },
            }
            for row in rows
        ]

    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 1 or self.min_uc > 1:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            while len(good_items) < len(item_sizes) or len(good_users) < len(user_sizes):
                if self.min_sc > 1:
                    item_sizes = df.groupby('sid').size()
                    good_items = item_sizes.index[item_sizes >= self.min_sc]
                    df = df[df['sid'].isin(good_items)]

                if self.min_uc > 1:
                    user_sizes = df.groupby('uid').size()
                    good_users = user_sizes.index[user_sizes >= self.min_uc]
                    df = df[df['uid'].isin(good_users)]

                item_sizes = df.groupby('sid').size()
                good_items = item_sizes.index[item_sizes >= self.min_sc]
                user_sizes = df.groupby('uid').size()
                good_users = user_sizes.index[user_sizes >= self.min_uc]
        return df
    
    def densify_index(self, df):
        print('Densifying index')
        umap = {u: i for i, u in enumerate(set(df['uid']), start=1)}
        smap = {s: i for i, s in enumerate(set(df['sid']), start=1)}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def split_df(self, df, user_count):
        print('Splitting')
        user_group = df.groupby('uid')
        user2items = user_group.progress_apply(
            lambda d: list(d.sort_values(by=['timestamp', 'sid'])['sid']))
        train, val, test = {}, {}, {}
        retain_train, forget_train = {}, {}
        forget_ratio = getattr(self.args, 'forget_ratio', 0.0)
        forget_rng = np.random.RandomState(getattr(self.args, 'forget_seed', 42))

        for i in range(user_count):
            user = i + 1
            items = user2items[user]
            train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]

            if forget_ratio > 0 and len(train[user]) > 1:
                user_full_train = train[user]
                n_forget = max(1, int(len(user_full_train) * forget_ratio))
                n_forget = min(n_forget, len(user_full_train) - 1)

                indices = list(range(len(user_full_train)))
                forget_rng.shuffle(indices)
                forget_idx = set(indices[:n_forget])
                retain_train[user] = [user_full_train[j] for j in range(len(user_full_train))
                                      if j not in forget_idx]
                forget_train[user] = [user_full_train[j] for j in range(len(user_full_train))
                                      if j in forget_idx]
            else:
                retain_train[user] = train[user]
                forget_train[user] = []

        return train, val, test, retain_train, forget_train

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc)
        forget_ratio = getattr(self.args, 'forget_ratio', 0.0)
        if forget_ratio > 0:
            folder_name += '_unlearn{}'.format(forget_ratio)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')
