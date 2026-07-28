import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np


def save_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def row_uid(row: Dict) -> int:
    return int(row.get("uid", row.get("user_id")))


def row_iid(row: Dict) -> int:
    return int(row.get("iid", row.get("item_id")))


class CollaborativePathAttributor:
    """Attribution for forget interaction -> CF neighborhood -> boundary path."""

    def __init__(self, split_data: Dict, dataset_data: Dict, args):
        self.split_data = split_data
        self.dataset_data = dataset_data
        self.args = args
        self.logs = {
            "retriever_path_available": False,
            "retriever_fallback": "disabled_or_unavailable",
        }
        self.user_sets = self._build_user_sets()
        self.item_users = self._build_item_users()
        self.retriever_probs = self._load_retriever_probs()

    def _build_user_sets(self) -> Dict[int, Set[int]]:
        train = self.dataset_data.get("retain_train", self.dataset_data.get("train", {}))
        return {int(uid): set(int(i) for i in seq) for uid, seq in train.items()}

    def _build_item_users(self) -> Dict[int, Set[int]]:
        item_users = defaultdict(set)
        for uid, items in self.user_sets.items():
            for iid in items:
                item_users[int(iid)].add(int(uid))
        return dict(item_users)

    def _load_retriever_probs(self):
        if not getattr(self.args, "enable_retriever_diagnosis", False):
            return None
        path = getattr(self.args, "llm_retrieved_path", None)
        if not path:
            return None
        try:
            with open(os.path.join(path, "retrieved.pkl"), "rb") as f:
                data = pickle.load(f)
            self.logs["retriever_path_available"] = True
            self.logs["retriever_fallback"] = None
            return {
                "val_probs": data.get("val_probs"),
                "test_probs": data.get("test_probs"),
            }
        except Exception as exc:
            self.logs["retriever_error"] = str(exc)
            return None

    def item_jaccard(self, a: int, b: int) -> float:
        ua = self.item_users.get(int(a), set())
        ub = self.item_users.get(int(b), set())
        if not ua or not ub:
            return 0.0
        return len(ua & ub) / len(ua | ub)

    def user_overlap(self, uid: int, forget_iid: int, candidate_iid: int) -> float:
        u_items = self.user_sets.get(int(uid), set())
        if not u_items:
            return 0.0
        present = float(candidate_iid in u_items)
        cooc = self.item_jaccard(forget_iid, candidate_iid)
        return float(np.clip(0.5 * present + 0.5 * cooc, 0.0, 1.0))

    def retriever_path_score(self, uid: int, candidate_iid: int) -> float:
        if not self.retriever_probs or not self.retriever_probs.get("val_probs"):
            return 0.0
        try:
            probs = self.retriever_probs["val_probs"][int(uid) - 1]
            if int(candidate_iid) < len(probs):
                return float(np.clip(probs[int(candidate_iid)], 0.0, 1.0))
        except Exception:
            return 0.0
        return 0.0

    def attribute(self, marginal_candidates: List[Dict]) -> Dict:
        records = []
        for cand in marginal_candidates:
            uid = int(cand["uid"])
            forget_iid = int(cand["forget_iid"])
            candidate_iid = int(cand["candidate_iid"])
            user_path = self.user_overlap(uid, forget_iid, candidate_iid)
            item_path = self.item_jaccard(forget_iid, candidate_iid)
            retriever_path = self.retriever_path_score(uid, candidate_iid)
            denom = (
                self.args.path_user_weight +
                self.args.path_item_weight +
                self.args.path_retriever_weight
            )
            if denom <= 0:
                denom = 1.0
            path_score = (
                self.args.path_user_weight * user_path +
                self.args.path_item_weight * item_path +
                self.args.path_retriever_weight * retriever_path
            ) / denom
            records.append({
                "uid": uid,
                "forget_iid": forget_iid,
                "candidate_iid": candidate_iid,
                "user_path_score": float(user_path),
                "item_path_score": float(item_path),
                "retriever_path_score": float(retriever_path),
                "collaborative_path_score": float(np.clip(path_score, 0.0, 1.0)),
            })
        return {
            "records": records,
            "logs": self.logs,
        }
