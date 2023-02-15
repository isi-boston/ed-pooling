import sys
import json
import traceback
from typing import Mapping
from better_events.better_core import BetterDocument


def load_documents(filename: str) -> Mapping[str, BetterDocument]:
    with open(filename, encoding='utf8') as f:
        data = json.load(f)
        results = {}
        for doc_id, doc in data.items():
            try:
                results[doc_id] = BetterDocument.from_json(doc)
            except Exception as e:
                sys.stderr.write(traceback.format_exc())
                sys.stderr.write(f"ERROR: COMPLETE FAILURE READING DOCUMENT {doc_id}\n")
        return results
