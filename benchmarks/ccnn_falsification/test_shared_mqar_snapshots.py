"""CI contract: published Zoology baselines must consume exact CCNN snapshots."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from run_official_zoology_fig2_baselines import (
    UPSTREAM_SHA, frozen_dataloader, read_frozen_snapshot
)


class FrozenMQARContracts(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.manifest = {
            "schema":"frozen_mqar_v1",
            "seed":11,
            "upstream_commit":UPSTREAM_SHA,
            "records":[]
        }
        for j, name in enumerate(("train","val","cal","test")):
            x = torch.full((3, 64), 100+j, dtype=torch.long)
            y = torch.full((3, 64), -100, dtype=torch.long)
            y[:, 14] = 4096 + j
            dig = hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()
            fn = f"seed11_{name}_L64_K4.pt"
            torch.save({"inputs":x,"labels":y},self.root/fn)
            self.manifest["records"].append({
                "seed":11,"split":name,"length":64,"pairs":4,
                "num_examples":3,"filename":fn,"sha256":dig
            })
        (self.root/"seed11_manifest.json").write_text(json.dumps(self.manifest))

    def tearDown(self):
        self.dir.cleanup()

    def test_load_same_frozen_examples_and_hashes(self):
        dl,digest = frozen_dataloader(self.root,11,"train",64,4,batch_size=2)
        rows=list(dl)
        self.assertEqual(len(rows),2)
        x = torch.cat([r[0] for r in rows])
        y = torch.cat([r[1] for r in rows])
        original,truth,expected = read_frozen_snapshot(self.root,11,"train",64,4)
        self.assertTrue(torch.equal(x,original))
        self.assertTrue(torch.equal(y,truth))
        self.assertEqual(digest,expected)
        self.assertEqual(len(rows[1][2]),1)
        self.assertEqual(rows[0][2][0]["num_kv_pairs"],4)

    def test_val_and_locked_test_are_different(self):
        _,val,_ = read_frozen_snapshot(self.root,11,"val",64,4)
        _,test,_ = read_frozen_snapshot(self.root,11,"test",64,4)
        self.assertFalse(torch.equal(val,test))

    def test_mutation_or_missing_manifest_is_rejected(self):
        path = self.root/"seed11_test_L64_K4.pt"
        saved=torch.load(path,weights_only=True)
        saved["inputs"][0,0]+=1
        torch.save(saved,path)
        with self.assertRaises(ValueError):
            read_frozen_snapshot(self.root,11,"test",64,4)
        with self.assertRaises(FileNotFoundError):
            read_frozen_snapshot(self.root,23,"train",64,4)

    def test_source_revision_is_verified(self):
        man=json.loads((self.root/"seed11_manifest.json").read_text())
        man["upstream_commit"]="wrong"
        (self.root/"seed11_manifest.json").write_text(json.dumps(man))
        with self.assertRaises(ValueError):
            read_frozen_snapshot(self.root,11,"train",64,4)


if __name__ == "__main__":
    unittest.main()
