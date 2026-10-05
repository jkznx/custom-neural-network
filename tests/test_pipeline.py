import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from dataset import load_polygons,native_mask
from evaluate import counts,ratio,is_detected
from model import LaneNet
from common import loss_function


class PipelineTests(unittest.TestCase):
    def test_polygon_union_and_bad_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'label.txt'
            path.write_text('0 0 0 0.8 0 0.8 0.8 0 0.8\n0 0.2 0.2 1 0.2 1 1 0.2 1')
            mask=np.asarray(native_mask(path,(20,20)))
            self.assertEqual(mask[10,10],255)
            path.write_text('0 .5 .5 .4 .4')
            with self.assertRaises(ValueError): load_polygons(path)
            path.write_text('0 nan 0 1 0 1 1')
            with self.assertRaises(ValueError): load_polygons(path)
            path.write_text('')
            self.assertEqual(load_polygons(path),[])
            path.unlink()
            with self.assertRaises(FileNotFoundError): load_polygons(path)

    def test_confusion_metrics(self):
        pred=np.array([[1,1],[0,0]])
        truth=np.array([[1,0],[1,0]])
        self.assertEqual(counts(pred,truth),(1,1,1,1))
        self.assertAlmostEqual(ratio(1,3),1/3)
        self.assertEqual(ratio(0,0,1),1)

    def test_forward_backward_all_layers(self):
        torch.set_num_threads(2)
        model=LaneNet()
        image=torch.randn(1,3,36,64)
        out=model(image)
        self.assertEqual(tuple(out.shape),(1,1,36,64))
        loss=loss_function(out,torch.zeros_like(out))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_detection_threshold_strictly_greater(self):
        self.assertFalse(is_detected(.6,True))
        self.assertTrue(is_detected(.60001,True))
        self.assertFalse(is_detected(1.,False))


if __name__=='__main__':
    unittest.main()
