"""Offline checks for the single supported CNN and retired-model rejection."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.cnn import CNN, get_model, get_parameters, set_parameters
from experiments.artifacts import load_config


class CNNTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    def test_user_architecture_shapes_and_training_gradients(self):
        model = CNN()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 20104)
        self.assertEqual([type(layer) for layer in model.conv_block1],
                         [nn.Conv2d, nn.BatchNorm2d, nn.ReLU])
        self.assertEqual([type(layer) for layer in model.conv_block2],
                         [nn.Conv2d, nn.BatchNorm2d, nn.ReLU])
        shapes = {}
        handles = []
        for name in ("conv_block1", "conv_block2", "pool", "global_avg_pool"):
            handles.append(getattr(model, name).register_forward_hook(
                lambda module, args, output, key=name: shapes.update({key: tuple(output.shape)})))
        output = model(torch.randn(1, 3, 28, 28))
        for handle in handles:
            handle.remove()
        self.assertEqual(shapes, {"conv_block1": (1, 32, 28, 28),
                                 "conv_block2": (1, 64, 28, 28),
                                 "pool": (1, 64, 14, 14), "global_avg_pool": (1, 64, 1, 1)})
        self.assertEqual(tuple(output.shape), (1, 8))
        nn.CrossEntropyLoss()(output, torch.tensor([3])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        self.assertFalse(any(isinstance(m, (nn.Dropout, nn.LeakyReLU)) for m in model.modules()))

    def test_default_class_count_and_state_exchange(self):
        self.assertIsInstance(get_model(), CNN)
        model, clone = get_model(num_classes=7), get_model(num_classes=7)
        model(torch.randn(2, 3, 28, 28))  # Include updated BatchNorm buffers.
        set_parameters(clone, get_parameters(model))
        model.eval()
        clone.eval()
        inputs = torch.randn(2, 3, 28, 28)
        self.assertEqual(tuple(model(inputs).shape), (2, 7))
        self.assertTrue(torch.equal(model(inputs), clone(inputs)))
        for key in model.state_dict():
            self.assertTrue(torch.equal(model.state_dict()[key], clone.state_dict()[key]))

    def test_retired_models_and_options_fail_before_running(self):
        for name in ("resnet18", "resnet", "resnet-18", "bloodcnn", "custom_cnn"):
            with self.subTest(model=name):
                with self.assertRaises(ValueError):
                    get_model(model_name=name)
        for options in ({"pretrained": True}, {"dropout_rate": 0.4}):
            with self.assertRaises(ValueError):
                get_model(**options)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            for model in ({"name": "resnet18"}, {"name": "custom_cnn"},
                          {"pretrained": True}, {"dropout_rate": 0.4}):
                config.write_text(yaml.safe_dump({"model": model}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(config)


if __name__ == "__main__":
    unittest.main()
