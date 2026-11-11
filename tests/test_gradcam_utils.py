import unittest

import torch
import torch.nn as nn

from visualization.gradcam_utils import GradCAM


class TinySpatialClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class GradCAMTest(unittest.TestCase):
    def test_standard_gradcam_produces_normalized_spatial_map(self):
        torch.manual_seed(0)
        model = TinySpatialClassifier().eval()
        image = torch.rand(1, 3, 11, 13)
        with GradCAM(model, model.features) as calculator:
            cam = calculator(image, target_class=1, output_size=(11, 13))
        self.assertEqual(cam.shape, (11, 13))
        self.assertTrue(float(cam.min()) >= 0.0)
        self.assertTrue(float(cam.max()) <= 1.0)


if __name__ == "__main__":
    unittest.main()
