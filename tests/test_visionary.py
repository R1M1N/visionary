"""
Testing Framework for Visionary

Provides comprehensive pytest suite covering core classes,
methods, fixture management for test data, and performance benchmarks.
"""

import sys
import os

# Add src folder to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


import pytest
import numpy as np
import cv2
import os
from visionary.utils.file import FileUtils
from visionary.utils.image import ImageUtils
from visionary.classification.core import Classification, Classifications

# Fixtures
@pytest.fixture(scope='module')
def sample_image():
    # Generate a dummy image
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture(scope='module')
def sample_classifications():
    classifications = Classifications()
    classifications.add(0, 'cat', 0.95)
    classifications.add(1, 'dog', 0.85)
    return classifications

# Test FileUtils
def test_list_files(tmp_path):
    d = tmp_path
    (d / "a.jpg").write_text("content")
    (d / "b.txt").write_text("content")
    jpg_files = FileUtils.list_files(str(d), extensions=['.jpg'])
    assert len(jpg_files) == 1
    assert jpg_files[0].name == 'a.jpg'

# Test ImageUtils
def test_load_image_and_preprocess(sample_image):
    image = sample_image
    assert image.shape == (100, 100, 3)
    processed = ImageUtils.preprocess_image(image, size=(50, 50), normalize=True)
    assert processed.shape == (50, 50, 3) and processed.dtype == np.float32

# Test Classification
def test_classifications(sample_classifications):
    top_class = sample_classifications.top_k(1)[0]
    assert top_class.class_name == 'cat'
    assert top_class.confidence == 0.95

# Performance benchmark example
import time

def test_image_preprocessing_performance(sample_image):
    image = sample_image
    start = time.time()
    for _ in range(100):
        ImageUtils.preprocess_image(image, size=(224, 224), normalize=True)
    end = time.time()
    duration = end - start
    assert duration < 1.0  # Expect under 1 second for 100 iterations
