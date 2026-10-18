"""CPU tests for true-2.5D windowing (adjacent-slice RGB)."""
import numpy as np

from src.data.dicom import adjacent_windows, imagenet_norm_chw


def test_adjacent_windows_uses_neighbors():
    # 5 distinct slices, each filled with its index so channels are identifiable
    norm = np.stack([np.full((2, 2), i, dtype=np.float32) for i in range(5)])  # (5,2,2)
    win = adjacent_windows(norm, centers=[2])  # single center at index 2
    assert win.shape == (1, 3, 2, 2)
    assert win[0, 0, 0, 0] == 1  # c-1
    assert win[0, 1, 0, 0] == 2  # c
    assert win[0, 2, 0, 0] == 3  # c+1


def test_adjacent_windows_clamps_edges():
    norm = np.stack([np.full((2, 2), i, dtype=np.float32) for i in range(4)])
    win = adjacent_windows(norm, centers=[0, 3])  # first and last
    # center 0: lo clamps to 0 -> channels (0,0,1)
    assert (win[0, 0] == 0).all() and (win[0, 1] == 0).all() and (win[0, 2] == 1).all()
    # center 3: hi clamps to 3 -> channels (2,3,3)
    assert (win[1, 0] == 2).all() and (win[1, 1] == 3).all() and (win[1, 2] == 3).all()


def test_true_2p5d_channels_differ_unlike_fake():
    norm = np.stack([np.full((2, 2), i, dtype=np.float32) for i in range(5)])
    win = adjacent_windows(norm, centers=[2])[0]  # (3,2,2)
    # the whole point: the 3 channels are DIFFERENT slices (cross-slice context)
    assert not np.allclose(win[0], win[1]) and not np.allclose(win[1], win[2])


def test_imagenet_norm_chw_shape():
    chw = np.random.rand(3, 8, 8).astype(np.float32)
    out = imagenet_norm_chw(chw)
    assert out.shape == (3, 8, 8)
    assert not np.allclose(out, chw)  # normalization changed values
