"""
Minimal smoke tests: build a model from a connectome CSV and run a forward pass on CPU,
and check that the CSV validator rejects malformed input. Runs under pytest, or directly:

    python tests/test_smoke.py
"""

import os
import re
import tempfile
import contextlib

import torch
import pandas as pd

try:
    import pytest
except ImportError:  # allow running directly without pytest installed
    pytest = None

from connectome_to_model.model.graph import Graph, validate_connectome_csv
from connectome_to_model.model.architectures import ConnectomicsConvGRU


@contextlib.contextmanager
def assert_raises(exc_type, match=None):
    """pytest.raises when available, else a minimal stand-in so the file runs standalone."""
    if pytest is not None:
        with pytest.raises(exc_type, match=match):
            yield
        return
    try:
        yield
    except exc_type as e:
        if match is not None:
            assert re.search(match, str(e)), f"{match!r} not found in {str(e)!r}"
    else:
        raise AssertionError(f"Expected {exc_type.__name__} was not raised")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_CSV = os.path.join(REPO_ROOT, 'graphs', 'sample_graph.csv')


def test_build_and_forward_cpu():
    """Build the sample A->B->C graph and run one forward pass entirely on CPU."""
    graph = Graph(SAMPLE_CSV, input_nodes=[0], output_nodes=[2],
                  dtype=torch.FloatTensor)  # forces CPU hidden states
    model = ConnectomicsConvGRU(graph, graph.find_input_sizes(), graph.find_input_dims(),
                                dropout=False).to('cpu').float()

    batch, seq_len = 2, 4
    in_dim = graph.nodes[0].input_dim
    h, w = graph.nodes[0].input_size
    x = torch.randn(batch, seq_len, in_dim, h, w)

    out = model([x])

    out_h, out_w = graph.nodes[2].input_size
    expected = (batch, graph.nodes[2].hidden_dim, out_h, out_w)
    assert tuple(out.shape) == expected, f"got {tuple(out.shape)}, expected {expected}"


def _write_csv(rows, columns):
    df = pd.DataFrame(rows, columns=columns)
    fd, path = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    df.to_csv(path, index=False)
    return path


def test_validator_rejects_missing_column():
    # Two areas, adjacency A/B, but 'kernel_w' is missing.
    cols = ['A', 'B', 'hidden_dim', 'input_dim', 'input_h', 'input_w',
            'kernel_h', 'basal_topdown_dim', 'apical_topdown_dim']
    rows = [[0, 1, 8, 1, 28, 28, 3, 0, 0],
            [0, 0, 8, 1, 28, 28, 3, 0, 0]]
    path = _write_csv(rows, cols)
    try:
        with assert_raises(ValueError, match="kernel_w"):
            Graph(path, input_nodes=[0], output_nodes=[1])
    finally:
        os.remove(path)


def test_validator_rejects_nonpositive_size():
    cols = ['A', 'B', 'hidden_dim', 'input_dim', 'input_h', 'input_w',
            'kernel_h', 'kernel_w', 'basal_topdown_dim', 'apical_topdown_dim']
    rows = [[0, 1, 8, 1, 0, 28, 3, 3, 0, 0],   # input_h = 0 is invalid
            [0, 0, 8, 1, 28, 28, 3, 3, 0, 0]]
    df = pd.DataFrame(rows, columns=cols)
    with assert_raises(ValueError, match="input_h"):
        validate_connectome_csv(df, input_nodes=[0], output_nodes=[1])


def test_validator_rejects_out_of_range_node():
    cols = ['A', 'B', 'hidden_dim', 'input_dim', 'input_h', 'input_w',
            'kernel_h', 'kernel_w', 'basal_topdown_dim', 'apical_topdown_dim']
    rows = [[0, 1, 8, 1, 28, 28, 3, 3, 0, 0],
            [0, 0, 8, 1, 28, 28, 3, 3, 0, 0]]
    df = pd.DataFrame(rows, columns=cols)
    with assert_raises(ValueError, match="output_nodes"):
        validate_connectome_csv(df, input_nodes=[0], output_nodes=[5])


if __name__ == "__main__":
    test_build_and_forward_cpu()
    test_validator_rejects_missing_column()
    test_validator_rejects_nonpositive_size()
    test_validator_rejects_out_of_range_node()
    print("All smoke tests passed.")
