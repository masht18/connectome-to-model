# Connectome CSV format

A connectome CSV describes a network of brain *areas*: how they connect and the shape of each
area. One **row per area**, and the columns split into two blocks.

## Layout

```
| <adjacency: first N columns> | <per-area parameter columns> |
```

where **N = the number of rows (areas)**.

### 1. Adjacency block (first N columns, position matters)

The first `N` columns form an `N x N` matrix. Entry `(row i, col j)` is the connection strength
from area `i` to area `j` (use `0`/`1` for absent/present). The column *headers* here are just
area names (e.g. `V1, V2, V4`) and are not interpreted — only that there are exactly `N` of them,
first, in order. Feedback is currently assumed reciprocal: a feedforward `i -> j` implies feedback
`j -> i` (see `Graph(reciprocal=...)`).

### 2. Per-area parameter columns (after the adjacency block; accessed by name)

These are read **by name**, so their order among themselves does not matter — but they must all
come *after* the `N` adjacency columns. All are required:

| Column | Meaning | Constraint |
|--------|---------|------------|
| `hidden_dim` | channels in the area's hidden state (≈ relative area size) | > 0 |
| `input_dim` | channels of the bottom-up input to the area | > 0 |
| `input_h`, `input_w` | spatial size (height, width) of the area | > 0 |
| `kernel_h`, `kernel_w` | convolution kernel / receptive-field size | > 0 |
| `basal_topdown_dim` | basal feedback channels (`0` = none) | ≥ 0 |
| `apical_topdown_dim` | apical/composite feedback channels (`0` = none) | ≥ 0 |

> **Note:** `output_h` / `output_w` columns appear in some older CSVs (e.g. `graphs/sample_graph.csv`)
> but are **ignored** by the loader — output spatial size is taken from the output area's
> `input_h`/`input_w`. You can leave them out.

## Validation

`Graph(...)` validates the CSV on load via `validate_connectome_csv` and raises a readable
`ValueError` naming the first problem (missing column, non-positive size, out-of-range
input/output node, etc.) instead of failing deep inside model construction.

## Minimal example (3 areas, A → B → C)

```csv
A,B,C,hidden_dim,input_dim,input_h,input_w,kernel_h,kernel_w,basal_topdown_dim,apical_topdown_dim
0,1,0,32,1,28,28,3,3,0,0
0,0,1,32,32,14,14,3,3,0,0
0,0,0,32,32,7,7,3,3,0,0
```

Here area `A` (row 0) feeds `B` (row 1) feeds `C` (row 2). With `input_nodes=[0]` the stimulus
enters `A`; with `output_nodes=[2]` the readout comes from `C`.
