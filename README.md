## 1 Description
Code base to convert csv connectome file into top-down and laterally recurrent artificial neural network.

## 2 Installation

TK

## 3 Connectome file

To start, fill out a csv connectome file, where each row corresponds to a connectome area. Given a connectome with N areas, the first N columns of each row must always contain connectivity data.

The following is an example for simple A-->B-->C connectome graph.

| A | B | C | hidden_dim |input_dim | input_h | input_w | kernel_h | kernel_w | basal_topdown_dim | apical_topdown_dim
| - | - | - | ---------- | ---------| --------| ---------- | ---------- | ---------- | ---------- | ---------- |
| 0 | 1 | 0 |     10     |    10    |    32   |     32     |    3      |     3     |     0     |     1     |
| 0 | 0 | 1 |     10     |    10    |    32   |     32     |    3      |     3     |     0     |     1     |
| 0 | 0 | 0 |     10     |    10    |    32   |     32     |    3      |     3     |     0     |     1     |

 In addition to connectivity data, tunable parameters for each area include:
- **hidden_dim:** relative size of area
- **kernel_h/w:** receptive field size of area
- **input_dim:** relative size of bottom-up inputs to area
- **basal_topdown_dim:** relative size of basal top-down inputs to the area
- **apical_topdown_dim:** relative size of threshold-shifting top-down inputs to the area
- **input h/w:** height and width of input to area

For more information on the different top-down mechanisms, refer to S1.

## 4 Connectome to graph

To turn the above connectome file into an aritifical neural network:
```
from model.graph import Graph, Architecture

input_node = [0, 2]                                  # Nodes which receive input, can be multiple
output_node = 1                                      # Output node
input_dims = [1, 0, 1]                               # Size of inputs to each of the nodes, leave 0 for nodes that do not receive input 
input_sizes = [(32, 32), (0, 0), (64, 64)]

# Load graph
graph_loc = '/path/to/connectome_file.csv'
graph = Graph(graph_loc, input_nodes=input_node, output_node=output_node)

# Build ANN
model = Architecture(graph, input_sizes, input_dims,
                    topdown=True).cuda().float()
```

## S1 Top-down feedback mechanisms

The ANN supports up to three different types biologically-based top-down feedback mechanisms.

1. **Default: Multiplicative/gain-modulating feedback.** Enabled by default. Can be turned off by passing topdown=False flag when building the Architecture. This type of feedback can only amplify extant signals. 
2. **Optional: Additive/threshold-shifting feedback.** Enabled if the connectome file has a non-zero positive apical_topdown_dim value. This feedback can cause silent neurons to fire or conversely silence a firing neuron if sufficiently strong.
3. **Optional: Basal feedback.** Enabled if the connectome file has a non-zero positive basal_topdown_dim value. These feedback signals are introduced much earlier in computation, at the same time as bottom-up inputs. Based on biological data from [Harris et al 2019](https://www.nature.com/articles/s41586-019-1716-z).
