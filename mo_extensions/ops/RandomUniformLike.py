import numpy as np
from mo.graph.graph import Node, Graph
from mo.ops.op import Op

def shape_infer(node):
    # Inputs: [shape]
    # Output: tensor with the input shape
    assert(len(node.in_nodes()) == 1)
    #node.out_node(0).shape = node.in_node(0).shape  # NC
    print('===> Debug: node_in_node(0): ', node.in_node(0))
    node.out_node(0).shape = node.in_node(0).shape 
    #node.out_node(0).shape = node.in_node(0).data.get_value() 
    #print('===> Debug value: ', node.in_node(0).shape)
    #print('===> Debug value: ', node.in_node(0).shape)

class RandomUniformLike(Op):
    op = 'RandomUniformLike'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': shape_infer
        }, attrs)
