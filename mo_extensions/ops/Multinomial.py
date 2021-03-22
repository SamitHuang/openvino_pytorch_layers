import numpy as np
from mo.graph.graph import Node, Graph
from mo.ops.op import Op

class Multinomial(Op):
    op = 'Multinomial'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': self.infer
        }, attrs)

    @staticmethod
    def infer(node):
        # Inputs: [batch_size, K]
        # Output: [batch_size, sample_size=1] 
        assert(len(node.in_nodes()) == 1)
        #print('===> Debug: node.in_node(0): ', node.in_node(0))
        node.out_node(0).shape = np.array([node.in_node(0).shape[0], 1])
        #print('===> Debug: node.out_node(0): ', node.out_node(0))
        #node.out_node(0).shape = node.in_node(0).data.get_value() 

