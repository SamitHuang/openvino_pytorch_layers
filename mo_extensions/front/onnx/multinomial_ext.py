from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from ...ops.Multinomial import Multinomial 

class MultinomialFrontExtractor(FrontExtractorOp):
    op = 'Multinomial'
    enabled = True

    @classmethod
    def extract(cls, node):
        Multinomial.update_node_stat(node)
        '''
        sample_size = onnx_attr(node, 'sample_size', 'ints', default=1)
        seed = onnx_attr(node, 'seed', 'f', default=1.0)
        dtype = onnx_attr(node, 'dtype', 'ints', default=6) # 6 for int
        attrs = {
                'sample_size': sample_size,
                'seed': seed}
        Multinomial.update_node_stat(node, attrs)
        '''
        return cls.enabled
