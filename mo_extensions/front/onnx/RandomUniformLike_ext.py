from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from ...ops.RandomUniformLike import RandomUniformLike 

class RandomUniformLikeFrontExtractor(FrontExtractorOp):
    op = 'RandomUniformLike'
    enabled = True

    @classmethod
    def extract(cls, node):
        RandomUniformLike.update_node_stat(node)
        '''
        low = onnx_attr(node, 'low', 'f', default=0)
        high = onnx_attr(node, 'high', 'f', default=1.0)
        attrs = {
                'low': low,
                'high': high}
        RandomUniformLike.update_node_stat(node, attrs)
        '''
        return cls.enabled
