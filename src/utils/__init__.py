from .data_util import dataset
from .loss_util import supervised_loss, graph_loss, cluster_loss
from .mnist_util import DataUtils as mnistdata
from .metric_util import *

LayerEnd = 'keras'
if LayerEnd == 'keras':
    from .kenn_util import forward
elif LayerEnd == 'tensorflow':
    from .tfnn_util import forward, print_layer