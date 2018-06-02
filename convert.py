import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorflow as tf
import os
import keras.backend as K

class PytorchToKeras(object):
    def __init__(self,pModel,kModel):
        super(PytorchToKeras,self)
        self.__source_layers = []
        self.__target_layers = []
        self.pModel = pModel
        self.kModel = kModel

        tf.keras.backend.set_learning_phase(0)

    def __retrieve_k_layers(self):

        for i,layer in enumerate(self.kModel.layers):
            if len(layer.weights) > 0:
                self.__target_layers.append(i)

    def __retrieve_p_layers(self,input_size):

        input = torch.randn(input_size)

        input = Variable(input.unsqueeze(0))

        hooks = []

        def add_hooks(module):

            def hook(module, input, output):
                if hasattr(module,"weight"):
                    self.__source_layers.append(module)

            if not isinstance(module, nn.ModuleList) and not isinstance(module,nn.Sequential) and module != self.pModel:
                hooks.append(module.register_forward_hook(hook))

        self.pModel.apply(add_hooks)


        self.pModel(input)
        for hook in hooks:
            hook.remove()

    def convert(self,input_size):
        self.__retrieve_k_layers()
        self.__retrieve_p_layers(input_size)

        for i,(source_layer,target_layer) in enumerate(zip(self.__source_layers,self.__target_layers)):

            weight_size = len(source_layer.weight.data.size())

            transpose_dims = []

            for i in range(weight_size):
                transpose_dims.append(weight_size - i - 1)

            self.kModel.layers[target_layer].set_weights([source_layer.weight.data.numpy().transpose(transpose_dims), source_layer.bias.data.numpy()])

    def save_model(self,output_file):
        self.kModel.save(output_file)
    def save_weights(self,output_file):
        self.kModel.save_weights(output_file)



def keras_to_tensorflow(keras_model, output_dir, model_name,out_prefix="output_", log_tensorboard=True):

    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)

    out_nodes = []

    for i in range(len(keras_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(keras_model.output[i], out_prefix + str(i + 1))

    sess = K.get_session()

    from tensorflow.python.framework import graph_util, graph_io

    init_graph = sess.graph.as_graph_def()

    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)

    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)

    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard

        import_pb_to_tensorboard.import_to_tensorboard(
            os.path.join(output_dir, model_name),
            output_dir)

