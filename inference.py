#!/usr/bin/env python3
import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        #Initialize any class variables desired
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request_handle = None
    
    def performance_counter(self, request_id):
        """
        Queries performance measures per layer to get feedback of what is the
        most time consuming layer.
        :param request_id: Index of Infer request value. Limited to device capabilities
        :return: Performance of the layer  
        """
        perf_count = self.exec_network.requests[request_id].get_perf_counts()
        return perf_count
    
    def load_model(self, model,cpu_extension, device, num_requests):
    
        #Load the model
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        # Initialize the plugin
        self.plugin = IECore()
        
        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        #Check for supported layers
        supported_layers = self.plugin.query_network(self.network, device_name="CPU")
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        
        if len(unsupported_layers) != 0:
            log.warning("Unsupported layers found: {}".format(unsupported_layers))
            log.warning("Check whether extensions are available to add to IECore.")
            #Add any necessary extensions
            if cpu_extension and "CPU" in device:
                self.plugin.add_extension(cpu_extension, device)
        
        self.exec_network = self.plugin.load_network(self.network, device)
        
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        
        #Return the loaded inference plugin
        
        return

    def get_input_shape(self):
        #Return the shape of the input layer
        input_shapes = {}
        for inp in self.network.inputs:
            input_shapes[inp] = (self.network.inputs[inp].shape)
        return input_shapes

    def exec_net(self,net_input,request_id):
        #tart an asynchronous request
        #Return any necessary information
        self.infer_request_handle = self.exec_network.start_async(
                request_id, 
                inputs=net_input)
        return

    def wait(self,request_id):
        #Wait for the request to be complete.
        #Return any necessary information
        status = self.exec_network.requests[0].wait(-1)
        return status
        

    def get_output(self):
        #Extract and return the output results
        return self.exec_network.requests[0].outputs[self.output_blob]
        
    def clean(self):
        del self.net_plugin
        del self.plugin
        del self.net
