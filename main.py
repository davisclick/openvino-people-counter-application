import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    #Connect to the MQTT client
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

      
def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    plugin = Network()
    client = connect_mqtt()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    
    single_image_mode = False
    cur_request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0
    current_count = 0
    

    #Load the model through `infer_network`
    plugin.load_model(args.model,args.cpu_extension,args.device,cur_request_id)
    
    
    net_input_shape = plugin.get_input_shape()
   
    #Handle the input stream
    if args.input == 'CAM':
        input_stream = 0
    elif args.input.endswith('.bmp') or args.input.endswith('.jpg'):
        single_image_mode = True
        input_stream = args.input
    else:
        input_stream = args.input
    
    cap = cv2.VideoCapture(input_stream)
    cap.open(input_stream)
    
    if not cap.isOpened():
        log.warning("Unable to open video source")
        
    # Grab the shape of the input 
    w = int(cap.get(3))
    h = int(cap.get(4))
    
    in_shape = net_input_shape['image_tensor']

    #Loop until stream is over
    while(cap.isOpened()):
        
        #Read from the video capture
        ret, frame = cap.read()
       
        if not ret:
            break
        key_pressed = cv2.waitKey(60)

        #Pre-process the image as needed
        p_frame = cv2.resize(frame, (in_shape[3], in_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        #Start asynchronous inference for specified request
        net_input = {'image_tensor': p_frame,'image_info': p_frame.shape[1:]}
        plugin.exec_net(net_input,cur_request_id)
        
        #TODO: Wait for the result
        inf_start = time.time()
        if plugin.wait(cur_request_id) == 0:
            
            det_time = time.time() - inf_start
           
            #Get the results of the inference request
            start_time = time.time()
            result = plugin.get_output()
            end_time = time.time()
            log.warning("Elapsed Time:", end_time-start_time)

            #Extract any desired stats from the results
            current_count = 0
            for obj in result[0][0]:
            # Draw bounding box for object when it's probability is more than
            #  the specified threshold
                if obj[2] > prob_threshold:
                    xmin = int(obj[3] * w)
                    ymin = int(obj[4] * h)
                    xmax = int(obj[5] * w)
                    ymax = int(obj[6] * h)
                
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    current_count = current_count + 1
            
            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            #Calculate and send relevant information on
            #current_count, total_count and duration to the MQTT server
            #Topic "person": keys of "count" and "total"
            # When new person enters the video
            if current_count > last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))

            # Person duration in the video is calculated
            if current_count < last_count:
                duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
                client.publish("person/duration",
                               json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count
            
            if key_pressed == 27:
                break

        #Send the frame to the FFMPEG server
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        #Write an output image if `single_image_mode`
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
        
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    plugin.clean()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
