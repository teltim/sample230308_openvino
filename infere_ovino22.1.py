"""
"""
import os, sys
import math, random, cv2
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

#import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
import torchvision

#from openvino.runtime import Core
from openvino.runtime import Core, PartialShape

#from PIL import Image
from PIL import Image, ImageFilter

if 1:
   if 1: # add for my test
        # int8 model via onnx runtime
        # quantize/dequantize linear layers
        model = "model.test-int8-Alex.onnx-rt-int8.cut.onnx"

        # int8 model via nncf package ovino22.1
        # fake-quantized layer
        #model = "model.test-int8-Alex.ovino-nncrf-int8.ovino22.1_mo.cut" + ".xml"

        # image 
        fig = "Zebra-1.jpg"
        #fig = "Accordion-842.jpg"

        np_save_name = "save_np_ovino22.1_" + fig + "_" + model

inmodel = "sample-model/"+model
baseImg = "sample-image/" + fig

input_size = (2048, 2048)
press_size = (255, 255)

normalize_GoogleOpenImage = torchvision.transforms.Normalize(
            mean=[0.4900, 0.4252, 0.3741],
            std=[0.2741, 0.2673, 0.2739])

transform_resize_totensor_donorm = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size[0],interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        #torchvision.transforms.Resize(input_size[0],interpolation=torchvision.transforms.InterpolationMode.NEAREST),
        #torchvision.transforms.Resize(input_size[0],interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.ToTensor(),
        normalize_GoogleOpenImage,
])

def get_input_image(fname, cname):
    print("\n>>>>> get_input_image")
    if 1: # use PIL, faster than cv2
        #image = Image.open(fname)
        image = Image.open(fname).convert('RGB')

        # do transform and add batch size
        # w/ normal process
        input_image = transform_resize_totensor_donorm( image )

        input_image = np.expand_dims(input_image, 0) 
        print("\ninput image = \n", input_image[0:1,0:1,0:1,0:9], "\n\n")
    return input_image 


def get_cutting_model(inmodel):
    print("\n>>>>> CALLED get_cutting_model()")
    core = Core()
    ir_model = core.read_model(model=inmodel)

    #ir_model.add_outputs([("cv1/WithoutBiases", 0)])
    #print(f"ir_model = {ir_model}")
    #print(f"len(ir_model.inputs) = {len(ir_model.inputs)}")

    if 1:
        #ir_input_layer = list(ir_model.inputs.keys())[0]
        ir_input_layer1 = next(iter(ir_model.inputs))
        ir_output_layer = next(iter(ir_model.outputs))
        print(f"ir_input_layer1 = {ir_input_layer1.any_name}")
        print(f"ir_output_layer = {ir_output_layer.any_name}")

    print("\n>>>>> ORIGINAL MODEL")
    print(f"input shape: {ir_input_layer1.shape}")
    print(f"output shape: {ir_output_layer.shape}")

    new_shape = PartialShape([1, 3, input_size[0], input_size[1]])

    ir_model.reshape({ir_input_layer1.any_name: new_shape})

    ir_compiled_model = core.compile_model(model=ir_model, device_name="CPU")

    print("\n>>>>> RESHAPED MODEL")
    print(f"model input shape: {ir_input_layer1.shape}")
    print(f"compiled_model input shape: {ir_compiled_model.input(index=0).shape}")
    print(f"compiled_model output shape: {ir_output_layer.shape}")
    return ir_model, ir_compiled_model


def get_feature_pixel(ir_model, ir_compiled_model, input_image, name, draw):
    print("\n>>>>> CALLED: get_feature_pixel()")
    print(f"ir_model = {ir_model}")

    if 1: # 1 input
        #ir_input_layer = next(iter(ir_model.inputs))
        input0Name = ir_model.input().get_any_name()
        input_data = {input0Name:input_image}

    method = cv2.INTER_NEAREST
    #method = cv2.INTER_LINEAR
    #method = cv2.INTER_CUBIC
    #method = cv2.INTER_LANCZOS4
    #method = cv2.INTER_AREA

    # do synchronous inference
    res = ir_compiled_model.infer_new_request(input_data)

    # TypeError: Inputs should be either list or dict! Current type: <class 'numpy.ndarray'>
    #print(type(res))
    # <class 'dict'>
    #print(res)

    matrix_list = list(res.values())
    matrix_len  = len(matrix_list) 

    if 1: # N output
        print(">>> this is N output")
        feature_list = [] # prepare list
        for i in range( matrix_len ):

            matrix = np.squeeze(matrix_list[i]) # remove batch size since this is a single oeration:q
            nite = matrix.shape[0] # n feature map
            #print(f"conv {i} features: {nite}, shape: {matrix.shape}")

            for j in range( nite ): # iteration over feature map
                fmap = matrix[j,  :,  :]
                resized_fmap = cv2.resize(src=fmap, dsize=(press_size)).astype(np.float32)
                #resized_fmap = cv2.resize(src=fmap, dsize=(press_size), interpolation=method).astype(np.float32)

                feature_list.append(resized_fmap)
                #print(f"{type(resized_fmap)}, {resized_fmap.dtype}")
                # <class 'numpy.ndarray'>, float32
    
    matrix = np.asarray(feature_list)
    #print("feature_list : ", matrix.shape, matrix.shape[0])
    # feature_list :  (256, 255, 255) 256

    if 1: # add for my output
        print(matrix.shape)
        print("\noutput features = \n", matrix[0:1,0:1,0:9], "\n\n")
        fig = plt.figure("")
        plt.imshow(matrix[0])
        #plt.show()
        np.save(np_save_name, matrix)
        plt.show()

    return matrix


# 1. get IR model
ir_model, ir_compiled_model = get_cutting_model(inmodel)

# 2. get images, 
input_base_img = get_input_image(baseImg, "imgB")

#
# 3. get feature map for base, and image1, and image2
#
matrixBase = get_feature_pixel(ir_model, ir_compiled_model, input_base_img, "base_img", 1)

