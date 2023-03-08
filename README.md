
## status
Some strage trouble, for a while, I can't upload output feature data,<br>
please run infere_ovinoXX.X.py under venv to generate *.npy data

## code
infere_ovino21.4.py<br>
a sample code to perform inference and extract feature, based on "open-vino21.1"

infere_ovino22.1.py<br> 
a sample code to perform inference and extract feature, based on "open-vino22.1"

read_npy.py<br>
a sample code to compare output features which are saved in .npy

## model
model is int8-Alex model<br>
int8 was done in onnx-runtime and nncf	

onnx-runtime: model.test-int8-Alex.onnx-rt-int8.cut.onnx<br>
nncf-package: model.test-int8-Alex.ovino-nncrf-int8.onnx	

IR conversion was done in each openvino version:<br>
ovino21.4_mo.cut<br>
ovino22.1_mo.cut  	

## generated output feature data
save_np_*,npy<br>   
saved output features via openvino21, and 22

save_np_ _vino-version_ _ _sample-image_ _ _sample-model_ .npy

e.g.<br>
save_np_ovino21.4_Accordion-842.jpg_model.test-int8-Alex.ovino-nncrf-int8.ovino21.4_mo.cut.xml.npy      
save_np_ovino22.1_Accordion-842.jpg_model.test-int8-Alex.ovino-nncrf-int8.ovino22.1_mo.cut.xml.npy    


## the difference we observe b/w openvino version and onnx-op layers is as follows     
for instance, usig "Accordion-842.jpg" <br>
difference b/w vino21-22 using fake-quantize (nncf-quantization) :  <br>
&emsp;min  0.0 <br>
&emsp;max  0.0 <br>
difference b/w vino21-22 using quant/dequant-linear (onnx-rt) :  <br>
&emsp;min  -0.03369951<br>
&emsp;max  +0.03252697

A model w/ fake-quantize layer outputs the same values b/w 21 and 22<br>
A model w/ quant/dequant-linear layer outputs different values b/w 21 and 22 
