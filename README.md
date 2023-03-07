
## status

## code
infere_ovino21.4.py<br>
a sample code to perform inference and extract feature, based on "open-vino21.1"

infere_ovino22.1.py<br> 
a sample code to perform inference and extract feature, based on "open-vino22.1"

read_npy.py<br>
a sample code to compare output features which are saved in .npy

## generated data
save_np_*,npy<br>   
saved output features via openvino21, and 22

save_np_ _vino-version_ _ _sample-image_ _ _sample-model_ .npy

e.g.<br>
save_np_ovino21.4_Accordion-842.jpg_model.test-int8-Alex.ovino-nncrf-int8.ovino21.4_mo.cut.xml.npy      
save_np_ovino22.1_Accordion-842.jpg_model.test-int8-Alex.ovino-nncrf-int8.ovino22.1_mo.cut.xml.npy    

## model
model is int8-Alex model<br>
int8 was done in onnx-runtime and nncf	

onnx-runtime: model.test-int8-Alex.onnx-rt-int8.cut.onnx<br>
nncf-package: model.test-int8-Alex.ovino-nncrf-int8.onnx	

IR conversion was done in each openvino version:<br>
ovino21.4_mo.cut<br>
ovino22.1_mo.cut  	
    
    	
