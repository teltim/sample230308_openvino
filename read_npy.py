"""

"""
import numpy as np
from matplotlib import pyplot as plt

if 1:
    # difference b/w vino21 and 22 
    #  using  fake-quantize layer  
    #  using  quant/dequant-linear layer
    
    #fig = "Zebra-1.jpg"
    fig = "Accordion-842.jpg"
    
    datA = "save_np_ovino21.4_"+fig+"_model.test-int8-Alex.ovino-nncrf-int8.ovino21.4_mo.cut.xml.npy"
    datB = "save_np_ovino22.1_"+fig+"_model.test-int8-Alex.ovino-nncrf-int8.ovino22.1_mo.cut.xml.npy"
    datX = "save_np_ovino21.4_"+fig+"_model.test-int8-Alex.onnx-rt-int8.cut.onnx.npy"
    datY = "save_np_ovino22.1_"+fig+"_model.test-int8-Alex.onnx-rt-int8.cut.onnx.npy"

    datA = np.load(datA)
    datB = np.load(datB)
    datX = np.load(datX)
    datY = np.load(datY)

    print("diff b/w vino21-22 w/ fake-quantize min ", (datA-datB).min() )
    print("diff b/w vino21-22 w/ fake-quantize max ", (datA-datB).max() )
    print("diff b/w vino21-22 w/ quant/dequant-linear min ", (datX-datY).min() )
    print("diff b/w vino21-22 w/ quant/dequant-linear max ", (datX-datY).max() )

   
    fig1 = plt.figure("vino21 and 22 ovino-nncf", figsize=(8,4), facecolor='lightblue')
    ax11 = fig1.add_subplot(1, 2, 1)
    img11= ax11.imshow(datA[10])
    ax12 = fig1.add_subplot(1, 2, 2)
    img12= ax12.imshow(datB[10])
    fig1.tight_layout()
   
    fig2 = plt.figure("vino21 and 22 onnx-rt", figsize=(8,4), facecolor='lightblue')
    ax21 = fig2.add_subplot(1, 2, 1)
    img21= ax21.imshow(datX[10])
    ax22 = fig2.add_subplot(1, 2, 2)
    img22= ax22.imshow(datY[10])
    fig2.tight_layout()
 
    #plt.show()

    fig3 = plt.figure("diff", figsize=(6,5))
    plt.title('diff b/w vino21-22 w/ fake-quantize')

    #plt.hist( (datA-datB).flatten(), range=(-0.5, 0.5), bins=200, histtype="step", label='1')
    plt.hist( (datX-datY).flatten(), range=(-0.1, 0.1), bins=200, histtype="step", label='1')

    #plt.legend(loc='upper left')
    plt.xlabel('feature diff (vino22.1 - vino21.4)', fontsize=16)
    plt.ylabel('entries', fontsize=16)
    fig3.tight_layout()

    plt.show()

