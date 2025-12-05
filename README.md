Efficient Real-Time Super-Resolution
====================================

**ECS 271 Course Project – UC Davis**

*Authors: Shu Zhang, Pichsorita Yim, Jan Remennik*

This project implements and compares **three super-resolution methods** under real-time constraints:

*   **Bicubic (baseline)**
    
*   **ResNet-based CNN**
    
*   **SwinIR-based Transformer**
    

The goal is to study the **trade-off between reconstruction quality and inference speed** on the DIV2K dataset (×4).

Dataset
-------

*   **Training:** DIV2K (800 images)
    
*   **Validation / Test:** DIV2K valid
    
*   **Scale:** ×4 bicubic downsampling
    

Models
------

*   **Bicubic baseline:** non-learned reference
    
*   **ResNet-SR:** lightweight residual CNN for fast inference
    
*   **SwinIR:** window-based Transformer for higher-quality reconstruction
    

One-Click Run (Recommended)
---------------------------

All training and evaluation can be executed using:
`   bash run_models.sh   `

This script will automatically:

1.  Train the ResNet model
    
2.  Train the SwinIR model
    
3.  Evaluate all models and save:
    
    *   PSNR / SSIM / inference time
        
    *   Visual results (LR / SR / HR)
        
    *   Training and evaluation plots
        

Results Output
--------------

*   **Trained models:** model/resnet/, model/swinir/
    
*   **Training curves:** images/, plots/\*.npz
    
*   **Visual comparisons:** analysis\_results\_test/
    

Project Structure
-----------------

```
analysis_results_test/           # visual comparisons
images/                          # training loss plots

model/                           # saved trained models
  resnet/
  swinir/

plots/                           # saved training arrays

src/
  dataset.py
  model_bicubic.py
  model_resnet_sr.py
  swinir.py
  training_resnet.py
  training_swinir.py
  evaluate_models.py
  plots.py
  utils.py

run_models.sh
```

Environment
-----------

`   pip install torch torchvision numpy matplotlib tqdm pillow   `