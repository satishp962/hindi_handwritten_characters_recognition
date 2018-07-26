# Hindi Handwritten Characters Recognition
### Trained on system having following configurations

 - Acer Aspire E1-570G
 - RAM - 4 GB
 - CPU - Intel I3 4th Generation
 - GPU - Nvidia Geforce 740M (CUDA Compatible GPU)

### Required Python modules
Run the **requirements.txt** to install dependencies

 - Tensorflow >= 1.1
 - Keras
 - OpenCV
 - Numpy
 - IMUtils
 -  Pickle
 - Pandas

### Files information

 - **dataset_ka_kha** - contains the training samples
 - **out** - contains the h5py file generated after training
 - **test_images** - contains the unseen images to test model performance
 - **load_images_into_pickels.py** - generates the data and labels list to be used in training
 - **predict_model.py** - use for prediction
 - **sample.csv** - label information
 - **train_model.py** - use to train the model

### Steps to run this example

 1. `python load_images_into_pickels.py`
 2. `python train_model.py`
 3. `predict_model.py`
## Output
![Output](https://github.com/satishp962/hindi_handwritten_characters_recognition/blob/master/output.PNG)

