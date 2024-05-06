# Sign-Language-Detection
This project deals take the live feed from your camera and predict the Sign Language as depicted in the system.

## General Info
The theme of this project is to use technology to improve accessibility by finding a creative solution to benefit the lives of those with a disability. 
We wanted to make it easy for 70 million deaf people across the world to be independent of translators for there daily communication needs, so we designed the app to work as a personal translator 24*7 for the deaf people.

## Screenshots





## Technology and Tools
* Python
* Tensorflow
* OpenCV
* Numpy

## Process
To execute the model.py on Google Colab, you can follow these steps:

* Set Up Google Colab: Open Google Colab in your browser and create a new notebook.
* Install Required Packages: Copy the code you provided into a code cell in the Colab notebook. The code includes several !pip install commands to install necessary packages. These will be executed directly within the notebook.
* Import Libraries: After installing the required packages, run the cell containing import statements. This will import the necessary libraries and modules for the rest of the code to execute.
* Set Up Roboflow: Make sure you have a Roboflow account and an API key. Replace "vXIrSGF6SPAaatoP3g53" with your actual API key.
* Download Dataset from Roboflow: The code uses the Roboflow API to download a dataset. Ensure that the project and version identifiers ("david-lee-d0rhs" and 6 respectively) are correct and correspond to your dataset.
* Define Dataset Parameters: Make sure the dataset_params dictionary corresponds to the structure of your downloaded dataset. Update the keys if necessary.
* Prepare Data Loaders: This section defines data loaders for training, validation, and testing. Ensure that the paths and parameters match your dataset structure.
* Set Training Parameters: Update the train_params dictionary according to your training requirements. You may adjust parameters such as learning rate, epochs, optimizer, etc.
* Download Pre-trained Model Weights: The code downloads pre-trained model weights. Ensure that the provided URL is accessible and correct.
* Train the Model: Execute the cell that initiates the training process using the trainer.train() method. This will train the model using the specified parameters and data loaders.
* Load Best Model: After training, load the best-performing model using the models.get() function and provide the checkpoint path.
* Test the Model: Run the cell that tests the model on the test dataset. This will evaluate the model's performance using specified metrics.
* Predictions: Finally, use the trained model to make predictions on images and videos. Ensure that the paths provided for images and videos are correct.
* Execute Code: Execute each code cell sequentially by clicking on the "Play" button next to the cell or by using Shift+Enter.
* Monitor Execution: Monitor the execution of each cell for any errors or warnings. If there are errors, debug the code accordingly.
* Save Output: If desired, save the output files generated by the model predictions.

After the Model is done move to PyCharm

To execute webcam.py in PyCharm, you can follow these steps:

* Set Up Environment: Ensure you have Python installed in your PyCharm environment. You may want to create a virtual environment to manage dependencies.
* Install Dependencies: Install the required packages. You can do this by opening the terminal in PyCharm and running the following command:
  pip install opencv-python-headless super-gradients
  This will install OpenCV and the super-gradients package.
* Import Libraries: Import the necessary libraries in your Python script:
* Load Model: Load the model using the models.get() function. Ensure you provide the correct number of classes and checkpoint path.
* Capture Webcam Feed: Use OpenCV to capture the webcam feed and make predictions using the loaded model. You can use a loop to continuously capture frames from the webcam.
* Run the Script: Run your Python script in PyCharm. Make sure you have a webcam connected to your system to capture the feed.
* Monitor Output: Monitor the output window where the webcam feed is displayed. You should see the predictions made by the model overlaid on the webcam feed.
* Close the Application: Close the application by pressing the 'q' key or by closing the output window.
