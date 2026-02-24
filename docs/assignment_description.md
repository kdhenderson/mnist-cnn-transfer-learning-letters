## Homework 3: CNN with MNIST and Transfer Learning

**Instructions from the Word document**:  

Create an example of transfer learning where you train on  half  the MNIST dataset digits (i.e. digits 0-4) and “transfer” the model to the second half (i.e, digits 5-9). 

Your homework is to train on all digits and make your own handwritten data set of the 5 characters {A, B, C, D, E}  and “transfer” your MNIST trained model over to the dataset you created. In other words, train on the full MNIST datasets (i.e. digits 0-9) and transfer on the {A, B, C, D, E} image dataset. 

Please do not use any other data resources from the web such as the mnist dataset. Figuring out the challenges of making your own handwritten character dataset is part of the exercise for this homework!   

Submission instructions: Include the images you generated as part of your submission. Include any code used for the character image preprocessing. 

Using OpenCV is encouraged. Pytorch or Tensorflow are both fine.


**Instructions from the class transcript**:  

You are to build a convolutional neural network (CNN) using the **MNIST dataset**, with a focus on **transfer learning**. The homework has three stages:

Stage 1: Base Training

- Train a CNN on digits **0–4** from the MNIST dataset.  
- The CNN must have **at least three convolutional layers**.

Stage 2: Transfer Learning

- **Remove** the final layer of the trained model.  
- **Retrain only the final layer** on digits **5–9** from MNIST.

Stage 3: Real-World Transfer

- **Create your own dataset**:
  - Write the letters **A–E** by hand on paper.
  - Take **pictures** of your handwritten letters using your phone.
  
- **Use the pretrained network** and **transfer learn to classify A–E**:
  - Only train the **final layer**.
  - **Preprocess your images** to align with MNIST format (e.g., 28×28 grayscale, normalized).

Tools and Help

- You are **encouraged to use Cursor and ChatGPT** to help write code and preprocess data.
- Try to see how much these tools reduce the workload—what historically took **15 hours** might now take **2 hours**.

Timeline

- You have **2 weeks** to complete the assignment.
