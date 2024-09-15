
# Pizza Classification using Deep Learning

This repository contains a Streamlit web application that uses deep learning to classify images of pizzas as either cooked or not cooked. The project leverages a Convolutional Neural Network (CNN) model built using TensorFlow and Keras to make binary classifications. 

## Project Overview

This project was part of my undergraduate coursework, where I explored deep learning techniques to create a functional and interactive web application. Users can upload an image of their pizza, and the model will predict whether it is cooked to perfection or not. The application uses Streamlit for an intuitive user interface and Keras for model deployment.

An interesting aspect of this project is its connection to my passion for pizza making. During one of my semesters, I built a fire brick pizza oven from scratch. This hands-on experience with pizza making inspired me to develop this deep learning model to analyze pizza images.

## Features

- **Image Upload:** Users can upload an image of their pizza directly into the web app.
- **Image Examples:** Provides example images of cooked and uncooked pizzas for reference.
- **Deep Learning Model:** Uses a pre-trained Convolutional Neural Network (CNN) to classify the pizza images.
- **Real-Time Prediction:** Offers a fun and interactive way to find out if your pizza is "bussin" or "disgusting."
- **Streamlit Interface:** Utilizes the Streamlit framework to create an easy-to-use web application.

## Demo

The application uses a deep learning model to determine whether a pizza is cooked or not. The user simply uploads an image, and the model predicts the status of the pizza, providing a fun and interactive experience.

## Project Files

- `app.py`: Main application file that sets up the Streamlit interface, handles image uploads, and performs predictions using the deep learning model.
- `cooked.jpg`: Example image of a cooked pizza.
- `Uncooked.jpg`: Example image of an uncooked pizza.
- `background.jpg`: Background image for the application.
- `requirements.txt`: File listing all the required packages to run the application.

## Getting Started

### Prerequisites

To run this project, you need Python installed on your machine along with the required libraries. Install the dependencies by running:

```bash
pip install -r requirements.txt
```

### Running the Application

To run the Streamlit app, execute the following command:

```bash
streamlit run app.py
```

## Usage

1. Open the Streamlit web application.
2. Upload an image of your pizza using the file uploader.
3. Click the button to find out if your pizza is cooked.
4. The application will display the result along with a fun message.

## Technical Details

- **Libraries Used:** TensorFlow, Keras, Streamlit, NumPy, PIL, Requests
- **Model Architecture:** Convolutional Neural Network (CNN)
- **Image Processing:** The uploaded image is resized to 256x256 pixels, normalized, and passed through the CNN model for prediction.

## Example Code Snippet

Here's a snippet of the main prediction logic in the `app.py` file:

```python
image = Image.open(file)
img_resized = image.resize((256, 256))
img_array = np.array(img_resized)
img_tensor = tf.convert_to_tensor(img_array)
img_tensor = tf.expand_dims(img_tensor, axis=0)
img_tensor = tf.cast(img_tensor, tf.float32) / 255.0

yhat = new_model.predict(img_tensor)

if 0 <= yhat <= 0.20: 
    st.subheader(f'THIS IS COOKED PIZZA!') 
    st.balloons()   
elif 0.21 <= yhat <= 0.40:
    st.subheader('It's cooked but not crispy')
elif 0.41 <= yhat <= 0.60:
    st.subheader('Needs more time')
elif 0.61 <= yhat <= 1.0:
    st.subheader(f'ITS RAWW - Ramsey')
```

## Building the Brick Oven

During one of my undergraduate semesters, I took on a personal project to build a fire brick pizza oven from scratch. This project not only enhanced my skills in crafting and engineering but also deepened my appreciation for the art of pizza making. This experience played a significant role in inspiring this deep learning project, combining my passion for both technology and culinary arts.

## Contact

For any inquiries or contributions, please feel free to reach out to me:

- **Email:** [abdelsal272@gmail.com](mailto:abdelsal272@gmail.com)
- **LinkedIn:** [Omar Abdelsalam](https://www.linkedin.com/in/-omarabdelsalam/)
- **GitHub:** [OmarAbdelsalam-Tech](https://github.com/OmarAbdelsalam-Tech)
