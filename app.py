import streamlit as st
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# app style
app_style = """
    <style>
        body {
            background-color: #caf0f8; /* Set background color to light gray */
            font-family: Arial, sans-serif; /* Change font family */
        }
        .title {
            color: #3366ff; /* Set title color to blue */
            text-align: center;
            font-size: 2.5em; /* Increase title font size */
        }
        .header {
            color: #009900; /* Set header color to green */
            font-size: 1.5em; /* Increase header font size */
        }
        .sub-header{
            color: #a7c957;
            font-size: 1.1em; /* Increase header font size */
        }
    </style>
"""


st.markdown(app_style, unsafe_allow_html=True)

# Title of the web app
st.markdown('<h1 class="title">Face Recognition App</h1>', unsafe_allow_html=True)



# Section 1: What is Face Recognition?

col1, col2 = st.columns(2)
with col1:
    st.markdown('<h2 class="header"> What is Face Recognition? </h2> ', unsafe_allow_html=True)
    st.write("Facial Recognition is a way of recognizing a human face using biometrics. "
            "It consists of comparing features of a person’s face with a database of known faces to find a match. "
            "When the match is found correctly, the system is said to have ‘recognized’ the face. "
            "Face Recognition is used for a variety of purposes, like unlocking phone screens, identifying criminals, "
            "and authorizing visitors.")
with col2:
    # st.markdown('<img src="media\Picture1.png">', unsafe_allow_html=True)
    image1 = Image.open('media\Picture1.png')

    st.image(image1, caption='')

# Section 2: How do Computers Recognize Faces?
st.markdown('<h2 class="header"> How do Computers Recognize Faces?</h2> ', unsafe_allow_html=True)
st.write("The Face Recognition system uses Machine Learning to analyze and process facial features from images or videos. "
         "Features can include anything, from the distance between your eyes to the size of your nose. "
         "These features, which are unique to each person, are also known as Facial Landmarks. "
         "The machine learns patterns in these landmarks by training Artificial Neural Networks. "
         "The machine can then identify people faces by matching these learned patterns against new facial data.")

# Section 3: Teach the Computer to Recognize your Face
st.markdown('<h2 class="header"> Teach the Computer to Recognize your Face </h2> ', unsafe_allow_html=True)
image2 = Image.open('media\Screenshot 2023-11-09 151026.png')
st.image(image2, caption='First Step')

st.markdown('<h3 class="sub-header"> Step 1 – Collect Data </h3> ', unsafe_allow_html=True)
st.write("We want our model to learn how to recognize your face. We will need two kinds of images for this - images of you, and images of people who are not you. This way, the model will learn to recognize how you look and also recognize how you don’t look.")

col1, col2 = st.columns(2)
with col1:
    st.write("1. Let’s start by giving the machine lots of images of you in different places, in different poses, and at different angles.")
    # Upload 'me' class images for training
    st.subheader("Label: 'me'")
    me_files = st.file_uploader("Choose images...", type=["jpg", "png"], accept_multiple_files=True, key="me")

with col2:
    st.write("2. Next, let’s give it images of people that are not you, so the machine understands the difference.")
    # Upload 'not me' class images for training
    st.subheader("Label: 'not me'")
    not_me_files = st.file_uploader("Choose images...", type=["jpg", "png"], accept_multiple_files=True, key="not_me")


# step-2
st.markdown('<h2 class="header"> Teach the Computer to Recognize your Face </h2> ', unsafe_allow_html=True)
image2 = Image.open('media\step-2.png')
st.image(image2, caption='Second Step')
st.markdown('<h3 class="sub-header"> Step 2 – Train the Machine </h3> ', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.write("""Next, we need to train the machine (or model) to recognize pictures of you. The model uses the samples of images you provided for this. 
This method is called “Supervised learning” because of the way you ‘supervised’ the training.   
The model learns from the patterns in the photos you’ve taken. It mostly takes into consideration the facial features or Facial Landmarks and associates the landmark of each face with the corresponding label.
""")
    
with col2:
    # st.markdown('<img src="Picture1.png">', unsafe_allow_html=True)
    image1 = Image.open('media\Picture2.png')

    st.image(image1, caption='')

# ste-2 - 2
st.markdown('<h3 class="sub-header"> What do you mean by Machine Learning? </h3> ', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.write("""Machine learning is the process of making systems that learn and improve by themselves. The model learns from the data and makes predictions. It then checks with your label to see if it predicted the label correctly. If it didn’t, then it tries again. It keeps repeating this process with an aim to get better at the predictions.
""")
    
with col2:
    # st.markdown('<img src="media\Picture1.png">', unsafe_allow_html=True)
    image1 = Image.open('media\Picture3.png')

    st.image(image1, caption='')

image1 = Image.open('media\Picture4.png')

st.image(image1, caption='')

# Slider for number of epochs
st.markdown('<h2 class="header"> Train the Machine </h2> ', unsafe_allow_html=True)
st.write("""Now let us set up our Machine Learning model!  Enter the number of epochs for which you would like the model to train:
 """)
epochs_duplicate = st.slider("Number of Epochs", 10, 100, 10)
epochs = (epochs_duplicate//10)
st.write("""Once your model is all set, you can start training your model - 
 """)
# Train the model
if st.button("Train Model"):
    if me_files and not_me_files:
        # Process uploaded images
        st.write("Processing uploaded images...")
        processed_images = []
        labels = []

        for uploaded_file in me_files:
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            processed_images.append(img)
            labels.append(1)  # 'me' class

        for uploaded_file in not_me_files:
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            processed_images.append(img)
            labels.append(0)  # 'not me' class

        X_train = np.vstack(processed_images)
        y_train = np.array(labels)

        # Train the model
        st.write(f"Training with {epochs_duplicate} epochs...")
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(2, activation='softmax')(x)  # 2 classes: 'me' and 'not me'
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs)  # Training with user-defined epochs

        # Calculate training accuracy
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        st.write(f"Training complete! Training Accuracy: {train_acc*100:.2f}%")

        # Save the model
        model.save('model.h5')

# Section 5: Test the Model
st.markdown('<h2 class="header"> Test the model </h2> ', unsafe_allow_html=True)

# Upload a test image
test_image = st.file_uploader("Upload a test image...", type=["jpg", "png"])

if test_image and 'model.h5' in os.listdir():
    st.write("Processing test image...")
    img = image.load_img(test_image, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Make prediction
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)  # 2 classes: 'me' and 'not me'
    model = Model(inputs=model.input, outputs=predictions)

    model.load_weights('model.h5')
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Display result
    if predicted_class == 1:
        st.write("Result: This is you!")
    else:
        st.write("Result: This is not you.")
