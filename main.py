import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('Food.h5')


def classify_image(image):
    img = np.array(image.resize((224, 224))) / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    class_idx = np.argmax(predictions[0])
    class_label = class_names[class_idx]
    confidence = predictions[0][class_idx]

    return class_label, confidence
class_names = ["apple_pie","baby_back_ribs","baklava","beef_carpaccio","beef_tartare",
                "beet_salad","beignets","bibimbap","bread_pudding","breakfast_burrito","bruschetta",
                "caesar_salad","cannoli","caprese_salad" ,"carrot_cake" ,"ceviche","cheesecake","cheese_plate","chicken_curry",
                "chicken_quesadilla","chicken_wings","chocolate_cake","chocolate_mousse","churros","clam_chowder","club_sandwich",
                "crab_cakes","creme_brule","croque_madame","cup_cakes","deviled_eggs","donuts","dumplings","edamame","eggs_benedict"
                ,"escargots","falafel","filet_mignon","fish_and_chips","foie_gras","french_fries","french_onion_soup","french_toast"
                ,"fried_calamari","fried_rice","frozen_yogurt","garlic_bread","gnocchi","greek_salad","grilled_cheese_sandwich"
                ,"grilled_salmon","guacamole","gyoza","hamburger","hot_and_sour_soup","hot_dog","huevos_rancheros","hummus"
                ,"ice_cream","lasagna","lobster_bisque","lobster_roll_sandwich","macaroni_and_cheese","macarons","miso_soup"
                ,"mussels","nachos","omelette","onion_rings","oysters","pad_thai","paella","pancakes","panna_cotta","peking_duck"
                ,"pho","pizza","pork_chop","poutine","prime_rib","pulled_pork_sandwich","ramen","ravioli","red_velvet_cake","risotto"
                ,"samosa","sashimi","scallops","seaweed_salad","shrimp_and_grits","spaghetti_bolognese","spaghetti_carbonara","spring_rolls"
                ,"steak","strawberry_shortcake","sushi","tacos","takoyaki","tiramisu","tuna_tartare","waffles"]

st.title("Image Classifier")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    class_label, confidence = classify_image(image)
    st.write("Class Label: ", class_label)
    st.write("Confidence: ", confidence)

