import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image
from skimage import io

#mosquito category
m_category = ['Aegypti', 'Albopictus', 'Other']

# load the save model
model = tf.keras.models.load_model('/content/keras_model.h5')

#print(model.summary())

img = "/content/drive/MyDrive/aegypti1a.jpg"
test_image = image.load_img(img, target_size=(224, 224))

# convert image in to array
test_image = image.img_to_array(test_image)
# print(test_image.shape)

# expand tha array with another demention
test_image = np.expand_dims(test_image, axis=0)

# predict the category
result = model.predict(test_image)

print(result)
print(type(np.logical_or(result[0][0] , result[0][1])))


show_img = io.imread(img)
io.imshow(show_img)

# image validation


Aegypti_Out = np.logical_and(result[0][0]>result[0][1] ,result[0][0]>result[0][2])

Albopictus_Out = np.logical_and(result[0][1]>result[0][0] ,result[0][1]>result[0][2])

Other_Out = np.logical_and(result[0][2]>result[0][1] ,result[0][2]>result[0][0])

if Aegypti_Out:
    print("The image is Aegypti")
elif Albopictus_Out:
    print("The image is Albopictus")  
elif Other_Out:
    print("The image is Other")  
else:
    print("Cant find !")