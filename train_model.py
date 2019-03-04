##first we have to set up the LeNet
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class LeNet:
    
    def build(width, height, depth, classes):
        #inititalize the model
        model = Sequential()
        inputShape = (height, width, depth)
        
        #if we're using channels first then update the input shape
        if K.image_data_format()== "channels_first":
            inputShape = (depth, height, width)
            
        #first set of conv relu and pool layers
        model.add(Conv2D(20, (5,5), padding="same", input_shape = inputShape))
        #20 filters
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        #second set of conv relu and pool layers
        model.add(Conv2D(50, (5,5), padding="same"))
        #50 filters
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        #first set of FC and RELU
        model.add(Flatten())
        model.add(Dense(500)) #FC with 500 nodes
        model.add(Activation("relu"))
        #softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model

##path to the SMILES dataset
dataset = "data/SMILEsmileD/"

data = []
labels = []

##loop over input images
for imagePath in sorted(paths.list_images(dataset)):
    #load image, preprocess and store
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width = 28) #resize to 28x28 for LeNet
    image = img_to_array(image)
    data.append(image)
    
    #get class labels and add to list
    label = imagePath.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)

#normalization 
data = np.array(data, dtype = "float") / 255.0
#one hot encoding
labels = np.array(labels)
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels),2)

# accounting for data imbalance by computing class weights
classTotals = labels.sum(axis = 0) #total number of examples per class
classWeights = classTotals.max() / classTotals
#every instance of "smiling" as 2.56 instances of "not smiling"

#split
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.21, stratify = labels, random_state = 42)
#train LeNet
model = LeNet.build(width = 28, height = 28, depth = 1, classes = 2)
model.compile(loss = "binary_crossentropy", 
              optimizer = "adam", metrics = ["accuracy"])
H = model.fit(trainX, trainY, validation_data = (testX, testY),
             class_weight = classWeights, batch_size = 64, 
              epochs = 15, verbose = 1)
#evaluate the network
predictions = model.predict(testX, batch_size = 64)
print(classification_report(testY.argmax(axis = 1),
                           predictions.argmax(axis = 1), 
                            target_names = le.classes_))
#save the model
model.save("model/smiles_lenet.hdf5")
#plot the train test loss and acc
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,15), H.history["acc"], label="acc")
plt.plot(np.arange(0,15), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Acc")
plt.legend()
plt.show()