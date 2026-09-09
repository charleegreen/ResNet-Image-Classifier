import tensorflow as tf
###############################################################
#globals
###############################################################
image_size = (224,224)
batch_size = 32


AUTOTUNE = tf.data.AUTOTUNE
##############################################################


# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    "/home/cs421_share/celebdataset/train",
    #"celebdataset/train",
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical',  # 'binary' if 2 classes
    shuffle=True,
    seed=42
)


num_classes = len(train_ds.class_names)

#was having issues with overfitting, primarily because test data is too small, so augment to make test data larger
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])


val_ds = tf.keras.utils.image_dataset_from_directory(
    "/home/cs421_share/celebdataset/train",
    #"celebdataset/train",
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical',  # 'binary' if 2 classes
    shuffle=True,
    seed=42
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "/home/cs421_share/celebdataset/train",
    #"celebdataset/train",
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)



# Normalize pixel values to [0,1]
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Normalize first, then augment, then cache/prefetch
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))    #this is where the augmented data gets added
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# Val gets normalization ONLY — no augmentation
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

###########################################################
#Build model from scratch
#just copy and pasted after the green layers... while changing this size of inputs
###########################################################

#this is the pair of conv layers plus the skip connections for the 3x3, 64
def blue_layers(name):
    #place holder for the input since reusing function...
    block_input= tf.keras.Input(shape=(56,56,64), name=name+"_input")   #orig image was 224 /2(stride) then /2(stride) makes image 56
    
    #key take away, padding=same means dont change x,y dimensions.
    x= tf.keras.layers.Conv2D(64,3, padding='same', use_bias=False)(block_input)  
    x= tf.keras.layers.BatchNormalization()(x)
    x= tf.keras.layers.ReLU()(x)
    
    x= tf.keras.layers.Conv2D(64,3, padding='same', use_bias=False)(x)  
    x= tf.keras.layers.BatchNormalization()(x)
    
    x=tf.keras.layers.Add()([x,block_input])    #SKIP CONNECTION
    
    x= tf.keras.layers.ReLU()(x)

    blue_model = tf.keras.Model(block_input, x, name=name)
    
    print(blue_model.summary())
    return blue_model

def green_layer1(name):
    
    block_input= tf.keras.Input(shape=(56,56,64), name=name+"_input")   #orig image was 224 /2(stride) then /2(stride) makes image 56
    
    #key take away, padding=same means dont change x,y dimensions.
    x= tf.keras.layers.Conv2D(128,3, padding='same',strides=2, use_bias=False)(block_input)  
    x= tf.keras.layers.BatchNormalization()(x)
    x= tf.keras.layers.ReLU()(x)
 
    x= tf.keras.layers.Conv2D(128,3, padding='same', use_bias=False)(x)  
    x= tf.keras.layers.BatchNormalization()(x)
    
    shortcut=tf.keras.layers.Conv2D(128,1,strides=2,use_bias=False)(block_input) 
    #gets the input to be the same size as the rest for adding
    shortcut=tf.keras.layers.BatchNormalization()(shortcut)
    
    x=tf.keras.layers.Add()([x,shortcut]) #mad about this line.....
    
    x= tf.keras.layers.ReLU()(x)
    
    green1_model= tf.keras.Model(block_input, x, name=name)
    
    print(green1_model.summary())
    return green1_model

def green_layer2(name):
    
    block_input= tf.keras.Input(shape=(28,28,128), name=name+"_input")  
    #56/2 for the stride and then 128 is input from last section
    x= tf.keras.layers.Conv2D(128,3, padding='same', use_bias=False)(block_input)  
    x= tf.keras.layers.BatchNormalization()(x)
    x= tf.keras.layers.ReLU()(x)

    x= tf.keras.layers.Conv2D(128,3, padding='same', use_bias=False)(block_input)  
    x= tf.keras.layers.BatchNormalization()(x)
    
    x=tf.keras.layers.Add()([x,block_input]) 
    
    x= tf.keras.layers.ReLU()(x)

    green2_model= tf.keras.Model(block_input, x, name=name)
    
    print(green2_model.summary())
    return green2_model
    
def orange_layer1(name):
    block_input= tf.keras.Input(shape=(28,28,128), name=name+"_input")   
    
    x= tf.keras.layers.Conv2D(256,3, padding='same',strides=2, use_bias=False)(block_input)  
    x= tf.keras.layers.BatchNormalization()(x)
    x= tf.keras.layers.ReLU()(x)
 
    x= tf.keras.layers.Conv2D(256,3, padding='same', use_bias=False)(x)  
    x= tf.keras.layers.BatchNormalization()(x)
    
    shortcut=tf.keras.layers.Conv2D(256,1,strides=2,use_bias=False)(block_input) 
    #gets the input to be the same size as the rest for adding
    shortcut=tf.keras.layers.BatchNormalization()(shortcut)
    
    x=tf.keras.layers.Add()([x,shortcut]) 
    
    x= tf.keras.layers.ReLU()(x)
    
    orange1_model= tf.keras.Model(block_input, x, name=name)
    
    print(orange1_model.summary())
    return orange1_model

def orange_layer2(name):
    
    block_input= tf.keras.Input(shape=(14,14,256), name=name+"_input")  
    #28/2=14 for the stride , and 256 from the last section
    x= tf.keras.layers.Conv2D(256,3, padding='same', use_bias=False)(block_input)  
    x= tf.keras.layers.BatchNormalization()(x)
    x= tf.keras.layers.ReLU()(x)

    x= tf.keras.layers.Conv2D(256,3, padding='same', use_bias=False)(block_input)  
    x= tf.keras.layers.BatchNormalization()(x)
    
    x=tf.keras.layers.Add()([x,block_input]) 
    
    x= tf.keras.layers.ReLU()(x)

    orange2_model= tf.keras.Model(block_input, x, name=name)
    
    print(orange2_model.summary())
    return orange2_model

def purple_layer1(name):
    block_input= tf.keras.Input(shape=(14,14,256), name=name+"_input")   #orig image was 224 /2(stride) then /2(stride) makes image 56
    
    #key take away, padding=same means dont change x,y dimensions.
    x= tf.keras.layers.Conv2D(512,3, padding='same',strides=2, use_bias=False)(block_input)  
    x= tf.keras.layers.BatchNormalization()(x)
    x= tf.keras.layers.ReLU()(x)
 
    x= tf.keras.layers.Conv2D(512,3, padding='same', use_bias=False)(x)  
    x= tf.keras.layers.BatchNormalization()(x)
    
    shortcut=tf.keras.layers.Conv2D(512,1,strides=2,use_bias=False)(block_input) 
    #gets the input to be the same size as the rest for adding
    shortcut=tf.keras.layers.BatchNormalization()(shortcut)
    
    x=tf.keras.layers.Add()([x,shortcut]) 
    
    x= tf.keras.layers.ReLU()(x)
    
    purple1_model= tf.keras.Model(block_input, x, name=name)
    
    print(purple1_model.summary())
    return purple1_model

def purple_layer2(name):
    
    block_input= tf.keras.Input(shape=(7,7,512), name=name+"_input")   
    #14/2= 7 from the stride, and 512 from the previous section.
    x= tf.keras.layers.Conv2D(512,3, padding='same', use_bias=False)(block_input)  
    x= tf.keras.layers.BatchNormalization()(x)
    x= tf.keras.layers.ReLU()(x)

    x= tf.keras.layers.Conv2D(512,3, padding='same', use_bias=False)(block_input)  
    x= tf.keras.layers.BatchNormalization()(x)
    
    x=tf.keras.layers.Add()([x,block_input]) 
    
    x= tf.keras.layers.ReLU()(x)

    purple2_model= tf.keras.Model(block_input, x, name=name)
    
    print(purple2_model.summary())
    return purple2_model

###########################################################
###########################################################
#start of the model
stem_input = tf.keras.Input(shape=(224,224,3), name='stem')

#input of this is the placeholder we named stem_outputm output is x
x= tf.keras.layers.Conv2D(64,7,strides=2, padding='same', use_bias=False)(stem_input)   #we use use_bias=False doing it in batchNorm

#batch norm. output is now called x
x= tf.keras.layers.BatchNormalization()(x)

#now the relu
x= tf.keras.layers.ReLU()(x)

#maxpooling comes after the relu
x= tf.keras.layers.MaxPooling2D(pool_size=3, strides=2,padding='same')(x)

#blue layers
blue_layers_model1 = blue_layers("bl0")
blue_layers_model2 = blue_layers("bl1")
blue_layers_model3 = blue_layers("bl2")
x=blue_layers_model1(x) #maybe use real var names... instead of x
x=blue_layers_model2(x)
x=blue_layers_model3(x)

#green layers
green_layer1_model = green_layer1("gr0")
green_layer2_model1 = green_layer2("gr1")
green_layer2_model2 = green_layer2("gr2")
green_layer2_model3 = green_layer2("gr3")

x=green_layer1_model(x)
x=green_layer2_model1(x)
x=green_layer2_model2(x)
x=green_layer2_model3(x)

#orange layers
orange_layer1_model = orange_layer1("or0")
orange_layer2_model1 = orange_layer2("or1")
orange_layer2_model2 = orange_layer2("or2")
orange_layer2_model3 = orange_layer2("or3")

x=orange_layer1_model(x)
x=orange_layer2_model1(x)
x=orange_layer2_model2(x)
x=orange_layer2_model3(x)

#purple layers
purple_layer1_model = purple_layer1("pr0")
purple_layer2_model1 = purple_layer2("pr1")
purple_layer2_model2 = purple_layer2("pr2")
purple_layer2_model3 = purple_layer2("pr3")

x=purple_layer1_model(x)
x=purple_layer2_model1(x)
x=purple_layer2_model2(x)
x=purple_layer2_model3(x)


x=tf.keras.layers.GlobalAveragePooling2D()(x)
x=tf.keras.layers.Dense(num_classes,activation = 'softmax')(x)


full_model= tf.keras.Model(stem_input, x, name="FullModel")


print(full_model.summary())

############################################################
#early stopping techniques
callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=8,
    restore_best_weights=True)

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)
#############################################################                   


                      
full_model.compile(loss='categorical_crossentropy', optimizer= "nadam", metrics=['accuracy'])
full_model.fit(train_ds, validation_data = val_ds, epochs=200, callbacks=[lr_callback, callback], verbose=1)

full_model.save("resnet.keras")  


