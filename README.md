# Image Classification using MNIST and Fashion MNIST Data Set

### Description

[The MNIST database](https://en.wikipedia.org/wiki/MNIST_database) (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used as a first test for new classification algorithms. 
We follow this tradition to investigate the performance of _Artificial Neural Networks_ of different complexity on MNIST. However, since MNIST is too easy for accessing the full power of modern machine learning algorithms (_Convolutional Neural Network_) we will extend our analysis to the recently introduced, harder [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

### Some model performences:

1. **Training CNN with 'relu' activation and hidden layer on MNIST Dataset**

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_2 (Conv2D)           (None, 28, 28, 32)        320       
                                                                 
 activation_51 (Activation)  (None, 28, 28, 32)        0         
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 14, 14, 32)       0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (None, 14, 14, 64)        51264     
                                                                 
 activation_52 (Activation)  (None, 14, 14, 64)        0         
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 7, 7, 64)         0         
 2D)                                                             
                                                                 
 flatten_2 (Flatten)         (None, 3136)              0         
                                                                 
 dense_49 (Dense)            (None, 128)               401536    
                                                                 
 activation_53 (Activation)  (None, 128)               0         
                                                                 
 dense_50 (Dense)            (None, 10)                1290      
                                                                 
 activation_54 (Activation)  (None, 10)                0         
                                                                 
=================================================================
Total params: 454,410
Trainable params: 454,410
Non-trainable params: 0
_________________________________________________________________
```

2. **Training CNN with 'relu' activation and hidden layer on Fashion MNIST Dataset**

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_5 (Conv2D)           (None, 28, 28, 32)        320       
                                                                 
 activation_58 (Activation)  (None, 28, 28, 32)        0         
                                                                 
 max_pooling2d_5 (MaxPooling  (None, 14, 14, 32)       0         
 2D)                                                             
                                                                 
 conv2d_6 (Conv2D)           (None, 14, 14, 64)        51264     
                                                                 
 activation_59 (Activation)  (None, 14, 14, 64)        0         
                                                                 
 max_pooling2d_6 (MaxPooling  (None, 7, 7, 64)         0         
 2D)                                                             
                                                                 
 flatten_4 (Flatten)         (None, 3136)              0         
                                                                 
 dense_53 (Dense)            (None, 128)               401536    
                                                                 
 activation_60 (Activation)  (None, 128)               0         
                                                                 
 dense_54 (Dense)            (None, 10)                1290      
                                                                 
 activation_61 (Activation)  (None, 10)                0         
                                                                 
=================================================================
Total params: 454,410
Trainable params: 454,410
Non-trainable params: 0
_________________________________________________________________
```

3. **Traning CNN with batch normalisation and 'sigmoid' activation function on MNIST Dataset**

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_11 (Conv2D)          (None, 28, 28, 32)        320       
                                                                 
 batch_normalization_12 (Bat  (None, 28, 28, 32)       128       
 chNormalization)                                                
                                                                 
 activation_70 (Activation)  (None, 28, 28, 32)        0         
                                                                 
 max_pooling2d_11 (MaxPoolin  (None, 14, 14, 32)       0         
 g2D)                                                            
                                                                 
 conv2d_12 (Conv2D)          (None, 14, 14, 64)        51264     
                                                                 
 batch_normalization_13 (Bat  (None, 14, 14, 64)       256       
 chNormalization)                                                
                                                                 
 activation_71 (Activation)  (None, 14, 14, 64)        0         
                                                                 
 max_pooling2d_12 (MaxPoolin  (None, 7, 7, 64)         0         
 g2D)                                                            
                                                                 
 flatten_7 (Flatten)         (None, 3136)              0         
                                                                 
 dense_59 (Dense)            (None, 128)               401536    
                                                                 
 batch_normalization_14 (Bat  (None, 128)              512       
 chNormalization)                                                
                                                                 
 activation_72 (Activation)  (None, 128)               0         
                                                                 
 dense_60 (Dense)            (None, 10)                1290      
                                                                 
 batch_normalization_15 (Bat  (None, 10)               40        
 chNormalization)                                                
                                                                 
 activation_73 (Activation)  (None, 10)                0         
                                                                 
=================================================================
Total params: 455,346
Trainable params: 454,878
Non-trainable params: 468
_________________________________________________________________
```

4. **Traning CNN with batch normalisation and 'sigmoid' activation function on Fashion MNIST Dataset**

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_13 (Conv2D)          (None, 28, 28, 32)        320       
                                                                 
 batch_normalization_16 (Bat  (None, 28, 28, 32)       128       
 chNormalization)                                                
                                                                 
 activation_74 (Activation)  (None, 28, 28, 32)        0         
                                                                 
 max_pooling2d_13 (MaxPoolin  (None, 14, 14, 32)       0         
 g2D)                                                            
                                                                 
 conv2d_14 (Conv2D)          (None, 14, 14, 64)        51264     
                                                                 
 batch_normalization_17 (Bat  (None, 14, 14, 64)       256       
 chNormalization)                                                
                                                                 
 activation_75 (Activation)  (None, 14, 14, 64)        0         
                                                                 
 max_pooling2d_14 (MaxPoolin  (None, 7, 7, 64)         0         
 g2D)                                                            
                                                                 
 flatten_8 (Flatten)         (None, 3136)              0         
                                                                 
 dense_61 (Dense)            (None, 128)               401536    
                                                                 
 batch_normalization_18 (Bat  (None, 128)              512       
 chNormalization)                                                
                                                                 
 activation_76 (Activation)  (None, 128)               0         
                                                                 
 dense_62 (Dense)            (None, 10)                1290      
                                                                 
 batch_normalization_19 (Bat  (None, 10)               40        
 chNormalization)                                                
                                                                 
 activation_77 (Activation)  (None, 10)                0         
                                                                 
=================================================================
Total params: 455,346
Trainable params: 454,878
Non-trainable params: 468
_________________________________________________________________
```
