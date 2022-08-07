
import numpy as np
import random

from keras.models import Model
from keras.layers import Dense, Dropout, Input, Reshape
from keras.layers import Lambda, Activation,Conv2D, MaxPooling2D 
from keras import backend as K
from keras.optimizers import Adam
import wave
from scipy.fftpack import fft

class ARCH_A(): 
    def __init__(self,basic_path,train_wavefile,PersianDict,NumberDict,export_path):

        self.basic_path=basic_path
        self.export_path=export_path
        self.train_wavefile=train_wavefile
        self.train_wavefile_length=len(train_wavefile)
        self.PersianDict=PersianDict
        self.NumberDict=NumberDict
        self.epoch = 1
        self.batch_size = 12
        self.save_step = 500

        self.Activation_hidden_layer='relu'
        self.kernel_initializer_hidden_layer='he_normal'
        self.CHARACTER_COUNT = 30
        self.MAX_STRING_OUTPUT = 178
        self.MAX_AUDIO_SAMPLE = 1600
        self.FEATURES_DIM = 200
        self._model_archA, self.base_model_archA = self.create_archA()


    def create_archA(self):
        #definition input layer
        input_layer = Input(shape=(self.MAX_AUDIO_SAMPLE, self.FEATURES_DIM, 1))

        #definition hidden layer
        LA_1 = Conv2D(32, (3,3), use_bias=False, activation=self.Activation_hidden_layer, padding='same', kernel_initializer=self.kernel_initializer_hidden_layer)(input_layer)
        LA_1 = Dropout(0.05)(LA_1)
        LA_2 = Conv2D(32, (3,3), use_bias=True, activation=self.Activation_hidden_layer, padding='same', kernel_initializer=self.kernel_initializer_hidden_layer)(LA_1)
        LA_3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(LA_2) # 池化层
        LA_3 = Dropout(0.05)(LA_3)
        LA_4 = Conv2D(64, (3,3), use_bias=True, activation=self.Activation_hidden_layer, padding='same', kernel_initializer=self.kernel_initializer_hidden_layer)(LA_3)
        LA_4 = Dropout(0.1)(LA_4)
        LA_5 = Conv2D(64, (3,3), use_bias=True, activation=self.Activation_hidden_layer, padding='same', kernel_initializer=self.kernel_initializer_hidden_layer)(LA_4)
        LA_6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(LA_5) # 池化层
        LA_6 = Dropout(0.1)(LA_6)
        LA_7 = Conv2D(128, (3,3), use_bias=True, activation=self.Activation_hidden_layer, padding='same', kernel_initializer=self.kernel_initializer_hidden_layer)(LA_6)
        LA_7 = Dropout(0.15)(LA_7)
        LA_8 = Conv2D(128, (3,3), use_bias=True, activation=self.Activation_hidden_layer, padding='same', kernel_initializer=self.kernel_initializer_hidden_layer)(LA_7)
        LA_9 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(LA_8) # 池化层
        LA_16 = Reshape((200*2, 3200*2))(LA_9)
        LA_16 = Dropout(0.3)(LA_16)
        LA_17 = Dense(2100, activation=self.Activation_hidden_layer, use_bias=True, kernel_initializer=self.kernel_initializer_hidden_layer)(LA_16)
        LA_17 = Dropout(0.3)(LA_17)
        LA_18 = Dense(self.CHARACTER_COUNT, use_bias=True, kernel_initializer=self.kernel_initializer_hidden_layer)(LA_17)

        #definition output layer
        y_pred = Activation('softmax')(LA_18)
        basic_model = Model(inputs = input_layer, outputs = y_pred)
        labels = Input(shape=[self.MAX_STRING_OUTPUT], dtype='float32')
        input_length = Input(shape=[1], dtype='int64')
        label_length = Input(shape=[1], dtype='int64')

        #definition lossfunction layer
        output_loss = Lambda(self.ctc_lambda_func, output_shape=(1,),name='ctc')([y_pred, labels, input_length, label_length])
        model = Model(inputs=[input_layer, labels, input_length, label_length], outputs=output_loss)
        model.summary()

        #definition optimizer
        opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = opt)

        return model, basic_model

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, :, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    #get sample from dataset
    def get_sample(self,random_id):

        filename = self.train_wavefile[random_id][0]
        string = self.train_wavefile[random_id][1]
        audio,fs=self.read_audio(self.basic_path + filename);
        x = np.zeros((1,79992),'int16');
        if(len(audio)< (5*fs)):
            audio=np.concatenate((audio,x),axis=1);

        feature = self.feature_extraction(audio,fs)
        feature = feature.reshape(feature.shape[0],feature.shape[1],1)

        label = self.convert_string_to_number(string)
        return feature,label
    
    #create generator from dataset
    def create_generator(self):
        text = np.zeros((self.batch_size,1), dtype = np.float)
        while True:
            X = np.zeros((self.batch_size, self.MAX_AUDIO_SAMPLE, self.FEATURES_DIM, 1), dtype = np.float)
            Y = np.zeros((self.batch_size, self.MAX_STRING_OUTPUT), dtype=np.int16)
            feature_length = []
            text_length = []

            for i in range(self.batch_size):
                ran_num = random.randint(0,self.train_wavefile_length - 1) 
                feature, data_text = self.get_sample(ran_num)  
                feature_length.append(feature.shape[0] // 8 + feature.shape[0] % 8)
                X[i,0:len(feature)] = feature
                Y[i,0:len(data_text)] = data_text
                text_length.append([len(data_text)])
            text_length = np.matrix(text_length)
            feature_length = np.array([feature_length]).T
            yield [X, Y, feature_length, text_length ], text
        pass
        
    #train dataset
    def train(self):
        
        generated_data = self.create_generator()       
        for epoch in range(self.epoch): 
            print('Epoch number is %d .' % epoch)
            step = 0 
            while True:
                try:
                    print('Number trained data is %d' % (step*self.save_step) )                    
                    self._model_archA.fit_generator(generated_data, self.save_step);
                    step=step+ 1
                except StopIteration:
                    print('Your data is not valid')
                    break
                
                name='archA_'+str(epoch)+'_'+str(step * self.save_step)
                self.Exportmodel(name)
                self.Reinit()

                
    def Reinit(self):
        num_data = self.train_wavefile_length
        data_count=32    
        try:
            ran_num = random.randint(0,num_data - 1)                        
            for i in range(data_count):
                f, l = self.get_sample((ran_num + i) % num_data) 
        except StopIteration:
            print('ERORE')
        
            
    #save model    
    def Exportmodel(self,name):
        fullname=self.export_path+name
        self._model_archA.save_weights(fullname + '.model')
        self.base_model_archA.save_weights(fullname + '.model.base')

    #load model
    def Importmodel(self,name):
        self._model_archA.load_weights(name)
        self.base_model_archA.load_weights(name + '.base')
       
    #test audio file
    def Test(self,path):
        
        batch_size = 1
        
        audio,fs = self.read_audio(path)
        x = np.zeros((1,79992),'int16');
        if(len(audio)< (5*fs)):
            audio=np.concatenate((audio,x),axis=1);

        features = np.array(self.feature_extraction(audio, fs),dtype = np.float)   
        length_audio_test = len(features)//8
        
        w=features.shape[0];h=features.shape[1];
        features = features.reshape(w,h,1)
                 
        input_net = np.zeros((batch_size),dtype = np.int32);input_net[0] = length_audio_test
        
        X = np.zeros((batch_size, self.MAX_AUDIO_SAMPLE , self.FEATURES_DIM, 1), dtype=np.float)
        
        for i in range(batch_size):
            X[i,0:len(features)] = features
        
        
        output = self.base_model_archA.predict(x = X)
        output =output[:, :, :]        
        rawtext = K.ctc_decode(output, input_net, greedy = True, beam_width=100, top_paths=1)     
        rawtext = K.get_value(rawtext[0][0]);rawtext=rawtext[0]
        
        text=self.convert_number_to_string(rawtext)        
        print(text)
        
        return text

        
        


    # convert string text to number
    def convert_string_to_number(self,string):        
        numstring=[]
        for i in range(len(string)):
            s=string[i]
            try:            
                index=self.PersianDict[s]
                numstring.append(index)
            except:
                pass
        numstring=np.array(numstring)
        return numstring
        
    # convert text number to string  
    def convert_number_to_string(self,string):        
        numstring=[]
        for i in range(len(string)):
            s=string[i]
            try:            
                index=self.NumberDict[s]
                numstring.append(index)
            except:
                pass
        numstring=''.join(numstring)
        return numstring
        

        
    #feature extraction function
    def feature_extraction(self,audio, fs):
        
        window_length = 25
        pad_start=160
        pad_finish=400
        resample_rate=10
        
        #create base window frame
        x=np.linspace(0, pad_finish - 1, pad_finish, dtype = np.int64)
        w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (pad_finish - 1) )
        
        #get range frame
        frame_length = int(len(audio[0])/fs*1000 - window_length) // resample_rate

        #get int wave file
        wavint = np.array(audio)

        #create buffer for features
        features = np.zeros((frame_length, self.FEATURES_DIM), dtype = np.float)
        features_temp = np.zeros((1, pad_finish), dtype = np.float)
        
        #get feature
        for i in range(0, frame_length):

            features_temp = wavint[0, i*pad_start:pad_finish+(i*pad_start)]*w
            features_temp = np.abs(fft(features_temp)) / wavint.shape[1]
            features[i]=features_temp[0:self.FEATURES_DIM]
        features = np.log(features + 1)
        return features

    #read audio function
    def read_audio(self,filename):
        audio = wave.open(filename,"rb");
        dataraw=audio.readframes(audio.getnframes())
        fs=audio.getframerate()        
        audioInt = np.fromstring(dataraw, dtype = np.short)
        audioInt.shape = -1, audio.getnchannels()
        audioInt = audioInt.T
        audio.close()
        return audioInt, fs
