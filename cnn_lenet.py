#!/usr/bin/env python
# coding: utf-8

# In[1]:


#-----------------------------
#  Configuracoes
#-----------------------------

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--parttest", help="% para teste", required=True)
parser.add_argument("-v", "--partvalid", help="% para validacao", required=True)
parser.add_argument("-e", "--nepocas", help="Numero de epocas", required=True)
args = parser.parse_args()

part_test = float(args.parttest)
part_valid = float(args.partvalid)
n_epocas = int(args.nepocas)
repeticao_model = 5
semente_model = True #False: pesos iniciais aleatorios, True: semente varia um a um até repeticao_model, int: pesos sempre os mesmo a partir desse int
arquit="Lenet Model"
dirfiles = "FB115_54"


# In[2]:


import os, datetime
import pickle
import numpy as np
#from functools import partial
from sklearn.model_selection import train_test_split
#from keras.utils import to_categorical

date_name = "./results/cnn/"+str(datetime.datetime.now())[:19] #Criando arquivo de resultados
os.mkdir(date_name)
temp_process = []

#-----------------------------
#  Leitura dos dados
#-----------------------------

# Cada arquivo possui 2000 imagens de 54x115 pixels cada e com 3 canais (RGB)
files = os.listdir(dirfiles)

for idx, f in enumerate(files):
    pathFile = open(os.path.join(dirfiles, f),"rb")
    if (idx==0):
        [X_all, Y_all] = pickle.load(pathFile)
        print(X_all.shape)
    else:
        [aux_X_all, aux_Y_all] = pickle.load(pathFile)
        X_all = np.append(X_all, aux_X_all, axis=0)
        Y_all = np.append(Y_all, aux_Y_all, axis=0)
        print(X_all.shape)

#Y_all = Y_all[:,0] #somente primeira coluna
#Y_all = to_categorical(Y_all)

X_train_all, X_test, y_train_all, y_test = train_test_split(X_all, Y_all, test_size = part_test, random_state = 13) # Treino e teste
X_train, X_valid, y_train, y_valid = train_test_split(X_train_all, y_train_all, test_size = part_valid, random_state = 13) # Treino e validação


# In[55]:


#-----------------------------
#  Gerando dados para as tradicionais
#-----------------------------

import matplotlib.pyplot as plt

for d1 in os.listdir('./faber_gray_data_set/train'):
    os.remove('./faber_gray_data_set/train/'+d1)
for d1 in os.listdir('./faber_gray_data_set/test'):
    os.remove('./faber_gray_data_set/test/'+d1)

i = 0
for imgs in X_train:
    #print('./faber_gray_data_set/train/'+str(int(y_train[i][0]))+"_"+str(i)+".png")
    plt.imsave("./faber_gray_data_set/train/"+str(int(y_train[i][0]))+"_"+str(i)+".png", imgs[:,:,1], cmap='gray',)
    i = i + 1
i = 0
for imgs in X_test:
    #print('./faber_gray_data_set/test/'+str(int(y_train[i][0]))+"_"+str(i)+".png")
    plt.imsave("./faber_gray_data_set/test/"+str(int(y_test[i][0]))+"_"+str(i)+".png", imgs[:,:,1], cmap='gray',)
    i = i + 1


# In[2]:


#Metricas para avaliacao do modelo

loss = []
val_loss = []
accuracy = []
val_accuracy = []

prediction = []

#------------------------
# MODELO
#------------------------

from numpy.random import seed
from tensorflow import set_random_seed
import keras
import keras.layers as layers
from keras.models import Sequential
#from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard

from sklearn import metrics

sgd = keras.optimizers.SGD(lr=0.01, nesterov=True)

for i in range(repeticao_model):
    
    temp_in = datetime.datetime.now() #tempo inicial de processamento
    
    #Inicializando e definindo os pesos da rede
    if(semente_model == True):
        seed(i)
        set_random_seed(i)
        print("semente_model =", i)
    elif(type(semente_model) == int):
        seed(semente_model)
        set_random_seed(semente_model)
        print("semente_model = ", semente_model)
    
    #Inicio do modelo-----------------------------------------------------------------------------------------------------------------------------
    model = keras.Sequential()

    model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(54, 115, 3)))
    model.add(layers.AveragePooling2D())
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(units=120, activation='relu'))
    model.add(layers.Dense(units=84, activation='relu'))
    model.add(layers.Dense(units=2, activation = 'softmax'))

    model.summary()
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
    #Final do modelo-----------------------------------------------------------------------------------------------------------------------------

    history = model.fit(
        x = X_train,
        y = y_train,
        epochs = n_epocas,
        validation_data = (X_valid, y_valid)
    )
    
    temp_fi = datetime.datetime.now() #tempo final de processamento
    
    temp_process.append(temp_fi - temp_in)
    loss.append(history.history['loss'])
    val_loss.append(history.history['val_loss'])
    accuracy.append(history.history['accuracy'])
    val_accuracy.append(history.history['val_accuracy'])
    
    #-----------------------------
    #  Teste do modelo
    #-----------------------------
    
    prediction.append(model.predict_classes(X_test))


# In[3]:


epoca = []
metrica = []

k = ['Train loss', 'Val loss', 'Train Acc', 'Val Acc']

for i in range(repeticao_model*4):
    for j in range(1,n_epocas+1):
        epoca.append(j)
        metrica.append(k[int(i/repeticao_model)])


# In[4]:


import pandas as pd

dados = np.array([loss, val_loss, accuracy, val_accuracy]).flatten().tolist()
d = {'Variação': dados, 'Métricas': metrica, 'Épocas': epoca}
df = pd.DataFrame(data=d)

df.to_csv(date_name+"/dados_analise.csv", index = False)
df


# In[5]:


import seaborn as sns
#import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(12,9)})
sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.95'}) #darkgrid, whitegrid, dark, white, and ticks
sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})

ax = sns.lineplot(x="Épocas", y="Variação", hue="Métricas",
                  data=df)

ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5), ncol=1)
ax.set(xlim=(1,n_epocas), ylim=(None,None))
#sns.despine()

plt.savefig(date_name+"/analise.png", bbox_inches='tight')


# In[6]:


tam=len(temp_process)
temp_total = temp_process[0]
for i in range(1,tam):
    temp_total = temp_process[i] + temp_total
    
temp_med = temp_total/tam
print()


# In[7]:


real = []

for i in y_test:
    real.append(int(i[1]))

print("Gerando metricas...") #sklearn: acuracia, auc, matriz de confusao...

acuracia = []
auc = []
mat_confu = []

for i in range(repeticao_model):
    acuracia.append(metrics.accuracy_score(np.array(real), prediction[i]))
    auc.append(metrics.roc_auc_score(np.array(real), prediction[i]))
    mat_confu.append(metrics.confusion_matrix(np.array(real), prediction[i]))
    
VP = []
VN = []
FP = []
FN = []

for i in range(repeticao_model):
    VP.append(mat_confu[i][0][0])
    VN.append(mat_confu[i][0][1])
    FP.append(mat_confu[i][1][0])
    FN.append(mat_confu[i][1][1])

dn = {'VP': VP, 'VN': VN, 'FP': FP, 'FN': FN}
dfn = pd.DataFrame(data=dn)

dfn.to_csv(date_name+"/matriz_confusao.csv", index = False)
#print(mat_confu)
dfn


# In[8]:


print("Gravando metricas...")

fs = open(date_name+"/relatorio_de_processamento.txt", "w")

fs.writelines("\n*********************" + " Informações " + "*********************")
fs.write("\n\nArquiteura:\t\t\t"+arquit+"\nTeste:\t\t\t\t"+str(part_test)+ "\nValidação:\t\t\t"+str(part_valid)+ 
         "\nNúmero de repetições:\t\t"+str(repeticao_model)+ "\nNúmero de épocas:\t\t"+str(n_epocas)+ 
         "\nSemente do modelo:\t\t"+ str(semente_model)+ "\nTempo total de processamento:\t"+ str(temp_total)+
         " h\nTempo médio de processamento:\t"+str(temp_med)+" h")

fs.writelines("\n\n***********************" + " Métricas " + "***********************")
#fs.write("\n\nPrincipais:\n")
fs.write("\n\nAcurácia média:\t\t\t"+str(np.mean(acuracia))+"\nAUC média:\t\t\t"+str(np.mean(auc))+
         "\n\nMatriz de confusão somada:\n\n"+ str(np.sum(mat_confu, axis = 0)))
#fs.write("\n\nGerais:\n\n"+relatorio)

fs.writelines("\n\n******************" + " Arquitetura gerada " + "******************\n\n")
model.summary(print_fn=lambda x: fs.write(x + '\n'))

fs.close()


# In[9]:


import smtplib

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def envia_email(fromaddr, toaddr, assunto, menssagem, filename, caminho):
    
    #toaddr = 'E-mail de destino'
    msg = MIMEMultipart()

    msg['From'] = fromaddr 
    msg['To'] = toaddr
    msg['Subject'] = assunto #"Titulo do e-mail"
    
    fromaddr = 'avisa.ic@pidtec.com.br'

    body = menssagem #"\nCorpo da mensagem"

    msg.attach(MIMEText(body, 'plain'))

    #filename = 'teste.pdf'
    for i in filename:
        pp = caminho+"/"+str(i)

        attachment = open(pp,'rb')

        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % i)

        msg.attach(part)

        attachment.close()

    server = smtplib.SMTP( 'smtp.hostinger.com.br', 587)
    server.starttls()
    server.login(fromaddr, 's^Lg0~Th')
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr.split(','), text)
    server.quit()
    print('\nEmail enviado com sucesso!')
    
fnames = ["analise.png", "relatorio_de_processamento.txt", "matriz_confusao.csv"] #"dados_analise.csv",
destinatarios = "gustavo.ubeda@unesp.br, andre.rossi@unesp.br" #"email1, email2, ..."
#email_resposta, email_destino, assunto, mensagem, anexos, caminho para os anexos
envia_email('avisa.ic@pidtec.com.br', destinatarios, "CNN_lenet (grid): "+date_name[-19:], 
            "Para: "+str(X_all.shape)+" imagens\n\nEm anexo os resultados gerados:\n\nAtt, Gustavo Ubeda", fnames, date_name)

