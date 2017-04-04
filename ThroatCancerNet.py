import pandas as pd
import sklearn
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.preprocessing import Normalizer
from Utils import *
import pickle
import numpy as np

from Utils import *

"""DATA PREPROC"""
def PreprocData():
  #separate inputs and outputs
  raw = pd.read_excel("USETHISFM.xlsx")
  #missingpercentage = pd.read_csv("Missing.csv")

  #indexmissing = np.where(missingpercentage >= 0.5)
  #namemissing = raw[indexmissing[0]]

  #print(namemissing)

  OUTPUT_COLS = ["'Histol-definitive'"]  # ,"'mal_wo_NIFT'",	"'capsule_s'",
# "'capsule'",	"'caps_invas_s'",	"'vasc_invas_s'",
# "'p-size-s'",	"'#LNM_s'",	"'#LN_resec_s'",
# "'#LN_resec'",	"'size_LN_s'",	"'conc. Cancer_s'",	"'T_dx_s'",
# "'N_dx_s'",	"'M_dx_s'",	"'Stage_s'",	"'pt_dx'",	"'Recurrence_s'","'malignant'",	"'cancer risk'"]
# DROP_COLS=["'cancer risk'","'ID'"] #,"'malignant'"]

  DROP_COLS = ["'ID'"]
  ## ,namemissing]
# DROP_COLS=["'ID'","'mal_wo_NIFT'",	"'capsule_s'",
#           "'capsule'",	"'caps_invas_s'",	"'vasc_invas_s'",
#           "'p-size-s'",	"'#LNM_s'",	"'#LN_resec_s'",
#           "'#LN_resec'",	"'size_LN_s'",	"'conc. Cancer_s'",	"'T_dx_s'",
#           "'N_dx_s'",	"'M_dx_s'",	"'Stage_s'",	"'pt_dx'",	"'Recurrence_s'","'malignant'",	"'cancer risk'"]

  raw = raw.drop(DROP_COLS, axis=1)
  raw = raw.replace("NaN", 0)

  outputs=raw[OUTPUT_COLS]
  inputs=raw.drop(OUTPUT_COLS,axis=1)
  #normalize inputs
  inNormalizer=sklearn.preprocessing.Normalizer().fit(inputs)
  outNormalizer=sklearn.preprocessing.Normalizer().fit(outputs)
  inputsNormed=inNormalizer.transform(inputs)
  outputsNormed=outNormalizer.transform(outputs)
  #separate into train and test set
  return (inputsNormed,outputsNormed),(inNormalizer,outNormalizer),(list(inputs),list(outputs))


BATCH_SIZE=50
data,normalizers,labels=PreprocData()

trainingData,testingData=SeparateData(data[0],data[1],0.4,BATCH_SIZE,True)
#create graph
#netTF=FullyConnectedNetwork([100,100,100],trainingData[0].shape[1],trainingData[1].shape[1])
netTF=FullyConnectedNetwork([10],trainingData[0].shape[1],trainingData[1].shape[1])

#pass graph components to interface
net=TFinterface(netTF.graph,netTF.OutputLayerTF,netTF.ErrorTF,netTF.TrainTF,netTF.GradsTF,netTF.InitVarsTF,"inputsPL","outputsPL","dropoutPL","outMasksPL")
#start graph
net.StartSession()
#training
errorVals=net.GenAndRunBatchTraining(trainingData[0],trainingData[1],BATCH_SIZE,100,0.5)
#testing
outputs,error=net.Test(testingData[0],testingData[1])

#get gradient values on testing data
gradVals=net.GetGradientValues(testingData[0],testingData[1],[1]) #,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) #[1,0])
gradSums=SumGradients(gradVals)

#Plot training and testing error
axs=GenAxs(1,1)
#axs=GenAxs(1,1)
PlotTrainingError(axs[0],errorVals,error)
plt.show()

axs=GenAxs(1,1)
PlotGradSums(axs[0],gradSums,labels[0])
plt.show()

df = pd.DataFrame(gradSums)
df.to_csv("SaveGradSums_HisDef.csv")

dfL = pd.DataFrame(labels[0])
dfL.to_csv("SaveLabels.csv")