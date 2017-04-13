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

def GetNanDropLabels(raw,acceptableNanProp):
  labels=list(raw)
  minAcceptable=acceptableNanProp*raw.shape[0]
  #print(minAcceptable)
  nullSums= raw.isnull().sum()
  ret=[]
  for label in labels:
    if nullSums[label]>minAcceptable: ret.append(label)
  return ret


def PreprocData():
  #separate inputs and outputs
  raw = pd.read_excel("TyroidData_Categorized_PV.xlsx")
  dropLabels=GetNanDropLabels(raw,0.9)


  OUTPUT_COLS = ["Histol-definitive_1_1.1_1.3",	"Histol-definitive_1.2_1.4_1.5",	"Histol-definitive_2.1", "Histol-definitive_2_2.2_2.3_2.4_2.5_2.6_2.7",	"Histol-definitive_3",	"Histol-definitive_4_4.1_5_5.1",	"Histol-definitive_6",	"Histol-definitive_7",	"malignant",	"mal_wo_NIFT",	"ETE_p_s_modified",	"R_status_s_modified",	"caps_invas_s_modified", "vasc_invas_s_modified",	"ENE_s",	"#LNM_s",	"T_dx_s_modified",	"N_dx_s_modified",	"M_dx_s_modified",	"Stage_s_modified",	"cancer risk", "Recurrence_s_modified",	"disease_status_last_f-u_s_modified"]
  #for label in dropLabels:
  #  if label in OUTPUT_COLS:OUTPUT_COLS.remove(label)

  DROP_COLS = ["ID","ETE_p_s","Gender","Other_ca","echogen_1","calcif_1","US_pat_1","vascul_1","Beth_gp_s","OP_s","thy_dysfx","Scint_scan_s","echogen_MM","calcifications_MM","US_pat_MM_mod","vasc_MM","echogen_CL","margin_CL_classified","US_pat_CL_mod","vasc_CL","Histol-definitive","R_status_s","caps_invas_s","vasc_invas_s","T_dx_s","N_dx_s","M_dx_s","Stage_s","Recurrence_s","disease_status_last_f-u_s"]+dropLabels

  raw = raw.drop(DROP_COLS, axis=1)
 # raw.to_excel("test.xlsx")
  outputs=raw[OUTPUT_COLS]
  outputsMissing=GenMissingDataColumns(outputs)
  outputs = outputs.replace(" ", 0)
  outputs = outputs.replace("NaN",0)
  inputs=raw.drop(OUTPUT_COLS,axis=1)
  inputsMissing = GenMissingDataColumns(inputs)
  inputs=inputs.replace(" ",0)
  inputs=inputs.replace("NaN",0)
  #normalize inputs
  inNormalizer=sklearn.preprocessing.StandardScaler().fit(inputs)
  outNormalizer=sklearn.preprocessing.StandardScaler().fit(outputs)
  inputsNormed=inNormalizer.transform(inputs)
  outputsNormed=outNormalizer.transform(outputs)
  inputs=ConcatMissingColumns(inputs,inputsMissing)
  outputs=ConcatMissingColumns(outputs,outputsMissing)

  #separate into train and test set
  return (inputsNormed,outputsNormed),(inNormalizer,outNormalizer),(list(inputs),list(outputs)),(outputs.shape[1],inputs.shape[1])


BATCH_SIZE=50
data,normalizers,labels,IOsizes=PreprocData()

trainingData,testingData=SeparateData(data[0],data[1],0.9,BATCH_SIZE,True)
#create graph
#netTF=FullyConnectedNetwork([100,100,100],trainingData[0].shape[1],trainingData[1].shape[1])
netTF=FullyConnectedNetworkWithMissingOutputs([100],trainingData[0].shape[1],int(trainingData[1].shape[1]/2))

#pass graph components to interface
net=TFinterface(netTF.graph,netTF.OutputLayerTF,netTF.ErrorTF,netTF.TrainTF,netTF.GradsTF,netTF.InitVarsTF,"inputsPL","outputsPL","dropoutPL","outMasksPL")
#start graph
net.StartSession()
#training
errorVals=net.GenAndRunBatchTraining(trainingData[0],trainingData[1],BATCH_SIZE,100,0.5)
#testing
outputs,error=net.Test(testingData[0],testingData[1])

#get gradient values on testing data
gradVals=net.GetGradientValues(testingData[0],testingData[1],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) #,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) #[1,0])
gradSums=SumGradients(gradVals)

#Plot training and testing error
axs=GenAxs(1,1)
#axs=GenAxs(1,1)
PlotTrainingError(axs[0],errorVals,error)
predictedOutputs=normalizers[1].inverse_transform(outputs)#[:,0:outputsWithZeros.shape[1]/2]
outputFrame=pd.DataFrame(predictedOutputs,columns=labels[1])
outputFrame.to_csv("predictedOutputs.csv")
actualOutputs=normalizers[1].inverse_transform(testingData[1][:,0:testingData[1].shape[1]/2])#[:,0:testingData[1].shape[1]/2]
actualFrame=pd.DataFrame(actualOutputs,columns=labels[1])
actualFrame.to_csv("actualOutputs.csv")

print(predictedOutputs)
plt.show()

#axs=GenAxs(1,1)
#PlotGradSums(axs[0],gradSums,labels[0])
#plt.show()

#df = pd.DataFrame(gradSums)
#df.to_csv("SaveGradSums_new.csv")

#dfL = pd.DataFrame(labels[0])
#dfL.to_csv("SaveLabels.csv")