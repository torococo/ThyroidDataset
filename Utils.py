import struct

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing.dummy import Pool
import csv
from matplotlib.patches import Rectangle

#region CONSTANTS
DEFAULT_CMAP='viridis'
#endregion

# BINARY DATA WRITING UTILS
def WriteBinData(outFile,typeChar,count,*vals):
  fmt=">"+typeChar*count
#  if count==1: struct.pack(fmt,*vals)
#  else:
  outFile.write(struct.pack(fmt,*vals))

def GetBinData(rawFile,typeChar,count=1):
  """get data as char"""
  #typechars: i:int,d:double,q:long,f:float
  size=struct.calcsize(typeChar)*count
  fmt=">"+typeChar*count
  val=rawFile.read(size)
  if count==1:
    return struct.unpack(fmt,val)[0]
  return np.asarray(struct.unpack(fmt,val))

# MISC UTILS

def AllCombos(Objs):
  allCombos=[]
  for x,namex in enumerate(Objs):
    for y,namey in enumerate(Objs):
      if x<y:
        allCombos.append((namex,namey))
  return allCombos

# PLOTTING UTILS

def SetLabels(ax, title=None, xLabel=None, yLabel=None, xLim=None, yLim=None, xTicks=None, yTicks=None, xTickLabels=None, yTickLabels=None,bEqualAxis=False):
  if title is not None: ax.set_title(title)
  if xLabel is not None: ax.set_xlabel(xLabel)
  if yLabel is not None: ax.set_ylabel(yLabel)
  if xTicks is not None: ax.set_xticks(xTicks)
  if yTicks is not None: ax.set_yticks(yTicks)
  if xTickLabels is not None: ax.set_xticklabels(xTickLabels)
  if yTickLabels is not None: ax.set_yticklabels(yTickLabels)
  if xLim is not None: ax.set_xlim(*xLim)
  if yLim is not None: ax.set_ylim(*yLim)
  if bEqualAxis: ax.axis('equal')

def GenAxs(rows,cols):
  f,axs=plt.subplots(rows,cols)
  ret=[]
  if rows==1 and cols==1:
    return [axs]
  if rows==1 or cols==1:
    for i in range(rows*cols):
      ret.append(axs[i])
  else:
    for i in range(rows):
      for j in range(cols):
        ret.append(axs[i][j])
  return ret

def Legend(ax, labels, colors=None):
  if colors:
    rects = [Rectangle((0, 0), 1, 1, fc=colors[i], fill=True, edgecolor='none', linewidth=0) for i in
             range(len(labels))]
  else:
    rects = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0) for _ in range(len(labels))]
  ax.legend(rects, labels)

def PlotScatter(ax,xs,ys,size,weights,cmap,bColorBar=False,colorBarLabel=''):
  cax=ax.scatter(xs,ys,c=weights,cmap=cmap,s=size)
  if bColorBar:plt.colorbar(cax,ax=ax,label=colorBarLabel)

def PlotLine(ax,xs,ys,lineFmt):
  ax.plot(xs,ys,lineFmt)

def PlotBar(ax,barPositions,barVals,barWidth,color,stdDevs=None):
  ax.bar(barPositions,barVals,barWidth,color=color,yerr=stdDevs)

def PlotBinMeanStd(ax,xs,ys,nBins,lineFmt,thickness=1,xLim=None,yLim=None):
  if xLim is None: xLim=(min(xs),max(xs))
  if yLim is None: yLim=(min(ys),max(ys))
  if lineFmt is not None:
    counts,edges=np.histogram(xs,bins=nBins)
    weights,edges=np.histogram(xs,bins=nBins,weights=ys)
    weightsSq,edges=np.histogram(xs,bins=nBins,weights=ys*ys)
    mean=weights/counts
    std=np.sqrt(weightsSq/counts-mean*mean)
    mids=(edges[1:]+edges[:-1])/2
    ax.errorbar(mids,mean,yerr=std,fmt=lineFmt,linewidth=thickness,elinewidth=thickness)

def PlotImgColormap(ax, vals, colorMap=DEFAULT_CMAP, bColorBar=True, weightLim=None, center=None, colorBarLabel='', extent=None):
  if weightLim is None: weightLim=(vals.min(),vals.max())
  if center is not None:
    maxDelta=np.max(np.abs(np.array([center-weightLim[0],center-weightLim[1]])))
    weightLim=(center+maxDelta,center-maxDelta)
  cax=ax.imshow(vals,cmap=colorMap,interpolation='nearest',origin='lower',aspect='auto',vmin=weightLim[0],vmax=weightLim[1],extent=extent)
  if bColorBar:plt.colorbar(cax,ax=ax,label=colorBarLabel)

def Histogram2D(xs, ys, nBins, weights=None, xLim=None, yLim=None):
  if xLim is None: xLim=(min(xs),max(xs))
  if yLim is None: yLim=(min(ys),max(ys))
  counts,xEdges,yEdges=np.histogram2d(xs,ys,bins=nBins,range=(xLim,yLim),weights=weights)
  extent=[xEdges[0],xEdges[-1],yEdges[0],yEdges[-1]]
  return counts,extent

def Histogram3D(xs,ys,zs,nBins,weights=None,xLim=None,yLim=None,zLim=None):
  inPoints=np.array(np.vstack((xs,ys,zs)).T,dtype=np.float32)
  if xLim is None: xLim=(min(xs),max(xs))
  if yLim is None: yLim=(min(ys),max(ys))
  if zLim is None: zLim=(min(zs),max(zs))
  ranges=np.array([xLim,yLim,zLim])
  print(np.isfinite(ranges))
  counts,edges=np.histogramdd(inPoints,bins=nBins,range=ranges,weights=weights)
  return counts

def ClearTerminal(): print(chr(27) + "[2J")

def MeanHistogram3D(xs,ys,zs,nBins,weights,xLim=None,yLim=None,zLim=None):
  counts=Histogram3D(xs,ys,zs,nBins,None,xLim,yLim,zLim)
  vals=Histogram3D(xs,ys,zs,nBins,weights,xLim,yLim,zLim)
  means=vals/counts
  return means.T

def MeanHistogram2D(xs,ys,nBins,weights,xLim=None,yLim=None):
  counts,extent=Histogram2D(xs, ys, nBins, None, xLim, yLim)
  vals,extent=Histogram2D(xs, ys, nBins, weights, xLim, yLim)
  means=vals/counts
  return means.T,extent
  #cax=ax.imshow(means.T,extent=extent,origin='lower',aspect='auto',interpolation="Nearest",cmap=cmap)
  #if bColorBar:plt.colorbar(cax,ax=ax,label=colorBarLabel)

def RunThreads(nRuns,nThreads,RunF):
  #RunF will take thread index as only arg
  pool=Pool(nThreads)
  ret=pool.map(RunF,range(nRuns))
  pool.close()
  pool.join()
  return ret

def StackPlot(ax,pops,stepTimes):#pops should be [[pop1],[pop2],...]
  ax.stackplot(stepTimes,pops,baseline='sym')

# PARAM SWEEP CLASS

class FSM:
  #edgefuns are lists with [state,chars,CharFunction]
  def __init__(self,stateFuns,initState):
    self.stateFuns=stateFuns
    self.state=initState
    self.CurrFun=stateFuns[initState]

  #reads file 1 char at a time
  def ReadFile(self,path):
    f=open(path,'r')
    for c in f.read():
      self.ProcChar(c)

  #advances the FSM by 1 character
  def ProcChar(self,c):
    self.state=self.CurrFun(c)
    self.CurrFun=self.stateFuns[self.state]

class HyperSweeper:

  def __init__(self,name,Run):
    self.Run=Run
    self.name=name
    self.data=[]
    self.paramNames=[]
    self.RandomParamFs=[]

  def AddParam(self,name,RandomVal):#random val should return an evenly distributed value in some interval
    if len(self.data) > 0:
      raise Exception("can't add Params with data already existing")
    self.paramNames.append(name)
    self.RandomParamFs.append(RandomVal)

  def GetParamDict(self,index):
    vals=self.data[index]
    return dict(zip(self.paramNames,self.data[index]))

  def _RandomRun(self,index=None):
    args={}
    vals=[]
    for i in range(len(self.paramNames)):
      val=self.RandomParamFs[i]()
      args[self.paramNames[i]]=val
      vals.append(val)
    vals.append(self.Run(args))
    return vals

  def RandomRunSingle(self):
    self.data.append(self._RandomRun())

  def RandomRunThreads(self,nRuns,nThreads):
    RunThreads(nRuns,nThreads,self._RandomRun)

  def SaveCSV(self,path):
    df=pd.DataFrame(self.data,columns=self.paramNames+['Score'])
    df.to_csv(path,index=False)

# TF DATA PROC

def SeparateData(inputs,outputs,trainProp,batchSize,bShuffle):
  if inputs.shape[0]!=outputs.shape[0]: raise Exception("inputs and outputs must have the same number of rows!")
  nEntries=inputs.shape[0]
  nTraining=((nEntries*trainProp)//batchSize)*batchSize
  indices=np.arange(0,nEntries)
  if bShuffle:
    np.random.shuffle(indices)
    print(indices[0:nTraining])
    out1=inputs[indices[0:nTraining]]

    out2=outputs[indices[0:nTraining]]
    out3=inputs[indices[nTraining:]]
    out4=outputs[indices[nTraining:]]
  return[inputs[indices[0:nTraining]],outputs[indices[0:nTraining]]],[inputs[indices[nTraining:]],outputs[indices[nTraining:]]]

def GenBatchSet(inputs,outputs,batchSize):
  if inputs.shape[0]!=outputs.shape[0]: raise Exception("inputs and outputs must have the same number of rows!")
  if inputs.shape[0]%batchSize!=0: raise Exception("batch size does not divide evenly by input size!")
  nEntries=inputs.shape[0]
  indices=np.arange(0,nEntries)
  np.random.shuffle(indices)
  batchInputs=inputs[indices].reshape((nEntries//batchSize,batchSize,inputs.shape[1]))
  batchOutputs=outputs[indices].reshape((nEntries//batchSize,batchSize,outputs.shape[1]))
  return batchInputs,batchOutputs

def ColTransform(df, columnToTransform, colInfos, bDropOriginal=False):
  #pass in a list of lists with [name,[[name,val]],default]
  for colInfo in colInfos:
    df[colInfo[0]]=colInfo[2]
  for colInfo in colInfos:
    for [subName,subVal] in colInfo[1]:
      df.loc[df[columnToTransform] == subName, colInfo[0]]=subVal
  if bDropOriginal:
    df=df.drop(columnToTransform, axis=1)
  return df

def MultiColTransform(df, columnsToTransform, colInfos, bDropOriginal=False):
  #pass in a list of lists with [name,[[[name],val]],[default]]
  for colInfo in colInfos:
    df[colInfo[0]]=colInfo[2]
  for colInfo in colInfos:
    for [subNames,subVal] in colInfo[1]:
      mask=np.ones((df.shape[0]), dtype=bool)
      for i in range(len(subNames)):
        subName=subNames[i]
        column=df[columnsToTransform[i]]
        np.logical_and(mask,column.where(column==subName),mask)
      df[mask]=subVal
  if bDropOriginal:
    for column in columnsToTransform:
      df.drop(column)
  return df


def GenMissingDataColumns(data):return pd.isnull(data)

def ConcatMissingColumns(data,missingColumns,labels):
  outputLabels=labels+[label+"_exists" for label in labels]
  return pd.DataFrame(np.concatenate((data, missingColumns), axis=1), columns=outputLabels)


#optionsDict should be dict of key:value pairs, where keys are strings
def TerminalMenu(displayText,optionsDict):
  choice=""
  while choice not in optionsDict:
    os.system('clear')
    print(displayText)
    choice=input("\n>>  ").lower()
  return optionsDict[choice]

def MinSecString(seconds):
  mins=str(int(seconds/60))
  secs=str(int(seconds%60))
  if len(secs)==1:secs="0"+secs
  return mins+":"+secs

class TFinterface:
  def __init__(self,graph,OutputTensor,ErrorTensor,TrainingTensor,GradientTensor,InitVarsFunction,inputsPlaceholderName,outputsPlaceholderName,dropoutPlaceholderName,gradientMasksPlaceholderName):
    self.graph=graph
    self.OutputTF=OutputTensor
    self.ErrorTF=ErrorTensor
    self.TrainTF=TrainingTensor
    self.GradientTF=GradientTensor
    self.InitVarsTF=InitVarsFunction
    self.sInputsPL=inputsPlaceholderName
    self.sOutputsPL=outputsPlaceholderName
    self.sDropoutPL=dropoutPlaceholderName
    self.sGradientMasksPL=gradientMasksPlaceholderName
    self.sess=None

  def StartSession(self):
    self.sess=tf.Session(graph=self.graph)
    self.sess.run(self.InitVarsTF)

  def Train(self,batchIn,batchOut,dropoutKeepProb):
    outputs,error,trainRes=self.sess.run((self.OutputTF,self.ErrorTF,self.TrainTF),feed_dict={self.sInputsPL+":0":batchIn,self.sOutputsPL+":0":batchOut,self.sDropoutPL+":0":dropoutKeepProb})
    return outputs,error,trainRes

  def Test(self,testIn,testOut):
    outputs,error=self.sess.run((self.OutputTF,self.ErrorTF),feed_dict={self.sInputsPL+":0":testIn,self.sOutputsPL+":0":testOut,self.sDropoutPL+":0":1})
    return outputs,error

  def GetGradientValues(self,inputSet,outputSet,outputMask):
    masks=np.tile(outputMask,(outputSet.shape[0],1))
    gradVals=self.sess.run(self.GradientTF,feed_dict={self.sInputsPL+":0":inputSet,self.sOutputsPL+":0":outputSet,self.sDropoutPL+":0":1,self.sGradientMasksPL+":0":masks})
    return np.asarray(gradVals[0])

  def GenAndRunBatchTraining(self,trainInputs,trainOutputs,batchSize,nInputSweeps,dropKeepProb):
    errorVals=[]
    for sweep in range(nInputSweeps):
      Batches=GenBatchSet(trainInputs,trainOutputs,batchSize)
      for iBatch in range(len(Batches[0])):
        trainOut,error,trainRes=self.Train(Batches[0][iBatch],Batches[1][iBatch],dropKeepProb)
        errorVals.append(error)
    return errorVals

def SumGradients(gradientValues):
  return np.sum(np.abs(gradientValues),axis=0)

def PlotGradSums(ax,gradSums,inputLabels):
  barLocs=np.arange(gradSums.shape[0])
  PlotBar(ax,np.arange(gradSums.shape[0]),gradSums,1,'r')
  SetLabels(ax,title="absolute value gradient sums",xTicks=barLocs+0.5)
  ax.set_xticklabels(inputLabels,rotation=45)

def PlotTrainingError(ax,errorVals,testError):
  batches=np.arange(1,len(errorVals)+1)
  PlotLine(ax,batches,errorVals,'k-')
  PlotLine(ax,[1,len(errorVals)+1],[testError,testError],'r-')
  Legend(ax,["train error","test error"],colors=["k","r"])
  SetLabels(ax,title="Train and Test Error Function Values",xLabel="Batch",yLabel="Error",yLim=[0,max(errorVals)])


#def InputEliminator(TFinterface,inputs,outputs,inputLabels):
#  dropped

class FullyConnectedNetwork:
  def __init__(self,layerSizes,nInputs,nOutputs):
    self.sess=None
    self.graph=tf.Graph()
    with self.graph.as_default():
      #input placeholder definition
      inputsTF=tf.placeholder(tf.float32,[None,nInputs],name='inputsPL')
      outputsTF=tf.placeholder(tf.float32,[None,nOutputs],name='outputsPL')
      dropoutProbTF=tf.placeholder(tf.float32,name='dropoutPL')

      # network definition
      LastLayerTF=inputsTF
      for size in layerSizes:
        LastLayerTF=slim.fully_connected(LastLayerTF,size)
        tf.nn.dropout(LastLayerTF,dropoutProbTF)
      LastLayerTF = slim.fully_connected(LastLayerTF,nOutputs,activation_fn=None)
      self.OutputLayerTF=LastLayerTF

      #error function
      self.ErrorTF=tf.reduce_mean(tf.square(LastLayerTF-outputsTF))

      #training
      self.TrainTF=tf.train.AdamOptimizer().minimize(self.ErrorTF)

      #getting gradient values
      outMasksTF=tf.placeholder(tf.float32,[None,nOutputs],name='outMasksPL')
      self.GradsTF=tf.gradients(self.OutputLayerTF,inputsTF,outMasksTF)

      #variable initialization
      self.InitVarsTF=tf.global_variables_initializer()

class FullyConnectedNetworkWithMissingOutputs:
  def __init__(self,layerSizes,nInputs,nOutputs):
    #bInputsWithNan and bOutsWithNan are arrays with true if that output column has nan in it
    self.sess=None
    self.graph=tf.Graph()
    with self.graph.as_default():
      #input placeholder definition
      inputsTF=tf.placeholder(tf.float32,[None,nInputs],name='inputsPL')
      outputsTF=tf.placeholder(tf.float32,[None,nOutputs*2],name='outputsPL')
      dropoutProbTF=tf.placeholder(tf.float32,name='dropoutPL')
      outputValsTF,outputExistsTF=tf.split(outputsTF,2,1)

      # network definition
      LastLayerTF=inputsTF
      for size in layerSizes:
        LastLayerTF=slim.fully_connected(LastLayerTF,size)
        tf.nn.dropout(LastLayerTF,dropoutProbTF)
      LastLayerTF = slim.fully_connected(LastLayerTF,nOutputs,activation_fn=None)
      self.OutputLayerTF=LastLayerTF

      #error function
      self.ErrorTF=(tf.reduce_mean(tf.square((LastLayerTF-outputValsTF)*outputExistsTF)))/tf.reduce_mean(outputExistsTF)

      #training
      self.TrainTF=tf.train.AdamOptimizer().minimize(self.ErrorTF)

      #getting gradient values
      outMasksTF=tf.placeholder(tf.float32,[None,nOutputs],name='outMasksPL')
      self.GradsTF=tf.gradients(self.OutputLayerTF,inputsTF,outMasksTF)

      #variable initialization
      self.InitVarsTF=tf.global_variables_initializer()
