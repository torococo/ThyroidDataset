import pandas as pd
from Utils import *

df=pd.DataFrame([['a','b'],['c','d']],columns=['first','second'])
print(df)
df=ColTransform(df,'first',[['new1',[['a','1']],'0']],True)
print(df)
