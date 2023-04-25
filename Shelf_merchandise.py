# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:03:19 2023

@author: Akshays
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import optimize
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
np.seterr(divide='ignore', invalid='ignore')
import time
from time import sleep
from time import time
import math
import webbrowser
from math import isnan
from pulp import *
from collections import Counter
from more_itertools import unique_everseen
import numpy as np
import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import time
from time import sleep

rad = st.sidebar.radio('Navigation',['Shelf Space Optimization','Market Basket'])

if rad=='Shelf Space Optimization':
    st.header('SHELF SPACE OPTIMIZATION')

    st.image('Shelf_optimization.jpg',use_column_width=True)
    
    file1 = st.file_uploader('Upload a file',key='f1')

    if file1 is not None:
        sales=pd.read_excel(file1,header=None)
        
        lift=sales.iloc[2:,1:]
        lift=np.array(lift)
        lift = lift.astype(np.int)

        brands=sales.iloc[0:1,:]
        brands=np.array(brands)
        brands=np.delete(brands,0)
        brands=brands.tolist()

        ff=Counter(brands)

        all_brands=ff.items()

        #define the optimization function

        prob=LpProblem("SO",LpMaximize)

        #define decision variables

        dec_var=LpVariable.matrix("dec_var",(range(len(lift)),range(len(lift[0]))),0,1,LpBinary)


        #Compute the sum product of decision variables and lifts

        prodt_matrix=[dec_var[i][j]*lift[i][j] for i in range(len(lift))

        for j in range(len(lift[0]))]


        prob+=lpSum(prodt_matrix)

        order=list(unique_everseen(brands))
        order_map = {}
        for pos, item in enumerate(order):

            order_map[item] = pos


        #brands in order as in input file

        brands_lift=sorted(all_brands,key=lambda x: order_map[x[0]])
        
        st.header('Enter the shelf constraints')
        
        a,b,c = st.columns(3)

        with a:
           min1 = st.number_input("Maximum no of products in Shelf 1", min_value=0, max_value=5)
        with b:
            min2 = st.number_input('Maximum no of products in Shelf 2', min_value=0, max_value=5)
        with c:
            min3 = st.number_input('Maximum no of products in Shelf 3', min_value=0, max_value=5)
            
        c,d = st.columns(2)

        with c:
            min4 = st.number_input('Maximum no of products in Shelf 4', min_value=0, max_value=5)
        with d:
            min5 = st.number_input('Maximum no of products in Shelf 5', min_value=0, max_value=5)
            
        
        

        #DEFINE CONSTRAINTS

        #1) Each shelf can have only one product i.e. sum (each row)<=input constraints


        row_con=[min1,min2,min3,min4,min5]

        for i in range(len(lift)):

            prob+=lpSum(dec_var[i])<=row_con[i]
            
            
        #2) Each product can be displayed only on a limited number of shelves i.e. Column constraints

        #Constraints are given as

        dec_var=np.array(dec_var)

        col_data=[]

        for j in range(len(brands)):

            col_data.append(list(zip(*dec_var))[j])

            prob+=lpSum(col_data[j])<=1
            
            
        prob.writeLP("SO.lp")

        prob.solve()

        if(st.button('Submit')):
            
            

            progress = st.progress(0)
            for i in range(100):
                sleep(0.1)
                progress.progress(i+1)
            st.success('Shelf Optimization Completed')
            st.subheader('Data Provided')
            st.table(sales)
            Matrix=[[0 for X in range(len(lift[0]))] for y in range(len(lift))]

            for v in prob.variables():
            
                Matrix[int(v.name.split("_")[2])][int(v.name.split("_")[3])]=v.varValue
            
                matrix=np.int_(Matrix)
            
            w = list(sales.iloc[0,1:])
            dfs = pd.DataFrame(data = matrix,columns=w,index=['Shelf1','Shelf2','Shelf3','Shelf4','Shelf5'])
            st.subheader('Planned Shelf')
            st.dataframe(data=dfs)
            
            val_Matrix=[[0 for X in range(len(lift[0])) ] for y in range(len(lift))]
            for x in range(len(lift)):
                for y in range(len(lift[0])):
                    if(matrix[x][y]==1):
                        val_Matrix[x][y]=lift[x][y]
                        
            dfv = pd.DataFrame(data = val_Matrix,columns=w,index=['Shelf1','Shelf2','Shelf3','Shelf4','Shelf5'])
            st.subheader('Estimated Sales as per plan')
            st.dataframe(data=dfv)
            st.write('Maximum Sales obtained:',value(prob.objective))
            
if rad=='Market Basket':
    st.title('Market Basket Analysis')
    st.image('market.jpg',width=800)
    f=st.file_uploader('Upload file')

    if f is not None:
        df2=pd.read_csv(f)
        #df1=df[df['STORE CODE']==32468][['TELEPHONE NUMBER','CATEGORY','CLASS','SUB CLASS','NET SOLD QTY','TRANS DATE']].reset_index(drop=True)
        #df1=df1.dropna()
        #df1.insert(1,'Item',df1['CATEGORY']+'_'+df1['CLASS']+'_'+df1['SUB CLASS'])
        #df2=df1[['TELEPHONE NUMBER','Item','NET SOLD QTY','TRANS DATE']]
        df2.dropna(axis=0, subset=['TELEPHONE NUMBER'], inplace=True)
        
        mybasket = (df2
              .groupby(['TELEPHONE NUMBER', 'Item'])['NET SOLD QTY']
              .sum().unstack().reset_index().fillna(0)
              .set_index('TELEPHONE NUMBER'))
        
        def my_encode_units(x):
            if x <= 0:
                return 0
            if x >= 1:
                return 1

        my_basket_sets = mybasket.applymap(my_encode_units)
        
        my_frequent_itemsets = apriori(my_basket_sets, min_support=0.04, use_colnames=True)
        
        my_rules = association_rules(my_frequent_itemsets, metric="lift", min_threshold=1)
        
        s=my_rules.sort_values("confidence",ascending=False).reset_index(drop=True)
        
        
        st.write(list(s['antecedents'][0])[0],'--------->',list(s['consequents'][0])[0])
        st.write(list(s['antecedents'][1])[0],'--------->',list(s['consequents'][1])[0])
        st.write(list(s['antecedents'][2])[0],'--------->',list(s['consequents'][2])[0])
        st.write(list(s['antecedents'][3])[0],'--------->',list(s['consequents'][3])[0])
       