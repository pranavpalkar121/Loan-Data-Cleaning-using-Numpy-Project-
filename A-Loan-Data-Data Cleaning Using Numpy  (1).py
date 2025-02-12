#!/usr/bin/env python
# coding: utf-8

# ## Importing the Packages

# In[2]:


import numpy as np


# In[3]:


np.set_printoptions(suppress = True, linewidth = 100, precision = 2)


# ## Importing the Data

# In[5]:


raw_data_np =np.genfromtxt(r"E:\Python\PRJ\Loan Data\loan-data.csv", delimiter=';', skip_header=1, autostrip=True)
raw_data_np


# In[6]:


raw_data_np.dtype.name


# ## Checking for Incomplete Data

# In[8]:


np.isnan(raw_data_np).sum()


# In[9]:


temp_fill= np.nanmax(raw_data_np) + 1
temp_mean= np.nanmean(raw_data_np, axis=0)


# In[10]:


np.nanmean(raw_data_np, axis=0)


# In[11]:


temp_mean


# In[12]:


temp_stats = np.array([np.nanmin(raw_data_np, axis=0), 
                       temp_mean,
                      np.nanmax(raw_data_np, axis=0)]) 
temp_stats


# In[13]:


temp_stats


# In[ ]:





# ## Splitting the Dataset

# ### Splitting the Columns

# In[16]:


column_strings = np.argwhere(np.isnan(temp_mean)).squeeze()
column_strings


# In[17]:


column_numeric = np.argwhere(np.isnan(temp_mean)== False).squeeze()
column_numeric


# ### Re-importing the Dataset

# In[19]:


loan_data_strings =np.genfromtxt(r"E:\Python\PRJ\Loan Data\loan-data.csv", 
                           delimiter=';', 
                           skip_header=1, 
                           autostrip=True,
                           usecols=column_strings,
                           dtype= 'str')
loan_data_strings


# In[20]:


loan_data_numeric =np.genfromtxt(r"E:\Python\PRJ\Loan Data\loan-data.csv", 
                           delimiter=';', 
                           skip_header=1, 
                           autostrip=True,
                           usecols=column_numeric,
                           filling_values=temp_fill)
loan_data_numeric


# ### The Names of the Columns

# In[22]:


header_full =np.genfromtxt(r"E:\Python\PRJ\Loan Data\loan-data.csv", 
                           delimiter=';', 
                           skip_footer= raw_data_np.shape[0],  ## ignoring all rows 
                           autostrip=True,
                           dtype= 'str')
header_full


# In[23]:


header_string= header_full[column_strings]
header_string


# In[24]:


header_numeric= header_full[column_numeric]
header_numeric


# ## Creating Checkpoints:

# In[26]:


def checkpoint(file_name, checkpoint_header, checkpoint_data):
    np.savez(file_name, header = checkpoint_header, data = checkpoint_data)
    checkpoint_variable = np.load(file_name + ".npz")
    return(checkpoint_variable)


# In[27]:


checkpoint_test = checkpoint(r'E:\Python\PRJ\Loan Data\test', header_string, loan_data_strings)


# ## Manipulating String Columns

# In[29]:


header_string


# In[30]:


header_string[0]= 'issue_date'


# In[31]:


header_string


# In[32]:


loan_data_strings


# ### Issue Date

# In[34]:


loan_data_strings[:,0]  ## taking all rows with 0 th column(Issue Date )


# In[35]:


np.unique(loan_data_strings[:,0] )


# In[36]:


## removing -15
loan_data_strings[:,0]= np.chararray.strip(loan_data_strings[:,0], '-15')


# In[37]:


months = np.array(['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])


# In[38]:


for i in range(13):
    loan_data_strings[:,0]= np.where(loan_data_strings[:,0]== months[i],
                                    i,
                                    loan_data_strings[:,0])


# In[39]:


np.unique(loan_data_strings[:,0] )


# ### Loan Status

# In[41]:


header_string


# In[42]:


np.unique(loan_data_strings[:,1])


# In[43]:


np.unique(loan_data_strings[:,1]).size


# In[44]:


status_bad= np.array(['','Charged Off','Default','Late (31-120 days)'])


# In[45]:


loan_data_strings[:,1]= np.where(np.isin(loan_data_strings[:,1], status_bad),0,1)


# In[46]:


np.unique(loan_data_strings[:,1])  ## we have made output 0 and 1 


# ### Term

# In[48]:


header_string


# In[49]:


np.unique(loan_data_strings[:,2])


# In[50]:


loan_data_strings[:,2]=np.chararray.strip(loan_data_strings[:,2],' months')
np.unique(loan_data_strings[:,2])


# In[51]:


header_string[2]= "term_months"


# In[52]:


loan_data_strings[:,2] = np.where(loan_data_strings[:,2]== '', '60', loan_data_strings[:,2])


# In[53]:


np.unique(loan_data_strings[:,2])


# ### Grade and Subgrade

# In[55]:


header_string


# In[56]:


np.unique(loan_data_strings[:,3])


# In[57]:


np.unique(loan_data_strings[:,4])


# #### Filling Sub Grade

# In[59]:


for i in np.unique(loan_data_strings[:,3])[1:]:
    loan_data_strings[:,4] = np.where((loan_data_strings[:,4] == '') & (loan_data_strings[:,3] == i),
                                      i + '5',
                                      loan_data_strings[:,4])


# In[60]:


np.unique(loan_data_strings[:,4], return_counts=True)  ## checking counts for each value, As you can see empty value is repeated 9  times


# In[61]:


## assiging H1 as null values 
loan_data_strings[:,4] = np.where((loan_data_strings[:,4] == ''),
                                      'H1',
                                      loan_data_strings[:,4])


# In[62]:


np.unique(loan_data_strings[:,4])


# #### Removing Grade

# In[64]:


## now we dont need grade as we have subgrade 
loan_data_strings= np.delete(loan_data_strings,3, axis = 1)


# In[65]:


np.unique(loan_data_strings[:,3])


# In[66]:


header_string = np.delete(header_string, 3)


# In[67]:


header_string


# #### Converting Sub Grade

# In[69]:


np.unique(loan_data_strings[:,3])


# In[70]:


keys = list(np.unique(loan_data_strings[:,3]))                         
values= list(range(1,np.unique(loan_data_strings[:,3]).shape[0] + 1))
dict_sub_grade = dict(zip(keys, values))


# In[71]:


dict_sub_grade


# In[72]:


for i in np.unique(loan_data_strings[:,3]):
        loan_data_strings[:,3] = np.where(loan_data_strings[:,3] == i, 
                                          dict_sub_grade[i],
                                          loan_data_strings[:,3])


# In[73]:


dict_sub_grade.items()


# ### Verification Status

# In[75]:


header_string


# In[76]:


np.unique(loan_data_strings[:,4])


# In[77]:


## we will treat '' and 'Not Verified' as bad values and will give 0 
loan_data_strings[:,4]= np.where((loan_data_strings[:,4]== '') | ( loan_data_strings[:,4] == 'Not Verified'),
                                 0,1)   ## if it is having values ''and not verified make it 0 or else 1 


# In[78]:


np.unique(loan_data_strings[:,4])


# ### URL

# In[80]:


loan_data_strings[:,5]


# In[81]:


np.chararray.strip(loan_data_strings[:,5], 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=')


# In[82]:


loan_data_strings[:,5]=np.chararray.strip(loan_data_strings[:,5], 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=')


# In[83]:


loan_data_strings[:,5]


# In[84]:


header_string


# In[85]:


## we have id column which we need to verify is it equal to this or not
header_full


# In[86]:


header_string


# In[87]:


loan_data_numeric[:,0].astype(dtype= 'int32')


# In[88]:


loan_data_strings[:,5].astype(dtype= 'int32')


# In[89]:


## checking np. array_equal


# In[90]:


header_string


# In[91]:


np.array_equal(loan_data_numeric[:,0].astype(dtype= 'int32'), loan_data_strings[:,5].astype(dtype= 'int32'))  ## its true


# In[92]:


header_string


# In[93]:


loan_data_strings= np.delete(loan_data_strings, 5, axis = 1)
header_string= np.delete(header_string, 5)


# In[94]:


header_full


# ### State Address

# In[96]:


header_string


# In[97]:


## changing name of column 
header_string[5]= 'State_address'


# In[98]:


loan_data_strings[:,5]


# In[99]:


np.unique(loan_data_strings[:,5]).size


# In[100]:


state_names , state_counts = np.unique(loan_data_strings[:,5], return_counts= True)
state_count_sorted= np.argsort(-state_counts)


# In[101]:


state_names[state_count_sorted], state_counts[state_count_sorted]


# In[102]:


loan_data_strings[:,5]= np.where(loan_data_strings[:,5]== '', 0 ,loan_data_strings[:,5])


# In[103]:


states_west = np.array(['WA', 'OR','CA','NV','ID','MT', 'WY','UT','CO', 'AZ','NM','HI','AK'])
states_south = np.array(['TX','OK','AR','LA','MS','AL','TN','KY','FL','GA','SC','NC','VA','WV','MD','DE','DC'])
states_midwest = np.array(['ND','SD','NE','KS','MN','IA','MO','WI','IL','IN','MI','OH'])
states_east = np.array(['PA','NY','NJ','CT','MA','VT','NH','ME','RI'])


# https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf

# In[105]:


loan_data_strings[:,5]


# In[106]:


loan_data_strings[:,5]= np.where(np.isin(loan_data_strings[:,5], states_west), 1, loan_data_strings[:,5])
loan_data_strings[:,5]= np.where(np.isin(loan_data_strings[:,5], states_south), 2, loan_data_strings[:,5])
loan_data_strings[:,5]= np.where(np.isin(loan_data_strings[:,5], states_midwest), 3, loan_data_strings[:,5])
loan_data_strings[:,5]= np.where(np.isin(loan_data_strings[:,5], states_east), 4, loan_data_strings[:,5])


# In[107]:


np.unique(loan_data_strings[:,5])


# ## Converting to Numbers

# In[109]:


loan_data_strings


# In[110]:


loan_data_strings=loan_data_strings.astype('int')


# In[111]:


loan_data_strings


# ### Checkpoint 1: Strings

# In[113]:


Checkpont_string=checkpoint(r'E:\Python\PRJ\Loan Data\Checkpont_string', header_string, loan_data_strings)


# In[114]:


Checkpont_string['header']


# In[115]:


Checkpont_string['data']


# ## Manipulating Numeric Columns

# In[117]:


loan_data_numeric


# In[118]:


np.isnan(loan_data_numeric).sum()


# ### Substitute "Filler" Values

# In[120]:


header_numeric


# #### ID

# In[122]:


temp_fill


# In[123]:


np.isin(loan_data_numeric[:,0], temp_fill).sum()


# #### Temporary Stats

# In[125]:


temp_stats[:,(0,  2,  4,  6,  7, 13)]  ## rather that putting like this, put it in a column with a numeric value 


# In[126]:


temp_stats[:,column_numeric]


# In[127]:


column_numeric


# #### Funded Amount

# In[129]:


header_numeric


# In[130]:


temp_stats


# In[131]:


loan_data_numeric[:,2]


# In[132]:


loan_data_numeric[:,2] = np.where(loan_data_numeric[:,2] == temp_fill, 
                                  temp_stats[0, column_numeric[2]],
                                  loan_data_numeric[:,2])
loan_data_numeric[:,2]


# #### Loaned Amount, Interest Rate, Total Payment, Installment

# In[134]:


header_numeric


# In[135]:


for i in [1,3,4,5]:
    loan_data_numeric[:,i] = np.where(loan_data_numeric[:,i] == temp_fill, 
                                  temp_stats[2, column_numeric[i]],
                                  loan_data_numeric[:,i])
    


# In[136]:


loan_data_numeric


# ### Currency Change

# #### The Exchange Rate

# In[139]:


EUR_USD= np.genfromtxt(r"E:\Python\PRJ\Loan Data\EUR-USD.csv", delimiter=',', autostrip=True, dtype = 'str')
EUR_USD


# In[140]:


EUR_USD= np.genfromtxt(r"E:\Python\PRJ\Loan Data\EUR-USD.csv", 
                       delimiter=',', 
                       autostrip=True,
                       skip_header=1,
                      usecols=3)
EUR_USD


# In[141]:


loan_data_strings[:,0]    ## it shows value of issues month 


# In[142]:


exchange_rate = loan_data_strings[:,0]
for i in range(1,13):
    exchange_rate = np.where(exchange_rate ==i , 
                            EUR_USD[i-1],## index start at 0 so we have go one step back
                            exchange_rate)   
exchange_rate = np.where(exchange_rate ==0  , 
                          np.mean(EUR_USD),## replacing with mean at the place of null
                          exchange_rate)
exchange_rate


# In[143]:


exchange_rate.shape


# In[144]:


loan_data_numeric.shape


# In[145]:


exchange_rate = np.reshape(exchange_rate, (10000, 1))


# In[146]:


loan_data_numeric= np.hstack((loan_data_numeric, exchange_rate))


# In[147]:


loan_data_numeric


# In[148]:


header_numeric= np.concatenate((header_numeric, np.array(['exchange_rate'])))
header_numeric


# In[149]:


exchange_rate.shape


# In[150]:


header_numeric.shape


# #### From USD to EUR

# In[152]:


header_numeric


# In[153]:


column_dollars = np.array([1,2,4,5])


# In[154]:


loan_data_numeric[:,6]


# In[155]:


for i in column_dollars:
    loan_data_numeric = np.hstack((loan_data_numeric, 
                                   np.reshape(loan_data_numeric[:,i] / loan_data_numeric[:,6], (10000,1))))


# In[156]:


loan_data_numeric  ## no you can see we have added calculated exchange rate at the end\


# In[157]:


loan_data_numeric.shape


# #### Expanding the header

# In[159]:


header_additional = np.array([column_name + '_EUR' for column_name in header_numeric[column_dollars]])


# In[160]:


header_numeric= np.concatenate((header_numeric, header_additional))


# In[161]:


header_numeric


# In[162]:


header_numeric[column_dollars] = np.array([column_name + '_USD' for column_name in header_numeric[column_dollars]])


# In[163]:


header_numeric  ## here we have added correct name of all columns


# In[164]:


columns_index_order = [0,1,7,2,8,3,4,9,5,10,6]


# In[165]:


header_numeric= header_numeric[columns_index_order]


# In[166]:


loan_data_numeric= loan_data_numeric[:,columns_index_order]


# In[167]:


loan_data_numeric  ## chnaged order as per convenience 


# ### Interest Rate

# In[169]:


header_numeric


# In[170]:


loan_data_numeric[:,5]


# In[171]:


loan_data_numeric[:,5]= loan_data_numeric[:,5]/100


# In[172]:


loan_data_numeric[:,5]


# ### Checkpoint 2: Numeric

# In[174]:


Checkpoint_numeric = checkpoint(r'E:\Python\PRJ\Loan Data\Checkpoint_numeric', header_numeric, loan_data_numeric)


# In[180]:


Checkpoint_numeric['header'], Checkpoint_numeric['data']


# ## Creating the "Complete" Dataset

# In[183]:


Checkpont_string['data'].shape


# In[185]:


Checkpoint_numeric['data'].shape


# In[187]:


np.hstack((Checkpoint_numeric['data'],Checkpont_string['data'])).shape


# In[189]:


loan_data= np.hstack((Checkpoint_numeric['data'],Checkpont_string['data']))


# In[191]:


loan_data


# In[193]:


np.isnan(loan_data).sum()


# In[195]:


header_full= np.concatenate((Checkpoint_numeric['header'],Checkpont_string['header']))
header_full


# ## Sorting the New Dataset

# In[198]:


np.argsort(loan_data[:,0])


# In[200]:


## now we have to sort it as per first column so we have to use argsort, 
## as we need to change other rows as per first column sorting 
loan_data= loan_data[np.argsort(loan_data[:,0])]


# In[202]:


loan_data


# In[206]:


## checking ,  are we getting proper sorting series 
np.argsort(loan_data[:,0])  ## see the series is of 0,1,2,3,4,5 etc which means its sorted correctly \|


# ## Storing the New Dataset

# In[210]:


loan_data= np.vstack((header_full, loan_data))


# In[212]:


np.savetxt(r'E:\Python\PRJ\Loan Data\Cleaned data\loan_data_preprocessed.csv', loan_data,
           fmt="%s",
          delimiter=',')

