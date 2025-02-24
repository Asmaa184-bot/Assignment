#!/usr/bin/env python
# coding: utf-8

# In[1]:


x1, x2 = 0.1, 0.5  

w11, w12 = 0.2, -0.3  
w21, w22 = 0.4, 0.1  
w_out1, w_out2 = -0.5, 0.3  

b1, b2 = 0.5, 0.7  

def tanh(x):
    e = 2.718281828459045
    return (e**x - e**-x) / (e**x + e**-x)  

out_h1 = tanh(w11 * x1 + w12 * x2 + b1)  
out_h2 = tanh(w21 * x1 + w22 * x2 + b1)  

out_o = tanh(w_out1 * out_h1 + w_out2 * out_h2 + b2)  

print(out_o)


# In[ ]:




