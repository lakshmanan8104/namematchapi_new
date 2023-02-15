#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def encrypt(basename_one,basename_two):
    import rsa
    
    with open('publicKey.pem', 'rb') as p:
        publicKey = rsa.PublicKey.load_pkcs1(p.read())
    encMessage_a = rsa.encrypt(basename_one.encode('utf-8'),
                         publicKey)
    
    encMessage_b = rsa.encrypt(basename_two.encode('utf-8'),
                         publicKey)
    encMessage_a1=encMessage_a.hex()
    encMessage_b1=encMessage_b.hex()
    return(encMessage_a1,encMessage_b1)

