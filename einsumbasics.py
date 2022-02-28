import numpy as np
from numpy import einsum

w = np.arange(6).reshape(2,3).astype(np.float32)
print(w)
x = np.ones((3,1),dtype=np.float32)
print(x)

y = einsum('ik,kj->ij',w,x)
print(y)

w = np.arange(6).reshape(2,3).astype(np.float32)
print(w)
x = np.ones((1,3),dtype=np.float32)
print(x)

y = einsum('ik,jk->ij',w,x)
print(y)

m = np.arange(9).reshape(3,3).astype(np.float32)
print(m)

diag = einsum('ii->i',m)
print(diag)

trace = einsum('ii->',m)
print(trace)

colsum = einsum('ij->j',m)
print(colsum)

rowsum = einsum('ij->i',m).reshape(3,1)
print(rowsum)

trans = einsum('ij->ji',m)
print(trans)

a = np.ones((3,),dtype=np.float32)
b = np.ones((3,),dtype=np.float32) *2
print(a)
print(b)

d = einsum('i,i->',a,b)
print(d)
i = einsum('i,i->',a,b)
print(i)
o = einsum('i,j->ij',a,b)
print(o)

ma = np.arange(9).reshape(3,3)
print(ma)
ve =np.ones(3)
print(ve)

#Vector matrix multiplication
#x'A
sum1 = einsum('i,ij->j',ve,ma)
print(sum1)

print(np.matmul(ve,ma))

#Ax
sum2 = einsum('ij,j->i',ma,ve)
print(sum2)
 
print(np.matmul(ma,np.transpose(ve)))