import numpy as np
import matplotlib.pyplot as plt

"""
Constructing Minotaur transition
"""
# 1. Minotaur can not stay still
idx = np.zeros((30,30))

# Four cornors
idx[[0+1,0+5],0] = 1
idx[[4-1,4+5],4] = 1
idx[[25-5,25+1],25] = 1
idx[[29-5,29-1],29] = 1

# Four edges
for i in 5*np.arange(4)+5:
    idx[[i+1,i-5,i+5],i] = 1

for i in np.arange(3)+1:
    idx[[i-1,i+5,i+1],i] = 1

for i in 5*(np.arange(4)+1)+4:
    idx[[i-5,i-1,i+5],i] = 1

for i in np.arange(3)+26:
    idx[[i-1,i+1,i-5],i] = 1

# Inner points
base = np.arange(3)+6
for i in np.arange(3)+1:
    base = np.append(base,np.arange(3)+6+i*5)

for i in base:
    idx[[i-1,i+1,i-5,i+5],i] = 1

P_m = np.zeros((30,30))
for i in np.arange(30):
    P_m[:,i] = idx[:,i]/np.sum(idx[:,i])

#print(P_m)

"""
Constructing player transition
"""
# 0. Not moving
P_p1 = np.eye(30)

# 1. Move left
P_p2 = np.zeros((30,30))
ind = np.arange(5)+5
ind = np.append(ind,np.arange(8)+13)
ind = np.append(ind,np.array([23]))
ind = np.append(ind,np.arange(5)+25)

for i in np.arange(30):
    if i in ind:
        P_p2[i-5,i] = 1
    else:
        P_p2[i,i] = 1

# 2. Move right
P_p3 = np.zeros((30,30))
ind = np.arange(5)
ind = np.append(ind,np.arange(8)+8)
ind = np.append(ind,np.array([18]))
ind = np.append(ind,np.arange(4)+20)

for i in np.arange(30):
    if i in ind:
        P_p3[i+5,i] = 1
    else:
        P_p3[i,i] = 1

# 3. Move up
P_p4 = np.zeros((30,30))
ind = np.arange(6)*5+1
ind = np.append(ind,np.arange(4)*5+2)
ind = np.append(ind,np.arange(6)*5+3)
ind = np.append(ind,np.array([4,29]))

for i in np.arange(30):
    if i in ind:
        P_p4[i-1,i] = 1
    else:
        P_p4[i,i] = 1


# 4. Move down
P_p5 = np.zeros((30,30))
ind = np.arange(6)*5
ind = np.append(ind,np.arange(4)*5+1)
ind = np.append(ind,np.arange(6)*5+2)
ind = np.append(ind,np.array([3,28]))

for i in np.arange(30):
    if i in ind:
        P_p5[i+1,i] = 1
    else:
        P_p5[i,i] = 1

"""
Transition of the whole matrix
"""

P = np.zeros((900,900,5))
P[:,:,0]=np.kron(P_p1,P_m)
P[:,:,1]=np.kron(P_p2,P_m)
P[:,:,2]=np.kron(P_p3,P_m)
P[:,:,3]=np.kron(P_p4,P_m)
P[:,:,4]=np.kron(P_p5,P_m)

# Rewrite the death state
for i in np.arange(30):
    if i != 24:
        for j in np.arange(5):
            P[:,30*i+i,j] = np.zeros((900))
            P[30*i+i,30*i+i,j]=1

Psum = np.zeros((900,5))
for i in np.arange(5):
    Psum[:,i] = P[:,:,i].sum(axis=0)

# print(Psum == np.ones((900,5)))
np.savetxt('Psum_value.txt',Psum[:,0])
"""
Reward matrix
"""
R = np.zeros((900,5))
R[np.arange(29)+29*30,1] = 1

"""
Backward induction
"""

a = np.zeros((900,15))
u = np.zeros((900,16))
umax = np.zeros((15,1))

for i in np.arange(14,-1,-1):
    for j in np.arange(900):
        r_compare = np.zeros((5,1))
        for k in np.arange(5):
            r_compare[k,0] = R[j,k] + P[:,j,k].dot(u[:,i+1])
        a[j,i] = r_compare.argmax()
        u[j,i] = r_compare.max()
    umax[i,0] = u[:,i].max()
print(u[:,0].max())
print(umax)

np.savetxt('u_value.txt',u[:,0])

"""
Draw plots
"""
plt.pcolor(np.reshape(u[:,0].T,(30,30)))
plt.show()

plt.pcolor(np.reshape(a[:,0].T,(30,30)))
plt.show()
