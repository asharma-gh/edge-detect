from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def edge_detect_grad(imm, sigma=1):
    # Gradient edge detection
    xgrads = []
    ygrads = []
    grad_map = np.zeros(imm.shape)
    for ii in reversed(range(0, imm.shape[0]-1, sigma)):
        for jj in (range(0, imm.shape[1]-1, sigma)):
            # starting from bottom left
            I_ij = imm[ii][jj] 
            I_ij1 = imm[ii][jj+1]
            I_i1j = imm[ii+1][jj]
            I_i1j1 = imm[ii+1][jj+1]
        
            # Compute xgrads
            xgrad = ((1/(2*sigma)) * ((I_i1j1 - I_ij1) + (I_i1j - I_ij)))
            ygrad = ((1/(2*sigma)) * ((I_i1j1 - I_i1j) + (I_ij1 - I_ij)))
            # Compute magnitude of gradients
            grad_map[ii][jj] = np.sqrt(xgrad**2 + ygrad**2)
            # Store for plot
            xgrads.append(xgrad)
            ygrads.append(ygrad)

    return grad_map, xgrads, ygrads

# edge detection with discrete laplacian
def edge_detect_lap(imm, sigma=1):
    dlap = np.zeros(imm.shape)
    for ii in reversed(range(1, imm.shape[0]-2, sigma)):
        for jj in (range(1, imm.shape[1]-1, sigma)):
            # Top Row
            I_ij1 = imm[ii][jj+1]
            # Mid row
            I_1ij = imm[ii-1][jj]
            I_ij = imm[ii][jj] 
            I_i1j = imm[ii+1][jj]
            # Bottom Row
            I_ij1 = imm[ii][jj+1]

            dIdx2 = 1/(sigma**2) * (I_1ij -(2*I_ij) + I_i1j)
            dIdy2 = 1/(sigma**2) * (I_i1j -(2*I_ij) + I_ij1)

            dlap[ii][jj] = dIdx2 + dIdy2
    return dlap

def edge_detect_can(imm):
    pass


## Main
imm = Image.open('digital_images_week2_quizzes_lena.gif')
ar = np.array(imm, dtype=np.float64)
gmap, gx, gy = edge_detect_grad(ar,1)
dlap = edge_detect_lap(ar, 1)
# plot
plt.imshow(dlap, interpolation='nearest')
#plt.plot(gx, gy, "bo")
plt.axhline(color='red', lw=0.5)
plt.axvline(color='red', lw=0.5)

plt.show()
