import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import h5py
import scipy.interpolate
from matplotlib.pyplot import figure
import glob

# def load_lookup_tables( fname_red, fname_green, fname_blue, fname_sigp, fname_sigh, maglat, plot=True):
# _, Qvec, E0vec, greenmat = process_brightbin(fname_green,plot=plot)
# _, _, _, redmat = process_brightbin(fname_red,plot=plot)
# _, _, _, bluemat = process_brightbin(fname_blue,plot=plot)
# _, _, _, altvec, sigPmat = process_sig3dbin(fname_sigp)
# _, _, _, _, sigHmat = process_sig3dbin(fname_sigh)
# SigPmat = integrator(sigPmat, altvec, maglat)
# SigHmat = integrator(sigHmat, altvec, maglat)
# lookup_table = {
# 	'Qvec': Qvec,
# 	'E0vec': E0vec,
# 	'greenmat': greenmat,
# 	'redmat': redmat,
# 	'bluemat': bluemat,
# 	'altvec': altvec,
# 	'sigPmat': sigPmat,
# 	'sigHmat': sigHmat,
# 	'SigPmat': SigPmat,
# 	'SigHmat': SigHmat
# }
# return lookup_table

# Should we add in a max blue brightness? Yes I think so add it from notebook
# def calculate_E0_Q(redbright,greenbright,bluebright,lookup_tables,minE0=150):
# minE0_ind = np.where(lookup_table['E0vec']<minE0)[0][0]
# maxbluebright = ...
# q, maxq, minq = q_interp(lookup_table['bluemat'],lookup_table['Qvec'],lookup_table['E0vec'],bluebright,minE0ind=minE0ind,maxbluebright=maxbluebright,interp='linear',plot=False):
# e0_interp_general(testmat,Qvec,E0vec,testvec,qinvec,degen_bounds=None):
#........
# return Q,E0,Qmin,Qmax,E0min,E0max

# Process one of the GLOW brightness lookup tables
def process_brightbin(fname,plot=True):
    with open(fname) as f:
        # Open file
        recs = np.fromfile(f, dtype='float32')
        # Parameters
        params = recs[:20]
        # Dimensions of Q,E0
        nq = int(recs[20])
        ne = int(recs[21])
        # Q vector and E0 vector
        Qvec = recs[22:22+nq]
        E0vec = recs[22+nq:22+nq+ne]
        # Brightness matrix
        bright = recs[22+nq+ne:].reshape(ne,nq)
        # Pcolor as a sanity check
        if plot:
            plt.pcolormesh(Qvec,E0vec,bright,shading='auto')
            plt.xlabel('Q')
            plt.ylabel('E0')
            plt.title(fname.split('/')[-1][1:5])
            plt.colorbar()
            plt.show()
        # Return data
        return params,Qvec,E0vec,bright
    
# Process one of the GLOW conductance lookup tables
def process_sig3dbin(fname):
    with open(fname) as f:
        # Open file
        recs = np.fromfile(f, dtype='float32')
        # Parameters
        params = recs[:20]
        # Dimensions of Q,E0,alt
        nq = int(recs[20])
        ne = int(recs[21])
        nalt = int(recs[22])
        # Q vector, E0 vector, alt vector
        Qvec = recs[23:23+nq]
        E0vec = recs[23+nq:23+nq+ne]
        altvec = recs[23+nq+ne:23+nq+ne+nalt]
        # Brightness data cube
        sig3d = recs[23+nq+ne+nalt:].reshape(nalt,ne,nq)
        # Return data
        return params,Qvec,E0vec,altvec,sig3d
    
    
# Uses a GLOW lookup table to estimate Q from blue line brightness
# We also specify a minimum E0 index to crop the lookup table to. 

# If the lookup table is not cropped, the inversion becomes nonlinear at very
# small values of E0 and worsens results.

# Note that the uncertainty values returned are valid only under the assumption that
# true E0 is greater than or equal to the chosen cutoff
def q_interp(bright428,Qvec,E0vec,bluevec,minE0ind=0,maxbluebright=np.inf,interp='linear',plot=False):
    # Initialize vector of Q values
    qvec = []
    # Initialize vector keeping track of uncertainty due to the lack of knowledge of E0 
    #uncertvec = []
    maxqvec = []
    minqvec = []

    # Pcolor the whole lookup table
    if plot:
        plt.pcolormesh(Qvec,E0vec,bright428,shading='auto')
        plt.xlabel('Q')
        plt.ylabel('E0')

    # Iterate through blue brightness data points
    for blue in bluevec:
        # Check to see if the pixel is too bright
        if blue>maxbluebright:
            qvec.append(np.nan)
            maxqvec.append(np.nan)
            minqvec.append(np.nan)
            continue
        # Initialize vector storing all possible values Q could take for the given blue brightness
        qcross = []
        # Initialize vector keeping track of the corresponding E0s for those Q values
        e0cross = []
        # Iterate through all E0s and find the Qs that correspond to the blue brightness data point
        for e0i in range(minE0ind,len(E0vec)):
            try:
                if interp=='nearest':
                    qcross.append(Qvec[np.where(np.diff(np.sign(bright428[e0i,:]-blue)))[0][0]])
                    e0cross.append(E0vec[e0i])
                # We interpolate between Q values
                elif interp=='linear':
                    # The index immediately before the interpolation range 
                    guessind = np.where(np.diff(np.sign(bright428[e0i,:]-blue)))[0][0]
                    # The forward difference slope of brightness vs Q for this index
                    m = (bright428[e0i,guessind+1] - bright428[e0i,guessind])/(Qvec[guessind+1]-Qvec[guessind])
                    # The interpolated value of Q
                    qi = (blue-bright428[e0i,guessind])/m + Qvec[guessind]
                    qcross.append(qi)
                    e0cross.append(E0vec[e0i])
            except:
                # Interpolation failed, for some reason.
                # Probably the blue brightness was too high or too low.
                pass
            
        qcross = np.asarray(qcross)

        # If at least one valid solution was found
        if len(qcross)>0:
            # We take the median Q value found
            qvec.append(np.median(qcross))
            # We keep track of how much error we may have accrued by taking that median value
            #uncertvec.append(np.std(qcross))
            maxqvec.append(np.amax(qcross))
            minqvec.append(np.amin(qcross))
            
            # Plot the curves of possible Q solutions for each data point.
            # For large-dimensional input, this gets busy/useless/slow pretty fast!
            if plot:
                plt.plot(qcross,e0cross)
                plt.scatter(np.median(qcross),e0cross[np.argmin(np.abs(qcross-np.median(qcross)))],color='black',s=50)
        # If no good solution was found
        else:
            qvec.append(np.nan)
            maxqvec.append(np.nan)
            minqvec.append(np.nan)

    # Plot the minimum value of E0 considered for the inversion
    if plot:
        plt.plot([Qvec[0],Qvec[-1]],[E0vec[minE0ind],E0vec[minE0ind]],color='black',linewidth=5)
        plt.title('Solution sets for a given blue line brightness ')

    return np.asarray(qvec),np.asarray(maxqvec),np.asarray(minqvec)



def e0_interp_general(testmat,Qvec,E0vec,testvec,qinvec,degen_bounds=None):
    e0out = []
    
    
    # indvec = np.asarray([np.argmin(np.abs(Qvec-qin)) for qin in qinvec])
    
    # Closest Q index that undershoots the actual Q
    indvec = []
    for qin in qinvec:
        try:
            indvec.append(np.where(Qvec<qin)[0][-1])
        except:
            indvec.append(np.nan)
            
    for i in range(len(testvec)):
        if np.isnan(indvec[i]):
            e0out.append(np.nan)
        else:
            #curve = testmat[:,indvec[i]]-testvec[i]

            # Linear interpolation in Q
            fracind = (qinvec[i]-Qvec[indvec[i]])/(Qvec[indvec[i]+1]-Qvec[indvec[i]])
            curve0 = (1-fracind)*testmat[:,indvec[i]] + (fracind)*testmat[:,indvec[i]+1]
            curve = np.copy(curve0) - testvec[i]
            
            try:
                crossings = np.where(np.diff(np.sign(curve)))[0]
                guessind = crossings[0]
                
                # Hopefully there is only one nontrivial solution for E0
                if len(crossings)>1:
                    # Multiple nontrivial crossings. We (maybe) cannot trust this result
                    if (len(crossings)>2) or (np.diff(crossings)[0] != 1):
                        # print('bad, mult cross')
                        # print(E0vec[crossings])
                        
                        if degen_bounds == None:
                            #print('degenerate output')
                            # This crashes the evaluation of e0i
                            guessind = np.nan
                        else:
                            # print('trying to break degeneracy')
                            # We attempt to resolve the degeneracy
                            crossings_mask = (E0vec[crossings]>degen_bounds[-1]) | (E0vec[crossings]<degen_bounds[0])
                            #print(E0vec[crossings])
                            if len(crossings[~crossings_mask]) == 1:
                                #print('degeneracy broken!')
                                guessind = crossings[~crossings_mask][0]
                            else:
                                print('failed to break degeneracy')
                                print(E0vec[crossings])
                                # This crashes the evaluation of e0i
                                guessind = np.nan
                        
                # Linear interpolation in E0
                m = (curve[guessind+1] - curve[guessind])/(E0vec[guessind+1]-E0vec[guessind])
                e0i = -curve[guessind]/m + E0vec[guessind]

            except:
                e0i = np.nan
                
            # if np.isnan(e0i):
            #     print('nan')
            e0out.append(e0i)
            # try:
            #     e0i = E0vec[np.where(np.diff(np.sign(curve)))[0][0]]
            #     #print(np.diff(np.sign(curve)))
            #     #print(np.where(np.diff(np.sign(curve)))[0])
            # except:
            #     e0i = np.nan
            # e0out.append(e0i)
    return np.asarray(e0out)

def e0_interp_general_nearest(testmat,Qvec,E0vec,testvec,qinvec):
    e0out = []
    
    # Nearest neighbor interpolation
    indvec = np.asarray([np.argmin(np.abs(Qvec-qin)) for qin in qinvec])
    
    for i in range(len(testvec)):
        if np.isnan(indvec[i]):
            e0out.append(np.nan)
        else:
            # Nearest neighbor interpolation
            curve = testmat[:,indvec[i]]-testvec[i]
            try:
                e0i = E0vec[np.where(np.diff(np.sign(curve)))[0][0]]
                #print(np.diff(np.sign(curve)))
                #print(np.where(np.diff(np.sign(curve)))[0])
            except:
                e0i = np.nan
            e0out.append(e0i)
    return np.asarray(e0out)
