import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate


# Height-integrate a 3d conductivity datacube to get a 2d conductance matrix
# Accounts for magnetic field angle from vertical to first order
def sig_integrator(sigmat,altvec,maglat):
    # Cumulative trapezoidal integration
    Sigmat = scipy.integrate.cumtrapz(sigmat,altvec/100,axis=0)[-1]
    # First order account for magnetic field angle from vertical
    Sigmat /= np.sin(maglat*np.pi/180)
    return Sigmat

# Given a set of filenames, reads in the GLOW lookup tables and packages them into a struct
def load_lookup_tables(fname_red, fname_green, fname_blue, fname_sigp, fname_sigh, maglat, plot=True):
    # Read in: run parameters,Q vector, E0 vector, green brightness matrix from bin file
    params, Qvec, E0vec, greenmat = process_brightbin(fname_green,plot=plot)
    # Read in red and blue brightness matrices from bin files
    _, _, _, redmat = process_brightbin(fname_red,plot=plot)
    _, _, _, bluemat = process_brightbin(fname_blue,plot=plot)
    # Read in altitude vector and Pedersen conductivity datacube from bin file
    _, _, _, altvec, sigPmat = process_sig3dbin(fname_sigp)
    # Read in Hall conductivity datacube from bin file
    _, _, _, _, sigHmat = process_sig3dbin(fname_sigh)
    # Height-integrate conductivity datacubes to get conductance. First order correction for magnetic field angle.
    SigPmat = sig_integrator(sigPmat, altvec, maglat)
    SigHmat = sig_integrator(sigHmat, altvec, maglat)
    # Put everything into a Python dict
    lookup_table = {
        'Params': params,
    	'Qvec': Qvec,
    	'E0vec': E0vec,
    	'greenmat': greenmat,
    	'redmat': redmat,
    	'bluemat': bluemat,
    	'altvec': altvec,
    	'sigPmat': sigPmat,
    	'sigHmat': sigHmat,
    	'SigPmat': SigPmat,
    	'SigHmat': SigHmat
    }
    return lookup_table

# Given RGB brightness arrays (calibrated, in Rayleighs) and a lookup table for the correct night, estimates E0 and Q
# Setting minE0 constrains uncertainty values in Q, since for some nights some strange stuff happens at the bottom of the lookup tables.
# We often assume that visual signatures are insignificant below 150 eV, but that parameter can be set lower or higher as desired
# The generous option sets Q,E0 to zero instead of NaN when inversion fails but certain conditions are met (very dim pixels)
def calculate_E0_Q(redbright,greenbright,bluebright,lookup_table,minE0=150,generous=False):
    # Save the initial shape of arrays. They will be flattened and later reshaped back to this
    shape = greenbright.shape

    # Reshape brightness arrays to vectors
    redvec = redbright.reshape(-1)
    greenvec = greenbright.reshape(-1)
    bluevec = bluebright.reshape(-1)

    # Cuts off the lookup table appropriately
    minE0ind = np.where(lookup_table['E0vec']>minE0)[0][0]
    
    # Estimates Q from blue brightness, along with error bars
    qvec, maxqvec, minqvec = q_interp(lookup_table['bluemat'],lookup_table['Qvec'],lookup_table['E0vec'],bluevec,minE0ind=minE0ind,maxbluebright='auto',interp='linear',plot=False)

    # Estimates E0 from red/green ratio and estimated Q value
    e0vec = e0_interp_general(lookup_table['redmat']/lookup_table['greenmat'],lookup_table['Qvec'],lookup_table['E0vec'],(redvec/greenvec),qvec)
    e0vecext1 = e0_interp_general(lookup_table['redmat']/lookup_table['greenmat'],lookup_table['Qvec'],lookup_table['E0vec'],(redvec/greenvec),maxqvec)
    e0vecext2 = e0_interp_general(lookup_table['redmat']/lookup_table['greenmat'],lookup_table['Qvec'],lookup_table['E0vec'],(redvec/greenvec),minqvec)

    mine0vec = np.minimum(e0vecext1,e0vecext2)
    maxe0vec = np.maximum(e0vecext1,e0vecext2)

    if generous:
        qvec[np.where(bluevec<np.amin(lookup_table['bluemat']))] = 0#lookup_table['E0vec'][0]
        e0vec[np.where((redvec == 0) | (greenvec == 0))] = 0#lookup_table['E0vec'][0]
        e0vec[np.where((redvec/greenvec) > np.amax(lookup_table['redmat']/lookup_table['greenmat']))] = 0#lookup_table['E0vec'][0]

    return qvec.reshape(shape),e0vec.reshape(shape),minqvec.reshape(shape),maxqvec.reshape(shape),mine0vec.reshape(shape),maxe0vec.reshape(shape)

# Given a processed lookup table dict from load_lookup_tables and arrays of  Q and E0, interpolates to calculate conductances
# The generous option tries to make sense of zeros in Q/E0 arrays by setting conductances to their minimum values
# Note that this function may throw an error when provided with a Q/E0 value that is nonzero but below the mininum entry in the table.
# This should be fixed soon! -Alex
def calculate_Sig(q,e0,lookup_table,generous=False):
    # Saves shape of input
    shape = q.shape
    
    # Reshapes inputs to vectors
    qvec = q.reshape(-1)
    e0vec = e0.reshape(-1)
    
    # Linearly interpolates conductances
    SigP_interp = scipy.interpolate.RegularGridInterpolator([lookup_table['E0vec'],lookup_table['Qvec']],lookup_table['SigPmat'])
    SigH_interp = scipy.interpolate.RegularGridInterpolator([lookup_table['E0vec'],lookup_table['Qvec']],lookup_table['SigHmat'])

    # Initializes conductance vectors
    SigPout = np.zeros_like(qvec)
    SigHout = np.zeros_like(qvec)

    # Removes nans or zeros from Q and E0 vecs that would cause the interpolator to throw an error
    mask = (np.isnan(qvec) | np.isnan(e0vec)) | ((qvec == 0) | (e0vec == 0))

    # Reshapes input to the format the interpolator wants
    invec = np.asarray([e0vec[np.where(~mask)],qvec[np.where(~mask)]]).T
    
    # Puts NaNs where the interpolator would have failed
    SigPout[np.where(mask)] = np.nan
    SigPout[np.where(~mask)] = SigP_interp(invec)

    SigHout[np.where(mask)] = np.nan
    SigHout[np.where(~mask)] = SigH_interp(invec)


    # Tries to make sense of zeros in Q/E0 vectots
    if generous:
        SigPout[np.where( (qvec == 0) | (e0vec == 0) )] = np.amin(SigPout)
        SigHout[np.where( (qvec == 0) | (e0vec == 0) )] = np.amin(SigHout)
    return SigPout.reshape(shape),SigHout.reshape(shape)



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

# Note that the uncertainty values returned are valid only under the assumption that
# true E0 is greater than or equal to the chosen cutoff

# maxbluebright should be kept at 'auto' unless there's a good reason to change it.
# The Q interpolation has a built-in routine to estimate the max blue brightness that works well
# Changing maxbluebright only affects the error bars on returned Q, not Q itself
def q_interp(bright428,Qvec,E0vec,bluevec,minE0ind=0,maxbluebright='auto',interp='linear',plot=False):
    # Automatically estimate where the inversion table "runs out of room" for very bright blue values
    # This involves a recursive evaluation...
    if maxbluebright == 'auto':
    	# Generate 50 blue brightnesses and invert to Q
    	# Note that this function recursively calls itself (once), when used with the 'auto' parameter for maxbluebright
    	# This lets it determine a reasonable bound for blue brightness, above which inversions may be inaccurate
    	testbluevec = np.linspace(0,np.amax(bright428),50)
    	_,testmaxqvec,_ =  q_interp(bright428,Qvec,E0vec,testbluevec,minE0ind=minE0ind,maxbluebright=np.inf,interp=interp,plot=False)
    	# Find where the upper Q bound hits a ceiling, and mark it as the maximum blue brightness where upper Q bound can accurately be determined
    	medval = np.median(np.diff(testmaxqvec[np.where(~np.isnan(testmaxqvec))]))
    	firstbadind = np.where((np.diff(testmaxqvec)<(medval/2)))[0][0]
    	maxbluebright = testbluevec[firstbadind]
    	if plot:
    	    plt.scatter(testbluevec,testmaxqvec,color='black')
    	    plt.scatter(testbluevec[firstbadind:],testmaxqvec[firstbadind:],color='red')
    	    plt.title('Max possible Q, ceiling hit in red')
    	    plt.xlabel('blue brightness')
    	    plt.show()
    
    # Initialize vector of Q values
    qvec = []
    # Initialize vectors keeping track of max/min Q values due to lack of knowledge of E0 
    maxqvec = []
    minqvec = []

    # Pcolor the whole lookup table
    if plot:
        plt.pcolormesh(Qvec,E0vec,bright428,shading='auto')
        plt.xlabel('Q')
        plt.ylabel('E0')

    # Iterate through blue brightness data points
    for blue in bluevec:
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
                # We linearly interpolate between Q values
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
            	# no Q consistent with this E0
                pass
        qcross = np.asarray(qcross)

        # If at least one valid solution was found
        if len(qcross)>0:
            # We take the Q value for very high E0
            qvec.append(qcross[-1])
            # We keep track of how much error we may have accrued by choosing that value
            if blue>maxbluebright:
            	maxqvec.append(np.nan)
            else:
            	maxqvec.append(np.amax(qcross))
            minqvec.append(np.amin(qcross))
            # Plot the curves of possible Q solutions for each data point.
            # For large-dimensional input, this gets busy/useless/slow pretty fast!
            if plot:
                plt.plot(qcross,e0cross)
                plt.scatter(qcross[-1],e0cross[np.argmin(np.abs(qcross-qcross[-1]))],color='black',s=50)
        # If no good solution was found
        else:
            # Extremely dim pixel
            if blue < np.amin(bright428):
            	qvec.append(0.)
            	maxqvec.append(Qvec[0])
            	minqvec.append(0.)
            else:
            	qvec.append(np.nan)
            	maxqvec.append(np.nan)
            	minqvec.append(np.nan)

    # Plot the minimum value of E0 considered for the inversion
    if plot:
        plt.plot([Qvec[0],Qvec[-1]],[E0vec[minE0ind],E0vec[minE0ind]],color='black',linewidth=5)
        plt.title('Solution sets for a given blue line brightness ')

    return np.asarray(qvec),np.asarray(maxqvec),np.asarray(minqvec)


# Interpolates E0 from a "general" lookup table, aka any function F(Qvec,E0vec).
# We highly recommend that this be used with the table red_brightness/green_brightness.

# If you specify degen_bounds = [min_E0, max_E0], they will be used in the event of a degeneracy
# (multiple valid E0 values found). In the event that the degeneracy is not resolved, E0 will
# be set to NaN.
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
                                print('degeneracy broken!')
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
                
            e0out.append(e0i)
            
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
