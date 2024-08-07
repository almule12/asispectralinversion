from bs4 import BeautifulSoup
import requests
import numpy as np
import os
from os.path import exists
import wget
#import h5py

# Turns 24 hour string time into float of seconds past midnight
def seconds_since_midnight(time):
    return 60*60*float(time[:2]) + 60*float(time[2:4]) + float(time[4:])

# Shifts a string of time 'hhmmss' by some number of minutes
# Can use fractional minutes (not intended for that) but should
# Make a whole number of seconds, then
def shift_time(timestr,shift_min):
    
    hr = float(timestr[:2])
    mn = float(timestr[2:4])
    sc = float(timestr[4:])
    
    t = 60*60*hr + 60*mn + sc
    t += 60*shift_min
    
    hr = np.floor( t/(60*60) )
    mn = np.floor( (t - 60*60*hr)/60 )
    sc = np.floor( t - 60*60*hr - 60*mn )
    
    def num2str(num):
        if num>=10:
            strout = str(int(num))
        else:
            strout = '0'+str(int(num))
        return strout
    return num2str(hr)+num2str(mn)+num2str(sc)

# Given a date, start and end time, finds links to every DASC frame
def genlinks(date,starttime,endtime):
    # Seconds from midnight of start time
    startsecs = seconds_since_midnight(starttime)
    # Seconds from midnight of end time
    endsecs = seconds_since_midnight(endtime)
    # Hour and year in string form
    hr = starttime[:2]
    year = date[:4]

    # Construct base url
    url = 'http://optics.gi.alaska.edu/amisr_archive/PKR/DASC/PNG/'+year+'/'+date+'/'+hr
    
    # Pull and process file list
    soup = BeautifulSoup(requests.get(url).text,'html.parser')
    # First 5 links are not actual events
    rawlinks = soup.find_all('a')[5:]
    # Turning into a numpy array
    links = np.asarray(rawlinks).flatten()
    if len(links)==0:
        print('no imagery found')
        raise Exception('no links')

    # Extracting the necessary information
    # Each row is: 
    # [seconds since midnight, color, time]

    # Initializing array with zeros
    newlinks = np.zeros([len(links),3])
    for i in range(len(links)):
        # Time and color from filename
        time,color = links[i].split('.')[0].split('_')[2:]
        # Seconds past midnight
        newlinks[i,0] = seconds_since_midnight(time)
        newlinks[i,1] = color
        newlinks[i,2] = time

    # Finding time of each frame, separated by color
    bluesecs = newlinks[:,0][np.where(newlinks[:,1]==428)]
    greensecs = newlinks[:,0][np.where(newlinks[:,1]==558)]
    redsecs = newlinks[:,0][np.where(newlinks[:,1]==630)]
    
    # We arbitrarily choose our first frame to be blue so that the frames are taken in 'b,g,r' order:
    try:
        s0 = bluesecs[np.where(bluesecs<startsecs)[0][-1]]
    except:
        s0 = bluesecs[0]
    # Therefore our last frame must be red
    try:
        s1 = redsecs[np.where(redsecs>endsecs)[0][0]]
    except:
        s1 = redsecs[-1]
        
    # We pull out the indices of the first and last frames we want to download
    startind = np.where(newlinks[:,0]>=s0)[0][0]
    endind = np.where(newlinks[:,0]<=s1)[0][-1]

    linkstrim = list(links[startind:endind+1])
    linksout = [url+'/'+link for link in linkstrim]
    return linksout,linkstrim

# Pulls all DASC png imagery between <starttime> and <endtime> (UT) on the date of 
# <date>. An example input would be download_imagery('20230314','0730','0745')
def download_imagery(date,starttime,endtime):
    # Start hour and end hour are the same
    if starttime[:2]==endtime[:2]:
        links,fnames = genlinks(date,starttime,endtime)
    # We do two pulls
    else:
        #print('crossing hour')
        # End of the first hour
        endtime0 = starttime[:2]+'5959'
        print(starttime)
        print(endtime0)

        links0,fnames0 = genlinks(date,starttime,endtime0)
        # Start of the second hour
        starttime1 = endtime[:2]+'0000'
        links1,fnames1 = genlinks(date,starttime1,endtime)
        
        print(starttime1)
        print(endtime)
        # Concatenate
        links = links0+links1
        fnames = fnames0+fnames1
    try:
        os.mkdir(date)
    except:
        pass
    
    for i in range(len(links)):
        if exists(date+'/'+fnames[i]):
            #print('file exists')
            continue
        else:
            wget.download(links[i],out=date)
