import numpy as np
import os
import time
import multiprocessing as mp

#FOLDER = "./node_files"
#from parameters import *

PARALLEL = True
WORKERS = 6

NODE_PATH = "Volumes/My Passport/CHI_2x2/stage1/node_files"
COOR_PATH = NODE_PATH + "/coor"
STRESS_PATH = NODE_PATH + "/stress"

def csvToPy(fileName):

    s = time.time()
        
    # read csv into numpy array
    oldName = FOLDER + '/' + fileName
    data = np.genfromtxt(oldName, delimiter=',')
    # change type
    data = data.astype(np.float32)
    # dump into numpy file
    newName = FOLDER + '/' + fileName.replace(".csv", '')
    np.save(newName, data)

    e = time.time()

    return "processed %s" % fileName, e - s

def convertFolder(FOLDER):

    files = os.listdir(FOLDER)
    valid = []
    for f in files:
        if (f.endswith(".csv")):
            valid += [f]
    files = valid

    finished = 0
    if(PARALLEL):
        with mp.Pool(processes=WORKERS) as pool:
            for message, elapsed in \
                pool.imap_unordered(csvToPy, files):

                finished += 1
                perc = (finished / len(files)) * 100
                progress = "%.2f" % perc + chr(37) # % character
                print(message + ", (%s)...%.3f" % (progress, elapsed))
    else:
        
        for fileName in files:
            message, elapsed = csvToPy(fileName)

            finished += 1
            perc = (finished / len(files)) * 100
            progress = "%.2f" % perc + chr(37) # % character
            print(message + ", (%s)...%.3f" % (progress, elapsed))

def compileData(args):

    files, folder, newName = args
    s = time.time()

    sortedFiles = sorted(files, 
        key = lambda name: \
            int(name[name.rfind('_') + 1:name.rfind('.')]))
    
    fullPaths = [folder + '/' + fileName for fileName in sortedFiles]
    
    stack = [np.genfromtxt(path, delimiter=',') for path in fullPaths]
    compiled = np.concatenate(stack, axis=0)

    np.save(newName, compiled)

    e = time.time()

    return "processed %s" % newName, e - s

def compileFolder(sourceFolder, tarFolder):
    
    trials = {}
    # list files into trials
    for fileName in os.listdir(sourceFolder):

        if(not fileName.endswith(".csv")): continue
        
        trialName = fileName[:fileName.rfind('_')]
        trials[trialName] = trials.get(trialName, []) + [fileName]
    
    files = [(trials[name], sourceFolder, tarFolder + '/' + name) for name in trials]

    finished = 0
    if(PARALLEL):
        with mp.Pool(processes=WORKERS) as pool:
            for message, elapsed in \
                pool.imap_unordered(compileData, files):

                finished += 1
                perc = (finished / len(files)) * 100
                progress = "%.2f" % perc + chr(37) # % character
                print(message + ", (%s)...%.3f" % (progress, elapsed))
    else:
        for fileName in files:
            message, elapsed = compileData(fileName)

            finished += 1
            perc = (finished / len(files)) * 100
            progress = "%.2f" % perc + chr(37) # % character
            print(message + ", (%s)...%.3f" % (progress, elapsed))

    pass

def main():

    # coor
    print("processing coor files:")
    print('=' * 80)
    compileFolder(COOR_PATH, NODE_PATH)
    # stress
    print("processing stress files:")
    print('=' * 80)
    compileFolder(STRESS_PATH, NODE_PATH)

    print("finished!")

if __name__ == "__main__":
    main()