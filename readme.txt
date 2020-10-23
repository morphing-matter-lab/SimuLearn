Date: Oct. 20. 2020
written by Humphrey Yang (hanliny@cs.cmu.edu)
This file is subjected to changes.

1. Ownership

    This code is uploaded and maintained by the Morphing Matter Lab at Carnegie
    Mellon University. Please cite this work using this DOI: 
    https://dl.acm.org/doi/10.1145/3379337.3415867

2. Dependencies:

    2a. System requirements
    the python codes would run on both PC and Mac as long as the dependencies 
    are installed, yet the design tool can only be ran on Windows because it 
    requires Human UI, which currently only supports Windows. It is 
    receommended to use a copmuter with a GPU installed but the files run on 
    CPU, too. Note that empirically, you will need at least 2 GB of available 
    RAM (system RAM or VRAM) on you device to use the design tool, and at least
     8 GB for training the model.

    2b. For data processing and machine learning:
    Pytorch version 1.6.0*
    Numpy version 1.17.0

    2c. For the design tool
    Rhinoceros 3D version 6SR29 with grasshopper
    Human UI version 0.8.1.3 for grasshopper
    GH_CPython version 0.1-alpha for grasshopper

    *both GPU and CPU version of pytorch should work with this implementation, 
    but we have only tested the GPU version with NVIDIA graphics card running 
    CUDA 10.1. Other versions may be compatible as well but further tests are 
    needed.

3. Dataset download

    you can find the additonal data from this link:
    https://drive.google.com/file/d/1FsPuSQb4Q9J0SjqPoJf8WytITNVv0tQi/view?usp=sharing

    The things contained in this compressed files are:
    3a. A very small subset of our FEA results and input files for demonstrating 
    the dataprocessing pipeline.
    3b. A dataset that is extracted from 2,000 trials of FEA.
    3c. A trained machine learning model for use by the design tool.

4. Usage instructions:

    4a. parameters.py
    This file contains the parameters used for parsing FEA results, extracting 
    datasets, and training machine learning models. These parameters are used 
    as global variables across all files.

    4b. FEA result extraction
    Run dataperser.py to extract data from FEA results. The file does three 
    things - reading the input file from "./input_files" folder to understand 
    the feature sampling nodes, getting feature values from FEA nodal files 
    (already converted to numpy binary files using Abaqus2Matlab, saved to 
    "./node_files" folder), and rotating and mirroring the simulation trials 
    to create orientational variations. The extracted dataset are saved to the
    "./data" folder. The code runs in two stages, the first stage extracts 
    feature from individual files and saves them to the "./extracted" folder, 
    the second stage compiles all extracted files into a single dataset and 
    saves them to the "./data" folder. This file can be ran in parallel by 
    setting the parameters in parameters.py. Note that in order to try out 
    this step, you will need to download additional files from the download 
    link metioned in 3., in which we have provided a few FEA input and nodal 
    files to demonstrate the implementaion. 

    4c. SimuLearn machine learning model training
    Run main.py to train the machine learning model - the simulator. Note that 
    you would have to train stage 1 and 2 simulators separately. The 
    parameters can be modified in parameters.py, and this implementation runs on 
    both GPU and CPU. The script loads the dataset from the "./data" folder and 
    the trained models are saved to the "./models" folder.

    4d. Using the design tool
    Run Rhinoceros 3D and grasshopper to lauch the script "design tool.gh". 
    Note that in order to do this, you would need to download additional files 
    (the trained simulator) from the download link in 3. In particular, the 
    design tool will load the trained model from the "./selected_models" folder
    to simulate user designs. When launching the design tool, a user interface
    will pop up and promt the user the locate the folder contraining the 
    design tool. Once located, the design tool window will change and the 
    user will be free to make and simulate their own design. Simulation 
    results will be written into the "./simulation" folder and then read by 
    the design tool to visualize the results. In addition, the "./images" 
    folder contains some visuals used by the design tool.