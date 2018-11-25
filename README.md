# asr

Create a new directory 'asr' which consists of all the folders - Models,moodles,results,TIMIT.

### The directory path is "/home/<USERNAME>/asr"

Please change the directory paths (just the initial parts with the path shown above) in the following
files present anywhere in the directory:
1.test_wavs
2.train_wavs
3.model_path

Use the directory TIMIT for test and train data.
Used SOX converter to convert the given .NIST to .WAV format.
The directory contains .wav files.

The file timit.hdf is large and hence, it was added just once. Please copy the files in other directories 
mentioned below.

Add the timit.hdf files in the following directories:
1. Models/with_energycoeff/test_with_energy_coeff
2. Models/with_energycoeff/train_with_energy_coeff
3. Models/without_energycoeff/test_without_energy_coeff
4. Models/without_energycoeff/test_without_energy_coeff

