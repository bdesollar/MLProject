# **ML Project**

## Overview
---

I spearheaded this project along with one other fellow students in the creation of this ML Project The main objective of this
project was to test out different learners and see which one proved to have the best results. There are multiple types of data that 
were tested against and differnt values we were trying to predict. For example in Used_Car_Type_Reg we used data from a Kaggle competition 
we entered ( Where we got 1st :) ) to predict the "type" of vehicle. (More about the competition and our 1st place finish found 
[here](https://www.kaggle.com/competitions/used-cars-type-classification/overview)). In this project we used supervised learning to solve 
both classification and regression problems. Used multiple models including nerual networks, Linearregression, Lasso, RandomForest, amongst others.
We chose these ones based upon the results during testing and the amount of data that needed to be tested. Used jupyter notebooks to be able to
document and code all in one file to make it easier to read and write the code needed, as well as the results.



## How to install

---

1) To use this you will first need to download all files as a zip file and unpack it or pull it to github desktop
2) Then you will need to setup a virtual enviroment
   1) Run the following commands
      1) `pip install virtualenv`
      2) `cd project_folder`
      3) `virtualenv venv`
      4) `source venv/bin/activate`
3) After that you will need to run `pip install notebook` in the terminal
4) Following that, run `pip install -r requirements.txt` in the terminal
5) The files of data to use for training and testing can be found [here](https://www.kaggle.com/competitions/used-car-price-regression-aiml-2022/data) for used car reg (regresssion) and [here](https://www.kaggle.com/competitions/used-cars-type-classification/data) for used car clas (classification)
   1) Put them in teh directory you have Used_Car_Reg in as well as in Used_Car_Clas
6) Then run `jupyter notebook` to launch the jupiter notebook and be able to test out the learners

## How to use

---

To use this you will just need to run all cells in the Jupyter Notebook. Click on the project type you want to test and the run all cells in
the *.ipynb file. The last cell will print the results to a csv file and that were uploaded to kaggle for comparison.

## Credits

---

This project was only possible due to the contributions of my fellow groupmates:

- Matt Mcdonell - Respnsible for the doucumenation as well as parsing the data for Bank Churners
- Myself - The majority of the code, including the parsing, testing, and implementation of the models and their results


## License

---

MIT License

Copyright (c) 2022 Ben DeSollar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



