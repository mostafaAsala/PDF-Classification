this folder you should put training data in it.

the configurations are as follows

you have two csv files:

links.csv: headers:-	PDF_URL,PROJECT,PDF_ID,classificstion,VENDOR_CODE
	this file contains the initial links PDF_URL , and the classificstion for each link those are the most important features
debug_data_out.csv: heaers:	PDF_URL,PROJECT,PDF_ID,classificstion,VENDOR_CODE,text
 	this file is the second step if you have an already text extracted files you can put it in this name and run training
you have three folders:
test: contains test.csv file containing test data
train: contains train.csv file containing train data
text_files: containing all Previously extracted text name {PDF_ID}.txt

