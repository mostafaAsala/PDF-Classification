import io
import requests
import pandas as pd
import traceback
import pdfplumber
from sklearn.calibration import LabelEncoder
import constants
import fitz
import traceback
import os
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import joblib
import datetime
import pytesseract
from pdf2image import convert_from_bytes, convert_from_path
from sklearn.model_selection import GridSearchCV
#from IronPdf import PdfDocument
import xgboost as xgb
from xgboost.callback import TrainingCallback
from sklearn.neural_network import MLPClassifier
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from transformers import BertTokenizer, BertModel
from sklearn.base import BaseEstimator, TransformerMixin
import torch
from imblearn.under_sampling import RandomUnderSampler
import constants  # Assuming this module is imported and contains required constants
from datetime import datetime







import logging
"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"
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a callback for logging during training
class PrintLossCallback(TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        # This method is called after each iteration
        for eval_name, metrics in evals_log.items():
            for metric_name, log in metrics.items():
                print(f"Epoch {epoch + 1} - {eval_name} {metric_name}: {log[-1]}")
        return False  # Return True to stop training early, False to continue
print_loss_callback = PrintLossCallback()

pytesseract.pytesseract.tesseract_cmd = constants.tesseract_path
# Define a function to convert sparse matrix to text
def identity_func(x):
    return x

current_hour = datetime.now()
debug=True
def debug(func):
    if debug==True:
        func()



def is_Searchable(pdf_path,top_margin=constants.top_margin,bottom_margin=constants.bottom_margin,thresh = 30):
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        total_pages = 5 if total_pages>5 else total_pages
        searchable_pages = 0
        index=0
        for page in pdf.pages:
            if index > total_pages:
                break
            index+=1
            # Extract text from the main content area (excluding header and footer)
            page_text = ""
            for obj in page.chars:
                if obj['top'] > top_margin and obj['bottom'] < page.height - bottom_margin*page.height:  # Adjust these values as needed
                    page_text += obj['text']

            # Check if the extracted text contains any searchable content
            if len(page_text)>300:
                searchable_pages += 1
    percentage_searchable = (searchable_pages / total_pages) * 100
    return percentage_searchable > thresh




def is_valid_pdf_link(url):
    # Check if the URL starts with "http://download.siliconexpert.com/pdfs" and ends with ".pdf"
    return (url.startswith("http://download.siliconexpert.com/pdfs") or url.startswith("https://download.siliconexpert.com/pdfs") ) and url.endswith(".pdf")






def download_pdf(url, save_path):
    if not is_valid_pdf_link(url):
        return "Invalid PDF link."
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Save the content to a file
            with open(save_path, 'wb') as pdf_file:
                pdf_file.write(response.content)
            #print(f"PDF downloaded successfully to '{save_path}'.")
            return True
             
        else:
            print( f"Failed to download PDF. Status code: {response.status_code}")
            return False

    except Exception as e:
       print( f"Error downloading PDF: {e}")
       return False


def extract_text_from_pdf2(pdf_path,txt_path,top_margin=constants.top_margin,bottom_margin=constants.bottom_margin):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract text from each page
            text = ""
            pgn=0
            
            for page in pdf.pages:
                page_text=""
                for obj in page.extract_text_lines(): 
                    if obj['top'] > top_margin and obj['bottom'] < page.height - bottom_margin*page.height:  # Adjust these values as needed
                         page_text += obj['text'] +'\n'

                pgn+=1
                if pgn>3:break
                text += page_text+" "
        if len(text)>300:
            with open(txt_path ,'w',encoding="utf-8") as file:
                file.write(text)
            # Append the new record to the dictionary
            return True
        else:
            return False
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        return False

def extract_text_from_pdf_Production(pdf_path,top_margin=constants.top_margin,bottom_margin=constants.bottom_margin):
    try:
        text=""
        response = requests.get(pdf_path)
        if response.status_code == 200:
            stream = response.content

            with fitz.open(stream=BytesIO(stream)) as doc:
                doclen= 3 if len(doc)>3 else len(doc)
                for page_num in range(doclen):
                    page = doc.load_page(page_num)
                    page_height = page.rect.height  # Height of the page
                    y_min = constants.top_margin * page_height  # Calculate 0.1 of the page height
                    y_max = page_height -constants.bottom_margin * page_height  # Calculate 0.9 of the page height
                    page_text = ""
                    words = page.get_text("words")
                    filtered_words = []  # List to store words within the specified range
                    for w in words:
                        y1 = w[3]  # Vertical coordinate of the bottom of the word
                        if y_min <= y1 <= y_max:  # Check if the word's vertical coordinate is within the range
                            filtered_words.append(w[4])  # Append the word to the list of filtered words
                    page_text += " ".join(filtered_words)  # Concatenate the filtered words
                    text+= page_text + ' '
            if len(text) < 300:
                print("empty pdf_ trying OCR to extract")
                data = process_pdf(response)
                if data['status']==0:
                    return ""
                else:
                    return data['text']
            else:
                return text
            
    except Exception as e:
        print(traceback.format_exc())
        return ""
def extract_text_from_pdf(pdf_path, txt_path, top_margin=constants.top_margin, bottom_margin=constants.bottom_margin):
    try:
        text = ""
        
        
        with fitz.open(stream=BytesIO(pdf_path)) as doc:
            doclen= 3 if len(doc)>3 else len(doc)
            for page_num in range(doclen):
                page = doc.load_page(page_num)
                page_height = page.rect.height  # Height of the page
                y_min = constants.top_margin * page_height  # Calculate 0.1 of the page height
                y_max = page_height -constants.bottom_margin * page_height  # Calculate 0.9 of the page height
                page_text = ""
                words = page.get_text("words")
                filtered_words = []  # List to store words within the specified range
                for w in words:
                    y1 = w[3]  # Vertical coordinate of the bottom of the word
                    if y_min <= y1 <= y_max:  # Check if the word's vertical coordinate is within the range
                        filtered_words.append(w[4])  # Append the word to the list of filtered words
                page_text += " ".join(filtered_words)  # Concatenate the filtered words
                text+= page_text + ' '
        if len(text) > 300:
            with open(txt_path, "w",encoding="utf-8") as file:
                file.write(text)
            return True
        else:
            return text
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        return None


# Function to extract text from image
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)


def process_pdf(pdf_bytes, max_pages=3):
    bytes_ = io.BytesIO(pdf_bytes.content)

    try:
        # Convert PDF to images
        images = convert_from_bytes(bytes_.getvalue(),poppler_path=constants.poopler_Path)
        # Perform OCR on each image, up to max_pages 
        text = ''
        for i, image in enumerate(images):
            if i >= max_pages:
                break
            text += pytesseract.image_to_string(image)
        if len(text)<300:
            status=0
        else:
            status = 1  # Success
    except Exception as e:
        text = ""
        status = 0  # Error
        print("error", traceback.format_exc())
    return {"text": text, "status": status}



def count_of_images(link):
    doc = fitz.open(link)
    num_images = 0
    for page_index in range(len(doc)):
        image_list = doc[page_index].get_images()
 
        # printing number of images found in this page
        if image_list:
            num_images += len(image_list)

def extract_images(pdf_path, output_folder):
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images()
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                width = base_image["width"] 
                height = base_image["height"] 
                
                #print((width,height), (page.rect.width, page.rect.height))






def data_extract2(data,filename="data.csv",start=0,finish=float('inf')):
    new_df={ 'PDF_URL':[],
             'Text':[],
             'datasheet':[]
            }
 
    #new_df=pd.DataFrame(columns=['PDF_URL', 'Text','datasheet'])
    # Iterate through each row in the dataframe
    for index, row in data.iterrows():
        if index<start:
            continue
        if index>finish:
            break
        url = row['PDF_URL']
        if index%10==0:
            print(index)
        if index%100==0 and index>start:
            pd.DataFrame(new_df).to_csv(filename,sep=',')
        
       
        try:
           # Send a GET request to the URL
            response = requests.get(url)
            # Check if the request was successful
            if response.status_code == 200:
                # Open the PDF file
                with open("temp_pdf.pdf", 'wb') as f:
                    f.write(response.content)
                # Read the PDF file using pdfplumber
                with pdfplumber.open("temp_pdf.pdf") as pdf:
                    # Extract text from each page
                    text = ""
                    pgn=0
                    for page in pdf.pages:
                        pgn+=1
                        if pgn>3:
                            pgn=0
                            break
                        text += page.extract_text()
                text=  text.replace(',',' ')
                text=  text.replace('\n',' ')
                text=  text.replace('\t',' ')
                if len(text)>300:
                    new_row= {'PDF_URL': url, 'Text': text, 'datasheet':row['datasheet']}
                    # Append the new record to the dictionary
                    for key, value in new_row.items():
                        new_df[key].append(value)
                # Append the URL and extracted text to the new dataframe
                #new_df = new_df.append({'PDF_URL': url, 'Text': text, 'datasheet':row['datasheet']}, ignore_index=True)
                #new_df = pd.concat([new_df, new_row], ignore_index=True)
            else:
                print(f"Failed to retrieve URL: {url}")

        except Exception as e:
            print(f"Error processing URL: {url}, Error: {e}")
            print(traceback.format_exc())
    
    return pd.DataFrame(new_df)
# Display the new dataframe

def process_links(df:pd.DataFrame):
    for item in df.items:
        print(item)


def get_links(file_path = constants.not_datasheet_link_file):

    df = pd.read_csv(file_path,encoding='utf-8',low_memory=False)
    df = df[['PDF_ID','PDF_URL','FILE_SIZE','PAGES','VENDOR_CODE','PROJECT']]
    boolean_function = df['PDF_URL'].apply(lambda x: not(is_valid_pdf_link(x)))
    x = df[boolean_function].index
    print(x)
    boolean_function.to_csv("Test\\beforezero2.csv",encoding="utf-8")
    df = df.drop(df[boolean_function].index)
    
    df.to_csv("Test\\beforezero.csv",encoding="utf-8")

    return df

def get_all_files(pdf_path=constants.not_datasheet_link_file, datashet_path=constants.datasheet_link_file,use_Exist=True):
    if os.path.exists(constants.link_file_final) and use_Exist:
        print("Get files From Info")
        df = pd.read_csv(constants.link_file_final,encoding='utf-8',low_memory=False,dtype={"PDF_URL": str})
        print(df.info())
    else:
        df1 = get_links(pdf_path)
        
        df1.to_csv("Test\\first.csv",encoding='utf-8')
        #df1 = df1[df1['PROJECT']!='Datasheet']
        df1.to_csv("Test\\inittial.csv",encoding='utf-8')
        print(len(df1))
        df2 = get_links(datashet_path)
        df2.to_csv("Test\\ds.csv")
        df1.to_csv("Test\\second.csv",encoding='utf-8')
        mask = df1['PDF_ID'].astype(str).isin(df2['PDF_ID'].astype(str).tolist())
        
        mask.to_csv("Test\\mask.csv",encoding='utf-8')
        df1 = df1[~mask]
        
        df1.to_csv("Test\\third.csv",encoding='utf-8')
        
        print(len(df2))
        df1['datasheet'] = 0
        df2['datasheet'] = 1
        df = pd.concat([df1,df2])
        print(len(df))
        df.head()
        df = df.dropna(subset=['PDF_ID'])
        print(len(df))
        
        df = df.drop_duplicates(subset=['PDF_ID'])
        df = df.reset_index(drop=True)
        print(len(df))
        
        df = df.sample(frac=1).reset_index(drop=True)
        df.to_csv(constants.link_file_final,encoding='utf-8')
    return df

def data_extract(filename="final_data.csv",file_Path = "C:\\Users\\Public\\Documents\\WorkPrograms\\ManualExtractionAuto\\Handlers\\Machine Learning\\",start=0, finish=1000000):
    
    previously_downloaded_pdf = os.listdir( constants.pdf_files_path)
    previously_extracted_text =  os.listdir(constants.result_text_path)
    data = get_all_files(pdf_path=constants.not_datasheet_link_file,datashet_path=constants.datasheet_link_file)
    data['result_path'] = None
    data['pdf_path']=None
    new_df = pd.read_csv(constants.link_file_final,encoding='utf-8',low_memory=False)
    new_df = new_df.dropna()
    #new_df=pd.DataFrame(columns=['PDF_URL', 'Text','datasheet'])
    # Iterate through each row in the dataframe
    for index, row in data.iterrows():
        if index<start:
            continue
        if index>finish:
            break
        url = row['PDF_URL']
        if index%10==0:
            print(index)
        if index%10==0 and index>start:
            new_df.to_csv(constants.link_file_final,encoding='utf-8')
        
        if row['PDF_ID'] in new_df['PDF_ID']:
            print("duplicate url")
            continue
        
        
        try:
            found=True
            result_path = constants.pdf_files_path+str(row['PDF_ID'])+'.pdf'
            if str(row['PDF_ID'])+'.pdf'  not in previously_downloaded_pdf:
                if (download_pdf(row['PDF_URL'],result_path)):
                    row['pdf_path'] = result_path
                    data.at[index, 'pdf_path'] = result_path
                else:
                    found=False
            else:
                row['pdf_path'] = result_path
            extracted = True
            text_path = constants.result_text_path + str(row['PDF_ID'])+'.txt'
            if found == True and (str(row['PDF_ID'])+'.txt' not in previously_extracted_text):
                
                if(extract_text_from_pdf(row['pdf_path'],text_path)!= None):
                    row['result_path'] = text_path
                else:
                    extracted=False
            else:
                row['result_path'] = text_path

            if extracted==True:    
                if row['PDF_ID'] not in new_df['PDF_ID'] :
                    new_df.loc[len(new_df)] = row
                else:
                    new_df.loc[new_df['PDF_ID'] == row['PDF_ID']] = row
                
        except Exception as e:
            print(f"Error processing URL: {url}, Error: {e}")
            print(traceback.format_exc())

    
    return pd.DataFrame(new_df)


def data_extract3(dataLabel_List=[],exclude_list=['Delete_HTML','In_Complete_DS'],start=0, finish=10000000):
    previously_extracted_text =  os.listdir(constants.result_text_path)
    error_data ={}
    data = get_all_files(pdf_path=constants.not_datasheet_link_file,datashet_path=constants.datasheet_link_file)
    data = data[data['PROJECT']!= 'In_Complete_DS' ]
    
    for index, row in data.iterrows():
        if index<start  or str(row['PROJECT']) in (exclude_list):
            print("excluded...")
            continue
        if index>finish:
            break
        url = row['PDF_URL']
        if not is_valid_pdf_link(url):
            continue
        if index%10==0:
            print(index)
        if index%100==0:
            pd.DataFrame(error_data).to_csv("data\\ErrorExtraction.csv",encoding='utf-8',sep=',')
        try:
            
            text_path = constants.result_text_path + str(row['PDF_ID'])+'.txt'
            try:
                 previously_extracted_text.index(str(row['PDF_ID'])+'.txt')>-1
                 found = True
                 print("old data")
            except:
                found=False

            if found==False:
                print("new extraction")
                response = requests.get(row['PDF_URL'])
                if response.status_code == 200:
                    stream = response.content

                    
                    if(extract_text_from_pdf(stream,text_path)!=None):
                        previously_extracted_text+=text_path
                    else:
                        print("error Extracting: "+row['PDF_URL'])
                        error_data+= {'PDF_ID':row['PDF_ID'], 'PDF_URL':row['PDF_URL'] }

                else: 
                    print("error Downloading: "+row['PDF_URL'])
        except Exception as e:
            print(f"Error processing URL: {url}, Error: {e}")
            print(traceback.format_exc())

    
    return
def analyse_extracted(data_links:pd.DataFrame):
    previously_extracted_text =  os.listdir(constants.result_text_path)
    previously_extracted_text = [int(i.split(sep='.')[0]) for i in previously_extracted_text]
    data_links['PDF_ID'] = data_links['PDF_ID'].astype(int)
    #previously_extracted_set = set(previously_extracted_text)

    print(previously_extracted_text)
    print(data_links['PDF_ID'])
    #mask=data_links[data_links['PDF_ID'].apply(lambda x: any(proj in x for proj in previously_extracted_text))]
    mask = data_links['PDF_ID'].isin(previously_extracted_text)
    print(mask)
    extracted_data = data_links[mask]
    count_per_type_extracted = extracted_data['PROJECT'].value_counts()
    count_per_type_orig = data_links['PROJECT'].value_counts()

    return extracted_data,count_per_type_extracted,count_per_type_orig
#Delete_HTML

# Function to read text files based on file paths
def read_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except:
        print("error reading file \""+ file_path+"\"")
        print(traceback.format_exc())
        return ""




class TrainingModels:
    def __init__(self):
        self.train_tfidf_vectorizer = TfidfVectorizer(token_pattern=r'\b(?!\d{1,})(?=\w{2,})\w+\b', stop_words='english', lowercase=True, max_features=600)
        
        self.classifiers = [
            #('LinearSVC', LinearSVC()),
            ('XGB',xgb.XGBClassifier(verbosity=1, callbacks=[print_loss_callback]))
            #('MultinomialNB', MultinomialNB()),
            ('RandomForest', RandomForestClassifier(random_state=42)),
            #('MLP', MLPClassifier(random_state=42))  # Adding MLPClassifier
            #('RandomForest_search', GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1))
        ]
        #self.model_ready = self.load_models()
        self.data = pd.DataFrame()
        self.tfidf_Params = []
        self.Pipelines = None
        self.selected_Labels = constants.selected_labels
        self.PreLoadText=False
        self.set_preload_text_= False
        print("init classifier")
    def set_preload_text(self,value=True):
        self.set_preload_text_=value
    def label_other(self,x,labels =constants.selected_labels):
            
            if str(x) in labels:
                return x
            else:
                return 'Others'
    def Load_data(self):
        self.data = get_all_files()
        print(self.data['PROJECT'])
        print("Data prepared")
        previously_extracted_text = os.listdir(constants.result_text_path)
        previously_extracted_text = [str(i.split(sep='.')[0]) for i in previously_extracted_text]
        pd.DataFrame(previously_extracted_text).to_csv("data\\Log\\previously_extracted_text.csv")
        
        self.data['PDF_ID'] = self.data['PDF_ID'].astype(str)
        self.data['PDF_ID']=self.data['PDF_ID'].apply(lambda x: str(x).replace('.0',''))
        self.data.to_csv("data\\Log\\justImported.csv")
        print(self.data.head()['PDF_ID'])
        mask = self.data['PDF_ID'].isin(previously_extracted_text)
        self.data = self.data[mask]
        self.data.to_csv("data\\Log\\afterjustImported.csv")
        
        # Apply a function to label elements not in the list as 'other'
        
        self.data['classification'] = self.data.apply(lambda x: self.label_other(x['PROJECT']), axis=1)
        print(self.data['classification'])
        print("Data Loaded")

    def load_Text(self):
        self.data.to_csv("data\\Log\\Before_debug"+str(current_hour.day)+"_"+str(current_hour.hour)+".csv")
        
        self.data['text'] = self.data['PDF_ID'].astype(str).apply(lambda x: read_text(constants.result_text_path +str(x) + '.txt'))
        
        self.data = self.data[self.data['text']!=""]
    
    
    def analyze(self):
        self.data[self.data['datasheet']==1]['PROJECT'] ='Datasheet' 
        info = self.data.info()
        valus_count = self.data['PROJECT'].value_counts()
        valus_count.to_csv("Log\\total_data"+str(current_hour.hour)+".csv")
        info.to_csv("Log\\total_data_info"+str(current_hour.hour)+".csv")

        pass
#fast API
    def train_multi_level(self,Classes_sample=['Datasheet','Package Brief','Manual and Guide'],selected_classes =['Datasheet','Package Brief','Manual and Guide'],Model_name="Datasheet"):
            
        if self.set_preload_text_:
            print("gotData...")
            if os.path.exists("data\\Log\\debug_data_out.csv"):
                self.data = pd.read_csv("data\\Log\\debug_data_out.csv",encoding="utf-8",low_memory=False)
                self.data['classification'] = self.data.apply(lambda x: self.label_other(x['classification']), axis=1)
            else:
                self.data.to_csv("data\\Log\\debug_data_out.csv",encoding="utf-8")
        else:
            print("loading Text....")
            self.load_Text()
            self.data.to_csv("data\\Log\\debug_data_out.csv",encoding="utf-8")

        mask=self.data['classification'].isin(constants.not_selected_Labels)
        self.data = self.data[~mask]

        self.data=self.data
        self.data = self.data[self.data['PROJECT']!='In_Complete_DS']
        traindata = self.data.dropna(subset=['text'])
        traindata = traindata[traindata['text']!='']
        traindata = traindata.drop_duplicates(subset=['PDF_ID'])

        #hirarchey classes configurations
        traindata = traindata[traindata['classification'].isin(Classes_sample)]
        traindata['classification'] = traindata.apply(lambda x: self.label_other(x['PROJECT_1'],selected_classes), axis=1)
        
        X = traindata['text']
        
        y = traindata['classification']
        traindata['classification'].value_counts().to_csv("data\\Log\\train"+str(current_hour.day)+'_'+str(current_hour.month)+'_'+str(current_hour.month)+"_"+str(current_hour.hour)+".csv")
        
        self.data['PROJECT'].value_counts().to_csv("data\\Log\\project"+str(current_hour.day)+'_'+str(current_hour.month)+"_"+str(current_hour.hour)+".csv")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # Balance the training data using RandomUnderSampler
        undersampler = RandomUnderSampler(random_state=42)
        
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train.values.reshape(-1, 1), y_train)
        X_train_resampled = pd.DataFrame(X_train_resampled,columns=['Text'])['Text'].astype(str)
        #y_train_resampled = pd.DataFrame(y_train_resampled,columns=['Text'])['Text']
        
        y_train_resampled.value_counts().to_csv("data\\Log\\train_resampled"+str(current_hour.day)+'_'+str(current_hour.month)+"_"+str(current_hour.hour)+".csv")
        #y_train_resampled = np.array(y_train_resampled)
        #print(type(X_train_resampled), type(y_train_resampled))
        
        #print("type : ddddddddddddddddddddddddddddddddddd     ",y_train_resampled.shape,"  ",X_train_resampled.shape)
        print('training the model')
        print("training "+"_model")
        # Train and evaluate each classifier
        for name, clf in self.classifiers:
            print("training "+name+"_model")
            # Create a pipeline for each classifier
            clf_pipeline = Pipeline([
                ('tfidf', self.train_tfidf_vectorizer),
                ('clf', clf)
            ])
            
            if name=='XGB':
                # Sample data
                data = self.selected_Labels

                # Initialize the LabelEncoder
                label_encoder = LabelEncoder()

                # Fit and transform the data
                encoded_data = label_encoder.fit_transform(data)

                y_train_resampled = encoded_data
            # Train the model
            clf_pipeline.fit(X_train_resampled, y_train_resampled)
            
            
            # Evaluate the model
            y_pred = clf_pipeline.predict(X_test)
            print(f"Classification Report for {name}:")
            print(classification_report(y_test, y_pred))
            results_df = pd.DataFrame({
                'y_test': y_test,
                'y_pred': y_pred
            })

            # Save the DataFrame to a CSV file
            results_df.to_csv('classification_results.csv', index=False)

            # Save the trained model
            model_filename = f"LVL2_{Model_name}_{name}_model"+str(current_hour.day)+'_'+str(current_hour.month)+'_'+str(current_hour.hour)+".pkl"
            joblib.dump(clf_pipeline, model_filename)
            print(f"Model for {name} saved as {model_filename}")

        pass


    def train(self):
        
        if self.set_preload_text_:
            print("gotData...")
            if os.path.exists("data\\Log\\debug_data_out.csv"):
                self.data = pd.read_csv("data\\Log\\debug_data_out.csv",encoding="utf-8",low_memory=False)
                self.data['classification'] = self.data.apply(lambda x: self.label_other(x['classification']), axis=1)
            else:
                self.data.to_csv("data\\Log\\debug_data_out.csv",encoding="utf-8")
        else:
            print("loading Text....")
            self.load_Text()
            self.data.to_csv("data\\Log\\debug_data_out.csv",encoding="utf-8")

        mask=self.data['classification'].isin(constants.not_selected_Labels)
        self.data = self.data[~mask]

        self.data=self.data
        self.data = self.data[self.data['PROJECT']!='In_Complete_DS']
        traindata = self.data.dropna(subset=['text'])
        traindata = traindata[traindata['text']!='']
        traindata = traindata.drop_duplicates(subset=['PDF_ID'])
        X = traindata
        y = traindata['classification']
        traindata['classification'].value_counts().to_csv("data\\Log\\train"+str(current_hour.day)+'_'+str(current_hour.month)+'_'+str(current_hour.month)+"_"+str(current_hour.hour)+".csv")
        
        self.data['PROJECT'].value_counts().to_csv("data\\Log\\project"+str(current_hour.day)+'_'+str(current_hour.month)+"_"+str(current_hour.hour)+".csv")

        X_train_f, X_test_f, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42,stratify=y)
        X_train = X_train_f['text']
        X_test = X_test_f['text']
        
        # Balance the training data using RandomUnderSampler
        #undersampler = RandomOverSampler(random_state=42)
        X_train_resampled = X_train
        y_train_resampled =   y_train
        print(X_train)
        #X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train.values.reshape(-1, 1), y_train)
        #X_train_resampled = pd.DataFrame(X_train_resampled,columns=['Text'])['Text'].astype(str)
        #y_train_resampled = pd.DataFrame(y_train_resampled,columns=['Text'])['Text']
        
        y_train_resampled.value_counts().to_csv("data\\Log\\train_resampled"+str(current_hour.day)+'_'+str(current_hour.month)+"_"+str(current_hour.hour)+".csv")
        #y_train_resampled = np.array(y_train_resampled)
        #print(type(X_train_resampled), type(y_train_resampled))
        
        #print("type : ddddddddddddddddddddddddddddddddddd     ",y_train_resampled.shape,"  ",X_train_resampled.shape)
        print('training the model')
        print("training "+"_model")
        # Train and evaluate each classifier
        for name, clf in self.classifiers:
            print("training "+name+"_model")
            # Create a pipeline for each classifier
            clf_pipeline = Pipeline([
                ('tfidf', self.train_tfidf_vectorizer),
                ('clf', clf)
            ])
            
            if name=='XGB':
                # Sample data
                data = self.selected_Labels

                # Initialize the LabelEncoder
                label_encoder = LabelEncoder()

                # Fit and transform the data
                encoded_data = label_encoder.fit_transform(data)

                y_train_resampled = encoded_data
            # Train the model
            clf_pipeline.fit(X_train_resampled, y_train_resampled)
            
            
            # Evaluate the model
            y_pred = clf_pipeline.predict(X_test)
            print(f"Classification Report for {name}:")
            print(classification_report(y_test, y_pred))
            X_test_f['Pred'] = y_pred
            

            # Save the DataFrame to a CSV file
            X_test_f.to_csv('classification_results'+str(current_hour.day)+'_'+str(current_hour.month)+'_'+str(current_hour.hour)+'.csv', index=False)

            # Save the trained model
            model_filename = f"{name}_model"+str(current_hour.day)+'_'+str(current_hour.month)+'_'+str(current_hour.hour)+".pkl"
            joblib.dump(clf_pipeline, model_filename)
            print(f"Model for {name} saved as {model_filename}")

    def load_models(self,date="7_6_12"):
        print(len(self.classifiers))
        self.Pipelines={}
        for (name, clf) in self.classifiers:
            try:
                
                print(f'{name}_model'+date+'.pkl')
                clf = joblib.load(f'{name}_model'+date+'.pkl')
                self.Pipelines[name]=(name,clf)
            except Exception as e :
                print("exception: "+str(e))
                return 0
        return 1

    def preprocess(self, text):
        return self.train_tfidf_vectorizer.transform([text])
    def predict_pdf(self,url):
        
        pdf_link = url
        if is_valid_pdf_link(pdf_link):
            extracted_text = extract_text_from_pdf_Production(pdf_link)
            if extracted_text is not None and extracted_text!="":
                print("text extracted successfully")
                print(extracted_text)
                
                predictions = self.predict([extracted_text])
                print(predictions)
                return predictions
            else: 
                print("data can't be extracted, file must be searchable.")
                return("data can't be extracted, file must be searchable.")
        else:
            print("invalid url, url must contain siliconecpert domain and be pdf file")
            return "invalid url, url must contain siliconecpert domain and be pdf file"
    def predict_Files(self,df:pd.DataFrame):
            predictions = []
            extracted_df =[]
            urls=[]
            df['Extracted']=1
            for index,row in df.iterrows():
                url = row['URL']
                pdf_link = url
                
                if is_valid_pdf_link(pdf_link):
                    extracted_text = extract_text_from_pdf_Production(pdf_link)
                    if extracted_text is not None and extracted_text!="":
                        print("text extracted successfully", index)
                        
                        extracted_df.append(extracted_text)
                        urls.append(url)
                        # Assign the new value to the new column
                        df.at[index, 'Extracted'] = 1

                    else: 
                        print("data can't be extracted, file must be searchable.")
                else:
                    print("invalid url, url must contain siliconecpert domain and be pdf file")
            
            original_df = df[df['Extracted']==1]
            predictions= self.predict(extracted_df)
            predictions = pd.DataFrame(predictions)
            print(predictions)
            predictions = pd.concat([original_df,predictions],axis=1)
            #predictions['URL'] = urls

            #ret = [{'url':urls[i],'Prediction':data['']} for i,data in predictions.iterrows()]
            #result_df = pd.DataFrame(ret)
            #print(result_df)
            #df['Prediction'] = predictions
            return predictions
            
    def predict(self, text_list=[]):
        # Preprocess input text
        #X_processed = self.preprocess(text)
        
        # Make predictions
        predictions = {}
        #print(self.Pipelines)
        confidence_scores={}
        
        #print(text)
        for name in self.Pipelines:
            
            clf = self.Pipelines[name][1]
            print(clf)
            predictions[name] = clf.predict(text_list)
            predictions[name+"_conf"] = [max(i) for i in clf.predict_proba(text_list)]
        
        print (predictions) 
        return predictions 
    

    def expand_Models(self):
        
        self.load_models()
        name ,clf = self.Pipelines['RandomForest_search']
        clf:Pipeline
        grid_search = clf.named_steps['clf']

        
        # Use GridSearchCV to find the best parameters
        rf_model:RandomForestClassifier = grid_search.best_estimator_
        print(rf_model.get_params())
        return rf_model
    

