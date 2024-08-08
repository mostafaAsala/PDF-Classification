import io
import requests
import traceback
import pdfplumber
import fitz
from io import BytesIO
import pytesseract
from pdf2image import convert_from_bytes
import utils.constants as constants
import os
import pandas as pd
# Function to read text files based on file paths

def read_text(file_path):
    """
    Description:
        This function reads the content of a text file from a given file path and returns the text as a string. If an error occurs during the reading process, it prints an error message and returns an empty string.

    Parameters:
        file_path (str): The path to the text file.
    Return Type:
        str: The content of the text file or an empty string if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except:
        print("error reading file \""+ file_path+"\"")
        print(traceback.format_exc())
        return ""


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
            return True
             
        else:
            print( f"Failed to download PDF. Status code: {response.status_code}")
            return False

    except Exception as e:
       print( f"Error downloading PDF: {e}")
       return False

def process_pdf(pdf_bytes, max_pages=3):
    
    try:
        bytes_ = io.BytesIO(pdf_bytes.content)

        # Convert PDF to images
        images = convert_from_bytes(bytes_.getvalue(),poppler_path=constants.poopler_Path)
        # Perform OCR on each image, up to max_pages 
        text = ''
        for i, image in enumerate(images):
            if i >= max_pages:
                break
            text += pytesseract.image_to_string(image)
        if len(text)<constants.min_Letters_in_text:
            status=0
            text=""
        else:
            status = 1  # Success
    except Exception as e:
        text = ""
        status = 0  # Error
    return {"text": text, "status": status}


def extract_text_from_pdf_Production(pdf_path,top_margin=constants.top_margin,bottom_margin=constants.bottom_margin,pages=3):
    doc_len=0
    try:
        text=""
        response = requests.get(pdf_path)
        if response.status_code == 200:
            stream = response.content

            with fitz.open(stream=BytesIO(stream)) as doc:
                doc_len=len(doc)
                doclen= pages if len(doc)>pages else len(doc)
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
            if len(text) < constants.min_Letters_in_text:
                print("empty pdf_ trying OCR to extract")
                data = process_pdf(response)
                if data['status']==0:
                    return 0,""
                else:
                    return doc_len,data['text']
            else:
                return doc_len,text
            
    except Exception as e:
        print(traceback.format_exc())
        return 0,""
    

def get_links(file_path = constants.link_file_final):

    df = pd.read_csv(file_path,encoding='utf-8',low_memory=False)
    df = df[['PDF_ID','PDF_URL','FILE_SIZE','PAGES','VENDOR_CODE','PROJECT']]
    boolean_function = df['PDF_URL'].apply(lambda x: not(is_valid_pdf_link(x)))
    x = df[boolean_function].index
    print(x)
    df = df.drop(df[boolean_function].index)
    return df


def get_all_files(pdf_path=constants.link_file_final, datashet_path=constants.link_file_final,use_Exist=True):
    if os.path.exists(constants.link_file_final) and use_Exist:
        print("Get files From Info")
        df = pd.read_csv(constants.link_file_final,encoding='utf-8',low_memory=False,dtype={"PDF_URL": str})
        print(df.info())
    else:
        df1 = get_links(pdf_path)
        
        print(len(df1))
        df2 = get_links(datashet_path)
        mask = df1['PDF_ID'].astype(str).isin(df2['PDF_ID'].astype(str).tolist())
        
        df1 = df1[~mask]
        
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



def data_extract3(dataLabel_List=[],exclude_list=['Delete_HTML','In_Complete_DS'],start=0, finish=10000000):
    previously_extracted_text =  os.listdir(constants.result_text_path)
    error_data ={}
    data = get_all_files(pdf_path=constants.link_file_final,datashet_path=constants.link_file_final)
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