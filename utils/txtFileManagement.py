import csv
import os
import pandas as pd

def remove_o(directory = r"data\text_files"):
    # Specify the directory containing the files
    
    dirs =os.listdir(directory)
    # Iterate over each file in the directory
    for filename in dirs:
        # Check if the file ends with '.0'
        if filename.endswith('.0.txt'):
            # Construct the new filename by removing '.0'
            new_filename = filename.replace('.0', '')
            try:
                # Rename the file
                os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
                print(f"Renamed {filename} to {new_filename}")
            except:
                print("error: "+f"Renamed {filename} to {new_filename}" )


def export_id(directory = r"data\text_files",filename = 'data\\all_Text.csv'):
    
    dirs =os.listdir(directory)
    dirs = [ d.split('.')[0] for d in dirs]
    # Open the file in write mode
    
    with open(filename, mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        
        # Write each row from the list to the CSV file
        for row in dirs:
            writer.writerow([row])



def get_ids_from_debug(source_path =r'C:\Users\161070\Downloads\FULL_DATA_out.csv',dest_path = r'C:\Users\161070\OneDrive - Arrow Electronics, Inc\Desktop\ids_file.csv' ):
    df = pd.read_csv(source_path,low_memory=False,  delimiter=',', encoding='utf-8')
    df_ids = df['pdf_url']
    df_ids.to_csv(dest_path,encoding="utf-8")


def update_classification(id_source = r'C:\Users\161070\OneDrive - Arrow Electronics, Inc\Desktop\datasheet_filters.csv',data_classification=r'C:\Users\161070\OneDrive - Arrow Electronics, Inc\Desktop\Document Classification\data\Log\debug_data_out.csv'):
    df= pd.read_csv(id_source,low_memory=False)
    data = pd.read_csv(data_classification,low_memory=False)
    datasheet_list = df['PDF_ID'].to_list()
    print(data.columns)
    # Convert the list to a set for faster lookup
    print(data['classification'].value_counts())
    print(len(data))
    
    # Convert the list to a set for faster lookup
    datasheet_set = set(datasheet_list)

    # Get the boolean mask where PDF_ID is in datasheet_set
    mask = data['PDF_ID'].isin(datasheet_set)
    d = data[mask]
    print(d['classification'].value_counts())
    # Get the indices of rows where the condition is True
    indices = data[mask].index.tolist()
    print(len(data))
    
def save_csv_to_files(path = r"C:\Users\161070\Downloads\TABLE_EXPORT_DATA_text.csv", dir = r'C:\Users\161070\OneDrive - Arrow Electronics, Inc\Desktop\Classification_Final\data\text_files'):
    
    
    files = os.listdir(dir)
    df= pd.read_csv(path,encoding='utf-8',low_memory=False)
    df = df.dropna(subset=['text'])
    for index,item in df.iterrows():
        
        x=item['PDF_ID']
        x=str(x)
        x=x.replace('.0','')
        txt_path = dir +'\\'+ x+'.txt'
        if txt_path not in files:
            text = item['text']

            with open(txt_path, "w",encoding="utf-8") as file:
                    file.write(str(text))
                    

def join_Files(file1=r"C:\Users\161070\OneDrive - Arrow Electronics, Inc\Desktop\datasheet_all\TABLE_EXPORT_DATA_1.csv",file2=r"C:\Users\161070\Downloads\FULL_DATA_out.csv",res=r"C:\Users\161070\Downloads\FULL_DATA_out2.csv"):
    
    print("file1 reading")
    
    
    df1= pd.read_csv(file1,encoding='utf-8',low_memory=False)
    print("file2 reading")
    df2 = pd.read_csv(file2,encoding='utf-8',low_memory=False)
    
    print("merging...")
    df = df1.merge(df2,on='PDF_URL', how='inner')
    df['classification']= df['PROJECT']
    
    print(df.head())
    df.to_csv(res,encoding='utf-8')

    pass

def check_csv(path=r'C:\Users\161070\OneDrive - Arrow Electronics, Inc\Desktop\Document Classification\data\Log\debug_data_out_last.csv'):
    df = pd.read_csv(path,low_memory=False,encoding="utf-8")
    print(df.columns)
    print(len(df))
    
    print(df['classification'].value_counts())
    print(df['PROJECT'].value_counts())
    pass

def check_csv(path=r'C:\Users\161070\OneDrive - Arrow Electronics, Inc\Desktop\Classification_Final\data\debug_data_out.csv'):
    df = pd.read_csv(path,low_memory=False,encoding="utf-8")
    df.loc[df['PROJECT_1'].str.contains('Declaration', na=False), 'classification'] = 'Environmental'

    dd= df[((df['PROJECT_1'].str.contains('Declaration', na=False)))][['PROJECT','PROJECT_1','classification']]
    print(dd)
    print(dd['classification'].unique())
    df.to_csv(path,encoding="utf-8")
    
    pass



def remove_duplicates(file= r'C:\Users\161070\OneDrive - Arrow Electronics, Inc\Desktop\Document Classification\data\Log\debug_data_out5_.csv'
    ):
    df = pd.read_csv(file,encoding='utf-8',low_memory=False)
    df = df.drop_duplicates(subset=['PDF_URL','PDF_ID'])
    df.to_csv(file,encoding='utf-8')



def notRelated():
    dfp1=r"\\10.199.104.106\SW_backup\Ali\filelist2.txt"
    dfp2= r"C:\Users\161070\Downloads\Book32.xlsx"
    df1 = pd.read_csv(dfp1,low_memory=False)
    
    print(df1.head()) 
    df2 = pd.read_excel(dfp2)
    print(df2.head())
    df1 = df1[~df1['URL'].str.contains('-7-2024')]
    df1 = df1[~df1['URL'].str.contains('-6-2024')]
    df1 = df1[df1['URL'].str.contains('.pdf')]
    df1 = df1[~df1['URL'].isin(df2['ACROBAT_REPORT_PATH'])]
    dfp1=r"\\10.199.104.106\SW_backup\Ali\filelist3.txt"
    
    df1.to_csv(dfp1,index=False,sep='\t')


export_id()