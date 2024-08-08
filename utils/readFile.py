import pandas as pd

df1 = pd.read_csv(r'C:\Users\161070\Downloads\News_text.csv',low_memory=False)


df9= pd.read_csv(r"C:\Users\161070\OneDrive - Arrow Electronics, Inc\Desktop\Classification_Final\data\debug_data_out.csv",low_memory=False)
#colums :- PDF_URL,PDF_ID,VENDOR_CODE,PROJECT_1,PROJECT,text
df = pd.concat([df1,df9],axis=0)
df = df.drop_duplicates(subset=['PDF_ID'])
print(df.info())
df['text'] = df['text'].replace('\n',' ')
df['text'] = df['text'].replace('\t',' ')
df['text'] = df['text'].replace(',',' ')
print(len(df))
print
df.to_csv(r'C:\Users\161070\OneDrive - Arrow Electronics, Inc\Desktop\Classification_Final\data\debug_data_out.csv',encoding="utf-8")