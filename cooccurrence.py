import pygsheets
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler, Binarizer

creds = pygsheets.authorize(service_file='Test Database-f3e687069ec2.json')
#client = gspread.authorize(creds)

workbook = creds.open("Cooccurrence")
sheet = workbook[0]

keyword_report = pd.read_csv("Search keyword report.csv")
keyword_list = keyword_report["Keyword"].tolist()

count_model = CountVectorizer(ngram_range=(1,1), stop_words = 'english')
X = count_model.fit_transform(keyword_list)
print(X)
Xc = (X.T * X)
Xc.setdiag(0)
names = count_model.get_feature_names()

df = pd.DataFrame(data = Xc.toarray(), columns = names, index = names)

scaler = MinMaxScaler()
output_norm = scaler.fit_transform(df)
binarizer = Binarizer(threshold = 0.8)
output_binary = binarizer.fit_transform(output_norm)

df_binary = pd.DataFrame(data = output_binary, columns = names, index = names)
df_norm = pd.DataFrame(data = output_norm, columns = names, index = names)

#writer = pd.ExcelWriter('cooccurrence2.xlsx', engine='xlsxwriter')
#df.to_excel(writer, sheet_name='Sheet1')
df_norm.to_csv("cooccurrence_norm.csv", sep = ";")
df_binary.to_csv("cooccurrence_binary.csv", sep = ";")
df.to_csv("cooccurrence_unprocessed.csv", sep = ";")