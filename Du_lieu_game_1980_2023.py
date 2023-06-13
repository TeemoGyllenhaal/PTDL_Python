import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\kulin\OneDrive\Máy tính\games.csv")
print(df)
print(df['Plays'])
print(df.head())
print(df.tail())
print(df.loc[1])

print(df.isna().sum())
print(df[['Title', 'Release Date', 'Team', 'Rating']][df.isna().any(axis=1)])
df.dropna(inplace=True)
print(df[['Title', 'Release Date', 'Team', 'Rating']][df.isna().any(axis=1)])
col_drop = ['Summary', 'Reviews']
df = df.drop(col_drop, axis=1)

print(df.duplicated(subset=df.columns.difference(['Number'])).sum())
print(df[df.duplicated(subset=df.columns.difference(['Number'])) == True].head())

df = df.drop_duplicates(subset=df.columns.difference(['Number']), keep='first')

# Chuyển các giá trị số với kí tự đặc biệt sang dạng integer 
col_num = ['Times Listed', 'Number of Reviews', 'Plays', 'Playing', 'Backlogs', 'Wishlist']
for col in col_num:
    df[col] = df[col].str.replace("K", "000", regex=False).str.replace(".", "").astype('int')

# Chuyển các giá trị chữ sang một dạng list
import ast
df['Genres'] = df['Genres'].apply(ast.literal_eval)
df['Team'] = df['Team'].apply(ast.literal_eval)

# Chuyển cột 'Release Date' đến dạng ngày chuẩn
index_to_drop = df[df['Release Date'] == 'releases on TBD'].index
df = df.drop(index=index_to_drop, axis=0)
df["Release Date"] = pd.to_datetime(df["Release Date"])



#Tạo cột Year mới để phục vụ các bài tính sau
df["Year"] = df["Release Date"].dt.year
print (df['Year'])

trung_binh = df['Rating'].mean()            # Tính trung bình cột "Rating" bằng hàm mean()
print("Trung bình: ",trung_binh)
trung_vi = df['Rating'].median()            # Tính trung vị cột "Rating" bằng hàm median()
print("Trung vị: ",trung_vi)
phuong_sai = df["Rating"].var()             # Tính phương sai cột "Rating" bằng hàm var()
print("Phương sai: ",phuong_sai)
do_lech_chuan = df["Rating"].std()          # Tính độ lệch chuẩn cột "Rating" bằng hàm std()
print("Độ lệch chuẩn: ",do_lech_chuan)

nho_nhat = df['Rating'].min()
print("Điểm đánh giá thấp nhất: ",nho_nhat)
print("Game bị đánh giá thấp nhất ",df.loc[df['Rating'] == nho_nhat])
lon_nhat = df['Rating'].max()
print("Điểm đánh giá cao nhất: ", lon_nhat)
print("Game được đánh giá cao nhất : ",df.loc[df['Rating']==lon_nhat])



tan_so = df['Year'].value_counts().sort_index()
print(tan_so)

tu_phan_vi = df['Rating'].quantile([0.25, 0.5, 0.75])
print("Tứ phân vị:\n",tu_phan_vi)


rating = df.groupby("Rating")
print(rating)
rating_1 = rating["Plays"].sum()
print("Số lượng lượt chơi game ở từng thang điểm:\n",rating_1)
rating_2 = rating["Title"].count()
print("Số lượng game nằm ở từng thang điểm:\n",rating_2)


year = df.groupby("Year")
year_1 = year["Plays"].sum()
print("Số lượng lượt chơi game ở từng năm:\n" , year_1)    
year_2 = year["Rating"].mean()
print("Trung bình điểm rating từng năm:\n", year_2)

import seaborn as sns
plt.figure(figsize=(15, 5))
sns.lineplot(data=year_2)
sns.despine(top=True, right=True)
plt.title("Trung bình đánh giá game bởi năm phát hành", fontsize=16, pad=15)
plt.xlabel("Năm")
plt.ylabel("Điểm đánh giá trung bình", labelpad=15, fontsize=15)
plt.xticks(range(1980,2023,4),fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# # print(b)

product_counts = df['Year'].value_counts().sort_index()
plt.bar(product_counts.index, product_counts.values)
plt.title('Số lượng sản phẩm phát hành theo năm')
plt.xlabel('Năm')
plt.ylabel('Số lượng')
plt.show()

colors = plt.cm.rainbow(np.linspace(0, 1, len(year_1)))
plt.bar(year_1.index, year_1.values, color = colors)
plt.title('Số lượng người chơi theo năm')
plt.xlabel('Năm')
plt.ylabel('Số lượng')
plt.show()


all_genres = [genre for genres_list in df['Genres'] for genre in genres_list]
genre_counts = pd.Series(all_genres).value_counts()
print("Số lượng game đã ra mắt ở từng thể loại:\n ",genre_counts)
fig, ax = plt.subplots(figsize=(16, 9))
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='rainbow')
ax.set_title("Top thể loại game phổ biến", fontsize=15)
# Thêm giá trị phần trăm
for i, v in enumerate(genre_counts.values):
    ax.text(v, i, str(round(v / len(df) * 100, 2)) + ' %', color="black", ha="left", va="center", fontsize=10)
ax.set_xlim(right=max(genre_counts.values) * 1.1)
sns.despine(right=True, top=True)
plt.show()

# b.to_excel(r"C:\Users\kulin\OneDrive\Máy tính\27_05_23.xlsx", index=False)

xep_hang = ["Tệ", "Tạm được", "Khá hay", "Hay", "Cực phê"]
dieu_kien = [0, 2.5, 3, 4, 4.2, 5]
df["Xếp hạng Game"] = pd.cut(df["Rating"], bins=dieu_kien, labels=xep_hang)
tan_so_1 = df["Xếp hạng Game"].value_counts().sort_index()
print(tan_so_1)
my_explode = [0, 0, 0, 0, 0.1]
my_colors = ['#E74C3C', '#F39C12', '#27AE60', '#3498DB', '#9B59B6']
plt.pie(tan_so_1, labels=xep_hang, autopct="%.1f%%", colors=my_colors,startangle=90, explode=my_explode , shadow= True, wedgeprops={'edgecolor': 'black', 'linewidth': 1})
plt.title("Phân bố xếp hạng game")
plt.legend(loc='upper center', bbox_to_anchor=(0, 0.2))
plt.show()


from scipy.stats import t

do_tin_cay = float(input("Nhập giá trị tin cậy: "))
a = (1 - do_tin_cay)
print ("Giá trị a: ",a)
n = int(input("Nhập số lượng giá trị mẫu: "))

column_name = 'Rating'
mau_ngau_nhien = df.sample(n, replace=False)[column_name]
print (mau_ngau_nhien)

# Tính giá trị trung bình mẫu n 
mean_rating = mau_ngau_nhien.mean()
print ("Giá trị trung bình mẫu: ",mean_rating)
var_rating = mau_ngau_nhien.var()
print ("Giá trị phương sai mẫu hiệu chỉnh: ",var_rating )
std_dev = np.std(mau_ngau_nhien, ddof=1)
print("Độ lệch mẫu hiệu chỉnh: ", std_dev)

gia_tri_student =  abs(t.ppf((a)/2, n-1))
print ("Giá trị t là: ", gia_tri_student)
# gia_tri_student = 2.093

import math
dang_truoc = mean_rating - (math.sqrt(var_rating)/math.sqrt(n)*gia_tri_student)
dang_sau = mean_rating + (math.sqrt(var_rating)/math.sqrt(n)*gia_tri_student)
sai_so = gia_tri_student * std_dev / np.sqrt(n)
print ("Giá trị sai số của ước lượng là: ", sai_so)
khoang_uoc_luong = (mean_rating - sai_so, mean_rating + sai_so)
print("Khoảng ước lượng giá trị trung bình:", khoang_uoc_luong)




# Bài toán kiểm định 
# giả thiết H0 : vào năm 2019 chất lượng game không giảm ( = giá trị trung bình )
# giả thiết H1 : vào năm 2019 chất lượng game giảm 




do_tin_cay_2 = float(input("Nhập giá trị tin cậy: "))
a1 = (1 - do_tin_cay_2)
print ("Giá trị a: ",a1)

n1 = int(input("Nhập số lượng giá trị mẫu: "))
mau_ngau_nhien_1 = df[df['Year'] == 2019].sample(n1, replace=False)['Rating']
print(mau_ngau_nhien_1)

mean_rating_1 = mau_ngau_nhien_1.mean()
print ("Giá trị trung bình mẫu: ",mean_rating_1)
std_dev_1 = np.std(mau_ngau_nhien_1, ddof=1)
print("Độ lệch chuẩn mẫu hiệu chỉnh: ", std_dev_1)

x = df[df["Year"] == 2018]
gia_tri_trung_binh_kiem_dinh = x["Rating"].mean()
print("Giá trị trung bình game năm 2018: ",gia_tri_trung_binh_kiem_dinh)
#Tính u
import scipy.stats as stats
gia_tri_student_1 =  t.ppf(a1, n-1)
print ("Giá trị phân phối t-student cho miền bác bỏ W: là: ", gia_tri_student_1)

bieu_tuong_vo_cung = '\u221e'
W = ((bieu_tuong_vo_cung), gia_tri_student_1)
print("Miền bác bỏ W (phía trái):", W)
t_observed = ((mean_rating_1 - gia_tri_trung_binh_kiem_dinh)*np.sqrt(n)) / (std_dev_1)
print("Giá trị t quan sát là: ", t_observed)
if t_observed > gia_tri_student_1:
    print("Giá trị t quan sát nằm ngoài miền bác bỏ.")
    print("Chấp thuận H0, tức là điểm game không giảm đi vào năm 2019")
else:
    print("Giá trị t quan sát không nằm ngoài miền bác bỏ.")
    print("Bác bỏ H0, chấp thuận H1, tức là giá trị game đã thực sự giảm đi so với năm 2018.")

import pandas as pd
from scipy import stats


print("cách 2: làm bằng t_test\n")
# Tách dữ liệu rating của game vào năm 2019 và các năm trước
data_2018 = df[df["Year"]== 2018 ]["Rating"]
data_2019 = df[df["Year"]== 2019 ]["Rating"]

# Thực hiện kiểm định t-test
t_statistic, p_value = stats.ttest_ind(data_2018, data_2019, equal_var=False)

# In kết quả
print("Giá trị thống kê t:", t_statistic)
print("Giá trị p:", p_value)
# Kiểm tra giả thiết
if p_value > 0.05:
    print("Chấp thuận H0, tức là chất lượng của game 2019 không đổi so với năm trước")
else:
    print("Có đủ bằng chứng để bác bỏ giả thiết H0.")
    print("Bác bỏ H0, chấp thuận H1, tức là chất lượng game đã thực sự giảm đi so với năm 2018.")


#Tương quan
import numpy as np
# Lọc dữ liệu cho năm 2022
df_2022 = df[df['Year'] == 2022]  # Giả sử cột chứa năm có tên là 'Năm'
# Lấy mẫu ngẫu nhiên giá trị từ dataframe của năm 2022
n2 = int(input("Nhập số lượng giá trị mẫu: "))
sample_2022 = df_2022.sample(n2, random_state=98)  # Số 42 là một số ngẫu nhiên để cố định mẫu lấy
# Tính toán hệ số tương quan Pearson cho mẫu lấy
correlation = np.corrcoef(sample_2022['Rating'], sample_2022['Plays'])[0, 1]
print("Hệ số tương quan Pearson của mẫu lấy năm 2022 (",n2,"giá trị): ", correlation)
from scipy.stats import pearsonr
pearsonr_2022 = pearsonr(sample_2022['Rating'], sample_2022['Plays'])
print(pearsonr_2022)
#Qua giá trị p >0.05 nên không có giá trị ý nghĩa thống kê của tương quan 2 biến này
#Vẽ biểu đồ
# Dữ liệu mẫu
x = sample_2022['Rating']
y = sample_2022['Plays']
# Tính hệ số tương quan
correlation = np.corrcoef(x, y)[0, 1]
# Tính hệ số góc và hệ số điểm của đường thẳng tương quan
slope = abs(correlation) * (np.std(y) / np.std(x))
intercept = np.mean(y) - slope * np.mean(x)
# Vẽ scatter plot
plt.scatter(x, y)
plt.xlabel('Rating')
plt.ylabel('Plays')
plt.title('Sự tương quan giữa điểm số với số lượng người chơi vào năm 2022')
# Vẽ đường thẳng tương quan thuận tuyến tính
x_line = np.linspace(min(x), max(x), 100)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, color='red', linestyle='--', label='Tương quan tuyến tính')
plt.legend()
plt.show()


#Hồi quy
from sklearn.linear_model import LinearRegression

# Chuẩn bị dữ liệu
x = df_2022['Rating'].values.reshape(-1, 1)  # Biến độc lập: Rating
y = df_2022['Plays'].values  # Biến phụ thuộc: Plays
# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(x, y)

# In ra hệ số hồi quy và hệ số chặn
print('Hệ số hồi quy:', model.coef_[0])
print('Hệ số chặn:', model.intercept_)
print('Phương trình hồi quy có dạng: y= ', model.coef_[0],'x + (',model.intercept_,')')
# Dự đoán giá trị Plays từ biến Rating
rating_new = float(input("Nhập giá trị điểm để dự đoán số lượt chơi: "))
rating_new_1 = np.array([rating_new])  # Giá trị Rating mới để dự đoán
plays_predicted = model.predict(rating_new_1.reshape(-1, 1))
print('Giá trị số lượt chơi vào năm 2022 được dự đoán là:', np.round(plays_predicted), 'lượt chơi')
