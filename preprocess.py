import pandas as pd
import numpy as np
import seaborn as sns

class Preprocess:
    data = None

    def __init__(self):
        sns.set_style("ticks")


        df = pd.read_csv("combined_data_1.csv", header = None, names = ['Cust_Id','Rating','Date'], usecols = [0,1,2])

        df['Rating'] = df['Rating'].astype(float)
        
        df = df.drop(df.columns[[2]],axis = 1)
        
        movie_count = df.isnull().sum()[1]  # 1. indexi(Rating sütunu) boş olanları yani filmlerin sayısını çekiyor.

        # %% Data Temizleme
        df_nan = pd.DataFrame(pd.isnull(df.Rating))  # Oy gözükmeyen satırlar true, oylar ise false
        df_nan = df_nan[df_nan['Rating'] == True]  # True olanları listeliyor 
        df_nan = df_nan.reset_index()  # Bu sayede filmlere ait oyların başlangıç ve bitiş indexlerini görebiliyoruz

        movie_np = []
        movie_id = 1
        # Hangi satırın hangi filme ait oy olduğunu yazan bir movie_np dizisi oluşturuyor
        # Bu sayede örneğin 134589. indexin 23 id li filme ait oy olduğunu biliyoruz
        for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
            
            temp = np.full((1, i - j - 1), movie_id) # her değerin movie_id değerine eşit olduğu bir satır matris oluşuyor
            movie_np = np.append(movie_np, temp) # bu matrisin transpozesi alınarak movie_np dizisine ekleniyor
            movie_id += 1

        # Yukarıdaki döngü son filme geldiğinde yani 59.indexteyken 60. index olmadığı için,
        # genel uzunluktan çıkarılarak 60. filmin uzunluğu bulunuyor.
        last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
        movie_np = np.append(movie_np, last_record)

        # 1:,2: gibi satırları silip Raiting sütunun yanına o satıra ait movie_id değerlerini ekliyor
        # bu işlemden sonra df matrisi tamamen oy sayısı kadar satıra sahip oluyor
        df = df[pd.notnull(df['Rating'])]

        df['Movie_Id'] = movie_np.astype(int)
        df['Cust_Id'] = df['Cust_Id'].astype(int)

        # %% Data Parçalama

        f = ['count', 'mean']
        # movie_benchmark ve cust_benchmark  hesaplamalardan önce anlamsız verilerin hesaplanması için bir değer buluyor
        
        # Her film için olması gereken minimum oy sayısını hesaplıyor
        # drop_movie_list bu hesabın altında kalanlar
        df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)
        df_movie_summary.index = df_movie_summary.index.map(int)
        movie_benchmark = round(df_movie_summary['count'].quantile(0.8), 0)
        drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

        # Her kullanıcı için oy vermesi gereken minimum film sayısını hesaplıyor
        # drop_cust_list bu hesabın altında kalanlar
        df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)
        df_cust_summary.index = df_cust_summary.index.map(int)
        cust_benchmark = round(df_cust_summary['count'].quantile(0.8), 0)
        drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

        # verileri temizliyor
        df = df[~df['Movie_Id'].isin(drop_movie_list)]
        df = df[~df['Cust_Id'].isin(drop_cust_list)]
 
        # %% Data Dönüştürme
        df_p = pd.pivot_table(df,values='Rating',index='Cust_Id',columns='Movie_Id')
      
        #1-2 oy alan filmler 0
        #3-4-5 oy alan filmler 1
        for col in df_p:
           df_p.loc[df_p[col] <= 2.0, col] = 0
           df_p.loc[df_p[col] > 2.0, col] = 1
        
        #nan değerler 0 olarak değiştiriliyor
        df_p = df_p.replace(np.nan,0)
       
        print(df_p.shape)
        
        #seçilen film
        movie = df_p.iloc[:,5]
        
        self.select_movie  = movie.head(48000)
        self.predict_movie = movie.tail(12000)
        
        df_p = df_p.drop(df_p.columns[[5]], axis=1) 
        
        self.data_train = df_p.head(48000)
        self.data_predict = df_p.tail(12000)
        
        
        

    def get_training_inputs(self):
        return self.data_train # eğitim verisi için datanın ilk %80'lik kısmı yollanıyor

    def get_training_outputs(self):
        return self.select_movie # eğitim verisi için yollanan datanın sonuç çıktısı yollanıyor

    def get_test_inputs(self):
        return self.data_predict # test verisi için datanın son %20'lik kısmı yollanıyor

    def get_test_outputs(self):
        return self.predict_movie # Tahmin sonucu çıkan veri ile kıyaslanması için gerçek veri yollanıyor

