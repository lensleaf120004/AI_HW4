# polynominal
# svm2.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import SVC


# 統一指定資料中選取的項目欄位
SELECTED_ITEMS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Gender",               # 性別會轉成0和1

    "Geography_France",     # 國家位碼表示 [1,0,0]
    "Geography_Germany",    # [0,1,0]
    "Geography_Spain",      # [0,0,1]
]


"""整理 : 讀取訓練、驗證與測試資料"""
def load_data():

    # 讀取 train.csv 的 train 資料 : 含所有項目欄位
    train_data = pd.read_csv("train.csv")
    # 讀取train資料的正確解答 : Exited
    train_gt   = pd.read_csv("train_gt.csv")
    # 讀取 val.csv 的 validation 資料 : 含所有項目欄位
    val_df     = pd.read_csv("val.csv")
    # 讀取 test.csv 的 test 資料 : 含所有項目欄位
    test_df    = pd.read_csv("test.csv")

    # ---- 特別處理 Gender 項目欄位：將 'Female' / 'Male' 轉成 0 / 1 ----
    for df in (train_data, val_df, test_df):
        df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

    # ---- 特別處理 Geography 項目欄位： France / Germany / Spain ----
    # 將 train 和 val 還有 test 資料中的 Geography 欄位都轉化成位碼
    train_data = pd.get_dummies(train_data, columns=["Geography"])
    val_df     = pd.get_dummies(val_df,   columns=["Geography"])
    test_df    = pd.get_dummies(test_df,  columns=["Geography"])


    # 從 train 資料中取出指定的項目欄位
    x_train = train_data[SELECTED_ITEMS]
    # 從 train 資料的正確解答中取出 Exited 值
    y_train = train_gt["Exited"]

    # 從 validation 資料中取出指定的項目欄位
    x_val  = val_df[SELECTED_ITEMS]
    # 從 test 資料中取出指定的項目欄位
    x_test = test_df[SELECTED_ITEMS]

    return x_train, y_train, x_val, x_test



"""輸出 : 將預測結果輸出成.csv檔"""
def save_predictions(y_pred, filename):
    
    out_data = pd.DataFrame({"Exited": y_pred})
    out_data.to_csv(filename, index=False)



"""主程式"""
def main():
    
    # 呼叫函式 => 建立 input 資料
    x_train, y_train, x_val, x_test = load_data()

    # =========== 資料前處理 Step1. 標準化 ===========

    ## 建立一個標準化物件
    scaler = StandardScaler()
    
    ## fit 會計算參數 x_train 的平均值和標準差、transform會利用fit計算出的平均值和標準差將 x_train 標準化
    ## 標準化後的 x_train 存成 x_train_sta
    ## 只有 train 先 fit 再 transform 、 val 和 test 不另外 fit
    x_train_standard = scaler.fit_transform(x_train)
    
    x_val_standard   = scaler.transform(x_val)
    x_test_standard  = scaler.transform(x_test)

    # =========== 資料前處理 Step2. 正規化 ===========

    ## 建立一個正規化物件 : 使用 l2 正規化，將每筆數據向量長度縮放成 1
    normalizer = Normalizer(norm="l2")
    
    ## 對每一個已標準化後的訓練資料們再做正規化
    x_train_normal = normalizer.fit_transform(x_train_standard)
    x_val_normal   = normalizer.transform(x_val_standard)
    x_test_normal  = normalizer.transform(x_test_standard)

    # =========== Polynomial kernel SVM 相關參數設定 ===========
    svm = SVC(
        kernel="poly",
        degree=7,
        gamma=1.0,
        coef0=1.0,
        C=1.0,
        random_state=40,
    )

    # 訓練模型 (用前處理完畢的訓練資料 x_train_norm 來訓練，y_train 是 train 資料的正解)
    svm.fit(x_train_normal, y_train)

    # 對前處理完的 validation 資料做預測得到 val_pred
    val_pred  = svm.predict(x_val_normal)
    # 對前處理完的 test 資料做預測得到 test_pred
    test_pred = svm.predict(x_test_normal)

    # 呼叫函式將 val_pred、test_pred 的預測結果存成.csv檔 => 建立 output 資料
    save_predictions(val_pred, "val_pred.csv")
    save_predictions(test_pred, "test_pred.csv")


if __name__ == "__main__":
    main()
