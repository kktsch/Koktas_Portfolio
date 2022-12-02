import pandas as pd

def PreProcess(path):

    data = pd.read_csv(path)
    data = data.drop_duplicates()

    df_obj = data.select_dtypes(['object'])
    data[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    data.CustomerNo = data.CustomerNo.astype(str)

    data_cancelled = data[data["TransactionNo"].str.contains("C")]
    data = data[~data["TransactionNo"].str.contains("C")]
    data = data.sort_values(by="Quantity", ascending=False).iloc[10:,:]

    data['Revenue'] = data.Price*data.Quantity
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values("Date")
    data = data.set_index("Date",drop=True)

    revenue = data.resample("D").sum().Revenue
    train = revenue[:-30].to_frame() 
    test = revenue[-30:].to_frame()

    data_train = train.copy()
    data_train.columns = ['Value']
    train = data_train.iloc[:-30]
    valid = data_train.iloc[-30:]
    
    return train, valid