import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

import os

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

import joblib
model_loaded = joblib.load(resource_path("25-11-2022-AttritionML-FinalModel.pkl"))

def FileDialog():
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = ([("Excel files", ".xlsx .xls")]))
    
    
    label_file['text'] = filename
    LoadExcelData()
    return None

    
def LoadExcelData():
    
    def SaveFile(df):
        file = filedialog.asksaveasfilename(filetypes=(("Excel files", "*.xlsx"),
                                           ("All files", "*.*") ))
        if file is None: # asksaveasfile return `None` if dialog closed with "cancel".
            return
        df.to_excel(file + ".xlsx", index=False, sheet_name="Results")
        frame1 = tk.Frame(root, height=120, width=400, bg="#609AB9")
        frame1.place(x=120,y=130)
        frame1.pack_propagate(0)

        lbl1 = tk.Label(frame1, text="Load the employee data.", font=("Arial",15), bg="#609AB9")
        lbl1.pack()

        button = tk.Button(frame1, text="BrowseFiles", command=FileDialog)
        button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    
    def PrepareData(data):
        data = data[['EmployeeID', 'Age', 'BusinessTravel', 'Department', 'DistanceFromHome',
       'Education', 'EducationField', 'EmployeeCount', 'Gender', 'JobLevel',
       'JobRole', 'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked',
       'Over18', 'PercentSalaryHike', 'StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
       'YearsSinceLastPromotion', 'YearsWithCurrManager',
       'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance',
       'JobInvolvement', 'PerformanceRating']]
        
        X = data.drop(['EmployeeCount', 'StandardHours', 'EmployeeID'], axis=1)

        X["JobInvolvement-Performance"] = X["JobInvolvement"]*X["PerformanceRating"]
        X = X.drop(['JobInvolvement','PerformanceRating'], axis=1)
        X["Satisfaction"] = X["JobSatisfaction"]*X["EnvironmentSatisfaction"]*X["WorkLifeBalance"]
        X = X.drop(['JobSatisfaction','EnvironmentSatisfaction','WorkLifeBalance'], axis=1)

        X_num = X.drop(['BusinessTravel', 'Department', 'EducationField', 'Gender', 'DistanceFromHome', 'JobLevel',
           'JobRole', 'MaritalStatus', 'Over18', 'Education','StockOptionLevel','JobInvolvement-Performance','Satisfaction'], axis=1)
        X_num_tr = StandardScaler().fit_transform(X_num)

        cat_attribs = X[['BusinessTravel', 'Department', 'EducationField', 'Gender',
           'JobRole', 'MaritalStatus', 'Over18']]
        X_cat_1hot = pd.get_dummies(cat_attribs, prefix_sep="_", drop_first=True)

        label_attribs = X[['DistanceFromHome', 'Education','JobLevel','StockOptionLevel',
                     'JobInvolvement-Performance','Satisfaction']]
        X_cat_label = label_attribs.apply(LabelEncoder().fit_transform)

        X_prepared = np.concatenate((X_num_tr,X_cat_1hot,X_cat_label),axis=1)

        return X_prepared

    def Predict():
        X = PrepareData(df)
        y_pred = model_loaded.predict(X).tolist()
        y_pred_str = [str(x).replace("1","Yes") if x==1 else str(x).replace("0","No") for x in y_pred]
        
        new_df = df.copy()
        new_df['Attrition_Predicted'] = y_pred_str

        frame3 = tk.Frame(root, height=120, width=400, bg="#609AB9")
        frame3.place(x=120,y=130)
        frame3.pack_propagate(0)
        
        p = ttk.Progressbar(frame3 ,orient='horizontal',length=200,mode="determinate",takefocus=True,maximum=100)
        p.pack()            
        for i in range(1000):                
            p.step()            
            root.update()
        p.destroy()
        
        lbl3 = tk.Label(frame3, text="Analysis is done.", font=("Arial",15), bg="#609AB9")
        lbl3.pack()

        button2 = tk.Button(frame3, text="Save to computer",command=lambda: SaveFile(new_df))
        button2.place(relx=0.5, rely=0.5, anchor=tk.CENTER)    
    
    file_path = label_file['text']
    
    try:      
        excel_filename = r"{}".format(file_path)
        df = pd.read_excel(excel_filename)
        if df.isnull().any(axis=1).sum()>0:
            tk.messagebox.showerror("Information", "The table has empty cells.")
            return None

        
        frame2 = tk.Frame(root, height=120, width=400, bg="#609AB9")
        frame2.place(x=120,y=130)
        frame2.pack_propagate(0)
        
        lbl2 = tk.Label(frame2, text="File is loaded.", font=("Arial",15), bg="#609AB9")
        lbl2.pack()
        
        button2 = tk.Button(frame2, text="Start Analysis", command=Predict)
        button2.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
    #except ValueError:
     #   tk.messagebox.showerror("Information", "The file you have chosen is invalid.")
      #  return None
    except FileNotFoundError:
        tk.messagebox.showerror("Information", f"No file {file_path}")
        return None

root = tk.Tk()

root.title("Attrition Extractor")
root.configure(bg="#609AB9")
root.geometry("640x360+250+100")
root.pack_propagate(False)
root.resizable(height=False, width=False)

frame1 = tk.Frame(root, height=120, width=400, bg="#609AB9")
frame1.place(x=120,y=130)
frame1.pack_propagate(0)

lbl1 = tk.Label(frame1, text="Load the employee data.", font=("Arial",15), bg="#609AB9")
lbl1.pack()

button = tk.Button(frame1, text="BrowseFiles", command=FileDialog)
button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

label_file = tk.Label(frame1, text="No File Selected")

root.mainloop()