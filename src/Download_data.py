from sklearn.datasets import fetch_20newsgroups 
import pandas as pd

#  you Can Go to this Link to Download Data

# Link = https://drive.google.com/drive/folders/16z_B4fkKhq-UQqqv8f7CiBy64SFKZKyw?usp=drive_link 

# Download it and put it in a data dir


def collect_data():
    dataset = fetch_20newsgroups(subset='all', 
                    shuffle=False, remove=('headers', 'footers', 'quotes'),categories=[
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',])    
    df = pd.DataFrame()
    df["data"] = dataset["data"]
    df["target"] = dataset["target"]
    df["target_names"] = df.target.apply(lambda row: dataset["target_names"][row])
    return df