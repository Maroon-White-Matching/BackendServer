import json
import pandas as pd

def deleteContent(fName):
    with open(fName, "w"):
        pass

def initialize_json(filename,ccr,cl,cr):
    deleteContent(filename)
    f = open(filename, "a")   
    f.write('{"scoring_weights": { "cr":' + ccr + ',"cl":' +  cl + ',"ccr":' + cr + '}, "group1": [], "group2": []} ')
    f.close()

def write_json(dataframe, filename='data.json'):
    with open(filename,'w') as f:
        json.dump(dataframe, f, indent=4)

def createContent(dataframe, filename):
    for x in range(len(dataframe)):
        with open(filename) as json_file:
            json_data = json.load(json_file)
            if dataframe.at[x,'Role:'] == 'Coach':
                temp = json_data['group1']
            else:
                temp = json_data['group2']
            y = {
                "name": dataframe.at[x,'UID'],
                "person": dataframe.at[x,'Full Name (First Middle Last)'],
                "role": dataframe.at[x,'Role:'],
                "ccr": [dataframe.at[x,'Do you enjoy asking difficult questions?'], dataframe.at[x,'Do you enjoy receiving hard questions?'], dataframe.at[x,'How important is the consistency of  fellow-coach meetings?'],
                    dataframe.at[x,'How important is the flexibility of fellow-coach meetings?'], dataframe.at[x,'How important is the directness of feedback?'], dataframe.at[x,'How important is the directness of feedback?'], 
                    dataframe.at[x,'Rank the following: [Relationship & Connection]'], dataframe.at[x,'Rank the following: [Goal Orientation/Achievement]'], dataframe.at[x,'Rank the following: [Development of leadership identity]']],
                "cl": dataframe.at[x,'distance'],
                "cr": [dataframe.at[x, 'EXT'],dataframe.at[x, 'NEU'],dataframe.at[x, 'AGR'],dataframe.at[x, 'CON'],dataframe.at[x, 'OPN']]
            }
            temp.append(y)
        write_json(json_data, filename)
    
def unhash(key, data):
    search_space = key.iloc[:,[27,2]].copy()
    frame = search_space.set_index('Full Name (First Middle Last)')['UID'].to_dict()  
    json_frame = pd.DataFrame(list(data.items()), columns = ['coach','fellow'])
    dictionary = {v : k for k, v in frame.items()}
    json_frame.coach = json_frame.coach.replace(dictionary)
    json_frame.fellow = json_frame.fellow.replace(dictionary)
    return json_frame




    