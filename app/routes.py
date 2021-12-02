import pandas as pd
import gspread
from app import app, db
from app.models import Users, Result

import sys
sys.path.append('app/utils')

import match 
import nlp
import json_helper

from app.utils.match import *
from app.utils.json_helper import * 
from app.utils.nlp import *

from oauth2client.service_account import ServiceAccountCredentials
from flask_cors import CORS, cross_origin
from flask import jsonify
from flask import request
from flask_jwt_extended import create_access_token
from flask_jwt_extended import get_jwt_identity
from flask_jwt_extended import jwt_required

# Json file for crafting + formating api calls
filename="app/lib/data/data.json"

def pull_data():
    scope = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("Copy of Fellow-Coach Matching Form Responses").sheet1
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    return df

def statistics():
    data = pull_data()
    data = apply_nlp(data)
    data = big5(data)
    
    stats = pd.DataFrame() 

    # Copying number of personality types
    stats['EXT'] = data['EXT'].astype(int)
    stats['NEU'] = data['NEU'].astype(int)
    stats['AGR'] = data['AGR'].astype(int)
    stats['CON'] = data['CON'].astype(int)
    stats['OPN'] = data['OPN'].astype(int)
    stats = stats.sum()

    # Getting the number of unique Coaches and Mentors
    stats['Coaches'] = data['Role:'].value_counts()['Coach']
    stats['Fellows'] = data['Role:'].value_counts()['Fellow']

    return stats


def matching_algorithm(ccr,cl,cr): 
    data = pull_data()
    data = apply_nlp(data)
    data = padding(data)
    data = df_column_uniquify(data)
    data = KNN_dictionary(data)
    data = big5(data)

    # Creates JSON with correct format to pass to matching algo
    initialize_json(filename,ccr,cl,cr)
    createContent(data, filename)
    temp = json.load(open(filename, "r"))
    hashed_match_result = apply(temp)
    match_results = unhash(data,hashed_match_result)

    return match_results
    

@app.route("/")
def home_view():
        return "<h1> Online </h1>"

@app.route("/login", methods=["POST"])
def login():
    username = request.json.get("username", None)
    password = request.json.get("password", None)

    exists = False
    name = None
    role = None

    users = Users.query.all()
    for row in users:
        if(row.username == username and row.password == password):
            exists = True
            name= row.name
            role = row.role
            break

    if exists == False:
        return jsonify({"msg": "Bad username or password"}), 401

    if role == 'Pending Approval':
        return jsonify({"msg": "User has not been approved"}), 301

    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token, name=name, role=role)

@app.route("/create", methods=["POST"])
def create():
    name = request.json.get("name", None)
    username = request.json.get("username", None)
    password = request.json.get("password", None)

    users = Users.query.all()
    for row in users:
        if(row.username == username):
            return jsonify({"msg": "Username already exists"}), 410

    db.session.add(Users(name=name,username=username,password=password, role='Pending Approval'))
    db.session.commit()
    return jsonify({"msg": "Success"}), 200

@app.route("/users",methods=['GET'])
@jwt_required()
def sendUsers():
    users = Users.query.order_by(Users.id.desc()).all()
    respId = []
    respName = []
    respRole = []

    for row in users:
        respId.append(row.id)
        respName.append(row.name)
        respRole.append(row.role)
        
    return jsonify({"id": respId, "name": respName, "role": respRole})


@app.route("/update", methods=["POST"])
@jwt_required()
def update():
    id = request.json.get("id", None)
    value = request.json.get("value", None)
    user = Users.query.get(id)
    user.role = value
    db.session.commit()
    return jsonify({"msg": "Success"}), 200


@app.route("/delete", methods=["POST"])
@jwt_required()
def delete():
    id = request.json.get("id", None)
    duser = Users.query.get(id)
    db.session.delete(duser)
    db.session.commit()
    return jsonify({"msg": "Success"}), 200

@app.route("/retrieve", methods=["GET"])
@jwt_required()
def retrieve():
    scope = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("Fellow-Coach Matching Form Responses").sheet1
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    df.to_csv("app/lib/data/FormResponses.csv", index=False)
    return jsonify(data), 200 

@app.route("/results", methods=["POST"])
@jwt_required()
def results():
    ccr = request.json.get("ccr", None)
    cl = request.json.get("cl", None)
    cr = request.json.get("cr", None)

    results = matching_algorithm(ccr,cl,cr)
    db.session.query(Result).delete()
    for index, row in results.iterrows():
        db.session.add(Result(id = index, coach = row.coach, fellow = row.fellow))
        db.session.commit()

    return results.to_json(default_handler=str), 200 

@app.route("/stats", methods=["GET"])
@jwt_required()
def display():
    stats = statistics()
    return stats.to_json(), 200 

@app.route("/finalMatches",methods=['GET'])
@jwt_required()
def sendFinalMatches():
    users = Result.query.order_by(Result.id.desc()).all()
    respId = []
    respCoach = []
    respFellow = []

    for row in users:
        respId.append(row.id)
        respCoach.append(row.coach)
        respFellow.append(row.fellow)
        
    return jsonify({"id": respId, "coach": respCoach, "fellow": respFellow})
