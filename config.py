import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-never-know-anything' 
    SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL'].replace("://", "ql://", 1)
    # SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:49Caolo2020@localhost/capstone'
    #SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://postgres:postgres@localhost/postgres'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
