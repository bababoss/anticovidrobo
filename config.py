import os
from sqlalchemy_utils.functions import database_exists, create_database
basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SECRET_KEY = 'change-this-when-you-are-ready'
    #SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']  > use this in case env is set for URI
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + basedir + '/covidinfo.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    if not database_exists(SQLALCHEMY_DATABASE_URI):
        create_database(SQLALCHEMY_DATABASE_URI)


class Production(Config):
    DEBUG = False

class Stage(Config):
    DEVELOPMENT = True
    DEBUG = True

class Development(Config):
    DEVELOPMENT = True
    DEBUG = True

class Testing(Config):
    TESTING = True


#set the environemnt variable as per the stage for e.g for dev set this : export APP_SETTINGS="config.Development"