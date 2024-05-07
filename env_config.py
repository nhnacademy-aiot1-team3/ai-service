from dotenv import load_dotenv
import os

def load_env_var():
    load_dotenv('parameters.env')
    env_var_list = {
        'db_url' : os.getenv('DB_URL'),
        'token' : os.getenv('DB_TOKEN'),
        'org' : os.getenv('DB_ORG'),
        'bucket' : os.getenv('DB_BUCKET')
    }
    return env_var_list