from env_config import load_env_var
from dataset_manager import DatasetManager

# Main
def main():
    env_var_list = load_env_var()
    dataset_manager = DatasetManager(env_var_list['db_url'], env_var_list['token'], env_var_list['org'], env_var_list['bucket'])
    result_df = dataset_manager.query_temperature('gyeongnam')
    dataset_manager.data_preprocessing(result_df)
    
    dataset_manager.close_connection()

if __name__ == '__main__':
    main()