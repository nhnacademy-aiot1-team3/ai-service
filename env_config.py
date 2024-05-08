from dotenv import load_dotenv  # type: ignore
import os


def load_env_var():
    """
    .env 파일에서 환경 변수를 로드합니다.

    이 함수는 `dotenv` 모듈의 `load_dotenv` 함수를 사용하여 'parameters.env'라는 이름의 .env 파일에서 환경 변수를 로드합니다.
    그런 다음 데이터베이스 구성과 관련된 특정 환경 변수를 검색하고 이를 사전 형식으로 반환합니다.

    반환값:
        dict: 다음과 같은 환경 변수를 포함하는 사전입니다:
            - 'db_url': 데이터베이스의 URL입니다.
            - 'token': 데이터베이스 액세스에 필요한 토큰입니다.
            - 'org': 데이터베이스와 관련된 조직입니다.
            - 'bucket': 데이터베이스에서 사용되는 버킷 이름입니다.

    참고:
        .env 파일을 만들고 그 안에 필요한 환경 변수를 정의하십시오.
        이 파일의 이름은 'parameters.env'여야 합니다.
        필요한 환경 변수는 'DB_URL', 'DB_TOKEN', 'DB_ORG', 'DB_BUCKET'입니다.
        .env 파일은 Python 스크립트와 동일한 디렉토리에 있어야 합니다.
    """
    load_dotenv('parameters.env')
    env_var_list = {
        'db_url' : os.getenv('DB_URL'),
        'token' : os.getenv('DB_TOKEN'),
        'org' : os.getenv('DB_ORG'),
        'bucket' : os.getenv('DB_BUCKET')
    }
    return env_var_list