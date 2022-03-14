import time
import yaml
import os
import shutil
import boto3
import gdown
import tarfile
import logging as log
from botocore.exceptions import ClientError
from .error import AWSCredentialError, DownloadDataError

DOWNLOAD_ERROR = f"""
Failed to download. Please download from folowing link in data/raw folder.
    https://drive.google.com/uc?id=1-021ruCLpzp2tH5hU4PFm0r0AJBiulQU

if you are using Colad try:
    `!gdown --id 1-021ruCLpzp2tH5hU4PFm0r0AJBiulQU --output data/raw` 
                        or 
    `!gdown --id 1I1LR7XjyEZ-VBQ-Xruh31V7xExMjlVvi --output data/raw`

For extracting:
    ```python
    from src.utils.utils import extract_data
    extract_data(os.path.join("data", "raw", "Task06_Lung.tar"))
    ```
"""

def read_yaml(file:str) -> dict:
    with open(file, 'r') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def write_yaml(data:dict, file:str) -> None:
    """
    A function to write YAML file
    :param data: data to write
    :param file: file to write
    """
    with open(file, 'w') as f:
        yaml.dump(data, f)

def get_value(file:str, key:str) -> str:
    """
    A function to get value from yaml file
    :param file: yaml file
    :param key: key to get value

    Example:
    ```yaml
    MLFLOW_TRACKING_URI: abcd
    DATABASE:
        HOST: abcd
        USER: abcd
        PASSWORD: abcd
    ```
    >>> get_value("config.yaml", "MLFLOW_TRACKING_URI")
    abcd
    >>> get_value("config.yaml", "DATABASE.HOST")
    abcd
    """
    file_path = os.path.join(os.getcwd(), file)
    data = read_yaml(file_path)
    for k in key.split("."):
        data = data[k]
    return data

def get_public_dns(instance_id:str=None) -> str:
    """
    A function to get public DNS for an instance
    :param instance_id: instance id
    """
    ec2_client = boto3.client("ec2", region_name="ap-south-1")
    reservations = ec2_client.describe_instances(InstanceIds=[instance_id]).get("Reservations")

    ips = []
    for reservation in reservations:
        for instance in reservation['Instances']:
            if instance["State"]["Name"] == "running":
                ips.append(instance.get("PublicDnsName"))
            elif instance["State"]["Name"] == "pending":
                log.info("Instance is not running yet. Please wait for a while tryig again.")
                time.sleep(20)
                ips.append(get_public_dns(instance_id))
            else:
                log.warning(f"Instance {instance_id} is not running.")
                log.info(f"Starting instance with id={instance_id}")
                ec2_client.start_instances(InstanceIds=[instance_id])
                time.sleep(10)
                ips.append(get_public_dns(instance_id))
    return ips[0]

def check_aws_credential() -> None:
    """
    A function to check if AWS credential is set
    """

    sts = boto3.client('sts')
    try:
        sts.get_caller_identity()
    except ClientError:
        log.error("AWS Credentials are not valid\nTry to run 'aws configure' to set them.")
        raise AWSCredentialError("AWS Credentials are not valid. Please run 'aws configure' to set them.")

def create_dir() -> None:
    """
    A function to setup the Directory
    """
    # Checking for folder
    dir = ["data", os.path.join("data", "raw"), os.path.join("data", "processed"), os.path.join("data", "processed", "train"), 
           os.path.join("data", "interim"), "logs", "models", "reports"]
    log.info("Creating directories.")
    for path in dir:
        os.makedirs(path, exist_ok=True)

def move_file(source, dest) -> None:
    """
    A function to move file from source to dest
    :param source: source file
    :param dest: destination file

    """
    allfiles = os.listdir(source)
    log.info(f"Moving all files from {source} to {dest}")
    for f in allfiles:
        try:
            shutil.move(os.path.join(source, f), os.path.join(dest, f))
        except FileNotFoundError:
            continue
    os.rmdir(source)
    f.write("move")
            

def extract_data(file_path) -> None:
    """
    A function to extract data from tar file

    :param file_path: path to tar file
    """
    log.info("Extracting data from tar file")
    my_tar = tarfile.open(file_path)
    my_tar.extractall(os.path.join("data", "raw"))
    my_tar.close()
    with open(os.path.join("data", "raw", "status"), "w") as f:
        f.write("extract")
    
def download_data() -> None:
    """
    A function to download data from Google Drive.
    """
    ids = ["1I1LR7XjyEZ-VBQ-Xruh31V7xExMjlVvi", "1-021ruCLpzp2tH5hU4PFm0r0AJBiulQU"]
    idx = 0
    tried = 0
    output = os.path.join("data", "raw", "Task06_Lung.tar")
    while True:
        if os.path.exists(output):
            with open(os.path.join("data", "raw", "status"), "w") as f:
                f.write("down")
            break
        else:
            log.info(f"Downloading data from Drive with id = {ids} tries = {tried+1}")
            gdown.download(id=ids[idx], output=output, quiet=False)
            tried += 1

        if tried == 3:
            idx = 1
            log.warning(f"Changing id to {ids[idx]}.")
            time.sleep(5)
        if tried == 6:
            log.error("Failed to download data from Drive.")
            break

    if os.path.exists(output):
        log.info("Data is been downloaded successfully.")
    else:
        log.error(DOWNLOAD_ERROR)
        raise DownloadDataError(DOWNLOAD_ERROR)

def setup() -> None:
    """
    A function to setup the project
    """
    # Creating directories
    create_dir()

    # Downloading data
    with open(os.path.join("data", "raw", "status"), "r") as f:
        status = f.read()
        if status == "":
            download_data()

    # Extracting data
    with open(os.path.join("data", "raw", "status"), "r") as f:
        status = f.read()
        if status == "down":
            extract_data(os.path.join("data", "raw", "Task06_Lung.tar"))

    # Moving files
    with open(os.path.join("data", "raw", "status"), "r") as f:
        status = f.read()
        if status == "extract":
            move_file(os.path.join("data", "raw", "Task06_Lung"), os.path.join("data", "raw"))
            os.remove(os.path.join("data", "raw", "Task06_Lung.tar"))

    # Checking AWS credential
    check_aws_credential()

    # Geting public DNS for mlflow server
    dns = get_public_dns(instance_id="i-0f7a0f90a77792a82")
    log.info(f"Public DNS for mlflow server: {dns}")
    yaml_dict = read_yaml("config.yaml")
    yaml_dict["MLFLOW_TRACKING_URI"] = f"http://{dns}:8000"

    log.info(f"Updating config.yaml with MLFLOW_TRACKING_URI: {yaml_dict['MLFLOW_TRACKING_URI']}")
    write_yaml(yaml_dict, "config.yaml")

    log.info("Setup complete")
