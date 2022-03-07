import time
import yaml
import os
import boto3
import gdown
import tarfile
import logging as log
from botocore.exceptions import ClientError
from .error import AWSCredentialError

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
    
def download_data() -> None:
    """
    A function to download data from Google Drive.
    """
    id = "1I1LR7XjyEZ-VBQ-Xruh31V7xExMjlVvi"
    output = os.path.join("data", "raw", "Task06_Lung.tar")
    gdown.download(id=id, output=output, quiet=False, resume=True)
    log.info("Downloading data from Drive")
    try:
        log.info("Extracting data from tar file")
        my_tar = tarfile.open(output)
        my_tar.extractall(os.path.join("data", "raw"))
        my_tar.close()
    except FileNotFoundError:
        link = "https://drive.google.com/uc?id=1-021ruCLpzp2tH5hU4PFm0r0AJBiulQU"
        log.error("Not able to download.")
        print(f"Not able to download.\nPlease download from folowing link and extract it in data/raw folder.\n{link}")

def setup():
    """
    A function to setup the project
    """
    # Creating directories
    create_dir()

    # Downloading data
    download_data()

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