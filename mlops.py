
import os
import copy
import subprocess
import json
import boto3
import mlflow
import mlflow.sagemaker as mfs


def get_account_id():
    try:
        client = boto3.session.Session().client("sts")
        caller_id = client.get_caller_identity()
        account_id = caller_id['Account']
        return account_id
    except Exception as e:
        print("Failed to get account id from assume role")
        return None


def get_region():
    try:
        session = boto3.session.Session()
        region = session.region_name or 'us-east-1'
        return region
    except Exception as e:
        print("Failed to get current region from server")
        return None


def create_repository(repository_name="mlflow-pyfunc", region_name=get_region()):
    client = boto3.session.Session().client('ecr', region_name=region_name)
    try:
        response = client.describe_repositories(
            repositoryNames=[repository_name])
        print(f"Repository '{repository_name}' already exist.")
    except client.exceptions.RepositoryNotFoundException:
        print(
            f"Repository '{repository_name}' not found, will create a new named repository")
        try:
            tags = [
                {
                    "Key": "Name",
                    "Value": repository_name
                },
                {
                    "Key": "Description",
                    "Value": "MlFlow automation repo"
                }
            ]
            encryptionConfiguration = {"encryptionType": "AES256"}
            response = client.create_repository(
                repositoryName=repository_name, tags=tags, encryptionConfiguration=encryptionConfiguration)
            print(f"Repository '{repository_name}' created")
        except Exception as e:
            raise e
    except Exception as e:
        raise e


def get_ecr_url(image_name="mlflow-pyfunc", region=get_region()):
    _full_template = "{account}.dkr.ecr.{region}.amazonaws.com/{image}:{version}"
    account_id = get_account_id()
    image_ecr_url = _full_template.format(
        account=account_id,
        region=region,
        image=image_name,
        version=mlflow.version.VERSION
    )
    return image_ecr_url


def check_sagemaker_endpoint_status(app_name, region=get_region()):
    try:
        sage_client = boto3.session.Session().client('sagemaker', region_name=region)
        endpoint_description = sage_client.describe_endpoint(
            EndpointName=app_name)
        endpoint_status = endpoint_description["EndpointStatus"]
        return endpoint_status
    except Exception as e:
        if "Could not find endpoint" in e.response["Error"]["Message"]:
            return None
        else:
            print("rain")
            raise e


def execute_shell(command, wait=True, current_env=True, working_dir=None):
    env = None
    stdout = None
    if current_env:
        env = copy.deepcopy(os.environ)

    process = subprocess.Popen(
        command, shell=True, stdout=stdout, env=env, cwd=working_dir)
    if wait:
        process.wait()
        return process.returncode
    return 0


def create_mlflow_base_docker_image(image_name="mlflow-pyfunc"):
    command = f"mlflow sagemaker build-and-push-container --no-push --container {image_name}"
    status_code = execute_shell(command=command)
    if status_code != 0:
        mgs = f"Failed building the image"
        print(msg)
        raise ValueError(msg)


def push_mlflow_base_docker_image(image_name="mlflow-pyfunc"):
    mfs.push_image_to_ecr(image=image_name)


def get_sagemager_execution_roler_arn(execution_role_name="mlflow_sagemaker"):
    account_id = get_account_id()
    execution_role_arn = f"arn:aws:iam::{account_id}:role/{execution_role_name}"
    return execution_role_arn


def deploy_model(
    app_name,
    experiment_id,
    run_id,
    bucket,
    region=get_region(),
    execution_role_name="mlflow_sagemaker",
    sagemaker_instance_type=mfs.DEFAULT_SAGEMAKER_INSTANCE_TYPE,
    sagemaker_instance_count=mfs.DEFAULT_SAGEMAKER_INSTANCE_COUNT,
    sagemaker_model_mode=mfs.DEPLOYMENT_MODE_CREATE,
    artifact_root=None
):
    # Verify Model before deploy
    status = check_sagemaker_endpoint_status(app_name=app_name)
    if status and sagemaker_model_mode is mfs.DEPLOYMENT_MODE_CREATE:
        msg = f"AWS Sagemaker endpoint {app_name} exist."
        print(msg)
        raise ValueError(msg)

    if not artifact_root:
        artifact_root = f"s3://{bucket}/prod/mlflow"

    image_ecr_url = get_ecr_url(region=region)
    execution_role_arn = get_sagemager_execution_roler_arn(execution_role_name)
    model_uri = f"{artifact_root}/{experiment_id}/{run_id}/artifacts/model"

    mfs.deploy(app_name=app_name, execution_role_arn=execution_role_arn, bucket=bucket, model_uri=model_uri, image_url=image_ecr_url,
               region_name=region, mode=sagemaker_model_mode, instance_type=sagemaker_instance_type, instance_count=sagemaker_instance_count)


def destroy_model(app_name, region=get_region()):
    # Verify Model before deploy
    status = check_sagemaker_endpoint_status(app_name=app_name)
    if status:
        mfs.delete(app_name=app_name, region_name=region, archive=False)
    else:
        msg = f"Model {app_name} Endpoint not present"
        print(msg)
        raise ValueError(msg)


def get_sagemaker_active_endpoints(region=get_region()):
    sage_client = boto3.session.Session().client('sagemaker', region_name=region)
    app_endpoints = sage_client.list_endpoints()["Endpoints"]
    return app_endpoints


def inferance_sagemaker_endpoint(app_name, input_json, format="pandas-split", region=get_region()):
    client = boto3.session.Session().client("sagemaker-runtime", region)

    response = client.invoke_endpoint(
        EndpointName=app_name,
        Body=input_json,
        ContentType='application/json; format=pandas-split',
    )
    preds = response["Body"].read().decode("ascii")
    preds = json.loads(preds)
    return preds


def get_json_data(file_name):
    data = None
    if (os.path.exists(file_name)):
        with open(file_name, 'r') as f:
            data = f.read()
        return json.loads(data)
    else:
        print(f"Input file does not exists : {file_name}")
    return None


def push_image_to_sagemaker(
    repository_name="mlflow-pyfunc",
    image_name="mlflow-pyfunc"
):
    # Create or verify the repository in AWS ECR
    create_repository(repository_name=repository_name)

    # Create base docker image for AWS Sagemaker
    create_mlflow_base_docker_image(image_name=image_name)

    # Push base docker image to AWS ECR
    push_mlflow_base_docker_image(image_name=image_name)


if __name__ == "__main__":
    # Create or verify the repository in AWS ECR
    push_image_to_sagemaker()

    app_name = os.environ['APP_NAME']
    experiment_id = os.environ['EXPERIMENT_ID']
    run_id = os.environ['RUN_ID']
    bucket = os.environ['BUCKET']
    input_test_file = os.environ['INPUT_TEST_FILE']

    # Verify Model before deploy
    status = check_sagemaker_endpoint_status(app_name=app_name)
    if status:
        msg = f"Model {app_name} Endpoint name already exist"
        print(msg)
        raise ValueError(msg)

    # Deploy model
    deploy_model(
        app_name=app_name,
        experiment_id=experiment_id,
        run_id=run_id,
        bucket=bucket)

    # Verify Model status
    status = str(check_sagemaker_endpoint_status(app_name=app_name))
    print(f"Model {app_name} Endpoint status is '{status}'")

    if "InService" in status:
        # Infer model here
        input_json = get_json_data(input_test_file)
        if input_json:
            print(f"********************************")
            print(f"Input test json data received {input_json}")
            print(f"********************************")
            prediction = inferance_sagemaker_endpoint(
                app_name=app_name, input_json=input_json)
            print(f"Received prediction response: {prediction}")
        else:
            msg = f"Failed to get test json data. Received str({input_json})"
            print(msg)
    else:
        msg = f"AWS Sagemaker endpoint is InService. Current status is {status}"
        print(msg)

    # Delete model
    destroy_model(app_name=app_name)
