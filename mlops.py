
import os
import copy
import subprocess
import json
import boto3
import mlflow
import mlflow.sagemaker as mfs
import datetime


def get_account_id():
    try:
        client = boto3.session.Session().client("sts")
        caller_id = client.get_caller_identity()
        account_id = caller_id['Account']
        return account_id
    except Exception as e:
        print(f"Failed to get account id from assume role")
        print(e)
        return None


def get_region():
    try:
        session = boto3.session.Session()
        region = session.region_name or 'us-east-1'
        return region
    except Exception as e:
        print(f"Failed to get current region from server")
        print(e)
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


def convert_to_date_time(mlflow_epoc):
    dt = datetime.datetime.fromtimestamp(mlflow_epoc / 1000)
    date = f'{dt:%Y-%m-%d}'
    time = f'{dt:%H:%M:%S%z}'
    return date, time


def get_mlflow_model_details(run_id, tracking_uri):
    run_info = {}
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    # Get run from run ID of MlFlow model
    model_run = client.get_run(run_id)

    experiment_id = model_run.info.experiment_id
    run_info["experiment_id"] = experiment_id
    run_info["life_cycle_stage"] = model_run.info.lifecycle_stage
    run_info["status"] = model_run.info.status
    run_info["artifact_uri"] = model_run.info.artifact_uri

    start_time = model_run.info.start_time
    end_time = model_run.info.end_time

    if start_time:
        start_date, start_time = convert_to_date_time(mlflow_epoc=start_time)
        run_info["start_date"] = start_date
        run_info["start_time"] = start_time

    if end_time:
        end_date, end_time = convert_to_date_time(mlflow_epoc=end_time)
        run_info["end_date"] = end_date
        run_info["end_time"] = end_time

    run_info["accuracy_score"] = model_run.data.metrics['accuracy score']
    run_info["average_precision"] = model_run.data.metrics['average precision']
    run_info["f1_score"] = model_run.data.metrics['f1-score']

    git_sha = model_run.data.tags['mlflow.source.git.commit']
    run_info["git_sha"] = git_sha
    if git_sha:
        run_info["git_sha"] = git_sha[:8]

    # Get experiment from experiment id
    experiment = client.get_experiment(experiment_id=experiment_id)

    run_info["experiment_name"] = experiment.name

    return run_info


def deploy_model(
    app_name,
    run_id,
    bucket,
    tracking_uri,
    region=get_region(),
    execution_role_name="mlflow_sagemaker",
    sagemaker_instance_type=mfs.DEFAULT_SAGEMAKER_INSTANCE_TYPE,
    sagemaker_instance_count=mfs.DEFAULT_SAGEMAKER_INSTANCE_COUNT,
    sagemaker_model_mode=mfs.DEPLOYMENT_MODE_CREATE
):
    # Verify Model before deploy
    status = check_sagemaker_endpoint_status(app_name=app_name)
    if status and sagemaker_model_mode is mfs.DEPLOYMENT_MODE_CREATE:
        msg = f"AWS Sagemaker endpoint {app_name} exist."
        print(msg)
        raise ValueError(msg)

    run_info = get_mlflow_model_details(run_id, tracking_uri)

    image_ecr_url = get_ecr_url(region=region)
    execution_role_arn = get_sagemager_execution_roler_arn(execution_role_name)
    model_uri = run_info["artifact_uri"] + "/model"

    print(
        f"Model URI is '{model_uri}' for runid '{run_id}' of MlFLow server '{tracking_uri}'")

    mfs.deploy(app_name=app_name, execution_role_arn=execution_role_arn, bucket=bucket, model_uri=model_uri, image_url=image_ecr_url,
               region_name=region, mode=sagemaker_model_mode, instance_type=sagemaker_instance_type, instance_count=sagemaker_instance_count)

    run_info['end_point'] = app_name
    print("Model is deployed successfuly on AWS Sagemaker")
    print(f"Model information =>\n{run_info}")


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


def get_test_data(file_name):
    data = None
    if (os.path.exists(file_name)):
        with open(file_name, 'r') as f:
            data = f.read()
        if data:
            data = data.encode("utf-8")
        return data
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


def write_prediction_output(file_name, prediction):
    base_dir = os.path.dirname(file_name)

    if (os.path.exists(base_dir)):
        with open(file_name, 'w') as f:
            f.write(str(prediction))
    else:
        print(
            f"Prediction folder not exist : {file_name}; Not saving the file")


if __name__ == "__main__":
    # Create or verify the repository in AWS ECR
    push_image_to_sagemaker()

    app_name = os.environ['APP_NAME']
    run_id = os.environ['RUN_ID']
    bucket = os.environ['BUCKET']
    tracking_uri = os.environ['TRACKING_URI']
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
        run_id=run_id,
        bucket=bucket,
        tracking_uri=tracking_uri)

    # Verify Model status
    status = str(check_sagemaker_endpoint_status(app_name=app_name))
    print(f"Model {app_name} Endpoint status is '{status}'")

    if "InService" in status:
        # Infer model here
        input_json = get_test_data(input_test_file)
        if input_json:
            print(f"********************************")
            print(f"Input test json data received {input_json}")
            print(f"********************************")
            prediction = inferance_sagemaker_endpoint(
                app_name=app_name, input_json=input_json)
            print(f"Received prediction response: {prediction}")

            base_dir = os.path.dirname(input_test_file)
            pred_file = os.path.join(base_dir, f"inference-{app_name}.pred")
            write_prediction_output(pred_file, prediction)
        else:
            msg = f"Failed to get test json data. Received str({input_json})"
            print(msg)
    else:
        msg = f"AWS Sagemaker endpoint is InService. Current status is {status}"
        print(msg)
        raise ValueError(msg)

    # Delete model
    destroy_model(app_name=app_name)
