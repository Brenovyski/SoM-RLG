#in development skeleton only
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow.models import Variable
import os
import json
import requests
import logging
import time

# Default arguments for the DAG
default_args = {
    'owner': 'tominagabreno',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['suguru.ben@usp.br'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Instantiate the DAG
dag = DAG(
    'som_benchmarking',
    default_args=default_args,
    description='Benchmarking multiple models using SoM',
    schedule_interval=None,
    catchup=False,
)

# List of models to benchmark - only proprietary models here
MODEL_LIST = [
    'chatgpt-4o-latest',
    'gpt-4o-mini',
    'gpt-4-turbo',
    'gemini-exp-1121',
    'Gemini 1.5 Flash',
    'Gemini 1.5 Flash-8B',
    # 'Gemini 1.5 Pro',  # Uncomment if you have access
    # 'Claude 3.5 Sonnet',  # Uncomment if you have access
]

# List of purposes (tasks) - as following the benchmark proposed in github
PURPOSES = [
    'coco_ovseg',
    'ade20k_ovseg',
    'flickr30k_grounding',
    'refcocog_refseg',
]

# Paths to the datasets
DATASET_PATHS = {
    'coco_ovseg': '/path/to/coco_ovseg',
    'ade20k_ovseg': '/path/to/ade20k_ovseg',
    'flickr30k_grounding': '/path/to/flickr30k_grounding',
    'refcocog_refseg': '/path/to/refcocog_refseg',
}

# Function to load images for a given purpose
def load_images(purpose, **kwargs):
    images_dir = os.path.join(DATASET_PATHS[purpose], 'som_images')
    images = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith('.jpg')]
    # Push the list of images to XCom for downstream tasks
    kwargs['ti'].xcom_push(key=f'{purpose}_images', value=images)
    logging.info(f'Loaded {len(images)} images for purpose: {purpose}')

# Function to apply the Set of Marks algorithm
def apply_som(purpose, **kwargs):
    from som_algorithm import apply_som_to_image  # Your implemented SoM function
    ti = kwargs['ti']
    images = ti.xcom_pull(key=f'{purpose}_images', task_ids=f'load_images_{purpose}')
    som_images = []

    for image_path in images:
        som_image_path = apply_som_to_image(image_path)
        som_images.append(som_image_path)

    # Push the list of SoM images to XCom
    ti.xcom_push(key=f'{purpose}_som_images', value=som_images)
    logging.info(f'Applied SoM to {len(som_images)} images for purpose: {purpose}')

# Function to prepare prompts
def prepare_prompts(purpose, **kwargs):
    ti = kwargs['ti']
    som_images = ti.xcom_pull(key=f'{purpose}_som_images', task_ids=f'apply_som_{purpose}')
    prompts = []

    if purpose == 'coco_ovseg':
        vocabulary = load_coco_vocabulary()  # Implement this function to load COCO vocabulary
        prompt_template = f"I have labeled a bright numeric ID at the center for each visual object in the image. Please enumerate their names. You must answer by selecting from the following names: {vocabulary}"
        prompts = [prompt_template for _ in som_images]
    elif purpose == 'ade20k_ovseg':
        vocabulary = load_ade20k_vocabulary()  # Implement this function to load ADE20K vocabulary
        prompt_template = f"I have labeled a bright numeric ID at the center for each visual object in the image. Please enumerate their names. You must answer by selecting from the following names: {vocabulary}"
        prompts = [prompt_template for _ in som_images]
    elif purpose == 'flickr30k_grounding':
        prompts = []
        for image_path in som_images:
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            json_path = os.path.join(DATASET_PATHS[purpose], f'{image_id}.json')
            with open(json_path, 'r') as f:
                data = json.load(f)
                caption = data['caption']
                phrases = data['phrases']
                prompt = f"I have labeled a bright numeric ID at the center for each visual object in the image. Given the image showing {caption}, find the corresponding regions for {', '.join(phrases)}."
                prompts.append(prompt)
    elif purpose == 'refcocog_refseg':
        prompts = []
        for image_path in som_images:
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            json_path = os.path.join(DATASET_PATHS[purpose], f'{image_id}.json')
            with open(json_path, 'r') as f:
                data = json.load(f)
                expressions = data['expressions']
                prompt = f"I have labeled a bright numeric ID at the center for each visual object in the image. Please tell me the IDs for: {'; '.join(expressions)}."
                prompts.append(prompt)
    else:
        logging.error(f'Unknown purpose: {purpose}')

    ti.xcom_push(key=f'{purpose}_prompts', value=prompts)
    logging.info(f'Prepared {len(prompts)} prompts for purpose: {purpose}')

# Function to send requests to model APIs
def send_requests_to_models(purpose, **kwargs):
    ti = kwargs['ti']
    som_images = ti.xcom_pull(key=f'{purpose}_som_images', task_ids=f'apply_som_{purpose}')
    prompts = ti.xcom_pull(key=f'{purpose}_prompts', task_ids=f'prepare_prompts_{purpose}')

    results = []

    for model in MODEL_LIST:
        model_results = []
        for image_path, prompt in zip(som_images, prompts):
            response = send_request_to_model(model, image_path, prompt)
            model_results.append({
                'image': image_path,
                'prompt': prompt,
                'response': response
            })
            # Respect API rate limits if necessary
            time.sleep(5)  # Adjust sleep time as per API rate limits
        results.append({
            'model': model,
            'results': model_results
        })
        logging.info(f'Completed requests for model: {model} for purpose: {purpose}')

    ti.xcom_push(key=f'{purpose}_model_results', value=results)

# Function to store results
def store_results(purpose, **kwargs):
    ti = kwargs['ti']
    model_results = ti.xcom_pull(key=f'{purpose}_model_results', task_ids=f'send_requests_{purpose}')
    output_dir = '/path/to/output'  # Ensure this directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{purpose}_results.json')

    with open(output_file, 'w') as f:
        json.dump(model_results, f, indent=4)

    logging.info(f'Stored results for purpose: {purpose} in {output_file}')

# Function to send request to a model API (ensuring no context is maintained)
def send_request_to_model(model_name, image_path, prompt):
    # Load image data
    with open(image_path, 'rb') as f:
        image_data = f.read()

    # Prepare API request for different models
    if model_name.startswith('chatgpt') or model_name.startswith('gpt-4'):
        response = send_openai_request(model_name, image_data, prompt)
    elif model_name.startswith('gemini'):
        response = send_gemini_request(model_name, image_data, prompt)
    elif model_name.startswith('claude'):
        response = send_anthropic_request(model_name, image_data, prompt)
    else:
        logging.error(f'Unknown model: {model_name}')
        return None

    # Handle the response
    if response:
        return response
    else:
        logging.error(f'Failed to get response from model: {model_name}')
        return None

# Implement the API request functions for each model
def send_openai_request(model_name, image_data, prompt):
    # Securely retrieve your API key from Airflow Variables
    openai_api_key = Variable.get('OPENAI_API_KEY')
    headers = {
        'Authorization': f'Bearer {openai_api_key}',
    }

    # Construct the request payload
    # Note: As of my knowledge cutoff in September 2021, OpenAI's GPT models do not support image inputs via API.
    # Adjust the code based on the actual API specifications when available.

    data = {
        'model': model_name,
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        # Include image data if the API supports it
    }

    try:
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f'OpenAI API error: {response.status_code} - {response.text}')
            return None
    except Exception as e:
        logging.error(f'Exception when calling OpenAI API: {e}')
        return None

def send_gemini_request(model_name, image_data, prompt):
    # Implement the request to Google's Gemini API
    # Securely retrieve your API key from Airflow Variables
    gemini_api_key = Variable.get('GEMINI_API_KEY')
    headers = {
        'Authorization': f'Bearer {gemini_api_key}',
    }

    data = {
        'model': model_name,
        'prompt': prompt,
        # Include image data if the API supports it
    }

    # Replace 'https://gemini.googleapis.com/v1/models' with the actual endpoint
    try:
        response = requests.post('https://gemini.googleapis.com/v1/models', headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f'Gemini API error: {response.status_code} - {response.text}')
            return None
    except Exception as e:
        logging.error(f'Exception when calling Gemini API: {e}')
        return None

def send_anthropic_request(model_name, image_data, prompt):
    # Implement the request to Anthropic's Claude API
    # Securely retrieve your API key from Airflow Variables
    anthropic_api_key = Variable.get('ANTHROPIC_API_KEY')
    headers = {
        'x-api-key': anthropic_api_key,
    }

    data = {
        'model': model_name,
        'prompt': prompt,
        # Include image data if the API supports it
    }

    # Replace 'https://api.anthropic.com/v1/complete' with the actual endpoint
    try:
        response = requests.post('https://api.anthropic.com/v1/complete', headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f'Anthropic API error: {response.status_code} - {response.text}')
            return None
    except Exception as e:
        logging.error(f'Exception when calling Anthropic API: {e}')
        return None

# Implement functions to load vocabularies
def load_coco_vocabulary():
    # Implement this function to load the COCO vocabulary
    # For example, read from a file or define a list
    return '[COCO Vocabulary]'

def load_ade20k_vocabulary():
    # Implement this function to load the ADE20K vocabulary
    return '[ADE20K Vocabulary]'

# Create tasks dynamically for each purpose
for purpose in PURPOSES:
    load_images_task = PythonOperator(
        task_id=f'load_images_{purpose}',
        python_callable=load_images,
        op_kwargs={'purpose': purpose},
        dag=dag,
    )

    apply_som_task = PythonOperator(
        task_id=f'apply_som_{purpose}',
        python_callable=apply_som,
        op_kwargs={'purpose': purpose},
        dag=dag,
    )

    prepare_prompts_task = PythonOperator(
        task_id=f'prepare_prompts_{purpose}',
        python_callable=prepare_prompts,
        op_kwargs={'purpose': purpose},
        dag=dag,
    )

    send_requests_task = PythonOperator(
        task_id=f'send_requests_{purpose}',
        python_callable=send_requests_to_models,
        op_kwargs={'purpose': purpose},
        dag=dag,
    )

    store_results_task = PythonOperator(
        task_id=f'store_results_{purpose}',
        python_callable=store_results,
        op_kwargs={'purpose': purpose},
        dag=dag,
    )

    # Set task dependencies
    load_images_task >> apply_som_task >> prepare_prompts_task >> send_requests_task >> store_results_task