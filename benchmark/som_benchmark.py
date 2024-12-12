import os
import json
import csv
import time
from PIL import Image

from gpt4v import request_gpt4v
from gemini import request_genai_model

OUTPUT_DIR = 'results/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_coco_vocabulary():
    return ", ".join([
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ])

def load_ade20k_vocabulary():
    print("Loading ADE20K vocabulary.")
    # Assuming the vocabulary is defined somewhere
    return ", ".join([
        # List of object categories for ADE20K
    ])

def build_flickr30k_prompts(image_data_list, dataset_path):
    print("Building prompts for Flickr30K.")
    prompts = []
    for data in image_data_list:
        json_path = data['json_path']
        with open(json_path, 'r') as f:
            data_json = json.load(f)
            caption = data_json.get('caption', '')
            phrases = data_json.get('phrases', [])
            prompt = f"I have labeled a bright numeric ID at the center for each visual object in the image. Given the image showing {caption}, find the corresponding regions for {', '.join(phrases)}."
            prompts.append(prompt)
    print(f"Built {len(prompts)} prompts for Flickr30K.")
    return prompts

def build_refcocog_prompts_seg(image_data_list, dataset_path):
    print("Building segmentation prompts for RefCOCOg.")
    prompts = []
    for data in image_data_list:
        json_path = data['json_path']
        with open(json_path, 'r') as f:
            data_json = json.load(f)
            expressions = data_json.get('expressions', [])
            prompt = f"I have labeled a bright numeric ID at the center for each visual object in the image. Please tell me the IDs for: {'; '.join(expressions)}."
            prompts.append(prompt)
    print(f"Built {len(prompts)} segmentation prompts for RefCOCOg.")
    return prompts

def build_refcocog_prompts_comp(image_data_list, dataset_path):
    print("Building comprehension prompts for RefCOCOg.")
    prompts = []
    for data in image_data_list:
        json_path = data['json_path']
        with open(json_path, 'r') as f:
            data_json = json.load(f)
            expressions = data_json.get('expressions', [])
            prompt = f"I have labeled a bright numeric ID at the center for each visual object in the image. Which one corresponds to the expression(s): {'; '.join(expressions)}?"
            prompts.append(prompt)
    print(f"Built {len(prompts)} comprehension prompts for RefCOCOg.")
    return prompts

BENCHMARKS = {
    'open_vocab_seg_coco': {
        'purpose': 'coco_ovseg',
        'dataset_name': 'COCO',
        'dataset_path': 'datasets/coco_ovseg',
        'prompt_builder': lambda data_list: [
            f'''I have labeled a bright numeric ID at the center for each visual object in the image. Please enumerate their names. You must answer by selecting from the following names: {load_coco_vocabulary()},
            if the name are not in the vocabuary, please try do infer what it is. For any marks mentioned in your answer, please highlight them with [], write in only 1 line'''
            for _ in data_list
        ],
        'additional_files': []
    },
    # 'open_vocab_seg_ade20k': {
    #     'purpose': 'ade20k_ovseg',
    #     'dataset_name': 'ADE20K',
    #     'dataset_path': 'datasets/ade20k_ovseg',
    #     'prompt_builder': lambda data_list: [
    #         f"I have labeled a bright numeric ID at the center for each visual object in the image. Please enumerate their names. You must answer by selecting from the following names: {load_ade20k_vocabulary()}"
    #         for _ in data_list
    #     ],
    #     'additional_files': []
    # },
    #phrase_grounding_flickr30k': {
    #   'purpose': 'flickr30k_grounding',
    #   'dataset_name': 'Flickr30K',
    #   'dataset_path': 'datasets/flickr30k_grounding',
    #   'prompt_builder': lambda data_list: build_flickr30k_prompts(data_list, 'datasets/flickr30k_grounding'),
    #   'additional_files': ['json', 'wbox.jpg']
    #,
    #refcocog_refseg_seg': {
    #   'purpose': 'refcocog_refseg',
    #   'dataset_name': 'RefCOCOg',
    #   'dataset_path': 'datasets/refcocog_refseg',
    #   'prompt_builder': lambda data_list: build_refcocog_prompts_seg(data_list, 'datasets/refcocog_refseg'),
    #   'additional_files': ['json']
    #,
    #refcocog_refseg_comp': {
    #   'purpose': 'refcocog_refseg',
    #   'dataset_name': 'RefCOCOg',
    #   'dataset_path': 'datasets/refcocog_refseg',
    #   'prompt_builder': lambda data_list: build_refcocog_prompts_comp(data_list, 'datasets/refcocog_refseg'),
    #   'additional_files': ['json']
    #,
}

MODEL_LIST = [
    # GPT-4 models
    'gpt-4o',
    #'gemini-1.5-flash',
    'gemini-1.5-pro',
    'gemini-2.0-flash-exp'
]

def load_images(benchmark_key):
    print(f"Loading images for benchmark: {benchmark_key}")
    benchmark = BENCHMARKS[benchmark_key]
    images_dir = os.path.join(benchmark['dataset_path'], 'som_images')
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    additional_files = benchmark.get('additional_files', [])

    image_data_list = []

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        base_name, _ = os.path.splitext(image_file)
        data_entry = {'image_path': image_path}

        for ext in additional_files:
            if ext == 'json':
                file_path = os.path.join(benchmark['dataset_path'], f'{base_name}.json')
                if os.path.exists(file_path):
                    data_entry['json_path'] = file_path
            else:
                file_path = os.path.join(images_dir, f'{base_name}_{ext}')
                if os.path.exists(file_path):
                    data_entry[f'{ext}_path'] = file_path

        image_data_list.append(data_entry)

    print(f"Loaded {len(image_data_list)} images and additional data for benchmark: {benchmark_key}")
    return image_data_list

def prepare_prompts(benchmark_key, image_data_list):
    print(f"Preparing prompts for benchmark: {benchmark_key}")
    benchmark = BENCHMARKS[benchmark_key]
    prompts = benchmark['prompt_builder'](image_data_list)
    print(f"Prepared {len(prompts)} prompts for benchmark: {benchmark_key}")
    return prompts

def send_requests_to_models(benchmark_key, image_data_list, prompts):
    print(f"Sending requests to models for benchmark: {benchmark_key}")
    results = []

    for idx, (data_entry, prompt) in enumerate(zip(image_data_list, prompts), 1):
        image_path = data_entry['image_path']
        pil_image = Image.open(image_path)
        files = []

        # Collect additional files if any
        if 'json_path' in data_entry:
            files.append(data_entry['json_path'])
        if 'wbox.jpg_path' in data_entry:
            files.append(data_entry['wbox.jpg_path'])

        print(f"Processing image {idx}/{len(image_data_list)}: {os.path.basename(image_path)}")

        for model_idx, model_name in enumerate(MODEL_LIST, 1):
            print(f"  Sending request to model {model_idx}/{len(MODEL_LIST)}: {model_name}")
            if model_name.startswith('chatgpt') or model_name.startswith('gpt-4'):
                try:
                    response_text = request_gpt4v(prompt, model_name=model_name, pil_image=pil_image, files=files)
                except Exception as e:
                    print(f"    ERROR: Failed to get response from {model_name}. Exception: {e}")
                    response_text = f"Error: {e}"
            elif model_name.lower().startswith('gemini'):
                try:
                    response_text = request_genai_model(model_name, prompt, pil_image=pil_image, files=files)
                except Exception as e:
                    print(f"    ERROR: Failed to get response from {model_name}. Exception: {e}")
                    response_text = f"Error: {e}"
            else:
                print(f"    ERROR: Unknown model: {model_name}")
                response_text = "Error: Unknown model"

            # Print response for each model
            print(f"    Response from {model_name}: {response_text[:200]}")  # Print first 200 characters

            results.append({
                'image': os.path.basename(image_path),
                'model': model_name,
                'prompt': prompt,
                'response': response_text
            })

            time.sleep(1)  # Adjust time if necessary

    print(f"Completed requests for benchmark: {benchmark_key}")
    return results

def store_results(benchmark_key, model_results):
    print(f"Storing results for benchmark: {benchmark_key}")
    output_file = os.path.join(OUTPUT_DIR, f'{benchmark_key}_results.csv')

    fieldnames = ['image', 'model', 'prompt', 'response']
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in model_results:
            writer.writerow(res)

    print(f'Stored results for benchmark: {benchmark_key} in {output_file}')

def main():
    print("Starting SoM Benchmark")
    for benchmark_idx, benchmark_key in enumerate(BENCHMARKS.keys(), 1):
        print(f'\nStarting benchmark {benchmark_idx}/{len(BENCHMARKS)}: {benchmark_key}')

        # Load images and additional files
        image_data_list = load_images(benchmark_key)

        # Prepare prompts
        prompts = prepare_prompts(benchmark_key, image_data_list)

        # Send requests to models
        model_results = send_requests_to_models(benchmark_key, image_data_list, prompts)

        # Store results
        store_results(benchmark_key, model_results)

        print(f'Completed benchmark: {benchmark_key}')
    print("All benchmarks completed.")

if __name__ == "__main__":
    main()