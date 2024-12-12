# Understanding Instances and Metrics in the Benchmark

In this document, we provide detailed explanations of the "Number of Instances" and the evaluation metrics used in the benchmark. This will help in understanding how to verify the results manually after the benchmark is completed.

---

## Table of Contents

1. [Understanding the Number of Instances](#understanding-the-number-of-instances)
   - [1. Open-Vocabulary Segmentation on COCO](#1-open-vocabulary-segmentation-on-coco)
   - [2. Open-Vocabulary Segmentation on ADE20K](#2-open-vocabulary-segmentation-on-ade20k)
   - [3. Phrase Grounding on Flickr30K](#3-phrase-grounding-on-flickr30k)
   - [4. Referring Expression Comprehension and Segmentation on RefCOCOg](#4-referring-expression-comprehension-and-segmentation-on-refcocog)
2. [Explanation of Metrics and Manual Verification](#explanation-of-metrics-and-manual-verification)
   - [1. Precision](#1-precision)
   - [2. Recall @ 1](#2-recall--1)
   - [3. Accuracy @ 0.5 (ACC @ 0.5)](#3-accuracy--05-acc--05)
   - [4. Mean Intersection over Union (mIoU)](#4-mean-intersection-over-union-miou)
3. [General Steps to Verify Benchmark Results Manually](#general-steps-to-verify-benchmark-results-manually)
4. [Additional Notes](#additional-notes)
5. [Summary](#summary)

---

## Understanding the Number of Instances

In the benchmark, the **"Number of Instances"** refers to the total count of individual elements—such as objects, phrases, or regions—that are annotated and need to be identified, segmented, or grounded by the models being evaluated.

### 1. Open-Vocabulary Segmentation on COCO

- **Dataset**: [COCO](https://cocodataset.org/#home)
- **Number of Images**: 100
- **Number of Instances**: 567
- **Marks**: Numeric IDs and Masks
- **Metric**: Precision

#### Explanation

- **Instances Defined**: Each individual object within an image annotated with a segmentation mask.
- **Reason for 567 Instances**: Across 100 images, there are a total of 567 annotated objects. This averages to about 5-6 objects per image.

#### Example

An image depicting a street scene may contain:

- 3 cars
- 2 pedestrians
- 1 bicycle

This results in **6 instances** in that image.

---

### 2. Open-Vocabulary Segmentation on ADE20K

- **Dataset**: [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
- **Number of Images**: 100
- **Number of Instances**: 488
- **Marks**: Numeric IDs and Masks
- **Metric**: Precision

#### Explanation

- **Instances Defined**: Individual objects or regions annotated with segmentation masks.
- **Reason for 488 Instances**: The 100 images contain a total of 488 annotated objects or regions.

#### Example

An image of a room interior may include:

- 2 chairs
- 1 table
- 1 lamp
- 1 rug

This results in **5 instances** in that image.

---

### 3. Phrase Grounding on Flickr30K

- **Dataset**: [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/)
- **Number of Images**: 100
- **Number of Instances**: 274
- **Marks**: Numeric IDs, Masks, and Boxes
- **Metric**: Recall @ 1

#### Explanation

- **Instances Defined**: Noun phrases from image captions corresponding to specific regions.
- **Reason for 274 Instances**: Across 100 images, there are 274 phrases to ground, averaging about 2-3 phrases per image.

#### Example

Image Caption: "A woman holding a red umbrella stands next to a dog."

Instances:

- "woman"
- "red umbrella"
- "dog"

---

### 4. Referring Expression Comprehension and Segmentation on RefCOCOg

- **Dataset**: [RefCOCOg](https://github.com/lichengunc/refer)
- **Number of Images**: 100
- **Number of Instances**: 177
- **Marks**: Numeric IDs and Masks
- **Metrics**:
  - **Referring Comprehension**: ACC @ 0.5
  - **Referring Segmentation**: mIoU

#### Explanation

- **Instances Defined**: Referring expressions that uniquely identify objects in the images.
- **Reason for 177 Instances**: The dataset includes 177 referring expressions across 100 images.

#### Example

Referring Expressions:

- "The laptop behind the beer bottle"
- "Laptop turned on"

---

### 5. Video Object Segmentation on DAVIS

- **Dataset**: [DAVIS](https://davischallenge.org/)
- **Number of Videos**: 71
- **Number of Instances**: 157
- **Marks**: Numeric IDs and Masks
- **Metric**: J&F Measure

#### Explanation

- **Instances Defined**: Objects that need to be segmented throughout the video frames.
- **Reason for 157 Instances**: Across 71 videos, there are 157 objects to track and segment.

#### Example

A video showing a person skiing with a dog following includes instances:

- "Person"
- "Dog"

---

## Explanation of Metrics and Manual Verification

Below are detailed explanations of each metric used in the benchmark and how to verify the results manually.

### 1. Precision

**Used in**: Open-Vocabulary Segmentation on COCO and ADE20K

#### Definition

Precision measures the proportion of correctly predicted instances among all predicted instances.

**Formula**:

\[
\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
\]

#### Manual Verification Steps

1. **Collect Predictions**: For each region, record the predicted and ground-truth class labels.
2. **Identify True Positives and False Positives**:
   - TP: Predicted label matches ground truth.
   - FP: Predicted label does not match ground truth.
3. **Calculate Precision**: Use the formula above.

#### Example

| Region ID | Predicted Label | Ground Truth Label | Correct Prediction |
|-----------|-----------------|--------------------|--------------------|
| 1         | Cat             | Cat                | Yes                |
| 2         | Dog             | Dog                | Yes                |
| 3         | Car             | Bus                | No                 |
| ...       | ...             | ...                | ...                |

---

### 2. Recall @ 1

**Used in**: Phrase Grounding on Flickr30K

#### Definition

Recall @ 1 measures the proportion of phrases where the correct region is among the top prediction.

**Formula**:

\[
\text{Recall @ 1} = \frac{\text{True Positives (TP)}}{\text{Total Relevant Instances (Actual Positives)}}
\]

#### Manual Verification Steps

1. **Collect Predictions**: For each phrase, record the predicted region ID.
2. **Compare with Ground Truth**: Check if the predicted region matches the ground truth.
3. **Calculate Recall @ 1**: Use the formula above.

#### Example

| Phrase             | Predicted Region ID | Ground Truth Region ID | Correct Prediction |
|--------------------|---------------------|------------------------|--------------------|
| "A man in glasses" | 2                   | 2                      | Yes                |
| "A piece of paper" | 3                   | 3                      | Yes                |
| ...                | ...                 | ...                    | ...                |

---

### 3. Accuracy @ 0.5 (ACC @ 0.5)

**Used in**: Referring Expression Comprehension on RefCOCOg

#### Definition

ACC @ 0.5 measures the proportion of instances where the Intersection over Union (IoU) between the predicted and ground-truth regions is at least 0.5.

**IoU Formula**:

\[
\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}
\]

**ACC @ 0.5 Formula**:

\[
\text{ACC @ 0.5} = \frac{\text{Number of Instances with IoU} \geq 0.5}{\text{Total Number of Instances}}
\]

#### Manual Verification Steps

1. **Collect Predictions**: Record predicted regions for each expression.
2. **Compute IoU**: Calculate IoU between predicted and ground-truth regions.
3. **Determine Correct Predictions**: IoU ≥ 0.5 is considered correct.
4. **Calculate ACC @ 0.5**: Use the formula above.

#### Example

| Expression                    | IoU  | Correct Prediction |
|-------------------------------|------|--------------------|
| "The laptop behind the bottle" | 0.65 | Yes                |
| "Man wearing a red hat"       | 0.40 | No                 |
| ...                           | ...  | ...                |

---

### 4. Mean Intersection over Union (mIoU)

**Used in**: Referring Expression Segmentation on RefCOCOg

#### Definition

mIoU is the average IoU across all instances.

**Formula**:

\[
\text{mIoU} = \frac{1}{N} \sum_{i=1}^{N} \text{IoU}_i
\]

#### Manual Verification Steps

1. **Collect Predictions**: Obtain predicted masks for each expression.
2. **Compute IoU**: Calculate IoU between predicted and ground-truth masks.
3. **Calculate mIoU**: Sum all IoUs and divide by the number of instances.

#### Example

| Expression                | IoU  |
|---------------------------|------|
| "Girl with the red balloon" | 0.70 |
| "Cat sitting on the mat"  | 0.60 |
| ...                       | ...  |

---

## General Steps to Verify Benchmark Results Manually

1. **Collect Data**: Access predicted outputs and ground-truth annotations.
2. **Understand Metrics**: Ensure clarity on definitions and calculations.
3. **Sample Verification**: Start with a small subset to confirm understanding.
4. **Use Tools**: Utilize libraries like NumPy, OpenCV, or PIL for calculations.
5. **Document Calculations**: Keep detailed records for transparency.
6. **Cross-Verification**: Have results independently verified if possible.

---

## Additional Notes

- **Thresholds**: Apply thresholds consistently (e.g., IoU ≥ 0.5).
- **Units Consistency**: Ensure all measurements are in the same units.
- **Multiple Instances**: Correctly match predictions to ground-truth instances.
- **Edge Cases**: Be aware of predictions with zero overlap.

---

## Summary

- **Precision**: Proportion of correct class label predictions among all predictions.
- **Recall @ 1**: Proportion of phrases where the correct region is the top prediction.
- **ACC @ 0.5**: Accuracy of predictions with IoU ≥ 0.5.
- **mIoU**: Average IoU across all instances.

Understanding these metrics and following the manual verification steps ensures accurate evaluation of model performance in the benchmark.

---
