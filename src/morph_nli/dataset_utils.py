"""
This file handles the dataset loading/preparation
"""

import datasets

def get_snli_data(partition='validation'):
    dataset_name = 'snli'
    snli = datasets.load_dataset('snli')[partition]
    features = snli.features['label'].names
    return (dataset_name, snli, features, 'premise', 'hypothesis', lambda pred: pred)

def get_mnli_matched_data(partition='validation_matched'):
    dataset_name = 'glue/mnli/matched'
    mnli     = datasets.load_dataset('glue', 'mnli')[partition]
    features = mnli.features['label'].names
    return (dataset_name, mnli, features, 'premise', 'hypothesis', lambda pred: pred)

def get_mnli_mismatched_data(partition='validation_mismatched'):
    dataset_name = 'glue/mnli/mismatched'
    mnli     = datasets.load_dataset('glue', 'mnli')[partition]
    features = mnli.features['label'].names
    return (dataset_name, mnli, features, 'premise', 'hypothesis', lambda pred: pred)

def get_anli_data(partition='dev_r1'):
    dataset_name = 'anli'
    anli     = datasets.load_dataset(dataset_name)[partition]
    features = anli.features['label'].names
    return (dataset_name, anli, features, 'premise', 'hypothesis', lambda pred: pred)

def get_sick_data(partition='validation'):
    dataset_name = 'sick'
    sick     = datasets.load_dataset(dataset_name)[partition]
    features = sick.features['label'].names
    return (dataset_name, sick, features, 'sentence_A', 'sentence_B', lambda pred: pred)

def get_hans_data(partition='validation'):
    dataset_name = 'hans'
    hans     = datasets.load_dataset(dataset_name)[partition]
    features = hans.features['label'].names
    return (dataset_name, hans, features, 'premise', 'hypothesis', lambda pred: ['entailment' if x == 'entailment' else 'non-entailment' for x in pred])

