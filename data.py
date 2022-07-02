import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from datasets import load_dataset

class MVSADataset(torch.utils.data.Dataset):
    def __init__(self):
        self.transform = transforms.Compose([
                                            transforms.Resize(256),                    
                                            transforms.CenterCrop(224),                
                                            transforms.ToTensor(),                     
                                            transforms.Normalize(                     
                                            mean=[0.485, 0.456, 0.406],                
                                            std=[0.229, 0.224, 0.225]                  
                                            )])
        self.path = "data/full"
        
        raw_data = load_dataset('csv', data_files=[f'{self.path}/data.csv'])
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.data = raw_data.map(self.tokenize, batched=True)
        self.data.set_format(type='torch', columns=['id', 'labels', 'text', 'input_ids', 'attention_mask'])

    def tokenize(self, input_sentence):
        return self.tokenizer(input_sentence['text'], padding='max_length', truncation=True)

    def __len__(self):
        return len(self.data['train'])
    
    def __getitem__(self, idx):
        f = os.path.join(f"{self.path}/imgs/", f"{idx}.jpg")
        img = Image.open(f)
        return (self.transform(img), (self.data['train']['input_ids'][idx], self.data['train']['attention_mask'][idx])), self.data['train']['labels'][idx]

class HatefulMemes(torch.utils.data.Dataset):
    def __init__(self):
        self.transform = transforms.Compose([
                                            transforms.Resize(256),                    
                                            transforms.CenterCrop(224),                
                                            transforms.ToTensor(),                     
                                            transforms.Normalize(                     
                                            mean=[0.485, 0.456, 0.406],                
                                            std=[0.229, 0.224, 0.225]                  
                                            )])

        self.path = 'hateful/full.jsonl'

        raw_data = load_dataset('json', data_files=[self.path])
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.data = raw_data.map(self.tokenize, batched=True)
        self.data.set_format(type='torch', columns=['label', 'input_ids', 'attention_mask'])

    def tokenize(self, input_sentence):
        return self.tokenizer(input_sentence['text'], padding='max_length', truncation=True)

    def __len__(self):
        return len(self.data['train'])

    def __getitem__(self, idx):
        return (self.transform(Image.open(f"hateful/{self.data['train']['img'][idx]}").convert('RGB')), (self.data['train']['input_ids'][idx], self.data['train']['attention_mask'][idx])), self.data['train']['label'][idx]