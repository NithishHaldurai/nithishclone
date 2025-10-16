import json
import datetime
from typing import List, Dict

class SimpleDataCollector:
    def __init__(self, data_file="user_data.json"):
        self.data_file = data_file
        self.conversations = self.load_data()
    
    def load_data(self) -> List[Dict]:
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_data(self):
        with open(self.data_file, 'w') as f:
            json.dump(self.conversations, f, indent=2)
    
    def add_conversation(self, user_input: str, user_response: str):
        conversation = {
            'timestamp': str(datetime.datetime.now()),
            'input': user_input,
            'response': user_response,
            'response_length': len(user_response.split()),
            'is_question': '?' in user_input
        }
        self.conversations.append(conversation)
        self.save_data()
        print(f"✓ Added conversation {len(self.conversations)}")

    def get_training_data(self):
        """Prepare data for training"""
        inputs = [conv['input'] for conv in self.conversations]
        responses = [conv['response'] for conv in self.conversations]
        return inputs, responses