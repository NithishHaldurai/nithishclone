import json
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
import numpy as np
import re
from collections import Counter

class GenerativeUserClone:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.vectorizer = None
        self.knn_model = None
        self.conversations = []
        self.models_loaded = False
        self.user_style = {}
        
    def train(self, inputs, responses):
        """Train the retrieval part of the system"""
        print("Training retrieval model...")
        
        # Store conversations for retrieval
        self.conversations = list(zip(inputs, responses))
        
        # Analyze user's style patterns
        self.analyze_user_style(responses)
        
        # Train TF-IDF vectorizer on inputs
        self.vectorizer = TfidfVectorizer(
            max_features=100, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        X = self.vectorizer.fit_transform(inputs)
        
        # Train KNN model
        self.knn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
        self.knn_model.fit(X)
        
        # Save models
        joblib.dump(self.vectorizer, f"{self.model_dir}/vectorizer.joblib")
        joblib.dump(self.knn_model, f"{self.model_dir}/knn_model.joblib")
        joblib.dump(self.conversations, f"{self.model_dir}/conversations.joblib")
        joblib.dump(self.user_style, f"{self.model_dir}/user_style.joblib")
        
        self.models_loaded = True
        print(f"✓ Retrieval model trained with {len(inputs)} conversations")
        print("✓ Ready for generative chatting!")
    
    def load_models(self):
        """Load trained models"""
        try:
            if not os.path.exists(f"{self.model_dir}/vectorizer.joblib"):
                return False
                
            self.vectorizer = joblib.load(f"{self.model_dir}/vectorizer.joblib")
            self.knn_model = joblib.load(f"{self.model_dir}/knn_model.joblib")
            self.conversations = joblib.load(f"{self.model_dir}/conversations.joblib")
            self.user_style = joblib.load(f"{self.model_dir}/user_style.joblib")
            self.models_loaded = True
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def learn_from_chat(self, user_input, user_response, was_correct=True):
        """Learn from chat interactions"""
        if was_correct:
            # Add this as a new conversation pattern
            self.conversations.append((user_input, user_response))
            
            # Update the vectorizer and KNN model with new data
            try:
                if self.vectorizer and self.knn_model:
                    # Get all current inputs
                    all_inputs = [conv[0] for conv in self.conversations]
                    
                    # Retrain vectorizer with new data
                    X = self.vectorizer.fit_transform(all_inputs)
                    
                    # Retrain KNN model
                    self.knn_model.fit(X)
                    
                    # Save updated models
                    joblib.dump(self.vectorizer, f"{self.model_dir}/vectorizer.joblib")
                    joblib.dump(self.knn_model, f"{self.model_dir}/knn_model.joblib")
                    joblib.dump(self.conversations, f"{self.model_dir}/conversations.joblib")
                    
            except Exception as e:
                print(f"Note: Couldn't update models in real-time: {e}")

    def analyze_user_style(self, responses):
        """Analyze user's response style patterns"""
        if not responses:
            self.user_style = {}
            return
        
        styles = {
            'common_words': [],
            'response_length': 0,
            'punctuation_style': 'neutral',
            'formality_level': 'neutral',
            'common_starters': [],
            'common_enders': []
        }
        
        all_words = []
        lengths = []
        punctuations = []
        
        for response in responses:
            # Words analysis
            words = response.lower().split()
            all_words.extend(words)
            
            # Length analysis
            lengths.append(len(words))
            
            # Punctuation analysis
            if '!' in response:
                punctuations.append('excited')
            elif '...' in response or '..' in response:
                punctuations.append('thoughtful')
            elif response.endswith('.'):
                punctuations.append('neutral')
            else:
                punctuations.append('casual')
            
            # Starters and enders
            if len(words) > 1:
                styles['common_starters'].append(words[0])
                styles['common_enders'].append(words[-1])
        
        # Calculate most common patterns
        styles['common_words'] = Counter(all_words).most_common(10)
        styles['response_length'] = int(np.mean(lengths)) if lengths else 5
        
        if punctuations:
            styles['punctuation_style'] = max(set(punctuations), key=punctuations.count)
        
        # Formality analysis
        formal_words = {'however', 'therefore', 'furthermore', 'moreover', 'thus'}
        casual_words = {'lol', 'haha', 'omg', 'btw', 'imo', 'hey', 'hi', 'yo'}
        
        formal_count = sum(1 for word in all_words if word in formal_words)
        casual_count = sum(1 for word in all_words if word in casual_words)
        
        if formal_count > casual_count:
            styles['formality_level'] = 'formal'
        elif casual_count > formal_count:
            styles['formality_level'] = 'casual'
        
        self.user_style = styles
    
    def is_simple_greeting(self, user_input):
        """Check if input is a simple greeting that should always generate new response"""
        user_input_lower = user_input.lower().strip()
        
        # Very simple greetings
        simple_greetings = {
            'hi', 'hello', 'hey', 'hy', 'helo', 'yo', 'sup', 
            'hi!', 'hello!', 'hey!', 'hi.', 'hello.', 'hey.',
            'hii', 'helloo', 'heyy'
        }
        
        # Check for exact matches with simple greetings
        if user_input_lower in simple_greetings:
            return True
        
        # Check for very short inputs that are likely greetings
        if len(user_input_lower) <= 5 and any(word in user_input_lower for word in ['hi', 'hey', 'hello']):
            return True
            
        return False
    
    def should_use_personal_response(self, user_input, similarity_score):
        """Decide whether to use personal response or generate new one"""
        # NEVER use personal responses for simple greetings
        if self.is_simple_greeting(user_input):
            return False
        
        # For other questions, be more strict about using personal responses
        question_words = {'what', 'how', 'why', 'when', 'where', 'which', 'who', '?'}
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in question_words):
            return similarity_score < 0.6  # Use personal only for close matches
        
        # Default: use personal if reasonably similar
        return similarity_score < 0.8
    
    def find_retrieval_response(self, user_input):
        """Try to find a response from user's personal data with similarity score"""
        user_input_lower = user_input.lower().strip()
        
        # Strategy 1: Exact or very close match
        for inp, resp in self.conversations:
            if self.is_very_similar(user_input_lower, inp.lower()):
                return resp, "personal", 0.0  # Perfect match
        
        # Strategy 2: Semantic similarity
        try:
            input_vec = self.vectorizer.transform([user_input])
            distances, indices = self.knn_model.kneighbors(input_vec)
            
            similarity_score = distances[0][0]
            best_match_idx = indices[0][0]
            
            return self.conversations[best_match_idx][1], "similar_personal", similarity_score
        except:
            pass
        
        return None, "no_match", 1.0
    
    def is_very_similar(self, input1, input2):
        """Check if two inputs are very similar"""
        input1_clean = re.sub(r'[^\w\s]', '', input1)
        input2_clean = re.sub(r'[^\w\s]', '', input2)
        
        if input1_clean == input2_clean:
            return True
        if input1_clean in input2_clean or input2_clean in input1_clean:
            return True
        
        return False
    
    def classify_input_type(self, user_input):
        """Better classification of input type for appropriate responses"""
        user_input_lower = user_input.lower().strip()
        
        # Simple greetings (already handled separately, but double-check)
        if self.is_simple_greeting(user_input):
            return 'greeting'
        
        # Status updates about how someone is doing
        status_words = {'good', 'fine', 'ok', 'okay', 'great', 'awesome', 'excellent', 'well'}
        if any(word in user_input_lower for word in status_words):
            return 'status_update'
        
        # Negative or minimal responses
        minimal_words = {'nothing', 'not much', 'nm', 'same', 'idk', "don't know", 'nothing much'}
        if any(word in user_input_lower for word in minimal_words):
            return 'minimal_response'
        
        # Eating/food related
        food_words = {'eat', 'ate', 'eating', 'food', 'hungry', 'meal'}
        if any(word in user_input_lower for word in food_words):
            return 'food_related'
        
        # Life questions
        life_words = {'life', 'living', 'existence', 'world'}
        if any(word in user_input_lower for word in life_words):
            return 'life_question'
        
        # Listening/attention questions
        attention_words = {'listen', 'listening', 'hear', 'paying attention'}
        if any(word in user_input_lower for word in attention_words):
            return 'attention_check'
        
        # Questions
        if '?' in user_input_lower:
            personal_questions = {'you', 'your', 'yourself'}
            if any(word in user_input_lower for word in personal_questions):
                return 'question_personal'
            else:
                return 'question_factual'
        
        # Agreement
        agreement_words = {'yes', 'yeah', 'yep', 'sure', 'right', 'true', 'correct', 'ok'}
        if any(word in user_input_lower for word in agreement_words):
            return 'agreement'
        
        # Thanks
        thanks_words = {'thanks', 'thank you', 'thx', 'ty'}
        if any(word in user_input_lower for word in thanks_words):
            return 'thanks'
        
        # Default
        return 'casual_statement'
    
    def generate_creative_response(self, user_input):
        """Generate a creative response based on user's style"""
        user_input_lower = user_input.lower()
        
        # Check for multiple questions first
        multi_response = self.handle_multiple_questions(user_input)
        if multi_response:
            return multi_response
        
        # Get the input type
        input_type = self.classify_input_type(user_input)
        
        # Enhanced response templates with COMPLETE responses
        response_templates = {
            'greeting': [
                "Hey there! How's it going?",
                "Hi! Nice to see you!",
                "Hello! What's new?",
                "Hey! How are you doing?",
                "Hi there! How can I help you today?",
                "Hello! Great to talk to you!",
                "Hey! What's on your mind?",
                "Hi! How's your day going?"
            ],
            'status_update': [
                "That's great to hear!",
                "Awesome! What have you been up to?",
                "Nice! How's everything else going?",
                "Glad to hear that!",
                "That's wonderful! Anything new?",
                "Cool! What's happening with you?"
            ],
            'minimal_response': [
                "I see!",
                "Okay, cool!",
                "No worries!",
                "That's fine!",
                "Alright then!",
                "Sounds good to me!",
                "Got it!",
                "No problem!"
            ],
            'food_related': [
                "Nice! What did you have to eat?",
                "Sounds good! I hope it was tasty!",
                "Food is always great! What was your favorite part?",
                "Yum! I'm getting hungry just thinking about food!",
                "Good to hear you ate! Was it delicious?"
            ],
            'life_question': [
                "Life is pretty interesting these days!",
                "Life has its ups and downs, but overall it's good!",
                "I think life is what you make of it!",
                "Life is full of surprises, don't you think?",
                "I'm enjoying life! How about you?"
            ],
            'attention_check': [
                "Yes, I'm listening! What's up?",
                "I'm here and paying attention!",
                "Of course I'm listening! Go ahead.",
                "Yes, I hear you! What would you like to talk about?",
                "I'm all ears! What's on your mind?"
            ],
            'question_personal': [
                "That's an interesting question! What do you think?",
                "I'm not sure about that myself. What's your opinion?",
                "That's something worth thinking about!",
                "I'm still learning about that. What's your perspective?",
                "That's a great question! I'd love to know your thoughts too."
            ],
            'question_factual': [
                "From what I know, it's quite fascinating.",
                "I believe there are different perspectives on that.",
                "Based on what I've learned, it's an interesting topic.",
                "I think that depends on how you look at it.",
                "In my understanding, it's something worth exploring."
            ],
            'agreement': [
                "I agree!",
                "That's right!",
                "Exactly what I was thinking!",
                "You've got a point there!",
                "I think so too!",
                "Definitely!",
                "Absolutely!",
                "For sure!"
            ],
            'thanks': [
                "You're welcome!",
                "No problem at all!",
                "Happy to help!",
                "Anytime!",
                "Of course! Glad I could help!"
            ],
            'casual_statement': [
                "I see what you mean!",
                "That makes sense!",
                "Interesting point!",
                "I get what you're saying.",
                "That's a good way to look at it!",
                "I understand where you're coming from.",
                "Cool! Tell me more about that.",
                "Nice! What else is new?"
            ]
        }
        
        # Select template based on input type
        templates = response_templates.get(input_type, response_templates['casual_statement'])
        
        # Apply user's style to the response
        response = random.choice(templates)
        response = self.apply_user_style(response, user_input)
        
        return response
    
    def handle_multiple_questions(self, user_input):
        """Handle inputs with multiple questions"""
        user_input_lower = user_input.lower()
        
        # Check if input contains multiple questions/conjunctions
        conjunctions = {' and ', ' also ', ' plus ', ' furthermore ', ' moreover '}
        has_multiple = any(conj in user_input_lower for conj in conjunctions)
        
        if not has_multiple:
            return None
        
        # Try to find answers for different parts
        responses = []
        
        # Common question patterns to split on
        if ' and ' in user_input_lower:
            parts = user_input.split(' and ')
            for part in parts:
                part = part.strip()
                if part and len(part) > 3:  # Meaningful part
                    response = self.find_single_response(part)
                    if response:
                        responses.append(response)
        
        # If we found multiple responses, combine them
        if len(responses) > 1:
            # Combine with natural language
            combined = self.combine_responses(responses)
            return combined
        
        return None
    
    def find_single_response(self, user_input):
        """Find response for a single question/statement"""
        # Try personal response first
        personal_response, _, similarity_score = self.find_retrieval_response(user_input)
        if personal_response and self.should_use_personal_response(user_input, similarity_score):
            return personal_response
        
        # Otherwise generate
        return self.generate_creative_response(user_input)
    
    def combine_responses(self, responses):
        """Combine multiple responses naturally"""
        if len(responses) == 2:
            connectors = [
                "{} and also {}",
                "{}, and {}",
                "Well, {} plus {}",
                "{} - and to add to that, {}",
                "{}. Also, {}"
            ]
            return random.choice(connectors).format(responses[0], responses[1])
        else:
            return ". ".join(responses) + "."
    
    def apply_user_style(self, response, user_input):
        """Apply user's personal style to the generated response"""
        if not self.user_style:
            return response
        
        # Apply punctuation style
        punctuation_style = self.user_style.get('punctuation_style', 'neutral')
        if punctuation_style == 'excited' and not response.endswith('!'):
            response = response.rstrip('.') + '!'
        elif punctuation_style == 'thoughtful' and not any(p in response for p in ['.', '!', '?']):
            response = response + '...'
        elif punctuation_style == 'casual' and response.endswith('.'):
            response = response.rstrip('.')
        
        # Apply formality level
        formality = self.user_style.get('formality_level', 'neutral')
        if formality == 'casual':
            # Make it more casual
            if response.startswith('Hello'):
                response = response.replace('Hello', 'Hey', 1)
            if 'How are you' in response:
                response = response.replace('How are you', 'How are ya')
        
        return response
    
    def generate_response(self, user_input: str) -> str:
        """Generate response using smart hybrid approach"""
        if not self.models_loaded:
            if not self.load_models():
                return "Please train the model first (option 2)."
        
        if not self.conversations:
            return self.generate_creative_response(user_input)
        
        # Step 1: For simple greetings, ALWAYS generate new response
        if self.is_simple_greeting(user_input):
            return self.generate_creative_response(user_input)
        
        # Step 2: Try to find personal response with similarity score
        personal_response, match_type, similarity_score = self.find_retrieval_response(user_input)
        
        # Step 3: Decide whether to use personal response or generate new one
        if personal_response and self.should_use_personal_response(user_input, similarity_score):
            return personal_response
        else:
            # Step 4: Generate creative response
            return self.generate_creative_response(user_input)