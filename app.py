import os
import json
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime

app = Flask(__name__)

# Add the CLI version directory to Python path
CLI_VERSION_PATH = r'D:\user-clone-simple\cli-version'
sys.path.insert(0, CLI_VERSION_PATH)

# Import your ML clone system
try:
    from user_clone import GenerativeUserClone
    from data_collector import SimpleDataCollector
    ML_CLONE_AVAILABLE = True
    print("✅ ML Clone system imported successfully!")
except ImportError as e:
    print(f"❌ Failed to import ML clone: {e}")
    ML_CLONE_AVAILABLE = False

# Initialize your ML components
if ML_CLONE_AVAILABLE:
    clone = GenerativeUserClone()
    collector = SimpleDataCollector()
    
    # Try to load trained models
    if clone.load_models():
        print("✅ ML models loaded successfully!")
    else:
        print("⚠️  ML models not trained yet")
else:
    clone = None
    collector = None

# Store current session conversations (not saved to training data)
current_session_conversations = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/api/conversations')
def get_conversations():
    """Get only current session conversations - no old data"""
    try:
        # Return only current session conversations (last 20 messages)
        recent_messages = current_session_conversations[-20:]
        
        # Format for frontend
        formatted_messages = []
        for i, conv in enumerate(recent_messages):
            formatted_messages.append({
                'id': f"user_{i}",
                'username': 'Visitor',
                'content': conv['input'],
                'timestamp': conv['timestamp'],
                'type': 'user_message'
            })
            
            formatted_messages.append({
                'id': f"ai_{i}",
                'username': 'Clone',
                'content': conv['response'],
                'timestamp': conv['timestamp'],
                'type': 'ai_message'
            })
        
        return jsonify(formatted_messages)
        
    except Exception as e:
        print(f"Error getting conversations: {e}")
        return jsonify({'error': 'Failed to get conversations'}), 500

@app.route('/api/send_message', methods=['POST'])
def send_message():
    """Send message to ML clone - store only in current session"""
    try:
        if not ML_CLONE_AVAILABLE:
            return jsonify({'status': 'error', 'message': 'ML system not available'}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON data received'}), 400
            
        user_input = data.get('content', '')
        
        if not user_input.strip():
            return jsonify({'status': 'error', 'message': 'Message content is empty'}), 400
        
        # Use your actual ML clone to generate response
        if clone.models_loaded:
            response = clone.generate_response(user_input)
            response_type = "ml_generated"
        else:
            # Fallback to simple response if models not trained
            response = "The AI clone is currently learning. Please try again later."
            response_type = "fallback"
        
        # Store in current session only (not in training data)
        conversation = {
            'timestamp': datetime.now().isoformat(),
            'input': user_input,
            'response': response,
            'response_type': response_type
        }
        current_session_conversations.append(conversation)
        
        print(f"Session conversation #{len(current_session_conversations)}: {user_input} -> {response}")
        return jsonify({
            'status': 'success', 
            'ai_response': response
        })
        
    except Exception as e:
        print(f"Error sending message: {e}")
        return jsonify({'status': 'error', 'message': f'Internal server error: {str(e)}'}), 500

@app.route('/api/stats')
def get_stats():
    """Get basic system stats"""
    try:
        stats = {
            'ml_system_available': ML_CLONE_AVAILABLE,
            'ml_models_loaded': clone.models_loaded if ML_CLONE_AVAILABLE else False,
            'current_session_messages': len(current_session_conversations)
        }
        
        return jsonify({
            'status': 'success',
            'stats': stats
        })
        
    except Exception as e:
        print(f"Error getting stats: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to get statistics'}), 500

@app.route('/api/clear_session', methods=['POST'])
def clear_session():
    """Clear current session conversations"""
    try:
        current_session_conversations.clear()
        return jsonify({
            'status': 'success',
            'message': 'Session cleared'
        })
    except Exception as e:
        print(f"Error clearing session: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to clear session'}), 500

if __name__ == '__main__':
    print("=== Clean ML User Clone Web Interface ===")
    print(f"ML System Available: {ML_CLONE_AVAILABLE}")
    
    if ML_CLONE_AVAILABLE:
        print(f"Models loaded: {clone.models_loaded}")
    
    print("Features:")
    print("  ✅ Clean interface for strangers")
    print("  ✅ No old conversation history")
    print("  ✅ No feedback requests")
    print("  ✅ Session-only storage")
    print("  ✅ Real ML clone responses")
    
    print("Running on: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)