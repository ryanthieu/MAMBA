from flask import Flask, jsonify, request, render_template
import json
import random

app = Flask(__name__)

# Load data from JSONL files
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

questions = load_jsonl('/workspace/MAMBA/question.jsonl')
responses = load_jsonl('/workspace/MAMBA/mamba2-hybrid-8b-3t-128k.jsonl')

# Create a mapping from question_id to question
questions_dict = {q['question_id']: q for q in questions}

# Function to match responses withs their corresponding questions
def match_questions_responses():
    matched_data = []
    for response in responses:
        question_id = response['question_id']
        question = questions_dict.get(question_id)
        if question:
            matched_data.append({
                'question_id': question_id,
                'category': question.get('category'),
                'question_turns': question.get('turns'),
                'response_turns': response['choices'][0]['turns']
            })
    return matched_data

matched_examples = match_questions_responses()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search_page():
    return render_template('search.html')

@app.route('/search_examples')
def search_examples():
    query = request.args.get('q', '').lower()
    category = request.args.get('category', '').lower()
    question_id = request.args.get('id', '').strip()
    
    filtered_examples = []

    for example in matched_examples:
        if (not query or any(query in text.lower() for text in example['question_turns'] + example['response_turns'])) and \
           (not category or example['category'].lower() == category) and \
           (not question_id or str(example['question_id']) == question_id):
            filtered_examples.append(example)

    return jsonify(filtered_examples)

@app.route('/random')
def get_random_example():
    example = random.choice(matched_examples)
    return jsonify(example)

@app.route('/example/next/<int:id>')
def next_example(id):
    if 0 <= id < len(matched_examples) - 1:
        return jsonify(matched_examples[id + 1])
    return jsonify({'error': 'No more examples'})

@app.route('/example/previous/<int:id>')
def previous_example(id):
    if id > 0:
        return jsonify(matched_examples[id - 1])
    return jsonify({'error': 'No previous examples'})

if __name__ == '__main__':
    app.run(debug=True)
