#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# simple_bot.py
import re

# very simple intent matching using keywords / regex
def detect_intent(message):
    m = message.lower()
    if re.search(r'\b(hi|hello|hey)\b', m):
        return "greeting"
    if re.search(r'\b(order|buy|price|cost)\b', m):
        return "order_query"
    if re.search(r'\b(return|refund|cancel)\b', m):
        return "returns"
    if re.search(r'\b(thanks|thank you)\b', m):
        return "thanks"
    return "fallback"

def handle_intent(intent, message):
    if intent == "greeting":
        return "Hi! 👋 How can I help you today?"
    if intent == "order_query":
        return "Are you asking about order status, prices, or how to place an order?"
    if intent == "returns":
        return "You can return items within 14 days. Do you want return instructions?"
    if intent == "thanks":
        return "You're welcome! Anything else I can help with?"
    return "Sorry, I didn't understand. Can you rephrase?"

def main():
    print("SimpleBot: Hello! (type 'quit' to exit)")
    while True:
        text = input("You: ").strip()
        if text.lower() in ("quit", "exit"):
            print("SimpleBot: Bye! 👋")
            break
        intent = detect_intent(text)
        reply = handle_intent(intent, text)
        print("SimpleBot:", reply)

if __name__ == "__main__":
    main()


# In[ ]:


# dialogflow_webhook.py
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    # Dialogflow v2 format: queryResult in req
    query_text = req.get('queryResult', {}).get('queryText')
    intent = req.get('queryResult', {}).get('intent', {}).get('displayName')
    params = req.get('queryResult', {}).get('parameters', {})

    # debug print
    print("Received intent:", intent, "text:", query_text, "params:", params)

    # simple logic
    if intent == "OrderStatus":
        order_id = params.get("order_id")
        if order_id:
            reply = f"Order {order_id} is in transit and should arrive in 2 days."
        else:
            reply = "Please provide your order ID."
    elif intent == "Default Welcome Intent":
        reply = "Welcome! How can I help you with your order today?"
    else:
        reply = "Sorry, I don't have an answer for that yet."

    # Dialogflow expects this shape
    return jsonify({
        "fulfillmentText": reply,
        # optional: webhookPayload or fulfillmentMessages for rich responses
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)


# In[ ]:


return jsonify({
  "fulfillmentMessages": [
    {"text": {"text": [reply]}},
    {
      "card": {
        "title": "Order Help",
        "subtitle": "Track, cancel or return",
        "buttons": [
          {"text": "Track order", "postback": "track_order"},
          {"text": "Cancel order", "postback": "cancel_order"}
        ]
      }
    }
  ]
})


# In[ ]:


import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv('intents.csv')

# Train model
X = data['patterns']
y = data['intent']

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

model.fit(X, y)

# Response dictionary
responses = {}
for intent in data['intent'].unique():
    responses[intent] = data[data['intent'] == intent]['responses'].values[0]

# Chat function
def chatbot_response(text):
    intent = model.predict([text])[0]
    return random.choice(responses[intent].split('|'))

# Run chatbot
print("🤖 Chatbot: Hello! I’m your support assistant. Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("🤖 Chatbot: Goodbye! Have a nice day 😊")
        break
    print("🤖 Chatbot:", chatbot_response(user_input))


# In[ ]:


import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# In[ ]:


file_path = r"C:\Users\koskr\Desktop\future_inturn\task_3\customer_support1\twcs\twcs.csv"


# In[ ]:


data = pd.read_csv(file_path)


# In[ ]:


print(data)


# In[ ]:


X = data['patterns']
y = data['intent']


# In[ ]:



model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

model.fit(X, y)


# In[ ]:



responses = {}
for intent in data['intent'].unique():
    responses[intent] = data[data['intent'] == intent]['responses'].values[0]


# In[ ]:


def chatbot_response(text):
    intent = model.predict([text])[0]
    return random.choice(responses[intent].split('|'))

# Run chatbot
print("🤖 Chatbot: Hello! I’m your support assistant. Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("🤖 Chatbot: Goodbye! Have a nice day 😊")
        break
    print("🤖 Chatbot:", chatbot_response(user_input))


# In[ ]:





# In[ ]:




