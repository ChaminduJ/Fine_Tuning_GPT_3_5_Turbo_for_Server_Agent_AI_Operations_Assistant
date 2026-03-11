import os
import json
import pandas as pd
import openai
import random
from tenacity import retry, stop_after_attempt, wait_exponential

prompt = """
A model that acts as an AI Operations Assistant for a web hosting automation platform called Server Agent.

Server Agent is a Python-based microservices platform used by hosting companies to automate infrastructure operations.

The platform manages and automates tasks including:

• Website provisioning
• DNS configuration
• SSL certificate installation
• WordPress deployment and management
• Backup scheduling and restoration
• Server monitoring
• Security and firewall configuration
• PHP runtime configuration
• Cross-server website migrations

The model will receive questions from:
1) Hosting customers
2) Support engineers
3) DevOps engineers

The model must respond with clear step-by-step instructions explaining how to perform tasks using the Server Agent platform.

Responses must:
• Be technically accurate
• Be structured with numbered steps
• Be easy to understand
• Help troubleshoot hosting operations
• Follow DevOps best practices

The assistant should behave like an expert DevOps engineer who understands Server Agent infrastructure automation.
"""

temperature = 0.3
number_of_examples = 120


openai.api_key = "YOUR_OPENAI_API_KEY"

N_RETRIES = 3


# This generates 120 Server Agent support examples like:
#
# prompt
# -----------
# How can I migrate a WordPress website to another server using Server Agent?
# -----------
#
# response
# -----------
# To migrate a WordPress site using Server Agent:
#
# 1. Log in to the Server Agent dashboard
# 2. Navigate to Site Management
# 3. Select the website you want to migrate
# 4. Click the Migration option
# 5. Choose the destination server
# 6. Start the migration process
#
# Server Agent will automatically transfer files, database, and configuration settings.
# -----------
@retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=70))
def generate_example(prompt, prev_examples, temperature=.5):

    messages=[
        {
            "role": "system",
            "content": f"""
You are generating training data for a machine learning model.

The model will be an AI Operations Assistant for the Server Agent hosting automation platform.

Generate a training example consisting of:

prompt
-----------
A realistic question related to hosting infrastructure operations using Server Agent.
-----------

response
-----------
A clear, technical, step-by-step answer explaining how to solve the problem.
-----------

Each example should involve tasks such as:

• Provisioning websites
• Configuring DNS
• Installing SSL certificates
• Deploying WordPress
• Setting up automated backups
• Migrating websites between servers
• Managing PHP versions
• Configuring firewall rules
• Troubleshooting hosting issues

Examples should become slightly more complex each time and remain diverse.
"""
        }
    ]

    if len(prev_examples) > 0:
        if len(prev_examples) > 8:
            prev_examples = random.sample(prev_examples, 8)

        for example in prev_examples:
            messages.append({
                "role": "assistant",
                "content": example
            })

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=temperature,
        max_tokens=1000,
    )

    return response.choices[0].message['content']

prev_examples = []

for i in range(number_of_examples):
    print(f'Generating example {i}')
    example = generate_example(prompt, prev_examples, temperature)
    prev_examples.append(example)

print(prev_examples)

#This creates the system instruction used during inference.
# example:
# You are an AI assistant specialized in the Server Agent hosting automation platform. Provide clear step-by-step instructions to help users perform hosting operations such as website provisioning, DNS configuration, SSL installation, backups, WordPress deployment, and server migrations.
def generate_system_message(prompt):
  response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
      {
        "role": "system",
        "content": """
You will be given a high-level description of the model we are training.

Generate a concise system prompt describing how the assistant should behave.
"""
      },
      {
        "role": "user",
        "content": prompt.strip(),
      }
    ],
    temperature=temperature,
    max_tokens=500,
  )

  return response.choices[0].message['content']


system_message = generate_system_message(prompt)

print(system_message)

#Convert Dataset to DataFrame and Convert Dataset to Fine-Tune Format
prompts = []
responses = []

for example in prev_examples:
    try:
        split_example = example.split('-----------')
        prompts.append(split_example[1].strip())
        responses.append(split_example[3].strip())
    except:
        pass

df = pd.DataFrame({
    'prompt': prompts,
    'response': responses
})

df = df.drop_duplicates()

print('Total training examples:', len(df))

training_examples = []

for index, row in df.iterrows():
    training_example = {
        "messages": [
            {"role": "system", "content": system_message.strip()},
            {"role": "user", "content": row['prompt']},
            {"role": "assistant", "content": row['response']}
        ]
    }

    training_examples.append(training_example)

with open('training_examples.jsonl', 'w') as f:
    for example in training_examples:
        f.write(json.dumps(example) + '\n')

#Upload Dataset
file_id = openai.File.create(
  file=open("/content/training_examples.jsonl", "rb"),
  purpose='fine-tune'
).id

# Train the model! You may need to wait a few minutes before running the next cell to allow for the file to process on OpenAI's servers.
job = openai.FineTuningJob.create(
    training_file=file_id,
    model="gpt-3.5-turbo"
)

job_id = job.id

#Now, just wait until the fine-tuning run is done, and you'll have a ready-to-use model!
openai.FineTuningJob.list_events(id=job_id, limit=10)

#Once your model is trained, run the next cell to grab the fine-tuned model name.
model_name_pre_object = openai.FineTuningJob.retrieve(job_id)
model_name = model_name_pre_object.fine_tuned_model
print(model_name)

#Test the Fine-Tuned Model
# expected response:
# To enable SSL for a domain in Server Agent:
# 1. Log into the Server Agent dashboard
# 2. Navigate to Site Management
# 3. Select the domain
# 4. Open the SSL settings
# 5. Click Enable SSL
# 6. Server Agent will automatically issue a Let's Encrypt certificate and configure HTTPS
response = openai.ChatCompletion.create(
    model=model_name,
    messages=[
      {
        "role": "system",
        "content": system_message,
      },
      {
          "role": "user",
          "content": "How do I enable SSL for a domain using Server Agent?"
      }
    ],
)

print(response.choices[0].message['content'])