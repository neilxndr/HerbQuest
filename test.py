import openai

# Set your OpenAI API key here
api_key = "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"

# Initialize the OpenAI API client
openai.api_key = api_key

# Define a prompt
prompt = "Translate the following English text to French: 'Hello, how are you?'"

# Call the OpenAI API to generate a response
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=50,  # You can adjust the number of tokens as needed
)

# Extract and print the generated text
generated_text = response.choices[0].text
print("Generated Text:")
print(generated_text)
