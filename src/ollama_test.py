from ollama import chat, generate

path = "test/photo_1.jpg"


content_wPath = f"whats in the image, be concise. {path}"
content = "whats in the image, be concise."

# response = chat(
#     model="gemma3:12b", 
#     messages=[{'role': 'user', 'content': content_wPath}]
# )


# response = chat(
#     model="llava:latest", 
#     messages=[{'role': 'user', 'content': content}], 
#     images=[path]
# )

stream = generate(
    model="gemma3:12b", 
    prompt='Rate the attractiveness of the person in this image. Respond with only both of these options and a confidence score (0-1): "attractive: 0.XX" and "not attractive: 0.XX", exactly this, nothing else!',
    images=[path]
)
for chunk in stream:
    if chunk[0] == "response":
        print(chunk[1])

stream = generate(
    model="gemma3:12b", 
    prompt='whats in the image? be concise',
    images=[path]
)

for chunk in stream:
    if chunk[0] == "response":
        print(chunk[1])