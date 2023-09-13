import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

system_msg = 'You are controlling a robot that responds to specific commands.'

# get the user message from the text file
prompt_common = open('prompts/prompt_common.txt', 'r').read()
prompt_common_2 = open('prompts/prompt_common_2.txt', 'r').read()
prompt_open_drawer = open('prompts/prompt_open_drawer.txt', 'r').read()
task = 'open_drawer'
# create responses directory
os.makedirs('responses/' + task, exist_ok=True)
vars = ['top', 'middle', 'bottom']
for var in vars:
    prompt_copy = prompt_open_drawer
    # replace {var} with top or middle or bottom
    prompt_copy = prompt_copy.replace('{var}', var)

    user_msg = prompt_copy + prompt_common + prompt_common_2
    response = openai.ChatCompletion.create(
              model="gpt-4",
              messages=[{"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
              ])
    content = response.choices[0].message.content

    # save the content to a txt file
    with open('responses/' + task + '/' + var + '.txt', 'w') as f:
        f.write(content)
pass