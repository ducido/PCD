# import simpler_env
# import pandas as pd
# from tqdm import tqdm

# tasks = (
#     "google_robot_pick_coke_can",
#     "google_robot_move_near",
#     "google_robot_close_drawer",
#     "google_robot_open_drawer",
#     "google_robot_place_apple_in_closed_top_drawer",
#     "widowx_carrot_on_plate",
#     "widowx_spoon_on_towel",
#     "widowx_put_eggplant_in_basket",
#     "widowx_stack_cube",
# )

# seen = set()
# data = []

# for task in tqdm(tasks, desc="Tasks"):
#     env = simpler_env.make(task)
    
#     for i in tqdm(range(100), desc=f"{task}", leave=False):
#         obs, _ = env.reset(seed=i)
        
#         instruction = env.unwrapped.get_language_instruction()
#         is_final_subtask = env.unwrapped.is_final_subtask()
        
#         key = (task, instruction, is_final_subtask)

#         if key not in seen:
#             seen.add(key)
#             data.append({
#                 "task": task,
#                 "instruction": instruction,
#                 "is_final_subtask": is_final_subtask
#             })

#     del env

# # tạo dataframe
# df = pd.DataFrame(data)

# # save file
# df.to_csv("robot_tasks.csv", index=False)

# print(df.head())



import simpler_env
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

df = pd.read_csv("robot_tasks.csv")

# =====================
# 1. Load model
# =====================
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# =====================
# 3. Paraphrase function
# =====================
def paraphrase_batch(instructions):
    prompts = []
    
    for instr in instructions:
        prompt = f"""
                    Rewrite this robot command.

                    Rules:
                    - Keep structure: VERB OBJECT1 RELATION OBJECT2
                    - Only replace words with simple synonyms
                    - Allowed verbs: move, place, put, bring, pick, grab
                    - Allowed relations: near, next to, close to, on, into
                    - Do NOT add extra words
                    - Output exactly one short command

                    Command: {instr}
                    """

        messages = [
            {"role": "system", "content": "You rewrite robot commands."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(text)

    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32,              # giảm length để tránh lan man
        do_sample=True,
        temperature=0.7,               # ↓ giảm randomness → ổn định hơn
        top_p=0.9
    )

    outputs = []

    for instr, input_ids, output_ids in zip(instructions, model_inputs.input_ids, generated_ids):
        gen = output_ids[len(input_ids):]
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()

        # =====================
        # Post-process cleanup
        # =====================
        text = text.split("\n")[0]              # lấy dòng đầu
        text = text.replace('"', '').strip()
        text = text.replace("Command:", "").strip()

        # =====================
        # Filter bad outputs
        # =====================
        def is_bad(orig, para):
            para_low = para.lower()

            banned = ["assistant", "human", "please", "write", "code", "translate", "instruction"]
            if any(b in para_low for b in banned):
                return True

            # quá dài → suspect
            if len(para.split()) > len(orig.split()) + 4:
                return True

            # quá khác → suspect (simple heuristic)
            if len(para) < 3:
                return True

            return False

        if is_bad(instr, text):
            text = instr  # fallback

        outputs.append(text)

    return outputs


# =====================
# 4. Apply paraphrase
# =====================
batch_size = 16
paraphrased = []

for i in tqdm(range(0, len(df), batch_size), desc="Paraphrasing"):
    batch_instr = df["instruction"].iloc[i:i+batch_size].tolist()
    batch_out = paraphrase_batch(batch_instr)
    paraphrased.extend(batch_out)

df["paraphrased_instruction"] = paraphrased

# =====================
# 5. Save
# =====================
df.to_csv("robot_tasks_with_paraphrase.csv", index=False)

print(df.head())